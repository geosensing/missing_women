"""
Add geographic attributes from OSM and itinerary data.

Two road type sources for verification:
- osm_highway: from OpenStreetMap (ground truth)
- itinerary_road_type: from sampling itineraries (what we intended)

Comparing them reveals sampling accuracy or OSM data changes.

OSM enrichment is optional - if PBF file not found, we skip it.

Inputs:
    annotations DataFrame (from 06_assign_frame_gps)
    data/osm/{osm_file}.osm.pbf (optional, configured per city)
    sampling/{city}/itineraries/*.csv

Returns:
    DataFrame with added columns:
        osm_highway, osm_road_name, osm_surface, osm_distance_m (if OSM available)
        itinerary_road_type, itinerary_distance_m
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from sklearn.neighbors import BallTree

    HAS_GEO = True
except ImportError:
    HAS_GEO = False


def load_itineraries(city_sampling_dir: Path, city: str) -> pd.DataFrame:
    """
    Load itinerary data for a specific city.

    Args:
        city_sampling_dir: Path to sampling/{city} directory
        city: City name for region tagging

    Returns:
        DataFrame with columns: lat, lon, road_type, itinerary_id, region
    """
    rows = []

    itin_dir = city_sampling_dir / "itineraries"

    if not itin_dir.exists():
        return pd.DataFrame(columns=["lat", "lon", "road_type", "itinerary_id", "region"])

    for csv_path in itin_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "lat" in df.columns and "lon" in df.columns:
            for _, row in df.iterrows():
                road_type = row.get("highway_type", row.get("type", row.get("highway")))
                rows.append(
                    {
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "road_type": road_type,
                        "itinerary_id": row.get("route_id", row.get("itinerary_id")),
                        "region": city,
                    }
                )
        elif "start_lat" in df.columns and "start_long" in df.columns:
            sample_lat = df["start_lat"].dropna().iloc[0] if len(df) > 0 else 0
            swap_needed = sample_lat > 30

            for _, row in df.iterrows():
                road_type = row.get("type", row.get("highway"))
                if swap_needed:
                    lat_val, lon_val = row["start_long"], row["start_lat"]
                else:
                    lat_val, lon_val = row["start_lat"], row["start_long"]
                rows.append(
                    {
                        "lat": lat_val,
                        "lon": lon_val,
                        "road_type": road_type,
                        "itinerary_id": row.get("itinerary_id"),
                        "region": city,
                    }
                )
                if "end_lat" in df.columns and "end_long" in df.columns:
                    if swap_needed:
                        lat_val, lon_val = row["end_long"], row["end_lat"]
                    else:
                        lat_val, lon_val = row["end_lat"], row["end_long"]
                    rows.append(
                        {
                            "lat": lat_val,
                            "lon": lon_val,
                            "road_type": road_type,
                            "itinerary_id": row.get("itinerary_id"),
                            "region": city,
                        }
                    )

    return pd.DataFrame(rows)


def assign_itinerary_road_type(
    df: pd.DataFrame, itineraries: pd.DataFrame, max_distance_m: float = 100.0
) -> pd.DataFrame:
    """
    Assign itinerary road type based on nearest itinerary point.

    Uses BallTree for efficient nearest neighbor lookup.
    """
    result = df.copy()
    result["itinerary_road_type"] = None
    result["itinerary_distance_m"] = None

    if not HAS_GEO:
        print("  Skipping itinerary matching (geopandas/sklearn not installed)")
        return result

    valid_mask = df["gps_lat"].notna() & df["gps_lon"].notna()
    if not valid_mask.any():
        return result

    itin_valid = itineraries.dropna(subset=["lat", "lon"])
    if itin_valid.empty:
        return result

    itin_coords = np.radians(itin_valid[["lat", "lon"]].astype(float).values)
    tree = BallTree(itin_coords, metric="haversine")

    frame_coords = np.radians(df.loc[valid_mask, ["gps_lat", "gps_lon"]].astype(float).values)

    distances, indices = tree.query(frame_coords, k=1)
    distances_m = distances.flatten() * 6371000

    for i, (idx, dist_m, itin_idx) in enumerate(
        zip(df.index[valid_mask], distances_m, indices.flatten())
    ):
        if dist_m <= max_distance_m:
            result.at[idx, "itinerary_road_type"] = itin_valid.iloc[itin_idx]["road_type"]
            result.at[idx, "itinerary_distance_m"] = dist_m
        else:
            result.at[idx, "itinerary_distance_m"] = dist_m

    return result


def _first(value):
    """OSM tag values can be a list/array (e.g. ['residential', 'service']); take the first.

    osmnx returns Python lists for multi-tag edges, but a network read back from
    GeoJSON returns numpy arrays; collapse both to a single scalar tag so the column
    stays object-of-strings and is parquet-writable.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) > 0 else None
    return value


def load_local_osm_roads(network_geojson: Path) -> "gpd.GeoDataFrame | None":
    """
    Load the OSM road network saved during route sampling.

    ``sampling/{city}/network/streets.geojson`` is the exact OSM drive network the
    itineraries were sampled from (LineStrings with a ``highway`` tag). Using it for
    road-type attribution is offline, reproducible, and avoids the Overpass API.
    """
    if not HAS_GEO or not network_geojson.exists():
        return None
    try:
        cols = ["highway", "name"]
        roads = gpd.read_file(network_geojson, columns=cols)
        keep = [c for c in [*cols, "geometry"] if c in roads.columns]
        return roads[keep]
    except Exception as e:
        print(f"  Error reading local OSM network: {e}")
        return None


def load_osm_roads(bbox: tuple) -> "gpd.GeoDataFrame | None":
    """
    Download the drivable OSM road network within a bounding box via osmnx.

    Args:
        bbox: (minlat, minlon, maxlat, maxlon)

    Returns:
        GeoDataFrame of road edges (EPSG:4326), or None if unavailable.
    """
    if not HAS_GEO:
        return None

    try:
        import osmnx as ox
    except ImportError:
        print("  osmnx not installed, skipping OSM enrichment")
        return None

    # City-wide drive networks are large; allow Overpass more time than the default.
    ox.settings.requests_timeout = 600

    try:
        minlat, minlon, maxlat, maxlon = bbox
        # osmnx >= 2.0 takes bbox=(left, bottom, right, top) = (minlon, minlat, maxlon, maxlat)
        G = ox.graph_from_bbox(bbox=(minlon, minlat, maxlon, maxlat), network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False)
        keep = [c for c in ["highway", "name", "surface", "geometry"] if c in edges.columns]
        return edges[keep].reset_index(drop=True)
    except Exception as e:
        print(f"  Error loading OSM: {e}")
        return None


def match_to_nearest_road(
    df: pd.DataFrame, roads_gdf: "gpd.GeoDataFrame | None", max_distance_m: float = 50.0
) -> pd.DataFrame:
    """
    Match frames to the nearest OSM road via a spatial nearest-join.

    Returns df with added osm_highway, osm_road_name, osm_surface, osm_distance_m.
    """
    result = df.copy()
    result["osm_highway"] = None
    result["osm_road_name"] = None
    result["osm_surface"] = None
    result["osm_distance_m"] = None

    if roads_gdf is None or not HAS_GEO or len(roads_gdf) == 0:
        return result

    valid_mask = df["gps_lat"].notna() & df["gps_lon"].notna()
    if not valid_mask.any():
        return result

    # Project to a metric CRS (UTM 43N covers all four cities) so distances are metres.
    pts = gpd.GeoDataFrame(
        {"_idx": df.index[valid_mask]},
        geometry=gpd.points_from_xy(df.loc[valid_mask, "gps_lon"], df.loc[valid_mask, "gps_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32643")
    roads = roads_gdf.to_crs("EPSG:32643")

    joined = gpd.sjoin_nearest(
        pts, roads, max_distance=max_distance_m, distance_col="osm_distance_m"
    )
    # Ties can yield multiple rows per point; keep the nearest.
    joined = joined.sort_values("osm_distance_m").drop_duplicates("_idx")

    for _, r in joined.iterrows():
        idx = r["_idx"]
        result.at[idx, "osm_highway"] = _first(r.get("highway"))
        result.at[idx, "osm_road_name"] = _first(r.get("name"))
        result.at[idx, "osm_surface"] = _first(r.get("surface"))
        result.at[idx, "osm_distance_m"] = r["osm_distance_m"]

    return result


def enrich_with_geo(
    annotations: pd.DataFrame,
    city_sampling_dir: Path,
    city: str,
    use_osm: bool = False,
) -> pd.DataFrame:
    """
    Add geographic attributes to annotations.

    Args:
        annotations: DataFrame with gps_lat, gps_lon columns
        city_sampling_dir: Path to sampling/{city} directory
        city: City name for region tagging
        use_osm: If True, download the OSM drive network for the frames' bounding
            box via osmnx and attach the nearest-road ``osm_highway`` (ground-truth
            road type). Requires network access.

    Returns:
        DataFrame with itinerary and OSM road attributes
    """
    print("Loading itineraries...")
    itineraries = load_itineraries(city_sampling_dir, city)
    print(f"  Found {len(itineraries)} itinerary points")

    print("Matching to itinerary road types...")
    result = assign_itinerary_road_type(annotations, itineraries)
    n_matched = result["itinerary_road_type"].notna().sum()
    print(f"  Matched {n_matched}/{len(result)} frames to itineraries")

    result["osm_highway"] = None
    result["osm_road_name"] = None
    result["osm_surface"] = None
    result["osm_distance_m"] = None

    if not use_osm:
        print("Skipping OSM enrichment")
        return result

    valid_coords = annotations[["gps_lat", "gps_lon"]].dropna()
    if valid_coords.empty:
        print("Skipping OSM enrichment (no GPS coordinates)")
        return result

    # Prefer the committed, route-clipped road index (small, offline, reproducible from a
    # clone — built by build_osm_road_index.py). Fall back to the full local sampling
    # network, then to an osmnx/Overpass download.
    project_root = city_sampling_dir.parent.parent
    road_index = project_root / "data" / city / "osm_roads.parquet"
    roads = None
    if HAS_GEO and road_index.exists():
        roads = gpd.read_parquet(road_index)
        print(f"  Using committed OSM road index: {road_index} ({len(roads)} roads)")
    if roads is None:
        network_geojson = city_sampling_dir / "network" / "streets.geojson"
        roads = load_local_osm_roads(network_geojson)
        if roads is not None:
            print(f"  Using local OSM network: {network_geojson}")
    if roads is None:
        # Robust quantiles, not min/max: a few GPS glitches (wrong-continent fixes)
        # would otherwise blow the bbox up to thousands of km and time out Overpass.
        lat, lon = valid_coords["gps_lat"], valid_coords["gps_lon"]
        bbox = (
            lat.quantile(0.005) - 0.01,
            lon.quantile(0.005) - 0.01,
            lat.quantile(0.995) + 0.01,
            lon.quantile(0.995) + 0.01,
        )
        print("Downloading OSM road network (osmnx)...")
        roads = load_osm_roads(bbox)

    if roads is not None:
        print(f"  {len(roads)} road segments; matching frames to nearest road...")
        result = match_to_nearest_road(result, roads)
        n_osm = result["osm_highway"].notna().sum()
        print(f"  Matched {n_osm}/{len(result)} frames to OSM roads")

    return result


if __name__ == "__main__":
    import argparse
    import importlib.util

    def import_from_path(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City to process")
    parser.add_argument("--osm", action="store_true", help="Download + match OSM road type")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    scripts_dir = Path(__file__).parent

    parse_annotations_mod = import_from_path(
        "parse_annotations", scripts_dir / "05_parse_annotations.py"
    )
    assign_frame_gps_mod = import_from_path(
        "assign_frame_gps", scripts_dir / "06_assign_frame_gps.py"
    )
    parse_all_annotations = parse_annotations_mod.parse_all_annotations
    assign_frame_gps = assign_frame_gps_mod.assign_frame_gps

    labelstudio_dir = project_root / "data" / args.city / "labelstudio"
    gps_index_dir = project_root / "data" / args.city / "gps_index"
    sampling_dir = project_root / "sampling" / args.city

    annotations = parse_all_annotations(labelstudio_dir, args.city)
    video_meta = pd.read_parquet(gps_index_dir / "video_metadata.parquet")
    gps_df = pd.read_parquet(gps_index_dir / "gps_timeseries.parquet")

    with_gps = assign_frame_gps(annotations, video_meta, gps_df)
    result = enrich_with_geo(with_gps, sampling_dir, args.city, use_osm=args.osm)

    print("\n=== Summary ===")
    print(f"Total frames: {len(result)}")
    print(f"With itinerary match: {result['itinerary_road_type'].notna().sum()}")
    if "osm_highway" in result.columns:
        print(f"With OSM match: {result['osm_highway'].notna().sum()}")
