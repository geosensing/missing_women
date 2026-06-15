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
    from shapely.geometry import Point
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


def load_osm_roads(pbf_path: Path, bbox: tuple) -> "gpd.GeoDataFrame | None":
    """
    Load road network from OSM PBF file within bounding box.

    Args:
        pbf_path: Path to .osm.pbf file
        bbox: (minlat, minlon, maxlat, maxlon)

    Returns:
        GeoDataFrame with road geometries, or None if not available
    """
    if not HAS_GEO:
        return None

    try:
        import osmnx as ox
    except ImportError:
        print("  osmnx not installed, skipping OSM enrichment")
        return None

    if not pbf_path.exists():
        print(f"  OSM file not found: {pbf_path}")
        return None

    try:
        minlat, minlon, maxlat, maxlon = bbox
        north, south, east, west = maxlat, minlat, maxlon, minlon

        G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False)
        return edges
    except Exception as e:
        print(f"  Error loading OSM: {e}")
        return None


def match_to_nearest_road(
    df: pd.DataFrame, roads_gdf: "gpd.GeoDataFrame | None", max_distance_m: float = 50.0
) -> pd.DataFrame:
    """
    Match frames to nearest OSM road.

    Returns df with added osm_* columns.
    """
    result = df.copy()
    result["osm_highway"] = None
    result["osm_road_name"] = None
    result["osm_surface"] = None
    result["osm_distance_m"] = None

    if roads_gdf is None or not HAS_GEO:
        return result

    valid_mask = df["gps_lat"].notna() & df["gps_lon"].notna()
    if not valid_mask.any():
        return result

    roads_gdf = roads_gdf.to_crs("EPSG:32643")

    for idx in df.index[valid_mask]:
        lat, lon = df.at[idx, "gps_lat"], df.at[idx, "gps_lon"]
        point = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:32643").iloc[0]

        distances = roads_gdf.geometry.distance(point)
        nearest_idx = distances.idxmin()
        nearest_dist = distances.loc[nearest_idx]

        if nearest_dist <= max_distance_m:
            road = roads_gdf.loc[nearest_idx]
            result.at[idx, "osm_highway"] = road.get("highway")
            result.at[idx, "osm_road_name"] = road.get("name")
            result.at[idx, "osm_surface"] = road.get("surface")
            result.at[idx, "osm_distance_m"] = nearest_dist

    return result


def enrich_with_geo(
    annotations: pd.DataFrame,
    city_sampling_dir: Path,
    city: str,
    osm_pbf_path: Path | None = None,
) -> pd.DataFrame:
    """
    Add geographic attributes to annotations.

    Args:
        annotations: DataFrame with gps_lat, gps_lon columns
        city_sampling_dir: Path to sampling/{city} directory
        city: City name for region tagging
        osm_pbf_path: Optional path to OSM PBF file

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

    if osm_pbf_path and osm_pbf_path.exists():
        valid_coords = annotations[["gps_lat", "gps_lon"]].dropna()
        if not valid_coords.empty:
            bbox = (
                valid_coords["gps_lat"].min() - 0.01,
                valid_coords["gps_lon"].min() - 0.01,
                valid_coords["gps_lat"].max() + 0.01,
                valid_coords["gps_lon"].max() + 0.01,
            )
            print("Loading OSM road network...")
            roads = load_osm_roads(osm_pbf_path, bbox)

            if roads is not None:
                print("Matching to OSM roads...")
                result = match_to_nearest_road(result, roads)
                n_osm = result["osm_highway"].notna().sum()
                print(f"  Matched {n_osm}/{len(result)} frames to OSM roads")
    else:
        print("Skipping OSM enrichment (PBF file not available)")
        result["osm_highway"] = None
        result["osm_road_name"] = None
        result["osm_surface"] = None
        result["osm_distance_m"] = None

    return result


if __name__ == "__main__":
    import argparse
    import importlib.util

    import yaml

    def import_from_path(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City to process")
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

    with open(project_root / "cities.yaml") as f:
        city_config = yaml.safe_load(f)[args.city]

    labelstudio_dir = project_root / "data" / args.city / "labelstudio"
    gps_index_dir = project_root / "output" / args.city / "gps_index"
    sampling_dir = project_root / "sampling" / args.city
    osm_path = project_root / "data" / "osm" / city_config["osm_file"]

    annotations = parse_all_annotations(labelstudio_dir, args.city)
    video_meta = pd.read_parquet(gps_index_dir / "video_metadata.parquet")
    gps_df = pd.read_parquet(gps_index_dir / "gps_timeseries.parquet")

    with_gps = assign_frame_gps(annotations, video_meta, gps_df)
    result = enrich_with_geo(with_gps, sampling_dir, args.city, osm_path)

    print("\n=== Summary ===")
    print(f"Total frames: {len(result)}")
    print(f"With itinerary match: {result['itinerary_road_type'].notna().sum()}")
    if "osm_highway" in result.columns:
        print(f"With OSM match: {result['osm_highway'].notna().sum()}")
