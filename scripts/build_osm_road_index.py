#!/usr/bin/env python3
"""
Build a compact, committed OSM road index per city for road-type enrichment.

Step 07 matches each annotated frame to the nearest OSM road to get its road type. The
full city road networks (``sampling/{city}/network/streets.geojson``) are 260--320 MB
and gitignored, and Mumbai has none at all (07 would download via osmnx, needing
internet). Neither is reproducible from a clone.

This script clips the road network to only the roads near where the camera actually went
(within 100 m of the GPS track) and writes ``data/{city}/osm_roads.parquet`` (~2 MB,
committed). Step 07 prefers that index, so the road-type step becomes small, offline, and
reproducible for every city.

Source network per city:
    - if ``sampling/{city}/network/streets.geojson`` exists, clip that (offline);
    - otherwise download the GPS-track bounding box via osmnx (Mumbai; needs internet
      once, then the committed parquet is used thereafter).

Usage:
    python scripts/build_osm_road_index.py                 # all four cities
    python scripts/build_osm_road_index.py --city mumbai
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import geopandas as gpd
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
DATA = PROJECT_ROOT / "data"

# UTM 43N: metric CRS covering all four cities, so buffers/distances are in metres.
METRIC_CRS = "EPSG:32643"
BUFFER_M = 100.0  # keep roads within 100 m of the track (07 matches within 50 m)
MAX_TRACK_POINTS = 20000  # downsample the GPS track for an efficient spatial join

spec = importlib.util.spec_from_file_location("enrich_with_geo", SCRIPTS / "07_enrich_with_geo.py")
enrich = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enrich)


def load_track_points(city: str) -> pd.DataFrame:
    """GPS track (camera path) for a city, from the committed gps_index."""
    path = DATA / city / "gps_index" / "gps_timeseries.parquet"
    df = pd.read_parquet(path, columns=["lat", "lon"]).dropna()
    if len(df) > MAX_TRACK_POINTS:
        df = df.iloc[:: len(df) // MAX_TRACK_POINTS]
    return df


def source_roads(city: str, track: pd.DataFrame) -> "gpd.GeoDataFrame | None":
    """Full road network for a city: local sampling network, else osmnx download."""
    local = enrich.load_local_osm_roads(
        PROJECT_ROOT / "sampling" / city / "network" / "streets.geojson"
    )
    if local is not None:
        print(f"  source: local sampling network ({len(local)} roads)")
        return local
    lat, lon = track["lat"], track["lon"]
    bbox = (
        lat.quantile(0.005) - 0.01,
        lon.quantile(0.005) - 0.01,
        lat.quantile(0.995) + 0.01,
        lon.quantile(0.995) + 0.01,
    )
    print("  source: osmnx download (no local network)...")
    return enrich.load_osm_roads(bbox)


def build_city(city: str) -> None:
    print(f"\n{city}")
    track = load_track_points(city)
    roads = source_roads(city, track)
    if roads is None or len(roads) == 0:
        print("  no road network available; skipping")
        return

    roads_m = roads.to_crs(METRIC_CRS).reset_index(drop=True)
    pts_m = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(track["lon"], track["lat"]), crs="EPSG:4326"
    ).to_crs(METRIC_CRS)

    near = gpd.sjoin_nearest(roads_m, pts_m, max_distance=BUFFER_M, how="inner")
    clipped = roads_m.loc[sorted(near.index.unique())].to_crs("EPSG:4326").copy()

    # osmnx returns list-valued tags for multi-tag edges; collapse to a scalar so the
    # column is GeoParquet-writable and matches how 07 reads highway tags.
    for col in ("highway", "name"):
        if col in clipped.columns:
            clipped[col] = clipped[col].map(enrich._first)
    keep = [c for c in ["highway", "name", "geometry"] if c in clipped.columns]
    clipped = clipped[keep]

    out = DATA / city / "osm_roads.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_parquet(out, index=False)
    size_mb = out.stat().st_size / 1e6
    print(
        f"  -> {out}  ({len(clipped)} roads, {size_mb:.1f} MB; {100 * len(clipped) / len(roads):.1f}% of full)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--city",
        type=str,
        default="mumbai,navi_mumbai,bangalore,delhi",
        help="Comma-separated list of cities",
    )
    args = parser.parse_args()
    for city in (c.strip() for c in args.city.split(",") if c.strip()):
        build_city(city)


if __name__ == "__main__":
    main()
