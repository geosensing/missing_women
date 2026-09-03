#!/usr/bin/env python3
"""Plot GPS tracks from recorded videos on a Delhi map.

Creates a static map (PNG/PDF) showing video coverage with tracks
colored by video_id.
"""

import argparse
import re
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString

DELHI_BOUNDS = {
    "lat_min": 28.4,
    "lat_max": 28.9,
    "lon_min": 76.8,
    "lon_max": 77.5,
}


def parse_dms(dms_str: str) -> float:
    """Parse DMS coordinate string to decimal degrees."""
    pattern = r"(\d+)\s*deg\s*(\d+)'\s*([\d.]+)\"\s*([NSEW])"
    match = re.match(pattern, dms_str.strip())
    if not match:
        raise ValueError(f"Cannot parse DMS string: {dms_str}")

    degrees = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    direction = match.group(4)

    decimal = degrees + minutes / 60 + seconds / 3600

    if direction in ("S", "W"):
        decimal = -decimal

    return decimal


def main():
    parser = argparse.ArgumentParser(
        description="Plot GPS tracks from recorded videos on a Delhi map"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input GPS timeseries CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output image file (PNG or PDF)",
    )
    parser.add_argument(
        "-n",
        "--downsample",
        type=int,
        default=5,
        help="Keep every Nth point (default: 5)",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply spline smoothing to tracks",
    )
    parser.add_argument(
        "--smooth-factor",
        type=float,
        default=0.0001,
        help="Smoothing factor for splines (default: 0.0001)",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading GPS data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows")

    print("Parsing coordinates...")
    df["lat"] = df["gps_latitude"].apply(parse_dms)
    df["lon"] = df["gps_longitude"].apply(parse_dms)

    print("Filtering to Delhi bounding box...")
    total_before = len(df)
    delhi_mask = (
        (df["lat"] >= DELHI_BOUNDS["lat_min"])
        & (df["lat"] <= DELHI_BOUNDS["lat_max"])
        & (df["lon"] >= DELHI_BOUNDS["lon_min"])
        & (df["lon"] <= DELHI_BOUNDS["lon_max"])
    )
    df = df[delhi_mask].copy()
    filtered_out = total_before - len(df)
    print(f"Filtered out {filtered_out:,} points outside Delhi bounds")
    print(f"Remaining: {len(df):,} points")

    print(f"Downsampling to every {args.downsample}th point...")
    df = df.iloc[:: args.downsample].copy()
    print(f"After downsampling: {len(df):,} points")

    video_ids = df["video_id"].unique()
    print(f"Creating tracks for {len(video_ids)} videos...")
    if args.smooth:
        print(f"Applying spline smoothing (factor={args.smooth_factor})...")

    lines = []
    for video_id in video_ids:
        video_df = df[df["video_id"] == video_id].sort_values("gps_datetime")
        if len(video_df) < 2:
            continue

        lons = video_df["lon"].values
        lats = video_df["lat"].values

        if args.smooth and len(video_df) >= 4:
            try:
                tck, _ = splprep([lons, lats], s=args.smooth_factor, k=3)
                u_new = np.linspace(0, 1, len(video_df) * 2)
                smooth_lon, smooth_lat = splev(u_new, tck)
                coords = list(zip(smooth_lon, smooth_lat))
            except Exception:
                coords = list(zip(lons, lats))
        else:
            coords = list(zip(lons, lats))

        lines.append({"video_id": video_id, "geometry": LineString(coords)})

    gdf = gpd.GeoDataFrame(lines, crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    _, ax = plt.subplots(figsize=(12, 12))

    gdf.plot(ax=ax, linewidth=1.5, alpha=0.7, legend=False)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    ax.set_axis_off()
    ax.set_title("Delhi Video GPS Coverage", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Map saved to {args.output}")


if __name__ == "__main__":
    main()
