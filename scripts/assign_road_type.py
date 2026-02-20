#!/usr/bin/env python3
"""Assign road type to annotations based on nearest point from itinerary data."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROAD_TYPES = ["primary", "secondary", "tertiary", "residential"]


def load_annotations(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load annotations CSV and filter to rows with valid GPS coordinates."""
    df = pd.read_csv(csv_path)
    df_with_gps = df.dropna(subset=["gps_lat", "gps_lon"])
    df_with_gps = df_with_gps[
        (df_with_gps["gps_lat"] > 1) & (df_with_gps["gps_lon"] > 1)
    ]
    print(
        f"Loaded {len(df)} annotations, {len(df_with_gps)} have valid GPS coordinates"
    )
    return df, df_with_gps


def load_itineraries(base_path: Path) -> pd.DataFrame:
    """Load and combine itinerary data from both Navi Mumbai and Mumbai city."""
    dfs = []

    navi_mumbai_path = base_path / "itineraries/output/detailed_itineraries.csv"
    if navi_mumbai_path.exists():
        navi = pd.read_csv(navi_mumbai_path)
        navi = navi.dropna(subset=["lat", "lon", "highway_type"])
        navi = navi[["lat", "lon", "highway_type"]]
        print(f"Loaded {len(navi)} Navi Mumbai itinerary points")
        dfs.append(navi)

    mumbai_city_path = base_path / "itineraries/itineraries-mumbai-city.csv"
    if mumbai_city_path.exists():
        mc = pd.read_csv(mumbai_city_path)
        mc_start = pd.DataFrame(
            {
                "lat": mc["start_long"],
                "lon": mc["start_lat"],
                "highway_type": mc["type"],
            }
        )
        mc_end = pd.DataFrame(
            {"lat": mc["end_long"], "lon": mc["end_lat"], "highway_type": mc["type"]}
        )
        mc_combined = pd.concat([mc_start, mc_end], ignore_index=True)
        mc_combined = mc_combined.dropna(subset=["lat", "lon", "highway_type"])
        print(f"Loaded {len(mc_combined)} Mumbai city itinerary points")
        dfs.append(mc_combined)

    if not dfs:
        raise FileNotFoundError("No itinerary files found")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["lat", "lon"])
    print(f"Combined: {len(combined)} unique itinerary points")

    highway_counts = combined["highway_type"].value_counts()
    print(f"Road type distribution:\n{highway_counts}")

    return combined


def assign_road_types(
    df_with_gps: pd.DataFrame, itineraries_df: pd.DataFrame
) -> pd.DataFrame:
    """Assign road type to each annotation based on nearest itinerary point."""
    itinerary_coords = np.radians(itineraries_df[["lat", "lon"]].values)

    tree = BallTree(itinerary_coords, metric="haversine")

    annotation_coords = np.radians(df_with_gps[["gps_lat", "gps_lon"]].values)

    distances, indices = tree.query(annotation_coords, k=1)

    R = 6371000
    distances_m = distances.flatten() * R

    road_types = itineraries_df.iloc[indices.flatten()]["highway_type"].values

    result = df_with_gps.copy()
    result["road_type"] = road_types
    result["road_distance_m"] = distances_m

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assign road types to annotations based on nearest itinerary point"
    )
    parser.add_argument(
        "--annotations",
        "-a",
        default="data/mumbai_annotations_with_exif.csv",
        help="Path to annotations CSV with GPS coordinates",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/mumbai_annotations_with_exif.csv",
        help="Output CSV path (default: overwrite input)",
    )

    args = parser.parse_args()

    base_path = Path(__file__).parent.parent
    annotations_path = base_path / args.annotations
    output_path = base_path / args.output

    if not annotations_path.exists():
        print(f"Annotations file not found: {annotations_path}")
        return 1

    df_full, df_with_gps = load_annotations(annotations_path)
    itineraries_df = load_itineraries(base_path)

    print(f"\nAssigning road types to {len(df_with_gps)} annotations...")
    df_with_road_types = assign_road_types(df_with_gps, itineraries_df)

    df_full["road_type"] = None
    df_full["road_distance_m"] = None
    df_full.loc[df_with_road_types.index, "road_type"] = df_with_road_types[
        "road_type"
    ].values
    df_full.loc[df_with_road_types.index, "road_distance_m"] = df_with_road_types[
        "road_distance_m"
    ].values

    df_full.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n" + "=" * 60)
    print("ROAD TYPE ASSIGNMENT COMPLETE")
    print("=" * 60)

    road_type_counts = df_full["road_type"].value_counts()
    print("\nRoad type distribution in annotations:")
    print(road_type_counts)

    target_types = df_full[df_full["road_type"].isin(ROAD_TYPES)]
    print(
        f"\nAnnotations on target road types ({', '.join(ROAD_TYPES)}): {len(target_types)}"
    )

    valid_distances = df_full["road_distance_m"].dropna()
    print("\nDistance statistics:")
    print(f"  Mean: {valid_distances.mean():.1f}m")
    print(f"  Median: {valid_distances.median():.1f}m")
    print(f"  Max: {valid_distances.max():.1f}m")
    print(f"  % within 100m: {(valid_distances < 100).mean() * 100:.1f}%")
    print(f"  % within 500m: {(valid_distances < 500).mean() * 100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
