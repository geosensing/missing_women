#!/usr/bin/env python3
"""
Hyderabad Data Collection Sampler

This script generates optimized data collection itineraries for Hyderabad by:
1. Extracting Hyderabad district boundary from OpenStreetMap
2. Downloading the street network via OSMnx
3. Sampling random points along streets
4. Getting OSRM distance matrix for all points
5. Using allocator.greedy_grow_itineraries() to create ~100km routes
6. Generating visualizations and Google Maps links

Supports --resume flag to skip completed steps.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.sampler_utils import (
    create_folium_map,
    create_itineraries,
    create_static_visualizations,
    osrm_distance_matrix,
    save_itineraries,
    save_map_urls,
    split_road_segment,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "sampling" / "hyderabad"
N_SAMPLE_POINTS = 1000
MAX_ITINERARY_DISTANCE = 100000  # 100 km in meters
SEGMENT_LENGTH = 500  # meters
RANDOM_SEED = 42

os.makedirs(f"{OUTPUT_DIR}/boundaries", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/network", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sampled_points", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/itineraries", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)


def check_step_complete(step_name):
    """Check if a step has already been completed."""
    checks = {
        "boundary": Path(f"{OUTPUT_DIR}/boundaries/hyderabad_boundary.geojson").exists(),
        "network": Path(f"{OUTPUT_DIR}/network/street_network.graphml").exists(),
        "sampled_points": Path(f"{OUTPUT_DIR}/sampled_points/sampled_points.csv").exists(),
        "itineraries": Path(f"{OUTPUT_DIR}/itineraries/itineraries_summary.csv").exists(),
    }
    return checks.get(step_name, False)


def main():
    parser = argparse.ArgumentParser(description="Hyderabad Data Collection Sampler")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from completed steps"
    )
    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 50)
    print("Hyderabad Data Collection Sampler")
    print("=" * 50)

    # 1. Extract Hyderabad Boundary from OSM
    if args.resume and check_step_complete("boundary"):
        print("\n1. Loading existing Hyderabad boundary...")
        boundary_gdf = gpd.read_file(f"{OUTPUT_DIR}/boundaries/hyderabad_boundary.geojson")
        hyderabad_boundary = boundary_gdf.geometry.union_all()
    else:
        print("\n1. Extracting Hyderabad district boundary from OpenStreetMap...")
        gdf_admin = ox.features_from_place("Hyderabad, India", tags={"boundary": "administrative"})

        hyderabad_gdf = gdf_admin[
            (gdf_admin["name"] == "Hyderabad") & (gdf_admin["admin_level"] == "6")
        ]

        if len(hyderabad_gdf) == 0:
            hyderabad_gdf = gdf_admin[
                (gdf_admin["name"] == "Hyderabad") & (gdf_admin["admin_level"].isin(["4", "5"]))
            ]

        if len(hyderabad_gdf) == 0:
            raise ValueError("Could not find Hyderabad boundary in OSM data")

        hyderabad_boundary = hyderabad_gdf.geometry.union_all()

        boundary_gdf = gpd.GeoDataFrame(geometry=[hyderabad_boundary], crs="EPSG:4326")
        boundary_gdf.to_file(f"{OUTPUT_DIR}/boundaries/hyderabad_boundary.geojson", driver="GeoJSON")

        _, ax = plt.subplots(figsize=(10, 10))
        hyderabad_gdf.plot(ax=ax, facecolor="lightblue", edgecolor="blue", linewidth=2)
        ax.set_title("Hyderabad District Boundary")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/visualizations/boundary.png", dpi=150)
        plt.close()

        print(f"   Boundary bounds: {hyderabad_boundary.bounds}")

    # 2. Download Street Network via OSMnx
    if args.resume and check_step_complete("network"):
        print("\n2. Loading existing street network...")
        G = ox.load_graphml(f"{OUTPUT_DIR}/network/street_network.graphml")
        nodes, edges = ox.graph_to_gdfs(G)
    else:
        print("\n2. Downloading street network via OSMnx...")
        minx, miny, maxx, maxy = hyderabad_boundary.bounds
        print(f"   Bounding box: N={maxy:.4f}, S={miny:.4f}, E={maxx:.4f}, W={minx:.4f}")

        G = ox.graph_from_polygon(hyderabad_boundary, network_type="drive")
        print(f"   Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")

        ox.save_graphml(G, f"{OUTPUT_DIR}/network/street_network.graphml")

        nodes, edges = ox.graph_to_gdfs(G)
        edges.to_file(f"{OUTPUT_DIR}/network/streets.geojson", driver="GeoJSON")

        _, ax = plt.subplots(figsize=(12, 12))
        edges.plot(ax=ax, linewidth=0.3, color="gray")
        boundary_gdf.boundary.plot(ax=ax, color="blue", linewidth=2)
        ax.set_title("Hyderabad Street Network")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/visualizations/street_network.png", dpi=150)
        plt.close()

    # 3. Sample Points Along Streets
    if args.resume and check_step_complete("sampled_points"):
        print("\n3. Loading existing sampled points...")
        df_sampled = pd.read_csv(f"{OUTPUT_DIR}/sampled_points/sampled_points.csv")
        sampled_gdf = gpd.read_file(f"{OUTPUT_DIR}/sampled_points/sampled_points.geojson")
    else:
        print("\n3. Sampling points along streets...")

        all_segment_points = []
        road_types_of_interest = [
            "residential",
            "primary",
            "secondary",
            "tertiary",
            "trunk",
            "unclassified",
        ]

        for idx, row in edges.iterrows():
            highway = row.get("highway", "")
            if isinstance(highway, list):
                highway = highway[0] if highway else ""

            if highway in road_types_of_interest:
                points = split_road_segment(row.geometry, SEGMENT_LENGTH)
                for lon, lat in points:
                    all_segment_points.append({"lon": lon, "lat": lat, "highway": highway})

        print(f"   Total segment points available: {len(all_segment_points)}")

        n_samples = min(N_SAMPLE_POINTS, len(all_segment_points))
        random.seed(RANDOM_SEED)
        sampled_indices = random.sample(range(len(all_segment_points)), n_samples)
        sampled_points = [all_segment_points[i] for i in sampled_indices]

        df_sampled = pd.DataFrame(sampled_points)
        df_sampled["point_id"] = range(len(df_sampled))

        geometry = [Point(row["lon"], row["lat"]) for _, row in df_sampled.iterrows()]
        sampled_gdf = gpd.GeoDataFrame(df_sampled, geometry=geometry, crs="EPSG:4326")

        sampled_gdf.to_file(
            f"{OUTPUT_DIR}/sampled_points/sampled_points.geojson", driver="GeoJSON"
        )
        df_sampled.to_csv(f"{OUTPUT_DIR}/sampled_points/sampled_points.csv", index=False)

        print(f"   Sampled {len(sampled_gdf)} points")

        _, ax = plt.subplots(figsize=(12, 12))
        edges.plot(ax=ax, linewidth=0.2, color="lightgray")
        sampled_gdf.plot(ax=ax, color="red", markersize=5, alpha=0.6)
        boundary_gdf.boundary.plot(ax=ax, color="blue", linewidth=2)
        ax.set_title(f"Sampled Points ({len(sampled_gdf)} points)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/visualizations/sampled_points.png", dpi=150)
        plt.close()

    # 4. Compute full OSRM distance matrix
    print("\n4. Computing OSRM distance matrix for all points...")
    coords = df_sampled[["lon", "lat"]].to_numpy()
    distance_matrix = osrm_distance_matrix(coords, use_local=False)

    if distance_matrix is None:
        raise ValueError("OSRM distance matrix computation failed")

    distance_matrix = np.where(distance_matrix is None, np.inf, distance_matrix)
    distance_matrix = distance_matrix.astype(float)

    print(f"   Distance matrix shape: {distance_matrix.shape}")

    # 5. Create itineraries
    print("\n5. Creating ~100km itineraries...")
    all_itineraries, optimized_indices, itinerary_distances = create_itineraries(
        df_sampled, distance_matrix, MAX_ITINERARY_DISTANCE, rng
    )

    # 6. Save itineraries
    print("\n6. Saving itineraries...")
    df_itineraries, df_summary = save_itineraries(
        all_itineraries, optimized_indices, distance_matrix, df_sampled, OUTPUT_DIR
    )

    # 7. Generate map URLs
    print("\n7. Generating map URLs...")
    save_map_urls(all_itineraries, df_itineraries, OUTPUT_DIR)

    # 8. Visualizations
    print("\n8. Creating visualizations...")
    create_static_visualizations(all_itineraries, edges, boundary_gdf, OUTPUT_DIR)

    # 9. Create Interactive Folium Map
    print("\n9. Creating interactive Folium map...")
    create_folium_map(all_itineraries, boundary_gdf, hyderabad_boundary, OUTPUT_DIR, "Hyderabad")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Boundary: {OUTPUT_DIR}/boundaries/hyderabad_boundary.geojson")
    print(f"Street network: {OUTPUT_DIR}/network/street_network.graphml")
    print(f"Sampled points: {OUTPUT_DIR}/sampled_points/sampled_points.geojson")
    print(f"Itineraries: {OUTPUT_DIR}/itineraries/itineraries.csv")
    print(f"Google Maps links: {OUTPUT_DIR}/itineraries/google_maps_itineraries.csv")
    print(f"OSRM links: {OUTPUT_DIR}/itineraries/osrm_itineraries.csv")
    print(f"\nTotal itineraries: {len(all_itineraries)}")
    print(f"Total sampled points: {len(df_sampled)}")
    print(f"Random seed: {RANDOM_SEED}")
    print("\nDone!")


if __name__ == "__main__":
    main()
