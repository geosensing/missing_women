#!/usr/bin/env python3
"""
Delhi Nearest-Neighbor Chain Sampler

This script generates optimized data collection itineraries for Delhi using
the nearest-neighbor chain algorithm (same as Bangalore):
1. Load existing Delhi sampled points (1000 points)
2. Compute OSRM distance matrix
3. Apply nearest-neighbor sort to create one chain through all points
4. Cut chain into 20km chunks
5. Generate visualizations and Google Maps links

Outputs to: sampling/delhi_nn/
"""

import os
import sys
import urllib.parse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.sampler_utils import (
    generate_osrm_urls,
    osrm_distance_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
INPUT_DIR = PROJECT_ROOT / "sampling" / "delhi"
OUTPUT_DIR = PROJECT_ROOT / "sampling" / "delhi_nn"
MAX_ITINERARY_DISTANCE = 20000  # 20 km in meters (same as Bangalore)

os.makedirs(f"{OUTPUT_DIR}/itineraries", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

print("=" * 50)
print("Delhi Nearest-Neighbor Chain Sampler")
print("=" * 50)

# 1. Load existing sampled points
print("\n1. Loading existing sampled points...")
df_sampled = pd.read_csv(f"{INPUT_DIR}/sampled_points/sampled_points.csv")
print(f"   Loaded {len(df_sampled)} points")

# Load boundary and network for visualizations
boundary_gdf = gpd.read_file(f"{INPUT_DIR}/boundaries/delhi_boundary.geojson")
delhi_boundary = boundary_gdf.geometry.union_all()

import osmnx as ox

G = ox.load_graphml(f"{INPUT_DIR}/network/street_network.graphml")
_, edges = ox.graph_to_gdfs(G)

# 2. Compute OSRM distance matrix
print("\n2. Computing OSRM distance matrix...")
coords = df_sampled[["lon", "lat"]].to_numpy()
print(f"   Computing distance matrix for {len(coords)} points...")
distance_matrix = osrm_distance_matrix(coords, use_local=False)

if distance_matrix is None:
    raise ValueError("OSRM distance matrix computation failed")

distance_matrix = np.where(distance_matrix is None, np.inf, distance_matrix)
distance_matrix = distance_matrix.astype(float)
print(f"   Distance matrix shape: {distance_matrix.shape}")


# 3. Nearest-neighbor sort
def nearest_neighbor_sort(distance_matrix):
    """Sort all points into a single chain using nearest-neighbor heuristic."""
    n = len(distance_matrix)
    visited = np.zeros(n, dtype=bool)
    sorted_indices = [0]
    visited[0] = True

    for _ in range(1, n):
        last_index = sorted_indices[-1]
        remaining_indices = np.where(~visited)[0]
        distances = distance_matrix[last_index, remaining_indices]
        distances = np.where(distances is None, np.inf, distances)
        nearest_index = remaining_indices[np.argmin(distances)]
        sorted_indices.append(nearest_index)
        visited[nearest_index] = True

    return sorted_indices


print("\n3. Sorting points by nearest neighbor...")
sorted_order = nearest_neighbor_sort(distance_matrix)
df_sorted = df_sampled.iloc[sorted_order].reset_index(drop=True)

df_distance_matrix = pd.DataFrame(
    distance_matrix, columns=df_sampled.point_id, index=df_sampled.point_id
)

distances = []
for i in range(len(df_sorted) - 1):
    loc1 = df_sorted.loc[i, "point_id"]
    loc2 = df_sorted.loc[i + 1, "point_id"]
    distance = df_distance_matrix.loc[loc1, loc2]
    if distance is None:
        distance = 0
    distances.append(distance)

df_sorted["osrm_travel_distance"] = [None] + distances

# 4. Create itineraries by cutting the chain into 20km chunks
print("\n4. Creating 20km itineraries...")

itineraries = []
current_itinerary = []
current_distance = 0
itinerary_id = 1

for i, row in df_sorted.iterrows():
    travel_distance = (
        row["osrm_travel_distance"] if i > 0 and row["osrm_travel_distance"] else 0
    )

    if current_distance + travel_distance > MAX_ITINERARY_DISTANCE and len(
        current_itinerary
    ) > 0:
        itineraries.append({"itinerary_id": itinerary_id, "segments": current_itinerary})
        current_itinerary = []
        current_distance = 0
        itinerary_id += 1

    current_itinerary.append(row.to_dict())
    current_distance += travel_distance

if current_itinerary:
    itineraries.append({"itinerary_id": itinerary_id, "segments": current_itinerary})

print(f"   Created {len(itineraries)} itineraries")

# 5. Save itineraries
print("\n5. Saving itineraries...")

flat_data = []
for itinerary in itineraries:
    itin_id = itinerary["itinerary_id"]
    for segment in itinerary["segments"]:
        segment["itinerary_id"] = itin_id
        flat_data.append(segment)

df_itineraries = pd.DataFrame(flat_data)
df_itineraries.to_csv(f"{OUTPUT_DIR}/itineraries/itineraries.csv", index=False)

summary_data = []
for itinerary in itineraries:
    itin_id = itinerary["itinerary_id"]
    n_points = len(itinerary["segments"])
    total_dist = sum(
        s.get("osrm_travel_distance", 0) or 0 for s in itinerary["segments"]
    )
    summary_data.append(
        {
            "itinerary_id": itin_id,
            "n_points": n_points,
            "total_distance_m": total_dist,
            "total_distance_km": total_dist / 1000,
        }
    )

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(f"{OUTPUT_DIR}/itineraries/itineraries_summary.csv", index=False)
print(f"   Summary:\n{df_summary.to_string()}")

itin_geometry = []
for itinerary in itineraries:
    points = [(s["lon"], s["lat"]) for s in itinerary["segments"]]
    if len(points) >= 2:
        line = LineString(points)
        itin_geometry.append({"itinerary_id": itinerary["itinerary_id"], "geometry": line})

itin_gdf = gpd.GeoDataFrame(itin_geometry, crs="EPSG:4326")
itin_gdf.to_file(f"{OUTPUT_DIR}/itineraries/itineraries.geojson", driver="GeoJSON")

# 6. Generate Google Maps URLs
print("\n6. Generating Google Maps URLs...")


def generate_google_maps_urls(segments, max_waypoints=20):
    base_url = "https://www.google.com/maps/dir/"
    urls = []

    for i in range(0, len(segments), max_waypoints - 1):
        waypoint_chunk = segments[i : i + max_waypoints]
        waypoints = [f"{seg['lat']},{seg['lon']}" for seg in waypoint_chunk]
        url = base_url + "/".join(waypoints)
        urls.append(urllib.parse.quote(url, safe=":/,"))

    return urls


itinerary_links = []
for itinerary in itineraries:
    itin_id = itinerary["itinerary_id"]
    urls = generate_google_maps_urls(itinerary["segments"])

    for idx, url in enumerate(urls):
        itinerary_links.append(
            {"itinerary_id": itin_id, "part": idx + 1, "google_maps_url": url}
        )

df_maps = pd.DataFrame(itinerary_links)
df_maps.to_csv(f"{OUTPUT_DIR}/itineraries/google_maps_itineraries.csv", index=False)
print(f"   Generated {len(df_maps)} Google Maps URLs")

osrm_urls = generate_osrm_urls(df_itineraries, 100)
df_osrm = pd.DataFrame(osrm_urls)
df_osrm.to_csv(f"{OUTPUT_DIR}/itineraries/osrm_itineraries.csv", index=False)
print(f"   Generated {len(df_osrm)} OSRM URLs")

# 7. Visualizations
print("\n7. Creating visualizations...")

fig, ax = plt.subplots(figsize=(14, 14))
edges.plot(ax=ax, linewidth=0.1, color="lightgray")
boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1)

colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(itineraries))))

for i, itinerary in enumerate(itineraries):
    points = [(s["lon"], s["lat"]) for s in itinerary["segments"]]
    if len(points) >= 2:
        xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.7,
        )
        ax.scatter(xs, ys, color=colors[i % len(colors)], s=10, alpha=0.5)

ax.set_title(f"Delhi NN Itineraries ({len(itineraries)} routes @ 20km)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visualizations/all_routes.png", dpi=150)
plt.close()

for itinerary in itineraries[:10]:
    itin_id = itinerary["itinerary_id"]
    points = [(s["lon"], s["lat"]) for s in itinerary["segments"]]

    if len(points) < 2:
        continue

    fig, ax = plt.subplots(figsize=(10, 10))

    xs, ys = zip(*points)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    buffer = 0.01

    bbox_edges = edges.cx[minx - buffer : maxx + buffer, miny - buffer : maxy + buffer]
    bbox_edges.plot(ax=ax, linewidth=0.3, color="lightgray")

    ax.plot(xs, ys, color="blue", linewidth=2, alpha=0.7)
    ax.scatter(xs, ys, color="red", s=30, zorder=5)

    ax.scatter(xs[0], ys[0], color="green", s=100, marker="^", zorder=6, label="Start")
    ax.scatter(xs[-1], ys[-1], color="purple", s=100, marker="s", zorder=6, label="End")

    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)
    ax.set_title(f"Itinerary {itin_id} ({len(points)} points)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/visualizations/route_{itin_id}.png", dpi=150)
    plt.close()

# 8. Create Interactive Folium Map
print("\n8. Creating interactive Folium map...")

try:
    import folium

    center_lat = (delhi_boundary.bounds[1] + delhi_boundary.bounds[3]) / 2
    center_lon = (delhi_boundary.bounds[0] + delhi_boundary.bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

    folium.GeoJson(
        boundary_gdf.__geo_interface__,
        style_function=lambda x: {
            "fillColor": "lightblue",
            "color": "blue",
            "weight": 2,
            "fillOpacity": 0.1,
        },
        name="Delhi Boundary",
    ).add_to(m)

    colors_list = [
        "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
        "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
        "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    ]

    for i, itinerary in enumerate(itineraries):
        itin_id = itinerary["itinerary_id"]
        color = colors_list[i % len(colors_list)]

        points = [(s["lat"], s["lon"]) for s in itinerary["segments"]]

        fg = folium.FeatureGroup(name=f"Itinerary {itin_id}")

        folium.PolyLine(points, color=color, weight=3, opacity=0.7).add_to(fg)

        for j, seg in enumerate(itinerary["segments"]):
            folium.CircleMarker(
                location=[seg["lat"], seg["lon"]],
                radius=3,
                color=color,
                fill=True,
                popup=f"Itinerary {itin_id}, Point {j + 1}",
            ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl().add_to(m)

    m.save(f"{OUTPUT_DIR}/visualizations/itineraries_map.html")
    print(f"   Saved interactive map to {OUTPUT_DIR}/visualizations/itineraries_map.html")

except ImportError:
    print("   Folium not installed. Skipping interactive map generation.")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Input sampled points: {INPUT_DIR}/sampled_points/sampled_points.csv")
print(f"Itineraries: {OUTPUT_DIR}/itineraries/itineraries.csv")
print(f"Google Maps links: {OUTPUT_DIR}/itineraries/google_maps_itineraries.csv")
print(f"OSRM links: {OUTPUT_DIR}/itineraries/osrm_itineraries.csv")
print(f"\nTotal itineraries: {len(itineraries)}")
print(f"Total sampled points: {len(df_sampled)}")
print(f"Max itinerary distance: {MAX_ITINERARY_DISTANCE / 1000} km")
print("\nComparison:")
print(f"  Delhi greedy (100km): 16 itineraries")
print(f"  Delhi NN (20km): {len(itineraries)} itineraries")
print("\nDone!")
