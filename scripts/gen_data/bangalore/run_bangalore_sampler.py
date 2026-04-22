#!/usr/bin/env python3
"""
Bangalore Data Collection Sampler

This script generates optimized data collection itineraries for Bangalore by:
1. Extracting Bangalore district boundary from GADM
2. Downloading the street network from OpenStreetMap via OSMnx
3. Sampling random points along streets
4. Creating optimized routes using OSRM distance matrix
5. Generating visualizations and Google Maps links
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import math
import requests
import urllib.parse
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, mapping
from shapely.ops import unary_union
from scipy.spatial.distance import pdist, squareform
import osmnx as ox

# Configuration
OUTPUT_DIR = "../../sampling/bangalore"
GADM_PATH = "../../maps/gadm41_IND_2.json"
N_SAMPLE_POINTS = 1000
MAX_ITINERARY_DISTANCE = 20000  # 20 km in meters
SEGMENT_LENGTH = 500  # meters
MAX_DISTANCE_MATRIX_SIZE = 100

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create output directories
os.makedirs(f"{OUTPUT_DIR}/boundaries", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/network", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/sampled_points", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/itineraries", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/visualizations", exist_ok=True)

print("=" * 50)
print("Bangalore Data Collection Sampler")
print("=" * 50)

# 1. Extract Bangalore Boundary from GADM
print("\n1. Extracting Bangalore boundary from GADM...")
with open(GADM_PATH) as f:
    gadm_data = json.load(f)

bangalore_features = [
    f for f in gadm_data['features']
    if f['properties']['NAME_2'] == 'Bangalore'
]

print(f"   Found {len(bangalore_features)} Bangalore feature(s)")

bangalore_gdf = gpd.GeoDataFrame.from_features(bangalore_features, crs="EPSG:4326")
bangalore_boundary = bangalore_gdf.union_all()

boundary_gdf = gpd.GeoDataFrame(geometry=[bangalore_boundary], crs="EPSG:4326")
boundary_gdf.to_file(f"{OUTPUT_DIR}/boundaries/bangalore_boundary.geojson", driver="GeoJSON")

fig, ax = plt.subplots(figsize=(10, 10))
bangalore_gdf.plot(ax=ax, facecolor='lightblue', edgecolor='blue', linewidth=2)
ax.set_title("Bangalore District Boundary")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visualizations/boundary.png", dpi=150)
plt.close()

print(f"   Boundary bounds: {bangalore_boundary.bounds}")

# 2. Download Street Network via OSMnx
print("\n2. Downloading street network via OSMnx...")
minx, miny, maxx, maxy = bangalore_boundary.bounds
print(f"   Bounding box: N={maxy:.4f}, S={miny:.4f}, E={maxx:.4f}, W={minx:.4f}")

G = ox.graph_from_polygon(bangalore_boundary, network_type='drive')
print(f"   Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")

ox.save_graphml(G, f"{OUTPUT_DIR}/network/street_network.graphml")

nodes, edges = ox.graph_to_gdfs(G)
edges.to_file(f"{OUTPUT_DIR}/network/streets.geojson", driver="GeoJSON")

fig, ax = plt.subplots(figsize=(12, 12))
edges.plot(ax=ax, linewidth=0.3, color='gray')
bangalore_gdf.boundary.plot(ax=ax, color='blue', linewidth=2)
ax.set_title("Bangalore Street Network")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visualizations/street_network.png", dpi=150)
plt.close()

# 3. Sample Points Along Streets
print("\n3. Sampling points along streets...")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def split_road_segment(geom, segment_length=SEGMENT_LENGTH):
    if geom.geom_type != 'LineString':
        return []

    coords = list(geom.coords)
    if len(coords) < 2:
        return []

    points = []

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        segment_dist = haversine(lat1, lon1, lat2, lon2)

        if segment_dist <= segment_length:
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            points.append((mid_lon, mid_lat))
        else:
            num_segments = int(np.ceil(segment_dist / segment_length))
            for j in range(num_segments):
                frac = (j + 0.5) / num_segments
                interp_lat = lat1 + frac * (lat2 - lat1)
                interp_lon = lon1 + frac * (lon2 - lon1)
                points.append((interp_lon, interp_lat))

    return points

all_segment_points = []
road_types_of_interest = ['residential', 'primary', 'secondary', 'tertiary', 'trunk', 'unclassified']

for idx, row in edges.iterrows():
    highway = row.get('highway', '')
    if isinstance(highway, list):
        highway = highway[0] if highway else ''

    if highway in road_types_of_interest:
        points = split_road_segment(row.geometry)
        for lon, lat in points:
            all_segment_points.append({
                'lon': lon,
                'lat': lat,
                'highway': highway
            })

print(f"   Total segment points available: {len(all_segment_points)}")

n_samples = min(N_SAMPLE_POINTS, len(all_segment_points))
random.seed(42)
sampled_indices = random.sample(range(len(all_segment_points)), n_samples)
sampled_points = [all_segment_points[i] for i in sampled_indices]

df_sampled = pd.DataFrame(sampled_points)
df_sampled['point_id'] = range(len(df_sampled))

geometry = [Point(row['lon'], row['lat']) for _, row in df_sampled.iterrows()]
sampled_gdf = gpd.GeoDataFrame(df_sampled, geometry=geometry, crs="EPSG:4326")

sampled_gdf.to_file(f"{OUTPUT_DIR}/sampled_points/sampled_points.geojson", driver="GeoJSON")
df_sampled.to_csv(f"{OUTPUT_DIR}/sampled_points/sampled_points.csv", index=False)

print(f"   Sampled {len(sampled_gdf)} points")

fig, ax = plt.subplots(figsize=(12, 12))
edges.plot(ax=ax, linewidth=0.2, color='lightgray')
sampled_gdf.plot(ax=ax, color='red', markersize=5, alpha=0.6)
bangalore_gdf.boundary.plot(ax=ax, color='blue', linewidth=2)
ax.set_title(f"Sampled Points ({len(sampled_gdf)} points)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visualizations/sampled_points.png", dpi=150)
plt.close()

# 4. Generate OSRM Distance Matrix and Optimize Routes
print("\n4. Computing OSRM distance matrix...")

def osrm_distance_matrix(X, Y=None, chunksize=MAX_DISTANCE_MATRIX_SIZE):
    api_base = 'http://router.project-osrm.org/table/v1/driving/'

    n_X = len(X)
    if Y is None:
        Y = X
    n_Y = len(Y)
    m = chunksize * 1.0
    Xsplits = math.ceil(n_X / m)
    Ysplits = math.ceil(n_Y / m)
    o = None
    count = 0
    for s in np.array_split(X, Xsplits):
        c = None
        for d in np.array_split(Y, Ysplits):
            a = ';'.join([','.join([str(x) for x in b]) for b in (list(s) + list(d))])
            sources = ';'.join([str(k) for k in range(0, len(s))])
            destinations = ';'.join([str(k) for k in range(len(s), len(s) + len(d))])
            url = (api_base + a + '?annotations=distance,duration&sources=' + sources +
                   '&destinations=' + destinations)
            count += 1
            r = requests.get(url)
            if r.status_code != 200:
                print(f"   OSRM Table API request error: {r.text}")
                break
            dm = r.json()['distances']
            arr = np.array(dm)
            if c is None:
                c = arr
            else:
                c = np.concatenate((c, arr), axis=1)
        if o is None:
            o = c
        else:
            o = np.concatenate((o, c), axis=0)
        print(f"   Current OSRM API requests count: {count}")
    print(f"   Total API requests: {count}")
    return o

coords = df_sampled[['lon', 'lat']].to_numpy()
print(f"   Computing distance matrix for {len(coords)} points...")
distance_matrix = osrm_distance_matrix(coords)

def nearest_neighbor_sort(distance_matrix):
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

print("   Sorting points by nearest neighbor...")
sorted_order = nearest_neighbor_sort(distance_matrix)
df_sorted = df_sampled.iloc[sorted_order].reset_index(drop=True)

df_distance_matrix = pd.DataFrame(distance_matrix,
                                   columns=df_sampled.point_id,
                                   index=df_sampled.point_id)

distances = []
for i in range(len(df_sorted) - 1):
    loc1 = df_sorted.loc[i, 'point_id']
    loc2 = df_sorted.loc[i + 1, 'point_id']
    distance = df_distance_matrix.loc[loc1, loc2]
    if distance is None:
        distance = 0
    distances.append(distance)

df_sorted['osrm_travel_distance'] = [None] + distances

# 5. Create Itineraries
print("\n5. Creating itineraries...")

itineraries = []
current_itinerary = []
current_distance = 0
itinerary_id = 1

for i, row in df_sorted.iterrows():
    travel_distance = row["osrm_travel_distance"] if i > 0 and row["osrm_travel_distance"] else 0

    if current_distance + travel_distance > MAX_ITINERARY_DISTANCE and len(current_itinerary) > 0:
        itineraries.append({"itinerary_id": itinerary_id, "segments": current_itinerary})
        current_itinerary = []
        current_distance = 0
        itinerary_id += 1

    current_itinerary.append(row.to_dict())
    current_distance += travel_distance

if current_itinerary:
    itineraries.append({"itinerary_id": itinerary_id, "segments": current_itinerary})

print(f"   Created {len(itineraries)} itineraries")

flat_data = []
for itinerary in itineraries:
    itinerary_id = itinerary["itinerary_id"]
    for segment in itinerary["segments"]:
        segment["itinerary_id"] = itinerary_id
        flat_data.append(segment)

df_itineraries = pd.DataFrame(flat_data)
df_itineraries.to_csv(f"{OUTPUT_DIR}/itineraries/itineraries.csv", index=False)

summary_data = []
for itinerary in itineraries:
    itin_id = itinerary["itinerary_id"]
    n_points = len(itinerary["segments"])
    total_dist = sum(s.get('osrm_travel_distance', 0) or 0 for s in itinerary["segments"])
    summary_data.append({
        'itinerary_id': itin_id,
        'n_points': n_points,
        'total_distance_m': total_dist
    })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(f"{OUTPUT_DIR}/itineraries/itineraries_summary.csv", index=False)
print(f"   Summary:\n{df_summary.to_string()}")

itin_geometry = []
for itinerary in itineraries:
    points = [(s['lon'], s['lat']) for s in itinerary['segments']]
    if len(points) >= 2:
        line = LineString(points)
        itin_geometry.append({
            'itinerary_id': itinerary['itinerary_id'],
            'geometry': line
        })

itin_gdf = gpd.GeoDataFrame(itin_geometry, crs="EPSG:4326")
itin_gdf.to_file(f"{OUTPUT_DIR}/itineraries/itineraries.geojson", driver="GeoJSON")

# 6. Generate Google Maps URLs
print("\n6. Generating Google Maps URLs...")

def generate_google_maps_urls(segments, max_waypoints=20):
    base_url = "https://www.google.com/maps/dir/"
    urls = []

    for i in range(0, len(segments), max_waypoints - 1):
        waypoint_chunk = segments[i:i + max_waypoints]

        waypoints = [f"{seg['lat']},{seg['lon']}" for seg in waypoint_chunk]
        url = base_url + "/".join(waypoints)
        urls.append(urllib.parse.quote(url, safe=':/,'))

    return urls

itinerary_links = []
for itinerary in itineraries:
    itinerary_id = itinerary["itinerary_id"]
    urls = generate_google_maps_urls(itinerary["segments"])

    for idx, url in enumerate(urls):
        itinerary_links.append({"itinerary_id": itinerary_id, "part": idx + 1, "google_maps_url": url})

df_maps = pd.DataFrame(itinerary_links)
df_maps.to_csv(f"{OUTPUT_DIR}/itineraries/google_maps_itineraries.csv", index=False)
print(f"   Generated {len(df_maps)} Google Maps URLs")

OSRM_BASE_URL = "https://map.project-osrm.org/?z=15"

def generate_osrm_urls(df, max_waypoints=20):
    urls = []
    for itinerary_id, group in df.groupby("itinerary_id"):
        waypoints = group[["lon", "lat"]].values.tolist()

        part = 1
        for i in range(0, len(waypoints), max_waypoints):
            chunk = waypoints[i:i+max_waypoints]
            center_lon, center_lat = chunk[0]
            loc_params = "&".join([f"loc={lat},{lon}" for lon, lat in chunk])
            url = f"{OSRM_BASE_URL}&center={center_lat},{center_lon}&{loc_params}&hl=en&alt=0&srv=1"
            urls.append({"itinerary_id": itinerary_id, "part": part, "osrm_maps_url": url})
            part += 1

    return urls

osrm_urls = generate_osrm_urls(df_itineraries, 100)
df_osrm = pd.DataFrame(osrm_urls)
df_osrm.to_csv(f"{OUTPUT_DIR}/itineraries/osrm_itineraries.csv", index=False)
print(f"   Generated {len(df_osrm)} OSRM URLs")

# 7. Visualizations
print("\n7. Creating visualizations...")

fig, ax = plt.subplots(figsize=(14, 14))
edges.plot(ax=ax, linewidth=0.1, color='lightgray')
bangalore_gdf.boundary.plot(ax=ax, color='black', linewidth=1)

colors = plt.cm.tab20(np.linspace(0, 1, len(itineraries)))

for i, itinerary in enumerate(itineraries):
    points = [(s['lon'], s['lat']) for s in itinerary['segments']]
    if len(points) >= 2:
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=colors[i], linewidth=1.5, alpha=0.7,
                label=f"Itinerary {itinerary['itinerary_id']}")
        ax.scatter(xs, ys, color=colors[i], s=10, alpha=0.5)

ax.set_title(f"All Itineraries ({len(itineraries)} routes)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visualizations/all_routes.png", dpi=150)
plt.close()

for itinerary in itineraries[:10]:
    itin_id = itinerary['itinerary_id']
    points = [(s['lon'], s['lat']) for s in itinerary['segments']]

    if len(points) < 2:
        continue

    fig, ax = plt.subplots(figsize=(10, 10))

    xs, ys = zip(*points)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    buffer = 0.01

    bbox_edges = edges.cx[minx-buffer:maxx+buffer, miny-buffer:maxy+buffer]
    bbox_edges.plot(ax=ax, linewidth=0.3, color='lightgray')

    ax.plot(xs, ys, color='blue', linewidth=2, alpha=0.7)
    ax.scatter(xs, ys, color='red', s=30, zorder=5)

    ax.scatter(xs[0], ys[0], color='green', s=100, marker='^', zorder=6, label='Start')
    ax.scatter(xs[-1], ys[-1], color='purple', s=100, marker='s', zorder=6, label='End')

    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)
    ax.set_title(f"Itinerary {itin_id} ({len(points)} points)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/visualizations/route_{itin_id}.png", dpi=150)
    plt.close()

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Boundary: {OUTPUT_DIR}/boundaries/bangalore_boundary.geojson")
print(f"Street network: {OUTPUT_DIR}/network/street_network.graphml")
print(f"Sampled points: {OUTPUT_DIR}/sampled_points/sampled_points.geojson")
print(f"Itineraries: {OUTPUT_DIR}/itineraries/itineraries.csv")
print(f"Google Maps links: {OUTPUT_DIR}/itineraries/google_maps_itineraries.csv")
print(f"OSRM links: {OUTPUT_DIR}/itineraries/osrm_itineraries.csv")
print(f"\nTotal itineraries: {len(itineraries)}")
print(f"Total sampled points: {len(df_sampled)}")
print("\nDone!")
