#!/usr/bin/env python3
"""
Common utilities for city samplers.

Shared functions for:
- Haversine distance calculation
- Road segment splitting
- OSRM distance matrix computation
- URL generation (Google Maps, OSRM)
- Itinerary creation and saving
- Visualization generation
"""

import math
import urllib.parse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString

from allocator.core.itinerary import greedy_grow_itineraries, tsp_optimize_route


def haversine(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in meters."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def split_road_segment(geom, segment_length=500):
    """Split a road geometry into points at specified interval."""
    if geom.geom_type != "LineString":
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


def osrm_distance_matrix(X, Y=None, chunksize=100):
    """Compute OSRM distance matrix in chunks."""
    api_base = "http://router.project-osrm.org/table/v1/driving/"

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
            a = ";".join([",".join([str(x) for x in b]) for b in (list(s) + list(d))])
            sources = ";".join([str(k) for k in range(0, len(s))])
            destinations = ";".join([str(k) for k in range(len(s), len(s) + len(d))])
            url = (
                api_base
                + a
                + "?annotations=distance,duration&sources="
                + sources
                + "&destinations="
                + destinations
            )
            count += 1
            r = requests.get(url)
            if r.status_code != 200:
                print(f"   OSRM Table API request error: {r.text}")
                break
            dm = r.json()["distances"]
            arr = np.array(dm)
            if c is None:
                c = arr
            else:
                c = np.concatenate((c, arr), axis=1)
        if o is None:
            o = c
        else:
            o = np.concatenate((o, c), axis=0)
    print(f"   OSRM API requests: {count}")
    return o


def generate_google_maps_urls(segments, max_waypoints=20):
    """Generate Google Maps direction URLs for an itinerary."""
    base_url = "https://www.google.com/maps/dir/"
    urls = []

    for i in range(0, len(segments), max_waypoints - 1):
        waypoint_chunk = segments[i : i + max_waypoints]
        waypoints = [f"{seg['lat']},{seg['lon']}" for seg in waypoint_chunk]
        url = base_url + "/".join(waypoints)
        urls.append(urllib.parse.quote(url, safe=":/,"))

    return urls


def generate_osrm_urls(df, max_waypoints=20):
    """Generate OSRM map URLs for itineraries."""
    OSRM_BASE_URL = "https://map.project-osrm.org/?z=15"
    urls = []
    for itinerary_id, group in df.groupby("itinerary_id"):
        waypoints = group[["lon", "lat"]].values.tolist()

        part = 1
        for i in range(0, len(waypoints), max_waypoints):
            chunk = waypoints[i : i + max_waypoints]
            center_lon, center_lat = chunk[0]
            loc_params = "&".join([f"loc={lat},{lon}" for lon, lat in chunk])
            url = f"{OSRM_BASE_URL}&center={center_lat},{center_lon}&{loc_params}&hl=en&alt=0&srv=1"
            urls.append({"itinerary_id": itinerary_id, "part": part, "osrm_maps_url": url})
            part += 1

    return urls


def haversine_distance_matrix(coords):
    """Compute haversine distance matrix (fast, no API calls)."""
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine(coords[i, 1], coords[i, 0], coords[j, 1], coords[j, 0])
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix


def create_itineraries(df_sampled, distance_matrix, max_distance, rng):
    """
    Create itineraries using allocator.greedy_grow_itineraries + TSP optimization.

    Returns:
        all_itineraries: List of dicts with itinerary_id and segments
        optimized_indices: List of optimized point indices per itinerary
        itinerary_distances: List of distances per itinerary
    """
    print("\n   Creating itineraries using greedy_grow_itineraries...")

    itinerary_indices, itinerary_distances = greedy_grow_itineraries(
        distance_matrix,
        max_distance=max_distance,
        start_method="furthest",
        rng=rng,
    )

    print(f"   Created {len(itinerary_indices)} itineraries")

    print("   Applying TSP optimization...")
    optimized_itineraries = []
    for indices in itinerary_indices:
        if len(indices) > 2:
            optimized = tsp_optimize_route(indices, distance_matrix)
        else:
            optimized = indices
        optimized_itineraries.append(optimized)

    all_itineraries = []
    for itin_idx, indices in enumerate(optimized_itineraries):
        segments = []
        for point_idx in indices:
            row = df_sampled.iloc[point_idx]
            segments.append({
                "point_id": int(row["point_id"]),
                "lon": row["lon"],
                "lat": row["lat"],
                "highway": row["highway"],
            })
        all_itineraries.append({"itinerary_id": itin_idx + 1, "segments": segments})

    return all_itineraries, optimized_itineraries, itinerary_distances


def save_itineraries(all_itineraries, optimized_indices, distance_matrix, df_sampled, output_dir):
    """Save itineraries to CSV and GeoJSON files."""
    flat_data = []
    for itinerary in all_itineraries:
        itin_id = itinerary["itinerary_id"]
        prev_point_id = None
        for seg_idx, segment in enumerate(itinerary["segments"]):
            segment_copy = segment.copy()
            segment_copy["itinerary_id"] = itin_id
            if seg_idx > 0 and prev_point_id is not None:
                curr_point_id = segment_copy["point_id"]
                prev_idx = df_sampled[df_sampled["point_id"] == prev_point_id].index[0]
                curr_idx = df_sampled[df_sampled["point_id"] == curr_point_id].index[0]
                segment_copy["osrm_travel_distance"] = distance_matrix[prev_idx, curr_idx]
            else:
                segment_copy["osrm_travel_distance"] = None
            prev_point_id = segment_copy["point_id"]
            flat_data.append(segment_copy)

    df_itineraries = pd.DataFrame(flat_data)
    df_itineraries.to_csv(f"{output_dir}/itineraries/itineraries.csv", index=False)

    summary_data = []
    for itin_idx, itinerary in enumerate(all_itineraries):
        itin_id = itinerary["itinerary_id"]
        n_points = len(itinerary["segments"])

        indices = optimized_indices[itin_idx]
        actual_dist = 0.0
        for i in range(len(indices) - 1):
            actual_dist += distance_matrix[indices[i], indices[i + 1]]

        summary_data.append({
            "itinerary_id": itin_id,
            "n_points": n_points,
            "total_distance_m": actual_dist,
            "total_distance_km": actual_dist / 1000,
        })

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(f"{output_dir}/itineraries/itineraries_summary.csv", index=False)
    print(f"   Summary:\n{df_summary.to_string()}")

    itin_geometry = []
    for itinerary in all_itineraries:
        points = [(s["lon"], s["lat"]) for s in itinerary["segments"]]
        if len(points) >= 2:
            line = LineString(points)
            itin_geometry.append({"itinerary_id": itinerary["itinerary_id"], "geometry": line})

    itin_gdf = gpd.GeoDataFrame(itin_geometry, crs="EPSG:4326")
    itin_gdf.to_file(f"{output_dir}/itineraries/itineraries.geojson", driver="GeoJSON")

    return df_itineraries, df_summary


def save_map_urls(all_itineraries, df_itineraries, output_dir):
    """Generate and save Google Maps and OSRM URLs."""
    itinerary_links = []
    for itinerary in all_itineraries:
        itin_id = itinerary["itinerary_id"]
        urls = generate_google_maps_urls(itinerary["segments"])

        for idx, url in enumerate(urls):
            itinerary_links.append(
                {"itinerary_id": itin_id, "part": idx + 1, "google_maps_url": url}
            )

    df_maps = pd.DataFrame(itinerary_links)
    df_maps.to_csv(f"{output_dir}/itineraries/google_maps_itineraries.csv", index=False)
    print(f"   Generated {len(df_maps)} Google Maps URLs")

    osrm_urls = generate_osrm_urls(df_itineraries, 100)
    df_osrm = pd.DataFrame(osrm_urls)
    df_osrm.to_csv(f"{output_dir}/itineraries/osrm_itineraries.csv", index=False)
    print(f"   Generated {len(df_osrm)} OSRM URLs")


def create_static_visualizations(all_itineraries, edges, boundary_gdf, output_dir):
    """Create static matplotlib visualizations."""
    fig, ax = plt.subplots(figsize=(14, 14))
    edges.plot(ax=ax, linewidth=0.1, color="lightgray")
    boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=1)

    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(all_itineraries))))

    for i, itinerary in enumerate(all_itineraries):
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

    ax.set_title(f"All Itineraries ({len(all_itineraries)} routes)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualizations/all_routes.png", dpi=150)
    plt.close()

    for itinerary in all_itineraries[:10]:
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
        plt.savefig(f"{output_dir}/visualizations/route_{itin_id}.png", dpi=150)
        plt.close()


def create_folium_map(all_itineraries, boundary_gdf, boundary, output_dir, city_name):
    """Create interactive Folium map."""
    try:
        import folium

        center_lat = (boundary.bounds[1] + boundary.bounds[3]) / 2
        center_lon = (boundary.bounds[0] + boundary.bounds[2]) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

        folium.GeoJson(
            boundary_gdf.__geo_interface__,
            style_function=lambda x: {
                "fillColor": "lightblue",
                "color": "blue",
                "weight": 2,
                "fillOpacity": 0.1,
            },
            name=f"{city_name} Boundary",
        ).add_to(m)

        colors_list = [
            "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
            "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
        ]

        for i, itinerary in enumerate(all_itineraries):
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

        m.save(f"{output_dir}/visualizations/itineraries_map.html")
        print(f"   Saved interactive map to {output_dir}/visualizations/itineraries_map.html")

    except ImportError:
        print("   Folium not installed. Skipping interactive map generation.")
        print("   Install with: pip install folium")
