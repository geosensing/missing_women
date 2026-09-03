#!/usr/bin/env python3
"""
Mumbai Data Preparation Script

Converts the road segment CSV to standard GeoJSON format for visualization.
Note: In the CSV, start_lat/start_long columns are swapped (start_lat contains longitude).
"""

import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "sampling/mumbai/itineraries/itineraries-mumbai-city-osrm.csv"
OUTPUT_DIR = PROJECT_ROOT / "sampling/mumbai"


def fetch_mumbai_boundary():
    """Fetch Mumbai boundary from OpenStreetMap via Overpass API."""
    print("Fetching Mumbai boundary from OpenStreetMap...")

    import osmnx as ox

    queries = [
        "Greater Mumbai, Maharashtra, India",
        "Mumbai Suburban, Maharashtra, India",
        {"city": "Mumbai", "state": "Maharashtra", "country": "India"},
    ]

    for query in queries:
        try:
            print(f"  Trying: {query}")
            gdf = ox.geocode_to_gdf(query)
            if len(gdf) > 0:
                print(f"  Success with: {query}")
                return gdf.geometry.iloc[0]
        except Exception as e:
            print(f"  Failed: {e}")

    try:
        print("  Trying Nominatim API directly...")
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": "Mumbai, Maharashtra, India",
            "format": "json",
            "polygon_geojson": 1,
            "limit": 1,
        }
        headers = {"User-Agent": "MissingWomenResearch/1.0"}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data and "geojson" in data[0]:
            from shapely.geometry import shape

            geom = shape(data[0]["geojson"])
            print("  Success with Nominatim API")
            return geom
    except Exception as e:
        print(f"  Nominatim failed: {e}")

    return None


def create_boundary_from_data(df):
    """Create a bounding box boundary from the itinerary data."""
    print("Creating boundary from itinerary data extent...")

    lats = pd.concat([df["start_long"], df["end_long"]])
    lons = pd.concat([df["start_lat"], df["end_lat"]])

    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()

    buffer = 0.01

    from shapely.geometry import box

    return box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)


def convert_to_geojson(df):
    """Convert road segment CSV to GeoJSON with LineStrings per itinerary."""
    print("Converting CSV to GeoJSON...")

    features = []

    for itinerary_id in sorted(df["itinerary_id"].unique()):
        itin_df = df[df["itinerary_id"] == itinerary_id].copy()

        coords = []
        total_length = 0

        for _, row in itin_df.iterrows():
            start_lon = row["start_lat"]
            start_lat = row["start_long"]
            end_lon = row["end_lat"]
            end_lat = row["end_long"]

            if not coords:
                coords.append((start_lon, start_lat))
            coords.append((end_lon, end_lat))

            total_length += row["LENGTH_M"]

        if len(coords) >= 2:
            line = LineString(coords)

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "itinerary_id": int(itinerary_id),
                        "n_segments": len(itin_df),
                        "length_m": total_length,
                    },
                    "geometry": line.__geo_interface__,
                }
            )
            print(
                f"  Itinerary {itinerary_id}: {len(itin_df)} segments, {len(coords)} points, {total_length:.0f}m"
            )

    geojson = {"type": "FeatureCollection", "features": features}

    return geojson


def main():
    print("=" * 50)
    print("Mumbai Data Preparation")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR / "boundaries", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "itineraries", exist_ok=True)
    os.makedirs(OUTPUT_DIR / "visualizations", exist_ok=True)

    print(f"\nReading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Rows: {len(df)}")
    print(f"  Itineraries: {df['itinerary_id'].nunique()}")

    geojson = convert_to_geojson(df)

    output_geojson = OUTPUT_DIR / "itineraries/itineraries.geojson"
    with open(output_geojson, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"\nSaved GeoJSON: {output_geojson}")
    print(f"  Features: {len(geojson['features'])}")

    summary_data = []
    for feature in geojson["features"]:
        props = feature["properties"]
        summary_data.append(
            {
                "itinerary_id": props["itinerary_id"],
                "n_segments": props["n_segments"],
                "length_m": props["length_m"],
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_csv = OUTPUT_DIR / "itineraries/itineraries_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary CSV: {summary_csv}")

    boundary = fetch_mumbai_boundary()
    if boundary is None:
        boundary = create_boundary_from_data(df)

    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs="EPSG:4326")
    boundary_path = OUTPUT_DIR / "boundaries/mumbai_boundary.geojson"
    boundary_gdf.to_file(boundary_path, driver="GeoJSON")
    print(f"\nSaved boundary: {boundary_path}")

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"\nOutput files:")
    print(f"  - {output_geojson}")
    print(f"  - {summary_csv}")
    print(f"  - {boundary_path}")


if __name__ == "__main__":
    main()
