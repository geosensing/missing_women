#!/usr/bin/env python3
"""Generate interactive Folium maps for itineraries visualization."""

import argparse
from pathlib import Path

import folium
import geopandas as gpd

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


def get_centroid(gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    """Get centroid of GeoDataFrame."""
    bounds = gdf.total_bounds
    return (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2


def create_itinerary_map(
    itineraries_path: Path,
    boundary_path: Path | None = None,
    sampled_points_path: Path | None = None,
    output_path: Path | None = None,
) -> folium.Map:
    """Create an interactive map with itineraries."""
    itineraries_gdf = gpd.read_file(itineraries_path)
    center = get_centroid(itineraries_gdf)

    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    if boundary_path and boundary_path.exists():
        boundary_gdf = gpd.read_file(boundary_path)
        folium.GeoJson(
            boundary_gdf,
            name="Boundary",
            style_function=lambda _: {
                "fillColor": "transparent",
                "color": "#333",
                "weight": 2,
                "dashArray": "5, 5",
            },
        ).add_to(m)

    for i, (_, row) in enumerate(itineraries_gdf.iterrows()):
        color = COLORS[i % len(COLORS)]
        itinerary_id = row.get("itinerary_id", row.get("route_id", i + 1))

        popup_content = f"<b>Route {itinerary_id}</b>"
        if "length_m" in row:
            popup_content += f"<br>Length: {row['length_m']:.0f} m"
        if "travel_time_min" in row:
            popup_content += f"<br>Time: {row['travel_time_min']:.1f} min"
        if "n_points" in row:
            popup_content += f"<br>Points: {row['n_points']}"

        geom = row.geometry
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                coords = [(c[1], c[0]) for c in line.coords]
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=folium.Popup(popup_content, max_width=200),
                    tooltip=f"Route {itinerary_id}",
                ).add_to(m)
        else:
            coords = [(c[1], c[0]) for c in geom.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=folium.Popup(popup_content, max_width=200),
                tooltip=f"Route {itinerary_id}",
            ).add_to(m)

    if sampled_points_path and sampled_points_path.exists():
        points_gdf = gpd.read_file(sampled_points_path)
        points_group = folium.FeatureGroup(name="Sampled Points", show=False)
        for _, row in points_gdf.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=2,
                color="#333",
                fill=True,
                fillOpacity=0.7,
            ).add_to(points_group)
        points_group.add_to(m)

    folium.LayerControl().add_to(m)

    if output_path:
        m.save(str(output_path))
        print(f"Saved map to {output_path}")

    return m


def main():
    parser = argparse.ArgumentParser(description="Generate itinerary maps")
    parser.add_argument(
        "--city",
        choices=["bangalore", "navi_mumbai", "all"],
        default="all",
        help="City to generate map for",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent / "sampling"

    cities = {
        "bangalore": {
            "itineraries": base_dir / "bangalore/itineraries/itineraries.geojson",
            "boundary": base_dir / "bangalore/boundaries/bangalore_boundary.geojson",
            "sampled_points": base_dir / "bangalore/sampled_points/sampled_points.geojson",
            "output": base_dir / "bangalore/visualizations/itineraries_map.html",
        },
        "navi_mumbai": {
            "itineraries": base_dir / "navi_mumbai/itineraries/itineraries.geojson",
            "boundary": base_dir / "navi_mumbai/boundaries/boundary.geojson",
            "sampled_points": base_dir / "navi_mumbai/sampled_points/sampled_points.geojson",
            "output": base_dir / "navi_mumbai/visualizations/itineraries_map.html",
        },
    }

    selected = cities.keys() if args.city == "all" else [args.city]

    for city in selected:
        config = cities[city]
        if not config["itineraries"].exists():
            print(f"Skipping {city}: itineraries file not found")
            continue

        print(f"Generating map for {city}...")
        create_itinerary_map(
            itineraries_path=config["itineraries"],
            boundary_path=config["boundary"],
            sampled_points_path=config["sampled_points"],
            output_path=config["output"],
        )


if __name__ == "__main__":
    main()
