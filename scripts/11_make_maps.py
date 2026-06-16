#!/usr/bin/env python3
"""
Generate maps for the Streetscope project.

Creates both interactive HTML maps (Folium) and static PDF maps (matplotlib + contextily).

Maps are per-city (the cities span the whole country, so a single combined map is
not legible), plus one all-India overview that anchors where data was collected.

Outputs (in figs/), one set per city plus an overview:
  - map_locations_{city}.html   Interactive map of that city's collection points
  - map_locations_{city}.pdf    Static PDF with basemap tiles
  - map_sexratio_{city}.html    Interactive map colored by proportion women
  - map_sexratio_{city}.pdf     Static PDF colored by proportion women
  - map_overview_india.pdf      All cities as labelled points on an India basemap

Usage:
    python scripts/11_make_maps.py --cities mumbai,navi_mumbai,bangalore,delhi
"""

from __future__ import annotations

import argparse
from pathlib import Path

import contextily as cx
import folium
import geopandas as gpd
import matplotlib
from folium.plugins import MarkerCluster

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "mumbai": "#2166ac",
    "navi_mumbai": "#b2182b",
    "bangalore": "#1b7837",
    "delhi": "#762a83",
}

CITY_LABELS = {
    "mumbai": "Mumbai",
    "navi_mumbai": "Navi Mumbai",
    "bangalore": "Bangalore",
    "delhi": "Delhi",
}

CITY_BOUNDS = {
    "mumbai": {"lat": (18.85, 19.30), "lon": (72.75, 73.05)},
    "navi_mumbai": {"lat": (18.90, 19.25), "lon": (72.80, 73.10)},
    "bangalore": {"lat": (12.85, 13.15), "lon": (77.45, 77.75)},
    "delhi": {"lat": (28.40, 28.90), "lon": (76.70, 77.50)},
}


def load_city_data(cities: list[str]) -> pd.DataFrame:
    """Load analysis_data.parquet for each city."""
    dfs = []
    for city in cities:
        path = DATA / city / "analysis_data.parquet"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        df = pd.read_parquet(path)
        df["city"] = city
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No data found for any city")
    return pd.concat(dfs, ignore_index=True)


def filter_valid_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with valid GPS coordinates within city bounds."""
    geo = df.dropna(subset=["gps_lat", "gps_lon"]).copy()

    valid_rows = []
    for city in geo["city"].unique():
        bounds = CITY_BOUNDS.get(city)
        if bounds is None:
            continue
        mask = (
            (geo["city"] == city)
            & (geo["gps_lat"] >= bounds["lat"][0])
            & (geo["gps_lat"] <= bounds["lat"][1])
            & (geo["gps_lon"] >= bounds["lon"][0])
            & (geo["gps_lon"] <= bounds["lon"][1])
        )
        valid_rows.append(geo[mask])

    if not valid_rows:
        return pd.DataFrame()

    return pd.concat(valid_rows, ignore_index=True)


def make_locations_map(geo: pd.DataFrame, city: str) -> None:
    """Interactive map of one city's data collection points."""
    if len(geo) == 0:
        print(f"  WARNING: no valid GPS for {city} locations map")
        return

    color = COLORS.get(city, "#666666")
    label = CITY_LABELS.get(city, city)

    m = folium.Map(
        location=[geo["gps_lat"].mean(), geo["gps_lon"].mean()],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    cluster = MarkerCluster(name=f"{label} (n={len(geo):,})")
    for _, row in geo.iterrows():
        popup_text = f"""
        <b>{label}</b><br>
        Video: {row.get("base_video_id", "N/A")}<br>
        Frame: {row.get("frame_number", "N/A")}<br>
        People: {row.get("total_people", "N/A")}<br>
        Prop women: {row.get("prop_women", 0):.2f}
        """
        folium.CircleMarker(
            location=[row["gps_lat"], row["gps_lon"]],
            radius=5,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(cluster)
    cluster.add_to(m)
    folium.LayerControl().add_to(m)

    out_path = FIGS / f"map_locations_{city}.html"
    m.save(str(out_path))
    print(f"  -> {out_path.name} ({len(geo):,} points)")


def get_color_for_prop(prop: float) -> str:
    """Return color from red (0) to green (0.5+) for proportion women."""
    if pd.isna(prop):
        return "#808080"
    prop = min(prop, 0.5)
    ratio = prop / 0.5
    r = int(255 * (1 - ratio))
    g = int(255 * ratio)
    return f"#{r:02x}{g:02x}00"


def make_sex_ratio_map(geo: pd.DataFrame, city: str) -> None:
    """Interactive map of one city's points colored by proportion women."""
    geo = geo[geo["total_people"] > 0].copy()
    if len(geo) == 0:
        print(f"  WARNING: no valid GPS for {city} sex ratio map")
        return

    label = CITY_LABELS.get(city, city)
    m = folium.Map(
        location=[geo["gps_lat"].mean(), geo["gps_lon"].mean()],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    for _, row in geo.iterrows():
        prop = row.get("prop_women", 0)
        color = get_color_for_prop(prop)
        popup_text = f"""
        <b>{label}</b><br>
        Video: {row.get("base_video_id", "N/A")}<br>
        People: {row.get("total_people", "N/A")}<br>
        Women: {row.get("total_women", "N/A")}<br>
        <b>Prop women: {prop:.2f}</b>
        """
        folium.CircleMarker(
            location=[row["gps_lat"], row["gps_lon"]],
            radius=6,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid grey; font-size: 12px;">
        <b>Proportion Women</b><br>
        <i style="background: #ff0000; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 0.00<br>
        <i style="background: #ffff00; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 0.25<br>
        <i style="background: #00ff00; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 0.50+
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = FIGS / f"map_sexratio_{city}.html"
    m.save(str(out_path))
    print(f"  -> {out_path.name} ({len(geo):,} points)")


def make_locations_pdf(geo: pd.DataFrame, city: str) -> None:
    """PDF map of one city's collection points with basemap tiles."""
    if len(geo) == 0:
        print(f"  WARNING: no valid GPS for {city} locations PDF")
        return

    color = COLORS.get(city, "#666666")
    label = CITY_LABELS.get(city, city)
    gdf = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo["gps_lon"], geo["gps_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color=color, markersize=8, alpha=0.5, label=f"{label} (n={len(gdf):,})")
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white")
    ax.set_title(f"{label}: Data Collection Locations", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out_path = FIGS / f"map_locations_{city}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} ({len(geo):,} points)")


def make_sex_ratio_pdf(geo: pd.DataFrame, city: str) -> None:
    """PDF map of one city's points colored by proportion women."""
    geo = geo[geo["total_people"] > 0].copy()
    if len(geo) == 0:
        print(f"  WARNING: no valid GPS for {city} sex ratio PDF")
        return

    label = CITY_LABELS.get(city, city)
    gdf = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo["gps_lon"], geo["gps_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    cmap = LinearSegmentedColormap.from_list("sex_ratio", ["#d73027", "#fee08b", "#1a9850"])

    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf["prop_women"].clip(0, 0.5),
        cmap=cmap,
        vmin=0,
        vmax=0.5,
        s=15,
        alpha=0.7,
        edgecolors="none",
    )
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Proportion Women", fontsize=10)
    cbar.set_ticks([0, 0.125, 0.25, 0.375, 0.5])
    cbar.set_ticklabels(["0.00", "0.12", "0.25", "0.38", "0.50+"])

    ax.set_axis_off()
    ax.set_title(f"{label}: Proportion Women by Location", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out_path = FIGS / f"map_sexratio_{city}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} ({len(geo):,} points)")


def make_overview_pdf(df: pd.DataFrame) -> None:
    """All-India overview: one labelled point per city, sized by sample count."""
    geo = filter_valid_gps(df)
    if len(geo) == 0:
        print("  WARNING: no valid GPS for overview map")
        return

    agg = (
        geo.groupby("city")
        .agg(lat=("gps_lat", "mean"), lon=("gps_lon", "mean"), n=("gps_lat", "size"))
        .reset_index()
    )
    gdf = gpd.GeoDataFrame(
        agg,
        geometry=gpd.points_from_xy(agg["lon"], agg["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8, 10))
    for _, r in gdf.iterrows():
        color = COLORS.get(r["city"], "#666666")
        label = CITY_LABELS.get(r["city"], r["city"])
        ax.scatter(
            r.geometry.x,
            r.geometry.y,
            s=140,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=3,
            label=f"{label} (n={int(r['n']):,})",
        )
        ax.annotate(
            label,
            (r.geometry.x, r.geometry.y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    # Pad the extent so the surrounding country provides geographic context.
    ax.margins(0.25)
    cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white")
    ax.set_title("Streetscope data collection — city overview", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out_path = FIGS / "map_overview_india.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} ({len(gdf)} cities)")


def main():
    parser = argparse.ArgumentParser(description="Generate maps")
    parser.add_argument(
        "--cities",
        type=str,
        default="mumbai,navi_mumbai,bangalore,delhi",
        help="Comma-separated list of cities",
    )
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",")]

    print("=" * 60)
    print("GENERATING MAPS")
    print("=" * 60)
    print(f"Cities: {cities}")

    df = load_city_data(cities)
    print(f"Loaded {len(df):,} rows")

    for city in cities:
        geo = filter_valid_gps(df[df["city"] == city])
        print(f"\n{CITY_LABELS.get(city, city)} ({len(geo):,} valid-GPS points):")
        make_locations_map(geo, city)
        make_sex_ratio_map(geo, city)
        make_locations_pdf(geo, city)
        make_sex_ratio_pdf(geo, city)

    print("\nOverview:")
    make_overview_pdf(df)

    print("\n" + "=" * 60)
    print(f"Done. Maps saved to: {FIGS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
