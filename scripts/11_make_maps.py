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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_config import CITY_BOUNDS
from matplotlib.colors import ListedColormap, to_hex

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
    """Filter to rows with valid GPS coordinates within city bounds.

    Logs per-city how many frames are dropped (no GPS fix vs GPS outside the
    city bounding box, i.e. garbage fixes) so no data disappears silently.
    """
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
        n_total = (df["city"] == city).sum()
        n_kept = mask.sum()
        n_out_of_bounds = int(df.loc[df["city"] == city, "gps_out_of_bounds"].fillna(False).sum())
        n_missing = n_total - n_kept - n_out_of_bounds
        print(
            f"  {city}: {n_total:,} frames -> {n_missing:,} no GPS, "
            f"{n_out_of_bounds:,} outside bounds (garbage fix), {n_kept:,} mapped"
        )
        valid_rows.append(geo[mask])

    if not valid_rows:
        return pd.DataFrame()

    return pd.concat(valid_rows, ignore_index=True)


# Break a track line wherever consecutive fixes jump more than this (GPS glitch or a
# gap between disjoint segments sharing a video_id), so no false straight-line diagonals
# are drawn across the map.
TRACK_JUMP_M = 300.0


def load_tracks(city: str, sampled_base_ids: set[str], bounds: dict) -> list[pd.DataFrame]:
    """Return the driven GPS track for each sampled itinerary, downsampled for plotting.

    Restricts the full ``gps_index`` timeseries to the videos that actually appear in the
    plotted sample (reconciling the ``13``/``day13`` prefix inconsistency), clips to the
    city bounding box, and thins each track by rounding to ~11 m and dropping consecutive
    duplicates. One DataFrame per video, ordered by time.
    """
    gps_dir = DATA / city / "gps_index"
    ts_path, vm_path = gps_dir / "gps_timeseries.parquet", gps_dir / "video_metadata.parquet"
    if not (ts_path.exists() and vm_path.exists()):
        return []

    vm = pd.read_parquet(vm_path)

    def norm(v: str) -> str:
        return str(v).removeprefix("day")

    wanted = {norm(b) for b in sampled_base_ids}
    keep_ids = set(vm.loc[vm["base_video_id"].map(lambda v: norm(v) in wanted), "video_id"])
    if not keep_ids:
        return []

    ts = pd.read_parquet(ts_path, columns=["video_id", "gps_datetime", "lat", "lon"])
    ts = ts[
        ts["video_id"].isin(keep_ids)
        & ts["lat"].between(*bounds["lat"])
        & ts["lon"].between(*bounds["lon"])
    ].sort_values(["video_id", "gps_datetime"])

    tracks = []
    for _, grp in ts.groupby("video_id", sort=False):
        g = grp.copy()
        g["rlat"], g["rlon"] = g["lat"].round(4), g["lon"].round(4)
        keep = (g["rlat"] != g["rlat"].shift()) | (g["rlon"] != g["rlon"].shift())
        g = g[keep]
        if len(g) > 1:
            tracks.append(g[["lat", "lon"]])
    return tracks


def _track_segments(track: pd.DataFrame) -> list[pd.DataFrame]:
    """Split one track into contiguous segments, cutting where a jump exceeds TRACK_JUMP_M."""
    dlat = track["lat"].diff().to_numpy() * 111_000
    dlon = track["lon"].diff().to_numpy() * 105_000
    jump = np.hypot(dlat, dlon) > TRACK_JUMP_M
    seg_id = jump.cumsum()
    return [track.iloc[np.where(seg_id == s)[0]] for s in np.unique(seg_id)]


def make_locations_map(geo: pd.DataFrame, city: str, tracks: list[pd.DataFrame]) -> None:
    """Interactive map of one city's collection points over its driven itineraries."""
    if len(geo) == 0:
        print(f"  WARNING: no valid GPS for {city} locations map")
        return

    color = COLORS.get(city, "#666666")
    label = CITY_LABELS.get(city, city)

    m = folium.Map(
        location=[geo["gps_lat"].mean(), geo["gps_lon"].mean()],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    track_group = folium.FeatureGroup(name=f"{label} itineraries")
    for track in tracks:
        for seg in _track_segments(track):
            if len(seg) > 1:
                folium.PolyLine(
                    list(zip(seg["lat"], seg["lon"])),
                    color=color,
                    weight=1.5,
                    opacity=0.35,
                ).add_to(track_group)
    track_group.add_to(m)

    point_group = folium.FeatureGroup(name=f"Annotated frames (n={len(geo):,})")
    for _, row in geo.iterrows():
        pw = row.get("prop_women")
        pw_text = f"{pw:.0%}" if pd.notna(pw) else "--"
        popup_text = f"""
        <b>{label}</b><br>
        Video: {row.get("base_video_id", "N/A")}<br>
        Frame: {row.get("frame_number", "N/A")}<br>
        People: {row.get("total_people", "N/A")}<br>
        Percent women: {pw_text}
        """
        folium.CircleMarker(
            location=[row["gps_lat"], row["gps_lon"]],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.5,
            weight=0.5,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(point_group)
    point_group.add_to(m)

    m.fit_bounds(
        [[geo["gps_lat"].min(), geo["gps_lon"].min()], [geo["gps_lat"].max(), geo["gps_lon"].max()]]
    )
    folium.LayerControl().add_to(m)

    out_path = FIGS / f"map_locations_{city}.html"
    m.save(str(out_path))
    print(f"  -> {out_path.name} ({len(geo):,} points, {len(tracks)} tracks)")


def get_color_for_prop(prop: float) -> str:
    """Sequential single-hue color (light -> dark purple) for proportion women, capped at 0.5."""
    if pd.isna(prop):
        return "#808080"
    # Start at 0.15 so 0% is visible against a light basemap.
    return to_hex(plt.get_cmap("Purples")(0.15 + 0.85 * min(prop, 0.5) / 0.5))


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
        tiles="CartoDB positron",
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
        <b>Percent Women</b><br>
        <i style="background: {c0}; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 0%<br>
        <i style="background: {c25}; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 25%<br>
        <i style="background: {c50}; width: 12px; height: 12px; display: inline-block; border-radius: 50%;"></i> 50%+
    </div>
    """.format(
        c0=get_color_for_prop(0.0), c25=get_color_for_prop(0.25), c50=get_color_for_prop(0.5)
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    out_path = FIGS / f"map_sexratio_{city}.html"
    m.save(str(out_path))
    print(f"  -> {out_path.name} ({len(geo):,} points)")


def _to_mercator(track: pd.DataFrame) -> pd.DataFrame:
    """Project a lat/lon track to Web Mercator (x, y) columns."""
    g = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(track["lon"], track["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=3857)
    return pd.DataFrame({"x": g.geometry.x.to_numpy(), "y": g.geometry.y.to_numpy()})


def make_locations_pdf(
    geo: pd.DataFrame, city: str, tracks: list[pd.DataFrame], n_total: int | None = None
) -> None:
    """PDF map of one city's collection points over its driven itineraries."""
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
    for track in tracks:
        for seg in _track_segments(track):
            if len(seg) > 1:
                m = _to_mercator(seg)
                ax.plot(m["x"], m["y"], color=color, lw=0.7, alpha=0.3, solid_capstyle="round")
    ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        s=7,
        color=color,
        alpha=0.55,
        edgecolors="none",
        zorder=5,
        label=f"{label} (n={len(gdf):,})",
    )
    ax.margins(0.05)
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white")
    ax.set_title(
        f"{label}: Data Collection Locations & Itineraries", fontsize=12, fontweight="bold"
    )
    if n_total is not None and n_total > len(geo):
        ax.annotate(
            f"{n_total - len(geo):,} of {n_total:,} frames not shown (no GPS fix)",
            xy=(0.99, 0.01),
            xycoords="axes fraction",
            ha="right",
            fontsize=8,
            color="#555555",
        )

    fig.tight_layout()
    out_path = FIGS / f"map_locations_{city}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} ({len(geo):,} points, {len(tracks)} tracks)")


def make_sex_ratio_pdf(
    geo: pd.DataFrame, city: str, cell_m: int = 500, min_people: int = 20
) -> None:
    """PDF map of ~500m cells colored by person-weighted female share.

    Frame-level coloring is noise (a 1-person frame is forced to 0% or 100%),
    so points are binned to cells; cells with < min_people are drawn only as
    coverage dots.
    """
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
    gdf["cell_x"] = (gdf.geometry.x / cell_m).round() * cell_m
    gdf["cell_y"] = (gdf.geometry.y / cell_m).round() * cell_m

    cells = (
        gdf.groupby(["cell_x", "cell_y"])
        .agg(people=("total_people", "sum"), women=("total_women", "sum"))
        .reset_index()
    )
    cells = cells[cells["people"] >= min_people]
    cells["share"] = cells["women"] / cells["people"]

    # Truncate Purples so the lightest data color is still visible on the pale basemap.
    cmap = ListedColormap(plt.get_cmap("Purples")(np.linspace(0.3, 1.0, 256)))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(gdf.geometry.x, gdf.geometry.y, s=4, color="#777777", alpha=0.4, edgecolors="none")
    sc = ax.scatter(
        cells["cell_x"],
        cells["cell_y"],
        c=cells["share"].clip(0, 0.5),
        cmap=cmap,
        vmin=0,
        vmax=0.5,
        s=np.sqrt(cells["people"]) * 14,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
    )
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Percent female (person-weighted)", fontsize=10)
    cbar.set_ticks([0, 0.125, 0.25, 0.375, 0.5])
    cbar.set_ticklabels(["0%", "12%", "25%", "38%", "50%+"])

    ax.set_axis_off()
    ax.set_title(f"{label}: Share Female by Location", fontsize=12, fontweight="bold")
    ax.annotate(
        f"~{cell_m}m cells with >={min_people} people, sized by people observed; "
        "gray dots: all frames",
        xy=(0.01, 0.985),
        xycoords="axes fraction",
        va="top",
        fontsize=8,
        color="#555555",
    )

    fig.tight_layout()
    out_path = FIGS / f"map_sexratio_{city}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path.name} ({len(cells):,} cells from {len(geo):,} frames)")


def make_overview_pdf(geo: pd.DataFrame) -> None:
    """All-India overview: one labelled point per city, sized by sample count."""
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
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
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

    city_geos = []
    for city in cities:
        sub = df[df["city"] == city]
        geo = filter_valid_gps(sub)
        city_geos.append(geo)
        bounds = CITY_BOUNDS.get(city)
        tracks = (
            load_tracks(city, set(geo["base_video_id"].dropna().astype(str)), bounds)
            if bounds is not None and len(geo)
            else []
        )
        print(
            f"\n{CITY_LABELS.get(city, city)} ({len(geo):,} valid-GPS points, {len(tracks)} tracks):"
        )
        make_locations_map(geo, city, tracks)
        make_sex_ratio_map(geo, city)
        make_locations_pdf(geo, city, tracks, n_total=len(sub))
        make_sex_ratio_pdf(geo, city)

    print("\nOverview:")
    make_overview_pdf(pd.concat(city_geos, ignore_index=True))

    print("\n" + "=" * 60)
    print(f"Done. Maps saved to: {FIGS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
