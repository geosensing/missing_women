#!/usr/bin/env python3
"""
Publication Figures and Tables for Streetscope Project
=========================================================
Produces publication-quality PDFs and LaTeX tables from analysis_data.parquet files.

Outputs:
  figs/
    - fig2_distribution.pdf        Side-by-side histograms of prop_women
    - fig3_multipanel.pdf          4-panel: mode, road type, POI, time of day
    - fig4_weekday_weekend.pdf     Weekday vs weekend bar chart
    - map_locations.pdf            GPS scatter colored by city
    - map_sexratio.pdf             Heatmap of prop_women by location

  tabs/
    - table1_city_summary.tex      City-level summary
    - tableS1_road_type.tex        Prop women by road type per city
    - tableS2_temporal.tex         Weekday/weekend, time-of-day coverage
    - tableS3_poi_infrastructure.tex  POI effects, infrastructure counts

Usage:
    python scripts/10_analysis.py --cities mumbai,navi_mumbai
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import inference
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data"
FIGS = ROOT / "figs"
TABS = ROOT / "tabs"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "mumbai": "#2166ac",
    "navi_mumbai": "#b2182b",
    "bangalore": "#1b7837",
}
PARITY_C = "#999999"

CITY_LABELS = {
    "mumbai": "Mumbai",
    "navi_mumbai": "Navi Mumbai",
    "bangalore": "Bangalore",
}

# Map raw OSM highway tags onto the four design road-type buckets so the
# OSM "ground truth" road type is comparable to the itinerary road type.
OSM_ROAD_CLASS = {
    "trunk": "primary",
    "trunk_link": "primary",
    "primary": "primary",
    "primary_link": "primary",
    "secondary": "secondary",
    "secondary_link": "secondary",
    "tertiary": "tertiary",
    "tertiary_link": "tertiary",
    "residential": "residential",
    "living_street": "residential",
    "unclassified": "residential",
    "service": "residential",
}


def osm_road_class(highway) -> str | None:
    """Bucket a raw OSM highway tag into a design road-type, or None."""
    if highway is None or (isinstance(highway, float) and pd.isna(highway)):
        return None
    if isinstance(highway, list):  # osmnx can return a list of tags per edge
        highway = highway[0] if highway else None
    return OSM_ROAD_CLASS.get(highway)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_all_cities(cities: list[str]) -> pd.DataFrame:
    """Load and concat analysis_data.parquet for each city."""
    dfs = []
    for city in cities:
        path = OUTPUT / city / "analysis_data.parquet"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        df = pd.read_parquet(path)
        df["city"] = city
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No data found for any city")
    return pd.concat(dfs, ignore_index=True)


def pct(x: float) -> str:
    """Format as percentage for LaTeX."""
    return f"{100 * x:.1f}\\%"


def write_tex(path: Path, lines: list[str]) -> None:
    """Write LaTeX file."""
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# TABLE FUNCTIONS
# =============================================================================


def make_table1_city_summary(df: pd.DataFrame, cities: list[str]) -> None:
    """City-level summary table with images, people, prop_women, sex ratio, and cluster-robust CIs."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Women's share of the visible street population by city.}",
        r"\label{tab:city_summary}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{l" + "r" * len(cities) + "}",
        r"\toprule",
        " & " + " & ".join(CITY_LABELS.get(c, c) for c in cities) + r" \\",
        r"\midrule",
    ]

    row_data = {}
    for city in cities:
        sub = df[df["city"] == city]
        valid = sub[sub["total_people"] > 0]

        s = inference.summarize(valid)

        row_data[city] = {
            "n_images": len(sub),
            "n_clusters": s["n_clusters"],
            "n_people": sub["total_people"].sum(),
            "weighted": s["weighted"],
            "weighted_ci_lower": s["weighted_ci_lower"],
            "weighted_ci_upper": s["weighted_ci_upper"],
            "unweighted": s["unweighted"],
            "unweighted_ci_lower": s["unweighted_ci_lower"],
            "unweighted_ci_upper": s["unweighted_ci_upper"],
            "sex_ratio": (
                (sub["total_women"].sum() / sub["total_men"].sum() * 1000)
                if sub["total_men"].sum() > 0
                else 0
            ),
            "ped_prop": (
                sub["women_count"].sum() / (sub["women_count"].sum() + sub["men_count"].sum())
                if (sub["women_count"].sum() + sub["men_count"].sum()) > 0
                else 0
            ),
            "ped_sex_ratio": (
                (sub["women_count"].sum() / sub["men_count"].sum() * 1000)
                if sub["men_count"].sum() > 0
                else 0
            ),
            "icc": s["icc"],
            "design_effect": s["design_effect"],
        }

    def fmt_ci(d: dict, key: str) -> str:
        return f"{d[key]:.3f} [{d[key + '_ci_lower']:.3f}, {d[key + '_ci_upper']:.3f}]"

    lines.append(
        "Images annotated & " + " & ".join(f"{row_data[c]['n_images']:,}" for c in cities) + r" \\"
    )
    lines.append(
        "Video sessions (clusters) & "
        + " & ".join(f"{row_data[c]['n_clusters']:,}" for c in cities)
        + r" \\"
    )
    lines.append(
        "Total people classified & "
        + " & ".join(f"{row_data[c]['n_people']:,}" for c in cities)
        + r" \\"
    )
    lines.append(r"\addlinespace")
    lines.append(
        "Prop.\\ female (person-weighted) & "
        + " & ".join(fmt_ci(row_data[c], "weighted") for c in cities)
        + r" \\"
    )
    lines.append(
        "Prop.\\ female (image-level mean) & "
        + " & ".join(fmt_ci(row_data[c], "unweighted") for c in cities)
        + r" \\"
    )
    lines.append(r"\addlinespace")
    lines.append(
        "Sex ratio (F/1000 M) & "
        + " & ".join(f"{row_data[c]['sex_ratio']:.0f}" for c in cities)
        + r" \\"
    )
    lines.append(r"\addlinespace")
    lines.append(
        "Pedestrian prop.\\ female & "
        + " & ".join(f"{row_data[c]['ped_prop']:.3f}" for c in cities)
        + r" \\"
    )
    lines.append(
        "Pedestrian sex ratio (F/1000 M) & "
        + " & ".join(f"{row_data[c]['ped_sex_ratio']:.0f}" for c in cities)
        + r" \\"
    )
    lines.append(r"\addlinespace")
    lines.append(
        r"\multicolumn{" + str(len(cities) + 1) + r"}{l}{\textit{Clustering diagnostics}} \\"
    )
    lines.append(
        "ICC (prop.\\ female by video) & "
        + " & ".join(f"{row_data[c]['icc']:.3f}" for c in cities)
        + r" \\"
    )
    lines.append(
        "Design effect & "
        + " & ".join(f"{row_data[c]['design_effect']:.2f}" for c in cities)
        + r" \\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item 95\% confidence intervals in brackets use cluster-robust standard errors (clustered by video session).",
        r"\item Person-weighted estimates weight each individual equally; image-level means weight each frame equally.",
        r"\item Sex ratio is women per 1,000 men. Gender inferred from visible appearance.",
        r"\item Design effect = $1 + (\bar{n}_k - 1) \times \text{ICC}$, where $\bar{n}_k$ is average cluster size.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    write_tex(TABS / "table1_city_summary.tex", lines)
    print("  -> table1_city_summary.tex")


def make_tableS1_road_type(df: pd.DataFrame, cities: list[str]) -> None:
    """Prop women by itinerary_road_type per city with cluster-robust CIs."""
    road_types = ["primary", "secondary", "tertiary", "residential"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Women's share of visible people by road type.}",
        r"\label{tab:road_type}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"City & Road type & Prop.\ female [95\% CI] & Sex ratio & N (clusters) \\",
        r"\midrule",
    ]

    for city in cities:
        sub = df[df["city"] == city]
        city_label = CITY_LABELS.get(city, city)
        first = True
        for rt in road_types:
            rt_sub = sub[sub["itinerary_road_type"] == rt]
            valid = rt_sub[rt_sub["total_people"] > 0]
            if len(valid) == 0:
                continue
            total_women = rt_sub["total_women"].sum()
            total_men = rt_sub["total_men"].sum()

            s = inference.summarize(valid)
            sr = (total_women / total_men * 1000) if total_men > 0 else 0
            city_col = city_label if first else ""
            prop_ci = (
                f"{s['weighted']:.3f} [{s['weighted_ci_lower']:.3f}, {s['weighted_ci_upper']:.3f}]"
            )
            lines.append(
                f"{city_col} & {rt.title()} & {prop_ci} & {sr:.0f} & {s['n_obs']} ({s['n_clusters']}) \\\\"
            )
            first = False
        lines.append(r"\addlinespace")

    has_osm = "osm_highway" in df.columns and df["osm_highway"].notna().any()
    if has_osm:
        osm = df.copy()
        osm["osm_road_class"] = osm["osm_highway"].map(osm_road_class)
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\textit{OSM ground truth (nearest mapped road)}} \\")
        lines.append(r"\midrule")
        for city in cities:
            sub = osm[osm["city"] == city]
            city_label = CITY_LABELS.get(city, city)
            first = True
            for rt in road_types:
                rt_sub = sub[sub["osm_road_class"] == rt]
                valid = rt_sub[rt_sub["total_people"] > 0]
                if len(valid) == 0:
                    continue
                total_women = rt_sub["total_women"].sum()
                total_men = rt_sub["total_men"].sum()
                s = inference.summarize(valid)
                sr = (total_women / total_men * 1000) if total_men > 0 else 0
                city_col = city_label if first else ""
                prop_ci = f"{s['weighted']:.3f} [{s['weighted_ci_lower']:.3f}, {s['weighted_ci_upper']:.3f}]"
                lines.append(
                    f"{city_col} & {rt.title()} & {prop_ci} & {sr:.0f} & {s['n_obs']} ({s['n_clusters']}) \\\\"
                )
                first = False
            lines.append(r"\addlinespace")

    itin_note = r"\item Road type assigned from itinerary classification."
    if has_osm:
        itin_note = (
            r"\item Itinerary road type is the sampling design's intended class; "
            r"OSM ground truth is the nearest mapped road's highway tag."
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        itin_note,
        r"\item 95\% CIs use cluster-robust standard errors (clustered by video session).",
        r"\item Person-weighted estimates.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    write_tex(TABS / "tableS1_road_type.tex", lines)
    print("  -> tableS1_road_type.tex")


def make_tableS2_temporal(df: pd.DataFrame, cities: list[str]) -> None:
    """Weekday/weekend, time-of-day coverage with cluster-robust CIs."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Temporal patterns and data-collection coverage.}",
        r"\label{tab:temporal}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"City & Period & Prop.\ female [95\% CI] & Images (clusters) \\",
        r"\midrule",
    ]

    for city in cities:
        sub = df[(df["city"] == city) & df["is_weekend"].notna()]
        city_label = CITY_LABELS.get(city, city)

        for is_we, label in [(False, "Weekday"), (True, "Weekend")]:
            we_sub = sub[sub["is_weekend"] == is_we]
            valid = we_sub[we_sub["total_people"] > 0]
            if len(valid) == 0:
                continue
            s = inference.summarize(valid)
            prop_ci = (
                f"{s['weighted']:.3f} [{s['weighted_ci_lower']:.3f}, {s['weighted_ci_upper']:.3f}]"
            )
            lines.append(
                f"{city_label} & {label} & {prop_ci} & {len(we_sub):,} ({s['n_clusters']}) \\\\"
            )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Share of images by time window}} \\")
    lines.append(r"\midrule")

    time_bins = [
        (7, 11, "Morning (7-11)"),
        (11, 15, "Midday (11-15)"),
        (15, 19, "Evening (15-19)"),
    ]

    for city in cities:
        sub = df[(df["city"] == city) & df["frame_hour"].notna()]
        city_label = CITY_LABELS.get(city, city)
        n_total = len(sub)

        for start, end, label in time_bins:
            bin_sub = sub[(sub["frame_hour"] >= start) & (sub["frame_hour"] < end)]
            share = len(bin_sub) / n_total if n_total > 0 else 0
            lines.append(f"{city_label} & {label} & {pct(share)} & {len(bin_sub):,} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item 95\% CIs use cluster-robust standard errors (clustered by video session).",
        r"\item Coverage spans daytime hours only.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    write_tex(TABS / "tableS2_temporal.tex", lines)
    print("  -> tableS2_temporal.tex")


def make_tableS3_poi_infrastructure(df: pd.DataFrame) -> None:
    """POI effects, infrastructure counts with cluster-robust CIs."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Points of interest and infrastructure summaries.}",
        r"\label{tab:poi_infra}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Characteristic & Present & Prop.\ female [95\% CI] & Images (clusters) \\",
        r"\midrule",
    ]

    pois = [
        ("bus_station", "Bus station"),
        ("railway_station", "Railway station"),
        ("street_vendor", "Street vendor"),
    ]

    valid = df[df["total_people"] > 0]

    for field, label in pois:
        for val, val_label in [(True, "Yes"), (False, "No")]:
            sub = valid[valid[field] == val]
            if len(sub) == 0:
                continue
            s = inference.summarize(sub)
            prop_ci = (
                f"{s['weighted']:.3f} [{s['weighted_ci_lower']:.3f}, {s['weighted_ci_upper']:.3f}]"
            )
            lines.append(
                f"{label} & {val_label} & {prop_ci} & {len(sub):,} ({s['n_clusters']}) \\\\"
            )
        lines.append(r"\addlinespace")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Infrastructure coverage (all cities pooled)}} \\")
    lines.append(r"\midrule")

    infra_fields = [
        ("footpath", "Footpath"),
        ("potholes", "Potholes"),
        ("litter", "Litter"),
    ]

    for field, label in infra_fields:
        for val, val_label in [(True, "Yes"), (False, "No")]:
            sub = df[df[field] == val]
            if len(sub) == 0:
                continue
            pct_all = len(sub) / len(df) * 100
            lines.append(f"{label} & {val_label} & -- & {len(sub):,} ({pct_all:.0f}\\%) \\\\")
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item 95\% CIs use cluster-robust standard errors (clustered by video session).",
        r"\item Infrastructure fields are sparse; not all images have all fields coded.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    write_tex(TABS / "tableS3_poi_infrastructure.tex", lines)
    print("  -> tableS3_poi_infrastructure.tex")


# =============================================================================
# FIGURE FUNCTIONS
# =============================================================================


def make_fig2_distribution(df: pd.DataFrame, cities: list[str]) -> None:
    """Side-by-side histograms of prop_women with cluster-robust CIs."""
    valid = df[df["total_people"] > 0]

    fig, axes = plt.subplots(1, len(cities), figsize=(5.5, 2.2), sharey=True)
    if len(cities) == 1:
        axes = [axes]

    bins = np.arange(-0.025, 1.075, 0.05)

    for ax, city in zip(axes, cities):
        color = COLORS.get(city, "#666666")
        city_data = valid[valid["city"] == city]
        vals = city_data["prop_women"].dropna()

        s = inference.summarize(city_data)

        ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(x=0.5, color=PARITY_C, linestyle="--", linewidth=0.7, label="Parity")
        ax.axvline(
            x=s["weighted"],
            color=color,
            linestyle="-",
            linewidth=1.2,
            label=f"Mean = {s['weighted']:.2f}",
        )
        ax.axvspan(s["weighted_ci_lower"], s["weighted_ci_upper"], alpha=0.15, color=color)
        ax.set_xlabel("Proportion female")
        ax.set_title(CITY_LABELS.get(city, city), fontsize=9, fontweight="bold")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
        ax.set_xlim(-0.05, 1.05)

    axes[0].set_ylabel("Number of images")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_distribution.pdf")
    plt.close(fig)
    print("  -> fig2_distribution.pdf")


def make_fig3_multipanel(df: pd.DataFrame, cities: list[str]) -> None:
    """4-panel: mode, road type, POI, time of day."""
    valid = df[df["total_people"] > 0]

    fig = plt.figure(figsize=(6.5, 6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    w = 0.32

    # --- Panel A: Pedestrian vs Two-wheeler ---
    ax = fig.add_subplot(gs[0, 0])
    mode_data = {}
    for city in cities:
        c = df[df["city"] == city]
        tw_women = c["women_twowheeler"].sum()
        tw_men = c["men_twowheeler"].sum()
        ped_women = c["women_count"].sum()
        ped_men = c["men_count"].sum()
        mode_data[city] = {
            "Pedestrian": (ped_women / (ped_women + ped_men) if (ped_women + ped_men) > 0 else 0),
            "Two-wheeler": (tw_women / (tw_women + tw_men) if (tw_women + tw_men) > 0 else 0),
        }

    x = np.arange(2)
    for i, city in enumerate(cities):
        color = COLORS.get(city, "#666666")
        vals = [mode_data[city]["Pedestrian"], mode_data[city]["Two-wheeler"]]
        bars = ax.bar(
            x + i * w,
            vals,
            w,
            color=color,
            alpha=0.85,
            label=CITY_LABELS.get(city, city),
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    ax.set_xticks(x + w * (len(cities) - 1) / 2)
    ax.set_xticklabels(["Pedestrian", "Two-wheeler"])
    ax.set_ylabel("Proportion female")
    ax.set_ylim(0, 0.50)
    ax.axhline(y=0.5, color=PARITY_C, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("A  Transport mode", fontsize=8.5, fontweight="bold", loc="left")

    # --- Panel B: Road type ---
    ax = fig.add_subplot(gs[0, 1])
    road_types = ["primary", "secondary", "tertiary", "residential"]
    road_labels = ["Primary", "Secondary", "Tertiary", "Residential"]

    road_data = {}
    for city in cities:
        c = df[df["city"] == city]
        road_data[city] = {}
        for rt in road_types:
            rt_sub = c[c["itinerary_road_type"] == rt]
            if len(rt_sub) > 0 and rt_sub["total_people"].sum() > 0:
                road_data[city][rt] = rt_sub["total_women"].sum() / rt_sub["total_people"].sum()
            else:
                road_data[city][rt] = 0

    x = np.arange(len(road_types))
    for i, city in enumerate(cities):
        color = COLORS.get(city, "#666666")
        vals = [road_data[city][rt] for rt in road_types]
        bars = ax.bar(
            x + i * w,
            vals,
            w,
            color=color,
            alpha=0.85,
            label=CITY_LABELS.get(city, city),
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set_xticks(x + w * (len(cities) - 1) / 2)
    ax.set_xticklabels(road_labels, fontsize=7)
    ax.set_ylabel("Proportion female")
    ax.set_ylim(0, 0.35)
    ax.axhline(y=0.5, color=PARITY_C, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("B  Road type", fontsize=8.5, fontweight="bold", loc="left")

    # --- Panel C: POI ---
    ax = fig.add_subplot(gs[1, 0])
    poi_data = {}
    for poi in ["bus_station", "street_vendor"]:
        for val in [True, False]:
            sub = valid[valid[poi] == val]
            if len(sub) > 0:
                poi_data[(poi, val)] = {
                    "pf": sub["total_women"].sum() / sub["total_people"].sum(),
                    "n": len(sub),
                }

    pois = [("bus_station", "Bus\nstation"), ("street_vendor", "Street\nvendor")]
    x = np.arange(len(pois))
    for i, (pres, pres_label, color) in enumerate(
        [(False, "Absent", "#bdbdbd"), (True, "Present", "#e6550d")]
    ):
        vals = [poi_data.get((p[0], pres), {}).get("pf", 0) for p in pois]
        ns = [poi_data.get((p[0], pres), {}).get("n", 0) for p in pois]
        bars = ax.bar(x + i * w, vals, w, color=color, alpha=0.85, label=pres_label)
        for bar, val, n in zip(bars, vals, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}\n({n})",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels([p[1] for p in pois], fontsize=7)
    ax.set_ylabel("Proportion female")
    ax.set_ylim(0, 0.35)
    ax.legend(title="POI", fontsize=6.5, title_fontsize=6.5, frameon=False)
    ax.set_title("C  Point of interest", fontsize=8.5, fontweight="bold", loc="left")

    # --- Panel D: Time of day ---
    ax = fig.add_subplot(gs[1, 1])
    for city in cities:
        color = COLORS.get(city, "#666666")
        ct = valid[(valid["city"] == city) & valid["frame_hour"].notna()].copy()
        ct["hour_bin"] = ct["frame_hour"].astype(int)
        hourly = (
            ct.groupby("hour_bin")
            .apply(
                lambda g: pd.Series(
                    {
                        "pf": (
                            g["total_women"].sum() / g["total_people"].sum()
                            if g["total_people"].sum() > 0
                            else np.nan
                        ),
                        "n": len(g),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        hourly = hourly[hourly["n"] >= 5]
        ax.plot(
            hourly["hour_bin"],
            hourly["pf"],
            "o-",
            color=color,
            markersize=3.5,
            linewidth=1,
            label=CITY_LABELS.get(city, city),
            alpha=0.85,
        )

    ax.axhline(y=0.5, color=PARITY_C, linestyle="--", linewidth=0.5)
    ax.set_xlabel("Hour of day (IST)")
    ax.set_ylabel("Proportion female")
    ax.set_ylim(0, 0.50)
    ax.set_xlim(6.5, 19.5)
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("D  Time of day", fontsize=8.5, fontweight="bold", loc="left")

    fig.savefig(FIGS / "fig3_multipanel.pdf")
    plt.close(fig)
    print("  -> fig3_multipanel.pdf")


def make_fig4_weekday_weekend(df: pd.DataFrame, cities: list[str]) -> None:
    """Bar chart weekday vs weekend."""
    valid = df[(df["total_people"] > 0) & df["is_weekend"].notna()]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    w = 0.32

    items = []
    for city in cities:
        ct = valid[valid["city"] == city]
        for we, label in [(False, "Weekday"), (True, "Weekend")]:
            sub = ct[ct["is_weekend"] == we]
            if len(sub) > 0:
                items.append(
                    {
                        "city": city,
                        "day": label,
                        "pf": sub["total_women"].sum() / sub["total_people"].sum(),
                        "n": len(sub),
                    }
                )

    x = np.arange(2)
    for i, city in enumerate(cities):
        color = COLORS.get(city, "#666666")
        vals = [it["pf"] for it in items if it["city"] == city]
        ns = [it["n"] for it in items if it["city"] == city]
        bars = ax.bar(
            x + i * w,
            vals,
            w,
            color=color,
            alpha=0.85,
            label=CITY_LABELS.get(city, city),
        )
        for bar, val, n in zip(bars, vals, ns):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.1%}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    ax.set_xticks(x + w * (len(cities) - 1) / 2)
    ax.set_xticklabels(["Weekday", "Weekend"])
    ax.set_ylabel("Proportion female")
    ax.set_ylim(0, 0.35)
    ax.axhline(y=0.5, color=PARITY_C, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_weekday_weekend.pdf")
    plt.close(fig)
    print("  -> fig4_weekday_weekend.pdf")


def make_fig5_pedestrian_loess(df: pd.DataFrame, cities: list[str]) -> None:
    """LOESS of prop pedestrian women vs total pedestrians, by city."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    valid = df[df["men_count"].notna() & df["women_count"].notna()].copy()
    valid["total_pedestrians"] = valid["men_count"] + valid["women_count"]
    valid = valid[valid["total_pedestrians"] > 0]
    valid["prop_ped_women"] = valid["women_count"] / valid["total_pedestrians"]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for city in cities:
        sub = valid[valid["city"] == city].copy()
        color = COLORS.get(city, "#666666")
        label = CITY_LABELS.get(city, city)

        x = sub["total_pedestrians"].values
        y = sub["prop_ped_women"].values

        ax.scatter(x, y, c=color, s=8, alpha=0.25, edgecolors="none")

        smoothed = lowess(y, x, frac=0.3, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2, label=label)

    ax.axhline(y=0.5, color=PARITY_C, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Total pedestrians")
    ax.set_ylabel("Prop. pedestrian women")
    ax.set_ylim(0, 0.6)
    ax.set_xlim(0, valid["total_pedestrians"].quantile(0.99))
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig5_pedestrian_loess.pdf")
    plt.close(fig)
    print("  -> fig5_pedestrian_loess.pdf")


def make_map_locations(df: pd.DataFrame, cities: list[str]) -> None:
    """GPS scatter colored by city."""
    geo = df.dropna(subset=["gps_lat", "gps_lon"]).copy()

    if len(geo) == 0:
        print("  WARNING: No GPS data available for map")
        return

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for city in cities:
        color = COLORS.get(city, "#666666")
        sub = geo[geo["city"] == city]
        ax.scatter(
            sub["gps_lon"],
            sub["gps_lat"],
            c=color,
            s=4,
            alpha=0.35,
            linewidths=0,
            label=f"{CITY_LABELS.get(city, city)} (n={len(sub):,})",
        )

    ax.legend(
        fontsize=7,
        loc="lower left",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
    )
    ax.set_title("Data collection locations", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "map_locations.pdf")
    plt.close(fig)
    print("  -> map_locations.pdf")


def make_map_sexratio(df: pd.DataFrame) -> None:
    """Heatmap of prop_women by location."""
    geo = df[(df["total_people"] > 0) & df["gps_lat"].notna() & df["gps_lon"].notna()].copy()

    if len(geo) == 0:
        print("  WARNING: No GPS data available for sex ratio map")
        return

    fig, ax = plt.subplots(figsize=(6, 5.5))

    cmap = LinearSegmentedColormap.from_list(
        "sexratio", ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"]
    )

    sizes = np.clip(geo["total_people"].values ** 0.5 * 4, 5, 50)
    sc = ax.scatter(
        geo["gps_lon"],
        geo["gps_lat"],
        c=geo["prop_women"].clip(0, 0.5),
        cmap=cmap,
        vmin=0,
        vmax=0.5,
        s=sizes,
        alpha=0.55,
        linewidths=0.2,
        edgecolors="grey",
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.02, aspect=25)
    cbar.set_label("Proportion female", fontsize=7)
    cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.set_ticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    cbar.ax.tick_params(labelsize=6)

    n_zero = (geo["prop_women"] == 0).sum()
    ax.set_title(
        f"Female share by location (n={len(geo):,}; {100 * n_zero / len(geo):.0f}% with zero women)",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "map_sexratio.pdf")
    plt.close(fig)
    print("  -> map_sexratio.pdf")


def main():
    parser = argparse.ArgumentParser(description="Publication figures and tables")
    parser.add_argument(
        "--cities",
        type=str,
        default="mumbai,navi_mumbai",
        help="Comma-separated list of cities (default: mumbai,navi_mumbai)",
    )
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",")]

    print("=" * 60)
    print("STREETSCOPE: PUBLICATION OUTPUTS")
    print("=" * 60)
    print(f"Cities: {cities}")

    df = load_all_cities(cities)
    print(f"Loaded {len(df):,} total rows")

    print("\n" + "-" * 40)
    print("TABLES")
    print("-" * 40)
    make_table1_city_summary(df, cities)
    make_tableS1_road_type(df, cities)
    make_tableS2_temporal(df, cities)
    make_tableS3_poi_infrastructure(df)

    print("\n" + "-" * 40)
    print("FIGURES")
    print("-" * 40)
    make_fig2_distribution(df, cities)
    make_fig3_multipanel(df, cities)
    make_fig4_weekday_weekend(df, cities)
    make_fig5_pedestrian_loess(df, cities)

    print("\n" + "=" * 60)
    print("COMPLETE.")
    print(f"  Figures: {FIGS}")
    print(f"  Tables:  {TABS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
