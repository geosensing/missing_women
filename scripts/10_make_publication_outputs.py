#!/usr/bin/env python3
"""
Publication outputs for Streetscope
===================================
Produces publication-quality PDFs and LaTeX tables from analysis_data.parquet files.

Outputs:
  figs/
    - fig2_distribution.pdf        Side-by-side histograms of prop_women
    - fig3_multipanel.pdf          Mode and road-type estimates with 95% CIs
    - figS3_context_patterns.pdf   POI and broad time-window estimates with 95% CIs
    - fig5_pedestrian_crowdsize.pdf  Female share by pedestrian crowd size per city

  (Maps are produced separately by 11_make_maps.py.)

  tabs/
    - table1_city_summary.tex      City-level summary
    - tableS1_road_type.tex        Prop women by road type per city
    - tableS2_temporal.tex         Weekday/weekend, time-of-day coverage
    - tableS3_poi_infrastructure.tex  POI effects, infrastructure counts

Usage:
    python scripts/10_make_publication_outputs.py --cities mumbai,navi_mumbai,bangalore,delhi
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
from analysis_config import TOPCODE_SENSITIVITY_VALUES
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data"
FIGS = ROOT / "figs"
TABS = ROOT / "tabs"
FIGS.mkdir(parents=True, exist_ok=True)
TABS.mkdir(parents=True, exist_ok=True)

from figstyle import (  # noqa: E402
    ACCENT,
    BAND_GRAY,
    BAR_GRAY,
    CITY_LABELS,
    apply_style,
)

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


apply_style()


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
    combined = pd.concat(dfs, ignore_index=True)
    # Canonical road type is the OSM ground truth: the nearest mapped road's highway
    # tag, bucketed into the four design classes. It is populated for ~90%+ of GPS
    # frames in every city once the pipeline runs OSM enrichment. The itinerary proxy
    # (itinerary_road_type) is too sparse outside Mumbai/Navi Mumbai to report.
    combined["road_class"] = combined["osm_highway"].map(osm_road_class)
    return combined


def pct(x: float) -> str:
    """Format as percentage for LaTeX."""
    return f"{100 * x:.1f}\\%"


def pct_ci(s: dict, key: str = "weighted") -> str:
    """Percent cell with cluster-robust 95% CI.

    Tables display percent (matching prose and figure axes), never raw
    proportions. A single collection day leaves the cluster-robust CI
    undefined; print '--' rather than nan.
    """
    lo, hi = s[f"{key}_ci_lower"], s[f"{key}_ci_upper"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{100 * s[key]:.1f} [--]"
    return f"{100 * s[key]:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def write_tex(path: Path, lines: list[str]) -> None:
    """Write LaTeX file."""
    path.write_text("\n".join(lines) + "\n")


# =============================================================================
# TABLE FUNCTIONS
# =============================================================================


def compute_city_summary(df: pd.DataFrame, cities: list[str]) -> dict:
    """Per-city primary pedestrian estimates and secondary combined-mode estimates."""
    row_data = {}
    for city in cities:
        sub = df[df["city"] == city]
        valid = sub[sub["total_pedestrians"] > 0]

        s = inference.summarize(valid)
        combined = inference.summarize(
            sub,
            women_col="total_women",
            people_col="total_people",
        )

        row_data[city] = {
            "n_images": len(sub),
            "n_clusters": s["n_clusters"],
            "n_clusters_eff": s["n_clusters_eff"],
            "n_pedestrians": sub["total_pedestrians"].sum(),
            "n_people": sub["total_people"].sum(),
            "weighted": s["weighted"],
            "weighted_ci_lower": s["weighted_ci_lower"],
            "weighted_ci_upper": s["weighted_ci_upper"],
            "unweighted": s["unweighted"],
            "unweighted_ci_lower": s["unweighted_ci_lower"],
            "unweighted_ci_upper": s["unweighted_ci_upper"],
            "sex_ratio": (
                (sub["women_count"].sum() / sub["men_count"].sum() * 1000)
                if sub["men_count"].sum() > 0
                else 0
            ),
            "combined": combined["weighted"],
            "combined_ci_lower": combined["weighted_ci_lower"],
            "combined_ci_upper": combined["weighted_ci_upper"],
            "twowheeler": (
                sub["women_twowheeler"].sum() / sub["total_twowheeler"].sum()
                if sub["total_twowheeler"].sum() > 0
                else float("nan")
            ),
            "combined_img": combined["unweighted"],
            "combined_img_ci_lower": combined["unweighted_ci_lower"],
            "combined_img_ci_upper": combined["unweighted_ci_upper"],
        }
    return row_data


def compute_mode_props(df: pd.DataFrame, cities: list[str]) -> dict:
    """Per-city proportion female among pedestrians and two-wheeler riders."""
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
    return mode_data


CITY_MACRO_NAMES = {
    "mumbai": "Mumbai",
    "navi_mumbai": "NaviMumbai",
    "bangalore": "Bangalore",
    "delhi": "Delhi",
}


def compute_binary_summaries(df: pd.DataFrame, field: str) -> dict[bool, dict]:
    """Return shared present/absent pedestrian summaries for a binary field."""
    valid = df[df["total_pedestrians"] > 0]
    return {value: inference.summarize(valid[valid[field] == value]) for value in (True, False)}


def compute_weekpart_summaries(df: pd.DataFrame, cities: list[str]) -> dict[str, dict[bool, dict]]:
    """Return shared weekday/weekend pedestrian summaries for each city."""
    valid = df[(df["total_pedestrians"] > 0) & df["is_weekend"].notna()]
    return {
        city: {
            is_weekend: inference.summarize(
                valid[(valid["city"] == city) & (valid["is_weekend"] == is_weekend)]
            )
            for is_weekend in (False, True)
        }
        for city in cities
    }


def compute_topcode_sensitivity(df: pd.DataFrame, cities: list[str]) -> dict[int, dict[str, float]]:
    """Return pedestrian shares under each replacement for interval-censored counts."""
    estimates = {}
    for replacement in TOPCODE_SENSITIVITY_VALUES:
        city_estimates = {}
        for city in cities:
            city_data = df[df["city"] == city].copy()
            for count_col in ["women_count", "men_count"]:
                flag_col = f"{count_col}_topcoded"
                city_data.loc[city_data[flag_col], count_col] = replacement
            city_data["total_pedestrians"] = city_data["women_count"] + city_data["men_count"]
            city_estimates[city] = inference.summarize(city_data)["weighted"]
        estimates[replacement] = city_estimates
    return estimates


def make_results_macros(df: pd.DataFrame, cities: list[str]) -> None:
    """Write manuscript headline values from the same city summaries as Table 1."""
    summary = compute_city_summary(df, cities)
    road = compute_road_props(df, cities)
    lines = ["% Generated by scripts/10_make_publication_outputs.py; do not edit by hand."]
    total_images = sum(int(summary[city]["n_images"]) for city in cities)
    total_people = sum(int(summary[city]["n_people"]) for city in cities)
    lines.extend(
        [
            rf"\newcommand{{\TotalImages}}{{{total_images:,}}}",
            rf"\newcommand{{\TotalPersonSightings}}{{{total_people:,}}}",
        ]
    )

    for city in cities:
        name = CITY_MACRO_NAMES[city]
        values = summary[city]
        lines.extend(
            [
                rf"\newcommand{{\{name}Images}}{{{int(values['n_images']):,}}}",
                rf"\newcommand{{\{name}AllModeShare}}{{{100 * values['combined']:.1f}}}",
                rf"\newcommand{{\{name}AllModeImageShare}}{{{100 * values['combined_img']:.1f}}}",
                rf"\newcommand{{\{name}PedestrianShare}}{{{100 * values['weighted']:.1f}}}",
                rf"\newcommand{{\{name}PedestrianSexRatio}}{{{values['sex_ratio']:.0f}}}",
                rf"\newcommand{{\{name}TwoWheelerShare}}{{{100 * values['twowheeler']:.1f}}}",
            ]
        )

    lines.extend(
        [
            rf"\newcommand{{\MinAllModeShare}}{{{100 * min(summary[c]['combined'] for c in cities):.1f}}}",
            rf"\newcommand{{\MaxAllModeShare}}{{{100 * max(summary[c]['combined'] for c in cities):.1f}}}",
            rf"\newcommand{{\MinMaleAllModeShare}}{{{100 * (1 - max(summary[c]['combined'] for c in cities)):.1f}}}",
            rf"\newcommand{{\MaxMaleAllModeShare}}{{{100 * (1 - min(summary[c]['combined'] for c in cities)):.1f}}}",
            rf"\newcommand{{\MinPedestrianSexRatio}}{{{min(summary[c]['sex_ratio'] for c in cities):.0f}}}",
            rf"\newcommand{{\MaxPedestrianSexRatio}}{{{max(summary[c]['sex_ratio'] for c in cities):.0f}}}",
            rf"\newcommand{{\MinTwoWheelerShare}}{{{100 * min(summary[c]['twowheeler'] for c in cities):.1f}}}",
            rf"\newcommand{{\MaxTwoWheelerShare}}{{{100 * max(summary[c]['twowheeler'] for c in cities):.1f}}}",
        ]
    )
    road_values = [road[city][road_type] for city in cities for road_type in ROAD_TYPES]
    road_spreads = [max(road[city].values()) - min(road[city].values()) for city in cities]
    lines.extend(
        [
            rf"\newcommand{{\MinRoadTypeSpreadPP}}{{{100 * min(road_spreads):.1f}}}",
            rf"\newcommand{{\MaxRoadTypeSpreadPP}}{{{100 * max(road_spreads):.1f}}}",
            rf"\newcommand{{\MaxRoadTypeShare}}{{{100 * max(road_values):.1f}}}",
        ]
    )
    bus = compute_binary_summaries(df, "bus_station")
    vendor = compute_binary_summaries(df, "street_vendor")
    railway = compute_binary_summaries(df, "railway_station")[True]
    lines.extend(
        [
            rf"\newcommand{{\BusStationPresentShare}}{{{100 * bus[True]['weighted']:.1f}}}",
            rf"\newcommand{{\BusStationAbsentShare}}{{{100 * bus[False]['weighted']:.1f}}}",
            rf"\newcommand{{\StreetVendorPresentShare}}{{{100 * vendor[True]['weighted']:.1f}}}",
            rf"\newcommand{{\StreetVendorAbsentShare}}{{{100 * vendor[False]['weighted']:.1f}}}",
            rf"\newcommand{{\RailwayStationImages}}{{{railway['n_obs']:,}}}",
            rf"\newcommand{{\RailwayStationClusters}}{{{railway['n_clusters']:,}}}",
        ]
    )

    weekpart = compute_weekpart_summaries(df, cities)
    weekend_clusters = [weekpart[city][True]["n_clusters"] for city in cities]
    weekpart_differences = [
        abs(weekpart[city][False]["weighted"] - weekpart[city][True]["weighted"]) for city in cities
    ]
    lines.extend(
        [
            rf"\newcommand{{\MaxWeekdayWeekendDifferencePP}}{{{100 * max(weekpart_differences):.1f}}}",
            rf"\newcommand{{\MinWeekendCollectionDays}}{{{min(weekend_clusters)}}}",
            rf"\newcommand{{\MaxWeekendCollectionDays}}{{{max(weekend_clusters)}}}",
        ]
    )

    max_leave_one_day_shift = 0.0
    for city in cities:
        name = CITY_MACRO_NAMES[city]
        day_name, dominant_share, max_shift = temporal_sensitivity_summary(df[df["city"] == city])
        max_leave_one_day_shift = max(max_leave_one_day_shift, max_shift)
        lines.extend(
            [
                rf"\newcommand{{\{name}DominantDay}}{{{day_name}}}",
                rf"\newcommand{{\{name}DominantDayShare}}{{{100 * dominant_share:.1f}}}",
            ]
        )
    lines.append(rf"\newcommand{{\MaxLeaveOneDayShiftPP}}{{{100 * max_leave_one_day_shift:.1f}}}")
    topcode = compute_topcode_sensitivity(df, cities)
    primary_replacement = min(TOPCODE_SENSITIVITY_VALUES)
    upper_replacement = max(TOPCODE_SENSITIVITY_VALUES)
    topcode_shifts = [
        abs(topcode[upper_replacement][city] - topcode[primary_replacement][city])
        for city in cities
    ]
    lines.extend(
        [
            rf"\newcommand{{\MinTopcodeShiftPP}}{{{100 * min(topcode_shifts):.1f}}}",
            rf"\newcommand{{\MaxTopcodeShiftPP}}{{{100 * max(topcode_shifts):.1f}}}",
        ]
    )
    write_tex(TABS / "results_macros.tex", lines)
    print("  -> results_macros.tex")


ROAD_TYPES = ["primary", "secondary", "tertiary", "residential"]

# Partition of the observed collection hours (~06:00-22:00 IST); every frame with a
# known hour falls in exactly one window.
TIME_BINS = [
    (6, 11, "Morning (6-11)"),
    (11, 15, "Midday (11-15)"),
    (15, 23, "Evening (15-22)"),
]

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_FULL_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def compute_road_props(df: pd.DataFrame, cities: list[str]) -> dict:
    """Per-city proportion female by road class."""
    road_data = {}
    for city in cities:
        c = df[df["city"] == city]
        road_data[city] = {}
        for rt in ROAD_TYPES:
            rt_sub = c[c["road_class"] == rt]
            if len(rt_sub) > 0 and rt_sub["total_pedestrians"].sum() > 0:
                road_data[city][rt] = (
                    rt_sub["women_count"].sum() / rt_sub["total_pedestrians"].sum()
                )
            else:
                road_data[city][rt] = 0
    return road_data


def make_table1_city_summary(df: pd.DataFrame, cities: list[str]) -> None:
    """City-level primary pedestrian summary with collection-day clustered CIs."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Women's share among classified sightings in collected street imagery by city.}",
        r"\label{tab:city_summary}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{@{\extracolsep{0pt}}l" + "r" * len(cities) + "}",
        r"\toprule",
        " & " + " & ".join(CITY_LABELS.get(c, c) for c in cities) + r" \\",
        r"\midrule",
    ]

    row_data = compute_city_summary(df, cities)

    def fmt_ci(d: dict, key: str) -> str:
        return pct_ci(d, key)

    lines.append(
        "Images annotated & " + " & ".join(f"{row_data[c]['n_images']:,}" for c in cities) + r" \\"
    )
    lines.append(
        "Collection days (clusters) & "
        + " & ".join(f"{row_data[c]['n_clusters']:,}" for c in cities)
        + r" \\"
    )
    lines.append(
        "\\quad effective (Kish) & "
        + " & ".join(f"{row_data[c]['n_clusters_eff']:.1f}" for c in cities)
        + r" \\"
    )
    lines.append(
        "Adult pedestrian sightings & "
        + " & ".join(f"{row_data[c]['n_pedestrians']:,}" for c in cities)
        + r" \\"
    )
    lines.append(r"\addlinespace")
    lines.append(
        "Pedestrians: women, person-weighted (\\%) & "
        + " & ".join(fmt_ci(row_data[c], "weighted") for c in cities)
        + r" \\"
    )
    lines.append(
        "Pedestrians: women, image-level mean (\\%) & "
        + " & ".join(fmt_ci(row_data[c], "unweighted") for c in cities)
        + r" \\"
    )
    lines.append(r"\addlinespace")
    lines.append(
        "Women per 1,000 men & "
        + " & ".join(f"{row_data[c]['sex_ratio']:.0f}" for c in cities)
        + r" \\"
    )
    lines.append(r"\addlinespace")
    lines.append(
        "All modes: women, person-weighted (\\%) & "
        + " & ".join(fmt_ci(row_data[c], "combined") for c in cities)
        + r" \\"
    )
    lines.append(
        "Two-wheelers: women (\\%) & "
        + " & ".join(f"{100 * row_data[c]['twowheeler']:.1f}" for c in cities)
        + r" \\"
    )
    lines.append(
        "All modes: women, image-level mean (\\%) & "
        + " & ".join(fmt_ci(row_data[c], "combined_img") for c in cities)
        + r" \\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item The first two share rows and the sex ratio describe pedestrians only, which is"
        r" the primary estimand: the share of classified pedestrian sightings coded as"
        r" women. The all-mode rows additionally include two-wheeler riders.",
        r"\item 95\% confidence intervals use collection-day cluster-robust standard errors"
        r" and quantify variability across observed fieldwork days; the route imagery is not a"
        r" population sample.",
        r"\item Collection days are the clustering unit, but they contributed very unequal"
        r" numbers of frames, so the effective count is well below the raw count. The Kish"
        r" effective number, $(\sum n_c)^2 / \sum n_c^2$, is the one that governs interval"
        r" coverage. In Mumbai and Navi Mumbai it falls to about six and five, and intervals"
        r" for those cities should be read as indicative.",
        r"\item Person-weighted estimates weight each classified sighting equally; image-level means weight each frame equally.",
        r"\item Sex ratio is women per 1,000 men. Gender inferred from visible appearance.",
        r"\item ``10+'' is represented by its known minimum, 11; Table~\ref{tab:topcode} reports sensitivity.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    write_tex(TABS / "table1_city_summary.tex", lines)
    print("  -> table1_city_summary.tex")


def make_tableS1_road_type(df: pd.DataFrame, cities: list[str]) -> None:
    """Prop women by road type (OSM ground truth) per city with cluster-robust CIs."""
    road_types = ["primary", "secondary", "tertiary", "residential"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Women's share among classified pedestrian sightings by road type.}",
        r"\label{tab:road_type}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{@{\extracolsep{0pt}}llccc}",
        r"\toprule",
        r"City & Road type & Women (\%) [95\% CI] & Women per 1,000 men & Images (clusters) \\",
        r"\midrule",
    ]

    for city in cities:
        sub = df[df["city"] == city]
        city_label = CITY_LABELS.get(city, city)
        first = True
        for rt in road_types:
            rt_sub = sub[sub["road_class"] == rt]
            valid = rt_sub[rt_sub["total_pedestrians"] > 0]
            if len(valid) == 0:
                continue
            total_women = rt_sub["women_count"].sum()
            total_men = rt_sub["men_count"].sum()

            s = inference.summarize(valid)
            sr = (total_women / total_men * 1000) if total_men > 0 else 0
            city_col = city_label if first else ""
            prop_ci = pct_ci(s)
            lines.append(
                f"{city_col} & {rt.title()} & {prop_ci} & {sr:.0f} & {s['n_obs']} ({s['n_clusters']}) \\\\"
            )
            first = False
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item Road type is the OSM highway class of the nearest mapped road to each "
        r"frame's GPS fix, bucketed into the four design classes.",
        r"\item 95\% CIs use collection-day cluster-robust standard errors.",
        r"\item Person-weighted estimates.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    write_tex(TABS / "tableS1_road_type.tex", lines)
    print("  -> tableS1_road_type.tex")


def leave_one_day_out(sub: pd.DataFrame) -> tuple[float, list[tuple[str, float, dict]]]:
    """City estimate with each single day of the week removed, ordered by sample share.

    Returns the baseline restricted to frames with a known day of week, so the
    comparison isolates day composition rather than also dropping frames whose
    timestamp is missing.

    Collection days are the clusters, so removing a day of the week removes whole
    clusters: the intervals widen as well as shift, and the comparison spends
    precision rather than only testing composition.
    """
    valid = sub[(sub["total_pedestrians"] > 0) & sub["frame_dayofweek"].notna()]
    total = float(valid["total_pedestrians"].sum())
    if total <= 0:
        return float("nan"), []

    baseline = float(inference.summarize(valid)["weighted"])

    results: list[tuple[str, float, dict]] = []
    for day in sorted(pd.unique(valid["frame_dayofweek"])):
        dropped = valid[valid["frame_dayofweek"] == day]
        kept = valid[valid["frame_dayofweek"] != day]
        if len(kept) == 0:
            continue
        share = float(dropped["total_pedestrians"].sum()) / total
        results.append((DAY_LABELS[int(day)], share, inference.summarize(kept)))
    return baseline, sorted(results, key=lambda r: -r[1])


def temporal_sensitivity_summary(sub: pd.DataFrame) -> tuple[str, float, float]:
    """Return dominant weekday, its sighting share, and the largest exclusion shift."""
    valid = sub[(sub["total_pedestrians"] > 0) & sub["frame_dayofweek"].notna()]
    by_day = valid.groupby("frame_dayofweek")["total_pedestrians"].sum()
    if by_day.empty:
        raise ValueError("temporal sensitivity requires at least one dated pedestrian sighting")

    dominant_day_number = int(by_day.idxmax())
    day_name = DAY_FULL_NAMES[dominant_day_number]
    dominant_share = float(by_day.loc[dominant_day_number] / by_day.sum())

    baseline, results = leave_one_day_out(valid)
    max_shift = max(
        (abs(summary["weighted"] - baseline) for _, _, summary in results),
        default=0.0,
    )
    return day_name, dominant_share, max_shift


def make_tableS2_temporal(df: pd.DataFrame, cities: list[str]) -> None:
    """Weekday/weekend, time-of-day coverage with cluster-robust CIs."""
    weekpart = compute_weekpart_summaries(df, cities)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Temporal patterns and data-collection coverage.}",
        r"\label{tab:temporal}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{@{\extracolsep{0pt}}llcc}",
        r"\toprule",
        r"City & Period & Women (\%) [95\% CI] & Images (clusters) \\",
        r"\midrule",
    ]

    for city in cities:
        sub = df[(df["city"] == city) & df["is_weekend"].notna()]
        city_label = CITY_LABELS.get(city, city)

        for is_we, label in [(False, "Weekday"), (True, "Weekend")]:
            we_sub = sub[sub["is_weekend"] == is_we]
            valid = we_sub[we_sub["total_pedestrians"] > 0]
            if len(valid) == 0:
                continue
            s = weekpart[city][is_we]
            prop_ci = pct_ci(s)
            lines.append(
                f"{city_label} & {label} & {prop_ci} & {len(we_sub):,} ({s['n_clusters']}) \\\\"
            )

    max_shift = 0.0
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{4}{l}{\textit{Women (\%) excluding the most-sampled day of week}} \\"
    )
    lines.append(r"\midrule")

    for city in cities:
        city_label = CITY_LABELS.get(city, city)
        baseline, results = leave_one_day_out(df[df["city"] == city])
        if not results:
            continue

        max_shift = max(max_shift, max(abs(s["weighted"] - baseline) for _, _, s in results))

        day, share, s = results[0]
        prop_ci = pct_ci(s)
        label = f"Drop {day} ({pct(share)} of sightings)"
        lines.append(
            f"{city_label} & {label} & {prop_ci} & {s['n_obs']:,} ({s['n_clusters']}) \\\\"
        )

    time_bins = TIME_BINS

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Women (\%) by time window}} \\")
    lines.append(r"\midrule")

    for city in cities:
        sub = df[(df["city"] == city) & df["frame_hour"].notna()]
        city_label = CITY_LABELS.get(city, city)

        for start, end, label in time_bins:
            bin_sub = sub[(sub["frame_hour"] >= start) & (sub["frame_hour"] < end)]
            valid = bin_sub[bin_sub["total_pedestrians"] > 0]
            if len(valid) == 0:
                continue
            s = inference.summarize(valid)
            prop_ci = pct_ci(s)
            lines.append(
                f"{city_label} & {label} & {prop_ci} & {len(bin_sub):,} ({s['n_clusters']}) \\\\"
            )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Share of images by time window}} \\")
    lines.append(r"\midrule")

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
        r"\item 95\% CIs use collection-day cluster-robust standard errors.",
        r"\item An interval is unavailable when a subgroup contains only one collection day.",
        r"\item Collection spans roughly 06:00--22:00 IST; the three windows partition it.",
        r"\item Fieldwork days were not spread evenly over the week, so each city's imagery is "
        r"dominated by one or two days. Dropping the most-sampled day removes whole collection-day "
        r"clusters and therefore widens the interval as well as moving the point estimate. Across "
        r"every single-day exclusion in every city, the largest movement in the person-weighted "
        rf"share is {100 * max_shift:.1f} percentage points, and every leave-one-day-out "
        r"estimate lies inside the "
        r"corresponding full-sample interval.",
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
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{@{\extracolsep{0pt}}llcc}",
        r"\toprule",
        r"Characteristic & Present & Women (\%) [95\% CI] & Images (clusters) \\",
        r"\midrule",
    ]

    pois = [
        ("bus_station", "Bus station"),
        ("railway_station", "Railway station"),
        ("street_vendor", "Street vendor"),
    ]

    valid = df[df["total_pedestrians"] > 0]

    for field, label in pois:
        summaries = compute_binary_summaries(df, field)
        for val, val_label in [(True, "Yes"), (False, "No")]:
            summary = summaries[val]
            if summary["n_obs"] == 0:
                continue
            prop_ci = pct_ci(summary)
            lines.append(
                f"{label} & {val_label} & {prop_ci} & "
                f"{summary['n_obs']:,} ({summary['n_clusters']}) \\\\"
            )
        lines.append(r"\addlinespace")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Infrastructure and disorder (all cities pooled)}} \\")
    lines.append(r"\midrule")

    infra_fields = [
        ("footpath", "Footpath"),
        ("potholes", "Potholes"),
        ("litter", "Litter"),
    ]

    for field, label in infra_fields:
        for val, val_label in [(True, "Yes"), (False, "No")]:
            sub = valid[valid[field] == val]
            if len(sub) == 0:
                continue
            s = inference.summarize(sub)
            prop_ci = pct_ci(s)
            lines.append(
                f"{label} & {val_label} & {prop_ci} & {len(sub):,} ({s['n_clusters']}) \\\\"
            )
        lines.append(r"\addlinespace")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{l}{\textit{Points of interest by city}} \\")
    lines.append(r"\midrule")
    lines.append(r"Characteristic & Comparison & Present: \% ($n$) & Absent: \% ($n$) \\")
    lines.append(r"\midrule")

    for field, label in [("bus_station", "Bus station"), ("street_vendor", "Street vendor")]:
        for city in sorted(valid["city"].unique()):
            csub = valid[valid["city"] == city]
            row = [f"{CITY_LABELS.get(city, city)}: {label}"]
            cells = []
            for val in [True, False]:
                sub = csub[csub[field] == val]
                if len(sub) == 0:
                    cells.append("--")
                    continue
                s = inference.summarize(sub)
                cells.append(f"{100 * s['weighted']:.1f} (n={len(sub):,})")
            lines.append(f"{row[0]} & Yes/No & {cells[0]} & {cells[1]} \\\\")
        lines.append(r"\addlinespace")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item 95\% CIs use collection-day cluster-robust standard errors.",
        r"\item Unfilled infrastructure fields count as absent. Explicit ``Not visible''"
        r" and ``N/A'' responses remain missing; construction debris counts as litter/disorder.",
        r"\item Per-city POI rows report point estimates with frame counts;"
        r" cells with no frames are dashed.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]

    write_tex(TABS / "tableS3_poi_infrastructure.tex", lines)
    print("  -> tableS3_poi_infrastructure.tex")


def make_tableS5_topcode_sensitivity(df: pd.DataFrame, cities: list[str]) -> None:
    """Report how interval-censored ``10+`` counts affect the primary estimate."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of women's share among classified pedestrians (\%) to ``10+'' count replacement.}",
        r"\label{tab:topcode}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{@{\extracolsep{0pt}}l" + "r" * len(cities) + "}",
        r"\toprule",
        "Replacement & " + " & ".join(CITY_LABELS.get(c, c) for c in cities) + r" \\",
        r"\midrule",
    ]
    sensitivity = compute_topcode_sensitivity(df, cities)
    for replacement in TOPCODE_SENSITIVITY_VALUES:
        values = [sensitivity[replacement][city] for city in cities]
        label = f"{replacement}" + (" (known minimum; primary)" if replacement == 11 else "")
        lines.append(label + " & " + " & ".join(f"{100 * value:.1f}" for value in values) + r" \\")
    affected = {
        city: int(
            df.loc[df["city"] == city, ["women_count_topcoded", "men_count_topcoded"]]
            .any(axis=1)
            .sum()
        )
        for city in cities
    }
    affected_note = ", ".join(f"{CITY_LABELS.get(c, c)} {n}" for c, n in affected.items())
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item ``10+'' is interval-censored: 11 is its known minimum, not an exact count.",
        rf"\item Frames affected in pedestrian counts: {affected_note}.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    write_tex(TABS / "tableS5_topcode_sensitivity.tex", lines)
    print("  -> tableS5_topcode_sensitivity.tex")


# =============================================================================
# FIGURE FUNCTIONS
# =============================================================================


def make_fig2_distribution(df: pd.DataFrame, cities: list[str]) -> None:
    """Side-by-side histograms of image-level pedestrian female share.

    A line over a histogram reads as a summary of that distribution, so the
    two summary lines are drawn in distinct styles and named in the legend:
    the dotted line is the mean of the plotted distribution (each image counts
    equally); the solid line is the person-weighted share (each person counts
    equally), the paper's headline estimate. They differ where women's
    presence covaries with crowd size.
    """
    valid = df[df["total_pedestrians"] > 0]

    fig, axes = plt.subplots(1, len(cities), figsize=(5.5, 2.2), sharey=True)
    if len(cities) == 1:
        axes = [axes]

    bins = np.arange(-0.025, 1.075, 0.05)

    for ax, city in zip(axes, cities):
        city_data = valid[valid["city"] == city]
        vals = city_data["prop_female"].dropna()

        s = inference.summarize(city_data)

        ax.hist(vals, bins=bins, color=BAR_GRAY, edgecolor="white", linewidth=0.3)
        ax.axvline(x=0.5, color=ACCENT, linestyle="--", linewidth=0.8)
        ax.axvline(x=s["unweighted"], color="black", linestyle=":", linewidth=1)
        ax.axvline(x=s["weighted"], color="black", linestyle="-", linewidth=1.2)
        ax.text(
            s["weighted"] + 0.04,
            0.95,
            f"{s['weighted']:.1%}",
            transform=ax.get_xaxis_transform(),
            fontsize=6.5,
            va="top",
        )
        ax.set_title(CITY_LABELS.get(city, city), fontsize=9, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    axes[0].set_ylabel("Number of images")
    fig.supxlabel("Women (%) per image", fontsize=8)
    fig.legend(
        handles=[
            Line2D(
                [], [], color="black", linestyle="-", linewidth=1.2, label="Person-weighted share"
            ),
            Line2D([], [], color="black", linestyle=":", linewidth=1, label="Image-level mean"),
            Line2D([], [], color=ACCENT, linestyle="--", linewidth=0.8, label="Parity"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=6.5,
        bbox_to_anchor=(0.5, 1.1),
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_distribution.pdf")
    plt.close(fig)
    print("  -> fig2_distribution.pdf")


def plot_share_interval(
    ax, summary: dict, y: float, *, color: str = "0.1", markersize: float = 4.5
) -> None:
    """Draw one person-weighted estimate and its 95% interval."""
    estimate = summary["weighted"]
    lower = summary["weighted_ci_lower"]
    upper = summary["weighted_ci_upper"]
    if np.isfinite(lower) and np.isfinite(upper):
        ax.hlines(y, lower, upper, color="0.68", linewidth=1.2, zorder=1)
    ax.plot(estimate, y, "o", color=color, markersize=markersize, zorder=2)


def style_share_axis(ax, *, show_xlabel: bool = True) -> None:
    """Apply the common percentage-share scale used across estimate figures."""
    ax.set_xlim(0, 0.52)
    ax.set_xticks(np.arange(0, 0.51, 0.1))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.axvline(0.5, color=ACCENT, linestyle="--", linewidth=0.7, zorder=0)
    if show_xlabel:
        ax.set_xlabel("Women (%)", fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def mode_summaries(df: pd.DataFrame, city: str) -> dict[str, dict]:
    """Return pedestrian and two-wheeler estimates with clustered intervals."""
    city_data = df[df["city"] == city]
    return {
        "Pedestrian": inference.summarize(city_data),
        "Two-wheeler": inference.summarize(
            city_data,
            women_col="women_twowheeler",
            people_col="total_twowheeler",
        ),
    }


def make_fig3_multipanel(df: pd.DataFrame, cities: list[str]) -> None:
    """Main estimate figure: transport mode and road type, directly labeled."""
    fig = plt.figure(figsize=(6.5, 6.1))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1, 2.15], hspace=0.58)

    # Panel A: one directly labeled paired comparison per city.
    ax = fig.add_subplot(outer[0])
    for row, city in enumerate(cities):
        summaries = mode_summaries(df, city)
        pedestrian = summaries["Pedestrian"]
        twowheeler = summaries["Two-wheeler"]
        ax.plot(
            [twowheeler["weighted"], pedestrian["weighted"]],
            [row, row],
            color="0.82",
            linewidth=1,
            zorder=0,
        )
        plot_share_interval(ax, twowheeler, row, color="0.45")
        plot_share_interval(ax, pedestrian, row, color="0.05")
        ax.text(
            twowheeler["weighted"] + 0.009,
            row,
            f"Two-wheeler {twowheeler['weighted']:.1%}",
            ha="left",
            va="center",
            fontsize=6.3,
            color="0.35",
        )
        ax.text(
            pedestrian["weighted"] + 0.009,
            row,
            f"Pedestrian {pedestrian['weighted']:.1%}",
            ha="left",
            va="center",
            fontsize=6.3,
            color="0.05",
        )
    ax.set_yticks(range(len(cities)))
    ax.set_yticklabels([CITY_LABELS.get(city, city) for city in cities], fontsize=7)
    ax.set_ylim(-0.65, len(cities) - 0.45)
    ax.invert_yaxis()
    style_share_axis(ax)
    ax.set_title("A  Transport mode", fontsize=8.5, fontweight="bold", loc="left")

    # Panel B: city identity is carried by facet titles, not symbols.
    road_grid = outer[1].subgridspec(2, 2, hspace=0.7, wspace=0.32)
    road_labels = ["Primary", "Secondary", "Tertiary", "Residential"]
    road_axes = []
    for cell, city in zip(road_grid, cities):
        road_ax = fig.add_subplot(cell)
        road_axes.append(road_ax)
        city_data = df[(df["city"] == city) & (df["road_class"].isin(ROAD_TYPES))]
        for row, road_type in enumerate(ROAD_TYPES):
            summary = inference.summarize(city_data[city_data["road_class"] == road_type])
            plot_share_interval(road_ax, summary, row)
        road_ax.set_yticks(range(len(ROAD_TYPES)))
        road_ax.set_yticklabels(road_labels, fontsize=6.5)
        road_ax.set_ylim(-0.55, len(ROAD_TYPES) - 0.45)
        road_ax.invert_yaxis()
        style_share_axis(road_ax)
        road_ax.set_title(CITY_LABELS.get(city, city), fontsize=7.5, fontweight="bold")
    road_axes[0].text(
        0,
        1.28,
        "B  Road type",
        transform=road_axes[0].transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
    )

    fig.savefig(FIGS / "fig3_multipanel.pdf")
    plt.close(fig)
    print("  -> fig3_multipanel.pdf")


def make_figS3_context_patterns(df: pd.DataFrame, cities: list[str]) -> None:
    """Supplementary estimates for points of interest and broad time windows."""
    valid = df[df["total_pedestrians"] > 0]
    fig = plt.figure(figsize=(6.5, 6.1))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1, 2.15], hspace=0.58)

    # Panel A: pooled POI comparisons, directly labeled.
    ax = fig.add_subplot(outer[0])
    pois = [("bus_station", "Bus station"), ("street_vendor", "Street vendor")]
    for row, (field, label) in enumerate(pois):
        absent = inference.summarize(valid[valid[field] == False])  # noqa: E712
        present = inference.summarize(valid[valid[field] == True])  # noqa: E712
        for offset, name, summary, color in [
            (-0.12, "Present", present, "0.05"),
            (0.12, "Absent", absent, "0.45"),
        ]:
            y = row + offset
            plot_share_interval(ax, summary, y, color=color)
            ax.text(
                summary["weighted"] + 0.009,
                y,
                f"{name} {summary['weighted']:.1%}",
                ha="left",
                va="center",
                fontsize=6.3,
                color=color,
            )
        ax.text(-0.01, row, label, ha="right", va="center", fontsize=7)
    ax.set_yticks([])
    ax.set_ylim(-0.65, len(pois) - 0.45)
    ax.invert_yaxis()
    style_share_axis(ax)
    ax.set_title("A  Point of interest", fontsize=8.5, fontweight="bold", loc="left")

    # Panel B: broad windows avoid implying a smooth hourly process.
    time_grid = outer[1].subgridspec(2, 2, hspace=0.7, wspace=0.32)
    time_labels = ["Morning", "Midday", "Evening"]
    time_axes = []
    for cell, city in zip(time_grid, cities):
        time_ax = fig.add_subplot(cell)
        time_axes.append(time_ax)
        city_data = valid[(valid["city"] == city) & valid["frame_hour"].notna()]
        for row, (start, end, _) in enumerate(TIME_BINS):
            summary = inference.summarize(
                city_data[(city_data["frame_hour"] >= start) & (city_data["frame_hour"] < end)]
            )
            plot_share_interval(time_ax, summary, row)
        time_ax.set_yticks(range(len(TIME_BINS)))
        time_ax.set_yticklabels(time_labels, fontsize=6.5)
        time_ax.set_ylim(-0.55, len(TIME_BINS) - 0.45)
        time_ax.invert_yaxis()
        style_share_axis(time_ax)
        time_ax.set_title(CITY_LABELS.get(city, city), fontsize=7.5, fontweight="bold")
    time_axes[0].text(
        0,
        1.28,
        "B  Time of day",
        transform=time_axes[0].transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
    )

    fig.savefig(FIGS / "figS3_context_patterns.pdf")
    plt.close(fig)
    print("  -> figS3_context_patterns.pdf")


def make_fig5_pedestrian_crowdsize(df: pd.DataFrame, cities: list[str]) -> None:
    """Person-weighted female share by pedestrian crowd size, per city.

    LOESS is unusable here: x is heavily tied small integers (in Delhi 36% of
    frames have exactly 1 pedestrian, more than the smoothing window), which
    degenerates the local fit at the boundary.
    """
    min_frames = 10

    valid = df[df["men_count"].notna() & df["women_count"].notna()].copy()
    valid["total_pedestrians"] = valid["men_count"] + valid["women_count"]
    valid = valid[valid["total_pedestrians"] > 0]
    valid["prop_ped_women"] = valid["women_count"] / valid["total_pedestrians"]

    binned = {}
    for city in cities:
        sub = valid[valid["city"] == city]
        rows = []
        for crowd_size, group in sub.groupby(sub["total_pedestrians"].astype(int)):
            if len(group) < min_frames:
                continue
            summary = inference.summarize(group)
            rows.append(
                {
                    "crowd_size": crowd_size,
                    "share": summary["weighted"],
                    "lower": summary["weighted_ci_lower"],
                    "upper": summary["weighted_ci_upper"],
                    "n_frames": len(group),
                }
            )
        binned[city] = pd.DataFrame(rows).set_index("crowd_size")
    max_bin = max(b.index.max() for b in binned.values() if len(b) > 0)

    y_top = min(1.0, max(b["upper"].max() for b in binned.values() if len(b) > 0) + 0.03)

    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4), sharex=True, sharey=True)
    for ax, city in zip(axes.flat, cities):
        by_size = binned[city]
        ax.fill_between(
            by_size.index,
            by_size["lower"],
            by_size["upper"],
            color=BAND_GRAY,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            by_size.index,
            by_size["share"],
            marker="o",
            color="0.15",
            markersize=2.8,
            linewidth=1,
            zorder=2,
        )
        ax.set_title(CITY_LABELS.get(city, city), fontsize=8, fontweight="bold")
        ax.set_ylim(0, y_top)
        ax.set_xlim(0.5, max_bin + 0.5)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    fig.supxlabel("Total pedestrians in frame", fontsize=8)
    fig.supylabel("Women among pedestrians (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_pedestrian_crowdsize.pdf")
    plt.close(fig)
    print("  -> fig5_pedestrian_crowdsize.pdf")


README = ROOT / "README.md"
KEY_FINDINGS_START = "<!-- key-findings:start -->"
KEY_FINDINGS_END = "<!-- key-findings:end -->"
DATA_COVERAGE_START = "<!-- data-coverage:start -->"
DATA_COVERAGE_END = "<!-- data-coverage:end -->"


def update_readme_key_findings(df: pd.DataFrame, cities: list[str]) -> None:
    """Regenerate the Key Findings tables in README.md between marker comments."""
    summary = compute_city_summary(df, cities)
    mode = compute_mode_props(df, cities)
    road = compute_road_props(df, cities)

    city_names = [CITY_LABELS.get(c, c) for c in cities]

    lines = [
        "### Summary",
        "",
        "| City | Images | Pedestrians | Women (%) [95% CI] | Women per 1,000 men |",
        "|------|--------|-------------|--------------------|-----------------------|",
    ]
    for c, name in zip(cities, city_names):
        d = summary[c]
        lines.append(
            f"| {name} | {d['n_images']:,} | {d['n_pedestrians']:,} "
            f"| {d['weighted']:.1%} [{d['weighted_ci_lower']:.1%}, {d['weighted_ci_upper']:.1%}] "
            f"| {d['sex_ratio']:.0f} |"
        )

    lines += [
        "",
        "### By Mode",
        "",
        "| Mode | " + " | ".join(city_names) + " |",
        "|------|" + "------|" * len(cities),
    ]
    for m in ["Pedestrian", "Two-wheeler"]:
        lines.append(f"| {m}s | " + " | ".join(f"{mode[c][m]:.1%}" for c in cities) + " |")

    lines += [
        "",
        "### By Road Type",
        "",
        "| City | Primary | Secondary | Tertiary | Residential |",
        "|------|---------|-----------|----------|-------------|",
    ]
    for c, name in zip(cities, city_names):
        lines.append(f"| {name} | " + " | ".join(f"{road[c][rt]:.1%}" for rt in ROAD_TYPES) + " |")

    text = README.read_text()
    if KEY_FINDINGS_START not in text or KEY_FINDINGS_END not in text:
        raise ValueError(f"README.md is missing {KEY_FINDINGS_START} / {KEY_FINDINGS_END} markers")
    start = text.index(KEY_FINDINGS_START) + len(KEY_FINDINGS_START)
    end = text.index(KEY_FINDINGS_END)
    text = text[:start] + "\n" + "\n".join(lines) + "\n" + text[end:]

    if DATA_COVERAGE_START not in text or DATA_COVERAGE_END not in text:
        raise ValueError(
            f"README.md is missing {DATA_COVERAGE_START} / {DATA_COVERAGE_END} markers"
        )

    coverage = {}
    for city in cities:
        city_df = df[df["city"] == city]
        valid_gps = city_df["gps_valid"].mean()
        itinerary_match = city_df["itinerary_road_type"].notna().mean()
        hours = city_df["frame_hour"].dropna()
        coverage[city] = {
            "gps": valid_gps,
            "itinerary": itinerary_match,
            "hours": f"{int(hours.min())}-{int(hours.max())}",
        }

    coverage_lines = [
        "| Metric | " + " | ".join(city_names) + " |",
        "|--------|" + "--------|" * len(cities),
        "| Frames with valid in-city GPS | "
        + " | ".join(f"{coverage[c]['gps']:.1%}" for c in cities)
        + " |",
        "| Itinerary road class matched | "
        + " | ".join(f"{coverage[c]['itinerary']:.1%}" for c in cities)
        + " |",
        "| Hour range (IST) | " + " | ".join(coverage[c]["hours"] for c in cities) + " |",
    ]
    start = text.index(DATA_COVERAGE_START) + len(DATA_COVERAGE_START)
    end = text.index(DATA_COVERAGE_END)
    text = text[:start] + "\n" + "\n".join(coverage_lines) + "\n" + text[end:]

    README.write_text(text)
    print("  -> README.md key findings and data coverage")


def main():
    parser = argparse.ArgumentParser(description="Publication figures and tables")
    parser.add_argument(
        "--cities",
        type=str,
        default="mumbai,navi_mumbai,bangalore,delhi",
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
    make_results_macros(df, cities)
    make_table1_city_summary(df, cities)
    make_tableS1_road_type(df, cities)
    make_tableS2_temporal(df, cities)
    make_tableS3_poi_infrastructure(df)
    make_tableS5_topcode_sensitivity(df, cities)

    print("\n" + "-" * 40)
    print("FIGURES")
    print("-" * 40)
    make_fig2_distribution(df, cities)
    make_fig3_multipanel(df, cities)
    make_figS3_context_patterns(df, cities)
    make_fig5_pedestrian_crowdsize(df, cities)

    print("\n" + "-" * 40)
    print("README")
    print("-" * 40)
    update_readme_key_findings(df, cities)

    print("\n" + "=" * 60)
    print("COMPLETE.")
    print(f"  Figures: {FIGS}")
    print(f"  Tables:  {TABS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
