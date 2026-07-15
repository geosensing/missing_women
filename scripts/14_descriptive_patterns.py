#!/usr/bin/env python3
"""
Descriptive Pattern Mining for Streetscope Project
=========================================================
Descriptive, policy-relevant patterns in women's street presence that have no
home in the publication tables: accompaniment/group composition, place rankings
(named corridors and ~100m GPS cells), and a joint descriptive regression
showing which bivariate correlates survive together.

Outputs:
  tabs/
    - descriptive_patterns.md      Markdown report with all tables

Usage:
    python scripts/14_descriptive_patterns.py --cities mumbai,navi_mumbai,bangalore,delhi
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

analysis = importlib.import_module("10_analysis")

ROOT = Path(__file__).resolve().parents[1]
TABS = ROOT / "tabs"
TABS.mkdir(parents=True, exist_ok=True)

MIN_ROAD_FRAMES = 30
MIN_CELL_FRAMES = 20

TIME_BINS = [
    (7, 11, "Morning (7-11)"),
    (11, 15, "Midday (11-15)"),
    (15, 19, "Evening (15-19)"),
]


def city_label(city: str) -> str:
    return analysis.CITY_LABELS.get(city, city)


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def accompaniment_section(df: pd.DataFrame, cities: list[str]) -> list[str]:
    """Solo-pedestrian share, group composition, and clustering vs binomial benchmark."""
    ped = df[(df["men_count"] + df["women_count"]) > 0].copy()
    ped["n_ped"] = ped["men_count"] + ped["women_count"]

    lines = ["## Accompaniment and group composition", ""]

    rows = []
    for city in cities + ["all"]:
        sub = ped if city == "all" else ped[ped["city"] == city]
        p = sub["women_count"].sum() / sub["n_ped"].sum()
        solo = sub[sub["n_ped"] == 1]
        with_w = sub[sub["women_count"] > 0]
        dist = with_w["women_count"].clip(upper=4).value_counts(normalize=True).sort_index()
        rows.append(
            [
                "All cities" if city == "all" else city_label(city),
                f"{p:.1%}",
                f"{solo['women_count'].mean():.1%} (n={len(solo):,})",
                f"{len(with_w) / len(sub):.1%}",
                " / ".join(f"{dist.get(k, 0):.0%}" for k in [1, 2, 3, 4]),
            ]
        )
    lines += md_table(
        [
            "City",
            "Ped. female share",
            "Solo pedestrians female",
            "Frames with any woman",
            "Women per such frame: 1 / 2 / 3 / 4+",
        ],
        rows,
    )

    lines += [
        "",
        "### P(frame contains a woman) by crowd size: observed vs binomial",
        "",
        "Expected under independent mixing: `1 - (1 - p_city)^n` averaged over frames",
        "in the bucket, with `p_city` the city's pedestrian female share. Observed below",
        "expected means women cluster together in fewer frames than chance.",
        "",
    ]
    buckets = [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, np.inf, "7+")]
    rows = []
    for city in cities:
        sub = ped[ped["city"] == city]
        p = sub["women_count"].sum() / sub["n_ped"].sum()
        cells = []
        for lo, hi, _ in buckets:
            b = sub[(sub["n_ped"] >= lo) & (sub["n_ped"] <= hi)]
            if len(b) == 0:
                cells.append("--")
                continue
            obs = (b["women_count"] > 0).mean()
            exp = (1 - (1 - p) ** b["n_ped"]).mean()
            cells.append(f"{obs:.0%} vs {exp:.0%}")
        rows.append([city_label(city)] + cells)
    lines += md_table(["City"] + [f"n={b[2]} (obs vs exp)" for b in buckets], rows)
    return lines


def place_rankings_section(df: pd.DataFrame) -> list[str]:
    """Named corridors and ~100m GPS cells ranked by person-weighted female share."""
    lines = ["## Place rankings", ""]

    named = df[df["osm_road_name"].notna()]
    roads = named.groupby(["city", "osm_road_name"]).agg(
        n=("total_people", "size"),
        people=("total_people", "sum"),
        women=("total_women", "sum"),
    )
    roads = roads[roads["n"] >= MIN_ROAD_FRAMES]
    roads["pf"] = roads["women"] / roads["people"]
    roads = roads.sort_values("pf")

    def road_rows(chunk: pd.DataFrame) -> list[list[str]]:
        return [
            [name, city_label(city), f"{r.pf:.1%}", f"{int(r.people):,}", int(r.n)]
            for (city, name), r in chunk.iterrows()
        ]

    lines += [f"### Named corridors ({MIN_ROAD_FRAMES}+ frames): lowest female share", ""]
    lines += md_table(
        ["Road", "City", "Female share", "People", "Frames"], road_rows(roads.head(10))
    )
    lines += ["", f"### Named corridors ({MIN_ROAD_FRAMES}+ frames): highest female share", ""]
    lines += md_table(
        ["Road", "City", "Female share", "People", "Frames"], road_rows(roads.tail(10).iloc[::-1])
    )

    gps = df[df["gps_lat"].notna()].copy()
    gps["cell_lat"] = gps["gps_lat"].round(3)
    gps["cell_lon"] = gps["gps_lon"].round(3)
    cells = gps.groupby(["city", "cell_lat", "cell_lon"]).agg(
        n=("total_people", "size"),
        people=("total_people", "sum"),
        women=("total_women", "sum"),
        road=("osm_road_name", lambda s: s.mode().iat[0] if s.notna().any() else "--"),
    )
    cells = cells[cells["n"] >= MIN_CELL_FRAMES]
    cells["pf"] = cells["women"] / cells["people"]
    cells = cells.sort_values("pf")

    def cell_rows(chunk: pd.DataFrame) -> list[list[str]]:
        return [
            [f"{lat:.3f}, {lon:.3f}", city_label(city), r.road, f"{r.pf:.1%}", f"{int(r.people):,}"]
            for (city, lat, lon), r in chunk.iterrows()
        ]

    lines += ["", f"### ~100m grid cells ({MIN_CELL_FRAMES}+ frames): lowest female share", ""]
    lines += md_table(
        ["Cell (lat, lon)", "City", "Nearest road", "Female share", "People"],
        cell_rows(cells.head(10)),
    )
    lines += ["", f"### ~100m grid cells ({MIN_CELL_FRAMES}+ frames): highest female share", ""]
    lines += md_table(
        ["Cell (lat, lon)", "City", "Nearest road", "Female share", "People"],
        cell_rows(cells.tail(10).iloc[::-1]),
    )
    return lines


def regression_section(df: pd.DataFrame) -> list[str]:
    """Joint descriptive WLS of frame-level female share on all correlates."""
    d = df[(df["total_people"] > 0) & df["frame_hour"].notna()].copy()
    d["prop_female_all"] = d["total_women"] / d["total_people"]
    d["window"] = pd.cut(
        d["frame_hour"],
        bins=[b[0] for b in TIME_BINS] + [TIME_BINS[-1][1]],
        labels=[b[2] for b in TIME_BINS],
        right=False,
    )
    d["road_class"] = d["road_class"].fillna("unmatched")
    for col in ["street_vendor", "bus_station", "litter", "potholes", "is_weekend"]:
        d[col] = (d[col] == True).astype(int)  # noqa: E712  (object columns with None)
    d["log_people"] = np.log(d["total_people"])
    d = d.dropna(subset=["window", "prop_female_all"])

    model = smf.wls(
        "prop_female_all ~ C(city, Treatment('mumbai'))"
        " + C(road_class, Treatment('residential'))"
        " + C(window, Treatment('Midday (11-15)'))"
        " + is_weekend + street_vendor + bus_station + litter + potholes + log_people",
        data=d,
        weights=d["total_people"],
    )
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": d["base_video_id"]})

    pretty = {
        "C(city, Treatment('mumbai'))[T.navi_mumbai]": "Navi Mumbai (vs Mumbai)",
        "C(city, Treatment('mumbai'))[T.bangalore]": "Bangalore (vs Mumbai)",
        "C(city, Treatment('mumbai'))[T.delhi]": "Delhi (vs Mumbai)",
        "C(road_class, Treatment('residential'))[T.primary]": "Primary road (vs residential)",
        "C(road_class, Treatment('residential'))[T.secondary]": "Secondary road (vs residential)",
        "C(road_class, Treatment('residential'))[T.tertiary]": "Tertiary road (vs residential)",
        "C(road_class, Treatment('residential'))[T.unmatched]": "Road unmatched (vs residential)",
        "C(window, Treatment('Midday (11-15)'))[T.Morning (7-11)]": "Morning 7-11 (vs midday)",
        "C(window, Treatment('Midday (11-15)'))[T.Evening (15-19)]": "Evening 15-19 (vs midday)",
        "is_weekend": "Weekend",
        "street_vendor": "Street vendor present",
        "bus_station": "Bus station present",
        "litter": "Litter present",
        "potholes": "Potholes present",
        "log_people": "log(people in frame)",
    }

    lines = [
        "## Joint descriptive regression",
        "",
        f"WLS of frame-level female share, weighted by people per frame (n={len(d):,} frames,",
        f"{d['base_video_id'].nunique()} video-session clusters); cluster-robust SEs.",
        "Coefficients in percentage points. Descriptive, not causal.",
        "",
    ]
    rows = []
    for name, label in pretty.items():
        b = fit.params[name] * 100
        lo, hi = fit.conf_int().loc[name] * 100
        star = "*" if fit.pvalues[name] < 0.05 else ""
        rows.append([label, f"{b:+.1f}{star}", f"[{lo:+.1f}, {hi:+.1f}]"])
    lines += md_table(["Correlate", "pp diff", "95% CI"], rows)
    lines += ["", "`*` p < 0.05. Base: Mumbai, residential road, midday, weekday, no POI/disorder."]
    return lines


def main():
    parser = argparse.ArgumentParser(description="Descriptive pattern mining")
    parser.add_argument("--cities", type=str, default="mumbai,navi_mumbai,bangalore,delhi")
    args = parser.parse_args()
    cities = [c.strip() for c in args.cities.split(",")]

    df = analysis.load_all_cities(cities)
    valid = df[df["total_people"] > 0]
    print(f"Loaded {len(df):,} rows ({len(valid):,} with people)")

    lines = [
        "# Descriptive patterns: where do we see more women?",
        "",
        f"Generated by `scripts/14_descriptive_patterns.py` for cities: "
        f"{', '.join(city_label(c) for c in cities)}.",
        "Shares are person-weighted (`sum(women) / sum(people)`) unless noted.",
        "",
    ]
    lines += accompaniment_section(valid, cities)
    lines += [""]
    lines += place_rankings_section(valid)
    lines += [""]
    lines += regression_section(valid)

    out = TABS / "descriptive_patterns.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
