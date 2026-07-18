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
MIN_PLACE_VIDEOS = 2

TIME_BINS = analysis.TIME_BINS


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
        "Two independent-mixing benchmarks, `1 - (1 - p)^n` averaged over frames in the",
        "bucket: `city` uses the city-wide pedestrian female share; `video` uses each",
        "video session's own share, so it conditions on place and time. Observed below",
        "the city benchmark but close to the video benchmark means women concentrate in",
        "particular places/times rather than clustering socially within a scene.",
        "",
    ]
    buckets = [(1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, np.inf, "7+")]
    rows = []
    for city in cities:
        sub = ped[ped["city"] == city]
        vid_p = sub.groupby("base_video_id").apply(
            lambda g: g["women_count"].sum() / g["n_ped"].sum(), include_groups=False
        )
        sub = sub.join(vid_p.rename("p_vid"), on="base_video_id")
        p = sub["women_count"].sum() / sub["n_ped"].sum()
        cells = []
        for lo, hi, _ in buckets:
            b = sub[(sub["n_ped"] >= lo) & (sub["n_ped"] <= hi)]
            if len(b) == 0:
                cells.append("--")
                continue
            obs = (b["women_count"] > 0).mean()
            exp_city = (1 - (1 - p) ** b["n_ped"]).mean()
            exp_vid = (1 - (1 - b["p_vid"]) ** b["n_ped"]).mean()
            cells.append(f"{obs:.0%} vs {exp_city:.0%} / {exp_vid:.0%}")
        rows.append([city_label(city)] + cells)
    lines += md_table(["City"] + [f"n={b[2]} (obs vs city/video exp)" for b in buckets], rows)
    return lines


def place_rankings_section(df: pd.DataFrame) -> list[str]:
    """Named corridors and ~100m GPS cells ranked by person-weighted female share."""
    lines = ["## Place rankings", ""]

    named = df[df["osm_road_name"].notna()]
    roads = named.groupby(["city", "osm_road_name"]).agg(
        n=("total_people", "size"),
        people=("total_people", "sum"),
        women=("total_women", "sum"),
        n_videos=("base_video_id", "nunique"),
    )
    roads = roads[(roads["n"] >= MIN_ROAD_FRAMES) & (roads["n_videos"] >= MIN_PLACE_VIDEOS)]
    roads["pf"] = roads["women"] / roads["people"]
    roads = roads.sort_values("pf")
    k_roads = min(10, len(roads) // 2)

    def road_rows(chunk: pd.DataFrame) -> list[list[str]]:
        return [
            [name, city_label(city), f"{r.pf:.1%}", f"{int(r.people):,}", int(r.n), int(r.n_videos)]
            for (city, name), r in chunk.iterrows()
        ]

    road_header = ["Road", "City", "Female share", "People", "Frames", "Videos"]
    lines += [
        f"### Named corridors ({MIN_ROAD_FRAMES}+ frames, {MIN_PLACE_VIDEOS}+ video sessions): "
        "lowest female share",
        "",
    ]
    lines += md_table(road_header, road_rows(roads.head(k_roads)))
    lines += [
        "",
        f"### Named corridors ({MIN_ROAD_FRAMES}+ frames, {MIN_PLACE_VIDEOS}+ video sessions): "
        "highest female share",
        "",
    ]
    lines += md_table(road_header, road_rows(roads.tail(k_roads).iloc[::-1]))
    lines += [
        "",
        "Corridors seen in a single video session are excluded: their shares reflect one",
        "walk's idiosyncrasy as much as the place.",
    ]

    gps = df[df["gps_lat"].notna()].copy()
    gps["cell_lat"] = gps["gps_lat"].round(3)
    gps["cell_lon"] = gps["gps_lon"].round(3)
    cells = gps.groupby(["city", "cell_lat", "cell_lon"]).agg(
        n=("total_people", "size"),
        people=("total_people", "sum"),
        women=("total_women", "sum"),
        n_videos=("base_video_id", "nunique"),
        road=("osm_road_name", lambda s: s.mode().iat[0] if s.notna().any() else "--"),
    )
    # No multi-video requirement at ~100m granularity (few cells are revisited);
    # the Videos column flags single-walk cells instead.
    cells = cells[cells["n"] >= MIN_CELL_FRAMES]
    cells["pf"] = cells["women"] / cells["people"]
    cells = cells.sort_values("pf")
    k_cells = min(10, len(cells) // 2)

    def cell_rows(chunk: pd.DataFrame) -> list[list[str]]:
        return [
            [
                f"{lat:.3f}, {lon:.3f}",
                city_label(city),
                r.road,
                f"{r.pf:.1%}",
                f"{int(r.people):,}",
                int(r.n_videos),
            ]
            for (city, lat, lon), r in chunk.iterrows()
        ]

    cell_header = ["Cell (lat, lon)", "City", "Nearest road", "Female share", "People", "Videos"]
    lines += [
        "",
        f"### ~100m grid cells ({MIN_CELL_FRAMES}+ frames): lowest female share",
        "",
    ]
    lines += md_table(cell_header, cell_rows(cells.head(k_cells)))
    lines += [
        "",
        f"### ~100m grid cells ({MIN_CELL_FRAMES}+ frames): highest female share",
        "",
    ]
    lines += md_table(cell_header, cell_rows(cells.tail(k_cells).iloc[::-1]))
    lines += [
        "",
        "Cells with Videos = 1 reflect a single walk; treat their shares as suggestive.",
    ]
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
    for col in ["street_vendor", "bus_station", "litter", "potholes", "footpath", "is_weekend"]:
        d[col] = (d[col] == True).astype(int)  # noqa: E712  (object columns with None)
    d["log_people"] = np.log(d["total_people"])
    d = d.dropna(subset=["window", "prop_female_all"])

    model = smf.wls(
        "prop_female_all ~ C(city, Treatment('mumbai'))"
        " + C(road_class, Treatment('residential'))"
        " + C(window, Treatment('Midday (11-15)'))"
        " + is_weekend + street_vendor + bus_station + litter + potholes + footpath"
        " + log_people",
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
        "C(window, Treatment('Midday (11-15)'))[T.Morning (6-11)]": "Morning 6-11 (vs midday)",
        "C(window, Treatment('Midday (11-15)'))[T.Evening (15-22)]": "Evening 15-22 (vs midday)",
        "is_weekend": "Weekend",
        "street_vendor": "Street vendor present",
        "bus_station": "Bus station present",
        "litter": "Litter present",
        "potholes": "Potholes present",
        "footpath": "Footpath present",
        "log_people": "log(people in frame)",
    }

    lines = [
        "## Joint descriptive regression",
        "",
        f"WLS of frame-level female share, weighted by people per frame (n={len(d):,} frames,",
        f"{d['base_video_id'].nunique()} video-session clusters); cluster-robust SEs.",
        "Sample: all frames with at least one person and a known hour (the windows",
        "partition the full 06:00-22:00 collection span). Coefficients in percentage",
        "points. Descriptive, not causal.",
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
    lines += [
        "",
        "## Limitations",
        "",
        '- Per-frame counts are top-coded at 10 ("10+" recorded as 11), attenuating',
        "  shares toward 0.5 in the densest scenes.",
        "- Collection spans roughly 06:00-22:00 IST only; nothing here speaks to night.",
        "- Road class is measured with error: itinerary and OSM road types agree on",
        "  ~72% of frames where both exist.",
        "- The same individuals can appear in multiple frames of one video session;",
        "  the estimand is the visible street population, and clustering by session",
        "  handles the standard errors, not the repeated sightings themselves.",
    ]
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
