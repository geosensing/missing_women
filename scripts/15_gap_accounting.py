#!/usr/bin/env python3
"""
Accounting for the street gap
=============================
The headline compares the observed pedestrian sex ratio to the residential sex
ratio and reports the difference as women missing from the street. That number is
not adjusted here: women kept at home are the largest component of the absence,
not a confound to remove.

What this script does is locate the gap, not explain it. Published mobility
statistics say how much of it opens at the front door and in trip counts, and
what is left over is what street imagery adds. Two factors, both from Goel (2023)
Table 1 for urban India:

    women are less likely to leave home at all     47.3% vs 86%
    and, having left, make fewer trips             1.32 vs 2.93 trips per day

Applying each in turn to the residential ratio splits the gap three ways: women
not out of the home, women out but making fewer trips, and a residual that no
published mobility statistic accounts for. The residual is the quantity this
instrument uniquely measures -- a time-use survey records who left the house, not
who is standing on a road.

Outputs
    tabs/tableS6_gap_accounting.tex
    console summary

Usage
    python scripts/15_gap_accounting.py
    python scripts/15_gap_accounting.py --cities mumbai,delhi
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABS = ROOT / "tabs"
SUPPLEMENTARY = ROOT / "supplementary" / "mobility_accounting.md"
TABS.mkdir(parents=True, exist_ok=True)

GAP_BLOCK_START = "<!-- GENERATED: gap-accounting:start -->"
GAP_BLOCK_END = "<!-- GENERATED: gap-accounting:end -->"
SENSITIVITY_BLOCK_START = "<!-- GENERATED: gap-sensitivity:start -->"
SENSITIVITY_BLOCK_END = "<!-- GENERATED: gap-sensitivity:end -->"

C14_PATH = DATA / "reference" / "DDWCT-0000C-14.xls"
C14_MD5 = "6900b9d53313f1699b3891b75865ed50"
C14_CATALOG_URL = "https://censusindia.gov.in/nada/index.php/catalog/1640"
C14_COLUMNS = [
    "table",
    "state_code",
    "town_code",
    "area_name",
    "age_group",
    "total_persons",
    "total_males",
    "total_females",
    "rural_persons",
    "rural_males",
    "rural_females",
    "urban_persons",
    "urban_males",
    "urban_females",
]
AGE_GROUPS_20_PLUS = (
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80+",
)

# Exact city-level municipal units in the official all-India C-14 City workbook.
# State and town codes are checked alongside names so a similarly named census
# town cannot silently enter the benchmark.
C14_CITY_GEOGRAPHIES = {
    "mumbai": (27, 802794, "Greater Mumbai (M Corp.)"),
    "navi_mumbai": (27, 802788, "Navi Mumbai (M Corp.)"),
    "bangalore": (29, 803162, "BBMP (M. Corp.+OG)"),
    "delhi": (7, 800441, "DMC (U) (M Corp.)"),
}

CITY_LABELS = {
    "mumbai": "Mumbai",
    "navi_mumbai": "Navi Mumbai",
    "bangalore": "Bangalore",
    "delhi": "Delhi",
}
MACRO_PREFIX = {
    "mumbai": "Mumbai",
    "navi_mumbai": "NaviMumbai",
    "bangalore": "Bangalore",
    "delhi": "Delhi",
}

# Goel (2023), Travel Behaviour and Society 32:100559, Table 1, urban India,
# Time Use Survey 2019. Mobility rate is the share making at least one trip on
# the reporting day; trip rate is trips per day over all respondents, so it
# already embeds the mobility rate.
MOBILITY_RATE = {"female": 0.473, "male": 0.86}
TRIP_RATE = {"female": 1.32, "male": 2.93}
MOBILITY_CITE = "Goel (2023) Table 1, TUS 2019, urban India"


def file_md5(path: Path) -> str:
    """Return the MD5 used by the Census download record for file identity."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adult_residential_benchmarks(path: Path = C14_PATH) -> dict[str, dict]:
    """Compute exact age-20+ municipal sex ratios from Census 2011 C-14 City."""
    if not path.exists():
        raise FileNotFoundError(f"Census C-14 workbook not found: {path}")
    checksum = file_md5(path)
    if checksum != C14_MD5:
        raise ValueError(f"unexpected Census C-14 workbook MD5: {checksum}")

    census = pd.read_excel(
        path,
        sheet_name="Sheet1",
        header=None,
        skiprows=6,
        names=C14_COLUMNS,
        engine="xlrd",
    )
    census["state_code"] = pd.to_numeric(census["state_code"], errors="coerce")
    census["town_code"] = pd.to_numeric(census["town_code"], errors="coerce")
    census["age_group"] = census["age_group"].astype(str).str.strip()

    benchmarks = {}
    expected_ages = set(AGE_GROUPS_20_PLUS)
    for city, (state_code, town_code, area_name) in C14_CITY_GEOGRAPHIES.items():
        city_rows = census[census["state_code"].eq(state_code) & census["town_code"].eq(town_code)]
        names = set(city_rows["area_name"].dropna().astype(str))
        if names != {area_name}:
            raise ValueError(
                f"{city}: expected {area_name!r} at {state_code}/{town_code}, got {names}"
            )

        adult_rows = city_rows[city_rows["age_group"].isin(expected_ages)]
        observed_ages = set(adult_rows["age_group"])
        if observed_ages != expected_ages or len(adult_rows) != len(expected_ages):
            raise ValueError(
                f"{city}: incomplete or duplicated age-20+ rows; got {sorted(observed_ages)}"
            )

        men = int(pd.to_numeric(adult_rows["total_males"], errors="raise").sum())
        women = int(pd.to_numeric(adult_rows["total_females"], errors="raise").sum())
        if men <= 0 or women <= 0:
            raise ValueError(f"{city}: invalid age-20+ counts: {women=} {men=}")
        benchmarks[city] = {
            "state_code": state_code,
            "town_code": town_code,
            "area_name": area_name,
            "age_min": 20,
            "men": men,
            "women": women,
            "ratio": women / men * 1000,
        }
    return benchmarks


def out_of_home_factor() -> float:
    """Female-to-male ratio of the probability of leaving home at all."""
    return MOBILITY_RATE["female"] / MOBILITY_RATE["male"]


def trip_factor() -> float:
    """Female-to-male ratio of trips per day, over all respondents."""
    return TRIP_RATE["female"] / TRIP_RATE["male"]


def conditional_trip_factor() -> float:
    """Female-to-male ratio of trips among people who left home.

    The intensive margin alone. By construction this is trip_factor divided by
    out_of_home_factor, so the two published rates imply it rather than report it.
    """
    return trip_factor() / out_of_home_factor()


def observed_sex_ratio(city: str) -> tuple[float, int, int]:
    """Pedestrian sex ratio in the annotated imagery, women per 1,000 men."""
    df = pd.read_parquet(DATA / city / "analysis_data.parquet")
    women = float(df["women_count"].sum())
    men = float(df["men_count"].sum())
    if men <= 0:
        raise ValueError(f"{city}: no men counted, cannot form a ratio")
    return women / men * 1000, int(women), int(men)


def account(observed: float, residential: float) -> dict[str, float]:
    """Split the street gap into two explained parts and a residual.

    All quantities are women per 1,000 men, so the parts add to the gap:

        residential - observed
            = [residential - out_of_home]   women not out of the home
            + [out_of_home - trip_scaled]   women out, but making fewer trips
            + [trip_scaled - observed]      unexplained by published mobility

    The residual is what the imagery adds. It is not a claim about mechanism:
    it is the part of the shortfall that survives everything a time-use survey
    can see.
    """
    out_of_home = residential * out_of_home_factor()
    trip_scaled = residential * trip_factor()
    total_gap = residential - observed

    parts = {
        "out_of_home": out_of_home,
        "trip_scaled": trip_scaled,
        "total_gap": total_gap,
        "gap_home": residential - out_of_home,
        "gap_trips": out_of_home - trip_scaled,
        "gap_residual": trip_scaled - observed,
        "shortfall": 1 - observed / residential,
    }
    for name in ("home", "trips", "residual"):
        parts[f"share_{name}"] = parts[f"gap_{name}"] / total_gap if total_gap else float("nan")
    return parts


def build_rows(cities: list[str]) -> list[dict]:
    benchmarks = adult_residential_benchmarks()
    rows = []
    for city in cities:
        if city not in benchmarks:
            print(f"WARNING: no adult residential benchmark configured for {city}, skipping")
            continue
        benchmark = benchmarks[city]
        residential = benchmark["ratio"]
        observed, women, men = observed_sex_ratio(city)
        rows.append(
            {
                "city": city,
                "label": CITY_LABELS.get(city, city),
                "residential": float(residential),
                "source": (
                    f"{benchmark['area_name']}, Census 2011 C-14 City, age 20+ "
                    f"({benchmark['women']:,} women; {benchmark['men']:,} men)"
                ),
                "census_women": benchmark["women"],
                "census_men": benchmark["men"],
                "observed": observed,
                "women": women,
                "men": men,
                **account(observed, float(residential)),
            }
        )
    return rows


def residual_sensitivity(rows: list[dict], tolerance: float = 0.10) -> float:
    """Largest residual-share movement under a proportional benchmark change."""
    worst = 0.0
    for row in rows:
        for scale in (1 - tolerance, 1 + tolerance):
            alt = account(row["observed"], row["residential"] * scale)
            worst = max(worst, abs(alt["share_residual"] - row["share_residual"]))
    return worst


def write_macros(rows: list[dict]) -> None:
    """Write the residential benchmarks and gap results consumed by the manuscript."""
    lines = [r"% Generated by scripts/15_gap_accounting.py; do not edit by hand."]
    for row in rows:
        prefix = MACRO_PREFIX[row["city"]]
        values = {
            "ResidentialSexRatio": row["residential"],
            "StreetShortfall": 100 * row["shortfall"],
            "GapNotOut": 100 * row["share_home"],
            "GapFewerTrips": 100 * row["share_trips"],
            "GapResidual": 100 * row["share_residual"],
            "ObservedOfResidential": 100 * row["observed"] / row["residential"],
        }
        for suffix, value in values.items():
            lines.append(rf"\newcommand{{\{prefix}{suffix}}}{{{value:.0f}}}")

    shortfalls = [100 * row["shortfall"] for row in rows]
    observed_fractions = [100 * row["observed"] / row["residential"] for row in rows]
    lines.extend(
        [
            rf"\newcommand{{\MinStreetShortfall}}{{{min(shortfalls):.0f}}}",
            rf"\newcommand{{\MaxStreetShortfall}}{{{max(shortfalls):.0f}}}",
            rf"\newcommand{{\MinObservedOfResidential}}{{{min(observed_fractions):.0f}}}",
            rf"\newcommand{{\MaxObservedOfResidential}}{{{max(observed_fractions):.0f}}}",
        ]
    )
    for macro, key in (
        ("GapNotOut", "share_home"),
        ("GapFewerTrips", "share_trips"),
        ("GapResidual", "share_residual"),
    ):
        values = [100 * row[key] for row in rows]
        lines.append(rf"\newcommand{{\Min{macro}}}{{{min(values):.0f}}}")
        lines.append(rf"\newcommand{{\Max{macro}}}{{{max(values):.0f}}}")
    (TABS / "gap_macros.tex").write_text("\n".join(lines) + "\n")
    print("  -> gap_macros.tex")


def replace_generated_block(text: str, start: str, end: str, body: str) -> str:
    """Replace one delimited generated block without touching surrounding prose."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"expected one {start!r} / {end!r} block")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{start}\n{body.rstrip()}\n{end}{after}"


def write_supplementary(rows: list[dict]) -> None:
    """Synchronize mutable accounting values in the prose supplement."""
    text = SUPPLEMENTARY.read_text()
    gap_lines = [
        "| City | Residential (20+) | Observed | Shortfall | Not out of home | "
        "Out, fewer trips | **Residual** |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sensitivity_lines = [
        "| City | Residual | Range under ±10% on the census ratio |",
        "|---|---:|---:|",
    ]
    for row in rows:
        gap_lines.append(
            f"| {row['label']} | {row['residential']:.0f} | {row['observed']:.0f} | "
            f"{100 * row['shortfall']:.0f}% | {100 * row['share_home']:.0f}% | "
            f"{100 * row['share_trips']:.0f}% | **{100 * row['share_residual']:.0f}%** |"
        )
        alternatives = [
            account(row["observed"], row["residential"] * scale)["share_residual"]
            for scale in (0.9, 1.1)
        ]
        sensitivity_lines.append(
            f"| {row['label']} | {100 * row['share_residual']:.0f}% | "
            f"[{100 * min(alternatives):.1f}%, {100 * max(alternatives):.1f}%] |"
        )

    text = replace_generated_block(text, GAP_BLOCK_START, GAP_BLOCK_END, "\n".join(gap_lines))
    text = replace_generated_block(
        text,
        SENSITIVITY_BLOCK_START,
        SENSITIVITY_BLOCK_END,
        "\n".join(sensitivity_lines),
    )
    SUPPLEMENTARY.write_text(text)
    print("  -> supplementary/mobility_accounting.md")


def write_table(rows: list[dict], sensitivity: float) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Accounting for the street gap.}",
        r"\label{tab:gap_accounting}",
        r"\scriptsize",
        r"\begin{threeparttable}",
        r"\begin{tabular}{@{\extracolsep{0pt}}lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Women per 1,000 men} & & "
        r"\multicolumn{3}{c}{Share of the gap} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){5-7}",
        r"City & Residential & Observed & Shortfall & Not out & Fewer trips "
        r"& Residual \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['residential']:.0f} & {row['observed']:.0f} & "
            f"{100 * row['shortfall']:.0f}\\% & {100 * row['share_home']:.0f}\\% & "
            f"{100 * row['share_trips']:.0f}\\% & "
            f"{100 * row['share_residual']:.0f}\\% \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\item Shortfall is the headline comparison of the observed pedestrian sex ratio to "
        r"the age-20+ residential sex ratio. It is reported unadjusted.",
        r"\item Residential ratios are computed from the 20--24 through 80+ rows in the "
        r"official Census 2011 C-14 City table; age not stated is excluded "
        r"\citep{census2011c14city}.",
        r"\item The three shares decompose that shortfall. Women are less likely to leave home "
        rf"at all ({100 * MOBILITY_RATE['female']:.1f}\% against "
        rf"{100 * MOBILITY_RATE['male']:.0f}\% of men) and, having left, make fewer trips "
        rf"({TRIP_RATE['female']} against {TRIP_RATE['male']} per day; "
        rf"{conditional_trip_factor():.2f} conditional on being mobile). Both rates are for "
        r"urban India from the 2019 Time Use Survey \citep{goel2023gendergap}. Scaling the "
        r"residential ratio by each factor in turn partitions the gap.",
        r"\item The partition locates the gap; it does not explain it. The headline is "
        r"reported unadjusted because women kept at home are the largest component of the "
        r"absence, not a confound to remove.",
        r"\item The residual is the part only street imagery can see: women who left home, "
        r"made their trips, and are still not among pedestrian sightings. "
        r"A time-use survey records who left the house, not who is standing on a road.",
        r"\item The mobility rates are national urban averages applied to four specific cities. "
        r"If women in these cities are more mobile than the urban average, the front-door and "
        r"trip-count shares are overstated and the residual is a lower bound.",
        rf"\item Scaling a residential ratio by $\pm$10\% moves the residual share by at most "
        rf"{100 * sensitivity:.0f} percentage points.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    (TABS / "tableS6_gap_accounting.tex").write_text("\n".join(lines) + "\n")
    print("  -> tableS6_gap_accounting.tex")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", default="mumbai,navi_mumbai,bangalore,delhi")
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    rows = build_rows(cities)
    if not rows:
        raise SystemExit("no cities with a configured residential sex ratio")

    print(f"\nMobility factors ({MOBILITY_CITE}):")
    print(f"  left home at all            {out_of_home_factor():.3f}")
    print(f"  trips per person            {trip_factor():.3f}")
    print(f"  trips per mobile person     {conditional_trip_factor():.3f}")

    print(
        f"\n{'city':<13}{'resid':>7}{'obs':>6}{'shortfall':>11}   "
        f"{'not out':>8}{'fewer trips':>13}{'residual':>10}"
    )
    for row in rows:
        print(
            f"{row['label']:<13}{row['residential']:>7.0f}{row['observed']:>6.0f}"
            f"{100 * row['shortfall']:>10.0f}%   "
            f"{100 * row['share_home']:>7.0f}%{100 * row['share_trips']:>12.0f}%"
            f"{100 * row['share_residual']:>9.0f}%"
        )

    sensitivity = residual_sensitivity(rows)
    print(
        f"\n+/-10% on a census ratio moves the residual share by at most {100 * sensitivity:.1f} pp"
    )
    print("\nAdult residential sex ratios used (age 20+, Census 2011 C-14 City):")
    for row in rows:
        print(f"  {row['label']:<13} {row['residential']:>4.0f}   {row['source']}")

    print()
    write_macros(rows)
    write_table(rows, sensitivity)
    write_supplementary(rows)


if __name__ == "__main__":
    main()
