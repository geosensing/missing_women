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
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABS = ROOT / "tabs"
TABS.mkdir(parents=True, exist_ok=True)

CITY_LABELS = {
    "mumbai": "Mumbai",
    "navi_mumbai": "Navi Mumbai",
    "bangalore": "Bangalore",
    "delhi": "Delhi",
}

# Residential sex ratio, women per 1,000 men, Census 2011, all four at the
# municipal-corporation level. The unit has to be held fixed across cities: the
# same city differs by 10-20 points between its district, its municipal
# corporation, and its urban agglomeration, so mixing units silently changes the
# headline.
#
# The municipal corporation is the right unit here because it is the sampling
# frame -- segments were drawn from the municipal street network.
#
# The manuscript draft used 838 for Mumbai and 910 for Navi Mumbai. Neither is a
# municipal-corporation sex ratio. 910 is not Navi Mumbai's figure under any age
# band: its overall ratio is 837 and its child (0-6) ratio is 902. See
# supplementary/mobility_accounting.md.
RESIDENTIAL_SEX_RATIO = {
    "mumbai": (853, "Greater Mumbai M Corp, Census 2011 (12,442,373 people)"),
    "navi_mumbai": (837, "Navi Mumbai M Corp, Census 2011 (1,120,547 people)"),
    "bangalore": (923, "Bruhat Bengaluru Mahanagara Palike, Census 2011"),
    "delhi": (876, "Delhi Municipal Corporation, Census 2011"),
}

# Goel (2023), Travel Behaviour and Society 32:100559, Table 1, urban India,
# Time Use Survey 2019. Mobility rate is the share making at least one trip on
# the reporting day; trip rate is trips per day over all respondents, so it
# already embeds the mobility rate.
MOBILITY_RATE = {"female": 0.473, "male": 0.86}
TRIP_RATE = {"female": 1.32, "male": 2.93}
MOBILITY_CITE = "Goel (2023) Table 1, TUS 2019, urban India"


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
    rows = []
    for city in cities:
        if city not in RESIDENTIAL_SEX_RATIO:
            print(f"WARNING: no residential sex ratio configured for {city}, skipping")
            continue
        residential, source = RESIDENTIAL_SEX_RATIO[city]
        observed, women, men = observed_sex_ratio(city)
        rows.append(
            {
                "city": city,
                "label": CITY_LABELS.get(city, city),
                "residential": float(residential),
                "source": source,
                "observed": observed,
                "women": women,
                "men": men,
                **account(observed, float(residential)),
            }
        )
    return rows


def residual_sensitivity(rows: list[dict], tolerance: float = 0.10) -> float:
    """Largest movement in the residual share if a census ratio is wrong.

    The residual is the quantity being claimed, so this is the check that matters
    while RESIDENTIAL_SEX_RATIO is unreconciled. If it is large, the claim is not
    usable until the manuscript's figures are read off.
    """
    worst = 0.0
    for row in rows:
        for scale in (1 - tolerance, 1 + tolerance):
            alt = account(row["observed"], row["residential"] * scale)
            worst = max(worst, abs(alt["share_residual"] - row["share_residual"]))
    return worst


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
        r" & \multicolumn{2}{c}{Sex ratio (F/1000 M)} & & "
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
        r"the residential sex ratio. It is reported unadjusted.",
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
    print("\nResidential sex ratios used (all municipal corporation, Census 2011):")
    for row in rows:
        print(f"  {row['label']:<13} {row['residential']:>4.0f}   {row['source']}")

    print()
    write_table(rows, sensitivity)


if __name__ == "__main__":
    main()
