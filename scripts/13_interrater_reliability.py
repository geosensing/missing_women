#!/usr/bin/env python3
"""
Inter-rater reliability (IRR) for Streetscope annotations.

The analysis dataset (data/{city}/analysis_data.parquet) keeps only the primary
annotator, so annotator agreement cannot be measured from it. Rather than persist a
separate all-annotations file, this script rebuilds the full multi-annotator frame
IN MEMORY from the committed Label Studio JSON and measures agreement between the
primary annotator and reviewers on the images both rated.

It reuses the canonical parsing/derivation from steps 05 and 08 (convert_counts,
add_computed_fields, fill_infrastructure_columns, get_primary_annotator), so the
quantities match the analysis dataset exactly. No GPS/OSM enrichment is needed for IRR.

Metrics
    Continuous (women_count, men_count, total_pedestrians, prop_female):
        ICC(2,1) absolute agreement, Pearson r, mean |difference|.
    Binary infrastructure flags (footpath, potholes, ...):
        Cohen's kappa + percent agreement. Kappa is undefined when a flag has no
        variance in the overlap (every image agreed absent) and is shown as "--".

Outputs
    tabs/tableS4_irr.tex     (pooled across cities; per-city overlap counts in the note)
    console summary          (per city + pooled)

Usage
    python scripts/13_interrater_reliability.py
    python scripts/13_interrater_reliability.py --cities delhi,mumbai
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent
DATA = PROJECT_ROOT / "data"
TABS = PROJECT_ROOT / "tabs"

CITY_LABELS = {
    "mumbai": "Mumbai",
    "navi_mumbai": "Navi Mumbai",
    "bangalore": "Bangalore",
    "delhi": "Delhi",
}

# (column, LaTeX label)
CONTINUOUS = [
    ("women_count", "Women count"),
    ("men_count", "Men count"),
    ("total_pedestrians", "Total pedestrians"),
    ("prop_female", r"Prop.\ female"),
]
BINARY = [
    ("footpath", "Footpath"),
    ("lane_markings", "Lane markings"),
    ("potholes", "Potholes"),
    ("litter", "Litter"),
    ("bus_station", "Bus station"),
    ("railway_station", "Railway station"),
    ("street_vendor", "Street vendor"),
]


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


parse_mod = _import("parse_annotations", SCRIPTS / "05_parse_annotations.py")
build_mod = _import("build_analysis_data", SCRIPTS / "08_build_analysis_data.py")


def load_all_annotations(city: str) -> pd.DataFrame:
    """Rebuild the full multi-annotator frame for a city in memory.

    Mirrors step 08 up to the count/feature derivation, skipping GPS/OSM enrichment
    (steps 06/07) which IRR does not need.
    """
    df = parse_mod.parse_all_annotations(DATA / city / "labelstudio", city)
    df = build_mod.convert_counts(df)
    df = build_mod.add_computed_fields(df)
    df = build_mod.fill_infrastructure_columns(df)
    return df


def icc21(matrix: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, single rater, absolute agreement."""
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    if n < 2:
        return np.nan
    grand = m.mean()
    ss_rows = k * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((m.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((m - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom > 0 else np.nan


def build_pairs(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Pair the primary annotator with each reviewer on images both rated.

    One row per (image, reviewer) overlap; columns ``<measure>_p`` (primary) and
    ``<measure>_r`` (reviewer). A repeat annotation by the same person is collapsed to
    its first record so only distinct-annotator pairs are compared.
    """
    df = build_mod.keep_latest_annotation(df.dropna(subset=["annotator"]))
    primary = build_mod.get_primary_annotator(df)
    prim = df[df["annotator"] == primary].set_index("image")
    reviewers = df[df["annotator"] != primary]
    measures = [c for c, _ in CONTINUOUS] + [c for c, _ in BINARY]

    rows = []
    for _, r in reviewers.iterrows():
        img = r["image"]
        if img not in prim.index:
            continue
        p = prim.loc[img]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[0]
        rec = {}
        for col in measures:
            rec[f"{col}_p"] = p.get(col)
            rec[f"{col}_r"] = r.get(col)
        rows.append(rec)
    return pd.DataFrame(rows), primary


def continuous_metrics(pairs: pd.DataFrame, col: str) -> dict:
    a = pd.to_numeric(pairs.get(f"{col}_p"), errors="coerce")
    b = pd.to_numeric(pairs.get(f"{col}_r"), errors="coerce")
    ok = a.notna() & b.notna()
    a, b = a[ok].to_numpy(), b[ok].to_numpy()
    n = len(a)
    if n < 2:
        return {"n": n, "icc": np.nan, "r": np.nan, "mad": np.nan}
    r = pearsonr(a, b)[0] if (a.std() > 0 and b.std() > 0) else np.nan
    return {
        "n": n,
        "icc": icc21(np.column_stack([a, b])),
        "r": r,
        "mad": float(np.abs(a - b).mean()),
    }


def binary_metrics(pairs: pd.DataFrame, col: str) -> dict:
    a = pairs.get(f"{col}_p").map(lambda x: False if pd.isna(x) else bool(x))
    b = pairs.get(f"{col}_r").map(lambda x: False if pd.isna(x) else bool(x))
    n = len(a)
    agree = float((a.to_numpy() == b.to_numpy()).mean()) if n else np.nan
    if n < 2 or a.nunique() < 2 or b.nunique() < 2:
        kappa = np.nan  # undefined when a rater has a single class (no variance)
    else:
        kappa = float(cohen_kappa_score(a.astype(int), b.astype(int)))
    return {"n": n, "agree": agree, "kappa": kappa}


def _f(x: float, nd: int = 2) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_table(pooled_cont: dict, pooled_bin: dict, overlap_counts: dict, primary: str) -> None:
    n_overlap = sum(overlap_counts.values())
    per_city = ", ".join(f"{CITY_LABELS.get(c, c)} {n}" for c, n in overlap_counts.items())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inter-rater reliability: primary annotator vs.\ reviewers.}",
        r"\label{tab:irr}",
        r"\begin{threeparttable}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Measure & $n$ & ICC(2,1) & Pearson $r$ & Mean $|\Delta|$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Counts (continuous)}} \\",
    ]
    for col, label in CONTINUOUS:
        m = pooled_cont[col]
        lines.append(f"{label} & {m['n']} & {_f(m['icc'])} & {_f(m['r'])} & {_f(m['mad'])} \\\\")
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Infrastructure flags (binary)}} \\",
        r"Feature & $n$ & \% agree & \multicolumn{2}{c}{Cohen's $\kappa$} \\",
    ]
    for col, label in BINARY:
        m = pooled_bin[col]
        agree_pct = "--" if np.isnan(m["agree"]) else f"{100 * m['agree']:.1f}\\%"
        lines.append(
            f"{label} & {m['n']} & {agree_pct} & \\multicolumn{{2}}{{c}}{{{_f(m['kappa'])}}} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        rf"\item Pooled over {n_overlap} image$\times$reviewer overlaps ({per_city}).",
        r"\item Agreement between the primary annotator and any other annotator who rated "
        r"the same image. ICC(2,1) is the two-way random-effects, absolute-agreement "
        r"single-rater coefficient.",
        r"\item Kappa is undefined (``--'') for a flag with no variance in the overlap "
        r"(all images agreed absent).",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    TABS.mkdir(parents=True, exist_ok=True)
    (TABS / "tableS4_irr.tex").write_text("\n".join(lines) + "\n")
    print(f"  -> {TABS / 'tableS4_irr.tex'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities",
        type=str,
        default="mumbai,navi_mumbai,bangalore,delhi",
        help="Comma-separated list of cities",
    )
    args = parser.parse_args()
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]

    all_pairs = []
    overlap_counts = {}
    print("=" * 60)
    print("STREETSCOPE: INTER-RATER RELIABILITY")
    print("=" * 60)

    for city in cities:
        df = load_all_annotations(city)
        pairs, primary = build_pairs(df)
        overlap_counts[city] = len(pairs)
        all_pairs.append(pairs)
        print(
            f"\n{CITY_LABELS.get(city, city)}  (primary={primary}, {len(pairs)} overlapping images)"
        )
        if len(pairs) == 0:
            print("  no doubly-annotated images")
            continue
        for col, label in CONTINUOUS:
            m = continuous_metrics(pairs, col)
            print(
                f"  {label:<18} n={m['n']:<5} ICC={_f(m['icc'])}  r={_f(m['r'])}  mean|Δ|={_f(m['mad'])}"
            )
        for col, label in BINARY:
            m = binary_metrics(pairs, col)
            ap = "--" if np.isnan(m["agree"]) else f"{100 * m['agree']:.1f}%"
            print(f"  {label:<18} n={m['n']:<5} agree={ap:<7} kappa={_f(m['kappa'])}")

    pooled = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    if len(pooled) == 0:
        print("\nNo overlaps found; nothing to write.")
        return

    print(f"\n{'=' * 60}\nPOOLED ({len(pooled)} overlaps)\n{'=' * 60}")
    pooled_cont = {}
    for col, label in CONTINUOUS:
        m = continuous_metrics(pooled, col)
        pooled_cont[col] = m
        print(
            f"  {label:<18} n={m['n']:<5} ICC={_f(m['icc'])}  r={_f(m['r'])}  mean|Δ|={_f(m['mad'])}"
        )
    pooled_bin = {}
    for col, label in BINARY:
        m = binary_metrics(pooled, col)
        pooled_bin[col] = m
        ap = "--" if np.isnan(m["agree"]) else f"{100 * m['agree']:.1f}%"
        print(f"  {label:<18} n={m['n']:<5} agree={ap:<7} kappa={_f(m['kappa'])}")

    print("\n" + "=" * 60)
    write_table(pooled_cont, pooled_bin, overlap_counts, "primary")
    print("=" * 60)


if __name__ == "__main__":
    main()
