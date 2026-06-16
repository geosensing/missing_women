#!/usr/bin/env python3
"""
Design-based inference for the Streetscope project.

Thin adapter over the ``geoinference`` library (../geoinfer), which is the single
source of truth for every design-based estimate and standard error reported in
the paper. Frames collected along a video session are spatially and temporally
correlated, so all estimates cluster by ``base_video_id`` and use cluster-robust
standard errors (Horvitz-Thompson linearization with a t_{G-1} interval).

Two estimands are produced from one ``geoinfer.estimate()`` call:

- **weighted**  : person-weighted ratio  sum(women) / sum(people)  -> ``ratio``
- **unweighted**: image-level mean       mean(women_i / people_i)  -> ``photo_mean``

This module replaces the former hand-rolled ``clustered_inference.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from geoinference import PointDesign, estimate

WOMEN_COL = "total_women"
PEOPLE_COL = "total_people"
CLUSTER_COL = "base_video_id"


def _empty(n_obs: int, n_clusters: int) -> dict:
    return {
        "weighted": np.nan,
        "weighted_ci_lower": np.nan,
        "weighted_ci_upper": np.nan,
        "unweighted": np.nan,
        "unweighted_ci_lower": np.nan,
        "unweighted_ci_upper": np.nan,
        "icc": np.nan,
        "design_effect": np.nan,
        "n_obs": n_obs,
        "n_clusters": n_clusters,
    }


def summarize(
    data: pd.DataFrame,
    women_col: str = WOMEN_COL,
    people_col: str = PEOPLE_COL,
    cluster_col: str = CLUSTER_COL,
) -> dict:
    """Design-based summary of women's share for a (sub)set of frames.

    Filters to frames with at least one person, then runs ``geoinference.estimate``
    clustered by ``cluster_col``. Returns a flat dict with both the
    person-weighted ratio and the image-level mean, each with its cluster-robust
    95% CI, plus the clustering diagnostics (ICC, design effect). For a single
    cluster the point estimates are returned with NaN confidence intervals.
    """
    valid = data[data[people_col] > 0].dropna(subset=[women_col, people_col, cluster_col])
    n_obs = len(valid)
    n_clusters = valid[cluster_col].nunique()

    if n_obs == 0:
        return _empty(n_obs, n_clusters)

    result = estimate(
        valid,
        women_var=women_col,
        people_var=people_col,
        design=PointDesign(sampling="srs", cluster_var=cluster_col),
        bootstrap=False,
    )

    return {
        "weighted": result.ratio,
        "weighted_ci_lower": result.ratio_ci.recommended[0],
        "weighted_ci_upper": result.ratio_ci.recommended[1],
        "unweighted": result.photo_mean,
        "unweighted_ci_lower": result.photo_mean_ci.recommended[0],
        "unweighted_ci_upper": result.photo_mean_ci.recommended[1],
        "icc": result.diagnostics.icc,
        "design_effect": result.diagnostics.deff,
        "n_obs": result.n_obs,
        "n_clusters": result.n_clusters,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Design-based diagnostics per city")
    parser.add_argument("--cities", type=str, default="mumbai,navi_mumbai,bangalore,delhi")
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",")]
    root = Path(__file__).resolve().parents[1]

    for city in cities:
        path = root / "data" / city / "analysis_data.parquet"
        if not path.exists():
            print(f"{city}: no analysis_data.parquet, skipping")
            continue
        s = summarize(pd.read_parquet(path))
        print(f"\n{city.upper()}  (n={s['n_obs']:,}, clusters={s['n_clusters']})")
        print(
            f"  weighted   {s['weighted']:.3f} "
            f"[{s['weighted_ci_lower']:.3f}, {s['weighted_ci_upper']:.3f}]"
        )
        print(
            f"  unweighted {s['unweighted']:.3f} "
            f"[{s['unweighted_ci_lower']:.3f}, {s['unweighted_ci_upper']:.3f}]"
        )
        print(f"  ICC {s['icc']:.3f}   design effect {s['design_effect']:.2f}")
