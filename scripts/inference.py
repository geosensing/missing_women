#!/usr/bin/env python3
"""Descriptive clustered inference for annotated Streetscope frames.

The sample consists of face-triggered frames along collected routes, not a
probability sample of a city's streets or residents. Estimates therefore describe
classified person-sightings in the collected imagery. Confidence intervals use a
collection-day cluster sandwich and a t distribution with G-1 degrees of freedom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t

WOMEN_COL = "women_count"
PEOPLE_COL = "total_pedestrians"
CLUSTER_COL = "collection_day"


def _empty(n_obs: int, n_clusters: int) -> dict[str, float | int]:
    return {
        "weighted": np.nan,
        "weighted_ci_lower": np.nan,
        "weighted_ci_upper": np.nan,
        "unweighted": np.nan,
        "unweighted_ci_lower": np.nan,
        "unweighted_ci_upper": np.nan,
        "n_obs": n_obs,
        "n_clusters": n_clusters,
    }


def _cluster_interval(
    estimate: float,
    residual: pd.Series,
    clusters: pd.Series,
    denominator: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Small-sample cluster-sandwich interval for a scalar estimating equation."""
    scores = residual.groupby(clusters, observed=True).sum()
    n_clusters = len(scores)
    if n_clusters < 2 or denominator <= 0:
        return np.nan, np.nan
    variance = n_clusters / (n_clusters - 1) * float(np.square(scores).sum()) / denominator**2
    critical = t.ppf(1 - alpha / 2, df=n_clusters - 1)
    half_width = critical * np.sqrt(variance)
    return max(0.0, estimate - half_width), min(1.0, estimate + half_width)


def summarize(
    data: pd.DataFrame,
    women_col: str = WOMEN_COL,
    people_col: str = PEOPLE_COL,
    cluster_col: str = CLUSTER_COL,
) -> dict[str, float | int]:
    """Summarize women's share among positive-count frames.

    ``weighted`` is the ratio of summed women to summed people. ``unweighted``
    is the mean frame-level share. Both intervals allow arbitrary dependence
    within collection days. They quantify sampling variability across observed
    fieldwork days; they do not turn the route imagery into a population sample.
    """
    required = [women_col, people_col, cluster_col]
    valid = data.dropna(subset=required).copy()
    valid = valid[valid[people_col] > 0]
    n_obs = len(valid)
    n_clusters = valid[cluster_col].nunique()
    if n_obs == 0:
        return _empty(n_obs, n_clusters)

    women = pd.to_numeric(valid[women_col], errors="raise").astype(float)
    people = pd.to_numeric(valid[people_col], errors="raise").astype(float)
    clusters = valid[cluster_col]

    weighted = float(women.sum() / people.sum())
    weighted_ci = _cluster_interval(
        weighted,
        women - weighted * people,
        clusters,
        float(people.sum()),
    )

    shares = women / people
    unweighted = float(shares.mean())
    unweighted_ci = _cluster_interval(
        unweighted,
        shares - unweighted,
        clusters,
        float(n_obs),
    )

    return {
        "weighted": weighted,
        "weighted_ci_lower": weighted_ci[0],
        "weighted_ci_upper": weighted_ci[1],
        "unweighted": unweighted,
        "unweighted_ci_lower": unweighted_ci[0],
        "unweighted_ci_upper": unweighted_ci[1],
        "n_obs": n_obs,
        "n_clusters": n_clusters,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Descriptive clustered summaries per city")
    parser.add_argument("--cities", type=str, default="mumbai,navi_mumbai,bangalore,delhi")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    for city in (value.strip() for value in args.cities.split(",")):
        path = root / "data" / city / "analysis_data.parquet"
        if not path.exists():
            print(f"{city}: no analysis_data.parquet, skipping")
            continue
        summary = summarize(pd.read_parquet(path))
        print(f"\n{city.upper()}  (n={summary['n_obs']:,}, days={summary['n_clusters']})")
        print(
            f"  person-weighted {summary['weighted']:.3f} "
            f"[{summary['weighted_ci_lower']:.3f}, {summary['weighted_ci_upper']:.3f}]"
        )
        print(
            f"  image mean     {summary['unweighted']:.3f} "
            f"[{summary['unweighted_ci_lower']:.3f}, {summary['unweighted_ci_upper']:.3f}]"
        )
