#!/usr/bin/env python3
"""
Clustered Inference Functions for Missing Women Project
========================================================
Provides cluster-robust standard errors, block bootstrap, and mixed effects
models to account for spatial/temporal clustering within video sessions.

Key insight: Frames within the same video are spatially and temporally correlated.
Design effect ~2.5-3.0, ICC ~0.14 for prop_women by video.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM


def compute_icc(data: pd.DataFrame, outcome_col: str, cluster_col: str) -> dict:
    """
    Compute intraclass correlation coefficient (ICC) for an outcome by cluster.

    ICC = between-cluster variance / total variance

    Parameters
    ----------
    data : pd.DataFrame
        Data with outcome and cluster columns.
    outcome_col : str
        Column name for outcome variable.
    cluster_col : str
        Column name for cluster identifier.

    Returns
    -------
    dict
        Dictionary with 'icc', 'var_between', 'var_within', 'design_effect',
        'n_clusters', 'avg_cluster_size'.
    """
    df = data[[outcome_col, cluster_col]].dropna()
    if len(df) == 0:
        return {
            "icc": np.nan,
            "var_between": np.nan,
            "var_within": np.nan,
            "design_effect": np.nan,
            "n_clusters": 0,
            "avg_cluster_size": np.nan,
        }

    grand_mean = df[outcome_col].mean()
    cluster_means = df.groupby(cluster_col)[outcome_col].mean()
    cluster_sizes = df.groupby(cluster_col).size()

    n_clusters = len(cluster_means)
    n_total = len(df)
    avg_cluster_size = n_total / n_clusters

    sse_within = 0.0
    sse_between = 0.0

    for cluster in cluster_means.index:
        cluster_data = df[df[cluster_col] == cluster][outcome_col]
        cluster_mean = cluster_means[cluster]
        n_k = len(cluster_data)

        sse_within += ((cluster_data - cluster_mean) ** 2).sum()
        sse_between += n_k * (cluster_mean - grand_mean) ** 2

    ms_between = sse_between / (n_clusters - 1) if n_clusters > 1 else 0
    ms_within = sse_within / (n_total - n_clusters) if n_total > n_clusters else 0

    n_0 = (n_total - (cluster_sizes**2).sum() / n_total) / (n_clusters - 1)

    var_between = max(0, (ms_between - ms_within) / n_0)
    var_within = ms_within

    total_var = var_between + var_within
    icc = var_between / total_var if total_var > 0 else 0

    design_effect = 1 + (avg_cluster_size - 1) * icc

    return {
        "icc": icc,
        "var_between": var_between,
        "var_within": var_within,
        "design_effect": design_effect,
        "n_clusters": n_clusters,
        "avg_cluster_size": avg_cluster_size,
    }


def compute_cluster_robust_ci(
    data: pd.DataFrame,
    outcome_col: str,
    cluster_col: str,
    weight_col: str | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compute mean and cluster-robust 95% CI using OLS with cluster-robust SEs.

    Parameters
    ----------
    data : pd.DataFrame
        Data with outcome and cluster columns.
    outcome_col : str
        Column name for outcome variable.
    cluster_col : str
        Column name for cluster identifier.
    weight_col : str, optional
        Column name for weights (e.g., total_people for person-weighting).
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    dict
        Dictionary with 'mean', 'se_clustered', 'se_naive', 'ci_lower', 'ci_upper',
        'n_obs', 'n_clusters', 'design_effect_empirical'.
    """
    df = data[[outcome_col, cluster_col]].dropna()
    if weight_col:
        df = df.join(data[[weight_col]]).dropna()

    n_clusters = df[cluster_col].nunique()

    if len(df) < 2 or n_clusters < 2:
        if len(df) > 0:
            mean_val = df[outcome_col].mean()
        else:
            mean_val = np.nan
        return {
            "mean": mean_val,
            "se_clustered": np.nan,
            "se_naive": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": len(df),
            "n_clusters": n_clusters,
            "design_effect_empirical": np.nan,
        }

    y = df[outcome_col].values
    X = np.ones((len(y), 1))
    clusters = df[cluster_col].values

    if weight_col:
        weights = df[weight_col].values
        model = sm.WLS(y, X, weights=weights)
    else:
        model = sm.OLS(y, X)

    result_naive = model.fit()
    result_clustered = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})

    mean = result_clustered.params[0]
    se_clustered = result_clustered.bse[0]
    se_naive = result_naive.bse[0]

    t_crit = stats.t.ppf(1 - alpha / 2, result_clustered.df_resid)
    ci_lower = mean - t_crit * se_clustered
    ci_upper = mean + t_crit * se_clustered

    design_effect_empirical = (se_clustered / se_naive) ** 2 if se_naive > 0 else np.nan

    return {
        "mean": mean,
        "se_clustered": se_clustered,
        "se_naive": se_naive,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": len(df),
        "n_clusters": n_clusters,
        "design_effect_empirical": design_effect_empirical,
    }


def compute_group_comparison(
    data: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    cluster_col: str,
    weight_col: str | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compare means across groups with cluster-robust inference.

    Uses OLS with cluster-robust SEs:
        y = beta_0 + beta_1 * group + epsilon

    Parameters
    ----------
    data : pd.DataFrame
        Data with outcome, group, and cluster columns.
    outcome_col : str
        Column name for outcome variable.
    group_col : str
        Column name for binary group indicator (0/1 or False/True).
    cluster_col : str
        Column name for cluster identifier.
    weight_col : str, optional
        Column name for weights.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    dict
        Dictionary with group means, difference, SE, CI, p-value, t-stat.
    """
    df = data[[outcome_col, group_col, cluster_col]].dropna()
    if weight_col:
        df = df.join(data[[weight_col]]).dropna()

    df[group_col] = df[group_col].astype(float)
    n_clusters = df[cluster_col].nunique()

    if len(df) < 4 or df[group_col].nunique() < 2 or n_clusters < 2:
        return {
            "mean_0": np.nan,
            "mean_1": np.nan,
            "difference": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_obs": len(df),
            "n_clusters": n_clusters,
        }

    y = df[outcome_col].values
    X = sm.add_constant(df[group_col].values)
    clusters = df[cluster_col].values

    if weight_col:
        weights = df[weight_col].values
        model = sm.WLS(y, X, weights=weights)
    else:
        model = sm.OLS(y, X)

    result = model.fit(cov_type="cluster", cov_kwds={"groups": clusters})

    mean_0 = result.params[0]
    difference = result.params[1]
    mean_1 = mean_0 + difference
    se = result.bse[1]
    t_stat = result.tvalues[1]
    p_value = result.pvalues[1]

    t_crit = stats.t.ppf(1 - alpha / 2, result.df_resid)
    ci_lower = difference - t_crit * se
    ci_upper = difference + t_crit * se

    return {
        "mean_0": mean_0,
        "mean_1": mean_1,
        "difference": difference,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_obs": len(df),
        "n_clusters": len(np.unique(clusters)),
    }


def block_bootstrap_ci(
    data: pd.DataFrame,
    cluster_col: str,
    estimator: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """
    Compute block bootstrap confidence interval.

    Resamples entire clusters (videos) with replacement to preserve
    within-cluster correlation structure.

    Parameters
    ----------
    data : pd.DataFrame
        Data with cluster column.
    cluster_col : str
        Column name for cluster identifier.
    estimator : Callable
        Function that takes a DataFrame and returns a scalar estimate.
    n_bootstrap : int
        Number of bootstrap iterations (default 1000).
    alpha : float
        Significance level (default 0.05 for 95% CI).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with 'estimate', 'se', 'ci_lower', 'ci_upper', 'n_bootstrap'.
    """
    rng = np.random.default_rng(seed)
    clusters = data[cluster_col].unique()
    n_clusters = len(clusters)

    point_estimate = estimator(data)
    bootstrap_estimates = []

    for _ in range(n_bootstrap):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        boot_data = pd.concat(
            [data[data[cluster_col] == c] for c in sampled_clusters], ignore_index=True
        )
        try:
            est = estimator(boot_data)
            if np.isfinite(est):
                bootstrap_estimates.append(est)
        except Exception:
            continue

    if len(bootstrap_estimates) < 100:
        return {
            "estimate": point_estimate,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_bootstrap": len(bootstrap_estimates),
        }

    bootstrap_estimates = np.array(bootstrap_estimates)
    se = np.std(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

    return {
        "estimate": point_estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": len(bootstrap_estimates),
    }


def fit_mixed_effects(
    data: pd.DataFrame,
    formula: str,
    groups: str,
) -> dict:
    """
    Fit mixed effects model with random intercepts for clusters.

    Parameters
    ----------
    data : pd.DataFrame
        Data with variables in formula and groups column.
    formula : str
        Model formula (e.g., 'prop_women ~ is_weekend + C(city)').
    groups : str
        Column name for random effect groups (clusters).

    Returns
    -------
    dict
        Dictionary with model summary information.
    """
    df = data.dropna(subset=[groups])

    try:
        model = MixedLM.from_formula(formula, groups=groups, data=df)
        result = model.fit()

        return {
            "converged": result.converged,
            "params": result.params.to_dict(),
            "bse": result.bse.to_dict(),
            "pvalues": result.pvalues.to_dict(),
            "random_effects_var": result.cov_re.iloc[0, 0]
            if hasattr(result.cov_re, "iloc")
            else float(result.cov_re),
            "residual_var": result.scale,
            "icc": result.cov_re.iloc[0, 0]
            / (result.cov_re.iloc[0, 0] + result.scale)
            if hasattr(result.cov_re, "iloc")
            else float(result.cov_re) / (float(result.cov_re) + result.scale),
            "n_obs": result.nobs,
            "n_groups": result.nobs,
            "aic": result.aic,
            "bic": result.bic,
            "llf": result.llf,
        }
    except Exception as e:
        return {"error": str(e), "converged": False}


def compute_city_summary_with_clustering(
    data: pd.DataFrame,
    cities: list[str],
    cluster_col: str = "base_video_id",
) -> pd.DataFrame:
    """
    Compute city-level summary statistics with cluster-robust CIs.

    Parameters
    ----------
    data : pd.DataFrame
        Data with city, cluster, and outcome columns.
    cities : list[str]
        List of city names.
    cluster_col : str
        Column name for cluster identifier.

    Returns
    -------
    pd.DataFrame
        Summary table with mean, CI, and clustering diagnostics per city.
    """
    results = []

    for city in cities:
        sub = data[data["city"] == city]
        valid = sub[sub["total_people"] > 0]

        if len(valid) == 0:
            continue

        weighted = compute_cluster_robust_ci(
            valid, "prop_women", cluster_col, weight_col="total_people"
        )

        unweighted = compute_cluster_robust_ci(valid, "prop_women", cluster_col)

        icc_info = compute_icc(valid, "prop_women", cluster_col)

        results.append(
            {
                "city": city,
                "n_images": len(sub),
                "n_clusters": icc_info["n_clusters"],
                "avg_cluster_size": icc_info["avg_cluster_size"],
                "prop_women_weighted": weighted["mean"],
                "prop_women_weighted_ci_lower": weighted["ci_lower"],
                "prop_women_weighted_ci_upper": weighted["ci_upper"],
                "prop_women_unweighted": unweighted["mean"],
                "prop_women_unweighted_ci_lower": unweighted["ci_lower"],
                "prop_women_unweighted_ci_upper": unweighted["ci_upper"],
                "icc": icc_info["icc"],
                "design_effect": icc_info["design_effect"],
                "se_inflation": np.sqrt(weighted["design_effect_empirical"])
                if not np.isnan(weighted["design_effect_empirical"])
                else np.nan,
            }
        )

    return pd.DataFrame(results)


def run_diagnostics(data: pd.DataFrame, cluster_col: str = "base_video_id") -> None:
    """
    Print clustering diagnostics for the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Data with cluster column and prop_women.
    cluster_col : str
        Column name for cluster identifier.
    """
    valid = data[data["total_people"] > 0].copy()

    print("=" * 60)
    print("CLUSTERING DIAGNOSTICS")
    print("=" * 60)

    icc_info = compute_icc(valid, "prop_women", cluster_col)
    print(f"\nCluster column: {cluster_col}")
    print(f"N observations: {len(valid):,}")
    print(f"N clusters: {icc_info['n_clusters']:,}")
    print(f"Avg cluster size: {icc_info['avg_cluster_size']:.1f}")
    print(f"\nICC (prop_women): {icc_info['icc']:.3f}")
    print(f"Design effect: {icc_info['design_effect']:.2f}")
    print(f"SE inflation factor: {np.sqrt(icc_info['design_effect']):.2f}")

    naive = compute_cluster_robust_ci(valid, "prop_women", cluster_col)
    print(f"\nNaive SE: {naive['se_naive']:.4f}")
    print(f"Clustered SE: {naive['se_clustered']:.4f}")
    print(f"Empirical SE inflation: {np.sqrt(naive['design_effect_empirical']):.2f}")

    print(f"\n95% CI (naive): [{naive['mean'] - 1.96*naive['se_naive']:.3f}, "
          f"{naive['mean'] + 1.96*naive['se_naive']:.3f}]")
    print(f"95% CI (clustered): [{naive['ci_lower']:.3f}, {naive['ci_upper']:.3f}]")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run clustering diagnostics")
    parser.add_argument(
        "--cities",
        type=str,
        default="mumbai,navi_mumbai",
        help="Comma-separated list of cities",
    )
    args = parser.parse_args()

    cities = [c.strip() for c in args.cities.split(",")]

    ROOT = Path(__file__).resolve().parents[1]
    OUTPUT = ROOT / "data"

    dfs = []
    for city in cities:
        path = OUTPUT / city / "analysis_data.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["city"] = city
            dfs.append(df)

    if not dfs:
        print("No data found")
        exit(1)

    data = pd.concat(dfs, ignore_index=True)

    run_diagnostics(data)

    print("\n" + "=" * 60)
    print("CITY SUMMARY WITH CLUSTER-ROBUST CIs")
    print("=" * 60)

    summary = compute_city_summary_with_clustering(data, cities)
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("WEEKDAY VS WEEKEND COMPARISON")
    print("=" * 60)

    valid = data[(data["total_people"] > 0) & data["is_weekend"].notna()]
    comparison = compute_group_comparison(
        valid, "prop_women", "is_weekend", "base_video_id"
    )
    print(f"\nWeekday mean: {comparison['mean_0']:.3f}")
    print(f"Weekend mean: {comparison['mean_1']:.3f}")
    print(f"Difference (weekend - weekday): {comparison['difference']:.4f}")
    print(f"SE (cluster-robust): {comparison['se']:.4f}")
    print(f"95% CI: [{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}]")
    print(f"t-stat: {comparison['t_stat']:.2f}")
    print(f"p-value: {comparison['p_value']:.4f}")
