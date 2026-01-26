"""
Metrics for evaluating weighting methods in causal inference.
It includes:
- Stabilized IPTW weights computation: P(T=t)/P(T=t|X)
- Weight trimming based on quantiles: [q_lo, q_hi]
- Effective Sample Size (ESS) calculation: (sum of weights)^2 / sum of weights^2
- Standardized Mean Difference (SMD) computation: both unweighted and weighted
- Average Treatment Effect (ATE) estimation using weighted outcomes
- SMD table generation for multiple features before and after weighting

ESS: it indicates the equivalent number of independent samples represented by the weighted sample.
A higher ESS suggests that the weights are more balanced and less variable, leading to more reliable estimates.

SMD: it measures the difference in means between treatment groups relative to the pooled standard deviation.
A lower SMD indicates better covariate balance between treatment groups.

"""

from typing import Union

import numpy as np
import pandas as pd

def stabilized_iptw_weights(treat: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """
    Stabilized weights: P(T=t)/P(T=t|X)
    :param treat: treatment assignments
    :param ps: propensity scores P(T=1|X)
    :return: stabilized weights
    """
    p_t = treat.mean()
    w = np.empty_like(ps, dtype=float)
    w[treat == 1] = p_t / ps[treat == 1]
    w[treat == 0] = (1 - p_t) / (1 - ps[treat == 0])
    return w

def trim_weights(w: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    """
    Trim weights to be within quantile range [q_lo, q_hi].
    :param w: weights
    :param q_lo: quantile low bound
    :param q_hi: quantile high bound
    :return: numpy array of trimmed weights
    """
    lo, hi = np.quantile(w, [q_lo, q_hi])
    return np.clip(w, lo, hi)

def ess(weights: np.ndarray) -> float:
    """
    Effective Sample Size (ESS):
    It is calculated as (sum of weights)^2 / sum of weights^2
    :param weights: array-like weights
    :return: effective sample size
    """
    w = np.asarray(weights, dtype=float)
    return (w.sum() ** 2) / (np.sum(w ** 2) + 1e-12)

def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    """
    Compute weighted mean.
    :param x: data
    :param w: weights
    :return: returns weighted mean
    """
    return np.sum(w * x) / (np.sum(w) + 1e-12)

def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    """
    Compute weighted variance.
    :param x: data
    :param w: weights
    :return: returns weighted variance
    """
    mu = weighted_mean(x, w)
    return np.sum(w * (x - mu) ** 2) / (np.sum(w) + 1e-12)

def smd_unweighted(x_t: np.ndarray, x_c: np.ndarray) -> float:
    m1, m0 = np.nanmean(x_t), np.nanmean(x_c)
    v1, v0 = np.nanvar(x_t), np.nanvar(x_c)
    denom = np.sqrt(0.5 * (v1 + v0) + 1e-12)
    return (m1 - m0) / denom

def smd_weighted(x: np.ndarray, tau: np.ndarray, w: np.ndarray) -> float:
    x1, x0 = x[tau == 1], x[tau == 0]
    w1, w0 = w[tau == 1], w[tau == 0]

    m1, m0 = weighted_mean(x1, w1), weighted_mean(x0, w0)
    v1, v0 = weighted_var(x1, w1), weighted_var(x0, w0)
    denom = np.sqrt(0.5 * (v1 + v0) + 1e-12)
    return (m1 - m0) / denom

def ate_weighted(y_hat: np.ndarray, tau: np.ndarray, w: np.ndarray) -> float:
    y1 = weighted_mean(y_hat[tau == 1], w[tau == 1])
    y0 = weighted_mean(y_hat[tau == 0], w[tau == 0])
    return float(y1 - y0)

def smd_table(df_x: pd.DataFrame, treat: np.ndarray, w: Union[np.ndarray | None] = None,
              max_features: Union[int | None] = None) -> pd.DataFrame:
    """
    Compute SMD table for all features in df_x.
    SMD is computed before weighting and after weighting (if weights provided).
    SMD = standardized mean difference.
    :param df_x: dataframe of features
    :param treat: treatment assignments
    :param w: weights (optional)
    :param max_features: number of features to compute (optional)
    :return: dataframe with columns: feature, smd_pre, smd_post
    """
    rows = []
    cols = df_x.columns if max_features is None else df_x.columns[:max_features]

    for c in cols:
        x = df_x[c].to_numpy(dtype=float, copy=False)
        x1 = x[treat == 1]
        x0 = x[treat == 0]
        smd_pre = smd_unweighted(x1, x0)
        smd_post = np.nan
        if w is not None:
            smd_post = smd_weighted(x, treat, w)
        rows.append({"feature": c, "smd_pre": smd_pre, "smd_post": smd_post})

    out = pd.DataFrame(rows).sort_values("smd_pre", key=lambda s: np.abs(s), ascending=False)
    return out.reset_index(drop=True)
