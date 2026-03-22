from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Logistic sigmoid.
    Parameters
    ----------
    x : np.ndarray
        Input array.
    Returns
    -------
    np.ndarray
        Sigmoid of the input array.
    """
    return 1.0 / (1.0 + np.exp(-x))


def compute_artificial_propensity(
    X_scaled: pd.DataFrame,
    weights: dict[str, float],
    intercept: float,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the artificial propensity score.

    Parameters
    ----------
    X_scaled : pd.DataFrame
        Standardized covariates.
    weights : dict[str, float]
        Per-covariate weights used in the artificial treatment policy.
    intercept : float
        Intercept added to the linear predictor.
    clip_min : float
        Lower clipping bound for the propensity score.
    clip_max : float
        Upper clipping bound for the propensity score.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Raw propensity score and clipped propensity score.

    Raises
    ------
    ValueError
        If weights are empty or refer to unknown covariates.
    """
    if not weights:
        raise ValueError("Weights dictionary is empty. Cannot build propensity score.")

    linear_predictor = np.full(shape=len(X_scaled), fill_value=intercept, dtype=float)

    for col, weight in weights.items():
        if col not in X_scaled.columns:
            raise ValueError(f"Weight specified for unknown covariate: {col}")
        linear_predictor += weight * X_scaled[col].values

    ps_raw = sigmoid(linear_predictor)
    ps = np.clip(ps_raw, clip_min, clip_max)

    return ps_raw, ps