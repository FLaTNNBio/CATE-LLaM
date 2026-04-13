from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    """
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_mean_inclusion_probability(
    linear_score: np.ndarray,
    intercept: float,
    strength: float,
) -> float:
    """
    Compute the mean inclusion probability under a given intercept.
    """
    logits = intercept + strength * linear_score
    probs = sigmoid(logits)
    return float(probs.mean())


def calibrate_intercept(
    linear_score: pd.Series | np.ndarray,
    target_inclusion_rate: float,
    *,
    strength: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
    lower_bound: float = -50.0,
    upper_bound: float = 50.0,
) -> tuple[float, dict]:
    """
    Calibrate the intercept so that the mean inclusion probability matches
    the target inclusion rate.

    Parameters
    ----------
    linear_score : pd.Series | np.ndarray
        Linear predictor without intercept.
    target_inclusion_rate : float
        Desired mean inclusion probability, must be in (0, 1).
    strength : float, default=1.0
        Global multiplier applied to the linear score.
    max_iter : int, default=200
        Maximum number of bisection iterations.
    tol : float, default=1e-8
        Convergence tolerance on the mean probability.
    lower_bound : float, default=-50.0
        Lower search bound for the intercept.
    upper_bound : float, default=50.0
        Upper search bound for the intercept.

    Returns
    -------
    tuple[float, dict]
        Calibrated intercept and metadata.

    Raises
    ------
    ValueError
        If target_inclusion_rate is not in (0, 1).
    RuntimeError
        If calibration fails.
    """
    if not (0.0 < target_inclusion_rate < 1.0):
        raise ValueError(
            f"target_inclusion_rate must be in (0, 1), got {target_inclusion_rate}"
        )

    linear_score_np = np.asarray(linear_score, dtype=float)

    low = float(lower_bound)
    high = float(upper_bound)

    low_mean = compute_mean_inclusion_probability(linear_score_np, low, strength)
    high_mean = compute_mean_inclusion_probability(linear_score_np, high, strength)

    if not (low_mean <= target_inclusion_rate <= high_mean):
        raise RuntimeError(
            "Failed to bracket the target inclusion rate during intercept calibration. "
            f"Observed bracket means: low={low_mean:.6f}, high={high_mean:.6f}, "
            f"target={target_inclusion_rate:.6f}"
        )

    intercept = 0.0
    achieved_rate = None
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        intercept = (low + high) / 2.0
        achieved_rate = compute_mean_inclusion_probability(
            linear_score_np, intercept, strength
        )

        if abs(achieved_rate - target_inclusion_rate) <= tol:
            break

        if achieved_rate < target_inclusion_rate:
            low = intercept
        else:
            high = intercept
    else:
        raise RuntimeError(
            "Intercept calibration did not converge within the maximum number of iterations."
        )

    metadata = {
        "target_inclusion_rate": float(target_inclusion_rate),
        "achieved_mean_probability": float(achieved_rate),
        "strength": float(strength),
        "intercept": float(intercept),
        "iterations": int(n_iter),
        "tolerance": float(tol),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
    }

    return float(intercept), metadata