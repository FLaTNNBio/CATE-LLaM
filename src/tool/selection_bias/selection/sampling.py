from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    """
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_selection_probabilities(
    linear_score: pd.Series,
    intercept: float,
    *,
    strength: float = 1.0,
    output_column: str = "selection_probability",
) -> tuple[pd.DataFrame, dict]:
    """
    Compute selection probabilities from the linear score.

    Parameters
    ----------
    linear_score : pd.Series
        Linear predictor without intercept.
    intercept : float
        Intercept term.
    strength : float, default=1.0
        Global multiplier applied to the linear score.
    output_column : str, default="selection_probability"
        Name of the output probability column.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A one-column dataframe with probabilities and metadata.
    """
    logits = intercept + strength * linear_score.to_numpy(dtype=float)
    probs = sigmoid(logits)

    df_probs = pd.DataFrame(
        {output_column: probs},
        index=linear_score.index,
    )

    metadata = {
        "output_column": output_column,
        "probability_summary": {
            "mean": float(df_probs[output_column].mean()),
            "std": float(df_probs[output_column].std(ddof=0)),
            "min": float(df_probs[output_column].min()),
            "max": float(df_probs[output_column].max()),
        },
    }

    return df_probs, metadata


def sample_selection_indicator(
    probabilities: pd.Series,
    *,
    rng: np.random.Generator,
    output_column: str = "selection_indicator",
) -> tuple[pd.DataFrame, dict]:
    """
    Sample the binary selection indicator from Bernoulli probabilities.

    Parameters
    ----------
    probabilities : pd.Series
        Selection probabilities in [0, 1].
    rng : np.random.Generator
        Random number generator.
    output_column : str, default="selection_indicator"
        Name of the output selection column.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A one-column dataframe with sampled indicator and metadata.
    """
    probs = probabilities.to_numpy(dtype=float)

    if ((probs < 0.0) | (probs > 1.0)).any():
        raise ValueError("Probabilities must lie in [0, 1].")

    sampled = rng.binomial(n=1, p=probs, size=len(probs))

    df_sel = pd.DataFrame(
        {output_column: sampled.astype(int)},
        index=probabilities.index,
    )

    n_selected = int(df_sel[output_column].sum())
    n_total = int(len(df_sel))

    metadata = {
        "output_column": output_column,
        "n_rows": n_total,
        "n_selected": n_selected,
        "realized_inclusion_rate": (n_selected / n_total) if n_total > 0 else 0.0,
    }

    return df_sel, metadata