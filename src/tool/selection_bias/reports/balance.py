from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def _safe_float(value: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
    return float(value)


def standardized_mean_difference(
    series_before: pd.Series,
    series_after: pd.Series,
) -> float | None:
    """
    Compute the standardized mean difference (SMD) between two samples.

    Parameters
    ----------
    series_before : pd.Series
        Reference sample.
    series_after : pd.Series
        Comparison sample.

    Returns
    -------
    float | None
        Standardized mean difference, or None if it cannot be computed.
    """
    x1 = series_before.dropna().to_numpy(dtype=float)
    x2 = series_after.dropna().to_numpy(dtype=float)

    if len(x1) == 0 or len(x2) == 0:
        return None

    mean1 = float(np.mean(x1))
    mean2 = float(np.mean(x2))
    var1 = float(np.var(x1, ddof=0))
    var2 = float(np.var(x2, ddof=0))

    pooled_sd = math.sqrt((var1 + var2) / 2.0)

    if pooled_sd == 0.0:
        if mean1 == mean2:
            return 0.0
        return None

    return float((mean2 - mean1) / pooled_sd)


def compute_balance_table(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    columns: Sequence[str],
) -> dict:
    """
    Compute pre/post balance diagnostics for a set of columns.

    Parameters
    ----------
    df_before : pd.DataFrame
        Reference dataframe (e.g. eligible sample).
    df_after : pd.DataFrame
        Comparison dataframe (e.g. selected sample).
    columns : Sequence[str]
        Columns to compare.

    Returns
    -------
    dict
        Per-column balance diagnostics.
    """
    balance = {}

    for col in columns:
        if col not in df_before.columns or col not in df_after.columns:
            balance[col] = {"error": "column_not_found"}
            continue

        s1 = df_before[col]
        s2 = df_after[col]

        mean_before = _safe_float(s1.dropna().mean()) if not s1.dropna().empty else None
        mean_after = _safe_float(s2.dropna().mean()) if not s2.dropna().empty else None

        balance[col] = {
            "mean_before": mean_before,
            "mean_after": mean_after,
            "mean_difference": (
                _safe_float(mean_after - mean_before)
                if mean_before is not None and mean_after is not None
                else None
            ),
            "smd": standardized_mean_difference(s1, s2),
            "missing_before": int(s1.isna().sum()),
            "missing_after": int(s2.isna().sum()),
        }

    return balance


def summarize_absolute_smd(balance_table: dict) -> dict:
    """
    Summarize the absolute SMD values from a balance table.

    Parameters
    ----------
    balance_table : dict
        Output of compute_balance_table.

    Returns
    -------
    dict
        Aggregate SMD summary.
    """
    smds = [
        abs(item["smd"])
        for item in balance_table.values()
        if isinstance(item, dict) and item.get("smd") is not None
    ]

    if not smds:
        return {
            "n_columns_with_valid_smd": 0,
            "mean_absolute_smd": None,
            "max_absolute_smd": None,
        }

    return {
        "n_columns_with_valid_smd": int(len(smds)),
        "mean_absolute_smd": float(np.mean(smds)),
        "max_absolute_smd": float(np.max(smds)),
    }