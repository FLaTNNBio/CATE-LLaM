from __future__ import annotations

from typing import Sequence

import pandas as pd


SUPPORTED_MISSING_POLICIES = {"drop", "mean-impute"}


def validate_missing_policy(policy: str) -> None:
    """
    Validate the missing-data policy.

    Parameters
    ----------
    policy : str
        Missing-data handling policy.

    Raises
    ------
    ValueError
        If the policy is not supported.
    """
    if policy not in SUPPORTED_MISSING_POLICIES:
        raise ValueError(
            f"Unsupported missing-data policy '{policy}'. "
            f"Supported policies: {sorted(SUPPORTED_MISSING_POLICIES)}"
        )


def summarize_missingness(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> dict:
    """
    Summarize missingness for a set of columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Sequence[str]
        Columns to inspect.

    Returns
    -------
    dict
        Missingness summary.
    """
    per_column = {}
    total_rows = int(len(df))

    for col in columns:
        n_missing = int(df[col].isna().sum())
        per_column[col] = {
            "n_missing": n_missing,
            "missing_rate": (n_missing / total_rows) if total_rows > 0 else 0.0,
        }

    rows_with_any_missing = int(df[list(columns)].isna().any(axis=1).sum())

    return {
        "n_rows": total_rows,
        "columns": list(columns),
        "rows_with_any_missing": rows_with_any_missing,
        "row_missing_rate": (rows_with_any_missing / total_rows) if total_rows > 0 else 0.0,
        "per_column": per_column,
    }


def apply_missing_policy(
    df: pd.DataFrame,
    columns: Sequence[str],
    policy: str = "drop",
) -> tuple[pd.DataFrame, dict]:
    """
    Apply a missing-data handling policy to the specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Sequence[str]
        Columns whose missing values should be handled.
    policy : str, default="drop"
        Missing-data handling policy. Supported:
        - 'drop'
        - 'mean-impute'

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Processed dataframe and metadata describing the operation.

    Raises
    ------
    ValueError
        If the policy is unsupported.
    """
    validate_missing_policy(policy)

    df_out = df.copy()
    before_n_rows = int(len(df_out))
    before_summary = summarize_missingness(df_out, columns)

    metadata = {
        "policy": policy,
        "before": before_summary,
        "after": None,
        "dropped_rows": 0,
        "imputed_columns": [],
        "imputation_values": {},
    }

    if policy == "drop":
        mask_complete = df_out[list(columns)].notna().all(axis=1)
        dropped_rows = int((~mask_complete).sum())
        df_out = df_out.loc[mask_complete].copy()

        metadata["dropped_rows"] = dropped_rows

    elif policy == "mean-impute":
        imputed_columns = []
        imputation_values = {}

        for col in columns:
            if df_out[col].isna().any():
                mean_value = float(df_out[col].mean())
                df_out[col] = df_out[col].fillna(mean_value)
                imputed_columns.append(col)
                imputation_values[col] = mean_value

        metadata["imputed_columns"] = imputed_columns
        metadata["imputation_values"] = imputation_values

    after_summary = summarize_missingness(df_out, columns)
    metadata["after"] = after_summary
    metadata["n_rows_before"] = before_n_rows
    metadata["n_rows_after"] = int(len(df_out))

    return df_out, metadata