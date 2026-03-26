from __future__ import annotations

from typing import Sequence

import pandas as pd


def _is_binary_series(series: pd.Series) -> bool:
    observed = set(series.dropna().unique().tolist())
    return observed.issubset({0, 1})


def summarize_binary_column(series: pd.Series) -> dict:
    """
    Summarize a binary column.
    """
    non_missing = series.dropna()
    n_non_missing = int(non_missing.shape[0])
    n_missing = int(series.isna().sum())

    if n_non_missing == 0:
        prevalence = None
        value_counts = {}
    else:
        prevalence = float(non_missing.mean()) if _is_binary_series(non_missing) else None
        value_counts = {
            str(k): int(v) for k, v in non_missing.value_counts().to_dict().items()
        }

    return {
        "n_non_missing": n_non_missing,
        "n_missing": n_missing,
        "missing_rate": float(n_missing / len(series)) if len(series) > 0 else 0.0,
        "prevalence_of_1": prevalence,
        "value_counts": value_counts,
    }


def summarize_numeric_column(series: pd.Series) -> dict:
    """
    Summarize a numeric column.
    """
    non_missing = series.dropna()
    n_non_missing = int(non_missing.shape[0])
    n_missing = int(series.isna().sum())

    if n_non_missing == 0:
        return {
            "n_non_missing": 0,
            "n_missing": n_missing,
            "missing_rate": float(n_missing / len(series)) if len(series) > 0 else 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "n_non_missing": n_non_missing,
        "n_missing": n_missing,
        "missing_rate": float(n_missing / len(series)) if len(series) > 0 else 0.0,
        "mean": float(non_missing.mean()),
        "std": float(non_missing.std(ddof=0)),
        "min": float(non_missing.min()),
        "max": float(non_missing.max()),
    }


def summarize_dataset(
    df: pd.DataFrame,
    *,
    treatment_column: str | None = None,
    outcome_column: str | None = None,
    probability_column: str | None = None,
    include_covariate_summaries: bool = False,
    covariates: Sequence[str] | None = None,
) -> dict:
    """
    Summarize a dataset used in the selection-bias pipeline.
    """
    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "total_missing_values": int(df.isna().sum().sum()),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
    }

    if treatment_column is not None and treatment_column in df.columns:
        summary["treatment"] = summarize_binary_column(df[treatment_column])

    if outcome_column is not None and outcome_column in df.columns:
        summary["outcome"] = summarize_binary_column(df[outcome_column])

    if probability_column is not None and probability_column in df.columns:
        summary["selection_probability"] = summarize_numeric_column(df[probability_column])

    if include_covariate_summaries:
        if covariates is None:
            raise ValueError(
                "covariates must be provided when include_covariate_summaries=True."
            )

        covariate_summaries = {}
        for col in covariates:
            if col not in df.columns:
                covariate_summaries[col] = {"error": "column_not_found"}
                continue

            if _is_binary_series(df[col].dropna()):
                covariate_summaries[col] = summarize_binary_column(df[col])
            else:
                covariate_summaries[col] = summarize_numeric_column(df[col])

        summary["covariates"] = covariate_summaries

    return summary