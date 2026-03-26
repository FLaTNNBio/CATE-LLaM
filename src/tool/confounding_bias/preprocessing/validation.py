from __future__ import annotations

from typing import Sequence

import pandas as pd


def validate_dataframe_not_none(df: pd.DataFrame) -> None:
    """
    Validate that the input dataframe is not None.
    """
    if df is None:
        raise ValueError("Input dataframe is None.")


def validate_dataframe_not_empty(df: pd.DataFrame) -> None:
    """
    Validate that the input dataframe is not empty.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")


def validate_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    """
    Validate that all required columns are present in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_cols : Sequence[str]
        Columns that must exist in df.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_duplicate_columns(columns: Sequence[str], *, name: str = "columns") -> None:
    """
    Validate that a sequence of column names does not contain duplicates.
    """
    seen = set()
    duplicates = []

    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)

    if duplicates:
        raise ValueError(f"Duplicate entries found in {name}: {duplicates}")


def validate_required_inputs(
    df: pd.DataFrame,
    covariates: Sequence[str],
    outcome_col: str,
    original_treatment_col: str,
    id_column: str | None = None,
) -> None:
    validate_dataframe_not_none(df)
    validate_dataframe_not_empty(df)

    if not covariates:
        raise ValueError("At least one covariate must be provided.")

    validate_no_duplicate_columns(covariates, name="covariates")

    required = list(covariates) + [outcome_col, original_treatment_col]
    if id_column is not None:
        required.append(id_column)

    validate_columns(df, required)