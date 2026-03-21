from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_dataframe_not_none(df: pd.DataFrame) -> None:
    """
    Validate that the input dataframe is not None.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Raises
    ------
    ValueError
        If df is None.
    """
    if df is None:
        raise ValueError("Input dataframe is None.")


def validate_dataframe_not_empty(df: pd.DataFrame) -> None:
    """
    Validate that the input dataframe is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Raises
    ------
    ValueError
        If df is empty.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
) -> None:
    """
    Validate that all required columns are present in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_columns : Sequence[str]
        Columns that must exist in df.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def validate_no_duplicate_columns(columns: Sequence[str], *, name: str = "columns") -> None:
    """
    Validate that a sequence of column names does not contain duplicates.

    Parameters
    ----------
    columns : Sequence[str]
        Column names to validate.
    name : str, default="columns"
        Human-readable name used in the error message.

    Raises
    ------
    ValueError
        If duplicate column names are found.
    """
    seen = set()
    duplicates = []

    for col in columns:
        if col in seen and col not in duplicates:
            duplicates.append(col)
        seen.add(col)

    if duplicates:
        raise ValueError(f"Duplicate entries found in {name}: {duplicates}")


def validate_columns_not_all_missing(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """
    Validate that each specified column is not entirely missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Sequence[str]
        Columns to validate.

    Raises
    ------
    ValueError
        If one or more columns contain only missing values.
    """
    all_missing = [col for col in columns if df[col].isna().all()]
    if all_missing:
        raise ValueError(
            f"The following columns contain only missing values: {all_missing}"
        )


def validate_columns_not_constant(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """
    Validate that each specified column is not constant after excluding missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Sequence[str]
        Columns to validate.

    Raises
    ------
    ValueError
        If one or more columns are constant.
    """
    constant_columns = []

    for col in columns:
        non_missing = df[col].dropna()
        if non_missing.empty:
            continue
        if non_missing.nunique() <= 1:
            constant_columns.append(col)

    if constant_columns:
        raise ValueError(
            f"The following columns are constant and cannot be used for selection: "
            f"{constant_columns}"
        )


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """
    Validate that the specified columns are numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Sequence[str]
        Columns to validate.

    Raises
    ------
    ValueError
        If one or more columns are not numeric.
    """
    non_numeric = [col for col in columns if not is_numeric_dtype(df[col])]
    if non_numeric:
        raise ValueError(
            f"The following columns are not numeric: {non_numeric}. "
            "Selection covariates must be numeric or encoded before validation."
        )


def validate_binary_column(
    df: pd.DataFrame,
    column: str,
    *,
    allow_missing: bool = False,
) -> None:
    """
    Validate that a column is binary, optionally allowing missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name.
    allow_missing : bool, default=False
        Whether missing values are allowed.

    Raises
    ------
    ValueError
        If the column is not binary.
    """
    series = df[column]

    if not allow_missing and series.isna().any():
        raise ValueError(f"Binary column '{column}' contains missing values.")

    observed_values = set(series.dropna().unique().tolist())
    valid_values = {0, 1}

    if not observed_values.issubset(valid_values):
        raise ValueError(
            f"Column '{column}' is not binary. "
            f"Observed values: {sorted(observed_values)}"
        )


def validate_selection_inputs(
    df: pd.DataFrame,
    covariates: Sequence[str],
    treatment_column: str | None = None,
    outcome_column: str | None = None,
    require_numeric_covariates: bool = True,
    validate_treatment_as_binary: bool = False,
    id_column: str | None = None
) -> None:
    """
    Perform end-to-end validation for selection bias inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    covariates : Sequence[str]
        Covariates used in the selection model.
    treatment_column : str | None, default=None
        Optional treatment column.
    outcome_column : str | None, default=None
        Optional outcome column, used only for existence checks here.
    require_numeric_covariates : bool, default=True
        Whether all covariates must be numeric.
    validate_treatment_as_binary : bool, default=False
        Whether to check that the treatment column is binary.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    validate_dataframe_not_none(df)
    validate_dataframe_not_empty(df)

    if not covariates:
        raise ValueError("At least one covariate must be provided.")

    validate_no_duplicate_columns(covariates, name="covariates")

    required_columns = list(covariates)
    if treatment_column is not None:
        required_columns.append(treatment_column)
    if outcome_column is not None:
        required_columns.append(outcome_column)
    if id_column is not None:
        required_columns.append(id_column)

    validate_required_columns(df, required_columns)
    validate_columns_not_all_missing(df, covariates)
    validate_columns_not_constant(df, covariates)

    if require_numeric_covariates:
        validate_numeric_columns(df, covariates)

    if treatment_column is not None and validate_treatment_as_binary:
        validate_binary_column(df, treatment_column, allow_missing=False)