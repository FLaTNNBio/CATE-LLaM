from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from .validation import validate_required_inputs


def get_required_columns_for_cleaning(
    covariates: Sequence[str],
    outcome_col: str,
    original_treatment_col: str,
    id_column: str | None = None,
) -> list[str]:
    """
    Return the columns that must be present and non-missing for a row
    to be kept in the cleaned dataset.

    """
    required = list(covariates) + [outcome_col, original_treatment_col]

    if id_column is not None:
        required.append(id_column)

    # remove duplicates while preserving order
    required = list(dict.fromkeys(required))
    return required


def drop_missing_required_rows(
    df: pd.DataFrame,
    required_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Drop rows with missing values only on the required columns,
    while preserving all original dataframe columns.
    """
    before = int(len(df))

    mask_complete = df[list(required_columns)].notna().all(axis=1)
    cleaned_df = df.loc[mask_complete].reset_index(drop=True).copy()

    after = int(len(cleaned_df))

    logging.info("Rows before dropna: %d", before)
    logging.info("Rows after dropna:  %d", after)

    if after == 0:
        raise ValueError("No rows left after dropping missing values.")

    metadata = {
        "required_columns_for_cleaning": list(required_columns),
        "n_rows_before": before,
        "n_rows_after": after,
        "dropped_rows": before - after,
    }

    return cleaned_df, metadata


def select_and_clean_data(
    df: pd.DataFrame,
    covariates: Sequence[str],
    outcome_col: str,
    original_treatment_col: str,
    id_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate required columns and drop rows with missing values on the
    required columns only, while keeping all original dataset columns.
    """
    validate_required_inputs(
        df=df,
        covariates=covariates,
        outcome_col=outcome_col,
        original_treatment_col=original_treatment_col,
        id_column=id_column,
    )

    required_columns = get_required_columns_for_cleaning(
        covariates=covariates,
        outcome_col=outcome_col,
        original_treatment_col=original_treatment_col,
        id_column=id_column,
    )

    cleaned_df, metadata = drop_missing_required_rows(
        df=df,
        required_columns=required_columns,
    )

    return cleaned_df, metadata