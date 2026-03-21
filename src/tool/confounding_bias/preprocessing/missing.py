from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from .validation import validate_required_inputs


def select_required_columns(
    df: pd.DataFrame,
    covariates: Sequence[str],
    outcome_col: str,
    original_treatment_col: str,
    id_column: str | None = "id"
) -> pd.DataFrame:
    """
    Select only the columns required for the confounding-bias transformation.
    
    """
    required = list(covariates) + [outcome_col, original_treatment_col]
    if id_column is not None:
        required = [id_column] + required
    required = list(dict.fromkeys(required))  # per evitare duplicati
    return df[required].copy()


def drop_missing_required_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Drop rows with missing values from the provided dataframe.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned dataframe and metadata.
    """
    before = int(len(df))
    cleaned_df = df.dropna().reset_index(drop=True)
    after = int(len(cleaned_df))

    logging.info("Rows before dropna: %d", before)
    logging.info("Rows after dropna:  %d", after)

    if after == 0:
        raise ValueError("No rows left after dropping missing values.")

    metadata = {
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
    id_column: str | None = "id",
) -> tuple[pd.DataFrame, dict]:
    """
    Select the required columns and drop rows with missing values only on those columns.
    
    """
    validate_required_inputs(
        df=df,
        covariates=covariates,
        outcome_col=outcome_col,
        original_treatment_col=original_treatment_col,
        id_column=id_column,
    )

    work_df = select_required_columns(
        df=df,
        covariates=covariates,
        outcome_col=outcome_col,
        original_treatment_col=original_treatment_col,
        id_column=id_column,
    )

    work_df, metadata = drop_missing_required_rows(work_df)
    return work_df, metadata