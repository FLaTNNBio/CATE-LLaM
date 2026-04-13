from __future__ import annotations

from typing import Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype


def is_binary_series(series: pd.Series) -> bool:
    """
    Check whether a series is binary with observed values in {0, 1}.

    Parameters
    ----------
    series : pd.Series
        Input series.

    Returns
    -------
    bool
        True if the series is binary, False otherwise.
    """
    observed_values = set(series.dropna().unique().tolist())
    return observed_values.issubset({0, 1})


def standardize_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    skip_binary: bool = True,
    output_suffix: str = "_scaled",
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Standardize selected numeric columns using z-score scaling.

    New scaled columns are added to the dataframe; original columns are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : Sequence[str]
        Columns to standardize.
    skip_binary : bool, default=True
        Whether to skip binary columns.
    output_suffix : str, default="_scaled"
        Suffix added to scaled column names.

    Returns
    -------
    tuple[pd.DataFrame, list[str], dict]
        - dataframe with scaled columns added
        - list of scaled column names to use downstream
        - metadata with scaling statistics

    Raises
    ------
    ValueError
        If a requested column is non-numeric or has zero variance.
    """
    df_out = df.copy()

    scaled_columns: list[str] = []
    scaling_metadata = {
        "skip_binary": skip_binary,
        "output_suffix": output_suffix,
        "scaled_columns": [],
        "skipped_binary_columns": [],
        "statistics": {},
    }

    for col in columns:
        if not is_numeric_dtype(df_out[col]):
            raise ValueError(f"Cannot scale non-numeric column '{col}'.")

        if skip_binary and is_binary_series(df_out[col]):
            scaled_columns.append(col)
            scaling_metadata["skipped_binary_columns"].append(col)
            continue

        mean_value = float(df_out[col].mean())
        std_value = float(df_out[col].std(ddof=0))

        if std_value == 0.0:
            raise ValueError(
                f"Cannot standardize column '{col}' because its standard deviation is zero."
            )

        scaled_col = f"{col}{output_suffix}"
        df_out[scaled_col] = (df_out[col] - mean_value) / std_value

        scaled_columns.append(scaled_col)
        scaling_metadata["scaled_columns"].append(scaled_col)
        scaling_metadata["statistics"][col] = {
            "scaled_column": scaled_col,
            "mean": mean_value,
            "std": std_value,
        }

    return df_out, scaled_columns, scaling_metadata