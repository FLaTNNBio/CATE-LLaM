# Count Missingness in Each Column of a DataFrame
import os

import pandas as pd

from src.config import ANALYTIC_DIR


def count_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of missing values in each column of the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame to check for missing values.

    Returns:
    pd.DataFrame: A DataFrame with columns 'column_name' and 'missing_count'.
    """
    missing_counts = df.isnull().sum()
    missing_df = pd.DataFrame({
        'column_name': missing_counts.index,
        'missing_count': missing_counts.values
    })
    return missing_df


if __name__ == "__main__":
    path = ANALYTIC_DIR / "analytic_rbc_v1.parquet"
    df = pd.read_parquet(path)

    print("Dataset general info:")
    print(df.info())

    print("Checking missingness in analytic_rbc_v1.parquet...")

    print(f"Reading data from {path}")
    missingness_df = count_missingness(df)
    print("Missingness in each column:")
    print(missingness_df)