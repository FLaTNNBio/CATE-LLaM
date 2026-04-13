# Compare missingness between: v1 and v1 fixed
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
    path = ANALYTIC_DIR / "analytic_rbc_v1_f.parquet"
    path_fixed = ANALYTIC_DIR / "analytic_rbc_v1_fixed.parquet"

    df_v1 = pd.read_parquet(path)
    df_v1_fixed = pd.read_parquet(path_fixed)


    print("v1 Dataset general info:")
    print(df_v1.info())

    print("v1_fixed Dataset general info:")
    print(df_v1_fixed.info())

    # Compare missingness for each column and save it in a csv
    print("Comparison missingness...")
    missingness_v1 = count_missingness(df_v1)
    missingness_v1_fixed = count_missingness(df_v1_fixed)

    comparison_df = missingness_v1.merge(
        missingness_v1_fixed,
        on='column_name',
        suffixes=('_v1', '_v1_fixed')
    )

    comparison_df['missing_difference'] = comparison_df['missing_count_v1'] - comparison_df['missing_count_v1_fixed']
    comparison_path = ANALYTIC_DIR / "missingness_comparison_v1_vs_v1_fixed.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Missingness comparison saved to {comparison_path}")

