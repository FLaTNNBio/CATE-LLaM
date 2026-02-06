import pandas as pd

from src.config import ANALYTIC_DIR

dataset = ANALYTIC_DIR / ("analytic_sepsis_early_diuretics_v1.parquet")
pd.read_parquet(dataset).info()

with pd.option_context('display.max_columns', None):
    df = pd.read_parquet(dataset)
    print(df.head(10))
    print(df.describe(include='all'))
    num_cols = df.select_dtypes(include='number').columns.tolist()
    print("Numeric columns:")
    for i in range(0, len(num_cols), 10):
        print("  " + ", ".join(num_cols[i:i+10]))

    print("\nAll column names:")
    # Print make cols 10 per line
    for i in range(0, len(df.columns), 10):
        print("  " + ", ".join(df.columns[i:i+10]))

    print("=== Missingness ===")

    n_total = len(df)
    for c in df.columns:
        n_missing = df[c].isna().sum()
        if n_missing > 0:
            print(f"- Column '{c}' has {n_missing} missing values ({n_missing / n_total:.2%})")
    print("No printed columns have no missing values.")

    treatment_col = "treat_early"
    # Show Number of treated vs untreated
    n_treated = df[treatment_col].sum()
    n_untreated = n_total - n_treated
    print(f"- Number of treated (treat_early=1): {n_treated} ({n_treated / n_total:.2%})")
    print(f"- Number of untreated (treat_early=0): {n_untreated} ({n_untreated / n_total:.2%})")

    # Show outcome rates per treatment group
    outcome_col = "y_28d_mort_inhosp"
    outcome_treated = df.loc[df[treatment_col] == 1, outcome_col].mean()
    outcome_untreated = df.loc[df[treatment_col] == 0, outcome_col].mean()
    print(f"- Overall outcome rate for outcome '{outcome_col}':")
    print(f"- Outcome rate (hospital mortality) for treated group: {outcome_treated:.2%}")
    print(f"- Outcome rate (hospital mortality) for untreated group: {outcome_untreated:.2%}")

    outcome_col2 = "y_hosp_mort"
    outcome_treated2 = df.loc[df[treatment_col] == 1, outcome_col2].mean()
    outcome_untreated2 = df.loc[df[treatment_col] == 0, outcome_col2].mean()
    print(f"- Overall outcome rate for outcome '{outcome_col2}':")
    print(f"- Outcome rate (hospital mortality) for treated group: {outcome_treated2:.2%}")
    print(f"- Outcome rate (hospital mortality) for untreated group: {outcome_untreated2:.2%}")
