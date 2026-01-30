# Check Dataset HFNC vs NIV

import pandas as pd
from src.config import ANALYTIC_DIR

PARQUET_PATH = ANALYTIC_DIR / "analytic_resp_v1.parquet"

def sanity_check():
    df = pd.read_parquet(PARQUET_PATH)

    # Total stays
    n_total = len(df)

    # Outcome rates and counts per treatment group
    mask_hfnc = df["t0_support"] == 'HFNC'
    mask_niv = df["t0_support"] == 'NIV'

    t1 = mask_hfnc.sum()
    t0 = mask_niv.sum()

    outcome_t1 = df.loc[mask_hfnc, "y_intub_48h"].mean() if t1 > 0 else float("nan")
    outcome_t0 = df.loc[mask_niv, "y_intub_48h"].mean() if t0 > 0 else float("nan")

    # Overall intubation rate
    intub_rate_48h = df["y_intub_48h"].mean()

    print("=== Respiratory Dataset Statistics ===")
    print(f"- Total stays: {n_total}")
    print(f"- Stays with HFNC (t0_support=1): {t1}")# ({t1/n_total:.2%})")
    print(f"- Stays with NIV (t0_support=0): {t0}")#({t0/n_total:.2%})")
    print(f"- Overall intubation rate within 48h: {intub_rate_48h:.2%}")
    print(f"- Intubation rate within 48h for HFNC group: {outcome_t1:.2%}")
    print(f"- Intubation rate within 48h for NIV group: {outcome_t0:.2%}")

    for c in df.columns:
        n_missing = df[c].isna().sum()
        if n_missing > 0:
            print(f"- Column '{c}' has {n_missing} missing values ({n_missing / n_total:.2%})")
    print("No printed columns have no missing values.")

    print("All columns names:")
    # print cols 10 per line
    for i in range(0, len(df.columns), 10):
        print("  " + ", ".join(df.columns[i:i+10]))

    print("=== End of Sanity Check ===")

if __name__ =="__main__":
    sanity_check()