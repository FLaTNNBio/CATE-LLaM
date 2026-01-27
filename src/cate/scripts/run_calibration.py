import argparse

from src.cate.calibration import cate_calibration_table, CalibrationConfig
import pandas as pd

# Add --data argument to pass path of df from cmd
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=False, help="Path to dr_tau_test.parquet", default="artifacts/cate/dr_tau_test.parquet")
args = ap.parse_args()

print(f"Loading data from {args.data}")

df = pd.read_parquet(args.data)
tab = cate_calibration_table(
    tau_hat=df["tau_hat"].values,
    y=df["y_hosp_mort"].values,
    t=df["t_vaso6h"].values,
    ps_hat=df["ps_hat"].values,
    cfg=CalibrationConfig(n_bins=10, ps_clip=(0.01, 0.99), method="ato"),
)

print("CATE calibration table (ATO weights):")
print(tab)

tab.to_csv("artifacts/cate/calibration_ato_deciles.csv", index=False)
print("Saved calibration table to artifacts/cate/calibration_ato_deciles.csv")