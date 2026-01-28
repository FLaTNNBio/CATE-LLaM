import argparse

from src.cate.calibration import cate_calibration_table, CalibrationConfig
import pandas as pd
from src.config import get_config, CONFIGS

# Add --data argument to pass path of df from cmd
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=False, help="Path to dr_tau_test.parquet", default="artifacts/cate/dr_tau_test.parquet")
ap.add_argument("--out_dir", required=False, help="Path to output directory")
ap.add_argument("--dataset", required=True,  choices=list(CONFIGS.keys()), default="rbc_v1", help="Which dataset config to use")

args = ap.parse_args()

cfg = get_config(args.dataset)

if args.data is None:
    args.data = cfg.data_path
if args.out_dir is None:
    args.out_dir = cfg.out_dir

print(f"Loading data from {args.data}")

df = pd.read_parquet(args.data)
tab = cate_calibration_table(
    tau_hat=df["tau_hat"].values,
    y=df[cfg.outcome_col].values,
    t=df[cfg.treatment_col].values,
    ps_hat=df["ps_hat"].values,
    cfg=CalibrationConfig(n_bins=10, ps_clip=(0.01, 0.99), method="ato"),
)

print("CATE calibration table (ATO weights):")
print(tab)

tab.to_csv("artifacts/cate/calibration_ato_deciles.csv", index=False)
print("Saved calibration table to artifacts/cate/calibration_ato_deciles.csv")