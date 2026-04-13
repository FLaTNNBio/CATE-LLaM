from src.config import PROJECT_ROOT
import pandas as pd

IN_PATH = PROJECT_ROOT / "data" / "analytic" / "analytic_v0_extended_prepared.parquet"
OUT_CSV = PROJECT_ROOT / "data" / "analytic" / "analytic_v0_extended_prepared.csv"

df = pd.read_parquet(IN_PATH)
df.to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV, "shape:", df.shape)
