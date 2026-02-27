import pandas as pd
from pathlib import Path
from src.config import ANALYTIC_DIR

def add_id_col(input_path: Path, output_path: Path) -> None:
    df = pd.read_parquet(input_path)
    df.insert(0, "id", range(1, len(df) + 1))
    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    input_path = ANALYTIC_DIR / "aids" / "aids_rct.parquet"
    output_path = ANALYTIC_DIR / "aids" / "aids_rct_id.parquet"
    add_id_col(input_path, output_path)
    print(f"Added 'id' column to {input_path} and saved to {output_path}")