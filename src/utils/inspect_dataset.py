"""
Quick inspection utility for tabular datasets (Parquet / CSV / CSV.GZ).

Usage:
    python -m src.utils.inspect_dataset path/to/file.parquet
    python -m src.utils.inspect_dataset path/to/file.csv.gz
"""

import sys
from pathlib import Path
import duckdb


def inspect_dataset(path: Path, n_rows: int = 5) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    con = duckdb.connect(database=":memory:")

    # Decide how to read the file
    if path.suffix == ".parquet":
        source = f"read_parquet('{path.as_posix()}')"
    elif path.suffix in [".csv", ".gz"]:
        source = f"read_csv_auto('{path.as_posix()}', union_by_name=true)"
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print("=" * 80)
    print(f"Inspecting: {path}")
    print("=" * 80)

    # Row / column counts
    print("\nBasic shape:")
    print(
        con.execute(f"""
        SELECT
          COUNT(*) AS n_rows
        FROM {source};
        """).fetchdf()
    )

    # Column info
    print("\nColumns:")
    cols = con.execute(f"""
        DESCRIBE SELECT * FROM {source};
    """).fetchdf()
    print(cols)

    # Head
    print(f"\nFirst {n_rows} rows:")
    print(
        con.execute(f"""
        SELECT * FROM {source}
        LIMIT {n_rows};
        """).fetchdf()
    )

    # Missingness
    # print("\nMissingness (fraction of NULLs per column):")
    # for col in cols["column_name"]:
    #     q = f"""
    #     SELECT
    #       1 - (COUNT({col})::DOUBLE / COUNT(*)) AS null_frac
    #     FROM {source};
    #     """
    #     null_frac = con.execute(q).fetchone()[0]
    #     print(f"{col:30s}: {null_frac:.3f}")

    print("\nMissingness (fraction of NULLs per column):")

    col_names = list(cols["column_name"])
    # Build a single aggregate query that computes null fraction per column
    exprs = []
    for c in col_names:
        safe = c.replace('"', '""')
        exprs.append(f"SUM(CASE WHEN \"{safe}\" IS NULL THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS \"{safe}\"")

    q = f"SELECT {', '.join(exprs)} FROM {source};"
    row = con.execute(q).fetchone()

    for name, null_frac in zip(col_names, row):
        print(f"{name:30s}: {null_frac:.3f}")

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.inspect_dataset <path_to_dataset>")
        sys.exit(1)

    dataset_path = Path(sys.argv[1]).resolve()
    inspect_dataset(dataset_path)
