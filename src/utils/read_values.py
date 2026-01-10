"""
    Read different values on column to understand the dataset.
    DuckDB
"""
import sys
from typing import Tuple
import duckdb


import sys
from pathlib import Path
from typing import Union, List, Optional
import duckdb


def read_parquet(path: Union[str, Path], columns: Union[str, List[str]] = "*", where: Optional[str] = None, **kwargs):
    con = duckdb.connect(database=":memory:")
    p = Path(path)

    # Decide how to read the file
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        source = f"read_parquet('{p.as_posix()}')"
    elif suffix in [".csv", ".gz"]:
        source = f"read_csv_auto('{p.as_posix()}', union_by_name=true)"
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

    print("=" * 80)
    print(f"Inspecting: {p}")
    print("=" * 80)

    if isinstance(columns, (list, tuple)):
        columns = ", ".join(columns)

    # Build optional WHERE clause
    where_clause = f"WHERE {where}" if where else ""

    print("Columns:", columns)
    print("Values: ")
    print(
        con.execute(f"""
        SELECT DISTINCT {columns}
        FROM {source}
        {where_clause}
        LIMIT 1000
        """
        ).fetchdf()
    )

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python read_values.py <path/to/file[.csv,.parquet,.gz]>")
        exit(1)

    pth = args[0]
    colu = args[1:]

    read_parquet(pth, colu)
