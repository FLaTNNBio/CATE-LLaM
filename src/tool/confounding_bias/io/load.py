from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """
    Load the input dataset from a Parquet file.

    Parameters
    ----------
    path : Path
        Path to the input Parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file is not a Parquet file or if the loaded dataframe is empty.
    RuntimeError
        If pandas fails while reading the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {path}")

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Input file must be a .parquet file, got: {path.suffix}")

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Parquet file '{path}': {exc}") from exc

    if df.empty:
        raise ValueError("Input Parquet is empty.")

    return df