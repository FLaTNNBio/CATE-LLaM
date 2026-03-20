from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


SUPPORTED_INPUT_SUFFIXES = {".csv", ".parquet"}


def load_dataframe(path: PathLike) -> pd.DataFrame:
    """
    Load a tabular dataset from CSV or Parquet.

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file extension is not supported.
    RuntimeError
        If pandas fails to load the file.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported input format '{suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_INPUT_SUFFIXES)}"
        )

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            # Defensive fallback, should never happen because of earlier validation
            raise ValueError(f"Unhandled input format: {suffix}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataframe from '{file_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Loaded dataframe is empty: {file_path}")

    return df


def infer_input_format(path: PathLike) -> str:
    """
    Infer the input file format from the file extension.

    Parameters
    ----------
    path : str | Path
        Input file path.

    Returns
    -------
    str
        One of: 'csv', 'parquet'.

    Raises
    ------
    ValueError
        If the extension is unsupported.
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"

    raise ValueError(
        f"Unsupported input format '{suffix}'. "
        f"Supported formats: {sorted(SUPPORTED_INPUT_SUFFIXES)}"
    )