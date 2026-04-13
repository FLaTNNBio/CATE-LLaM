from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

import pandas as pd


PathLike = Union[str, Path]

SUPPORTED_DATAFRAME_OUTPUT_SUFFIXES = {".csv", ".parquet"}
SUPPORTED_REPORT_SUFFIXES = {".json"}


def ensure_parent_dir(path: PathLike) -> Path:
    """
    Ensure the parent directory of a file path exists.

    Parameters
    ----------
    path : str | Path
        Target file path.

    Returns
    -------
    Path
        Normalized Path object.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_dataframe(
    df: pd.DataFrame,
    path: PathLike,
    *,
    index: bool = False,
) -> Path:
    """
    Save a dataframe to CSV or Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    path : str | Path
        Output file path.
    index : bool, default=False
        Whether to write the dataframe index.

    Returns
    -------
    Path
        Saved file path.

    Raises
    ------
    ValueError
        If the output format is unsupported.
    RuntimeError
        If saving fails.
    """
    if df is None:
        raise ValueError("Cannot save dataframe: received None")

    file_path = ensure_parent_dir(path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_DATAFRAME_OUTPUT_SUFFIXES:
        raise ValueError(
            f"Unsupported dataframe output format '{suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_DATAFRAME_OUTPUT_SUFFIXES)}"
        )

    try:
        if suffix == ".csv":
            df.to_csv(file_path, index=index)
        elif suffix == ".parquet":
            df.to_parquet(file_path, index=index)
        else:
            raise ValueError(f"Unhandled dataframe output format: {suffix}")
    except Exception as exc:
        raise RuntimeError(f"Failed to save dataframe to '{file_path}': {exc}") from exc

    return file_path


def save_json_report(
    report: Mapping[str, Any],
    path: PathLike,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    """
    Save a report dictionary as a JSON file.

    Parameters
    ----------
    report : Mapping[str, Any]
        Report content.
    path : str | Path
        Output JSON file path.
    indent : int, default=2
        JSON indentation level.
    sort_keys : bool, default=False
        Whether to sort keys in output JSON.

    Returns
    -------
    Path
        Saved file path.

    Raises
    ------
    ValueError
        If the report path does not end with .json.
    RuntimeError
        If saving fails.
    """
    if report is None:
        raise ValueError("Cannot save report: received None")

    file_path = ensure_parent_dir(path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_REPORT_SUFFIXES:
        raise ValueError(
            f"Unsupported report format '{suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_REPORT_SUFFIXES)}"
        )

    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
            f.write("\n")
    except Exception as exc:
        raise RuntimeError(f"Failed to save JSON report to '{file_path}': {exc}") from exc

    return file_path