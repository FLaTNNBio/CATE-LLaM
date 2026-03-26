from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def ensure_parent_dir(path: Path) -> Path:
    """
    Ensure that the parent directory of the given path exists.

    Parameters
    ----------
    path : Path
        Target file path.

    Returns
    -------
    Path
        The same normalized path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    """
    Save a dataframe to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save.
    path : Path
        Output Parquet path.
    index : bool, default=False
        Whether to save the dataframe index.

    Returns
    -------
    Path
        Saved file path.

    Raises
    ------
    ValueError
        If the path does not have a .parquet suffix or df is None.
    RuntimeError
        If saving fails.
    """
    if df is None:
        raise ValueError("Cannot save dataframe: received None.")

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Output dataset must be a .parquet file, got: {path.suffix}")

    output_path = ensure_parent_dir(path)

    try:
        df.to_parquet(output_path, index=index)
    except Exception as exc:
        raise RuntimeError(f"Failed to save Parquet file '{output_path}': {exc}") from exc

    return output_path


def save_json_report(
    report: Mapping[str, Any],
    path: Path,
    *,
    indent: int = 2,
) -> Path:
    """
    Save a report dictionary as JSON.

    Parameters
    ----------
    report : Mapping[str, Any]
        Report content.
    path : Path
        Output JSON path.
    indent : int, default=2
        JSON indentation level.

    Returns
    -------
    Path
        Saved file path.

    Raises
    ------
    ValueError
        If the path does not have a .json suffix or report is None.
    RuntimeError
        If saving fails.
    """
    if report is None:
        raise ValueError("Cannot save report: received None.")

    if path.suffix.lower() != ".json":
        raise ValueError(f"Output report must be a .json file, got: {path.suffix}")

    output_path = ensure_parent_dir(path)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=indent)
            f.write("\n")
    except Exception as exc:
        raise RuntimeError(f"Failed to save JSON report '{output_path}': {exc}") from exc

    return output_path