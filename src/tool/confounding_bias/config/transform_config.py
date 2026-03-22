from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .defaults import (
    DEFAULT_CLIP_MAX,
    DEFAULT_CLIP_MIN,
    DEFAULT_COVARIATES,
    DEFAULT_INTERCEPT,
    DEFAULT_NEW_TREATMENT,
    DEFAULT_ORIGINAL_TREATMENT,
    DEFAULT_OUTCOME,
    DEFAULT_PS_COL,
    DEFAULT_SEED,
)


@dataclass
class TransformConfig:
    input_path: Path
    output_parquet: Path
    output_report: Path

    covariates: list[str]

    outcome_col: str = DEFAULT_OUTCOME
    original_treatment_col: str = DEFAULT_ORIGINAL_TREATMENT
    new_treatment_col: str = DEFAULT_NEW_TREATMENT
    ps_col: str = DEFAULT_PS_COL

    clip_min: float = DEFAULT_CLIP_MIN
    clip_max: float = DEFAULT_CLIP_MAX
    intercept: float = DEFAULT_INTERCEPT
    seed: int = DEFAULT_SEED
    
    id_column: str | None = "id"

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_parquet = Path(self.output_parquet)
        self.output_report = Path(self.output_report)

        self._validate_covariates()
        self._validate_clip_bounds()
        self._validate_seed()
        self._validate_column_names()

    def _validate_covariates(self) -> None:
        if not self.covariates:
            raise ValueError("covariates must contain at least one column name.")

        if not all(isinstance(col, str) and col.strip() for col in self.covariates):
            raise ValueError("All covariates must be non-empty strings.")

        seen = set()
        duplicates = []
        for col in self.covariates:
            if col in seen and col not in duplicates:
                duplicates.append(col)
            seen.add(col)

        if duplicates:
            raise ValueError(f"Duplicate covariates found: {duplicates}")

    def _validate_clip_bounds(self) -> None:
        if not (0.0 < float(self.clip_min) < float(self.clip_max) < 1.0):
            raise ValueError(
                "clip_min and clip_max must satisfy 0 < clip_min < clip_max < 1."
            )

    def _validate_seed(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")

    def _validate_column_names(self) -> None:
        column_names = [
            self.outcome_col,
            self.original_treatment_col,
            self.new_treatment_col,
            self.ps_col,
        ]

        if not all(isinstance(col, str) and col.strip() for col in column_names):
            raise ValueError("Column name fields must be non-empty strings.")

    def to_dict(self) -> dict:
        return {
            "input_path": str(self.input_path),
            "output_parquet": str(self.output_parquet),
            "output_report": str(self.output_report),
            "id_column": self.id_column,
            "covariates": list(self.covariates),
            "outcome_col": self.outcome_col,
            "original_treatment_col": self.original_treatment_col,
            "new_treatment_col": self.new_treatment_col,
            "ps_col": self.ps_col,
            "clip_min": float(self.clip_min),
            "clip_max": float(self.clip_max),
            "intercept": float(self.intercept),
            "seed": int(self.seed),
        }


def default_transform_config(
    *,
    input_path: str | Path,
    output_parquet: str | Path,
    output_report: str | Path,
) -> TransformConfig:
    """
    Convenience factory using the project defaults.
    """
    return TransformConfig(
        input_path=Path(input_path),
        output_parquet=Path(output_parquet),
        output_report=Path(output_report),
        covariates=list(DEFAULT_COVARIATES),
    )