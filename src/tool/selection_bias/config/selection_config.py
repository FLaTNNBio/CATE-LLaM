from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


from .defaults import (
    DEFAULT_INCLUDE_TREATMENT_IN_SELECTION,
    DEFAULT_INDICATOR_COLUMN,
    DEFAULT_KEEP_SELECTION_COLUMNS,
    DEFAULT_LINEAR_SCORE_COLUMN,
    DEFAULT_MISSING_POLICY,
    DEFAULT_PROBABILITY_COLUMN,
    DEFAULT_SAVE_ANNOTATED_DATASET,
    DEFAULT_SCALED_SUFFIX,
    DEFAULT_SEED,
    DEFAULT_SELECTION_STRENGTH,
    DEFAULT_TARGET_INCLUSION_RATE,
    DEFAULT_TREATMENT_WEIGHT,
    DEFAULT_VALIDATE_TREATMENT_AS_BINARY,
    SUPPORTED_MISSING_POLICIES,
)


@dataclass
class SelectionBiasConfig:
    """
    Configuration for the pre-treatment selection bias tool.
    """

    input_path: str | Path
    output_selected_path: str | Path
    output_report_path: str | Path

    covariates: list[str]

    treatment_column: str | None = None
    outcome_column: str | None = None

    output_annotated_path: str | Path | None = None

    missing_policy: str = DEFAULT_MISSING_POLICY
    target_inclusion_rate: float = DEFAULT_TARGET_INCLUSION_RATE
    selection_strength: float = DEFAULT_SELECTION_STRENGTH
    seed: int = DEFAULT_SEED

    include_treatment_in_selection: bool = DEFAULT_INCLUDE_TREATMENT_IN_SELECTION
    treatment_weight: float = DEFAULT_TREATMENT_WEIGHT
    validate_treatment_as_binary: bool = DEFAULT_VALIDATE_TREATMENT_AS_BINARY

    keep_selection_columns: bool = DEFAULT_KEEP_SELECTION_COLUMNS
    save_annotated_dataset: bool = DEFAULT_SAVE_ANNOTATED_DATASET

    scaled_suffix: str = DEFAULT_SCALED_SUFFIX

    linear_score_column: str = DEFAULT_LINEAR_SCORE_COLUMN
    probability_column: str = DEFAULT_PROBABILITY_COLUMN
    indicator_column: str = DEFAULT_INDICATOR_COLUMN

    # Weights on ORIGINAL covariate names, not on scaled names.
    # Example: {"age": 0.8, "cd40": 1.2, "symptom": -0.5}
    feature_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_selected_path = Path(self.output_selected_path)
        self.output_report_path = Path(self.output_report_path)

        if self.output_annotated_path is not None:
            self.output_annotated_path = Path(self.output_annotated_path)

        self._validate_covariates()
        self._validate_missing_policy()
        self._validate_target_inclusion_rate()
        self._validate_selection_strength()
        self._validate_seed()
        self._validate_treatment_settings()
        self._validate_output_settings()
        self._validate_feature_weights()

    def _validate_covariates(self) -> None:
        if not self.covariates:
            raise ValueError("covariates must contain at least one column name.")

        if not all(isinstance(col, str) and col.strip() for col in self.covariates):
            raise ValueError("All covariates must be non-empty strings.")

        duplicates = []
        seen = set()
        for col in self.covariates:
            if col in seen and col not in duplicates:
                duplicates.append(col)
            seen.add(col)

        if duplicates:
            raise ValueError(f"Duplicate covariates found: {duplicates}")

    def _validate_missing_policy(self) -> None:
        if self.missing_policy not in SUPPORTED_MISSING_POLICIES:
            raise ValueError(
                f"Unsupported missing_policy '{self.missing_policy}'. "
                f"Supported policies: {sorted(SUPPORTED_MISSING_POLICIES)}"
            )

    def _validate_target_inclusion_rate(self) -> None:
        if not (0.0 < float(self.target_inclusion_rate) < 1.0):
            raise ValueError(
                "target_inclusion_rate must be strictly between 0 and 1."
            )

    def _validate_selection_strength(self) -> None:
        if float(self.selection_strength) <= 0.0:
            raise ValueError("selection_strength must be > 0.")

    def _validate_seed(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")

    def _validate_treatment_settings(self) -> None:
        if self.include_treatment_in_selection and not self.treatment_column:
            raise ValueError(
                "treatment_column must be provided when "
                "include_treatment_in_selection=True."
            )

    def _validate_output_settings(self) -> None:
        if self.save_annotated_dataset and self.output_annotated_path is None:
            raise ValueError(
                "output_annotated_path must be provided when "
                "save_annotated_dataset=True."
            )

        generated_columns = [
            self.linear_score_column,
            self.probability_column,
            self.indicator_column,
        ]

        if len(set(generated_columns)) != len(generated_columns):
            raise ValueError(
                "Generated output column names must be distinct: "
                f"{generated_columns}"
            )

    def _validate_feature_weights(self) -> None:
        if self.feature_weights is None:
            return

        unknown = [col for col in self.feature_weights if col not in self.covariates]
        if unknown:
            raise ValueError(
                "feature_weights contains columns not listed in covariates: "
                f"{unknown}"
            )

    def resolved_feature_weights(
        self,
        model_feature_columns: list[str],
    ) -> dict[str, float] | None:
        """
        Resolve weights from original covariate names to actual model feature columns.

        This is useful because after preprocessing the model may use:
        - scaled columns for continuous covariates, e.g. 'age_scaled'
        - original columns for binary covariates, e.g. 'symptom'

        Parameters
        ----------
        model_feature_columns : list[str]
            Actual columns used by the selection model.

        Returns
        -------
        dict[str, float] | None
            Weights keyed by model feature column names.
        """
        if self.feature_weights is None:
            return None

        resolved: dict[str, float] = {}

        for feature_col in model_feature_columns:
            if feature_col in self.feature_weights:
                resolved[feature_col] = float(self.feature_weights[feature_col])
                continue

            if feature_col.endswith(self.scaled_suffix):
                original_col = feature_col[: -len(self.scaled_suffix)]
                if original_col in self.feature_weights:
                    resolved[feature_col] = float(self.feature_weights[original_col])
                    continue

            # Fallback: if not explicitly specified, assign neutral weight 1.0
            resolved[feature_col] = 1.0

        return resolved

    def to_dict(self) -> dict:
        """
        Convert the config to a JSON-serializable dictionary.
        """
        return {
            "input_path": str(self.input_path),
            "output_selected_path": str(self.output_selected_path),
            "output_report_path": str(self.output_report_path),
            "output_annotated_path": (
                str(self.output_annotated_path)
                if self.output_annotated_path is not None
                else None
            ),
            "covariates": list(self.covariates),
            "treatment_column": self.treatment_column,
            "outcome_column": self.outcome_column,
            "missing_policy": self.missing_policy,
            "target_inclusion_rate": float(self.target_inclusion_rate),
            "selection_strength": float(self.selection_strength),
            "seed": int(self.seed),
            "include_treatment_in_selection": bool(self.include_treatment_in_selection),
            "treatment_weight": float(self.treatment_weight),
            "validate_treatment_as_binary": bool(self.validate_treatment_as_binary),
            "keep_selection_columns": bool(self.keep_selection_columns),
            "save_annotated_dataset": bool(self.save_annotated_dataset),
            "scaled_suffix": self.scaled_suffix,
            "linear_score_column": self.linear_score_column,
            "probability_column": self.probability_column,
            "indicator_column": self.indicator_column,
            "feature_weights": (
                {k: float(v) for k, v in self.feature_weights.items()}
                if self.feature_weights is not None
                else None
            ),
        }