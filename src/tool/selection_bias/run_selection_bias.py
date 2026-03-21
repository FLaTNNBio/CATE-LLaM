"""
NOTE: This script is intended to be run from the command line. Example usage:
python -m selection_bias.run_selection_bias --input ../../data/analytic/aids/aids_obs.parquet --output-selected ../results/selection_bias/aids_obs_selected.parquet --output-report ../results/selection_bias/report.json --covariates age wtkg karnof oprior preanti strat cd40 symptom --treatment-column treat_obs --outcome-column label --target-inclusion-rate 0.70 --selection-strength 1.0 --seed 42 --feature-weights age=0.8 wtkg=-0.2 karnof=1.0 oprior=0.4 preanti=0.3 strat=-0.1 cd40=0.9 symptom=-0.5 --verbose

USER GUIDE:
This script applies a pre-treatment selection bias to a dataset. It takes an input dataset, applies a selection mechanism based on specified covariates and parameters, and outputs a selected dataset along with a JSON report.
The main steps are:
1. Parse CLI arguments to build a configuration object.
2. Load the input dataset.
3. Run the selection bias pipeline, which includes:
   - Validating inputs
   - Applying missing data policies
    - Standardizing covariates
    - Building a linear predictor for selection
    - Calibrating the intercept to achieve the target inclusion rate
    - Computing selection probabilities
    - Sampling selection indicators
4. Build a report summarizing the selection process and its impact on the dataset.
5. Save the selected dataset, annotated dataset (if configured), and JSON report to the specified paths.

This script is designed to be flexible and configurable, allowing users to specify various aspects of the selection mechanism and output. 
It also includes robust validation and error handling to ensure that inputs are appropriate for the selection process.

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from selection_bias.config import SelectionBiasConfig
from selection_bias.io import load_dataframe, save_dataframe, save_json_report
from selection_bias.reports import build_selection_report
from selection_bias.selection import run_selection_pipeline


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Apply pre-treatment selection bias to a dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-selected",
        required=True,
        help="Path to save the selected dataset (.csv or .parquet).",
    )
    parser.add_argument(
        "--output-report",
        required=True,
        help="Path to save the JSON report.",
    )
    parser.add_argument(
        "--output-annotated",
        default=None,
        help="Optional path to save the annotated dataset (.csv or .parquet).",
    )

    parser.add_argument(
        "--covariates",
        nargs="+",
        required=True,
        help="Covariates used in the selection model.",
    )
    parser.add_argument(
        "--treatment-column",
        default=None,
        help="Optional treatment column.",
    )
    parser.add_argument(
        "--outcome-column",
        default=None,
        help="Optional outcome column (used for reporting).",
    )

    parser.add_argument(
        "--missing-policy",
        default="drop",
        choices=["drop", "mean-impute"],
        help="Missing-data handling policy for selection covariates.",
    )
    parser.add_argument(
        "--target-inclusion-rate",
        type=float,
        default=0.70,
        help="Target mean inclusion probability in (0, 1).",
    )
    parser.add_argument(
        "--selection-strength",
        type=float,
        default=1.0,
        help="Global strength multiplier for the selection score.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--include-treatment-in-selection",
        action="store_true",
        help="Include treatment in the selection model.",
    )
    parser.add_argument(
        "--treatment-weight",
        type=float,
        default=0.0,
        help="Weight for treatment if included in the selection model.",
    )
    parser.add_argument(
        "--no-validate-treatment-as-binary",
        action="store_true",
        help="Disable binary validation for the treatment column.",
    )

    parser.add_argument(
        "--drop-selection-columns",
        action="store_true",
        help="Drop generated selection columns from the selected output dataset.",
    )
    parser.add_argument(
        "--no-save-annotated",
        action="store_true",
        help="Do not save the annotated dataset, even if --output-annotated is provided.",
    )

    parser.add_argument(
        "--linear-score-column",
        default="selection_linear_score",
        help="Name of the generated linear score column.",
    )
    parser.add_argument(
        "--probability-column",
        default="selection_probability",
        help="Name of the generated selection probability column.",
    )
    parser.add_argument(
        "--indicator-column",
        default="selection_indicator",
        help="Name of the generated selection indicator column.",
    )
    parser.add_argument(
        "--scaled-suffix",
        default="_scaled",
        help="Suffix used for scaled continuous covariates.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_feature_weights(raw_weights: list[str] | None) -> dict[str, float] | None:
    """
    Parse feature weights from CLI tokens like:
        age=0.8 cd40=1.2 symptom=-0.5

    Parameters
    ----------
    raw_weights : list[str] | None
        Raw CLI tokens.

    Returns
    -------
    dict[str, float] | None
        Parsed mapping, or None if no weights were provided.

    Raises
    ------
    ValueError
        If parsing fails.
    """
    if not raw_weights:
        return None

    weights: dict[str, float] = {}
    for item in raw_weights:
        if "=" not in item:
            raise ValueError(
                f"Invalid feature weight '{item}'. Expected format: column=value"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Invalid feature weight '{item}': empty column name")

        try:
            weights[key] = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value in feature weight '{item}'"
            ) from exc

    return weights


def build_config_from_args(args: argparse.Namespace) -> SelectionBiasConfig:
    """
    Build the SelectionBiasConfig object from parsed CLI arguments.
    """
    feature_weights = parse_feature_weights(getattr(args, "feature_weights", None))

    save_annotated_dataset = not args.no_save_annotated
    output_annotated_path = args.output_annotated

    if save_annotated_dataset and output_annotated_path is None:
        # Sensible default: place annotated output next to selected output.
        selected_path = Path(args.output_selected)
        output_annotated_path = selected_path.with_name(
            f"{selected_path.stem}_annotated{selected_path.suffix}"
        )

    config = SelectionBiasConfig(
        input_path=args.input,
        output_selected_path=args.output_selected,
        output_report_path=args.output_report,
        output_annotated_path=output_annotated_path,
        covariates=args.covariates,
        treatment_column=args.treatment_column,
        outcome_column=args.outcome_column,
        missing_policy=args.missing_policy,
        target_inclusion_rate=args.target_inclusion_rate,
        selection_strength=args.selection_strength,
        seed=args.seed,
        include_treatment_in_selection=args.include_treatment_in_selection,
        treatment_weight=args.treatment_weight,
        validate_treatment_as_binary=not args.no_validate_treatment_as_binary,
        keep_selection_columns=not args.drop_selection_columns,
        save_annotated_dataset=save_annotated_dataset,
        scaled_suffix=args.scaled_suffix,
        linear_score_column=args.linear_score_column,
        probability_column=args.probability_column,
        indicator_column=args.indicator_column,
        feature_weights=feature_weights,
    )

    return config


def main() -> None:
    parser = build_parser()

    parser.add_argument(
        "--feature-weights",
        nargs="*",
        default=None,
        help=(
            "Optional per-covariate weights in the form col=value. "
            "Example: --feature-weights age=0.8 cd40=1.2 symptom=-0.5"
        ),
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    LOGGER.info("Building configuration")
    config = build_config_from_args(args)

    LOGGER.info("Loading input dataset from %s", config.input_path)
    df_input = load_dataframe(config.input_path)
    LOGGER.info("Loaded dataframe with shape: %s", df_input.shape)

    LOGGER.info("Running selection bias pipeline")
    df_annotated, df_selected, pipeline_metadata = run_selection_pipeline(
        df=df_input,
        config=config,
    )

    LOGGER.info(
        "Selection completed | input=%d eligible=%d selected=%d",
        len(df_input),
        pipeline_metadata["n_rows_eligible"],
        len(df_selected),
    )

    LOGGER.info("Building report")
    report = build_selection_report(
        config=config,
        df_input=df_input,
        df_annotated=df_annotated,
        df_selected=df_selected,
        pipeline_metadata=pipeline_metadata,
    )

    LOGGER.info("Saving selected dataset to %s", config.output_selected_path)
    save_dataframe(df_selected, config.output_selected_path)

    if config.save_annotated_dataset and config.output_annotated_path is not None:
        LOGGER.info("Saving annotated dataset to %s", config.output_annotated_path)
        save_dataframe(df_annotated, config.output_annotated_path)

    LOGGER.info("Saving JSON report to %s", config.output_report_path)
    save_json_report(report, config.output_report_path)

    LOGGER.info("Run completed successfully.")


if __name__ == "__main__":
    main()