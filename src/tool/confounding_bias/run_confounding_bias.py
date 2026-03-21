"""
Command-line tool to create a pseudo-observational version of an RCT dataset using an artificial propensity score.
python -m confounding_bias.run_confounding_bias --input ../../data/analytic/aids/aids_rct_id.parquet --output-parquet ../results/confounding_bias/actg175_observational.parquet --output-report ../results/confounding_bias/actg175_observational_report.json --covariates age wtkg karnof oprior preanti strat cd40 symptom --outcome-col label --original-treatment-col treat --new-treatment-col treat_obs --ps-col ps_artificial --clip-min 0.05 --clip-max 0.95 --intercept 0.0 --seed 42 --verbose
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from confounding_bias.config import (
    DEFAULT_CLIP_MAX,
    DEFAULT_CLIP_MIN,
    DEFAULT_COVARIATES,
    DEFAULT_INTERCEPT,
    DEFAULT_NEW_TREATMENT,
    DEFAULT_ORIGINAL_TREATMENT,
    DEFAULT_OUTCOME,
    DEFAULT_PS_COL,
    DEFAULT_SEED,
    TransformConfig,
)
from confounding_bias.io import load_data, save_json_report, save_parquet
from confounding_bias.preprocessing import select_and_clean_data, standardize_covariates
from confounding_bias.propensity import run_propensity_pipeline
from confounding_bias.reports import build_report, compute_balance_table
from confounding_bias.utils import setup_logging


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a pseudo-observational version of an RCT dataset "
            "using an artificial propensity score."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input Parquet.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Path to output transformed Parquet.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Path to output JSON report.",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        nargs="+",
        default=DEFAULT_COVARIATES,
        help="List of pre-treatment covariates to use in the artificial propensity score.",
    )
    parser.add_argument(
        "--outcome-col",
        type=str,
        default=DEFAULT_OUTCOME,
    )
    parser.add_argument(
        "--original-treatment-col",
        type=str,
        default=DEFAULT_ORIGINAL_TREATMENT,
    )
    parser.add_argument(
        "--new-treatment-col",
        type=str,
        default=DEFAULT_NEW_TREATMENT,
    )
    parser.add_argument(
        "--ps-col",
        type=str,
        default=DEFAULT_PS_COL,
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=DEFAULT_CLIP_MIN,
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=DEFAULT_CLIP_MAX,
    )
    parser.add_argument(
        "--intercept",
        type=float,
        default=DEFAULT_INTERCEPT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser


def build_config_from_args(args: argparse.Namespace) -> TransformConfig:
    """
    Build the transformation config from parsed CLI arguments.
    """
    return TransformConfig(
        input_path=args.input,
        output_parquet=args.output_parquet,
        output_report=args.output_report,
        covariates=args.covariates,
        outcome_col=args.outcome_col,
        original_treatment_col=args.original_treatment_col,
        new_treatment_col=args.new_treatment_col,
        ps_col=args.ps_col,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        intercept=args.intercept,
        seed=args.seed,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    config = build_config_from_args(args)

    LOGGER.info("Loading data from %s", config.input_path)
    df = load_data(config.input_path)
    n_original_rows = len(df)

    LOGGER.info("Selecting and cleaning data")
    work_df, cleaning_metadata = select_and_clean_data(
        df=df,
        covariates=config.covariates,
        outcome_col=config.outcome_col,
        original_treatment_col=config.original_treatment_col,
    )

    LOGGER.info("Standardizing covariates")
    X_scaled, _ = standardize_covariates(
        work_df=work_df,
        covariates=config.covariates,
    )

    LOGGER.info("Running artificial propensity-score pipeline")
    work_df, propensity_metadata = run_propensity_pipeline(
        work_df=work_df,
        X_scaled=X_scaled,
        config=config,
    )

    LOGGER.info("Computing covariate balance table")
    balance_df = compute_balance_table(
        df=work_df,
        covariates=config.covariates,
        treatment_col=config.new_treatment_col,
    )

    LOGGER.info("Saving transformed dataset to %s", config.output_parquet)
    save_parquet(work_df, config.output_parquet, index=False)

    report = build_report(
        work_df=work_df,
        covariates=config.covariates,
        outcome_col=config.outcome_col,
        original_treatment_col=config.original_treatment_col,
        new_treatment_col=config.new_treatment_col,
        ps_col=config.ps_col,
        auc=propensity_metadata["assignment_auc_predicting_new_treatment_from_X"],
        weights=propensity_metadata["weights_used"],
        balance_df=balance_df,
        n_original_rows=n_original_rows,
    )

    LOGGER.info("Saving report to %s", config.output_report)
    save_json_report(report, config.output_report, indent=2)

    print("\n=== Transformation completed ===")
    print(f"Rows retained: {len(work_df)} / {n_original_rows}")
    print(f"Original treatment rate: {work_df[config.original_treatment_col].mean():.3f}")
    print(f"New treatment rate:      {work_df[config.new_treatment_col].mean():.3f}")
    print(f"Mean artificial PS:      {work_df[config.ps_col].mean():.3f}")
    print(
        "AUC[Treat_obs ~ X]:      "
        f"{propensity_metadata['assignment_auc_predicting_new_treatment_from_X']:.3f}"
    )
    print("\nTop covariate shifts by |SMD|:")
    print(balance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()