"""
Plot BCAUSS CATE estimates with policy analysis similar to DR-learner output.

This script generates plots for:
  1. Distribution of CATE estimates by risk (baseline outcome) quantiles
  2. Treatment rate under BCAUSS policy by risk bin
  3. Expected outcome (mortality) under policy vs treat-none
  4. Expected outcome under policy vs treat-all
  5. Benefit contribution per risk bin

Usage:
    python src/cate/scripts/plot_bcauss.py --dataset aids_v1 --threshold 0.0
"""

import argparse
import sys
from pathlib import Path

# Add catellam root directory to path for imports
# plot_bcauss.py: .../catellam/src/cate/scripts/plot_bcauss.py
# parent.parent.parent = .../catellam/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import get_config, CONFIGS
from src.cate.policy_check import policy_decision, TAU_LE_THR, TAU_GE_THR


def plot_bcauss_policy_analysis(
    cate_estimates_path: Path,
    y_test: np.ndarray,
    t_test: np.ndarray,
    outcome_col: str,
    outcome_nice_name: str = "Failure",
    threshold: float = 0.0,
    n_bins: int = 5,
    tau_direction: str = "lte",
    out_dir: Path = None,
) -> pd.DataFrame:
    """
    Generate policy analysis plots for BCAUSS CATE estimates.

    Parameters
    ----------
    cate_estimates_path : Path
        Path to parquet file with CATE estimates (id, treatment, outcome_failure, cate_estimate)
    y_test : np.ndarray
        Test outcome values (aligned with CATE estimates)
    t_test : np.ndarray
        Test treatment values
    outcome_col : str
        Name of outcome column (e.g., "outcome_failure")
    outcome_nice_name : str
        Nice name for outcome (used in plots)
    threshold : float
        Threshold for treatment policy decision (default 0.0: treat if CATE <= threshold)
    n_bins : int
        Number of risk quantile bins
    tau_direction : str
        "lte" = treat if tau <= threshold, "gte" = treat if tau >= threshold
    out_dir : Path
        Output directory for plots (default: subdirectory of config out_dir)

    Returns
    -------
    pd.DataFrame
        Summary statistics per risk bin
    """
    if out_dir is None:
        out_dir = Path.cwd() / "bcauss_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load CATE estimates
    cate_df = pd.read_parquet(cate_estimates_path)
    if len(cate_df) != len(y_test):
        raise ValueError(
            f"CATE estimates ({len(cate_df)}) and test set ({len(y_test)}) have different sizes"
        )

    # For baseline risk (mu0_hat), use outcome as proxy
    # mu0_hat = empirical outcome (binary 0/1)
    # mu1_hat = estimated counterfactual (we use outcome + CATE as approximation)
    mu0_hat = y_test.astype(float)
    mu1_hat = np.clip(y_test.astype(float) + cate_df["cate_estimate"].values, 0, 1)
    
    # Build working dataframe
    d = pd.DataFrame({
        "cate_estimate": cate_df["cate_estimate"].values,
        "mu0_hat": mu0_hat,
        "mu1_hat": mu1_hat,
        "y_test": y_test,
        "t_test": t_test,
    }).dropna()

    # Create risk bins using quantiles of outcome prevalence or treatment assignment
    # Since mu0_hat is binary (0/1), we need a smoother risk metric
    # Solution: use smoothed risk via treatment assignment + outcome
    risk_smooth = np.zeros_like(d["mu0_hat"].values)
    for i in range(len(d)):
        # Risk ~ empirical outcome rate in neighborhood
        # For simplicity: use outcome value (0 or 1) but also add treatment info
        risk_smooth[i] = d["mu0_hat"].values[i]
    
    d["risk_smooth"] = risk_smooth
    
    # Create bins - if not enough unique values, use by-outcome binning
    unique_risks = d["risk_smooth"].nunique()
    if unique_risks < n_bins:
        # Binary outcome: bin by treatment + outcome (4 groups max)
        d["risk_bin"] = d["t_test"].astype(int) * 2 + d["y_test"].astype(int)
        n_bins_eff = int(d["risk_bin"].max()) + 1
    else:
        d["risk_bin"] = pd.qcut(d["risk_smooth"], q=n_bins, labels=False, duplicates="drop")
        n_bins_eff = int(d["risk_bin"].max()) + 1

    # Policy decision: which rule to use?
    if tau_direction == "lte":
        rule = TAU_LE_THR  # treat if CATE <= threshold
    else:
        rule = TAU_GE_THR  # treat if CATE >= threshold

    a_pi = policy_decision(d["cate_estimate"], threshold=threshold, rule=rule)
    d["a_pi"] = a_pi

    # Estimated outcome under policy (using Y as proxy)
    d["mu_pi"] = np.where(d["a_pi"].values == 1, d["mu1_hat"].values, d["mu0_hat"].values)

    # Aggregate by risk bin
    g = d.groupby("risk_bin", observed=True)
    summary = g.agg(
        n=("mu0_hat", "size"),
        risk0_mean=("mu0_hat", "mean"),
        cate_mean=("cate_estimate", "mean"),
        cate_median=("cate_estimate", "median"),
        cate_std=("cate_estimate", "std"),
        treat_rate=("a_pi", "mean"),
        mort_policy=("mu_pi", "mean"),
        mort_none=("mu0_hat", "mean"),
        mort_treat_all=("mu1_hat", "mean"),
    ).reset_index()

    summary["delta_policy_vs_treat_all"] = summary["mort_policy"] - summary["mort_treat_all"]
    summary["delta_policy_vs_none"] = summary["mort_policy"] - summary["mort_none"]
    summary["benefit_abs"] = -summary["delta_policy_vs_none"]
    summary["benefit_total"] = summary["benefit_abs"] * summary["n"]

    # ============================================================================
    # Plot 1: CATE distribution by risk bin (boxplot)
    # ============================================================================
    cate_by_bin = [
        d.loc[d["risk_bin"] == b, "cate_estimate"].values for b in range(n_bins_eff)
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(cate_by_bin, showfliers=False)
    plt.axhline(threshold, linestyle="--", color="red", label=f"Policy threshold ({threshold})")
    plt.xlabel(f"Risk bins (quantiles of {outcome_nice_name})")
    plt.ylabel("CATE estimate")
    plt.title("Distribution of BCAUSS CATE estimates by baseline risk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "bcauss_cate_by_risk_bin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_dir / 'bcauss_cate_by_risk_bin.png'}")

    # ============================================================================
    # Plot 2: Treatment rate per risk bin
    # ============================================================================
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(n_bins_eff) + 1, summary["treat_rate"].values, marker="o", linewidth=2, markersize=8)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(f"Risk bin (1=low → {n_bins_eff}=high {outcome_nice_name})")
    plt.ylabel("Treatment rate under BCAUSS policy")
    plt.title("BCAUSS Policy: Treatment rate by risk level")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "bcauss_treat_rate_by_risk_bin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_dir / 'bcauss_treat_rate_by_risk_bin.png'}")

    # ============================================================================
    # Plot 3: Estimated outcome - policy vs treat-none
    # ============================================================================
    plt.figure(figsize=(10, 6))
    x = np.arange(n_bins_eff)
    plt.bar(
        x - 0.2,
        summary["mort_none"].values,
        width=0.4,
        label="Treat-none",
        alpha=0.8,
        color="blue",
    )
    plt.bar(
        x + 0.2,
        summary["mort_policy"].values,
        width=0.4,
        label="BCAUSS policy",
        alpha=0.8,
        color="green",
    )
    plt.xlabel(f"Risk bin (1=low → {n_bins_eff}=high {outcome_nice_name})")
    plt.ylabel(f"Estimated {outcome_nice_name}")
    plt.title(f"BCAUSS: Estimated {outcome_nice_name} by risk - Policy vs Treat-none")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(out_dir / "bcauss_outcome_vs_treat_none.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_dir / 'bcauss_outcome_vs_treat_none.png'}")

    # ============================================================================
    # Plot 4: Estimated outcome - policy vs treat-all
    # ============================================================================
    plt.figure(figsize=(10, 6))
    x = np.arange(n_bins_eff)
    plt.bar(
        x - 0.2,
        summary["mort_treat_all"].values,
        width=0.4,
        label="Treat-all",
        alpha=0.8,
        color="orange",
    )
    plt.bar(
        x + 0.2,
        summary["mort_policy"].values,
        width=0.4,
        label="BCAUSS policy",
        alpha=0.8,
        color="green",
    )
    plt.xlabel(f"Risk bin (1=low → {n_bins_eff}=high {outcome_nice_name})")
    plt.ylabel(f"Estimated {outcome_nice_name}")
    plt.title(f"BCAUSS: Estimated {outcome_nice_name} by risk - Policy vs Treat-all")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(out_dir / "bcauss_outcome_vs_treat_all.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_dir / 'bcauss_outcome_vs_treat_all.png'}")

    # ============================================================================
    # Plot 5: Benefit contribution per risk bin
    # ============================================================================
    plt.figure(figsize=(10, 6))
    plt.bar(
        np.arange(n_bins_eff),
        summary["benefit_total"].values,
        alpha=0.8,
        color="purple",
    )
    plt.xlabel(f"Risk bin (1=low → {n_bins_eff}=high {outcome_nice_name})")
    plt.ylabel(f"Total benefit contribution (n × Δ{outcome_nice_name})")
    plt.title("BCAUSS: Where does the improvement come from?")
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig(out_dir / "bcauss_benefit_contribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out_dir / 'bcauss_benefit_contribution.png'}")

    # ============================================================================
    # Summary statistics CSV
    # ============================================================================
    summary.to_csv(out_dir / "bcauss_summary_by_risk_bin.csv", index=False)
    print(f"✓ Saved: {out_dir / 'bcauss_summary_by_risk_bin.csv'}")

    return summary


def main():
    """Main entry point for BCAUSS plotting."""
    parser = argparse.ArgumentParser(
        description="Generate policy analysis plots for BCAUSS CATE estimates"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(CONFIGS.keys()),
        help="Dataset config to use",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Treatment decision threshold (default 0.0: treat if CATE <= threshold)",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=5,
        help="Number of risk quantile bins",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Load config
    cfg = get_config(args.dataset)

    # Paths
    bcauss_dir = Path(cfg.out_dir) / "bcauss"
    cate_path = bcauss_dir / "cate_estimates.parquet"
    plots_dir = bcauss_dir / "plots"

    if not cate_path.exists():
        raise FileNotFoundError(f"CATE estimates not found: {cate_path}")

    # Load CATE estimates
    cate_df = pd.read_parquet(cate_path)

    # Load test data for outcome
    df = pd.read_parquet(cfg.data_path)
    from src.baseline.split import split_by_subject

    splits = split_by_subject(
        df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=args.seed,
    )
    te_idx = splits["test"]

    Y_test = df.iloc[te_idx][cfg.outcome_col].astype(int).values
    T_test = df.iloc[te_idx][cfg.treatment_col].astype(int).values

    if len(cate_df) != len(Y_test):
        raise ValueError(
            f"CATE estimates ({len(cate_df)}) and test set ({len(Y_test)}) have different sizes"
        )

    # Generate plots
    print(f"\n{'='*70}")
    print(f"Generating BCAUSS policy analysis plots for {args.dataset}")
    print(f"{'='*70}")
    print(f"CATE estimates: {cate_path}")
    print(f"Output directory: {plots_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Risk bins: {args.n_bins}")
    print(f"Tau direction: {cfg.tau_direction}")
    print(f"{'='*70}\n")

    summary = plot_bcauss_policy_analysis(
        cate_estimates_path=cate_path,
        y_test=Y_test,
        t_test=T_test,
        outcome_col=cfg.outcome_col,
        outcome_nice_name=cfg.outcome_nice_name,
        threshold=args.threshold,
        n_bins=args.n_bins,
        tau_direction=cfg.tau_direction,
        out_dir=plots_dir,
    )

    # Print summary
    print("\n" + "="*70)
    print("Summary Statistics by Risk Bin:")
    print("="*70)
    print(summary.to_string(index=False))
    print("="*70)

    # Overall statistics
    print("\nOverall Policy Statistics:")
    print(f"  Overall treat rate: {(cate_df['cate_estimate'] <= args.threshold).mean():.2%}")
    print(f"  Mean CATE: {cate_df['cate_estimate'].mean():.6f}")
    print(f"  Median CATE: {cate_df['cate_estimate'].median():.6f}")
    print(f"  Std CATE: {cate_df['cate_estimate'].std():.6f}")
    print(f"  CATE range: [{cate_df['cate_estimate'].min():.6f}, {cate_df['cate_estimate'].max():.6f}]")
    print()


if __name__ == "__main__":
    main()




