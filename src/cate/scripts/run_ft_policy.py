"""
Personalized policy: treat each individual based on their CATE estimate.
- Treat if tau_hat < 0 (treatment predicted to reduce mortality)
- Compare DR policy value vs treat-all and treat-none
- Report treatment counts, percentages, and DR policy values

Usage:
    python -m src.cate.scripts.run_personalized_policy --dataset diur_v1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline.split import split_by_subject
from src.baseline.features import default_feature_columns, coerce_numeric_columns
from src.baseline.models import HGBConfig, make_hgb_pipeline
from src.baseline.summary import save_json
from src.config import get_config, CONFIGS

from src.cate.policy import dr_policy_value, bootstrap_policy_value, PolicyValueConfig


def main():
    ap = argparse.ArgumentParser(description="Personalized policy from CATE estimates.")
    ap.add_argument("--dataset", required=True, choices=list(CONFIGS.keys()), help="Which dataset config to use")
    ap.add_argument("--data", required=False, help="Path to analytic parquet file")
    ap.add_argument("--tau_pred", required=False, help="Path to dr_tau_test.parquet")
    ap.add_argument("--out_dir", help="Output directory")
    ap.add_argument("--tau_threshold", type=float, default=0.0,
                    help="Treat if tau_hat < threshold (default: 0.0 -> treat if beneficial)")
    ap.add_argument("--n_boot", type=int, default=200, help="Bootstrap replicates for CIs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = get_config(args.dataset)

    if args.data is None:
        args.data = cfg.data_path
    if args.out_dir is None:
        args.out_dir = cfg.out_dir
    if args.tau_pred is None:
        args.tau_pred = str(cfg.out_dir) + "/dr_tau_test.parquet"

    seed = int(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load full data
    df = pd.read_parquet(args.data)

    num_cols, cat_cols, dropped = default_feature_columns(
        df,
        id_col=cfg.id_col,
        subject_col=cfg.subject_col,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=cfg.drop_cols,
    )
    df[num_cols] = coerce_numeric_columns(df, num_cols)
    feat_cols = num_cols + cat_cols

    splits = split_by_subject(
        df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=seed,
    )
    tr_idx, te_idx = splits["train"], splits["test"]

    X = df[feat_cols]
    T = df[cfg.treatment_col].astype(int).values
    Y = df[cfg.outcome_col].astype(int).values

    X_tr, T_tr, Y_tr = X.iloc[tr_idx], T[tr_idx], Y[tr_idx]
    X_te, T_te, Y_te = X.iloc[te_idx], T[te_idx], Y[te_idx]

    # --- load tau predictions and align by id_col
    pred = pd.read_parquet(args.tau_pred)
    te_ids = df.iloc[te_idx][cfg.id_col].values
    te_map = pd.DataFrame({cfg.id_col: te_ids, "_pos": np.arange(len(te_ids))})
    pred2 = pred.merge(te_map, on=cfg.id_col, how="inner").sort_values("_pos")
    if len(pred2) != len(te_ids):
        raise ValueError(
            f"tau_pred does not match test split: {len(pred2)} vs {len(te_ids)}"
        )

    tau_hat = pred2["tau_hat"].values.astype(float)

    # --- propensity scores
    if "ps_hat" in pred2.columns:
        ps_hat = pred2["ps_hat"].values.astype(float)
    else:
        ps_cfg = HGBConfig(random_state=seed)
        ps_model = make_hgb_pipeline(num_cols, cat_cols, ps_cfg)
        ps_model.fit(X_tr, T_tr)
        ps_hat = ps_model.predict_proba(X_te)[:, 1]

    # --- outcome nuisance models fit on train
    out_cfg = HGBConfig(random_state=seed, max_depth=3, min_samples_leaf=50)
    out1 = make_hgb_pipeline(num_cols, cat_cols, out_cfg)
    out0 = make_hgb_pipeline(num_cols, cat_cols, out_cfg)
    out1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    out0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
    mu1_hat = out1.predict_proba(X_te)[:, 1]
    mu0_hat = out0.predict_proba(X_te)[:, 1]

    # --- personalized policy: treat if tau_hat < threshold (treatment reduces outcome)
    tau_thr = args.tau_threshold
    pi_personalized = (tau_hat < tau_thr).astype(int)
    pi_none = np.zeros(len(Y_te), dtype=int)
    pi_all = np.ones(len(Y_te), dtype=int)

    pvcfg = PolicyValueConfig(
        ps_clip=cfg.ps_clip,
        ipw_trim_quantiles=None,
    )

    # --- DR policy values
    v_none = dr_policy_value(y=Y_te, t=T_te, pi=pi_none, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=pvcfg)
    v_all  = dr_policy_value(y=Y_te, t=T_te, pi=pi_all,  ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=pvcfg)
    v_pi   = dr_policy_value(y=Y_te, t=T_te, pi=pi_personalized, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=pvcfg)

    # --- bootstrap CIs
    ci_none = bootstrap_policy_value(y=Y_te, t=T_te, pi=pi_none, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat,
                                     n_boot=args.n_boot, seed=seed, cfg=pvcfg)
    ci_all  = bootstrap_policy_value(y=Y_te, t=T_te, pi=pi_all,  ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat,
                                     n_boot=args.n_boot, seed=seed, cfg=pvcfg)
    ci_pi   = bootstrap_policy_value(y=Y_te, t=T_te, pi=pi_personalized, ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat,
                                     n_boot=args.n_boot, seed=seed, cfg=pvcfg)

    # --- treatment stats
    n_test = len(Y_te)
    n_treated = int(pi_personalized.sum())
    n_control = n_test - n_treated
    pct_treated = 100.0 * n_treated / n_test
    pct_control = 100.0 * n_control / n_test

    # tau distribution summary
    tau_summary = {
        "mean": float(np.mean(tau_hat)),
        "sd": float(np.std(tau_hat, ddof=1)),
        "p01": float(np.quantile(tau_hat, 0.01)),
        "p25": float(np.quantile(tau_hat, 0.25)),
        "p50": float(np.quantile(tau_hat, 0.50)),
        "p75": float(np.quantile(tau_hat, 0.75)),
        "p99": float(np.quantile(tau_hat, 0.99)),
        "n_negative": int((tau_hat < 0).sum()),
        "n_positive": int((tau_hat >= 0).sum()),
    }

    # --- console output
    print("=" * 60)
    print("  PERSONALIZED POLICY EVALUATION")
    print(f"  Dataset : {args.dataset}")
    print(f"  Outcome : {cfg.outcome_col}  (lower = better)")
    print(f"  Tau threshold for treatment: tau_hat < {tau_thr}")
    print("=" * 60)
    print(f"\n--- Test set size: {n_test}")
    print(f"    Treated by personalized policy : {n_treated:>6}  ({pct_treated:.1f}%)")
    print(f"    Control by personalized policy : {n_control:>6}  ({pct_control:.1f}%)")
    print(f"\n--- tau_hat distribution on test set ---")
    print(f"    mean={tau_summary['mean']:.4f}  sd={tau_summary['sd']:.4f}")
    print(f"    p01={tau_summary['p01']:.4f}  p25={tau_summary['p25']:.4f}  "
          f"p50={tau_summary['p50']:.4f}  p75={tau_summary['p75']:.4f}  p99={tau_summary['p99']:.4f}")
    print(f"    tau_hat < 0 (beneficial): {tau_summary['n_negative']}  |  tau_hat >= 0 (not beneficial): {tau_summary['n_positive']}")
    print(f"\n--- DR Policy Values (expected {cfg.outcome_col}; lower = better) ---")
    print(f"    Treat-none       : {v_none:.4f}  95% CI [{ci_none['ci_lo']:.4f}, {ci_none['ci_hi']:.4f}]")
    print(f"    Treat-all        : {v_all:.4f}  95% CI [{ci_all['ci_lo']:.4f}, {ci_all['ci_hi']:.4f}]")
    print(f"    Personalized     : {v_pi:.4f}  95% CI [{ci_pi['ci_lo']:.4f}, {ci_pi['ci_hi']:.4f}]")
    print(f"\n    Δ Personalized vs Treat-none : {v_pi - v_none:+.4f}")
    print(f"    Δ Personalized vs Treat-all  : {v_pi - v_all:+.4f}")
    winner = "Personalized" if v_pi <= min(v_none, v_all) else ("Treat-all" if v_all <= v_none else "Treat-none")
    print(f"\n    Best policy: {winner}")
    print("=" * 60)

    # --- save detailed per-person predictions
    out_preds = pd.DataFrame({
        cfg.id_col: pred2[cfg.id_col].values,
        cfg.subject_col: pred2[cfg.subject_col].values if cfg.subject_col in pred2.columns else np.nan,
        cfg.treatment_col: T_te,
        cfg.outcome_col: Y_te,
        "tau_hat": tau_hat,
        "ps_hat": ps_hat,
        "mu1_hat": mu1_hat,
        "mu0_hat": mu0_hat,
        "pi_personalized": pi_personalized,
        "pi_none": pi_none,
        "pi_all": pi_all,
    })
    pred_path = out_dir / "personalized_policy_predictions.parquet"
    out_preds.to_parquet(pred_path, index=False)
    print(f"\nSaved per-person predictions -> {pred_path}")

    # --- save summary JSON
    results = {
        "meta": {
            "dataset": args.dataset,
            "data": str(args.data),
            "tau_pred": str(args.tau_pred),
            "n_test": n_test,
            "tau_threshold": tau_thr,
            "outcome": cfg.outcome_col,
            "treatment": cfg.treatment_col,
            "ps_clip": {"lo": float(cfg.ps_clip[0]), "hi": float(cfg.ps_clip[1])},
            "n_boot": args.n_boot,
            "seed": seed,
        },
        "treatment_allocation": {
            "n_treated": n_treated,
            "n_control": n_control,
            "pct_treated": round(pct_treated, 2),
            "pct_control": round(pct_control, 2),
        },
        "tau_summary": tau_summary,
        "policy_values": {
            "treat_none": v_none,
            "treat_all": v_all,
            "personalized": v_pi,
            "delta_vs_none": float(v_pi - v_none),
            "delta_vs_all": float(v_pi - v_all),
        },
        "bootstrap_ci": {
            "treat_none": ci_none,
            "treat_all": ci_all,
            "personalized": ci_pi,
        },
        "best_policy": winner,
    }

    summary_path = out_dir / "personalized_policy_summary.json"
    save_json(summary_path, results)
    print(f"Saved summary -> {summary_path}")

    # -----------------------------------------------------------------------
    # CATE vs ATE: does heterogeneity help beyond the average effect?
    # -----------------------------------------------------------------------
    _lo, _hi = cfg.ps_clip
    _e_clipped = np.clip(ps_hat, _lo, _hi)

    # DR-ATE: doubly robust average treatment effect
    dr_scores = (
        (mu1_hat - mu0_hat)
        + T_te * (Y_te - mu1_hat) / (_e_clipped + 1e-12)
        - (1 - T_te) * (Y_te - mu0_hat) / (1 - _e_clipped + 1e-12)
    )
    dr_ate = float(np.mean(dr_scores))
    dr_ate_se = float(np.std(dr_scores, ddof=1) / np.sqrt(len(dr_scores)))

    # Naive ATE (unadjusted difference in means)
    naive_ate = float(np.mean(Y_te[T_te == 1]) - np.mean(Y_te[T_te == 0]))

    # IPW ATE
    ipw_scores = (
        T_te * Y_te / (_e_clipped + 1e-12)
        - (1 - T_te) * Y_te / (1 - _e_clipped + 1e-12)
    )
    ipw_ate = float(np.mean(ipw_scores))

    # CATE summary
    tau_mean     = float(np.mean(tau_hat))
    tau_sd       = float(np.std(tau_hat, ddof=1))
    tau_p05      = float(np.quantile(tau_hat, 0.05))
    tau_p95      = float(np.quantile(tau_hat, 0.95))
    frac_benefit = float(np.mean(tau_hat < 0))

    # Heterogeneity SNR: if CATE ~ ATE for everyone, tau_sd ~ 0
    het_snr = tau_sd / (abs(dr_ate) + 1e-9)

    # Bias reduction signal: how much does DR-ATE differ from naive ATE?
    bias_reduction = float(naive_ate - dr_ate)

    # Value of personalization vs best uniform policy
    best_uniform = min(v_none, v_all)
    value_of_personalization = float(best_uniform - v_pi)  # positive = CATE policy is better

    cate_vs_ate = {
        "ate": {
            "naive_ate": naive_ate,
            "ipw_ate": ipw_ate,
            "dr_ate": dr_ate,
            "dr_ate_se": dr_ate_se,
            "bias_reduction_naive_vs_dr": bias_reduction,
            "interpretation": (
                "Negative ATE = treatment reduces mortality on average. "
                "bias_reduction_naive_vs_dr: large absolute value = strong confounding corrected by DR."
            ),
        },
        "cate_summary": {
            "mean_tau": tau_mean,
            "sd_tau": tau_sd,
            "p05_tau": tau_p05,
            "p95_tau": tau_p95,
            "frac_predicted_benefit": frac_benefit,
            "interpretation": (
                "mean_tau ≈ DR-ATE if no heterogeneity. "
                "Large sd_tau relative to |ATE| suggests meaningful heterogeneity."
            ),
        },
        "heterogeneity_signal": {
            "het_snr": het_snr,
            "value_of_personalization_vs_best_uniform": value_of_personalization,
            "interpretation": (
                "het_snr = sd(tau_hat)/|DR_ATE|. Values >> 1 suggest noise dominates signal. "
                "value_of_personalization > 0 means CATE policy beats best uniform policy on DR value."
            ),
        },
    }

    results["cate_vs_ate"] = cate_vs_ate

    # re-save JSON with cate_vs_ate section added
    save_json(summary_path, results)

    print("\n=== CATE vs ATE (Heterogeneity & Bias Reduction Check) ===")
    print(f"Naive ATE              : {naive_ate:+.4f}  (unadjusted, biased)")
    print(f"IPW ATE                : {ipw_ate:+.4f}  (propensity weighted)")
    print(f"DR ATE                 : {dr_ate:+.4f}  (SE={dr_ate_se:.4f}, doubly robust)")
    print(f"Bias corrected by DR   : {bias_reduction:+.4f}  (naive - DR)")
    print(f"Mean CATE (tau_hat)    : {tau_mean:+.4f}  (SD={tau_sd:.4f})")
    print(f"Frac predicted benefit : {frac_benefit:.3f}  (tau < 0)")
    print(f"Heterogeneity SNR      : {het_snr:.2f}  (SD/|ATE|, >> 1 = noisy)")
    print(f"Value of personalization vs best uniform: {value_of_personalization:+.4f}")
    print("=" * 60)


    # --- optional plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: tau_hat distribution + threshold
        axes[0].hist(tau_hat, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[0].axvline(tau_thr, color="red", linestyle="--", linewidth=2, label=f"Threshold={tau_thr}")
        axes[0].set_xlabel("tau_hat (CATE)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of CATE Estimates")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        treat_patch = plt.matplotlib.patches.Patch(color="steelblue", alpha=0.5, label=f"Treat (tau<{tau_thr}): {n_treated} ({pct_treated:.1f}%)")
        ctrl_patch  = plt.matplotlib.patches.Patch(color="lightgray",  alpha=0.8, label=f"Control (tau≥{tau_thr}): {n_control} ({pct_control:.1f}%)")
        axes[0].legend(handles=[treat_patch, ctrl_patch], loc="upper right")

        # Right: policy value comparison with CIs
        policies = ["Treat-none", "Treat-all", "Personalized"]
        values   = [v_none, v_all, v_pi]
        cis      = [ci_none, ci_all, ci_pi]
        colors   = ["tomato", "cornflowerblue", "mediumseagreen"]
        yerr_lo  = [v - c["ci_lo"] for v, c in zip(values, cis)]
        yerr_hi  = [c["ci_hi"] - v for v, c in zip(values, cis)]

        x_pos = np.arange(len(policies))
        bars = axes[1].bar(x_pos, values, color=colors, alpha=0.85, width=0.5)
        axes[1].errorbar(x_pos, values, yerr=[yerr_lo, yerr_hi],
                         fmt="none", color="black", capsize=6, linewidth=2)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(policies)
        axes[1].set_ylabel(f"DR Policy Value ({cfg.outcome_col})")
        axes[1].set_title("Policy Comparison (lower = better)")
        axes[1].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plot_path = out_dir / "personalized_policy_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot -> {plot_path}")

        # --- Threshold curve plot (top-fraction policy sweep)
        from src.cate.policy import threshold_curve

        thresholds = np.linspace(0.05, 0.95, 20)
        pvcfg_curve = PolicyValueConfig(ps_clip=cfg.ps_clip, ipw_trim_quantiles=None)

        curve = threshold_curve(
            Y=Y_te,
            t=T_te,
            mu1_hat=mu1_hat,
            mu0_hat=mu0_hat,
            ps_hat=ps_hat,
            tau_hat=tau_hat,
            cfg=pvcfg_curve,
            thresholds=thresholds,
            n_boot=args.n_boot,
            seed=seed,
            direction=cfg.tau_direction,
            policy_kind="top_frac_benefit",
        )

        curve.to_csv(out_dir / "ft_policy_threshold_curve.csv", index=False)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.plot(curve["threshold"], curve["value_dr"], marker='o', color='green', label='Policy Value')
        ax2.axhline(y=v_none, color='r', linestyle='--', label='Treat None')
        ax2.axhline(y=v_all,  color='b', linestyle='--', label='Treat All')
        ax2.set_xlabel("Threshold (top fraction treated)")
        ax2.set_ylabel(f"Doubly Robust Policy Value ({cfg.outcome_col})")
        ax2.set_title("Threshold Curve for Treatment Policy")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        curve_plot_path = out_dir / "ft_policy_threshold_curve.png"
        fig2.savefig(curve_plot_path, dpi=150)
        plt.close(fig2)
        print(f"Saved threshold curve plot -> {curve_plot_path}")


    except ImportError:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
