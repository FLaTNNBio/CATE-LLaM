import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import src

from src.baseline import BaselineConfig, split_by_subject
from src.baseline.features import default_feature_columns, coerce_numeric_columns
from src.baseline.models import HGBConfig, make_hgb_pipeline, clip_ps
from src.baseline.metrics import (
    stabilized_iptw_weights,
    trim_weights,
    ess,
    ate_weighted,
    smd_table,
)
from src.baseline.summary import (
    ps_overlap_summary,
    ci_from_bootstrap,
    weights_summary,
    balance_summary,
    bootstrap_ate_iptw,
    bootstrap_ate_aipw,
    save_json,
)

# ---------------------------
# ATO (overlap) weights
# ---------------------------
def overlap_weights(treat: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """
    Overlap weights (ATO):
      w = 1-ps for treated
      w = ps   for control
    """
    treat = np.asarray(treat, dtype=int)
    ps = np.asarray(ps, dtype=float)
    w = np.empty_like(ps, dtype=float)
    w[treat == 1] = 1.0 - ps[treat == 1]
    w[treat == 0] = ps[treat == 0]
    return w


def _predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    """
    Predict probability of Treatment=1
    using a trained model.
    :param model: model to use
    :param x: data to predict
    :return: numpy array of predictions
    """
    return model.predict_proba(x)[:, 1]

def compute_and_save_summary(
    *,
    name: str,
    out_dir: Path,
    X_sub: pd.DataFrame,
    T_sub: np.ndarray,
    Y_sub: np.ndarray,
    ps_raw: np.ndarray,
    ps_clip: np.ndarray,
    ps_clip_range: tuple[float, float],
    weight_trim_q: tuple[float, float],
    num_cols: list[str],
    cat_cols: list[str],
    ps_cfg: HGBConfig,
    out_cfg: HGBConfig,
    n_folds_aipw: int,
    n_boot_iptw: int,
    n_boot_aipw: int,
    seed: int,
) -> dict:
    """
    Compute and save summary statistics for a given subset (e.g., test set or full data).
    Stats include:
        - PS overlap before and after clipping
        - Weight summaries for IPTW and ATO
        - Balance summaries (SMD) for IPTW and ATO
        - Point estimates for ATE via IPTW, ATO, and AIPW
        - Bootstrap confidence intervals for IPTW, ATO, and AIPW
    Saves SMD tables to CSV files in the specified output directory.

    :param name: name of the subset (e.g., "test" or "full")
    :param out_dir: directory to save outputs
    :param X_sub: DataFrame of covariates for the subset
    :param T_sub: Treatment assignments for the subset
    :param Y_sub: Outcomes for the subset
    :param ps_raw: Propensity Scores before clipping
    :param ps_clip: Propensity Scores after clipping
    :param ps_clip_range: PS clipping range (low, high)
    :param weight_trim_q: Weight trimming quantiles (q_low, q_high) - Trim weights outside these quantiles
    :param num_cols: Column names of numeric features
    :param cat_cols: Column names of categorical features
    :param ps_cfg: Configuration for propensity score model
    :param out_cfg: Configuration for outcome model
    :param n_folds_aipw: Number of folds for AIPW cross-fitting
    :param n_boot_iptw: Number of bootstrap samples for IPTW CI
    :param n_boot_aipw: Number of bootstrap samples for AIPW CI
    :param seed: Seed for reproducibility
    :return: Results summary dictionary
    """
    # weights
    w_iptw = stabilized_iptw_weights(T_sub, ps_clip)
    w_iptw = trim_weights(w_iptw, *weight_trim_q)
    w_ato = overlap_weights(T_sub, ps_clip)
    w_ato = trim_weights(w_ato, *weight_trim_q)

    # ATEs
    ate_iptw = ate_weighted(Y_sub, T_sub, w_iptw)
    ate_ato = ate_weighted(Y_sub, T_sub, w_ato)

    from src.baseline.estimator import aipw_crossfit
    ate_aipw = aipw_crossfit(
        x=X_sub,
        treat=T_sub,
        outcome=Y_sub,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ps_cfg=ps_cfg,
        out_cfg=out_cfg,
        n_folds=n_folds_aipw,
        ps_clip_range=ps_clip_range,
    )

    # SMD (numeric only)
    smd_iptw = smd_table(X_sub[num_cols], T_sub, w=w_iptw)
    smd_ato = smd_table(X_sub[num_cols], T_sub, w=w_ato)

    # save SMDs
    smd_iptw_path = out_dir / f"smd_iptw_{name}.csv"
    smd_ato_path = out_dir / f"smd_ato_{name}.csv"
    smd_iptw.to_csv(smd_iptw_path, index=False)
    smd_ato.to_csv(smd_ato_path, index=False)

    # summary dict
    summary = {
        "scope": name,
        "ps_overlap_preclip": ps_overlap_summary(ps_raw, T_sub, clip_range=ps_clip_range),
        "ps_overlap_postclip": ps_overlap_summary(ps_clip, T_sub, clip_range=ps_clip_range),
        "weights_iptw": weights_summary(w_iptw, n_total=len(w_iptw)),
        "weights_ato": weights_summary(w_ato, n_total=len(w_ato)),
        "balance_iptw": balance_summary(smd_iptw, top_k=20),
        "balance_ato": balance_summary(smd_ato, top_k=20),
        "point_estimates": {
            "ate_iptw": float(ate_iptw),
            "ate_ato": float(ate_ato),
            "ate_aipw": float(ate_aipw),
        },
        "outputs": {
            "smd_iptw_csv": str(smd_iptw_path),
            "smd_ato_csv": str(smd_ato_path),
        }
    }
    # -----------
    # Bootstrap
    # IPTW and AIPW bootstrap CIs
    # -----------
    # Bootstrap: we repeat the entire estimation procedure on bootstrap samples

    # IPTW bootstrap
    ci_iptw = bootstrap_ate_iptw(
        outcome=Y_sub, treat=T_sub, ps=ps_clip,
        n_boot=n_boot_iptw, seed=seed,
        ps_clip_range=ps_clip_range,
        weight_trim_q=weight_trim_q,
    )

    # AIPW bootstrap
    ci_aipw = bootstrap_ate_aipw(
        X=X_sub, Y=Y_sub, T=T_sub,
        num_cols=num_cols, cat_cols=cat_cols,
        ps_cfg=ps_cfg, out_cfg=out_cfg,
        n_folds=n_folds_aipw,
        ps_clip_range=ps_clip_range,
        n_boot=n_boot_aipw, seed=seed,
    )

    # ATO bootstrap (cheap)
    rng = np.random.default_rng(seed)
    n = len(Y_sub)
    idx = np.arange(n)
    reps_ato = np.empty(n_boot_iptw, dtype=float)
    for b in range(n_boot_iptw):
        b_idx = rng.choice(idx, size=n, replace=True)
        Yb, Tb, psb = Y_sub[b_idx], T_sub[b_idx], ps_clip[b_idx]
        wb = overlap_weights(Tb, psb)
        wb = trim_weights(wb, *weight_trim_q)
        reps_ato[b] = ate_weighted(Yb, Tb, wb)

    ci_ato = {"type": "ato_bootstrap", "n_boot": int(n_boot_iptw), "alpha": 0.05, "ci": ci_from_bootstrap(reps_ato)}
    summary["bootstrap_ci"] = {"iptw": ci_iptw, "aipw": ci_aipw, "ato": ci_ato}

    return summary



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to analytic_v0_extended_prepared.parquet")
    ap.add_argument("--out_dir", default="artifacts", help="Directory for outputs")
    ap.add_argument("--n_boot_iptw", type=int, default=100)
    ap.add_argument("--n_boot_aipw", type=int, default=100)
    ap.add_argument("--scope", choices=["test", "full", "both"], default="test",
                    help="Compute summary on test set, full dataset, or both.")
    ap.add_argument("--full_ps", choices=["train_fit", "crossfit"], default="train_fit",
                    help="How to obtain PS on full: train_fit uses PS model fit on train; crossfit refits via CV (slower, cleaner).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # debug: ensure you run what you edited
    print("baseline package:", src.baseline.__file__)
    print("estimator file:", src.baseline.estimator.__file__)
    print("features file:", src.baseline.features.__file__)

    # Config and data
    cfg = BaselineConfig()
    df = pd.read_parquet(args.data)

    # Feature list
    num_cols, cat_cols, dropped = default_feature_columns(
        df,
        id_col=cfg.id_col,
        subject_col=cfg.subject_col,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=cfg.drop_cols,
    )
    df[num_cols] = coerce_numeric_columns(df, num_cols)

    print(f"Dropped columns (auto): {dropped[:20]} ... total={len(dropped)}")
    print(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")

    # Split per subject_id
    splits = split_by_subject(
        df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )
    train_idx = splits["train"]
    test_idx = splits["test"]

    feat_cols = num_cols + cat_cols

    # ---------------------------
    # MAIN DATA ARRAYS
    # --------------------------
    X = df[feat_cols]                               # Covariates
    T = df[cfg.treatment_col].astype(int).values    # Treatment assignments
    Y = df[cfg.outcome_col].astype(int).values      # Outcomes

    # Split data
    # Train
    X_tr, T_tr, Y_tr = X.iloc[train_idx], T[train_idx], Y[train_idx]
    # Test
    X_te, T_te, Y_te = X.iloc[test_idx], T[test_idx], Y[test_idx]

    # ---------------------------
    # Fit PS model on TRAIN
    # ---------------------------
    ps_cfg = HGBConfig(random_state=cfg.random_state)
    ps_model = make_hgb_pipeline(num_cols, cat_cols, ps_cfg)
    ps_model.fit(X_tr, T_tr)

    #ps_tr_raw = _predict_proba(ps_model, X_tr)
    ps_te_raw = _predict_proba(ps_model, X_te)

    # clip PS (for weights/stability)
    ps_clip_range = cfg.ps_clip
    #ps_tr = clip_ps(ps_tr_raw, *ps_clip_range)
    ps_te = clip_ps(ps_te_raw, *ps_clip_range)

    # ---------------------------
    # WEIGHT TRIM Q AND ESS IPTW
    # ---------------------------
    weight_trim_q = cfg.weight_trim_quantiles
    w_iptw = stabilized_iptw_weights(T_te, ps_te)
    w_iptw = trim_weights(w_iptw, *weight_trim_q)
    ess_iptw = ess(w_iptw)

    w_ato = overlap_weights(T_te, ps_te)
    w_ato = trim_weights(w_ato, *weight_trim_q)
    ess_ato = ess(w_ato)

    # ---------------------------
    # Full PS
    # ---------------------------

    ps_full_raw = _predict_proba(ps_model, X)
    ps_full = clip_ps(ps_full_raw, *ps_clip_range)


    out_cfg = HGBConfig(random_state=cfg.random_state, max_depth=3, min_samples_leaf=50)

    summaries = {}

    if args.scope in ("test", "both"):
        summaries["test"] = compute_and_save_summary(
            name="test",
            out_dir=out_dir,
            X_sub=X_te, T_sub=T_te, Y_sub=Y_te, # Note: test set
            ps_raw=ps_te_raw, ps_clip=ps_te,
            ps_clip_range=ps_clip_range,
            weight_trim_q=weight_trim_q,
            num_cols=num_cols, cat_cols=cat_cols,
            ps_cfg=ps_cfg, out_cfg=out_cfg,
            n_folds_aipw=cfg.n_folds,
            n_boot_iptw=args.n_boot_iptw,
            n_boot_aipw=args.n_boot_aipw,
            seed=cfg.random_state,
        )

    if args.scope in ("full", "both"):
        # full uses X, T, Y and ps_full
        summaries["full"] = compute_and_save_summary(
            name="full",
            out_dir=out_dir,
            X_sub=X, T_sub=T, Y_sub=Y, # Note: full set
            ps_raw=ps_full_raw, ps_clip=ps_full,
            ps_clip_range=ps_clip_range,
            weight_trim_q=weight_trim_q,
            num_cols=num_cols, cat_cols=cat_cols,
            ps_cfg=ps_cfg, out_cfg=out_cfg,
            n_folds_aipw=cfg.n_folds,
            n_boot_iptw=args.n_boot_iptw,
            n_boot_aipw=args.n_boot_aipw,
            seed=cfg.random_state,
        )

        # Payload: Add to summary also META info
        payload = {
            "meta": {
                "data": str(args.data),
                "n_total": int(len(df)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_num_features": int(len(num_cols)),
                "n_cat_features": int(len(cat_cols)),
                "ps_clip_range": {"lo": float(ps_clip_range[0]), "hi": float(ps_clip_range[1])},
                "weight_trim_quantiles": {"q_lo": float(weight_trim_q[0]), "q_hi": float(weight_trim_q[1])},
                "n_folds_aipw": int(cfg.n_folds),
                "scope": args.scope,
                "full_ps_mode": args.full_ps,
            },
            "summaries": summaries,
        }

        # Print test results summary
        if "test" in summaries:
            pe = summaries["test"]["point_estimates"]
            print("=== TEST RESULTS ===")
            print(f"ATE IPTW : {pe['ate_iptw']:.4f} | ESS: {ess_iptw:.1f}")
            print(f"ATE ATO  : {pe['ate_ato']:.4f} | ESS: {ess_ato:.1f}")
            print(f"ATE AIPW : {pe['ate_aipw']:.4f}")

        # Save JSON summary
        json_path = out_dir / "baseline_summary_recap.json"
        save_json(json_path, payload)
        print(f"Saved baseline summary -> {json_path}")


if __name__ == "__main__":
    main()
    # How to run:
    # python -m src.baseline.scripts.run_baseline --data <path_to_data.parquet> [--out_dir <output_directory>] [--scope test|full|both]
    # Scope:
    #   test: compute summary only on test set
    #   full: compute summary only on full dataset
    #   both: compute summary on both test set and full dataset
    #
    # Example:
    # python -m src.baseline.scripts.run_baseline --data "data/analytic_v0_extended_prepared.parquet" --out_dir artifacts/baseline --scope both
