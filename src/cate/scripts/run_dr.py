from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline.config import BaselineConfig
from src.baseline.models import HGBConfig, make_hgb_pipeline
from src.baseline.split import split_by_subject
from src.baseline.features import default_feature_columns, coerce_numeric_columns
from src.baseline.summary import save_json

from src.cate.dr_learner import DRLearner, DRLearnerConfig
from src.config import get_config, CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="Path to analytic_v0_extended_prepared.parquet")
    ap.add_argument("--dataset", choices=list(CONFIGS.keys()), default="rbc_v1", help="Which dataset config to use")
    ap.add_argument("--out_dir",  help="Output directory")
    ap.add_argument("--n_folds", type=int, default=5, help="Override cross-fitting folds (default: cfg.n_folds)")
    ap.add_argument("--seed", type=int, default=42, help="Override random seed (default: cfg.random_state)")
    args = ap.parse_args()

    cfg = get_config(args.dataset)

    if args.data is None:
        args.data = cfg.data_path
    if args.out_dir is None:
        args.out_dir = cfg.out_dir

    seed = cfg.random_state if args.seed is None else int(args.seed)
    n_folds = cfg.n_folds if args.n_folds is None else int(args.n_folds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)

    # --- feature selection (uses cfg.drop_cols -> includes has_hba1c_1)
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

    print(f"Dropped columns (auto): {dropped[:20]} ... total={len(dropped)}")
    print(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")
    print(f"Total features used: {len(feat_cols)}")

    # --- split by subject_id
    splits = split_by_subject(
        df,
        subject_col=cfg.subject_col,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=seed,
    )
    tr_idx = splits["train"]
    te_idx = splits["test"]

    X = df[feat_cols]
    T = df[cfg.treatment_col].astype(int).values
    Y = df[cfg.outcome_col].astype(int).values

    X_tr, T_tr, Y_tr = X.iloc[tr_idx], T[tr_idx], Y[tr_idx]
    X_te, T_te, Y_te = X.iloc[te_idx], T[te_idx], Y[te_idx]

    # --- fit DR learner
    dr_cfg = DRLearnerConfig(
        n_folds=n_folds,
        ps_clip=cfg.ps_clip,
        ps_cfg=HGBConfig(random_state=seed),
        out_cfg=HGBConfig(random_state=seed, max_depth=3, min_samples_leaf=50),
        tau_cfg=HGBConfig(random_state=seed, max_depth=3, min_samples_leaf=50),
    )

    learner = DRLearner(cfg=dr_cfg, num_cols=num_cols, cat_cols=cat_cols)
    learner.fit(X_tr, T_tr, Y_tr)

    # --- predict CATE on test
    tau_hat = learner.predict_tau(X_te)

    # PS Estimation
    ps_cfg = HGBConfig(random_state=seed)
    ps_model = make_hgb_pipeline(num_cols, cat_cols, ps_cfg)
    ps_model.fit(X_tr, T_tr)
    ps_hat_te = ps_model.predict_proba(X_te)[:, 1]

    # --- save predictions
    out_pred = pd.DataFrame({
        cfg.subject_col: df.iloc[te_idx][cfg.subject_col].values,
        cfg.id_col: df.iloc[te_idx][cfg.id_col].values,
        "ps_hat": ps_hat_te,
        "tau_hat": tau_hat,
        cfg.treatment_col: T_te,
        cfg.outcome_col: Y_te,
    })
    pred_path = out_dir / "dr_tau_test.parquet"
    out_pred.to_parquet(pred_path, index=False)
    print(f"Saved tau predictions -> {pred_path}")

    # --- save diagnostics
    art = learner._artifacts
    summary = {
        "meta": {
            "data": str(args.data),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "n_num_features": int(len(num_cols)),
            "n_cat_features": int(len(cat_cols)),
            "n_folds": int(n_folds),
            "seed": int(seed),
            "ps_clip": {"lo": float(cfg.ps_clip[0]), "hi": float(cfg.ps_clip[1])},
            "dropped_cols": list(cfg.drop_cols) if cfg.drop_cols is not None else [],
        },
        "phi_summary_train_oof": {
            "mean": art.phi_mean,
            "sd": art.phi_sd,
            "p01": art.phi_p01,
            "p50": art.phi_p50,
            "p99": art.phi_p99,
        },
        "ps_clip_train_oof": {
            "frac_clipped_lo": art.frac_ps_clipped_lo,
            "frac_clipped_hi": art.frac_ps_clipped_hi,
        },
        "tau_hat_test": {
            "mean": float(np.mean(tau_hat)),
            "sd": float(np.std(tau_hat, ddof=1)),
            "p01": float(np.quantile(tau_hat, 0.01)),
            "p50": float(np.quantile(tau_hat, 0.50)),
            "p99": float(np.quantile(tau_hat, 0.99)),
        },
    }

    summary_path = out_dir / "dr_summary.json"
    save_json(summary_path, summary)
    print(f"Saved DR summary -> {summary_path}")


if __name__ == "__main__":
    main()


