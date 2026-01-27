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
    return model.predict_proba(x)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to analytic_v0_extended_prepared.parquet")
    ap.add_argument("--out_dir", default="artifacts", help="Directory for outputs")
    ap.add_argument("--n_boot_iptw", type=int, default=100)
    ap.add_argument("--n_boot_aipw", type=int, default=100)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # debug: ensure you run what you edited
    print("baseline package:", src.baseline.__file__)
    print("estimator file:", src.baseline.estimator.__file__)
    print("features file:", src.baseline.features.__file__)

    cfg = BaselineConfig()
    df = pd.read_parquet(args.data)

    # feature list
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

    # split per subject_id
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
    X = df[feat_cols]
    T = df[cfg.treatment_col].astype(int).values
    Y = df[cfg.outcome_col].astype(int).values

    X_tr, T_tr, Y_tr = X.iloc[train_idx], T[train_idx], Y[train_idx]
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
    # IPTW (ATE) on TEST
    # ---------------------------
    weight_trim_q = cfg.weight_trim_quantiles
    w_iptw = stabilized_iptw_weights(T_te, ps_te)
    w_iptw = trim_weights(w_iptw, *weight_trim_q)

    ate_iptw = ate_weighted(Y_te, T_te, w_iptw)
    ess_iptw = ess(w_iptw)

    # balance (numeric only to avoid categorical SMD hassles)
    smd_iptw = smd_table(X_te[num_cols], T_te, w=w_iptw)

    # ---------------------------
    # ATO (Overlap weights) on TEST
    # ---------------------------
    w_ato = overlap_weights(T_te, ps_te)
    # (optional) trimming ATO weights is usually not necessary, but safe:
    w_ato = trim_weights(w_ato, *weight_trim_q)

    ate_ato = ate_weighted(Y_te, T_te, w_ato)
    ess_ato = ess(w_ato)
    smd_ato = smd_table(X_te[num_cols], T_te, w=w_ato)

    # ---------------------------
    # AIPW (DR) on TEST (cross-fit)
    # ---------------------------
    out_cfg = HGBConfig(random_state=cfg.random_state, max_depth=3, min_samples_leaf=50)

    # Use your existing cross-fit implementation inside estimator.py
    from src.baseline.estimator import aipw_crossfit
    ate_aipw = aipw_crossfit(
        x=X_te,
        treat=T_te,
        outcome=Y_te,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ps_cfg=ps_cfg,
        out_cfg=out_cfg,
        n_folds=cfg.n_folds,
        ps_clip_range=ps_clip_range,
    )

    # ---------------------------
    # Print headline results
    # ---------------------------
    print("=== BASELINE RESULTS (TEST) ===")
    print(f"ATE IPTW : {ate_iptw:.4f} | ESS: {ess_iptw:.1f}")
    print(f"ATE ATO  : {ate_ato:.4f} | ESS: {ess_ato:.1f}")
    print(f"ATE AIPW : {ate_aipw:.4f}")

    # save SMDs
    smd_iptw_path = out_dir / "baseline_smd_iptw.csv"
    smd_ato_path = out_dir / "baseline_smd_ato.csv"
    smd_iptw.to_csv(smd_iptw_path, index=False)
    smd_ato.to_csv(smd_ato_path, index=False)
    print(f"Saved SMD IPTW -> {smd_iptw_path}")
    print(f"Saved SMD ATO  -> {smd_ato_path}")

    # ---------------------------
    # Build JSON summary
    # ---------------------------
    summary = {
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
        },
        "ps_overlap_test_preclip": ps_overlap_summary(ps_te_raw, T_te, clip_range=ps_clip_range),
        "ps_overlap_test_postclip": ps_overlap_summary(ps_te, T_te, clip_range=ps_clip_range),
        "weights_iptw": weights_summary(w_iptw, n_total=len(w_iptw)),
        "weights_ato": weights_summary(w_ato, n_total=len(w_ato)),
        "balance_iptw": balance_summary(smd_iptw, top_k=20),
        "balance_ato": balance_summary(smd_ato, top_k=20),
        "point_estimates": {
            "ate_iptw": float(ate_iptw),
            "ate_ato": float(ate_ato),
            "ate_aipw": float(ate_aipw),
        },
    }

    # ---------------------------
    # Bootstrap CI
    # ---------------------------
    ci_iptw = bootstrap_ate_iptw(
        outcome=Y_te,
        treat=T_te,
        ps=ps_te,
        n_boot=args.n_boot_iptw,
        seed=cfg.random_state,
        ps_clip_range=ps_clip_range,
        weight_trim_q=weight_trim_q,
    )

    ci_aipw = bootstrap_ate_aipw(
        X=X_te,
        Y=Y_te,
        T=T_te,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ps_cfg=ps_cfg,
        out_cfg=out_cfg,
        n_folds=cfg.n_folds,
        ps_clip_range=ps_clip_range,
        n_boot=args.n_boot_aipw,
        seed=cfg.random_state,
    )

    # ATO bootstrap CI (cheap): same as IPTW but with overlap weights
    # We'll implement inline quickly:
    rng = np.random.default_rng(cfg.random_state)
    n = len(Y_te)
    idx = np.arange(n)
    reps_ato = np.empty(args.n_boot_iptw, dtype=float)
    for b in range(args.n_boot_iptw):
        b_idx = rng.choice(idx, size=n, replace=True)
        Yb, Tb, psb = Y_te[b_idx], T_te[b_idx], ps_te[b_idx]
        wb = overlap_weights(Tb, psb)
        wb = trim_weights(wb, *weight_trim_q)
        reps_ato[b] = ate_weighted(Yb, Tb, wb)
    # reuse ci_from_bootstrap already in summary.py
    from src.baseline.summary import ci_from_bootstrap
    ci_ato = {"type": "ato_bootstrap", "n_boot": int(args.n_boot_iptw), "alpha": 0.05, "ci": ci_from_bootstrap(reps_ato)}

    summary["bootstrap_ci"] = {"iptw": ci_iptw, "aipw": ci_aipw, "ato": ci_ato}

    # save JSON
    json_path = out_dir / "baseline_summary.json"
    save_json(json_path, summary)
    print(f"Saved baseline summary -> {json_path}")


if __name__ == "__main__":
    main()
