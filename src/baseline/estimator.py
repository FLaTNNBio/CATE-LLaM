"""
Baseline estimators for causal inference: IPTW, AIPW with cross-fitting.
Dataclass BaselineResults: stores results including PS AUC, ATE estimates, ESS, and SMD table.

Main functions:
- _predict_proba: predict probabilities using a trained model.
    Probabilities for the positive class of binary classification.
- aipw_crossfit: cross-fitted AIPW estimator for ATE.
    Uses K-Fold cross-fitting to estimate propensity scores and outcome models.
    K-Fold splits the data into n_folds subsets, training on n-1 folds and validating on the held-out fold.
- run_baseline: runs baseline estimators (IPTW, AIPW) and returns results.

"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from .models import HGBConfig, make_hgb_pipeline, clip_ps
from .metrics import (
    stabilized_iptw_weights, trim_weights, ess, ate_weighted, smd_table
)

@dataclass
class BaselineResults:
    feature_cols: list[str]

    # PS diagnostics
    ps_auc_train: float
    ps_auc_test: float

    # IPTW
    ate_iptw: float
    ess_iptw: float

    # AIPW
    ate_aipw: float

    # balance
    smd: pd.DataFrame

def _predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    """
    Predict probability using a trained model.
    :param model: model to use
    :param x: data to predict
    :return: numpy array of predictions
    """
    # sklearn HGB: predict_proba available
    return model.predict_proba(x)[:, 1]


def aipw_crossfit(
    x: pd.DataFrame,
    treat: np.ndarray,
    outcome: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    ps_cfg: HGBConfig,
    out_cfg: HGBConfig,
    n_folds: int,
    ps_clip_range: tuple[float, float],
) -> float:
    """
    Cross-fitted AIPW / Doubly-Robust estimator for ATE.

    Inputs:
      x: covariates (can include NaN; categoricals allowed as strings)
      treat: treatment assignment (0/1)
      outcome: observed outcome (0/1)
    """
    n = len(outcome)
    assert len(treat) == n and len(x) == n

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=ps_cfg.random_state)

    ps_hat = np.zeros(n, dtype=float)
    mu1_hat = np.zeros(n, dtype=float)
    mu0_hat = np.zeros(n, dtype=float)

    lo, hi = ps_clip_range

    for tr_idx, te_idx in kf.split(x):
        x_tr, x_te = x.iloc[tr_idx], x.iloc[te_idx]
        t_tr, y_tr = treat[tr_idx], outcome[tr_idx]

        # --- Propensity model e(x)
        ps_model = make_hgb_pipeline(num_cols, cat_cols, ps_cfg)
        ps_model.fit(x_tr, t_tr)
        ps_hat[te_idx] = _predict_proba(ps_model, x_te)

        # --- Outcome models m1(x), m0(x)
        # Fit on treated/control subsets of the TRAIN fold
        out1 = make_hgb_pipeline(num_cols, cat_cols, out_cfg)
        out0 = make_hgb_pipeline(num_cols, cat_cols, out_cfg)

        mask1 = (t_tr == 1)
        mask0 = (t_tr == 0)

        # safety: ensure both groups exist in each fold
        if mask1.sum() == 0 or mask0.sum() == 0:
            raise ValueError(
                "A fold has only treated or only control samples. "
                "Reduce n_folds or stratify folds by treatment."
            )

        out1.fit(x_tr[mask1], y_tr[mask1])
        out0.fit(x_tr[mask0], y_tr[mask0])

        mu1_hat[te_idx] = _predict_proba(out1, x_te)
        mu0_hat[te_idx] = _predict_proba(out0, x_te)

    ps_hat = clip_ps(ps_hat, lo, hi)

    # AIPW score for each i:
    # tau_i = (mu1 - mu0) + T*(Y - mu1)/ps - (1-T)*(Y - mu0)/(1-ps)
    term1 = (mu1_hat - mu0_hat)
    term2 = treat * (outcome - mu1_hat) / ps_hat
    term3 = (1 - treat) * (outcome - mu0_hat) / (1 - ps_hat)
    tau_i = term1 + term2 - term3

    return float(np.mean(tau_i))

def run_baseline(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    treatment_col: str,
    outcome_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_folds: int = 5,
    ps_clip_range: tuple[float, float] = (0.01, 0.99),
    weight_trim_q: tuple[float, float] = (0.01, 0.99),
    seed: int = 42,
) -> BaselineResults:
    """
    Run baseline estimators (IPTW, AIPW) and return results.
    :param df: dataframe
    :param num_cols: numerical feature columns to use
    :param cat_cols: categorical feature columns to use
    :param treatment_col: treatment column name
    :param outcome_col: outcome column name
    :param train_idx: train indices from split
    :param test_idx: test indices from split
    :param n_folds: number of folds for cross-fitting
    :param ps_clip_range: propensity score clipping range
    :param weight_trim_q:  weight trimming quantiles
    :param seed: seed for reproducibility
    :return: BaselineResults dataclass with results
    """
    feat_cols = num_cols + cat_cols
    x = df[feat_cols]

    treat = df[treatment_col].astype(int).values # treatment assignments
    y_obs = df[outcome_col].astype(int).values # outcomes

    # split data
    # tr = train
    # te = test
    # x= features, t= treatment, y= outcome
    x_tr, t_tr, y_tr = x.iloc[train_idx], treat[train_idx], y_obs[train_idx]
    x_te, t_te, y_te = x.iloc[test_idx], treat[test_idx], y_obs[test_idx]

    # --- PS model for IPTW (fit on train, evaluate on test)
    ps_cfg = HGBConfig(random_state=seed)
    ps_model = make_hgb_pipeline(num_cols, cat_cols, ps_cfg)

    dt_cols = [c for c in x_tr.columns if pd.api.types.is_datetime64_any_dtype(x_tr[c])]
    assert not dt_cols, f"Datetime leaked into X: {dt_cols}"

    ps_model.fit(x_tr, t_tr)

    # predict PS
    ps_tr = _predict_proba(ps_model, x_tr)
    ps_te = _predict_proba(ps_model, x_te)

    # clip PS
    ps_tr = clip_ps(ps_tr, *ps_clip_range)
    ps_te = clip_ps(ps_te, *ps_clip_range)

    # PS AUC
    ps_auc_train = roc_auc_score(t_tr, ps_tr)
    ps_auc_test = roc_auc_score(t_te, ps_te)

    # --- IPTW on test (paper style: train nuisance, evaluate estimand on held-out)
    w_te = stabilized_iptw_weights(t_te, ps_te)
    w_te = trim_weights(w_te, *weight_trim_q)

    # ATE and ESS
    ate_iptw = ate_weighted(y_te, t_te, w_te)
    ess_iptw = ess(w_te)

    # --- SMD table before/after weighting ---
    smd = smd_table(x_te[num_cols], t_te, w=w_te)

    # --- AIPW cross-fitted on test (cross-fitting inside test for unbiased value estimate)
    # You can also do cross-fitting on train and then evaluate on test; but this variant gives a clean OOS estimate
    out_cfg = HGBConfig(random_state=seed, max_depth=3, min_samples_leaf=50)
    ate_aipw = aipw_crossfit(
        x=x_te,
        treat=t_te,
        outcome=y_te,
        num_cols=num_cols,
        cat_cols=cat_cols,
        ps_cfg=ps_cfg,
        out_cfg=out_cfg,
        n_folds=n_folds,
        ps_clip_range=ps_clip_range,
    )

    return BaselineResults(
        feature_cols=feat_cols,
        ps_auc_train=float(ps_auc_train),
        ps_auc_test=float(ps_auc_test),
        ate_iptw=float(ate_iptw),
        ess_iptw=float(ess_iptw),
        ate_aipw=float(ate_aipw),
        smd=smd,
    )
