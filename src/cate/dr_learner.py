"""
Doubly-Robust Learner for Conditional Average Treatment Effect (CATE) estimation.

This implementation follows the DR-learner approach:
1. Fit nuisance models for propensity score e(x) and outcome models m1(x), m0(x) using cross-fitting.
2. Construct the doubly-robust pseudo-outcome ϕ.
3. Fit a regression model τ(x) on the pseudo-outcome to estimate CATE.

Design choices:
- Cross-fitting reduces leakage in nuisance estimates.
- Clipping propensity scores avoids extreme weights.
- Outcome models predict probabilities for binary Y.
- tau-model used as regressor on pseudo-outcome (HGBRegressor).

Output:
  - τ(x) = E[Y(1) - Y(0) | X=x]
  - Means to predict individual treatment effects.

Inputs:
- DataFrame of covariates X (NumPy arrays for T, Y)
- T: binary treatment (0/1)
- Y: binary outcome (0/1)
- Can handle NaN in X and categorical features (pipeline as in baseline)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.baseline.models import HGBConfig, make_hgb_pipeline, clip_ps


def _predict_proba_1(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(Y=1|X) from a fitted classifier pipeline."""
    return model.predict_proba(X)[:, 1]


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return num / (den + eps)


def dr_pseudo_outcome(
    y: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    m1: np.ndarray,
    m0: np.ndarray,
) -> np.ndarray:
    """
    Doubly-robust pseudo-outcome for CATE:
      phi = (m1 - m0) + T*(Y - m1)/e - (1-T)*(Y - m0)/(1-e)
    """
    t = t.astype(int)
    term1 = (m1 - m0)
    term2 = t * _safe_div((y - m1), e)
    term3 = (1 - t) * _safe_div((y - m0), (1 - e))
    return term1 + term2 - term3

@dataclass(frozen=True)
class DRLearnerConfig:
    n_folds: int = 5
    ps_clip: tuple[float, float] = (0.01, 0.99)

    # nuisance model configs (use default_factory to avoid mutable defaults)
    ps_cfg: HGBConfig = field(default_factory=lambda: HGBConfig(random_state=42))
    out_cfg: HGBConfig = field(default_factory=lambda: HGBConfig(
        random_state=42, max_depth=3, min_samples_leaf=50
    ))

    # tau model config (regressor)
    tau_cfg: HGBConfig = field(default_factory=lambda: HGBConfig(
        random_state=42, max_depth=3, min_samples_leaf=50
    ))


@dataclass
class DRLearnerArtifacts:
    """
    Artifacts useful for debugging and paper reporting.
    """
    cfg: DRLearnerConfig
    num_cols: list[str]
    cat_cols: list[str]

    e_oof: np.ndarray
    m1_oof: np.ndarray
    m0_oof: np.ndarray
    phi_oof: np.ndarray

    # fraction of samples clipped at bounds for e(x)
    frac_ps_clipped_lo: float
    frac_ps_clipped_hi: float

    # quick phi summary (helps detect exploding pseudo-outcomes)
    phi_mean: float
    phi_sd: float
    phi_p01: float
    phi_p50: float
    phi_p99: float


class DRLearner:
    """
      DR-learner for CATE with cross-fitted nuisance models and an HGB regressor for tau(x).
    """

    def __init__(self, cfg: DRLearnerConfig, num_cols: list[str], cat_cols: list[str]):
        self.cfg = cfg
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self._artifacts: Optional[DRLearnerArtifacts] = None
        # fitted tau model on full train (after cross-fit pseudo-outcome)
        self._tau_model: Optional[object] = None

    @property
    def fit_result(self) -> DRLearnerArtifacts:
        if self._artifacts is None:
            raise RuntimeError("DRLearner is not fit yet.")
        return self._artifacts

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> DRLearnerArtifacts:
        """
        Fit DR-learner on training data.
        Produces cross-fitted nuisance predictions and fits a tau regressor on pseudo-outcome.
        """
        X = X.reset_index(drop=True)
        t = np.asarray(t, dtype=int)
        y = np.asarray(y, dtype=int)

        n = len(y)
        assert len(t) == n and len(X) == n

        kf = KFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=self.cfg.ps_cfg.random_state)

        e_oof = np.zeros(n, dtype=float) # propensity scores e(x) out-of-fold (meaning not trained on that fold)
        m1_oof = np.zeros(n, dtype=float) # outcome model m1(x) out-of-fold
        m0_oof = np.zeros(n, dtype=float) # outcome model m0(x) out-of-fold

        lo, hi = self.cfg.ps_clip # clip range for propensity scores

        # --- cross-fit nuisance models
        for tr_idx, te_idx in kf.split(X):
            x_tr, x_te = X.iloc[tr_idx], X.iloc[te_idx]
            t_tr, y_tr = t[tr_idx], y[tr_idx]

            # propensity e(x)
            ps_model = make_hgb_pipeline(self.num_cols, self.cat_cols, self.cfg.ps_cfg)
            ps_model.fit(x_tr, t_tr)
            e_hat = _predict_proba_1(ps_model, x_te)
            e_hat = clip_ps(e_hat, lo, hi)
            e_oof[te_idx] = e_hat

            # outcome models m1, m0 (probability of Y=1)
            out1 = make_hgb_pipeline(self.num_cols, self.cat_cols, self.cfg.out_cfg)
            out0 = make_hgb_pipeline(self.num_cols, self.cat_cols, self.cfg.out_cfg)

            mask1 = (t_tr == 1)
            mask0 = (t_tr == 0)
            if mask1.sum() == 0 or mask0.sum() == 0:
                raise ValueError(
                    "A fold has only treated or only control samples. "
                    "Reduce n_folds or stratify folds by treatment."
                )

            out1.fit(x_tr[mask1], y_tr[mask1])
            out0.fit(x_tr[mask0], y_tr[mask0])

            m1_oof[te_idx] = _predict_proba_1(out1, x_te)
            m0_oof[te_idx] = _predict_proba_1(out0, x_te)

        # --- pseudo-outcome on train (OOF)
        phi_oof = dr_pseudo_outcome(y=y, t=t, e=e_oof, m1=m1_oof, m0=m0_oof)

        # --- tau model: regress phi on X
        # We need a regressor pipeline. If your make_hgb_pipeline only builds classifiers,
        # add a make_hgb_regressor_pipeline() in baseline.models and use it here.
        from src.baseline.models import make_hgb_regressor_pipeline  # <-- to implement next

        tau_model = make_hgb_regressor_pipeline(self.num_cols, self.cat_cols, self.cfg.tau_cfg)
        tau_model.fit(X, phi_oof)

        # diagnostics
        frac_lo = float(np.mean(e_oof <= lo + 1e-15))
        frac_hi = float(np.mean(e_oof >= hi - 1e-15))
        q01, q50, q99 = np.quantile(phi_oof, [0.01, 0.50, 0.99])

        self._tau_model = tau_model
        self._artifacts = DRLearnerArtifacts(
            cfg=self.cfg,
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            e_oof=e_oof,
            m1_oof=m1_oof,
            m0_oof=m0_oof,
            phi_oof=phi_oof,
            frac_ps_clipped_lo=frac_lo,
            frac_ps_clipped_hi=frac_hi,
            phi_mean=float(np.mean(phi_oof)),
            phi_sd=float(np.std(phi_oof, ddof=1)),
            phi_p01=float(q01),
            phi_p50=float(q50),
            phi_p99=float(q99),
        )
        return self._artifacts

    def predict_tau(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict CATE tau(x) for new samples.
        """
        if self._tau_model is None:
            raise RuntimeError("DRLearner not fit yet.")
        X = X.reset_index(drop=True)
        tau = self._tau_model.predict(X)
        return np.asarray(tau, dtype=float)
