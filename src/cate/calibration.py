"""
CATE calibration table construction.

What is CATE calibration?
It is similar in spirit to probability calibration for classifiers,
meaning that we want to check if the model's predicted CATE values
correspond to actual observed effects.

In particular, we want to check that higher predicted CATE values correspond
to higher observed treatment effects.

Formal procedure:
1. Take the predicted CATE values τ^(x) on a held-out test set.
2. Divide the samples into bins (e.g., deciles or quintiles) based on the predicted CATE.
3. For each bin, estimate the average treatment effect (ATE) using a robust method (e.g., AIPW or ATO).
4. Check if the estimated ATE increases with the bin index
    (i.e., higher predicted CATE bins should have higher observed ATE).

Good calibration expected:

Decile τ̂	Estimated ATE
-------------------------
low	        small or ~0
medium	    ~medium
high	    large
-------------------------

Coherent with data seen for now:
    the preferred method is to use ATO weights
    to estimate the ATE within each bin.

"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.baseline.models import clip_ps
from src.baseline.metrics import ate_weighted, ess


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


@dataclass(frozen=True)
class CalibrationConfig:
    n_bins: int = 10
    ps_clip: tuple[float, float] = (0.01, 0.99)
    method: str = "ato"  # "ato" (recommended) or "diff"


def cate_calibration_table(
    *,
    tau_hat: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    ps_hat: np.ndarray | None = None,
    cfg: CalibrationConfig = CalibrationConfig(),
) -> pd.DataFrame:
    """
    Build a CATE calibration table by binning samples by tau_hat (e.g., deciles),
    and estimating an average treatment effect within each bin.

    Parameters
    ----------
    tau_hat : np.ndarray
        Predicted CATE on the evaluation split (e.g., test).
    y : np.ndarray
        Observed binary outcome (0/1) aligned with tau_hat.
    t : np.ndarray
        Observed treatment (0/1) aligned with tau_hat.
    ps_hat : np.ndarray | None
        Propensity score predictions P(T=1|X) aligned with tau_hat.
        Required for method="ato".
    cfg : CalibrationConfig
        Calibration settings.

    Returns
    -------
    pd.DataFrame
        One row per bin with:
          - size, treated/control counts
          - mean tau_hat per bin
          - within-bin effect estimates (unweighted diff and optionally ATO)
          - ESS for weighted estimator
    """
    tau_hat = np.asarray(tau_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)

    n = len(tau_hat)
    if len(y) != n or len(t) != n:
        raise ValueError("tau_hat, y, t must have the same length")

    if cfg.method not in ("ato", "diff"):
        raise ValueError("cfg.method must be one of: 'ato', 'diff'")

    if cfg.method == "ato" and ps_hat is None:
        raise ValueError("ps_hat is required for cfg.method='ato'")

    if ps_hat is not None:
        ps_hat = np.asarray(ps_hat, dtype=float)
        if len(ps_hat) != n:
            raise ValueError("ps_hat must have same length as tau_hat")

    # Bin by quantiles of tau_hat (qcut). duplicates='drop' avoids errors if tau has ties.
    s_tau = pd.Series(tau_hat)
    bin_idx = pd.qcut(s_tau, q=cfg.n_bins, labels=False, duplicates="drop")

    rows: list[dict] = []
    lo, hi = cfg.ps_clip

    for b in sorted(bin_idx.dropna().unique()):
        mask = (bin_idx == b).to_numpy()

        tb = t[mask]
        yb = y[mask]
        taub = tau_hat[mask]

        n_b = int(mask.sum())
        n_t1 = int((tb == 1).sum())
        n_t0 = int((tb == 0).sum())

        # sanity: unweighted difference in means (NOT causal, but useful as check)
        # (works only if both groups present)
        if n_t1 > 0 and n_t0 > 0:
            diff = float(yb[tb == 1].mean() - yb[tb == 0].mean())
        else:
            diff = np.nan

        out = {
            "bin": int(b),
            "n": n_b,
            "n_treated": n_t1,
            "n_control": n_t0,
            "tau_mean": float(np.mean(taub)),
            "tau_p10": float(np.quantile(taub, 0.10)),
            "tau_p50": float(np.quantile(taub, 0.50)),
            "tau_p90": float(np.quantile(taub, 0.90)),
            "diff_unweighted": diff,
        }

        # Within-bin causal estimate using ATO weights (preferred)
        if cfg.method == "ato":
            psb = ps_hat[mask]
            psb = clip_ps(psb, lo, hi)

            wb = overlap_weights(tb, psb)

            # Handle degenerate bins (all treated or all control)
            if n_t1 == 0 or n_t0 == 0:
                out["ate_ato"] = np.nan
                out["ess_ato"] = np.nan
            else:
                out["ate_ato"] = float(ate_weighted(yb, tb, wb))
                out["ess_ato"] = float(ess(wb))

        rows.append(out)

    df_out = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    return df_out
