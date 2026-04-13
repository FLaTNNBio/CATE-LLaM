"""
    Summary statistics for propensity scores, weights, and covariate balance.
    Main functions:
    - ps_overlap_summary: summarize propensity score overlap diagnostics.
    - weights_summary: summarize weight distribution and effective sample size.
    - balance_summary: summarize covariate balance from SMD table.
"""
from typing import Any
import numpy as np
import pandas as pd
import json
from pathlib import Path


def _as_float(x) -> float:
    return float(x) if x is not None else float("nan")


def _quantiles(arr: np.ndarray, qs=(0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)) -> dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    vals = np.quantile(arr, qs)
    out = {}
    for q, v in zip(qs, vals):
        key = "min" if q == 0.0 else "max" if q == 1.0 else f"p{int(q*100):02d}"
        out[key] = float(v)
    return out


def ps_overlap_summary(
    ps: np.ndarray,
    treat: np.ndarray,
    clip_range: tuple[float, float] = (0.01, 0.99),
) -> dict[str, Any]:
    """
    Summarize propensity score overlap (positivity) diagnostics.

    Inputs:
      ps: propensity scores (prob of T=1), shape (n,)
      T: treatment indicator 0/1, shape (n,)
      clip_range: (lo, hi) used in pipeline

    Output keys:
      - overall quantiles
      - quantiles by treated/control
      - fractions outside clip range (pre-clip)
      - simple overlap flags
    """
    ps = np.asarray(ps, dtype=float)
    treat = np.asarray(treat, dtype=int)
    assert ps.shape[0] == treat.shape[0], "ps and T must have same length"

    lo, hi = clip_range
    treated = ps[treat == 1]
    control = ps[treat == 0]

    frac_below_lo = float(np.mean(ps < lo))
    frac_above_hi = float(np.mean(ps > hi))

    out = {
        "clip_range": {"lo": float(lo), "hi": float(hi)},
        "n": int(ps.size),
        "n_treated": int((treat == 1).sum()),
        "n_control": int((treat == 0).sum()),
        "overall": _quantiles(ps),
        "treated": _quantiles(treated),
        "control": _quantiles(control),
        "frac_ps_below_lo": frac_below_lo,
        "frac_ps_above_hi": frac_above_hi,
        "frac_ps_outside_clip": float(frac_below_lo + frac_above_hi),
    }

    # Simple overlap heuristics (useful to log; can be plotted later)
    # If treated ps are mostly very high and control ps mostly very low, overlap is likely weak.
    out["heuristics"] = {
        "treated_p05": out["treated"].get("p05", float("nan")),
        "treated_p95": out["treated"].get("p95", float("nan")),
        "control_p05": out["control"].get("p05", float("nan")),
        "control_p95": out["control"].get("p95", float("nan")),
        "possible_separation": bool(
            (out["treated"].get("p05", 1.0) > 0.8) and (out["control"].get("p95", 0.0) < 0.2)
        ),
    }

    return out


def weights_summary(
    w: np.ndarray,
    n_total: int | None = None,
    quantiles=(0.5, 0.9, 0.95, 0.99, 1.0),
) -> dict[str, Any]:
    """
    Summarize weight distribution and effective sample size (ESS).

    Inputs:
      w: weights array
      n_total: optional reference N to compute ess_ratio; if None uses len(w)
      quantiles: weight quantiles to report (include 1.0 for max)

    Output:
      - ess, ess_ratio
      - weight quantiles
      - basic sanity stats
    """
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return {
            "n": 0,
            "ess": float("nan"),
            "ess_ratio": float("nan"),
            "weights": {},
            "sanity": {},
        }

    N = int(n_total) if n_total is not None else int(w.size)

    # ESS = (sum w)^2 / sum w^2
    sw = w.sum()
    ess = float((sw * sw) / (np.sum(w * w) + 1e-12))
    ess_ratio = float(ess / max(N, 1))

    q_vals = np.quantile(w, list(quantiles))
    w_q = {}
    for q, v in zip(quantiles, q_vals):
        key = "max" if q == 1.0 else f"p{int(q*100):02d}"
        w_q[key] = float(v)

    out = {
        "n": int(w.size),
        "ess": ess,
        "ess_ratio": ess_ratio,
        "weights": w_q,
        "sanity": {
            "min": float(np.min(w)),
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "max": float(np.max(w)),
        },
    }
    return out


def balance_summary(
    smd_df: pd.DataFrame,
    post_col: str = "smd_post",
    feature_col: str = "feature",
    top_k: int = 20,
    thresholds: tuple[float, float] = (0.1, 0.2),
) -> dict[str, Any]:
    """
    Summarize covariate balance from an SMD table.

    Expected columns in smd_df:
      - feature (name)
      - smd_pre
      - smd_post (or chosen post_col)

    Output:
      - max_abs_post
      - fractions above thresholds
      - top_k features by abs(post)
    """
    if smd_df is None or len(smd_df) == 0:
        return {
            "n_features": 0,
            "max_abs_post": float("nan"),
            "frac_abs_post_gt_0.1": float("nan"),
            "frac_abs_post_gt_0.2": float("nan"),
            "top": [],
        }

    if post_col not in smd_df.columns:
        raise ValueError(f"balance_summary expects column '{post_col}' in smd_df")

    post = pd.to_numeric(smd_df[post_col], errors="coerce").to_numpy(dtype=float)
    abs_post = np.abs(post)
    abs_post = abs_post[np.isfinite(abs_post)]

    if abs_post.size == 0:
        return {
            "n_features": int(len(smd_df)),
            "max_abs_post": float("nan"),
            "frac_abs_post_gt_0.1": float("nan"),
            "frac_abs_post_gt_0.2": float("nan"),
            "top": [],
        }

    thr1, thr2 = thresholds
    frac_thr1 = float(np.mean(abs_post > thr1))
    frac_thr2 = float(np.mean(abs_post > thr2))

    # Top-K offenders by abs(post)
    tmp = smd_df.copy()
    tmp[post_col] = pd.to_numeric(tmp[post_col], errors="coerce")
    tmp["abs_post"] = tmp[post_col].abs()
    tmp = tmp.sort_values("abs_post", ascending=False).head(top_k)

    top = []
    for _, r in tmp.iterrows():
        top.append({
            "feature": str(r.get(feature_col)),
            "smd_post": _as_float(r.get(post_col)),
            "smd_pre": _as_float(r.get("smd_pre")) if "smd_pre" in tmp.columns else float("nan"),
        })

    return {
        "n_features": int(len(smd_df)),
        "max_abs_post": float(np.max(abs_post)),
        f"frac_abs_post_gt_{thr1}": frac_thr1,
        f"frac_abs_post_gt_{thr2}": frac_thr2,
        "top": top,
    }

# ============= Bootstrap CI helpers =============

def ci_from_bootstrap(samples: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """
    Helpers to compute bootstrap confidence intervals.
    Compute mean, standard deviation, and (1-alpha) confidence interval from bootstrap samples.
    :param samples: array of bootstrap samples
    :param alpha: significance level for CI (default 0.05 for 95% CI)
    :return: dict with keys: mean, sd, ci_lo, ci_hi, n_boot
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}

    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return {
        "mean": float(np.mean(samples)),
        "sd": float(np.std(samples, ddof=1)),
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot": int(samples.size),
    }

from typing import Callable

def bootstrap_ate_iptw(
    outcome: np.ndarray,
    treat: np.ndarray,
    ps: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
    ps_clip_range: tuple[float, float] = (0.01, 0.99),
    weight_trim_q: tuple[float, float] = (0.01, 0.99),
    ate_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float] | None = None,
    weights_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    trim_fn: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
) -> dict[str, Any]:
    """
    Bootstrap CI for IPTW ATE using fixed ps (computed out-of-sample already).
    This is fast and sufficient for baseline uncertainty reporting.

    Inputs:
      outcome (Y), treat (T), ps: arrays on the evaluation set (e.g., test)
    """
    rng = np.random.default_rng(seed)
    n = len(outcome)
    idx = np.arange(n)

    # default functions (imported lazily to avoid circular imports)
    if ate_fn is None or weights_fn is None or trim_fn is None:
        from src.baseline.metrics import ate_weighted, stabilized_iptw_weights, trim_weights
        ate_fn = ate_weighted if ate_fn is None else ate_fn
        weights_fn = stabilized_iptw_weights if weights_fn is None else weights_fn
        trim_fn = trim_weights if trim_fn is None else trim_fn

    lo, hi = ps_clip_range
    qlo, qhi = weight_trim_q

    reps = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        b_idx = rng.choice(idx, size=n, replace=True)
        Yb, Tb, psb = outcome[b_idx], treat[b_idx], ps[b_idx]

        psb = np.clip(psb, lo, hi)
        wb = weights_fn(Tb, psb)
        wb = trim_fn(wb, qlo, qhi)

        reps[b] = ate_fn(Yb, Tb, wb)

    return {
        "type": "iptw_bootstrap",
        "n_boot": int(n_boot),
        "alpha": 0.05,
        "ci": ci_from_bootstrap(reps, alpha=0.05),
    }

def bootstrap_ate_aipw(
    X: pd.DataFrame,
    Y: np.ndarray,
    T: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    ps_cfg,
    out_cfg,
    n_folds: int,
    ps_clip_range: tuple[float, float],
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Bootstrap CI for AIPW ATE by re-running cross-fitted AIPW on each bootstrap resample.
    Slower but closer to end-to-end uncertainty.

    Inputs are for the evaluation set (e.g. test).
    """
    rng = np.random.default_rng(seed)
    n = len(Y)
    idx = np.arange(n)

    from .estimator import aipw_crossfit  # avoid circular import at module import time

    reps = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        b_idx = rng.choice(idx, size=n, replace=True)

        Xb = X.iloc[b_idx].reset_index(drop=True)
        Yb = Y[b_idx]
        Tb = T[b_idx]

        reps[b] = aipw_crossfit(
            x=Xb,
            treat=Tb,
            outcome=Yb,
            num_cols=num_cols,
            cat_cols=cat_cols,
            ps_cfg=ps_cfg,
            out_cfg=out_cfg,
            n_folds=n_folds,
            ps_clip_range=ps_clip_range,
        )

    return {
        "type": "aipw_bootstrap",
        "n_boot": int(n_boot),
        "alpha": 0.05,
        "ci": ci_from_bootstrap(reps, alpha=0.05),
    }

# ============= JSON helpers =============

def to_jsonable(obj: Any) -> Any:
    """
    Recursively convert common numpy/pandas objects to JSON-serializable Python types.
    """
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas
    if isinstance(obj, (pd.Series,)):
        return obj.to_list()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient="records")

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    # pathlib
    if isinstance(obj, Path):
        return str(obj)

    # fallback
    return obj


def save_json(path: str | Path, data: dict[str, Any], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = to_jsonable(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)
