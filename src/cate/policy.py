"""
Policy value estimation for treatment policies derived from CATE estimates.

Idea:
    - Use τ^(x) to define a policy π(x)
    - Estimate expected mortality if everyone followed that policy
    - Use doubly robust (DR) estimator
        V(π) = E[ μ_π(X) + I(T=π(X))/P(T|X) * (Y - μ_T(X)) ]
    where μ_π = π*μ1 + (1-π)*μ0
    and P(T|X) = e(X) if T=1 else 1-e(X)

Policies supported:
    - treat-if-tau>0
    - treat-none
    - treat-all

Output:
    - Estimated policy value V(π) with bootstrap CI
    - Lower is better since outcome is mortality (0/1)
    
"""

POLICIES = ["tau_gt_0", "tau_lt_0", "top_frac_benefit"]

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _clip_ps(ps: np.ndarray, lo: float, hi: float) -> np.ndarray:
    ps = np.asarray(ps, dtype=float)
    return np.clip(ps, lo, hi)


def policy_from_tau(tau_hat: np.ndarray, kind: str="tau_lt_0", top_frac: float=0.2) -> np.ndarray:
    """
    Build a deterministic treatment policy pi(x) in {0,1} from tau_hat.

    kind:
      - "tau_gt_0": treat if tau_hat > 0
      - "tau_lt_0": treat if tau_hat < 0
      - "top_frac_benefit": treat top fraction (most negative tau_hat)
    """
    tau_hat = np.asarray(tau_hat, dtype=float)

    if kind == "tau_lt_0":
        return (tau_hat < 0.0).astype(int)

    if kind == "tau_gt_0":
        return (tau_hat > 0.0).astype(int)

    if kind == "top_frac_benefit":
        # treat chi ha tau più negativo (massimo beneficio)
        if not (0.0 < top_frac < 1.0):
            raise ValueError("top_frac must be in (0,1)")
        thr = np.quantile(tau_hat, top_frac)  # lower = more negative
        return (tau_hat <= thr).astype(int)

    raise ValueError("unknown kind")



@dataclass(frozen=True)
class PolicyValueConfig:
    ps_clip: tuple[float, float] = (0.01, 0.99)
    # Optional trimming of the IPW factor match / p_t, e.g. (0.05, 0.95)
    ipw_trim_quantiles: tuple[float, float] | None = None
    eps: float = 1e-12


def dr_policy_value(
    *,
    y: np.ndarray,
    t: np.ndarray,
    pi: np.ndarray,
    ps_hat: np.ndarray,
    mu1_hat: np.ndarray,
    mu0_hat: np.ndarray,
    cfg: PolicyValueConfig = PolicyValueConfig(),
) -> float:
    """
    Doubly-robust policy value:
      V(pi) = E[ mu_pi(X) + I(T=pi(X))/P(T|X) * (Y - mu_T(X)) ]

    If cfg.ipw_trim_quantiles is set, we trim the IPW multiplier I(T=pi)/P(T|X)
    by quantiles computed over matched units only (match==1).
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    pi = np.asarray(pi, dtype=int)
    ps_hat = np.asarray(ps_hat, dtype=float)
    mu1_hat = np.asarray(mu1_hat, dtype=float)
    mu0_hat = np.asarray(mu0_hat, dtype=float)

    n = len(y)
    if not (len(t) == len(pi) == len(ps_hat) == len(mu1_hat) == len(mu0_hat) == n):
        raise ValueError("All inputs must have the same length")

    lo, hi = cfg.ps_clip
    e = _clip_ps(ps_hat, lo, hi)

    # mu under the policy
    mu_pi = pi * mu1_hat + (1 - pi) * mu0_hat

    # mu of observed treatment
    mu_t = t * mu1_hat + (1 - t) * mu0_hat

    # probability of observed treatment
    p_t = t * e + (1 - t) * (1 - e)

    # indicator that observed action matches policy action
    match = (t == pi).astype(float)
    ipw = match / (p_t + cfg.eps)  # only nonzero where match==1

    # Optional IPW trimming (paired with how you run baseline)
    if cfg.ipw_trim_quantiles is not None:
        q_lo, q_hi = cfg.ipw_trim_quantiles
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("ipw_trim_quantiles must satisfy 0 <= q_lo < q_hi <= 1")

        m = (match == 1.0)
        if np.any(m):
            ipw_m = ipw[m]
            lo_w = float(np.quantile(ipw_m, q_lo))
            hi_w = float(np.quantile(ipw_m, q_hi))
            ipw[m] = np.clip(ipw[m], lo_w, hi_w)

    score = mu_pi + ipw * (y - mu_t)
    return float(np.mean(score))


def bootstrap_policy_value(
    *,
    y: np.ndarray,
    t: np.ndarray,
    pi: np.ndarray,
    ps_hat: np.ndarray,
    mu1_hat: np.ndarray,
    mu0_hat: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
    alpha: float = 0.05,
    cfg: PolicyValueConfig = PolicyValueConfig(),
) -> dict:
    """
    Nonparametric bootstrap CI for DR policy value.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)

    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        b_idx = rng.choice(idx, size=n, replace=True)
        reps[b] = dr_policy_value(
            y=y[b_idx],
            t=t[b_idx],
            pi=pi[b_idx],
            ps_hat=ps_hat[b_idx],
            mu1_hat=mu1_hat[b_idx],
            mu0_hat=mu0_hat[b_idx],
            cfg=cfg,
        )

    lo_q = float(np.quantile(reps, alpha / 2))
    hi_q = float(np.quantile(reps, 1 - alpha / 2))

    return {
        "type": "dr_policy_bootstrap",
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "mean": float(np.mean(reps)),
        "sd": float(np.std(reps, ddof=1)),
        "ci_lo": lo_q,
        "ci_hi": hi_q,
    }

def bootstrap_policy_deltas(
    *,
    y: np.ndarray,
    t: np.ndarray,
    ps_hat: np.ndarray,
    mu1_hat: np.ndarray,
    mu0_hat: np.ndarray,
    pi: np.ndarray,
    cfg: PolicyValueConfig,
    n_boot: int = 300,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    Paired bootstrap for deltas:
      delta_none = V(pi) - V(none)
      delta_all  = V(pi) - V(all)

    Uses the SAME bootstrap resample for all three values to get correct CI on deltas.
    Also returns p_better = P(delta_none < 0) (lower outcome is better).
    """
    y = np.asarray(y)
    t = np.asarray(t)
    ps_hat = np.asarray(ps_hat)
    mu1_hat = np.asarray(mu1_hat)
    mu0_hat = np.asarray(mu0_hat)
    pi = np.asarray(pi).astype(int)

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)

    deltas_none = np.empty(n_boot, dtype=float)
    deltas_all = np.empty(n_boot, dtype=float)
    vals_pi = np.empty(n_boot, dtype=float)

    pi_none = np.zeros_like(pi)
    pi_all = np.ones_like(pi)

    for b in range(n_boot):
        b_idx = rng.choice(idx, size=n, replace=True)

        v_none = dr_policy_value(
            y=y[b_idx], t=t[b_idx], pi=pi_none[b_idx],
            ps_hat=ps_hat[b_idx], mu1_hat=mu1_hat[b_idx], mu0_hat=mu0_hat[b_idx],
            cfg=cfg
        )
        v_all = dr_policy_value(
            y=y[b_idx], t=t[b_idx], pi=pi_all[b_idx],
            ps_hat=ps_hat[b_idx], mu1_hat=mu1_hat[b_idx], mu0_hat=mu0_hat[b_idx],
            cfg=cfg
        )
        v_pi = dr_policy_value(
            y=y[b_idx], t=t[b_idx], pi=pi[b_idx],
            ps_hat=ps_hat[b_idx], mu1_hat=mu1_hat[b_idx], mu0_hat=mu0_hat[b_idx],
            cfg=cfg
        )

        vals_pi[b] = v_pi
        deltas_none[b] = v_pi - v_none
        deltas_all[b] = v_pi - v_all

    def _summ(x: np.ndarray) -> dict:
        lo_q = float(np.quantile(x, alpha / 2))
        hi_q = float(np.quantile(x, 1 - alpha / 2))
        return {
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "ci_lo": lo_q,
            "ci_hi": hi_q,
        }

    out = {
        "type": "paired_dr_policy_bootstrap",
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "value_pi": _summ(vals_pi),
        "delta_vs_none": _summ(deltas_none),
        "delta_vs_all": _summ(deltas_all),
        "p_better_vs_none": float(np.mean(deltas_none < 0.0)),
    }
    return out



def threshold_curve(
    tau_hat: np.ndarray,
    ps_hat: np.ndarray,
    mu1_hat: np.ndarray,
    mu0_hat: np.ndarray,
    Y: np.ndarray,
    t: np.ndarray,
    thresholds: np.ndarray,
    cfg: PolicyValueConfig,
    n_boot: int = 50,
    seed: int = 42,
    direction: str = "gte",  # "gte" or "lte"
) -> pd.DataFrame:

    tau_hat = np.asarray(tau_hat, dtype=float)

    def make_pi(thr: float) -> np.ndarray:
        if direction == "gte":
            return (tau_hat >= thr).astype(int)
        if direction == "lte":
            return (tau_hat <= thr).astype(int)
        raise ValueError("direction must be 'gte' or 'lte'")
    # Baselines (point estimates)
    v_none = dr_policy_value(
        y=Y, t=t, pi=np.zeros_like(tau_hat, dtype=int),
        ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=cfg
    )
    v_all = dr_policy_value(
        y=Y, t=t, pi=np.ones_like(tau_hat, dtype=int),
        ps_hat=ps_hat, mu0_hat=mu0_hat, mu1_hat=mu1_hat, cfg=cfg
    )

    res = []
    for thr in thresholds:
        pi_policy = make_pi(float(thr))
        v_pi = dr_policy_value(
            y=Y, t=t, pi=pi_policy,
            ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=cfg
        )
        res.append({
            "threshold": round(float(thr), 4),
            "treat_rate": round(float(np.mean(pi_policy)), 4),
            "value_dr": round(float(v_pi), 4),
            "delta_vs_none": round(float(v_pi - v_none), 4),
            "delta_vs_all": round(float(v_pi - v_all), 4),
        })

    out = pd.DataFrame(res).sort_values("threshold").reset_index(drop=True)

    # best threshold (min risk)
    best_thr = float(out.loc[out["value_dr"].idxmin(), "threshold"])

    # refine around best
    fine_thresholds = np.arange(best_thr - 0.05, best_thr + 0.051, 0.01)
    for thr in fine_thresholds:
        pi_policy = make_pi(float(thr))
        v_pi = dr_policy_value(
            y=Y, t=t, pi=pi_policy,
            ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat, cfg=cfg
        )
        out = pd.concat([out, pd.DataFrame([{
            "threshold": round(float(thr), 4),
            "treat_rate": round(float(np.mean(pi_policy)), 4),
            "value_dr": round(float(v_pi), 4),
            "delta_vs_none": round(float(v_pi - v_none), 4),
            "delta_vs_all": round(float(v_pi - v_all), 4),
        }])], ignore_index=True)

    out = out.drop_duplicates(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)

    # bootstrap CI per ogni threshold
    ci_rows = []
    for thr in out["threshold"].values:
        pi_policy = make_pi(float(thr))
        ci = bootstrap_policy_value(
            y=Y, t=t, pi=pi_policy,
            ps_hat=ps_hat, mu1_hat=mu1_hat, mu0_hat=mu0_hat,
            n_boot=n_boot, seed=seed, cfg=cfg
        )
        ci_rows.append({
            "threshold": round(float(thr), 4),
            "value_boot_mean": round(float(ci["mean"]), 4),
            "value_boot_sd": round(float(ci["sd"]), 4),
            "value_boot_ci_lo": round(float(ci["ci_lo"]), 4),
            "value_boot_ci_hi": round(float(ci["ci_hi"]), 4),
        })

    ci_df = pd.DataFrame(ci_rows)
    out = out.merge(ci_df, on="threshold", how="left")
    return out


def _curve_loop_(Y: np.ndarray, cfg: PolicyValueConfig,
                 fine_thresholds: np.ndarray,
                 mu0_hat: np.ndarray, mu1_hat: np.ndarray,
                 ps_hat: np.ndarray, res: list[Any], t: np.ndarray[tuple[Any, ...]],
                 tau_hat: np.ndarray[tuple[Any, ...]], v_all: float, v_none: float):
    for threshold in fine_thresholds:
        pi_policy = (tau_hat >= threshold).astype(int)
        pvcfg = PolicyValueConfig(ps_clip=cfg.ps_clip)
        v_pi = dr_policy_value(y=Y, t=t, pi=pi_policy,
                               ps_hat=ps_hat,
                               mu1_hat=mu1_hat,
                               mu0_hat=mu0_hat,
                               cfg=pvcfg)
        res.append(
            {
                "threshold": round(float(threshold), 4),
                "treat_rate": round(float(np.mean(pi_policy)), 4),
                "value_dr": round(float(v_pi), 4),
                "delta_vs_none": round(float(v_pi - v_none), 4),
                "delta_vs_all": round(float(v_pi - v_all), 4),
            }
        )
