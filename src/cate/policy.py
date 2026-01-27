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

from dataclasses import dataclass
import numpy as np


def _clip_ps(ps: np.ndarray, lo: float, hi: float) -> np.ndarray:
    ps = np.asarray(ps, dtype=float)
    return np.clip(ps, lo, hi)


def policy_from_tau(
    tau_hat: np.ndarray,
    kind: str = "tau_gt_0",
    top_frac: float = 0.2,
) -> np.ndarray:
    """
    Build a deterministic treatment policy pi(x) in {0,1} from tau_hat.

    kind:
      - "tau_gt_0": treat if tau_hat > 0
      - "top_frac": treat only top fraction by tau_hat (e.g., top 20%)
    """
    tau_hat = np.asarray(tau_hat, dtype=float)

    if kind == "tau_gt_0":
        return (tau_hat > 0.0).astype(int)

    if kind == "top_frac":
        if not (0.0 < top_frac < 1.0):
            raise ValueError("top_frac must be in (0,1)")
        thr = np.quantile(tau_hat, 1.0 - top_frac)
        return (tau_hat >= thr).astype(int)

    raise ValueError("kind must be one of: 'tau_gt_0', 'top_frac'")


@dataclass(frozen=True)
class PolicyValueConfig:
    ps_clip: tuple[float, float] = (0.01, 0.99)


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
    Doubly-robust (AIPW-style) estimator of the expected outcome under a policy pi(x).

    For binary outcome Y (mortality) lower is better.
    Deterministic policy pi in {0,1}.

    V(pi) = E[ mu_pi(X) + I(T=pi(X))/P(T|X) * (Y - mu_T(X)) ]

    where:
      mu_pi = pi*mu1 + (1-pi)*mu0
      P(T|X) = e(X) if T=1 else 1-e(X)
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

    # DR score
    score = mu_pi + match * (y - mu_t) / (p_t + 1e-12)
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
