from __future__ import annotations

import numpy as np


def sample_treatment(ps: np.ndarray, seed: int) -> np.ndarray:
    """
    Sample the new pseudo-observational treatment from the artificial PS.

    Parameters
    ----------
    ps : np.ndarray
        Clipped propensity scores.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Sampled binary treatment vector.
    """
    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=ps, size=len(ps))