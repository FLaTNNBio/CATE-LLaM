from __future__ import annotations


DEFAULT_COVARIATES = [
    "age",
    "wtkg",
    "karnof",
    "oprior",
    "preanti",
    "strat",
    "cd40",
    "symptom",
]

DEFAULT_OUTCOME = "label"
DEFAULT_ORIGINAL_TREATMENT = "treat"
DEFAULT_NEW_TREATMENT = "treat_obs"
DEFAULT_PS_COL = "ps_artificial"

DEFAULT_CLIP_MIN = 0.05
DEFAULT_CLIP_MAX = 0.95
DEFAULT_INTERCEPT = 0.0
DEFAULT_SEED = 42