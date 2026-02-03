import pandas as pd
from src.config import ANALYTIC_DIR

# -------------------------
# Config: input/output
# -------------------------

IN_PATH = ANALYTIC_DIR / "analytic_resp_v1.parquet"
OUT_PATH = ANALYTIC_DIR / "analytic_resp_v1_clean.parquet"

# -------------------------
# Columns to drop (agreed)
# -------------------------
DROP_COLS = [
    # outcome timing / hospital timing (not needed for modeling)
    "intub_time",
    "trach_time",
    "deathtime",
    "dischtime",

    # extremely missing / not useful now
    "flow_rate_l_min",
    "albumin",
    "glucose",

    # (optional) almost-always-zero outcome; keep only if you plan to analyze it
    "y_trach_48h",
]

# Keep IDs for modeling parquet
ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Core outcome/treatment fields
KEEP_ALWAYS = [
    "intime",
    "t0_time",
    "t0_support",
    "n_hfnc_2h",
    "n_niv_2h",
    "t_hfnc",
    "y_intub_48h",
    "y_hosp_mort",     # optional secondary outcome; keep
]

# Covariates we consider "useful enough" to keep in model parquet
COVARIATES = [
    # demog
    "age", "gender", "race",

    # vitals / severity
    "hr", "rr", "spo2", "temp_c",
    "nibp_sys", "nibp_dia", "nibp_mean",
    "o2_flow",

    # neuro
    "gcs_eye", "gcs_verbal", "gcs_motor",

    # gas exchange
    "pCO2", "ph",

    # labs (keep; missingness handled via indicators)
    "hemoglobin", "platelets", "wbc", "creatinine",
    "bicarbonate", "sodium", "potassium", "lactate",
    "inr", "ptt",

    # anthropometrics (keep but you may choose not to use in PS base)
    "admission_weight_kg", "height_cm", "bmi",
]

# Existing has_* columns you want to DROP because they are pointless / noisy
DROP_HAS = [
    "has_hr",   # near 0 missingness -> not informative
]

# has_* indicators that are useful to have for MNAR-ish variables (add if missing)
ADD_HAS_FOR = [
    # gas exchange
    #"pCO2", "ph",

    # GCS (even if low missingness, harmless; you can drop later if you want)
    #"gcs_eye", "gcs_verbal", "gcs_motor",

    # vitals often ok, but o2_flow missing 18% -> useful indicator
    #"o2_flow",

    # labs with moderate/high missingness
    #"hemoglobin", "platelets", "wbc", "creatinine",
    #"lactate", "inr", "ptt",
]


def _safe_drop(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols_present = [c for c in cols if c in df.columns]
    return df.drop(columns=cols_present)


def _make_binary_support(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures:
      - t0_support_bin: HFNC=1, NIV=0, else NA
      - t_hfnc present and integer-like (0/1)
    """
    if "t0_support" in df.columns:
        s = df["t0_support"].astype(str).str.upper().str.strip()
        df["t_hfnc"] = s.map({"HFNC": 1, "NIV": 0}).astype("Int64")
    else:
        df["t_hfnc"] = pd.NA
    return df


def _add_has_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            has_col = f"has_{c.lower()}"
            if has_col not in df.columns:
                df[has_col] = (~df[c].isna()).astype("Int64")
    return df

def remove_hypercapnic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional function to remove hypercapnic patients from the dataset.
    Hypercapnic patients are defined as those with pCO2 > 55 or pH < 7.30.
    """
    if "pCO2" in df.columns and "ph" in df.columns:
        mask_hypercapnic = (df["pCO2"] > 55) | (df["ph"] < 7.30)
        df = df[~mask_hypercapnic].copy()
    return df

def union_mort_intub(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional function to create a combined outcome variable that indicates
    whether a patient was either intubated within 48 hours or died during hospitalization.
    """
    if "y_intub_48h" in df.columns and "y_hosp_mort" in df.columns:
        df["y_intub_48h"] = ((df["y_intub_48h"] == 1) | (df["y_hosp_mort"] == 1)).astype("Int64")
    return df


def main() -> None:
    df = pd.read_parquet(IN_PATH)

    # Optional: remove hypercapnic patients
    df = remove_hypercapnic(df)

    # Optional: union of mortality and intubation outcomes
    df = union_mort_intub(df)

    # 1) Drop agreed columns
    df = _safe_drop(df, DROP_COLS)

    # 2) Drop useless has_* columns
    df = _safe_drop(df, DROP_HAS)

    # 3) Add/standardize treatment encoding
    df = _make_binary_support(df)

    # 4) Add missing has_* indicators (only if useful)
    df = _add_has_indicators(df, ADD_HAS_FOR)

    # 5) Keep only a clean subset (IDs + core + covariates + has_*)
    keep = []
    for c in ID_COLS + KEEP_ALWAYS + COVARIATES + ["t0_support_bin"]:
        if c in df.columns:
            keep.append(c)

    # Keep all existing has_* indicators except those we dropped
    has_cols = sorted([c for c in df.columns if c.startswith("has_")])
    keep += [c for c in has_cols if c not in keep]

    # Deduplicate while preserving order
    seen = set()
    keep_unique = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            keep_unique.append(c)

    df = df[keep_unique].copy()

    # 6) Basic type hygiene
    # outcomes as int (nullable)
    for y in ["y_intub_48h", "y_hosp_mort"]:
        if y in df.columns:
            df[y] = pd.to_numeric(df[y], errors="coerce").astype("Int64")

    # age numeric
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # 7) Write out
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    # Quick summary
    print(f"Wrote: {OUT_PATH}")
    print("Shape:", df.shape)
    print("Treatment counts (t_hfnc):")
    if "t_hfnc" in df.columns:
        print(df["t_hfnc"].value_counts(dropna=False))
    print("Outcome rate y_intub_48h:")
    if "y_intub_48h" in df.columns:
        print(df["y_intub_48h"].mean())


if __name__ == "__main__":
    main()
