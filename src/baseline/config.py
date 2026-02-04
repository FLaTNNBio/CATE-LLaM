from dataclasses import dataclass, field
from typing import Optional, Sequence

@dataclass(frozen=True)
class BaselineConfig:
    # data
    data_path: str = "data/analytic/analytic_v0.parquet"

    # output
    out_dir: str = "artifacts/cate/baseline"

    # columns
    id_col: str = "stay_id"
    subject_col: str = "subject_id"
    treatment_col: str = "t_vaso6h"
    outcome_col: str = "y_hosp_mort"

    # split
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42

    # weighting / overlap
    ps_clip: tuple[float, float] = (0.01, 0.99)        # clip e(x) before weights
    weight_trim_quantiles: tuple[float, float] = (0.01, 0.99)  # trim weights

    # cross-fitting
    n_folds: int = 5

    # feature control
    drop_cols: Optional[Sequence[str]] = field(default_factory=list)

    # tau direction
    tau_direction: str = "lte"  # "gte" or "lte"

    # Policy Tau
    policy: str = "tau_gt_0" # ["tau_gt_0", "tau_lt_0", "top_frac_benefit"]


VASO_V0 = BaselineConfig(
    data_path="data/analytic/analytic_v0.parquet",
    treatment_col="t_vaso6h",
    outcome_col="y_hosp_mort",
    drop_cols=["has_hba1c_1"],
    out_dir="artifacts/cate/vaso_v0",
    id_col="stay_id",
    subject_col="subject_id"
)

RBC_V1_FIXED = BaselineConfig(
    data_path="data/analytic/analytic_rbc_v1_f.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="t_rbc_3h",
    outcome_col="y_hosp_mort",
    # Empty for now
    drop_cols=["elig_hb_threshold","max_elig_hb_threshold",
               "min_elig_hb_threshold", "treat_window_hours", "y_24h_mort", "y_hosp_mort",
               "elig_within_hours", "y_hosp_mort", "t_rbc_3h", "hadm_id", "intime"], # "t0_hb", "hemoglobin", "has_lactate", "bicarbonate", "potassium"],
    # Same weighting settings initially used in RBC analyses
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/rbc_v1_fixed"
)


RBC_V2 = BaselineConfig(
    data_path="data/analytic/analytic_sepsis_steroids_mit_v2.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="t_rbc_3h",
    outcome_col="y_hosp_mort",
    # Empty for now
    drop_cols= [ "treat_window_hours", "y_24h_mort", "y_hosp_mort", "Fi02_bg", "lactate", "BMI",
                 "stay_id", "hadm_id", "intime", "rbc_units_proxy", "po2", "has_lactate", "chloride",
                 "elig_within_hours", "y_hosp_mort", "t_rbc_3h", "hb_threshold", "elig_hb_threshold",
                 "max_elig_hb_threshold", "min_elig_hb_threshold"], # "t0_hb", "hemoglobin", "has_lactate",
    # "bicarbonate", "potassium"],
    # Same weighting settings initially used in RBC analyses
    ps_clip=(0.05, 0.95),
    weight_trim_quantiles=(0.05, 0.95),
    out_dir="artifacts/cate/rbc_v2"
)


RESP_V1 = BaselineConfig(
    data_path="data/analytic/analytic_resp_v1_clean.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="t_hfnc",
    outcome_col="y_intub_48h",
    drop_cols=["intime", "t0_time","t0_support", "n_hfnc_2h", "n_niv_2h", "y_hosp_mort",
               "stay_id", "subject_id", "hadm_id", "y_intub_48h", "t_hfnc", "o2_flow"],
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.05, 0.95),
    out_dir="artifacts/cate/resp_v1",
    tau_direction="lte",
    policy="tau_lt_0"
)

CRRT_V1 = BaselineConfig(
    data_path="data/analytic/analytic_crrt_v1.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="treat_crrt",
    outcome_col="y_hosp_mort",
    drop_cols=["intime", "t0_time","t0_support", "n_hfnc_2h", "n_niv_2h", "y_hosp_mort",
               "stay_id", "subject_id", "hadm_id", "y_intub_48h", "treat_crrt" , "o2_flow",
               "flow_rate_l_min","crrt_mode", "crrt_kcl", "glucose"],
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/crrt_v1",
    tau_direction="lte",
    policy="tau_lt_0",
    n_folds=2
)

SEPSIS_V1 = BaselineConfig(
    data_path="data/analytic/analytic_sepsis_steroids_clean_v1.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="treat_steroid",
    outcome_col="y_28d_mort_inhosp",
    drop_cols=["intime", "t0_time","t0_support", "y_hosp_mort", # "y_28d_mort_inhosp"
               "enroll_end", "treat_steroid", "steroid_unparsed_events",
               "steroid_total_events", "hc_equiv_mg_0_24h", "lactate",
               "stay_id", "subject_id", "hadm_id"], # "glucose"
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/sepsis_v1",
    tau_direction="lte",
    policy="tau_lt_0",
    n_folds=5
)

SEPSIS_V2 = BaselineConfig(
    data_path="data/analytic/analytic_sepsis_steroids_mit_v2.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="treat_steroid",
    outcome_col="y_28d_mort_inhosp",
    drop_cols=["intime", "t0_time","t0_support", "y_hosp_mort", "SO2_bg",# "y_28d_mort_inhosp"
               "enroll_end", "steroid_unparsed_events", "treat_steroid",
               "steroid_total_events", "hc_equiv_mg_0_24h", "lactate",
               "baseline_lookback_hours", "elig_within_hours", "treat_window_hours",
               "stay_id", "subject_id", "hadm_id"], # "glucose"
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/sepsis_v2",
    tau_direction="lte",
    policy="tau_lt_0",
    n_folds=5
)