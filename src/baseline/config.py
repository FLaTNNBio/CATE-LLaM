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
    data_path="data/analytic/analytic_rbc_v1_fixed.parquet",
    id_col="stay_id",
    subject_col="subject_id",
    treatment_col="t_rbc_3h",
    outcome_col="y_hosp_mort",
    # Empty for now
    drop_cols=["elig_hb_threshold","max_elig_hb_threshold",
               "min_elig_hb_threshold", "treat_window_hours", "elig_within_hours"], # "t0_hb", "hemoglobin", "has_lactate", "bicarbonate", "potassium"],
    # Same weighting settings initially used in RBC analyses
    ps_clip=(0.01, 0.99),
    weight_trim_quantiles=(0.01, 0.99),
    out_dir="artifacts/cate/rbc_v1_fixed"
)
