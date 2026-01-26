from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass(frozen=True)
class BaselineConfig:
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
    drop_cols: Optional[Sequence[str]] = None  # extra cols to drop if needed
