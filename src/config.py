# src/config.py
from pathlib import Path

# Project root = parent of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Common directories
MIMIC_DIR = PROJECT_ROOT / "mimic"
HOSP_DIR = MIMIC_DIR / "hosp"
ICU_DIR = MIMIC_DIR / "icu"

DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
ANALYTIC_DIR = DATA_DIR / "analytic"

COHORT_PARQUET = INTERMEDIATE_DIR / "cohort.parquet"
COHORT_CLEAN_PARQUET = INTERMEDIATE_DIR / "cohort_clean.parquet"

# Ensure output dirs exist
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
ANALYTIC_DIR.mkdir(parents=True, exist_ok=True)

# Check for correct import and running folder
if not (PROJECT_ROOT / "mimic").exists():
    raise RuntimeError(
        f"PROJECT_ROOT seems wrong: {PROJECT_ROOT}"
    )


def register_parquet_view(con, view_name: str = "cohort", parquet_path: Path = COHORT_CLEAN_PARQUET) -> None:
    """
    Register a parquet view
    :param con: connection to duckdb database
    :param view_name: name of the parquet view, defaults to "cohort"
    :param parquet_path: path to the parquet file, defaults to COHORT_CLEAN_PARQUET
    """
    parquet_path = parquet_path.resolve()
    con.execute(f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT * FROM read_parquet('{parquet_path.as_posix()}');
    """)

