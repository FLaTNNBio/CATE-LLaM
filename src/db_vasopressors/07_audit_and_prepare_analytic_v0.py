"""
Audit + prepare analytic_v0:
- add missingness indicators has_*
- apply conservative clipping (winsorization by hard bounds) for numeric columns
- output analytic_v0_prepared.parquet

We clip ONLY when units are stable/known (vitals) and use very conservative bounds for labs.
"""
import duckdb

from src.config import ANALYTIC_DIR, INTERMEDIATE_DIR

IN_PATH = ANALYTIC_DIR / "analytic_v0.parquet"
OUT_PATH = ANALYTIC_DIR / "analytic_v0_prepared.parquet"

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Conservative bounds (units-aware)
CLIP_BOUNDS = {
    # Vitals (units stable given our itemid choices)
    "hr": (20.0, 250.0),          # bpm
    "rr": (3.0, 80.0),            # breaths/min
    "spo2": (50.0, 100.0),        # %
    "temp_c": (25.0, 45.0),       # Celsius (we used Temperature Celsius itemid)
    "nibp_sys": (50.0, 250.0),    # mmHg
    "nibp_dia": (20.0, 150.0),    # mmHg
    "nibp_mean": (30.0, 200.0),   # mmHg

    # Labs (very conservative; still catches obvious data entry issues)
    "creatinine": (0.0, 25.0),    # mg/dL typical; 25 is very high already
    "lactate": (0.0, 30.0),       # mmol/L; 30 extremely high
    "wbc": (0.0, 300.0),          # K/uL; 300 absurdly high but conservative
    "hemoglobin": (0.0, 25.0),    # g/dL; conservative
    "platelets": (0.0, 2000.0),   # K/uL; conservative
    "sodium": (90.0, 200.0),      # mmol/L; conservative
    "potassium": (1.0, 10.0),     # mmol/L; conservative
    "bicarbonate": (0.0, 60.0),   # mmol/L; conservative
    # Optional
    "hba1c": (3.0, 20.0),         # %
}

# Variables for which we add missingness indicators
HAS_FLAGS = [
    "creatinine", "lactate",
    "wbc", "hemoglobin", "platelets",
    "sodium", "potassium", "bicarbonate",
    "hr", "rr", "spo2", "temp_c",
    "nibp_sys", "nibp_dia", "nibp_mean",
    "hba1c",
]


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input parquet: {IN_PATH}")

    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='4GB';")

    con.execute(f"""
    CREATE OR REPLACE VIEW v0 AS
    SELECT * FROM read_parquet('{IN_PATH.as_posix()}');
    """)

    # Helper: build clipping expressions
    clip_exprs = []
    for col, (lo, hi) in CLIP_BOUNDS.items():
        # Only clip if the column exists; we'll check later via TRY to be safe.
        clip_exprs.append(
            f"""
            CASE
              WHEN {col} IS NULL THEN NULL
              WHEN {col} < {lo} THEN {lo}
              WHEN {col} > {hi} THEN {hi}
              ELSE {col}
            END AS {col}
            """.strip()
        )

    # Build has_* indicators (only if column exists; if not, they will error - so we keep to known columns)
    has_exprs = [f"CASE WHEN {c} IS NULL THEN 0 ELSE 1 END AS has_{c}" for c in HAS_FLAGS]

    # Report: how many values are out of bounds before clipping
    print("Out-of-bounds counts BEFORE clipping (non-NULL):")
    for col, (lo, hi) in CLIP_BOUNDS.items():
        q = f"""
        SELECT
          '{col}' AS col,
          SUM(CASE WHEN {col} < {lo} THEN 1 ELSE 0 END) AS n_below,
          SUM(CASE WHEN {col} > {hi} THEN 1 ELSE 0 END) AS n_above,
          COUNT({col}) AS n_nonnull
        FROM v0;
        """
        print(con.execute(q).fetchdf())

    # Create prepared table:
    # - keep all original columns, but override clipped columns with their clipped versions
    # DuckDB SELECT * EXCLUDE (...) + add back is convenient.
    # We'll explicitly exclude the columns we clip, then add them back clipped.
    clip_cols = list(CLIP_BOUNDS.keys())
    exclude_clause = ", ".join(clip_cols)

    con.execute(f"""
    CREATE OR REPLACE TABLE v0_prepared AS
    SELECT
      * EXCLUDE ({exclude_clause}),
      {", ".join(clip_exprs)},
      {", ".join(has_exprs)}
    FROM v0;
    """)

    # Report: out-of-bounds after clipping (should be 0)
    print("\nOut-of-bounds counts AFTER clipping (non-NULL):")
    for col, (lo, hi) in CLIP_BOUNDS.items():
        q = f"""
        SELECT
          '{col}' AS col,
          SUM(CASE WHEN {col} < {lo} THEN 1 ELSE 0 END) AS n_below,
          SUM(CASE WHEN {col} > {hi} THEN 1 ELSE 0 END) AS n_above,
          COUNT({col}) AS n_nonnull
        FROM v0_prepared;
        """
        print(con.execute(q).fetchdf())

    # Quick missingness summary
    print("\nMissingness summary (selected columns):")
    miss_q = """
    SELECT
      COUNT(*) AS n,
      1 - COUNT(creatinine)::DOUBLE/COUNT(*) AS creatinine_null_frac,
      1 - COUNT(lactate)::DOUBLE/COUNT(*) AS lactate_null_frac,
      1 - COUNT(hr)::DOUBLE/COUNT(*) AS hr_null_frac,
      1 - COUNT(nibp_mean)::DOUBLE/COUNT(*) AS nibp_mean_null_frac
    FROM v0_prepared;
    """
    print(con.execute(miss_q).fetchdf())

    # Export
    con.execute(f"""
    COPY v0_prepared
    TO '{OUT_PATH.as_posix()}'
    (FORMAT PARQUET);
    """)

    print(f"\nPrepared dataset written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
