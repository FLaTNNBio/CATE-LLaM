# src/pipeline/06_build_analytic_v0.py
"""
Build analytic_v0 dataset:
- start from analytic_skeleton (outcome + treatment)
- left join labs_6h and vitals_6h on stay_id
- optionally left join hba1c_baseline if file exists

Outputs:
- data/analytic/analytic_v0.parquet
"""

import duckdb

from src.config import ANALYTIC_DIR, INTERMEDIATE_DIR

SKELETON_PATH = ANALYTIC_DIR / "analytic_skeleton.parquet"
LABS_PATH = INTERMEDIATE_DIR / "labs_6h_v2.parquet"      # from v2 labs script
VITALS_PATH = INTERMEDIATE_DIR / "vitals_6h.parquet"     # from final vitals script
HBA1C_PATH = INTERMEDIATE_DIR / "hba1c_baseline.parquet" # optional

OUT_PATH = ANALYTIC_DIR / "analytic_v0.parquet"
DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not SKELETON_PATH.exists():
        raise FileNotFoundError(f"Missing: {SKELETON_PATH}")
    if not LABS_PATH.exists():
        raise FileNotFoundError(f"Missing: {LABS_PATH} (run labs step)")
    if not VITALS_PATH.exists():
        raise FileNotFoundError(f"Missing: {VITALS_PATH} (run vitals step)")

    include_hba1c = HBA1C_PATH.exists()

    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='4GB';")

    con.execute(f"""
    CREATE OR REPLACE VIEW skel AS
    SELECT * FROM read_parquet('{SKELETON_PATH.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW labs AS
    SELECT * FROM read_parquet('{LABS_PATH.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW vitals AS
    SELECT * FROM read_parquet('{VITALS_PATH.as_posix()}');
    """)

    if include_hba1c:
        con.execute(f"""
        CREATE OR REPLACE VIEW hba1c AS
        SELECT * FROM read_parquet('{HBA1C_PATH.as_posix()}');
        """)

    # Build analytic v0
    if include_hba1c:
        con.execute("""
        CREATE OR REPLACE TABLE analytic_v0 AS
        SELECT
          s.*,
          -- labs (v2 columns)
          l.creatinine,
          l.lactate,
          l.wbc,
          l.hemoglobin,
          l.platelets,
          l.sodium,
          l.potassium,
          l.bicarbonate,
          -- vitals
          v.hr,
          v.rr,
          v.spo2,
          v.temp_c,
          v.nibp_sys,
          v.nibp_dia,
          v.nibp_mean,
          -- optional chronic
          h.hba1c,
          h.has_hba1c
        FROM skel s
        LEFT JOIN labs l
          ON s.stay_id = l.stay_id
        LEFT JOIN vitals v
          ON s.stay_id = v.stay_id
        LEFT JOIN hba1c h
          ON s.stay_id = h.stay_id;
        """)
    else:
        con.execute("""
        CREATE OR REPLACE TABLE analytic_v0 AS
        SELECT
          s.*,
          l.creatinine,
          l.lactate,
          l.wbc,
          l.hemoglobin,
          l.platelets,
          l.sodium,
          l.potassium,
          l.bicarbonate,
          v.hr,
          v.rr,
          v.spo2,
          v.temp_c,
          v.nibp_sys,
          v.nibp_dia,
          v.nibp_mean
        FROM skel s
        LEFT JOIN labs l
          ON s.stay_id = l.stay_id
        LEFT JOIN vitals v
          ON s.stay_id = v.stay_id;
        """)

    # Sanity checks
    n_skel = con.execute("SELECT COUNT(*) FROM skel;").fetchone()[0]
    n_v0 = con.execute("SELECT COUNT(*) FROM analytic_v0;").fetchone()[0]

    print("Rows:")
    print({"skeleton": n_skel, "analytic_v0": n_v0})

    # Check duplicates on stay_id
    dup = con.execute("""
    SELECT COUNT(*) AS n_dup
    FROM (
      SELECT stay_id, COUNT(*) c
      FROM analytic_v0
      GROUP BY stay_id
      HAVING COUNT(*) > 1
    );
    """).fetchone()[0]
    print("\nDuplicate stay_id rows (should be 0):", dup)

    # Quick missingness summary (core)
    print("\nCore missingness:")
    print(con.execute("""
    SELECT
      1 - COUNT(creatinine)::DOUBLE/COUNT(*) AS creatinine_null_frac,
      1 - COUNT(lactate)::DOUBLE/COUNT(*) AS lactate_null_frac,
      1 - COUNT(hr)::DOUBLE/COUNT(*) AS hr_null_frac,
      1 - COUNT(nibp_mean)::DOUBLE/COUNT(*) AS nibp_mean_null_frac
    FROM analytic_v0;
    """).fetchdf())

    # Export
    con.execute(f"""
    COPY analytic_v0
    TO '{OUT_PATH.as_posix()}'
    (FORMAT PARQUET);
    """)
    print(f"\nAnalytic v0 written to: {OUT_PATH}")
    if include_hba1c:
        print("HbA1c joined: YES")
    else:
        print("HbA1c joined: NO (file not found)")


if __name__ == "__main__":
    main()
