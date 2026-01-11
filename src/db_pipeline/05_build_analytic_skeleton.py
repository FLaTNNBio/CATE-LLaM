"""
Build the analytic skeleton dataset:
- One row per ICU stay
- Outcome + treatment only (no covariates yet)

This is used to validate alignment before adding features.
"""

import duckdb
from config import INTERMEDIATE_DIR, ANALYTIC_DIR

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
TREATMENT_PATH = INTERMEDIATE_DIR / "treatment_vaso6h.parquet"
OUT_PATH = ANALYTIC_DIR / "analytic_skeleton.parquet"

con = duckdb.connect(database=":memory:")

# ------------------------------------------------------------------
# Register Parquet files as views
# ------------------------------------------------------------------

con.execute(f"""
CREATE VIEW cohort AS
SELECT * FROM read_parquet('{COHORT_PATH.as_posix()}');
""")

con.execute(f"""
CREATE VIEW treatment AS
SELECT * FROM read_parquet('{TREATMENT_PATH.as_posix()}');
""")

# ------------------------------------------------------------------
# Build analytic skeleton
# Left join ensures every ICU stay has a treatment assignment
# ------------------------------------------------------------------

con.execute("""
CREATE TABLE analytic_skeleton AS
SELECT
  c.subject_id,
  c.hadm_id,
  c.stay_id,
  c.intime,
  c.y_hosp_mort,
  t.t_vaso6h
FROM cohort c
LEFT JOIN treatment t
  ON c.stay_id = t.stay_id;
""")

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

con.execute(f"""
COPY analytic_skeleton
TO '{OUT_PATH.as_posix()}'
(FORMAT PARQUET);
""")

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------

print("Rows in analytic_skeleton:")
print(con.execute("SELECT COUNT(*) FROM analytic_skeleton").fetchdf())

print("\nMissing treatment assignments:")
print(con.execute("""
SELECT COUNT(*) AS n_missing
FROM analytic_skeleton
WHERE t_vaso6h IS NULL
""").fetchdf())

print("\nTreatment prevalence:")
print(con.execute("""
SELECT AVG(t_vaso6h) AS p_treated
FROM analytic_skeleton
""").fetchdf())

print("\nOutcome by treatment (naive, confounded):")
print(con.execute("""
SELECT
  t_vaso6h,
  COUNT(*) AS n,
  AVG(y_hosp_mort) AS mort_rate
FROM analytic_skeleton
GROUP BY t_vaso6h
ORDER BY t_vaso6h
""").fetchdf())

print(f"\nAnalytic skeleton written to: {OUT_PATH}")