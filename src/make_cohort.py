"""
Build the base ICU cohort for causal analysis on MIMIC-IV.

Unit of analysis:
- One row per ICU stay
- Only the FIRST ICU stay per patient
- Adult patients only (age >= 18)

Outputs:
- cohort.parquet with baseline identifiers and hospital mortality outcome
"""
import os
import duckdb


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# src/make_cohort.py
import duckdb
from src.config import (
    PROJECT_ROOT,
    HOSP_DIR,
    ICU_DIR,
    INTERMEDIATE_DIR
)

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
COHORT_PARQUET = INTERMEDIATE_DIR / "cohort.parquet"

OUT_DIR = PROJECT_ROOT / "data" / "intermediate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Connect to DuckDB
# ---------------------------------------------------------------------

con = duckdb.connect(str(DB_PATH))

# ---------------------------------------------------------------------
# Create views over raw CSV files
# DuckDB reads CSVs directly without loading everything into memory
# ---------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE VIEW icustays AS
SELECT *
FROM read_csv_auto('{ICU_DIR / "icustays.csv.gz"}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW admissions AS
SELECT *
FROM read_csv_auto('{HOSP_DIR / "admissions.csv.gz"}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW patients AS
SELECT *
FROM read_csv_auto('{HOSP_DIR / "patients.csv.gz"}', union_by_name=true);
""")

# ---------------------------------------------------------------------
# Build ICU cohort
# - Join ICU stays with hospital admissions and patient demographics
# - Keep only the first ICU stay per subject
# - Restrict to adult patients
#
# Output: First ICU stay per adult (>=18yo) patient
# ---------------------------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE cohort AS
WITH base AS (
    SELECT
        i.subject_id        AS subject_id,
        i.hadm_id           AS hadm_id,
        i.stay_id           AS stay_id,
        i.intime            AS intime,
        i.outtime           AS outtime,
        a.admittime         AS admittime,
        a.dischtime         AS dischtime,
        a.deathtime         AS deathtime,
        p.gender            AS gender,
        p.anchor_age        AS anchor_age
    FROM icustays i
    JOIN admissions a
        ON i.hadm_id = a.hadm_id
    JOIN patients p
        ON i.subject_id = p.subject_id
    WHERE i.hadm_id IS NOT NULL
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY subject_id
            ORDER BY intime
        ) AS rn
    FROM base
)
SELECT
    subject_id,
    hadm_id,
    stay_id,
    intime,
    outtime,
    admittime,
    dischtime,
    deathtime,
    gender,
    anchor_age
FROM ranked
WHERE rn = 1
  AND anchor_age >= 18;
""")

# ---------------------------------------------------------------------
# OUTCOME DEFINING
# Define outcome: hospital mortality
# Death occurring before hospital discharge
# Outcome of code: Added Column 'y_hosp_mort' : 1 death; 0 not dead.
# ---------------------------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE cohort_outcome AS
SELECT
    *,
    CASE
        WHEN deathtime IS NOT NULL
         AND deathtime <= dischtime
        THEN 1
        ELSE 0
    END AS y_hosp_mort
FROM cohort;
""")

# ---------------------------------------------------------------------
# Export cohort to Parquet
# ---------------------------------------------------------------------

out_path = OUT_DIR / "cohort.parquet"

con.execute(f"""
COPY cohort_outcome
TO '{out_path}'
(FORMAT PARQUET);
""")

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

print("Cohort size:")
print(con.execute("SELECT COUNT(*) AS n FROM cohort_outcome").fetchdf())

print("\nHospital mortality rate:")
print(con.execute(
    "SELECT AVG(y_hosp_mort) AS hosp_mort_rate FROM cohort_outcome"
).fetchdf())

print("\nCheck ICU time alignment (should be near zero):")
print(con.execute("""
SELECT
    SUM(
        CASE
            WHEN intime < admittime
              OR intime > dischtime
            THEN 1
            ELSE 0
        END
    ) AS bad_time_alignment
FROM cohort_outcome;
""").fetchdf())

print(f"\nParquet file written to: {out_path}")


# Create a "clean" cohort for causal analysis:
# - Drop paradoxical cases where ICU intime is AFTER hospital dischtime: 85/65,366 -> ~0.13%
# - Keep cases where ICU intime is slightly before admittime (~405/65,366 - likely logging mismatch)
# OUTPUT: cohort_clean.parquet

con.execute("""
CREATE OR REPLACE TABLE cohort_outcome_clean AS
SELECT *
FROM cohort_outcome
WHERE NOT (intime > dischtime);
""")

# Export clean cohort
clean_path = OUT_DIR / "cohort_clean.parquet"
con.execute(f"""
COPY cohort_outcome_clean
TO '{clean_path}'
(FORMAT PARQUET);
""")
