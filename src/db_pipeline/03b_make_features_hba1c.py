"""
Extract baseline HbA1c as a chronic covariate.
We take the LAST HbA1c measurement BEFORE ICU admission
within a lookback window of 365 days.
"""

import duckdb
from config import HOSP_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
D_LABITEMS = HOSP_DIR / "d_labitems.csv.gz"
LABEVENTS = HOSP_DIR / "labevents.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "hba1c_baseline.parquet"

con = duckdb.connect(str(DB_PATH))
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
con.execute("SET memory_limit='8GB';")

# Cohort
con.execute(f"""
CREATE OR REPLACE VIEW cohort_pq AS
SELECT stay_id, hadm_id, intime
FROM read_parquet('{COHORT_PATH.as_posix()}');
""")

# Lab tables
con.execute(f"""
CREATE OR REPLACE VIEW d_labitems_v AS
SELECT * FROM read_csv_auto('{D_LABITEMS.as_posix()}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW labevents_v AS
SELECT * FROM read_csv_auto('{LABEVENTS.as_posix()}', union_by_name=true);
""")

# HbA1c itemid (specific and safe)
con.execute("""
CREATE OR REPLACE TABLE hba1c_item AS
SELECT itemid, label
FROM d_labitems_v
WHERE lower(label) LIKE '%hemoglobin a1c%';
""")

print("HbA1c itemids:")
print(con.execute("SELECT * FROM hba1c_item").fetchdf())

# Extract last HbA1c before ICU (within 1 year)
con.execute("""
CREATE OR REPLACE TABLE hba1c_long AS
WITH filtered AS (
  SELECT
    c.stay_id,
    le.charttime,
    le.valuenum,
    c.intime
  FROM cohort_pq c
  JOIN labevents_v le
    ON le.hadm_id = c.hadm_id
  JOIN hba1c_item h
    ON le.itemid = h.itemid
  WHERE le.charttime IS NOT NULL
    AND le.valuenum IS NOT NULL
    AND le.charttime <= c.intime
    AND le.charttime >= c.intime - INTERVAL '365' DAY
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY stay_id
      ORDER BY charttime DESC
    ) AS rn
  FROM filtered
)
SELECT
  stay_id,
  valuenum AS hba1c
FROM ranked
WHERE rn = 1;
""")

# LEFT JOIN with cohort to keep cardinality
con.execute("""
CREATE OR REPLACE TABLE hba1c_baseline AS
SELECT
  c.stay_id,
  h.hba1c,
  CASE WHEN h.hba1c IS NOT NULL THEN 1 ELSE 0 END AS has_hba1c
FROM cohort_pq c
LEFT JOIN hba1c_long h
  ON c.stay_id = h.stay_id;
""")

# Export
con.execute(f"""
COPY hba1c_baseline
TO '{OUT_PATH.as_posix()}'
(FORMAT PARQUET);
""")

print("\nHbA1c coverage:")
print(con.execute("""
SELECT
  AVG(has_hba1c) AS frac_with_hba1c
FROM hba1c_baseline;
""").fetchdf())

print(f"\nParquet written to: {OUT_PATH}")
