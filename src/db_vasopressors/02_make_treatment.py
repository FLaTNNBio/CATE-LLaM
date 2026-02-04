"""
Create early vasopressor treatment indicator for MIMIC-IV ICU cohort.

Treatment definition:
- Unit of analysis: ICU stay (stay_id)
- Time zero: icustays.intime (already stored in cohort.parquet)
- Treatment window: [intime, intime + 6 hours]
- T = 1 if ANY vasopressor infusion/administration starts within the window
- Else T = 0

Outputs:
- treatment_vaso6h.parquet with:
  stay_id, t_vaso6h, first_vaso_time, hours_to_first_vaso, vaso_events_in_window
"""

import duckdb
from src.config import (ICU_DIR, COHORT_CLEAN_PARQUET, INTERMEDIATE_DIR, register_parquet_view)


D_ITEMS_CSV = ICU_DIR / "d_items.csv.gz"
INPUTEVENTS_CSV = ICU_DIR / "inputevents.csv.gz"
OUT_DIR = INTERMEDIATE_DIR

# Connect to DuckDB
con = duckdb.connect(database=":memory:")
# Register cohort as a view
register_parquet_view(con, "cohort", COHORT_CLEAN_PARQUET)

# ---------------------------------------------------------------------
# Register ICU tables needed for treatment extraction
# ---------------------------------------------------------------------

con.execute(f"""
CREATE OR REPLACE VIEW d_items AS
SELECT * FROM read_csv_auto('{D_ITEMS_CSV}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW inputevents AS
SELECT * FROM read_csv_auto('{INPUTEVENTS_CSV}', union_by_name=true);
""")

# ---------------------------------------------------------------------
# 1) Build a table of vasopressor itemids from d_items labels.
#    We do a conservative pattern match on common vasopressor names.
#    You can extend this list later if needed.
# ---------------------------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE vaso_itemids AS
SELECT DISTINCT itemid, label
FROM d_items
WHERE
  lower(label) LIKE '%norepinephrine%'
  OR lower(label) LIKE '%epinephrine%'
  OR lower(label) LIKE '%dopamine%'
  OR lower(label) LIKE '%vasopressin%'
  OR lower(label) LIKE '%phenylephrine%';
""")

# Quick check: how many itemids did we catch?
print("Vasopressor itemids found:")
print(con.execute("SELECT COUNT(*) AS n_itemids FROM vaso_itemids").fetchdf())
print(con.execute("SELECT * FROM vaso_itemids ORDER BY label LIMIT 30").fetchdf())

# ---------------------------------------------------------------------
# 2) Compute early vasopressor treatment (within 6h from ICU intime).
#    Strategy:
#    - Join inputevents with cohort to get intime per stay
#    - Filter to vasopressor itemids
#    - Keep events with starttime in [intime, intime + 6 hours]
#    - Aggregate per stay_id to create binary T and extra diagnostics
# ---------------------------------------------------------------------

con.execute("""
CREATE OR REPLACE TABLE treatment_vaso6h AS
WITH vaso_events AS (
  SELECT
    ie.stay_id,
    ie.itemid,
    ie.starttime,
    c.intime
  FROM inputevents ie
  JOIN cohort c
    ON ie.stay_id = c.stay_id
  JOIN vaso_itemids v
    ON ie.itemid = v.itemid
  WHERE ie.starttime IS NOT NULL
),
vaso_in_window AS (
  SELECT
    stay_id,
    MIN(starttime) AS first_vaso_time,
    COUNT(*)       AS vaso_events_in_window
  FROM vaso_events
  WHERE starttime >= intime
    AND starttime <  intime + INTERVAL '6' HOUR
  GROUP BY stay_id
)
SELECT
  c.stay_id,
  CASE WHEN w.stay_id IS NOT NULL THEN 1 ELSE 0 END AS t_vaso6h,
  w.first_vaso_time,
  CASE
    WHEN w.first_vaso_time IS NOT NULL
    THEN EXTRACT(EPOCH FROM (w.first_vaso_time - c.intime)) / 3600.0
    ELSE NULL
  END AS hours_to_first_vaso,
  COALESCE(w.vaso_events_in_window, 0) AS vaso_events_in_window
FROM cohort c
LEFT JOIN vaso_in_window w
  ON c.stay_id = w.stay_id;
""")

# ---------------------------------------------------------------------
# Export to Parquet
# ---------------------------------------------------------------------

out_path = OUT_DIR / "treatment_vaso6h.parquet"
con.execute(f"""
COPY treatment_vaso6h
TO '{out_path}'
(FORMAT PARQUET);
""")

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

print("\nTreatment prevalence (vasopressors within 6h):")
print(con.execute("SELECT AVG(t_vaso6h) AS p_treated FROM treatment_vaso6h").fetchdf())

print("\nHours to first vasopressor (treated only):")
print(con.execute("""
SELECT
  COUNT(*) AS n_treated,
  MIN(hours_to_first_vaso) AS min_h,
  AVG(hours_to_first_vaso) AS mean_h,
  MAX(hours_to_first_vaso) AS max_h
FROM treatment_vaso6h
WHERE t_vaso6h = 1;
""").fetchdf())

print(f"\nParquet file written to: {out_path}")
