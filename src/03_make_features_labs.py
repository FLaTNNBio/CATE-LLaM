"""
Baseline lab features (v2): pick ONE itemid per canonical lab (most frequent in 6h window),
then create a wide table with one row per stay_id (same cardinality as cohort).

Window: [intime, intime + 6 hours)
Aggregation: FIRST value in the window for each selected lab.
"""
import duckdb
from config import HOSP_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
D_LABITEMS = HOSP_DIR / "d_labitems.csv.gz"
LABEVENTS = HOSP_DIR / "labevents.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "labs_6h_v2.parquet"

con = duckdb.connect(str(DB_PATH))
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
con.execute("SET memory_limit='4GB';")

# Use unique view names to avoid collisions
con.execute(f"""
CREATE OR REPLACE VIEW cohort_pq AS
SELECT subject_id, hadm_id, stay_id, intime
FROM read_parquet('{COHORT_PATH.as_posix()}');
""")

con.execute(f"""
CREATE OR REPLACE VIEW d_labitems_v AS
SELECT * FROM read_csv_auto('{D_LABITEMS.as_posix()}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW labevents_v AS
SELECT * FROM read_csv_auto('{LABEVENTS.as_posix()}', union_by_name=true);
""")

# Canonical labs we want (tight-ish patterns)
# We also explicitly exclude non-blood/specimen variants via negative filters.
con.execute("""
CREATE OR REPLACE TABLE canonical_labs AS
SELECT * FROM (VALUES
  ('creatinine',   '%creatinine%'),
  ('lactate',      '%lactate%'),
  ('wbc',          'wbc'),
  ('hemoglobin',   'hemoglobin'),
  ('platelets',    '%platelet%'),
  ('sodium',       '%sodium%'),
  ('potassium',    '%potassium%'),
  ('bicarbonate',  'bicarbonate')
) AS t(lab_name, pattern);
""")

# Candidate itemids per canonical lab, excluding obvious unwanted specimen types
con.execute("""
CREATE OR REPLACE TABLE lab_item_candidates AS
SELECT
  c.lab_name,
  d.itemid,
  d.label
FROM canonical_labs c
JOIN d_labitems_v d
  ON lower(d.label) LIKE lower(c.pattern)
WHERE
  -- exclude non-blood / special fluids / derived variants
  lower(d.label) NOT LIKE '%urine%'
  AND lower(d.label) NOT LIKE '%csf%'
  AND lower(d.label) NOT LIKE '%ascites%'
  AND lower(d.label) NOT LIKE '%pleural%'
  AND lower(d.label) NOT LIKE '%stool%'
  AND lower(d.label) NOT LIKE '%joint%'
  AND lower(d.label) NOT LIKE '%other fluid%'
  AND lower(d.label) NOT LIKE '%clearance%'
  AND lower(d.label) NOT LIKE '%ratio%'
  AND lower(d.label) NOT LIKE '%calculated%'
  AND lower(d.label) NOT LIKE '%a1c%';
""")

print("Candidate itemids per lab (sample):")
print(con.execute("""
SELECT lab_name, COUNT(*) AS n_candidates
FROM lab_item_candidates
GROUP BY lab_name
ORDER BY lab_name;
""").fetchdf())

# Count frequency in the actual 6h window and pick the most frequent itemid per lab
con.execute("""
CREATE OR REPLACE TABLE lab_item_choice AS
WITH window_events AS (
  SELECT
    c.stay_id,
    cand.lab_name,
    le.itemid
  FROM cohort_pq c
  JOIN labevents_v le
    ON le.hadm_id = c.hadm_id
  JOIN lab_item_candidates cand
    ON le.itemid = cand.itemid
  WHERE le.charttime IS NOT NULL
    AND le.valuenum IS NOT NULL
    AND le.charttime >= c.intime
    AND le.charttime <  c.intime + INTERVAL '6' HOUR
),
counts AS (
  SELECT lab_name, itemid, COUNT(*) AS n
  FROM window_events
  GROUP BY lab_name, itemid
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY lab_name ORDER BY n DESC) AS rn
  FROM counts
)
SELECT lab_name, itemid
FROM ranked
WHERE rn = 1;
""")

print("\nChosen itemid per lab:")
print(con.execute("""
SELECT
  c.lab_name,
  c.itemid,
  d.label
FROM lab_item_choice c
JOIN d_labitems_v d
  ON c.itemid = d.itemid
ORDER BY c.lab_name;
""").fetchdf())

# Extract first value for each chosen lab in the 6h window
con.execute("""
CREATE OR REPLACE TABLE labs_long AS
WITH chosen AS (
  SELECT * FROM lab_item_choice
),
filtered AS (
  SELECT
    c.stay_id,
    ch.lab_name,
    le.charttime,
    le.valuenum,
    c.intime
  FROM cohort_pq c
  JOIN labevents_v le
    ON le.hadm_id = c.hadm_id
  JOIN chosen ch
    ON le.itemid = ch.itemid
  WHERE le.charttime IS NOT NULL
    AND le.valuenum IS NOT NULL
    AND le.charttime >= c.intime
    AND le.charttime <  c.intime + INTERVAL '6' HOUR
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY stay_id, lab_name ORDER BY charttime) AS rn
  FROM filtered
)
SELECT stay_id, lab_name, valuenum
FROM ranked
WHERE rn = 1;
""")

# Pivot to wide
con.execute("""
CREATE OR REPLACE TABLE labs_wide AS
SELECT *
FROM (
  PIVOT labs_long
  ON lab_name IN ('creatinine','lactate','wbc','hemoglobin','platelets','sodium','potassium','bicarbonate')
  USING first(valuenum)
);
""")

# LEFT JOIN with cohort to guarantee 1 row per stay_id
con.execute("""
CREATE OR REPLACE TABLE labs_6h_v2 AS
SELECT
  c.stay_id,
  w.creatinine,
  w.lactate,
  w.wbc,
  w.hemoglobin,
  w.platelets,
  w.sodium,
  w.potassium,
  w.bicarbonate
FROM cohort_pq c
LEFT JOIN labs_wide w
  ON c.stay_id = w.stay_id;
""")

# Export
con.execute(f"""
COPY labs_6h_v2
TO '{OUT_PATH.as_posix()}'
(FORMAT PARQUET);
""")

print("\nFinal labs table size (should match cohort):")
print(con.execute("SELECT COUNT(*) AS n FROM labs_6h_v2").fetchdf())

print("\nNull fraction per lab:")
print(con.execute("""
SELECT
  1 - COUNT(creatinine)::DOUBLE/COUNT(*) AS creatinine_null_frac,
  1 - COUNT(lactate)::DOUBLE/COUNT(*) AS lactate_null_frac,
  1 - COUNT(wbc)::DOUBLE/COUNT(*) AS wbc_null_frac,
  1 - COUNT(hemoglobin)::DOUBLE/COUNT(*) AS hemoglobin_null_frac,
  1 - COUNT(platelets)::DOUBLE/COUNT(*) AS platelets_null_frac,
  1 - COUNT(sodium)::DOUBLE/COUNT(*) AS sodium_null_frac,
  1 - COUNT(potassium)::DOUBLE/COUNT(*) AS potassium_null_frac,
  1 - COUNT(bicarbonate)::DOUBLE/COUNT(*) AS bicarbonate_null_frac
FROM labs_6h_v2;
""").fetchdf())

print(f"\nParquet written to: {OUT_PATH}")
