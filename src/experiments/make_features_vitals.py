"""
Extract baseline vital sign features from chartevents and aggregate to one row per ICU stay.

Unit: stay_id
Time zero: cohort.intime
Window: [intime, intime + 6 hours)

We:
1) find itemids for a small set of standard vitals via d_items label patterns
2) filter chartevents to cohort stay_ids, those itemids, and the baseline window
3) aggregate per stay_id (mean values per vital)

Output: vitals_6h.parquet
"""

import duckdb
from config import ICU_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
D_ITEMS = ICU_DIR / "d_items.csv.gz"
CHARTEVENTS = ICU_DIR / "chartevents.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "vitals_6h.parquet"

con = duckdb.connect(str(DB_PATH))
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
con.execute("SET memory_limit='4GB';")  # adjust if needed

# Cohort
con.execute(f"""
CREATE OR REPLACE VIEW cohort_pq AS
SELECT stay_id, intime
FROM read_parquet('{COHORT_PATH.as_posix()}');
""")

# ICU dictionaries and events
con.execute(f"""
CREATE OR REPLACE VIEW d_items_v AS
SELECT * FROM read_csv_auto('{D_ITEMS.as_posix()}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW chartevents_v AS
SELECT * FROM read_csv_auto('{CHARTEVENTS.as_posix()}', union_by_name=true);
""")

# 1) Find candidate itemids for standard vitals (tight patterns)
con.execute("""
CREATE OR REPLACE TABLE vital_candidates AS
SELECT DISTINCT
  itemid,
  label
FROM d_items_v
WHERE
  -- Heart Rate
  lower(label) LIKE '%heart rate%'

  OR
  -- Respiratory Rate
  lower(label) LIKE '%respiratory rate%'

  OR
  -- SpO2
  lower(label) LIKE '%spo2%'
  OR lower(label) LIKE '%o2 saturation%'

  OR
  -- Temperature
  lower(label) LIKE '%temperature%'
  OR lower(label) LIKE '%temp%'

  OR
  -- Mean Arterial Pressure
  lower(label) LIKE '%mean bp%'
  OR lower(label) LIKE '%mean arterial%'
;
""")

print("Vital itemid candidates:")
print(con.execute("SELECT COUNT(*) AS n_itemids FROM vital_candidates").fetchdf())
print(con.execute("SELECT * FROM vital_candidates ORDER BY label").fetchdf())

# 2) Map each candidate itemid to a canonical vital name
# We prefer to keep it simple and deterministic:
con.execute("""
CREATE OR REPLACE TABLE vital_map AS
SELECT
  itemid,
  CASE
    WHEN lower(label) = 'heart rate' THEN 'hr'
    WHEN lower(label) IN ('mean arterial pressure', 'map') OR lower(label) LIKE '%mean bp%' THEN 'map'
    WHEN lower(label) = 'respiratory rate' THEN 'rr'
    WHEN lower(label) IN ('spo2', 'o2 saturation pulseoxymetry') THEN 'spo2'
    WHEN lower(label) LIKE 'temperature%' OR lower(label) = 'temperature' THEN 'temp'
    ELSE NULL
  END AS vital
FROM vital_candidates
WHERE
  CASE
    WHEN lower(label) = 'heart rate' THEN 1
    WHEN lower(label) IN ('mean arterial pressure', 'map') OR lower(label) LIKE '%mean bp%' THEN 1
    WHEN lower(label) = 'respiratory rate' THEN 1
    WHEN lower(label) IN ('spo2', 'o2 saturation pulseoxymetry') THEN 1
    WHEN lower(label) LIKE 'temperature%' OR lower(label) = 'temperature' THEN 1
    ELSE 0
  END = 1;
""")

# 3) Filter chartevents to baseline window and usable numeric values
# We use valuenum and ignore text values.
con.execute("""
CREATE OR REPLACE TABLE vitals_long AS
SELECT
  c.stay_id,
  m.vital,
  ce.charttime,
  ce.valuenum,
  c.intime
FROM cohort_pq c
JOIN chartevents_v ce
  ON ce.stay_id = c.stay_id
JOIN vital_map m
  ON ce.itemid = m.itemid
WHERE ce.charttime IS NOT NULL
  AND ce.valuenum IS NOT NULL
  AND ce.charttime >= c.intime
  AND ce.charttime <  c.intime + INTERVAL '6' HOUR;
""")

# 4) Aggregate: mean per (stay_id, vital)
con.execute("""
CREATE OR REPLACE TABLE vitals_agg AS
SELECT
  stay_id,
  vital,
  AVG(valuenum) AS mean_value
FROM vitals_long
GROUP BY stay_id, vital;
""")

# 5) Pivot to wide: one row per stay_id (hr/map/rr/spo2/temp)
con.execute("""
CREATE OR REPLACE TABLE vitals_wide AS
SELECT *
FROM (
  PIVOT vitals_agg
  ON vital IN ('hr','map','rr','spo2','temp')
  USING first(mean_value)
);
""")

# 6) LEFT JOIN with cohort to keep full cardinality
con.execute("""
CREATE OR REPLACE TABLE vitals_6h AS
SELECT
  c.stay_id,
  w.hr,
  w.map,
  w.rr,
  w.spo2,
  w.temp
FROM cohort_pq c
LEFT JOIN vitals_wide w
  ON c.stay_id = w.stay_id;
""")

# Export
con.execute(f"""
COPY vitals_6h
TO '{OUT_PATH.as_posix()}'
(FORMAT PARQUET);
""")

# Sanity checks
print("\nFinal vitals table size (should match cohort):")
print(con.execute("SELECT COUNT(*) AS n FROM vitals_6h").fetchdf())

print("\nNull fraction per vital:")
print(con.execute("""
SELECT
  1 - COUNT(hr)::DOUBLE/COUNT(*) AS hr_null_frac,
  1 - COUNT(map)::DOUBLE/COUNT(*) AS map_null_frac,
  1 - COUNT(rr)::DOUBLE/COUNT(*) AS rr_null_frac,
  1 - COUNT(spo2)::DOUBLE/COUNT(*) AS spo2_null_frac,
  1 - COUNT(temp)::DOUBLE/COUNT(*) AS temp_null_frac
FROM vitals_6h;
""").fetchdf())

print(f"\nParquet written to: {OUT_PATH}")
