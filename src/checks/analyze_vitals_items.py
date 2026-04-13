"""
Analyze candidate vital itemids:
- coverage in first 6h
- avg number of measurements per stay
- value range (min/max)

Used to select the best itemids per vital.
"""

import duckdb
from src.config import ICU_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
D_ITEMS = ICU_DIR / "d_items.csv.gz"
CHARTEVENTS = ICU_DIR / "chartevents.csv.gz"

con = duckdb.connect(str(DB_PATH))
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
con.execute("SET memory_limit='8GB';")

# Views
con.execute(f"""
CREATE OR REPLACE VIEW cohort_pq AS
SELECT stay_id, intime
FROM read_parquet('{COHORT_PATH.as_posix()}');
""")

con.execute(f"""
CREATE OR REPLACE VIEW d_items_v AS
SELECT * FROM read_csv_auto('{D_ITEMS.as_posix()}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW chartevents_v AS
SELECT * FROM read_csv_auto('{CHARTEVENTS.as_posix()}', union_by_name=true);
""")

# Reuse your wide candidate selection
con.execute("""
CREATE OR REPLACE VIEW vital_candidates_v AS
SELECT DISTINCT
  itemid,
  label,
  CASE
    WHEN lower(label) LIKE '%heart rate%' THEN 'hr'
    WHEN lower(label) LIKE '%respiratory rate%' THEN 'rr'
    WHEN lower(label) LIKE '%spo2%' OR lower(label) LIKE '%o2 saturation%' THEN 'spo2'
    WHEN lower(label) LIKE '%temperature%' OR lower(label) LIKE '%temp%' THEN 'temp'
    WHEN lower(label) LIKE '%mean bp%' OR lower(label) LIKE '%mean arterial%' THEN 'map'
    ELSE NULL
  END AS vital
FROM d_items_v
WHERE
  lower(label) LIKE '%heart rate%'
  OR lower(label) LIKE '%respiratory rate%'
  OR lower(label) LIKE '%spo2%'
  OR lower(label) LIKE '%o2 saturation%'
  OR lower(label) LIKE '%temperature%'
  OR lower(label) LIKE '%temp%'
  OR lower(label) LIKE '%mean bp%'
  OR lower(label) LIKE '%mean arterial%';
""")

# Score each itemid
df = con.execute("""
WITH windowed AS (
  SELECT
    c.stay_id,
    v.vital,
    v.itemid,
    ce.valuenum
  FROM cohort_pq c
  JOIN chartevents_v ce
    ON ce.stay_id = c.stay_id
  JOIN vital_candidates_v v
    ON ce.itemid = v.itemid
  WHERE ce.charttime >= c.intime
    AND ce.charttime <  c.intime + INTERVAL '6' HOUR
    AND ce.valuenum IS NOT NULL
),
agg AS (
  SELECT
    vital,
    itemid,
    COUNT(DISTINCT stay_id)::DOUBLE / (SELECT COUNT(*) FROM cohort_pq) AS coverage,
    COUNT(*)::DOUBLE / COUNT(DISTINCT stay_id) AS avg_measures,
    MIN(valuenum) AS min_val,
    MAX(valuenum) AS max_val
  FROM windowed
  GROUP BY vital, itemid
)
SELECT
  a.vital,
  a.itemid,
  d.label,
  ROUND(a.coverage, 3) AS coverage,
  ROUND(a.avg_measures, 1) AS avg_measures,
  a.min_val,
  a.max_val
FROM agg a
JOIN d_items_v d
  ON a.itemid = d.itemid
ORDER BY vital, coverage DESC;
""").fetchdf()

print(df)
