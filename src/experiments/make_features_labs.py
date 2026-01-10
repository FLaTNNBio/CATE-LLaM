"""
Extract baseline lab features from MIMIC-IV and aggregate to one row per ICU stay.

Unit: stay_id (ICU stay)
Time zero: cohort.intime
Window: [intime, intime + 6 hours)

We:
1) identify target lab itemids via d_labitems label patterns
2) filter labevents to those itemids and to the cohort hadm_ids
3) restrict to the baseline window relative to ICU intime
4) aggregate to 1 row per stay_id (we take the FIRST value in the window)

Output: labs_6h.parquet (stay_id + lab feature columns)
"""

import duckdb
from config import HOSP_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
D_LABITEMS = HOSP_DIR / "d_labitems.csv.gz"
LABEVENTS = HOSP_DIR / "labevents.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "labs_6h.parquet"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

# Allow DuckDB to spill intermediates to disk instead of RAM
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
con.execute("SET memory_limit='8GB';")  # adjust if you want (e.g., '2GB')

# Clean heavy views before execution to be sure and avoid RAM explosion
con.execute("DROP VIEW IF EXISTS cohort_pq;")
con.execute("DROP VIEW IF EXISTS labevents_v;")
con.execute("DROP VIEW IF EXISTS d_labitems_v;")

# Register cohort parquet as a view with a UNIQUE name (avoid collisions)
con.execute(f"""
CREATE OR REPLACE VIEW cohort_pq AS
SELECT subject_id, hadm_id, stay_id, intime
FROM read_parquet('{COHORT_PATH.as_posix()}');
""")

# Register lab tables
con.execute(f"""
CREATE OR REPLACE VIEW d_labitems_v AS
SELECT * FROM read_csv_auto('{D_LABITEMS.as_posix()}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW labevents_v AS
SELECT * FROM read_csv_auto('{LABEVENTS.as_posix()}', union_by_name=true);
""")


# ------------------------------------------------------------------
# 1) Choose lab itemids via label patterns
# (You can refine patterns later)
# ------------------------------------------------------------------
con.execute("""
CREATE OR REPLACE TABLE target_lab_itemids AS
SELECT DISTINCT itemid, label
FROM d_labitems_v
WHERE
  lower(label) LIKE '%creatinine%'
  OR lower(label) LIKE '%lactate%'
  OR lower(label) = 'wbc'
  OR lower(label) LIKE '%white blood cell%'
  OR lower(label) LIKE '%hemoglobin%'
  OR lower(label) LIKE '%platelet%'
  OR lower(label) LIKE '%sodium%'
  OR lower(label) LIKE '%potassium%'
  OR lower(label) LIKE '%bicarbonate%'
  OR lower(label) = 'co2';
""")

print("Target lab itemids found:")
print(con.execute("SELECT COUNT(*) AS n_itemids FROM target_lab_itemids").fetchdf())
print(con.execute("SELECT * FROM target_lab_itemids ORDER BY label LIMIT 50").fetchdf())

# ------------------------------------------------------------------
# 2) Filter labevents to:
# - cohort hadm_id
# - selected lab itemids
# - baseline time window relative to ICU intime
# ------------------------------------------------------------------
# We'll take the FIRST lab value in the window for each (stay_id, itemid).
# Use a window function to rank by charttime.

con.execute("""
CREATE OR REPLACE TABLE labs_windowed AS
WITH filtered AS (
  SELECT
    c.stay_id,
    le.itemid,
    le.charttime,
    le.valuenum,
    c.intime
  FROM cohort_pq c
  JOIN labevents_v le
    ON c.hadm_id = le.hadm_id
  JOIN target_lab_itemids t
    ON le.itemid = t.itemid
  WHERE le.charttime IS NOT NULL
    AND le.valuenum IS NOT NULL
    AND le.charttime >= c.intime
    AND le.charttime <  c.intime + INTERVAL '6' HOUR
),
ranked AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY stay_id, itemid
      ORDER BY charttime
    ) AS rn
  FROM filtered
)
SELECT
  stay_id,
  itemid,
  valuenum
FROM ranked
WHERE rn = 1;
""")

# ------------------------------------------------------------------
# 3) Pivot into wide format: one row per stay_id
# DuckDB supports PIVOT
# We create human-readable column names by mapping itemid -> canonical name
# For simplicity, we use itemid-based names first; we can rename later.
# ------------------------------------------------------------------

# Create a stable list of itemids to pivot
itemids = con.execute("SELECT DISTINCT itemid FROM labs_windowed ORDER BY itemid").fetchall()
itemids = [str(x[0]) for x in itemids]

if not itemids:
    raise RuntimeError("No lab events found in the 6h window. Check charttime/valuenum and patterns.")

pivot_in = ", ".join(itemids)

con.execute(f"""
CREATE OR REPLACE TABLE labs_6h AS
PIVOT (
  SELECT stay_id, itemid, valuenum
  FROM labs_windowed
)
ON itemid IN ({pivot_in})
USING first(valuenum);
""")

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
con.execute(f"""
COPY labs_6h
TO '{OUT_PATH.as_posix()}'
(FORMAT PARQUET);
""")

# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------
print("\nLabs feature table size:")
print(con.execute("SELECT COUNT(*) AS n FROM labs_6h").fetchdf())

print("\nMissingness overview (fraction of NULL per column, first 15 cols):")
# DuckDB doesn't have a single function for this; do a small sample check
cols = con.execute("PRAGMA table_info('labs_6h')").fetchdf()["name"].tolist()
cols = [c for c in cols if c != "stay_id"]

# Show missingness for a subset to keep it readable
for c in cols[:15]:
    q = f"SELECT 1 - (COUNT({c})::DOUBLE / COUNT(*)) AS null_frac FROM labs_6h;"
    val = con.execute(q).fetchone()[0]
    print(f"{c}: {val:.3f}")

print(f"\nParquet file written to: {OUT_PATH}")
