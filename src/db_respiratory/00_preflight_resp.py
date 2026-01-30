import duckdb
from src.config import INTERMEDIATE_DIR, ICU_DIR, ANALYTIC_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Core ICU cohort with stay_id, hadm_id, subject_id, intime
COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"

# OMR-derived anthropometrics produced previously (stay_id, admission_weight_kg, height_cm, bmi)
# NOTE: this file (as you observed) often stores lb/in; we deterministically convert below.
GENERAL_PATH = INTERMEDIATE_DIR / "general.parquet"

# Raw event tables
CHARTEVENTS = ICU_DIR / "chartevents.csv.gz"
D_ITEMS = ICU_DIR / "d_items.csv.gz"

ANALYTIC_DIR.mkdir(parents=True, exist_ok=True)

def _csv(path) -> str:
    return path.as_posix().replace("'", "''")

con = duckdb.connect(str(DB_PATH))
con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
con.execute("SET memory_limit='8GB';")

# -------------------------
# Load core inputs as views
# -------------------------
con.execute(f"""
CREATE OR REPLACE VIEW v_cohort AS
SELECT subject_id, hadm_id, stay_id, intime
FROM read_parquet('{_csv(COHORT_PATH)}');
""")

con.execute(f"""
CREATE OR REPLACE VIEW v_chart AS
SELECT * FROM read_csv_auto('{_csv(CHARTEVENTS)}', union_by_name=true);
""")

con.execute(f"""
CREATE OR REPLACE VIEW v_ditems AS
SELECT * FROM read_csv_auto('{_csv(D_ITEMS)}', union_by_name=true);
""")

print(f"- Loaded core views: v_cohort, v_chart, v_ditems")

# -------------------------

con.execute(f"""
CREATE OR REPLACE TABLE v_t0 AS
    WITH ranked AS (
      SELECT
        stay_id, hadm_id, subject_id, intime,
        charttime AS t0_time,
        support_type AS t0_support,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime) AS rn
      FROM v_resp_support
    )
    SELECT
      stay_id, hadm_id, subject_id, intime,
      t0_time,
      t0_support
    FROM ranked
    WHERE rn = 1;
""")

print(f"- Generated v_t0")


con.execute(f"""
CREATE OR REPLACE TABLE v_resp_itemids AS
    SELECT itemid, label
    FROM v_ditems
    WHERE lower(label) LIKE '%oxygen%'
       OR lower(label) LIKE '%o2%'
       OR lower(label) LIKE '%ventilation%'
       OR lower(label) LIKE '%ventilator%'
       OR lower(label) LIKE '%resp%';
""")

print(f"- Generated v_resp_itemids with candidate respiratory itemids")

con.execute(f"""
CREATE OR REPLACE TABLE v_resp_support AS
    WITH ce AS (
      SELECT
        c.stay_id,
        c.hadm_id,
        c.subject_id,
        c.intime,
        ch.charttime,
        ch.itemid,
        ch.value
      FROM v_cohort c
      JOIN v_chart ch
        ON ch.stay_id = c.stay_id
      JOIN v_resp_itemids di
        ON di.itemid = ch.itemid
      WHERE ch.charttime >= c.intime
        AND ch.charttime <  c.intime + INTERVAL '24' HOUR
        AND ch.value IS NOT NULL
    ),
    classified AS (
      SELECT
        *,
        CASE
          WHEN lower(value) LIKE '%high flow%' OR lower(value) LIKE '%hfnc%'
            THEN 'HFNC'
          WHEN lower(value) LIKE '%bipap%' OR lower(value) LIKE '%bi-pap%'
            OR lower(value) LIKE '%cpap%'
            OR lower(value) LIKE '%noninvasive%' OR lower(value) LIKE '%non-invasive%'
            OR lower(value) LIKE '%niv%'
            THEN 'NIV'
          ELSE NULL
        END AS support_type
      FROM ce
    )
    SELECT *
    FROM classified
    WHERE support_type IS NOT NULL;
""")

print(f"- Generated v_resp_support with respiratory support classification")

# CHECK 1

#sp_types = con.execute("SELECT DISTINCT support_type FROM v_resp_support GROUP BY 1;").fetchall()
#print(f"- Identified support types: {[st[0] for st in sp_types]}")

#t0_support_cases = con.execute("SELECT t0_support, COUNT(*) FROM v_t0 GROUP BY 1;").fetchall()
#print(f"- t0_support distribution: {t0_support_cases}")

# res = con.execute("""
#     SELECT support_type, lower(value), COUNT(*)
#     FROM v_resp_support
#     GROUP BY 1,2 ORDER BY COUNT(*) DESC LIMIT 50;
# """).fetchall()
# print("- Sample of support_type and values:")
# for row in res:
#     print(f"  {row[0]} | {row[1]} : {row[2]}")

# CHECK 2

item_dominance = con.execute("""
SELECT support_type, itemid, COUNT(*) AS n
    FROM v_resp_support
    GROUP BY 1,2
    ORDER BY n DESC
    LIMIT 30;
""").fetchall()
print("- Top itemid dominance per support_type:")
for row in item_dominance:
    print(f"  {row[0]} | itemid={row[1]} : {row[2]}")

most_common_values = con.execute("""
    SELECT support_type, lower(value) AS v, itemid, COUNT(*) AS n
    FROM v_resp_support
    GROUP BY 1,2,3
    ORDER BY n DESC
    LIMIT 50;
""").fetchall()
print("- Most common values per support_type:")
for row in most_common_values:
    print(f"  {row[0]} | {row[1]} | itemid={row[2]} : {row[3]}")

events_per_stay = con.execute("""
    WITH first AS (
      SELECT stay_id, MIN(charttime) AS t_first
      FROM v_resp_support
      GROUP BY stay_id
    ),
    win AS (
      SELECT r.stay_id, r.support_type
      FROM v_resp_support r
      JOIN first f USING (stay_id)
      WHERE r.charttime >= f.t_first
        AND r.charttime <  f.t_first + INTERVAL '2' HOUR
    )
    SELECT stay_id,
           SUM(CASE WHEN support_type='HFNC' THEN 1 ELSE 0 END) AS n_hfnc,
           SUM(CASE WHEN support_type='NIV'  THEN 1 ELSE 0 END) AS n_niv
    FROM win
    GROUP BY stay_id;
""").fetchdf()
print(f"- Sample of events per stay in first 2 hours:")
print(events_per_stay.head())





print("\nPreflight respiratory data preparation completed.")