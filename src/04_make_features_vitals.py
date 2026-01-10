# src/pipeline/04_make_features_vitals.py
"""
Build baseline vitals features from chartevents (v0, manually curated itemids).

Unit: stay_id
Time zero: cohort.intime
Window: [intime, intime + 6 hours)

Aggregation: mean value per stay_id and vital (from selected itemids only)

Output: data/intermediate/vitals_6h.parquet
"""
import duckdb

from src.config import ICU_DIR, INTERMEDIATE_DIR
from src.constants import VITAL_ITEMIDS

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
CHARTEVENTS = ICU_DIR / "chartevents.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "vitals_6h.parquet"

WINDOW_HOURS = 6


def _flat_itemids(vital_itemids: dict[str, list[int]]) -> list[int]:
    ids: list[int] = []
    for _, v in vital_itemids.items():
        ids.extend(v)
    return sorted(set(ids))


def main() -> None:
    if not COHORT_PATH.exists():
        raise FileNotFoundError(f"Missing cohort parquet: {COHORT_PATH}")
    if not CHARTEVENTS.exists():
        raise FileNotFoundError(f"Missing chartevents: {CHARTEVENTS}")

    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='4GB';")

    # Views
    con.execute(f"""
    CREATE OR REPLACE VIEW cohort_pq AS
    SELECT stay_id, intime
    FROM read_parquet('{COHORT_PATH.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW chartevents_v AS
    SELECT * FROM read_csv_auto('{CHARTEVENTS.as_posix()}', union_by_name=true);
    """)

    # Build itemid -> vital_name mapping table
    rows = []
    for vital_name, itemids in VITAL_ITEMIDS.items():
        for iid in itemids:
            rows.append((iid, vital_name))

    values_sql = ",\n".join([f"({iid}, '{vital}')" for iid, vital in rows])

    con.execute(f"""
    CREATE OR REPLACE TABLE vital_map AS
    SELECT * FROM (VALUES
      {values_sql}
    ) AS t(itemid, vital);
    """)

    itemids_all = _flat_itemids(VITAL_ITEMIDS)
    itemids_in = ", ".join(map(str, itemids_all))

    # Extract windowed numeric vitals
    con.execute(f"""
    CREATE OR REPLACE TABLE vitals_long AS
    SELECT
      c.stay_id,
      vm.vital,
      ce.charttime,
      ce.valuenum
    FROM cohort_pq c
    JOIN chartevents_v ce
      ON ce.stay_id = c.stay_id
    JOIN vital_map vm
      ON ce.itemid = vm.itemid
    WHERE ce.itemid IN ({itemids_in})
      AND ce.charttime IS NOT NULL
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >= c.intime
      AND ce.charttime <  c.intime + INTERVAL '{WINDOW_HOURS}' HOUR
    ;
    """)

    # Aggregate mean per stay_id and vital
    con.execute("""
    CREATE OR REPLACE TABLE vitals_agg AS
    SELECT
      stay_id,
      vital,
      AVG(valuenum) AS mean_value
    FROM vitals_long
    GROUP BY stay_id, vital;
    """)

    # Pivot to wide
    con.execute("""
    CREATE OR REPLACE TABLE vitals_wide AS
    SELECT *
    FROM (
      PIVOT vitals_agg
      ON vital IN ('hr','rr','spo2','temp_c','nibp_sys','nibp_dia','nibp_mean')
      USING first(mean_value)
    );
    """)

    # Ensure full cardinality (left join with cohort)
    con.execute("""
    CREATE OR REPLACE TABLE vitals_6h AS
    SELECT
      c.stay_id,
      w.hr,
      w.rr,
      w.spo2,
      w.temp_c,
      w.nibp_sys,
      w.nibp_dia,
      w.nibp_mean
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
    print("Final vitals table size (should match cohort):")
    print(con.execute("SELECT COUNT(*) AS n FROM vitals_6h;").fetchdf())

    print("\nNull fraction per vital:")
    print(con.execute("""
    SELECT
      1 - COUNT(hr)::DOUBLE/COUNT(*) AS hr_null_frac,
      1 - COUNT(rr)::DOUBLE/COUNT(*) AS rr_null_frac,
      1 - COUNT(spo2)::DOUBLE/COUNT(*) AS spo2_null_frac,
      1 - COUNT(temp_c)::DOUBLE/COUNT(*) AS temp_null_frac,
      1 - COUNT(nibp_sys)::DOUBLE/COUNT(*) AS nibp_sys_null_frac,
      1 - COUNT(nibp_dia)::DOUBLE/COUNT(*) AS nibp_dia_null_frac,
      1 - COUNT(nibp_mean)::DOUBLE/COUNT(*) AS nibp_mean_null_frac
    FROM vitals_6h;
    """).fetchdf())

    print(f"\nParquet written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
