import duckdb
from src.config import HOSP_DIR, INTERMEDIATE_DIR, ICU_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT = INTERMEDIATE_DIR / "cohort_clean.parquet"
LABEVENTS = HOSP_DIR / "labevents.csv.gz"
INPUTEVENTS = ICU_DIR / "inputevents.csv.gz"  # inputevents

# === PARAMETERS ===
HB_ITEMID = 51222          # hemoglobin
HB_THRESHOLD = 7.5
ELIG_WITHIN_HOURS = 48
WINDOW_HOURS = 3

RBC_ITEMIDS = [225168, 226368, 227070]
PLASMA_ITEMIDS = [220970]      # Fresh Frozen Plasma
PLATELET_ITEMIDS = [225170]    # Platelets


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='8GB';")

    # Base tables
    con.execute(f"""
    CREATE OR REPLACE VIEW v_cohort AS
    SELECT stay_id, hadm_id, intime
    FROM read_parquet('{COHORT.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_labs AS
    SELECT * FROM read_csv_auto('{LABEVENTS.as_posix()}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_inputs AS
    SELECT * FROM read_csv_auto('{INPUTEVENTS.as_posix()}', union_by_name=true);
    """)

    rbc_list = ", ".join(str(x) for x in RBC_ITEMIDS)
    plasma_list = ", ".join(str(x) for x in PLASMA_ITEMIDS)
    platelet_list = ", ".join(str(x) for x in PLATELET_ITEMIDS)

    # 1) time-zero: first Hb < 7.5 within 48h
    con.execute(f"""
    CREATE OR REPLACE TABLE v_t0 AS
    WITH hb AS (
      SELECT
        c.stay_id,
        le.charttime,
        le.valuenum AS hb
      FROM v_cohort c
      JOIN v_labs le
        ON le.hadm_id = c.hadm_id
      WHERE le.itemid = {HB_ITEMID}
        AND le.charttime IS NOT NULL
        AND le.valuenum IS NOT NULL
        AND le.charttime >= c.intime
        AND le.charttime <  c.intime + INTERVAL '{ELIG_WITHIN_HOURS}' HOUR
        AND le.valuenum < {HB_THRESHOLD}
    ),
    ranked AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime) AS rn
      FROM hb
    )
    SELECT stay_id, charttime AS t0_time
    FROM ranked
    WHERE rn = 1;
    """)

    # 2) Assign early RBC
    con.execute(f"""
    CREATE OR REPLACE TABLE v_assign AS
    SELECT
      t0.stay_id,
      CASE WHEN EXISTS (
        SELECT 1
        FROM v_inputs ie
        WHERE ie.stay_id = t0.stay_id
          AND ie.itemid IN ({rbc_list})
          AND ie.starttime IS NOT NULL
          AND ie.starttime >= t0.t0_time
          AND ie.starttime <  t0.t0_time + INTERVAL '{WINDOW_HOURS}' HOUR
      ) THEN 1 ELSE 0 END AS t_rbc_early
    FROM v_t0 t0;
    """)

    # 3) Among CONTROLS (no early RBC), check plasma / platelets
    res = con.execute(f"""
    SELECT
      COUNT(*) AS n_controls,

      SUM(CASE WHEN EXISTS (
        SELECT 1 FROM v_inputs ie
        WHERE ie.stay_id = a.stay_id
          AND ie.itemid IN ({plasma_list})
          AND ie.starttime >= t0.t0_time
          AND ie.starttime <  t0.t0_time + INTERVAL '{WINDOW_HOURS}' HOUR
      ) THEN 1 ELSE 0 END) AS n_plasma_only,

      SUM(CASE WHEN EXISTS (
        SELECT 1 FROM v_inputs ie
        WHERE ie.stay_id = a.stay_id
          AND ie.itemid IN ({platelet_list})
          AND ie.starttime >= t0.t0_time
          AND ie.starttime <  t0.t0_time + INTERVAL '{WINDOW_HOURS}' HOUR
      ) THEN 1 ELSE 0 END) AS n_platelets_only,

      SUM(CASE WHEN EXISTS (
        SELECT 1 FROM v_inputs ie
        WHERE ie.stay_id = a.stay_id
          AND ie.itemid IN ({plasma_list}, {platelet_list})
          AND ie.starttime >= t0.t0_time
          AND ie.starttime <  t0.t0_time + INTERVAL '{WINDOW_HOURS}' HOUR
      ) THEN 1 ELSE 0 END) AS n_any_non_rbc

    FROM v_assign a
    JOIN v_t0 t0
      ON a.stay_id = t0.stay_id
    WHERE a.t_rbc_early = 0;
    """).fetchdf()

    print("\n=== Non-RBC transfusions among CONTROLS (Hb < 7.5, Δ=3h) ===")
    print(res)

    # Percentages
    props = con.execute("""
    SELECT
      n_controls,
      n_plasma_only::DOUBLE / n_controls AS p_plasma,
      n_platelets_only::DOUBLE / n_controls AS p_platelets,
      n_any_non_rbc::DOUBLE / n_controls AS p_any_non_rbc
    FROM (
      SELECT
        COUNT(*) AS n_controls,
        SUM(CASE WHEN EXISTS (
          SELECT 1 FROM v_inputs ie
          WHERE ie.stay_id = a.stay_id
            AND ie.itemid IN (220970)
            AND ie.starttime >= t0.t0_time
            AND ie.starttime <  t0.t0_time + INTERVAL '3' HOUR
        ) THEN 1 ELSE 0 END) AS n_plasma_only,
        SUM(CASE WHEN EXISTS (
          SELECT 1 FROM v_inputs ie
          WHERE ie.stay_id = a.stay_id
            AND ie.itemid IN (225170)
            AND ie.starttime >= t0.t0_time
            AND ie.starttime <  t0.t0_time + INTERVAL '3' HOUR
        ) THEN 1 ELSE 0 END) AS n_platelets_only,
        SUM(CASE WHEN EXISTS (
          SELECT 1 FROM v_inputs ie
          WHERE ie.stay_id = a.stay_id
            AND ie.itemid IN (220970,225170)
            AND ie.starttime >= t0.t0_time
            AND ie.starttime <  t0.t0_time + INTERVAL '3' HOUR
        ) THEN 1 ELSE 0 END) AS n_any_non_rbc
      FROM v_assign a
      JOIN v_t0 t0
        ON a.stay_id = t0.stay_id
      WHERE a.t_rbc_early = 0
    );
    """).fetchdf()

    print("\nProportions among controls:")
    print(props)

if __name__ == "__main__":
    main()
