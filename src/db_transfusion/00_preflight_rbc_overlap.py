import duckdb
from src.config import HOSP_DIR, INTERMEDIATE_DIR, ICU_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT = INTERMEDIATE_DIR / "cohort_clean.parquet"          # stay_id, hadm_id, subject_id, intime
LABEVENTS = HOSP_DIR / "labevents.csv.gz"
INPUTEVENTS = INTERMEDIATE_DIR / "inputevents_filtered.parquet"  # or read_csv_auto ICU/inputevents.csv.gz
INPUTEVENTS_NO_FILTERED = ICU_DIR / "inputevents.csv.gz"
# If no filtered parquet per inputevents, use ICU_DIR/inputevents.csv.gz with read_csv_auto.

# >>>> SET THESE <<<<
HB_ITEMID = 51222  # (d_labitems): check right code (lab) - should be ok
RBC_ITEMIDS = [225168, 226368, 227070]  #: Packed Red Blood Cells # (check itemid d_items/inputevents)
# 225168 – Packed Red Blood Cells (PRBC)
# 226368 – OR Packed RBC Intake
# 227070 – PACU Packed RBC Intake

THRESHOLDS = [7.0, 7.5, 8.0]
WINDOW_HOURS = [2, 3, 6]
ELIG_WITHIN_HOURS_FROM_INTIME = 48


def main():
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='8GB';")

    con.execute(f"""
    CREATE OR REPLACE VIEW v_cohort AS
    SELECT stay_id, hadm_id, intime
    FROM read_parquet('{COHORT.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_labs AS
    SELECT * FROM read_csv_auto('{LABEVENTS.as_posix()}', union_by_name=true);
    """)

    # If filtered parquet per inputevents ok; else use ICU_DIR/inputevents.csv.gz with read_csv_auto.
    #con.execute(f"""
    #CREATE OR REPLACE VIEW v_inputs AS
    #SELECT * FROM read_parquet('{INPUTEVENTS.as_posix()}');
    #""")

    con.execute(f"""
    CREATE OR REPLACE VIEW v_inputs AS
    SELECT * FROM read_csv_auto('{INPUTEVENTS_NO_FILTERED.as_posix()}', union_by_name=true);
    """)


    rbc_list = ", ".join(str(x) for x in RBC_ITEMIDS)

    for thr in THRESHOLDS:
        # time_zero: first Hb below threshold within first 48h of ICU
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
            AND le.charttime <  c.intime + INTERVAL '{ELIG_WITHIN_HOURS_FROM_INTIME}' HOUR
            AND le.valuenum < {thr}
        ),
        ranked AS (
          SELECT *,
                 ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime) AS rn
          FROM hb
        )
        SELECT stay_id, charttime AS t0_time, hb AS hb_t0
        FROM ranked
        WHERE rn = 1;
        """)

        n_elig = con.execute("SELECT COUNT(*) FROM v_t0;").fetchone()[0]
        print(f"\n=== Hb < {thr} === eligible stays: {n_elig}")

        if n_elig == 0:
            continue

        for wh in WINDOW_HOURS:
            con.execute(f"""
            CREATE OR REPLACE TABLE v_assign AS
            SELECT
              t0.stay_id,
              t0.t0_time,
              t0.hb_t0,
              CASE WHEN EXISTS (
                SELECT 1
                FROM v_inputs ie
                WHERE ie.stay_id = t0.stay_id
                  AND ie.itemid IN ({rbc_list})
                  AND ie.starttime IS NOT NULL
                  AND ie.starttime >= t0.t0_time
                  AND ie.starttime <  t0.t0_time + INTERVAL '{wh}' HOUR
              ) THEN 1 ELSE 0 END AS t_rbc_early
            FROM v_t0 t0;
            """)

            out = con.execute("""
            SELECT
              AVG(t_rbc_early)::DOUBLE AS p_treated,
              SUM(CASE WHEN t_rbc_early=1 THEN 1 ELSE 0 END) AS n_treated,
              SUM(CASE WHEN t_rbc_early=0 THEN 1 ELSE 0 END) AS n_control
            FROM v_assign;
            """).fetchdf()

            print(f"  window {wh}h ->", out.to_string(index=False))


if __name__ == "__main__":
    main()
