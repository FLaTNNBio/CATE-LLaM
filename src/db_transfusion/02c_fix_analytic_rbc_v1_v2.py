import duckdb
from src.config import HOSP_DIR, ICU_DIR, INTERMEDIATE_DIR, ANALYTIC_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = ANALYTIC_DIR / "analytic_rbc_v1.parquet"
OUT_PATH = ANALYTIC_DIR / "analytic_rbc_v1_fixed.parquet"

LABEVENTS = HOSP_DIR / "labevents.csv.gz"
CHARTEVENTS = ICU_DIR / "chartevents.csv.gz"

LOOKBACK_HOURS = 6
VITALS_GRACE_HOURS = 1
LABS_GRACE_HOURS = 1

TEMP_C_ITEMID = 223762
TEMP_F_ITEMID = 223761

HR_ITEMID = 220045
RR_ITEMID = 220210
SPO2_ITEMID = 220277

NIBP_SYS = 220179
NIBP_DIA = 220180
NIBP_MEAN = 220181

ART_SYS_CANDS = [220050, 225309]
ART_DIA_CANDS = [220051, 225310]
ART_MEAN_CANDS = [220052, 225312]

WBC_ITEMIDS = [51300, 51301, 51516]


def _csv(path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='8GB';")

    con.execute(f"""
    CREATE OR REPLACE VIEW v_ana AS
    SELECT * FROM read_parquet('{_csv(IN_PATH)}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_keys AS
    SELECT stay_id, hadm_id, t0_time
    FROM v_ana;
    """)

    # Source tables
    con.execute(f"""
    CREATE OR REPLACE VIEW v_chart_raw AS
    SELECT * FROM read_csv_auto('{_csv(CHARTEVENTS)}', union_by_name=true);
    """)
    con.execute(f"""
    CREATE OR REPLACE VIEW v_labs_raw AS
    SELECT * FROM read_csv_auto('{_csv(LABEVENTS)}', union_by_name=true);
    """)

    # Cast timestamps explicitly to avoid silent string comparisons
    con.execute("""
    CREATE OR REPLACE VIEW v_chart AS
    SELECT
      stay_id,
      itemid,
      CAST(charttime AS TIMESTAMP) AS charttime,
      CAST(valuenum AS DOUBLE) AS valuenum
    FROM v_chart_raw
    WHERE charttime IS NOT NULL;
    """)
    con.execute("""
    CREATE OR REPLACE VIEW v_labs AS
    SELECT
      hadm_id,
      itemid,
      CAST(charttime AS TIMESTAMP) AS charttime,
      CAST(valuenum AS DOUBLE) AS valuenum
    FROM v_labs_raw
    WHERE charttime IS NOT NULL;
    """)

    # DEBUG sanity
    print("\n[DEBUG] Chart join rows:")
    print(con.execute("""
    SELECT COUNT(*) AS n_chart_rows
    FROM v_chart ce JOIN v_keys k ON ce.stay_id = k.stay_id;
    """).fetchdf())

    # Helper for last chart value
    def last_chart_value(out_table: str, itemids: list[int], out_col: str) -> None:
        ids = ", ".join(str(x) for x in itemids)
        con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        WITH w AS (
          SELECT
            k.stay_id,
            ce.charttime,
            ce.valuenum,
            ROW_NUMBER() OVER (PARTITION BY k.stay_id ORDER BY ce.charttime DESC) AS rn
          FROM v_keys k
          JOIN v_chart ce ON ce.stay_id = k.stay_id
          WHERE ce.itemid IN ({ids})
            AND ce.valuenum IS NOT NULL
            AND ce.charttime >  k.t0_time - INTERVAL '{LOOKBACK_HOURS}' HOUR
            AND ce.charttime <= k.t0_time + INTERVAL '{VITALS_GRACE_HOURS}' HOUR
        )
        SELECT stay_id, valuenum AS {out_col}
        FROM w
        WHERE rn = 1;
        """)

    last_chart_value("v_fix_hr", [HR_ITEMID], "hr_fix")
    last_chart_value("v_fix_rr", [RR_ITEMID], "rr_fix")
    last_chart_value("v_fix_spo2", [SPO2_ITEMID], "spo2_fix")
    last_chart_value("v_fix_nibp_sys", [NIBP_SYS], "nibp_sys_fix")
    last_chart_value("v_fix_nibp_dia", [NIBP_DIA], "nibp_dia_fix")
    last_chart_value("v_fix_nibp_mean", [NIBP_MEAN], "nibp_mean_fix")
    last_chart_value("v_fix_art_sys", ART_SYS_CANDS, "art_sys")
    last_chart_value("v_fix_art_dia", ART_DIA_CANDS, "art_dia")
    last_chart_value("v_fix_art_mean", ART_MEAN_CANDS, "art_mean")

    # Temperature with conversion
    con.execute(f"""
    CREATE OR REPLACE TABLE v_fix_temp AS
    WITH w AS (
      SELECT
        k.stay_id,
        ce.charttime,
        CASE
          WHEN ce.itemid = {TEMP_C_ITEMID} THEN ce.valuenum
          WHEN ce.itemid = {TEMP_F_ITEMID} THEN (ce.valuenum - 32.0) * 5.0 / 9.0
          ELSE NULL
        END AS temp_c_fix,
        ROW_NUMBER() OVER (PARTITION BY k.stay_id ORDER BY ce.charttime DESC) AS rn
      FROM v_keys k
      JOIN v_chart ce ON ce.stay_id = k.stay_id
      WHERE ce.itemid IN ({TEMP_C_ITEMID}, {TEMP_F_ITEMID})
        AND ce.valuenum IS NOT NULL
        AND ce.charttime >  k.t0_time - INTERVAL '{LOOKBACK_HOURS}' HOUR
        AND ce.charttime <= k.t0_time + INTERVAL '{VITALS_GRACE_HOURS}' HOUR
    )
    SELECT stay_id, temp_c_fix
    FROM w
    WHERE rn = 1;
    """)

    # BP prefer arterial
    con.execute("""
    CREATE OR REPLACE TABLE v_fix_bp AS
    SELECT
      k.stay_id,
      COALESCE(am.art_sys, ns.nibp_sys_fix) AS bp_sys_fix,
      COALESCE(ad.art_dia, nd.nibp_dia_fix) AS bp_dia_fix,
      COALESCE(amean.art_mean, nm.nibp_mean_fix) AS bp_mean_fix
    FROM v_keys k
    LEFT JOIN v_fix_art_sys am ON k.stay_id = am.stay_id
    LEFT JOIN v_fix_art_dia ad ON k.stay_id = ad.stay_id
    LEFT JOIN v_fix_art_mean amean ON k.stay_id = amean.stay_id
    LEFT JOIN v_fix_nibp_sys ns ON k.stay_id = ns.stay_id
    LEFT JOIN v_fix_nibp_dia nd ON k.stay_id = nd.stay_id
    LEFT JOIN v_fix_nibp_mean nm ON k.stay_id = nm.stay_id;
    """)

    # WBC aggregate
    wbc_ids = ", ".join(str(x) for x in WBC_ITEMIDS)
    con.execute(f"""
    CREATE OR REPLACE TABLE v_fix_wbc AS
    WITH w AS (
      SELECT
        k.stay_id,
        le.charttime,
        le.valuenum AS wbc_fix,
        ROW_NUMBER() OVER (PARTITION BY k.stay_id ORDER BY le.charttime DESC) AS rn
      FROM v_keys k
      JOIN v_labs le ON le.hadm_id = k.hadm_id
      WHERE le.itemid IN ({wbc_ids})
        AND le.valuenum IS NOT NULL
        AND le.charttime >  k.t0_time - INTERVAL '{LOOKBACK_HOURS}' HOUR
        AND le.charttime <= k.t0_time + INTERVAL '{LABS_GRACE_HOURS}' HOUR
    )
    SELECT stay_id, wbc_fix
    FROM w
    WHERE rn = 1;
    """)

    # Assemble with EXCLUDE to ensure replacement (critical!)
    con.execute("""
    CREATE OR REPLACE TABLE analytic_rbc_v1_fixed AS
    SELECT
      a.* EXCLUDE (hr, rr, spo2, temp_c, nibp_sys, nibp_dia, nibp_mean, wbc, has_nibp_mean),

      COALESCE(h.hr_fix, a.hr) AS hr,
      COALESCE(r.rr_fix, a.rr) AS rr,
      COALESCE(s.spo2_fix, a.spo2) AS spo2,
      COALESCE(t.temp_c_fix, a.temp_c) AS temp_c,

      COALESCE(bp.bp_sys_fix, a.nibp_sys) AS nibp_sys,
      COALESCE(bp.bp_dia_fix, a.nibp_dia) AS nibp_dia,
      COALESCE(bp.bp_mean_fix, a.nibp_mean) AS nibp_mean,

      COALESCE(w.wbc_fix, a.wbc) AS wbc,

      CASE WHEN COALESCE(bp.bp_mean_fix, a.nibp_mean) IS NULL THEN 0 ELSE 1 END AS has_nibp_mean

    FROM v_ana a
    LEFT JOIN v_fix_hr h ON a.stay_id = h.stay_id
    LEFT JOIN v_fix_rr r ON a.stay_id = r.stay_id
    LEFT JOIN v_fix_spo2 s ON a.stay_id = s.stay_id
    LEFT JOIN v_fix_temp t ON a.stay_id = t.stay_id
    LEFT JOIN v_fix_bp bp ON a.stay_id = bp.stay_id
    LEFT JOIN v_fix_wbc w ON a.stay_id = w.stay_id;
    """)

    # Quick missingness check (key)
    print("\n[DEBUG] Missingness AFTER fix (key columns):")
    print(con.execute("""
    SELECT
      1 - COUNT(hr)::DOUBLE/COUNT(*) AS hr_null,
      1 - COUNT(rr)::DOUBLE/COUNT(*) AS rr_null,
      1 - COUNT(spo2)::DOUBLE/COUNT(*) AS spo2_null,
      1 - COUNT(temp_c)::DOUBLE/COUNT(*) AS temp_null,
      1 - COUNT(nibp_mean)::DOUBLE/COUNT(*) AS map_null,
      1 - COUNT(wbc)::DOUBLE/COUNT(*) AS wbc_null
    FROM analytic_rbc_v1_fixed;
    """).fetchdf())

    con.execute(f"""
    COPY analytic_rbc_v1_fixed
    TO '{_csv(OUT_PATH)}'
    (FORMAT PARQUET);
    """)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
