#
# Post-fix for analytic_rbc_v1:
# - add grace periods to reduce missingness (charting delays)
# - temperature: Celsius + Fahrenheit with conversion to Celsius
# - BP: prefer arterial BP when available, fallback to NIBP
# - WBC: aggregate multiple itemids
# - anthropometrics: wider grace (does not change quickly)
#
# Input:  analytic/analytic_rbc_v1.parquet
# Output: analytic/analytic_rbc_v1_fixed.parquet

import duckdb
from src.config import HOSP_DIR, ICU_DIR, INTERMEDIATE_DIR, ANALYTIC_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = ANALYTIC_DIR / "analytic_rbc_v1.parquet"
OUT_PATH = ANALYTIC_DIR / "analytic_rbc_v1_fixed.parquet"

LABEVENTS = HOSP_DIR / "labevents.csv.gz"
CHARTEVENTS = ICU_DIR / "chartevents.csv.gz"

# Windows
LOOKBACK_HOURS = 6

# Grace periods
VITALS_GRACE_HOURS = 1        # HR/RR/SpO2/BP/Temp often charted slightly after observation
LABS_GRACE_HOURS = 1          # conservative; you can set to 0 if you prefer
ANTHRO_GRACE_HOURS = 6        # weight/height are quasi-invariant; allow wider grace

# Temperature itemids
TEMP_C_ITEMID = 223762  # Temperature Celsius
TEMP_F_ITEMID = 223761  # Temperature Fahrenheit

# Vitals itemids
HR_ITEMID = 220045
RR_ITEMID = 220210
SPO2_ITEMID = 220277

# Non-invasive BP itemids
NIBP_SYS = 220179
NIBP_DIA = 220180
NIBP_MEAN = 220181

# Arterial BP itemids (commonly used in ICU)
# Some sites use 220050/051/052 (Arterial Blood Pressure ...), others use 225309/310/312 (ART BP ...)
ART_SYS_CANDS = [220050, 225309]
ART_DIA_CANDS = [220051, 225310]
ART_MEAN_CANDS = [220052, 225312]

# WBC itemids (aggregate to improve coverage)
WBC_ITEMIDS = [51300, 51301, 51516]  # adjust if you discover additional canonical ids in your d_labitems

# Helper constants for Fahrenheit->Celsius conversion
def f_to_c_sql(expr: str) -> str:
    return f"(({expr} - 32.0) * 5.0 / 9.0)"


def _csv(path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    ANALYTIC_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='8GB';")

    # Load analytic dataset (already filtered to eligible cohort)
    con.execute(f"""
    CREATE OR REPLACE VIEW v_ana AS
    SELECT *
    FROM read_parquet('{_csv(IN_PATH)}');
    """)

    # Load source event tables
    con.execute(f"""
    CREATE OR REPLACE VIEW v_chart AS
    SELECT * FROM read_csv_auto('{_csv(CHARTEVENTS)}', union_by_name=true);
    """)
    con.execute(f"""
    CREATE OR REPLACE VIEW v_labs AS
    SELECT * FROM read_csv_auto('{_csv(LABEVENTS)}', union_by_name=true);
    """)

    # Key table: ids + t0_time + hadm_id + stay_id
    con.execute("""
    CREATE OR REPLACE VIEW v_keys AS
    SELECT stay_id, hadm_id, t0_time
    FROM v_ana;
    """)

    # -------------------------
    # VITALS with grace period
    # window: (t0 - LOOKBACK) to (t0 + VITALS_GRACE)
    # last value in that window
    # -------------------------
    def last_chart_value(table_name: str, itemid_sql: str, colname: str) -> None:
        con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH w AS (
          SELECT
            k.stay_id,
            ce.charttime,
            ce.valuenum,
            ROW_NUMBER() OVER (
              PARTITION BY k.stay_id
              ORDER BY ce.charttime DESC
            ) AS rn
          FROM v_keys k
          JOIN v_chart ce
            ON ce.stay_id = k.stay_id
          WHERE ce.itemid IN ({itemid_sql})
            AND ce.charttime IS NOT NULL
            AND ce.valuenum IS NOT NULL
            AND ce.charttime >  k.t0_time - INTERVAL '{LOOKBACK_HOURS}' HOUR
            AND ce.charttime <= k.t0_time + INTERVAL '{VITALS_GRACE_HOURS}' HOUR
        )
        SELECT stay_id, valuenum AS {colname}
        FROM w
        WHERE rn = 1;
        """)

    # HR/RR/SpO2
    last_chart_value("v_fix_hr", str(HR_ITEMID), "hr_fix")
    last_chart_value("v_fix_rr", str(RR_ITEMID), "rr_fix")
    last_chart_value("v_fix_spo2", str(SPO2_ITEMID), "spo2_fix")

    # Temperature: Celsius + Fahrenheit (converted) with grace
    # We'll compute a unified "temp_c_fix" by converting Fahrenheit values to Celsius before ranking.
    con.execute(f"""
    CREATE OR REPLACE TABLE v_fix_temp AS
    WITH w AS (
      SELECT
        k.stay_id,
        ce.charttime,
        CASE
          WHEN ce.itemid = {TEMP_C_ITEMID} THEN ce.valuenum
          WHEN ce.itemid = {TEMP_F_ITEMID} THEN {f_to_c_sql("ce.valuenum")}
          ELSE NULL
        END AS temp_c,
        ROW_NUMBER() OVER (
          PARTITION BY k.stay_id
          ORDER BY ce.charttime DESC
        ) AS rn
      FROM v_keys k
      JOIN v_chart ce
        ON ce.stay_id = k.stay_id
      WHERE ce.itemid IN ({TEMP_C_ITEMID}, {TEMP_F_ITEMID})
        AND ce.charttime IS NOT NULL
        AND ce.valuenum IS NOT NULL
        AND ce.charttime >  k.t0_time - INTERVAL '{LOOKBACK_HOURS}' HOUR
        AND ce.charttime <= k.t0_time + INTERVAL '{VITALS_GRACE_HOURS}' HOUR
    )
    SELECT stay_id, temp_c AS temp_c_fix
    FROM w
    WHERE rn = 1;
    """)

    # Blood pressure: prefer arterial if present, otherwise NIBP.
    # We compute arterial and NIBP separately, then coalesce.
    art_sys_sql = ", ".join(str(x) for x in ART_SYS_CANDS)
    art_dia_sql = ", ".join(str(x) for x in ART_DIA_CANDS)
    art_mean_sql = ", ".join(str(x) for x in ART_MEAN_CANDS)

    last_chart_value("v_fix_art_sys", art_sys_sql, "art_sys")
    last_chart_value("v_fix_art_dia", art_dia_sql, "art_dia")
    last_chart_value("v_fix_art_mean", art_mean_sql, "art_mean")

    last_chart_value("v_fix_nibp_sys", str(NIBP_SYS), "nibp_sys_fix")
    last_chart_value("v_fix_nibp_dia", str(NIBP_DIA), "nibp_dia_fix")
    last_chart_value("v_fix_nibp_mean", str(NIBP_MEAN), "nibp_mean_fix")

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

    # -------------------------
    # LABS: WBC aggregation with optional grace
    # window: (t0 - LOOKBACK) to (t0 + LABS_GRACE)
    # Note: labs are by hadm_id
    # -------------------------
    wbc_sql = ", ".join(str(x) for x in WBC_ITEMIDS)
    con.execute(f"""
    CREATE OR REPLACE TABLE v_fix_wbc AS
    WITH w AS (
      SELECT
        k.stay_id,
        le.charttime,
        le.valuenum AS wbc,
        ROW_NUMBER() OVER (
          PARTITION BY k.stay_id
          ORDER BY le.charttime DESC
        ) AS rn
      FROM v_keys k
      JOIN v_labs le
        ON le.hadm_id = k.hadm_id
      WHERE le.itemid IN ({wbc_sql})
        AND le.charttime IS NOT NULL
        AND le.valuenum IS NOT NULL
        AND le.charttime >  k.t0_time - INTERVAL '{LOOKBACK_HOURS}' HOUR
        AND le.charttime <= k.t0_time + INTERVAL '{LABS_GRACE_HOURS}' HOUR
    )
    SELECT stay_id, wbc AS wbc_fix
    FROM w
    WHERE rn = 1;
    """)

    # -------------------------
    # ANTHROPOMETRICS: wider grace (invariant)
    # If missing in v_ana, optionally try to "borrow" from a wider window.
    # BUT: since your anthropometrics came from OMR/general and are not time-stamped here,
    # we cannot re-extract them from source tables without an OMR table.
    # So here we only keep them as-is. We *can*, however, keep the option open:
    # if you have an "omr" parquet with chartdate/seq, we can fill them.
    #
    # For now: just keep existing admission_weight_kg/height_cm/bmi.
    # -------------------------

    # -------------------------
    # Assemble fixed dataset
    # Strategy: replace a subset of columns with *_fix when available
    # -------------------------
    con.execute("""
    CREATE OR REPLACE TABLE analytic_rbc_v1_fixed AS
    SELECT
        a.* EXCLUDE (hr, rr, spo2, temp_c, nibp_sys, nibp_dia, nibp_mean, wbc),
  
      -- Replace vitals with fixed versions when available
      COALESCE(h.hr_fix, a.hr) AS hr,
      COALESCE(r.rr_fix, a.rr) AS rr,
      COALESCE(s.spo2_fix, a.spo2) AS spo2,
      COALESCE(t.temp_c_fix, a.temp_c) AS temp_c,

      -- Replace BP with preferred arterial->nibp when available
      COALESCE(bp.bp_sys_fix, a.nibp_sys) AS nibp_sys,
      COALESCE(bp.bp_dia_fix, a.nibp_dia) AS nibp_dia,
      COALESCE(bp.bp_mean_fix, a.nibp_mean) AS nibp_mean,

      -- Replace WBC
      COALESCE(w.wbc_fix, a.wbc) AS wbc

    FROM v_ana a
    LEFT JOIN v_fix_hr h ON a.stay_id = h.stay_id
    LEFT JOIN v_fix_rr r ON a.stay_id = r.stay_id
    LEFT JOIN v_fix_spo2 s ON a.stay_id = s.stay_id
    LEFT JOIN v_fix_temp t ON a.stay_id = t.stay_id
    LEFT JOIN v_fix_bp bp ON a.stay_id = bp.stay_id
    LEFT JOIN v_fix_wbc w ON a.stay_id = w.stay_id
    ;
    """)

    # Recompute has_* that depend on modified columns
    con.execute("""
    CREATE OR REPLACE TABLE analytic_rbc_v1_fixed2 AS
    SELECT
      -- keep all columns EXCEPT old has_nibp_mean (we recompute)
      * EXCLUDE (has_nibp_mean),

      CASE WHEN nibp_mean IS NULL THEN 0 ELSE 1 END AS has_nibp_mean
    FROM analytic_rbc_v1_fixed;
    """)

    # Sanity prints
    print("\nRowsows (fixed):")
    print(con.execute("SELECT COUNT(*) AS n FROM analytic_rbc_v1_fixed2;").fetchdf())

    print("\nMissingness (key columns) BEFORE vs AFTER fix:")
    print(con.execute("""
    WITH before AS (
      SELECT
        1 - COUNT(hr)::DOUBLE/COUNT(*) AS hr_null,
        1 - COUNT(rr)::DOUBLE/COUNT(*) AS rr_null,
        1 - COUNT(spo2)::DOUBLE/COUNT(*) AS spo2_null,
        1 - COUNT(temp_c)::DOUBLE/COUNT(*) AS temp_null,
        1 - COUNT(nibp_mean)::DOUBLE/COUNT(*) AS map_null,
        1 - COUNT(wbc)::DOUBLE/COUNT(*) AS wbc_null
      FROM v_ana
    ),
    after AS (
      SELECT
        1 - COUNT(hr)::DOUBLE/COUNT(*) AS hr_null,
        1 - COUNT(rr)::DOUBLE/COUNT(*) AS rr_null,
        1 - COUNT(spo2)::DOUBLE/COUNT(*) AS spo2_null,
        1 - COUNT(temp_c)::DOUBLE/COUNT(*) AS temp_null,
        1 - COUNT(nibp_mean)::DOUBLE/COUNT(*) AS map_null,
        1 - COUNT(wbc)::DOUBLE/COUNT(*) AS wbc_null
      FROM analytic_rbc_v1_fixed2
    )
    SELECT
      before.hr_null AS hr_before, after.hr_null AS hr_after,
      before.rr_null AS rr_before, after.rr_null AS rr_after,
      before.spo2_null AS spo2_before, after.spo2_null AS spo2_after,
      before.temp_null AS temp_before, after.temp_null AS temp_after,
      before.map_null AS map_before, after.map_null AS map_after,
      before.wbc_null AS wbc_before, after.wbc_null AS wbc_after
    FROM before, after;
    """).fetchdf())

    # Write parquet
    con.execute(f"""
    COPY analytic_rbc_v1_fixed2
    TO '{_csv(OUT_PATH)}'
    (FORMAT PARQUET);
    """)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
