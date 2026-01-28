import duckdb
from src.config import ANALYTIC_DIR, HOSP_DIR, ICU_DIR, INTERMEDIATE_DIR


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


    print("\n[DEBUG] Keys count:")
    print(con.execute("SELECT COUNT(*) AS n FROM v_keys;").fetchdf())

    print("\n[DEBUG] Chart events for these stays (any):")
    print(con.execute("""
    SELECT
      COUNT(*) AS n_chart_rows,
      COUNT(DISTINCT ce.stay_id) AS n_chart_stays
    FROM v_chart ce
    JOIN v_keys k ON ce.stay_id = k.stay_id;
    """).fetchdf())


    print("\n[DEBUG] HR rows within [-6h, +1h] for these stays:")
    print(con.execute(f"""
    SELECT
      COUNT(*) AS n_hr_rows,
      COUNT(DISTINCT k.stay_id) AS n_stays_with_hr
    FROM v_keys k
    JOIN v_chart ce ON ce.stay_id = k.stay_id
    WHERE ce.itemid = {HR_ITEMID}
      AND ce.charttime IS NOT NULL
      AND ce.valuenum IS NOT NULL
      AND ce.charttime >  k.t0_time - INTERVAL '{LOOKBACK_HOURS}' HOUR
      AND ce.charttime <= k.t0_time + INTERVAL '{VITALS_GRACE_HOURS}' HOUR
    """).fetchdf())

    print("\n[DEBUG] t0_time - intime distribution (hours):")
    print(con.execute("""
                      SELECT MIN(EXTRACT(EPOCH FROM (t0_time - intime)) / 3600.0)                      AS min_h,
                             quantile_cont(EXTRACT(EPOCH FROM (t0_time - intime)) / 3600.0, 0.50)      AS p50_h,
                             quantile_cont(EXTRACT(EPOCH FROM (t0_time - intime)) / 3600.0, 0.90)      AS p90_h,
                             SUM(CASE WHEN t0_time <= intime + INTERVAL '30' MINUTE THEN 1 ELSE 0 END) AS n_within_30m
                      FROM v_keys;
                      """).fetchdf())


if __name__ == "__main__":
    main()
