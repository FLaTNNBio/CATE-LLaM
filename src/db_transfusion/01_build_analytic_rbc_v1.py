#
# Build a trial-like analytic dataset for early RBC transfusion.
# - Time-zero (t0): first hemoglobin < threshold within 48h of ICU intime
# - Treatment: any RBC inputevent within 3h after t0
# - Baseline covariates: last observation in [t0-6h, t0]
# - Outcome: hospital mortality from admissions.hospital_expire_flag
#
# Output: analytic/analytic_rbc_v1.parquet (1 row per stay_id)

import duckdb
from src.config import HOSP_DIR, ICU_DIR, INTERMEDIATE_DIR, ANALYTIC_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Core ICU cohort with stay_id, hadm_id, subject_id, intime
COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"

# Demographics (stay_id, age, gender, race) produced previously
DEMOG_PATH = INTERMEDIATE_DIR / "demographics.parquet"

# OMR-derived anthropometrics produced previously (stay_id, admission_weight_kg, height_cm, bmi)
# NOTE: this file (as you observed) often stores lb/in; we deterministically convert below.
GENERAL_PATH = INTERMEDIATE_DIR / "general.parquet"

# Raw event tables
LABEVENTS = HOSP_DIR / "labevents.csv.gz"
CHARTEVENTS = ICU_DIR / "chartevents.csv.gz"
INPUTEVENTS = ICU_DIR / "inputevents.csv.gz"
ADMISSIONS = HOSP_DIR / "admissions.csv.gz"

# Design parameters
MAX_HB_THRESHOLD = 8.5
MIN_HB_THRESHOLD = 7.5  # for sanity checks
ELIG_WITHIN_HOURS = 48
BASELINE_LOOKBACK_HOURS = 6
TREAT_WINDOW_HOURS = 3

# Itemids (update here if you chose different canonical ids)
# Hb for eligibility + baseline Hb
HB_ITEMID = 51222  # Hemoglobin (lab)

# Early RBC treatment itemids (aggregate)
RBC_ITEMIDS = [225168, 226368, 227070]

# Baseline vitals in chartevents
VITAL_ITEMIDS = {
    "hr": 220045,         # Heart Rate
    "rr": 220210,         # Respiratory Rate
    "spo2": 220277,       # O2 saturation pulseoxymetry
    "temp_c": 223762,     # Temperature Celsius
    "nibp_sys": 220179,   # Non Invasive Blood Pressure systolic
    "nibp_dia": 220180,   # Non Invasive Blood Pressure diastolic
    "nibp_mean": 220181,  # Non Invasive Blood Pressure mean
}

# Baseline labs in labevents (core + extended)
LAB_ITEMIDS = {
    "hemoglobin": 51222,       # Hemoglobin
    "platelets": 51265,        # Platelet Count
    "wbc": 51516,              # WBC
    "creatinine": 50912,       # Creatinine
    "bicarbonate": 50882,      # Bicarbonate
    "sodium": 50983,           # Sodium
    "potassium": 50971,        # Potassium
    "lactate": 50813,          # Lactate
    "inr": 51237,              # INR(PT)
    "ptt": 51275,              # PTT
    "glucose": 50809,          # Glucose (blood gas)
    "albumin": 50862,          # Albumin
}

# Anthropometrics conversion (empirically: OMR exports are often lb / inches)
LB_TO_KG = 0.453592
IN_TO_CM = 2.54

OUT_PATH = ANALYTIC_DIR / "analytic_rbc_v1.parquet"


def _csv(path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    ANALYTIC_DIR.mkdir(parents=True, exist_ok=True)

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
    CREATE OR REPLACE VIEW v_demog AS
    SELECT stay_id, age, gender, race
    FROM read_parquet('{_csv(DEMOG_PATH)}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_general_raw AS
    SELECT stay_id, admission_weight_kg, height_cm, bmi
    FROM read_parquet('{_csv(GENERAL_PATH)}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_labs AS
    SELECT * FROM read_csv_auto('{_csv(LABEVENTS)}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_chart AS
    SELECT * FROM read_csv_auto('{_csv(CHARTEVENTS)}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_inputs AS
    SELECT * FROM read_csv_auto('{_csv(INPUTEVENTS)}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_adm AS
    SELECT
      hadm_id,
      dischtime,
      deathtime,
      hospital_expire_flag
    FROM read_csv_auto('{_csv(ADMISSIONS)}', union_by_name=true);
    """)

    # -------------------------
    # 1) Define time-zero (t0)
    # -------------------------
    con.execute(f"""
    CREATE OR REPLACE TABLE v_t0 AS
    WITH hb AS (
      SELECT
        c.subject_id,
        c.hadm_id,
        c.stay_id,
        c.intime,
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
        AND le.valuenum >= {MIN_HB_THRESHOLD}
        AND le.valuenum <= {MAX_HB_THRESHOLD}
    ),
    ranked AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime) AS rn
      FROM hb
    )
    SELECT
      subject_id,
      hadm_id,
      stay_id,
      intime,
      charttime AS t0_time,
      hb AS t0_hb
    FROM ranked
    WHERE rn = 1;
    """)

    # -------------------------
    # 2) Assign treatment (RBC within 3h after t0)
    # -------------------------
    rbc_list = ", ".join(str(x) for x in RBC_ITEMIDS)

    con.execute(f"""
    CREATE OR REPLACE TABLE v_treat AS
    WITH rbc_in_window AS (
      SELECT
        t0.stay_id,
        MIN(ie.starttime) AS rbc_first_time,
        -- "units" proxy: sum(amount) if present, else 1 per event
        SUM(CASE
              WHEN TRY_CAST(ie.amount AS DOUBLE) IS NOT NULL THEN TRY_CAST(ie.amount AS DOUBLE)
              ELSE 1
            END) AS rbc_units_24h
      FROM v_t0 t0
      JOIN v_inputs ie
        ON ie.stay_id = t0.stay_id
      WHERE ie.itemid IN ({rbc_list})
        AND ie.starttime IS NOT NULL
        AND ie.starttime >= t0.t0_time
        AND ie.starttime <  t0.t0_time + INTERVAL '{TREAT_WINDOW_HOURS}' HOUR
      GROUP BY t0.stay_id
    )
    SELECT
      t0.stay_id,
      CASE WHEN r.stay_id IS NULL THEN 0 ELSE 1 END AS t_rbc_3h,
      r.rbc_first_time,
      r.rbc_units_24h
    FROM v_t0 t0
    LEFT JOIN rbc_in_window r
      ON t0.stay_id = r.stay_id;
    """)

    # -------------------------
    # 3) Baseline covariates in [-6h, 0h] relative to t0 (last value pre t0)
    # -------------------------
    # Helper: build a "last value" table for chartevents vitals
    def last_value_chartevents(colname: str, itemid: int) -> None:
        con.execute(f"""
        CREATE OR REPLACE TABLE v_base_{colname} AS
        WITH w AS (
          SELECT
            t0.stay_id,
            ce.charttime,
            ce.valuenum,
            ROW_NUMBER() OVER (
              PARTITION BY t0.stay_id
              ORDER BY ce.charttime DESC
            ) AS rn
          FROM v_t0 t0
          JOIN v_chart ce
            ON ce.stay_id = t0.stay_id
          WHERE ce.itemid = {itemid}
            AND ce.charttime IS NOT NULL
            AND ce.valuenum IS NOT NULL
            AND ce.charttime >  t0.t0_time - INTERVAL '{BASELINE_LOOKBACK_HOURS}' HOUR
            AND ce.charttime <= t0.t0_time
        )
        SELECT stay_id, valuenum AS {colname}
        FROM w
        WHERE rn = 1;
        """)

    for k, iid in VITAL_ITEMIDS.items():
        last_value_chartevents(k, iid)

    # Helper: build a "last value" table for labevents baseline labs
    def last_value_labevents(colname: str, itemid: int) -> None:
        con.execute(f"""
        CREATE OR REPLACE TABLE v_base_lab_{colname} AS
        WITH w AS (
          SELECT
            t0.stay_id,
            le.charttime,
            le.valuenum,
            ROW_NUMBER() OVER (
              PARTITION BY t0.stay_id
              ORDER BY le.charttime DESC
            ) AS rn
          FROM v_t0 t0
          JOIN v_labs le
            ON le.hadm_id = t0.hadm_id
          WHERE le.itemid = {itemid}
            AND le.charttime IS NOT NULL
            AND le.valuenum IS NOT NULL
            AND le.charttime >  t0.t0_time - INTERVAL '{BASELINE_LOOKBACK_HOURS}' HOUR
            AND le.charttime <= t0.t0_time
        )
        SELECT stay_id, valuenum AS {colname}
        FROM w
        WHERE rn = 1;
        """)

    for k, iid in LAB_ITEMIDS.items():
        last_value_labevents(k, iid)

    # has_hb_prior: any Hb measurement strictly before t0 in the baseline window
    con.execute(f"""
    CREATE OR REPLACE TABLE v_has_hb_prior AS
    SELECT
      t0.stay_id,
      CASE WHEN EXISTS (
        SELECT 1
        FROM v_labs le
        WHERE le.hadm_id = t0.hadm_id
          AND le.itemid = {HB_ITEMID}
          AND le.charttime IS NOT NULL
          AND le.valuenum IS NOT NULL
          AND le.charttime >  t0.t0_time - INTERVAL '{BASELINE_LOOKBACK_HOURS}' HOUR
          AND le.charttime <  t0.t0_time
      ) THEN 1 ELSE 0 END AS has_hb_prior
    FROM v_t0 t0;
    """)

    # -------------------------
    # 4) Outcome from admissions
    # -------------------------
    con.execute("""
    CREATE OR REPLACE TABLE v_outcome AS
    SELECT
      t0.stay_id,
      a.dischtime,
      a.deathtime,
      CAST(a.hospital_expire_flag AS INTEGER) AS y_hosp_mort
    FROM v_t0 t0
    LEFT JOIN v_adm a
      ON t0.hadm_id = a.hadm_id;
    """)

    # -------------------------
    # 5) Anthropometrics (convert lb/in -> kg/cm and recompute BMI)
    # -------------------------
    con.execute(f"""
    CREATE OR REPLACE TABLE v_anthro AS
    SELECT
      g.stay_id,

      CASE WHEN g.admission_weight_kg IS NULL THEN NULL
           ELSE g.admission_weight_kg * {LB_TO_KG}
      END AS admission_weight_kg,

      CASE WHEN g.height_cm IS NULL THEN NULL
           ELSE g.height_cm * {IN_TO_CM}
      END AS height_cm,

      CASE
        WHEN g.admission_weight_kg IS NULL OR g.height_cm IS NULL THEN NULL
        WHEN g.height_cm * {IN_TO_CM} <= 0 THEN NULL
        ELSE (g.admission_weight_kg * {LB_TO_KG}) / POWER((g.height_cm * {IN_TO_CM}) / 100.0, 2)
      END AS bmi
    FROM v_general_raw g;
    """)

    # -------------------------
    # 6) Assemble final analytic dataset
    # -------------------------
    con.execute(f"""
    CREATE OR REPLACE TABLE analytic_rbc_v1 AS
    SELECT
      -- IDs & timing
      t0.subject_id,
      t0.hadm_id,
      t0.stay_id,
      t0.intime,
      t0.t0_time,
      t0.t0_hb,

      -- design params (explicit constants)
      {MIN_HB_THRESHOLD}::DOUBLE AS min_elig_hb_threshold,
      {MAX_HB_THRESHOLD}::DOUBLE AS max_elig_hb_threshold,
      {ELIG_WITHIN_HOURS}::INTEGER AS elig_within_hours,
      {TREAT_WINDOW_HOURS}::INTEGER AS treat_window_hours,

      -- treatment
      tr.t_rbc_3h,
      tr.rbc_first_time,
      tr.rbc_units_24h,

      -- outcomes
      o.y_hosp_mort,
      o.dischtime,
      o.deathtime,

      -- demographics
      d.age,
      d.gender,
      d.race,

      -- anthropometrics
      a.admission_weight_kg,
      a.height_cm,
      a.bmi,

      -- vitals (baseline last in [-6h,0h])
      hr.hr,
      rr.rr,
      spo2.spo2,
      temp_c.temp_c,
      nibp_sys.nibp_sys,
      nibp_dia.nibp_dia,
      nibp_mean.nibp_mean,

      -- labs (baseline last in [-6h,0h])
      hemoglobin.hemoglobin,
      platelets.platelets,
      wbc.wbc,
      creatinine.creatinine,
      bicarbonate.bicarbonate,
      sodium.sodium,
      potassium.potassium,
      lactate.lactate,
      inr.inr,
      ptt.ptt,
      glucose.glucose,
      albumin.albumin,

      -- essential missingness indicators (MNAR proxies)
      CASE WHEN lactate.lactate IS NULL THEN 0 ELSE 1 END AS has_lactate,
      CASE WHEN inr.inr IS NULL THEN 0 ELSE 1 END AS has_inr,
      CASE WHEN ptt.ptt IS NULL THEN 0 ELSE 1 END AS has_ptt,
      CASE WHEN creatinine.creatinine IS NULL THEN 0 ELSE 1 END AS has_creatinine,
      CASE WHEN platelets.platelets IS NULL THEN 0 ELSE 1 END AS has_platelets,
      CASE WHEN nibp_mean.nibp_mean IS NULL THEN 0 ELSE 1 END AS has_nibp_mean,
      hbp.has_hb_prior

    FROM v_t0 t0
    JOIN v_treat tr ON t0.stay_id = tr.stay_id
    LEFT JOIN v_outcome o ON t0.stay_id = o.stay_id
    LEFT JOIN v_demog d ON t0.stay_id = d.stay_id
    LEFT JOIN v_anthro a ON t0.stay_id = a.stay_id

    LEFT JOIN v_base_hr hr ON t0.stay_id = hr.stay_id
    LEFT JOIN v_base_rr rr ON t0.stay_id = rr.stay_id
    LEFT JOIN v_base_spo2 spo2 ON t0.stay_id = spo2.stay_id
    LEFT JOIN v_base_temp_c temp_c ON t0.stay_id = temp_c.stay_id
    LEFT JOIN v_base_nibp_sys nibp_sys ON t0.stay_id = nibp_sys.stay_id
    LEFT JOIN v_base_nibp_dia nibp_dia ON t0.stay_id = nibp_dia.stay_id
    LEFT JOIN v_base_nibp_mean nibp_mean ON t0.stay_id = nibp_mean.stay_id

    LEFT JOIN v_base_lab_hemoglobin hemoglobin ON t0.stay_id = hemoglobin.stay_id
    LEFT JOIN v_base_lab_platelets platelets ON t0.stay_id = platelets.stay_id
    LEFT JOIN v_base_lab_wbc wbc ON t0.stay_id = wbc.stay_id
    LEFT JOIN v_base_lab_creatinine creatinine ON t0.stay_id = creatinine.stay_id
    LEFT JOIN v_base_lab_bicarbonate bicarbonate ON t0.stay_id = bicarbonate.stay_id
    LEFT JOIN v_base_lab_sodium sodium ON t0.stay_id = sodium.stay_id
    LEFT JOIN v_base_lab_potassium potassium ON t0.stay_id = potassium.stay_id
    LEFT JOIN v_base_lab_lactate lactate ON t0.stay_id = lactate.stay_id
    LEFT JOIN v_base_lab_inr inr ON t0.stay_id = inr.stay_id
    LEFT JOIN v_base_lab_ptt ptt ON t0.stay_id = ptt.stay_id
    LEFT JOIN v_base_lab_glucose glucose ON t0.stay_id = glucose.stay_id
    LEFT JOIN v_base_lab_albumin albumin ON t0.stay_id = albumin.stay_id

    LEFT JOIN v_has_hb_prior hbp ON t0.stay_id = hbp.stay_id
    ;
    """)

    # -------------------------
    # 7) Sanity checks + export
    # -------------------------
    print("\nRows (analytic_rbc_v1):")
    print(con.execute("SELECT COUNT(*) AS n FROM analytic_rbc_v1;").fetchdf())

    print("\nTreatment prevalence:")
    print(con.execute("""
    SELECT AVG(t_rbc_3h)::DOUBLE AS p_treated,
           SUM(CASE WHEN t_rbc_3h=1 THEN 1 ELSE 0 END) AS n_treated,
           SUM(CASE WHEN t_rbc_3h=0 THEN 1 ELSE 0 END) AS n_control
    FROM analytic_rbc_v1;
    """).fetchdf())

    print("\nQuick anthropometrics percentiles (metric):")
    print(con.execute("""
    SELECT
      quantile_cont(admission_weight_kg, 0.50) AS w_p50,
      quantile_cont(height_cm, 0.50) AS h_p50,
      quantile_cont(bmi, 0.50) AS bmi_p50
    FROM analytic_rbc_v1;
    """).fetchdf())

    print("\nMissingness (selected):")
    print(con.execute("""
    SELECT
      1 - COUNT(nibp_mean)::DOUBLE/COUNT(*) AS nibp_mean_null_frac,
      1 - COUNT(lactate)::DOUBLE/COUNT(*) AS lactate_null_frac,
      1 - COUNT(inr)::DOUBLE/COUNT(*) AS inr_null_frac,
      1 - COUNT(albumin)::DOUBLE/COUNT(*) AS albumin_null_frac,
      1 - COUNT(glucose)::DOUBLE/COUNT(*) AS glucose_null_frac
    FROM analytic_rbc_v1;
    """).fetchdf())

    con.execute(f"""
    COPY analytic_rbc_v1
    TO '{_csv(OUT_PATH)}'
    (FORMAT PARQUET);
    """)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
