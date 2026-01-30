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
TREAT_WINDOW_HOURS = 4

# Grace periods for baseline extraction
VITALS_GRACE_HOURS = 1
LABS_GRACE_HOURS = 1
ANTHRO_GRACE_HOURS = 6

# Itemids (update here if you chose different canonical ids)
# Hb for eligibility + baseline Hb
HB_ITEMID = 51222  # Hemoglobin (lab)

# Early RBC treatment itemids (aggregate)
RBC_ITEMIDS = [225168, 226368, 227070]  # Packed Red Blood Cells

# Temperature itemids
TEMP_C_ITEMID = 223762
TEMP_F_ITEMID = 223761

# Arterial BP itemids (preferred over NIBP)
ART_SYS_CANDS = [220050, 225309]
ART_DIA_CANDS = [220051, 225310]
ART_MEAN_CANDS = [220052, 225312]

# WBC itemids (aggregate for better coverage)
WBC_ITEMIDS = [51300, 51301, 51516]


# Baseline vitals in chartevents
VITAL_ITEMIDS = {
    "hr": [220045],         # Heart Rate
    "rr": [220210],         # Respiratory Rate
    "spo2": [220277],       # O2 saturation pulseoxymetry
    "temp_c": [223762],     # Temperature Celsius
    "nibp_sys": [220179],   # Non Invasive Blood Pressure systolic
    "nibp_dia": [220180],   # Non Invasive Blood Pressure diastolic
    "nibp_mean": [220181],  # Non Invasive Blood Pressure mean
    "art_sys" : [220050, 225309], # Arterial Blood Pressure systolic
    "art_dia" : [220051, 225310], # Arterial Blood Pressure diastolic
    "art_mean" : [220052, 225312] # Arterial Blood Pressure mean
}

# Baseline labs in labevents (core + extended)
LAB_ITEMIDS = {
    "hemoglobin": 51222,       # Hemoglobin
    "platelets": 51265,        # Platelet Count
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

OUT_PATH = ANALYTIC_DIR / "analytic_rbc_v1_f.parquet"


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

    print(f"- Generated v_t0")

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

    print(f"- Generated v_treat: view with treatment assignment")

    # -------------------------
    # 3) Baseline covariates in [-6h, 0h] relative to t0 (last value pre t0)
    # -------------------------
    # Replace the per-vital loop with a single query that builds v_base_vitals
    # (temp handled separately as you already do)

    # Flatten itemids and build itemid -> column mapping
    vital_itemid_to_col = []
    all_vital_itemids = []

    for col, ids in VITAL_ITEMIDS.items():
        if col == "temp_c":
            continue
        for itemid in ids:
            vital_itemid_to_col.append((itemid, col))
            all_vital_itemids.append(itemid)

    all_ids_sql = ", ".join(str(x) for x in sorted(set(all_vital_itemids)))

    # For each itemid, pick the most recent value in the window, then pivot to columns
    select_cols_sql = ",\n         ".join(
        f"MAX(CASE WHEN itemid = {itemid} THEN valuenum END) AS {col}"
        for itemid, col in vital_itemid_to_col
    )

    con.execute(f"""
    CREATE OR REPLACE TABLE v_base_vitals AS
    WITH ranked AS (
      SELECT
        t0.stay_id,
        ce.itemid,
        ce.valuenum,
        ROW_NUMBER() OVER (
          PARTITION BY t0.stay_id, ce.itemid
          ORDER BY ce.charttime DESC
        ) AS rn
      FROM v_t0 t0
      JOIN v_chart ce
        ON ce.stay_id = t0.stay_id
      WHERE ce.itemid IN ({all_ids_sql})
        AND ce.charttime IS NOT NULL
        AND ce.valuenum IS NOT NULL
        AND ce.charttime >  t0.t0_time - INTERVAL '{BASELINE_LOOKBACK_HOURS}' HOUR
        AND ce.charttime <= t0.t0_time + INTERVAL '{VITALS_GRACE_HOURS}' HOUR
    ),
    last_per_item AS (
      SELECT stay_id, itemid, valuenum
      FROM ranked
      WHERE rn = 1
    )
    SELECT
      stay_id,
      {select_cols_sql}
    FROM last_per_item
    GROUP BY stay_id;
    """)

    print("- Generated baseline vitals table: v_base_vitals")

    # Temperature with conversion
    con.execute(f"""
        CREATE OR REPLACE TABLE v_base_temp AS
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
          FROM v_t0 k
          JOIN v_chart ce ON ce.stay_id = k.stay_id
          WHERE ce.itemid IN ({TEMP_C_ITEMID}, {TEMP_F_ITEMID})
            AND ce.valuenum IS NOT NULL
            AND ce.charttime >  k.t0_time - INTERVAL '{BASELINE_LOOKBACK_HOURS}' HOUR
            AND ce.charttime <= k.t0_time + INTERVAL '{VITALS_GRACE_HOURS}' HOUR
        )
        SELECT stay_id, temp_c_fix
        FROM w
        WHERE rn = 1;
        """)

    print(f"- Generated baseline vital table: v_base_temp_c")

    # LABS

    # WBC aggregate
    wbc_ids = ", ".join(str(x) for x in WBC_ITEMIDS)
    con.execute(f"""
        CREATE OR REPLACE TABLE v_base_wbc AS
        WITH w AS (
          SELECT
            k.stay_id,
            le.charttime,
            le.valuenum AS wbc_fix,
            ROW_NUMBER() OVER (PARTITION BY k.stay_id ORDER BY le.charttime DESC) AS rn
          FROM v_t0 k
          JOIN v_labs le ON le.hadm_id = k.hadm_id
          WHERE le.itemid IN ({wbc_ids})
            AND le.valuenum IS NOT NULL
            AND le.charttime >  k.t0_time - INTERVAL '{BASELINE_LOOKBACK_HOURS}' HOUR
            AND le.charttime <= k.t0_time + INTERVAL '{LABS_GRACE_HOURS}' HOUR
        )
        SELECT stay_id, wbc_fix
        FROM w
        WHERE rn = 1;
        """)

    print(f"- Generated baseline lab table: v_base_wbc with aggregate itemids")

    # Baseline labs in labevents: single query + pivot (includes WBC aggregation separately)
    # Creates: v_base_labs (one row per stay_id, one column per lab in LAB_ITEMIDS)
    # Keeps: v_base_wbc as its own aggregate across multiple itemids.

    # Build itemid -> column mapping (exclude wbc here if you keep v_base_wbc)
    lab_itemid_to_col = []
    all_lab_itemids = []

    for col, itemid in LAB_ITEMIDS.items():
        if col == "wbc":
            continue
        lab_itemid_to_col.append((int(itemid), col))
        all_lab_itemids.append(int(itemid))

    all_lab_ids_sql = ", ".join(str(x) for x in sorted(set(all_lab_itemids)))

    select_lab_cols_sql = ",\n         ".join(
        f"MAX(CASE WHEN itemid = {itemid} THEN valuenum END) AS {col}"
        for itemid, col in lab_itemid_to_col
    )

    con.execute(f"""
    CREATE OR REPLACE TABLE v_base_labs AS
    WITH ranked AS (
      SELECT
        t0.stay_id,
        le.itemid,
        le.valuenum,
        ROW_NUMBER() OVER (
          PARTITION BY t0.stay_id, le.itemid
          ORDER BY le.charttime DESC
        ) AS rn
      FROM v_t0 t0
      JOIN v_labs le
        ON le.hadm_id = t0.hadm_id
      WHERE le.itemid IN ({all_lab_ids_sql})
        AND le.charttime IS NOT NULL
        AND le.valuenum IS NOT NULL
        AND le.charttime >  t0.t0_time - INTERVAL '{BASELINE_LOOKBACK_HOURS}' HOUR
        AND le.charttime <= t0.t0_time + INTERVAL '{LABS_GRACE_HOURS}' HOUR
    ),
    last_per_item AS (
      SELECT stay_id, itemid, valuenum
      FROM ranked
      WHERE rn = 1
    )
    SELECT
      stay_id,
      {select_lab_cols_sql}
    FROM last_per_item
    GROUP BY stay_id;
    """)

    print("- Generated baseline labs table: v_base_labs")

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

    print(f"- Generated has_hb_prior table")

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

    print(f"- Generated v_outcome table")

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

    print(f"- Generated v_anthro table with conversions")

    print("\n- All baseline tables generated.")
    print("- Assembling final analytic dataset...")

    # -------------------------
    # 6) Assemble final analytic dataset
    # -------------------------
    con.execute(f"""
    CREATE OR REPLACE TABLE analytic_rbc_v1_f AS
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

      -- vitals (baseline last in window)
      v.hr,
      v.rr,
      v.spo2,
      temp.temp_c_fix AS temp_c,

      -- keep raw nibp + art as separate columns if you want
      v.nibp_sys,
      v.nibp_dia,
      v.nibp_mean,
      v.art_sys,
      v.art_dia,
      v.art_mean,

      -- preferred BP (arterial if available else nibp)
      COALESCE(v.art_sys,  v.nibp_sys)  AS bp_sys,
      COALESCE(v.art_dia,  v.nibp_dia)  AS bp_dia,
      COALESCE(v.art_mean, v.nibp_mean) AS bp_mean,

      -- labs (baseline last in window)
      labs.hemoglobin,
      labs.platelets,
      wbc.wbc_fix AS wbc,
      labs.creatinine,
      labs.bicarbonate,
      labs.sodium,
      labs.potassium,
      labs.lactate,
      labs.inr,
      labs.ptt,
      labs.glucose,
      labs.albumin,

      -- essential missingness indicators (MNAR proxies)
      CASE WHEN labs.lactate IS NULL THEN 0 ELSE 1 END AS has_lactate,
      CASE WHEN labs.inr IS NULL THEN 0 ELSE 1 END AS has_inr,
      CASE WHEN labs.ptt IS NULL THEN 0 ELSE 1 END AS has_ptt,
      CASE WHEN labs.creatinine IS NULL THEN 0 ELSE 1 END AS has_creatinine,
      CASE WHEN labs.platelets IS NULL THEN 0 ELSE 1 END AS has_platelets,
      CASE WHEN v.nibp_mean IS NULL THEN 0 ELSE 1 END AS has_nibp_mean,
      hbp.has_hb_prior

    FROM v_t0 t0
    JOIN v_treat tr ON t0.stay_id = tr.stay_id
    LEFT JOIN v_outcome o ON t0.stay_id = o.stay_id
    LEFT JOIN v_demog d ON t0.stay_id = d.stay_id
    LEFT JOIN v_anthro a ON t0.stay_id = a.stay_id

    LEFT JOIN v_base_vitals v ON t0.stay_id = v.stay_id
    LEFT JOIN v_base_temp temp ON t0.stay_id = temp.stay_id

    LEFT JOIN v_base_labs labs ON t0.stay_id = labs.stay_id
    LEFT JOIN v_base_wbc wbc ON t0.stay_id = wbc.stay_id

    LEFT JOIN v_has_hb_prior hbp ON t0.stay_id = hbp.stay_id
    ;
    """)

    print(f"- Assembled final analytic table: analytic_rbc_v1_f")

    # Drop intermediate views to free up space
    con.execute("""
    DROP VIEW IF EXISTS v_cohort;
    DROP VIEW IF EXISTS v_demog;
    DROP VIEW IF EXISTS v_general_raw;
    DROP VIEW IF EXISTS v_labs;
    DROP VIEW IF EXISTS v_chart;
    DROP VIEW IF EXISTS v_inputs;
    DROP VIEW IF EXISTS v_adm;
    """)

    drop = [ "nibp_sys", "nibp_dia", "nibp_mean",
             "art_sys", "art_dia", "art_mean" ]
    # Drop sys, art, mean columns that were coalesced into bp_sys, bp_dia, bp_mean
    for c in drop:
        con.execute(f"""
        ALTER TABLE analytic_rbc_v1_f
        DROP COLUMN IF EXISTS {c}
        """)

    # Rename bp_sys, bp_dia, bp_mean to nibp_sys, nibp_dia, nibp_mean for consistency and compatibility
    con.execute("""
    ALTER TABLE analytic_rbc_v1_f
    RENAME COLUMN bp_sys TO nibp_sys
    """)
    con.execute("""
    ALTER TABLE analytic_rbc_v1_f
    RENAME COLUMN bp_dia TO nibp_dia
        """)
    con.execute("""
    ALTER TABLE analytic_rbc_v1_f
    RENAME COLUMN bp_mean TO nibp_mean;
    """)

    # -------------------------
    # 7) Sanity checks + export
    # -------------------------
    print("\\nSanity checks:")
    print("\\nRows (analytic_rbc_v1):")
    print(con.execute("SELECT COUNT(*) AS n FROM analytic_rbc_v1_f;").fetchdf())

    # 5) Sanity checks: avoid median(), avoid ::DOUBLE casts
    print("\nQuick anthropometrics percentiles (metric):")
    print(con.execute("""
                      SELECT quantile(admission_weight_kg, 0.5) AS w_p50,
                             quantile(height_cm, 0.5)           AS h_p50,
                             quantile(bmi, 0.5)                 AS bmi_p50
                      FROM analytic_rbc_v1_f;
                      """).fetchdf())

    print("\\nMissingness (selected):")
    print(con.execute("""
                      SELECT 1.0 - COUNT(nibp_mean)::DOUBLE / COUNT(*) AS nibp_mean_null_frac,
                             1.0 - COUNT(lactate)::DOUBLE / COUNT(*)   AS lactate_null_frac,
                             1.0 - COUNT(inr)::DOUBLE / COUNT(*)       AS inr_null_frac,
                             1.0 - COUNT(albumin)::DOUBLE / COUNT(*)   AS albumin_null_frac,
                             1.0 - COUNT(glucose)::DOUBLE / COUNT(*)   AS glucose_null_frac,
                             1.0 - COUNT(hr)::DOUBLE / COUNT(*)        AS hr_null,
                             1.0 - COUNT(rr)::DOUBLE / COUNT(*)        AS rr_null,
                             1.0 - COUNT(spo2)::DOUBLE / COUNT(*)      AS spo2_null,
                             1.0 - COUNT(temp_c)::DOUBLE / COUNT(*)    AS temp_null,
                             1.0 - COUNT(wbc)::DOUBLE / COUNT(*)       AS wbc_null
                      FROM analytic_rbc_v1_f;
                      """).fetchdf())

    # -----------------
    # Export final analytic dataset
    # -----------------
    con.execute(f"""
    COPY analytic_rbc_v1_f
    TO '{_csv(OUT_PATH)}'
    (FORMAT PARQUET);
    """)
    print(f"\nWrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
