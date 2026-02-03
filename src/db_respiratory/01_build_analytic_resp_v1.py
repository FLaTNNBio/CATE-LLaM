#
# Build a trial-like analytic dataset for HFNC vs NIV treatments.
# - Time-zero (t0): first respiratory support (HFNC or NIV) within 24h of ICU admission
# - Treatment: HFNC vs NIV at t0 (HFNC=1, NIV=0)
# - Baseline covariates: last observation in [-6h, +1h] for vitals, [-6h, +1h] for labs
# - Eligibility: all patients with either HFNC or NIV within 24h of ICU admission
# - Outcome: intubation within 48h of t0
#
# Output: analytic/analytic_resp_v1.parquet (1 row per stay_id)

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
D_ITEMS = ICU_DIR / "d_items.csv.gz"
PROCEDUREEVENTS = ICU_DIR / "procedureevents.csv.gz"  # opzionale ma utile per intubazione

# Design parameters
ELIG_WITHIN_HOURS = 24
ASSIGN_WINDOW_HOURS = 2
OUTCOME_WINDOW_HOURS = 24

BASELINE_LOOKBACK_HOURS = 6
TREAT_WINDOW_HOURS = 4

# Grace periods for baseline extraction
VITALS_GRACE_HOURS = 0.5
LABS_GRACE_HOURS = 1
ANTHRO_GRACE_HOURS = 6

# O2 Delivery Device itemid
O2_DELIVERY_DEV = 226732 # Oxygen delivery device (chart)
INTUBATION_PROC = 224385 # Intubation (procedure)
TRACH_TIME = [225448, 226237] # Tracheostomy (procedure)

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
    "o2_flow": [223834],  # O2 Flow
    "flow_rate_l_min": [224691],  # Flow Rate (L/min)
    "hr": [220045],         # Heart Rate
    "rr": [220210],         # Respiratory Rate
    "spo2": [220277],       # O2 saturation pulseoxymetry
    "fio2" : [223835],      # FiO2
    "sofa" : [227428],      # SOFA Score
    "temp_c": [223762],     # Temperature Celsius
    "nibp_sys": [220179],   # Non Invasive Blood Pressure systolic
    "nibp_dia": [220180],   # Non Invasive Blood Pressure diastolic
    "nibp_mean": [220181],  # Non Invasive Blood Pressure mean
    "art_sys" : [220050, 225309], # Arterial Blood Pressure systolic
    "art_dia" : [220051, 225310], # Arterial Blood Pressure diastolic
    "art_mean" : [220052, 225312], # Arterial Blood Pressure mean
    "gcs_eye":   [220739],  # Glasgow Coma Scale eye opening
    "gcs_verbal":[223900],  # GCS verbal
    "gcs_motor": [223901]   # GCS motor
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
    "pCO2": 50818,             # pCO2
    "ph": 50820                # pH
}

# Baseline inputs in inputevents (vasopressors)
INPUT_ITEMIDS = {
    "Vasopressors": [221289,   # 221289,229617 : Epinephrine
                     229617,
                     221906,   # 221906 : Norepinephrine
                     222315,   # Vasopressin
                     221662,   # Dopamine 221662
                     ]
}


# Anthropometrics conversion (empirically: OMR exports are often lb / inches)
LB_TO_KG = 0.453592
IN_TO_CM = 2.54

OUT_PATH = ANALYTIC_DIR / "analytic_resp_v1.parquet"


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

    con.execute(f"""
    CREATE OR REPLACE VIEW v_ditems AS
    SELECT * FROM read_csv_auto('{_csv(D_ITEMS)}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_proc AS
    SELECT * FROM read_csv_auto('{_csv(PROCEDUREEVENTS)}', union_by_name=true);
    """)

    print("- Loaded core views: ")
    print("  v_cohort, v_demog, v_general_raw, v_labs, \n"
          "  v_chart, v_adm, v_ditems, v_proc")

    # -------------------------
    # 1) Find Respiratory Items
    # -------------------------

    con.execute(f"""
        -- ---- 0) Respiratory support events: ONLY O2 Delivery Device(s)
        CREATE OR REPLACE TABLE v_resp_support AS
        WITH ce AS (
          SELECT
            c.subject_id,
            c.hadm_id,
            c.stay_id,
            c.intime,
            ch.charttime,
            lower(trim(ch.value)) AS value_lc
          FROM v_cohort c
          JOIN v_chart ch
            ON ch.stay_id = c.stay_id
          WHERE ch.itemid = {O2_DELIVERY_DEV}               -- O2 Delivery Device(s)
            AND ch.charttime IS NOT NULL
            AND ch.value IS NOT NULL
            AND ch.charttime >= c.intime
            AND ch.charttime <  c.intime + INTERVAL '24' HOUR
        ),
        classified AS (
          SELECT
            *,
            CASE
              -- HFNC patterns
              WHEN value_lc LIKE '%high flow%' OR value_lc LIKE '%high-flow%' OR value_lc LIKE '%hfnc%'
                THEN 'HFNC'
        
              -- NIV patterns (mask-based non-invasive ventilation)
              WHEN value_lc LIKE '%bipap%' OR value_lc LIKE '%bi-pap%'
                OR value_lc LIKE '%cpap%'
                OR value_lc LIKE '%niv%'
                OR value_lc LIKE '%mask%'
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


    # -------------------------
    # 2) Define time-zero (t0)
    # -------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE v_t0 AS
        WITH first_any AS (
          SELECT
            stay_id, hadm_id, subject_id, intime,
            MIN(charttime) AS t_first
          FROM v_resp_support
          GROUP BY 1,2,3,4
        ),
        win AS (
          SELECT
            r.stay_id, r.hadm_id, r.subject_id, r.intime,
            f.t_first,
            r.support_type
          FROM v_resp_support r
          JOIN first_any f
            ON r.stay_id = f.stay_id
          WHERE r.charttime >= f.t_first
            AND r.charttime <  f.t_first + INTERVAL '2' HOUR
        ),
        counts AS (
          SELECT
            stay_id, hadm_id, subject_id, intime,
            MIN(t_first) AS t0_time,
            SUM(CASE WHEN support_type='HFNC' THEN 1 ELSE 0 END) AS n_hfnc_2h,
            SUM(CASE WHEN support_type='NIV'  THEN 1 ELSE 0 END) AS n_niv_2h
          FROM win
          GROUP BY 1,2,3,4
        ),
        labeled AS (
          SELECT
            *,
            CASE
              WHEN n_hfnc_2h > n_niv_2h THEN 'HFNC'
              WHEN n_niv_2h  > n_hfnc_2h THEN 'NIV'
              ELSE NULL
            END AS t0_support
          FROM counts
        )
        SELECT
          stay_id, hadm_id, subject_id, intime,
          t0_time,
          t0_support,
          n_hfnc_2h,
          n_niv_2h
        FROM labeled
        WHERE t0_support IS NOT NULL;
    """)

    print(f"- Generated v_t0")

    # -------------------------
    # 3) Assign treatment (HFNC vs NIV) HFNC=1, NIV=0;
    # -------------------------
    trach_time_tuple = "(" + ", ".join(str(x) for x in TRACH_TIME) + ")"
    con.execute(f"""
        CREATE OR REPLACE TABLE v_treat AS
        SELECT
          stay_id,
          CASE WHEN t0_support='HFNC' THEN 1 ELSE 0 END AS t_hfnc
        FROM v_t0;
    """)

    # -------------------------
    # 3a) Outcome from admissions - mortality (secondary outcome)
    # -------------------------
    con.execute("""
       CREATE OR REPLACE TABLE v_outcome_adm AS
       SELECT
         t0.stay_id,
         a.dischtime,
         a.deathtime,
         CAST(a.hospital_expire_flag AS INTEGER) AS y_hosp_mort
       FROM v_t0 t0
       LEFT JOIN v_adm a
         ON t0.hadm_id = a.hadm_id;
       """)

    print(f"- Generated v_outcome_adm table")

    # -------------------------
    # 3b) Outcome from procedureevents - intubation within 48h after t0 (primary outcome)
    # -------------------------

    con.execute(f"""
        -- ---- 3) Outcome: intubation within 48h after t0 (procedureevents)
        -- Use explicit itemid(s) from your d_items list:
        --   224385 = Intubation (procedureevents)
        -- Optionally track tracheostomy:
        --   225448 = Percutaneous Tracheostomy
        --   226237 = Open Tracheostomy
        CREATE OR REPLACE TABLE v_outcome_proc AS
        WITH proc_in_window AS (
          SELECT
            t0.stay_id,
            MIN(p.starttime) FILTER (WHERE p.itemid = {INTUBATION_PROC}) AS intub_time,
            MIN(p.starttime) FILTER (WHERE p.itemid IN {trach_time_tuple}) AS trach_time
          FROM v_t0 t0
          LEFT JOIN v_proc p
            ON p.stay_id = t0.stay_id
           AND p.starttime IS NOT NULL
           AND p.starttime >= t0.t0_time
           AND p.starttime <  t0.t0_time + INTERVAL '48' HOUR
          GROUP BY t0.stay_id
        )
        SELECT
          stay_id,
          CASE WHEN intub_time IS NULL THEN 0 ELSE 1 END AS y_intub_48h,
          intub_time,
          CASE WHEN trach_time IS NULL THEN 0 ELSE 1 END AS y_trach_48h,
          trach_time
        FROM proc_in_window;
    """)

    print(f"- Generated v_treat: view with treatment assignment")

    # 3c Exclude Pre-Intubated Patients
    con.execute(f"""
    CREATE OR REPLACE TABLE v_exclude_preintub AS
        SELECT
          t0.stay_id
        FROM v_t0 t0
        JOIN v_proc p
          ON p.stay_id = t0.stay_id
        WHERE p.itemid = {INTUBATION_PROC}
          AND p.starttime IS NOT NULL
          AND p.starttime < t0.t0_time;
    """)

    # -------------------------
    # 3d) Input Events - Vasopressors (before t0)
    # -------------------------

    input_itemid_tuple = "(" + ", ".join(str(x) for x in INPUT_ITEMIDS["Vasopressors"]) + ")"
    con.execute(f"""
        CREATE OR REPLACE TABLE v_inputs_vaso AS
        SELECT
            t0.stay_id,
            COUNT(*) AS n_vaso_pre_t0
        FROM v_t0 t0
        JOIN v_inputs ie
            ON ie.stay_id = t0.stay_id
        WHERE ie.itemid IN {input_itemid_tuple}
            AND ie.starttime IS NOT NULL
            AND ie.starttime < t0.t0_time
        GROUP BY t0.stay_id;
    """)

    # -------------------------
    # 4) Baseline covariates in [-6h, 0h] relative to t0 (last value pre t0)
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
        CREATE OR REPLACE TABLE analytic_resp_v1 AS
        SELECT
          -- IDs & timing
          t0.subject_id,
          t0.hadm_id,
          t0.stay_id,
          t0.intime,
          t0.t0_time,
          t0.t0_support,
          t0.n_hfnc_2h,
          t0.n_niv_2h,
        
          -- design params
          {ELIG_WITHIN_HOURS}::INTEGER AS elig_within_hours,
          {ASSIGN_WINDOW_HOURS}::INTEGER AS assign_window_hours,
          {OUTCOME_WINDOW_HOURS}::INTEGER AS outcome_window_hours,
        
          -- treatment
          tr.t_hfnc,
        
          -- outcomes
          op.y_intub_48h,
          op.intub_time,
          op.y_trach_48h,
          op.trach_time,
          oa.y_hosp_mort,
          oa.dischtime,
          oa.deathtime,
        
          -- demographics
          d.age,
          d.gender,
          d.race,
        
          -- anthropometrics
          a.admission_weight_kg,
          a.height_cm,
          a.bmi,
        
          -- vitals
          v.hr,
          v.rr,
          v.spo2,
          v.o2_flow,
          v.flow_rate_l_min,
          temp.temp_c_fix AS temp_c,
        
          -- keep raw nibp + art as separate columns if you want
          v.nibp_sys,
          v.nibp_dia,
          v.nibp_mean,
          v.art_sys,
          v.art_dia,
          v.art_mean,
          v.gcs_eye,
          v.gcs_verbal,
          v.gcs_motor,
    
          -- preferred BP (arterial if available else nibp)
          COALESCE(v.art_sys,  v.nibp_sys)  AS bp_sys,
          COALESCE(v.art_dia,  v.nibp_dia)  AS bp_dia,
          COALESCE(v.art_mean, v.nibp_mean) AS bp_mean,
        
          -- labs
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
          labs.pCO2,
          labs.ph,
          
          -- inputs
            COALESCE(i.n_vaso_pre_t0, 0) AS n_vaso_pre_t0,
        
          -- missingness indicators
          CASE WHEN labs.lactate     IS NULL THEN 0 ELSE 1 END AS has_lactate,
          CASE WHEN labs.inr         IS NULL THEN 0 ELSE 1 END AS has_inr,
          CASE WHEN labs.ptt         IS NULL THEN 0 ELSE 1 END AS has_ptt,
          CASE WHEN labs.creatinine  IS NULL THEN 0 ELSE 1 END AS has_creatinine,
          CASE WHEN labs.platelets   IS NULL THEN 0 ELSE 1 END AS has_platelets,
          CASE WHEN v.nibp_mean      IS NULL THEN 0 ELSE 1 END AS has_nibp_mean,
          CASE WHEN v.hr             IS NULL THEN 0 ELSE 1 END AS has_hr
        FROM v_t0 t0
        JOIN v_treat tr ON t0.stay_id = tr.stay_id
        LEFT JOIN v_outcome_proc op ON t0.stay_id = op.stay_id
        LEFT JOIN v_outcome_adm  oa ON t0.stay_id = oa.stay_id
        LEFT JOIN v_demog d ON t0.stay_id = d.stay_id
        LEFT JOIN v_anthro a ON t0.stay_id = a.stay_id
        LEFT JOIN v_base_vitals v ON t0.stay_id = v.stay_id
        LEFT JOIN v_base_temp temp ON t0.stay_id = temp.stay_id
        LEFT JOIN v_base_labs labs ON t0.stay_id = labs.stay_id
        LEFT JOIN v_inputs_vaso i ON t0.stay_id = i.stay_id
        LEFT JOIN v_base_wbc wbc ON t0.stay_id = wbc.stay_id
        LEFT JOIN v_exclude_preintub x ON t0.stay_id = x.stay_id WHERE x.stay_id IS NULL;
    """)

    print(f"- Assembled final analytic table: analytic_resp_v1")

    drop = ["nibp_sys", "nibp_dia", "nibp_mean",
            "art_sys", "art_dia", "art_mean"]
    # Drop sys, art, mean columns that were coalesced into bp_sys, bp_dia, bp_mean
    for c in drop:
        con.execute(f"""
            ALTER TABLE analytic_resp_v1
            DROP COLUMN IF EXISTS {c}
        """)

    # Rename bp_sys, bp_dia, bp_mean to nibp_sys, nibp_dia, nibp_mean for consistency and compatibility
    con.execute("""
        ALTER TABLE analytic_resp_v1
            RENAME COLUMN bp_sys TO nibp_sys
    """)
    con.execute("""
        ALTER TABLE analytic_resp_v1
            RENAME COLUMN bp_dia TO nibp_dia
    """)
    con.execute("""
        ALTER TABLE analytic_resp_v1
            RENAME COLUMN bp_mean TO nibp_mean;
    """)

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

    # -------------------------
    # 7) Sanity checks + export
    # -------------------------
    print("\nSanity checks:")
    print("\nRows (analytic_rbc_v1):")
    print(con.execute("SELECT COUNT(*) AS n FROM analytic_resp_v1;").fetchdf())

    # 5) Sanity checks: avoid median(), avoid ::DOUBLE casts
    print("\nQuick anthropometrics percentiles (metric):")
    print(con.execute("""
                      SELECT quantile(admission_weight_kg, 0.5) AS w_p50,
                             quantile(height_cm, 0.5)           AS h_p50,
                             quantile(bmi, 0.5)                 AS bmi_p50
                      FROM analytic_resp_v1;
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
                      FROM analytic_resp_v1;
                      """).fetchdf())

    # -----------------
    # Export final analytic dataset
    # -----------------
    con.execute(f"""
    COPY analytic_resp_v1
    TO '{_csv(OUT_PATH)}'
    (FORMAT PARQUET);
    """)
    print(f"\nWrote: {OUT_PATH}")


    # Secondary check:
    print("\n Secondary check of dataset:")

    print("\n Treatment counts (HFNC vs NIV):")
    print(con.execute(f"""
    SELECT t0_support, COUNT(*) FROM v_t0 GROUP BY 1;
    """).fetchdf())

    print("\n Overall intubation rate within 48h:")
    print(con.execute("""
          SELECT AVG(y_intub_48h)::DOUBLE AS intub_rate_48h
          FROM v_outcome_proc;
    """).fetchdf())


if __name__ == "__main__":
    main()
