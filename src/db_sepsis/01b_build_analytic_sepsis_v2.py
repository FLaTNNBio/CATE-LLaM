# Corticosteroids in Sepsis target-trial emulation (MIMIC-IV) — MIT tables only
# Output: analytic/analytic_sepsis_steroids_mit_v2.parquet (1 row per stay_id)

import duckdb
from pathlib import Path
from dataclasses import dataclass

from src.config import MIT_MIMIC, ANALYTIC_DIR


# -------------------------
# Config
# -------------------------

@dataclass
class Params:
    enroll_hours: int = 24
    baseline_lookback_hours: int = 2
    grace_hours: int = 24
    exclude_steroids_before_hours: int = 10  # "more than 10 hrs before enrollment window" (supplement)
    hc_equiv_threshold_mg: float = 160.0     # hydrocortisone-equivalent mg in [0,24h]
    memory_limit: str = "8GB"


HC_EQ_MULT = {
    "hydrocortisone": 1.0,
    "prednisone": 4.0,
    "prednisolone": 4.0,
    "dexamethasone": 25.0,
    "methylprednisolone": 5.0,
}

def cleanup_previous_run(con: duckdb.DuckDBPyConnection) -> None:
    """
    Drop temp leftover objects from previous runs to ensure idempotency.
    - Drop any VIEW/TABLE/SEQUENCE starting with 'v_'.
    - Drop also working tables note (t0, treat, outcome, base_*, ecc.)
    in 'main' schema to avoid collisions.
    """
    # 1) drop TABLE/VIEW starting with v_*
    rows = con.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
        AND table_name LIKE 'v\\_%' ESCAPE '\\'
    """).fetchall()

    for name, ttype in rows:
        if ttype == "VIEW":
            con.execute(f'DROP VIEW IF EXISTS "{name}";')
        else:
            con.execute(f'DROP TABLE IF EXISTS "{name}";')

    # 2) drop of working tables “notes” (in main) to avoid collisions (t0, treat, ecc.)
    #    Includes base_* e qa_*
    rows2 = con.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND (
            table_name IN ('t0', 'treat', 'outcome', 'excl_steroids', 'steroid_events')
                OR table_name LIKE 'base\\_%' ESCAPE '\\'
                OR table_name LIKE 'qa\\_%' ESCAPE '\\'
                    OR table_name LIKE 'analytic\\_%' ESCAPE '\\'
            )
        """).fetchall()
    for name, ttype in rows2:
        if ttype == "VIEW":
            con.execute(f'DROP VIEW IF EXISTS "{name}";')
        else:
            con.execute(f'DROP TABLE IF EXISTS "{name}";')

    # 3) optional: drop possible temp views
    con.execute("DROP VIEW IF EXISTS _tmp_work_objects;")



def table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = ? AND table_name = ?
    LIMIT 1
    """
    return con.execute(q, [schema, table]).fetchone() is not None


def col_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str, col: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = ? AND table_name = ? AND column_name = ?
    LIMIT 1
    """
    return con.execute(q, [schema, table, col]).fetchone() is not None


def main(db_path: str, out_path: str, tmp_dir: str) -> None:
    p = Params()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute(f"SET temp_directory='{str(Path(tmp_dir))}';")
    con.execute(f"SET memory_limit='{p.memory_limit}';")

    # Cleanup temp objects from previous runs
    # to ensure idempotency (views, tables, ecc.)
    cleanup_previous_run(con)

    # -------------------------
    # Preconditions (fail fast)
    # -------------------------
    required = [
        ("mimiciv_icu", "icustays"),
        ("mimiciv_hosp", "admissions"),
        ("mimiciv_derived", "age"),
        ("mimiciv_derived", "sepsis3"),
        ("mimiciv_derived", "vitalsign"),
        ("mimiciv_derived", "chemistry"),
        ("mimiciv_derived", "complete_blood_count"),
        ("mimiciv_derived", "coagulation"),
        ("mimiciv_derived", "bg"),
        ("mimiciv_hosp", "prescriptions"),
        ("mimiciv_derived", "icustay_detail"),
        ("mimiciv_derived", "weight_durations"),
        ("mimiciv_derived", "height"),
        ("mimiciv_derived", "sofa"),
    ]
    missing = [(s, t) for (s, t) in required if not table_exists(con, s, t)]
    if missing:
        raise RuntimeError(
            "Mancano tabelle richieste nel DB (MIT concepts). "
            f"Assenti: {missing}. "
            "Importale/creale prima con mimic-code concepts_duckdb."
        )

    # Optional tables (ALT/AST/Bilirubin in enzyme)
    has_enzyme = table_exists(con, "mimiciv_derived", "enzyme")
    has_liver = table_exists(con, "mimiciv_derived", "liver_function")  # alcuni setup la chiamano così

    # -------------------------
    # 0) Base Views (From MIT tables)
    # -------------------------

    con.execute("""
    CREATE OR REPLACE VIEW v_icustays AS
    SELECT subject_id, hadm_id, stay_id, intime, outtime
    FROM mimiciv_icu.icustays
    WHERE intime IS NOT NULL;
    """)

    con.execute("""
    CREATE OR REPLACE VIEW v_adm AS
    SELECT hadm_id, dischtime, deathtime, hospital_expire_flag
    FROM mimiciv_hosp.admissions;
    """)

    # Age + demo (icustay_detail for gender/race; age in derived.age)
    con.execute("""
    CREATE OR REPLACE VIEW v_demog AS
    SELECT
      d.stay_id,
      a.age,
      d.gender,
      d.race
    FROM mimiciv_derived.icustay_detail d
    JOIN mimiciv_derived.age a
      ON a.subject_id = d.subject_id AND a.hadm_id = d.hadm_id;
    """)

    # Sepsis-3: use sepsis_time (GREATEST infection suspect and SOFA-time)
    con.execute("""
    CREATE OR REPLACE VIEW v_sepsis AS
    SELECT
      stay_id,
      GREATEST(suspected_infection_time, sofa_time) AS sepsis_time
    FROM mimiciv_derived.sepsis3
    WHERE sepsis3 = TRUE
      AND suspected_infection_time IS NOT NULL
      AND sofa_time IS NOT NULL;
    """)

    #MIT: weight in kg, height in cm) + BMI (first available)
    con.execute(
        """ CREATE OR REPLACE TABLE base_bmi 
        AS WITH weight_duration AS ( 
            SELECT stay_id, weight AS weight_kg, ROW_NUMBER() 
            OVER (PARTITION BY stay_id ORDER BY starttime) AS rn 
            FROM mimiciv_derived.weight_durations 
            WHERE weight IS NOT NULL 
        ), first_day AS ( 
            SELECT stay_id, weight AS weight_kg 
            FROM mimiciv_derived.first_day_weight 
            WHERE weight IS NOT NULL 
        ), height_data AS ( 
            SELECT stay_id, height AS height_cm, ROW_NUMBER() 
            OVER (PARTITION BY stay_id ORDER BY charttime) AS rn 
            FROM mimiciv_derived.height WHERE height IS NOT NULL 
        ), all_stays AS ( 
            SELECT DISTINCT stay_id 
            FROM mimiciv_icu.icustays 
            WHERE intime IS NOT NULL 
        ) 
        SELECT s.stay_id, 
            COALESCE(fd.weight_kg, wd.weight_kg) AS weight_kg, 
            h.height_cm, 
            CASE WHEN 
                COALESCE(fd.weight_kg, wd.weight_kg) IS NOT NULL 
                AND h.height_cm IS NOT NULL AND h.height_cm > 0 
            THEN COALESCE(fd.weight_kg, wd.weight_kg) / POWER(h.height_cm/100.0, 2) 
            ELSE NULL END AS bmi 
        FROM all_stays s 
        LEFT JOIN first_day fd ON fd.stay_id = s.stay_id 
        LEFT JOIN (
            SELECT stay_id, weight_kg 
            FROM weight_duration 
            WHERE rn=1
            ) wd ON wd.stay_id = s.stay_id 
        LEFT JOIN (SELECT stay_id, height_cm FROM height_data WHERE rn=1) h 
        ON h.stay_id = s.stay_id; 
    """)


    # -------------------------
    # 1) Time zero + eligibility (adult, first ICU stay, sepsis within 24h)
    # -------------------------

    con.execute(f"""
    CREATE OR REPLACE TABLE t0 AS
    WITH base AS (
      SELECT
        i.subject_id, i.hadm_id, i.stay_id,
        i.intime AS t0_time,
        i.intime + INTERVAL '{p.enroll_hours}' HOUR AS enroll_end
      FROM v_icustays i
    ),
    adult AS (
      SELECT b.*
      FROM base b
      JOIN v_demog d ON d.stay_id = b.stay_id
      WHERE d.age >= 18
    ),
    first_stay AS (
      SELECT *
      FROM (
        SELECT
          a.*,
          ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.t0_time) AS rn
        FROM adult a
      )
      WHERE rn = 1
    ),
    sepsis_in_window AS (
      SELECT
        f.*,
        s.sepsis_time
      FROM first_stay f
      JOIN v_sepsis s ON s.stay_id = f.stay_id
      WHERE s.sepsis_time >= f.t0_time
        AND s.sepsis_time <  f.enroll_end
    )
    SELECT * FROM sepsis_in_window;
    """)

    # -------------------------
    # 2) Steroids exposure from mimiciv_hosp.prescriptions (in-DB)
    #    - exclusion: steroid start < t0 - 10h
    #    - treatment: hc-equivalent cumulative mg in [t0, t0+24h] >= 160
    #    - control: 0 mg in [0,24h]
    #    NB: prescriptions è spesso il punto più “sporco” (dose free-text).
    #        qui facciamo parsing conservativo ma migliore di un LIKE nudo.
    # -------------------------

    con.execute("""
    CREATE OR REPLACE VIEW v_rx AS
    SELECT
      hadm_id,
      starttime,
      stoptime,
      drug,
      dose_val_rx,
      dose_unit_rx,
      doses_per_24_hrs,
      route
    FROM mimiciv_hosp.prescriptions
    WHERE starttime IS NOT NULL;
    """)

    con.execute("""
    CREATE OR REPLACE TABLE steroid_events AS
    WITH matched AS (
      SELECT
        t.stay_id,
        t.t0_time,
        t.enroll_end,
        r.starttime AS event_time,
        lower(coalesce(r.drug,'')) AS drug_lc,
        lower(coalesce(r.dose_unit_rx,'')) AS unit_lc,
        lower(coalesce(r.route,'')) AS route_lc,
        r.doses_per_24_hrs,
        r.dose_val_rx,
        CASE
          WHEN lower(r.drug) LIKE '%hydrocortisone%' THEN 'hydrocortisone'
          WHEN lower(r.drug) LIKE '%prednisone%' THEN 'prednisone'
          WHEN lower(r.drug) LIKE '%prednisolone%' THEN 'prednisolone'
          WHEN lower(r.drug) LIKE '%dexamethasone%' THEN 'dexamethasone'
          WHEN lower(r.drug) LIKE '%methylpred%' THEN 'methylprednisolone'
          ELSE NULL
        END AS drug_key
      FROM t0 t
      JOIN v_rx r
        ON r.hadm_id = t.hadm_id
    ),
    parsed AS (
      SELECT
        stay_id,
        t0_time,
        enroll_end,
        event_time,
        drug_key,
        unit_lc,
        route_lc,
        doses_per_24_hrs,
        -- parse numerico: "20", "20.5", "20 mg", ecc.
        CASE
          WHEN dose_val_rx IS NULL THEN NULL
          WHEN try_cast(dose_val_rx AS DOUBLE) IS NOT NULL THEN try_cast(dose_val_rx AS DOUBLE)
          ELSE try_cast(regexp_extract(dose_val_rx, '([0-9]+\\.?[0-9]*)', 1) AS DOUBLE)
        END AS dose_num
      FROM matched
      WHERE drug_key IS NOT NULL
    ),
    norm AS (
      SELECT
        stay_id,
        t0_time,
        enroll_end,
        event_time,
        drug_key,
        -- conversione unità (conservativa):
        CASE
          WHEN dose_num IS NULL THEN NULL
          WHEN unit_lc LIKE '%mg%' THEN dose_num
          WHEN unit_lc LIKE '%g%'  THEN dose_num * 1000.0
          WHEN unit_lc LIKE '%mcg%' OR unit_lc LIKE '%µg%' THEN dose_num / 1000.0
          ELSE NULL
        END AS dose_mg_single,
        doses_per_24_hrs,
        CASE
          WHEN dose_num IS NULL THEN 1
          WHEN (unit_lc LIKE '%mg%' OR unit_lc LIKE '%g%' OR unit_lc LIKE '%mcg%' OR unit_lc LIKE '%µg%') THEN 0
          ELSE 1
        END AS is_unparsed
      FROM parsed
    )
    SELECT
      stay_id,
      event_time,
      drug_key,
      dose_mg_single,
      -- se doses_per_24_hrs è presente, stimiamo “mg/day”; altrimenti usiamo dose singola come proxy
      CASE
        WHEN dose_mg_single IS NULL THEN NULL
        WHEN doses_per_24_hrs IS NOT NULL THEN dose_mg_single * doses_per_24_hrs
        ELSE dose_mg_single
      END AS dose_mg_per_day_proxy,
      is_unparsed
    FROM norm;
    """)

    # Exclusion: any steroid earlier than t0 - 10h
    con.execute(f"""
    CREATE OR REPLACE TABLE excl_steroids AS
    SELECT
      t.stay_id,
      MIN(s.event_time) AS first_steroid_time
    FROM t0 t
    JOIN steroid_events s
      ON s.stay_id = t.stay_id
    WHERE s.event_time < t.t0_time - INTERVAL '{p.exclude_steroids_before_hours}' HOUR
    GROUP BY t.stay_id;
    """)

    # Treatment assignment: cumulative hc-equivalent in [0,24h]
    con.execute(f"""
    CREATE OR REPLACE TABLE treat AS
    WITH win AS (
      SELECT
        t.stay_id,
        s.drug_key,
        s.dose_mg_per_day_proxy,
        s.is_unparsed,
        CASE s.drug_key
          WHEN 'hydrocortisone' THEN {HC_EQ_MULT["hydrocortisone"]}
          WHEN 'prednisone' THEN {HC_EQ_MULT["prednisone"]}
          WHEN 'prednisolone' THEN {HC_EQ_MULT["prednisolone"]}
          WHEN 'dexamethasone' THEN {HC_EQ_MULT["dexamethasone"]}
          WHEN 'methylprednisolone' THEN {HC_EQ_MULT["methylprednisolone"]}
          ELSE NULL
        END AS hc_mult
      FROM t0 t
      JOIN steroid_events s
        ON s.stay_id = t.stay_id
      WHERE s.event_time >= t.t0_time
        AND s.event_time <  t.enroll_end
    ),
    agg AS (
      SELECT
        stay_id,
        SUM(CASE WHEN dose_mg_per_day_proxy IS NOT NULL THEN dose_mg_per_day_proxy * hc_mult ELSE 0 END) AS hc_equiv_mg_0_24h,
        COUNT(*) AS steroid_events_0_24h,
        SUM(is_unparsed) AS steroid_unparsed_0_24h,
        SUM(CASE WHEN dose_mg_per_day_proxy IS NOT NULL THEN 1 ELSE 0 END) AS steroid_parsed_0_24h
      FROM win
      GROUP BY stay_id
    )
    SELECT
      t.stay_id,
      COALESCE(a.hc_equiv_mg_0_24h, 0.0) AS hc_equiv_mg_0_24h,
      COALESCE(a.steroid_events_0_24h, 0) AS steroid_events_0_24h,
      COALESCE(a.steroid_unparsed_0_24h, 0) AS steroid_unparsed_0_24h,
      COALESCE(a.steroid_parsed_0_24h, 0) AS steroid_parsed_0_24h,
      CASE
        WHEN COALESCE(a.hc_equiv_mg_0_24h, 0.0) >= {p.hc_equiv_threshold_mg} THEN 1
        WHEN COALESCE(a.hc_equiv_mg_0_24h, 0.0) = 0.0 THEN 0
        ELSE NULL
      END AS treat_steroid
    FROM t0 t
    LEFT JOIN agg a ON a.stay_id = t.stay_id;
    """)

    # -------------------------
    # 3) Outcome: 28-day in-hospital mortality proxy (deathtime <= t0+28d)
    # -------------------------
    con.execute("""
    CREATE OR REPLACE TABLE outcome AS
    SELECT
      t.stay_id,
      a.dischtime,
      a.deathtime,
      CAST(a.hospital_expire_flag AS INTEGER) AS y_hosp_mort,
      CASE
        WHEN a.deathtime IS NOT NULL
         AND a.deathtime <= t.t0_time + INTERVAL '28' DAY THEN 1
        ELSE 0
      END AS y_28d_mort_inhosp,
      CASE
        WHEN a.deathtime IS NOT NULL AND a.deathtime <= t.enroll_end THEN 1
        WHEN a.dischtime IS NOT NULL AND a.dischtime <= t.enroll_end THEN 1
        ELSE 0
      END AS died_or_disch_before_enroll_end
    FROM t0 t
    LEFT JOIN v_adm a ON a.hadm_id = t.hadm_id;
    """)

    # -------------------------
    # 4) Baseline covariates (paper feature set)
    #    Window: (t0-6h, t0+1h]
    # -------------------------

    # Vitals: heart_rate, resp_rate, spo2, temperature, sbp, mbp
    con.execute(f"""
    CREATE OR REPLACE TABLE base_vitals AS
    WITH w AS (
      SELECT
        t.stay_id,
        v.charttime,
        v.heart_rate,
        v.resp_rate,
        v.spo2,
        v.temperature,
        COALESCE(v.sbp, v.sbp_ni) AS sbp,
        COALESCE(v.mbp, v.mbp_ni) AS mbp
      FROM t0 t
      JOIN mimiciv_derived.vitalsign v
        ON v.stay_id = t.stay_id
      WHERE v.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND v.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    )
    SELECT
      stay_id,
      arg_max(heart_rate, charttime) FILTER (WHERE heart_rate IS NOT NULL) AS heart_rate,
      arg_max(resp_rate,  charttime) FILTER (WHERE resp_rate  IS NOT NULL) AS respiratory_rate,
      arg_max(spo2,       charttime) FILTER (WHERE spo2       IS NOT NULL) AS spo2,
      arg_max(temperature,charttime) FILTER (WHERE temperature IS NOT NULL) AS temperature,
      arg_max(sbp, charttime) FILTER (WHERE sbp IS NOT NULL) AS systolic_abp,
      arg_max(mbp, charttime) FILTER (WHERE mbp IS NOT NULL) AS map
    FROM w
    GROUP BY stay_id;
    """)

    # Labs core: chemistry + bg + coag + cbc
    con.execute(f"""
    CREATE OR REPLACE TABLE base_chem AS
    WITH w AS (
      SELECT
        t.stay_id,
        c.charttime,
        c.bun,
        c.creatinine,
        c.chloride,
        c.glucose,
        c.sodium,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY c.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.chemistry c
        ON c.hadm_id = t.hadm_id
      WHERE c.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND c.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    )
    SELECT stay_id, bun, creatinine, chloride, glucose, sodium
    FROM w
    WHERE rn = 1;
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE base_bg AS
    WITH w AS (
      SELECT
        t.stay_id,
        b.charttime,
        b.lactate,
        b.fio2,
        b.po2,
        -- alcune versioni hanno so2, altre no: gestiamo dopo
        b.ph,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY b.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.bg b
        ON b.hadm_id = t.hadm_id
      WHERE b.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND b.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    )
    SELECT stay_id, lactate, fio2, po2, ph
    FROM w
    WHERE rn = 1;
    """)

    # FiO2: ventilator_setting, oxygen_delivery, bg (in order of priority)
    con.execute("""
    CREATE OR REPLACE TABLE base_fio2 AS
    WITH vent AS (
      SELECT
        t.stay_id,
        v.charttime,
        v.fio2,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY v.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.ventilator_setting v
        ON v.stay_id = t.stay_id
      WHERE v.fio2 IS NOT NULL
        AND v.charttime >= t.t0_time
        AND v.charttime <  t.enroll_end
    ),
    bg AS (
      SELECT
        t.stay_id,
        b.charttime,
        b.fio2,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY b.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.bg b
        ON b.hadm_id = t.hadm_id
      WHERE b.fio2 IS NOT NULL
        AND b.charttime >= t.t0_time
        AND b.charttime <  t.enroll_end
    )
    SELECT
      t.stay_id,
      COALESCE(v.fio2, b.fio2) AS fio2
    FROM t0 t
    LEFT JOIN (SELECT stay_id, fio2 FROM vent WHERE rn=1) v USING (stay_id)
    LEFT JOIN (SELECT stay_id, fio2 FROM bg   WHERE rn=1) b USING (stay_id);
    """)



    # Optional: so2 from bg if present
    bg_has_so2 = col_exists(con, "mimiciv_derived", "bg", "so2")
    if bg_has_so2:
        con.execute(f"""
        CREATE OR REPLACE TABLE base_bg_so2 AS
        WITH w AS (
          SELECT
            t.stay_id,
            b.charttime,
            b.so2,
            ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY b.charttime DESC) AS rn
          FROM t0 t
          JOIN mimiciv_derived.bg b
            ON b.hadm_id = t.hadm_id
          WHERE b.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
            AND b.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
            AND b.so2 IS NOT NULL
        )
        SELECT stay_id, so2
        FROM w WHERE rn=1;
        """)
    else:
        con.execute("""
        CREATE OR REPLACE TABLE base_bg_so2 AS
        SELECT stay_id, NULL::DOUBLE AS so2
        FROM t0;
        """)

    con.execute(f"""
    CREATE OR REPLACE TABLE base_coag AS
    WITH w AS (
      SELECT
        t.stay_id,
        co.charttime,
        co.inr,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY co.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.coagulation co
        ON co.hadm_id = t.hadm_id
      WHERE co.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND co.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    )
    SELECT stay_id, inr
    FROM w WHERE rn=1;
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE base_cbc AS
    WITH w AS (
      SELECT
        t.stay_id,
        c.charttime,
        c.hemoglobin,
        c.platelet AS platelet,
        c.wbc,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY c.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.complete_blood_count c
        ON c.hadm_id = t.hadm_id
      WHERE c.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND c.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    )
    SELECT stay_id, hemoglobin, platelet, wbc
    FROM w WHERE rn=1;
    """)

    # SOFA baseline (take record in same baseline window; keep components)
    con.execute(f"""
    CREATE OR REPLACE TABLE base_sofa AS
    WITH w AS (
      SELECT
        t.stay_id,
        s.starttime,
        s.sofa_24hours AS sofa_score,
        s.respiration_24hours    AS sofa_resp,
        s.coagulation_24hours    AS sofa_coag,
        s.liver_24hours          AS sofa_liver,
        s.cardiovascular_24hours AS sofa_cv,
        s.cns_24hours            AS sofa_cns,
        s.renal_24hours          AS sofa_renal,
        s.gcs_min AS gcs,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY s.starttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.sofa s
        ON s.stay_id = t.stay_id
      WHERE s.starttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND s.starttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    )
    SELECT
      stay_id,
      sofa_score,
      sofa_resp, sofa_coag, sofa_liver, sofa_cv, sofa_cns, sofa_renal,
      gcs
    FROM w WHERE rn=1;
    """)

    # CGS:
    con.execute(f"""
    CREATE OR REPLACE TABLE base_gcs AS
    WITH w AS (
      SELECT
        t.stay_id,
        g.charttime,
        g.gcs,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY g.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.gcs g
        ON g.stay_id = t.stay_id
      WHERE g.gcs IS NOT NULL
        AND g.charttime >= t.t0_time
        AND g.charttime <  t.enroll_end
    )
    SELECT stay_id, gcs
    FROM w WHERE rn=1;
    """)


    # ALT/AST/Bilirubin: da enzyme o liver_function se presente, altrimenti NULL (ma *esplicito*)
    if has_enzyme and all(
        col_exists(con, "mimiciv_derived", "enzyme", c) for c in ["alt", "ast", "bilirubin_total"]
    ):
        con.execute(f"""
        CREATE OR REPLACE TABLE base_liverlabs AS
        WITH w AS (
          SELECT
            t.stay_id,
            e.charttime,
            e.alt,
            e.ast,
            e.bilirubin_total AS bilirubin,
            ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY e.charttime DESC) AS rn
          FROM t0 t
          JOIN mimiciv_derived.enzyme e
            ON e.hadm_id = t.hadm_id
          WHERE e.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
            AND e.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
        )
        SELECT stay_id, alt, ast, bilirubin
        FROM w WHERE rn=1;
        """)
    elif has_liver and all(
        col_exists(con, "mimiciv_derived", "liver_function", c) for c in ["alt", "ast", "bilirubin_total"]
    ):
        con.execute(f"""
        CREATE OR REPLACE TABLE base_liverlabs AS
        WITH w AS (
          SELECT
            t.stay_id,
            lf.charttime,
            lf.alt,
            lf.ast,
            lf.bilirubin_total AS bilirubin,
            ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY lf.charttime DESC) AS rn
          FROM t0 t
          JOIN mimiciv_derived.liver_function lf
            ON lf.hadm_id = t.hadm_id
          WHERE lf.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
            AND lf.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
        )
        SELECT stay_id, alt, ast, bilirubin
        FROM w WHERE rn=1;
        """)
    else:
        con.execute("""
        CREATE OR REPLACE TABLE base_liverlabs AS
        SELECT stay_id, NULL::DOUBLE AS alt, NULL::DOUBLE AS ast, NULL::DOUBLE AS bilirubin
        FROM t0;
        """)

    # -------------------------
    # 5) Assemble dataset + exclusions
    #    - Exclude steroid start < t0 - 10h
    #    - Keep only treat_steroid in {0,1} (drop intermediate >0 and <160)
    # -------------------------

    out_table = "analytic_sepsis_steroids_mit_v2"
    con.execute(f"""
    CREATE OR REPLACE TABLE {out_table} AS
    SELECT
      t.subject_id,
      t.hadm_id,
      t.stay_id,
      t.t0_time AS intime,
      t.enroll_end,
      t.sepsis_time,

      tr.treat_steroid,
      tr.hc_equiv_mg_0_24h,
      tr.steroid_events_0_24h,
      tr.steroid_unparsed_0_24h,
      tr.steroid_parsed_0_24h,

      o.y_28d_mort_inhosp,
      o.y_hosp_mort,
      o.dischtime,
      o.deathtime,
      o.died_or_disch_before_enroll_end,

      d.age,
      d.gender,
      d.race,

      bmi.weight_kg,
      bmi.height_cm,
      bmi.bmi,

      -- Paper feature set (Supplementary Text 1)
      s.sofa_score,
      s.sofa_resp, s.sofa_coag, s.sofa_liver, s.sofa_cv, s.sofa_cns, s.sofa_renal,

      l.alt,
      l.ast,
      l.bilirubin,

      chem.bun,
      chem.creatinine,
      chem.chloride,
      chem.glucose,
      chem.sodium,

      COALESCE(fio2.fio2, bg.fio2) AS fio2,
      bg.po2,
      so2.so2,
      bg.lactate,

      v.temperature,
      cbc.wbc,
      v.spo2 AS spo2,  -- “SO2” in paper is sat (ABG); keep both for now
      v.respiratory_rate,
      v.heart_rate,
      v.systolic_abp,
      v.map,
      COALESCE(s.gcs, gcs.gcs) AS gcs,
      cbc.hemoglobin,
      coag.inr,
      cbc.platelet

    FROM t0 t
    LEFT JOIN treat tr        ON tr.stay_id = t.stay_id
    LEFT JOIN outcome o       ON o.stay_id  = t.stay_id
    LEFT JOIN v_demog d       ON d.stay_id  = t.stay_id
    -- LEFT JOIN v_anthro a      ON a.stay_id  = t.stay_id

    LEFT JOIN base_sofa s     ON s.stay_id  = t.stay_id
    LEFT JOIN base_liverlabs l ON l.stay_id = t.stay_id

    LEFT JOIN base_chem chem  ON chem.stay_id = t.stay_id
    LEFT JOIN base_bg bg      ON bg.stay_id   = t.stay_id
    LEFT JOIN base_bg_so2 so2 ON so2.stay_id  = t.stay_id
    LEFT JOIN base_vitals v   ON v.stay_id    = t.stay_id
    LEFT JOIN base_cbc cbc    ON cbc.stay_id  = t.stay_id
    LEFT JOIN base_coag coag  ON coag.stay_id = t.stay_id
    LEFT JOIN base_bmi bmi    ON bmi.stay_id  = t.stay_id
    LEFT JOIN base_gcs gcs    ON gcs.stay_id  = t.stay_id
    LEFT JOIN base_fio2 fio2  ON fio2.stay_id = t.stay_id

    LEFT JOIN excl_steroids ex ON ex.stay_id = t.stay_id

    WHERE ex.stay_id IS NULL
      AND tr.treat_steroid IS NOT NULL;
    """)

    # -------------------------
    # 6) QA: missingness table (conteggio + %)
    # -------------------------

    con.execute(f"""
    CREATE OR REPLACE TABLE qa_missingness AS
    WITH base AS (
      SELECT * FROM {out_table}
    ),
    n AS (SELECT COUNT(*) AS N FROM base),
    rows AS (
      SELECT 'ALT' AS cov, SUM(CASE WHEN alt IS NULL THEN 1 ELSE 0 END) AS miss FROM base UNION ALL
      SELECT 'AST', SUM(CASE WHEN ast IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Bilirubin', SUM(CASE WHEN bilirubin IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'BUN', SUM(CASE WHEN bun IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Creatinine', SUM(CASE WHEN creatinine IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Chloride', SUM(CASE WHEN chloride IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Glucose', SUM(CASE WHEN glucose IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Sodium', SUM(CASE WHEN sodium IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'FiO2', SUM(CASE WHEN fio2 IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'PaO2', SUM(CASE WHEN po2 IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'SO2_bg', SUM(CASE WHEN so2 IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'SpO2', SUM(CASE WHEN spo2 IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Temperature', SUM(CASE WHEN temperature IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'WBC', SUM(CASE WHEN wbc IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Lactate', SUM(CASE WHEN lactate IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Heart_rate', SUM(CASE WHEN heart_rate IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Respiratory_rate', SUM(CASE WHEN respiratory_rate IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Systolic_ABP', SUM(CASE WHEN systolic_abp IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'MAP', SUM(CASE WHEN map IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'GCS', SUM(CASE WHEN gcs IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Hemoglobin', SUM(CASE WHEN hemoglobin IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'INR', SUM(CASE WHEN inr IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Platelet', SUM(CASE WHEN platelet IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'BMI', SUM(CASE WHEN bmi IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Age', SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) FROM base
    )
    SELECT
      cov,
      miss AS missing_n,
      ROUND(miss::DOUBLE / (SELECT N FROM n), 4) AS missing_frac
    FROM rows
    ORDER BY missing_frac DESC, cov;
    """)

    # quick prints (puoi commentare)
    print("\nCohort size:")
    print(con.execute(f"SELECT COUNT(*) AS n FROM {out_table};").fetchdf())

    print("\nTreated/control:")
    print(con.execute(f"""
      SELECT treat_steroid, COUNT(*) AS n
      FROM {out_table}
      GROUP BY 1
      ORDER BY 1;
    """).fetchdf())

    print("\nDose distribution (hc_equiv_mg_0_24h):")
    print(con.execute(f"""
      SELECT
        quantile(hc_equiv_mg_0_24h, 0.0)  AS p0,
        quantile(hc_equiv_mg_0_24h, 0.5)  AS p50,
        quantile(hc_equiv_mg_0_24h, 0.9)  AS p90,
        quantile(hc_equiv_mg_0_24h, 0.99) AS p99
      FROM {out_table};
    """).fetchdf())

    print("\nMissingness (top 15):")
    print(con.execute("SELECT * FROM qa_missingness LIMIT 15;").fetchdf())

    print("\nUnparsed steroid events (0-24h):")
    print(con.execute(f"""
      SELECT
        SUM(steroid_unparsed_0_24h) AS unparsed_total,
        SUM(steroid_events_0_24h)   AS total_events,
        AVG(CASE WHEN steroid_events_0_24h>0 THEN steroid_unparsed_0_24h::DOUBLE/steroid_events_0_24h ELSE 0 END) AS avg_unparsed_frac
      FROM {out_table};
    """).fetchdf())

    # -------------------------
    # 7) Export
    # -------------------------
    con.execute(f"COPY {out_table} TO '{str(out)}' (FORMAT PARQUET);")
    print(f"\nWrote: {out}")

    con.close()


if __name__ == "__main__":
    # Esempio:
    #   python build_sepsis_steroids_mit_v2.py \
    #     --db /path/to/mimic.duckdb \
    #     --out analytic/analytic_sepsis_steroids_mit_v2.parquet \
    #     --tmp intermediate/duckdb_tmp
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=False, help="Path al duckdb con tabelle MIT (mimiciv_icu/hosp/derived).")
    ap.add_argument("--out", required=False, help="Output parquet.")
    ap.add_argument("--tmp", default="duckdb_tmp", help="Temp dir per duckdb.")
    args = ap.parse_args()

    if not args.db:
        args.db = MIT_MIMIC
    if not args.out:
        args.out = ANALYTIC_DIR / "analytic_sepsis_steroids_mit_v2.parquet"

    main(db_path=args.db, out_path=args.out, tmp_dir=args.tmp)
