# Build a trial-like analytic dataset for EARLY vs DELAYED loop diuretics in sepsis (MIT tables only)
# - Cohort: adult, first ICU stay, Sepsis-3 within 24h of ICU intime
# - Landmark time t0: ICU intime + 6h
# - Exposure window: [t0, t0+24h]
# - Treatment:
#     Early  = first loop diuretic in [t0, t0+6h)
#     Delayed= first loop diuretic in [t0+6h, t0+24h)
#   Primary analysis: restrict to patients with ANY loop diuretic in [t0, t0+24h) (reduces PS extremes)
# - Baseline covariates: last value in [t0-6h, t0] (no leakage)
# - Outcome: 28-day in-hospital mortality proxy (deathtime <= t0+28d)
#
# Output: analytic/analytic_sepsis_early_diuretics_mit_v1.parquet
from dataclasses import dataclass
from pathlib import Path
import duckdb

from src.config import MIT_MIMIC, ANALYTIC_DIR


@dataclass
class Params:
    landmark_hours: int = 12
    sepsis_within_hours: int = 24

    baseline_lookback_hours: int = 6

    early_hours: int = 12
    exposure_hours: int = 24

    grace_hours: int = 6  # allow some leakage for covariate measurement after t0 (e.g., labs drawn at t0 may be recorded shortly after)

    memory_limit: str = "8GB"


def table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    return con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema=? AND table_name=?
        LIMIT 1
        """,
        [schema, table],
    ).fetchone() is not None


def cleanup_previous_run(con: duckdb.DuckDBPyConnection) -> None:
    # drop v_* and known working tables in main
    rows = con.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema='main' AND table_name LIKE 'v\\_%' ESCAPE '\\'
    """).fetchall()
    for name, ttype in rows:
        if ttype == "VIEW":
            con.execute(f'DROP VIEW IF EXISTS "{name}";')
        else:
            con.execute(f'DROP TABLE IF EXISTS "{name}";')

    rows2 = con.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema='main'
          AND (
            table_name IN ('t0','treat','outcome','diur_events','qa_missingness')
            OR table_name LIKE 'base\\_%' ESCAPE '\\'
            -- OR table_name LIKE 'analytic\\_%' ESCAPE '\\'
          )
    """).fetchall()
    for name, ttype in rows:
        if ttype == "VIEW":
            con.execute(f'DROP VIEW IF EXISTS "{name}";')
        else:
            con.execute(f'DROP TABLE IF EXISTS "{name}";')


def main(db_path: str, out_path: str, tmp_dir: str) -> None:
    p = Params()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute(f"SET temp_directory='{str(Path(tmp_dir))}';")
    con.execute(f"SET memory_limit='{p.memory_limit}';")

    cleanup_previous_run(con)

    # ---------- Required MIT tables ----------
    required = [
        ("mimiciv_icu", "icustays"),
        ("mimiciv_hosp", "admissions"),
        ("mimiciv_derived", "age"),
        ("mimiciv_derived", "icustay_detail"),
        ("mimiciv_derived", "sepsis3"),
        ("mimiciv_derived", "vitalsign"),
        ("mimiciv_derived", "chemistry"),
        ("mimiciv_derived", "complete_blood_count"),
        ("mimiciv_derived", "coagulation"),
        ("mimiciv_derived", "gcs"),
        ("mimiciv_derived", "urine_output_rate"),
        ("mimiciv_derived", "ventilation"),
        ("mimiciv_derived", "ventilator_setting"),
        ("mimiciv_derived", "oxygen_delivery"),
        ("mimiciv_derived", "bg"),
        ("mimiciv_derived", "height"),
        ("mimiciv_derived", "first_day_weight"),
        ("mimiciv_derived", "norepinephrine"),
        ("mimiciv_derived", "epinephrine"),
        ("mimiciv_derived", "dopamine"),
        ("mimiciv_derived", "dobutamine"),
    ]
    missing = [(s, t) for (s, t) in required if not table_exists(con, s, t)]
    if missing:
        raise RuntimeError(f"Mancano tabelle MIT richieste: {missing}")

    has_rx = table_exists(con, "mimiciv_hosp", "prescriptions")
    has_inputs = table_exists(con, "mimiciv_icu", "inputevents")
    if not (has_rx or has_inputs):
        raise RuntimeError("Serve mimiciv_hosp.prescriptions o mimiciv_icu.inputevents per identificare i diuretici.")

    # ---------- Core views ----------
    con.execute("""
    CREATE OR REPLACE VIEW v_icustays AS
    SELECT subject_id, hadm_id, stay_id, intime
    FROM mimiciv_icu.icustays
    WHERE intime IS NOT NULL;
    """)

    con.execute("""
    CREATE OR REPLACE VIEW v_adm AS
    SELECT hadm_id, dischtime, deathtime, hospital_expire_flag
    FROM mimiciv_hosp.admissions;
    """)

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

    con.execute("""
    CREATE OR REPLACE VIEW v_anthro AS
    WITH h AS (
      SELECT stay_id, height,
             ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime) AS rn
      FROM mimiciv_derived.height
      WHERE height IS NOT NULL
    ),
    w AS (
      SELECT stay_id, weight
      FROM mimiciv_derived.first_day_weight
      WHERE weight IS NOT NULL
    )
    SELECT
      COALESCE(w.stay_id, h.stay_id) AS stay_id,
      w.weight AS weight_kg,
      h.height AS height_cm,
      CASE
        WHEN w.weight IS NOT NULL AND h.height IS NOT NULL AND h.height > 0
          THEN w.weight / POWER(h.height/100.0, 2)
        ELSE NULL
      END AS bmi
    FROM w
    FULL OUTER JOIN (SELECT stay_id, height FROM h WHERE rn=1) h
      ON h.stay_id = w.stay_id;
    """)

    # ---------- 1) Define cohort + landmark t0 ----------
    # sepsis_time from sepsis3 (use GREATEST(suspected_infection_time, sofa_time) as in MIT scripts usually)
    con.execute(f"""
    CREATE OR REPLACE TABLE t0 AS
    WITH base AS (
      SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime
      FROM v_icustays i
      JOIN v_demog d ON d.stay_id=i.stay_id
      WHERE d.age >= 18
    ),
    first_stay AS (
      SELECT *
      FROM (
        SELECT
          b.*,
          ROW_NUMBER() OVER (PARTITION BY b.subject_id ORDER BY b.intime) AS rn
        FROM base b
      ) WHERE rn=1
    ),
    sepsis AS (
      SELECT
        s.stay_id,
        GREATEST(s.suspected_infection_time, s.sofa_time) AS sepsis_time
      FROM mimiciv_derived.sepsis3 s
      WHERE s.sepsis3 = TRUE
        AND s.suspected_infection_time IS NOT NULL
        AND s.sofa_time IS NOT NULL
    ),
    joined AS (
      SELECT
        f.*,
        s.sepsis_time,
        s.sepsis_time + INTERVAL '6' HOUR AS t0_time,
        s.sepsis_time + INTERVAL '30' HOUR AS exposure_end   -- 6h + 24h
      FROM first_stay f
      JOIN sepsis s ON s.stay_id=f.stay_id
      WHERE s.sepsis_time >= f.intime
        AND s.sepsis_time <  f.intime + INTERVAL '24' HOUR
    )
    SELECT
      subject_id, hadm_id, stay_id, intime,
      sepsis_time, t0_time, exposure_end
    FROM joined;
    """)

    # ---------- 2) Diuretic exposure (loop diuretics) ----------
    # We prefer prescriptions if available (cleaner), else fallback to inputevents.
    # Drugs: furosemide, bumetanide, torsemide, ethacrynic acid.
    if has_rx:
        con.execute("""
        CREATE OR REPLACE TABLE diur_events AS
        SELECT
          t.stay_id,
          r.starttime AS event_time,
          LOWER(COALESCE(r.drug, '')) AS drug_lc,
          'rx' AS source
        FROM t0 t
        JOIN mimiciv_hosp.prescriptions r
          ON r.hadm_id = t.hadm_id
        WHERE r.starttime IS NOT NULL
          AND r.starttime >= t.t0_time
          AND r.starttime <  t.exposure_end
          AND (
            LOWER(r.drug) LIKE '%furosemide%' OR
            LOWER(r.drug) LIKE '%bumetanide%' OR
            LOWER(r.drug) LIKE '%torsemide%' OR
            LOWER(r.drug) LIKE '%ethacryn%'
          );
        """)
    else:
        # Fallback: inputevents name fields differ; we only use itemid if you know them.
        # Here we do string match on label fields if present; if not, you must provide itemids.
        con.execute("""
        CREATE OR REPLACE TABLE diur_events AS
        SELECT
          t.stay_id,
          ie.starttime AS event_time,
          LOWER(COALESCE(ie.label, COALESCE(ie.ordercategoryname, ''))) AS drug_lc,
          'input' AS source
        FROM t0 t
        JOIN mimiciv_icu.inputevents ie
          ON ie.stay_id = t.stay_id
        WHERE ie.starttime IS NOT NULL
          AND ie.starttime >= t.t0_time
          AND ie.starttime <  t.exposure_end
          AND (
            LOWER(COALESCE(ie.label, COALESCE(ie.ordercategoryname, ''))) LIKE '%furosemide%' OR
            LOWER(COALESCE(ie.label, COALESCE(ie.ordercategoryname, ''))) LIKE '%bumetanide%' OR
            LOWER(COALESCE(ie.label, COALESCE(ie.ordercategoryname, ''))) LIKE '%torsemide%' OR
            LOWER(COALESCE(ie.label, COALESCE(ie.ordercategoryname, ''))) LIKE '%ethacryn%'
          );
        """)

    # Assign early vs delayed using FIRST diuretic time in [t0,t0+24h)
    con.execute(f"""
    CREATE OR REPLACE TABLE treat AS
    WITH first_diur AS (
      SELECT
        stay_id,
        MIN(event_time) AS first_diur_time
      FROM diur_events
      GROUP BY stay_id
    )
    SELECT
      t.stay_id,
      f.first_diur_time,
      CASE
        WHEN f.first_diur_time >= t.t0_time
         AND f.first_diur_time <  t.t0_time + INTERVAL '{p.early_hours}' HOUR
        THEN 1
        WHEN f.first_diur_time >= t.t0_time + INTERVAL '{p.early_hours}' HOUR
         AND f.first_diur_time <  t.exposure_end
        THEN 0
        ELSE NULL
      END AS treat_early
    FROM t0 t
    JOIN first_diur f ON f.stay_id = t.stay_id
    WHERE f.first_diur_time < t.exposure_end;
    """)

    # ---------- 3) Outcome (28d in-hosp proxy) ----------
    con.execute("""
    CREATE OR REPLACE TABLE outcome AS
    SELECT
      t.stay_id,
      a.dischtime,
      a.deathtime,
      CAST(a.hospital_expire_flag AS INTEGER) AS y_hosp_mort,
      CASE
        WHEN a.deathtime IS NOT NULL AND a.deathtime <= t.t0_time + INTERVAL '28' DAY THEN 1
        ELSE 0
      END AS y_28d_mort_inhosp
    FROM t0 t
    LEFT JOIN v_adm a ON a.hadm_id=t.hadm_id;
    """)

    # ---------- 4) Baseline covariates in [t0-6h, t0] ----------
    # Vitals
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
      JOIN mimiciv_derived.vitalsign v ON v.stay_id=t.stay_id
      WHERE v.charttime IS NOT NULL
        AND v.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND v.charttime <= t.t0_time
    )
    SELECT
      stay_id,
      arg_max(heart_rate, charttime) FILTER (WHERE heart_rate IS NOT NULL) AS hr,
      arg_max(resp_rate,  charttime) FILTER (WHERE resp_rate  IS NOT NULL) AS rr,
      arg_max(spo2,       charttime) FILTER (WHERE spo2       IS NOT NULL) AS spo2,
      arg_max(temperature,charttime) FILTER (WHERE temperature IS NOT NULL) AS temp_c,
      arg_max(sbp, charttime) FILTER (WHERE sbp IS NOT NULL) AS sbp,
      arg_max(mbp, charttime) FILTER (WHERE mbp IS NOT NULL) AS map
    FROM w
    GROUP BY stay_id;
    """)

    # Chemistry
    con.execute(f"""
    CREATE OR REPLACE TABLE base_chem AS
    WITH w AS (
      SELECT
        t.stay_id, c.charttime,
        c.creatinine, c.bun, c.sodium, c.potassium, c.chloride, c.bicarbonate, c.glucose
      FROM t0 t
      JOIN mimiciv_derived.chemistry c ON c.hadm_id=t.hadm_id
      WHERE c.charttime IS NOT NULL
        AND c.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND c.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    )
    SELECT
      stay_id,
      arg_max(creatinine, charttime) FILTER (WHERE creatinine IS NOT NULL) AS creatinine,
      arg_max(bun,        charttime) FILTER (WHERE bun IS NOT NULL) AS bun,
      arg_max(sodium,     charttime) FILTER (WHERE sodium IS NOT NULL) AS sodium,
      arg_max(potassium,  charttime) FILTER (WHERE potassium IS NOT NULL) AS potassium,
      arg_max(chloride,   charttime) FILTER (WHERE chloride IS NOT NULL) AS chloride,
      arg_max(bicarbonate,charttime) FILTER (WHERE bicarbonate IS NOT NULL) AS bicarbonate,
      arg_max(glucose,    charttime) FILTER (WHERE glucose IS NOT NULL) AS glucose
    FROM w
    GROUP BY stay_id;
    """)

    # CBC
    con.execute(f"""
    CREATE OR REPLACE TABLE base_cbc AS
    WITH w AS (
      SELECT t.stay_id, c.charttime, c.hemoglobin, c.platelet AS platelets, c.wbc
      FROM t0 t
      JOIN mimiciv_derived.complete_blood_count c ON c.hadm_id=t.hadm_id
      WHERE c.charttime IS NOT NULL
        AND c.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND c.charttime <= t.t0_time
    )
    SELECT
      stay_id,
      arg_max(hemoglobin, charttime) FILTER (WHERE hemoglobin IS NOT NULL) AS hemoglobin,
      arg_max(platelets,  charttime) FILTER (WHERE platelets  IS NOT NULL) AS platelets,
      arg_max(wbc,        charttime) FILTER (WHERE wbc        IS NOT NULL) AS wbc
    FROM w
    GROUP BY stay_id;
    """)

    # Coag + Lactate (bg) + pH
    con.execute(f"""
    CREATE OR REPLACE TABLE base_coag_bg AS
    WITH coag AS (
      SELECT t.stay_id, co.charttime, co.inr, co.ptt
      FROM t0 t
      JOIN mimiciv_derived.coagulation co ON co.hadm_id=t.hadm_id
      WHERE co.charttime IS NOT NULL
        AND co.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND co.charttime <= t.t0_time + INTERVAL '{p.grace_hours}' HOUR
    ),
    bg AS (
      SELECT t.stay_id, b.charttime, b.lactate, b.ph, b.po2, b.pco2, b.fio2
      FROM t0 t
      JOIN mimiciv_derived.bg b ON b.hadm_id=t.hadm_id
      WHERE b.charttime IS NOT NULL
        AND b.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND b.charttime <= t.t0_time
    )
    SELECT
      t.stay_id,
      arg_max(coag.inr, coag.charttime) FILTER (WHERE coag.inr IS NOT NULL) AS inr,
      arg_max(coag.ptt, coag.charttime) FILTER (WHERE coag.ptt IS NOT NULL) AS ptt,
      arg_max(bg.lactate, bg.charttime) FILTER (WHERE bg.lactate IS NOT NULL) AS lactate,
      arg_max(bg.ph,      bg.charttime) FILTER (WHERE bg.ph IS NOT NULL) AS ph,
      arg_max(bg.po2,     bg.charttime) FILTER (WHERE bg.po2 IS NOT NULL) AS po2,
      arg_max(bg.pco2,    bg.charttime) FILTER (WHERE bg.pco2 IS NOT NULL) AS pco2,
      arg_max(bg.fio2,    bg.charttime) FILTER (WHERE bg.fio2 IS NOT NULL) AS fio2_bg
    FROM t0 t
    LEFT JOIN coag ON coag.stay_id=t.stay_id
    LEFT JOIN bg   ON bg.stay_id=t.stay_id
    GROUP BY t.stay_id;
    """)

    # FiO2 from ventilator_setting / oxygen_delivery (baseline) with fallback to bg
    con.execute(f"""
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
        AND v.charttime >  t.t0_time - INTERVAL '6' HOUR
        AND v.charttime <= t.t0_time
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
        AND b.charttime >  t.t0_time - INTERVAL '6' HOUR
        AND b.charttime <= t.t0_time
    )
    SELECT
      t.stay_id,
      COALESCE(v.fio2, b.fio2) AS fio2
    FROM t0 t
    LEFT JOIN (SELECT stay_id, fio2 FROM vent WHERE rn=1) v USING (stay_id)
    LEFT JOIN (SELECT stay_id, fio2 FROM bg   WHERE rn=1) b USING (stay_id);

    """)

    # GCS
    con.execute(f"""
    CREATE OR REPLACE TABLE base_gcs AS
    WITH w AS (
      SELECT t.stay_id, g.charttime, g.gcs,
             ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY g.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.gcs g ON g.stay_id=t.stay_id
      WHERE g.gcs IS NOT NULL
        AND g.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND g.charttime <= t.t0_time
    )
    SELECT stay_id, gcs
    FROM w WHERE rn=1;
    """)

    # Urine output rate (baseline)
    con.execute(f"""
    CREATE OR REPLACE TABLE base_uo AS
    WITH w AS (
      SELECT
        t.stay_id,
        u.charttime,
        u.uo_mlkghr_6hr,
        u.urineoutput_6hr,
        u.uo,  -- second fallback 
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY u.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.urine_output_rate u
        ON u.stay_id = t.stay_id
      WHERE u.charttime IS NOT NULL
        AND u.charttime >  t.t0_time - INTERVAL '6' HOUR
        AND u.charttime <= t.t0_time
        AND (
          u.uo_mlkghr_6hr IS NOT NULL
          OR u.urineoutput_6hr IS NOT NULL
          OR u.uo IS NOT NULL
        )
    )
    SELECT
      stay_id,
      -- Prefer ml/kg/hr (best), 2nd choice 6h vol, last uo
      CAST(uo_mlkghr_6hr AS DOUBLE) AS uo_mlkghr_6hr,
      urineoutput_6hr,
      uo
    FROM w
    WHERE rn = 1;
    """)

    # Ventilation status active at t0
    con.execute("""
    CREATE OR REPLACE TABLE base_vent AS
    WITH w AS (
      SELECT
        t.stay_id,
        CASE
          WHEN v.starttime <= t.t0_time AND (v.endtime IS NULL OR v.endtime > t.t0_time)
          THEN 1 ELSE 0
        END AS vent_active
      FROM t0 t
      LEFT JOIN mimiciv_derived.ventilation v ON v.stay_id=t.stay_id
    )
    SELECT stay_id, MAX(vent_active) AS vent_active_at_t0
    FROM w
    GROUP BY stay_id;
    """)

    # Vasopressors active + max rate in baseline window
    con.execute(f"""
    CREATE OR REPLACE TABLE base_vaso AS
    WITH vaso AS (
      SELECT stay_id, starttime, endtime, vaso_rate FROM mimiciv_derived.norepinephrine
      UNION ALL
      SELECT stay_id, starttime, endtime, vaso_rate FROM mimiciv_derived.epinephrine
      UNION ALL
      SELECT stay_id, starttime, endtime, vaso_rate FROM mimiciv_derived.dopamine
      UNION ALL
      SELECT stay_id, starttime, endtime, vaso_rate FROM mimiciv_derived.dobutamine
    ),
    w AS (
      SELECT
        t.stay_id,
        CASE
          WHEN v.starttime <= t.t0_time AND (v.endtime IS NULL OR v.endtime > t.t0_time)
          THEN 1 ELSE 0
        END AS vaso_active,
        CASE
          WHEN v.starttime IS NOT NULL
           AND v.starttime <= t.t0_time
           AND (v.endtime IS NULL OR v.endtime >= t.t0_time - INTERVAL '6' HOUR)
          THEN COALESCE(v.vaso_rate, 0)
          ELSE 0
        END AS rate_in_window
      FROM t0 t
      LEFT JOIN vaso v ON v.stay_id=t.stay_id
    )
    SELECT
      stay_id,
      MAX(vaso_active) AS vaso_active_at_t0,
      MAX(CASE WHEN rate_in_window > 0 THEN 1 ELSE 0 END) AS vaso_any_baseline,
      MAX(rate_in_window) AS vaso_max_rate_baseline
    FROM w
    GROUP BY stay_id;
    """)

    # ---------- 5) Assemble analytic table (primary: only early/delayed) ----------
    out_table = "analytic_sepsis_early_diuretics_mit_v1"

    con.execute(f"""
    CREATE OR REPLACE TABLE {out_table} AS
    SELECT
      t.subject_id, t.hadm_id, t.stay_id,
      t.intime,
      t.sepsis_time,
      t.t0_time,
      t.exposure_end,

      tr.treat_early,
      tr.first_diur_time,

      o.y_28d_mort_inhosp,
      o.y_hosp_mort,
      o.dischtime,
      o.deathtime,

      d.age,
      d.gender,
      d.race,

      a.weight_kg,
      a.height_cm,
      a.bmi,

      v.hr, v.rr, v.spo2, v.temp_c, v.sbp, v.map,

      chem.creatinine, chem.bun, chem.sodium, chem.potassium, chem.chloride, chem.bicarbonate, chem.glucose,

      cbc.hemoglobin, cbc.platelets, cbc.wbc,

      cbg.inr, cbg.ptt, cbg.lactate, cbg.ph, cbg.po2, cbg.pco2,
      fio2.fio2,

      g.gcs,
      uo.uo_mlkghr_6hr,
      uo.urineoutput_6hr,

      vent.vent_active_at_t0,

      vaso.vaso_active_at_t0,
      vaso.vaso_any_baseline,
      vaso.vaso_max_rate_baseline,

      -- Missingness indicators (MNAR proxies)
      CASE WHEN chem.creatinine IS NULL THEN 0 ELSE 1 END AS has_creatinine,
      CASE WHEN uo.uo_mlkghr_6hr IS NULL THEN 0 ELSE 1 END AS has_uo_mlkghr_6hr,
      CASE WHEN fio2.fio2 IS NULL THEN 0 ELSE 1 END AS has_fio2,
      CASE WHEN cbg.lactate IS NULL THEN 0 ELSE 1 END AS has_lactate,
      CASE WHEN g.gcs IS NULL THEN 0 ELSE 1 END AS has_gcs

    FROM t0 t
    JOIN treat tr ON tr.stay_id=t.stay_id              -- restrict to those with any diuretic in [t0, t0+24h)
    LEFT JOIN outcome o ON o.stay_id=t.stay_id
    LEFT JOIN v_demog d ON d.stay_id=t.stay_id
    LEFT JOIN v_anthro a ON a.stay_id=t.stay_id
    LEFT JOIN base_vitals v ON v.stay_id=t.stay_id
    LEFT JOIN base_chem chem ON chem.stay_id=t.stay_id
    LEFT JOIN base_cbc cbc ON cbc.stay_id=t.stay_id
    LEFT JOIN base_coag_bg cbg ON cbg.stay_id=t.stay_id
    LEFT JOIN base_fio2 fio2 ON fio2.stay_id=t.stay_id
    LEFT JOIN base_gcs g ON g.stay_id=t.stay_id
    LEFT JOIN base_uo uo ON uo.stay_id=t.stay_id
    LEFT JOIN base_vent vent ON vent.stay_id=t.stay_id
    LEFT JOIN base_vaso vaso ON vaso.stay_id=t.stay_id

    WHERE tr.treat_early IS NOT NULL
    AND COALESCE(vaso.vaso_active_at_t0, 0) = 0;
    """)

    # ---------- QA ----------
    print("\nCohort size:")
    print(con.execute(f"SELECT COUNT(*) AS n FROM {out_table};").fetchdf())

    print("\nTreated/control (early vs delayed):")
    print(con.execute(f"""
        SELECT treat_early, COUNT(*) AS n, AVG(y_28d_mort_inhosp) AS mort28
        FROM {out_table}
        GROUP BY 1
        ORDER BY 1;
    """).fetchdf())

    con.execute(f"""
    CREATE OR REPLACE TABLE qa_missingness AS
    WITH base AS (SELECT * FROM {out_table}),
    n AS (SELECT COUNT(*) AS N FROM base),

    rows AS (
      SELECT 'fio2' AS cov, SUM(CASE WHEN fio2 IS NULL THEN 1 ELSE 0 END) AS miss FROM base UNION ALL
      SELECT 'uo_mlkghr_6hr', SUM(CASE WHEN uo_mlkghr_6hr IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'urineoutput_6hr', SUM(CASE WHEN urineoutput_6hr IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'lactate', SUM(CASE WHEN lactate IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'creatinine', SUM(CASE WHEN creatinine IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'gcs', SUM(CASE WHEN gcs IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'bmi', SUM(CASE WHEN bmi IS NULL THEN 1 ELSE 0 END) FROM base
    )
    SELECT cov, miss AS missing_n, ROUND(miss::DOUBLE/(SELECT N FROM n), 4) AS missing_frac
    FROM rows
    ORDER BY missing_frac DESC;
    """)

    print("\nMissingness (selected):")
    print(con.execute("SELECT * FROM qa_missingness;").fetchdf())

    con.execute(f"COPY {out_table} TO '{str(out)}' (FORMAT PARQUET);")
    print(f"\nWrote: {out}")

    con.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=False, help="DuckDB path with MIT tables (mimiciv_* schemas).")
    ap.add_argument("--out", help="Output parquet.")
    ap.add_argument("--tmp", default="duckdb_tmp", help="DuckDB temp dir.")
    args = ap.parse_args()

    if not args.db:
        args.db = MIT_MIMIC
    if not args.out:
        args.out = ANALYTIC_DIR / "analytic_sepsis_early_diuretics_v1.parquet"


    main(db_path=args.db, out_path=args.out, tmp_dir=args.tmp)
