# Build a trial-like analytic dataset for early RBC transfusion (MIT tables only)
# - Time-zero (t0): first hemoglobin < threshold within 48h of ICU intime
# - Treatment: any RBC inputevent within 3h after t0
# - Baseline covariates: last observation in [t0-6h, t0]
# - Outcome: hospital mortality from admissions.hospital_expire_flag
#
# Output: analytic/analytic_rbc_mit_v2.parquet (1 row per stay_id)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import duckdb

from src.config import MIT_MIMIC, ANALYTIC_DIR


@dataclass
class Params:
    hb_threshold: float = 8.5          # eligibility: first Hb <= hb_threshold
    hb_min_sanity: float = 7.5         # optional sanity lower bound (avoid junk/outliers)
    elig_within_hours: int = 48

    baseline_lookback_hours: int = 6

    treat_window_hours: int = 3        # "within 3h after t0"
    memory_limit: str = "8GB"


# RBC itemids (MIMIC-IV ICU inputevents)
# Verify in your setup if needed.
RBC_ITEMIDS = [225168, 226368, 227070]


def table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = ? AND table_name = ?
    LIMIT 1
    """
    return con.execute(q, [schema, table]).fetchone() is not None


def cleanup_previous_run(con: duckdb.DuckDBPyConnection) -> None:
    """
    Drop objects from previous runs to avoid collisions:
    - any TABLE/VIEW in schema 'main' starting with v_
    - known working tables (t0, treat, outcome, base_*, qa_*, analytic_rbc_*)
    """
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

    rows2 = con.execute("""
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND (
            table_name IN ('t0','treat','outcome','rbc_events')
            OR table_name LIKE 'base\\_%' ESCAPE '\\'
            OR table_name LIKE 'qa\\_%' ESCAPE '\\'
            OR table_name LIKE 'analytic_rbc\\_%' ESCAPE '\\'
          )
    """).fetchall()

    for name, ttype in rows2:
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

    # -------------------------
    # Preconditions
    # -------------------------
    required = [
        ("mimiciv_icu", "icustays"),
        ("mimiciv_hosp", "admissions"),
        ("mimiciv_derived", "age"),
        ("mimiciv_derived", "icustay_detail"),
        ("mimiciv_derived", "complete_blood_count"),
        ("mimiciv_derived", "vitalsign"),
        ("mimiciv_derived", "chemistry"),
        ("mimiciv_derived", "coagulation"),
        ("mimiciv_derived", "bg"),
        ("mimiciv_derived", "height"),
        ("mimiciv_derived", "first_day_weight"),
        ("mimiciv_icu", "inputevents"),
    ]
    missing = [(s, t) for (s, t) in required if not table_exists(con, s, t)]
    if missing:
        raise RuntimeError(
            f"Mancano tabelle richieste nel DB (MIT concepts). Assenti: {missing}"
        )

    # -------------------------
    # Base views (MIT tables only)
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

    # Anthropometrics (use first_day_weight + height; BMI baseline-friendly)
    con.execute("""
    CREATE OR REPLACE VIEW v_anthro AS
    WITH h AS (
      SELECT
        stay_id,
        height,
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

    # -------------------------
    # 1) Define time-zero (t0): first Hb <= threshold within 48h of ICU intime
    #    Hb from mimiciv_derived.complete_blood_count (already harmonized)
    # -------------------------
    con.execute(f"""
    CREATE OR REPLACE TABLE t0 AS
    WITH base AS (
      SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime
      FROM v_icustays i
      JOIN v_demog d
        ON d.stay_id = i.stay_id
      WHERE d.age >= 18
    ),
    hb AS (
      SELECT
        b.subject_id,
        b.hadm_id,
        b.stay_id,
        b.intime,
        c.charttime,
        c.hemoglobin AS hb
      FROM base b
      JOIN mimiciv_derived.complete_blood_count c
        ON c.hadm_id = b.hadm_id
      WHERE c.charttime IS NOT NULL
        AND c.hemoglobin IS NOT NULL
        AND c.charttime >= b.intime
        AND c.charttime <  b.intime + INTERVAL '{p.elig_within_hours}' HOUR
        AND c.hemoglobin <= {p.hb_threshold}
        AND c.hemoglobin >= {p.hb_min_sanity}
    ),
    ranked AS (
      SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime) AS rn
      FROM hb
    )
    SELECT
      subject_id,
      hadm_id,
      stay_id,
      intime,
      charttime AS t0_time,
      hb AS t0_hb,
      intime + INTERVAL '{p.elig_within_hours}' HOUR AS elig_end
    FROM ranked
    WHERE rn = 1;
    """)

    # -------------------------
    # 2) Treatment: any RBC inputevent within 3h after t0
    # -------------------------
    rbc_ids_sql = ", ".join(str(x) for x in RBC_ITEMIDS)

    con.execute(f"""
    CREATE OR REPLACE TABLE rbc_events AS
    SELECT
      t.stay_id,
      ie.starttime,
      ie.itemid,
      ie.amount,
      ie.amountuom,
      ie.rate,
      ie.rateuom
    FROM t0 t
    JOIN mimiciv_icu.inputevents ie
      ON ie.stay_id = t.stay_id
    WHERE ie.itemid IN ({rbc_ids_sql})
      AND ie.starttime IS NOT NULL
      AND ie.starttime >= t.t0_time
      AND ie.starttime <  t.t0_time + INTERVAL '{p.treat_window_hours}' HOUR;
    """)

    con.execute("""
    CREATE OR REPLACE TABLE treat AS
    WITH agg AS (
      SELECT
        stay_id,
        MIN(starttime) AS rbc_first_time,
        COUNT(*) AS rbc_event_count,
        -- proxy units: sum(amount) if numeric else count(events)
        SUM(CASE WHEN try_cast(amount AS DOUBLE) IS NOT NULL THEN try_cast(amount AS DOUBLE) ELSE 0 END) AS rbc_amount_sum,
        SUM(CASE WHEN try_cast(amount AS DOUBLE) IS NULL THEN 1 ELSE 0 END) AS rbc_amount_unparsed
      FROM rbc_events
      GROUP BY stay_id
    )
    SELECT
      t.stay_id,
      CASE WHEN a.stay_id IS NULL THEN 0 ELSE 1 END AS t_rbc_3h,
      a.rbc_first_time,
      a.rbc_event_count,
      CASE
        WHEN a.rbc_amount_sum > 0 THEN a.rbc_amount_sum
        WHEN a.rbc_event_count IS NOT NULL THEN a.rbc_event_count::DOUBLE
        ELSE 0.0
      END AS rbc_units_proxy,
      COALESCE(a.rbc_amount_unparsed, 0) AS rbc_amount_unparsed
    FROM t0 t
    LEFT JOIN agg a
      ON a.stay_id = t.stay_id;
    """)

    # -------------------------
    # 3) Baseline covariates: last observation in [t0-6h, t0]
    #    Reuse a sensible subset similar to steroids pipeline:
    #    - vitals: hr, rr, spo2, temp, sbp, mbp
    #    - labs: bun, creatinine, sodium, chloride, glucose
    #    - cbc: wbc, platelet, hemoglobin
    #    - coag: inr
    #    - bg: lactate, ph, po2, fio2 (optional; may be sparse)
    # -------------------------

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
      JOIN mimiciv_derived.vitalsign v
        ON v.stay_id = t.stay_id
      WHERE v.charttime IS NOT NULL
        AND v.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND v.charttime <= t.t0_time
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

    # Chemistry
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
      WHERE c.charttime IS NOT NULL
        AND c.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND c.charttime <= t.t0_time
    )
    SELECT stay_id, bun, creatinine, chloride, glucose, sodium
    FROM w
    WHERE rn=1;
    """)

    # CBC
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
      WHERE c.charttime IS NOT NULL
        AND c.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND c.charttime <= t.t0_time
    )
    SELECT stay_id, hemoglobin, platelet, wbc
    FROM w
    WHERE rn=1;
    """)

    # Coagulation
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
      WHERE co.charttime IS NOT NULL
        AND co.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND co.charttime <= t.t0_time
    )
    SELECT stay_id, inr
    FROM w WHERE rn=1;
    """)

    # Blood gas (optional sparse confounders)
    con.execute(f"""
    CREATE OR REPLACE TABLE base_bg AS
    WITH w AS (
      SELECT
        t.stay_id,
        b.charttime,
        b.lactate,
        b.ph,
        b.po2,
        b.fio2,
        ROW_NUMBER() OVER (PARTITION BY t.stay_id ORDER BY b.charttime DESC) AS rn
      FROM t0 t
      JOIN mimiciv_derived.bg b
        ON b.hadm_id = t.hadm_id
      WHERE b.charttime IS NOT NULL
        AND b.charttime >  t.t0_time - INTERVAL '{p.baseline_lookback_hours}' HOUR
        AND b.charttime <= t.t0_time
    )
    SELECT stay_id, lactate, ph, po2, fio2
    FROM w WHERE rn=1;
    """)

    # -------------------------
    # 4) Outcome: hospital mortality
    # -------------------------
    con.execute("""
    CREATE OR REPLACE TABLE outcome AS
    SELECT
      t.stay_id,
      a.dischtime,
      a.deathtime,
      CAST(a.hospital_expire_flag AS INTEGER) AS y_hosp_mort
    FROM t0 t
    LEFT JOIN v_adm a
      ON a.hadm_id = t.hadm_id;
    """)

    # -------------------------
    # 5) Assemble final analytic dataset
    # -------------------------
    out_table = "analytic_rbc_mit_v2"

    con.execute(f"""
    CREATE OR REPLACE TABLE {out_table} AS
    SELECT
      -- ids & timing
      t.subject_id,
      t.hadm_id,
      t.stay_id,
      t.intime,
      t.t0_time,
      t.t0_hb,

      -- design constants
      {p.hb_threshold}::DOUBLE AS hb_threshold,
      {p.elig_within_hours}::INTEGER AS elig_within_hours,
      {p.baseline_lookback_hours}::INTEGER AS baseline_lookback_hours,
      {p.treat_window_hours}::INTEGER AS treat_window_hours,

      -- treatment
      tr.t_rbc_3h,
      tr.rbc_first_time,
      tr.rbc_event_count,
      tr.rbc_units_proxy,
      tr.rbc_amount_unparsed,

      -- outcome
      o.y_hosp_mort,
      o.dischtime,
      o.deathtime,

      -- demographics
      d.age,
      d.gender,
      d.race,

      -- anthropometrics
      a.weight_kg,
      a.height_cm,
      a.bmi,

      -- baseline vitals
      v.heart_rate,
      v.respiratory_rate,
      v.spo2,
      v.temperature,
      v.systolic_abp,
      v.map,

      -- baseline labs
      chem.bun,
      chem.creatinine,
      chem.chloride,
      chem.glucose,
      chem.sodium,

      cbc.hemoglobin,
      cbc.platelet,
      cbc.wbc,

      coag.inr,

      bg.lactate,
      bg.ph,
      bg.po2,
      bg.fio2,

      -- missingness indicators (MNAR proxies)
      CASE WHEN chem.creatinine IS NULL THEN 0 ELSE 1 END AS has_creatinine,
      CASE WHEN chem.bun IS NULL THEN 0 ELSE 1 END AS has_bun,
      CASE WHEN cbc.platelet IS NULL THEN 0 ELSE 1 END AS has_platelet,
      CASE WHEN cbc.wbc IS NULL THEN 0 ELSE 1 END AS has_wbc,
      CASE WHEN coag.inr IS NULL THEN 0 ELSE 1 END AS has_inr,
      CASE WHEN bg.lactate IS NULL THEN 0 ELSE 1 END AS has_lactate,
      CASE WHEN v.map IS NULL THEN 0 ELSE 1 END AS has_map

    FROM t0 t
    JOIN treat tr      ON tr.stay_id = t.stay_id
    LEFT JOIN outcome o ON o.stay_id = t.stay_id
    LEFT JOIN v_demog d ON d.stay_id = t.stay_id
    LEFT JOIN v_anthro a ON a.stay_id = t.stay_id
    LEFT JOIN base_vitals v ON v.stay_id = t.stay_id
    LEFT JOIN base_chem chem ON chem.stay_id = t.stay_id
    LEFT JOIN base_cbc cbc ON cbc.stay_id = t.stay_id
    LEFT JOIN base_coag coag ON coag.stay_id = t.stay_id
    LEFT JOIN base_bg bg ON bg.stay_id = t.stay_id
    ;
    """)

    # -------------------------
    # 6) QA: counts + missingness (top)
    # -------------------------
    con.execute(f"""
    CREATE OR REPLACE TABLE qa_missingness AS
    WITH base AS (SELECT * FROM {out_table}),
    n AS (SELECT COUNT(*) AS N FROM base),
    rows AS (
      SELECT 'BMI' AS cov, SUM(CASE WHEN bmi IS NULL THEN 1 ELSE 0 END) AS miss FROM base UNION ALL
      SELECT 'MAP', SUM(CASE WHEN map IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Systolic_ABP', SUM(CASE WHEN systolic_abp IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Lactate', SUM(CASE WHEN lactate IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'INR', SUM(CASE WHEN inr IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'WBC', SUM(CASE WHEN wbc IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Platelet', SUM(CASE WHEN platelet IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Creatinine', SUM(CASE WHEN creatinine IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'BUN', SUM(CASE WHEN bun IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'FiO2_bg', SUM(CASE WHEN fio2 IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'PaO2', SUM(CASE WHEN po2 IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'SpO2', SUM(CASE WHEN spo2 IS NULL THEN 1 ELSE 0 END) FROM base UNION ALL
      SELECT 'Heart_rate', SUM(CASE WHEN heart_rate IS NULL THEN 1 ELSE 0 END) FROM base
    )
    SELECT
      cov,
      miss AS missing_n,
      ROUND(miss::DOUBLE / (SELECT N FROM n), 4) AS missing_frac
    FROM rows
    ORDER BY missing_frac DESC, cov;
    """)

    print("\nCohort size:")
    print(con.execute(f"SELECT COUNT(*) AS n FROM {out_table};").fetchdf())

    print("\nTreated/control:")
    print(con.execute(f"""
      SELECT t_rbc_3h, COUNT(*) AS n
      FROM {out_table}
      GROUP BY 1
      ORDER BY 1;
    """).fetchdf())

    print("\nHb at t0 distribution:")
    print(con.execute(f"""
      SELECT
        quantile(t0_hb, 0.0) AS p0,
        quantile(t0_hb, 0.5) AS p50,
        quantile(t0_hb, 0.9) AS p90,
        quantile(t0_hb, 0.99) AS p99
      FROM {out_table};
    """).fetchdf())

    print("\nMissingness (top 15):")
    print(con.execute("SELECT * FROM qa_missingness LIMIT 15;").fetchdf())

    # -------------------------
    # 7) Export
    # -------------------------
    con.execute(f"COPY {out_table} TO '{str(out)}' (FORMAT PARQUET);")
    print(f"\nWrote: {out}")

    con.close()


if __name__ == "__main__":
    # Example:
    # python build_rbc_mit_v2.py --db /path/to/mimic.duckdb --out analytic/analytic_rbc_mit_v2.parquet --tmp duckdb_tmp
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
