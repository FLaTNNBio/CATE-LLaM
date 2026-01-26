import duckdb
from src.config import ANALYTIC_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Inputs
IN_V0_PREP = ANALYTIC_DIR / "analytic_v0_prepared.parquet"
DEMOG = INTERMEDIATE_DIR / "demographics.parquet"
GENERAL = INTERMEDIATE_DIR / "general.parquet"              # currently contains imperial units from OMR (lb/in)
LABS_EXT = INTERMEDIATE_DIR / "labs_extended_6h.parquet"

# Output
OUT_PATH = ANALYTIC_DIR / "analytic_v0_extended_prepared.parquet"

# Unit conversion factors
LB_TO_KG = 0.453592
IN_TO_CM = 2.54

# Conservative bounds (after conversion to metric)
CLIP = {
    "age": (0.0, 120.0),

    # anthropometrics in METRIC
    "admission_weight_kg": (20.0, 400.0),
    "height_cm": (80.0, 250.0),
    "bmi": (10.0, 80.0),

    # extended labs (units typical in MIMIC lab events)
    "hematocrit": (0.0, 80.0),         # %
    "glucose": (0.0, 1500.0),          # mg/dL
    "magnesium": (0.0, 10.0),          # mg/dL-ish (conservative)
    "albumin": (0.0, 10.0),            # g/dL
    "ast": (0.0, 5000.0),              # U/L
    "alt": (0.0, 5000.0),              # U/L
    "bilirubin_total": (0.0, 80.0),    # mg/dL
    "crp": (0.0, 500.0),               # conservative (varies by lab/unit)
    "calcium_total": (0.0, 20.0),      # mg/dL
    "inr": (0.0, 20.0),
    "ptt": (0.0, 200.0),               # seconds-ish
}


def clip_expr(col: str, lo: float, hi: float) -> str:
    return f"""
    CASE
      WHEN {col} IS NULL THEN NULL
      WHEN {col} < {lo} THEN {lo}
      WHEN {col} > {hi} THEN {hi}
      ELSE {col}
    END AS {col}
    """.strip()


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='4GB';")

    # Views (prefixed to avoid name collisions)
    con.execute(f"CREATE OR REPLACE VIEW v0 AS SELECT * FROM read_parquet('{IN_V0_PREP.as_posix()}');")
    con.execute(f"CREATE OR REPLACE VIEW v_demog AS SELECT * FROM read_parquet('{DEMOG.as_posix()}');")
    con.execute(f"CREATE OR REPLACE VIEW v_gen_raw AS SELECT * FROM read_parquet('{GENERAL.as_posix()}');")
    con.execute(f"CREATE OR REPLACE VIEW v_labsx AS SELECT * FROM read_parquet('{LABS_EXT.as_posix()}');")

    # 1) Join everything into a raw extended table.
    # 2) Convert OMR anthropometrics from imperial -> metric deterministically (based on observed percentiles):
    #    - admission_weight_kg currently holds pounds
    #    - height_cm currently holds inches
    # 3) Recompute BMI using metric values.
    con.execute(f"""
    CREATE OR REPLACE TABLE v0_ext_raw AS
    SELECT
      v0.*,

      -- demographics
      d.gender,
      d.race,
      d.age,

      -- general (raw from OMR; empirically appears to be lb and inches)
      -- convert: lb -> kg, inches -> cm
      CASE
        WHEN g.admission_weight_kg IS NULL THEN NULL
        ELSE g.admission_weight_kg * {LB_TO_KG}
      END AS admission_weight_kg,

      CASE
        WHEN g.height_cm IS NULL THEN NULL
        ELSE g.height_cm * {IN_TO_CM}
      END AS height_cm,

      -- recompute BMI from converted metric measures when possible;
      -- otherwise keep NULL (do not trust raw BMI if units differ across systems)
      CASE
        WHEN g.admission_weight_kg IS NULL OR g.height_cm IS NULL THEN NULL
        WHEN g.height_cm * {IN_TO_CM} <= 0 THEN NULL
        ELSE (g.admission_weight_kg * {LB_TO_KG}) / POWER((g.height_cm * {IN_TO_CM}) / 100.0, 2)
      END AS bmi,

      -- extended labs
      x.hematocrit,
      x.glucose,
      x.magnesium,
      x.albumin,
      x.ast,
      x.alt,
      x.bilirubin_total,
      x.crp,
      x.calcium_total,
      x.inr,
      x.ptt

    FROM v0
    LEFT JOIN v_demog d
      ON v0.stay_id = d.stay_id
    LEFT JOIN v_gen_raw g
      ON v0.stay_id = g.stay_id
    LEFT JOIN v_labsx x
      ON v0.stay_id = x.stay_id;
    """)

    # Add has_* indicators for new columns
    new_has = [
        "age", "gender", "race",
        "admission_weight_kg", "height_cm", "bmi",
        "hematocrit", "glucose", "magnesium", "albumin", "ast", "alt",
        "bilirubin_total", "crp", "calcium_total", "inr", "ptt",
    ]
    has_exprs = [f"CASE WHEN {c} IS NULL THEN 0 ELSE 1 END AS has_{c}" for c in new_has]

    # Clip selected numeric columns (post-conversion)
    clip_cols = list(CLIP.keys())
    exclude_clause = ", ".join(clip_cols)
    clip_exprs = [clip_expr(c, lo, hi) for c, (lo, hi) in CLIP.items()]

    con.execute(f"""
    CREATE OR REPLACE TABLE v0_ext_prepared AS
    SELECT
      * EXCLUDE ({exclude_clause}),
      {", ".join(clip_exprs)},
      {", ".join(has_exprs)}
    FROM v0_ext_raw;
    """)

    # Sanity checks
    print("Rows (v0_ext_prepared):")
    print(con.execute("SELECT COUNT(*) AS n FROM v0_ext_prepared;").fetchdf())

    print("\nDuplicate stay_id rows (should be 0):")
    print(con.execute("""
    SELECT COUNT(*) AS n_dup
    FROM (
      SELECT stay_id, COUNT(*) c
      FROM v0_ext_prepared
      GROUP BY stay_id
      HAVING COUNT(*) > 1
    );
    """).fetchdf())

    # Check plausibility of converted units
    print("\nAnthropometrics percentiles (after conversion & clipping):")
    print(con.execute("""
    SELECT
      quantile_cont(admission_weight_kg, 0.01) AS w_p01,
      quantile_cont(admission_weight_kg, 0.50) AS w_p50,
      quantile_cont(admission_weight_kg, 0.99) AS w_p99,
      quantile_cont(height_cm, 0.01) AS h_p01,
      quantile_cont(height_cm, 0.50) AS h_p50,
      quantile_cont(height_cm, 0.99) AS h_p99,
      quantile_cont(bmi, 0.01) AS bmi_p01,
      quantile_cont(bmi, 0.50) AS bmi_p50,
      quantile_cont(bmi, 0.99) AS bmi_p99
    FROM v0_ext_prepared;
    """).fetchdf())

    print("\nMissingness quick check (new core):")
    print(con.execute("""
    SELECT
      1 - COUNT(age)::DOUBLE/COUNT(*) AS age_null_frac,
      1 - COUNT(admission_weight_kg)::DOUBLE/COUNT(*) AS weight_null_frac,
      1 - COUNT(height_cm)::DOUBLE/COUNT(*) AS height_null_frac,
      1 - COUNT(bmi)::DOUBLE/COUNT(*) AS bmi_null_frac,
      1 - COUNT(glucose)::DOUBLE/COUNT(*) AS glucose_null_frac,
      1 - COUNT(inr)::DOUBLE/COUNT(*) AS inr_null_frac,
      1 - COUNT(crp)::DOUBLE/COUNT(*) AS crp_null_frac
    FROM v0_ext_prepared;
    """).fetchdf())

    # Export
    con.execute(f"""
    COPY v0_ext_prepared
    TO '{OUT_PATH.as_posix()}'
    (FORMAT PARQUET);
    """)
    print(f"\nAnalytic v0 extended prepared written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
