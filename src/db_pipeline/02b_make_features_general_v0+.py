import duckdb
from src.config import HOSP_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
OMR = HOSP_DIR / "omr.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "general.parquet"

LOOKBACK_DAYS = 365
FORWARD_DAYS = 7


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='8GB';")

    con.execute(f"""
    CREATE OR REPLACE VIEW v_cohort AS
    SELECT stay_id, subject_id, intime
    FROM read_parquet('{COHORT_PATH.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_omr AS
    SELECT * FROM read_csv_auto('{OMR.as_posix()}', union_by_name=true);
    """)

    # Inspectable patterns (OMR result_name varies a bit across versions)
    con.execute("""
    CREATE OR REPLACE TABLE v_omr_filtered AS
    SELECT
      c.stay_id,
      o.chartdate,
      o.result_name,
      o.result_value
    FROM v_cohort c
    JOIN v_omr o
      ON o.subject_id = c.subject_id
    WHERE o.chartdate IS NOT NULL
      AND o.result_value IS NOT NULL
      AND o.chartdate >= (DATE(c.intime) - INTERVAL '365' DAY)
      AND o.chartdate <= (DATE(c.intime) + INTERVAL '7' DAY)
      AND (
        lower(o.result_name) LIKE '%weight%'
        OR lower(o.result_name) LIKE '%height%'
        OR lower(o.result_name) LIKE '%bmi%'
      );
    """)

    # Normalize to three targets (weight_kg, height_cm, bmi)
    # Some OMR entries are in inches/pounds; we avoid unit guessing by preferring explicit metric labels.
    # If your OMR uses imperial, we can add conversion rules later.
    con.execute("""
    CREATE OR REPLACE TABLE v_omr_norm AS
    SELECT
      stay_id,
      chartdate,
      CASE
        WHEN lower(result_name) LIKE '%bmi%' THEN 'bmi'
        WHEN lower(result_name) LIKE '%height%' AND (lower(result_name) LIKE '%cm%' OR lower(result_name) LIKE '%cent%') THEN 'height_cm'
        WHEN lower(result_name) LIKE '%weight%' AND (lower(result_name) LIKE '%kg%' OR lower(result_name) LIKE '%kilo%') THEN 'weight_kg'
        -- fallback (less safe): accept height/weight without explicit unit
        WHEN lower(result_name) LIKE '%height%' THEN 'height_cm'
        WHEN lower(result_name) LIKE '%weight%' THEN 'weight_kg'
        ELSE NULL
      END AS target,
      TRY_CAST(result_value AS DOUBLE) AS val
    FROM v_omr_filtered
    WHERE TRY_CAST(result_value AS DOUBLE) IS NOT NULL
      AND (
        lower(result_name) LIKE '%bmi%'
        OR lower(result_name) LIKE '%height%'
        OR lower(result_name) LIKE '%weight%'
      );
    """)

    # Choose the closest measurement to intime per stay_id+target
    con.execute("""
    CREATE OR REPLACE TABLE v_general_long AS
    WITH joined AS (
      SELECT
        n.stay_id,
        n.target,
        n.val,
        -- distance in days between chartdate and intime date
        ABS(DATE_DIFF('day', DATE(c.intime), n.chartdate)) AS abs_day_diff,
        -- prefer measurements on/near ICU date
        CASE WHEN n.chartdate <= DATE(c.intime) THEN 0 ELSE 1 END AS is_after
      FROM v_omr_norm n
      JOIN v_cohort c
        ON n.stay_id = c.stay_id
      WHERE n.target IS NOT NULL
    ),
    ranked AS (
      SELECT *,
        ROW_NUMBER() OVER (
          PARTITION BY stay_id, target
          ORDER BY abs_day_diff ASC, is_after ASC
        ) AS rn
      FROM joined
    )
    SELECT stay_id, target, val
    FROM ranked
    WHERE rn = 1;
    """)

    con.execute("""
    CREATE OR REPLACE TABLE v_general_wide AS
    SELECT *
    FROM (
      PIVOT v_general_long
      ON target IN ('weight_kg','height_cm','bmi')
      USING first(val)
    );
    """)

    con.execute("""
    CREATE OR REPLACE TABLE general AS
    SELECT
      c.stay_id,
      w.weight_kg AS admission_weight_kg,
      w.height_cm AS height_cm,
      CASE
        WHEN w.bmi IS NOT NULL THEN w.bmi
        WHEN w.weight_kg IS NULL OR w.height_cm IS NULL THEN NULL
        WHEN w.height_cm <= 0 THEN NULL
        ELSE w.weight_kg / POWER(w.height_cm / 100.0, 2)
      END AS bmi
    FROM v_cohort c
    LEFT JOIN v_general_wide w
      ON c.stay_id = w.stay_id;
    """)

    print("General rows:")
    print(con.execute("SELECT COUNT(*) AS n FROM general;").fetchdf())

    print("\nMissingness (weight/height/bmi):")
    print(con.execute("""
    SELECT
      1 - COUNT(admission_weight_kg)::DOUBLE/COUNT(*) AS weight_null_frac,
      1 - COUNT(height_cm)::DOUBLE/COUNT(*) AS height_null_frac,
      1 - COUNT(bmi)::DOUBLE/COUNT(*) AS bmi_null_frac
    FROM general;
    """).fetchdf())

    print("\nQuick plausibility (percentiles):")
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
    FROM general;
    """).fetchdf())

    con.execute(f"""
    COPY general
    TO '{OUT_PATH.as_posix()}'
    (FORMAT PARQUET);
    """)
    print(f"\nParquet written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
