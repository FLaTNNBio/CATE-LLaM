import duckdb

from src.config import HOSP_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
PATIENTS = HOSP_DIR / "patients.csv.gz"
ADMISSIONS = HOSP_DIR / "admissions.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "demographics.parquet"


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='4GB';")

    con.execute(f"""
    CREATE OR REPLACE VIEW v_cohort AS
    SELECT subject_id, hadm_id, stay_id, intime
    FROM read_parquet('{COHORT_PATH.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW patients AS
    SELECT * FROM read_csv_auto('{PATIENTS.as_posix()}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW admissions AS
    SELECT * FROM read_csv_auto('{ADMISSIONS.as_posix()}', union_by_name=true);
    """)

    # MIMIC-IV: no real DOB; use anchor_age + (year(intime) - anchor_year)
    con.execute("""
    CREATE OR REPLACE TABLE demographics AS
    SELECT
      c.stay_id,
      c.subject_id,
      c.hadm_id,
      p.gender AS gender,
      a.race AS race,
      -- compute age at ICU intime
      GREATEST(
        0,
        LEAST(
          120,
          p.anchor_age + (EXTRACT(YEAR FROM c.intime)::INT - p.anchor_year)
        )
      ) AS age
    FROM v_cohort c
    LEFT JOIN patients p
      ON c.subject_id = p.subject_id
    LEFT JOIN admissions a
      ON c.hadm_id = a.hadm_id;
    """)

    print("Demographics rows:")
    print(con.execute("SELECT COUNT(*) AS n FROM demographics;").fetchdf())

    print("\nMissingness (gender/race/age):")
    print(con.execute("""
    SELECT
      1 - COUNT(gender)::DOUBLE/COUNT(*) AS gender_null_frac,
      1 - COUNT(race)::DOUBLE/COUNT(*) AS race_null_frac,
      1 - COUNT(age)::DOUBLE/COUNT(*) AS age_null_frac
    FROM demographics;
    """).fetchdf())

    con.execute(f"""
    COPY demographics
    TO '{OUT_PATH.as_posix()}'
    (FORMAT PARQUET);
    """)
    print(f"\nParquet written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
