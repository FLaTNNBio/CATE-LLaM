import duckdb
from src.config import HOSP_DIR, INTERMEDIATE_DIR

DB_PATH = INTERMEDIATE_DIR / "mimic.duckdb"
TMP_DIR = INTERMEDIATE_DIR / "duckdb_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

COHORT_PATH = INTERMEDIATE_DIR / "cohort_clean.parquet"
D_LABITEMS = HOSP_DIR / "d_labitems.csv.gz"
LABEVENTS = HOSP_DIR / "labevents.csv.gz"

OUT_PATH = INTERMEDIATE_DIR / "labs_extended_6h.parquet"
WINDOW_HOURS = 6


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")
    con.execute("SET memory_limit='8GB';")

    con.execute(f"""
    CREATE OR REPLACE VIEW v_cohort AS
    SELECT stay_id, hadm_id, intime
    FROM read_parquet('{COHORT_PATH.as_posix()}');
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_dlab AS
    SELECT * FROM read_csv_auto('{D_LABITEMS.as_posix()}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE OR REPLACE VIEW v_labs AS
    SELECT * FROM read_csv_auto('{LABEVENTS.as_posix()}', union_by_name=true);
    """)

    con.execute("""
    CREATE OR REPLACE TABLE v_canonical_labs_ext AS
    SELECT * FROM (VALUES
      ('hematocrit',      '%hematocrit%'),
      ('glucose',         '%glucose%'),
      ('magnesium',       '%magnesium%'),
      ('albumin',         '%albumin%'),
      ('ast',             '%asparate aminotransferase%'), -- keep your original spelling variant if present
      ('ast',             '%aspartate aminotransferase%'), -- also correct spelling
      ('alt',             '%alanine aminotransferase%'),
      ('bilirubin_total', '%bilirubin, total%'),
      ('crp',             '%c-reactive protein%'),
      ('calcium_total',   '%calcium, total%'),
      ('inr',             '%inr%'),
      ('ptt',             '%ptt%')
    ) AS t(lab_name, pattern);
    """)

    con.execute("""
    CREATE OR REPLACE TABLE v_lab_item_candidates_ext AS
    SELECT DISTINCT
      c.lab_name,
      d.itemid,
      d.label
    FROM v_canonical_labs_ext c
    JOIN v_dlab d
      ON lower(d.label) LIKE lower(c.pattern)
    WHERE
      lower(d.label) NOT LIKE '%urine%'
      AND lower(d.label) NOT LIKE '%csf%'
      AND lower(d.label) NOT LIKE '%ascites%'
      AND lower(d.label) NOT LIKE '%pleural%'
      AND lower(d.label) NOT LIKE '%stool%'
      AND lower(d.label) NOT LIKE '%joint%'
      AND lower(d.label) NOT LIKE '%other fluid%'
      AND lower(d.label) NOT LIKE '%clearance%'
      AND lower(d.label) NOT LIKE '%ratio%'
      AND lower(d.label) NOT LIKE '%calculated%'
      AND lower(d.label) NOT LIKE '%24 hr%'
      AND lower(d.label) NOT LIKE '%24hr%';
    """)

    print("Candidates per extended lab:")
    print(con.execute("""
    SELECT lab_name, COUNT(*) AS n_candidates
    FROM v_lab_item_candidates_ext
    GROUP BY lab_name
    ORDER BY lab_name;
    """).fetchdf())

    con.execute(f"""
    CREATE OR REPLACE TABLE v_lab_item_choice_ext AS
    WITH window_events AS (
      SELECT
        c.stay_id,
        cand.lab_name,
        le.itemid
      FROM v_cohort c
      JOIN v_labs le
        ON le.hadm_id = c.hadm_id
      JOIN v_lab_item_candidates_ext cand
        ON le.itemid = cand.itemid
      WHERE le.charttime IS NOT NULL
        AND le.valuenum IS NOT NULL
        AND le.charttime >= c.intime
        AND le.charttime <  c.intime + INTERVAL '{WINDOW_HOURS}' HOUR
    ),
    counts AS (
      SELECT lab_name, itemid, COUNT(*) AS n
      FROM window_events
      GROUP BY lab_name, itemid
    ),
    ranked AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY lab_name ORDER BY n DESC) AS rn
      FROM counts
    )
    SELECT lab_name, itemid
    FROM ranked
    WHERE rn = 1;
    """)

    print("\nChosen itemid per extended lab:")
    print(con.execute("""
    SELECT
      c.lab_name,
      c.itemid,
      d.label
    FROM v_lab_item_choice_ext c
    JOIN v_dlab d
      ON c.itemid = d.itemid
    ORDER BY c.lab_name;
    """).fetchdf())

    con.execute(f"""
    CREATE OR REPLACE TABLE v_labs_ext_long AS
    WITH chosen AS (
      SELECT * FROM v_lab_item_choice_ext
    ),
    filtered AS (
      SELECT
        c.stay_id,
        ch.lab_name,
        le.charttime,
        le.valuenum
      FROM v_cohort c
      JOIN v_labs le
        ON le.hadm_id = c.hadm_id
      JOIN chosen ch
        ON le.itemid = ch.itemid
      WHERE le.charttime IS NOT NULL
        AND le.valuenum IS NOT NULL
        AND le.charttime >= c.intime
        AND le.charttime <  c.intime + INTERVAL '{WINDOW_HOURS}' HOUR
    ),
    ranked AS (
      SELECT *,
        ROW_NUMBER() OVER (PARTITION BY stay_id, lab_name ORDER BY charttime) AS rn
      FROM filtered
    )
    SELECT stay_id, lab_name, valuenum
    FROM ranked
    WHERE rn = 1;
    """)

    con.execute("""
    CREATE OR REPLACE TABLE v_labs_ext_wide AS
    SELECT *
    FROM (
      PIVOT v_labs_ext_long
      ON lab_name IN ('hematocrit','glucose','magnesium','albumin','ast','alt',
                      'bilirubin_total','crp','calcium_total','inr','ptt')
      USING first(valuenum)
    );
    """)

    con.execute("""
    CREATE OR REPLACE TABLE labs_extended_6h AS
    SELECT
      c.stay_id,
      w.hematocrit,
      w.glucose,
      w.magnesium,
      w.albumin,
      w.ast,
      w.alt,
      w.bilirubin_total,
      w.crp,
      w.calcium_total,
      w.inr,
      w.ptt
    FROM v_cohort c
    LEFT JOIN v_labs_ext_wide w
      ON c.stay_id = w.stay_id;
    """)

    print("\nExtended labs table size (should match cohort):")
    print(con.execute("SELECT COUNT(*) AS n FROM labs_extended_6h;").fetchdf())

    con.execute(f"""
    COPY labs_extended_6h
    TO '{OUT_PATH.as_posix()}'
    (FORMAT PARQUET);
    """)
    print(f"\nParquet written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
