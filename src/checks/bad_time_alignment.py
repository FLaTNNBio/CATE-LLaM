"""
Diagnostics for ICU time alignment in the cohort.

We check whether:
- ICU intime is before hospital admittime
- ICU intime is after hospital dischtime

This script reads the exported Parquet (cohort.parquet),
so it does NOT depend on DuckDB tables existing from previous runs.
"""

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# src/check_time_alignment.py
import duckdb
from src.config import COHORT_PARQUET, register_parquet_view

print(f"Reading cohort parquet from {COHORT_PARQUET}")
# ---------------------------------------------------------------------
# Connect to DuckDB
# ---------------------------------------------------------------------
con = duckdb.connect(database=":memory:")

# usage
register_parquet_view(con, "cohort", COHORT_PARQUET)

select_bad_alignment = """
SELECT
  SUM(CASE WHEN intime < admittime THEN 1 ELSE 0 END) AS n_intime_before_admit,
  SUM(CASE WHEN intime > dischtime THEN 1 ELSE 0 END) AS n_intime_after_discharge
FROM cohort;
"""

# How many hours is intime BEFORE admittime?
before_admit = """
SELECT
  COUNT(*) AS n,
  MIN(EXTRACT(EPOCH FROM (admittime - intime))/3600.0) AS min_h,
  AVG(EXTRACT(EPOCH FROM (admittime - intime))/3600.0) AS mean_h,
  MAX(EXTRACT(EPOCH FROM (admittime - intime))/3600.0) AS max_h
FROM cohort
WHERE intime < admittime;
"""

# How many hours is intime AFTER dischtime?
after_dischtime = """
SELECT
  COUNT(*) AS n,
  MIN(EXTRACT(EPOCH FROM (intime - dischtime))/3600.0) AS min_h,
  AVG(EXTRACT(EPOCH FROM (intime - dischtime))/3600.0) AS mean_h,
  MAX(EXTRACT(EPOCH FROM (intime - dischtime))/3600.0) AS max_h
FROM cohort
WHERE intime > dischtime;
"""

bad_alignments = con.execute(select_bad_alignment).fetchall()
print("# Bad Alignments: in < adm : ", bad_alignments[0][0])
print("# Bad Alignments: in > disch : ", bad_alignments[0][1])
before = con.execute(before_admit).fetchall()
print("# InTime Before AdmTime: Min: ", before[0][1], "Avg: ", before[0][2], "Max: ", before[0][3])
after = con.execute(after_dischtime).fetchall()
print("# InTime After DischTime : Min: ", after[0][1], "Avg: ", after[0][2], "Max: ", after[0][3])

# Output from Raw Dataset:
# InTime Before AdmTime: Min:  0.001388888888888889 Avg:  2.9787455418381303 Max:  23.98111111111111
# InTime After DischTime : Min:  0.08333333333333333 Avg:  11.134183006535944 Max:  22.966666666666665
# We keep the firsts, but discard the seconds to avoid incoherence