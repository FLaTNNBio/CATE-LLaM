import pandas as pd


df = pd.read_parquet("../../data/analytic/analytic_v0_extended_prepared.parquet")
df = df.groupby("subject_id")["stay_id"].nunique().value_counts().head()
print(df)