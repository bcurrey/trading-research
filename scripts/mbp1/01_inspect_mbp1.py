from pathlib import Path
import polars as pl

FILE = r"D:\TradingResearch\data_raw\mbp1\2026-02\unzipped\glbx-mdp3-20260220.mbp-1.csv.zst"

print("\nLoading sample...\n")

df = pl.read_csv(
    FILE,
    n_rows=10
)

print(df)

print("\nColumns:\n")
print(df.columns)

print("\nSchema:\n")
print(df.schema)