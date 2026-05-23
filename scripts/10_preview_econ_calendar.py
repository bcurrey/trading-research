import polars as pl
from pathlib import Path

RAW_DIR = Path(r"D:\TradingResearch\data_raw")

file_path = RAW_DIR / "forex_factory_cache.csv"

print("\nLoading economic calendar sample...\n")

df = pl.read_csv(file_path, n_rows=20, infer_schema_length=1000)

print(df)

print("\nColumns:")
for col in df.columns:
    print(col)

print("\nData types:")
for col, dtype in zip(df.columns, df.dtypes):
    print(f"{col}: {dtype}")