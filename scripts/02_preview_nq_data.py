import polars as pl

FILE_PATH = r"D:\TradingResearch\data_raw\NQ-Part 1-20180101-20201230.ohlcv-1m.csv.zst"

print("\nLoading sample data...\n")

df = pl.read_csv(
    FILE_PATH,
    n_rows=5
)

print(df)

print("\nColumns:")
print(df.columns)

print("\nData types:")
print(df.dtypes)