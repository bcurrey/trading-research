import polars as pl
from pathlib import Path

RAW_DIR = Path(r"D:\TradingResearch\data_raw")
OUTPUT_DIR = Path(r"D:\TradingResearch\data_parquet")

FILES = [
    RAW_DIR / "NQ-Part 1-20180101-20201230.ohlcv-1m.csv.zst",
    RAW_DIR / "NQ-Part 2-20210101-20260507.ohlcv-1m.csv.zst",
]

print("\nLoading NQ files...\n")

dfs = []

for file in FILES:
    print(f"Reading: {file.name}")

    df = pl.read_csv(file)

    dfs.append(df)

print("\nCombining files...\n")

df = pl.concat(dfs)

print(f"Rows before filtering: {len(df):,}")

# Remove spread symbols
df = df.filter(
    ~pl.col("symbol").str.contains("-")
)

print(f"Rows after removing spreads: {len(df):,}")

# Convert timestamp
df = df.with_columns(
    pl.col("ts_event").str.strptime(
        pl.Datetime,
        format="%Y-%m-%dT%H:%M:%S%.fZ",
        strict=False
    )
)

# For each timestamp, keep highest-volume contract
df = (
    df.sort(["ts_event", "volume"], descending=[False, True])
      .group_by("ts_event")
      .first()
      .sort("ts_event")
)

print(f"Rows after selecting dominant contract: {len(df):,}")

# Keep only needed columns
df = df.select([
    "ts_event",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume"
])

print("\nPreview:")
print(df.head())

output_file = OUTPUT_DIR / "NQ_continuous_1m.parquet"

print(f"\nWriting parquet file:\n{output_file}")

df.write_parquet(output_file)

print("\nDONE.")