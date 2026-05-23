import polars as pl
from pathlib import Path

RAW_DIR = Path(r"D:\TradingResearch\data_raw")
OUTPUT_DIR = Path(r"D:\TradingResearch\data_parquet")

file_path = RAW_DIR / "forex_factory_cache.csv"

print("\nLoading economic calendar...\n")

df = pl.read_csv(file_path, infer_schema_length=10000)

print(f"Rows loaded: {len(df):,}")

# Parse timezone-aware datetime and convert to UTC-naive datetime
df = df.with_columns([
    pl.col("DateTime")
    .str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%z", strict=False)
    .alias("event_time")
])

# Keep useful columns and standardize names
df = df.select([
    pl.col("event_time"),
    pl.col("Currency").alias("currency"),
    pl.col("Impact").alias("impact"),
    pl.col("Event").alias("event"),
    pl.col("Actual").alias("actual"),
    pl.col("Forecast").alias("forecast"),
    pl.col("Previous").alias("previous"),
    pl.col("Detail").alias("detail"),
])

# Filter to USD economic events
usd = df.filter(
    (pl.col("currency") == "USD") &
    (pl.col("impact") != "Non-Economic") &
    pl.col("event_time").is_not_null()
).sort("event_time")

print(f"USD economic rows: {len(usd):,}")

# Flag high impact
usd = usd.with_columns([
    (pl.col("impact") == "High Impact Expected").alias("is_high_impact"),
    pl.col("event_time").dt.date().alias("event_date_utc"),
])

# Flag major event names
usd = usd.with_columns([
    pl.when(pl.col("event").str.contains("CPI|Consumer Price", literal=False))
      .then(pl.lit("CPI"))
      .when(pl.col("event").str.contains("FOMC|Federal Funds|Fed Interest Rate", literal=False))
      .then(pl.lit("FOMC"))
      .when(pl.col("event").str.contains("Non-Farm|Nonfarm|NFP|Employment Change", literal=False))
      .then(pl.lit("NFP"))
      .when(pl.col("event").str.contains("PPI|Producer Price", literal=False))
      .then(pl.lit("PPI"))
      .when(pl.col("event").str.contains("PMI|ISM", literal=False))
      .then(pl.lit("PMI_ISM"))
      .when(pl.col("event").str.contains("Unemployment|Jobless", literal=False))
      .then(pl.lit("JOBS"))
      .when(pl.col("event").str.contains("Powell|Fed Chair", literal=False))
      .then(pl.lit("FED_SPEECH"))
      .otherwise(pl.lit("OTHER"))
      .alias("event_group")
])

print("\nPreview:")
print(
    usd.select([
        "event_time",
        "currency",
        "impact",
        "event",
        "event_group",
        "is_high_impact",
    ]).head(30)
)

output_file = OUTPUT_DIR / "econ_calendar_usd.parquet"

print(f"\nWriting:\n{output_file}")
usd.write_parquet(output_file)

print("\nDONE.")