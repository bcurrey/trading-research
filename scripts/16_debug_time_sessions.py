import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet")

print("\nBasic timestamp sample:")
print(
    df.select([
        "ts_event",
        "ts_ct",
        "hour_ct",
        "minute_ct",
        "minute_of_day_ct",
    ]).head(50)
)

print("\nMinute of day range:")
print(
    df.select([
        pl.col("minute_of_day_ct").min().alias("min_minute"),
        pl.col("minute_of_day_ct").max().alias("max_minute"),
        pl.col("hour_ct").min().alias("min_hour"),
        pl.col("hour_ct").max().alias("max_hour"),
    ])
)

print("\nBars by CT hour:")
print(
    df.group_by("hour_ct")
    .agg(pl.len().alias("bars"))
    .sort("hour_ct")
)

print("\nRows where hour should be RTH range:")
print(
    df.filter(
        (pl.col("hour_ct") >= 7) &
        (pl.col("hour_ct") <= 15)
    )
    .select([
        "ts_event",
        "ts_ct",
        "hour_ct",
        "minute_ct",
        "minute_of_day_ct",
        "open",
        "high",
        "low",
        "close",
    ])
    .head(100)
)