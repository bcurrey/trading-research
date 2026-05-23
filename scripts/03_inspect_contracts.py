import polars as pl
from pathlib import Path

RAW_DIR = Path(r"D:\TradingResearch\data_raw")

files = [
    RAW_DIR / "NQ-Part 1-20180101-20201230.ohlcv-1m.csv.zst",
    RAW_DIR / "NQ-Part 2-20210101-20260507.ohlcv-1m.csv.zst",
    RAW_DIR / "ES-20180101-20260429.ohlcv-1m.csv.zst",
    RAW_DIR / "MNQ-20180101-20260429.ohlcv-1m.csv.zst",
]

for file in files:
    print("\n" + "=" * 80)
    print(file.name)

    df = pl.scan_csv(str(file))

    summary = (
        df.group_by("symbol")
        .agg(
            pl.len().alias("rows"),
            pl.col("ts_event").min().alias("first_ts"),
            pl.col("ts_event").max().alias("last_ts"),
            pl.col("volume").sum().alias("total_volume"),
        )
        .sort("first_ts")
        .collect()
    )

    print(summary)