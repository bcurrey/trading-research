import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")

FILES = [
    "NQ_continuous_1m.parquet",
    "ES_continuous_1m.parquet",
    "MNQ_continuous_1m.parquet",
]

for file_name in FILES:
    print("\n" + "=" * 80)
    print(file_name)
    print("=" * 80)

    file_path = DATA_DIR / file_name

    df = pl.read_parquet(file_path)

    print(f"Rows: {len(df):,}")
    print(f"First timestamp: {df['ts_event'].min()}")
    print(f"Last timestamp:  {df['ts_event'].max()}")
    print(f"Unique contracts: {df['symbol'].n_unique()}")

    duplicate_count = (
        df.group_by("ts_event")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )

    print(f"Duplicate timestamps: {duplicate_count:,}")

    bad_price_rows = df.filter(
        (pl.col("high") < pl.col("low")) |
        (pl.col("open") <= 0) |
        (pl.col("high") <= 0) |
        (pl.col("low") <= 0) |
        (pl.col("close") <= 0)
    ).height

    print(f"Bad price rows: {bad_price_rows:,}")

    zero_volume_rows = df.filter(pl.col("volume") <= 0).height
    print(f"Zero/negative volume rows: {zero_volume_rows:,}")

    print("\nContracts used:")
    contract_summary = (
        df.group_by("symbol")
        .agg(
            pl.len().alias("rows"),
            pl.col("ts_event").min().alias("first_ts"),
            pl.col("ts_event").max().alias("last_ts"),
            pl.col("volume").sum().alias("total_volume"),
        )
        .sort("first_ts")
    )

    print(contract_summary)