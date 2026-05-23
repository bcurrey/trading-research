import polars as pl
from pathlib import Path

RAW_DIR = Path(r"D:\TradingResearch\data_raw")
OUTPUT_DIR = Path(r"D:\TradingResearch\data_parquet")

DATASETS = {
    "ES": [
        RAW_DIR / "ES-20180101-20260429.ohlcv-1m.csv.zst",
    ],
    "MNQ": [
        RAW_DIR / "MNQ-20180101-20260429.ohlcv-1m.csv.zst",
    ],
}

for market, files in DATASETS.items():
    print("\n" + "=" * 80)
    print(f"Building continuous dataset for {market}")
    print("=" * 80)

    dfs = []

    for file in files:
        print(f"Reading: {file.name}")
        df = pl.read_csv(file)
        dfs.append(df)

    df = pl.concat(dfs)

    print(f"Rows before filtering: {len(df):,}")

    df = df.filter(
        ~pl.col("symbol").str.contains("-")
    )

    print(f"Rows after removing spreads: {len(df):,}")

    df = df.with_columns(
        pl.col("ts_event").str.strptime(
            pl.Datetime,
            format="%Y-%m-%dT%H:%M:%S%.fZ",
            strict=False
        )
    )

    df = (
        df.sort(["ts_event", "volume"], descending=[False, True])
          .group_by("ts_event")
          .first()
          .sort("ts_event")
    )

    print(f"Rows after selecting dominant contract: {len(df):,}")

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

    output_file = OUTPUT_DIR / f"{market}_continuous_1m.parquet"

    print(f"\nWriting parquet file:\n{output_file}")

    df.write_parquet(output_file)

    print(f"\nDONE: {market}")