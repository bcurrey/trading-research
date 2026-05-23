import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "htf_zones"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

TIMEFRAMES = {
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
}

all_zones = []

for label, every in TIMEFRAMES.items():
    print(f"\nBuilding {label} candles...")

    htf = (
        df.group_by_dynamic(
            index_column="ts_event",
            every=every,
            period=every,
            closed="left",
        )
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
        ])
        .drop_nulls()
        .sort("ts_event")
    )

    htf = htf.with_columns([
        pl.col("high").shift(2).alias("high_2"),
        pl.col("low").shift(2).alias("low_2"),
    ])

    htf = htf.with_columns([
        (pl.col("low") > pl.col("high_2")).alias("bull_fvg"),
        (pl.col("high") < pl.col("low_2")).alias("bear_fvg"),
    ])

    bull = (
        htf.filter(pl.col("bull_fvg"))
        .select([
            pl.col("ts_event").alias("zone_time"),
            pl.lit(label).alias("timeframe"),
            pl.lit("bullish").alias("fvg_type"),
            pl.col("high_2").alias("zone_low"),
            pl.col("low").alias("zone_high"),
        ])
    )

    bear = (
        htf.filter(pl.col("bear_fvg"))
        .select([
            pl.col("ts_event").alias("zone_time"),
            pl.lit(label).alias("timeframe"),
            pl.lit("bearish").alias("fvg_type"),
            pl.col("high").alias("zone_low"),
            pl.col("low_2").alias("zone_high"),
        ])
    )

    zones = pl.concat([bull, bear])

    zones = zones.with_columns([
        (pl.col("zone_high") - pl.col("zone_low")).alias("zone_size"),
    ])

    zones = zones.filter(pl.col("zone_size") >= 5)

    print(f"{label} zones: {zones.height:,}")

    zones.write_parquet(OUT_DIR / f"nq_{label}_fvg_zones.parquet")
    zones.write_csv(OUT_DIR / f"nq_{label}_fvg_zones.csv")

    all_zones.append(zones)

all_zones_df = pl.concat(all_zones).sort(["zone_time", "timeframe"])

all_zones_df.write_parquet(OUT_DIR / "nq_all_htf_fvg_zones.parquet")
all_zones_df.write_csv(OUT_DIR / "nq_all_htf_fvg_zones.csv")

print("\nAll HTF FVG zones:")
print(
    all_zones_df
    .group_by(["timeframe", "fvg_type"])
    .agg([
        pl.len().alias("zones"),
        pl.col("zone_size").mean().round(2).alias("avg_zone_size"),
    ])
    .sort(["timeframe", "fvg_type"])
)

print("\nSaved to:")
print(OUT_DIR)
print("\nDONE.")