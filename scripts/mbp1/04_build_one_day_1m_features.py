from pathlib import Path
import polars as pl
import time

INPUT = Path(r"D:\TradingResearch\data_parquet\mbp1\symbol=NQH6\date=2026-02-20.parquet")
OUTPUT_DIR = Path(r"D:\TradingResearch\data_parquet\mbp1_1m_features\symbol=NQH6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "date=2026-02-20.parquet"

start = time.time()

print(f"\nBuilding 1m features from: {INPUT}\n")

lf = (
    pl.scan_parquet(INPUT)
    .with_columns([
        pl.col("ts_event_dt").dt.truncate("1m").alias("minute"),
        pl.col("mid_px").diff().alias("mid_change"),
        pl.col("bid_px_00").diff().alias("bid_change"),
        pl.col("ask_px_00").diff().alias("ask_change"),
    ])
    .with_columns([
        (pl.col("mid_change") > 0).cast(pl.Int8).alias("mid_up"),
        (pl.col("mid_change") < 0).cast(pl.Int8).alias("mid_down"),
        (pl.col("bid_change") > 0).cast(pl.Int8).alias("bid_lifted"),
        (pl.col("ask_change") < 0).cast(pl.Int8).alias("ask_hit"),
        (pl.col("spread_points") <= 1.0).cast(pl.Int8).alias("tight_spread"),
        (pl.col("top_size_imbalance_ratio") > 0.5).cast(pl.Int8).alias("bid_size_dominant"),
        (pl.col("top_size_imbalance_ratio") < -0.5).cast(pl.Int8).alias("ask_size_dominant"),
    ])
    .group_by("minute")
    .agg([
        pl.len().alias("updates"),

        pl.first("mid_px").alias("open_mid"),
        pl.max("mid_px").alias("high_mid"),
        pl.min("mid_px").alias("low_mid"),
        pl.last("mid_px").alias("close_mid"),

        pl.first("bid_px_00").alias("open_bid"),
        pl.last("bid_px_00").alias("close_bid"),
        pl.first("ask_px_00").alias("open_ask"),
        pl.last("ask_px_00").alias("close_ask"),

        pl.mean("spread_points").alias("avg_spread"),
        pl.median("spread_points").alias("median_spread"),
        pl.min("spread_points").alias("min_spread"),
        pl.max("spread_points").alias("max_spread"),

        pl.mean("bid_sz_00").alias("avg_bid_sz"),
        pl.mean("ask_sz_00").alias("avg_ask_sz"),
        pl.max("bid_sz_00").alias("max_bid_sz"),
        pl.max("ask_sz_00").alias("max_ask_sz"),

        pl.mean("top_size_imbalance_ratio").alias("avg_top_imbalance_ratio"),
        pl.sum("bid_size_dominant").alias("bid_dominant_updates"),
        pl.sum("ask_size_dominant").alias("ask_dominant_updates"),

        pl.sum("mid_up").alias("mid_up_updates"),
        pl.sum("mid_down").alias("mid_down_updates"),
        pl.sum("bid_lifted").alias("bid_lifted_updates"),
        pl.sum("ask_hit").alias("ask_hit_updates"),
        pl.sum("tight_spread").alias("tight_spread_updates"),
    ])
    .with_columns([
        (pl.col("close_mid") - pl.col("open_mid")).alias("mid_delta"),
        (pl.col("high_mid") - pl.col("low_mid")).alias("mid_range"),
        (pl.col("mid_up_updates") - pl.col("mid_down_updates")).alias("mid_update_delta"),
        (pl.col("bid_lifted_updates") - pl.col("ask_hit_updates")).alias("quote_pressure_delta"),
        (pl.col("bid_dominant_updates") - pl.col("ask_dominant_updates")).alias("size_pressure_delta"),
        (pl.col("tight_spread_updates") / pl.col("updates")).alias("tight_spread_pct"),
    ])
    .sort("minute")
)

df = lf.collect(streaming=True)

print("Rows:")
print(df.height)

print("\nPreview:")
print(df.head(10))

print("\nWriting parquet...")
df.write_parquet(
    OUTPUT,
    compression="zstd",
    statistics=True,
)

elapsed = time.time() - start

print(f"\nDONE in {elapsed:,.1f} seconds")
print(f"Saved: {OUTPUT}")