from pathlib import Path
import polars as pl

FILE = r"D:\TradingResearch\data_raw\mbp1\2026-02\unzipped\glbx-mdp3-20260220.mbp-1.csv.zst"

print("\nScanning file...\n")

lf = (
    pl.scan_csv(FILE)
    .with_columns([
        pl.col("ts_event").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ").alias("ts_event_dt"),
        pl.col("ts_recv").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ").alias("ts_recv_dt"),
    ])
    .with_columns([
        pl.col("symbol").str.contains("-").alias("is_spread"),
        (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("spread_points"),
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid_px"),
    ])
)

print("\nRows by symbol:\n")
print(
    lf.group_by("symbol")
    .agg(pl.len().alias("rows"))
    .sort("rows", descending=True)
    .collect()
)

print("\nOutright contracts only:\n")
print(
    lf.filter(~pl.col("is_spread"))
    .group_by("symbol")
    .agg([
        pl.len().alias("rows"),
        pl.min("ts_event_dt").alias("first_event"),
        pl.max("ts_event_dt").alias("last_event"),
        pl.mean("spread_points").alias("avg_spread"),
        pl.median("spread_points").alias("median_spread"),
        pl.mean("bid_sz_00").alias("avg_bid_sz"),
        pl.mean("ask_sz_00").alias("avg_ask_sz"),
    ])
    .sort("rows", descending=True)
    .collect()
)

print("\nDone.")