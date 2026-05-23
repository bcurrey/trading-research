from pathlib import Path
import polars as pl
import time

INPUT = Path(r"D:\TradingResearch\data_raw\mbp1\2026-02\unzipped\glbx-mdp3-20260220.mbp-1.csv.zst")
OUTPUT_DIR = Path(r"D:\TradingResearch\data_parquet\mbp1\symbol=NQH6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "date=2026-02-20.parquet"

SYMBOL = "NQH6"

start = time.time()

print(f"\nConverting: {INPUT.name}")
print(f"Symbol: {SYMBOL}")
print(f"Output: {OUTPUT}\n")

lf = (
    pl.scan_csv(INPUT)
    .filter(pl.col("symbol") == SYMBOL)
    .select([
        "ts_recv",
        "ts_event",
        "action",
        "side",
        "depth",
        "price",
        "size",
        "flags",
        "sequence",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "bid_ct_00",
        "ask_ct_00",
        "symbol",
    ])
    .with_columns([
        pl.col("ts_event").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ").alias("ts_event_dt"),
        pl.col("ts_recv").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ").alias("ts_recv_dt"),
        (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("spread_points"),
        ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2).alias("mid_px"),
        (pl.col("ask_sz_00") - pl.col("bid_sz_00")).alias("top_size_imbalance"),
        (
            (pl.col("bid_sz_00") - pl.col("ask_sz_00")) /
            (pl.col("bid_sz_00") + pl.col("ask_sz_00"))
        ).alias("top_size_imbalance_ratio"),
    ])
    .select([
        "ts_event_dt",
        "ts_recv_dt",
        "symbol",
        "action",
        "side",
        "depth",
        "price",
        "size",
        "flags",
        "sequence",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
        "bid_ct_00",
        "ask_ct_00",
        "spread_points",
        "mid_px",
        "top_size_imbalance",
        "top_size_imbalance_ratio",
    ])
)

df = lf.collect(streaming=True)

print("\nRows converted:")
print(df.height)

print("\nPreview:")
print(df.head(5))

print("\nWriting parquet...")
df.write_parquet(
    OUTPUT,
    compression="zstd",
    statistics=True,
)

elapsed = time.time() - start

print(f"\nDONE in {elapsed:,.1f} seconds")
print(f"Saved: {OUTPUT}")