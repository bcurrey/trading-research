from pathlib import Path
import re
import time
import polars as pl

RAW_DIR = Path(r"D:\TradingResearch\data_raw\mbp1\2026-02\unzipped")

TICK_OUT_DIR = Path(r"D:\TradingResearch\data_parquet\mbp1")
FEATURE_OUT_DIR = Path(r"D:\TradingResearch\data_parquet\mbp1_1m_features")

FRONT_SYMBOL = "NQH6"

FILES = sorted(RAW_DIR.glob("glbx-mdp3-202602*.mbp-1.csv.zst"))

def date_from_name(path: Path) -> str:
    m = re.search(r"(\d{8})", path.name)
    if not m:
        raise ValueError(f"Could not parse date from {path.name}")
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def convert_tick_file(input_file: Path, trade_date: str) -> Path:
    out_dir = TICK_OUT_DIR / f"symbol={FRONT_SYMBOL}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"date={trade_date}.parquet"

    if out_file.exists():
        print(f"  tick parquet exists, skipping: {out_file.name}")
        return out_file

    lf = (
        pl.scan_csv(input_file)
        .filter(pl.col("symbol") == FRONT_SYMBOL)
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

    if df.height == 0:
        print(f"  WARNING: no rows for {FRONT_SYMBOL}")
        return out_file

    df.write_parquet(out_file, compression="zstd", statistics=True)
    print(f"  tick rows: {df.height:,}")

    return out_file

def build_1m_features(tick_file: Path, trade_date: str) -> Path:
    out_dir = FEATURE_OUT_DIR / f"symbol={FRONT_SYMBOL}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"date={trade_date}.parquet"

    if out_file.exists():
        print(f"  1m features exist, skipping: {out_file.name}")
        return out_file

    lf = (
        pl.scan_parquet(tick_file)
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

    if df.height == 0:
        print("  WARNING: no 1m rows created")
        return out_file

    df.write_parquet(out_file, compression="zstd", statistics=True)
    print(f"  1m rows: {df.height:,}")

    return out_file

def main():
    print(f"\nFiles found: {len(FILES)}")
    print(f"Front symbol: {FRONT_SYMBOL}\n")

    total_start = time.time()

    for i, file in enumerate(FILES, start=1):
        trade_date = date_from_name(file)
        start = time.time()

        print(f"[{i}/{len(FILES)}] {trade_date} | {file.name}")

        try:
            tick_file = convert_tick_file(file, trade_date)
            build_1m_features(tick_file, trade_date)
        except Exception as e:
            print(f"  ERROR on {file.name}: {e}")

        elapsed = time.time() - start
        print(f"  finished in {elapsed:,.1f} sec\n")

    total_elapsed = time.time() - total_start
    print(f"DONE. Total time: {total_elapsed:,.1f} sec")

if __name__ == "__main__":
    main()