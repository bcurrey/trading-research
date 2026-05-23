from pathlib import Path
import re
import time
import polars as pl

RAW_ROOT = Path(r"D:\TradingResearch\data_raw")

TICK_OUT_DIR = Path(r"D:\TradingResearch\data_parquet\mbp1")
FEATURE_OUT_DIR = Path(r"D:\TradingResearch\data_parquet\mbp1_1m_features")

MONTH_FOLDERS = [
    RAW_ROOT / "mbp1" / "2026-02" / "unzipped",
    RAW_ROOT / "mbp1" / "2026-03" / "unzipped",
    RAW_ROOT / "mbp1" / "2026-04" / "unzipped",
]

def date_from_name(path: Path) -> str:
    m = re.search(r"(\d{8})", path.name)
    if not m:
        raise ValueError(f"Could not parse date from {path.name}")

    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def detect_front_contract(file_path: Path) -> str:

    scan = (
        pl.scan_csv(file_path)
        .filter(~pl.col("symbol").str.contains("-"))
        .group_by("symbol")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .limit(1)
        .collect()
    )

    if scan.height == 0:
        raise ValueError(f"No outright symbols found in {file_path.name}")

    return scan["symbol"][0]

def convert_tick_file(input_file: Path, trade_date: str, symbol: str) -> Path:

    out_dir = TICK_OUT_DIR / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"date={trade_date}.parquet"

    if out_file.exists():
        print(f"    tick parquet exists, skipping")
        return out_file

    lf = (
        pl.scan_csv(input_file)
        .filter(pl.col("symbol") == symbol)
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
            pl.col("ts_event")
            .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ")
            .alias("ts_event_dt"),

            pl.col("ts_recv")
            .str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.fZ")
            .alias("ts_recv_dt"),

            (pl.col("ask_px_00") - pl.col("bid_px_00"))
            .alias("spread_points"),

            (
                (pl.col("bid_px_00") + pl.col("ask_px_00")) / 2
            ).alias("mid_px"),

            (
                pl.col("ask_sz_00") - pl.col("bid_sz_00")
            ).alias("top_size_imbalance"),

            (
                (pl.col("bid_sz_00") - pl.col("ask_sz_00")) /
                (pl.col("bid_sz_00") + pl.col("ask_sz_00"))
            ).alias("top_size_imbalance_ratio"),
        ])
    )

    df = lf.collect(streaming=True)

    df.write_parquet(
        out_file,
        compression="zstd",
        statistics=True,
    )

    print(f"    tick rows: {df.height:,}")

    return out_file

def build_1m_features(tick_file: Path, trade_date: str, symbol: str):

    out_dir = FEATURE_OUT_DIR / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"date={trade_date}.parquet"

    if out_file.exists():
        print(f"    1m features exist, skipping")
        return

    lf = (
        pl.scan_parquet(tick_file)

        .with_columns([
            pl.col("ts_event_dt")
            .dt.truncate("1m")
            .alias("minute"),

            pl.col("mid_px").diff().alias("mid_change"),
            pl.col("bid_px_00").diff().alias("bid_change"),
            pl.col("ask_px_00").diff().alias("ask_change"),
        ])

        .with_columns([
            (pl.col("mid_change") > 0)
            .cast(pl.Int8)
            .alias("mid_up"),

            (pl.col("mid_change") < 0)
            .cast(pl.Int8)
            .alias("mid_down"),

            (pl.col("bid_change") > 0)
            .cast(pl.Int8)
            .alias("bid_lifted"),

            (pl.col("ask_change") < 0)
            .cast(pl.Int8)
            .alias("ask_hit"),

            (pl.col("spread_points") <= 1.0)
            .cast(pl.Int8)
            .alias("tight_spread"),

            (pl.col("top_size_imbalance_ratio") > 0.5)
            .cast(pl.Int8)
            .alias("bid_size_dominant"),

            (pl.col("top_size_imbalance_ratio") < -0.5)
            .cast(pl.Int8)
            .alias("ask_size_dominant"),
        ])

        .group_by("minute")

        .agg([
            pl.len().alias("updates"),

            pl.first("mid_px").alias("open_mid"),
            pl.max("mid_px").alias("high_mid"),
            pl.min("mid_px").alias("low_mid"),
            pl.last("mid_px").alias("close_mid"),

            pl.mean("spread_points").alias("avg_spread"),
            pl.median("spread_points").alias("median_spread"),

            pl.mean("bid_sz_00").alias("avg_bid_sz"),
            pl.mean("ask_sz_00").alias("avg_ask_sz"),

            pl.mean("top_size_imbalance_ratio")
            .alias("avg_top_imbalance_ratio"),

            pl.sum("mid_up").alias("mid_up_updates"),
            pl.sum("mid_down").alias("mid_down_updates"),

            pl.sum("bid_lifted")
            .alias("bid_lifted_updates"),

            pl.sum("ask_hit")
            .alias("ask_hit_updates"),
        ])

        .with_columns([
            (pl.col("close_mid") - pl.col("open_mid"))
            .alias("mid_delta"),

            (pl.col("high_mid") - pl.col("low_mid"))
            .alias("mid_range"),

            (
                pl.col("bid_lifted_updates") -
                pl.col("ask_hit_updates")
            ).alias("quote_pressure_delta"),
        ])

        .sort("minute")
    )

    df = lf.collect(streaming=True)

    df.write_parquet(
        out_file,
        compression="zstd",
        statistics=True,
    )

    print(f"    1m rows: {df.height:,}")

def process_file(file_path: Path):

    trade_date = date_from_name(file_path)

    print(f"\n{trade_date} | {file_path.name}")

    symbol = detect_front_contract(file_path)

    print(f"    detected front contract: {symbol}")

    tick_file = convert_tick_file(
        file_path,
        trade_date,
        symbol,
    )

    build_1m_features(
        tick_file,
        trade_date,
        symbol,
    )

def main():

    all_files = []

    for folder in MONTH_FOLDERS:
        files = sorted(folder.glob("*.csv.zst"))
        all_files.extend(files)

    print(f"\nTotal files found: {len(all_files)}\n")

    total_start = time.time()

    for i, file_path in enumerate(all_files, start=1):

        start = time.time()

        print(f"[{i}/{len(all_files)}]")

        try:
            process_file(file_path)

        except Exception as e:
            print(f"    ERROR: {e}")

        elapsed = time.time() - start

        print(f"    finished in {elapsed:,.1f} sec")

    total_elapsed = time.time() - total_start

    print(f"\nDONE")
    print(f"Total runtime: {total_elapsed:,.1f} sec")

if __name__ == "__main__":
    main()