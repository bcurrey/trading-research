# 29_trade_replay_export.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

BEST = ROOT / r"research_outputs\scaling_model_search\scaling_model_best_trades.csv"
FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
OUTDIR = ROOT / r"research_outputs\trade_replay_export"

OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading best trades...")
trades = pl.read_csv(BEST, try_parse_dates=True)
print(f"Trades loaded: {trades.height}")

print("Loading features...")
features = pl.read_parquet(FEATURES)
print(f"Feature rows loaded: {features.height:,}")

exports = []

for i, row in enumerate(trades.iter_rows(named=True), start=1):
    entry = row["entry_time"]

    start = entry - pl.duration(minutes=30)
    end = entry + pl.duration(minutes=90)

    clip = (
        features
        .filter((pl.col("ts_ct") >= start) & (pl.col("ts_ct") <= end))
        .with_columns(
            pl.lit(i).alias("trade_num"),
            pl.lit(str(entry)).alias("target_entry_time"),
            pl.lit(row["result_points"]).alias("trade_result_points"),
            pl.lit(row["entry_price"]).alias("trade_entry_price"),
            pl.lit(row["stop_price"]).alias("trade_stop_price"),
            pl.lit(row["target_price"]).alias("trade_target_price"),
        )
    )

    exports.append(clip)

if exports:
    replay = pl.concat(exports, how="vertical")
else:
    replay = pl.DataFrame()

replay.write_csv(OUTDIR / "best_model_replay_bars.csv")
trades.write_csv(OUTDIR / "best_model_trade_list.csv")

print(f"Saved replay bars: {OUTDIR / 'best_model_replay_bars.csv'}")
print(f"Saved trade list: {OUTDIR / 'best_model_trade_list.csv'}")
print("Done.")
