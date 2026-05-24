from pathlib import Path
from datetime import datetime
import traceback
import polars as pl

# rerun trigger: 2026-05-24 workflow retry
ROOT = Path(r"D:\TradingResearch")
OUT = ROOT / "research_outputs"
STUDY = OUT / "htf_fvg_study"
OUT.mkdir(parents=True, exist_ok=True)
STUDY.mkdir(parents=True, exist_ok=True)

STATUS = OUT / "run_status.txt"
SOURCE = STUDY / "62_2026_zone_comparison_scaleout_trades.csv"
SUMMARY = STUDY / "latest_4h_fvg_tp1_timing_summary.csv"
DETAIL = STUDY / "latest_4h_fvg_tp1_timing_detail.csv"
TEXT = STUDY / "latest_4h_fvg_tp1_timing_summary.txt"

def status(s, msg):
    STATUS.write_text(f"status={s}\ntimestamp={datetime.now()}\nmessage={msg}\n", encoding="utf-8")

try:
    print("4H FVG TP1 TIMING TEST")
    print(f"Reading: {SOURCE}")
    df = pl.read_csv(SOURCE, try_parse_dates=False)

    df = df.filter(pl.col("zone_type") == "4h_FVG")
    if df.height == 0:
        raise RuntimeError("No 4h_FVG rows found")

    if df.schema.get("tp1_hit") == pl.String:
        df = df.with_columns(pl.col("tp1_hit").str.to_lowercase().eq("true"))

    df = df.with_columns([
        pl.col("entry_time").str.strptime(pl.Datetime, strict=False).alias("entry_dt"),
        pl.col("tp1_time").str.strptime(pl.Datetime, strict=False).alias("tp1_dt"),
    ])

    detail = (
        df.filter(pl.col("tp1_hit") == True)
        .with_columns(((pl.col("tp1_dt") - pl.col("entry_dt")).dt.total_seconds() / 60).round(2).alias("minutes_to_tp1"))
        .select([
            "zone_type", "side", "trade_date", "entry_time", "tp1_time", "minutes_to_tp1",
            "entry", "tp1", "exit_reason", "tp1_hit", "tp2_hit", "tp3_hit",
            "pnl_dollars", "gross_points_3_contracts", "mfe_points_from_entry", "mae_points_from_entry"
        ])
        .sort("minutes_to_tp1")
    )

    total = df.height
    hit = detail.height
    miss = total - hit

    summary = detail.select([
        pl.lit("4h_FVG").alias("zone_type"),
        pl.lit(total).alias("total_trades"),
        pl.lit(hit).alias("tp1_hit_count"),
        pl.lit(miss).alias("tp1_miss_count"),
        (pl.lit(hit) / pl.lit(total) * 100).round(2).alias("tp1_hit_pct"),
        pl.col("minutes_to_tp1").mean().round(2).alias("avg_minutes_to_tp1"),
        pl.col("minutes_to_tp1").median().round(2).alias("median_minutes_to_tp1"),
        pl.col("minutes_to_tp1").min().round(2).alias("fastest_minutes_to_tp1"),
        pl.col("minutes_to_tp1").max().round(2).alias("slowest_minutes_to_tp1"),
    ])

    detail.write_csv(DETAIL)
    summary.write_csv(SUMMARY)

    r = summary.to_dicts()[0]
    text = "\n".join([
        "4H FVG TP1 Timing Summary",
        f"total_trades={r['total_trades']}",
        f"tp1_hit_count={r['tp1_hit_count']}",
        f"tp1_miss_count={r['tp1_miss_count']}",
        f"tp1_hit_pct={r['tp1_hit_pct']}",
        f"avg_minutes_to_tp1={r['avg_minutes_to_tp1']}",
        f"median_minutes_to_tp1={r['median_minutes_to_tp1']}",
        f"fastest_minutes_to_tp1={r['fastest_minutes_to_tp1']}",
        f"slowest_minutes_to_tp1={r['slowest_minutes_to_tp1']}",
        "",
    ])
    TEXT.write_text(text, encoding="utf-8")

    print(summary)
    print(detail)
    print(f"Saved: {SUMMARY}")
    print(f"Saved: {DETAIL}")
    print(f"Saved: {TEXT}")
    status("SUCCESS", "Calculated 4H FVG average time to TP1")

except Exception as e:
    print(traceback.format_exc())
    status("FAILED", f"{type(e).__name__}: {e}")
    raise
