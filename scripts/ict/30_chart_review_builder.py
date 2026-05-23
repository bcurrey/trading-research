# 30_chart_review_builder.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

TRADES = ROOT / r"research_outputs\scaling_model_search\scaling_model_best_trades.csv"
FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\chart_review_builder"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading trades...")
trades = pl.read_csv(TRADES, try_parse_dates=True)

print("Loading features...")
features = pl.read_parquet(FEATURES)

review_rows = []

for row in trades.iter_rows(named=True):

    ts = row["entry_time"]

    context = (
        features
        .filter(
            (pl.col("ts_ct") >= ts - pl.duration(minutes=15)) &
            (pl.col("ts_ct") <= ts)
        )
        .sort("ts_ct")
    )

    if context.height == 0:
        continue

    latest = context.tail(1)

    review_rows.append({
        "entry_time": ts,
        "result_points": row["result_points"],
        "risk_points": row["risk_points"],
        "hour": row["hour"],
        "minute": row["minute"],

        "lower_wick_pct": latest["lower_wick_pct"][0] if "lower_wick_pct" in latest.columns else None,
        "body_pct": latest["body_pct"][0] if "body_pct" in latest.columns else None,
        "range_expansion_20": latest["range_expansion_20"][0] if "range_expansion_20" in latest.columns else None,
        "rel_vol_20": latest["rel_vol_20"][0] if "rel_vol_20" in latest.columns else None,

        "dist_pdh": latest["dist_pdh"][0] if "dist_pdh" in latest.columns else None,
        "dist_pdl": latest["dist_pdl"][0] if "dist_pdl" in latest.columns else None,
        "dist_dol": latest["dist_dol"][0] if "dist_dol" in latest.columns else None,
        "dist_vwap": latest["dist_vwap"][0] if "dist_vwap" in latest.columns else None,

        "dist_or_low_30m": latest["dist_or_low_30m"][0] if "dist_or_low_30m" in latest.columns else None,
        "dist_or_high_30m": latest["dist_or_high_30m"][0] if "dist_or_high_30m" in latest.columns else None,

        "liquidity_sweep_reclaim_count_last_30m":
            latest["liquidity_sweep_reclaim_count_last_30m"][0]
            if "liquidity_sweep_reclaim_count_last_30m" in latest.columns else None,

        "has_high_impact_usd_event":
            latest["has_high_impact_usd_event"][0]
            if "has_high_impact_usd_event" in latest.columns else None,
    })

review = pl.DataFrame(review_rows)

review.write_csv(OUTDIR / "chart_review_summary.csv")

wins = review.filter(pl.col("result_points") > 0)
losses = review.filter(pl.col("result_points") <= 0)

wins.write_csv(OUTDIR / "winning_examples.csv")
losses.write_csv(OUTDIR / "losing_examples.csv")

print(review)

print(f"\nSaved summary: {OUTDIR / 'chart_review_summary.csv'}")
print(f"Saved wins: {OUTDIR / 'winning_examples.csv'}")
print(f"Saved losses: {OUTDIR / 'losing_examples.csv'}")
