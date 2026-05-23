# 34_entry_bar_anatomy.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

TRADES = ROOT / r"research_outputs\final_model_candidate\final_model_best_trades.csv"
FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\entry_bar_anatomy"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading trades...")
trades = pl.read_csv(TRADES, try_parse_dates=True)

print("Loading features...")
features = pl.read_parquet(FEATURES)

joined = trades.join(
    features,
    left_on="entry_time",
    right_on="ts_ct",
    how="left"
)

cols = [
    "entry_time",
    "result_points",

    "open","high","low","close",

    "body",
    "body_abs",
    "body_pct",

    "upper_wick",
    "lower_wick",
    "upper_wick_pct",
    "lower_wick_pct",

    "bar_range",
    "range_expansion_20",

    "rel_vol_20",

    "dist_pdh",
    "dist_pdl",
    "dist_dol",
    "dist_vwap",

    "bear_displacement",
    "bull_fvg",
    "bear_fvg",

    "any_liquidity_sweep_reclaim_last_30m",
    "liquidity_sweep_reclaim_count_last_30m",
]

cols = [c for c in cols if c in joined.columns]

out = joined.select(cols)

summary_rows = []

numeric_cols = [
    c for c, d in out.schema.items()
    if d in [
        pl.Float64, pl.Float32,
        pl.Int64, pl.Int32,
        pl.UInt64, pl.UInt32
    ]
]

for c in numeric_cols:

    if c == "result_points":
        continue

    try:
        summary_rows.append({
            "feature": c,
            "mean": round(float(out[c].mean()), 4),
            "median": round(float(out[c].median()), 4),
            "min": round(float(out[c].min()), 4),
            "max": round(float(out[c].max()), 4),
        })
    except:
        pass

summary = pl.DataFrame(summary_rows)

out.write_csv(OUTDIR / "entry_bar_dataset.csv")
summary.write_csv(OUTDIR / "entry_bar_summary.csv")

print(summary.sort("feature"))

print(f"\nSaved dataset: {OUTDIR / 'entry_bar_dataset.csv'}")
print(f"Saved summary: {OUTDIR / 'entry_bar_summary.csv'}")
