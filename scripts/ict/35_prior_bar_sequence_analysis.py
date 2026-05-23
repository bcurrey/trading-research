# 35_prior_bar_sequence_analysis.py

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

OUTDIR = ROOT / r"research_outputs\prior_bar_sequence_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading trades...")
trades = pl.read_csv(TRADES, try_parse_dates=True)

print("Loading features...")
features = pl.read_parquet(FEATURES)

features = features.sort("ts_ct").with_row_index("idx")

joined = trades.join(
    features.select(["idx", "ts_ct"]),
    left_on="entry_time",
    right_on="ts_ct",
    how="left"
)

rows = []

for t in joined.iter_rows(named=True):

    idx = t["idx"]

    if idx is None or idx < 3:
        continue

    seq = features.slice(idx - 3, 4)

    if seq.height < 4:
        continue

    bars = seq.to_dicts()

    b1 = bars[0]
    b2 = bars[1]
    b3 = bars[2]
    b4 = bars[3]

    rows.append({

        "entry_time": t["entry_time"],
        "result_points": t["result_points"],

        "bar1_body_pct": b1.get("body_pct"),
        "bar2_body_pct": b2.get("body_pct"),
        "bar3_body_pct": b3.get("body_pct"),
        "entry_body_pct": b4.get("body_pct"),

        "bar1_lower_wick_pct": b1.get("lower_wick_pct"),
        "bar2_lower_wick_pct": b2.get("lower_wick_pct"),
        "bar3_lower_wick_pct": b3.get("lower_wick_pct"),
        "entry_lower_wick_pct": b4.get("lower_wick_pct"),

        "bar1_range_expansion": b1.get("range_expansion_20"),
        "bar2_range_expansion": b2.get("range_expansion_20"),
        "bar3_range_expansion": b3.get("range_expansion_20"),
        "entry_range_expansion": b4.get("range_expansion_20"),

        "bar1_close": b1.get("close"),
        "bar2_close": b2.get("close"),
        "bar3_close": b3.get("close"),
        "entry_close": b4.get("close"),

        "bar1_open": b1.get("open"),
        "bar2_open": b2.get("open"),
        "bar3_open": b3.get("open"),
        "entry_open": b4.get("open"),

        "entry_dist_pdh": b4.get("dist_pdh"),
        "entry_dist_vwap": b4.get("dist_vwap"),

        "entry_rel_vol": b4.get("rel_vol_20"),

        "entry_bear_displacement": b4.get("bear_displacement"),
        "entry_bull_fvg": b4.get("bull_fvg"),
        "entry_bear_fvg": b4.get("bear_fvg"),
    })

df = pl.DataFrame(rows)

summary_rows = []

numeric_cols = [
    c for c, d in df.schema.items()
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
            "mean": round(float(df[c].mean()), 4),
            "median": round(float(df[c].median()), 4),
            "min": round(float(df[c].min()), 4),
            "max": round(float(df[c].max()), 4),
        })
    except:
        pass

summary = pl.DataFrame(summary_rows)

df.write_csv(OUTDIR / "prior_bar_sequence_dataset.csv")
summary.write_csv(OUTDIR / "prior_bar_sequence_summary.csv")

print(summary.sort("feature"))

print(f"\nSaved dataset: {OUTDIR / 'prior_bar_sequence_dataset.csv'}")
print(f"Saved summary: {OUTDIR / 'prior_bar_sequence_summary.csv'}")
