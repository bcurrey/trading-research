# 42_trace_sniper_source.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

SNIPER = ROOT / r"research_outputs\final_model_candidate\final_model_best_trades.csv"
BROAD = ROOT / r"research_outputs\exact_engine_rebuild\exact_engine_trades.csv"
FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\trace_sniper_source"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading sniper trades...")
sniper = pl.read_csv(SNIPER, try_parse_dates=True)

print("Loading broad trades...")
broad = pl.read_csv(BROAD, try_parse_dates=True)

print("Loading features...")
features = pl.read_parquet(FEATURES).sort("ts_ct").with_row_index("idx")

sniper_keys = sniper.select("entry_time").unique()

broad_labeled = (
    broad.join(
        sniper_keys.with_columns(pl.lit(1).alias("is_sniper")),
        on="entry_time",
        how="left"
    )
    .with_columns(pl.col("is_sniper").fill_null(0))
)

matched = broad_labeled.filter(pl.col("is_sniper") == 1)
not_matched = broad_labeled.filter(pl.col("is_sniper") == 0)

print(f"Sniper trades: {sniper.height}")
print(f"Broad trades: {broad.height}")
print(f"Matched sniper inside broad: {matched.height}")

# Join sniper to entry feature row
sniper_features = sniper.join(
    features,
    left_on="entry_time",
    right_on="ts_ct",
    how="left"
)

# Join broad to entry feature row
broad_features = broad_labeled.join(
    features,
    left_on="entry_time",
    right_on="ts_ct",
    how="left"
)

def summarize(x, label):
    if x.height == 0:
        return None

    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)

    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

    return {
        "group": label,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    }

summary = pl.DataFrame([
    summarize(sniper, "SNIPER_FILE"),
    summarize(broad, "BROAD_ENGINE"),
    summarize(matched, "MATCHED_IN_BROAD"),
    summarize(not_matched, "BROAD_NOT_SNIPER"),
])

# Feature comparison
candidate_cols = [
    "hour", "minute", "risk", "result_points",
    "stage1_lower_wick_pct", "stage1_range_expansion",
    "stage2_body_pct", "stage2_lower_wick_pct",
    "dist_pdh", "dist_vwap",

    "open", "high", "low", "close",
    "body_pct", "lower_wick_pct", "upper_wick_pct",
    "range_expansion_20", "rel_vol_20",
    "dist_dol", "dist_pdl",
    "dist_or_high_30m", "dist_or_low_30m",
    "liquidity_sweep_reclaim_count_last_30m",
    "bear_displacement", "bull_displacement",
    "bear_fvg", "bull_fvg",
]

rows = []

for c in candidate_cols:
    if c not in sniper_features.columns or c not in broad_features.columns:
        continue

    try:
        s_mean = sniper_features[c].mean()
        b_mean = broad_features.filter(pl.col("is_sniper") == 0)[c].mean()

        if s_mean is None or b_mean is None:
            continue

        rows.append({
            "feature": c,
            "sniper_mean": round(float(s_mean), 5),
            "broad_not_sniper_mean": round(float(b_mean), 5),
            "diff": round(float(s_mean - b_mean), 5),
            "abs_diff": round(abs(float(s_mean - b_mean)), 5),
        })

    except:
        pass

feature_diff = pl.DataFrame(rows).sort("abs_diff", descending=True)

summary.write_csv(OUTDIR / "trace_summary.csv")
sniper_features.write_csv(OUTDIR / "sniper_with_features.csv")
broad_features.write_csv(OUTDIR / "broad_with_features.csv")
feature_diff.write_csv(OUTDIR / "sniper_vs_broad_feature_diff.csv")

print("\nSUMMARY")
print(summary)

print("\nTOP FEATURE DIFFERENCES")
print(feature_diff.head(40))

print(f"\nSaved: {OUTDIR}")
