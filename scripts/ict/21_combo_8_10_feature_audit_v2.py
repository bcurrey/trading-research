# 21_combo_8_10_feature_audit_v2.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

TRADES = ROOT / r"research_outputs\combo_8_10_audit\combo_8_10_trades.csv"
FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
OUTDIR = ROOT / r"research_outputs\combo_8_10_feature_audit"

OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading combo trades...")
trades = pl.read_csv(TRADES, try_parse_dates=True)

print("Loading feature dataset...")
features = pl.read_parquet(FEATURES)

features_small_cols = [
    "ts_ct","trade_date_ct","hour_ct","minute_ct",
    "open","high","low","close","volume",
    "vwap","vwap_day","dist_vwap","dist_vwap_day",
    "ema_20","ema_50","dist_ema_20","dist_ema_50",
    "atr_14","atr_50","rel_vol_20","rel_vol_50",
    "bar_range","body","body_abs","body_pct",
    "upper_wick","lower_wick","upper_wick_pct","lower_wick_pct",
    "range_expansion_20","body_expansion_20",
    "bear_displacement","bear_fvg","bear_fvg_size",
    "bull_fvg","bull_fvg_size",
    "pdh","pdl","dol","ny_open",
    "dist_dol","dist_pdh","dist_pdl","dist_ny_open",
    "dist_asia_high","dist_asia_low",
    "dist_london_high","dist_london_low",
    "dist_or_high_30m","dist_or_low_30m",
    "any_liquidity_sweep_reclaim_last_30m",
    "liquidity_sweep_reclaim_count_last_30m",
    "recent_bear_fvg_last_120m",
    "inside_bear_fvg_current_bar",
    "bear_ifvg_proxy_fast",
    "any_ifvg_proxy_fast",
]

# remove duplicates while preserving order
features_small_cols = list(dict.fromkeys([
    c for c in features_small_cols if c in features.columns
]))

features_small = features.select(features_small_cols)

joined = trades.join(
    features_small,
    left_on="entry_time",
    right_on="ts_ct",
    how="left"
)

joined = joined.with_columns(
    pl.when(pl.col("result_points") > 0)
    .then(pl.lit("WIN"))
    .otherwise(pl.lit("LOSS"))
    .alias("outcome")
)

numeric_cols = [
    c for c, d in joined.schema.items()
    if d in [
        pl.Float64, pl.Float32,
        pl.Int64, pl.Int32, pl.Int16, pl.Int8,
        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8
    ]
]

compare_rows = []

for c in numeric_cols:
    try:
        win_mean = joined.filter(pl.col("outcome") == "WIN")[c].mean()
        loss_mean = joined.filter(pl.col("outcome") == "LOSS")[c].mean()

        if win_mean is None or loss_mean is None:
            continue

        compare_rows.append({
            "feature": c,
            "win_mean": round(float(win_mean), 4),
            "loss_mean": round(float(loss_mean), 4),
            "abs_diff": round(abs(float(loss_mean - win_mean)), 4),
        })
    except:
        pass

compare = (
    pl.DataFrame(compare_rows)
    .sort("abs_diff", descending=True)
)

candidate_rows = []

def evaluate(name, filt):

    x = joined.filter(filt)

    if x.height == 0:
        return

    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)

    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

    candidate_rows.append({
        "filter": name,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    })

if "dist_vwap" in joined.columns:
    evaluate("dist_vwap_gt_0", pl.col("dist_vwap") > 0)
    evaluate("dist_vwap_lt_0", pl.col("dist_vwap") < 0)

if "rel_vol_20" in joined.columns:
    evaluate("rel_vol_gt_1.25", pl.col("rel_vol_20") > 1.25)
    evaluate("rel_vol_gt_1.5", pl.col("rel_vol_20") > 1.5)

if "range_expansion_20" in joined.columns:
    evaluate("range_exp_gt_1.5", pl.col("range_expansion_20") > 1.5)

if "liquidity_sweep_reclaim_count_last_30m" in joined.columns:
    evaluate(
        "multi_sweep",
        pl.col("liquidity_sweep_reclaim_count_last_30m") >= 2
    )

candidate = (
    pl.DataFrame(candidate_rows)
    .sort(["losses","profit_factor"], descending=[False,True])
)

joined.write_csv(OUTDIR / "combo_8_10_trades_with_features.csv")
compare.write_csv(OUTDIR / "winner_loser_feature_compare.csv")
candidate.write_csv(OUTDIR / "candidate_filters.csv")

print("\nTOP FEATURE DIFFERENCES")
print(compare.head(30))

print("\nTOP FILTERS")
print(candidate)

print("\nLOSERS")
print(
    joined
    .filter(pl.col("outcome") == "LOSS")
    .select([
        "entry_time",
        "hour",
        "result_points",
        "risk_points",
        "mfe_points",
        "mae_points"
    ])
)

print(f"\nSaved folder: {OUTDIR}")
