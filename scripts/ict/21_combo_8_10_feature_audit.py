# 21_combo_8_10_feature_audit.py
# Join combo_8_10 trades back to full feature dataset and compare winners vs losers.

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

OUT_JOINED = OUTDIR / "combo_8_10_trades_with_features.csv"
OUT_WIN_LOSS = OUTDIR / "winner_loser_feature_compare.csv"
OUT_LOSERS = OUTDIR / "losers_with_features.csv"
OUT_CANDIDATE_FILTERS = OUTDIR / "candidate_loss_filters.csv"

print("Loading combo trades...")
trades = pl.read_csv(TRADES, try_parse_dates=True)

print(f"Trades loaded: {trades.height}")

print("Loading feature dataset...")
features = pl.read_parquet(FEATURES)

print(f"Feature rows loaded: {features.height:,}")

# Join on entry_time -> ts_ct.
# If exact join misses due dtype/time precision, fallback to signal_time.
features_small_cols = [
    "ts_ct", "trade_date_ct", "hour_ct", "minute_ct",
    "open", "high", "low", "close", "volume",
    "vwap", "vwap_day", "dist_vwap", "dist_vwap_day",
    "ema_20", "ema_50", "dist_ema_20", "dist_ema_50",
    "atr_14", "atr_50", "rel_vol_20", "rel_vol_50",
    "bar_range", "body", "body_abs", "body_pct",
    "upper_wick", "lower_wick", "upper_wick_pct", "lower_wick_pct",
    "range_expansion_20", "body_expansion_20",
    "bear_displacement", "bear_fvg", "bear_fvg_size", "bear_fvg_min_2pt",
    "bull_fvg", "bull_fvg_size", "bull_fvg_min_2pt",
    "pdh", "pdl", "dol", "vwap", "ny_open",
    "dist_dol", "dist_pdh", "dist_pdl", "dist_ny_open",
    "dist_asia_high", "dist_asia_low", "dist_london_high", "dist_london_low",
    "dist_or_high_30m", "dist_or_low_30m",
    "pdh_sweep_reclaim_last_30m", "pdl_sweep_reclaim_last_30m",
    "pmh_sweep_reclaim_last_30m", "pml_sweep_reclaim_last_30m",
    "asia_high_sweep_reclaim_last_30m", "asia_low_sweep_reclaim_last_30m",
    "london_high_sweep_reclaim_last_30m", "london_low_sweep_reclaim_last_30m",
    "any_liquidity_sweep_reclaim_last_30m",
    "liquidity_sweep_reclaim_count_last_30m",
    "recent_bear_fvg_last_120m", "inside_bear_fvg_current_bar",
    "bear_ifvg_proxy_fast", "any_ifvg_proxy_fast",
    "has_high_impact_usd_event", "has_cpi", "has_fomc", "has_nfp",
    "has_ppi", "has_pmi_ism", "has_jobs", "has_fed_speech",
]

features_small_cols = [c for c in features_small_cols if c in features.columns]

features_small = features.select(features_small_cols)

joined = trades.join(
    features_small,
    left_on="entry_time",
    right_on="ts_ct",
    how="left"
)

missing = joined.filter(pl.col("close").is_null()).height if "close" in joined.columns else joined.height
print(f"Missing feature joins: {missing}")

joined = joined.with_columns(
    pl.when(pl.col("result_points") > 0).then(pl.lit("WIN")).otherwise(pl.lit("LOSS")).alias("outcome")
)

feature_cols = [
    c for c in joined.columns
    if c not in [
        "signal_time", "entry_time", "trade_date", "year", "month", "hour", "minute",
        "side", "entry_price", "stop_price", "target_price", "result_points", "result_rr",
        "exit_reason", "bars_held", "fill_bars", "outcome"
    ]
]

numeric_cols = []
for c in feature_cols:
    if joined.schema.get(c) in [
        pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8,
        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8
    ]:
        numeric_cols.append(c)

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
            "diff_loss_minus_win": round(float(loss_mean - win_mean), 4),
        })
    except Exception:
        pass

compare = (
    pl.DataFrame(compare_rows)
    .with_columns(pl.col("diff_loss_minus_win").abs().alias("abs_diff"))
    .sort("abs_diff", descending=True)
)

# Candidate simple filters: test thresholds from observed loss/win distribution.
candidate_filters = []

def test_filter(name: str, filt):
    x = joined.filter(filt)
    if x.height == 0:
        return
    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)
    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0
    candidate_filters.append({
        "filter": name,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "removed_trades": joined.height - x.height,
    })

for col in [
    "dist_vwap", "dist_vwap_day", "dist_dol", "dist_pdh", "dist_pdl",
    "dist_or_high_30m", "dist_or_low_30m", "rel_vol_20", "range_expansion_20",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "liquidity_sweep_reclaim_count_last_30m"
]:
    if col in joined.columns:
        vals = joined.select(pl.col(col)).drop_nulls()
        if vals.height >= 10:
            q25 = vals[col].quantile(0.25)
            q50 = vals[col].quantile(0.50)
            q75 = vals[col].quantile(0.75)
            for qname, qv in [("gt_q25", q25), ("gt_q50", q50), ("gt_q75", q75)]:
                test_filter(f"{col}_{qname}_{round(float(qv), 4)}", pl.col(col) > qv)
            for qname, qv in [("lt_q25", q25), ("lt_q50", q50), ("lt_q75", q75)]:
                test_filter(f"{col}_{qname}_{round(float(qv), 4)}", pl.col(col) < qv)

candidate_df = (
    pl.DataFrame(candidate_filters)
    .sort(["losses", "profit_factor", "net_points"], descending=[False, True, True])
)

joined.write_csv(OUT_JOINED)
compare.write_csv(OUT_WIN_LOSS)
joined.filter(pl.col("outcome") == "LOSS").write_csv(OUT_LOSERS)
candidate_df.write_csv(OUT_CANDIDATE_FILTERS)

print("\nWINNER vs LOSER FEATURE DIFF TOP 30")
print(compare.head(30))

print("\nCANDIDATE LOSS FILTERS TOP 30")
print(candidate_df.head(30))

print("\nLOSERS")
cols = [c for c in [
    "entry_time", "hour", "result_points", "risk_points", "mfe_points", "mae_points",
    "dist_vwap", "dist_vwap_day", "dist_dol", "dist_or_low_30m", "rel_vol_20",
    "range_expansion_20", "liquidity_sweep_reclaim_count_last_30m"
] if c in joined.columns]
print(joined.filter(pl.col("outcome") == "LOSS").select(cols))

print(f"\nSaved: {OUT_JOINED}")
print(f"Saved: {OUT_WIN_LOSS}")
print(f"Saved: {OUT_LOSERS}")
print(f"Saved: {OUT_CANDIDATE_FILTERS}")
