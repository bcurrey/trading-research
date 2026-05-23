# 37_two_stage_signal_backtest.py FIXED

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

SIGNALS = ROOT / r"research_outputs\two_stage_sequence_detector\two_stage_signals.csv"
FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
OUTDIR = ROOT / r"research_outputs\two_stage_signal_backtest"

OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading signals...")
signals = pl.read_csv(SIGNALS, try_parse_dates=True)

print("Loading features...")
features = (
    pl.read_parquet(FEATURES)
    .sort("ts_ct")
    .with_row_index("idx")
    .with_columns([
        pl.col("ts_ct").dt.year().alias("year"),
        pl.col("ts_ct").dt.strftime("%Y-%m").alias("month"),
    ])
)

signals = signals.join(
    features.select(["idx", "ts_ct", "open", "high", "low", "close"]),
    left_on="signal_time",
    right_on="ts_ct",
    how="left"
)

bars = features.select([
    "ts_ct",
    "year",
    "month",
    "hour_ct",
    "minute_ct",
    "open",
    "high",
    "low",
    "close"
]).to_dicts()

trades = []

for s in signals.iter_rows(named=True):

    idx = s["idx"]

    if idx is None:
        continue

    entry_idx = idx + 1

    if entry_idx >= len(bars):
        continue

    entry_bar = bars[entry_idx]

    entry = float(entry_bar["open"])
    stop = max(float(s["high"]), float(entry_bar["high"])) + 2.0

    risk = stop - entry

    if risk <= 0 or risk > 30:
        continue

    target = entry - (2.0 * risk)

    result = None
    exit_reason = None
    bars_held = None

    for j in range(entry_idx, min(entry_idx + 90, len(bars))):

        high = float(bars[j]["high"])
        low = float(bars[j]["low"])

        if high >= stop:
            result = entry - stop
            exit_reason = "stop"
            bars_held = j - entry_idx + 1
            break

        if low <= target:
            result = entry - target
            exit_reason = "target"
            bars_held = j - entry_idx + 1
            break

    if result is None:
        last = bars[min(entry_idx + 89, len(bars)-1)]
        result = entry - float(last["close"])
        exit_reason = "time"
        bars_held = 90

    trades.append({
        "signal_time": s["signal_time"],
        "entry_time": entry_bar["ts_ct"],
        "year": entry_bar["year"],
        "month": entry_bar["month"],
        "hour": entry_bar["hour_ct"],
        "minute": entry_bar["minute_ct"],
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk": round(risk, 2),
        "result_points": round(result, 2),
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "stage1_lower_wick_pct": s["stage1_lower_wick_pct"],
        "stage1_range_expansion": s["stage1_range_expansion"],
        "stage2_body_pct": s["stage2_body_pct"],
        "stage2_lower_wick_pct": s["stage2_lower_wick_pct"],
        "dist_pdh": s["dist_pdh"],
        "dist_vwap": s["dist_vwap"],
    })

df = pl.DataFrame(trades)

wins = df.filter(pl.col("result_points") > 0)
losses = df.filter(pl.col("result_points") <= 0)

gross_win = wins["result_points"].sum() if wins.height else 0
gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

summary = pl.DataFrame([{
    "trades": df.height,
    "wins": wins.height,
    "losses": losses.height,
    "winrate": round(100 * wins.height / df.height, 2),
    "net_points": round(df["result_points"].sum(), 2),
    "avg_trade": round(df["result_points"].mean(), 2),
    "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
}])

yearly = (
    df.group_by("year")
    .agg(
        pl.len().alias("trades"),
        (pl.col("result_points") > 0).sum().alias("wins"),
        (pl.col("result_points") <= 0).sum().alias("losses"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .with_columns((100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"))
    .sort("year")
)

df.write_csv(OUTDIR / "two_stage_backtest_trades.csv")
summary.write_csv(OUTDIR / "two_stage_backtest_summary.csv")
yearly.write_csv(OUTDIR / "two_stage_backtest_yearly.csv")

print(summary)
print(yearly)

print(f"\nSaved: {OUTDIR}")
