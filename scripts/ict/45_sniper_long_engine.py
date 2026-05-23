# 45_sniper_long_engine.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
OUTDIR = ROOT / r"research_outputs\sniper_long_engine"

OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading features...")
df = (
    pl.read_parquet(FEATURES)
    .sort("ts_ct")
    .with_row_index("idx")
    .with_columns([
        pl.col("ts_ct").dt.year().alias("year"),
        pl.col("ts_ct").dt.strftime("%Y-%m").alias("month"),
    ])
)

signals = df.filter(
    (pl.col("hour_ct").is_in([8, 10])) &
    (
        ((pl.col("hour_ct") == 8) & (pl.col("minute_ct") >= 30) & (pl.col("minute_ct") <= 50)) |
        ((pl.col("hour_ct") == 10) & (pl.col("minute_ct") >= 10) & (pl.col("minute_ct") <= 50))
    ) &
    (pl.col("body_pct") >= 0.66) &
    (pl.col("body_pct") <= 0.98) &
    (pl.col("lower_wick_pct") <= 0.12) &
    (pl.col("bull_fvg") == False) &
    (pl.col("bear_displacement") == False) &
    (pl.col("any_liquidity_sweep_reclaim_last_30m") == True) &
    (pl.col("liquidity_sweep_reclaim_count_last_30m") >= 1) &
    (pl.col("dist_pdh") < -78) &
    (pl.col("dist_vwap") < -20)
)

print(f"Signals found: {signals.height}")

bars = df.select([
    "idx", "ts_ct", "year", "month", "hour_ct", "minute_ct",
    "open", "high", "low", "close"
]).to_dicts()

sig_rows = signals.select([
    "idx", "ts_ct", "year", "month", "hour_ct", "minute_ct",
    "open", "high", "low", "close",
    "body_pct", "lower_wick_pct", "upper_wick_pct",
    "dist_pdh", "dist_vwap",
    "liquidity_sweep_reclaim_count_last_30m",
]).to_dicts()

trades = []

for s in sig_rows:

    idx = int(s["idx"])
    entry_idx = idx + 1

    if entry_idx >= len(bars):
        continue

    entry_bar = bars[entry_idx]

    entry = float(entry_bar["open"])
    stop = float(s["low"]) - 2.0
    risk = entry - stop

    if risk <= 0 or risk > 25:
        continue

    target = entry + (2.0 * risk)

    result = None
    exit_reason = None
    bars_held = None

    for j in range(entry_idx, min(entry_idx + 90, len(bars))):

        high = float(bars[j]["high"])
        low = float(bars[j]["low"])

        if low <= stop:
            result = stop - entry
            exit_reason = "stop"
            bars_held = j - entry_idx + 1
            break

        if high >= target:
            result = target - entry
            exit_reason = "target"
            bars_held = j - entry_idx + 1
            break

    if result is None:
        last = bars[min(entry_idx + 89, len(bars)-1)]
        result = float(last["close"]) - entry
        exit_reason = "time"
        bars_held = 90

    trades.append({
        "signal_time": s["ts_ct"],
        "entry_time": entry_bar["ts_ct"],
        "year": entry_bar["year"],
        "month": entry_bar["month"],
        "hour": entry_bar["hour_ct"],
        "minute": entry_bar["minute_ct"],
        "side": "long",
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "risk": round(risk, 2),
        "result_points": round(result, 2),
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "body_pct": s["body_pct"],
        "lower_wick_pct": s["lower_wick_pct"],
        "upper_wick_pct": s["upper_wick_pct"],
        "dist_pdh": s["dist_pdh"],
        "dist_vwap": s["dist_vwap"],
        "sweep_count": s["liquidity_sweep_reclaim_count_last_30m"],
    })

trades_df = pl.DataFrame(trades)

wins = trades_df.filter(pl.col("result_points") > 0)
losses = trades_df.filter(pl.col("result_points") <= 0)

gross_win = wins["result_points"].sum() if wins.height else 0
gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

eq = (
    trades_df.sort("entry_time")
    .with_columns(pl.col("result_points").cum_sum().alias("equity"))
    .with_columns(pl.col("equity").cum_max().alias("peak"))
    .with_columns((pl.col("equity") - pl.col("peak")).alias("dd"))
)

summary = pl.DataFrame([{
    "trades": trades_df.height,
    "wins": wins.height,
    "losses": losses.height,
    "winrate": round(100 * wins.height / trades_df.height, 2) if trades_df.height else 0,
    "net_points": round(trades_df["result_points"].sum(), 2) if trades_df.height else 0,
    "avg_trade": round(trades_df["result_points"].mean(), 2) if trades_df.height else 0,
    "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    "max_dd": round(eq["dd"].min(), 2) if eq.height else 0,
}])

yearly = (
    trades_df.group_by("year")
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

signals.write_csv(OUTDIR / "sniper_long_signals.csv")
trades_df.write_csv(OUTDIR / "sniper_long_trades.csv")
summary.write_csv(OUTDIR / "sniper_long_summary.csv")
yearly.write_csv(OUTDIR / "sniper_long_yearly.csv")
trades_df.filter(pl.col("result_points") <= 0).write_csv(OUTDIR / "sniper_long_losers.csv")

print("\nSUMMARY")
print(summary)

print("\nYEARLY")
print(yearly)

print("\nLOSERS")
print(trades_df.filter(pl.col("result_points") <= 0))

print(f"\nSaved: {OUTDIR}")
