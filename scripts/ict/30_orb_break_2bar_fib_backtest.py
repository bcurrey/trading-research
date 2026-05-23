# 30_orb_break_2bar_fib_backtest.py

from pathlib import Path
import sys
import polars as pl

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"D:\TradingResearch")
DATA = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
OUTDIR = ROOT / r"research_outputs\orb_break_2bar_fib_backtest"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("ORB Break + BSC 2Bar Fib Entry Backtest")
print("Rules:")
print("- ORB window: 09:30-09:45 ET / 08:30-08:45 CT")
print("- ORB break must happen first")
print("- Then 2-bar setup occurs after the ORB break")
print("- Test entries at 0.382 and 0.50 of signal candle")
print("- Max risk: 40 points")
print("- Bias: 15m candle bias")
print("- Base BSC 2Bar: 5m, strong body, min move, volume spike, EMA alignment")
print("=" * 90)

print(f"Loading: {DATA}")
df = pl.read_parquet(DATA)
print(f"Rows loaded: {df.height:,}")
print(f"Columns loaded: {len(df.columns)}")

def pick(cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

ts_col = pick(["ts_ct", "datetime_ct", "timestamp_ct", "ts", "datetime", "timestamp"])
open_col = pick(["open", "Open"])
high_col = pick(["high", "High"])
low_col = pick(["low", "Low"])
close_col = pick(["close", "Close"])
vol_col = pick(["volume", "Volume", "vol"])

missing = []
for name, col in [
    ("timestamp", ts_col),
    ("open", open_col),
    ("high", high_col),
    ("low", low_col),
    ("close", close_col),
    ("volume", vol_col),
]:
    if col is None:
        missing.append(name)

if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

df = (
    df
    .select([
        pl.col(ts_col).alias("ts"),
        pl.col(open_col).cast(pl.Float64).alias("open"),
        pl.col(high_col).cast(pl.Float64).alias("high"),
        pl.col(low_col).cast(pl.Float64).alias("low"),
        pl.col(close_col).cast(pl.Float64).alias("close"),
        pl.col(vol_col).cast(pl.Float64).alias("volume"),
    ])
    .sort("ts")
    .with_columns([
        pl.col("ts").dt.date().alias("trade_date"),
        pl.col("ts").dt.year().alias("year"),
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.minute().alias("minute"),
    ])
)

print("Building 5m bars...")
bars5 = (
    df
    .group_by_dynamic("ts", every="5m", period="5m", closed="left")
    .agg([
        pl.first("open").alias("open"),
        pl.max("high").alias("high"),
        pl.min("low").alias("low"),
        pl.last("close").alias("close"),
        pl.sum("volume").alias("volume"),
    ])
    .drop_nulls()
    .sort("ts")
    .with_columns([
        pl.col("ts").dt.date().alias("trade_date"),
        pl.col("ts").dt.year().alias("year"),
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.minute().alias("minute"),
    ])
)

print(f"5m bars: {bars5.height:,}")

print("Building 15m bias...")
bars15 = (
    df
    .group_by_dynamic("ts", every="15m", period="15m", closed="left")
    .agg([
        pl.first("open").alias("bias15_open"),
        pl.last("close").alias("bias15_close"),
    ])
    .drop_nulls()
    .sort("ts")
    .with_columns([
        pl.when(pl.col("bias15_close") > pl.col("bias15_open")).then(pl.lit(1))
        .when(pl.col("bias15_close") < pl.col("bias15_open")).then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("bias15")
    ])
)

bars5 = bars5.join_asof(
    bars15.select(["ts", "bias15"]),
    on="ts",
    strategy="backward",
)

print("Calculating EMA/VWAP/ORB...")
bars5 = (
    bars5
    .with_columns([
        pl.col("close").ewm_mean(span=200, adjust=False).alias("ema200"),
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("volume").shift(1).alias("prev_volume"),
    ])
    .with_columns([
        ((pl.col("close") * pl.col("volume")).cum_sum().over("trade_date") / pl.col("volume").cum_sum().over("trade_date")).alias("vwap"),
        (((pl.col("hour") == 8) & (pl.col("minute") >= 30) & (pl.col("minute") < 45))).alias("in_orb"),
        (((pl.col("hour") > 8) | ((pl.col("hour") == 8) & (pl.col("minute") >= 45))) & ((pl.col("hour") < 15))).alias("after_orb"),
    ])
)

orb = (
    bars5
    .filter(pl.col("in_orb"))
    .group_by("trade_date")
    .agg([
        pl.max("high").alias("orb_high"),
        pl.min("low").alias("orb_low"),
    ])
)

bars5 = bars5.join(orb, on="trade_date", how="left")

bars5 = (
    bars5
    .with_columns([
        ((pl.col("after_orb")) & (pl.col("close") > pl.col("orb_high"))).alias("orb_break_up_bar"),
        ((pl.col("after_orb")) & (pl.col("close") < pl.col("orb_low"))).alias("orb_break_down_bar"),
    ])
    .with_columns([
        pl.col("orb_break_up_bar").cum_max().over("trade_date").alias("orb_broke_up"),
        pl.col("orb_break_down_bar").cum_max().over("trade_date").alias("orb_broke_down"),
    ])
)

print("Generating 2Bar signals...")

MIN_MOVE = 65.0
MIN_BODY_PCT = 60.0
MAX_RISK = 40.0
MAX_HOLD_BARS = 24

bars5 = (
    bars5
    .with_columns([
        (pl.col("high") - pl.col("low")).clip(0.25, None).alias("range"),
        (pl.col("close") - pl.col("open")).alias("body_signed"),
        (pl.col("close") - pl.col("open")).abs().alias("body_abs"),
    ])
    .with_columns([
        (pl.col("body_abs") / pl.col("range") * 100).alias("body_pct"),
        (pl.col("open") - pl.col("close")).alias("bear_move"),
        (pl.col("close") - pl.col("open")).alias("bull_move"),
        (pl.col("volume") > pl.col("prev_volume")).alias("vol_spike"),
    ])
    .with_columns([
        (
            pl.col("after_orb")
            & pl.col("orb_broke_up")
            & (pl.col("close") > pl.col("ema200"))
            & (pl.col("bias15") == 1)
            & (pl.col("bull_move") >= MIN_MOVE)
            & (pl.col("body_pct") >= MIN_BODY_PCT)
            & pl.col("vol_spike")
        ).alias("long_signal"),
        (
            pl.col("after_orb")
            & pl.col("orb_broke_down")
            & (pl.col("close") < pl.col("ema200"))
            & (pl.col("bias15") == -1)
            & (pl.col("bear_move") >= MIN_MOVE)
            & (pl.col("body_pct") >= MIN_BODY_PCT)
            & pl.col("vol_spike")
        ).alias("short_signal"),
    ])
)

signals = (
    bars5
    .filter(pl.col("long_signal") | pl.col("short_signal"))
    .with_row_index("signal_id")
    .with_columns([
        pl.when(pl.col("long_signal")).then(pl.lit(1)).otherwise(pl.lit(-1)).alias("direction"),
        pl.when(pl.col("long_signal")).then(pl.lit("LONG")).otherwise(pl.lit("SHORT")).alias("side"),
    ])
    .with_columns([
        (pl.col("low") + (pl.col("high") - pl.col("low")) * 0.382).alias("long_entry_382"),
        (pl.col("low") + (pl.col("high") - pl.col("low")) * 0.500).alias("long_entry_50"),
        (pl.col("high") - (pl.col("high") - pl.col("low")) * 0.382).alias("short_entry_382"),
        (pl.col("high") - (pl.col("high") - pl.col("low")) * 0.500).alias("short_entry_50"),
    ])
    .with_columns([
        pl.when(pl.col("direction") == 1).then(pl.col("long_entry_382")).otherwise(pl.col("short_entry_382")).alias("entry_382"),
        pl.when(pl.col("direction") == 1).then(pl.col("long_entry_50")).otherwise(pl.col("short_entry_50")).alias("entry_50"),
        pl.when(pl.col("direction") == 1).then(pl.col("low")).otherwise(pl.col("high")).alias("stop"),
    ])
    .with_columns([
        (pl.when(pl.col("direction") == 1).then(pl.col("entry_382") - pl.col("stop")).otherwise(pl.col("stop") - pl.col("entry_382"))).alias("risk_382"),
        (pl.when(pl.col("direction") == 1).then(pl.col("entry_50") - pl.col("stop")).otherwise(pl.col("stop") - pl.col("entry_50"))).alias("risk_50"),
    ])
    .filter((pl.col("risk_382") > 0) & (pl.col("risk_50") > 0) & (pl.col("risk_50") <= MAX_RISK))
)

print(f"Qualified signals after risk filter: {signals.height:,}")

bars_for_replay = bars5.select(["ts", "trade_date", "year", "open", "high", "low", "close", "ema200", "vwap", "orb_high", "orb_low"])

bars_rows = bars_for_replay.to_dicts()
signals_rows = signals.to_dicts()

print("Replaying trades...")

def first_touch_after(signal_ts, trade_date, direction, entry, stop, risk, target_r=1.0):
    target = entry + direction * risk * target_r
    entered = False
    entry_time = None
    bars_held = 0
    mfe = 0.0
    mae = 0.0

    for b in bars_rows:
        if b["trade_date"] != trade_date:
            continue
        if b["ts"] <= signal_ts:
            continue

        if not entered:
            if direction == 1:
                if b["low"] <= entry:
                    entered = True
                    entry_time = b["ts"]
            else:
                if b["high"] >= entry:
                    entered = True
                    entry_time = b["ts"]
            if not entered:
                continue

        bars_held += 1

        if direction == 1:
            mfe = max(mfe, b["high"] - entry)
            mae = min(mae, b["low"] - entry)

            stop_hit = b["low"] <= stop
            target_hit = b["high"] >= target

            if stop_hit and target_hit:
                return "LOSS", -risk, entry_time, b["ts"], bars_held, target, mfe, mae, "both_same_bar_stop_first"
            if stop_hit:
                return "LOSS", -risk, entry_time, b["ts"], bars_held, target, mfe, mae, "stop"
            if target_hit:
                return "WIN", risk * target_r, entry_time, b["ts"], bars_held, target, mfe, mae, "target"

        else:
            mfe = max(mfe, entry - b["low"])
            mae = min(mae, entry - b["high"])

            stop_hit = b["high"] >= stop
            target_hit = b["low"] <= target

            if stop_hit and target_hit:
                return "LOSS", -risk, entry_time, b["ts"], bars_held, target, mfe, mae, "both_same_bar_stop_first"
            if stop_hit:
                return "LOSS", -risk, entry_time, b["ts"], bars_held, target, mfe, mae, "stop"
            if target_hit:
                return "WIN", risk * target_r, entry_time, b["ts"], bars_held, target, mfe, mae, "target"

        if bars_held >= MAX_HOLD_BARS:
            exit_px = b["close"]
            pnl = (exit_px - entry) * direction
            result = "WIN" if pnl > 0 else "LOSS"
            return result, pnl, entry_time, b["ts"], bars_held, target, mfe, mae, "time_exit"

    return "NO_ENTRY", 0.0, None, None, 0, target, 0.0, 0.0, "no_entry_or_no_exit"

trade_records = []

for s in signals_rows:
    for fib_name, entry_col, risk_col in [("fib382", "entry_382", "risk_382"), ("fib50", "entry_50", "risk_50")]:
        for target_r in [1.0, 1.5, 2.0]:
            result, pnl, entry_time, exit_time, bars_held, target, mfe, mae, exit_reason = first_touch_after(
                signal_ts=s["ts"],
                trade_date=s["trade_date"],
                direction=s["direction"],
                entry=s[entry_col],
                stop=s["stop"],
                risk=s[risk_col],
                target_r=target_r,
            )

            trade_records.append({
                "signal_id": s["signal_id"],
                "signal_time": s["ts"],
                "entry_time": entry_time,
                "exit_time": exit_time,
                "year": s["year"],
                "trade_date": s["trade_date"],
                "side": s["side"],
                "fib_entry": fib_name,
                "target_r": target_r,
                "entry_price": s[entry_col],
                "stop_price": s["stop"],
                "target_price": target,
                "risk_points": s[risk_col],
                "result": result,
                "pnl_points": pnl,
                "bars_held": bars_held,
                "mfe_points": mfe,
                "mae_points": mae,
                "exit_reason": exit_reason,
                "orb_high": s["orb_high"],
                "orb_low": s["orb_low"],
                "signal_open": s["open"],
                "signal_high": s["high"],
                "signal_low": s["low"],
                "signal_close": s["close"],
                "signal_volume": s["volume"],
                "body_pct": s["body_pct"],
                "bias15": s["bias15"],
                "ema200": s["ema200"],
                "vwap": s["vwap"],
            })

trades = pl.DataFrame(trade_records)

trades_path = OUTDIR / "orb_break_2bar_fib_trades.csv"
signals_path = OUTDIR / "orb_break_2bar_signals.csv"
summary_path = OUTDIR / "orb_break_2bar_summary.csv"
yearly_path = OUTDIR / "orb_break_2bar_yearly.csv"

signals.write_csv(signals_path)
trades.write_csv(trades_path)

completed = trades.filter(pl.col("result") != "NO_ENTRY")

print(f"Trade rows tested: {trades.height:,}")
print(f"Completed entries: {completed.height:,}")
print(f"No-entry/no-exit rows: {trades.height - completed.height:,}")

def summarize(group_cols):
    return (
        completed
        .with_columns([
            (pl.col("result") == "WIN").cast(pl.Int64).alias("is_win"),
            (pl.col("result") == "LOSS").cast(pl.Int64).alias("is_loss"),
            pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).alias("gross_win"),
            pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).alias("gross_loss"),
        ])
        .group_by(group_cols)
        .agg([
            pl.len().alias("trades"),
            pl.sum("is_win").alias("wins"),
            pl.sum("is_loss").alias("losses"),
            (pl.mean("is_win") * 100).round(2).alias("win_rate"),
            pl.sum("pnl_points").round(2).alias("net_points"),
            pl.mean("pnl_points").round(2).alias("avg_points"),
            pl.sum("gross_win").round(2).alias("gross_wins"),
            pl.sum("gross_loss").round(2).alias("gross_losses"),
            pl.mean("risk_points").round(2).alias("avg_risk"),
            pl.mean("mfe_points").round(2).alias("avg_mfe"),
            pl.mean("mae_points").round(2).alias("avg_mae"),
        ])
        .with_columns([
            pl.when(pl.col("gross_losses") > 0)
            .then((pl.col("gross_wins") / pl.col("gross_losses")).round(2))
            .otherwise(None)
            .alias("profit_factor")
        ])
        .sort(group_cols)
    )

summary = summarize(["fib_entry", "target_r", "side"])
yearly = summarize(["year", "fib_entry", "target_r", "side"])

summary.write_csv(summary_path)
yearly.write_csv(yearly_path)

print("")
print("=" * 90)
print("SUMMARY BY FIB / TARGET / SIDE")
print("=" * 90)
print(summary)

print("")
print("=" * 90)
print("YEARLY SUMMARY")
print("=" * 90)
print(yearly)

best = (
    summary
    .filter(pl.col("trades") >= 25)
    .sort(["profit_factor", "win_rate", "net_points"], descending=[True, True, True])
)

print("")
print("=" * 90)
print("BEST CONFIGS, MIN 25 TRADES")
print("=" * 90)
print(best.head(20))

print("")
print("Saved files:")
print(f"- {signals_path}")
print(f"- {trades_path}")
print(f"- {summary_path}")
print(f"- {yearly_path}")
print("Done.")
