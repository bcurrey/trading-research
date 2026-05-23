# 31_orb_break_2bar_orb_stop.py

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
OUTDIR = ROOT / r"research_outputs\orb_break_2bar_orb_stop"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("ORB Break + 2Bar Fib Entry + ORB Opposite-Side Stop")
print("Rules:")
print("- ORB window: 09:30-09:45 ET / 08:30-08:45 CT")
print("- ORB break must happen first")
print("- 2Bar setup occurs after ORB break")
print("- Keep filters: 65+ point body, 60%+ body, volume > prior candle, 15m bias match")
print("- Removed EMA 200 alignment filter")
print("- Entries tested: 0.382 and 0.50 retrace of signal candle")
print("- Stop loss: opposite side of ORB")
print("- Targets tested: 1R, 1.5R, 2R")
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

print("Calculating ORB and setup columns...")

bars5 = (
    bars5
    .with_columns([
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("volume").shift(1).alias("prev_volume"),
        (((pl.col("hour") == 8) & (pl.col("minute") >= 30) & (pl.col("minute") < 45))).alias("in_orb"),
        (((pl.col("hour") > 8) | ((pl.col("hour") == 8) & (pl.col("minute") >= 45))) & (pl.col("hour") < 15)).alias("after_orb"),
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

MIN_MOVE = 65.0
MIN_BODY_PCT = 60.0
MAX_HOLD_BARS = 24

bars5 = (
    bars5
    .with_columns([
        (pl.col("high") - pl.col("low")).clip(0.25, None).alias("bar_range"),
        (pl.col("close") - pl.col("open")).alias("body_signed"),
        (pl.col("close") - pl.col("open")).abs().alias("body_abs"),
    ])
    .with_columns([
        (pl.col("body_abs") / pl.col("bar_range") * 100).alias("body_pct"),
        (pl.col("open") - pl.col("close")).alias("bear_move"),
        (pl.col("close") - pl.col("open")).alias("bull_move"),
        (pl.col("volume") > pl.col("prev_volume")).alias("vol_spike"),
    ])
    .with_columns([
        (
            pl.col("after_orb")
            & pl.col("orb_broke_up")
            & (pl.col("bias15") == 1)
            & (pl.col("bull_move") >= MIN_MOVE)
            & (pl.col("body_pct") >= MIN_BODY_PCT)
            & pl.col("vol_spike")
        ).alias("long_signal"),
        (
            pl.col("after_orb")
            & pl.col("orb_broke_down")
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
        pl.when(pl.col("direction") == 1).then(pl.col("orb_low")).otherwise(pl.col("orb_high")).alias("stop"),
    ])
    .with_columns([
        pl.when(pl.col("direction") == 1).then(pl.col("entry_382") - pl.col("stop")).otherwise(pl.col("stop") - pl.col("entry_382")).alias("risk_382"),
        pl.when(pl.col("direction") == 1).then(pl.col("entry_50") - pl.col("stop")).otherwise(pl.col("stop") - pl.col("entry_50")).alias("risk_50"),
    ])
    .filter((pl.col("risk_382") > 0) & (pl.col("risk_50") > 0))
)

print("")
print("=" * 90)
print("SETUP COUNTS")
print("=" * 90)
print(f"Qualified signals: {signals.height:,}")
print(signals.group_by(["year", "side"]).agg(pl.len().alias("signals")).sort(["year", "side"]))

bars_rows = bars5.select([
    "ts", "trade_date", "year", "open", "high", "low", "close", "orb_high", "orb_low"
]).to_dicts()

signals_rows = signals.to_dicts()

print("")
print("Replaying trades...")

def replay_trade(signal_ts, trade_date, direction, entry, stop, risk, target_r):
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
            if direction == 1 and b["low"] <= entry:
                entered = True
                entry_time = b["ts"]
            elif direction == -1 and b["high"] >= entry:
                entered = True
                entry_time = b["ts"]
            else:
                continue

        bars_held += 1

        if direction == 1:
            mfe = max(mfe, b["high"] - entry)
            mae = min(mae, b["low"] - entry)
            stop_hit = b["low"] <= stop
            target_hit = b["high"] >= target
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
            pnl = (b["close"] - entry) * direction
            result = "WIN" if pnl > 0 else "LOSS"
            return result, pnl, entry_time, b["ts"], bars_held, target, mfe, mae, "time_exit"

    return "NO_ENTRY", 0.0, None, None, 0, target, 0.0, 0.0, "no_entry_or_no_exit"

records = []

for s in signals_rows:
    for fib_name, entry_col, risk_col in [
        ("fib382", "entry_382", "risk_382"),
        ("fib50", "entry_50", "risk_50"),
    ]:
        for target_r in [1.0, 1.5, 2.0]:
            result, pnl, entry_time, exit_time, bars_held, target, mfe, mae, exit_reason = replay_trade(
                s["ts"],
                s["trade_date"],
                s["direction"],
                s[entry_col],
                s["stop"],
                s[risk_col],
                target_r,
            )

            records.append({
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
            })

trades = pl.DataFrame(records)
completed = trades.filter(pl.col("result") != "NO_ENTRY")

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
            pl.median("risk_points").round(2).alias("median_risk"),
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
risk_buckets = (
    completed
    .with_columns([
        pl.when(pl.col("risk_points") <= 40).then(pl.lit("<=40"))
        .when(pl.col("risk_points") <= 60).then(pl.lit("40-60"))
        .when(pl.col("risk_points") <= 80).then(pl.lit("60-80"))
        .otherwise(pl.lit(">80"))
        .alias("risk_bucket")
    ])
)
risk_summary = (
    risk_buckets
    .with_columns([
        (pl.col("result") == "WIN").cast(pl.Int64).alias("is_win"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).alias("gross_loss"),
    ])
    .group_by(["fib_entry", "target_r", "side", "risk_bucket"])
    .agg([
        pl.len().alias("trades"),
        (pl.mean("is_win") * 100).round(2).alias("win_rate"),
        pl.sum("pnl_points").round(2).alias("net_points"),
        pl.mean("risk_points").round(2).alias("avg_risk"),
        pl.sum("gross_win").round(2).alias("gross_wins"),
        pl.sum("gross_loss").round(2).alias("gross_losses"),
    ])
    .with_columns([
        pl.when(pl.col("gross_losses") > 0)
        .then((pl.col("gross_wins") / pl.col("gross_losses")).round(2))
        .otherwise(None)
        .alias("profit_factor")
    ])
    .sort(["fib_entry", "target_r", "side", "risk_bucket"])
)

signals_path = OUTDIR / "signals.csv"
trades_path = OUTDIR / "trades.csv"
summary_path = OUTDIR / "summary.csv"
yearly_path = OUTDIR / "yearly.csv"
risk_path = OUTDIR / "risk_bucket_summary.csv"

signals.write_csv(signals_path)
trades.write_csv(trades_path)
summary.write_csv(summary_path)
yearly.write_csv(yearly_path)
risk_summary.write_csv(risk_path)

print("")
print("=" * 90)
print("TRADE COUNTS")
print("=" * 90)
print(f"Trade rows tested: {trades.height:,}")
print(f"Completed entries: {completed.height:,}")
print(f"No-entry/no-exit rows: {trades.height - completed.height:,}")

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

print("")
print("=" * 90)
print("RISK BUCKET SUMMARY")
print("=" * 90)
print(risk_summary)

print("")
print("=" * 90)
print("BEST CONFIGS, MIN 25 TRADES")
print("=" * 90)
best = (
    summary
    .filter(pl.col("trades") >= 25)
    .sort(["profit_factor", "win_rate", "net_points"], descending=[True, True, True])
)
print(best.head(20))

print("")
print("Saved files:")
print(f"- {signals_path}")
print(f"- {trades_path}")
print(f"- {summary_path}")
print(f"- {yearly_path}")
print(f"- {risk_path}")
print("Done.")
