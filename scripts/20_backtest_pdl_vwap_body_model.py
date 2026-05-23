import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "setup_scans"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# ---------------------------------------------------
# Strategy filters
# ---------------------------------------------------

signals = df.filter(
    (pl.col("pdl_sweep_close_back_above")) &
    (pl.col("hour_ct").is_in([8, 9])) &
    (pl.col("close") > pl.col("vwap_day")) &
    (pl.col("body_pct") >= 0.50)
)

print("\nSignals found:", signals.height)

# ---------------------------------------------------
# Backtest assumptions
# Long entry at signal candle close
# Stop = signal candle low
# TP = 1R, 1.5R, 2R
# Max hold = 60 minutes
# ---------------------------------------------------

trades = []

for row in signals.iter_rows(named=True):
    entry_time = row["ts_event"]
    trade_date = row["trade_date_ct"]

    entry = row["close"]
    stop = row["low"]
    risk = entry - stop

    if risk <= 0:
        continue

    tp_1r = entry + risk
    tp_15r = entry + risk * 1.5
    tp_2r = entry + risk * 2.0

    future = (
        df
        .filter(
            (pl.col("ts_event") > entry_time) &
            (pl.col("ts_event") <= entry_time + pl.duration(minutes=60))
        )
        .select(["ts_event", "high", "low", "close"])
    )

    if future.height == 0:
        continue

    outcome_1r = "timeout"
    outcome_15r = "timeout"
    outcome_2r = "timeout"

    exit_1r = future[-1, "close"]
    exit_15r = future[-1, "close"]
    exit_2r = future[-1, "close"]

    exit_time_1r = future[-1, "ts_event"]
    exit_time_15r = future[-1, "ts_event"]
    exit_time_2r = future[-1, "ts_event"]

    for bar in future.iter_rows(named=True):
        t = bar["ts_event"]
        high = bar["high"]
        low = bar["low"]

        if outcome_1r == "timeout":
            if low <= stop:
                outcome_1r = "stop"
                exit_1r = stop
                exit_time_1r = t
            elif high >= tp_1r:
                outcome_1r = "target"
                exit_1r = tp_1r
                exit_time_1r = t

        if outcome_15r == "timeout":
            if low <= stop:
                outcome_15r = "stop"
                exit_15r = stop
                exit_time_15r = t
            elif high >= tp_15r:
                outcome_15r = "target"
                exit_15r = tp_15r
                exit_time_15r = t

        if outcome_2r == "timeout":
            if low <= stop:
                outcome_2r = "stop"
                exit_2r = stop
                exit_time_2r = t
            elif high >= tp_2r:
                outcome_2r = "target"
                exit_2r = tp_2r
                exit_time_2r = t

    trades.append({
        "entry_time": entry_time,
        "trade_date": trade_date,
        "entry": entry,
        "stop": stop,
        "risk_points": risk,

        "tp_1r": tp_1r,
        "outcome_1r": outcome_1r,
        "exit_1r": exit_1r,
        "r_1r": (exit_1r - entry) / risk,
        "exit_time_1r": exit_time_1r,

        "tp_15r": tp_15r,
        "outcome_15r": outcome_15r,
        "exit_15r": exit_15r,
        "r_15r": (exit_15r - entry) / risk,
        "exit_time_15r": exit_time_15r,

        "tp_2r": tp_2r,
        "outcome_2r": outcome_2r,
        "exit_2r": exit_2r,
        "r_2r": (exit_2r - entry) / risk,
        "exit_time_2r": exit_time_2r,

        "hour_ct": row["hour_ct"],
        "weekday_ct": row["weekday_ct"],
        "body_pct": row["body_pct"],
        "atr_14": row["atr_14"],
        "rel_vol_20": row["rel_vol_20"],
        "vwap_day": row["vwap_day"],
        "close": row["close"],
        "prior_rth_low": row["prior_rth_low"],
    })

trades_df = pl.DataFrame(trades)

print("\nTrades created:", trades_df.height)

# ---------------------------------------------------
# Summary
# ---------------------------------------------------

summary = trades_df.select([
    pl.len().alias("trades"),

    (pl.col("r_1r") > 0).mean().alias("win_rate_1r"),
    pl.col("r_1r").mean().alias("avg_r_1r"),
    pl.col("r_1r").sum().alias("total_r_1r"),

    (pl.col("r_15r") > 0).mean().alias("win_rate_15r"),
    pl.col("r_15r").mean().alias("avg_r_15r"),
    pl.col("r_15r").sum().alias("total_r_15r"),

    (pl.col("r_2r") > 0).mean().alias("win_rate_2r"),
    pl.col("r_2r").mean().alias("avg_r_2r"),
    pl.col("r_2r").sum().alias("total_r_2r"),
])

summary = summary.with_columns([
    (pl.col("win_rate_1r") * 100).round(2).alias("win_rate_1r_pct"),
    (pl.col("win_rate_15r") * 100).round(2).alias("win_rate_15r_pct"),
    (pl.col("win_rate_2r") * 100).round(2).alias("win_rate_2r_pct"),

    pl.col("avg_r_1r").round(3),
    pl.col("avg_r_15r").round(3),
    pl.col("avg_r_2r").round(3),

    pl.col("total_r_1r").round(2),
    pl.col("total_r_15r").round(2),
    pl.col("total_r_2r").round(2),
])

print("\nSUMMARY:")
print(summary)

print("\nOutcome counts:")
print(
    trades_df.select([
        pl.col("outcome_1r").value_counts().alias("outcome_1r_counts"),
        pl.col("outcome_15r").value_counts().alias("outcome_15r_counts"),
        pl.col("outcome_2r").value_counts().alias("outcome_2r_counts"),
    ])
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

trades_file = OUT_DIR / "pdl_vwap_body_backtest_trades.csv"
summary_file = OUT_DIR / "pdl_vwap_body_backtest_summary.csv"

trades_df.write_csv(trades_file)
summary.write_csv(summary_file)

print("\nSaved:")
print(trades_file)
print(summary_file)

print("\nDONE.")