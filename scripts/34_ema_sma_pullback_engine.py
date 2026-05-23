import polars as pl
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "ema_sma_pullback_research"
OUT_DIR.mkdir(exist_ok=True)

print("\nEMA/SMA PULLBACK ENGINE")
print("Started:", datetime.now())

df_1m = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
}

MA_PAIRS = [
    (7, 14),
    (10, 20),
    (13, 26),
    (20, 40),
]

TARGETS = [25.0, 40.0, 60.0, 80.0]
MAX_HOLD_BARS = 24
MIN_RISK = 5.0
MAX_RISK = 60.0

all_trades = []

for tf_label, tf_every in TIMEFRAMES.items():

    print(f"\nBuilding {tf_label} candles...")

    df = (
        df_1m.group_by_dynamic(
            index_column="ts_event",
            every=tf_every,
            period=tf_every,
            closed="left",
        )
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("trade_date_ct").last().alias("trade_date_ct"),
            pl.col("hour_ct").last().alias("hour_ct"),
            pl.col("minute_ct").last().alias("minute_ct"),
            pl.col("minute_of_day_ct").last().alias("minute_of_day_ct"),
            pl.col("vwap_day").last().alias("vwap_day"),
            pl.col("atr_14").last().alias("atr_14"),
            pl.col("rel_vol_20").last().alias("rel_vol_20"),
            pl.col("or_high_30m").last().alias("or_high_30m"),
            pl.col("or_low_30m").last().alias("or_low_30m"),
            pl.col("premarket_high").last().alias("premarket_high"),
            pl.col("premarket_low").last().alias("premarket_low"),
            pl.col("prior_rth_high").last().alias("prior_rth_high"),
            pl.col("prior_rth_low").last().alias("prior_rth_low"),
        ])
        .drop_nulls()
        .sort("ts_event")
    )

    df = df.with_columns([
        (pl.col("high") - pl.col("low")).alias("bar_range"),
        (pl.col("close") - pl.col("open")).alias("body"),
        (pl.col("close") - pl.col("open")).abs().alias("body_abs"),
        pl.col("high").shift(1).alias("prior_high"),
        pl.col("low").shift(1).alias("prior_low"),
        pl.col("close").shift(1).alias("prior_close"),
    ])

    df = df.with_columns([
        pl.when(pl.col("bar_range") > 0)
        .then(pl.col("body_abs") / pl.col("bar_range"))
        .otherwise(None)
        .alias("body_pct"),
    ])

    df = df.with_columns([
        pl.when(pl.col("close") > pl.col("vwap_day"))
        .then(pl.lit("above_vwap"))
        .otherwise(pl.lit("below_vwap"))
        .alias("vwap_side"),

        pl.when(pl.col("atr_14") < 15)
        .then(pl.lit("low_atr"))
        .when(pl.col("atr_14") < 30)
        .then(pl.lit("mid_atr"))
        .otherwise(pl.lit("high_atr"))
        .alias("atr_bucket"),

        pl.when(pl.col("rel_vol_20") < 0.8)
        .then(pl.lit("low_vol"))
        .when(pl.col("rel_vol_20") < 1.2)
        .then(pl.lit("normal_vol"))
        .otherwise(pl.lit("high_vol"))
        .alias("volume_bucket"),

        pl.when(pl.col("body_pct") >= 0.65)
        .then(pl.lit("strong_body"))
        .when(pl.col("body_pct") >= 0.40)
        .then(pl.lit("medium_body"))
        .otherwise(pl.lit("weak_body"))
        .alias("body_bucket"),
    ])

    for ema_len, sma_len in MA_PAIRS:

        print(f"Testing {tf_label} EMA({ema_len}) / SMA({sma_len})...")

        ema_col = f"ema_{ema_len}"
        sma_col = f"sma_{sma_len}"

        df_pair = df.with_columns([
            pl.col("close").ewm_mean(span=ema_len, adjust=False).alias(ema_col),
            pl.col("close").rolling_mean(sma_len).alias(sma_col),
        ])

        df_pair = df_pair.with_columns([
            (pl.col(ema_col) - pl.col(ema_col).shift(3)).alias("ema_slope_3"),
            (pl.col(sma_col) - pl.col(sma_col).shift(3)).alias("sma_slope_3"),
        ])

        df_pair = df_pair.with_columns([
            (
                (pl.col("close") > pl.col(ema_col)) &
                (pl.col(ema_col) > pl.col(sma_col)) &
                (pl.col("ema_slope_3") >= 0) &
                (pl.col("sma_slope_3") >= 0)
            ).alias("trend_up"),

            (
                (pl.col("close") < pl.col(ema_col)) &
                (pl.col(ema_col) < pl.col(sma_col)) &
                (pl.col("ema_slope_3") <= 0) &
                (pl.col("sma_slope_3") <= 0)
            ).alias("trend_down"),
        ])

        df_pair = df_pair.with_columns([
            (
                pl.col("trend_up").shift(1).fill_null(False) &
                (pl.col("close") <= pl.col(ema_col)) &
                (pl.col("close") >= pl.col(sma_col))
            ).alias("long_pullback_between_ma"),

            (
                pl.col("trend_down").shift(1).fill_null(False) &
                (pl.col("close") >= pl.col(ema_col)) &
                (pl.col("close") <= pl.col(sma_col))
            ).alias("short_pullback_between_ma"),
        ])

        long_signals = (
            df_pair
            .filter(
                (pl.col("long_pullback_between_ma")) &
                (pl.col("minute_of_day_ct") >= 8 * 60 + 30) &
                (pl.col("minute_of_day_ct") <= 11 * 60 + 30)
            )
            .with_columns([
                pl.lit("long").alias("direction"),
                pl.lit(tf_label).alias("timeframe"),
                pl.lit(ema_len).alias("ema_len"),
                pl.lit(sma_len).alias("sma_len"),
            ])
        )

        short_signals = (
            df_pair
            .filter(
                (pl.col("short_pullback_between_ma")) &
                (pl.col("minute_of_day_ct") >= 8 * 60 + 30) &
                (pl.col("minute_of_day_ct") <= 11 * 60 + 30)
            )
            .with_columns([
                pl.lit("short").alias("direction"),
                pl.lit(tf_label).alias("timeframe"),
                pl.lit(ema_len).alias("ema_len"),
                pl.lit(sma_len).alias("sma_len"),
            ])
        )

        signals = pl.concat([long_signals, short_signals], how="diagonal_relaxed")

        print(f"Signals: {signals.height:,}")

        if signals.height == 0:
            continue

        for row in signals.iter_rows(named=True):
            entry_time = row["ts_event"]
            direction = row["direction"]
            entry = row["close"]

            sma_value = row[sma_col]

            if sma_value is None:
                continue

            if direction == "long":
                stop = min(row["low"], sma_value)
                risk = entry - stop
            else:
                stop = max(row["high"], sma_value)
                risk = stop - entry

            if risk <= 0:
                continue

            if risk < MIN_RISK or risk > MAX_RISK:
                continue

            future = (
                df_pair
                .filter(pl.col("ts_event") > entry_time)
                .head(MAX_HOLD_BARS)
                .select(["ts_event", "high", "low", "close", "prior_high", "prior_low", ema_col])
            )

            if future.height == 0:
                continue

            mech_status = "timeout"
            mech_exit = future[-1, "close"]

            for bar in future.iter_rows(named=True):
                if direction == "long":
                    if bar["low"] <= stop:
                        mech_status = "stop"
                        mech_exit = stop
                        break

                    if bar["prior_high"] is not None and bar["close"] > bar[ema_col] and bar["close"] > bar["prior_high"]:
                        mech_status = "mechanical_exit"
                        mech_exit = bar["close"]
                        break

                else:
                    if bar["high"] >= stop:
                        mech_status = "stop"
                        mech_exit = stop
                        break

                    if bar["prior_low"] is not None and bar["close"] < bar[ema_col] and bar["close"] < bar["prior_low"]:
                        mech_status = "mechanical_exit"
                        mech_exit = bar["close"]
                        break

            if direction == "long":
                mech_points = mech_exit - entry
                max_favorable = future.select(pl.col("high").max()).item() - entry
                max_adverse = entry - future.select(pl.col("low").min()).item()
            else:
                mech_points = entry - mech_exit
                max_favorable = entry - future.select(pl.col("low").min()).item()
                max_adverse = future.select(pl.col("high").max()).item() - entry

            trade = {
                "timeframe": tf_label,
                "ema_len": ema_len,
                "sma_len": sma_len,
                "direction": direction,
                "entry_time": entry_time,
                "trade_date_ct": row["trade_date_ct"],
                "hour_ct": row["hour_ct"],
                "minute_ct": row["minute_ct"],
                "entry": entry,
                "stop": stop,
                "risk_points": risk,

                "mechanical_status": mech_status,
                "mechanical_points": mech_points,
                "mechanical_r": mech_points / risk,

                "max_favorable": max_favorable,
                "max_adverse": max_adverse,

                "vwap_side": row["vwap_side"],
                "atr_bucket": row["atr_bucket"],
                "volume_bucket": row["volume_bucket"],
                "body_bucket": row["body_bucket"],
                "body_pct": row["body_pct"],
            }

            for target in TARGETS:
                status = "timeout"
                exit_price = future[-1, "close"]

                if direction == "long":
                    target_price = entry + target

                    for bar in future.iter_rows(named=True):
                        if bar["low"] <= stop:
                            status = "stop"
                            exit_price = stop
                            break

                        if bar["high"] >= target_price:
                            status = "target"
                            exit_price = target_price
                            break

                    points = exit_price - entry

                else:
                    target_price = entry - target

                    for bar in future.iter_rows(named=True):
                        if bar["high"] >= stop:
                            status = "stop"
                            exit_price = stop
                            break

                        if bar["low"] <= target_price:
                            status = "target"
                            exit_price = target_price
                            break

                    points = entry - exit_price

                trade[f"outcome_{int(target)}"] = status
                trade[f"points_{int(target)}"] = points
                trade[f"r_{int(target)}"] = points / risk

            all_trades.append(trade)

if not all_trades:
    print("No trades generated.")
    raise SystemExit

trades_df = pl.DataFrame(all_trades)

print(f"\nTotal trades: {trades_df.height:,}")

# -----------------------------
# Summaries
# -----------------------------

summary_rows = []

group_cols = [
    ["timeframe", "ema_len", "sma_len", "direction"],
    ["timeframe", "ema_len", "sma_len", "direction", "vwap_side"],
    ["timeframe", "ema_len", "sma_len", "direction", "atr_bucket"],
    ["timeframe", "ema_len", "sma_len", "direction", "volume_bucket"],
    ["timeframe", "ema_len", "sma_len", "direction", "body_bucket"],
]

for cols in group_cols:
    grouped = trades_df.group_by(cols).agg([
        pl.len().alias("trades"),
        (pl.col("mechanical_points") > 0).mean().alias("mechanical_win_rate"),
        pl.col("mechanical_points").mean().alias("mechanical_avg_points"),
        pl.col("mechanical_points").sum().alias("mechanical_total_points"),
        pl.col("mechanical_r").mean().alias("mechanical_avg_r"),
        pl.col("mechanical_r").sum().alias("mechanical_total_r"),
        pl.col("risk_points").mean().alias("avg_risk"),
        pl.col("max_favorable").mean().alias("avg_mfe"),
        pl.col("max_adverse").mean().alias("avg_mae"),
    ])

    grouped = grouped.with_columns([
        pl.lit(" | ".join(cols)).alias("grouping"),
    ])

    summary_rows.append(grouped)

summary_df = pl.concat(summary_rows, how="diagonal_relaxed")

summary_df = summary_df.with_columns([
    (pl.col("mechanical_win_rate") * 100).round(2).alias("mechanical_win_rate_pct"),
    pl.col("mechanical_avg_points").round(2),
    pl.col("mechanical_total_points").round(2),
    pl.col("mechanical_avg_r").round(3),
    pl.col("mechanical_total_r").round(2),
    pl.col("avg_risk").round(2),
    pl.col("avg_mfe").round(2),
    pl.col("avg_mae").round(2),
])

target_summaries = []

for target in [25, 40, 60, 80]:
    s = trades_df.group_by(["timeframe", "ema_len", "sma_len", "direction"]).agg([
        pl.len().alias("trades"),
        (pl.col(f"outcome_{target}") == "target").mean().alias("target_rate"),
        pl.col(f"points_{target}").mean().alias("avg_points"),
        pl.col(f"points_{target}").sum().alias("total_points"),
        pl.col(f"r_{target}").mean().alias("avg_r"),
        pl.col(f"r_{target}").sum().alias("total_r"),
        pl.col("risk_points").mean().alias("avg_risk"),
    ])

    s = s.with_columns([
        pl.lit(target).alias("target_points"),
        (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
        pl.col("avg_points").round(2),
        pl.col("total_points").round(2),
        pl.col("avg_r").round(3),
        pl.col("total_r").round(2),
        pl.col("avg_risk").round(2),
    ])

    target_summaries.append(s)

target_summary_df = pl.concat(target_summaries).sort("avg_points", descending=True)

print("\nTOP MECHANICAL EXIT RESULTS:")
print(
    summary_df
    .filter(pl.col("trades") >= 50)
    .sort("mechanical_avg_points", descending=True)
    .select([
        "grouping",
        "timeframe",
        "ema_len",
        "sma_len",
        "direction",
        "vwap_side",
        "atr_bucket",
        "volume_bucket",
        "body_bucket",
        "trades",
        "mechanical_win_rate_pct",
        "mechanical_avg_points",
        "mechanical_total_points",
        "mechanical_avg_r",
        "mechanical_total_r",
        "avg_risk",
        "avg_mfe",
        "avg_mae",
    ])
    .head(75)
)

print("\nTOP FIXED TARGET RESULTS:")
print(
    target_summary_df
    .filter(pl.col("trades") >= 50)
    .select([
        "timeframe",
        "ema_len",
        "sma_len",
        "direction",
        "target_points",
        "trades",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
    ])
    .head(75)
)

trades_file = OUT_DIR / "ema_sma_pullback_trades.csv"
summary_file = OUT_DIR / "ema_sma_pullback_mechanical_summary.csv"
target_file = OUT_DIR / "ema_sma_pullback_target_summary.csv"

trades_df.write_csv(trades_file)
summary_df.write_csv(summary_file)
target_summary_df.write_csv(target_file)

print("\nSaved:")
print(trades_file)
print(summary_file)
print(target_file)

print("\nFinished:", datetime.now())
print("DONE.")