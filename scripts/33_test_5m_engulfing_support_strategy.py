import polars as pl
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "engulfing_support_research"
OUT_DIR.mkdir(exist_ok=True)

print("\n5M ENGULFING SUPPORT STRATEGY TEST")
print("Started:", datetime.now())

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# -----------------------------
# Build 5m candles
# -----------------------------

m5 = (
    df.group_by_dynamic(
        index_column="ts_event",
        every="5m",
        period="5m",
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
        pl.col("atr_14").last().alias("atr_14_1m"),
        pl.col("prior_rth_high").last().alias("prior_rth_high"),
        pl.col("prior_rth_low").last().alias("prior_rth_low"),
        pl.col("premarket_high").last().alias("premarket_high"),
        pl.col("premarket_low").last().alias("premarket_low"),
        pl.col("or_high_30m").last().alias("or_high_30m"),
        pl.col("or_low_30m").last().alias("or_low_30m"),
    ])
    .drop_nulls()
    .sort("ts_event")
)

m5 = m5.with_columns([
    (pl.col("high") - pl.col("low")).alias("range"),
    (pl.col("close") - pl.col("open")).alias("body"),
    (pl.col("close") - pl.col("open")).abs().alias("body_abs"),
    (pl.col("high") - pl.max_horizontal(["open", "close"])).alias("upper_wick"),
    (pl.min_horizontal(["open", "close"]) - pl.col("low")).alias("lower_wick"),
    pl.col("volume").rolling_mean(20).alias("vol_avg_20"),
])

m5 = m5.with_columns([
    pl.when(pl.col("range") > 0)
    .then(pl.col("body_abs") / pl.col("range"))
    .otherwise(None)
    .alias("body_pct"),

    pl.when(pl.col("vol_avg_20") > 0)
    .then(pl.col("volume") / pl.col("vol_avg_20"))
    .otherwise(None)
    .alias("rel_vol_20"),

    pl.col("open").shift(1).alias("prev_open"),
    pl.col("close").shift(1).alias("prev_close"),
    pl.col("high").shift(1).alias("prev_high"),
    pl.col("low").shift(1).alias("prev_low"),
    pl.col("body").shift(1).alias("prev_body"),
    pl.col("volume").shift(1).alias("prev_volume"),
])

# -----------------------------
# Daily trend context
# "stocks trending up in recent weeks" translated to:
# daily close above 10-day and 20-day EMA, and 20-day slope positive
# -----------------------------

daily = (
    m5.group_by("trade_date_ct")
    .agg([
        pl.col("high").max().alias("day_high"),
        pl.col("low").min().alias("day_low"),
        pl.col("close").last().alias("day_close"),
    ])
    .sort("trade_date_ct")
)

daily = daily.with_columns([
    pl.col("day_close").ewm_mean(span=10, adjust=False).alias("daily_ema_10"),
    pl.col("day_close").ewm_mean(span=20, adjust=False).alias("daily_ema_20"),
])

daily = daily.with_columns([
    (pl.col("daily_ema_20") - pl.col("daily_ema_20").shift(5)).alias("daily_ema20_slope_5d"),
])

daily = daily.with_columns([
    (
        (pl.col("day_close") > pl.col("daily_ema_10")) &
        (pl.col("day_close") > pl.col("daily_ema_20")) &
        (pl.col("daily_ema20_slope_5d") > 0)
    ).alias("recent_weeks_uptrend"),
])

m5 = m5.join(
    daily.select(["trade_date_ct", "recent_weeks_uptrend", "daily_ema20_slope_5d"]),
    on="trade_date_ct",
    how="left",
)

# -----------------------------
# Intraday tested support
# Support = prior 5m swing low tested earlier same day
# -----------------------------

m5 = m5.with_columns([
    (
        (pl.col("low") < pl.col("low").shift(1)) &
        (pl.col("low") < pl.col("low").shift(-1))
    ).alias("swing_low"),
])

support_levels = (
    m5.filter(pl.col("swing_low"))
    .select([
        "trade_date_ct",
        pl.col("ts_event").alias("support_time"),
        pl.col("low").alias("support_price"),
    ])
)

signals = []

print("\nScanning 5m bullish engulfing at tested support...")

for row in m5.iter_rows(named=True):
    t = row["ts_event"]
    trade_date = row["trade_date_ct"]

    if row["minute_of_day_ct"] is None:
        continue

    if not (8 * 60 + 30 <= row["minute_of_day_ct"] <= 13 * 60):
        continue

    bullish_engulf = (
        row["prev_close"] is not None and
        row["prev_open"] is not None and
        row["prev_close"] < row["prev_open"] and
        row["close"] > row["open"] and
        row["open"] <= row["prev_close"] and
        row["close"] >= row["prev_open"]
    )

    if not bullish_engulf:
        continue

    prior_supports = support_levels.filter(
        (pl.col("trade_date_ct") == trade_date) &
        (pl.col("support_time") < t) &
        ((pl.col("support_price") - row["low"]).abs() <= 10)
    )

    if prior_supports.height == 0:
        continue

    nearest_support = (
        prior_supports
        .with_columns((pl.col("support_price") - row["low"]).abs().alias("dist"))
        .sort("dist")
        .head(1)
        .to_dicts()[0]
    )

    stop = min(row["low"], row["prev_low"])
    risk = row["close"] - stop

    if risk <= 0:
        continue

    signals.append({
        "ts_event": t,
        "trade_date_ct": trade_date,
        "entry": row["close"],
        "stop": stop,
        "risk_points": risk,
        "support_price": nearest_support["support_price"],
        "dist_to_support": abs(nearest_support["support_price"] - row["low"]),

        "hour_ct": row["hour_ct"],
        "minute_ct": row["minute_ct"],
        "minute_of_day_ct": row["minute_of_day_ct"],

        "recent_weeks_uptrend": row["recent_weeks_uptrend"],
        "daily_ema20_slope_5d": row["daily_ema20_slope_5d"],

        "above_vwap": row["close"] > row["vwap_day"],
        "dist_vwap": row["close"] - row["vwap_day"],

        "body_pct": row["body_pct"],
        "rel_vol_20": row["rel_vol_20"],
        "volume_vs_prev_red": row["volume"] / row["prev_volume"] if row["prev_volume"] and row["prev_volume"] > 0 else None,

        "near_pdl": abs(row["close"] - row["prior_rth_low"]) <= 20 if row["prior_rth_low"] is not None else False,
        "near_pmh": abs(row["close"] - row["premarket_high"]) <= 20 if row["premarket_high"] is not None else False,
        "near_pml": abs(row["close"] - row["premarket_low"]) <= 20 if row["premarket_low"] is not None else False,
        "inside_or": row["or_low_30m"] <= row["close"] <= row["or_high_30m"] if row["or_low_30m"] is not None and row["or_high_30m"] is not None else False,
    })

signals_df = pl.DataFrame(signals)

print("Signals found:", signals_df.height)

if signals_df.height == 0:
    raise SystemExit("No signals found.")

# -----------------------------
# Backtest
# Entry = 5m close
# Stop = low of engulfing setup
# Targets = 25/40/60/80
# Trail proxy = MFE/MAE over next 120m
# -----------------------------

TARGETS = [25.0, 40.0, 60.0, 80.0]
MAX_HOLD_MINUTES = 120

trades = []

for row in signals_df.iter_rows(named=True):
    entry_time = row["ts_event"]
    entry = row["entry"]
    stop = row["stop"]
    risk = row["risk_points"]

    if risk <= 0 or risk > 80:
        continue

    future = (
        m5.filter(
            (pl.col("ts_event") > entry_time) &
            (pl.col("ts_event") <= entry_time + pl.duration(minutes=MAX_HOLD_MINUTES))
        )
        .select(["ts_event", "high", "low", "close"])
    )

    if future.height == 0:
        continue

    result = dict(row)

    max_favorable = future.select(pl.col("high").max()).item() - entry
    max_adverse = entry - future.select(pl.col("low").min()).item()

    result["max_favorable_120m"] = max_favorable
    result["max_adverse_120m"] = max_adverse

    for target in TARGETS:
        status = "timeout"
        exit_price = future[-1, "close"]

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

        result[f"outcome_{int(target)}"] = status
        result[f"points_{int(target)}"] = points
        result[f"r_{int(target)}"] = points / risk

    trades.append(result)

trades_df = pl.DataFrame(trades)

print("Trades created:", trades_df.height)

# -----------------------------
# Summary helper
# -----------------------------

def summarize(label, data):
    if data.height < 20:
        return None

    rows = []

    for target in [25, 40, 60, 80]:
        rows.append(
            data.select([
                pl.lit(label).alias("segment"),
                pl.lit(target).alias("target_points"),
                pl.len().alias("trades"),
                (pl.col(f"outcome_{target}") == "target").mean().alias("target_rate"),
                pl.col(f"points_{target}").mean().alias("avg_points"),
                pl.col(f"points_{target}").sum().alias("total_points"),
                pl.col(f"r_{target}").mean().alias("avg_r"),
                pl.col(f"r_{target}").sum().alias("total_r"),
                pl.col("risk_points").mean().alias("avg_risk"),
                pl.col("max_favorable_120m").mean().alias("avg_mfe_120m"),
                pl.col("max_adverse_120m").mean().alias("avg_mae_120m"),
            ])
        )

    return pl.concat(rows)

summaries = []

base_summary = summarize("ALL", trades_df)
if base_summary is not None:
    summaries.append(base_summary)

segments = {
    "recent_weeks_uptrend": trades_df.filter(pl.col("recent_weeks_uptrend") == True),
    "not_recent_weeks_uptrend": trades_df.filter(pl.col("recent_weeks_uptrend") != True),
    "above_vwap": trades_df.filter(pl.col("above_vwap") == True),
    "below_vwap": trades_df.filter(pl.col("above_vwap") == False),
    "rel_vol_20_gt_1": trades_df.filter(pl.col("rel_vol_20") > 1),
    "volume_vs_prev_red_gt_1": trades_df.filter(pl.col("volume_vs_prev_red") > 1),
    "body_pct_gt_50": trades_df.filter(pl.col("body_pct") >= 0.50),
    "risk_8_to_40": trades_df.filter((pl.col("risk_points") >= 8) & (pl.col("risk_points") <= 40)),
    "near_pml": trades_df.filter(pl.col("near_pml") == True),
    "near_pdl": trades_df.filter(pl.col("near_pdl") == True),
    "inside_or": trades_df.filter(pl.col("inside_or") == True),
}

for name, seg in segments.items():
    s = summarize(name, seg)
    if s is not None:
        summaries.append(s)

summary_df = pl.concat(summaries)

summary_df = summary_df.with_columns([
    (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
    pl.col("avg_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_r").round(3),
    pl.col("total_r").round(2),
    pl.col("avg_risk").round(2),
    pl.col("avg_mfe_120m").round(2),
    pl.col("avg_mae_120m").round(2),
])

summary_df = summary_df.sort(["avg_points", "total_points"], descending=True)

print("\nTOP RESULTS:")
print(
    summary_df.select([
        "segment",
        "target_points",
        "trades",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "avg_mfe_120m",
        "avg_mae_120m",
    ]).head(50)
)

signals_file = OUT_DIR / "5m_engulfing_support_signals.csv"
trades_file = OUT_DIR / "5m_engulfing_support_trades.csv"
summary_file = OUT_DIR / "5m_engulfing_support_summary.csv"

signals_df.write_csv(signals_file)
trades_df.write_csv(trades_file)
summary_df.write_csv(summary_file)

print("\nSaved:")
print(signals_file)
print(trades_file)
print(summary_file)

print("\nFinished:", datetime.now())
print("DONE.")