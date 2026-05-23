import polars as pl
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "execution_models"
OUT_DIR.mkdir(exist_ok=True)

print("\nORB FAILURE EXECUTION MODEL")
print("Started:", datetime.now())

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# ---------------------------------------------------
# Add year/month and overnight levels if missing
# ---------------------------------------------------

df = df.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),
    pl.col("trade_date_ct").dt.month().alias("month"),
])

df = df.with_columns([
    (
        pl.when(pl.col("minute_of_day_ct") >= 17 * 60)
        .then(pl.col("trade_date_ct") + pl.duration(days=1))
        .otherwise(pl.col("trade_date_ct"))
    ).alias("overnight_trade_date")
])

overnight = (
    df
    .filter(
        (pl.col("minute_of_day_ct") >= 17 * 60) |
        (pl.col("minute_of_day_ct") < 8 * 60 + 30)
    )
    .group_by("overnight_trade_date")
    .agg([
        pl.col("high").max().alias("overnight_high"),
        pl.col("low").min().alias("overnight_low"),
        pl.col("open").first().alias("overnight_open"),
        pl.col("close").last().alias("overnight_close"),
    ])
    .rename({"overnight_trade_date": "trade_date_ct"})
)

df = df.join(overnight, on="trade_date_ct", how="left")

df = df.with_columns([
    (pl.col("overnight_high") - pl.col("overnight_low")).alias("overnight_range"),
    (pl.col("open") - pl.col("prior_rth_close")).alias("gap_from_prior_close"),
])

# ---------------------------------------------------
# Event window
# OR = 8:30-9:00 CT
# Entry/sweep window = 9:00-10:30 CT
# ---------------------------------------------------

event_window = df.filter(
    (pl.col("minute_of_day_ct") >= 9 * 60) &
    (pl.col("minute_of_day_ct") <= 10 * 60 + 30)
)

events = []

for trade_date in event_window.select("trade_date_ct").unique().to_series().to_list():
    day = event_window.filter(pl.col("trade_date_ct") == trade_date).sort("ts_event")

    if day.height == 0:
        continue

    first = day.row(0, named=True)

    or_high = first["or_high_30m"]
    or_low = first["or_low_30m"]

    if or_high is None or or_low is None:
        continue

    # first OR high break
    high_break = day.filter(
        (pl.col("high") > or_high) &
        (pl.col("close") > or_high)
    ).head(1)

    # first OR low break
    low_break = day.filter(
        (pl.col("low") < or_low) &
        (pl.col("close") < or_low)
    ).head(1)

    candidates = []

    if high_break.height > 0:
        r = high_break.to_dicts()[0]
        candidates.append(("OR High Breakout", "long", r))

    if low_break.height > 0:
        r = low_break.to_dicts()[0]
        candidates.append(("OR Low Breakdown", "short", r))

    if not candidates:
        continue

    # Use first OR break of the day, whichever came first
    candidates = sorted(candidates, key=lambda x: x[2]["ts_event"])
    event_type, breakout_direction, row = candidates[0]

    breakout_time = row["ts_event"]
    breakout_price = row["close"]

    future = df.filter(
        (pl.col("ts_event") > breakout_time) &
        (pl.col("ts_event") <= breakout_time + pl.duration(minutes=90))
    ).sort("ts_event")

    if future.height == 0:
        continue

    max_up = future.select(pl.col("high").max()).item() - breakout_price
    max_down = breakout_price - future.select(pl.col("low").min()).item()

    if breakout_direction == "long":
        reversed_40 = max_down >= 40
        reversed_50 = max_down >= 50
        continued_40 = max_up >= 40
        trap_direction = "short"
    else:
        reversed_40 = max_up >= 40
        reversed_50 = max_up >= 50
        continued_40 = max_down >= 40
        trap_direction = "long"

    # classify
    if reversed_50:
        outcome_class = "failed_breakout_reversal_50"
    elif reversed_40:
        outcome_class = "failed_breakout_reversal_40"
    elif continued_40:
        outcome_class = "successful_breakout_40"
    else:
        outcome_class = "chop_no_40"

    events.append({
        "trade_date_ct": trade_date,
        "year": row["year"],
        "month": row["month"],
        "event_type": event_type,
        "breakout_direction": breakout_direction,
        "trap_direction": trap_direction,
        "outcome_class": outcome_class,
        "breakout_time": breakout_time,
        "breakout_minute_ct": row["minute_of_day_ct"],
        "hour_ct": row["hour_ct"],
        "minute_ct": row["minute_ct"],
        "breakout_price": breakout_price,
        "or_high": or_high,
        "or_low": or_low,
        "or_range": row["or_range_30m"],
        "vwap_day": row["vwap_day"],
        "dist_vwap_day": row["dist_vwap_day"],
        "breakout_above_vwap": row["close"] > row["vwap_day"],
        "atr_14": row["atr_14"],
        "rel_vol_20": row["rel_vol_20"],
        "body_pct": row["body_pct"],
        "prior_rth_high": row["prior_rth_high"],
        "prior_rth_low": row["prior_rth_low"],
        "prior_rth_close": row["prior_rth_close"],
        "overnight_high": row["overnight_high"],
        "overnight_low": row["overnight_low"],
        "overnight_range": row["overnight_range"],
        "gap_from_prior_close": row["gap_from_prior_close"],
        "premarket_high": row["premarket_high"],
        "premarket_low": row["premarket_low"],
        "max_up_90m": max_up,
        "max_down_90m": max_down,
        "reversed_40": reversed_40,
        "reversed_50": reversed_50,
        "continued_40": continued_40,
    })

events_df = pl.DataFrame(events)

print("\nORB events found:", events_df.height)

# ---------------------------------------------------
# Buckets for condition comparison
# ---------------------------------------------------

events_df = events_df.with_columns([
    pl.when(pl.col("breakout_minute_ct") <= 9 * 60 + 15)
    .then(pl.lit("first_15m_after_or"))
    .when(pl.col("breakout_minute_ct") <= 9 * 60 + 30)
    .then(pl.lit("first_30m_after_or"))
    .otherwise(pl.lit("after_930"))
    .alias("breakout_time_bucket"),

    pl.when(pl.col("or_range") < 35)
    .then(pl.lit("small_or"))
    .when(pl.col("or_range") < 70)
    .then(pl.lit("normal_or"))
    .otherwise(pl.lit("large_or"))
    .alias("or_range_bucket"),

    pl.when(pl.col("overnight_range") < 80)
    .then(pl.lit("small_on_range"))
    .when(pl.col("overnight_range") < 180)
    .then(pl.lit("normal_on_range"))
    .otherwise(pl.lit("large_on_range"))
    .alias("overnight_range_bucket"),

    pl.when(pl.col("gap_from_prior_close") > 30)
    .then(pl.lit("gap_up_30_plus"))
    .when(pl.col("gap_from_prior_close") < -30)
    .then(pl.lit("gap_down_30_plus"))
    .otherwise(pl.lit("small_gap"))
    .alias("gap_bucket"),

    pl.when(pl.col("atr_14") < 15)
    .then(pl.lit("low_atr"))
    .when(pl.col("atr_14") < 30)
    .then(pl.lit("mid_atr"))
    .otherwise(pl.lit("high_atr"))
    .alias("atr_bucket"),

    pl.when(pl.col("rel_vol_20") < 0.8)
    .then(pl.lit("low_vol"))
    .when(pl.col("rel_vol_20") < 1.5)
    .then(pl.lit("normal_vol"))
    .otherwise(pl.lit("high_vol"))
    .alias("volume_bucket"),

    pl.when(pl.col("body_pct") >= 0.65)
    .then(pl.lit("strong_body"))
    .when(pl.col("body_pct") >= 0.40)
    .then(pl.lit("medium_body"))
    .otherwise(pl.lit("weak_body"))
    .alias("body_bucket"),

    pl.when(pl.col("breakout_above_vwap"))
    .then(pl.lit("above_vwap"))
    .otherwise(pl.lit("below_vwap"))
    .alias("vwap_side"),

    pl.when(pl.col("breakout_price") > pl.col("overnight_high"))
    .then(pl.lit("above_overnight_high"))
    .when(pl.col("breakout_price") < pl.col("overnight_low"))
    .then(pl.lit("below_overnight_low"))
    .otherwise(pl.lit("inside_overnight_range"))
    .alias("overnight_position"),

    pl.when(pl.col("breakout_price") > pl.col("prior_rth_high"))
    .then(pl.lit("above_pdh"))
    .when(pl.col("breakout_price") < pl.col("prior_rth_low"))
    .then(pl.lit("below_pdl"))
    .otherwise(pl.lit("inside_prior_day_range"))
    .alias("prior_day_position"),
])

# ---------------------------------------------------
# Condition comparison: failed reversals vs successful breakouts
# ---------------------------------------------------

condition_cols = [
    "event_type",
    "breakout_time_bucket",
    "or_range_bucket",
    "overnight_range_bucket",
    "gap_bucket",
    "atr_bucket",
    "volume_bucket",
    "body_bucket",
    "vwap_side",
    "overnight_position",
    "prior_day_position",
]

condition_rows = []

for col in condition_cols:
    result = (
        events_df
        .group_by(col)
        .agg([
            pl.len().alias("events"),
            (pl.col("outcome_class").str.contains("failed_breakout_reversal")).mean().alias("failure_rate"),
            (pl.col("outcome_class") == "successful_breakout_40").mean().alias("success_rate"),
            pl.col("max_up_90m").mean().alias("avg_max_up_90m"),
            pl.col("max_down_90m").mean().alias("avg_max_down_90m"),
        ])
        .with_columns([
            pl.lit(col).alias("condition"),
            (pl.col("failure_rate") * 100).round(2).alias("failure_rate_pct"),
            (pl.col("success_rate") * 100).round(2).alias("success_rate_pct"),
            pl.col("avg_max_up_90m").round(2),
            pl.col("avg_max_down_90m").round(2),
        ])
        .rename({col: "condition_value"})
    )

    condition_rows.append(result)

condition_df = pl.concat(condition_rows, how="diagonal_relaxed")

condition_df = condition_df.sort(["failure_rate_pct", "events"], descending=True)

print("\nTOP FAILURE CONDITIONS:")
print(
    condition_df
    .filter(pl.col("events") >= 50)
    .select([
        "condition",
        "condition_value",
        "events",
        "failure_rate_pct",
        "success_rate_pct",
        "avg_max_up_90m",
        "avg_max_down_90m",
    ])
    .head(50)
)

# ---------------------------------------------------
# Backtest simple ORB before/after
# Before:
#   follow breakout direction
# After:
#   follow breakout only when simple continuation filters pass
#   fade breakout when simple failure filters pass
# ---------------------------------------------------

def simulate_event(row, mode):
    """
    mode:
    follow = trade breakout direction
    fade = trade opposite direction
    """

    entry_time = row["breakout_time"]
    entry = row["breakout_price"]

    if mode == "follow":
        direction = row["breakout_direction"]
    else:
        direction = row["trap_direction"]

    # Practical execution stop:
    # follow long stop = OR midpoint or opposite OR side, whichever closer but at least 15 points
    # fade short stop = breakout candle extreme approximated by breakout price +/- 25
    or_mid = (row["or_high"] + row["or_low"]) / 2

    if direction == "long":
        stop = min(or_mid, entry - 15)
        risk = entry - stop
        target = entry + 50
    else:
        stop = max(or_mid, entry + 15)
        risk = stop - entry
        target = entry - 50

    if risk <= 0 or risk > 80:
        return None

    future = df.filter(
        (pl.col("ts_event") > entry_time) &
        (pl.col("ts_event") <= entry_time + pl.duration(minutes=90))
    ).sort("ts_event")

    if future.height == 0:
        return None

    status = "timeout"
    exit_price = future[-1, "close"]

    for bar in future.iter_rows(named=True):
        if direction == "long":
            if bar["low"] <= stop:
                status = "stop"
                exit_price = stop
                break
            if bar["high"] >= target:
                status = "target"
                exit_price = target
                break
        else:
            if bar["high"] >= stop:
                status = "stop"
                exit_price = stop
                break
            if bar["low"] <= target:
                status = "target"
                exit_price = target
                break

    points = exit_price - entry if direction == "long" else entry - exit_price

    return {
        "trade_date_ct": row["trade_date_ct"],
        "year": row["year"],
        "event_type": row["event_type"],
        "mode": mode,
        "direction": direction,
        "entry_time": entry_time,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_points": risk,
        "status": status,
        "points": points,
        "r": points / risk,
        "outcome_class": row["outcome_class"],
        "breakout_time_bucket": row["breakout_time_bucket"],
        "or_range_bucket": row["or_range_bucket"],
        "overnight_range_bucket": row["overnight_range_bucket"],
        "gap_bucket": row["gap_bucket"],
        "atr_bucket": row["atr_bucket"],
        "volume_bucket": row["volume_bucket"],
        "body_bucket": row["body_bucket"],
        "vwap_side": row["vwap_side"],
        "overnight_position": row["overnight_position"],
        "prior_day_position": row["prior_day_position"],
    }

trade_rows = []

for row in events_df.iter_rows(named=True):
    # Baseline always follows breakout
    base_trade = simulate_event(row, "follow")
    if base_trade:
        base_trade["strategy"] = "baseline_follow_all"
        trade_rows.append(base_trade)

    # Filtered continuation:
    # Follow only if volume decent, not weak body, not first 15m fakeout context
    follow_filter = (
        row["volume_bucket"] != "low_vol" and
        row["body_bucket"] != "weak_body" and
        row["or_range_bucket"] != "large_or"
    )

    if follow_filter:
        t = simulate_event(row, "follow")
        if t:
            t["strategy"] = "filtered_follow"
            trade_rows.append(t)

    # Fade filters:
    # Fade when failed-breakout-prone conditions are present
    fade_filter = (
        row["breakout_time_bucket"] == "first_15m_after_or" or
        row["volume_bucket"] == "low_vol" or
        row["or_range_bucket"] == "large_or" or
        (
            row["event_type"] == "OR High Breakout" and
            row["vwap_side"] == "below_vwap"
        ) or
        (
            row["event_type"] == "OR Low Breakdown" and
            row["vwap_side"] == "above_vwap"
        )
    )

    if fade_filter:
        t = simulate_event(row, "fade")
        if t:
            t["strategy"] = "filtered_fade"
            trade_rows.append(t)

trades_df = pl.DataFrame(trade_rows)

summary_df = (
    trades_df
    .group_by("strategy")
    .agg([
        pl.len().alias("trades"),
        (pl.col("status") == "target").mean().alias("target_rate"),
        pl.col("points").mean().alias("avg_points"),
        pl.col("points").sum().alias("total_points"),
        pl.col("r").mean().alias("avg_r"),
        pl.col("r").sum().alias("total_r"),
        pl.col("risk_points").mean().alias("avg_risk"),
    ])
    .with_columns([
        (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
        pl.col("avg_points").round(2),
        pl.col("total_points").round(2),
        pl.col("avg_r").round(3),
        pl.col("total_r").round(2),
        pl.col("avg_risk").round(2),
    ])
    .sort("avg_points", descending=True)
)

print("\nORB BEFORE / AFTER BACKTEST:")
print(summary_df)

by_year = (
    trades_df
    .group_by(["strategy", "year"])
    .agg([
        pl.len().alias("trades"),
        pl.col("points").sum().alias("year_points"),
        pl.col("r").sum().alias("year_r"),
    ])
    .sort(["strategy", "year"])
)

print("\nBY YEAR:")
print(by_year)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

events_file = OUT_DIR / "orb_failure_events.csv"
conditions_file = OUT_DIR / "orb_failure_condition_comparison.csv"
trades_file = OUT_DIR / "orb_failure_backtest_trades.csv"
summary_file = OUT_DIR / "orb_failure_filtered_backtest.csv"
year_file = OUT_DIR / "orb_failure_by_year.csv"

events_df.write_csv(events_file)
condition_df.write_csv(conditions_file)
trades_df.write_csv(trades_file)
summary_df.write_csv(summary_file)
by_year.write_csv(year_file)

print("\nSaved:")
print(events_file)
print(conditions_file)
print(trades_file)
print(summary_file)
print(year_file)

print("\nFinished:", datetime.now())
print("DONE.")