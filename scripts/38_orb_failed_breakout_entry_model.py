import polars as pl
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "execution_models"
OUT_DIR.mkdir(exist_ok=True)

print("\nORB FAILED BREAKOUT ENTRY MODEL")
print("Started:", datetime.now())

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# ---------------------------------------------------
# Add date features
# ---------------------------------------------------

df = df.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),
    pl.col("trade_date_ct").dt.month().alias("month"),
])

# ---------------------------------------------------
# Build overnight levels
# ---------------------------------------------------

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
# Data-driven volatility percentiles
# Use daily OR range, overnight range, and ATR at 9:00 CT
# ---------------------------------------------------

daily_context = (
    df
    .filter(pl.col("minute_of_day_ct") == 9 * 60)
    .select([
        "trade_date_ct",
        "or_range_30m",
        "overnight_range",
        "atr_14",
        "gap_from_prior_close",
    ])
    .drop_nulls()
)

or_p70 = daily_context.select(pl.col("or_range_30m").quantile(0.70)).item()
or_p80 = daily_context.select(pl.col("or_range_30m").quantile(0.80)).item()
or_p90 = daily_context.select(pl.col("or_range_30m").quantile(0.90)).item()

on_p70 = daily_context.select(pl.col("overnight_range").quantile(0.70)).item()
on_p80 = daily_context.select(pl.col("overnight_range").quantile(0.80)).item()
on_p90 = daily_context.select(pl.col("overnight_range").quantile(0.90)).item()

atr_p70 = daily_context.select(pl.col("atr_14").quantile(0.70)).item()
atr_p80 = daily_context.select(pl.col("atr_14").quantile(0.80)).item()
atr_p90 = daily_context.select(pl.col("atr_14").quantile(0.90)).item()

print("\nVolatility thresholds:")
print(f"OR range p70/p80/p90: {or_p70:.2f} / {or_p80:.2f} / {or_p90:.2f}")
print(f"Overnight range p70/p80/p90: {on_p70:.2f} / {on_p80:.2f} / {on_p90:.2f}")
print(f"ATR p70/p80/p90: {atr_p70:.2f} / {atr_p80:.2f} / {atr_p90:.2f}")

df = df.with_columns([
    pl.when(pl.col("or_range_30m") >= or_p90)
    .then(pl.lit("or_extreme"))
    .when(pl.col("or_range_30m") >= or_p80)
    .then(pl.lit("or_high"))
    .when(pl.col("or_range_30m") >= or_p70)
    .then(pl.lit("or_elevated"))
    .otherwise(pl.lit("or_normal"))
    .alias("or_vol_regime"),

    pl.when(pl.col("overnight_range") >= on_p90)
    .then(pl.lit("on_extreme"))
    .when(pl.col("overnight_range") >= on_p80)
    .then(pl.lit("on_high"))
    .when(pl.col("overnight_range") >= on_p70)
    .then(pl.lit("on_elevated"))
    .otherwise(pl.lit("on_normal"))
    .alias("overnight_vol_regime"),

    pl.when(pl.col("atr_14") >= atr_p90)
    .then(pl.lit("atr_extreme"))
    .when(pl.col("atr_14") >= atr_p80)
    .then(pl.lit("atr_high"))
    .when(pl.col("atr_14") >= atr_p70)
    .then(pl.lit("atr_elevated"))
    .otherwise(pl.lit("atr_normal"))
    .alias("atr_vol_regime"),
])

# ---------------------------------------------------
# Find OR breakout -> close back inside OR events
# This is the actual executable reversal confirmation
# ---------------------------------------------------

events = []

test_days = df.select("trade_date_ct").unique().to_series().to_list()

for trade_date in test_days:
    day = (
        df
        .filter(
            (pl.col("trade_date_ct") == trade_date) &
            (pl.col("minute_of_day_ct") >= 9 * 60) &
            (pl.col("minute_of_day_ct") <= 10 * 60 + 30)
        )
        .sort("ts_event")
    )

    if day.height == 0:
        continue

    first = day.row(0, named=True)

    or_high = first["or_high_30m"]
    or_low = first["or_low_30m"]

    if or_high is None or or_low is None:
        continue

    # Track whether OR high/low has been swept first
    high_swept = False
    low_swept = False
    high_sweep_price = None
    low_sweep_price = None
    high_sweep_time = None
    low_sweep_time = None

    event_found = False

    for row in day.iter_rows(named=True):
        if event_found:
            break

        # OR high sweep
        if not high_swept and row["high"] > or_high:
            high_swept = True
            high_sweep_price = row["high"]
            high_sweep_time = row["ts_event"]

        # OR low sweep
        if not low_swept and row["low"] < or_low:
            low_swept = True
            low_sweep_price = row["low"]
            low_sweep_time = row["ts_event"]

        # Failed OR high breakout: swept high, then close back inside OR
        if high_swept and row["close"] < or_high:
            entry_time = row["ts_event"]
            entry = row["close"]
            stop = high_sweep_price
            risk = stop - entry

            if risk > 0:
                events.append({
                    "trade_date_ct": trade_date,
                    "year": row["year"],
                    "month": row["month"],
                    "event_type": "failed_or_high_breakout",
                    "direction": "short",
                    "entry_time": entry_time,
                    "entry": entry,
                    "stop": stop,
                    "risk_points": risk,
                    "sweep_time": high_sweep_time,
                    "sweep_price": high_sweep_price,
                    "or_high": or_high,
                    "or_low": or_low,
                    "or_mid": (or_high + or_low) / 2,
                    "or_range": row["or_range_30m"],
                    "close_back_inside_price": row["close"],
                    "minute_of_day_ct": row["minute_of_day_ct"],
                    "hour_ct": row["hour_ct"],
                    "minute_ct": row["minute_ct"],
                    "vwap_day": row["vwap_day"],
                    "dist_vwap_day": row["dist_vwap_day"],
                    "entry_above_vwap": row["close"] > row["vwap_day"],
                    "atr_14": row["atr_14"],
                    "rel_vol_20": row["rel_vol_20"],
                    "body_pct": row["body_pct"],
                    "overnight_high": row["overnight_high"],
                    "overnight_low": row["overnight_low"],
                    "overnight_range": row["overnight_range"],
                    "gap_from_prior_close": row["gap_from_prior_close"],
                    "prior_rth_high": row["prior_rth_high"],
                    "prior_rth_low": row["prior_rth_low"],
                    "prior_rth_close": row["prior_rth_close"],
                    "or_vol_regime": row["or_vol_regime"],
                    "overnight_vol_regime": row["overnight_vol_regime"],
                    "atr_vol_regime": row["atr_vol_regime"],
                })
                event_found = True
                break

        # Failed OR low breakdown: swept low, then close back inside OR
        if low_swept and row["close"] > or_low:
            entry_time = row["ts_event"]
            entry = row["close"]
            stop = low_sweep_price
            risk = entry - stop

            if risk > 0:
                events.append({
                    "trade_date_ct": trade_date,
                    "year": row["year"],
                    "month": row["month"],
                    "event_type": "failed_or_low_breakdown",
                    "direction": "long",
                    "entry_time": entry_time,
                    "entry": entry,
                    "stop": stop,
                    "risk_points": risk,
                    "sweep_time": low_sweep_time,
                    "sweep_price": low_sweep_price,
                    "or_high": or_high,
                    "or_low": or_low,
                    "or_mid": (or_high + or_low) / 2,
                    "or_range": row["or_range_30m"],
                    "close_back_inside_price": row["close"],
                    "minute_of_day_ct": row["minute_of_day_ct"],
                    "hour_ct": row["hour_ct"],
                    "minute_ct": row["minute_ct"],
                    "vwap_day": row["vwap_day"],
                    "dist_vwap_day": row["dist_vwap_day"],
                    "entry_above_vwap": row["close"] > row["vwap_day"],
                    "atr_14": row["atr_14"],
                    "rel_vol_20": row["rel_vol_20"],
                    "body_pct": row["body_pct"],
                    "overnight_high": row["overnight_high"],
                    "overnight_low": row["overnight_low"],
                    "overnight_range": row["overnight_range"],
                    "gap_from_prior_close": row["gap_from_prior_close"],
                    "prior_rth_high": row["prior_rth_high"],
                    "prior_rth_low": row["prior_rth_low"],
                    "prior_rth_close": row["prior_rth_close"],
                    "or_vol_regime": row["or_vol_regime"],
                    "overnight_vol_regime": row["overnight_vol_regime"],
                    "atr_vol_regime": row["atr_vol_regime"],
                })
                event_found = True
                break

events_df = pl.DataFrame(events)

print("\nConfirmed ORB failure entries:", events_df.height)

if events_df.height == 0:
    raise SystemExit("No events found.")

# ---------------------------------------------------
# Add rule buckets
# ---------------------------------------------------

events_df = events_df.with_columns([
    pl.when(pl.col("minute_of_day_ct") <= 9 * 60 + 15)
    .then(pl.lit("entry_9_00_to_9_15"))
    .when(pl.col("minute_of_day_ct") <= 9 * 60 + 30)
    .then(pl.lit("entry_9_16_to_9_30"))
    .when(pl.col("minute_of_day_ct") <= 10 * 60)
    .then(pl.lit("entry_9_31_to_10_00"))
    .otherwise(pl.lit("entry_10_01_to_10_30"))
    .alias("entry_time_bucket"),

    pl.when(pl.col("rel_vol_20") >= 1.5)
    .then(pl.lit("high_volume"))
    .when(pl.col("rel_vol_20") >= 1.0)
    .then(pl.lit("normal_volume"))
    .otherwise(pl.lit("low_volume"))
    .alias("volume_bucket"),

    pl.when(pl.col("body_pct") >= 0.65)
    .then(pl.lit("strong_body"))
    .when(pl.col("body_pct") >= 0.40)
    .then(pl.lit("medium_body"))
    .otherwise(pl.lit("weak_body"))
    .alias("body_bucket"),

    pl.when(pl.col("risk_points") < 10)
    .then(pl.lit("risk_tiny"))
    .when(pl.col("risk_points") < 25)
    .then(pl.lit("risk_good"))
    .when(pl.col("risk_points") < 50)
    .then(pl.lit("risk_large"))
    .otherwise(pl.lit("risk_too_large"))
    .alias("risk_bucket"),

    pl.when(pl.col("entry_above_vwap"))
    .then(pl.lit("above_vwap"))
    .otherwise(pl.lit("below_vwap"))
    .alias("vwap_side"),

    pl.when(pl.col("gap_from_prior_close") > 30)
    .then(pl.lit("gap_up_30_plus"))
    .when(pl.col("gap_from_prior_close") < -30)
    .then(pl.lit("gap_down_30_plus"))
    .otherwise(pl.lit("small_gap"))
    .alias("gap_bucket"),

    pl.when(pl.col("entry") > pl.col("overnight_high"))
    .then(pl.lit("above_overnight_high"))
    .when(pl.col("entry") < pl.col("overnight_low"))
    .then(pl.lit("below_overnight_low"))
    .otherwise(pl.lit("inside_overnight_range"))
    .alias("overnight_position"),

    pl.when(pl.col("entry") > pl.col("prior_rth_high"))
    .then(pl.lit("above_pdh"))
    .when(pl.col("entry") < pl.col("prior_rth_low"))
    .then(pl.lit("below_pdl"))
    .otherwise(pl.lit("inside_prior_day_range"))
    .alias("prior_day_position"),
])

# ---------------------------------------------------
# Backtest actual entries
# Targets:
# 1. OR midpoint
# 2. Opposite OR side
# 3. Fixed 25/40/60 points
# ---------------------------------------------------

trade_rows = []

for row in events_df.iter_rows(named=True):
    entry_time = row["entry_time"]
    direction = row["direction"]
    entry = row["entry"]
    stop = row["stop"]
    risk = row["risk_points"]

    if risk <= 0 or risk > 80:
        continue

    future = (
        df
        .filter(
            (pl.col("ts_event") > entry_time) &
            (pl.col("ts_event") <= entry_time + pl.duration(minutes=90))
        )
        .sort("ts_event")
    )

    if future.height == 0:
        continue

    targets = {}

    if direction == "short":
        targets["or_mid"] = row["or_mid"]
        targets["opposite_or"] = row["or_low"]
        targets["fixed_25"] = entry - 25
        targets["fixed_40"] = entry - 40
        targets["fixed_60"] = entry - 60
    else:
        targets["or_mid"] = row["or_mid"]
        targets["opposite_or"] = row["or_high"]
        targets["fixed_25"] = entry + 25
        targets["fixed_40"] = entry + 40
        targets["fixed_60"] = entry + 60

    for target_name, target_price in targets.items():
        status = "timeout"
        exit_price = future[-1, "close"]

        for bar in future.iter_rows(named=True):
            if direction == "short":
                if bar["high"] >= stop:
                    status = "stop"
                    exit_price = stop
                    break
                if bar["low"] <= target_price:
                    status = "target"
                    exit_price = target_price
                    break
            else:
                if bar["low"] <= stop:
                    status = "stop"
                    exit_price = stop
                    break
                if bar["high"] >= target_price:
                    status = "target"
                    exit_price = target_price
                    break

        if direction == "short":
            points = entry - exit_price
        else:
            points = exit_price - entry

        trade_rows.append({
            **row,
            "target_name": target_name,
            "target_price": target_price,
            "status": status,
            "points": points,
            "r": points / risk,
        })

trades_df = pl.DataFrame(trade_rows)

print("\nTrade outcome rows:", trades_df.height)

# ---------------------------------------------------
# Summaries
# ---------------------------------------------------

def summarize(group_cols):
    return (
        trades_df
        .group_by(group_cols)
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

base_summary = summarize(["event_type", "direction", "target_name"])

print("\nBASE CONFIRMED FAILURE ENTRY RESULTS:")
print(
    base_summary
    .filter(pl.col("trades") >= 50)
    .select([
        "event_type",
        "direction",
        "target_name",
        "trades",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
    ])
    .head(50)
)

context_cols = [
    "entry_time_bucket",
    "or_vol_regime",
    "overnight_vol_regime",
    "atr_vol_regime",
    "volume_bucket",
    "body_bucket",
    "risk_bucket",
    "vwap_side",
    "gap_bucket",
    "overnight_position",
    "prior_day_position",
]

context_frames = []

for col in context_cols:
    s = summarize(["event_type", "direction", "target_name", col])
    s = s.with_columns([
        pl.lit(col).alias("context"),
        pl.col(col).cast(pl.Utf8).alias("context_value"),
    ])
    context_frames.append(s)

context_summary = pl.concat(context_frames, how="diagonal_relaxed")

context_summary = context_summary.filter(pl.col("trades") >= 40)

context_summary = context_summary.with_columns([
    (
        pl.col("avg_points") *
        (pl.col("trades").clip(upper_bound=500) / 500)
    ).round(4).alias("simple_score")
])

context_summary = context_summary.sort("simple_score", descending=True)

print("\nTOP CONTEXT RESULTS:")
print(
    context_summary
    .select([
        "event_type",
        "direction",
        "target_name",
        "context",
        "context_value",
        "trades",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "simple_score",
    ])
    .head(100)
)

# ---------------------------------------------------
# By year for top-level strategy
# ---------------------------------------------------

by_year = (
    trades_df
    .group_by(["event_type", "direction", "target_name", "year"])
    .agg([
        pl.len().alias("trades"),
        pl.col("points").sum().alias("year_points"),
        pl.col("r").sum().alias("year_r"),
    ])
    .sort(["event_type", "target_name", "year"])
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

events_file = OUT_DIR / "orb_confirmed_failure_entries.csv"
trades_file = OUT_DIR / "orb_confirmed_failure_trades.csv"
base_file = OUT_DIR / "orb_confirmed_failure_base_summary.csv"
context_file = OUT_DIR / "orb_confirmed_failure_context_summary.csv"
year_file = OUT_DIR / "orb_confirmed_failure_by_year.csv"

events_df.write_csv(events_file)
trades_df.write_csv(trades_file)
base_summary.write_csv(base_file)
context_summary.write_csv(context_file)
by_year.write_csv(year_file)

print("\nSaved:")
print(events_file)
print(trades_file)
print(base_file)
print(context_file)
print(year_file)

print("\nFinished:", datetime.now())
print("DONE.")