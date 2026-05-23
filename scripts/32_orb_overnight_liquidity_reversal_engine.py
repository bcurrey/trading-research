import polars as pl
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "orb_liquidity_research"
OUT_DIR.mkdir(exist_ok=True)

print("\n========================================")
print("ORB + OVERNIGHT LIQUIDITY RESEARCH ENGINE")
print("Started:", datetime.now())
print("========================================")

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# ---------------------------------------------------
# Build overnight high/low
# Overnight window:
# prior session 17:00 CT through 08:29 CT
# This uses your current ts_ct/trade_date_ct structure.
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
        pl.col("close").last().alias("overnight_close"),
    ])
    .rename({"overnight_trade_date": "trade_date_ct"})
)

df = df.join(overnight, on="trade_date_ct", how="left")

df = df.with_columns([
    (pl.col("overnight_high") - pl.col("overnight_low")).alias("overnight_range"),

    (pl.col("high") > pl.col("overnight_high")).alias("swept_overnight_high"),
    (pl.col("low") < pl.col("overnight_low")).alias("swept_overnight_low"),

    (
        (pl.col("high") > pl.col("overnight_high")) &
        (pl.col("close") < pl.col("overnight_high"))
    ).alias("overnight_high_sweep_reject"),

    (
        (pl.col("low") < pl.col("overnight_low")) &
        (pl.col("close") > pl.col("overnight_low"))
    ).alias("overnight_low_sweep_reclaim"),

    (pl.col("close") - pl.col("overnight_high")).alias("dist_overnight_high"),
    (pl.col("close") - pl.col("overnight_low")).alias("dist_overnight_low"),
])

# ---------------------------------------------------
# ORB events after opening range is formed
# OR: 8:30-9:00 CT
# Test events: 9:00-11:30 CT
# ---------------------------------------------------

test_window = df.filter(
    (pl.col("minute_of_day_ct") >= 9 * 60) &
    (pl.col("minute_of_day_ct") <= 11 * 60 + 30)
)

test_window = test_window.with_columns([
    (
        (pl.col("high") > pl.col("or_high_30m")) &
        (pl.col("close") > pl.col("or_high_30m"))
    ).alias("or_high_breakout_close_above"),

    (
        (pl.col("low") < pl.col("or_low_30m")) &
        (pl.col("close") < pl.col("or_low_30m"))
    ).alias("or_low_breakdown_close_below"),

    (
        (pl.col("high") > pl.col("or_high_30m")) &
        (pl.col("close") < pl.col("or_high_30m"))
    ).alias("or_high_failed_breakout"),

    (
        (pl.col("low") < pl.col("or_low_30m")) &
        (pl.col("close") > pl.col("or_low_30m"))
    ).alias("or_low_failed_breakdown"),
])

# ---------------------------------------------------
# Event definitions
# One first event per day per type
# ---------------------------------------------------

EVENTS = [
    ("OR high breakout continuation", "long", "or_high_breakout_close_above"),
    ("OR low breakdown continuation", "short", "or_low_breakdown_close_below"),
    ("OR high failed breakout reversal", "short", "or_high_failed_breakout"),
    ("OR low failed breakdown reversal", "long", "or_low_failed_breakdown"),
    ("Overnight high sweep reject", "short", "overnight_high_sweep_reject"),
    ("Overnight low sweep reclaim", "long", "overnight_low_sweep_reclaim"),
    ("PMH sweep reject", "short", "pmh_sweep_close_back_below"),
    ("PML sweep reclaim", "long", "pml_sweep_close_back_above"),
    ("PDH sweep reject", "short", "pdh_sweep_close_back_below"),
    ("PDL sweep reclaim", "long", "pdl_sweep_close_back_above"),
]

event_frames = []

for event_name, direction, col in EVENTS:
    x = (
        test_window
        .filter(pl.col(col) == True)
        .sort(["trade_date_ct", "ts_event"])
        .group_by("trade_date_ct")
        .first()
        .with_columns([
            pl.lit(event_name).alias("event_name"),
            pl.lit(direction).alias("direction"),
        ])
    )

    event_frames.append(x)

events = pl.concat(event_frames).sort("ts_event")

print("\nEvents found:")
print(
    events
    .group_by(["event_name", "direction"])
    .agg(pl.len().alias("events"))
    .sort("events", descending=True)
)

# ---------------------------------------------------
# Context buckets
# ---------------------------------------------------

events = events.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),

    pl.when(pl.col("close") > pl.col("vwap_day"))
    .then(pl.lit("above_vwap"))
    .otherwise(pl.lit("below_vwap"))
    .alias("vwap_side"),

    pl.when(pl.col("or_range_30m") < 35)
    .then(pl.lit("small_or"))
    .when(pl.col("or_range_30m") < 70)
    .then(pl.lit("normal_or"))
    .otherwise(pl.lit("large_or"))
    .alias("or_range_bucket"),

    pl.when(pl.col("overnight_range") < 80)
    .then(pl.lit("small_overnight_range"))
    .when(pl.col("overnight_range") < 180)
    .then(pl.lit("normal_overnight_range"))
    .otherwise(pl.lit("large_overnight_range"))
    .alias("overnight_range_bucket"),

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

    pl.when(pl.col("close") > pl.col("or_high_30m"))
    .then(pl.lit("above_or"))
    .when(pl.col("close") < pl.col("or_low_30m"))
    .then(pl.lit("below_or"))
    .otherwise(pl.lit("inside_or"))
    .alias("or_position"),

    pl.when(pl.col("close") > pl.col("overnight_high"))
    .then(pl.lit("above_overnight_high"))
    .when(pl.col("close") < pl.col("overnight_low"))
    .then(pl.lit("below_overnight_low"))
    .otherwise(pl.lit("inside_overnight_range"))
    .alias("overnight_position"),

    pl.when(pl.col("high") > pl.col("prior_rth_high"))
    .then(pl.lit("took_pdh"))
    .when(pl.col("low") < pl.col("prior_rth_low"))
    .then(pl.lit("took_pdl"))
    .otherwise(pl.lit("took_neither_pdh_pdl"))
    .alias("prior_day_liquidity_taken"),

    pl.when(pl.col("high") > pl.col("overnight_high"))
    .then(pl.lit("took_onh"))
    .when(pl.col("low") < pl.col("overnight_low"))
    .then(pl.lit("took_onl"))
    .otherwise(pl.lit("took_neither_onh_onl"))
    .alias("overnight_liquidity_taken"),
])

# ---------------------------------------------------
# Backtest event rows with actual sequence
# ---------------------------------------------------

TARGETS = [25.0, 40.0, 60.0, 80.0]
MAX_HOLDS = [30, 60, 90, 120]
RISK_WINDOWS = [
    ("risk_5_25", 5.0, 25.0),
    ("risk_8_35", 8.0, 35.0),
    ("risk_10_50", 10.0, 50.0),
]

trade_rows = []

print("\nBacktesting event outcomes...")

for row in events.iter_rows(named=True):
    entry_time = row["ts_event"]
    direction = row["direction"]

    entry = row["close"]

    if direction == "long":
        stop = row["low"]
        risk = entry - stop
    else:
        stop = row["high"]
        risk = stop - entry

    if risk <= 0:
        continue

    for hold in MAX_HOLDS:
        future = (
            df
            .filter(
                (pl.col("ts_event") > entry_time) &
                (pl.col("ts_event") <= entry_time + pl.duration(minutes=hold))
            )
            .select(["ts_event", "high", "low", "close"])
        )

        if future.height == 0:
            continue

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

            trade_rows.append({
                "event_name": row["event_name"],
                "direction": direction,
                "trade_date_ct": row["trade_date_ct"],
                "year": row["year"],
                "entry_time": entry_time,
                "target_points": target,
                "max_hold_minutes": hold,
                "entry": entry,
                "stop": stop,
                "risk_points": risk,
                "status": status,
                "points": points,
                "r": points / risk,

                "hour_ct": row["hour_ct"],
                "weekday_ct": row["weekday_ct"],
                "vwap_side": row["vwap_side"],
                "or_range_bucket": row["or_range_bucket"],
                "overnight_range_bucket": row["overnight_range_bucket"],
                "atr_bucket": row["atr_bucket"],
                "volume_bucket": row["volume_bucket"],
                "body_bucket": row["body_bucket"],
                "or_position": row["or_position"],
                "overnight_position": row["overnight_position"],
                "prior_day_liquidity_taken": row["prior_day_liquidity_taken"],
                "overnight_liquidity_taken": row["overnight_liquidity_taken"],
            })

trades = pl.DataFrame(trade_rows)

print(f"\nTrade outcome rows: {trades.height:,}")

# ---------------------------------------------------
# Summarize base event performance
# ---------------------------------------------------

base_summary = (
    trades
    .group_by(["event_name", "direction", "target_points", "max_hold_minutes"])
    .agg([
        pl.len().alias("rows"),
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

print("\nTOP BASE EVENT RESULTS:")
print(
    base_summary
    .filter(pl.col("rows") >= 100)
    .select([
        "event_name",
        "direction",
        "target_points",
        "max_hold_minutes",
        "rows",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
    ])
    .head(50)
)

# ---------------------------------------------------
# Regime / context scan
# ---------------------------------------------------

context_cols = [
    "hour_ct",
    "weekday_ct",
    "vwap_side",
    "or_range_bucket",
    "overnight_range_bucket",
    "atr_bucket",
    "volume_bucket",
    "body_bucket",
    "or_position",
    "overnight_position",
    "prior_day_liquidity_taken",
    "overnight_liquidity_taken",
]

context_results = []

for event_name in trades.select("event_name").unique().to_series().to_list():
    event_df = trades.filter(pl.col("event_name") == event_name)

    for target in TARGETS:
        for hold in MAX_HOLDS:
            base_x = event_df.filter(
                (pl.col("target_points") == target) &
                (pl.col("max_hold_minutes") == hold)
            )

            if base_x.height < 75:
                continue

            for col in context_cols:
                values = base_x.select(pl.col(col).unique()).to_series().to_list()

                for val in values:
                    x = base_x.filter(pl.col(col) == val)

                    if x.height < 40:
                        continue

                    year_stats = (
                        x
                        .group_by("year")
                        .agg(pl.col("points").sum().alias("year_points"))
                        .with_columns((pl.col("year_points") > 0).alias("positive_year"))
                    )

                    positive_years = year_stats.select(pl.col("positive_year").sum()).item()
                    total_years = year_stats.height

                    result = x.select([
                        pl.len().alias("rows"),
                        (pl.col("status") == "target").mean().alias("target_rate"),
                        pl.col("points").mean().alias("avg_points"),
                        pl.col("points").sum().alias("total_points"),
                        pl.col("r").mean().alias("avg_r"),
                        pl.col("r").sum().alias("total_r"),
                        pl.col("risk_points").mean().alias("avg_risk"),
                    ]).to_dicts()[0]

                    result.update({
                        "event_name": event_name,
                        "filter": f"{col}={val}",
                        "target_points": target,
                        "max_hold_minutes": hold,
                        "positive_years": positive_years,
                        "total_years": total_years,
                        "positive_year_rate": positive_years / total_years if total_years else 0,
                    })

                    context_results.append(result)

context_df = pl.DataFrame(context_results)

context_df = context_df.with_columns([
    (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
    (pl.col("positive_year_rate") * 100).round(2).alias("positive_year_rate_pct"),
    pl.col("avg_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_r").round(3),
    pl.col("total_r").round(2),
    pl.col("avg_risk").round(2),
    (
        pl.col("avg_points") *
        pl.col("positive_year_rate") *
        (pl.col("rows").clip(upper_bound=500) / 500)
    ).round(4).alias("context_score")
])

ranked_context = (
    context_df
    .filter(
        (pl.col("rows") >= 50) &
        (pl.col("avg_points") > 0) &
        (pl.col("positive_year_rate") >= 0.50)
    )
    .sort("context_score", descending=True)
)

print("\nTOP ORB / LIQUIDITY CONTEXT RESULTS:")
print(
    ranked_context
    .select([
        "event_name",
        "filter",
        "target_points",
        "max_hold_minutes",
        "rows",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "positive_years",
        "total_years",
        "positive_year_rate_pct",
        "context_score",
    ])
    .head(100)
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

events_file = OUT_DIR / "orb_liquidity_events.csv"
trades_file = OUT_DIR / "orb_liquidity_trade_outcomes.csv"
base_summary_file = OUT_DIR / "orb_liquidity_base_summary.csv"
context_file = OUT_DIR / "orb_liquidity_context_ranked.csv"

events.write_csv(events_file)
trades.write_csv(trades_file)
base_summary.write_csv(base_summary_file)
ranked_context.write_csv(context_file)

print("\nSaved:")
print(events_file)
print(trades_file)
print(base_summary_file)
print(context_file)

print("\nFinished:", datetime.now())
print("\nDONE.")