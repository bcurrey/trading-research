import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
ZONE_DIR = DATA_DIR / "htf_zones"
OUT_DIR = DATA_DIR / "htf_fvg_rejections"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")
zones = pl.read_parquet(ZONE_DIR / "nq_all_htf_fvg_zones.parquet").sort("zone_time")

# Morning session only
df = df.filter(
    (pl.col("minute_of_day_ct") >= 8 * 60 + 30) &
    (pl.col("minute_of_day_ct") <= 11 * 60 + 30)
)

TARGETS = [25.0, 40.0, 60.0]
MAX_HOLD_MINUTES = 90
MAX_ZONE_AGE_DAYS = 20

MIN_ZONE_SIZE = 8.0
MIN_ENTRY_DEPTH = 0.25
CONFIRM_BARS = 3
MIN_CONFIRM_MOVE = 8.0

MIN_RISK = 8.0
MAX_RISK = 60.0

events = []

print("\nScanning confirmed HTF FVG rejections...")
print("1m rows:", df.height)
print("HTF zones:", zones.height)

zones = zones.filter(pl.col("zone_size") >= MIN_ZONE_SIZE)

for z in zones.iter_rows(named=True):
    zone_time = z["zone_time"]
    timeframe = z["timeframe"]
    fvg_type = z["fvg_type"]
    zone_low = z["zone_low"]
    zone_high = z["zone_high"]
    zone_size = z["zone_size"]

    zone_end_time = zone_time + pl.duration(days=MAX_ZONE_AGE_DAYS)

    touch_bars = (
        df
        .filter(
            (pl.col("ts_event") > zone_time) &
            (pl.col("ts_event") <= zone_end_time) &
            (pl.col("high") >= zone_low) &
            (pl.col("low") <= zone_high)
        )
        .sort("ts_event")
    )

    if touch_bars.height == 0:
        continue

    # First touch only for now, but with confirmation rules
    touch = touch_bars.head(1).to_dicts()[0]
    touch_time = touch["ts_event"]

    after_touch = (
        df
        .filter(
            (pl.col("ts_event") >= touch_time) &
            (pl.col("ts_event") <= touch_time + pl.duration(minutes=CONFIRM_BARS))
        )
        .sort("ts_event")
    )

    if after_touch.height < CONFIRM_BARS:
        continue

    confirm_window = after_touch.head(CONFIRM_BARS)

    touch_high = touch["high"]
    touch_low = touch["low"]
    touch_close = touch["close"]

    if fvg_type == "bearish":
        # Need price to enter into bearish FVG from below, reject, close below zone_low,
        # then move lower after confirmation.
        entry_depth = (touch_high - zone_low) / zone_size

        if entry_depth < MIN_ENTRY_DEPTH:
            continue

        if touch_close >= zone_low:
            continue

        confirm_low = confirm_window.select(pl.col("low").min()).item()
        confirm_close = confirm_window[-1, "close"]

        if (touch_close - confirm_low) < MIN_CONFIRM_MOVE:
            continue

        direction = "short"
        entry_time = confirm_window[-1, "ts_event"]
        entry = confirm_close
        stop = zone_high
        risk = stop - entry

    else:
        # Need price to enter into bullish FVG from above, reject, close above zone_high,
        # then move higher after confirmation.
        entry_depth = (zone_high - touch_low) / zone_size

        if entry_depth < MIN_ENTRY_DEPTH:
            continue

        if touch_close <= zone_high:
            continue

        confirm_high = confirm_window.select(pl.col("high").max()).item()
        confirm_close = confirm_window[-1, "close"]

        if (confirm_high - touch_close) < MIN_CONFIRM_MOVE:
            continue

        direction = "long"
        entry_time = confirm_window[-1, "ts_event"]
        entry = confirm_close
        stop = zone_low
        risk = entry - stop

    if risk <= 0:
        continue

    if risk < MIN_RISK or risk > MAX_RISK:
        continue

    future = (
        df
        .filter(
            (pl.col("ts_event") > entry_time) &
            (pl.col("ts_event") <= entry_time + pl.duration(minutes=MAX_HOLD_MINUTES))
        )
        .select(["ts_event", "high", "low", "close"])
    )

    if future.height == 0:
        continue

    result = {
        "zone_time": zone_time,
        "touch_time": touch_time,
        "entry_time": entry_time,
        "trade_date_ct": touch["trade_date_ct"],
        "timeframe": timeframe,
        "fvg_type": fvg_type,
        "direction": direction,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "zone_size": zone_size,
        "entry_depth": entry_depth,
        "entry": entry,
        "stop": stop,
        "risk_points": risk,
        "hour_ct": touch["hour_ct"],
        "minute_ct": touch["minute_ct"],
        "weekday_ct": touch["weekday_ct"],
        "body_pct": touch["body_pct"],
        "atr_14": touch["atr_14"],
        "rel_vol_20": touch["rel_vol_20"],
        "dist_vwap_day": touch["dist_vwap_day"],
    }

    for target in TARGETS:
        status = "timeout"
        exit_price = future[-1, "close"]
        exit_time = future[-1, "ts_event"]

        if direction == "short":
            target_price = entry - target

            for bar in future.iter_rows(named=True):
                if bar["high"] >= stop:
                    status = "stop"
                    exit_price = stop
                    exit_time = bar["ts_event"]
                    break

                if bar["low"] <= target_price:
                    status = "target"
                    exit_price = target_price
                    exit_time = bar["ts_event"]
                    break

            points = entry - exit_price

        else:
            target_price = entry + target

            for bar in future.iter_rows(named=True):
                if bar["low"] <= stop:
                    status = "stop"
                    exit_price = stop
                    exit_time = bar["ts_event"]
                    break

                if bar["high"] >= target_price:
                    status = "target"
                    exit_price = target_price
                    exit_time = bar["ts_event"]
                    break

            points = exit_price - entry

        result[f"target_{int(target)}"] = target_price
        result[f"outcome_{int(target)}"] = status
        result[f"points_{int(target)}"] = points
        result[f"r_{int(target)}"] = points / risk
        result[f"exit_time_{int(target)}"] = exit_time

    events.append(result)

events_df = pl.DataFrame(events)

print("\nConfirmed rejection events:", events_df.height)

if events_df.height == 0:
    print("No confirmed rejection events found.")
    raise SystemExit

summaries = []

for target in [25, 40, 60]:
    summary = (
        events_df
        .group_by(["timeframe", "fvg_type", "direction"])
        .agg([
            pl.len().alias("trades"),
            (pl.col(f"outcome_{target}") == "target").mean().alias("target_rate"),
            pl.col(f"points_{target}").mean().alias("avg_points"),
            pl.col(f"points_{target}").sum().alias("total_points"),
            pl.col(f"r_{target}").mean().alias("avg_r"),
            pl.col(f"r_{target}").sum().alias("total_r"),
            pl.col("risk_points").mean().alias("avg_risk"),
            pl.col("zone_size").mean().alias("avg_zone_size"),
            pl.col("entry_depth").mean().alias("avg_entry_depth"),
        ])
        .with_columns([
            pl.lit(target).alias("target_points"),
            (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
            pl.col("avg_points").round(2),
            pl.col("total_points").round(2),
            pl.col("avg_r").round(3),
            pl.col("total_r").round(2),
            pl.col("avg_risk").round(2),
            pl.col("avg_zone_size").round(2),
            pl.col("avg_entry_depth").round(2),
        ])
    )

    summaries.append(summary)

summary_df = pl.concat(summaries).sort("avg_points", descending=True)

print("\nTOP CONFIRMED HTF FVG REJECTION RESULTS:")
print(
    summary_df.select([
        "timeframe",
        "fvg_type",
        "direction",
        "target_points",
        "trades",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "avg_zone_size",
        "avg_entry_depth",
    ])
    .head(50)
)

events_file = OUT_DIR / "confirmed_htf_fvg_rejection_events.csv"
summary_file = OUT_DIR / "confirmed_htf_fvg_rejection_summary.csv"

events_df.write_csv(events_file)
summary_df.write_csv(summary_file)

print("\nSaved:")
print(events_file)
print(summary_file)

print("\nDONE.")