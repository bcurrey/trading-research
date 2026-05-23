import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
ZONE_DIR = DATA_DIR / "htf_zones"
OUT_DIR = DATA_DIR / "htf_fvg_rejections"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")
zones = pl.read_parquet(ZONE_DIR / "nq_all_htf_fvg_zones.parquet").sort("zone_time")

# Morning only for now
df = df.filter(
    (pl.col("minute_of_day_ct") >= 8 * 60 + 30) &
    (pl.col("minute_of_day_ct") <= 11 * 60 + 30)
)

print("\n1m rows:", df.height)
print("HTF zones:", zones.height)

events = []

MAX_ZONE_AGE_DAYS = 20

for z in zones.iter_rows(named=True):
    zone_time = z["zone_time"]
    timeframe = z["timeframe"]
    fvg_type = z["fvg_type"]
    zone_low = z["zone_low"]
    zone_high = z["zone_high"]
    zone_size = z["zone_size"]

    zone_end_time = zone_time + pl.duration(days=MAX_ZONE_AGE_DAYS)

    touches = df.filter(
        (pl.col("ts_event") > zone_time) &
        (pl.col("ts_event") <= zone_end_time) &
        (pl.col("high") >= zone_low) &
        (pl.col("low") <= zone_high)
    )

    if touches.height == 0:
        continue

    # First touch per zone only
    touch = touches.sort("ts_event").head(1)

    row = touch.to_dicts()[0]

    if fvg_type == "bearish":
        # Rejection short: trades into bearish FVG and closes back below zone
        rejected = row["close"] < zone_low
        direction = "short"
        entry = row["close"]
        stop = row["high"]
        risk = stop - entry

    else:
        # Rejection long: trades into bullish FVG and closes back above zone
        rejected = row["close"] > zone_high
        direction = "long"
        entry = row["close"]
        stop = row["low"]
        risk = entry - stop

    if not rejected:
        continue

    if risk <= 0:
        continue

    events.append({
        "zone_time": zone_time,
        "touch_time": row["ts_event"],
        "trade_date_ct": row["trade_date_ct"],
        "timeframe": timeframe,
        "fvg_type": fvg_type,
        "direction": direction,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "zone_size": zone_size,
        "entry": entry,
        "stop": stop,
        "risk_points": risk,
        "hour_ct": row["hour_ct"],
        "minute_ct": row["minute_ct"],
        "weekday_ct": row["weekday_ct"],
        "body_pct": row["body_pct"],
        "upper_wick_pct": row["upper_wick_pct"],
        "lower_wick_pct": row["lower_wick_pct"],
        "atr_14": row["atr_14"],
        "rel_vol_20": row["rel_vol_20"],
        "vwap_day": row["vwap_day"],
        "close": row["close"],
        "dist_vwap_day": row["dist_vwap_day"],
        "future_high_60m": row["future_high_60m"],
        "future_low_60m": row["future_low_60m"],
        "fwd_points_60m": row["fwd_points_60m"],
    })

events_df = pl.DataFrame(events)

print("\nRejected HTF FVG touches:", events_df.height)

if events_df.height == 0:
    raise SystemExit("No rejection events found.")

# ---------------------------------------
# Backtest using conservative 60m future high/low
# ---------------------------------------

TARGETS = [25.0, 40.0, 60.0]
MIN_RISKS = [5.0, 8.0, 10.0]
MAX_RISKS = [25.0, 35.0, 50.0]

results = []

for target in TARGETS:
    for min_risk in MIN_RISKS:
        for max_risk in MAX_RISKS:

            x = events_df.filter(
                (pl.col("risk_points") >= min_risk) &
                (pl.col("risk_points") <= max_risk)
            )

            if x.height < 40:
                continue

            long_x = x.filter(pl.col("direction") == "long").with_columns([
                (pl.col("future_high_60m") - pl.col("entry")).alias("favorable"),
                (pl.col("entry") - pl.col("future_low_60m")).alias("adverse"),
            ])

            short_x = x.filter(pl.col("direction") == "short").with_columns([
                (pl.col("entry") - pl.col("future_low_60m")).alias("favorable"),
                (pl.col("future_high_60m") - pl.col("entry")).alias("adverse"),
            ])

            combined = pl.concat([long_x, short_x])

            combined = combined.with_columns([
                (pl.col("favorable") >= target).alias("hit_target"),
                (pl.col("adverse") >= pl.col("risk_points")).alias("hit_stop"),
            ])

            combined = combined.with_columns([
                pl.when(pl.col("hit_target") & ~pl.col("hit_stop"))
                .then(pl.lit(target))
                .when(pl.col("hit_stop"))
                .then(-pl.col("risk_points"))
                .otherwise(
                    pl.when(pl.col("direction") == "long")
                    .then(pl.col("fwd_points_60m"))
                    .otherwise(-pl.col("fwd_points_60m"))
                )
                .alias("points_result")
            ])

            summary = (
                combined
                .group_by(["timeframe", "fvg_type", "direction"])
                .agg([
                    pl.len().alias("trades"),
                    (pl.col("points_result") > 0).mean().alias("win_rate"),
                    pl.col("points_result").mean().alias("avg_points"),
                    pl.col("points_result").sum().alias("total_points"),
                    pl.col("risk_points").mean().alias("avg_risk"),
                    pl.col("zone_size").mean().alias("avg_zone_size"),
                ])
                .filter(pl.col("trades") >= 40)
                .with_columns([
                    pl.lit(target).alias("target_points"),
                    pl.lit(min_risk).alias("min_risk"),
                    pl.lit(max_risk).alias("max_risk"),
                ])
            )

            if summary.height > 0:
                results.append(summary)

if not results:
    print("\nNo valid summary results.")
    events_df.write_csv(OUT_DIR / "htf_fvg_rejection_events.csv")
    raise SystemExit

results_df = pl.concat(results)

results_df = results_df.with_columns([
    (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
    pl.col("avg_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_risk").round(2),
    pl.col("avg_zone_size").round(2),
])

ranked = results_df.sort("avg_points", descending=True)

print("\nTOP HTF FVG REJECTION RESULTS:")
print(
    ranked.select([
        "timeframe",
        "fvg_type",
        "direction",
        "target_points",
        "min_risk",
        "max_risk",
        "trades",
        "win_rate_pct",
        "avg_points",
        "total_points",
        "avg_risk",
        "avg_zone_size",
    ]).head(50)
)

events_file = OUT_DIR / "htf_fvg_rejection_events.csv"
summary_file = OUT_DIR / "htf_fvg_rejection_summary.csv"

events_df.write_csv(events_file)
ranked.write_csv(summary_file)

print("\nSaved:")
print(events_file)
print(summary_file)

print("\nDONE.")