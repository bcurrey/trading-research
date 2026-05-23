import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "event_engines"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# NY open through late morning
base = df.filter(
    (pl.col("minute_of_day_ct") >= 8 * 60 + 30) &
    (pl.col("minute_of_day_ct") <= 11 * 60 + 30)
)

events = []

def first_event(event_name, direction, condition):
    x = (
        base
        .filter(condition)
        .sort(["trade_date_ct", "ts_event"])
        .group_by("trade_date_ct")
        .first()
        .with_columns([
            pl.lit(event_name).alias("event_name"),
            pl.lit(direction).alias("direction"),
        ])
    )
    return x

events.append(
    first_event(
        "First PDL sweep reclaim",
        "long",
        pl.col("pdl_sweep_close_back_above")
    )
)

events.append(
    first_event(
        "First PDH sweep reject",
        "short",
        pl.col("pdh_sweep_close_back_below")
    )
)

events.append(
    first_event(
        "First PML sweep reclaim",
        "long",
        pl.col("pml_sweep_close_back_above")
    )
)

events.append(
    first_event(
        "First PMH sweep reject",
        "short",
        pl.col("pmh_sweep_close_back_below")
    )
)

events_df = pl.concat(events).sort("ts_event")

print("\nEvents found:")
print(
    events_df
    .group_by(["event_name", "direction"])
    .agg(pl.len().alias("events"))
    .sort("events", descending=True)
)

trades = []

TARGETS = [25.0, 40.0, 60.0]
MAX_HOLD_MINUTES = 90
MIN_RISK = 8.0
MAX_RISK = 50.0

for row in events_df.iter_rows(named=True):
    entry_time = row["ts_event"]
    direction = row["direction"]
    event_name = row["event_name"]

    entry = row["close"]

    if direction == "long":
        stop = row["low"]
        risk = entry - stop
    else:
        stop = row["high"]
        risk = stop - entry

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
        "event_name": event_name,
        "direction": direction,
        "trade_date": row["trade_date_ct"],
        "entry_time": entry_time,
        "entry": entry,
        "stop": stop,
        "risk_points": risk,
        "hour_ct": row["hour_ct"],
        "minute_ct": row["minute_ct"],
        "weekday_ct": row["weekday_ct"],
        "body_pct": row["body_pct"],
        "atr_14": row["atr_14"],
        "rel_vol_20": row["rel_vol_20"],
        "close_above_vwap": row["close"] > row["vwap_day"],
        "dist_vwap_day": row["dist_vwap_day"],
    }

    for target in TARGETS:
        status = "timeout"
        exit_price = future[-1, "close"]
        exit_time = future[-1, "ts_event"]

        if direction == "long":
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

        else:
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

        result[f"target_{int(target)}"] = target_price
        result[f"outcome_{int(target)}"] = status
        result[f"points_{int(target)}"] = points
        result[f"r_{int(target)}"] = points / risk
        result[f"exit_time_{int(target)}"] = exit_time

    trades.append(result)

trades_df = pl.DataFrame(trades)

print("\nTrades after risk filters:", trades_df.height)

def summarize(target):
    col_points = f"points_{target}"
    col_r = f"r_{target}"
    col_outcome = f"outcome_{target}"

    return (
        trades_df
        .group_by(["event_name", "direction"])
        .agg([
            pl.len().alias("trades"),
            (pl.col(col_outcome) == "target").mean().alias("target_rate"),
            pl.col(col_points).mean().alias("avg_points"),
            pl.col(col_points).sum().alias("total_points"),
            pl.col(col_r).mean().alias("avg_r"),
            pl.col(col_r).sum().alias("total_r"),
            pl.col("risk_points").mean().alias("avg_risk"),
        ])
        .with_columns([
            pl.lit(target).alias("target_points"),
            (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
            pl.col("avg_points").round(2),
            pl.col("total_points").round(2),
            pl.col("avg_r").round(3),
            pl.col("total_r").round(2),
            pl.col("avg_risk").round(2),
        ])
        .sort("avg_points", descending=True)
    )

summaries = []

for target in [25, 40, 60]:
    s = summarize(target)
    summaries.append(s)

    print(f"\n===== TARGET {target} POINTS =====")
    print(s)

summary_df = pl.concat(summaries)

print("\nBest combinations:")
print(
    summary_df
    .filter(pl.col("trades") >= 75)
    .sort("avg_points", descending=True)
    .select([
        "event_name",
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
)

trades_file = OUT_DIR / "first_liquidity_sweep_trades.csv"
summary_file = OUT_DIR / "first_liquidity_sweep_summary.csv"

trades_df.write_csv(trades_file)
summary_df.write_csv(summary_file)

print("\nSaved:")
print(trades_file)
print(summary_file)

print("\nDONE.")