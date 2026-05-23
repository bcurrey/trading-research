import polars as pl
from pathlib import Path
from itertools import product

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "overnight_analysis"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# ---------------------------------------
# Base event: first PMH sweep reject
# ---------------------------------------

base = df.filter(
    (pl.col("minute_of_day_ct") >= 8 * 60 + 30) &
    (pl.col("minute_of_day_ct") <= 11 * 60 + 30) &
    (pl.col("pmh_sweep_close_back_below"))
)

events = (
    base
    .sort(["trade_date_ct", "ts_event"])
    .group_by("trade_date_ct")
    .first()
    .sort("ts_event")
)

print("\nFirst PMH rejection events:", events.height)

# ---------------------------------------
# Add regime buckets
# ---------------------------------------

events = events.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),

    pl.when(pl.col("close") > pl.col("vwap_day"))
    .then(pl.lit("above_vwap"))
    .otherwise(pl.lit("below_vwap"))
    .alias("vwap_side"),

    pl.when(pl.col("dist_vwap_day").abs() < 10)
    .then(pl.lit("near_vwap"))
    .when(pl.col("dist_vwap_day").abs() < 25)
    .then(pl.lit("mid_vwap_dist"))
    .otherwise(pl.lit("far_vwap"))
    .alias("vwap_distance"),

    pl.when(pl.col("atr_14") < 15)
    .then(pl.lit("low_atr"))
    .when(pl.col("atr_14") < 30)
    .then(pl.lit("mid_atr"))
    .otherwise(pl.lit("high_atr"))
    .alias("atr_regime"),

    pl.when(pl.col("rel_vol_20") < 0.8)
    .then(pl.lit("low_vol"))
    .when(pl.col("rel_vol_20") < 1.2)
    .then(pl.lit("normal_vol"))
    .otherwise(pl.lit("high_vol"))
    .alias("volume_regime"),

    pl.when(pl.col("body_pct") >= 0.65)
    .then(pl.lit("strong_body"))
    .when(pl.col("body_pct") >= 0.40)
    .then(pl.lit("medium_body"))
    .otherwise(pl.lit("weak_body"))
    .alias("body_regime"),

    pl.when(pl.col("hour_ct") == 8)
    .then(pl.lit("8_ct"))
    .when(pl.col("hour_ct") == 9)
    .then(pl.lit("9_ct"))
    .when(pl.col("hour_ct") == 10)
    .then(pl.lit("10_ct"))
    .otherwise(pl.lit("11_ct"))
    .alias("hour_bucket"),
])

# ---------------------------------------
# Backtest helper
# Short entry at event close
# Stop = event candle high
# Targets = fixed NQ points
# ---------------------------------------

TARGETS = [25.0, 40.0, 60.0, 80.0]
MAX_HOLDS = [30, 60, 90, 120]
MIN_RISKS = [5.0, 8.0, 10.0]
MAX_RISKS = [25.0, 35.0, 50.0]

REGIME_COLS = [
    "hour_bucket",
    "weekday_ct",
    "vwap_side",
    "vwap_distance",
    "atr_regime",
    "volume_regime",
    "body_regime",
]

results = []
trade_rows = []

for target, max_hold, min_risk, max_risk in product(TARGETS, MAX_HOLDS, MIN_RISKS, MAX_RISKS):

    for combo_size in [1, 2, 3]:
        from itertools import combinations

        for cols in combinations(REGIME_COLS, combo_size):

            grouped_keys = list(cols)

            groups = (
                events
                .group_by(grouped_keys)
                .agg(pl.len().alias("event_count"))
                .filter(pl.col("event_count") >= 40)
            )

            for g in groups.iter_rows(named=True):

                filt = events

                label_parts = []

                for col in grouped_keys:
                    val = g[col]
                    filt = filt.filter(pl.col(col) == val)
                    label_parts.append(f"{col}={val}")

                trades = []

                for row in filt.iter_rows(named=True):

                    entry_time = row["ts_event"]
                    entry = row["close"]
                    stop = row["high"]
                    risk = stop - entry

                    if risk <= 0:
                        continue

                    if risk < min_risk or risk > max_risk:
                        continue

                    target_price = entry - target

                    future = (
                        df
                        .filter(
                            (pl.col("ts_event") > entry_time) &
                            (pl.col("ts_event") <= entry_time + pl.duration(minutes=max_hold))
                        )
                        .select(["ts_event", "high", "low", "close"])
                    )

                    if future.height == 0:
                        continue

                    status = "timeout"
                    exit_price = future[-1, "close"]

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
                    r = points / risk

                    trades.append({
                        "year": row["year"],
                        "points": points,
                        "r": r,
                        "status": status,
                        "risk": risk,
                    })

                if len(trades) < 40:
                    continue

                tdf = pl.DataFrame(trades)

                year_stats = (
                    tdf
                    .group_by("year")
                    .agg([
                        pl.len().alias("year_trades"),
                        pl.col("points").sum().alias("year_points"),
                    ])
                    .with_columns((pl.col("year_points") > 0).alias("positive_year"))
                )

                positive_years = year_stats.select(pl.col("positive_year").sum()).item()
                total_years = year_stats.height

                summary = tdf.select([
                    pl.len().alias("trades"),
                    (pl.col("status") == "target").mean().alias("target_rate"),
                    pl.col("points").mean().alias("avg_points"),
                    pl.col("points").sum().alias("total_points"),
                    pl.col("r").mean().alias("avg_r"),
                    pl.col("r").sum().alias("total_r"),
                    pl.col("risk").mean().alias("avg_risk"),
                ]).to_dicts()[0]

                summary.update({
                    "setup": "First PMH sweep reject",
                    "direction": "short",
                    "filter": " | ".join(label_parts),
                    "target_points": target,
                    "max_hold_minutes": max_hold,
                    "min_risk": min_risk,
                    "max_risk": max_risk,
                    "positive_years": positive_years,
                    "total_years": total_years,
                    "positive_year_rate": positive_years / total_years if total_years else 0,
                })

                results.append(summary)

if not results:
    print("\nNo results found.")
    raise SystemExit

results_df = pl.DataFrame(results)

results_df = results_df.with_columns([
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
        (pl.col("trades").clip(upper_bound=500) / 500)
    ).round(4).alias("edge_score")
])

ranked = (
    results_df
    .filter(
        (pl.col("trades") >= 75) &
        (pl.col("avg_points") > 0) &
        (pl.col("positive_year_rate") >= 0.55)
    )
    .sort("edge_score", descending=True)
)

print("\nTOP 50 PMH REJECTION REGIME RESULTS:")
print(
    ranked
    .select([
        "filter",
        "target_points",
        "max_hold_minutes",
        "min_risk",
        "max_risk",
        "trades",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "positive_years",
        "total_years",
        "positive_year_rate_pct",
        "edge_score",
    ])
    .head(50)
)

all_file = OUT_DIR / "pmh_rejection_regime_all_results.csv"
ranked_file = OUT_DIR / "pmh_rejection_regime_ranked_results.csv"

results_df.write_csv(all_file)
ranked.write_csv(ranked_file)

print("\nSaved:")
print(all_file)
print(ranked_file)

print("\nDONE.")