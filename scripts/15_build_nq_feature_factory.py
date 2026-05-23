import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")

print("\nLoading NQ master dataset...\n")
df = pl.read_parquet(DATA_DIR / "NQ_master.parquet").sort("ts_event")
print(f"Rows loaded: {len(df):,}")


# ---------------------------------------------------
# Time / session features
# Approx CT = UTC-6 for now. Later we can improve DST.
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("ts_event") - pl.duration(hours=6)).alias("ts_ct")
])

df = df.with_columns([
    pl.col("ts_ct").dt.date().alias("trade_date_ct"),
    pl.col("ts_ct").dt.hour().alias("hour_ct"),
    pl.col("ts_ct").dt.minute().alias("minute_ct"),
    pl.col("ts_ct").dt.weekday().alias("weekday_ct"),
])

df = df.with_columns([
    (
        (
            pl.col("hour_ct").cast(pl.Int32) * 60 +
            pl.col("minute_ct").cast(pl.Int32)
        )
        .cast(pl.Int32)
    ).alias("minute_of_day_ct")
])

df = df.with_columns([
    ((pl.col("minute_of_day_ct") >= 8 * 60 + 30) & (pl.col("minute_of_day_ct") <= 15 * 60)).alias("is_rth"),
    ((pl.col("minute_of_day_ct") >= 8 * 60 + 30) & (pl.col("minute_of_day_ct") < 9 * 60)).alias("is_opening_30m"),
    ((pl.col("minute_of_day_ct") >= 7 * 60) & (pl.col("minute_of_day_ct") < 8 * 60 + 30)).alias("is_premarket_window"),
    ((pl.col("minute_of_day_ct") >= 8 * 60 + 30) & (pl.col("minute_of_day_ct") <= 11 * 60 + 30)).alias("is_morning_trade_window"),
    ((pl.col("minute_of_day_ct") >= 11 * 60 + 30) & (pl.col("minute_of_day_ct") <= 13 * 60)).alias("is_lunch_window"),
    ((pl.col("minute_of_day_ct") >= 13 * 60) & (pl.col("minute_of_day_ct") <= 15 * 60)).alias("is_pm_window"),
])


# ---------------------------------------------------
# Candle anatomy
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("high") - pl.col("low")).alias("bar_range"),
    (pl.col("close") - pl.col("open")).alias("body"),
    (pl.col("close") - pl.col("open")).abs().alias("body_abs"),
    (pl.col("high") - pl.max_horizontal(["open", "close"])).alias("upper_wick"),
    (pl.min_horizontal(["open", "close"]) - pl.col("low")).alias("lower_wick"),
])

df = df.with_columns([
    pl.when(pl.col("bar_range") > 0)
    .then(pl.col("body_abs") / pl.col("bar_range"))
    .otherwise(None)
    .alias("body_pct"),

    pl.when(pl.col("bar_range") > 0)
    .then(pl.col("upper_wick") / pl.col("bar_range"))
    .otherwise(None)
    .alias("upper_wick_pct"),

    pl.when(pl.col("bar_range") > 0)
    .then(pl.col("lower_wick") / pl.col("bar_range"))
    .otherwise(None)
    .alias("lower_wick_pct"),

    pl.when(pl.col("body") > 0)
    .then(pl.lit("bull"))
    .when(pl.col("body") < 0)
    .then(pl.lit("bear"))
    .otherwise(pl.lit("doji"))
    .alias("candle_direction"),
])


# ---------------------------------------------------
# Displacement / impulse candles
# IMPORTANT: split into multiple with_columns blocks
# ---------------------------------------------------

df = df.with_columns([
    pl.col("bar_range").rolling_mean(20).alias("avg_range_20"),
    pl.col("body_abs").rolling_mean(20).alias("avg_body_20"),
])

df = df.with_columns([
    (pl.col("bar_range") / pl.col("avg_range_20")).alias("range_expansion_20"),
    (pl.col("body_abs") / pl.col("avg_body_20")).alias("body_expansion_20"),
])

df = df.with_columns([
    (
        (pl.col("body_pct") >= 0.65) &
        (pl.col("range_expansion_20") >= 1.5)
    ).alias("is_displacement_candle"),

    (
        (pl.col("body") > 0) &
        (pl.col("body_pct") >= 0.65) &
        (pl.col("range_expansion_20") >= 1.5)
    ).alias("bull_displacement"),

    (
        (pl.col("body") < 0) &
        (pl.col("body_pct") >= 0.65) &
        (pl.col("range_expansion_20") >= 1.5)
    ).alias("bear_displacement"),
])


# ---------------------------------------------------
# FVG detection
# Bullish FVG: current low > high from 2 bars ago
# Bearish FVG: current high < low from 2 bars ago
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("low") > pl.col("high").shift(2)).alias("bull_fvg"),
    (pl.col("high") < pl.col("low").shift(2)).alias("bear_fvg"),
])

df = df.with_columns([
    pl.when(pl.col("bull_fvg"))
    .then(pl.col("high").shift(2))
    .otherwise(None)
    .alias("bull_fvg_low"),

    pl.when(pl.col("bull_fvg"))
    .then(pl.col("low"))
    .otherwise(None)
    .alias("bull_fvg_high"),

    pl.when(pl.col("bear_fvg"))
    .then(pl.col("high"))
    .otherwise(None)
    .alias("bear_fvg_low"),

    pl.when(pl.col("bear_fvg"))
    .then(pl.col("low").shift(2))
    .otherwise(None)
    .alias("bear_fvg_high"),
])

df = df.with_columns([
    (pl.col("bull_fvg_high") - pl.col("bull_fvg_low")).alias("bull_fvg_size"),
    (pl.col("bear_fvg_high") - pl.col("bear_fvg_low")).alias("bear_fvg_size"),
])

df = df.with_columns([
    (
        pl.col("bull_fvg") &
        (pl.col("bull_fvg_size") >= 2.0)
    ).alias("bull_fvg_min_2pt"),

    (
        pl.col("bear_fvg") &
        (pl.col("bear_fvg_size") >= 2.0)
    ).alias("bear_fvg_min_2pt"),
])


# ---------------------------------------------------
# Prior candle / 2-bar structure
# ---------------------------------------------------

df = df.with_columns([
    pl.col("high").shift(1).alias("prior_high"),
    pl.col("low").shift(1).alias("prior_low"),
    pl.col("open").shift(1).alias("prior_open"),
    pl.col("close").shift(1).alias("prior_close"),
    pl.col("body").shift(1).alias("prior_body"),
    pl.col("body_pct").shift(1).alias("prior_body_pct"),
])

df = df.with_columns([
    (pl.col("high") > pl.col("prior_high")).alias("swept_prior_bar_high"),
    (pl.col("low") < pl.col("prior_low")).alias("swept_prior_bar_low"),

    (
        (pl.col("high") > pl.col("prior_high")) &
        (pl.col("close") < pl.col("prior_high"))
    ).alias("prior_high_sweep_close_back_below"),

    (
        (pl.col("low") < pl.col("prior_low")) &
        (pl.col("close") > pl.col("prior_low"))
    ).alias("prior_low_sweep_close_back_above"),
])


# ---------------------------------------------------
# Daily / session levels
# ---------------------------------------------------

rth_daily = (
    df.filter(pl.col("is_rth"))
    .group_by("trade_date_ct")
    .agg([
        pl.col("high").max().alias("rth_day_high"),
        pl.col("low").min().alias("rth_day_low"),
        pl.col("close").last().alias("rth_day_close"),
    ])
    .sort("trade_date_ct")
)

rth_daily = rth_daily.with_columns([
    pl.col("rth_day_high").shift(1).alias("prior_rth_high"),
    pl.col("rth_day_low").shift(1).alias("prior_rth_low"),
    pl.col("rth_day_close").shift(1).alias("prior_rth_close"),
])

df = df.join(
    rth_daily.select([
        "trade_date_ct",
        "prior_rth_high",
        "prior_rth_low",
        "prior_rth_close",
    ]),
    on="trade_date_ct",
    how="left",
)

premarket = (
    df.filter(pl.col("is_premarket_window"))
    .group_by("trade_date_ct")
    .agg([
        pl.col("high").max().alias("premarket_high"),
        pl.col("low").min().alias("premarket_low"),
    ])
)

df = df.join(premarket, on="trade_date_ct", how="left")

opening_range = (
    df.filter(pl.col("is_opening_30m"))
    .group_by("trade_date_ct")
    .agg([
        pl.col("high").max().alias("or_high_30m"),
        pl.col("low").min().alias("or_low_30m"),
    ])
)

df = df.join(opening_range, on="trade_date_ct", how="left")

df = df.with_columns([
    (pl.col("or_high_30m") - pl.col("or_low_30m")).alias("or_range_30m"),
    (pl.col("premarket_high") - pl.col("premarket_low")).alias("premarket_range"),
    (pl.col("prior_rth_high") - pl.col("prior_rth_low")).alias("prior_day_range"),
])


# ---------------------------------------------------
# Sweep flags
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("high") > pl.col("prior_rth_high")).alias("swept_prior_rth_high"),
    (pl.col("low") < pl.col("prior_rth_low")).alias("swept_prior_rth_low"),

    (
        (pl.col("high") > pl.col("prior_rth_high")) &
        (pl.col("close") < pl.col("prior_rth_high"))
    ).alias("pdh_sweep_close_back_below"),

    (
        (pl.col("low") < pl.col("prior_rth_low")) &
        (pl.col("close") > pl.col("prior_rth_low"))
    ).alias("pdl_sweep_close_back_above"),

    (
        (pl.col("high") > pl.col("premarket_high")) &
        (pl.col("close") < pl.col("premarket_high"))
    ).alias("pmh_sweep_close_back_below"),

    (
        (pl.col("low") < pl.col("premarket_low")) &
        (pl.col("close") > pl.col("premarket_low"))
    ).alias("pml_sweep_close_back_above"),

    (pl.col("close") > pl.col("or_high_30m")).alias("above_or_high"),
    (pl.col("close") < pl.col("or_low_30m")).alias("below_or_low"),
])


# ---------------------------------------------------
# Distance to key levels
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("close") - pl.col("prior_rth_high")).alias("dist_prior_rth_high"),
    (pl.col("close") - pl.col("prior_rth_low")).alias("dist_prior_rth_low"),
    (pl.col("close") - pl.col("premarket_high")).alias("dist_premarket_high"),
    (pl.col("close") - pl.col("premarket_low")).alias("dist_premarket_low"),
    (pl.col("close") - pl.col("or_high_30m")).alias("dist_or_high_30m"),
    (pl.col("close") - pl.col("or_low_30m")).alias("dist_or_low_30m"),
])


# ---------------------------------------------------
# Forward outcomes in points
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("close").shift(-5) - pl.col("close")).alias("fwd_points_5m"),
    (pl.col("close").shift(-15) - pl.col("close")).alias("fwd_points_15m"),
    (pl.col("close").shift(-30) - pl.col("close")).alias("fwd_points_30m"),
    (pl.col("close").shift(-60) - pl.col("close")).alias("fwd_points_60m"),
    (pl.col("close").shift(-120) - pl.col("close")).alias("fwd_points_120m"),
])

df = df.with_columns([
    pl.col("high").reverse().rolling_max(30).reverse().alias("future_high_30m"),
    pl.col("low").reverse().rolling_min(30).reverse().alias("future_low_30m"),
    pl.col("high").reverse().rolling_max(60).reverse().alias("future_high_60m"),
    pl.col("low").reverse().rolling_min(60).reverse().alias("future_low_60m"),
])

df = df.with_columns([
    (pl.col("future_high_30m") - pl.col("close")).alias("mfe_30m"),
    (pl.col("close") - pl.col("future_low_30m")).alias("mae_30m"),
    (pl.col("future_high_60m") - pl.col("close")).alias("mfe_60m"),
    (pl.col("close") - pl.col("future_low_60m")).alias("mae_60m"),
])


# ---------------------------------------------------
# Save
# ---------------------------------------------------

print("\nPreview:")
print(
    df.select([
        "ts_event",
        "ts_ct",
        "close",
        "body_pct",
        "range_expansion_20",
        "bull_fvg",
        "bear_fvg",
        "bull_fvg_size",
        "bear_fvg_size",
        "pdh_sweep_close_back_below",
        "pdl_sweep_close_back_above",
        "fwd_points_30m",
        "mfe_30m",
        "mae_30m",
    ]).head(30)
)

output_file = DATA_DIR / "NQ_feature_factory.parquet"

print(f"\nWriting:\n{output_file}")
df.write_parquet(output_file)

print("\nDONE.")