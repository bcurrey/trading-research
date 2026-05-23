import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUTPUT_DIR = Path(r"D:\TradingResearch\data_parquet")

print("\nLoading NQ continuous dataset...\n")

df = pl.read_parquet(DATA_DIR / "NQ_continuous_1m.parquet").sort("ts_event")

print(f"Rows loaded: {len(df):,}")

# ---------------------------------------------------
# Time features
# Note: source timestamps are UTC.
# Central Time = UTC-6 standard / UTC-5 daylight.
# For now, use UTC fields; later we can add proper CT conversion.
# ---------------------------------------------------

df = df.with_columns([
    pl.col("ts_event").dt.date().alias("trade_date_utc"),
    pl.col("ts_event").dt.hour().alias("hour_utc"),
    pl.col("ts_event").dt.minute().alias("minute_utc"),
    pl.col("ts_event").dt.weekday().alias("weekday"),
])

# ---------------------------------------------------
# Daily VWAP reset by UTC date for now
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("close") * pl.col("volume")).alias("pv")
])

df = df.with_columns([
    pl.col("pv").cum_sum().over("trade_date_utc").alias("cum_pv_day"),
    pl.col("volume").cum_sum().over("trade_date_utc").alias("cum_vol_day"),
])

df = df.with_columns([
    (pl.col("cum_pv_day") / pl.col("cum_vol_day")).alias("vwap_day"),
    (pl.col("close") - pl.col("cum_pv_day") / pl.col("cum_vol_day")).alias("dist_vwap_day"),
])

# ---------------------------------------------------
# Returns
# ---------------------------------------------------

df = df.with_columns([
    ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("return_1m"),
    ((pl.col("close") / pl.col("close").shift(5)) - 1).alias("return_5m"),
    ((pl.col("close") / pl.col("close").shift(15)) - 1).alias("return_15m"),
])

# ---------------------------------------------------
# EMA
# ---------------------------------------------------

df = df.with_columns([
    pl.col("close").ewm_mean(span=9).alias("ema_9"),
    pl.col("close").ewm_mean(span=20).alias("ema_20"),
    pl.col("close").ewm_mean(span=50).alias("ema_50"),
])

df = df.with_columns([
    (pl.col("close") - pl.col("ema_9")).alias("dist_ema_9"),
    (pl.col("close") - pl.col("ema_20")).alias("dist_ema_20"),
    (pl.col("close") - pl.col("ema_50")).alias("dist_ema_50"),
])

# ---------------------------------------------------
# True Range / ATR
# ---------------------------------------------------

df = df.with_columns([
    pl.max_horizontal([
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    ]).alias("true_range")
])

df = df.with_columns([
    pl.col("true_range").rolling_mean(window_size=14).alias("atr_14"),
    pl.col("true_range").rolling_mean(window_size=50).alias("atr_50"),
])

# ---------------------------------------------------
# Volume features
# ---------------------------------------------------

df = df.with_columns([
    pl.col("volume").rolling_mean(window_size=20).alias("vol_avg_20"),
    pl.col("volume").rolling_mean(window_size=50).alias("vol_avg_50"),
])

df = df.with_columns([
    (pl.col("volume") / pl.col("vol_avg_20")).alias("rel_vol_20"),
    (pl.col("volume") / pl.col("vol_avg_50")).alias("rel_vol_50"),
])

# ---------------------------------------------------
# Candle structure
# ---------------------------------------------------

df = df.with_columns([
    (pl.col("high") - pl.col("low")).alias("bar_range"),
    (pl.col("close") - pl.col("open")).alias("body"),
    (pl.col("close") - pl.col("open")).abs().alias("body_abs"),
])

df = df.with_columns([
    (pl.col("body_abs") / pl.col("bar_range")).alias("body_pct"),
    (pl.col("high") - pl.max_horizontal(["open", "close"])).alias("upper_wick"),
    (pl.min_horizontal(["open", "close"]) - pl.col("low")).alias("lower_wick"),
])

# ---------------------------------------------------
# Forward outcomes
# ---------------------------------------------------

df = df.with_columns([
    ((pl.col("close").shift(-5) / pl.col("close")) - 1).alias("fwd_return_5m"),
    ((pl.col("close").shift(-15) / pl.col("close")) - 1).alias("fwd_return_15m"),
    ((pl.col("close").shift(-30) / pl.col("close")) - 1).alias("fwd_return_30m"),
    ((pl.col("close").shift(-60) / pl.col("close")) - 1).alias("fwd_return_60m"),
])

# ---------------------------------------------------
# Save
# ---------------------------------------------------

df = df.drop([
    "pv",
    "cum_pv_day",
    "cum_vol_day",
])

print("\nPreview:")
print(
    df.select([
        "ts_event",
        "symbol",
        "close",
        "vwap_day",
        "dist_vwap_day",
        "atr_14",
        "rel_vol_20",
        "body_pct",
        "fwd_return_15m",
    ]).head(20)
)

output_file = OUTPUT_DIR / "NQ_session_features.parquet"

print(f"\nWriting:\n{output_file}")
df.write_parquet(output_file)

print("\nDONE.")