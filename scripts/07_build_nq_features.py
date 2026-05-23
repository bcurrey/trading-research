import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUTPUT_DIR = Path(r"D:\TradingResearch\data_parquet")

print("\nLoading NQ continuous dataset...\n")

df = pl.read_parquet(
    DATA_DIR / "NQ_continuous_1m.parquet"
)

print(f"Rows loaded: {len(df):,}")

# ---------------------------------------------------
# SORT
# ---------------------------------------------------

df = df.sort("ts_event")

# ---------------------------------------------------
# BASIC RETURNS
# ---------------------------------------------------

df = df.with_columns([
    (
        (pl.col("close") / pl.col("close").shift(1)) - 1
    ).alias("return_1m"),

    (
        (pl.col("close") / pl.col("close").shift(5)) - 1
    ).alias("return_5m"),

    (
        (pl.col("close") / pl.col("close").shift(15)) - 1
    ).alias("return_15m"),
])

# ---------------------------------------------------
# EMA
# ---------------------------------------------------

df = df.with_columns([
    pl.col("close")
    .ewm_mean(span=9)
    .alias("ema_9"),

    pl.col("close")
    .ewm_mean(span=20)
    .alias("ema_20"),

    pl.col("close")
    .ewm_mean(span=50)
    .alias("ema_50"),
])

# ---------------------------------------------------
# TRUE RANGE
# ---------------------------------------------------

df = df.with_columns([
    pl.max_horizontal([
        (pl.col("high") - pl.col("low")),

        (pl.col("high") - pl.col("close").shift(1)).abs(),

        (pl.col("low") - pl.col("close").shift(1)).abs()
    ]).alias("true_range")
])

# ---------------------------------------------------
# ATR
# ---------------------------------------------------

df = df.with_columns([
    pl.col("true_range")
    .rolling_mean(window_size=14)
    .alias("atr_14")
])

# ---------------------------------------------------
# VWAP
# ---------------------------------------------------

df = df.with_columns([
    (
        (
            (pl.col("close") * pl.col("volume"))
            .cum_sum()
        ) /
        (
            pl.col("volume")
            .cum_sum()
        )
    ).alias("vwap")
])

# ---------------------------------------------------
# DISTANCE FROM VWAP
# ---------------------------------------------------

df = df.with_columns([
    (
        pl.col("close") - pl.col("vwap")
    ).alias("dist_vwap")
])

# ---------------------------------------------------
# FORWARD RETURNS
# ---------------------------------------------------

df = df.with_columns([
    (
        (pl.col("close").shift(-5) / pl.col("close")) - 1
    ).alias("fwd_return_5m"),

    (
        (pl.col("close").shift(-15) / pl.col("close")) - 1
    ).alias("fwd_return_15m"),

    (
        (pl.col("close").shift(-30) / pl.col("close")) - 1
    ).alias("fwd_return_30m"),
])

# ---------------------------------------------------
# TIME FEATURES
# ---------------------------------------------------

df = df.with_columns([
    pl.col("ts_event").dt.hour().alias("hour"),

    pl.col("ts_event").dt.minute().alias("minute"),

    pl.col("ts_event").dt.weekday().alias("weekday"),
])

print("\nFeature preview:\n")

print(
    df.select([
        "ts_event",
        "close",
        "ema_9",
        "ema_20",
        "atr_14",
        "vwap",
        "dist_vwap",
        "fwd_return_15m"
    ]).head(10)
)

output_file = OUTPUT_DIR / "NQ_features.parquet"

print(f"\nWriting feature dataset:\n{output_file}")

df.write_parquet(output_file)

print("\nDONE.")