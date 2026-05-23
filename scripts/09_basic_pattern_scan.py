import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
REPORT_DIR = Path(r"D:\TradingResearch\reports")

print("\nLoading feature dataset...\n")

df = pl.read_parquet(DATA_DIR / "NQ_session_features.parquet")

print(f"Rows loaded: {len(df):,}")

# Remove rows with missing key fields
df = df.filter(
    pl.col("atr_14").is_not_null() &
    pl.col("rel_vol_20").is_not_null() &
    pl.col("fwd_return_30m").is_not_null() &
    pl.col("body_pct").is_finite()
)

print(f"Rows after filtering: {len(df):,}")

# Convert forward returns to points
df = df.with_columns([
    ((pl.col("close").shift(-15) - pl.col("close"))).alias("fwd_points_15m"),
    ((pl.col("close").shift(-30) - pl.col("close"))).alias("fwd_points_30m"),
    ((pl.col("close").shift(-60) - pl.col("close"))).alias("fwd_points_60m"),
])

# Define condition buckets
df = df.with_columns([
    pl.when(pl.col("dist_vwap_day") > 20).then(pl.lit("far_above_vwap"))
      .when(pl.col("dist_vwap_day") > 5).then(pl.lit("above_vwap"))
      .when(pl.col("dist_vwap_day") < -20).then(pl.lit("far_below_vwap"))
      .when(pl.col("dist_vwap_day") < -5).then(pl.lit("below_vwap"))
      .otherwise(pl.lit("near_vwap"))
      .alias("vwap_bucket"),

    pl.when(pl.col("rel_vol_20") >= 2.0).then(pl.lit("high_rel_vol"))
      .when(pl.col("rel_vol_20") >= 1.2).then(pl.lit("above_avg_vol"))
      .when(pl.col("rel_vol_20") <= 0.6).then(pl.lit("low_vol"))
      .otherwise(pl.lit("normal_vol"))
      .alias("volume_bucket"),

    pl.when(pl.col("body_pct") >= 0.7).then(pl.lit("strong_body"))
      .when(pl.col("body_pct") <= 0.25).then(pl.lit("wicky"))
      .otherwise(pl.lit("normal_body"))
      .alias("candle_bucket"),

    pl.when(pl.col("close") > pl.col("ema_20")).then(pl.lit("above_ema20"))
      .otherwise(pl.lit("below_ema20"))
      .alias("ema20_bucket"),

    pl.when(pl.col("return_15m") > 0.002).then(pl.lit("strong_up_prior_15m"))
      .when(pl.col("return_15m") < -0.002).then(pl.lit("strong_down_prior_15m"))
      .otherwise(pl.lit("flat_prior_15m"))
      .alias("prior_momentum_bucket"),
])

# Long outcome: +20 points in 30m
# Short outcome: -20 points in 30m
df = df.with_columns([
    (pl.col("fwd_points_30m") >= 20).alias("long_20pt_win_30m"),
    (pl.col("fwd_points_30m") <= -20).alias("short_20pt_win_30m"),
])

group_cols = [
    "vwap_bucket",
    "volume_bucket",
    "candle_bucket",
    "ema20_bucket",
    "prior_momentum_bucket",
]

summary = (
    df.group_by(group_cols)
    .agg([
        pl.len().alias("samples"),
        pl.col("long_20pt_win_30m").mean().alias("long_win_rate_20pt_30m"),
        pl.col("short_20pt_win_30m").mean().alias("short_win_rate_20pt_30m"),
        pl.col("fwd_points_15m").mean().alias("avg_fwd_points_15m"),
        pl.col("fwd_points_30m").mean().alias("avg_fwd_points_30m"),
        pl.col("fwd_points_60m").mean().alias("avg_fwd_points_60m"),
    ])
    .filter(pl.col("samples") >= 200)
    .sort("long_win_rate_20pt_30m", descending=True)
)

print("\nTop long-biased condition groups:")
print(summary.head(25))

long_report = REPORT_DIR / "basic_pattern_scan_long.csv"
summary.write_csv(long_report)

summary_short = summary.sort("short_win_rate_20pt_30m", descending=True)

print("\nTop short-biased condition groups:")
print(summary_short.head(25))

short_report = REPORT_DIR / "basic_pattern_scan_short.csv"
summary_short.write_csv(short_report)

print(f"\nSaved reports:")
print(long_report)
print(short_report)

print("\nDONE.")