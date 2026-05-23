import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet")

# ---------------------------------------------------
# Focus setup
# ---------------------------------------------------

setup = df.filter(
    (pl.col("is_morning_trade_window")) &
    (pl.col("pdl_sweep_close_back_above"))
)

print("\nPDL sweep setup rows:", setup.height)

# ---------------------------------------------------
# Add useful regime buckets
# ---------------------------------------------------

setup = setup.with_columns([

    # ATR regime
    pl.when(pl.col("atr_14") < 15)
    .then(pl.lit("low_atr"))
    .when(pl.col("atr_14") < 30)
    .then(pl.lit("mid_atr"))
    .otherwise(pl.lit("high_atr"))
    .alias("atr_regime"),

    # Volume regime
    pl.when(pl.col("rel_vol_20") < 0.8)
    .then(pl.lit("low_vol"))
    .when(pl.col("rel_vol_20") < 1.2)
    .then(pl.lit("normal_vol"))
    .otherwise(pl.lit("high_vol"))
    .alias("volume_regime"),

    # VWAP positioning
    pl.when(pl.col("close") > pl.col("vwap_day"))
    .then(pl.lit("above_vwap"))
    .otherwise(pl.lit("below_vwap"))
    .alias("vwap_side"),

    # Body quality
    pl.when(pl.col("body_pct") >= 0.7)
    .then(pl.lit("strong_body"))
    .when(pl.col("body_pct") >= 0.5)
    .then(pl.lit("medium_body"))
    .otherwise(pl.lit("weak_body"))
    .alias("body_quality"),
])

# ---------------------------------------------------
# Helper function
# ---------------------------------------------------

def summarize(group_col):

    result = (
        setup
        .group_by(group_col)
        .agg([
            pl.len().alias("signals"),

            (pl.col("fwd_points_30m") > 0)
            .mean()
            .alias("win_rate"),

            pl.col("fwd_points_30m")
            .mean()
            .alias("avg_30m"),

            pl.col("fwd_points_60m")
            .mean()
            .alias("avg_60m"),

            pl.col("mfe_30m")
            .mean()
            .alias("avg_mfe"),

            pl.col("mae_30m")
            .mean()
            .alias("avg_mae"),
        ])
        .with_columns([
            (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
            pl.col("avg_30m").round(2),
            pl.col("avg_60m").round(2),
            pl.col("avg_mfe").round(2),
            pl.col("avg_mae").round(2),
        ])
        .sort("avg_30m", descending=True)
    )

    print(f"\n===== {group_col} =====")
    print(result)

    return result


# ---------------------------------------------------
# Deep dives
# ---------------------------------------------------

summarize("hour_ct")

summarize("weekday_ct")

summarize("atr_regime")

summarize("volume_regime")

summarize("vwap_side")

summarize("body_quality")

print("\nDONE.")