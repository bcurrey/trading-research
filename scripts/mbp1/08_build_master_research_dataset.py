from pathlib import Path
import polars as pl
import time

START = time.time()

print("\nLOADING DATASETS...\n")

# ============================================
# PATHS
# ============================================

FEATURE_FACTORY = Path(
    r"D:\TradingResearch\data_parquet\NQ_feature_factory.parquet"
)

MBP_ROOT = Path(
    r"D:\TradingResearch\data_parquet\mbp1_1m_features"
)

OUTPUT = Path(
    r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet"
)

# ============================================
# LOAD FEATURE FACTORY
# ============================================

print("Loading feature factory...")

base = (
    pl.scan_parquet(FEATURE_FACTORY)

    .with_columns([
        pl.col("ts_ct")
        .cast(pl.Datetime)
        .dt.truncate("1m")
        .alias("minute")
    ])

    # ============================================
    # SESSION FLAGS
    # ============================================

    .with_columns([

        (
            (pl.col("hour_ct") >= 8) &
            (pl.col("hour_ct") < 15)
        ).cast(pl.Int8).alias("is_rth_session"),

        (
            (
                (pl.col("hour_ct") == 8) &
                (pl.col("minute_ct") >= 30)
            ) |
            (
                (pl.col("hour_ct") > 8) &
                (pl.col("hour_ct") < 9)
            )
        ).cast(pl.Int8).alias("is_or_window"),

        (
            (pl.col("hour_ct") >= 11) &
            (pl.col("hour_ct") < 13)
        ).cast(pl.Int8).alias("is_lunch_session"),

    ])
)

# ============================================
# LOAD MBP FEATURES
# ============================================

print("Loading MBP features...")

mbp_files = sorted(MBP_ROOT.rglob("*.parquet"))

mbp = (
    pl.scan_parquet([str(x) for x in mbp_files])

    .with_columns([
        pl.col("minute")
        .cast(pl.Datetime)
        .dt.truncate("1m")
        .alias("minute")
    ])

    .select([
        "minute",

        "updates",

        "avg_spread",
        "median_spread",

        "avg_bid_sz",
        "avg_ask_sz",

        "avg_top_imbalance_ratio",

        "mid_delta",
        "mid_range",

        "mid_up_updates",
        "mid_down_updates",

        "bid_lifted_updates",
        "ask_hit_updates",

        "quote_pressure_delta",
    ])
)

# ============================================
# JOIN DATASETS
# ============================================

print("Joining datasets...")

master = (

    base.join(
        mbp,
        on="minute",
        how="left",
    )

    # ============================================
    # FILL NULLS
    # ============================================

    .with_columns([

        pl.col("updates").fill_null(0),

        pl.col("avg_spread").fill_null(0),
        pl.col("median_spread").fill_null(0),

        pl.col("avg_bid_sz").fill_null(0),
        pl.col("avg_ask_sz").fill_null(0),

        pl.col("avg_top_imbalance_ratio").fill_null(0),

        pl.col("mid_delta").fill_null(0),
        pl.col("mid_range").fill_null(0),

        pl.col("mid_up_updates").fill_null(0),
        pl.col("mid_down_updates").fill_null(0),

        pl.col("bid_lifted_updates").fill_null(0),
        pl.col("ask_hit_updates").fill_null(0),

        pl.col("quote_pressure_delta").fill_null(0),

    ])

    # ============================================
    # ORDERFLOW FEATURES
    # ============================================

    .with_columns([

        (
            pl.col("mid_up_updates") -
            pl.col("mid_down_updates")
        ).alias("net_mid_pressure"),

        (
            pl.col("bid_lifted_updates") -
            pl.col("ask_hit_updates")
        ).alias("net_quote_pressure"),

        (
            pl.col("avg_bid_sz") -
            pl.col("avg_ask_sz")
        ).alias("book_size_delta"),

        (
            pl.col("updates") /
            (
                pl.col("mid_range") + 0.25
            )
        ).alias("update_velocity"),

    ])

    # ============================================
    # FAILED AUCTION FEATURES
    # ============================================

    .with_columns([

        (
            (pl.col("swept_prior_rth_low") == 1) &
            (pl.col("close") > pl.col("prior_rth_low"))
        ).cast(pl.Int8).alias("failed_pdl_break"),

        (
            (pl.col("swept_prior_rth_high") == 1) &
            (pl.col("close") < pl.col("prior_rth_high"))
        ).cast(pl.Int8).alias("failed_pdh_break"),

    ])

    # ============================================
    # BAR FEATURES
    # ============================================

    .with_columns([

        (
            pl.col("high") -
            pl.col("low")
        ).alias("bar_range_master"),

        (
            pl.col("close") -
            pl.col("open")
        ).alias("bar_delta_master"),

    ])

    # ============================================
    # FORWARD RETURNS
    # ============================================

    .with_columns([

        (
            pl.col("close")
            .shift(-5) -
            pl.col("close")
        ).alias("master_fwd_5m"),

        (
            pl.col("close")
            .shift(-15) -
            pl.col("close")
        ).alias("master_fwd_15m"),

        (
            pl.col("close")
            .shift(-30) -
            pl.col("close")
        ).alias("master_fwd_30m"),

        (
            pl.col("close")
            .shift(-60) -
            pl.col("close")
        ).alias("master_fwd_60m"),

    ])

)

print("\nCOLLECTING FINAL DATASET...\n")

df = master.collect(engine="streaming")

print("ROWS:")
print(f"{df.height:,}")

print("\nCOLUMNS:")
print(len(df.columns))

print("\nSAMPLE:")
print(df.head(5))

print("\nWRITING PARQUET...\n")

df.write_parquet(
    OUTPUT,
    compression="zstd",
    statistics=True,
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")
print(f"Saved: {OUTPUT}")