from pathlib import Path
import polars as pl
import time

START = time.time()

DATA = Path(
    r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet"
)

print("\nLoading master dataset...\n")

lf = pl.scan_parquet(DATA)

# =========================================================
# FAILED OR LOW BREAKDOWN LONG MODEL
# =========================================================

print("Building OR failure models...\n")

study = (

    lf

    # =====================================================
    # RTH ONLY
    # =====================================================

    .filter(
        pl.col("is_rth") == 1
    )

    # =====================================================
    # OPENING WINDOW
    # =====================================================

    .filter(
        (
            (pl.col("hour_ct") == 8) &
            (pl.col("minute_ct") >= 30)
        ) |
        (
            (pl.col("hour_ct") == 9) &
            (pl.col("minute_ct") <= 30)
        )
    )

    # =====================================================
    # FAILED PRIOR LOW SWEEP
    # =====================================================

    .with_columns([

        (
            (pl.col("swept_prior_rth_low") == 1) &
            (pl.col("close") > pl.col("prior_rth_low"))
        ).cast(pl.Int8).alias("failed_low_reclaim"),

        (
            (pl.col("close") > pl.col("vwap_day"))
        ).cast(pl.Int8).alias("above_vwap"),

        (
            (pl.col("close") < pl.col("vwap_day"))
        ).cast(pl.Int8).alias("below_vwap"),

        (
            pl.col("quote_pressure_delta") > 0
        ).cast(pl.Int8).alias("bull_quote_pressure"),

        (
            pl.col("net_mid_pressure") > 0
        ).cast(pl.Int8).alias("bull_mid_pressure"),

        (
            pl.col("avg_spread") <= 1.0
        ).cast(pl.Int8).alias("tight_spread"),

        (
            pl.col("update_velocity") > 1000
        ).cast(pl.Int8).alias("high_velocity"),

        (
            pl.col("rel_vol_20") > 1.25
        ).cast(pl.Int8).alias("high_rel_vol"),

        (
            pl.col("bar_delta_master") > 0
        ).cast(pl.Int8).alias("bull_bar"),

        (
            pl.col("close") >
            pl.col("open")
        ).cast(pl.Int8).alias("green_bar"),

    ])

    # =====================================================
    # CORE FAILED AUCTION SETUP
    # =====================================================

    .filter(
        pl.col("failed_low_reclaim") == 1
    )

    # =====================================================
    # TARGETS
    # =====================================================

    .with_columns([

        (
            pl.col("master_fwd_15m") >= 10
        ).cast(pl.Int8).alias("win_10pt"),

        (
            pl.col("master_fwd_15m") >= 20
        ).cast(pl.Int8).alias("win_20pt"),

        (
            pl.col("master_fwd_30m") >= 30
        ).cast(pl.Int8).alias("win_30pt"),

        (
            pl.col("master_fwd_60m") >= 40
        ).cast(pl.Int8).alias("win_40pt"),

    ])

)

print("\nCOLLECTING DATA...\n")

df = study.collect(engine="streaming")

print(f"Rows: {df.height:,}")

# =========================================================
# OVERALL RESULTS
# =========================================================

print("\n================ OVERALL ==================\n")

overall = df.select([

    pl.len().alias("trades"),

    pl.mean("win_10pt").alias("winrate_10pt"),
    pl.mean("win_20pt").alias("winrate_20pt"),
    pl.mean("win_30pt").alias("winrate_30pt"),
    pl.mean("win_40pt").alias("winrate_40pt"),

    pl.mean("master_fwd_15m").alias("avg_15m"),
    pl.mean("master_fwd_30m").alias("avg_30m"),
    pl.mean("master_fwd_60m").alias("avg_60m"),

])

print(overall)

# =========================================================
# CONDITION STUDIES
# =========================================================

conditions = [

    "bull_quote_pressure",
    "bull_mid_pressure",
    "tight_spread",
    "high_velocity",
    "high_rel_vol",
    "green_bar",
    "above_vwap",
    "below_vwap",

]

for cond in conditions:

    print(f"\n================ {cond} ==================\n")

    out = (

        df

        .group_by(cond)

        .agg([

            pl.len().alias("trades"),

            pl.mean("win_10pt").alias("winrate_10pt"),
            pl.mean("win_20pt").alias("winrate_20pt"),
            pl.mean("win_30pt").alias("winrate_30pt"),
            pl.mean("win_40pt").alias("winrate_40pt"),

            pl.mean("master_fwd_15m").alias("avg_15m"),
            pl.mean("master_fwd_30m").alias("avg_30m"),
            pl.mean("master_fwd_60m").alias("avg_60m"),

        ])

        .sort(cond)

    )

    print(out)

# =========================================================
# STACKED MODEL
# =========================================================

print("\n================ STACKED MODEL ==================\n")

stacked = (

    df

    .with_columns([

        (
            pl.col("bull_quote_pressure") +
            pl.col("bull_mid_pressure") +
            pl.col("tight_spread") +
            pl.col("high_velocity") +
            pl.col("high_rel_vol") +
            pl.col("green_bar")
        ).alias("stack_score")

    ])

    .group_by("stack_score")

    .agg([

        pl.len().alias("trades"),

        pl.mean("win_10pt").alias("winrate_10pt"),
        pl.mean("win_20pt").alias("winrate_20pt"),
        pl.mean("win_30pt").alias("winrate_30pt"),
        pl.mean("win_40pt").alias("winrate_40pt"),

        pl.mean("master_fwd_15m").alias("avg_15m"),
        pl.mean("master_fwd_30m").alias("avg_30m"),
        pl.mean("master_fwd_60m").alias("avg_60m"),

    ])

    .sort("stack_score")

)

print(stacked)

# =========================================================
# SAVE RESULTS
# =========================================================

OUTFILE = Path(
    r"D:\TradingResearch\data_parquet\orb_failed_auction_research.csv"
)

stacked.write_csv(OUTFILE)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")
print(f"Saved: {OUTFILE}")