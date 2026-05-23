from pathlib import Path
import polars as pl
import itertools
import time

START = time.time()

DATA = Path(
    r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet"
)

OUTPUT_DIR = Path(
    r"D:\TradingResearch\research_outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLOADING MASTER DATASET...\n")

lf = pl.scan_parquet(DATA)

# ============================================================
# BUILD SEQUENCE DATASET
# ============================================================

base = (

    lf

    # ========================================================
    # RTH ONLY
    # ========================================================

    .filter(
        pl.col("is_rth") == 1
    )

    # ========================================================
    # OPENING WINDOW
    # ========================================================

    .filter(

        (
            (pl.col("hour_ct") == 8) &
            (pl.col("minute_ct") >= 30)
        ) |

        (
            (pl.col("hour_ct") == 9) &
            (pl.col("minute_ct") <= 15)
        )

    )

    # ========================================================
    # MUST SWEEP PRIOR LOW
    # ========================================================

    .filter(
        pl.col("swept_prior_rth_low") == 1
    )

    # ========================================================
    # MUST RECLAIM PRIOR LOW
    # ========================================================

    .filter(
        pl.col("close") > pl.col("prior_rth_low")
    )

    # ========================================================
    # CORE STRUCTURE
    # ========================================================

    .with_columns([

        (
            pl.col("close") > pl.col("or_low_30m")
        ).cast(pl.Int8).alias("back_inside_or"),

        (
            pl.col("close") > pl.col("vwap_day")
        ).cast(pl.Int8).alias("above_vwap"),

        (
            pl.col("close") < pl.col("vwap_day")
        ).cast(pl.Int8).alias("below_vwap"),

        (
            pl.col("or_range_30m") > 80
        ).cast(pl.Int8).alias("large_or"),

        (
            pl.col("avg_spread") <= 1.0
        ).cast(pl.Int8).alias("tight_spread"),

        (
            pl.col("quote_pressure_delta") > 0
        ).cast(pl.Int8).alias("bull_quote_pressure"),

        (
            pl.col("net_mid_pressure") > 0
        ).cast(pl.Int8).alias("bull_mid_pressure"),

        (
            pl.col("is_displacement_candle") == 1
        ).cast(pl.Int8).alias("displacement"),

        (
            pl.col("bull_fvg") == 1
        ).cast(pl.Int8).alias("bull_fvg"),

        (
            pl.col("close") > pl.col("open")
        ).cast(pl.Int8).alias("green_bar"),

        (
            pl.col("close") < pl.col("open")
        ).cast(pl.Int8).alias("red_bar"),

        (
            pl.col("body_pct") > 0.7
        ).cast(pl.Int8).alias("strong_body"),

    ])

    # ========================================================
    # SEQUENCE FEATURES
    # ========================================================

    .with_columns([

        # ============================================
        # NEXT BAR CONDITIONS
        # ============================================

        pl.col("tight_spread")
        .shift(-1)
        .alias("next_tight_spread"),

        pl.col("displacement")
        .shift(-1)
        .alias("next_displacement"),

        pl.col("bull_quote_pressure")
        .shift(-1)
        .alias("next_bull_quote_pressure"),

        pl.col("bull_mid_pressure")
        .shift(-1)
        .alias("next_bull_mid_pressure"),

        pl.col("green_bar")
        .shift(-1)
        .alias("next_green_bar"),

        # ============================================
        # TWO BAR FOLLOW THROUGH
        # ============================================

        (
            pl.col("close")
            .shift(-2) >
            pl.col("close")
        ).cast(pl.Int8).alias("two_bar_followthrough"),

        # ============================================
        # CONTINUATION CONDITIONS
        # ============================================

        (
            pl.col("master_fwd_15m") > 20
        ).cast(pl.Int8).alias("strong_15m"),

        (
            pl.col("master_fwd_30m") > 30
        ).cast(pl.Int8).alias("strong_30m"),

        (
            pl.col("master_fwd_60m") > 40
        ).cast(pl.Int8).alias("strong_60m"),

    ])

)

print("\nCOLLECTING SEQUENCE DATA...\n")

df = base.collect(engine="streaming")

print(f"Base setups: {df.height:,}")

# ============================================================
# SEQUENCE COMBINATIONS
# ============================================================

conditions = [

    "back_inside_or",
    "above_vwap",
    "below_vwap",
    "large_or",
    "tight_spread",
    "bull_quote_pressure",
    "bull_mid_pressure",
    "displacement",
    "bull_fvg",
    "green_bar",
    "red_bar",
    "strong_body",

    # sequence
    "next_tight_spread",
    "next_displacement",
    "next_bull_quote_pressure",
    "next_bull_mid_pressure",
    "next_green_bar",
    "two_bar_followthrough",

]

results = []

print("\nRUNNING SEQUENCE TESTS...\n")

combo_count = 0

for r in range(3, 7):

    for combo in itertools.combinations(conditions, r):

        combo_count += 1

        filt = None

        for cond in combo:

            expr = pl.col(cond) == 1

            if filt is None:
                filt = expr
            else:
                filt = filt & expr

        sub = df.filter(filt)

        trades = sub.height

        if trades < 25:
            continue

        stats = sub.select([

            pl.mean("strong_15m").alias("win_15m"),
            pl.mean("strong_30m").alias("win_30m"),
            pl.mean("strong_60m").alias("win_60m"),

            pl.mean("master_fwd_15m").alias("avg_15m"),
            pl.mean("master_fwd_30m").alias("avg_30m"),
            pl.mean("master_fwd_60m").alias("avg_60m"),

            pl.median("master_fwd_30m").alias("med_30m"),

            pl.std("master_fwd_30m").alias("std_30m"),

        ]).row(0, named=True)

        expectancy = (
            stats["avg_30m"] *
            stats["win_30m"] *
            (trades / 100)
        )

        sequence_score = (
            stats["win_15m"] *
            stats["win_30m"] *
            stats["win_60m"] *
            trades
        )

        results.append({

            "conditions": " -> ".join(combo),

            "trades": trades,

            "expectancy_score": expectancy,

            "sequence_score": sequence_score,

            **stats

        })

print(f"\nCombos tested: {combo_count:,}")

# ============================================================
# RESULTS
# ============================================================

res = pl.DataFrame(results)

top_expectancy = (
    res
    .sort("expectancy_score", descending=True)
)

top_sequence = (
    res
    .sort("sequence_score", descending=True)
)

print("\n================ TOP EXPECTANCY ==================\n")
print(top_expectancy.head(50))

print("\n================ TOP SEQUENCE ==================\n")
print(top_sequence.head(50))

# ============================================================
# SAVE
# ============================================================

top_expectancy.write_csv(
    OUTPUT_DIR / "top_sequence_expectancy_models.csv"
)

top_sequence.write_csv(
    OUTPUT_DIR / "top_sequence_models.csv"
)

res.write_parquet(
    OUTPUT_DIR / "all_sequence_models.parquet",
    compression="zstd"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nFILES SAVED:")
print(OUTPUT_DIR / "top_sequence_expectancy_models.csv")
print(OUTPUT_DIR / "top_sequence_models.csv")
print(OUTPUT_DIR / "all_sequence_models.parquet")