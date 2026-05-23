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
# BASE FILTERS
# ============================================================

base = (

    lf

    .filter(
        pl.col("is_rth") == 1
    )

    # OR WINDOW
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

    # MUST SWEEP PRIOR RTH LOW
    .filter(
        pl.col("swept_prior_rth_low") == 1
    )

    # MUST RECLAIM
    .filter(
        pl.col("close") > pl.col("prior_rth_low")
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    .with_columns([

        (
            pl.col("close") > pl.col("vwap_day")
        ).cast(pl.Int8).alias("above_vwap"),

        (
            pl.col("close") < pl.col("vwap_day")
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
            pl.col("avg_spread") > 1.25
        ).cast(pl.Int8).alias("wide_spread"),

        (
            pl.col("update_velocity") > 500
        ).cast(pl.Int8).alias("high_velocity"),

        (
            pl.col("rel_vol_20") > 1.25
        ).cast(pl.Int8).alias("high_rel_vol"),

        (
            pl.col("close") > pl.col("open")
        ).cast(pl.Int8).alias("green_bar"),

        (
            pl.col("close") < pl.col("open")
        ).cast(pl.Int8).alias("red_bar"),

        (
            pl.col("bull_fvg") == 1
        ).cast(pl.Int8).alias("bull_fvg_present"),

        (
            pl.col("is_displacement_candle") == 1
        ).cast(pl.Int8).alias("displacement"),

        (
            pl.col("or_range_30m") <= 40
        ).cast(pl.Int8).alias("small_or"),

        (
            (pl.col("or_range_30m") > 40) &
            (pl.col("or_range_30m") <= 80)
        ).cast(pl.Int8).alias("medium_or"),

        (
            pl.col("or_range_30m") > 80
        ).cast(pl.Int8).alias("large_or"),

        (
            pl.col("bar_range_master") > pl.col("avg_range_20")
        ).cast(pl.Int8).alias("expansion_bar"),

        (
            pl.col("body_pct") > 0.6
        ).cast(pl.Int8).alias("strong_body"),

        (
            pl.col("lower_wick_pct") > 0.4
        ).cast(pl.Int8).alias("large_lower_wick"),

    ])

    # ========================================================
    # TARGETS
    # ========================================================

    .with_columns([

        (
            pl.col("master_fwd_15m") >= 10
        ).cast(pl.Int8).alias("win_10"),

        (
            pl.col("master_fwd_15m") >= 20
        ).cast(pl.Int8).alias("win_20"),

        (
            pl.col("master_fwd_30m") >= 30
        ).cast(pl.Int8).alias("win_30"),

        (
            pl.col("master_fwd_60m") >= 40
        ).cast(pl.Int8).alias("win_40"),

    ])

)

print("\nCOLLECTING BASE DATA...\n")

df = base.collect(engine="streaming")

print(f"Base setups: {df.height:,}")

# ============================================================
# CONDITIONS TO TEST
# ============================================================

conditions = [

    "above_vwap",
    "below_vwap",
    "bull_quote_pressure",
    "bull_mid_pressure",
    "tight_spread",
    "wide_spread",
    "high_velocity",
    "high_rel_vol",
    "green_bar",
    "red_bar",
    "bull_fvg_present",
    "displacement",
    "small_or",
    "medium_or",
    "large_or",
    "expansion_bar",
    "strong_body",
    "large_lower_wick",

]

results = []

# ============================================================
# TEST COMBINATIONS
# ============================================================

print("\nRUNNING MASSIVE CONDITION TESTS...\n")

combo_count = 0

for r in range(1, 5):

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

        if trades < 40:
            continue

        stats = sub.select([

            pl.mean("win_10").alias("win_10"),
            pl.mean("win_20").alias("win_20"),
            pl.mean("win_30").alias("win_30"),
            pl.mean("win_40").alias("win_40"),

            pl.mean("master_fwd_15m").alias("avg_15m"),
            pl.mean("master_fwd_30m").alias("avg_30m"),
            pl.mean("master_fwd_60m").alias("avg_60m"),

            pl.median("master_fwd_30m").alias("med_30m"),

            pl.std("master_fwd_30m").alias("std_30m"),

        ]).row(0, named=True)

        expectancy_score = (
            stats["avg_30m"] *
            stats["win_20"] *
            (trades / 100)
        )

        results.append({

            "conditions": " & ".join(combo),

            "trades": trades,

            "expectancy_score": expectancy_score,

            **stats

        })

print(f"\nCombos tested: {combo_count:,}")

# ============================================================
# RESULTS
# ============================================================

res = (
    pl.DataFrame(results)

    .sort(
        "expectancy_score",
        descending=True
    )
)

print("\n================ TOP 50 MODELS ==================\n")

print(
    res.head(50)
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

TOP_OUT = OUTPUT_DIR / "top_50_orb_failed_auction_models.csv"
ALL_OUT = OUTPUT_DIR / "all_orb_failed_auction_models.parquet"

res.head(50).write_csv(TOP_OUT)

res.write_parquet(
    ALL_OUT,
    compression="zstd"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nSaved:")
print(TOP_OUT)
print(ALL_OUT)