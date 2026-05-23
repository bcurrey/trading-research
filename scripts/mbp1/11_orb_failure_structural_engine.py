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
# BUILD TRUE OR FAILURE DATASET
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
    # OPENING WINDOW ONLY
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
    # MUST RECLAIM
    # ========================================================

    .filter(
        pl.col("close") > pl.col("prior_rth_low")
    )

    # ========================================================
    # ENGINEER STRUCTURAL FEATURES
    # ========================================================

    .with_columns([

        # ============================================
        # VWAP STATE
        # ============================================

        (
            pl.col("close") > pl.col("vwap_day")
        ).cast(pl.Int8).alias("above_vwap"),

        (
            pl.col("close") < pl.col("vwap_day")
        ).cast(pl.Int8).alias("below_vwap"),

        (
            (
                pl.col("close") > pl.col("vwap_day")
            ) &
            (
                pl.col("open") < pl.col("vwap_day")
            )
        ).cast(pl.Int8).alias("vwap_reclaim"),

        # ============================================
        # OR STRUCTURE
        # ============================================

        (
            pl.col("close") > pl.col("or_low_30m")
        ).cast(pl.Int8).alias("back_inside_or"),

        (
            pl.col("close") > pl.col("or_high_30m")
        ).cast(pl.Int8).alias("above_or_high"),

        (
            pl.col("or_range_30m") > 80
        ).cast(pl.Int8).alias("large_or"),

        (
            pl.col("or_range_30m") <= 40
        ).cast(pl.Int8).alias("small_or"),

        # ============================================
        # CANDLE STRUCTURE
        # ============================================

        (
            pl.col("close") > pl.col("open")
        ).cast(pl.Int8).alias("green_bar"),

        (
            pl.col("close") < pl.col("open")
        ).cast(pl.Int8).alias("red_bar"),

        (
            pl.col("body_pct") > 0.7
        ).cast(pl.Int8).alias("strong_body"),

        (
            pl.col("lower_wick_pct") > 0.35
        ).cast(pl.Int8).alias("large_lower_wick"),

        (
            pl.col("bar_range_master") >
            pl.col("avg_range_20")
        ).cast(pl.Int8).alias("expansion_bar"),

        # ============================================
        # FVG / DISPLACEMENT
        # ============================================

        (
            pl.col("bull_fvg") == 1
        ).cast(pl.Int8).alias("bull_fvg"),

        (
            pl.col("is_displacement_candle") == 1
        ).cast(pl.Int8).alias("displacement"),

        # ============================================
        # MBP / ORDERFLOW
        # ============================================

        (
            pl.col("avg_spread") <= 1.0
        ).cast(pl.Int8).alias("tight_spread"),

        (
            pl.col("avg_spread") > 1.25
        ).cast(pl.Int8).alias("wide_spread"),

        (
            pl.col("quote_pressure_delta") > 0
        ).cast(pl.Int8).alias("bull_quote_pressure"),

        (
            pl.col("net_mid_pressure") > 0
        ).cast(pl.Int8).alias("bull_mid_pressure"),

        (
            pl.col("update_velocity") > 500
        ).cast(pl.Int8).alias("high_velocity"),

        # ============================================
        # VOLUME / VOLATILITY
        # ============================================

        (
            pl.col("rel_vol_20") > 1.25
        ).cast(pl.Int8).alias("high_rel_vol"),

        (
            pl.col("atr_14") > pl.col("atr_50")
        ).cast(pl.Int8).alias("high_volatility"),

        # ============================================
        # TARGETS
        # ============================================

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

print("\nCOLLECTING STRUCTURAL DATA...\n")

df = base.collect(engine="streaming")

print(f"Base setups: {df.height:,}")

# ============================================================
# CONDITIONS
# ============================================================

conditions = [

    "above_vwap",
    "below_vwap",
    "vwap_reclaim",

    "back_inside_or",
    "above_or_high",

    "large_or",
    "small_or",

    "green_bar",
    "red_bar",
    "strong_body",
    "large_lower_wick",
    "expansion_bar",

    "bull_fvg",
    "displacement",

    "tight_spread",
    "wide_spread",

    "bull_quote_pressure",
    "bull_mid_pressure",
    "high_velocity",

    "high_rel_vol",
    "high_volatility",

]

results = []

print("\nRUNNING STRUCTURAL COMBO TESTS...\n")

combo_count = 0

for r in range(2, 6):

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

        if trades < 30:
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

        expectancy = (
            stats["avg_30m"] *
            stats["win_20"] *
            (trades / 100)
        )

        robustness = (
            stats["win_20"] *
            stats["win_30"] *
            trades
        )

        results.append({

            "conditions": " & ".join(combo),

            "trades": trades,

            "expectancy_score": expectancy,

            "robustness_score": robustness,

            **stats

        })

print(f"\nCombos tested: {combo_count:,}")

# ============================================================
# RESULTS
# ============================================================

res = pl.DataFrame(results)

# ============================================================
# TOP EXPECTANCY
# ============================================================

top_expectancy = (
    res
    .sort("expectancy_score", descending=True)
)

print("\n================ TOP EXPECTANCY ==================\n")
print(top_expectancy.head(50))

# ============================================================
# TOP ROBUSTNESS
# ============================================================

top_robustness = (
    res
    .sort("robustness_score", descending=True)
)

print("\n================ TOP ROBUSTNESS ==================\n")
print(top_robustness.head(50))

# ============================================================
# SAVE OUTPUTS
# ============================================================

top_expectancy.write_csv(
    OUTPUT_DIR / "top_expectancy_orb_models.csv"
)

top_robustness.write_csv(
    OUTPUT_DIR / "top_robust_orb_models.csv"
)

res.write_parquet(
    OUTPUT_DIR / "all_structural_orb_models.parquet",
    compression="zstd"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nFILES SAVED:")
print(OUTPUT_DIR / "top_expectancy_orb_models.csv")
print(OUTPUT_DIR / "top_robust_orb_models.csv")
print(OUTPUT_DIR / "all_structural_orb_models.parquet")