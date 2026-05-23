from pathlib import Path
import polars as pl
import numpy as np
import itertools
import time
import math

START = time.time()

DATA = Path(
    r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet"
)

OUTPUT_DIR = Path(
    r"D:\TradingResearch\research_outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLOADING DATASET...\n")

df = pl.read_parquet(DATA)

# ============================================================
# BUILD CORE SETUPS
# ============================================================

print("BUILDING SETUPS...\n")

setups = (

    df

    .filter(
        pl.col("is_rth") == 1
    )

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
    # STRUCTURAL CONDITIONS
    # ========================================================

    .filter(
        pl.col("swept_prior_rth_low") == 1
    )

    .filter(
        pl.col("close") > pl.col("prior_rth_low")
    )

    .with_columns([

        (
            pl.col("close") > pl.col("or_low_30m")
        ).cast(pl.Int8).alias("back_inside_or"),

        (
            pl.col("or_range_30m") > 80
        ).cast(pl.Int8).alias("large_or"),

        (
            pl.col("avg_spread") <= 1.0
        ).cast(pl.Int8).alias("tight_spread"),

        (
            pl.col("close") < pl.col("vwap_day")
        ).cast(pl.Int8).alias("below_vwap"),

        (
            pl.col("is_displacement_candle") == 1
        ).cast(pl.Int8).alias("displacement"),

    ])

    .filter(
        pl.col("back_inside_or") == 1
    )

    .filter(
        pl.col("large_or") == 1
    )

    .filter(
        pl.col("tight_spread") == 1
    )

)

print(f"Setups found: {setups.height:,}")

# ============================================================
# EXECUTION MODELS
# ============================================================

stop_models = [
    ("tight", 12),
    ("medium", 20),
    ("wide", 30),
]

target_models = [
    ("1R", 1.0),
    ("1_5R", 1.5),
    ("2R", 2.0),
    ("3R", 3.0),
]

be_trigger_models = [
    ("none", 999),
    ("0_5R", 0.5),
    ("1R", 1.0),
]

entry_models = [
    ("market", 0),
    ("5pt_pullback", -5),
    ("10pt_pullback", -10),
]

# ============================================================
# RUN SIMS
# ============================================================

results = []

print("\nRUNNING EXECUTION TESTS...\n")

combo_count = 0

for stop_name, stop_pts in stop_models:

    for target_name, target_r in target_models:

        for be_name, be_trigger_r in be_trigger_models:

            for entry_name, entry_offset in entry_models:

                combo_count += 1

                wins = 0
                losses = 0

                gross_win = 0
                gross_loss = 0

                equity = 0
                peak = 0
                max_dd = 0

                trade_results = []

                target_pts = stop_pts * target_r

                for row in setups.iter_rows(named=True):

                    entry = row["close"] + entry_offset

                    stop = entry - stop_pts
                    target = entry + target_pts

                    future_high = row["future_high_60m"]
                    future_low = row["future_low_60m"]

                    max_favorable = future_high - entry
                    max_adverse = entry - future_low

                    be_trigger = stop_pts * be_trigger_r

                    outcome = ""
                    pnl = 0

                    # ====================================
                    # BREAKEVEN LOGIC
                    # ====================================

                    moved_to_be = False

                    if max_favorable >= be_trigger:
                        moved_to_be = True

                    # ====================================
                    # TARGET ONLY
                    # ====================================

                    if future_high >= target and future_low > stop:

                        pnl = target_pts
                        outcome = "WIN"

                    # ====================================
                    # STOP ONLY
                    # ====================================

                    elif future_low <= stop and future_high < target:

                        pnl = -stop_pts
                        outcome = "LOSS"

                    # ====================================
                    # BOTH HIT
                    # ====================================

                    elif future_high >= target and future_low <= stop:

                        # optimistic path assumption
                        pnl = target_pts
                        outcome = "WIN_AMBIG"

                    # ====================================
                    # BREAKEVEN
                    # ====================================

                    elif moved_to_be and future_low <= entry:

                        pnl = 0
                        outcome = "BREAKEVEN"

                    # ====================================
                    # TIME EXIT
                    # ====================================

                    else:

                        pnl = row["master_fwd_60m"]

                        if pnl > 0:
                            outcome = "TIME_WIN"
                        else:
                            outcome = "TIME_LOSS"

                    # ====================================
                    # STATS
                    # ====================================

                    if pnl > 0:
                        wins += 1
                        gross_win += pnl

                    elif pnl < 0:
                        losses += 1
                        gross_loss += abs(pnl)

                    equity += pnl

                    peak = max(peak, equity)

                    dd = peak - equity

                    max_dd = max(max_dd, dd)

                    trade_results.append(pnl)

                # ========================================
                # METRICS
                # ========================================

                total = len(trade_results)

                winrate = (
                    wins / total
                    if total > 0
                    else 0
                )

                pf = (
                    gross_win / gross_loss
                    if gross_loss > 0
                    else 0
                )

                avg_trade = (
                    sum(trade_results) / total
                    if total > 0
                    else 0
                )

                sharpe = 0

                if len(trade_results) > 5:

                    std = np.std(trade_results)

                    if std > 0:
                        sharpe = (
                            np.mean(trade_results) / std
                        ) * math.sqrt(252)

                expectancy_score = (
                    avg_trade *
                    pf *
                    (total / 100)
                )

                results.append({

                    "stop_model": stop_name,
                    "target_model": target_name,
                    "be_model": be_name,
                    "entry_model": entry_name,

                    "trades": total,

                    "winrate": round(winrate, 4),

                    "profit_factor": round(pf, 4),

                    "avg_trade": round(avg_trade, 2),

                    "net_points": round(equity, 2),

                    "max_drawdown": round(max_dd, 2),

                    "sharpe": round(sharpe, 2),

                    "expectancy_score": round(expectancy_score, 2),

                })

print(f"\nExecution models tested: {combo_count:,}")

# ============================================================
# RESULTS
# ============================================================

res = (
    pl.DataFrame(results)
    .sort("expectancy_score", descending=True)
)

print("\n================ TOP EXECUTION MODELS ==================\n")

print(res.head(50))

# ============================================================
# SAVE
# ============================================================

res.write_csv(
    OUTPUT_DIR / "execution_model_results.csv"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nFILE SAVED:")
print(OUTPUT_DIR / "execution_model_results.csv")