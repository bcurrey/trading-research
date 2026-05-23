from pathlib import Path
import polars as pl
import numpy as np
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

df = pl.read_parquet(DATA)

# ============================================================
# CORE MODEL FILTER
# ============================================================

print("BUILDING TRADE SETUPS...\n")

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
    # CORE STRUCTURAL MODEL
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
# TRADE PARAMETERS
# ============================================================

STOP_POINTS = 20
TARGET_POINTS = 40

print("\nSIMULATING TRADES...\n")

# ============================================================
# BUILD FORWARD ARRAYS
# ============================================================

future_highs = []
future_lows = []

highs = setups["future_high_60m"].to_list()
lows = setups["future_low_60m"].to_list()

for h, l in zip(highs, lows):

    future_highs.append(h)
    future_lows.append(l)

# ============================================================
# SIMULATE
# ============================================================

results = []

wins = 0
losses = 0

gross_win = 0
gross_loss = 0

equity = 0
peak = 0
max_dd = 0

equity_curve = []

for row in setups.iter_rows(named=True):

    entry = row["close"]

    stop = entry - STOP_POINTS
    target = entry + TARGET_POINTS

    future_high = row["future_high_60m"]
    future_low = row["future_low_60m"]

    hit_target = future_high >= target
    hit_stop = future_low <= stop

    pnl = 0
    outcome = ""

    # ========================================================
    # LOGIC
    # ========================================================

    if hit_target and not hit_stop:

        pnl = TARGET_POINTS
        wins += 1
        gross_win += pnl
        outcome = "WIN"

    elif hit_stop and not hit_target:

        pnl = -STOP_POINTS
        losses += 1
        gross_loss += abs(pnl)
        outcome = "LOSS"

    elif hit_target and hit_stop:

        # assume stop first (conservative)
        pnl = -STOP_POINTS
        losses += 1
        gross_loss += abs(pnl)
        outcome = "AMBIG_STOP"

    else:

        pnl = row["master_fwd_60m"]

        if pnl > 0:
            wins += 1
            gross_win += pnl
            outcome = "TIME_WIN"
        else:
            losses += 1
            gross_loss += abs(pnl)
            outcome = "TIME_LOSS"

    equity += pnl

    peak = max(peak, equity)

    dd = peak - equity

    max_dd = max(max_dd, dd)

    equity_curve.append(equity)

    results.append({

        "ts_event": row["ts_event"],
        "symbol": row["symbol"],

        "entry": entry,
        "stop": stop,
        "target": target,

        "future_high_60m": future_high,
        "future_low_60m": future_low,

        "outcome": outcome,
        "pnl": pnl,

        "master_fwd_15m": row["master_fwd_15m"],
        "master_fwd_30m": row["master_fwd_30m"],
        "master_fwd_60m": row["master_fwd_60m"],

    })

# ============================================================
# RESULTS DF
# ============================================================

trades = pl.DataFrame(results)

total = len(results)

winrate = wins / total if total > 0 else 0

profit_factor = (
    gross_win / gross_loss
    if gross_loss > 0
    else 0
)

expectancy = trades["pnl"].mean()

median_trade = trades["pnl"].median()

std_trade = trades["pnl"].std()

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================ RESULTS ==================\n")

summary = pl.DataFrame([{

    "trades": total,
    "wins": wins,
    "losses": losses,

    "winrate": round(winrate, 4),

    "profit_factor": round(profit_factor, 4),

    "avg_trade": round(expectancy, 2),
    "median_trade": round(median_trade, 2),

    "std_trade": round(std_trade, 2),

    "gross_win": round(gross_win, 2),
    "gross_loss": round(gross_loss, 2),

    "net_points": round(equity, 2),

    "max_drawdown": round(max_dd, 2),

}])

print(summary)

# ============================================================
# BEST/WORST DAYS
# ============================================================

print("\n================ OUTCOMES ==================\n")

print(
    trades.group_by("outcome")
    .agg([
        pl.len().alias("trades"),
        pl.mean("pnl").alias("avg_pnl"),
    ])
)

# ============================================================
# SAVE OUTPUTS
# ============================================================

summary.write_csv(
    OUTPUT_DIR / "trade_sim_summary.csv"
)

trades.write_csv(
    OUTPUT_DIR / "trade_sim_trades.csv"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nFILES SAVED:")
print(OUTPUT_DIR / "trade_sim_summary.csv")
print(OUTPUT_DIR / "trade_sim_trades.csv")