from pathlib import Path
import polars as pl
import numpy as np
import math
import time

START = time.time()

DATA = Path(
    r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet"
)

OUTPUT_DIR = Path(
    r"D:\TradingResearch\research_outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLOADING DATA...\n")

df = pl.read_parquet(DATA)

# ============================================================
# BUILD CORE SETUPS
# ============================================================

print("BUILDING CORE SETUPS...\n")

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

print(f"Core setups: {setups.height:,}")

# ============================================================
# SORT DATASET FOR BAR LOOKUPS
# ============================================================

df = df.sort("ts_event")

# ============================================================
# TRADE PARAMETERS
# ============================================================

ENTRY_PULLBACK = 10
STOP_POINTS = 20
TARGET_POINTS = 20

BE_TRIGGER = 15

MAX_HOLD_BARS = 60

print("\nRUNNING BAR-BY-BAR WALKFORWARD...\n")

results = []

wins = 0
losses = 0
breakevens = 0

gross_win = 0
gross_loss = 0

equity = 0
peak = 0
max_dd = 0

# ============================================================
# MAIN LOOP
# ============================================================

for row in setups.iter_rows(named=True):

    ts = row["ts_event"]

    entry = row["close"] - ENTRY_PULLBACK

    stop = entry - STOP_POINTS
    target = entry + TARGET_POINTS

    symbol = row["symbol"]

    # ========================================================
    # GET FUTURE BARS
    # ========================================================

    future = (

        df

        .filter(
            pl.col("symbol") == symbol
        )

        .filter(
            pl.col("ts_event") > ts
        )

        .head(MAX_HOLD_BARS)

    )

    in_trade = False
    moved_to_be = False

    pnl = 0
    outcome = ""
    bars_held = 0

    mae = 0
    mfe = 0

    # ========================================================
    # WALKFORWARD
    # ========================================================

    for bar in future.iter_rows(named=True):

        bars_held += 1

        high = bar["high"]
        low = bar["low"]

        # ====================================================
        # ENTRY CHECK
        # ====================================================

        if not in_trade:

            if low <= entry:

                in_trade = True

            else:
                continue

        # ====================================================
        # MAE / MFE
        # ====================================================

        current_mae = entry - low
        current_mfe = high - entry

        mae = max(mae, current_mae)
        mfe = max(mfe, current_mfe)

        # ====================================================
        # MOVE TO BREAKEVEN
        # ====================================================

        if current_mfe >= BE_TRIGGER:

            moved_to_be = True

        # ====================================================
        # TARGET
        # ====================================================

        if high >= target:

            pnl = TARGET_POINTS

            wins += 1
            gross_win += pnl

            outcome = "WIN"

            break

        # ====================================================
        # BREAKEVEN STOP
        # ====================================================

        if moved_to_be and low <= entry:

            pnl = 0

            breakevens += 1

            outcome = "BREAKEVEN"

            break

        # ====================================================
        # HARD STOP
        # ====================================================

        if low <= stop:

            pnl = -STOP_POINTS

            losses += 1
            gross_loss += abs(pnl)

            outcome = "LOSS"

            break

    # ========================================================
    # TIME EXIT
    # ========================================================

    if outcome == "":

        final_close = future[-1, "close"]

        pnl = final_close - entry

        if pnl > 0:

            wins += 1
            gross_win += pnl

            outcome = "TIME_WIN"

        elif pnl < 0:

            losses += 1
            gross_loss += abs(pnl)

            outcome = "TIME_LOSS"

        else:

            breakevens += 1

            outcome = "FLAT"

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    equity += pnl

    peak = max(peak, equity)

    dd = peak - equity

    max_dd = max(max_dd, dd)

    results.append({

        "ts_event": ts,
        "symbol": symbol,

        "entry": entry,
        "stop": stop,
        "target": target,

        "outcome": outcome,

        "pnl": round(pnl, 2),

        "bars_held": bars_held,

        "mae": round(mae, 2),
        "mfe": round(mfe, 2),

    })

# ============================================================
# RESULTS DATAFRAME
# ============================================================

trades = pl.DataFrame(results)

total = trades.height

winrate = wins / total if total > 0 else 0

profit_factor = (
    gross_win / gross_loss
    if gross_loss > 0
    else 0
)

avg_trade = trades["pnl"].mean()

median_trade = trades["pnl"].median()

std_trade = trades["pnl"].std()

sharpe = 0

if std_trade > 0:

    sharpe = (
        avg_trade / std_trade
    ) * math.sqrt(252)

# ============================================================
# SUMMARY
# ============================================================

summary = pl.DataFrame([{

    "trades": total,

    "wins": wins,
    "losses": losses,
    "breakevens": breakevens,

    "winrate": round(winrate, 4),

    "profit_factor": round(profit_factor, 4),

    "avg_trade": round(avg_trade, 2),

    "median_trade": round(median_trade, 2),

    "gross_win": round(gross_win, 2),
    "gross_loss": round(gross_loss, 2),

    "net_points": round(equity, 2),

    "max_drawdown": round(max_dd, 2),

    "sharpe": round(sharpe, 2),

}])

print("\n================ SUMMARY ==================\n")

print(summary)

print("\n================ OUTCOMES ==================\n")

print(

    trades

    .group_by("outcome")

    .agg([

        pl.len().alias("trades"),

        pl.mean("pnl").alias("avg_pnl"),

        pl.mean("bars_held").alias("avg_bars"),

        pl.mean("mae").alias("avg_mae"),

        pl.mean("mfe").alias("avg_mfe"),

    ])

)

# ============================================================
# SAVE
# ============================================================

summary.write_csv(
    OUTPUT_DIR / "bar_by_bar_summary.csv"
)

trades.write_csv(
    OUTPUT_DIR / "bar_by_bar_trades.csv"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nFILES SAVED:")
print(OUTPUT_DIR / "bar_by_bar_summary.csv")
print(OUTPUT_DIR / "bar_by_bar_trades.csv")