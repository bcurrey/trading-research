from pathlib import Path
import polars as pl
import itertools
import time
import math
import numpy as np

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
    # FAILED AUCTION CORE
    # ========================================================

    .filter(
        pl.col("swept_prior_rth_low") == 1
    )

    .filter(
        pl.col("close") > pl.col("prior_rth_low")
    )

    .with_columns([

        # ====================================================
        # CORE STRUCTURE
        # ====================================================

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

        # ====================================================
        # 15M REGIME PROXIES
        # ====================================================

        (
            pl.col("ema_9") > pl.col("ema_20")
        ).cast(pl.Int8).alias("bull_trend"),

        (
            pl.col("ema_20") > pl.col("ema_50")
        ).cast(pl.Int8).alias("strong_trend"),

        (
            pl.col("dist_ema_20").abs() < 15
        ).cast(pl.Int8).alias("compressed"),

        (
            pl.col("dist_ema_20").abs() > 40
        ).cast(pl.Int8).alias("expanded"),

        (
            pl.col("atr_14") > pl.col("atr_50")
        ).cast(pl.Int8).alias("high_volatility"),

        (
            pl.col("atr_14") < pl.col("atr_50")
        ).cast(pl.Int8).alias("low_volatility"),

        # ====================================================
        # DAILY REGIME
        # ====================================================

        (
            pl.col("premarket_range") > 80
        ).cast(pl.Int8).alias("large_premarket"),

        (
            pl.col("prior_day_range") > 250
        ).cast(pl.Int8).alias("large_prior_day"),

        (
            pl.col("prior_day_range") < 120
        ).cast(pl.Int8).alias("small_prior_day"),

        # ====================================================
        # CANDLE STRUCTURE
        # ====================================================

        (
            pl.col("is_displacement_candle") == 1
        ).cast(pl.Int8).alias("displacement"),

        (
            pl.col("bull_fvg") == 1
        ).cast(pl.Int8).alias("bull_fvg"),

        (
            pl.col("body_pct") > 0.7
        ).cast(pl.Int8).alias("strong_body"),

        (
            pl.col("lower_wick_pct") > 0.35
        ).cast(pl.Int8).alias("large_lower_wick"),

        # ====================================================
        # ORDERFLOW
        # ====================================================

        (
            pl.col("quote_pressure_delta") > 0
        ).cast(pl.Int8).alias("bull_quote_pressure"),

        (
            pl.col("net_mid_pressure") > 0
        ).cast(pl.Int8).alias("bull_mid_pressure"),

        (
            pl.col("update_velocity") > 500
        ).cast(pl.Int8).alias("high_velocity"),

    ])

    # ========================================================
    # FILTER TO CORE MODEL
    # ========================================================

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
# REGIME CONDITIONS
# ============================================================

conditions = [

    # HTF
    "bull_trend",
    "strong_trend",
    "compressed",
    "expanded",
    "high_volatility",
    "low_volatility",

    # daily
    "large_premarket",
    "large_prior_day",
    "small_prior_day",

    # structure
    "below_vwap",
    "displacement",
    "bull_fvg",
    "strong_body",
    "large_lower_wick",

    # orderflow
    "bull_quote_pressure",
    "bull_mid_pressure",
    "high_velocity",

]

# ============================================================
# EXECUTION SETTINGS
# ============================================================

ENTRY_PULLBACK = 10
STOP_POINTS = 20
TARGET_POINTS = 20
BE_TRIGGER = 15

MAX_HOLD_BARS = 60

# ============================================================
# MAIN TESTING
# ============================================================

results = []

combo_count = 0

print("\nRUNNING HTF REGIME TESTS...\n")

for r in range(1, 6):

    for combo in itertools.combinations(conditions, r):

        combo_count += 1

        filt = None

        for cond in combo:

            expr = pl.col(cond) == 1

            if filt is None:
                filt = expr
            else:
                filt = filt & expr

        sub = setups.filter(filt)

        trades = sub.height

        if trades < 25:
            continue

        wins = 0
        losses = 0
        breakevens = 0

        gross_win = 0
        gross_loss = 0

        equity = 0
        peak = 0
        max_dd = 0

        trade_pnls = []

        # ====================================================
        # BAR-BY-BAR WALKFORWARD
        # ====================================================

        for row in sub.iter_rows(named=True):

            ts = row["ts_event"]

            entry = row["close"] - ENTRY_PULLBACK

            stop = entry - STOP_POINTS
            target = entry + TARGET_POINTS

            symbol = row["symbol"]

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

            for bar in future.iter_rows(named=True):

                high = bar["high"]
                low = bar["low"]

                # ============================================
                # ENTRY
                # ============================================

                if not in_trade:

                    if low <= entry:

                        in_trade = True

                    else:
                        continue

                mfe = high - entry

                if mfe >= BE_TRIGGER:
                    moved_to_be = True

                # ============================================
                # TARGET
                # ============================================

                if high >= target:

                    pnl = TARGET_POINTS

                    wins += 1
                    gross_win += pnl

                    outcome = "WIN"

                    break

                # ============================================
                # BREAKEVEN
                # ============================================

                if moved_to_be and low <= entry:

                    pnl = 0

                    breakevens += 1

                    outcome = "BE"

                    break

                # ============================================
                # STOP
                # ============================================

                if low <= stop:

                    pnl = -STOP_POINTS

                    losses += 1
                    gross_loss += abs(pnl)

                    outcome = "LOSS"

                    break

            # ================================================
            # TIME EXIT
            # ================================================

            if outcome == "":

                final_close = future[-1, "close"]

                pnl = final_close - entry

                if pnl > 0:

                    wins += 1
                    gross_win += pnl

                elif pnl < 0:

                    losses += 1
                    gross_loss += abs(pnl)

                else:

                    breakevens += 1

            equity += pnl

            peak = max(peak, equity)

            dd = peak - equity

            max_dd = max(max_dd, dd)

            trade_pnls.append(pnl)

        # ====================================================
        # METRICS
        # ====================================================

        total = len(trade_pnls)

        if total == 0:
            continue

        winrate = wins / total

        pf = (
            gross_win / gross_loss
            if gross_loss > 0
            else 0
        )

        avg_trade = np.mean(trade_pnls)

        sharpe = 0

        std = np.std(trade_pnls)

        if std > 0:

            sharpe = (
                avg_trade / std
            ) * math.sqrt(252)

        expectancy = (
            avg_trade *
            pf *
            (total / 100)
        )

        results.append({

            "conditions": " & ".join(combo),

            "trades": total,

            "winrate": round(winrate, 4),

            "profit_factor": round(pf, 4),

            "avg_trade": round(avg_trade, 2),

            "net_points": round(equity, 2),

            "max_drawdown": round(max_dd, 2),

            "sharpe": round(sharpe, 2),

            "expectancy_score": round(expectancy, 2),

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

print("\n================ TOP HTF REGIMES ==================\n")

print(res.head(50))

# ============================================================
# SAVE
# ============================================================

res.write_csv(
    OUTPUT_DIR / "htf_regime_results.csv"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nFILE SAVED:")
print(OUTPUT_DIR / "htf_regime_results.csv")