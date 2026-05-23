from pathlib import Path
import polars as pl
import itertools
import numpy as np
import math
import time

START = time.time()

DATA = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUTPUT_DIR = Path(r"D:\TradingResearch\research_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLOADING DATA...\n")
df = pl.read_parquet(DATA).sort("ts_event")

print("BUILDING CORE SETUPS...\n")

base = (
    df
    .filter(pl.col("is_rth") == 1)
    .filter(
        ((pl.col("hour_ct") == 8) & (pl.col("minute_ct") >= 30)) |
        ((pl.col("hour_ct") == 9) & (pl.col("minute_ct") <= 15))
    )
    .filter(pl.col("swept_prior_rth_low") == 1)
    .filter(pl.col("close") > pl.col("prior_rth_low"))
    .with_columns([
        (pl.col("close") > pl.col("or_low_30m")).cast(pl.Int8).alias("back_inside_or"),
        (pl.col("or_range_30m") > 80).cast(pl.Int8).alias("large_or"),
        (pl.col("avg_spread") <= 1.0).cast(pl.Int8).alias("tight_spread"),
        (pl.col("close") < pl.col("vwap_day")).cast(pl.Int8).alias("below_vwap"),
        (pl.col("close") > pl.col("vwap_day")).cast(pl.Int8).alias("above_vwap"),
        (pl.col("atr_14") < pl.col("atr_50")).cast(pl.Int8).alias("low_volatility"),
        (pl.col("premarket_range") > 80).cast(pl.Int8).alias("large_premarket"),
        (pl.col("is_displacement_candle") == 1).cast(pl.Int8).alias("displacement"),
        (pl.col("bull_fvg") == 1).cast(pl.Int8).alias("bull_fvg"),
        (pl.col("body_pct") > 0.7).cast(pl.Int8).alias("strong_body"),
        (pl.col("lower_wick_pct") > 0.35).cast(pl.Int8).alias("large_lower_wick"),
        (pl.col("quote_pressure_delta") > 0).cast(pl.Int8).alias("bull_quote_pressure"),
        (pl.col("net_mid_pressure") > 0).cast(pl.Int8).alias("bull_mid_pressure"),
        (pl.col("close") > pl.col("open")).cast(pl.Int8).alias("green_bar"),
        (pl.col("close") < pl.col("open")).cast(pl.Int8).alias("red_bar"),
    ])
    .filter(pl.col("back_inside_or") == 1)
    .filter(pl.col("large_or") == 1)
    .filter(pl.col("tight_spread") == 1)
)

print(f"Core setups: {base.height:,}")

conditions = [
    "below_vwap",
    "above_vwap",
    "low_volatility",
    "large_premarket",
    "displacement",
    "bull_fvg",
    "strong_body",
    "large_lower_wick",
    "bull_quote_pressure",
    "bull_mid_pressure",
    "green_bar",
    "red_bar",
]

condition_sets = [()]
for r in range(1, 4):
    condition_sets.extend(list(itertools.combinations(conditions, r)))

entry_models = [
    ("5pt_pullback", 5),
    ("10pt_pullback", 10),
    ("15pt_pullback", 15),
]

stop_models = [
    ("15pt_stop", 15),
    ("20pt_stop", 20),
    ("25pt_stop", 25),
]

target_models = [
    ("20pt_target", 20),
    ("25pt_target", 25),
    ("30pt_target", 30),
]

be_triggers = [
    ("no_be", None),
    ("be_after_15", 15),
    ("be_after_20", 20),
]

MAX_HOLD_BARS = 60
MIN_TRADES = 25

def grade_model(winrate, pf, trades_per_year, avg_trade, profitable_years_pct):
    if winrate >= 0.60 and pf >= 1.50 and trades_per_year >= 20 and avg_trade >= 5 and profitable_years_pct >= 0.60:
        return "A+"
    if winrate >= 0.55 and pf >= 1.30 and trades_per_year >= 20 and avg_trade >= 3 and profitable_years_pct >= 0.55:
        return "A"
    if winrate >= 0.50 and pf >= 1.10 and avg_trade > 0:
        return "B"
    return "REJECT"

results = []

print("\nRUNNING FOCUSED WIN-RATE ENGINE...\n")

total_model_count = 0

for combo in condition_sets:
    sub = base

    for cond in combo:
        sub = sub.filter(pl.col(cond) == 1)

    if sub.height < MIN_TRADES:
        continue

    for entry_name, pullback in entry_models:
        for stop_name, stop_pts in stop_models:
            for target_name, target_pts in target_models:
                for be_name, be_trigger in be_triggers:

                    total_model_count += 1

                    pnls = []
                    trade_rows = []

                    wins = 0
                    losses = 0
                    breakevens = 0
                    gross_win = 0
                    gross_loss = 0
                    equity = 0
                    peak = 0
                    max_dd = 0

                    for row in sub.iter_rows(named=True):
                        ts = row["ts_event"]
                        symbol = row["symbol"]

                        entry = row["close"] - pullback
                        stop = entry - stop_pts
                        target = entry + target_pts

                        future = (
                            df
                            .filter(pl.col("symbol") == symbol)
                            .filter(pl.col("ts_event") > ts)
                            .head(MAX_HOLD_BARS)
                        )

                        if future.height == 0:
                            continue

                        in_trade = False
                        moved_to_be = False
                        pnl = 0
                        outcome = ""

                        for bar in future.iter_rows(named=True):
                            high = bar["high"]
                            low = bar["low"]

                            if not in_trade:
                                if low <= entry:
                                    in_trade = True
                                else:
                                    continue

                            if be_trigger is not None and high - entry >= be_trigger:
                                moved_to_be = True

                            if high >= target:
                                pnl = target_pts
                                outcome = "WIN"
                                break

                            if moved_to_be and low <= entry:
                                pnl = 0
                                outcome = "BREAKEVEN"
                                break

                            if low <= stop:
                                pnl = -stop_pts
                                outcome = "LOSS"
                                break

                        if outcome == "":
                            final = future.tail(1).row(0, named=True)
                            pnl = float(final["close"] - entry)
                            outcome = "TIME_WIN" if pnl > 0 else "TIME_LOSS" if pnl < 0 else "FLAT"

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
                        max_dd = max(max_dd, peak - equity)

                        pnls.append(pnl)
                        trade_rows.append({"year": row["ts_event"].year, "pnl": pnl})

                    total = len(pnls)
                    if total < MIN_TRADES:
                        continue

                    winrate = wins / total
                    pf = gross_win / gross_loss if gross_loss > 0 else 0
                    avg_trade = float(np.mean(pnls))
                    median_trade = float(np.median(pnls))
                    std = float(np.std(pnls))
                    sharpe = (avg_trade / std) * math.sqrt(252) if std > 0 else 0

                    trade_years = pl.DataFrame(trade_rows)
                    by_year = (
                        trade_years
                        .group_by("year")
                        .agg([
                            pl.len().alias("trades"),
                            pl.sum("pnl").alias("net_points"),
                        ])
                    )

                    years = by_year.height
                    profitable_years = by_year.filter(pl.col("net_points") > 0).height
                    profitable_years_pct = profitable_years / years if years > 0 else 0

                    first_year = int(by_year["year"].min())
                    last_year = int(by_year["year"].max())
                    span_years = max(1, last_year - first_year + 1)
                    trades_per_year = total / span_years
                    trades_per_week = trades_per_year / 52

                    tier = grade_model(winrate, pf, trades_per_year, avg_trade, profitable_years_pct)

                    size = {
                        "A+": "full",
                        "A": "half",
                        "B": "micro_or_paper",
                        "REJECT": "ignore",
                    }[tier]

                    results.append({
                        "tier": tier,
                        "suggested_size": size,
                        "conditions": " & ".join(combo) if combo else "CORE_ONLY",
                        "entry_model": entry_name,
                        "stop_model": stop_name,
                        "target_model": target_name,
                        "be_model": be_name,
                        "trades": total,
                        "trades_per_year": round(trades_per_year, 2),
                        "trades_per_week": round(trades_per_week, 2),
                        "wins": wins,
                        "losses": losses,
                        "breakevens": breakevens,
                        "winrate": round(winrate, 4),
                        "profit_factor": round(pf, 4),
                        "avg_trade": round(avg_trade, 2),
                        "median_trade": round(median_trade, 2),
                        "net_points": round(equity, 2),
                        "max_drawdown": round(max_dd, 2),
                        "sharpe": round(sharpe, 2),
                        "profitable_years_pct": round(profitable_years_pct, 4),
                        "profitable_years": profitable_years,
                        "years_tested": years,
                        "first_year": first_year,
                        "last_year": last_year,
                    })

print(f"\nModels tested: {total_model_count:,}")
print(f"Valid results: {len(results):,}")

res = pl.DataFrame(results)

res = res.with_columns([
    pl.when(pl.col("tier") == "A+").then(4)
    .when(pl.col("tier") == "A").then(3)
    .when(pl.col("tier") == "B").then(2)
    .otherwise(1)
    .alias("tier_rank")
])

ranked = (
    res
    .sort(
        ["tier_rank", "winrate", "profit_factor", "avg_trade", "trades_per_year"],
        descending=[True, True, True, True, True]
    )
)

print("\n================ TOP FOCUSED WIN-RATE MODELS ==================\n")
print(ranked.head(75))

print("\n================ TIER COUNTS ==================\n")
print(
    ranked
    .group_by("tier")
    .agg(pl.len().alias("models"))
    .sort("tier")
)

ranked.write_csv(OUTPUT_DIR / "focused_winrate_ranked_models.csv")
ranked.filter(pl.col("tier").is_in(["A+", "A"])).write_csv(
    OUTPUT_DIR / "focused_winrate_A_models.csv"
)
ranked.write_parquet(
    OUTPUT_DIR / "focused_winrate_all_models.parquet",
    compression="zstd"
)

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")

print("\nFILES SAVED:")
print(OUTPUT_DIR / "focused_winrate_ranked_models.csv")
print(OUTPUT_DIR / "focused_winrate_A_models.csv")
print(OUTPUT_DIR / "focused_winrate_all_models.parquet")