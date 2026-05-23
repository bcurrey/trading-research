from pathlib import Path
import polars as pl
import time

START = time.time()

DATA = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUTPUT_DIR = Path(r"D:\TradingResearch\research_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(DATA).sort("ts_event")

print("\nBUILDING TOP HTF TRADE SET...\n")

setups = (
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
        (pl.col("atr_14") < pl.col("atr_50")).cast(pl.Int8).alias("low_volatility"),
        (pl.col("premarket_range") > 80).cast(pl.Int8).alias("large_premarket"),
    ])
    .filter(pl.col("back_inside_or") == 1)
    .filter(pl.col("large_or") == 1)
    .filter(pl.col("tight_spread") == 1)
    .filter(pl.col("low_volatility") == 1)
    .filter(pl.col("large_premarket") == 1)
)

print(f"Top model setups: {setups.height:,}")

ENTRY_PULLBACK = 10
STOP_POINTS = 20
TARGET_POINTS = 20
BE_TRIGGER = 15
MAX_HOLD_BARS = 60

trades = []

equity = 0
peak = 0
max_dd = 0

for row in setups.iter_rows(named=True):
    ts = row["ts_event"]
    symbol = row["symbol"]

    entry = row["close"] - ENTRY_PULLBACK
    stop = entry - STOP_POINTS
    target = entry + TARGET_POINTS

    future = (
        df
        .filter(pl.col("symbol") == symbol)
        .filter(pl.col("ts_event") > ts)
        .head(MAX_HOLD_BARS)
    )

    in_trade = False
    moved_to_be = False
    pnl = 0
    outcome = ""
    bars_held = 0
    mae = 0
    mfe = 0
    exit_ts = None
    exit_price = None

    for bar in future.iter_rows(named=True):
        bars_held += 1
        high = bar["high"]
        low = bar["low"]

        if not in_trade:
            if low <= entry:
                in_trade = True
            else:
                continue

        mae = max(mae, entry - low)
        mfe = max(mfe, high - entry)

        if mfe >= BE_TRIGGER:
            moved_to_be = True

        if high >= target:
            pnl = TARGET_POINTS
            outcome = "WIN"
            exit_ts = bar["ts_event"]
            exit_price = target
            break

        if moved_to_be and low <= entry:
            pnl = 0
            outcome = "BREAKEVEN"
            exit_ts = bar["ts_event"]
            exit_price = entry
            break

        if low <= stop:
            pnl = -STOP_POINTS
            outcome = "LOSS"
            exit_ts = bar["ts_event"]
            exit_price = stop
            break

    if outcome == "":
        final = future.tail(1).row(0, named=True)
        exit_ts = final["ts_event"]
        exit_price = final["close"]
        pnl = exit_price - entry
        outcome = "TIME_WIN" if pnl > 0 else "TIME_LOSS" if pnl < 0 else "FLAT"

    equity += pnl
    peak = max(peak, equity)
    max_dd = max(max_dd, peak - equity)

    trades.append({
        "setup_ts": ts,
        "exit_ts": exit_ts,
        "trade_date_ct": row["trade_date_ct"],
        "hour_ct": row["hour_ct"],
        "minute_ct": row["minute_ct"],
        "symbol": symbol,
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "exit_price": round(exit_price, 2),
        "outcome": outcome,
        "pnl": round(pnl, 2),
        "bars_held": bars_held,
        "mae": round(mae, 2),
        "mfe": round(mfe, 2),
        "or_range_30m": row["or_range_30m"],
        "premarket_range": row["premarket_range"],
        "prior_day_range": row["prior_day_range"],
        "avg_spread": row["avg_spread"],
        "rel_vol_20": row["rel_vol_20"],
        "atr_14": row["atr_14"],
        "atr_50": row["atr_50"],
        "equity": round(equity, 2),
        "drawdown": round(peak - equity, 2),
    })

trades_df = pl.DataFrame(trades).with_columns([
    pl.col("setup_ts").dt.year().alias("year"),
    pl.col("setup_ts").dt.month().alias("month"),
])

summary = trades_df.select([
    pl.len().alias("trades"),
    (pl.col("pnl") > 0).mean().alias("winrate"),
    pl.sum("pnl").alias("net_points"),
    pl.mean("pnl").alias("avg_trade"),
    pl.median("pnl").alias("median_trade"),
    pl.max("drawdown").alias("max_drawdown"),
])

by_year = (
    trades_df
    .group_by("year")
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl") > 0).mean().alias("winrate"),
        pl.sum("pnl").alias("net_points"),
        pl.mean("pnl").alias("avg_trade"),
        pl.max("drawdown").alias("max_drawdown"),
    ])
    .sort("year")
)

by_month = (
    trades_df
    .group_by(["year", "month"])
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl") > 0).mean().alias("winrate"),
        pl.sum("pnl").alias("net_points"),
        pl.mean("pnl").alias("avg_trade"),
    ])
    .sort(["year", "month"])
)

by_outcome = (
    trades_df
    .group_by("outcome")
    .agg([
        pl.len().alias("trades"),
        pl.sum("pnl").alias("net_points"),
        pl.mean("pnl").alias("avg_pnl"),
        pl.mean("bars_held").alias("avg_bars"),
        pl.mean("mae").alias("avg_mae"),
        pl.mean("mfe").alias("avg_mfe"),
    ])
    .sort("trades", descending=True)
)

print("\nSUMMARY")
print(summary)

print("\nBY YEAR")
print(by_year)

print("\nBY OUTCOME")
print(by_outcome)

trades_df.write_csv(OUTPUT_DIR / "top_htf_model_trades.csv")
summary.write_csv(OUTPUT_DIR / "top_htf_model_summary.csv")
by_year.write_csv(OUTPUT_DIR / "top_htf_model_by_year.csv")
by_month.write_csv(OUTPUT_DIR / "top_htf_model_by_month.csv")
by_outcome.write_csv(OUTPUT_DIR / "top_htf_model_by_outcome.csv")

elapsed = time.time() - START

print("\nDONE")
print(f"Runtime: {elapsed:,.1f} sec")
print("\nFILES SAVED:")
print(OUTPUT_DIR / "top_htf_model_trades.csv")
print(OUTPUT_DIR / "top_htf_model_summary.csv")
print(OUTPUT_DIR / "top_htf_model_by_year.csv")
print(OUTPUT_DIR / "top_htf_model_by_month.csv")
print(OUTPUT_DIR / "top_htf_model_by_outcome.csv")