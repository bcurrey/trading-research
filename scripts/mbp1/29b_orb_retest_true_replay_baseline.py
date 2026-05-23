from pathlib import Path
import polars as pl

DATA_PATH = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT_DIR = Path(r"D:\TradingResearch\research_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_NAME = "29b_orb_retest_true_replay_baseline"

TRADES_OUT = OUT_DIR / f"{SCRIPT_NAME}_trades.csv"
SUMMARY_OUT = OUT_DIR / f"{SCRIPT_NAME}_summary.csv"

TP_VALUES = [20, 30, 40]
STOP_VALUES = [20, 30, 40]
RETEST_BUFFER_VALUES = [0, 2, 4]
MAX_RETEST_WAIT_BARS = 30
MAX_HOLD_BARS = 120

print("Loading data...")
df = pl.read_parquet(DATA_PATH).sort("ts_event")

required = [
    "ts_ct", "trade_date_ct", "hour_ct", "minute_ct",
    "open", "high", "low", "close",
    "is_rth", "is_opening_30m"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

rth = df.filter(pl.col("is_rth") == True).with_row_index("bar_id")

orb = (
    rth.filter(pl.col("is_opening_30m") == True)
    .group_by("trade_date_ct")
    .agg([
        pl.col("high").max().alias("or_high"),
        pl.col("low").min().alias("or_low"),
        (pl.col("high").max() - pl.col("low").min()).alias("or_range"),
    ])
)

trade_window = (
    rth.join(orb, on="trade_date_ct", how="left")
    .filter(
        (
            ((pl.col("hour_ct") > 9) | ((pl.col("hour_ct") == 9) & (pl.col("minute_ct") >= 0))) &
            ((pl.col("hour_ct") < 11) | ((pl.col("hour_ct") == 11) & (pl.col("minute_ct") <= 30)))
        )
    )
    .with_columns([
        (pl.col("high") > pl.col("or_high")).alias("breaks_or_high"),
        (pl.col("low") < pl.col("or_low")).alias("breaks_or_low"),
    ])
)

print(f"Rows loaded: {df.height:,}")
print(f"RTH rows: {rth.height:,}")
print(f"Trade-window rows: {trade_window.height:,}")

work = trade_window.select([
    "bar_id", "ts_ct", "trade_date_ct",
    "open", "high", "low", "close",
    "or_high", "or_low", "or_range",
    "breaks_or_high", "breaks_or_low",
]).to_dicts()

trades = []

configs = []
for tp in TP_VALUES:
    for stop in STOP_VALUES:
        for retest_buffer in RETEST_BUFFER_VALUES:
            configs.append({"tp": tp, "stop": stop, "retest_buffer": retest_buffer})

for cfg in configs:
    traded_dates = set()

    for i in range(len(work) - MAX_HOLD_BARS - 2):
        row = work[i]
        trade_date = row["trade_date_ct"]

        if trade_date in traded_dates:
            continue

        if row["or_high"] is None or row["or_low"] is None or row["or_range"] is None:
            continue

        or_range = float(row["or_range"])

        if or_range < 20 or or_range > 120:
            continue

        direction = None
        or_level = None

        if row["breaks_or_high"]:
            direction = "LONG"
            or_level = float(row["or_high"])
        elif row["breaks_or_low"]:
            direction = "SHORT"
            or_level = float(row["or_low"])
        else:
            continue

        entry_i = None
        entry = None

        for j in range(i + 1, min(i + MAX_RETEST_WAIT_BARS + 1, len(work))):
            pull = work[j]

            if pull["trade_date_ct"] != trade_date:
                break

            if direction == "LONG":
                touched = float(pull["low"]) <= or_level + cfg["retest_buffer"]
                held = float(pull["close"]) >= or_level
                if touched and held:
                    entry_i = j
                    entry = or_level
                    break

            if direction == "SHORT":
                touched = float(pull["high"]) >= or_level - cfg["retest_buffer"]
                held = float(pull["close"]) <= or_level
                if touched and held:
                    entry_i = j
                    entry = or_level
                    break

        if entry_i is None:
            continue

        if direction == "LONG":
            stop_price = entry - cfg["stop"]
            target_price = entry + cfg["tp"]
        else:
            stop_price = entry + cfg["stop"]
            target_price = entry - cfg["tp"]

        exit_i = None
        exit_price = None
        outcome = None

        for k in range(entry_i + 1, min(entry_i + MAX_HOLD_BARS + 1, len(work))):
            fwd = work[k]

            if fwd["trade_date_ct"] != trade_date:
                break

            if direction == "LONG":
                hit_stop = float(fwd["low"]) <= stop_price
                hit_target = float(fwd["high"]) >= target_price
            else:
                hit_stop = float(fwd["high"]) >= stop_price
                hit_target = float(fwd["low"]) <= target_price

            if hit_stop and hit_target:
                exit_i = k
                exit_price = stop_price
                outcome = "LOSS_SAME_BAR_CONSERVATIVE"
                break

            if hit_stop:
                exit_i = k
                exit_price = stop_price
                outcome = "LOSS"
                break

            if hit_target:
                exit_i = k
                exit_price = target_price
                outcome = "WIN"
                break

        if exit_i is None:
            exit_i = entry_i
            for k in range(entry_i + 1, min(entry_i + MAX_HOLD_BARS + 1, len(work))):
                if work[k]["trade_date_ct"] != trade_date:
                    break
                exit_i = k

            exit_price = float(work[exit_i]["close"])
            outcome = "TIME_EXIT"

        if direction == "LONG":
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price

        trades.append({
            "model": f"ORB_RETEST_TP{cfg['tp']}_SL{cfg['stop']}_BUF{cfg['retest_buffer']}",
            "trade_date": str(trade_date),
            "direction": direction,
            "break_time": str(row["ts_ct"]),
            "entry_time": str(work[entry_i]["ts_ct"]),
            "exit_time": str(work[exit_i]["ts_ct"]),
            "or_high": round(float(row["or_high"]), 2),
            "or_low": round(float(row["or_low"]), 2),
            "or_range": round(or_range, 2),
            "entry": round(entry, 2),
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "tp": cfg["tp"],
            "stop": cfg["stop"],
            "retest_buffer": cfg["retest_buffer"],
            "outcome": outcome,
            "pnl_points": round(pnl, 2),
        })

        traded_dates.add(trade_date)

if not trades:
    print("NO TRADES FOUND.")
    pl.DataFrame({"status": ["NO_TRADES"]}).write_csv(SUMMARY_OUT)
    raise SystemExit

trades_df = pl.DataFrame(trades)
keys = ["model", "tp", "stop", "retest_buffer"]

trades_df = trades_df.with_columns(pl.col("pnl_points").cum_sum().over(keys).alias("equity"))
trades_df = trades_df.with_columns(pl.col("equity").cum_max().over(keys).alias("equity_high"))
trades_df = trades_df.with_columns((pl.col("equity") - pl.col("equity_high")).alias("drawdown"))

summary = (
    trades_df.group_by(keys)
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.col("drawdown").min().round(2).alias("max_dd_points"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().alias("gross_loss"),
    ])
    .with_columns(
        pl.when(pl.col("gross_loss") > 0)
        .then((pl.col("gross_win") / pl.col("gross_loss")).round(3))
        .otherwise(None)
        .alias("profit_factor")
    )
    .with_columns(
        pl.when(
            (pl.col("trades") >= 80) &
            (pl.col("winrate") >= 0.48) &
            (pl.col("profit_factor") >= 1.15) &
            (pl.col("net_points") > 0) &
            (pl.col("avg_trade") > 0)
        )
        .then(pl.lit("CANDIDATE"))
        .otherwise(pl.lit("REJECT"))
        .alias("tier")
    )
    .sort(["tier", "profit_factor", "net_points", "winrate"], descending=[False, True, True, True])
)

trades_df.write_csv(TRADES_OUT)
summary.write_csv(SUMMARY_OUT)

print("\nSUMMARY")
print(summary)

print("\nCANDIDATES")
print(summary.filter(pl.col("tier") == "CANDIDATE"))

print(f"\nSaved trades: {TRADES_OUT}")
print(f"Saved summary: {SUMMARY_OUT}")
