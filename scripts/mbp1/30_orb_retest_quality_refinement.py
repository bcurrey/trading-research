from pathlib import Path
import polars as pl

DATA_PATH = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT_DIR = Path(r"D:\TradingResearch\research_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_NAME = "30_orb_retest_quality_refinement"

TRADES_OUT = OUT_DIR / f"{SCRIPT_NAME}_trades.csv"
SUMMARY_OUT = OUT_DIR / f"{SCRIPT_NAME}_summary.csv"
MONTHLY_OUT = OUT_DIR / f"{SCRIPT_NAME}_monthly.csv"
YEARLY_OUT = OUT_DIR / f"{SCRIPT_NAME}_yearly.csv"

TP = 20
STOP = 20
RETEST_BUFFER = 4
MAX_RETEST_WAIT_BARS = 30
MAX_HOLD_BARS = 120

print("Loading data...")
df = pl.read_parquet(DATA_PATH).sort("ts_event")

required = [
    "ts_ct", "trade_date_ct", "hour_ct", "minute_ct",
    "open", "high", "low", "close",
    "is_rth", "is_opening_30m",
    "vwap_day", "ema_20", "atr_14", "rel_vol_20",
    "is_displacement_candle", "body_abs", "bar_range",
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
        (pl.col("close") > pl.col("vwap_day")).alias("above_vwap"),
        (pl.col("close") < pl.col("vwap_day")).alias("below_vwap"),
        (pl.col("close") > pl.col("ema_20")).alias("above_ema20"),
        (pl.col("close") < pl.col("ema_20")).alias("below_ema20"),
    ])
)

print(f"Rows loaded: {df.height:,}")
print(f"RTH rows: {rth.height:,}")
print(f"Trade-window rows: {trade_window.height:,}")

work = trade_window.select([
    "bar_id", "ts_ct", "trade_date_ct",
    "open", "high", "low", "close",
    "vwap_day", "ema_20", "atr_14", "rel_vol_20",
    "is_displacement_candle", "body_abs", "bar_range",
    "or_high", "or_low", "or_range",
    "breaks_or_high", "breaks_or_low",
    "above_vwap", "below_vwap", "above_ema20", "below_ema20",
]).to_dicts()

FILTERS = [
    {"name": "BASE", "side": "BOTH", "vwap": False, "ema20": False, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "LONG_ONLY", "side": "LONG", "vwap": False, "ema20": False, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "SHORT_ONLY", "side": "SHORT", "vwap": False, "ema20": False, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "VWAP_ALIGN", "side": "BOTH", "vwap": True, "ema20": False, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "EMA20_ALIGN", "side": "BOTH", "vwap": False, "ema20": True, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "VWAP_EMA20_ALIGN", "side": "BOTH", "vwap": True, "ema20": True, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "DISPLACEMENT_BREAK", "side": "BOTH", "vwap": False, "ema20": False, "disp": True, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "OR_20_TO_80", "side": "BOTH", "vwap": False, "ema20": False, "disp": False, "or_min": 20, "or_max": 80, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "OR_30_TO_90", "side": "BOTH", "vwap": False, "ema20": False, "disp": False, "or_min": 30, "or_max": 90, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "REL_VOL_1_10", "side": "BOTH", "vwap": False, "ema20": False, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 1.10, "min_atr": 0.0},
    {"name": "ATR_8_PLUS", "side": "BOTH", "vwap": False, "ema20": False, "disp": False, "or_min": 20, "or_max": 120, "min_relvol": 0.0, "min_atr": 8.0},
    {"name": "PROD_STACK_LIGHT", "side": "BOTH", "vwap": True, "ema20": False, "disp": False, "or_min": 20, "or_max": 90, "min_relvol": 0.0, "min_atr": 0.0},
    {"name": "PROD_STACK_STRICT", "side": "BOTH", "vwap": True, "ema20": True, "disp": True, "or_min": 20, "or_max": 90, "min_relvol": 1.00, "min_atr": 0.0},
]

def filter_ok(row, direction, cfg):
    if cfg["side"] != "BOTH" and cfg["side"] != direction:
        return False

    if row["or_range"] is None:
        return False

    or_range = float(row["or_range"])
    if or_range < cfg["or_min"] or or_range > cfg["or_max"]:
        return False

    if cfg["vwap"]:
        if direction == "LONG" and not row["above_vwap"]:
            return False
        if direction == "SHORT" and not row["below_vwap"]:
            return False

    if cfg["ema20"]:
        if direction == "LONG" and not row["above_ema20"]:
            return False
        if direction == "SHORT" and not row["below_ema20"]:
            return False

    if cfg["disp"] and not row["is_displacement_candle"]:
        return False

    if cfg["min_relvol"] > 0:
        if row["rel_vol_20"] is None or float(row["rel_vol_20"]) < cfg["min_relvol"]:
            return False

    if cfg["min_atr"] > 0:
        if row["atr_14"] is None or float(row["atr_14"]) < cfg["min_atr"]:
            return False

    return True

trades = []

for cfg in FILTERS:
    traded_dates = set()

    for i in range(len(work) - MAX_HOLD_BARS - 2):
        row = work[i]
        trade_date = row["trade_date_ct"]

        if trade_date in traded_dates:
            continue

        if row["or_high"] is None or row["or_low"] is None:
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

        if not filter_ok(row, direction, cfg):
            continue

        entry_i = None
        entry = None

        for j in range(i + 1, min(i + MAX_RETEST_WAIT_BARS + 1, len(work))):
            pull = work[j]

            if pull["trade_date_ct"] != trade_date:
                break

            if direction == "LONG":
                touched = float(pull["low"]) <= or_level + RETEST_BUFFER
                held = float(pull["close"]) >= or_level
                if touched and held:
                    entry_i = j
                    entry = or_level
                    break

            if direction == "SHORT":
                touched = float(pull["high"]) >= or_level - RETEST_BUFFER
                held = float(pull["close"]) <= or_level
                if touched and held:
                    entry_i = j
                    entry = or_level
                    break

        if entry_i is None:
            continue

        if direction == "LONG":
            stop_price = entry - STOP
            target_price = entry + TP
        else:
            stop_price = entry + STOP
            target_price = entry - TP

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
            "model": cfg["name"],
            "trade_date": str(trade_date),
            "year": str(trade_date)[:4],
            "month": str(trade_date)[:7],
            "direction": direction,
            "break_time": str(row["ts_ct"]),
            "entry_time": str(work[entry_i]["ts_ct"]),
            "exit_time": str(work[exit_i]["ts_ct"]),
            "or_high": round(float(row["or_high"]), 2),
            "or_low": round(float(row["or_low"]), 2),
            "or_range": round(float(row["or_range"]), 2),
            "entry": round(entry, 2),
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "tp": TP,
            "stop": STOP,
            "retest_buffer": RETEST_BUFFER,
            "outcome": outcome,
            "pnl_points": round(pnl, 2),
            "break_above_vwap": bool(row["above_vwap"]),
            "break_below_vwap": bool(row["below_vwap"]),
            "break_above_ema20": bool(row["above_ema20"]),
            "break_below_ema20": bool(row["below_ema20"]),
            "break_displacement": bool(row["is_displacement_candle"]),
            "rel_vol_20": None if row["rel_vol_20"] is None else round(float(row["rel_vol_20"]), 3),
            "atr_14": None if row["atr_14"] is None else round(float(row["atr_14"]), 2),
        })

        traded_dates.add(trade_date)

if not trades:
    print("NO TRADES FOUND.")
    pl.DataFrame({"status": ["NO_TRADES"]}).write_csv(SUMMARY_OUT)
    raise SystemExit

trades_df = pl.DataFrame(trades)
keys = ["model"]

trades_df = trades_df.sort(["model", "trade_date", "entry_time"])
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
        (pl.col("pnl_points") > 0).sum().alias("wins"),
        (pl.col("pnl_points") < 0).sum().alias("losses"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().alias("gross_loss"),
        pl.col("month").n_unique().alias("active_months"),
        pl.col("year").n_unique().alias("active_years"),
    ])
    .with_columns([
        pl.when(pl.col("gross_loss") > 0)
        .then((pl.col("gross_win") / pl.col("gross_loss")).round(3))
        .otherwise(None)
        .alias("profit_factor"),
        (pl.col("trades") / pl.col("active_months")).round(2).alias("trades_per_month"),
    ])
    .with_columns([
        pl.when(
            (pl.col("trades") >= 300) &
            (pl.col("winrate") >= 0.50) &
            (pl.col("profit_factor") >= 1.30) &
            (pl.col("avg_trade") >= 3.0) &
            (pl.col("net_points") > 0)
        )
        .then(pl.lit("A_CANDIDATE"))
        .when(
            (pl.col("trades") >= 150) &
            (pl.col("winrate") >= 0.48) &
            (pl.col("profit_factor") >= 1.20) &
            (pl.col("avg_trade") > 1.5) &
            (pl.col("net_points") > 0)
        )
        .then(pl.lit("B_CANDIDATE"))
        .otherwise(pl.lit("REJECT"))
        .alias("tier")
    ])
    .sort(["tier", "profit_factor", "avg_trade", "net_points"], descending=[False, True, True, True])
)

monthly = (
    trades_df.group_by(["model", "month"])
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
    ])
    .sort(["model", "month"])
)

yearly = (
    trades_df.group_by(["model", "year"])
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
    ])
    .sort(["model", "year"])
)

trades_df.write_csv(TRADES_OUT)
summary.write_csv(SUMMARY_OUT)
monthly.write_csv(MONTHLY_OUT)
yearly.write_csv(YEARLY_OUT)

print("\nSUMMARY")
print(summary)

print("\nTOP CANDIDATES")
print(summary.filter(pl.col("tier") != "REJECT"))

print(f"\nSaved trades: {TRADES_OUT}")
print(f"Saved summary: {SUMMARY_OUT}")
print(f"Saved monthly: {MONTHLY_OUT}")
print(f"Saved yearly: {YEARLY_OUT}")
