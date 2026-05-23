from pathlib import Path
import polars as pl
import math

DATA_PATH = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT_DIR = Path(r"D:\TradingResearch\research_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_NAME = "33_orb_liquidity_target_test"

TRADES_OUT = OUT_DIR / f"{SCRIPT_NAME}_trades.csv"
SUMMARY_OUT = OUT_DIR / f"{SCRIPT_NAME}_summary.csv"
YEARLY_OUT = OUT_DIR / f"{SCRIPT_NAME}_yearly.csv"
RECENT_OUT = OUT_DIR / f"{SCRIPT_NAME}_recent_2024_2026.csv"

STOP = 20
FIXED_TP = 20
RETEST_BUFFER = 4
MAX_RETEST_WAIT_BARS = 30
MAX_HOLD_BARS = 120

MIN_LIQ_TP = 20
MAX_LIQ_TP = 80

print("Loading data...")
df = pl.read_parquet(DATA_PATH).sort("ts_event")

required = [
    "ts_ct", "trade_date_ct", "hour_ct", "minute_ct",
    "open", "high", "low", "close",
    "is_rth", "is_opening_30m",
    "vwap_day",
    "prior_rth_high", "prior_rth_low",
    "premarket_high", "premarket_low",
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

# Build London / Asia levels from full dataset.
# Asia used here: 19:00-23:59 CT from the prior evening / same trade_date_ct handling in data.
# London: 02:00-04:59 CT.
asia_levels = (
    df.filter(((pl.col("hour_ct") >= 19) & (pl.col("hour_ct") <= 23)))
    .group_by("trade_date_ct")
    .agg([
        pl.col("high").max().alias("asia_high"),
        pl.col("low").min().alias("asia_low"),
    ])
)

london_levels = (
    df.filter(((pl.col("hour_ct") >= 2) & (pl.col("hour_ct") <= 4)))
    .group_by("trade_date_ct")
    .agg([
        pl.col("high").max().alias("london_high"),
        pl.col("low").min().alias("london_low"),
    ])
)

trade_window = (
    rth.join(orb, on="trade_date_ct", how="left")
    .join(asia_levels, on="trade_date_ct", how="left")
    .join(london_levels, on="trade_date_ct", how="left")
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
    ])
)

print(f"Rows loaded: {df.height:,}")
print(f"Trade-window rows: {trade_window.height:,}")

work = trade_window.select([
    "bar_id", "ts_ct", "trade_date_ct",
    "open", "high", "low", "close",
    "vwap_day",
    "or_high", "or_low", "or_range",
    "prior_rth_high", "prior_rth_low",
    "premarket_high", "premarket_low",
    "asia_high", "asia_low",
    "london_high", "london_low",
    "breaks_or_high", "breaks_or_low",
    "above_vwap", "below_vwap",
]).to_dicts()

def valid_num(x):
    if x is None:
        return False
    try:
        return not math.isnan(float(x))
    except Exception:
        return False

def pick_liquidity_target(row, direction, entry):
    candidates = []

    levels = [
        ("PDH", row.get("prior_rth_high")),
        ("PDL", row.get("prior_rth_low")),
        ("PMH", row.get("premarket_high")),
        ("PML", row.get("premarket_low")),
        ("ASIA_H", row.get("asia_high")),
        ("ASIA_L", row.get("asia_low")),
        ("LDN_H", row.get("london_high")),
        ("LDN_L", row.get("london_low")),
    ]

    for name, lvl in levels:
        if not valid_num(lvl):
            continue

        lvl = float(lvl)

        if direction == "LONG":
            dist = lvl - entry
            if MIN_LIQ_TP <= dist <= MAX_LIQ_TP:
                candidates.append((dist, lvl, name))

        if direction == "SHORT":
            dist = entry - lvl
            if MIN_LIQ_TP <= dist <= MAX_LIQ_TP:
                candidates.append((dist, lvl, name))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[0])
    dist, lvl, name = candidates[0]
    return lvl, dist, name

MODELS = [
    {"name": "FIXED_TP20_BASELINE", "target_mode": "FIXED"},
    {"name": "LIQ_TP_20_80_ONLY", "target_mode": "LIQ_ONLY"},
    {"name": "LIQ_TP_20_80_FALLBACK20", "target_mode": "LIQ_FALLBACK"},
]

trades = []

for model in MODELS:
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

        if row["breaks_or_high"] and row["above_vwap"]:
            direction = "LONG"
            or_level = float(row["or_high"])
        elif row["breaks_or_low"] and row["below_vwap"]:
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

        liq_target, liq_dist, liq_name = pick_liquidity_target(row, direction, entry)

        if model["target_mode"] == "FIXED":
            tp_dist = FIXED_TP
            target_type = "FIXED20"
            target_name = "FIXED20"

        elif model["target_mode"] == "LIQ_ONLY":
            if liq_target is None:
                continue
            tp_dist = liq_dist
            target_type = "LIQ"
            target_name = liq_name

        else:
            if liq_target is None:
                tp_dist = FIXED_TP
                target_type = "FALLBACK20"
                target_name = "FALLBACK20"
            else:
                tp_dist = liq_dist
                target_type = "LIQ"
                target_name = liq_name

        if direction == "LONG":
            stop_price = entry - STOP
            target_price = entry + tp_dist
        else:
            stop_price = entry + STOP
            target_price = entry - tp_dist

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

        pnl = exit_price - entry if direction == "LONG" else entry - exit_price

        trades.append({
            "model": model["name"],
            "trade_date": str(trade_date),
            "year": str(trade_date)[:4],
            "month": str(trade_date)[:7],
            "direction": direction,
            "break_time": str(row["ts_ct"]),
            "entry_time": str(work[entry_i]["ts_ct"]),
            "exit_time": str(work[exit_i]["ts_ct"]),
            "entry": round(entry, 2),
            "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2),
            "target_type": target_type,
            "target_name": target_name,
            "tp_dist": round(float(tp_dist), 2),
            "outcome": outcome,
            "pnl_points": round(pnl, 2),
            "or_range": round(or_range, 2),
        })

        traded_dates.add(trade_date)

if not trades:
    print("NO TRADES FOUND.")
    pl.DataFrame({"status": ["NO_TRADES"]}).write_csv(SUMMARY_OUT)
    raise SystemExit

trades_df = pl.DataFrame(trades).sort(["model", "trade_date", "entry_time"])

keys = ["model"]

trades_df = trades_df.with_columns([
    pl.col("pnl_points").cum_sum().over(keys).alias("equity"),
])
trades_df = trades_df.with_columns([
    pl.col("equity").cum_max().over(keys).alias("equity_high"),
])
trades_df = trades_df.with_columns([
    (pl.col("equity") - pl.col("equity_high")).alias("drawdown"),
])

summary = (
    trades_df.group_by("model")
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.col("drawdown").min().round(2).alias("max_dd_points"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_loss"),
        pl.col("tp_dist").mean().round(2).alias("avg_tp_dist"),
    ])
    .with_columns([
        (pl.col("gross_win") / pl.col("gross_loss")).round(3).alias("profit_factor")
    ])
    .sort(["profit_factor", "net_points"], descending=True)
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

recent = (
    trades_df.filter(pl.col("year").is_in(["2024", "2025", "2026"]))
    .group_by("model")
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.col("tp_dist").mean().round(2).alias("avg_tp_dist"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_loss"),
    ])
    .with_columns([
        (pl.col("gross_win") / pl.col("gross_loss")).round(3).alias("profit_factor")
    ])
    .sort(["profit_factor", "net_points"], descending=True)
)

trades_df.write_csv(TRADES_OUT)
summary.write_csv(SUMMARY_OUT)
yearly.write_csv(YEARLY_OUT)
recent.write_csv(RECENT_OUT)

print("\nSUMMARY")
print(summary)

print("\nRECENT 2024-2026")
print(recent)

print("\n2026 ONLY")
print(
    trades_df.filter(pl.col("year") == "2026")
    .group_by("model")
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.col("tp_dist").mean().round(2).alias("avg_tp_dist"),
    ])
    .sort("net_points", descending=True)
)

print(f"\nSaved trades: {TRADES_OUT}")
print(f"Saved summary: {SUMMARY_OUT}")
print(f"Saved yearly: {YEARLY_OUT}")
print(f"Saved recent: {RECENT_OUT}")
