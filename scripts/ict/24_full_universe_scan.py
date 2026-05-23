# 24_full_universe_scan.py
# Full raw-feature scan for institutional combo pattern.
# Purpose: validate whether the 35-38 win/no-loss lower_wick edge exists outside prior trade subset.

from pathlib import Path
import polars as pl
import sys
import math

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
REFERENCE_TRADES = ROOT / r"research_outputs\combo_8_10_feature_audit\combo_8_10_trades_with_features.csv"
OUTDIR = ROOT / r"research_outputs\full_universe_scan"

OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "full_universe_trades.csv"
SUMMARY_OUT = OUTDIR / "full_universe_summary.csv"
YEARLY_OUT = OUTDIR / "full_universe_yearly.csv"
MONTHLY_OUT = OUTDIR / "full_universe_monthly.csv"
THRESHOLD_OUT = OUTDIR / "full_universe_threshold_sweep.csv"
LOSERS_OUT = OUTDIR / "full_universe_losers.csv"

print("Loading reference trades...")
ref = pl.read_csv(REFERENCE_TRADES, try_parse_dates=True)

# Reference q20/q25 thresholds from the actual discovered combo set.
base_ref = ref.filter(
    (pl.col("hour").is_in([8, 10])) &
    (pl.col("risk_points") >= 6) &
    (pl.col("risk_points") <= 20) &
    (pl.col("mae_points") > -10)
)

q20 = float(base_ref["lower_wick_pct"].quantile(0.20))
q25 = float(base_ref["lower_wick_pct"].quantile(0.25))
q30 = float(base_ref["lower_wick_pct"].quantile(0.30))
q35 = float(base_ref["lower_wick_pct"].quantile(0.35))

print(f"Reference thresholds: q20={q20:.5f}, q25={q25:.5f}, q30={q30:.5f}, q35={q35:.5f}")

print("Loading full feature dataset...")
df = pl.read_parquet(FEATURES)
print(f"Rows loaded: {df.height:,}")

# Required feature assumptions from scripts 15-23.
needed = [
    "ts_ct","trade_date_ct","hour_ct","minute_ct",
    "open","high","low","close","volume",
    "ema_20","ema_50",
    "range_expansion_20","rel_vol_20",
    "bear_displacement",
    "any_liquidity_sweep_reclaim_last_30m",
    "lower_wick_pct","body_pct",
]

missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = (
    df
    .with_row_index("idx")
    .with_columns(
        pl.col("ts_ct").dt.year().alias("year"),
        pl.col("ts_ct").dt.strftime("%Y-%m").alias("month"),
        pl.col("ts_ct").dt.weekday().alias("day_of_week"),
        (pl.col("high") - pl.col("low")).alias("bar_range_calc"),
    )
)

# Full universe signal definition:
# mimic original deep retrace core but scan raw rows.
signals = (
    df.filter(
        (pl.col("hour_ct").is_in([8, 10])) &
        (
            ((pl.col("hour_ct") == 8) & (pl.col("minute_ct") >= 30)) |
            (pl.col("hour_ct") == 10)
        ) &
        (pl.col("ema_20") < pl.col("ema_50")) &
        (pl.col("close") < pl.col("ema_20")) &
        (pl.col("bear_displacement") == True) &
        (pl.col("any_liquidity_sweep_reclaim_last_30m") == True) &
        (pl.col("range_expansion_20") > 1.15) &
        (pl.col("rel_vol_20") > 1.05)
    )
    .with_columns(
        (pl.col("high") - 0.25 * (pl.col("high") - pl.col("low"))).alias("entry_price"),
        (pl.col("high") + 2.0).alias("stop_price"),
    )
    .with_columns(
        (pl.col("stop_price") - pl.col("entry_price")).alias("risk_points")
    )
    .with_columns(
        (pl.col("entry_price") - 2.0 * pl.col("risk_points")).alias("target_price")
    )
    .filter(
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20)
    )
)

print(f"Raw signals before fill/lower wick: {signals.height:,}")

# Convert for replay.
replay_cols = ["ts_ct", "trade_date_ct", "year", "month", "hour_ct", "minute_ct", "open", "high", "low", "close"]
rows = df.select(replay_cols).to_dicts()

sig_rows = signals.select([
    "idx","ts_ct","trade_date_ct","year","month","day_of_week",
    "hour_ct","minute_ct",
    "open","high","low","close",
    "entry_price","stop_price","target_price","risk_points",
    "lower_wick_pct","body_pct","range_expansion_20","rel_vol_20",
]).to_dicts()

def simulate_short(start_i, entry, stop, target, max_hold=90):
    max_fav = 0.0
    max_adv = 0.0

    for j in range(start_i, min(len(rows), start_i + max_hold)):
        high = float(rows[j]["high"])
        low = float(rows[j]["low"])
        close = float(rows[j]["close"])

        max_fav = max(max_fav, entry - low)
        max_adv = max(max_adv, high - entry)

        stop_hit = high >= stop
        target_hit = low <= target

        if stop_hit and target_hit:
            return "stop_same_bar", entry - stop, j - start_i + 1, max_fav, -max_adv
        if stop_hit:
            return "stop", entry - stop, j - start_i + 1, max_fav, -max_adv
        if target_hit:
            return "target", entry - target, j - start_i + 1, max_fav, -max_adv

    last_i = min(len(rows) - 1, start_i + max_hold - 1)
    close = float(rows[last_i]["close"])
    return "time", entry - close, last_i - start_i + 1, max_fav, -max_adv

trades = []

for s in sig_rows:
    sig_i = int(s["idx"])

    # fill within 1 bar only, same as candidate.
    fill_i = sig_i + 1
    if fill_i >= len(rows):
        continue

    entry = float(s["entry_price"])
    stop = float(s["stop_price"])
    target = float(s["target_price"])
    risk = float(s["risk_points"])

    fill_bar = rows[fill_i]

    if not (float(fill_bar["high"]) >= entry and float(fill_bar["low"]) <= entry):
        continue

    exit_reason, result, bars_held, mfe, mae = simulate_short(fill_i, entry, stop, target)

    trades.append({
        "signal_time": s["ts_ct"],
        "entry_time": fill_bar["ts_ct"],
        "trade_date": fill_bar["trade_date_ct"],
        "year": fill_bar["year"],
        "month": fill_bar["month"],
        "day_of_week": s["day_of_week"],
        "hour": fill_bar["hour_ct"],
        "minute": fill_bar["minute_ct"],
        "side": "short",
        "entry_price": round(entry, 2),
        "stop_price": round(stop, 2),
        "target_price": round(target, 2),
        "risk_points": round(risk, 2),
        "result_points": round(result, 2),
        "result_rr": round(result / risk, 4) if risk > 0 else None,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "fill_bars": 1,
        "mfe_points": round(mfe, 2),
        "mae_points": round(mae, 2),
        "lower_wick_pct": s["lower_wick_pct"],
        "body_pct": s["body_pct"],
        "range_expansion_20": s["range_expansion_20"],
        "rel_vol_20": s["rel_vol_20"],
    })

trades_df = pl.DataFrame(trades).sort("entry_time")

print(f"Filled trades before lower_wick filter: {trades_df.height:,}")

def max_dd(x):
    if x.height == 0:
        return 0
    eq = (
        x.sort("entry_time")
        .with_columns(pl.col("result_points").cum_sum().alias("equity"))
        .with_columns(pl.col("equity").cum_max().alias("peak"))
        .with_columns((pl.col("equity") - pl.col("peak")).alias("dd"))
    )
    return round(eq["dd"].min(), 2)

def summarize(x, label):
    if x.height == 0:
        return None
    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)
    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0
    return {
        "label": label,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": max_dd(x),
        "first_trade": str(x["entry_time"].min()),
        "last_trade": str(x["entry_time"].max()),
    }

summary_rows = []

thresholds = {
    "no_lower_wick_filter": None,
    f"ref_q20_gt_{q20:.5f}": q20,
    f"ref_q25_gt_{q25:.5f}": q25,
    f"ref_q30_gt_{q30:.5f}": q30,
    f"ref_q35_gt_{q35:.5f}": q35,
}

for label, th in thresholds.items():
    if th is None:
        x = trades_df
    else:
        x = trades_df.filter(pl.col("lower_wick_pct") > th)

    s = summarize(x, label)
    if s:
        summary_rows.append(s)

summary = pl.DataFrame(summary_rows)

best = trades_df.filter(pl.col("lower_wick_pct") > q20)

yearly = (
    best.group_by("year")
    .agg(
        pl.len().alias("trades"),
        (pl.col("result_points") > 0).sum().alias("wins"),
        (pl.col("result_points") <= 0).sum().alias("losses"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .with_columns((100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"))
    .sort("year")
)

monthly = (
    best.group_by("month")
    .agg(
        pl.len().alias("trades"),
        (pl.col("result_points") > 0).sum().alias("wins"),
        (pl.col("result_points") <= 0).sum().alias("losses"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .with_columns((100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"))
    .sort("month")
)

trades_df.write_csv(TRADES_OUT)
summary.write_csv(SUMMARY_OUT)
yearly.write_csv(YEARLY_OUT)
monthly.write_csv(MONTHLY_OUT)
summary.write_csv(THRESHOLD_OUT)
best.filter(pl.col("result_points") <= 0).write_csv(LOSERS_OUT)

print("\nFULL UNIVERSE SUMMARY")
print(summary)

print("\nBEST Q20 YEARLY")
print(yearly)

print("\nBEST Q20 LOSERS")
print(best.filter(pl.col("result_points") <= 0))

print(f"\nSaved: {TRADES_OUT}")
print(f"Saved: {SUMMARY_OUT}")
print(f"Saved: {YEARLY_OUT}")
print(f"Saved: {MONTHLY_OUT}")
print(f"Saved losers: {LOSERS_OUT}")
