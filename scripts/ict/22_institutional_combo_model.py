# 22_institutional_combo_model.py
# Institutional-grade combo model testing from combo_8_10 feature audit.

from pathlib import Path
import polars as pl
import random
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\combo_8_10_feature_audit\combo_8_10_trades_with_features.csv"
OUTDIR = ROOT / r"research_outputs\institutional_combo_model"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "institutional_combo_summary.csv"
YEARLY_OUT = OUTDIR / "institutional_combo_yearly.csv"
MONTHLY_OUT = OUTDIR / "institutional_combo_monthly.csv"
EQUITY_OUT = OUTDIR / "institutional_combo_equity.csv"
MONTE_OUT = OUTDIR / "institutional_combo_monte_carlo.csv"
TRADES_OUT = OUTDIR / "institutional_combo_best_trades.csv"

print("Loading joined combo trades...")
df = pl.read_csv(INPUT, try_parse_dates=True)

print(f"Rows loaded: {df.height}")

# Only base model universe
base = df.filter(
    (pl.col("hour").is_in([8, 10])) &
    (pl.col("risk_points") >= 6) &
    (pl.col("risk_points") <= 20) &
    (pl.col("mae_points") > -10)
).sort("entry_time")

print(f"Base trades: {base.height}")

models = {
    "base_combo_8_10": True,

    "lower_wick_gt_q25": (
        pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)
    ),

    "body_pct_lt_q50": (
        pl.col("body_pct") < base["body_pct"].quantile(0.50)
    ),

    "range_exp_gt_q50": (
        pl.col("range_expansion_20") > base["range_expansion_20"].quantile(0.50)
    ),

    "deep_below_vwap_q25": (
        pl.col("dist_vwap") < base["dist_vwap"].quantile(0.25)
    ),

    "rel_vol_gt_q50": (
        pl.col("rel_vol_20") > base["rel_vol_20"].quantile(0.50)
    ),

    "wick_and_range": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("range_expansion_20") > base["range_expansion_20"].quantile(0.50))
    ),

    "body_and_wick": (
        (pl.col("body_pct") < base["body_pct"].quantile(0.50)) &
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25))
    ),

    "institutional_v1": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < base["body_pct"].quantile(0.75)) &
        (pl.col("range_expansion_20") > base["range_expansion_20"].quantile(0.25))
    ),

    "institutional_v2_strict": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.50)) &
        (pl.col("body_pct") < base["body_pct"].quantile(0.50)) &
        (pl.col("range_expansion_20") > base["range_expansion_20"].quantile(0.50))
    ),

    "institutional_v3_vwap": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < base["body_pct"].quantile(0.75)) &
        (pl.col("dist_vwap") < base["dist_vwap"].quantile(0.75))
    ),

    "institutional_v4_sweep": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("liquidity_sweep_reclaim_count_last_30m") >= 3)
    ),
}


def max_drawdown(points):
    eq = []
    running = 0.0
    peak = 0.0
    max_dd = 0.0

    for p in points:
        running += float(p)
        peak = max(peak, running)
        dd = running - peak
        max_dd = min(max_dd, dd)
        eq.append(running)

    return round(max_dd, 2)


def losing_streak(points):
    max_ls = 0
    cur = 0

    for p in points:
        if p <= 0:
            cur += 1
            max_ls = max(max_ls, cur)
        else:
            cur = 0

    return max_ls


def summarize(x: pl.DataFrame, name: str):
    if x.height == 0:
        return None

    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)

    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

    points = x.sort("entry_time")["result_points"].to_list()

    return {
        "model": name,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": max_drawdown(points),
        "max_losing_streak": losing_streak(points),
        "first_trade": str(x["entry_time"].min()),
        "last_trade": str(x["entry_time"].max()),
    }


summary_rows = []
model_frames = {}

for name, filt in models.items():
    if filt is True:
        x = base
    else:
        x = base.filter(filt)

    model_frames[name] = x
    row = summarize(x, name)

    if row:
        summary_rows.append(row)

summary = (
    pl.DataFrame(summary_rows)
    .sort(["losses", "profit_factor", "trades"], descending=[False, True, True])
)

summary.write_csv(SUMMARY_OUT)

# Pick best realistic model:
# prefer >=30 trades, zero/low losses, high PF, high avg trade
eligible = summary.filter(pl.col("trades") >= 30)

if eligible.height:
    best_name = eligible.sort(
        ["losses", "profit_factor", "avg_trade"],
        descending=[False, True, True]
    )["model"][0]
else:
    best_name = summary.sort(
        ["losses", "profit_factor", "avg_trade"],
        descending=[False, True, True]
    )["model"][0]

best = model_frames[best_name].sort("entry_time")

print(f"\nBEST MODEL: {best_name}")

# Yearly/monthly
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

equity = (
    best.with_columns(pl.col("result_points").cum_sum().alias("equity_points"))
    .with_columns(pl.col("equity_points").cum_max().alias("peak_points"))
    .with_columns((pl.col("equity_points") - pl.col("peak_points")).alias("drawdown_points"))
)

# Monte Carlo shuffle
points = best["result_points"].to_list()
mc_rows = []

for i in range(1000):
    shuffled = points[:]
    random.shuffle(shuffled)

    mc_rows.append({
        "sim": i + 1,
        "net_points": round(sum(shuffled), 2),
        "max_dd": max_drawdown(shuffled),
        "max_losing_streak": losing_streak(shuffled),
    })

mc = pl.DataFrame(mc_rows)

best.write_csv(TRADES_OUT)
yearly.write_csv(YEARLY_OUT)
monthly.write_csv(MONTHLY_OUT)
equity.write_csv(EQUITY_OUT)
mc.write_csv(MONTE_OUT)

print("\nSUMMARY")
print(summary)

print("\nBEST YEARLY")
print(yearly)

print("\nMONTE CARLO")
print(
    mc.select([
        pl.col("max_dd").min().alias("worst_dd"),
        pl.col("max_dd").mean().round(2).alias("avg_dd"),
        pl.col("max_losing_streak").max().alias("worst_losing_streak"),
        pl.col("max_losing_streak").mean().round(2).alias("avg_losing_streak"),
    ])
)

print(f"\nSaved: {SUMMARY_OUT}")
print(f"Saved: {YEARLY_OUT}")
print(f"Saved: {MONTHLY_OUT}")
print(f"Saved: {EQUITY_OUT}")
print(f"Saved: {MONTE_OUT}")
print(f"Saved best trades: {TRADES_OUT}")
