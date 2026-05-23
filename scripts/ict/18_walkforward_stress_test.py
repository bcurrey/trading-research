# 18_walkforward_stress_test.py
# Walkforward / yearly robustness test for 10ct combo model

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\deep_retrace_v1_candidate\deep_retrace_v1_trades.csv"
OUTDIR = ROOT / r"research_outputs\walkforward_stress_test"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "walkforward_summary.csv"
YEARLY_OUT = OUTDIR / "walkforward_yearly.csv"

df = pl.read_csv(INPUT, try_parse_dates=True)

combo = (
    df.filter(
        (pl.col("hour") == 10) &
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20) &
        (pl.col("mae_points") > -10)
    )
    .sort("entry_time")
)

equity = (
    combo.with_columns(
        pl.col("result_points").cum_sum().alias("equity")
    )
    .with_columns(
        pl.col("equity").cum_max().alias("peak")
    )
    .with_columns(
        (pl.col("equity") - pl.col("peak")).alias("dd")
    )
)

wins = combo.filter(pl.col("result_points") > 0)
losses = combo.filter(pl.col("result_points") <= 0)

gross_win = wins["result_points"].sum() if wins.height else 0
gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

summary = pl.DataFrame([{
    "trades": combo.height,
    "wins": wins.height,
    "losses": losses.height,
    "winrate": round(100 * wins.height / combo.height, 2),
    "net_points": round(combo["result_points"].sum(), 2),
    "avg_trade": round(combo["result_points"].mean(), 2),
    "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    "max_dd": round(equity["dd"].min(), 2),
}])

yearly = (
    combo.group_by("year")
    .agg(
        pl.len().alias("trades"),
        (pl.col("result_points") > 0).sum().alias("wins"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .with_columns(
        (100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate")
    )
    .sort("year")
)

summary.write_csv(SUMMARY_OUT)
yearly.write_csv(YEARLY_OUT)

print(summary)
print(yearly)

print(f"\nSaved: {SUMMARY_OUT}")
print(f"Saved: {YEARLY_OUT}")
