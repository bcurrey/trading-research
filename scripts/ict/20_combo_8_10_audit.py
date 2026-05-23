# 20_combo_8_10_audit.py
# Audit/export combo_8_10 model

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
OUTDIR = ROOT / r"research_outputs\combo_8_10_audit"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(INPUT, try_parse_dates=True)

combo = (
    df.filter(
        (pl.col("hour").is_in([8, 10])) &
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20) &
        (pl.col("mae_points") > -10)
    )
    .sort("entry_time")
)

combo = combo.with_columns(
    pl.col("entry_time").dt.weekday().alias("day_of_week")
)

equity = (
    combo.with_columns(pl.col("result_points").cum_sum().alias("equity_points"))
    .with_columns(pl.col("equity_points").cum_max().alias("peak_points"))
    .with_columns((pl.col("equity_points") - pl.col("peak_points")).alias("drawdown_points"))
)

wins = combo.filter(pl.col("result_points") > 0)
losses = combo.filter(pl.col("result_points") <= 0)

gross_win = wins["result_points"].sum() if wins.height else 0
gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

summary = pl.DataFrame([{
    "model": "combo_8_10",
    "first_trade": str(combo["entry_time"].min()) if combo.height else None,
    "last_trade": str(combo["entry_time"].max()) if combo.height else None,
    "trades": combo.height,
    "wins": wins.height,
    "losses": losses.height,
    "winrate": round(100 * wins.height / combo.height, 2) if combo.height else 0,
    "net_points": round(combo["result_points"].sum(), 2) if combo.height else 0,
    "avg_trade": round(combo["result_points"].mean(), 2) if combo.height else 0,
    "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    "max_dd": round(equity["drawdown_points"].min(), 2) if equity.height else 0,
    "avg_risk": round(combo["risk_points"].mean(), 2) if combo.height else 0,
    "avg_mfe": round(combo["mfe_points"].mean(), 2) if combo.height else 0,
    "avg_mae": round(combo["mae_points"].mean(), 2) if combo.height else 0,
}])

def grouped(by_col: str):
    return (
        combo.group_by(by_col)
        .agg(
            pl.len().alias("trades"),
            (pl.col("result_points") > 0).sum().alias("wins"),
            (pl.col("result_points") <= 0).sum().alias("losses"),
            pl.col("result_points").sum().round(2).alias("net_points"),
            pl.col("result_points").mean().round(2).alias("avg_trade"),
            pl.col("risk_points").mean().round(2).alias("avg_risk"),
            pl.col("mfe_points").mean().round(2).alias("avg_mfe"),
            pl.col("mae_points").mean().round(2).alias("avg_mae"),
        )
        .with_columns((100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"))
        .sort(by_col)
    )

yearly = grouped("year")
monthly = grouped("month")
dow = grouped("day_of_week")
hour = grouped("hour")

combo.write_csv(OUTDIR / "combo_8_10_trades.csv")
summary.write_csv(OUTDIR / "combo_8_10_summary.csv")
yearly.write_csv(OUTDIR / "combo_8_10_yearly.csv")
monthly.write_csv(OUTDIR / "combo_8_10_monthly.csv")
dow.write_csv(OUTDIR / "combo_8_10_day_of_week.csv")
hour.write_csv(OUTDIR / "combo_8_10_hour.csv")
losses.write_csv(OUTDIR / "combo_8_10_losers.csv")
wins.write_csv(OUTDIR / "combo_8_10_winners.csv")
equity.write_csv(OUTDIR / "combo_8_10_equity.csv")

print("\nSUMMARY")
print(summary)

print("\nYEARLY")
print(yearly)

print("\nHOUR")
print(hour)

print("\nDAY OF WEEK")
print(dow)

print("\nLOSERS")
if losses.height:
    print(losses.select(["entry_time", "hour", "result_points", "risk_points", "mfe_points", "mae_points"]))
else:
    print("No losers")

print(f"\nSaved folder: {OUTDIR}")
