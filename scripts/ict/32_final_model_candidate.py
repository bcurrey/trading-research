# 32_final_model_candidate.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")
TRADES = ROOT / r"research_outputs\scaling_model_search\scaling_model_best_trades.csv"
OUTDIR = ROOT / r"research_outputs\final_model_candidate"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(TRADES, try_parse_dates=True)

models = {
    "current_49": pl.lit(True),
    "pdh_lt_minus78": pl.col("dist_pdh") < -78,
    "pdh_dol": (pl.col("dist_pdh") < -78) & (pl.col("dist_dol") < 25),
    "pdh_vwap": (pl.col("dist_pdh") < -78) & (pl.col("dist_vwap") < -20),
    "pdh_vwap_relvol": (pl.col("dist_pdh") < -78) & (pl.col("dist_vwap") < -20) & (pl.col("rel_vol_20") < 2.2),
    "balanced": (pl.col("dist_pdh") < -78) & (pl.col("dist_vwap") < -20) & (pl.col("dist_dol") < 25) & (pl.col("rel_vol_20") < 2.2),
}

def summarize(x, label):
    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)
    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0
    eq = (
        x.sort("entry_time")
        .with_columns(pl.col("result_points").cum_sum().alias("equity"))
        .with_columns(pl.col("equity").cum_max().alias("peak"))
        .with_columns((pl.col("equity") - pl.col("peak")).alias("dd"))
    )
    return {
        "model": label,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2) if x.height else 0,
        "net_points": round(x["result_points"].sum(), 2) if x.height else 0,
        "avg_trade": round(x["result_points"].mean(), 2) if x.height else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": round(eq["dd"].min(), 2) if x.height else 0,
    }

rows = []
frames = {}

for name, filt in models.items():
    x = df.filter(filt).sort("entry_time")
    frames[name] = x
    rows.append(summarize(x, name))

summary = pl.DataFrame(rows).sort(["profit_factor", "net_points"], descending=[True, True])

best_name = summary.filter(pl.col("trades") >= 20)["model"][0]
best = frames[best_name]

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

summary.write_csv(OUTDIR / "final_model_summary.csv")
best.write_csv(OUTDIR / "final_model_best_trades.csv")
yearly.write_csv(OUTDIR / "final_model_yearly.csv")
best.filter(pl.col("result_points") <= 0).write_csv(OUTDIR / "final_model_losers.csv")

print(summary)
print("\nBEST:", best_name)
print(yearly)
print("\nSaved folder:", OUTDIR)
