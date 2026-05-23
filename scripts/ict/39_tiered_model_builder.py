# 39_tiered_model_builder.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\two_stage_signal_backtest\two_stage_backtest_trades.csv"
OUTDIR = ROOT / r"research_outputs\tiered_model_builder"

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(INPUT, try_parse_dates=True)

# -----------------------------
# TIERS
# -----------------------------

sniper = df.filter(
    (pl.col("dist_pdh") < -78) &
    (pl.col("dist_vwap") < -20) &
    (pl.col("stage1_lower_wick_pct") > 0.20) &
    (pl.col("stage2_lower_wick_pct") < 0.18)
)

elite = df.filter(
    (pl.col("dist_pdh") < -55) &
    (pl.col("dist_vwap") < -12) &
    (pl.col("stage1_lower_wick_pct") > 0.18) &
    (pl.col("stage2_lower_wick_pct") < 0.22) &
    (pl.col("stage2_body_pct").is_between(0.40, 0.90))
)

a_model = df.filter(
    (pl.col("dist_pdh") < -35) &
    (pl.col("dist_vwap") < -5) &
    (pl.col("stage1_lower_wick_pct") > 0.15)
)

b_model = df.filter(
    (pl.col("dist_pdh") < -15)
)

tiers = {
    "SNIPER": sniper,
    "ELITE": elite,
    "A": a_model,
    "B": b_model,
}

# -----------------------------
# SUMMARY
# -----------------------------

def summarize(x, label):

    if x.height == 0:
        return None

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
        "tier": label,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": round(eq["dd"].min(), 2),
    }

summary_rows = []

for name, frame in tiers.items():
    s = summarize(frame, name)
    if s:
        summary_rows.append(s)

summary = pl.DataFrame(summary_rows).sort(
    ["profit_factor", "avg_trade"],
    descending=[True, True]
)

# -----------------------------
# YEARLY
# -----------------------------

yearly_rows = []

for name, frame in tiers.items():

    if frame.height == 0:
        continue

    yearly = (
        frame.group_by("year")
        .agg(
            pl.len().alias("trades"),
            (pl.col("result_points") > 0).sum().alias("wins"),
            (pl.col("result_points") <= 0).sum().alias("losses"),
            pl.col("result_points").sum().round(2).alias("net_points"),
            pl.col("result_points").mean().round(2).alias("avg_trade"),
        )
        .with_columns([
            pl.lit(name).alias("tier"),
            (100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"),
        ])
    )

    yearly_rows.append(yearly)

yearly_df = pl.concat(yearly_rows, how="vertical")

# -----------------------------
# SAVE
# -----------------------------

summary.write_csv(OUTDIR / "tier_summary.csv")
yearly_df.write_csv(OUTDIR / "tier_yearly.csv")

for name, frame in tiers.items():
    frame.write_csv(OUTDIR / f"{name.lower()}_trades.csv")

print(summary)

print("\nYEARLY")
print(yearly_df)

print(f"\nSaved: {OUTDIR}")

