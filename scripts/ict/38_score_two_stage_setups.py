# 38_score_two_stage_setups.py

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
OUTDIR = ROOT / r"research_outputs\two_stage_scoring"

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(INPUT, try_parse_dates=True)

df = df.with_columns([
    pl.when(pl.col("dist_pdh") < -150).then(3)
      .when(pl.col("dist_pdh") < -100).then(2)
      .when(pl.col("dist_pdh") < -78).then(1)
      .otherwise(0)
      .alias("score_pdh_extension"),

    pl.when(pl.col("dist_vwap") < -75).then(3)
      .when(pl.col("dist_vwap") < -50).then(2)
      .when(pl.col("dist_vwap") < -20).then(1)
      .otherwise(0)
      .alias("score_vwap_extension"),

    pl.when(pl.col("stage1_lower_wick_pct") > 0.45).then(3)
      .when(pl.col("stage1_lower_wick_pct") > 0.30).then(2)
      .when(pl.col("stage1_lower_wick_pct") > 0.20).then(1)
      .otherwise(0)
      .alias("score_stage1_reclaim"),

    pl.when(pl.col("stage2_lower_wick_pct") < 0.08).then(2)
      .when(pl.col("stage2_lower_wick_pct") < 0.15).then(1)
      .otherwise(0)
      .alias("score_stage2_control"),

    pl.when(pl.col("stage2_body_pct").is_between(0.50, 0.80)).then(2)
      .when(pl.col("stage2_body_pct").is_between(0.40, 0.90)).then(1)
      .otherwise(0)
      .alias("score_stage2_body"),

    pl.when(pl.col("hour").is_in([8, 10])).then(2)
      .when(pl.col("hour").is_between(8, 11)).then(1)
      .otherwise(0)
      .alias("score_time"),
])

df = df.with_columns(
    (
        pl.col("score_pdh_extension") +
        pl.col("score_vwap_extension") +
        pl.col("score_stage1_reclaim") +
        pl.col("score_stage2_control") +
        pl.col("score_stage2_body") +
        pl.col("score_time")
    ).alias("quality_score")
)

df = df.with_columns(
    pl.when(pl.col("quality_score") >= 10).then(pl.lit("A+"))
      .when(pl.col("quality_score") >= 8).then(pl.lit("A"))
      .when(pl.col("quality_score") >= 6).then(pl.lit("B"))
      .otherwise(pl.lit("C"))
      .alias("tier")
)

def summarize(x, label):
    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)
    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

    return {
        "group": label,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2) if x.height else 0,
        "net_points": round(x["result_points"].sum(), 2) if x.height else 0,
        "avg_trade": round(x["result_points"].mean(), 2) if x.height else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    }

summary_rows = []

for tier in ["A+", "A", "B", "C"]:
    summary_rows.append(summarize(df.filter(pl.col("tier") == tier), tier))

for score in sorted(df["quality_score"].unique().to_list()):
    summary_rows.append(summarize(df.filter(pl.col("quality_score") == score), f"score_{score}"))

summary = pl.DataFrame(summary_rows)

yearly = (
    df.group_by(["tier", "year"])
    .agg(
        pl.len().alias("trades"),
        (pl.col("result_points") > 0).sum().alias("wins"),
        (pl.col("result_points") <= 0).sum().alias("losses"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .with_columns((100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"))
    .sort(["tier", "year"])
)

df.write_csv(OUTDIR / "two_stage_scored_trades.csv")
summary.write_csv(OUTDIR / "two_stage_score_summary.csv")
yearly.write_csv(OUTDIR / "two_stage_score_yearly.csv")

print(summary)
print(yearly)

print(f"\nSaved: {OUTDIR}")
