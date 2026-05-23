from pathlib import Path
import polars as pl

OUT_DIR = Path(r"D:\TradingResearch\research_outputs")
SCRIPT_NAME = "31_orb_top4_production_audit"

TRADES_IN = OUT_DIR / "30_orb_retest_quality_refinement_trades.csv"

SUMMARY_OUT = OUT_DIR / f"{SCRIPT_NAME}_summary.csv"
YEARLY_OUT = OUT_DIR / f"{SCRIPT_NAME}_yearly.csv"
MONTHLY_OUT = OUT_DIR / f"{SCRIPT_NAME}_monthly.csv"
DIRECTION_OUT = OUT_DIR / f"{SCRIPT_NAME}_direction.csv"
RECENT_OUT = OUT_DIR / f"{SCRIPT_NAME}_recent_2024_2026.csv"
ROLLING_OUT = OUT_DIR / f"{SCRIPT_NAME}_rolling_50.csv"
DECISION_OUT = OUT_DIR / f"{SCRIPT_NAME}_decision_table.csv"
PINE_SIGNALS_OUT = OUT_DIR / f"{SCRIPT_NAME}_pine_validation_trades.csv"

TOP4 = [
    "PROD_STACK_STRICT",
    "ATR_8_PLUS",
    "VWAP_EMA20_ALIGN",
    "VWAP_ALIGN",
]

print("Loading script 30 trades...")
df = pl.read_csv(TRADES_IN)

missing = [m for m in TOP4 if m not in df["model"].unique().to_list()]
if missing:
    raise ValueError(f"Missing expected models in trades file: {missing}")

df = (
    df.filter(pl.col("model").is_in(TOP4))
    .with_columns([
        pl.col("trade_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
        pl.col("entry_time").str.strptime(pl.Datetime, strict=False).alias("entry_dt"),
        pl.col("exit_time").str.strptime(pl.Datetime, strict=False).alias("exit_dt"),
    ])
    .sort(["model", "date", "entry_time"])
)

df = df.with_columns([
    pl.col("pnl_points").cum_sum().over("model").alias("equity"),
])

df = df.with_columns([
    pl.col("equity").cum_max().over("model").alias("equity_high"),
])

df = df.with_columns([
    (pl.col("equity") - pl.col("equity_high")).alias("drawdown"),
    (pl.col("pnl_points") > 0).cast(pl.Int8).alias("is_win"),
    (pl.col("pnl_points") < 0).cast(pl.Int8).alias("is_loss"),
])

# Consecutive loss helper
pdf = df.to_pandas()
rows = []

for model, g in pdf.groupby("model", sort=False):
    streak = 0
    max_streak = 0

    for _, r in g.iterrows():
        if r["pnl_points"] < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    rows.append({"model": model, "max_consecutive_losses": max_streak})

loss_streak_df = pl.DataFrame(rows)

summary = (
    df.group_by("model")
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.col("drawdown").min().round(2).alias("max_dd_points"),
        pl.col("month").n_unique().alias("active_months"),
        pl.col("year").n_unique().alias("active_years"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_loss"),
        pl.col("pnl_points").std().round(2).alias("pnl_std"),
    ])
    .with_columns([
        (pl.col("gross_win") / pl.col("gross_loss")).round(3).alias("profit_factor"),
        (pl.col("trades") / pl.col("active_months")).round(2).alias("trades_per_month"),
        (pl.col("net_points") / (-pl.col("max_dd_points"))).round(2).alias("net_to_dd"),
    ])
    .join(loss_streak_df, on="model", how="left")
)

yearly = (
    df.group_by(["model", "year"])
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_loss"),
    ])
    .with_columns([
        pl.when(pl.col("gross_loss") > 0)
        .then((pl.col("gross_win") / pl.col("gross_loss")).round(3))
        .otherwise(None)
        .alias("profit_factor")
    ])
    .sort(["model", "year"])
)

monthly = (
    df.group_by(["model", "month"])
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
    ])
    .sort(["model", "month"])
)

direction = (
    df.group_by(["model", "direction"])
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_loss"),
    ])
    .with_columns([
        pl.when(pl.col("gross_loss") > 0)
        .then((pl.col("gross_win") / pl.col("gross_loss")).round(3))
        .otherwise(None)
        .alias("profit_factor")
    ])
    .sort(["model", "direction"])
)

recent = (
    df.filter(pl.col("year").is_in(["2024", "2025", "2026"]))
    .group_by("model")
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_loss"),
    ])
    .with_columns([
        pl.when(pl.col("gross_loss") > 0)
        .then((pl.col("gross_win") / pl.col("gross_loss")).round(3))
        .otherwise(None)
        .alias("profit_factor")
    ])
    .sort(["profit_factor", "net_points"], descending=True)
)

rolling = (
    df.with_columns([
        pl.col("pnl_points").rolling_sum(window_size=50).over("model").alias("rolling_50_net"),
        (pl.col("pnl_points") > 0).cast(pl.Float64).rolling_mean(window_size=50).over("model").alias("rolling_50_wr"),
    ])
    .select([
        "model", "trade_date", "entry_time", "direction", "pnl_points",
        "rolling_50_net", "rolling_50_wr",
    ])
)

rolling_summary = (
    rolling.group_by("model")
    .agg([
        pl.col("rolling_50_net").min().round(2).alias("worst_rolling_50_net"),
        pl.col("rolling_50_net").mean().round(2).alias("avg_rolling_50_net"),
        pl.col("rolling_50_wr").min().round(4).alias("worst_rolling_50_wr"),
        pl.col("rolling_50_wr").mean().round(4).alias("avg_rolling_50_wr"),
    ])
)

bad_months = (
    monthly.group_by("model")
    .agg([
        (pl.col("net_points") < 0).sum().alias("negative_months"),
        pl.col("net_points").min().round(2).alias("worst_month_points"),
        pl.col("net_points").mean().round(2).alias("avg_month_points"),
    ])
)

positive_years = (
    yearly.group_by("model")
    .agg([
        (pl.col("net_points") > 0).sum().alias("positive_years"),
        pl.col("net_points").min().round(2).alias("worst_year_points"),
    ])
)

decision = (
    summary
    .join(recent.select([
        "model",
        pl.col("trades").alias("recent_trades"),
        pl.col("winrate").alias("recent_winrate"),
        pl.col("net_points").alias("recent_net_points"),
        pl.col("avg_trade").alias("recent_avg_trade"),
        pl.col("profit_factor").alias("recent_profit_factor"),
    ]), on="model", how="left")
    .join(rolling_summary, on="model", how="left")
    .join(bad_months, on="model", how="left")
    .join(positive_years, on="model", how="left")
    .with_columns([
        (
            (pl.col("profit_factor") * 25)
            + (pl.col("recent_profit_factor") * 25)
            + (pl.col("winrate") * 20)
            + (pl.col("recent_winrate") * 20)
            + (pl.col("net_to_dd") * 5)
            - (pl.col("negative_months") * 0.75)
            - (pl.col("max_consecutive_losses") * 1.0)
        ).round(2).alias("production_score")
    ])
    .with_columns([
        pl.when(
            (pl.col("profit_factor") >= 1.70) &
            (pl.col("recent_profit_factor") >= 1.40) &
            (pl.col("winrate") >= 0.62) &
            (pl.col("positive_years") >= 7) &
            (pl.col("worst_rolling_50_net") > -250)
        )
        .then(pl.lit("BUILD_INDICATOR_FIRST"))
        .when(
            (pl.col("profit_factor") >= 1.60) &
            (pl.col("winrate") >= 0.60)
        )
        .then(pl.lit("KEEP_AS_BACKUP"))
        .otherwise(pl.lit("REJECT_FOR_PRODUCTION"))
        .alias("decision")
    ])
    .sort(["decision", "production_score"], descending=[False, True])
)

# Export last 150 trades per model for visual validation.
pine_validation = (
    df.sort(["model", "date", "entry_time"])
    .with_columns([
        pl.int_range(pl.len()).over("model").alias("model_trade_index"),
        pl.len().over("model").alias("model_trade_count"),
    ])
    .filter(pl.col("model_trade_index") >= (pl.col("model_trade_count") - 150))
    .select([
        "model", "trade_date", "direction", "break_time", "entry_time", "exit_time",
        "or_high", "or_low", "or_range", "entry", "stop_price", "target_price",
        "outcome", "pnl_points", "break_above_vwap", "break_below_vwap",
        "break_above_ema20", "break_below_ema20", "break_displacement",
        "rel_vol_20", "atr_14"
    ])
)

summary.write_csv(SUMMARY_OUT)
yearly.write_csv(YEARLY_OUT)
monthly.write_csv(MONTHLY_OUT)
direction.write_csv(DIRECTION_OUT)
recent.write_csv(RECENT_OUT)
rolling.write_csv(ROLLING_OUT)
decision.write_csv(DECISION_OUT)
pine_validation.write_csv(PINE_SIGNALS_OUT)

print("\nSUMMARY")
print(summary.sort(["profit_factor", "net_points"], descending=True))

print("\nRECENT 2024-2026")
print(recent)

print("\nDIRECTION")
print(direction)

print("\nDECISION TABLE")
print(decision.select([
    "model", "trades", "winrate", "profit_factor", "net_points",
    "max_dd_points", "max_consecutive_losses", "recent_profit_factor",
    "negative_months", "positive_years", "worst_rolling_50_net",
    "production_score", "decision"
]))

print(f"\nSaved summary: {SUMMARY_OUT}")
print(f"Saved yearly: {YEARLY_OUT}")
print(f"Saved monthly: {MONTHLY_OUT}")
print(f"Saved direction: {DIRECTION_OUT}")
print(f"Saved recent: {RECENT_OUT}")
print(f"Saved rolling: {ROLLING_OUT}")
print(f"Saved decision: {DECISION_OUT}")
print(f"Saved Pine validation trades: {PINE_SIGNALS_OUT}")
