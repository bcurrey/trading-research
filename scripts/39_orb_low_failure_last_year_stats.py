import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
FILE = DATA_DIR / "execution_models" / "orb_confirmed_failure_trades.csv"

df = pl.read_csv(FILE, try_parse_dates=True)

# Data ends around 2026-05-07, so past year = 2025-05-08 forward
df = df.filter(
    (pl.col("trade_date_ct") >= pl.date(2025, 5, 8)) &
    (pl.col("event_type") == "failed_or_low_breakdown") &
    (pl.col("direction") == "long") &
    (pl.col("target_name") == "or_mid")
)

summary = df.select([
    pl.len().alias("trades"),
    (pl.col("status") == "target").mean().alias("win_rate"),
    pl.col("points").mean().alias("avg_points"),
    pl.col("points").sum().alias("total_points"),
    pl.col("r").mean().alias("avg_r"),
    pl.col("r").sum().alias("total_r"),
    pl.col("risk_points").mean().alias("avg_risk"),
])

summary = summary.with_columns([
    (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
    pl.col("avg_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_r").round(3),
    pl.col("total_r").round(2),
    pl.col("avg_risk").round(2),
])

print("\nLAST 12 MONTHS: Failed OR Low Breakdown Long → OR Mid Target")
print(summary)

print("\nBy year/month:")
print(
    df.with_columns([
        pl.col("trade_date_ct").dt.strftime("%Y-%m").alias("year_month")
    ])
    .group_by("year_month")
    .agg([
        pl.len().alias("trades"),
        (pl.col("status") == "target").mean().alias("win_rate"),
        pl.col("points").sum().alias("month_points"),
        pl.col("r").sum().alias("month_r"),
    ])
    .with_columns([
        (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
        pl.col("month_points").round(2),
        pl.col("month_r").round(2),
    ])
    .sort("year_month")
)

print("\nOutcome counts:")
print(
    df.group_by("status")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)