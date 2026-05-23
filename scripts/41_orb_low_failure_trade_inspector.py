import polars as pl
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "execution_models"
OUT_DIR.mkdir(exist_ok=True)

print("\nORB LOW FAILURE TRADE INSPECTOR")
print("Started:", datetime.now())

TRADES_FILE = OUT_DIR / "orb_confirmed_failure_trades.csv"
RANKED_FILE = OUT_DIR / "orb_low_failure_large_target_realistic.csv"

trades = pl.read_csv(TRADES_FILE, try_parse_dates=True)
ranked = pl.read_csv(RANKED_FILE)

print("\nTop realistic filters:")
print(
    ranked.select([
        "filter_type",
        "target_name",
        "filter",
        "trades",
        "target_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "positive_month_rate_pct",
        "worst_month_points",
        "execution_score",
    ]).head(20)
)

# ---------------------------------------------------
# Pick the top candidate from script 40
# ---------------------------------------------------

top = ranked.row(0, named=True)

target_name = top["target_name"]
filter_text = top["filter"]

print("\nSelected top candidate:")
print("Target:", target_name)
print("Filter:", filter_text)

# ---------------------------------------------------
# Focus same universe:
# Last 12 months
# Failed OR low breakdown long
# Risk 8-35
# Selected target
# ---------------------------------------------------

df = trades.filter(
    (pl.col("trade_date_ct") >= pl.date(2025, 5, 8)) &
    (pl.col("event_type") == "failed_or_low_breakdown") &
    (pl.col("direction") == "long") &
    (pl.col("target_name") == target_name) &
    (pl.col("risk_points") >= 8) &
    (pl.col("risk_points") <= 35)
)

# ---------------------------------------------------
# Apply top filter text dynamically
# Format:
# col=value | col=value | target=fixed_60
# ---------------------------------------------------

parts = [p.strip() for p in filter_text.split("|")]

for part in parts:
    if "=" not in part:
        continue

    col, val = part.split("=", 1)
    col = col.strip()
    val = val.strip()

    if col == "target":
        continue

    if col not in df.columns:
        print(f"Skipped missing filter column: {col}")
        continue

    df = df.filter(pl.col(col).cast(pl.Utf8) == val)

print("\nFiltered trades:", df.height)

if df.height == 0:
    raise SystemExit("No trades after applying filter.")

# ---------------------------------------------------
# Add month
# ---------------------------------------------------

df = df.with_columns([
    pl.col("trade_date_ct").dt.strftime("%Y-%m").alias("year_month")
])

# ---------------------------------------------------
# Main stats
# ---------------------------------------------------

summary = df.select([
    pl.len().alias("trades"),
    (pl.col("status") == "target").mean().alias("target_rate"),
    pl.col("points").mean().alias("avg_points"),
    pl.col("points").median().alias("median_points"),
    pl.col("points").sum().alias("total_points"),
    pl.col("r").mean().alias("avg_r"),
    pl.col("r").median().alias("median_r"),
    pl.col("r").sum().alias("total_r"),
    pl.col("risk_points").mean().alias("avg_risk"),
    pl.col("risk_points").median().alias("median_risk"),
])

summary = summary.with_columns([
    (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
    pl.col("avg_points").round(2),
    pl.col("median_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_r").round(3),
    pl.col("median_r").round(3),
    pl.col("total_r").round(2),
    pl.col("avg_risk").round(2),
    pl.col("median_risk").round(2),
])

print("\nTOP CANDIDATE SUMMARY:")
print(summary)

print("\nBy month:")
by_month = (
    df.group_by("year_month")
    .agg([
        pl.len().alias("trades"),
        (pl.col("status") == "target").mean().alias("target_rate"),
        pl.col("points").sum().alias("month_points"),
        pl.col("r").sum().alias("month_r"),
    ])
    .with_columns([
        (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
        pl.col("month_points").round(2),
        pl.col("month_r").round(2),
    ])
    .sort("year_month")
)

print(by_month)

print("\nOutcome counts:")
print(
    df.group_by("status")
    .agg(pl.len().alias("count"))
    .sort("count", descending=True)
)

# ---------------------------------------------------
# Winner / loser inspection table
# ---------------------------------------------------

inspection_cols = [
    "trade_date_ct",
    "entry_time",
    "event_type",
    "direction",
    "target_name",
    "status",
    "points",
    "r",
    "entry",
    "stop",
    "target_price",
    "risk_points",
    "or_high",
    "or_low",
    "or_mid",
    "or_range",
    "sweep_price",
    "close_back_inside_price",
    "entry_time_bucket",
    "or_vol_regime",
    "overnight_vol_regime",
    "atr_vol_regime",
    "volume_bucket",
    "body_bucket",
    "risk_bucket",
    "vwap_side",
    "gap_bucket",
    "overnight_position",
    "prior_day_position",
]

inspection_cols = [c for c in inspection_cols if c in df.columns]

winners = (
    df.filter(pl.col("points") > 0)
    .sort("points", descending=True)
    .select(inspection_cols)
    .head(30)
)

losers = (
    df.filter(pl.col("points") <= 0)
    .sort("points")
    .select(inspection_cols)
    .head(30)
)

print("\nTop winners:")
print(winners)

print("\nWorst losers:")
print(losers)

# ---------------------------------------------------
# Save focused files
# ---------------------------------------------------

selected_file = OUT_DIR / "orb_low_failure_selected_candidate_trades.csv"
winners_file = OUT_DIR / "orb_low_failure_selected_candidate_top_winners.csv"
losers_file = OUT_DIR / "orb_low_failure_selected_candidate_worst_losers.csv"
month_file = OUT_DIR / "orb_low_failure_selected_candidate_by_month.csv"
summary_file = OUT_DIR / "orb_low_failure_selected_candidate_summary.csv"

df.write_csv(selected_file)
winners.write_csv(winners_file)
losers.write_csv(losers_file)
by_month.write_csv(month_file)
summary.write_csv(summary_file)

print("\nSaved:")
print(selected_file)
print(winners_file)
print(losers_file)
print(month_file)
print(summary_file)

print("\nFinished:", datetime.now())
print("DONE.")