import polars as pl
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "execution_models"
OUT_DIR.mkdir(exist_ok=True)

print("\nORB FAILED AUCTION V1 MODEL")
print("Started:", datetime.now())

FILE = OUT_DIR / "orb_confirmed_failure_trades.csv"

df = pl.read_csv(FILE, try_parse_dates=True)

# ---------------------------------------------------
# Strategy V1:
# Failed OR Low Breakdown Reclaim Long
# Large target only
# ---------------------------------------------------

df = df.filter(
    (pl.col("event_type") == "failed_or_low_breakdown") &
    (pl.col("direction") == "long") &
    (pl.col("target_name") == "fixed_60")
)

# ---------------------------------------------------
# Add derived features
# ---------------------------------------------------

df = df.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),
    pl.col("trade_date_ct").dt.month().alias("month"),
    pl.col("trade_date_ct").dt.strftime("%Y-%m").alias("year_month"),

    (pl.col("or_low") - pl.col("sweep_price")).alias("sweep_depth_points"),
    (pl.col("entry") - pl.col("or_low")).alias("reclaim_strength_points"),
    (pl.col("risk_points") / pl.col("or_range")).alias("risk_vs_or_range"),
])

df = df.with_columns([
    pl.when(pl.col("sweep_depth_points") >= 15)
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("deep_enough_sweep"),

    pl.when(pl.col("reclaim_strength_points") >= 5)
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("strong_enough_reclaim"),

    pl.when(pl.col("minute_of_day_ct") <= 9 * 60 + 5)
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("early_fast_reclaim"),

    pl.when(pl.col("gap_from_prior_close") <= -30)
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("gap_down_30_plus"),

    pl.when(pl.col("vwap_side") == "below_vwap")
    .then(pl.lit(True))
    .otherwise(pl.lit(False))
    .alias("below_vwap_entry"),
])

# ---------------------------------------------------
# V1 rule set
# ---------------------------------------------------

v1 = df.filter(
    (pl.col("risk_points") >= 8) &
    (pl.col("risk_points") <= 35) &
    (pl.col("gap_down_30_plus")) &
    (pl.col("below_vwap_entry")) &
    (pl.col("early_fast_reclaim")) &
    (
        (pl.col("or_vol_regime").is_in(["or_high", "or_extreme"])) |
        (pl.col("overnight_vol_regime").is_in(["on_high", "on_extreme"])) |
        (pl.col("atr_vol_regime").is_in(["atr_high", "atr_extreme"]))
    )
)

print("\nV1 trades:", v1.height)

if v1.height == 0:
    raise SystemExit("No V1 trades found.")

# ---------------------------------------------------
# Stats helper
# ---------------------------------------------------

def max_drawdown(vals):
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for v in vals:
        equity += float(v)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return max_dd


def summarize(label, data):
    if data.height == 0:
        return None

    vals = data.sort("trade_date_ct").select("points").to_series().to_list()

    month_stats = (
        data
        .group_by("year_month")
        .agg([
            pl.len().alias("trades"),
            pl.col("points").sum().alias("month_points"),
            pl.col("r").sum().alias("month_r"),
        ])
        .with_columns((pl.col("month_points") > 0).alias("positive_month"))
    )

    year_stats = (
        data
        .group_by("year")
        .agg([
            pl.len().alias("trades"),
            pl.col("points").sum().alias("year_points"),
            pl.col("r").sum().alias("year_r"),
        ])
        .with_columns((pl.col("year_points") > 0).alias("positive_year"))
    )

    positive_months = month_stats.select(pl.col("positive_month").sum()).item()
    months_tested = month_stats.height
    worst_month = month_stats.select(pl.col("month_points").min()).item()

    positive_years = year_stats.select(pl.col("positive_year").sum()).item()
    years_tested = year_stats.height
    worst_year = year_stats.select(pl.col("year_points").min()).item()

    summary = data.select([
        pl.lit(label).alias("sample"),
        pl.len().alias("trades"),
        (pl.col("status") == "target").mean().alias("win_rate"),
        pl.col("points").mean().alias("avg_points"),
        pl.col("points").median().alias("median_points"),
        pl.col("points").sum().alias("total_points"),
        pl.col("r").mean().alias("avg_r"),
        pl.col("r").median().alias("median_r"),
        pl.col("r").sum().alias("total_r"),
        pl.col("risk_points").mean().alias("avg_risk"),
        pl.col("risk_points").median().alias("median_risk"),
        pl.col("sweep_depth_points").mean().alias("avg_sweep_depth"),
        pl.col("reclaim_strength_points").mean().alias("avg_reclaim_strength"),
    ])

    summary = summary.with_columns([
        (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
        pl.col("avg_points").round(2),
        pl.col("median_points").round(2),
        pl.col("total_points").round(2),
        pl.col("avg_r").round(3),
        pl.col("median_r").round(3),
        pl.col("total_r").round(2),
        pl.col("avg_risk").round(2),
        pl.col("median_risk").round(2),
        pl.col("avg_sweep_depth").round(2),
        pl.col("avg_reclaim_strength").round(2),

        pl.lit(positive_months).alias("positive_months"),
        pl.lit(months_tested).alias("months_tested"),
        pl.lit(round(positive_months / months_tested * 100, 2) if months_tested else None).alias("positive_month_rate_pct"),
        pl.lit(round(worst_month, 2) if worst_month is not None else None).alias("worst_month_points"),

        pl.lit(positive_years).alias("positive_years"),
        pl.lit(years_tested).alias("years_tested"),
        pl.lit(round(positive_years / years_tested * 100, 2) if years_tested else None).alias("positive_year_rate_pct"),
        pl.lit(round(worst_year, 2) if worst_year is not None else None).alias("worst_year_points"),

        pl.lit(round(max_drawdown(vals), 2)).alias("max_drawdown_points"),
    ])

    return summary, month_stats, year_stats

# ---------------------------------------------------
# Sample splits
# ---------------------------------------------------

max_date = v1.select(pl.col("trade_date_ct").max()).item()
last_12m_start = max_date.replace(year=max_date.year - 1)
last_3y_start = max_date.replace(year=max_date.year - 3)

splits = [
    ("full_sample", v1),
    ("last_3_years", v1.filter(pl.col("trade_date_ct") >= last_3y_start)),
    ("last_12_months", v1.filter(pl.col("trade_date_ct") >= last_12m_start)),
]

summary_frames = []
month_frames = []
year_frames = []

for label, data in splits:
    if data.height == 0:
        continue

    s, m, y = summarize(label, data)

    summary_frames.append(s)
    month_frames.append(m.with_columns(pl.lit(label).alias("sample")))
    year_frames.append(y.with_columns(pl.lit(label).alias("sample")))

summary_df = pl.concat(summary_frames)
month_df = pl.concat(month_frames)
year_df = pl.concat(year_frames)

print("\nV1 SUMMARY:")
print(summary_df)

print("\nV1 MONTHLY:")
print(month_df.sort(["sample", "year_month"]))

print("\nV1 YEARLY:")
print(year_df.sort(["sample", "year"]))

# ---------------------------------------------------
# Trade list for review
# ---------------------------------------------------

review_cols = [
    "trade_date_ct",
    "entry_time",
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
    "sweep_depth_points",
    "reclaim_strength_points",
    "gap_from_prior_close",
    "dist_vwap_day",
    "or_vol_regime",
    "overnight_vol_regime",
    "atr_vol_regime",
    "volume_bucket",
    "body_bucket",
    "vwap_side",
    "overnight_position",
    "prior_day_position",
]

review_cols = [c for c in review_cols if c in v1.columns]

print("\nV1 TRADE REVIEW:")
print(
    v1.sort("trade_date_ct")
    .select(review_cols)
    .head(100)
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

trades_file = OUT_DIR / "orb_failed_auction_v1_trades.csv"
summary_file = OUT_DIR / "orb_failed_auction_v1_summary.csv"
month_file = OUT_DIR / "orb_failed_auction_v1_monthly.csv"
year_file = OUT_DIR / "orb_failed_auction_v1_yearly.csv"

v1.write_csv(trades_file)
summary_df.write_csv(summary_file)
month_df.write_csv(month_file)
year_df.write_csv(year_file)

print("\nSaved:")
print(trades_file)
print(summary_file)
print(month_file)
print(year_file)

print("\nFinished:", datetime.now())
print("DONE.")