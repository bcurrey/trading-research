import polars as pl
from pathlib import Path
from itertools import combinations
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "execution_models"
OUT_DIR.mkdir(exist_ok=True)

print("\nORB LOW FAILURE LARGE TARGET FILTER TEST")
print("Started:", datetime.now())

FILE = DATA_DIR / "execution_models" / "orb_confirmed_failure_trades.csv"

df = pl.read_csv(FILE, try_parse_dates=True)

# ---------------------------------------------------
# Focus only the setup we care about:
# Failed OR low breakdown reclaim LONG
# Last 12 months only
# Larger target only
# ---------------------------------------------------

df = df.filter(
    (pl.col("trade_date_ct") >= pl.date(2025, 5, 8)) &
    (pl.col("event_type") == "failed_or_low_breakdown") &
    (pl.col("direction") == "long") &
    (pl.col("target_name").is_in(["fixed_25", "fixed_40", "fixed_60", "opposite_or"]))
)

print("\nRows loaded:", df.height)

if df.height == 0:
    raise SystemExit("No rows found.")

# ---------------------------------------------------
# Keep realistic trade constraints
# ---------------------------------------------------

df = df.filter(
    (pl.col("risk_points") >= 8) &
    (pl.col("risk_points") <= 35)
)

print("Rows after risk filter 8-35:", df.height)

# ---------------------------------------------------
# Add month/year
# ---------------------------------------------------

df = df.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),
    pl.col("trade_date_ct").dt.month().alias("month"),
    pl.col("trade_date_ct").dt.strftime("%Y-%m").alias("year_month"),
])

# ---------------------------------------------------
# Candidate context filters
# These columns should exist from script 38 output.
# ---------------------------------------------------

context_cols = [
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

# Remove columns that may not exist
context_cols = [c for c in context_cols if c in df.columns]

print("\nContext columns:")
print(context_cols)

# ---------------------------------------------------
# Helper summary
# ---------------------------------------------------

def summarize(label, data):
    if data.height < 8:
        return None

    month_stats = (
        data
        .group_by("year_month")
        .agg([
            pl.len().alias("month_trades"),
            pl.col("points").sum().alias("month_points"),
            pl.col("r").sum().alias("month_r"),
        ])
        .with_columns([
            (pl.col("month_points") > 0).alias("positive_month")
        ])
    )

    positive_months = month_stats.select(pl.col("positive_month").sum()).item()
    months_tested = month_stats.height
    worst_month_points = month_stats.select(pl.col("month_points").min()).item()

    result = data.select([
        pl.lit(label).alias("filter"),
        pl.len().alias("trades"),
        (pl.col("status") == "target").mean().alias("target_rate"),
        pl.col("points").mean().alias("avg_points"),
        pl.col("points").sum().alias("total_points"),
        pl.col("r").mean().alias("avg_r"),
        pl.col("r").sum().alias("total_r"),
        pl.col("risk_points").mean().alias("avg_risk"),
    ]).to_dicts()[0]

    result.update({
        "positive_months": positive_months,
        "months_tested": months_tested,
        "positive_month_rate": positive_months / months_tested if months_tested else 0,
        "worst_month_points": worst_month_points,
    })

    return result

results = []

# ---------------------------------------------------
# Baseline by target
# ---------------------------------------------------

for target in df.select("target_name").unique().to_series().to_list():
    x = df.filter(pl.col("target_name") == target)
    row = summarize(f"BASE | target={target}", x)
    if row:
        row["target_name"] = target
        row["filter_type"] = "base"
        results.append(row)

# ---------------------------------------------------
# Single context filters by target
# ---------------------------------------------------

for target in df.select("target_name").unique().to_series().to_list():
    base = df.filter(pl.col("target_name") == target)

    for col in context_cols:
        vals = base.select(pl.col(col).unique()).to_series().to_list()

        for val in vals:
            x = base.filter(pl.col(col) == val)
            row = summarize(f"{col}={val} | target={target}", x)
            if row:
                row["target_name"] = target
                row["filter_type"] = "single"
                results.append(row)

# ---------------------------------------------------
# Two-filter combinations by target
# ---------------------------------------------------

for target in df.select("target_name").unique().to_series().to_list():
    base = df.filter(pl.col("target_name") == target)

    for c1, c2 in combinations(context_cols, 2):
        vals1 = base.select(pl.col(c1).unique()).to_series().to_list()
        vals2 = base.select(pl.col(c2).unique()).to_series().to_list()

        for v1 in vals1:
            for v2 in vals2:
                x = base.filter(
                    (pl.col(c1) == v1) &
                    (pl.col(c2) == v2)
                )

                row = summarize(f"{c1}={v1} | {c2}={v2} | target={target}", x)

                if row:
                    row["target_name"] = target
                    row["filter_type"] = "two_filter"
                    results.append(row)

# ---------------------------------------------------
# Three-filter combinations, but require enough base count
# ---------------------------------------------------

for target in df.select("target_name").unique().to_series().to_list():
    base = df.filter(pl.col("target_name") == target)

    for c1, c2, c3 in combinations(context_cols, 3):
        vals1 = base.select(pl.col(c1).unique()).to_series().to_list()
        vals2 = base.select(pl.col(c2).unique()).to_series().to_list()
        vals3 = base.select(pl.col(c3).unique()).to_series().to_list()

        for v1 in vals1:
            for v2 in vals2:
                for v3 in vals3:
                    x = base.filter(
                        (pl.col(c1) == v1) &
                        (pl.col(c2) == v2) &
                        (pl.col(c3) == v3)
                    )

                    row = summarize(
                        f"{c1}={v1} | {c2}={v2} | {c3}={v3} | target={target}",
                        x,
                    )

                    if row:
                        row["target_name"] = target
                        row["filter_type"] = "three_filter"
                        results.append(row)

if not results:
    raise SystemExit("No results created.")

results_df = pl.DataFrame(results)

results_df = results_df.with_columns([
    (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
    (pl.col("positive_month_rate") * 100).round(2).alias("positive_month_rate_pct"),
    pl.col("avg_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_r").round(3),
    pl.col("total_r").round(2),
    pl.col("avg_risk").round(2),
    pl.col("worst_month_points").round(2),

    (
        pl.col("avg_points") *
        (pl.col("trades").clip(upper_bound=100) / 100) *
        pl.col("positive_month_rate")
    ).round(4).alias("execution_score")
])

# ---------------------------------------------------
# Rank only things that matter:
# not tiny targets, positive avg points, decent sample
# ---------------------------------------------------

ranked = (
    results_df
    .filter(
        (pl.col("trades") >= 10) &
        (pl.col("avg_points") >= 5) &
        (pl.col("total_points") > 0) &
        (pl.col("positive_month_rate") >= 0.45)
    )
    .sort("execution_score", descending=True)
)

print("\nTOP LARGE-TARGET FILTER RESULTS:")
print(
    ranked
    .select([
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
        "positive_months",
        "months_tested",
        "positive_month_rate_pct",
        "worst_month_points",
        "execution_score",
    ])
    .head(100)
)

# ---------------------------------------------------
# Also print realistic top candidates only
# ---------------------------------------------------

realistic = (
    ranked
    .filter(
        (pl.col("trades") >= 15) &
        (pl.col("avg_points") >= 8) &
        (pl.col("avg_r") > 0.20)
    )
    .sort("execution_score", descending=True)
)

print("\nREALISTIC CANDIDATES:")
print(
    realistic
    .select([
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
    ])
    .head(50)
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

all_file = OUT_DIR / "orb_low_failure_large_target_all_results.csv"
ranked_file = OUT_DIR / "orb_low_failure_large_target_ranked.csv"
realistic_file = OUT_DIR / "orb_low_failure_large_target_realistic.csv"

results_df.write_csv(all_file)
ranked.write_csv(ranked_file)
realistic.write_csv(realistic_file)

print("\nSaved:")
print(all_file)
print(ranked_file)
print(realistic_file)

print("\nFinished:", datetime.now())
print("DONE.")