import polars as pl
from pathlib import Path
from itertools import combinations, product
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "execution_models"
OUT_DIR.mkdir(exist_ok=True)

print("\nORB LOW RECLAIM STRATEGY LAB")
print("Started:", datetime.now())

FILE = OUT_DIR / "orb_confirmed_failure_trades.csv"

df = pl.read_csv(FILE, try_parse_dates=True)

# ---------------------------------------------------
# Focus strategy family:
# Failed OR low breakdown reclaim LONG only
# ---------------------------------------------------

df = df.filter(
    (pl.col("event_type") == "failed_or_low_breakdown") &
    (pl.col("direction") == "long")
)

print("\nRows loaded:", df.height)

# ---------------------------------------------------
# Add derived execution features
# ---------------------------------------------------

df = df.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),
    pl.col("trade_date_ct").dt.month().alias("month"),
    pl.col("trade_date_ct").dt.strftime("%Y-%m").alias("year_month"),

    # For failed OR low reclaim:
    # sweep_price is below OR low
    (pl.col("or_low") - pl.col("sweep_price")).alias("sweep_depth_points"),
    (pl.col("entry") - pl.col("or_low")).alias("reclaim_strength_points"),
    (pl.col("or_mid") - pl.col("entry")).alias("distance_to_or_mid"),
    (pl.col("or_high") - pl.col("entry")).alias("distance_to_or_high"),

    (pl.col("risk_points") / pl.col("or_range")).alias("risk_vs_or_range"),
])

df = df.with_columns([
    pl.when(pl.col("sweep_depth_points") < 5)
    .then(pl.lit("sweep_tiny"))
    .when(pl.col("sweep_depth_points") < 15)
    .then(pl.lit("sweep_normal"))
    .when(pl.col("sweep_depth_points") < 30)
    .then(pl.lit("sweep_deep"))
    .otherwise(pl.lit("sweep_extreme"))
    .alias("sweep_depth_bucket"),

    pl.when(pl.col("reclaim_strength_points") < 2)
    .then(pl.lit("weak_reclaim"))
    .when(pl.col("reclaim_strength_points") < 8)
    .then(pl.lit("normal_reclaim"))
    .when(pl.col("reclaim_strength_points") < 15)
    .then(pl.lit("strong_reclaim"))
    .otherwise(pl.lit("extreme_reclaim"))
    .alias("reclaim_strength_bucket"),

    pl.when(pl.col("distance_to_or_mid") < 10)
    .then(pl.lit("near_or_mid"))
    .when(pl.col("distance_to_or_mid") < 30)
    .then(pl.lit("mid_distance_to_or_mid"))
    .otherwise(pl.lit("far_from_or_mid"))
    .alias("or_mid_distance_bucket"),

    pl.when(pl.col("risk_points") < 8)
    .then(pl.lit("risk_too_small"))
    .when(pl.col("risk_points") < 15)
    .then(pl.lit("risk_8_15"))
    .when(pl.col("risk_points") < 25)
    .then(pl.lit("risk_15_25"))
    .when(pl.col("risk_points") <= 35)
    .then(pl.lit("risk_25_35"))
    .otherwise(pl.lit("risk_too_large"))
    .alias("risk_bucket_custom"),

    pl.when(pl.col("minute_of_day_ct") <= 9 * 60 + 5)
    .then(pl.lit("entry_900_905"))
    .when(pl.col("minute_of_day_ct") <= 9 * 60 + 10)
    .then(pl.lit("entry_906_910"))
    .when(pl.col("minute_of_day_ct") <= 9 * 60 + 15)
    .then(pl.lit("entry_911_915"))
    .when(pl.col("minute_of_day_ct") <= 9 * 60 + 30)
    .then(pl.lit("entry_916_930"))
    .otherwise(pl.lit("entry_after_930"))
    .alias("entry_time_fine"),

    pl.when(pl.col("dist_vwap_day") < -50)
    .then(pl.lit("far_below_vwap"))
    .when(pl.col("dist_vwap_day") < -20)
    .then(pl.lit("below_vwap"))
    .when(pl.col("dist_vwap_day") < 0)
    .then(pl.lit("near_below_vwap"))
    .otherwise(pl.lit("above_vwap"))
    .alias("vwap_distance_bucket"),
])

# ---------------------------------------------------
# Test windows
# ---------------------------------------------------

max_date = df.select(pl.col("trade_date_ct").max()).item()
last_12m_start = max_date.replace(year=max_date.year - 1)

print("\nMax date:", max_date)
print("Last 12m start:", last_12m_start)

WINDOWS = [
    ("full_sample", None),
    ("last_12_months", last_12m_start),
]

TARGETS = [
    "or_mid",
    "opposite_or",
    "fixed_25",
    "fixed_40",
    "fixed_60",
]

# ---------------------------------------------------
# Context columns to test
# ---------------------------------------------------

context_cols = [
    "entry_time_bucket",
    "entry_time_fine",
    "or_vol_regime",
    "overnight_vol_regime",
    "atr_vol_regime",
    "volume_bucket",
    "body_bucket",
    "risk_bucket",
    "risk_bucket_custom",
    "vwap_side",
    "vwap_distance_bucket",
    "gap_bucket",
    "overnight_position",
    "prior_day_position",
    "sweep_depth_bucket",
    "reclaim_strength_bucket",
    "or_mid_distance_bucket",
]

context_cols = [c for c in context_cols if c in df.columns]

print("\nContext columns:")
for c in context_cols:
    print("-", c)

# ---------------------------------------------------
# Summary function
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


def summarize(label, data, window_name, target_name, filter_type):
    if data.height < 8:
        return None

    vals = data.sort("trade_date_ct").select("points").to_series().to_list()
    dd = max_drawdown(vals)

    month_stats = (
        data
        .group_by("year_month")
        .agg([
            pl.len().alias("month_trades"),
            pl.col("points").sum().alias("month_points"),
            pl.col("r").sum().alias("month_r"),
        ])
        .with_columns((pl.col("month_points") > 0).alias("positive_month"))
    )

    year_stats = (
        data
        .group_by("year")
        .agg([
            pl.len().alias("year_trades"),
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

    row = data.select([
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
        pl.col("sweep_depth_points").mean().alias("avg_sweep_depth"),
        pl.col("reclaim_strength_points").mean().alias("avg_reclaim_strength"),
    ]).to_dicts()[0]

    row.update({
        "window": window_name,
        "target_name": target_name,
        "filter_type": filter_type,
        "filter": label,
        "positive_months": positive_months,
        "months_tested": months_tested,
        "positive_month_rate": positive_months / months_tested if months_tested else 0,
        "worst_month_points": worst_month,
        "positive_years": positive_years,
        "years_tested": years_tested,
        "positive_year_rate": positive_years / years_tested if years_tested else 0,
        "worst_year_points": worst_year,
        "max_drawdown_points": dd,
    })

    return row

# ---------------------------------------------------
# Run strategy lab
# ---------------------------------------------------

results = []

for window_name, start_date in WINDOWS:
    if start_date is None:
        window_df = df
    else:
        window_df = df.filter(pl.col("trade_date_ct") >= start_date)

    print(f"\nScanning window: {window_name} | rows: {window_df.height}")

    for target in TARGETS:
        base = window_df.filter(pl.col("target_name") == target)

        # Realistic risk only
        base = base.filter(
            (pl.col("risk_points") >= 8) &
            (pl.col("risk_points") <= 35)
        )

        if base.height < 8:
            continue

        # Base
        row = summarize(
            f"BASE | target={target}",
            base,
            window_name,
            target,
            "base",
        )
        if row:
            results.append(row)

        # Single filters
        for c in context_cols:
            vals = base.select(pl.col(c).drop_nulls().unique()).to_series().to_list()

            for v in vals:
                x = base.filter(pl.col(c).cast(pl.Utf8) == str(v))
                row = summarize(
                    f"{c}={v}",
                    x,
                    window_name,
                    target,
                    "single",
                )
                if row:
                    results.append(row)

        # Two-filter combos
        for c1, c2 in combinations(context_cols, 2):
            vals1 = base.select(pl.col(c1).drop_nulls().unique()).to_series().to_list()
            vals2 = base.select(pl.col(c2).drop_nulls().unique()).to_series().to_list()

            for v1, v2 in product(vals1, vals2):
                x = base.filter(
                    (pl.col(c1).cast(pl.Utf8) == str(v1)) &
                    (pl.col(c2).cast(pl.Utf8) == str(v2))
                )

                row = summarize(
                    f"{c1}={v1} | {c2}={v2}",
                    x,
                    window_name,
                    target,
                    "two_filter",
                )
                if row:
                    results.append(row)

        # Three-filter combos
        for c1, c2, c3 in combinations(context_cols, 3):
            vals1 = base.select(pl.col(c1).drop_nulls().unique()).to_series().to_list()
            vals2 = base.select(pl.col(c2).drop_nulls().unique()).to_series().to_list()
            vals3 = base.select(pl.col(c3).drop_nulls().unique()).to_series().to_list()

            for v1, v2, v3 in product(vals1, vals2, vals3):
                x = base.filter(
                    (pl.col(c1).cast(pl.Utf8) == str(v1)) &
                    (pl.col(c2).cast(pl.Utf8) == str(v2)) &
                    (pl.col(c3).cast(pl.Utf8) == str(v3))
                )

                row = summarize(
                    f"{c1}={v1} | {c2}={v2} | {c3}={v3}",
                    x,
                    window_name,
                    target,
                    "three_filter",
                )
                if row:
                    results.append(row)

if not results:
    raise SystemExit("No results generated.")

results_df = pl.DataFrame(results)

# ---------------------------------------------------
# Score and rank
# ---------------------------------------------------

results_df = results_df.with_columns([
    (pl.col("target_rate") * 100).round(2).alias("target_rate_pct"),
    (pl.col("positive_month_rate") * 100).round(2).alias("positive_month_rate_pct"),
    (pl.col("positive_year_rate") * 100).round(2).alias("positive_year_rate_pct"),

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
    pl.col("worst_month_points").round(2),
    pl.col("worst_year_points").round(2),
    pl.col("max_drawdown_points").round(2),

    (
        (pl.col("avg_points") * 2.0) +
        (pl.col("avg_r") * 10.0) +
        (pl.col("positive_month_rate") * 10.0) +
        (pl.col("trades").clip(upper_bound=50) / 5.0) -
        (pl.col("max_drawdown_points") / 25.0)
    ).round(4).alias("strategy_score")
])

ranked = (
    results_df
    .filter(
        (pl.col("trades") >= 12) &
        (pl.col("avg_points") >= 8) &
        (pl.col("avg_r") >= 0.20) &
        (pl.col("total_points") > 0) &
        (pl.col("positive_month_rate") >= 0.45)
    )
    .sort("strategy_score", descending=True)
)

strict = (
    ranked
    .filter(
        (pl.col("trades") >= 18) &
        (pl.col("avg_points") >= 10) &
        (pl.col("avg_r") >= 0.30)
    )
    .sort("strategy_score", descending=True)
)

# ---------------------------------------------------
# Print outputs
# ---------------------------------------------------

print("\nTOP RANKED STRATEGY CANDIDATES:")
print(
    ranked
    .select([
        "window",
        "filter_type",
        "target_name",
        "filter",
        "trades",
        "target_rate_pct",
        "avg_points",
        "median_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "positive_month_rate_pct",
        "worst_month_points",
        "max_drawdown_points",
        "avg_sweep_depth",
        "avg_reclaim_strength",
        "strategy_score",
    ])
    .head(100)
)

print("\nSTRICT CANDIDATES:")
print(
    strict
    .select([
        "window",
        "filter_type",
        "target_name",
        "filter",
        "trades",
        "target_rate_pct",
        "avg_points",
        "median_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "positive_month_rate_pct",
        "worst_month_points",
        "max_drawdown_points",
        "strategy_score",
    ])
    .head(50)
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

all_file = OUT_DIR / "orb_low_reclaim_strategy_lab_all.csv"
ranked_file = OUT_DIR / "orb_low_reclaim_strategy_lab_ranked.csv"
strict_file = OUT_DIR / "orb_low_reclaim_strategy_lab_strict.csv"

results_df.write_csv(all_file)
ranked.write_csv(ranked_file)
strict.write_csv(strict_file)

print("\nSaved:")
print(all_file)
print(ranked_file)
print(strict_file)

print("\nFinished:", datetime.now())
print("DONE.")