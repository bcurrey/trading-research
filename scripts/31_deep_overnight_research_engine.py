import polars as pl
from pathlib import Path
from itertools import combinations, product
from datetime import datetime

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "deep_overnight_research"
OUT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "NQ_feature_factory.parquet"

print("\n==============================")
print("DEEP OVERNIGHT RESEARCH ENGINE")
print("==============================")
print("Started:", datetime.now())

df = pl.read_parquet(INPUT_FILE).sort("ts_event")

# ---------------------------------------------------
# Research window
# ---------------------------------------------------

df = df.filter(
    (pl.col("minute_of_day_ct") >= 8 * 60 + 30) &
    (pl.col("minute_of_day_ct") <= 11 * 60 + 30)
)

print(f"\nRows in NY morning window: {df.height:,}")

# ---------------------------------------------------
# Add outcomes and buckets
# ---------------------------------------------------

df = df.with_columns([
    pl.col("trade_date_ct").dt.year().alias("year"),

    (pl.col("fwd_points_30m") >= 25).alias("long_25_30m"),
    (pl.col("fwd_points_60m") >= 25).alias("long_25_60m"),
    (pl.col("fwd_points_60m") >= 40).alias("long_40_60m"),
    (pl.col("fwd_points_120m") >= 40).alias("long_40_120m"),
    (pl.col("fwd_points_120m") >= 60).alias("long_60_120m"),

    (pl.col("fwd_points_30m") <= -25).alias("short_25_30m"),
    (pl.col("fwd_points_60m") <= -25).alias("short_25_60m"),
    (pl.col("fwd_points_60m") <= -40).alias("short_40_60m"),
    (pl.col("fwd_points_120m") <= -40).alias("short_40_120m"),
    (pl.col("fwd_points_120m") <= -60).alias("short_60_120m"),
])

df = df.with_columns([
    pl.when(pl.col("hour_ct") == 8).then(pl.lit("8"))
    .when(pl.col("hour_ct") == 9).then(pl.lit("9"))
    .when(pl.col("hour_ct") == 10).then(pl.lit("10"))
    .otherwise(pl.lit("11"))
    .alias("hour_bucket"),

    pl.when(pl.col("close") > pl.col("vwap_day"))
    .then(pl.lit("above_vwap"))
    .otherwise(pl.lit("below_vwap"))
    .alias("vwap_side"),

    pl.when(pl.col("dist_vwap_day").abs() < 10)
    .then(pl.lit("near_vwap"))
    .when(pl.col("dist_vwap_day").abs() < 25)
    .then(pl.lit("mid_vwap_dist"))
    .otherwise(pl.lit("far_vwap"))
    .alias("vwap_dist_bucket"),

    pl.when(pl.col("atr_14") < 12)
    .then(pl.lit("very_low_atr"))
    .when(pl.col("atr_14") < 20)
    .then(pl.lit("low_atr"))
    .when(pl.col("atr_14") < 35)
    .then(pl.lit("mid_atr"))
    .otherwise(pl.lit("high_atr"))
    .alias("atr_bucket"),

    pl.when(pl.col("rel_vol_20") < 0.7)
    .then(pl.lit("low_vol"))
    .when(pl.col("rel_vol_20") < 1.2)
    .then(pl.lit("normal_vol"))
    .when(pl.col("rel_vol_20") < 2.0)
    .then(pl.lit("high_vol"))
    .otherwise(pl.lit("extreme_vol"))
    .alias("volume_bucket"),

    pl.when(pl.col("body_pct") < 0.25)
    .then(pl.lit("weak_body"))
    .when(pl.col("body_pct") < 0.50)
    .then(pl.lit("medium_body"))
    .when(pl.col("body_pct") < 0.70)
    .then(pl.lit("strong_body"))
    .otherwise(pl.lit("very_strong_body"))
    .alias("body_bucket"),

    pl.when(pl.col("upper_wick_pct") >= 0.50)
    .then(pl.lit("large_upper_wick"))
    .otherwise(pl.lit("no_large_upper_wick"))
    .alias("upper_wick_bucket"),

    pl.when(pl.col("lower_wick_pct") >= 0.50)
    .then(pl.lit("large_lower_wick"))
    .otherwise(pl.lit("no_large_lower_wick"))
    .alias("lower_wick_bucket"),

    pl.when(pl.col("or_range_30m") < 35)
    .then(pl.lit("small_or"))
    .when(pl.col("or_range_30m") < 70)
    .then(pl.lit("normal_or"))
    .otherwise(pl.lit("large_or"))
    .alias("or_range_bucket"),

    pl.when(pl.col("prior_day_range") < 150)
    .then(pl.lit("small_prior_day"))
    .when(pl.col("prior_day_range") < 300)
    .then(pl.lit("normal_prior_day"))
    .otherwise(pl.lit("large_prior_day"))
    .alias("prior_day_range_bucket"),
])

# ---------------------------------------------------
# Feature lists
# ---------------------------------------------------

binary_features = [
    "pdh_sweep_close_back_below",
    "pdl_sweep_close_back_above",
    "pmh_sweep_close_back_below",
    "pml_sweep_close_back_above",
    "above_or_high",
    "below_or_low",
    "bull_fvg_min_2pt",
    "bear_fvg_min_2pt",
    "bull_displacement",
    "bear_displacement",
    "swept_prior_bar_high",
    "swept_prior_bar_low",
    "prior_high_sweep_close_back_below",
    "prior_low_sweep_close_back_above",
    "has_high_impact_usd_event",
    "has_cpi",
    "has_fomc",
    "has_nfp",
    "has_ppi",
    "has_pmi_ism",
    "has_jobs",
    "has_fed_speech",
]

bucket_features = [
    "hour_bucket",
    "weekday_ct",
    "vwap_side",
    "vwap_dist_bucket",
    "atr_bucket",
    "volume_bucket",
    "body_bucket",
    "upper_wick_bucket",
    "lower_wick_bucket",
    "or_range_bucket",
    "prior_day_range_bucket",
]

outcomes = [
    ("long_25_30m", "long", 25, 30),
    ("long_25_60m", "long", 25, 60),
    ("long_40_60m", "long", 40, 60),
    ("long_40_120m", "long", 40, 120),
    ("long_60_120m", "long", 60, 120),
    ("short_25_30m", "short", 25, 30),
    ("short_25_60m", "short", 25, 60),
    ("short_40_60m", "short", 40, 60),
    ("short_40_120m", "short", 40, 120),
    ("short_60_120m", "short", 60, 120),
]

MIN_ROWS_SINGLE = 200
MIN_ROWS_COMBO = 100

all_results = []

# ---------------------------------------------------
# Utility
# ---------------------------------------------------

def score_group(data: pl.DataFrame, label: str, outcome_col: str, direction: str, target: int, horizon: int):
    if data.height == 0:
        return None

    if direction == "long":
        avg_fwd = data.select(pl.col(f"fwd_points_{horizon}m").mean()).item() if f"fwd_points_{horizon}m" in data.columns else None
    else:
        avg_fwd = data.select((-pl.col(f"fwd_points_{horizon}m")).mean()).item() if f"fwd_points_{horizon}m" in data.columns else None

    years = (
        data
        .group_by("year")
        .agg([
            pl.len().alias("year_rows"),
            pl.col(outcome_col).mean().alias("year_hit_rate"),
        ])
        .with_columns([
            (pl.col("year_hit_rate") > 0.50).alias("positive_year")
        ])
    )

    positive_years = years.select(pl.col("positive_year").sum()).item()
    total_years = years.height

    result = data.select([
        pl.len().alias("rows"),
        pl.col(outcome_col).mean().alias("hit_rate"),
    ]).to_dicts()[0]

    result.update({
        "label": label,
        "outcome": outcome_col,
        "direction": direction,
        "target_points": target,
        "horizon_minutes": horizon,
        "avg_directional_fwd": avg_fwd,
        "positive_years": positive_years,
        "total_years": total_years,
        "positive_year_rate": positive_years / total_years if total_years else 0,
    })

    return result


# ---------------------------------------------------
# 1. Single binary feature scan
# ---------------------------------------------------

print("\n[1/5] Scanning single binary features...")

for feature in binary_features:
    if feature not in df.columns:
        continue

    feature_df = df.filter(pl.col(feature) == True)

    if feature_df.height < MIN_ROWS_SINGLE:
        continue

    for outcome_col, direction, target, horizon in outcomes:
        row = score_group(
            feature_df,
            f"{feature}=true",
            outcome_col,
            direction,
            target,
            horizon,
        )

        if row:
            row["scan_type"] = "single_binary"
            all_results.append(row)

# ---------------------------------------------------
# 2. Single bucket feature scan
# ---------------------------------------------------

print("\n[2/5] Scanning single bucket features...")

for bucket in bucket_features:
    vals = df.select(pl.col(bucket).unique()).to_series().to_list()

    for val in vals:
        group_df = df.filter(pl.col(bucket) == val)

        if group_df.height < MIN_ROWS_SINGLE:
            continue

        for outcome_col, direction, target, horizon in outcomes:
            row = score_group(
                group_df,
                f"{bucket}={val}",
                outcome_col,
                direction,
                target,
                horizon,
            )

            if row:
                row["scan_type"] = "single_bucket"
                all_results.append(row)

# ---------------------------------------------------
# 3. Binary + bucket combinations
# ---------------------------------------------------

print("\n[3/5] Scanning binary + bucket combinations...")

for feature in binary_features:
    if feature not in df.columns:
        continue

    base_feature_df = df.filter(pl.col(feature) == True)

    if base_feature_df.height < MIN_ROWS_COMBO:
        continue

    for bucket in bucket_features:
        vals = base_feature_df.select(pl.col(bucket).unique()).to_series().to_list()

        for val in vals:
            combo_df = base_feature_df.filter(pl.col(bucket) == val)

            if combo_df.height < MIN_ROWS_COMBO:
                continue

            for outcome_col, direction, target, horizon in outcomes:
                row = score_group(
                    combo_df,
                    f"{feature}=true | {bucket}={val}",
                    outcome_col,
                    direction,
                    target,
                    horizon,
                )

                if row:
                    row["scan_type"] = "binary_bucket"
                    all_results.append(row)

# ---------------------------------------------------
# 4. Binary + two-bucket combinations
# ---------------------------------------------------

print("\n[4/5] Scanning binary + two-bucket combinations...")

for feature in binary_features:
    if feature not in df.columns:
        continue

    base_feature_df = df.filter(pl.col(feature) == True)

    if base_feature_df.height < MIN_ROWS_COMBO:
        continue

    for b1, b2 in combinations(bucket_features, 2):
        vals1 = base_feature_df.select(pl.col(b1).unique()).to_series().to_list()
        vals2 = base_feature_df.select(pl.col(b2).unique()).to_series().to_list()

        for v1, v2 in product(vals1, vals2):
            combo_df = base_feature_df.filter(
                (pl.col(b1) == v1) &
                (pl.col(b2) == v2)
            )

            if combo_df.height < MIN_ROWS_COMBO:
                continue

            for outcome_col, direction, target, horizon in outcomes:
                row = score_group(
                    combo_df,
                    f"{feature}=true | {b1}={v1} | {b2}={v2}",
                    outcome_col,
                    direction,
                    target,
                    horizon,
                )

                if row:
                    row["scan_type"] = "binary_two_bucket"
                    all_results.append(row)

# ---------------------------------------------------
# 5. Binary + binary + bucket combinations
# ---------------------------------------------------

print("\n[5/5] Scanning binary + binary + bucket combinations...")

for f1, f2 in combinations(binary_features, 2):
    if f1 not in df.columns or f2 not in df.columns:
        continue

    base_combo_df = df.filter(
        (pl.col(f1) == True) &
        (pl.col(f2) == True)
    )

    if base_combo_df.height < MIN_ROWS_COMBO:
        continue

    for bucket in bucket_features:
        vals = base_combo_df.select(pl.col(bucket).unique()).to_series().to_list()

        for val in vals:
            combo_df = base_combo_df.filter(pl.col(bucket) == val)

            if combo_df.height < MIN_ROWS_COMBO:
                continue

            for outcome_col, direction, target, horizon in outcomes:
                row = score_group(
                    combo_df,
                    f"{f1}=true | {f2}=true | {bucket}={val}",
                    outcome_col,
                    direction,
                    target,
                    horizon,
                )

                if row:
                    row["scan_type"] = "binary_binary_bucket"
                    all_results.append(row)

# ---------------------------------------------------
# Final ranking
# ---------------------------------------------------

if not all_results:
    print("\nNo results generated.")
    raise SystemExit

results = pl.DataFrame(all_results)

results = results.with_columns([
    (pl.col("hit_rate") * 100).round(2).alias("hit_rate_pct"),
    (pl.col("positive_year_rate") * 100).round(2).alias("positive_year_rate_pct"),
    pl.col("avg_directional_fwd").round(2),
    (
        pl.col("avg_directional_fwd") *
        pl.col("positive_year_rate") *
        (pl.col("rows").clip(upper_bound=1000) / 1000)
    ).round(4).alias("research_score")
])

ranked = (
    results
    .filter(
        (pl.col("rows") >= 100) &
        (pl.col("avg_directional_fwd") > 0) &
        (pl.col("hit_rate") >= 0.48) &
        (pl.col("positive_year_rate") >= 0.55)
    )
    .sort("research_score", descending=True)
)

# ---------------------------------------------------
# Save everything
# ---------------------------------------------------

all_file = OUT_DIR / "deep_research_all_results.csv"
ranked_file = OUT_DIR / "deep_research_ranked_results.csv"

results.write_csv(all_file)
ranked.write_csv(ranked_file)

print("\n==============================")
print("TOP 100 RANKED RESEARCH CLUSTERS")
print("==============================")

print(
    ranked
    .select([
        "scan_type",
        "label",
        "outcome",
        "direction",
        "target_points",
        "horizon_minutes",
        "rows",
        "hit_rate_pct",
        "avg_directional_fwd",
        "positive_years",
        "total_years",
        "positive_year_rate_pct",
        "research_score",
    ])
    .head(100)
)

print("\nSaved:")
print(all_file)
print(ranked_file)

print("\nFinished:", datetime.now())
print("\nDONE.")