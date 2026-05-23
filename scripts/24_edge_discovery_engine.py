import polars as pl
from pathlib import Path
from itertools import product

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "edge_discovery"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

# ---------------------------------------------------
# Add helper columns
# ---------------------------------------------------

df = df.with_columns([
    pl.col("close").shift(1).alias("prev_close"),
    pl.col("vwap_day").shift(1).alias("prev_vwap"),
    pl.col("trade_date_ct").dt.year().alias("year"),
])

df = df.with_columns([
    (
        (pl.col("prev_close") < pl.col("prev_vwap")) &
        (pl.col("close") > pl.col("vwap_day"))
    ).alias("bull_vwap_reclaim"),

    (
        (pl.col("prev_close") > pl.col("prev_vwap")) &
        (pl.col("close") < pl.col("vwap_day"))
    ).alias("bear_vwap_reclaim"),

    (
        (pl.col("close") > pl.col("ema_20")) &
        (pl.col("close") > pl.col("ema_50")) &
        (pl.col("low") <= pl.col("ema_20"))
    ).alias("bull_trend_pullback"),

    (
        (pl.col("close") < pl.col("ema_20")) &
        (pl.col("close") < pl.col("ema_50")) &
        (pl.col("high") >= pl.col("ema_20"))
    ).alias("bear_trend_pullback"),
])

# ---------------------------------------------------
# Setup definitions
# ---------------------------------------------------

SETUPS = [
    ("PDL sweep reversal", "long", pl.col("pdl_sweep_close_back_above")),
    ("PDH sweep reversal", "short", pl.col("pdh_sweep_close_back_below")),
    ("PML sweep reversal", "long", pl.col("pml_sweep_close_back_above")),
    ("PMH sweep reversal", "short", pl.col("pmh_sweep_close_back_below")),
    ("OR high breakout", "long", pl.col("above_or_high")),
    ("OR low breakdown", "short", pl.col("below_or_low")),
    ("Bull displacement FVG", "long", pl.col("bull_displacement") & pl.col("bull_fvg_min_2pt")),
    ("Bear displacement FVG", "short", pl.col("bear_displacement") & pl.col("bear_fvg_min_2pt")),
    ("Bull VWAP reclaim", "long", pl.col("bull_vwap_reclaim")),
    ("Bear VWAP reclaim", "short", pl.col("bear_vwap_reclaim")),
    ("Bull trend pullback", "long", pl.col("bull_trend_pullback")),
    ("Bear trend pullback", "short", pl.col("bear_trend_pullback")),
]

HOUR_FILTERS = [
    ("8 only", [8]),
    ("9 only", [9]),
    ("10 only", [10]),
    ("8-9", [8, 9]),
    ("8-10", [8, 9, 10]),
    ("9-10", [9, 10]),
    ("morning full", [8, 9, 10, 11]),
]

VWAP_FILTERS = [
    ("any_vwap", None),
    ("above_vwap", "above"),
    ("below_vwap", "below"),
]

BODY_FILTERS = [
    ("any_body", None),
    ("body_50_plus", 0.50),
    ("body_65_plus", 0.65),
]

ATR_FILTERS = [
    ("any_atr", None),
    ("atr_15_plus", 15),
    ("atr_25_plus", 25),
]

VOL_FILTERS = [
    ("any_vol", None),
    ("normal_vol", "normal"),
    ("high_vol", "high"),
]

RISK_FILTERS = [
    ("risk_5_50", 5.0, 50.0),
    ("risk_8_40", 8.0, 40.0),
    ("risk_10_35", 10.0, 35.0),
]

TARGETS = [25.0, 40.0, 60.0]
MIN_TRADES = 75

results = []

# ---------------------------------------------------
# Conservative fast backtest
# Uses future high/low over 60 minutes.
# If target and stop both hit in same window, count as stop.
# This is intentionally conservative for discovery.
# ---------------------------------------------------

def evaluate_model(data, setup_name, direction, filter_name, target_points, min_risk, max_risk):
    if data.height < MIN_TRADES:
        return None

    if direction == "long":
        trades = data.with_columns([
            pl.col("close").alias("entry"),
            pl.col("low").alias("stop"),
            (pl.col("close") - pl.col("low")).alias("risk_points"),
            (pl.col("future_high_60m") - pl.col("close")).alias("favorable_points"),
            (pl.col("close") - pl.col("future_low_60m")).alias("adverse_points"),
        ])
    else:
        trades = data.with_columns([
            pl.col("close").alias("entry"),
            pl.col("high").alias("stop"),
            (pl.col("high") - pl.col("close")).alias("risk_points"),
            (pl.col("close") - pl.col("future_low_60m")).alias("favorable_points"),
            (pl.col("future_high_60m") - pl.col("close")).alias("adverse_points"),
        ])

    trades = trades.filter(
        (pl.col("risk_points") >= min_risk) &
        (pl.col("risk_points") <= max_risk)
    )

    if trades.height < MIN_TRADES:
        return None

    trades = trades.with_columns([
        (pl.col("favorable_points") >= target_points).alias("hit_target"),
        (pl.col("adverse_points") >= pl.col("risk_points")).alias("hit_stop"),
    ])

    # Conservative:
    # target only counts if target hit and stop did NOT also hit within 60m
    trades = trades.with_columns([
        (
            pl.col("hit_target") &
            ~pl.col("hit_stop")
        ).alias("clean_win"),

        pl.when(pl.col("hit_target") & ~pl.col("hit_stop"))
        .then(pl.lit(target_points))
        .when(pl.col("hit_stop"))
        .then(-pl.col("risk_points"))
        .otherwise(
            pl.when(direction == "long")
            .then(pl.col("fwd_points_60m"))
            .otherwise(-pl.col("fwd_points_60m"))
        )
        .alias("points_result")
    ])

    summary = trades.select([
        pl.len().alias("trades"),
        pl.col("clean_win").mean().alias("win_rate"),
        pl.col("points_result").mean().alias("avg_points"),
        pl.col("points_result").sum().alias("total_points"),
        pl.col("risk_points").mean().alias("avg_risk"),
        pl.col("favorable_points").mean().alias("avg_favorable"),
        pl.col("adverse_points").mean().alias("avg_adverse"),
    ])

    year_stats = (
        trades
        .group_by("year")
        .agg([
            pl.len().alias("year_trades"),
            pl.col("points_result").sum().alias("year_points"),
        ])
        .with_columns([
            (pl.col("year_points") > 0).alias("positive_year")
        ])
    )

    positive_years = year_stats.select(pl.col("positive_year").sum()).item()
    total_years = year_stats.height

    row = summary.to_dicts()[0]

    row.update({
        "setup": setup_name,
        "direction": direction,
        "filters": filter_name,
        "target_points": target_points,
        "min_risk": min_risk,
        "max_risk": max_risk,
        "positive_years": positive_years,
        "total_years": total_years,
        "positive_year_rate": positive_years / total_years if total_years else 0,
    })

    return row


# ---------------------------------------------------
# Scan engine
# ---------------------------------------------------

base = df.filter(pl.col("is_morning_trade_window"))

print("\nStarting edge discovery scan...")
print(f"Morning rows: {base.height:,}")

for setup_name, direction, setup_expr in SETUPS:
    setup_df = base.filter(setup_expr)

    print(f"\nScanning {setup_name}: {setup_df.height:,} raw signals")

    for hour_name, hours in HOUR_FILTERS:
        for vwap_name, vwap_side in VWAP_FILTERS:
            for body_name, body_min in BODY_FILTERS:
                for atr_name, atr_min in ATR_FILTERS:
                    for vol_name, vol_filter in VOL_FILTERS:
                        for risk_name, min_risk, max_risk in RISK_FILTERS:
                            for target in TARGETS:

                                filt = setup_df.filter(pl.col("hour_ct").is_in(hours))

                                if vwap_side == "above":
                                    filt = filt.filter(pl.col("close") > pl.col("vwap_day"))
                                elif vwap_side == "below":
                                    filt = filt.filter(pl.col("close") < pl.col("vwap_day"))

                                if body_min is not None:
                                    filt = filt.filter(pl.col("body_pct") >= body_min)

                                if atr_min is not None:
                                    filt = filt.filter(pl.col("atr_14") >= atr_min)

                                if vol_filter == "normal":
                                    filt = filt.filter(
                                        (pl.col("rel_vol_20") >= 0.8) &
                                        (pl.col("rel_vol_20") <= 1.2)
                                    )
                                elif vol_filter == "high":
                                    filt = filt.filter(pl.col("rel_vol_20") >= 1.2)

                                filter_name = " | ".join([
                                    hour_name,
                                    vwap_name,
                                    body_name,
                                    atr_name,
                                    vol_name,
                                    risk_name,
                                ])

                                row = evaluate_model(
                                    filt,
                                    setup_name,
                                    direction,
                                    filter_name,
                                    target,
                                    min_risk,
                                    max_risk,
                                )

                                if row is not None:
                                    results.append(row)

if not results:
    print("\nNo models passed minimum trade threshold.")
    raise SystemExit

results_df = pl.DataFrame(results)

# ---------------------------------------------------
# Rank results
# ---------------------------------------------------

results_df = results_df.with_columns([
    (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
    (pl.col("positive_year_rate") * 100).round(2).alias("positive_year_rate_pct"),

    pl.col("avg_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_risk").round(2),
    pl.col("avg_favorable").round(2),
    pl.col("avg_adverse").round(2),

    (
        pl.col("avg_points") *
        pl.col("positive_year_rate") *
        (pl.col("trades").clip(upper_bound=500) / 500)
    ).alias("edge_score")
])

results_df = results_df.with_columns([
    pl.col("edge_score").round(4)
])

ranked = (
    results_df
    .filter(
        (pl.col("avg_points") > 0) &
        (pl.col("win_rate") > 0.45) &
        (pl.col("positive_year_rate") >= 0.55) &
        (pl.col("trades") >= MIN_TRADES)
    )
    .sort("edge_score", descending=True)
)

print("\nTOP 30 CANDIDATE EDGES:")
print(
    ranked.select([
        "setup",
        "direction",
        "filters",
        "target_points",
        "trades",
        "win_rate_pct",
        "avg_points",
        "total_points",
        "avg_risk",
        "positive_years",
        "total_years",
        "positive_year_rate_pct",
        "edge_score",
    ]).head(30)
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

all_file = OUT_DIR / "edge_discovery_all_results.csv"
ranked_file = OUT_DIR / "edge_discovery_ranked_results.csv"

results_df.write_csv(all_file)
ranked.write_csv(ranked_file)

print("\nSaved:")
print(all_file)
print(ranked_file)

print("\nDONE.")