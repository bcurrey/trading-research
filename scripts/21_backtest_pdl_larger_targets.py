import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "setup_scans"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet").sort("ts_event")

signals = df.filter(
    (pl.col("pdl_sweep_close_back_above")) &
    (pl.col("hour_ct").is_in([8, 9])) &
    (pl.col("close") > pl.col("vwap_day")) &
    (pl.col("body_pct") >= 0.50)
)

print("\nSignals found:", signals.height)

trades = []

MIN_RISK = 8.0
MAX_RISK = 35.0
MIN_TARGET_POINTS = 25.0
MAX_HOLD_MINUTES = 60

for row in signals.iter_rows(named=True):
    entry_time = row["ts_event"]
    entry = row["close"]
    stop = row["low"]
    risk = entry - stop

    if risk <= 0:
        continue

    if risk < MIN_RISK:
        continue

    if risk > MAX_RISK:
        continue

    tp_15r = entry + risk * 1.5
    tp_2r = entry + risk * 2.0
    tp_25 = entry + 25.0
    tp_40 = entry + 40.0

    if risk * 1.5 < MIN_TARGET_POINTS:
        continue

    future = (
        df
        .filter(
            (pl.col("ts_event") > entry_time) &
            (pl.col("ts_event") <= entry_time + pl.duration(minutes=MAX_HOLD_MINUTES))
        )
        .select(["ts_event", "high", "low", "close"])
    )

    if future.height == 0:
        continue

    outcomes = {
        "15r": {"status": "timeout", "exit": future[-1, "close"], "exit_time": future[-1, "ts_event"], "target": tp_15r},
        "2r": {"status": "timeout", "exit": future[-1, "close"], "exit_time": future[-1, "ts_event"], "target": tp_2r},
        "25pt": {"status": "timeout", "exit": future[-1, "close"], "exit_time": future[-1, "ts_event"], "target": tp_25},
        "40pt": {"status": "timeout", "exit": future[-1, "close"], "exit_time": future[-1, "ts_event"], "target": tp_40},
    }

    for bar in future.iter_rows(named=True):
        t = bar["ts_event"]
        high = bar["high"]
        low = bar["low"]

        for key in outcomes:
            if outcomes[key]["status"] != "timeout":
                continue

            target = outcomes[key]["target"]

            if low <= stop:
                outcomes[key]["status"] = "stop"
                outcomes[key]["exit"] = stop
                outcomes[key]["exit_time"] = t

            elif high >= target:
                outcomes[key]["status"] = "target"
                outcomes[key]["exit"] = target
                outcomes[key]["exit_time"] = t

    trades.append({
        "entry_time": entry_time,
        "trade_date": row["trade_date_ct"],
        "entry": entry,
        "stop": stop,
        "risk_points": risk,

        "tp_15r": tp_15r,
        "outcome_15r": outcomes["15r"]["status"],
        "r_15r": (outcomes["15r"]["exit"] - entry) / risk,
        "points_15r": outcomes["15r"]["exit"] - entry,

        "tp_2r": tp_2r,
        "outcome_2r": outcomes["2r"]["status"],
        "r_2r": (outcomes["2r"]["exit"] - entry) / risk,
        "points_2r": outcomes["2r"]["exit"] - entry,

        "tp_25": tp_25,
        "outcome_25pt": outcomes["25pt"]["status"],
        "r_25pt": (outcomes["25pt"]["exit"] - entry) / risk,
        "points_25pt": outcomes["25pt"]["exit"] - entry,

        "tp_40": tp_40,
        "outcome_40pt": outcomes["40pt"]["status"],
        "r_40pt": (outcomes["40pt"]["exit"] - entry) / risk,
        "points_40pt": outcomes["40pt"]["exit"] - entry,

        "hour_ct": row["hour_ct"],
        "weekday_ct": row["weekday_ct"],
        "body_pct": row["body_pct"],
        "atr_14": row["atr_14"],
        "rel_vol_20": row["rel_vol_20"],
        "vwap_day": row["vwap_day"],
        "prior_rth_low": row["prior_rth_low"],
    })

trades_df = pl.DataFrame(trades)

print("\nTrades after larger-target filters:", trades_df.height)

if trades_df.height == 0:
    print("No trades passed filters.")
    raise SystemExit


summary = trades_df.select([
    pl.len().alias("trades"),

    (pl.col("r_15r") > 0).mean().alias("win_rate_15r"),
    pl.col("r_15r").mean().alias("avg_r_15r"),
    pl.col("r_15r").sum().alias("total_r_15r"),
    pl.col("points_15r").mean().alias("avg_points_15r"),

    (pl.col("r_2r") > 0).mean().alias("win_rate_2r"),
    pl.col("r_2r").mean().alias("avg_r_2r"),
    pl.col("r_2r").sum().alias("total_r_2r"),
    pl.col("points_2r").mean().alias("avg_points_2r"),

    (pl.col("r_25pt") > 0).mean().alias("win_rate_25pt"),
    pl.col("r_25pt").mean().alias("avg_r_25pt"),
    pl.col("r_25pt").sum().alias("total_r_25pt"),
    pl.col("points_25pt").mean().alias("avg_points_25pt"),

    (pl.col("r_40pt") > 0).mean().alias("win_rate_40pt"),
    pl.col("r_40pt").mean().alias("avg_r_40pt"),
    pl.col("r_40pt").sum().alias("total_r_40pt"),
    pl.col("points_40pt").mean().alias("avg_points_40pt"),
])

summary = summary.with_columns([
    (pl.col("win_rate_15r") * 100).round(2).alias("win_rate_15r_pct"),
    (pl.col("win_rate_2r") * 100).round(2).alias("win_rate_2r_pct"),
    (pl.col("win_rate_25pt") * 100).round(2).alias("win_rate_25pt_pct"),
    (pl.col("win_rate_40pt") * 100).round(2).alias("win_rate_40pt_pct"),

    pl.col("avg_r_15r").round(3),
    pl.col("avg_r_2r").round(3),
    pl.col("avg_r_25pt").round(3),
    pl.col("avg_r_40pt").round(3),

    pl.col("total_r_15r").round(2),
    pl.col("total_r_2r").round(2),
    pl.col("total_r_25pt").round(2),
    pl.col("total_r_40pt").round(2),

    pl.col("avg_points_15r").round(2),
    pl.col("avg_points_2r").round(2),
    pl.col("avg_points_25pt").round(2),
    pl.col("avg_points_40pt").round(2),
])

print("\nSUMMARY:")
print(summary)

print("\nOutcome counts:")
print(
    trades_df.select([
        pl.col("outcome_15r").value_counts().alias("outcome_15r_counts"),
        pl.col("outcome_2r").value_counts().alias("outcome_2r_counts"),
        pl.col("outcome_25pt").value_counts().alias("outcome_25pt_counts"),
        pl.col("outcome_40pt").value_counts().alias("outcome_40pt_counts"),
    ])
)

print("\nBy hour:")
print(
    trades_df
    .group_by("hour_ct")
    .agg([
        pl.len().alias("trades"),
        (pl.col("r_25pt") > 0).mean().alias("win_rate_25pt"),
        pl.col("points_25pt").mean().alias("avg_points_25pt"),
        pl.col("r_25pt").sum().alias("total_r_25pt"),
    ])
    .with_columns([
        (pl.col("win_rate_25pt") * 100).round(2).alias("win_rate_25pt_pct"),
        pl.col("avg_points_25pt").round(2),
        pl.col("total_r_25pt").round(2),
    ])
    .sort("hour_ct")
)

print("\nBy weekday:")
print(
    trades_df
    .group_by("weekday_ct")
    .agg([
        pl.len().alias("trades"),
        (pl.col("r_25pt") > 0).mean().alias("win_rate_25pt"),
        pl.col("points_25pt").mean().alias("avg_points_25pt"),
        pl.col("r_25pt").sum().alias("total_r_25pt"),
    ])
    .with_columns([
        (pl.col("win_rate_25pt") * 100).round(2).alias("win_rate_25pt_pct"),
        pl.col("avg_points_25pt").round(2),
        pl.col("total_r_25pt").round(2),
    ])
    .sort("total_r_25pt", descending=True)
)

trades_file = OUT_DIR / "pdl_larger_targets_trades.csv"
summary_file = OUT_DIR / "pdl_larger_targets_summary.csv"

trades_df.write_csv(trades_file)
summary.write_csv(summary_file)

print("\nSaved:")
print(trades_file)
print(summary_file)

print("\nDONE.")