import polars as pl
from pathlib import Path
from datetime import datetime
import random
import statistics

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "robustness_audit"
OUT_DIR.mkdir(exist_ok=True)

print("\nDEEP CANDIDATE ROBUSTNESS AUDIT")
print("Started:", datetime.now())

CANDIDATE_FILES = [
    DATA_DIR / "orb_liquidity_research" / "orb_liquidity_trade_outcomes.csv",
    DATA_DIR / "ema_sma_pullback_research" / "ema_sma_pullback_trades.csv",
    DATA_DIR / "htf_fvg_rejections" / "confirmed_htf_fvg_rejection_events.csv",
    DATA_DIR / "event_engines" / "first_liquidity_sweep_trades.csv",
]

loaded = []

for file in CANDIDATE_FILES:
    if file.exists():
        print(f"Loading: {file}")
        x = pl.read_csv(file, try_parse_dates=True)
        x = x.with_columns([
            pl.lit(file.parent.name).alias("source_folder"),
            pl.lit(file.name).alias("source_file"),
        ])
        loaded.append(x)
    else:
        print(f"Missing, skipped: {file}")

if not loaded:
    raise SystemExit("No candidate files found.")

# ---------------------------------------------------
# Normalize candidate datasets
# ---------------------------------------------------

normalized = []

for x in loaded:
    cols = x.columns
    source = x["source_folder"][0]

    if "trade_date_ct" not in cols:
        continue

    base_cols = [
        "source_folder",
        "source_file",
        "trade_date_ct",
    ]

    if "entry_time" in cols:
        base_cols.append("entry_time")

    if "event_name" in cols:
        base_cols.append("event_name")
    else:
        x = x.with_columns(pl.lit(source).alias("event_name"))
        base_cols.append("event_name")

    if "direction" in cols:
        base_cols.append("direction")
    else:
        x = x.with_columns(pl.lit("unknown").alias("direction"))
        base_cols.append("direction")

    if "risk_points" in cols:
        base_cols.append("risk_points")

    # Pull points/R columns from each file type
    point_cols = [c for c in x.columns if c.startswith("points_")]
    r_cols = [c for c in x.columns if c.startswith("r_")]

    if "mechanical_points" in x.columns:
        point_cols.append("mechanical_points")

    if "mechanical_r" in x.columns:
        r_cols.append("mechanical_r")

    for pc in point_cols:
        suffix = pc.replace("points_", "").replace("mechanical_points", "mechanical")

        rc = None
        if pc.startswith("points_"):
            possible_r = "r_" + pc.replace("points_", "")
            if possible_r in x.columns:
                rc = possible_r
        elif pc == "mechanical_points" and "mechanical_r" in x.columns:
            rc = "mechanical_r"

        temp = x.select([
            *[pl.col(c) for c in base_cols if c in x.columns],
            pl.lit(pc).alias("result_type"),
            pl.col(pc).cast(pl.Float64).alias("points"),
            pl.col(rc).cast(pl.Float64).alias("r") if rc else pl.lit(None).cast(pl.Float64).alias("r"),
        ])

        normalized.append(temp)

all_trades = pl.concat(normalized, how="diagonal_relaxed")

all_trades = all_trades.drop_nulls(["points"])

all_trades = all_trades.with_columns([
    pl.col("trade_date_ct").cast(pl.Date),
    pl.col("trade_date_ct").dt.year().alias("year"),
    pl.col("trade_date_ct").dt.month().alias("month"),
    pl.col("trade_date_ct").dt.strftime("%Y-%m").alias("year_month"),
])

print(f"\nNormalized trade-result rows: {all_trades.height:,}")

# ---------------------------------------------------
# Candidate key
# ---------------------------------------------------

all_trades = all_trades.with_columns([
    (
        pl.col("source_folder") + pl.lit(" | ") +
        pl.col("source_file") + pl.lit(" | ") +
        pl.col("event_name") + pl.lit(" | ") +
        pl.col("direction") + pl.lit(" | ") +
        pl.col("result_type")
    ).alias("candidate")
])

# ---------------------------------------------------
# Base summary
# ---------------------------------------------------

base_summary = (
    all_trades
    .group_by("candidate")
    .agg([
        pl.len().alias("trades"),
        pl.col("points").mean().alias("avg_points"),
        pl.col("points").sum().alias("total_points"),
        (pl.col("points") > 0).mean().alias("win_rate"),
        pl.col("r").mean().alias("avg_r"),
        pl.col("r").sum().alias("total_r"),
        pl.col("risk_points").mean().alias("avg_risk"),
    ])
    .filter(pl.col("trades") >= 50)
)

# ---------------------------------------------------
# Year and month stability
# ---------------------------------------------------

year_stats = (
    all_trades
    .group_by(["candidate", "year"])
    .agg([
        pl.len().alias("year_trades"),
        pl.col("points").sum().alias("year_points"),
    ])
    .with_columns((pl.col("year_points") > 0).alias("positive_year"))
)

year_summary = (
    year_stats
    .group_by("candidate")
    .agg([
        pl.len().alias("years_tested"),
        pl.col("positive_year").sum().alias("positive_years"),
        pl.col("year_points").min().alias("worst_year_points"),
        pl.col("year_points").mean().alias("avg_year_points"),
    ])
)

month_stats = (
    all_trades
    .group_by(["candidate", "year_month"])
    .agg([
        pl.len().alias("month_trades"),
        pl.col("points").sum().alias("month_points"),
    ])
    .with_columns((pl.col("month_points") > 0).alias("positive_month"))
)

month_summary = (
    month_stats
    .group_by("candidate")
    .agg([
        pl.len().alias("months_tested"),
        pl.col("positive_month").sum().alias("positive_months"),
        pl.col("month_points").min().alias("worst_month_points"),
        pl.col("month_points").mean().alias("avg_month_points"),
    ])
)

summary = (
    base_summary
    .join(year_summary, on="candidate", how="left")
    .join(month_summary, on="candidate", how="left")
)

summary = summary.with_columns([
    (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
    (pl.col("positive_years") / pl.col("years_tested") * 100).round(2).alias("positive_year_rate_pct"),
    (pl.col("positive_months") / pl.col("months_tested") * 100).round(2).alias("positive_month_rate_pct"),
    pl.col("avg_points").round(2),
    pl.col("total_points").round(2),
    pl.col("avg_r").round(3),
    pl.col("total_r").round(2),
    pl.col("avg_risk").round(2),
    pl.col("worst_year_points").round(2),
    pl.col("worst_month_points").round(2),
])

# ---------------------------------------------------
# Drawdown and Monte Carlo-style shuffle
# ---------------------------------------------------

def max_drawdown(vals):
    equity = 0.0
    peak = 0.0
    max_dd = 0.0

    for v in vals:
        equity += float(v)
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return max_dd


def monte_carlo_dd(vals, n=300):
    if len(vals) < 50:
        return None

    dds = []
    vals = [float(v) for v in vals]

    for _ in range(n):
        sample = vals[:]
        random.shuffle(sample)
        dds.append(max_drawdown(sample))

    return {
        "mc_dd_median": statistics.median(dds),
        "mc_dd_90pct": sorted(dds)[int(len(dds) * 0.90)],
        "mc_dd_95pct": sorted(dds)[int(len(dds) * 0.95)],
    }


audit_rows = []

candidates = summary.select("candidate").to_series().to_list()

print(f"\nRunning drawdown/Monte Carlo audit on {len(candidates)} candidates...")

for i, cand in enumerate(candidates, start=1):
    if i % 25 == 0:
        print(f"Audited {i}/{len(candidates)} candidates...")

    x = (
        all_trades
        .filter(pl.col("candidate") == cand)
        .sort("trade_date_ct")
    )

    vals = x.select("points").to_series().to_list()

    dd = max_drawdown(vals)
    mc = monte_carlo_dd(vals, n=300)

    row = {
        "candidate": cand,
        "max_drawdown_points": dd,
    }

    if mc:
        row.update(mc)
    else:
        row.update({
            "mc_dd_median": None,
            "mc_dd_90pct": None,
            "mc_dd_95pct": None,
        })

    audit_rows.append(row)

audit_df = pl.DataFrame(audit_rows)

final = summary.join(audit_df, on="candidate", how="left")

final = final.with_columns([
    pl.col("max_drawdown_points").round(2),
    pl.col("mc_dd_median").round(2),
    pl.col("mc_dd_90pct").round(2),
    pl.col("mc_dd_95pct").round(2),

    (
        (pl.col("avg_points") * 10) +
        (pl.col("positive_year_rate_pct") * 0.5) +
        (pl.col("positive_month_rate_pct") * 0.2) +
        (pl.col("total_points") / 100) -
        (pl.col("max_drawdown_points") / 50)
    ).round(4).alias("robustness_score")
])

ranked = (
    final
    .filter(
        (pl.col("trades") >= 75) &
        (pl.col("avg_points") > 0) &
        (pl.col("positive_year_rate_pct") >= 55) &
        (pl.col("positive_month_rate_pct") >= 45)
    )
    .sort("robustness_score", descending=True)
)

print("\nTOP ROBUST CANDIDATES:")
print(
    ranked
    .select([
        "candidate",
        "trades",
        "win_rate_pct",
        "avg_points",
        "total_points",
        "avg_r",
        "total_r",
        "avg_risk",
        "positive_years",
        "years_tested",
        "positive_year_rate_pct",
        "positive_month_rate_pct",
        "worst_year_points",
        "worst_month_points",
        "max_drawdown_points",
        "mc_dd_90pct",
        "robustness_score",
    ])
    .head(100)
)

# ---------------------------------------------------
# Save
# ---------------------------------------------------

all_file = OUT_DIR / "candidate_robustness_all.csv"
ranked_file = OUT_DIR / "candidate_robustness_ranked.csv"
trades_file = OUT_DIR / "normalized_candidate_trade_results.csv"

final.write_csv(all_file)
ranked.write_csv(ranked_file)
all_trades.write_csv(trades_file)

print("\nSaved:")
print(all_file)
print(ranked_file)
print(trades_file)

print("\nFinished:", datetime.now())
print("DONE.")