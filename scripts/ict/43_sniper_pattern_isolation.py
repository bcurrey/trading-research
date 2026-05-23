# 43_sniper_pattern_isolation.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

SNIPER = ROOT / r"research_outputs\trace_sniper_source\sniper_with_features.csv"
OUTDIR = ROOT / r"research_outputs\sniper_pattern_isolation"

OUTDIR.mkdir(parents=True, exist_ok=True)

df = pl.read_csv(SNIPER, try_parse_dates=True)

print(f"Sniper trades loaded: {df.height}")

features = [
    "hour", "minute",
    "risk_points", "result_points",
    "body_pct", "lower_wick_pct", "upper_wick_pct",
    "range_expansion_20", "rel_vol_20",
    "dist_pdh", "dist_pdl", "dist_dol", "dist_vwap",
    "dist_or_high_30m", "dist_or_low_30m",
    "bear_fvg", "bull_fvg",
    "bear_displacement", "bull_displacement",
    "any_liquidity_sweep_reclaim_last_30m",
    "liquidity_sweep_reclaim_count_last_30m",
]

features = [c for c in features if c in df.columns]

rows = []

for c in features:
    try:
        rows.append({
            "feature": c,
            "mean": round(float(df[c].mean()), 5),
            "median": round(float(df[c].median()), 5),
            "min": round(float(df[c].min()), 5),
            "max": round(float(df[c].max()), 5),
        })
    except:
        pass

summary = pl.DataFrame(rows)

# binary occurrence stats
binary_cols = [
    c for c in [
        "bear_fvg", "bull_fvg",
        "bear_displacement", "bull_displacement",
        "any_liquidity_sweep_reclaim_last_30m",
    ] if c in df.columns
]

binary_rows = []

for c in binary_cols:
    binary_rows.append({
        "feature": c,
        "true_count": int(df.filter(pl.col(c) == True).height),
        "false_count": int(df.filter(pl.col(c) == False).height),
        "true_pct": round(100 * df.filter(pl.col(c) == True).height / df.height, 2),
    })

binary = pl.DataFrame(binary_rows)

# time distribution
time = (
    df.group_by(["hour", "minute"])
    .agg(
        pl.len().alias("trades"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .sort(["hour", "minute"])
)

yearly = (
    df.group_by("year")
    .agg(
        pl.len().alias("trades"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .sort("year")
)

summary.write_csv(OUTDIR / "sniper_feature_summary.csv")
binary.write_csv(OUTDIR / "sniper_binary_summary.csv")
time.write_csv(OUTDIR / "sniper_time_distribution.csv")
yearly.write_csv(OUTDIR / "sniper_yearly.csv")
df.write_csv(OUTDIR / "sniper_trades.csv")

print("\nFEATURE SUMMARY")
print(summary)

print("\nBINARY SUMMARY")
print(binary)

print("\nTIME DISTRIBUTION")
print(time)

print("\nYEARLY")
print(yearly)

print(f"\nSaved: {OUTDIR}")
