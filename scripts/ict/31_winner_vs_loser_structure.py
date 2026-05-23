# 31_winner_vs_loser_structure.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\chart_review_builder\chart_review_summary.csv"
OUTDIR = ROOT / r"research_outputs\winner_vs_loser_structure"

OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading chart review summary...")
df = pl.read_csv(INPUT, try_parse_dates=True)

print(f"Rows loaded: {df.height}")

df = df.with_columns(
    pl.when(pl.col("result_points") > 0)
    .then(pl.lit("WIN"))
    .otherwise(pl.lit("LOSS"))
    .alias("outcome")
)

numeric_cols = [
    c for c, d in df.schema.items()
    if d in [
        pl.Float64, pl.Float32,
        pl.Int64, pl.Int32,
        pl.UInt64, pl.UInt32
    ]
]

rows = []

for c in numeric_cols:

    if c in ["result_points"]:
        continue

    try:
        win_mean = df.filter(pl.col("outcome") == "WIN")[c].mean()
        loss_mean = df.filter(pl.col("outcome") == "LOSS")[c].mean()

        rows.append({
            "feature": c,
            "win_mean": round(float(win_mean), 4) if win_mean is not None else None,
            "loss_mean": round(float(loss_mean), 4) if loss_mean is not None else None,
            "difference": round(float(win_mean - loss_mean), 4)
            if win_mean is not None and loss_mean is not None else None,
            "abs_difference": round(abs(float(win_mean - loss_mean)), 4)
            if win_mean is not None and loss_mean is not None else None,
        })

    except:
        pass

compare = (
    pl.DataFrame(rows)
    .sort("abs_difference", descending=True)
)

compare.write_csv(OUTDIR / "winner_vs_loser_feature_diff.csv")

print("\nTOP DIFFERENCES")
print(compare.head(40))

print(f"\nSaved: {OUTDIR / 'winner_vs_loser_feature_diff.csv'}")
