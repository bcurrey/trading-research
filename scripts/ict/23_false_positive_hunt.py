# 23_false_positive_hunt.py
# Hunt for overfit / false positives in lower_wick_gt_q25 institutional model.

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\combo_8_10_feature_audit\combo_8_10_trades_with_features.csv"
OUTDIR = ROOT / r"research_outputs\false_positive_hunt"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "false_positive_threshold_summary.csv"
YEARLY_OUT = OUTDIR / "false_positive_yearly.csv"
TRAIN_TEST_OUT = OUTDIR / "false_positive_train_test.csv"
BEST_TRADES_OUT = OUTDIR / "false_positive_best_trades.csv"

print("Loading combo feature trades...")
df = pl.read_csv(INPUT, try_parse_dates=True)

print(f"Rows loaded: {df.height}")

base = (
    df.filter(
        (pl.col("hour").is_in([8, 10])) &
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20) &
        (pl.col("mae_points") > -10)
    )
    .sort("entry_time")
)

print(f"Base trades: {base.height}")

def stats(x: pl.DataFrame, label: str):
    if x.height == 0:
        return None

    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)

    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

    eq = (
        x.sort("entry_time")
        .with_columns(pl.col("result_points").cum_sum().alias("equity"))
        .with_columns(pl.col("equity").cum_max().alias("peak"))
        .with_columns((pl.col("equity") - pl.col("peak")).alias("dd"))
    )

    return {
        "label": label,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": round(eq["dd"].min(), 2),
        "first_trade": str(x["entry_time"].min()),
        "last_trade": str(x["entry_time"].max()),
    }

rows = []

# Threshold sensitivity around the q25 winner filter.
for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    qv = base["lower_wick_pct"].quantile(q)
    x = base.filter(pl.col("lower_wick_pct") > qv)
    s = stats(x, f"lower_wick_gt_q{int(q*100)}_{round(float(qv), 5)}")
    if s:
        rows.append(s)

# Reverse condition: check if weak lower wick is where losses live.
for q in [0.25, 0.50, 0.75]:
    qv = base["lower_wick_pct"].quantile(q)
    x = base.filter(pl.col("lower_wick_pct") <= qv)
    s = stats(x, f"lower_wick_lte_q{int(q*100)}_{round(float(qv), 5)}")
    if s:
        rows.append(s)

# Combo robustness with related filters.
tests = {
    "q25_only": pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25),
    "q20_only": pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.20),
    "q30_only": pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.30),
    "q25_plus_body_lt_q75": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < base["body_pct"].quantile(0.75))
    ),
    "q25_plus_range_gt_q25": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("range_expansion_20") > base["range_expansion_20"].quantile(0.25))
    ),
    "q25_plus_relvol_gt_q25": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("rel_vol_20") > base["rel_vol_20"].quantile(0.25))
    ),
    "q25_plus_deep_vwap": (
        (pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25)) &
        (pl.col("dist_vwap") < base["dist_vwap"].quantile(0.75))
    ),
}

for name, filt in tests.items():
    x = base.filter(filt)
    s = stats(x, name)
    if s:
        rows.append(s)

summary = (
    pl.DataFrame(rows)
    .sort(["losses", "trades", "avg_trade"], descending=[False, True, True])
)

summary.write_csv(SUMMARY_OUT)

# Pick candidate: q20/q25 if both clean; otherwise lowest losses with >=30 trades.
eligible = summary.filter(pl.col("trades") >= 30)
best_label = eligible.sort(["losses", "avg_trade"], descending=[False, True])["label"][0]

print(f"\nBEST LABEL: {best_label}")

# Rebuild best filter based on label prefix.
if best_label.startswith("lower_wick_gt_q"):
    q_num = int(best_label.split("_q")[1].split("_")[0])
    q = q_num / 100
    qv = base["lower_wick_pct"].quantile(q)
    best = base.filter(pl.col("lower_wick_pct") > qv)
elif best_label == "q25_only":
    best = base.filter(pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25))
elif best_label == "q20_only":
    best = base.filter(pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.20))
elif best_label == "q30_only":
    best = base.filter(pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.30))
else:
    best = base.filter(pl.col("lower_wick_pct") > base["lower_wick_pct"].quantile(0.25))

yearly = (
    best.group_by("year")
    .agg(
        pl.len().alias("trades"),
        (pl.col("result_points") > 0).sum().alias("wins"),
        (pl.col("result_points") <= 0).sum().alias("losses"),
        pl.col("result_points").sum().round(2).alias("net_points"),
        pl.col("result_points").mean().round(2).alias("avg_trade"),
    )
    .with_columns((100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"))
    .sort("year")
)

# Train/test splits
train_test_rows = []

splits = [
    ("train_2019_2023", 2019, 2023),
    ("test_2024_2026", 2024, 2026),
    ("pre_2022", 2019, 2021),
    ("stress_2022", 2022, 2022),
    ("post_2022", 2023, 2026),
    ("recent_2025_2026", 2025, 2026),
]

for name, y1, y2 in splits:
    x = best.filter((pl.col("year") >= y1) & (pl.col("year") <= y2))
    s = stats(x, name)
    if s:
        train_test_rows.append(s)

train_test = pl.DataFrame(train_test_rows)

yearly.write_csv(YEARLY_OUT)
train_test.write_csv(TRAIN_TEST_OUT)
best.write_csv(BEST_TRADES_OUT)

print("\nTHRESHOLD SUMMARY")
print(summary)

print("\nBEST YEARLY")
print(yearly)

print("\nTRAIN TEST")
print(train_test)

print(f"\nSaved: {SUMMARY_OUT}")
print(f"Saved: {YEARLY_OUT}")
print(f"Saved: {TRAIN_TEST_OUT}")
print(f"Saved: {BEST_TRADES_OUT}")
