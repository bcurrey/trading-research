# 25_subset_delta_analysis.py
# Compare original clean combo_8_10 subset vs noisy full-universe expansion.

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

ORIGINAL = ROOT / r"research_outputs\combo_8_10_feature_audit\combo_8_10_trades_with_features.csv"
FULL = ROOT / r"research_outputs\full_universe_scan\full_universe_trades.csv"
OUTDIR = ROOT / r"research_outputs\subset_delta_analysis"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "subset_delta_summary.csv"
FEATURE_COMPARE_OUT = OUTDIR / "original_vs_extra_feature_compare.csv"
FILTER_OUT = OUTDIR / "recover_clean_subset_filters.csv"
BEST_TRADES_OUT = OUTDIR / "recovered_best_trades.csv"
EXTRA_TRADES_OUT = OUTDIR / "extra_noisy_trades.csv"

print("Loading original combo trades...")
orig = pl.read_csv(ORIGINAL, try_parse_dates=True)

print("Loading full universe trades...")
full = pl.read_csv(FULL, try_parse_dates=True)

print(f"Original rows: {orig.height}")
print(f"Full rows: {full.height}")

# Normalize join keys.
orig_keys = orig.select(["entry_time"]).unique()
full = full.with_columns(
    pl.col("entry_time").cast(pl.Datetime)
)

orig = orig.with_columns(
    pl.col("entry_time").cast(pl.Datetime)
)

full_labeled = (
    full.join(
        orig.select(["entry_time"]).unique().with_columns(pl.lit(1).alias("in_original")),
        on="entry_time",
        how="left"
    )
    .with_columns(
        pl.col("in_original").fill_null(0),
        pl.when(pl.col("in_original") == 1)
        .then(pl.lit("ORIGINAL"))
        .otherwise(pl.lit("EXTRA"))
        .alias("subset")
    )
)

extra = full_labeled.filter(pl.col("subset") == "EXTRA")

print(f"Extra rows: {extra.height}")

def summarize(x: pl.DataFrame, label: str):
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

summary_rows = [
    summarize(full_labeled, "FULL_131"),
    summarize(full_labeled.filter(pl.col("subset") == "ORIGINAL"), "ORIGINAL_MATCHED"),
    summarize(extra, "EXTRA_ONLY"),
    summarize(extra.filter(pl.col("result_points") > 0), "EXTRA_WINNERS"),
    summarize(extra.filter(pl.col("result_points") <= 0), "EXTRA_LOSERS"),
]
summary = pl.DataFrame([x for x in summary_rows if x is not None])

# Feature comparison original vs extra.
numeric_cols = [
    c for c, d in full_labeled.schema.items()
    if d in [
        pl.Float64, pl.Float32,
        pl.Int64, pl.Int32, pl.Int16, pl.Int8,
        pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
    ]
    and c not in ["in_original"]
]

compare_rows = []
for c in numeric_cols:
    try:
        o_mean = full_labeled.filter(pl.col("subset") == "ORIGINAL")[c].mean()
        e_mean = full_labeled.filter(pl.col("subset") == "EXTRA")[c].mean()
        ew_mean = full_labeled.filter((pl.col("subset") == "EXTRA") & (pl.col("result_points") > 0))[c].mean()
        el_mean = full_labeled.filter((pl.col("subset") == "EXTRA") & (pl.col("result_points") <= 0))[c].mean()

        if o_mean is None or e_mean is None:
            continue

        compare_rows.append({
            "feature": c,
            "orig_mean": round(float(o_mean), 5),
            "extra_mean": round(float(e_mean), 5),
            "extra_win_mean": round(float(ew_mean), 5) if ew_mean is not None else None,
            "extra_loss_mean": round(float(el_mean), 5) if el_mean is not None else None,
            "extra_minus_orig": round(float(e_mean - o_mean), 5),
            "abs_extra_minus_orig": round(abs(float(e_mean - o_mean)), 5),
        })
    except:
        pass

feature_compare = (
    pl.DataFrame(compare_rows)
    .sort("abs_extra_minus_orig", descending=True)
)

# Try recovery filters on full universe.
candidate_rows = []

def evaluate(name: str, filt):
    x = full_labeled.filter(filt)

    if x.height == 0:
        return

    s = summarize(x, name)
    if s:
        candidate_rows.append(s)

# Base columns to test.
test_cols = [
    "lower_wick_pct",
    "body_pct",
    "range_expansion_20",
    "rel_vol_20",
    "risk_points",
    "mfe_points",
    "mae_points",
]

for c in test_cols:
    if c not in full_labeled.columns:
        continue

    vals = full_labeled.select(pl.col(c)).drop_nulls()

    if vals.height < 20:
        continue

    for q in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]:
        qv = vals[c].quantile(q)

        evaluate(f"{c}_gt_q{int(q*100)}_{round(float(qv),5)}", pl.col(c) > qv)
        evaluate(f"{c}_lt_q{int(q*100)}_{round(float(qv),5)}", pl.col(c) < qv)

# Multi-factor filters based on observed likely original subset differences.
if all(c in full_labeled.columns for c in ["lower_wick_pct", "body_pct"]):
    evaluate(
        "wick_gt_q20_body_lt_q75",
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.20)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75))
    )

    evaluate(
        "wick_gt_q25_body_lt_q75",
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75))
    )

    evaluate(
        "wick_gt_q30_body_lt_q75",
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.30)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75))
    )

if all(c in full_labeled.columns for c in ["lower_wick_pct", "body_pct", "range_expansion_20"]):
    evaluate(
        "wick_body_range_v1",
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.20)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75)) &
        (pl.col("range_expansion_20") > full_labeled["range_expansion_20"].quantile(0.25))
    )

    evaluate(
        "wick_body_range_v2",
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.70)) &
        (pl.col("range_expansion_20") > full_labeled["range_expansion_20"].quantile(0.25))
    )

    evaluate(
        "wick_body_range_strict",
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.30)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.60)) &
        (pl.col("range_expansion_20") > full_labeled["range_expansion_20"].quantile(0.30))
    )

if all(c in full_labeled.columns for c in ["lower_wick_pct", "body_pct", "rel_vol_20"]):
    evaluate(
        "wick_body_relvol",
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75)) &
        (pl.col("rel_vol_20") > full_labeled["rel_vol_20"].quantile(0.25))
    )

filter_df = (
    pl.DataFrame(candidate_rows)
    .filter(pl.col("trades") >= 30)
    .sort(["profit_factor", "net_points", "trades"], descending=[True, True, True])
)

best_label = filter_df["label"][0]
print(f"\nBEST RECOVERY FILTER: {best_label}")

# Recreate best roughly by matching exact evaluated label if possible.
# Simpler: choose top row summary only and save all rows; deeper exact export later.
best_trades = full_labeled

if best_label.startswith("lower_wick_pct_gt_q"):
    q = int(best_label.split("_q")[1].split("_")[0]) / 100
    best_trades = full_labeled.filter(pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(q))
elif best_label.startswith("lower_wick_pct_lt_q"):
    q = int(best_label.split("_q")[1].split("_")[0]) / 100
    best_trades = full_labeled.filter(pl.col("lower_wick_pct") < full_labeled["lower_wick_pct"].quantile(q))
elif best_label == "wick_gt_q20_body_lt_q75":
    best_trades = full_labeled.filter(
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.20)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75))
    )
elif best_label == "wick_gt_q25_body_lt_q75":
    best_trades = full_labeled.filter(
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75))
    )
elif best_label == "wick_gt_q30_body_lt_q75":
    best_trades = full_labeled.filter(
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.30)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75))
    )
elif best_label == "wick_body_range_v1":
    best_trades = full_labeled.filter(
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.20)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75)) &
        (pl.col("range_expansion_20") > full_labeled["range_expansion_20"].quantile(0.25))
    )
elif best_label == "wick_body_range_v2":
    best_trades = full_labeled.filter(
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.70)) &
        (pl.col("range_expansion_20") > full_labeled["range_expansion_20"].quantile(0.25))
    )
elif best_label == "wick_body_range_strict":
    best_trades = full_labeled.filter(
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.30)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.60)) &
        (pl.col("range_expansion_20") > full_labeled["range_expansion_20"].quantile(0.30))
    )
elif best_label == "wick_body_relvol":
    best_trades = full_labeled.filter(
        (pl.col("lower_wick_pct") > full_labeled["lower_wick_pct"].quantile(0.25)) &
        (pl.col("body_pct") < full_labeled["body_pct"].quantile(0.75)) &
        (pl.col("rel_vol_20") > full_labeled["rel_vol_20"].quantile(0.25))
    )

summary.write_csv(SUMMARY_OUT)
feature_compare.write_csv(FEATURE_COMPARE_OUT)
filter_df.write_csv(FILTER_OUT)
best_trades.write_csv(BEST_TRADES_OUT)
extra.write_csv(EXTRA_TRADES_OUT)

print("\nSUBSET SUMMARY")
print(summary)

print("\nTOP ORIGINAL VS EXTRA FEATURE DIFFS")
print(feature_compare.head(30))

print("\nTOP RECOVERY FILTERS")
print(filter_df.head(40))

print("\nBEST TRADES SUMMARY")
print(pl.DataFrame([summarize(best_trades, best_label)]))

print(f"\nSaved: {SUMMARY_OUT}")
print(f"Saved: {FEATURE_COMPARE_OUT}")
print(f"Saved: {FILTER_OUT}")
print(f"Saved: {BEST_TRADES_OUT}")
print(f"Saved: {EXTRA_TRADES_OUT}")
