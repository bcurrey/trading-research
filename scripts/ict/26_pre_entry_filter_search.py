# 26_pre_entry_filter_search.py
# Search valid pre-entry filters only. No MAE/MFE/result leakage.

from pathlib import Path
import itertools
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\full_universe_scan\full_universe_trades.csv"
OUTDIR = ROOT / r"research_outputs\pre_entry_filter_search"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "pre_entry_filter_summary.csv"
YEARLY_OUT = OUTDIR / "pre_entry_filter_yearly.csv"
BEST_TRADES_OUT = OUTDIR / "pre_entry_best_trades.csv"
BEST_LOSERS_OUT = OUTDIR / "pre_entry_best_losers.csv"

print("Loading full universe trades...")
df = pl.read_csv(INPUT, try_parse_dates=True)

print(f"Rows loaded: {df.height}")

# Valid pre-entry columns only.
valid_cols = [
    "hour",
    "minute",
    "risk_points",
    "lower_wick_pct",
    "body_pct",
    "range_expansion_20",
    "rel_vol_20",
]

missing = [c for c in valid_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.sort("entry_time")

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

rows = []

# Baselines
baselines = {
    "all_full_universe": pl.lit(True),
    "hour_8_only": pl.col("hour") == 8,
    "hour_10_only": pl.col("hour") == 10,
    "hour_8_10": pl.col("hour").is_in([8, 10]),
}

for name, filt in baselines.items():
    s = summarize(df.filter(filt), name)
    if s:
        rows.append(s)

# Quantile threshold search.
search_cols = [
    "risk_points",
    "lower_wick_pct",
    "body_pct",
    "range_expansion_20",
    "rel_vol_20",
]

thresholds = {}

for c in search_cols:
    thresholds[c] = {}
    for q in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]:
        thresholds[c][q] = df[c].quantile(q)

# Single filters.
single_filters = []

for c, qdict in thresholds.items():
    for q, v in qdict.items():
        single_filters.append((f"{c}_gt_q{int(q*100)}_{round(float(v),5)}", pl.col(c) > v))
        single_filters.append((f"{c}_lt_q{int(q*100)}_{round(float(v),5)}", pl.col(c) < v))

for name, filt in single_filters:
    x = df.filter(filt)
    if x.height >= 20:
        s = summarize(x, name)
        if s:
            rows.append(s)

# Multi-factor searches.
candidate_filters = []

# Conservative set: wick/body/range/volume/risk/hour combos.
for wick_q in [0.10, 0.20, 0.25, 0.30, 0.40]:
    for body_q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]:
        candidate_filters.append((
            f"wick_gt_q{int(wick_q*100)}_body_lt_q{int(body_q*100)}",
            (pl.col("lower_wick_pct") > thresholds["lower_wick_pct"][wick_q]) &
            (pl.col("body_pct") < thresholds["body_pct"][body_q])
        ))

for wick_q in [0.10, 0.20, 0.25, 0.30, 0.40]:
    for range_q in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        candidate_filters.append((
            f"wick_gt_q{int(wick_q*100)}_range_gt_q{int(range_q*100)}",
            (pl.col("lower_wick_pct") > thresholds["lower_wick_pct"][wick_q]) &
            (pl.col("range_expansion_20") > thresholds["range_expansion_20"][range_q])
        ))

for wick_q in [0.10, 0.20, 0.25, 0.30, 0.40]:
    for rel_q in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        candidate_filters.append((
            f"wick_gt_q{int(wick_q*100)}_relvol_gt_q{int(rel_q*100)}",
            (pl.col("lower_wick_pct") > thresholds["lower_wick_pct"][wick_q]) &
            (pl.col("rel_vol_20") > thresholds["rel_vol_20"][rel_q])
        ))

for body_q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]:
    for range_q in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        candidate_filters.append((
            f"body_lt_q{int(body_q*100)}_range_gt_q{int(range_q*100)}",
            (pl.col("body_pct") < thresholds["body_pct"][body_q]) &
            (pl.col("range_expansion_20") > thresholds["range_expansion_20"][range_q])
        ))

# 3-factor combos.
for wick_q in [0.10, 0.20, 0.25, 0.30]:
    for body_q in [0.60, 0.70, 0.75, 0.80]:
        for range_q in [0.10, 0.20, 0.25, 0.30]:
            candidate_filters.append((
                f"wick_q{int(wick_q*100)}_body_q{int(body_q*100)}_range_q{int(range_q*100)}",
                (pl.col("lower_wick_pct") > thresholds["lower_wick_pct"][wick_q]) &
                (pl.col("body_pct") < thresholds["body_pct"][body_q]) &
                (pl.col("range_expansion_20") > thresholds["range_expansion_20"][range_q])
            ))

for wick_q in [0.10, 0.20, 0.25, 0.30]:
    for body_q in [0.60, 0.70, 0.75, 0.80]:
        for rel_q in [0.10, 0.20, 0.25, 0.30]:
            candidate_filters.append((
                f"wick_q{int(wick_q*100)}_body_q{int(body_q*100)}_rel_q{int(rel_q*100)}",
                (pl.col("lower_wick_pct") > thresholds["lower_wick_pct"][wick_q]) &
                (pl.col("body_pct") < thresholds["body_pct"][body_q]) &
                (pl.col("rel_vol_20") > thresholds["rel_vol_20"][rel_q])
            ))

# Hour-specific versions of promising combos.
expanded = []

for name, filt in candidate_filters:
    expanded.append((name, filt))
    expanded.append((name + "_hour8", filt & (pl.col("hour") == 8)))
    expanded.append((name + "_hour10", filt & (pl.col("hour") == 10)))

for name, filt in expanded:
    x = df.filter(filt)

    # Need enough sample to care.
    if x.height < 20:
        continue

    s = summarize(x, name)
    if s:
        rows.append(s)

summary = (
    pl.DataFrame(rows)
    .filter(pl.col("trades") >= 20)
    .sort(["profit_factor", "net_points", "trades"], descending=[True, True, True])
)

summary.write_csv(SUMMARY_OUT)

# Pick realistic best: 50+ trades preferred; otherwise 35+.
eligible_50 = summary.filter((pl.col("trades") >= 50) & (pl.col("profit_factor").is_not_null()))
eligible_35 = summary.filter((pl.col("trades") >= 35) & (pl.col("profit_factor").is_not_null()))

if eligible_50.height:
    best_label = eligible_50.sort(["profit_factor", "net_points"], descending=[True, True])["label"][0]
elif eligible_35.height:
    best_label = eligible_35.sort(["profit_factor", "net_points"], descending=[True, True])["label"][0]
else:
    best_label = summary.sort(["profit_factor", "net_points"], descending=[True, True])["label"][0]

print(f"\nBEST LABEL: {best_label}")

# Rebuild by lookup.
filter_map = dict(expanded + single_filters + list(baselines.items()))
best = df.filter(filter_map[best_label]).sort("entry_time")

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

best.write_csv(BEST_TRADES_OUT)
yearly.write_csv(YEARLY_OUT)
best.filter(pl.col("result_points") <= 0).write_csv(BEST_LOSERS_OUT)

print("\nTOP PRE-ENTRY FILTERS")
print(summary.head(40))

print("\nBEST SUMMARY")
print(pl.DataFrame([summarize(best, best_label)]))

print("\nBEST YEARLY")
print(yearly)

print("\nBEST LOSERS")
print(best.filter(pl.col("result_points") <= 0))

print(f"\nSaved: {SUMMARY_OUT}")
print(f"Saved: {YEARLY_OUT}")
print(f"Saved best trades: {BEST_TRADES_OUT}")
print(f"Saved best losers: {BEST_LOSERS_OUT}")
