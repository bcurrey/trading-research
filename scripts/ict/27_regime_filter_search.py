# 27_regime_filter_search.py
# Search no-leakage regime filters to improve the current valid pre-entry model.
# Starting model from script 26:
# lower_wick_pct < q40 on full_universe_trades.csv

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

FULL_TRADES = ROOT / r"research_outputs\full_universe_scan\full_universe_trades.csv"
FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
OUTDIR = ROOT / r"research_outputs\regime_filter_search"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "regime_filter_summary.csv"
YEARLY_OUT = OUTDIR / "regime_filter_yearly.csv"
BEST_TRADES_OUT = OUTDIR / "regime_filter_best_trades.csv"
BEST_LOSERS_OUT = OUTDIR / "regime_filter_best_losers.csv"

print("Loading full universe trades...")
trades = pl.read_csv(FULL_TRADES, try_parse_dates=True)
print(f"Trades loaded: {trades.height}")

print("Loading features...")
features = pl.read_parquet(FEATURES)
print(f"Feature rows loaded: {features.height:,}")

# Add pre-entry regime context from feature row at signal_time.
regime_cols = [
    "ts_ct",
    "atr_14", "atr_50",
    "dist_vwap", "dist_vwap_day",
    "dist_ema_20", "dist_ema_50",
    "dist_dol", "dist_pdh", "dist_pdl",
    "dist_or_high_30m", "dist_or_low_30m",
    "or_range_30m",
    "premarket_range",
    "prior_day_range",
    "liquidity_sweep_reclaim_count_last_30m",
    "has_high_impact_usd_event",
    "has_cpi", "has_fomc", "has_nfp", "has_ppi",
    "has_pmi_ism", "has_jobs", "has_fed_speech",
]

regime_cols = [c for c in regime_cols if c in features.columns]

feat = features.select(regime_cols)

df = trades.join(
    feat,
    left_on="signal_time",
    right_on="ts_ct",
    how="left"
)

print(f"Joined rows: {df.height}")

# Start from script 26 valid best filter.
q40 = df["lower_wick_pct"].quantile(0.40)
base = df.filter(pl.col("lower_wick_pct") < q40).sort("entry_time")

print(f"Base valid model trades: {base.height}")

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
filters = {}

def add_filter(name, filt):
    filters[name] = filt
    x = base.filter(filt)
    if x.height >= 20:
        s = summarize(x, name)
        if s:
            rows.append(s)

# Baseline
filters["base_lower_wick_lt_q40"] = pl.lit(True)
rows.append(summarize(base, "base_lower_wick_lt_q40"))

# Hour split
add_filter("hour_8_only", pl.col("hour") == 8)
add_filter("hour_10_only", pl.col("hour") == 10)

# Minute filters
add_filter("minute_lte_45", pl.col("minute") <= 45)
add_filter("minute_lte_40", pl.col("minute") <= 40)
add_filter("minute_gte_35", pl.col("minute") >= 35)

# Event filters
for c in [
    "has_high_impact_usd_event",
    "has_cpi", "has_fomc", "has_nfp", "has_ppi",
    "has_pmi_ism", "has_jobs", "has_fed_speech",
]:
    if c in base.columns:
        add_filter(f"{c}_false", pl.col(c) == False)
        add_filter(f"{c}_true", pl.col(c) == True)

# Numeric regime filters.
num_cols = [
    "atr_14", "atr_50",
    "dist_vwap", "dist_vwap_day",
    "dist_ema_20", "dist_ema_50",
    "dist_dol", "dist_pdh", "dist_pdl",
    "dist_or_high_30m", "dist_or_low_30m",
    "or_range_30m",
    "premarket_range",
    "prior_day_range",
    "liquidity_sweep_reclaim_count_last_30m",
    "risk_points", "body_pct", "range_expansion_20", "rel_vol_20",
]

for c in num_cols:
    if c not in base.columns:
        continue

    vals = base.select(pl.col(c)).drop_nulls()
    if vals.height < 20:
        continue

    for q in [0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]:
        qv = vals[c].quantile(q)

        add_filter(f"{c}_gt_q{int(q*100)}_{round(float(qv),5)}", pl.col(c) > qv)
        add_filter(f"{c}_lt_q{int(q*100)}_{round(float(qv),5)}", pl.col(c) < qv)

# Two-factor filters from likely useful regime concepts.
def combo(name, f1, f2):
    add_filter(name, f1 & f2)

if "dist_vwap" in base.columns and "body_pct" in base.columns:
    combo(
        "deep_vwap_body_high",
        pl.col("dist_vwap") < base["dist_vwap"].quantile(0.50),
        pl.col("body_pct") > base["body_pct"].quantile(0.40),
    )

if "dist_or_low_30m" in base.columns and "body_pct" in base.columns:
    combo(
        "below_orlow_body_high",
        pl.col("dist_or_low_30m") < base["dist_or_low_30m"].quantile(0.50),
        pl.col("body_pct") > base["body_pct"].quantile(0.40),
    )

if "atr_14" in base.columns and "rel_vol_20" in base.columns:
    combo(
        "atr_high_relvol_low",
        pl.col("atr_14") > base["atr_14"].quantile(0.40),
        pl.col("rel_vol_20") < base["rel_vol_20"].quantile(0.75),
    )

if "liquidity_sweep_reclaim_count_last_30m" in base.columns and "body_pct" in base.columns:
    combo(
        "multi_sweep_body_high",
        pl.col("liquidity_sweep_reclaim_count_last_30m") >= 3,
        pl.col("body_pct") > base["body_pct"].quantile(0.40),
    )

summary = (
    pl.DataFrame(rows)
    .filter(pl.col("trades") >= 20)
    .sort(["profit_factor", "net_points", "trades"], descending=[True, True, True])
)

summary.write_csv(SUMMARY_OUT)

# Pick best non-null PF, >=30 preferred, otherwise >=20.
eligible_30 = summary.filter((pl.col("trades") >= 30) & (pl.col("profit_factor").is_not_null()))
eligible_20 = summary.filter((pl.col("trades") >= 20) & (pl.col("profit_factor").is_not_null()))

if eligible_30.height:
    best_label = eligible_30.sort(["profit_factor", "net_points"], descending=[True, True])["label"][0]
else:
    best_label = eligible_20.sort(["profit_factor", "net_points"], descending=[True, True])["label"][0]

best = base.filter(filters[best_label]).sort("entry_time")

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

print(f"\nBEST LABEL: {best_label}")

print("\nTOP REGIME FILTERS")
print(summary.head(50))

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
