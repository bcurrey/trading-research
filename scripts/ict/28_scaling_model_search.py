# 28_scaling_model_search.py
# Scale the valid no-leakage model from script 27.
# Base:
#   full_universe_trades
#   lower_wick_pct < q40
# Then search softer/stacked pre-entry regime filters only.

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
OUTDIR = ROOT / r"research_outputs\scaling_model_search"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "scaling_model_summary.csv"
YEARLY_OUT = OUTDIR / "scaling_model_yearly.csv"
BEST_TRADES_OUT = OUTDIR / "scaling_model_best_trades.csv"
BEST_LOSERS_OUT = OUTDIR / "scaling_model_best_losers.csv"

print("Loading full universe trades...")
trades = pl.read_csv(FULL_TRADES, try_parse_dates=True)
print(f"Trades loaded: {trades.height}")

print("Loading features...")
features = pl.read_parquet(FEATURES)
print(f"Feature rows loaded: {features.height:,}")

regime_cols = [
    "ts_ct",
    "atr_14", "atr_50",
    "dist_vwap", "dist_vwap_day",
    "dist_ema_20", "dist_ema_50",
    "dist_dol", "dist_pdh", "dist_pdl",
    "dist_or_high_30m", "dist_or_low_30m",
    "or_range_30m", "premarket_range", "prior_day_range",
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
    how="left",
).sort("entry_time")

q_lw40 = df["lower_wick_pct"].quantile(0.40)
base = df.filter(pl.col("lower_wick_pct") < q_lw40)

print(f"Base lower_wick<q40 trades: {base.height}")

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

def test(name: str, filt):
    filters[name] = filt
    x = base.filter(filt)
    if x.height < 20:
        return
    s = summarize(x, name)
    if s:
        rows.append(s)

# Baseline
test("base_lw_lt_q40", pl.lit(True))

# Core discovered filter variants.
if "dist_pdh" in base.columns:
    for q in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        v = base["dist_pdh"].quantile(q)
        test(f"dist_pdh_lt_q{int(q*100)}_{round(float(v),2)}", pl.col("dist_pdh") < v)

# DOL / VWAP / OR variants.
for col in [
    "dist_dol",
    "dist_vwap",
    "dist_vwap_day",
    "dist_or_low_30m",
    "dist_or_high_30m",
    "dist_pdl",
    "or_range_30m",
    "premarket_range",
    "prior_day_range",
    "atr_14",
    "atr_50",
    "rel_vol_20",
    "body_pct",
    "range_expansion_20",
    "risk_points",
]:
    if col not in base.columns:
        continue

    vals = base.select(pl.col(col)).drop_nulls()
    if vals.height < 20:
        continue

    for q in [0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75]:
        v = vals[col].quantile(q)
        test(f"{col}_lt_q{int(q*100)}_{round(float(v),2)}", pl.col(col) < v)
        test(f"{col}_gt_q{int(q*100)}_{round(float(v),2)}", pl.col(col) > v)

# Time filters.
test("hour8", pl.col("hour") == 8)
test("hour10", pl.col("hour") == 10)
test("minute_lte_45", pl.col("minute") <= 45)
test("minute_lte_50", pl.col("minute") <= 50)
test("minute_gte_35", pl.col("minute") >= 35)
test("not_1050_plus", ~((pl.col("hour") == 10) & (pl.col("minute") >= 50)))

# Event exclusion filters.
for c in [
    "has_high_impact_usd_event", "has_cpi", "has_fomc", "has_nfp", "has_ppi",
    "has_pmi_ism", "has_jobs", "has_fed_speech",
]:
    if c in base.columns:
        test(f"exclude_{c}", pl.col(c) == False)

# Stacked filters around the current best.
if "dist_pdh" in base.columns:
    pdh_q60 = base["dist_pdh"].quantile(0.60)
    pdh_q70 = base["dist_pdh"].quantile(0.70)
    pdh_q75 = base["dist_pdh"].quantile(0.75)

    if "rel_vol_20" in base.columns:
        rv_q50 = base["rel_vol_20"].quantile(0.50)
        rv_q60 = base["rel_vol_20"].quantile(0.60)
        rv_q75 = base["rel_vol_20"].quantile(0.75)

        test("pdh60_relvol_lt_q75", (pl.col("dist_pdh") < pdh_q60) & (pl.col("rel_vol_20") < rv_q75))
        test("pdh70_relvol_lt_q75", (pl.col("dist_pdh") < pdh_q70) & (pl.col("rel_vol_20") < rv_q75))
        test("pdh75_relvol_lt_q75", (pl.col("dist_pdh") < pdh_q75) & (pl.col("rel_vol_20") < rv_q75))
        test("pdh70_relvol_lt_q60", (pl.col("dist_pdh") < pdh_q70) & (pl.col("rel_vol_20") < rv_q60))
        test("pdh75_relvol_lt_q60", (pl.col("dist_pdh") < pdh_q75) & (pl.col("rel_vol_20") < rv_q60))

    if "body_pct" in base.columns:
        body_q40 = base["body_pct"].quantile(0.40)
        body_q50 = base["body_pct"].quantile(0.50)
        body_q60 = base["body_pct"].quantile(0.60)
        body_q75 = base["body_pct"].quantile(0.75)

        test("pdh60_body_gt_q40", (pl.col("dist_pdh") < pdh_q60) & (pl.col("body_pct") > body_q40))
        test("pdh70_body_gt_q40", (pl.col("dist_pdh") < pdh_q70) & (pl.col("body_pct") > body_q40))
        test("pdh75_body_gt_q40", (pl.col("dist_pdh") < pdh_q75) & (pl.col("body_pct") > body_q40))
        test("pdh70_body_gt_q50", (pl.col("dist_pdh") < pdh_q70) & (pl.col("body_pct") > body_q50))
        test("pdh75_body_gt_q50", (pl.col("dist_pdh") < pdh_q75) & (pl.col("body_pct") > body_q50))

    if "range_expansion_20" in base.columns:
        rng_q25 = base["range_expansion_20"].quantile(0.25)
        rng_q40 = base["range_expansion_20"].quantile(0.40)
        rng_q50 = base["range_expansion_20"].quantile(0.50)

        test("pdh60_range_gt_q25", (pl.col("dist_pdh") < pdh_q60) & (pl.col("range_expansion_20") > rng_q25))
        test("pdh70_range_gt_q25", (pl.col("dist_pdh") < pdh_q70) & (pl.col("range_expansion_20") > rng_q25))
        test("pdh75_range_gt_q25", (pl.col("dist_pdh") < pdh_q75) & (pl.col("range_expansion_20") > rng_q25))
        test("pdh70_range_gt_q40", (pl.col("dist_pdh") < pdh_q70) & (pl.col("range_expansion_20") > rng_q40))

    if "dist_or_low_30m" in base.columns:
        orlow_q40 = base["dist_or_low_30m"].quantile(0.40)
        orlow_q50 = base["dist_or_low_30m"].quantile(0.50)
        orlow_q60 = base["dist_or_low_30m"].quantile(0.60)

        test("pdh70_orlow_lt_q60", (pl.col("dist_pdh") < pdh_q70) & (pl.col("dist_or_low_30m") < orlow_q60))
        test("pdh75_orlow_lt_q60", (pl.col("dist_pdh") < pdh_q75) & (pl.col("dist_or_low_30m") < orlow_q60))
        test("pdh70_orlow_gt_q40", (pl.col("dist_pdh") < pdh_q70) & (pl.col("dist_or_low_30m") > orlow_q40))

summary = (
    pl.DataFrame(rows)
    .filter(pl.col("profit_factor").is_not_null())
    .sort(["profit_factor", "trades", "net_points"], descending=[True, True, True])
)

summary.write_csv(SUMMARY_OUT)

# Pick balanced: prefer 45+ trades PF > 6, otherwise best PF with 30+.
balanced = summary.filter((pl.col("trades") >= 45) & (pl.col("profit_factor") >= 6))
if balanced.height:
    best_label = balanced.sort(["profit_factor", "net_points"], descending=[True, True])["label"][0]
else:
    eligible = summary.filter(pl.col("trades") >= 30)
    best_label = eligible.sort(["profit_factor", "net_points"], descending=[True, True])["label"][0]

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

print("\nTOP SCALING MODELS")
print(summary.head(60))

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
