from pathlib import Path
import polars as pl

OUT_DIR = Path(r"D:\TradingResearch\research_outputs")
SCRIPT_NAME = "32_pine_signal_export_builder"

TRADES_IN = OUT_DIR / "30_orb_retest_quality_refinement_trades.csv"

PINE_RULES_OUT = OUT_DIR / f"{SCRIPT_NAME}_rules.txt"
SIGNALS_OUT = OUT_DIR / f"{SCRIPT_NAME}_vwap_align_signals.csv"
RECENT_SIGNALS_OUT = OUT_DIR / f"{SCRIPT_NAME}_recent_200_signals.csv"
SUMMARY_OUT = OUT_DIR / f"{SCRIPT_NAME}_summary.csv"

MODELS = ["VWAP_ALIGN", "VWAP_EMA20_ALIGN"]

print("Loading trades from script 30...")
df = pl.read_csv(TRADES_IN)

df = (
    df.filter(pl.col("model").is_in(MODELS))
    .with_columns([
        pl.col("trade_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
        pl.col("year").cast(pl.Int64).alias("year_int"),
    ])
    .sort(["model", "date", "entry_time"])
)

summary = (
    df.group_by("model")
    .agg([
        pl.len().alias("trades"),
        (pl.col("pnl_points") > 0).mean().round(4).alias("winrate"),
        pl.col("pnl_points").sum().round(2).alias("net_points"),
        pl.col("pnl_points").mean().round(2).alias("avg_trade"),
        pl.when(pl.col("pnl_points") > 0).then(pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_win"),
        pl.when(pl.col("pnl_points") < 0).then(-pl.col("pnl_points")).otherwise(0).sum().round(2).alias("gross_loss"),
    ])
    .with_columns([
        (pl.col("gross_win") / pl.col("gross_loss")).round(3).alias("profit_factor")
    ])
    .sort("profit_factor", descending=True)
)

recent = (
    df.sort(["model", "date", "entry_time"])
    .with_columns([
        pl.int_range(pl.len()).over("model").alias("idx"),
        pl.len().over("model").alias("n"),
    ])
    .filter(pl.col("idx") >= pl.col("n") - 200)
)

signals = df.select([
    "model",
    "trade_date",
    "year",
    "month",
    "direction",
    "break_time",
    "entry_time",
    "exit_time",
    "or_high",
    "or_low",
    "or_range",
    "entry",
    "stop_price",
    "target_price",
    "outcome",
    "pnl_points",
    "break_above_vwap",
    "break_below_vwap",
    "break_above_ema20",
    "break_below_ema20",
    "break_displacement",
    "rel_vol_20",
    "atr_14",
])

rules = """
ORB RETEST PRODUCTION CANDIDATE — PINE REPLICATION RULES

PRIMARY MODEL TO BUILD FIRST:
VWAP_ALIGN

BACKTEST SOURCE:
D:\\TradingResearch\\research_outputs\\30_orb_retest_quality_refinement_trades.csv

TIMEZONE:
America/Chicago

CHART:
NQ / MNQ 1-minute chart.

OPENING RANGE:
8:30 AM CT through 8:59 AM CT.
OR High = highest high during opening range.
OR Low = lowest low during opening range.

TRADE WINDOW:
9:00 AM CT through 11:30 AM CT.

ONE TRADE:
One trade max per day.

LONG SIGNAL:
1. After OR completes, price breaks above OR High.
2. Breakout bar closes above VWAP.
3. Wait up to 30 bars for retest.
4. Retest condition:
   - low <= OR High + 4 points
   - close >= OR High
5. Entry is OR High.
6. Stop = entry - 20 points.
7. Target = entry + 20 points.

SHORT SIGNAL:
1. After OR completes, price breaks below OR Low.
2. Breakout bar closes below VWAP.
3. Wait up to 30 bars for retest.
4. Retest condition:
   - high >= OR Low - 4 points
   - close <= OR Low
5. Entry is OR Low.
6. Stop = entry + 20 points.
7. Target = entry - 20 points.

VWAP_EMA20_ALIGN VERSION:
Same as VWAP_ALIGN, but adds:
LONG breakout close must be above EMA20.
SHORT breakout close must be below EMA20.

IMPORTANT:
This indicator should not repaint.
OR levels lock after 8:59 AM CT.
Signals only appear after retest confirmation.
No same-bar TP/SL assumption is used in Pine; Pine indicator only marks signal/entry/SL/TP levels.

NEXT VALIDATION STEP:
Load the recent_200_signals CSV.
Compare plotted Pine signals against these dates/times.
"""

signals.write_csv(SIGNALS_OUT)
recent.write_csv(RECENT_SIGNALS_OUT)
summary.write_csv(SUMMARY_OUT)
PINE_RULES_OUT.write_text(rules.strip(), encoding="utf-8")

print("\nSUMMARY")
print(summary)

print(f"\nSaved rules: {PINE_RULES_OUT}")
print(f"Saved all signals: {SIGNALS_OUT}")
print(f"Saved recent signals: {RECENT_SIGNALS_OUT}")
print(f"Saved summary: {SUMMARY_OUT}")
