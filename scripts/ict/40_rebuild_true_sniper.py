# 40_rebuild_true_sniper.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

TRADES = ROOT / r"research_outputs\two_stage_signal_backtest\two_stage_backtest_trades.csv"

OUTDIR = ROOT / r"research_outputs\rebuild_true_sniper"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading trades...")
df = pl.read_csv(TRADES, try_parse_dates=True)

# -----------------------------------
# REBUILD ORIGINAL ELITE LOGIC
# -----------------------------------

models = {}

# Base original concepts
models["base_pdh_vwap"] = (
    (pl.col("dist_pdh") < -78) &
    (pl.col("dist_vwap") < -20)
)

# Add reclaim stabilization
models["reclaim_stabilization"] = (
    (pl.col("dist_pdh") < -78) &
    (pl.col("dist_vwap") < -20) &
    (pl.col("stage1_lower_wick_pct") > 0.25) &
    (pl.col("stage2_lower_wick_pct") < 0.15)
)

# Add calmer continuation
models["controlled_continuation"] = (
    (pl.col("dist_pdh") < -78) &
    (pl.col("dist_vwap") < -20) &
    (pl.col("stage2_body_pct").is_between(0.45, 0.80)) &
    (pl.col("stage2_lower_wick_pct") < 0.12)
)

# Add exhaustion + stabilization
models["exhaustion_then_control"] = (
    (pl.col("dist_pdh") < -90) &
    (pl.col("dist_vwap") < -25) &
    (pl.col("stage1_lower_wick_pct") > 0.30) &
    (pl.col("stage1_range_expansion") > 1.5) &
    (pl.col("stage2_body_pct").is_between(0.45, 0.75)) &
    (pl.col("stage2_lower_wick_pct") < 0.12)
)

# Time-sensitive versions
models["timing_sensitive"] = (
    (pl.col("dist_pdh") < -78) &
    (pl.col("dist_vwap") < -20) &
    (pl.col("minute") <= 50) &
    (pl.col("stage2_lower_wick_pct") < 0.12)
)

# Very selective
models["true_sniper_candidate"] = (
    (pl.col("dist_pdh") < -100) &
    (pl.col("dist_vwap") < -30) &
    (pl.col("stage1_lower_wick_pct") > 0.35) &
    (pl.col("stage1_range_expansion") > 1.75) &
    (pl.col("stage2_body_pct").is_between(0.50, 0.75)) &
    (pl.col("stage2_lower_wick_pct") < 0.10) &
    (pl.col("minute") <= 50)
)

def summarize(x, label):

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
        "model": label,
        "trades": x.height,
        "wins": wins.height,
        "losses": losses.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": round(eq["dd"].min(), 2),
    }

summary_rows = []
model_frames = {}

for name, filt in models.items():

    x = df.filter(filt).sort("entry_time")

    model_frames[name] = x

    s = summarize(x, name)

    if s:
        summary_rows.append(s)

summary = (
    pl.DataFrame(summary_rows)
    .sort(
        ["profit_factor", "avg_trade", "winrate"],
        descending=[True, True, True]
    )
)

print(summary)

summary.write_csv(OUTDIR / "true_sniper_summary.csv")

for name, frame in model_frames.items():

    if frame.height == 0:
        continue

    frame.write_csv(OUTDIR / f"{name}_trades.csv")

    yearly = (
        frame.group_by("year")
        .agg(
            pl.len().alias("trades"),
            (pl.col("result_points") > 0).sum().alias("wins"),
            (pl.col("result_points") <= 0).sum().alias("losses"),
            pl.col("result_points").sum().round(2).alias("net_points"),
            pl.col("result_points").mean().round(2).alias("avg_trade"),
        )
        .with_columns(
            (100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate")
        )
        .sort("year")
    )

    yearly.write_csv(OUTDIR / f"{name}_yearly.csv")

print(f"\nSaved: {OUTDIR}")

