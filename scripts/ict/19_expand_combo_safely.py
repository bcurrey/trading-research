# 19_expand_combo_safely.py
# Safely expand combo model without killing PF

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\deep_retrace_v1_candidate\deep_retrace_v1_trades.csv"
OUTDIR = ROOT / r"research_outputs\expand_combo_safely"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "expand_combo_summary.csv"

df = pl.read_csv(INPUT, try_parse_dates=True)

models = {
    "combo_base": (
        (pl.col("hour") == 10) &
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20) &
        (pl.col("mae_points") > -10)
    ),

    "combo_plus_9ct": (
        (pl.col("hour").is_in([9, 10])) &
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20) &
        (pl.col("mae_points") > -10)
    ),

    "combo_relaxed_mae": (
        (pl.col("hour") == 10) &
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20) &
        (pl.col("mae_points") > -12)
    ),

    "combo_8_10": (
        (pl.col("hour").is_in([8, 10])) &
        (pl.col("risk_points") >= 6) &
        (pl.col("risk_points") <= 20) &
        (pl.col("mae_points") > -10)
    ),

    "combo_high_mfe": (
        (pl.col("hour") == 10) &
        (pl.col("mfe_points") >= 20)
    ),

    "combo_large_avg_trade": (
        (pl.col("hour") == 10) &
        (pl.col("result_points") >= 10)
    ),
}

rows = []

for name, filt in models.items():

    x = df.filter(filt).sort("entry_time")

    if x.height == 0:
        continue

    wins = x.filter(pl.col("result_points") > 0)
    losses = x.filter(pl.col("result_points") <= 0)

    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

    eq = (
        x.with_columns(
            pl.col("result_points").cum_sum().alias("equity")
        )
        .with_columns(
            pl.col("equity").cum_max().alias("peak")
        )
        .with_columns(
            (pl.col("equity") - pl.col("peak")).alias("dd")
        )
    )

    rows.append({
        "model": name,
        "trades": x.height,
        "wins": wins.height,
        "winrate": round(100 * wins.height / x.height, 2),
        "net_points": round(x["result_points"].sum(), 2),
        "avg_trade": round(x["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": round(eq["dd"].min(), 2),
    })

summary = (
    pl.DataFrame(rows)
    .sort(
        ["profit_factor", "avg_trade"],
        descending=[True, True]
    )
)

summary.write_csv(SUMMARY_OUT)

print(summary)

print(f"\nSaved: {SUMMARY_OUT}")
