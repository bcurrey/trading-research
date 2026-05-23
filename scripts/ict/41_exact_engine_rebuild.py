# 41_exact_engine_rebuild.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

FEATURES = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\exact_engine_rebuild"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading features...")
df = (
    pl.read_parquet(FEATURES)
    .sort("ts_ct")
    .with_row_index("idx")
)

rows = df.to_dicts()

trades = []

# -----------------------------------------
# REBUILD ORIGINAL ENGINE LOGIC
# -----------------------------------------
# THEORY:
# Stage 1:
#   aggressive exhaustion reclaim
#
# Stage 2:
#   stabilization continuation
#
# Entry:
#   NEXT BAR AFTER STAGE 2
#
# Short model:
#   fade reclaim continuation
# -----------------------------------------

for i in range(2, len(rows) - 120):

    try:

        b1 = rows[i - 1]
        b2 = rows[i]
        b3 = rows[i + 1]

        # -----------------------------
        # STAGE 1
        # exhaustion reclaim
        # -----------------------------

        s1 = (
            b1["dist_pdh"] < -78 and
            b1["dist_vwap"] < -20 and
            b1["lower_wick_pct"] > 0.20 and
            b1["range_expansion_20"] > 1.25
        )

        # -----------------------------
        # STAGE 2
        # stabilization continuation
        # -----------------------------

        s2 = (
            b2["body_pct"] > 0.45 and
            b2["body_pct"] < 0.85 and
            b2["lower_wick_pct"] < 0.15 and
            b2["rel_vol_20"] < 2.5
        )

        if not (s1 and s2):
            continue

        # -----------------------------
        # ENTRY BAR
        # -----------------------------

        entry = float(b3["open"])

        # ORIGINAL STYLE STOP
        stop = max(
            float(b1["high"]),
            float(b2["high"]),
            float(b3["high"])
        ) + 2.0

        risk = stop - entry

        if risk <= 0:
            continue

        if risk > 25:
            continue

        target = entry - (2.0 * risk)

        result = None
        exit_reason = None
        bars_held = None

        # -----------------------------
        # FORWARD TEST
        # -----------------------------

        for j in range(i + 1, min(i + 90, len(rows))):

            r = rows[j]

            high = float(r["high"])
            low = float(r["low"])

            if high >= stop:
                result = entry - stop
                exit_reason = "stop"
                bars_held = j - i
                break

            if low <= target:
                result = entry - target
                exit_reason = "target"
                bars_held = j - i
                break

        if result is None:

            last = rows[min(i + 89, len(rows) - 1)]

            result = entry - float(last["close"])
            exit_reason = "time"
            bars_held = 90

        trades.append({

            "signal_time": b2["ts_ct"],
            "entry_time": b3["ts_ct"],

            "year": b3["ts_ct"].year,

            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),

            "risk": round(risk, 2),

            "result_points": round(result, 2),

            "bars_held": bars_held,
            "exit_reason": exit_reason,

            "stage1_lower_wick_pct": b1["lower_wick_pct"],
            "stage1_range_expansion": b1["range_expansion_20"],

            "stage2_body_pct": b2["body_pct"],
            "stage2_lower_wick_pct": b2["lower_wick_pct"],

            "dist_pdh": b2["dist_pdh"],
            "dist_vwap": b2["dist_vwap"],

            "hour": b3["hour_ct"],
            "minute": b3["minute_ct"],
        })

    except:
        pass

out = pl.DataFrame(trades)

wins = out.filter(pl.col("result_points") > 0)
losses = out.filter(pl.col("result_points") <= 0)

gross_win = wins["result_points"].sum() if wins.height else 0
gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

eq = (
    out.sort("entry_time")
    .with_columns(pl.col("result_points").cum_sum().alias("equity"))
    .with_columns(pl.col("equity").cum_max().alias("peak"))
    .with_columns((pl.col("equity") - pl.col("peak")).alias("dd"))
)

summary = pl.DataFrame([{
    "trades": out.height,
    "wins": wins.height,
    "losses": losses.height,
    "winrate": round(100 * wins.height / out.height, 2) if out.height else 0,
    "net_points": round(out["result_points"].sum(), 2) if out.height else 0,
    "avg_trade": round(out["result_points"].mean(), 2) if out.height else 0,
    "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    "max_dd": round(eq["dd"].min(), 2) if out.height else 0,
}])

yearly = (
    out.group_by("year")
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

out.write_csv(OUTDIR / "exact_engine_trades.csv")
summary.write_csv(OUTDIR / "exact_engine_summary.csv")
yearly.write_csv(OUTDIR / "exact_engine_yearly.csv")

print(summary)

print("\nYEARLY")
print(yearly)

print(f"\nSaved: {OUTDIR}")

