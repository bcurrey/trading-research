# 36_two_stage_sequence_detector.py

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

OUTDIR = ROOT / r"research_outputs\two_stage_sequence_detector"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading features...")
df = pl.read_parquet(FEATURES)

df = (
    df
    .sort("ts_ct")
    .with_row_index("idx")
)

# Stage 1:
# aggressive reclaim / exhaustion bar
stage1 = (
    (pl.col("lower_wick_pct") > 0.20) &
    (pl.col("range_expansion_20") > 1.25) &
    (pl.col("dist_pdh") < -78) &
    (pl.col("dist_vwap") < -20)
)

# Stage 2:
# stabilization continuation bar
stage2 = (
    (pl.col("body_pct") > 0.45) &
    (pl.col("body_pct") < 0.90) &
    (pl.col("lower_wick_pct") < 0.18) &
    (pl.col("rel_vol_20") < 2.5)
)

signals = []

rows = df.to_dicts()

for i in range(1, len(rows)):

    r1 = rows[i - 1]
    r2 = rows[i]

    try:

        s1 = (
            r1["lower_wick_pct"] > 0.20 and
            r1["range_expansion_20"] > 1.25 and
            r1["dist_pdh"] < -78 and
            r1["dist_vwap"] < -20
        )

        s2 = (
            r2["body_pct"] > 0.45 and
            r2["body_pct"] < 0.90 and
            r2["lower_wick_pct"] < 0.18 and
            r2["rel_vol_20"] < 2.5
        )

        if s1 and s2:

            signals.append({
                "signal_time": r2["ts_ct"],
                "hour": r2["hour_ct"],
                "minute": r2["minute_ct"],

                "stage1_lower_wick_pct": r1["lower_wick_pct"],
                "stage1_range_expansion": r1["range_expansion_20"],

                "stage2_body_pct": r2["body_pct"],
                "stage2_lower_wick_pct": r2["lower_wick_pct"],

                "dist_pdh": r2["dist_pdh"],
                "dist_vwap": r2["dist_vwap"],
            })

    except:
        pass

out = pl.DataFrame(signals)

out.write_csv(OUTDIR / "two_stage_signals.csv")

print(out)

print(f"\nSignals found: {out.height}")
print(f"Saved: {OUTDIR / 'two_stage_signals.csv'}")
