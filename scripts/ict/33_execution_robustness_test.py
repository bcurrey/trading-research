# 33_execution_robustness_test.py

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
BEST = ROOT / r"research_outputs\final_model_candidate\final_model_best_trades.csv"

OUTDIR = ROOT / r"research_outputs\execution_robustness_test"
OUTDIR.mkdir(parents=True, exist_ok=True)

print("Loading best trades...")
trades = pl.read_csv(BEST, try_parse_dates=True)

print("Loading features...")
features = pl.read_parquet(FEATURES)

results = []

for offset in [0, 1, 2, 3]:

    rows = []

    for t in trades.iter_rows(named=True):

        entry_time = t["entry_time"]

        future = (
            features
            .filter(pl.col("ts_ct") >= entry_time)
            .sort("ts_ct")
            .slice(offset, 90)
        )

        if future.height < 5:
            continue

        entry_bar = future.row(0, named=True)

        entry = float(entry_bar["close"])
        stop = float(entry_bar["high"]) + 2.0

        risk = stop - entry

        if risk <= 0 or risk > 25:
            continue

        target = entry - (2.0 * risk)

        outcome = None

        for r in future.iter_rows(named=True):

            high = float(r["high"])
            low = float(r["low"])

            if high >= stop:
                outcome = entry - stop
                break

            if low <= target:
                outcome = entry - target
                break

        if outcome is None:
            outcome = entry - float(future.row(-1, named=True)["close"])

        rows.append(outcome)

    if rows:

        wins = [x for x in rows if x > 0]
        losses = [x for x in rows if x <= 0]

        gross_win = sum(wins)
        gross_loss = abs(sum(losses)) if losses else 0

        results.append({
            "entry_delay_bars": offset,
            "trades": len(rows),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": round(100 * len(wins) / len(rows), 2),
            "net_points": round(sum(rows), 2),
            "avg_trade": round(sum(rows) / len(rows), 2),
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        })

summary = pl.DataFrame(results)

summary.write_csv(OUTDIR / "execution_robustness_summary.csv")

print(summary)

print(f"\nSaved: {OUTDIR / 'execution_robustness_summary.csv'}")
