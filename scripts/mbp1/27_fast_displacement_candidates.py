import polars as pl
from pathlib import Path
from itertools import product

DATA = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT = Path(r"D:\TradingResearch\research_outputs")

START = "2018-01-01"
END = "2026-05-01"

print("Loading candidate rows only...")

df = (
    pl.scan_parquet(DATA)
    .filter(
        (pl.col("trade_date_ct") >= pl.lit(START).str.to_date()) &
        (pl.col("trade_date_ct") <= pl.lit(END).str.to_date()) &
        (pl.col("is_rth") == True) &
        (pl.col("low") < pl.col("prior_rth_low")) &
        (pl.col("close") > pl.col("prior_rth_low")) &
        (pl.col("close") > pl.col("or_low_30m")) &
        (pl.col("dist_vwap_day") < 0) &
        (pl.col("close") > pl.col("open")) &
        (pl.col("bull_fvg") == True)
    )
    .with_columns([
        pl.col("ts_ct").alias("ts"),
        pl.col("trade_date_ct").alias("date"),
    ])
    .select([
        "ts", "date", "open", "high", "low", "close",
        "body_abs", "body_pct", "bull_fvg_low", "bull_fvg_high",
        "bull_fvg_size", "premarket_range", "minute_of_day_ct",
        "avg_spread", "quote_pressure_delta", "net_mid_pressure"
    ])
    .collect()
    .sort("ts")
)

print(f"Candidate rows: {df.height:,}")

rows = []

for disp, fvg_min, tp, sl in product([8, 10, 12], [2, 3, 4], [20, 25], [12, 15, 20]):
    x = df.filter(
        (pl.col("body_abs") >= disp) &
        (pl.col("body_pct") >= 0.60) &
        (pl.col("bull_fvg_size") >= fvg_min)
    )

    if x.height < 10:
        continue

    wins = 0
    losses = 0
    pnl = []

    for r in x.iter_rows(named=True):
        entry = float((r["bull_fvg_low"] + r["bull_fvg_high"]) / 2)
        mfe = float(r["high"] - entry)
        mae = float(r["low"] - entry)

        if mfe >= tp:
            wins += 1
            pnl.append(tp)
        elif abs(mae) >= sl:
            losses += 1
            pnl.append(-sl)

    if len(pnl) < 10:
        continue

    gross_win = sum(p for p in pnl if p > 0)
    gross_loss = abs(sum(p for p in pnl if p < 0))
    pf = gross_win / gross_loss if gross_loss else 999
    wr = wins / len(pnl)

    rows.append({
        "disp": disp,
        "fvg_min": fvg_min,
        "tp": tp,
        "sl": sl,
        "trades": len(pnl),
        "wins": wins,
        "losses": losses,
        "winrate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "net_points": round(sum(pnl), 2),
    })

out = pl.DataFrame(rows).sort(
    ["winrate", "profit_factor", "net_points"],
    descending=True
)

path = OUT / "27_fast_displacement_candidates.csv"
out.write_csv(path)

print(out.head(50))
print("Saved:", path)