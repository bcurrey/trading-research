from pathlib import Path
import polars as pl

ROOT = Path(r"D:\TradingResearch\data_parquet\mbp1_1m_features")

files = sorted(ROOT.rglob("*.parquet"))

rows = []

for f in files:
    symbol = f.parent.name.replace("symbol=", "")
    date = f.stem.replace("date=", "")

    df = pl.scan_parquet(f).select([
        pl.len().alias("rows"),
        pl.min("minute").alias("first_minute"),
        pl.max("minute").alias("last_minute"),
        pl.mean("updates").alias("avg_updates"),
        pl.max("updates").alias("max_updates"),
        pl.mean("avg_spread").alias("avg_spread"),
        pl.mean("mid_range").alias("avg_mid_range"),
    ]).collect()

    rows.append({
        "date": date,
        "symbol": symbol,
        **df.row(0, named=True),
    })

out = pl.DataFrame(rows).sort(["date", "symbol"])

print("\nSummary by symbol:")
print(
    out.group_by("symbol")
    .agg([
        pl.len().alias("days"),
        pl.min("date").alias("first_date"),
        pl.max("date").alias("last_date"),
        pl.sum("rows").alias("total_1m_rows"),
    ])
)

print("\nLowest-row days:")
print(out.sort("rows").head(15))

print("\nHighest-spread days:")
print(out.sort("avg_spread", descending=True).head(15))

out.write_csv(r"D:\TradingResearch\data_parquet\mbp1_1m_validation_summary.csv")

print("\nSaved:")
print(r"D:\TradingResearch\data_parquet\mbp1_1m_validation_summary.csv")