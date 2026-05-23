import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
FILE = DATA_DIR / "edge_discovery" / "edge_discovery_all_results.csv"

df = pl.read_csv(FILE)

print("\nRows tested:", df.height)

print("\nTop 30 by avg_points:")
print(
    df.sort("avg_points", descending=True)
    .select([
        "setup",
        "direction",
        "filters",
        "target_points",
        "trades",
        "win_rate_pct",
        "avg_points",
        "total_points",
        "avg_risk",
        "positive_years",
        "total_years",
        "positive_year_rate_pct",
        "edge_score",
    ])
    .head(30)
)

print("\nTop 30 by total_points:")
print(
    df.sort("total_points", descending=True)
    .select([
        "setup",
        "direction",
        "filters",
        "target_points",
        "trades",
        "win_rate_pct",
        "avg_points",
        "total_points",
        "avg_risk",
        "positive_years",
        "total_years",
        "positive_year_rate_pct",
        "edge_score",
    ])
    .head(30)
)

print("\nTop 30 with at least 200 trades and positive avg_points:")
print(
    df.filter(
        (pl.col("trades") >= 200) &
        (pl.col("avg_points") > 0)
    )
    .sort("avg_points", descending=True)
    .select([
        "setup",
        "direction",
        "filters",
        "target_points",
        "trades",
        "win_rate_pct",
        "avg_points",
        "total_points",
        "avg_risk",
        "positive_years",
        "total_years",
        "positive_year_rate_pct",
        "edge_score",
    ])
    .head(30)
)

print("\nDONE.")