import polars as pl
from pathlib import Path

OUT = Path(r"D:\TradingResearch\research_outputs")

A_PATH = OUT / "focused_winrate_A_models.csv"
RANKED_PATH = OUT / "focused_winrate_ranked_models.csv"
ALL_PATH = OUT / "focused_winrate_all_models.parquet"

print("Loading focused win-rate outputs...")

if A_PATH.exists():
    df = pl.read_csv(A_PATH)
    source = A_PATH
elif RANKED_PATH.exists():
    df = pl.read_csv(RANKED_PATH)
    source = RANKED_PATH
elif ALL_PATH.exists():
    df = pl.read_parquet(ALL_PATH)
    source = ALL_PATH
else:
    raise SystemExit("No focused win-rate output files found.")

print(f"\nUsing: {source}")
print(f"Rows: {df.height:,}")
print("\nColumns:")
print(df.columns)

# Normalize tier column
if "tier" not in df.columns:
    raise SystemExit("No tier column found.")

# Pull A/A+ models only
top = (
    df
    .filter(pl.col("tier").is_in(["A+", "A"]))
    .sort(
        ["tier", "win_rate", "profit_factor", "net_points"],
        descending=[False, True, True, True]
    )
)

print("\nA/A+ MODEL COUNT:")
print(top.group_by("tier").len().sort("tier"))

print("\nTOP 50 A/A+ MODELS:")
print(top.head(50))

out = OUT / "21_clean_A_Aplus_models_for_next_test.csv"
top.write_csv(out)

print("\nSaved:")
print(out)