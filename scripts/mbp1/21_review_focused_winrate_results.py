import polars as pl
from pathlib import Path

OUT = Path(r"D:\TradingResearch\research_outputs")
path = OUT / "focused_winrate_A_models.csv"

print("Loading:", path)

if not path.exists():
    print("FILE NOT FOUND")
    print("Available CSVs:")
    for f in OUT.glob("*.csv"):
        print(f.name)
    raise SystemExit

df = pl.read_csv(path)

print("Rows:", df.height)
print("Columns:", df.columns)

top = df.filter(pl.col("tier").is_in(["A+", "A"]))

print("A/A+ rows:", top.height)

top = top.sort(
    ["tier", "winrate", "profit_factor", "net_points"],
    descending=[False, True, True, True]
)

print(top.head(50))

out = OUT / "21_clean_A_Aplus_models_for_next_test.csv"
top.write_csv(out)

print("Saved:", out)
