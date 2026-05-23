import polars as pl
from pathlib import Path

OUT = Path(r"D:\TradingResearch\research_outputs")
IN = OUT / "21_clean_A_Aplus_models_for_next_test.csv"

df = pl.read_csv(IN)

print("Loaded A/A+ models:", df.height)

mbp_filters = [
    "NO_MBP_FILTER",
    "tight_spread",
    "positive_quote_pressure",
    "positive_mid_pressure",
    "tight_spread_plus_pressure",
]

rows = []

for r in df.iter_rows(named=True):
    for mbp in mbp_filters:
        row = dict(r)
        row["mbp_overlay"] = mbp
        row["next_test_priority"] = (
            "HIGH" if mbp in ["tight_spread_plus_pressure", "NO_MBP_FILTER"] else "MED"
        )
        rows.append(row)

out = pl.DataFrame(rows)

path = OUT / "22_A_models_with_mbp_overlay_test_plan.csv"
out.write_csv(path)

print("Rows:", out.height)
print(out.select([
    "tier", "conditions", "entry_model", "target_model",
    "winrate", "profit_factor", "net_points", "mbp_overlay",
    "next_test_priority"
]).head(80))

print("Saved:", path)