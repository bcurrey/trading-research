from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\TradingResearch")
INPUT = ROOT / r"ict_trade_study\ict_discord_trade_review_index_with_links.xlsx"
OUTDIR = ROOT / r"ict_trade_study\outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTDIR / "ict_keeper_trades_clean.csv"

df = pd.read_excel(INPUT, sheet_name="NQ Candidates")

print("\nAvailable columns:")
print(list(df.columns))

# Normalize column names
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

if "reviewer_keep" not in df.columns:
    raise ValueError(f"Missing reviewer_keep column. Columns: {df.columns.tolist()}")

keepers = df[df["reviewer_keep"].astype(str).str.strip().str.lower() == "yes"].copy()

print("\nReviewer keep counts:")
print(df["reviewer_keep"].value_counts(dropna=False))

resolved = {
    "date": "date",
    "entry_time": "entry_time",
    "entry_price": "entry_price",
    "side": "side",
    "result": "result",
}

print("\nResolved columns:")
print("keeper: reviewer_keep")
for k, v in resolved.items():
    print(f"{k}: {v}")

out = pd.DataFrame()
out["trade_id"] = keepers["trade_id"].fillna("").astype(str).str.strip()
out.loc[out["trade_id"].eq("") | out["trade_id"].str.lower().eq("nan"), "trade_id"] = [
    f"ICT_{i+1:03d}" for i in range((out["trade_id"].eq("") | out["trade_id"].str.lower().eq("nan")).sum())
]

out["source_row"] = keepers.index + 2

for standard, col in resolved.items():
    out[standard] = keepers[col] if col in keepers.columns else ""

for col in [
    "trades_per_post",
    "message_id",
    "discord_url",
    "open_discord",
    "timestamp",
    "date_in_text",
    "tickers",
    "priority",
    "side_hint",
    "result_hint",
    "points_hint",
    "attachment_count",
    "attachment_file_names",
    "attachment_paths",
    "youtube_links",
    "content",
    "sl",
    "tp",
]:
    if col in keepers.columns:
        out[col] = keepers[col]

out["side"] = out["side"].astype(str).str.lower().str.strip()
out["result"] = out["result"].astype(str).str.lower().str.strip()

out["missing_entry_time"] = out["entry_time"].isna() | (out["entry_time"].astype(str).str.strip() == "")
out["missing_entry_price"] = out["entry_price"].isna() | (out["entry_price"].astype(str).str.strip() == "")
out["missing_side"] = out["side"].isna() | (out["side"].astype(str).str.strip() == "") | out["side"].eq("nan")
out["missing_result"] = out["result"].isna() | (out["result"].astype(str).str.strip() == "") | out["result"].eq("nan")

out.to_csv(OUTPUT, index=False)

print(f"\nKeeper trades found: {len(out)}")
print(f"Saved: {OUTPUT}")

print("\nQuality check:")
print(
    out[
        [
            "trade_id",
            "date",
            "entry_time",
            "entry_price",
            "side",
            "result",
            "missing_entry_time",
            "missing_entry_price",
            "missing_side",
            "missing_result",
        ]
    ].to_string(index=False)
)