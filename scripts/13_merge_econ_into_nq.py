import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")

print("\nLoading NQ features...")
nq = pl.read_parquet(DATA_DIR / "NQ_session_features.parquet")

print("\nLoading econ daily flags...")
econ = pl.read_parquet(DATA_DIR / "econ_daily_flags.parquet")

print(f"NQ rows: {len(nq):,}")
print(f"Econ daily rows: {len(econ):,}")

# Build UTC date key on NQ
nq = nq.with_columns([
    pl.col("ts_event").dt.date().alias("event_date_utc")
])

# Merge daily econ flags into every 1m bar
merged = nq.join(
    econ,
    on="event_date_utc",
    how="left"
)

# Fill nulls for non-event days
bool_cols = [
    "has_high_impact_usd_event",
    "has_cpi",
    "has_fomc",
    "has_nfp",
    "has_ppi",
    "has_pmi_ism",
    "has_jobs",
    "has_fed_speech",
]

count_cols = [
    "usd_event_count",
    "high_impact_event_count",
    "cpi_event_count",
    "fomc_event_count",
    "nfp_event_count",
    "ppi_event_count",
    "pmi_ism_event_count",
    "jobs_event_count",
    "fed_speech_event_count",
]

merged = merged.with_columns([
    *[pl.col(c).fill_null(False) for c in bool_cols],
    *[pl.col(c).fill_null(0) for c in count_cols],
])

print("\nPreview:")
print(
    merged.select([
        "ts_event",
        "close",
        "has_high_impact_usd_event",
        "has_cpi",
        "has_fomc",
        "has_nfp",
        "has_ppi",
        "high_impact_event_count",
    ]).filter(pl.col("has_high_impact_usd_event") == True).head(20)
)

output_file = DATA_DIR / "NQ_master.parquet"

print(f"\nWriting:\n{output_file}")
merged.write_parquet(output_file)

print("\nDONE.")