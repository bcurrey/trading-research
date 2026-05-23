import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")

print("\nLoading USD economic calendar...\n")

econ = pl.read_parquet(DATA_DIR / "econ_calendar_usd.parquet")

print(f"Rows loaded: {len(econ):,}")

# Create daily flags by UTC date
daily = (
    econ.group_by("event_date_utc")
    .agg([
        pl.len().alias("usd_event_count"),
        pl.col("is_high_impact").sum().alias("high_impact_event_count"),
        (pl.col("event_group") == "CPI").sum().alias("cpi_event_count"),
        (pl.col("event_group") == "FOMC").sum().alias("fomc_event_count"),
        (pl.col("event_group") == "NFP").sum().alias("nfp_event_count"),
        (pl.col("event_group") == "PPI").sum().alias("ppi_event_count"),
        (pl.col("event_group") == "PMI_ISM").sum().alias("pmi_ism_event_count"),
        (pl.col("event_group") == "JOBS").sum().alias("jobs_event_count"),
        (pl.col("event_group") == "FED_SPEECH").sum().alias("fed_speech_event_count"),
    ])
    .sort("event_date_utc")
)

daily = daily.with_columns([
    (pl.col("high_impact_event_count") > 0).alias("has_high_impact_usd_event"),
    (pl.col("cpi_event_count") > 0).alias("has_cpi"),
    (pl.col("fomc_event_count") > 0).alias("has_fomc"),
    (pl.col("nfp_event_count") > 0).alias("has_nfp"),
    (pl.col("ppi_event_count") > 0).alias("has_ppi"),
    (pl.col("pmi_ism_event_count") > 0).alias("has_pmi_ism"),
    (pl.col("jobs_event_count") > 0).alias("has_jobs"),
    (pl.col("fed_speech_event_count") > 0).alias("has_fed_speech"),
])

print("\nPreview:")
print(daily.head(30))

output_file = DATA_DIR / "econ_daily_flags.parquet"

print(f"\nWriting:\n{output_file}")
daily.write_parquet(output_file)

print("\nDONE.")