import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
FILE = DATA_DIR / "NQ_feature_factory.parquet"

print("\nLoading feature factory...\n")
df = pl.read_parquet(FILE)

print(f"Rows: {df.height:,}")
print(f"Columns: {len(df.columns):,}")

print("\nDate range:")
print(
    df.select([
        pl.col("ts_event").min().alias("first_ts"),
        pl.col("ts_event").max().alias("last_ts"),
        pl.col("trade_date_ct").n_unique().alias("trade_days"),
    ])
)

print("\nKey feature counts:")
summary = df.select([
    pl.col("is_rth").sum().alias("rth_bars"),
    pl.col("is_morning_trade_window").sum().alias("morning_bars"),
    pl.col("bull_fvg").sum().alias("bull_fvg"),
    pl.col("bear_fvg").sum().alias("bear_fvg"),
    pl.col("bull_fvg_min_2pt").sum().alias("bull_fvg_min_2pt"),
    pl.col("bear_fvg_min_2pt").sum().alias("bear_fvg_min_2pt"),
    pl.col("bull_displacement").sum().alias("bull_displacement"),
    pl.col("bear_displacement").sum().alias("bear_displacement"),
    pl.col("pdh_sweep_close_back_below").sum().alias("pdh_sweep_reversal"),
    pl.col("pdl_sweep_close_back_above").sum().alias("pdl_sweep_reversal"),
    pl.col("pmh_sweep_close_back_below").sum().alias("pmh_sweep_reversal"),
    pl.col("pml_sweep_close_back_above").sum().alias("pml_sweep_reversal"),
    pl.col("above_or_high").sum().alias("above_or_high"),
    pl.col("below_or_low").sum().alias("below_or_low"),
])

print(summary)

print("\nForward return stats:")
print(
    df.select([
        pl.col("fwd_points_5m").mean().alias("avg_5m"),
        pl.col("fwd_points_15m").mean().alias("avg_15m"),
        pl.col("fwd_points_30m").mean().alias("avg_30m"),
        pl.col("fwd_points_60m").mean().alias("avg_60m"),
        pl.col("mfe_30m").mean().alias("avg_mfe_30m"),
        pl.col("mae_30m").mean().alias("avg_mae_30m"),
    ])
)

print("\nMorning-only setup counts:")
morning = df.filter(pl.col("is_morning_trade_window"))

print(
    morning.select([
        pl.col("bull_fvg_min_2pt").sum().alias("bull_fvg_min_2pt"),
        pl.col("bear_fvg_min_2pt").sum().alias("bear_fvg_min_2pt"),
        pl.col("bull_displacement").sum().alias("bull_displacement"),
        pl.col("bear_displacement").sum().alias("bear_displacement"),
        pl.col("pdh_sweep_close_back_below").sum().alias("pdh_sweep_reversal"),
        pl.col("pdl_sweep_close_back_above").sum().alias("pdl_sweep_reversal"),
    ])
)

print("\nDONE.")