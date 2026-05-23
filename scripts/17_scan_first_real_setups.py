import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")
OUT_DIR = DATA_DIR / "setup_scans"
OUT_DIR.mkdir(exist_ok=True)

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet")

# Only test realistic trade window first
base = df.filter(pl.col("is_morning_trade_window"))

print("\nRows in morning window:", base.height)


def summarize_setup(data: pl.DataFrame, name: str, direction: str):
    """
    direction:
      long  = positive forward points are good
      short = negative forward points are good
    """

    if direction == "long":
        edge_expr = pl.col("fwd_points_30m")
        win_expr = pl.col("fwd_points_30m") > 0
        mfe_expr = pl.col("mfe_30m")
        mae_expr = pl.col("mae_30m")
    else:
        edge_expr = -pl.col("fwd_points_30m")
        win_expr = pl.col("fwd_points_30m") < 0
        mfe_expr = pl.col("mae_30m")
        mae_expr = pl.col("mfe_30m")

    return data.select([
        pl.lit(name).alias("setup"),
        pl.lit(direction).alias("direction"),
        pl.len().alias("signals"),
        win_expr.mean().alias("win_rate_30m"),
        edge_expr.mean().alias("avg_edge_30m"),
        edge_expr.median().alias("median_edge_30m"),
        mfe_expr.mean().alias("avg_mfe_30m"),
        mae_expr.mean().alias("avg_mae_30m"),
        pl.col("fwd_points_30m").mean().alias("raw_avg_fwd_30m"),
        pl.col("fwd_points_60m").mean().alias("raw_avg_fwd_60m"),
    ])


setups = []

# 1. Prior day high sweep reversal = short idea
setups.append(
    summarize_setup(
        base.filter(pl.col("pdh_sweep_close_back_below")),
        "PDH sweep close back below",
        "short",
    )
)

# 2. Prior day low sweep reversal = long idea
setups.append(
    summarize_setup(
        base.filter(pl.col("pdl_sweep_close_back_above")),
        "PDL sweep close back above",
        "long",
    )
)

# 3. Premarket high sweep reversal = short idea
setups.append(
    summarize_setup(
        base.filter(pl.col("pmh_sweep_close_back_below")),
        "PMH sweep close back below",
        "short",
    )
)

# 4. Premarket low sweep reversal = long idea
setups.append(
    summarize_setup(
        base.filter(pl.col("pml_sweep_close_back_above")),
        "PML sweep close back above",
        "long",
    )
)

# 5. Bull displacement FVG continuation = long idea
setups.append(
    summarize_setup(
        base.filter(
            pl.col("bull_displacement") &
            pl.col("bull_fvg_min_2pt")
        ),
        "Bull displacement + FVG",
        "long",
    )
)

# 6. Bear displacement FVG continuation = short idea
setups.append(
    summarize_setup(
        base.filter(
            pl.col("bear_displacement") &
            pl.col("bear_fvg_min_2pt")
        ),
        "Bear displacement + FVG",
        "short",
    )
)

# 7. OR high breakout continuation = long idea
setups.append(
    summarize_setup(
        base.filter(pl.col("above_or_high")),
        "Above opening range high",
        "long",
    )
)

# 8. OR low breakdown continuation = short idea
setups.append(
    summarize_setup(
        base.filter(pl.col("below_or_low")),
        "Below opening range low",
        "short",
    )
)

results = pl.concat(setups)

results = results.with_columns([
    (pl.col("win_rate_30m") * 100).round(2).alias("win_rate_pct"),
    pl.col("avg_edge_30m").round(2),
    pl.col("median_edge_30m").round(2),
    pl.col("avg_mfe_30m").round(2),
    pl.col("avg_mae_30m").round(2),
    pl.col("raw_avg_fwd_30m").round(2),
    pl.col("raw_avg_fwd_60m").round(2),
])

results = results.sort("avg_edge_30m", descending=True)

print("\nSetup scan results:")
print(results)

out_file = OUT_DIR / "first_setup_scan_summary.parquet"
results.write_parquet(out_file)

csv_file = OUT_DIR / "first_setup_scan_summary.csv"
results.write_csv(csv_file)

print(f"\nSaved:")
print(out_file)
print(csv_file)
print("\nDONE.")