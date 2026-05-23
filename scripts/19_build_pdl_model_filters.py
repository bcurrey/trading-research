import polars as pl
from pathlib import Path

DATA_DIR = Path(r"D:\TradingResearch\data_parquet")

df = pl.read_parquet(DATA_DIR / "NQ_feature_factory.parquet")

# ---------------------------------------------------
# Base setup
# ---------------------------------------------------

setup = df.filter(
    (pl.col("is_morning_trade_window")) &
    (pl.col("pdl_sweep_close_back_above"))
)

print("\nBase setup count:", setup.height)

# ---------------------------------------------------
# Layered filters
# ---------------------------------------------------

models = []

def evaluate(name, filt):

    x = setup.filter(filt)

    result = x.select([
        pl.lit(name).alias("model"),
        pl.len().alias("signals"),

        (pl.col("fwd_points_30m") > 0)
        .mean()
        .alias("win_rate"),

        pl.col("fwd_points_30m")
        .mean()
        .alias("avg_30m"),

        pl.col("fwd_points_60m")
        .mean()
        .alias("avg_60m"),

        pl.col("mfe_30m")
        .mean()
        .alias("avg_mfe"),

        pl.col("mae_30m")
        .mean()
        .alias("avg_mae"),
    ])

    return result


# ---------------------------------------------------
# Model 1
# Only 8-9 CT
# ---------------------------------------------------

models.append(
    evaluate(
        "8-9 CT only",

        (pl.col("hour_ct").is_in([8, 9]))
    )
)

# ---------------------------------------------------
# Model 2
# 8-9 CT + above VWAP
# ---------------------------------------------------

models.append(
    evaluate(
        "8-9 CT + above VWAP",

        (
            pl.col("hour_ct").is_in([8, 9])
        ) &
        (
            pl.col("close") > pl.col("vwap_day")
        )
    )
)

# ---------------------------------------------------
# Model 3
# Add body strength filter
# ---------------------------------------------------

models.append(
    evaluate(
        "8-9 CT + above VWAP + body > 0.5",

        (
            pl.col("hour_ct").is_in([8, 9])
        ) &
        (
            pl.col("close") > pl.col("vwap_day")
        ) &
        (
            pl.col("body_pct") >= 0.5
        )
    )
)

# ---------------------------------------------------
# Model 4
# Exclude Thursdays
# weekday 4 from your output
# ---------------------------------------------------

models.append(
    evaluate(
        "8-9 CT + VWAP + body + no Thursday",

        (
            pl.col("hour_ct").is_in([8, 9])
        ) &
        (
            pl.col("close") > pl.col("vwap_day")
        ) &
        (
            pl.col("body_pct") >= 0.5
        ) &
        (
            pl.col("weekday_ct") != 4
        )
    )
)

# ---------------------------------------------------
# Model 5
# Add ATR filter
# ---------------------------------------------------

models.append(
    evaluate(
        "FULL FILTER STACK",

        (
            pl.col("hour_ct").is_in([8, 9])
        ) &
        (
            pl.col("close") > pl.col("vwap_day")
        ) &
        (
            pl.col("body_pct") >= 0.5
        ) &
        (
            pl.col("weekday_ct") != 4
        ) &
        (
            pl.col("atr_14") >= 15
        )
    )
)

results = pl.concat(models)

results = results.with_columns([
    (pl.col("win_rate") * 100).round(2).alias("win_rate_pct"),
    pl.col("avg_30m").round(2),
    pl.col("avg_60m").round(2),
    pl.col("avg_mfe").round(2),
    pl.col("avg_mae").round(2),
])

print("\nPDL MODEL RESULTS:\n")
print(results)

print("\nDONE.")