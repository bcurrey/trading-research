# 17_expand_10ct_model.py
# Expand elite 10 CT model while preserving edge

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"research_outputs\deep_retrace_v1_candidate\deep_retrace_v1_trades.csv"
OUTDIR = ROOT / r"research_outputs\expand_10ct_model"

OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "expand_10ct_summary.csv"
YEARLY_OUT = OUTDIR / "expand_10ct_yearly.csv"


def summarize(df: pl.DataFrame, name: str):

    if df.height == 0:
        return {
            "model": name,
            "trades": 0,
            "wins": 0,
            "winrate": 0,
            "net_points": 0,
            "avg_trade": 0,
            "profit_factor": 0,
            "max_dd": 0,
        }

    wins = df.filter(pl.col("result_points") > 0)
    losses = df.filter(pl.col("result_points") <= 0)

    gross_win = wins["result_points"].sum() if wins.height else 0
    gross_loss = abs(losses["result_points"].sum()) if losses.height else 0

    eq = (
        df.sort("entry_time")
        .with_columns(
            pl.col("result_points").cum_sum().alias("equity")
        )
        .with_columns(
            pl.col("equity").cum_max().alias("peak")
        )
        .with_columns(
            (pl.col("equity") - pl.col("peak")).alias("dd")
        )
    )

    return {
        "model": name,
        "trades": df.height,
        "wins": wins.height,
        "winrate": round(100 * wins.height / df.height, 2),
        "net_points": round(df["result_points"].sum(), 2),
        "avg_trade": round(df["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_dd": round(eq["dd"].min(), 2),
    }


def yearly(df: pl.DataFrame, name: str):

    if df.height == 0:
        return pl.DataFrame()

    return (
        df.group_by("year")
        .agg(
            pl.len().alias("trades"),
            (pl.col("result_points") > 0).sum().alias("wins"),
            pl.col("result_points").sum().round(2).alias("net_points"),
            pl.col("result_points").mean().round(2).alias("avg_trade"),
        )
        .with_columns(
            (100 * pl.col("wins") / pl.col("trades")).round(2).alias("winrate"),
            pl.lit(name).alias("model")
        )
        .select([
            "model",
            "year",
            "trades",
            "wins",
            "winrate",
            "net_points",
            "avg_trade"
        ])
        .sort(["model", "year"])
    )


def apply_model(df: pl.DataFrame, model: str):

    x = df.clone()

    if model == "10ct_elite":
        return x.filter(
            (pl.col("hour") == 10)
        )

    if model == "10ct_fill2":
        return x.filter(
            (pl.col("hour") == 10) &
            (pl.col("fill_bars") <= 2)
        )

    if model == "8_10_fill2":
        return x.filter(
            (pl.col("hour").is_in([8, 10])) &
            (pl.col("fill_bars") <= 2)
        )

    if model == "risk_6_20":
        return x.filter(
            (pl.col("risk_points") >= 6) &
            (pl.col("risk_points") <= 20)
        )

    if model == "10ct_low_mae":
        return x.filter(
            (pl.col("hour") == 10) &
            (pl.col("mae_points") > -8)
        )

    if model == "10ct_high_mfe":
        return x.filter(
            (pl.col("hour") == 10) &
            (pl.col("mfe_points") >= 20)
        )

    if model == "10ct_combo":
        return x.filter(
            (pl.col("hour") == 10) &
            (pl.col("risk_points") >= 6) &
            (pl.col("risk_points") <= 20) &
            (pl.col("mae_points") > -10)
        )

    return x


def main():

    print("Loading trades...")

    df = pl.read_csv(
        INPUT,
        try_parse_dates=True
    )

    print(f"Trades loaded: {df.height}")

    models = [
        "10ct_elite",
        "10ct_fill2",
        "8_10_fill2",
        "risk_6_20",
        "10ct_low_mae",
        "10ct_high_mfe",
        "10ct_combo",
    ]

    summary_rows = []
    yearly_frames = []

    for m in models:

        test = apply_model(df, m)

        result = summarize(test, m)
        summary_rows.append(result)

        yearly_frames.append(
            yearly(test, m)
        )

        print(f"\n{m}")
        print(result)

    summary_df = (
        pl.DataFrame(summary_rows)
        .sort(
            ["profit_factor", "avg_trade"],
            descending=[True, True]
        )
    )

    yearly_df = pl.concat(yearly_frames, how="vertical")

    summary_df.write_csv(SUMMARY_OUT)
    yearly_df.write_csv(YEARLY_OUT)

    print("\nFINAL SUMMARY")
    print(summary_df)

    print(f"\nSaved: {SUMMARY_OUT}")
    print(f"Saved: {YEARLY_OUT}")


if __name__ == "__main__":
    main()
