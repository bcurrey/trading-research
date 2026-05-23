# 16_trade_controls_comparison.py
# Compare trade management / daily control variants
#
# Run:
# cd D:\TradingResearch
# py -3.12 scripts\ict\16_trade_controls_comparison.py

from pathlib import Path
import polars as pl
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

ROOT = Path(r"D:\TradingResearch")

INPUT_CSV = ROOT / r"research_outputs\deep_retrace_v1_candidate\deep_retrace_v1_trades.csv"
OUT_DIR = ROOT / r"research_outputs\trade_controls_comparison"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUT_DIR / "trade_controls_summary.csv"
YEARLY_OUT = OUT_DIR / "trade_controls_yearly.csv"
EQUITY_OUT = OUT_DIR / "trade_controls_equity.csv"


def summarize(df: pl.DataFrame, group_name: str) -> dict:
    if df.height == 0:
        return {
            "variant": group_name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0,
            "avg_trade": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
        }

    wins_df = df.filter(pl.col("result_points") > 0)
    losses_df = df.filter(pl.col("result_points") <= 0)

    gross_win = wins_df["result_points"].sum() if wins_df.height else 0
    gross_loss = abs(losses_df["result_points"].sum()) if losses_df.height else 0

    equity = (
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
        "variant": group_name,
        "trades": df.height,
        "wins": wins_df.height,
        "losses": losses_df.height,
        "win_rate": round(100 * wins_df.height / df.height, 2),
        "net_points": round(df["result_points"].sum(), 2),
        "avg_trade": round(df["result_points"].mean(), 2),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "max_drawdown": round(equity["dd"].min(), 2),
    }


def apply_controls(df: pl.DataFrame, variant: str) -> pl.DataFrame:

    x = df.sort("entry_time")

    if variant == "baseline":
        return x

    if variant == "8_10_ct_only":
        return x.filter(pl.col("hour").is_in([8, 10]))

    if variant == "10_ct_only":
        return x.filter(pl.col("hour") == 10)

    if variant == "one_trade_day":
        return (
            x.group_by("trade_date")
            .first()
            .sort("entry_time")
        )

    if variant == "max_2_trades_day":
        return (
            x.with_columns(
                pl.int_range(0, pl.len()).over("trade_date").alias("trade_num")
            )
            .filter(pl.col("trade_num") < 2)
            .drop("trade_num")
        )

    if variant == "one_trade_hour":
        return (
            x.group_by(["trade_date", "hour"])
            .first()
            .sort("entry_time")
        )

    if variant == "stop_after_win":
        out = []

        for _, g in x.partition_by("trade_date", as_dict=True).items():
            day = g.sort("entry_time")

            for row in day.iter_rows(named=True):
                out.append(row)
                if row["result_points"] > 0:
                    break

        return pl.DataFrame(out) if out else x.head(0)

    if variant == "stop_after_loss":
        out = []

        for _, g in x.partition_by("trade_date", as_dict=True).items():
            day = g.sort("entry_time")

            for row in day.iter_rows(named=True):
                out.append(row)
                if row["result_points"] <= 0:
                    break

        return pl.DataFrame(out) if out else x.head(0)

    return x


def yearly_summary(df: pl.DataFrame, variant: str) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame()

    out = (
        df.group_by("year")
        .agg(
            pl.len().alias("trades"),
            (pl.col("result_points") > 0).sum().alias("wins"),
            pl.col("result_points").sum().round(2).alias("net_points"),
            pl.col("result_points").mean().round(2).alias("avg_trade"),
        )
        .with_columns(
            (100 * pl.col("wins") / pl.col("trades")).round(2).alias("win_rate"),
            pl.lit(variant).alias("variant")
        )
        .select([
            "variant",
            "year",
            "trades",
            "wins",
            "win_rate",
            "net_points",
            "avg_trade"
        ])
        .sort(["variant", "year"])
    )

    return out


def main():

    print("Loading trades...")

    df = pl.read_csv(
        INPUT_CSV,
        try_parse_dates=True
    )

    print(f"Trades loaded: {df.height}")

    variants = [
        "baseline",
        "8_10_ct_only",
        "10_ct_only",
        "one_trade_day",
        "max_2_trades_day",
        "one_trade_hour",
        "stop_after_win",
        "stop_after_loss",
    ]

    summary_rows = []
    yearly_frames = []
    equity_frames = []

    for v in variants:

        test_df = apply_controls(df, v)

        summary_rows.append(
            summarize(test_df, v)
        )

        yearly_frames.append(
            yearly_summary(test_df, v)
        )

        eq = (
            test_df.sort("entry_time")
            .with_columns(
                pl.col("result_points").cum_sum().alias("equity")
            )
            .select([
                "entry_time",
                "trade_date",
                "result_points",
                "equity"
            ])
            .with_columns(
                pl.lit(v).alias("variant")
            )
        )

        equity_frames.append(eq)

        print(f"\n{v}")
        print(summarize(test_df, v))

    summary_df = pl.DataFrame(summary_rows).sort(
        ["profit_factor", "net_points"],
        descending=[True, True]
    )

    yearly_df = pl.concat(yearly_frames, how="vertical")
    equity_df = pl.concat(equity_frames, how="vertical")

    summary_df.write_csv(SUMMARY_OUT)
    yearly_df.write_csv(YEARLY_OUT)
    equity_df.write_csv(EQUITY_OUT)

    print("\nFINAL SUMMARY")
    print(summary_df)

    print(f"\nSaved: {SUMMARY_OUT}")
    print(f"Saved: {YEARLY_OUT}")
    print(f"Saved: {EQUITY_OUT}")


if __name__ == "__main__":
    main()
