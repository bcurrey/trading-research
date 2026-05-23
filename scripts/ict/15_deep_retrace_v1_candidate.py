# 15_deep_retrace_v1_candidate.py
# Deep Retrace V1 Candidate
# Run from:
#   cd D:\TradingResearch
#   py -3.12 scripts\ict\15_deep_retrace_v1_candidate.py

from __future__ import annotations

from pathlib import Path
import math
import polars as pl


ROOT = Path(r"D:\TradingResearch")
DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
OUT_DIR = ROOT / r"research_outputs\deep_retrace_v1_candidate"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_CSV = OUT_DIR / "deep_retrace_v1_trades.csv"
SUMMARY_CSV = OUT_DIR / "deep_retrace_v1_summary.csv"
YEARLY_CSV = OUT_DIR / "deep_retrace_v1_yearly.csv"
MONTHLY_CSV = OUT_DIR / "deep_retrace_v1_monthly.csv"
TIME_CSV = OUT_DIR / "deep_retrace_v1_time_of_day.csv"
VARIANT_CSV = OUT_DIR / "deep_retrace_v1_variants.csv"
DAILY_CSV = OUT_DIR / "deep_retrace_v1_daily.csv"
EQUITY_CSV = OUT_DIR / "deep_retrace_v1_equity_curve.csv"


# -----------------------------
# Helpers
# -----------------------------

def pick_col(df: pl.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried: {candidates}")
    return None


def add_time_cols(df: pl.DataFrame, ts_col: str) -> pl.DataFrame:
    # Dataset is assumed to already be Central-time aware in current pipeline.
    # If ts is UTC in your file, convert upstream the same way scripts 13/14 did.
    return df.with_columns(
        pl.col(ts_col).dt.date().alias("trade_date"),
        pl.col(ts_col).dt.year().alias("year"),
        pl.col(ts_col).dt.strftime("%Y-%m").alias("month"),
        pl.col(ts_col).dt.hour().alias("hour"),
        pl.col(ts_col).dt.minute().alias("minute"),
    )


def summarize(trades: pl.DataFrame, group_col: str = "group") -> pl.DataFrame:
    if trades.height == 0:
        return pl.DataFrame({
            group_col: [],
            "trades": [],
            "wins": [],
            "losses": [],
            "win_rate": [],
            "net_points": [],
            "avg_points": [],
            "avg_rr": [],
            "profit_factor": [],
            "avg_risk": [],
            "avg_mfe": [],
            "avg_mae": [],
            "max_drawdown_points": [],
        })

    return (
        trades
        .group_by(group_col)
        .agg(
            pl.len().alias("trades"),
            (pl.col("result_points") > 0).sum().alias("wins"),
            (pl.col("result_points") <= 0).sum().alias("losses"),
            (100 * (pl.col("result_points") > 0).mean()).round(2).alias("win_rate"),
            pl.col("result_points").sum().round(2).alias("net_points"),
            pl.col("result_points").mean().round(2).alias("avg_points"),
            pl.col("result_rr").mean().round(2).alias("avg_rr"),
            pl.when((pl.col("result_points").filter(pl.col("result_points") < 0).sum().abs()) > 0)
              .then(
                  pl.col("result_points").filter(pl.col("result_points") > 0).sum()
                  / pl.col("result_points").filter(pl.col("result_points") < 0).sum().abs()
              )
              .otherwise(None)
              .round(2)
              .alias("profit_factor"),
            pl.col("risk_points").mean().round(2).alias("avg_risk"),
            pl.col("mfe_points").mean().round(2).alias("avg_mfe"),
            pl.col("mae_points").mean().round(2).alias("avg_mae"),
        )
        .sort(group_col)
    )


def add_max_dd_by_group(trades: pl.DataFrame, summary_df: pl.DataFrame, group_col: str) -> pl.DataFrame:
    if trades.height == 0 or summary_df.height == 0:
        return summary_df.with_columns(pl.lit(None).alias("max_drawdown_points"))

    dd = (
        trades
        .sort([group_col, "entry_time"])
        .with_columns(
            pl.col("result_points").cum_sum().over(group_col).alias("equity")
        )
        .with_columns(
            pl.col("equity").cum_max().over(group_col).alias("peak")
        )
        .with_columns(
            (pl.col("equity") - pl.col("peak")).alias("drawdown")
        )
        .group_by(group_col)
        .agg(pl.col("drawdown").min().round(2).alias("max_drawdown_points"))
    )

    return summary_df.join(dd, on=group_col, how="left")


def simulate_short_trade(
    rows: list[dict],
    start_i: int,
    entry: float,
    stop: float,
    target: float,
    max_hold_bars: int,
) -> tuple[str, float, int, float, float]:
    """
    Returns:
      exit_reason, result_points, bars_held, mfe_points, mae_points
    Conservative same-bar handling:
      For shorts, if stop and target both hit in same bar, assume stop first.
    """
    entry_i = start_i
    max_favorable = 0.0
    max_adverse = 0.0

    for j in range(entry_i, min(len(rows), entry_i + max_hold_bars)):
        high = float(rows[j]["high"])
        low = float(rows[j]["low"])
        close = float(rows[j]["close"])

        favorable = entry - low
        adverse = high - entry
        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        stop_hit = high >= stop
        target_hit = low <= target

        if stop_hit and target_hit:
            return "stop_same_bar", entry - stop, j - entry_i + 1, max_favorable, -max_adverse
        if stop_hit:
            return "stop", entry - stop, j - entry_i + 1, max_favorable, -max_adverse
        if target_hit:
            return "target", entry - target, j - entry_i + 1, max_favorable, -max_adverse

    last_i = min(len(rows) - 1, entry_i + max_hold_bars - 1)
    close = float(rows[last_i]["close"])
    return "time", entry - close, last_i - entry_i + 1, max_favorable, -max_adverse


def main() -> None:
    print("Loading NQ dataset...")
    df = pl.read_parquet(DATA_PATH)
    print(f"Rows loaded: {df.height:,}")

    ts_col = pick_col(df, ["ts", "timestamp", "datetime", "date_time", "time"])
    open_col = pick_col(df, ["open", "Open"])
    high_col = pick_col(df, ["high", "High"])
    low_col = pick_col(df, ["low", "Low"])
    close_col = pick_col(df, ["close", "Close"])
    vol_col = pick_col(df, ["volume", "Volume", "vol"], required=False)

    ema20_col = pick_col(df, ["ema_20", "ema20"])
    ema50_col = pick_col(df, ["ema_50", "ema50"])

    # Flexible feature names based on prior scripts.
    range_exp_col = pick_col(df, ["range_expansion", "range_expansion_20", "bar_range_expansion", "range_vs_avg"], required=False)
    rel_vol_col = pick_col(df, ["rel_volume", "relative_volume", "volume_rel", "rvol"], required=False)

    sweep_cols = [c for c in [
        "any_liquidity_sweep_reclaim_30m",
        "liq_sweep_reclaim_30m",
        "any_sweep_reclaim_30m",
        "sweep_reclaim_30m",
        "pdh_sweep_reclaim_30m",
        "pdl_sweep_reclaim_30m",
        "asia_high_sweep_reclaim_30m",
        "asia_low_sweep_reclaim_30m",
        "london_high_sweep_reclaim_30m",
        "london_low_sweep_reclaim_30m",
    ] if c in df.columns]

    bear_disp_col = pick_col(df, ["bear_displacement", "bear_disp", "is_bear_displacement"], required=False)

    df = df.rename({
        ts_col: "ts",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
        ema20_col: "ema_20",
        ema50_col: "ema_50",
    })

    df = add_time_cols(df, "ts")

    # If explicit columns are missing, calculate reasonable replacements.
    df = df.with_columns(
        (pl.col("high") - pl.col("low")).alias("bar_range"),
        (pl.col("close") - pl.col("open")).alias("body"),
    )

    if range_exp_col and range_exp_col != "range_expansion":
        df = df.rename({range_exp_col: "range_expansion"})
    elif not range_exp_col:
        df = df.with_columns(
            (pl.col("bar_range") / pl.col("bar_range").rolling_mean(20)).alias("range_expansion")
        )

    if rel_vol_col and rel_vol_col != "rel_volume":
        df = df.rename({rel_vol_col: "rel_volume"})
    elif not rel_vol_col:
        if vol_col:
            if vol_col != "volume":
                df = df.rename({vol_col: "volume"})
            df = df.with_columns(
                (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("rel_volume")
            )
        else:
            df = df.with_columns(pl.lit(1.10).alias("rel_volume"))

    if bear_disp_col and bear_disp_col != "bear_displacement":
        df = df.rename({bear_disp_col: "bear_displacement"})
    elif not bear_disp_col:
        df = df.with_columns(
            (
                (pl.col("close") < pl.col("open")) &
                (pl.col("bar_range") >= pl.col("bar_range").rolling_mean(20) * 1.15)
            ).alias("bear_displacement")
        )

    if sweep_cols:
        df = df.with_columns(
            pl.any_horizontal([pl.col(c).fill_null(False).cast(pl.Boolean) for c in sweep_cols]).alias("any_sweep_reclaim_30m")
        )
    else:
        # Fallback: use True so missing naming does not kill the script.
        # If script 14 used a named sweep column, add it to sweep_cols above if needed.
        df = df.with_columns(pl.lit(True).alias("any_sweep_reclaim_30m"))

    # Base model from script 14.
    signal_df = (
        df
        .with_row_index("idx")
        .filter(
            (pl.col("hour").is_in([8, 9, 10])) &
            (
                ((pl.col("hour") == 8) & (pl.col("minute") >= 30)) |
                (pl.col("hour").is_in([9, 10]))
            ) &
            (pl.col("ema_20") < pl.col("ema_50")) &
            (pl.col("close") < pl.col("ema_20")) &
            (pl.col("bear_displacement") == True) &
            (pl.col("any_sweep_reclaim_30m") == True) &
            (pl.col("range_expansion") > 1.15) &
            (pl.col("rel_volume") > 1.05)
        )
        .with_columns(
            (pl.col("high") - 0.25 * (pl.col("high") - pl.col("low"))).alias("entry_price"),
            (pl.col("high") + 2.0).alias("stop_price"),
        )
        .with_columns(
            (pl.col("stop_price") - pl.col("entry_price")).alias("risk_points")
        )
        .with_columns(
            (pl.col("entry_price") - 2.0 * pl.col("risk_points")).alias("target_price")
        )
    )

    print(f"Signals found: {signal_df.height:,}")

    rows = df.select(["ts", "open", "high", "low", "close", "trade_date", "year", "month", "hour", "minute"]).to_dicts()
    signals = signal_df.select([
        "idx", "ts", "trade_date", "year", "month", "hour", "minute",
        "open", "high", "low", "close",
        "entry_price", "stop_price", "target_price", "risk_points",
        "range_expansion", "rel_volume"
    ]).to_dicts()

    trades = []

    for s in signals:
        sig_i = int(s["idx"])
        entry = float(s["entry_price"])
        stop = float(s["stop_price"])
        target = float(s["target_price"])
        risk = float(s["risk_points"])

        if not math.isfinite(risk) or risk <= 0:
            continue

        # Candidate filter: risk 8-20.
        if risk < 8 or risk > 20:
            continue

        # Fill must happen within next 1 bar after signal.
        fill_i = sig_i + 1
        if fill_i >= len(rows):
            continue

        fill_bar = rows[fill_i]
        fill_hit = float(fill_bar["high"]) >= entry and float(fill_bar["low"]) <= entry
        if not fill_hit:
            continue

        exit_reason, result_points, bars_held, mfe, mae = simulate_short_trade(
            rows=rows,
            start_i=fill_i,
            entry=entry,
            stop=stop,
            target=target,
            max_hold_bars=90,
        )

        trades.append({
            "signal_time": s["ts"],
            "entry_time": fill_bar["ts"],
            "trade_date": fill_bar["trade_date"],
            "year": fill_bar["year"],
            "month": fill_bar["month"],
            "hour": fill_bar["hour"],
            "minute": fill_bar["minute"],
            "side": "short",
            "entry_price": round(entry, 2),
            "stop_price": round(stop, 2),
            "target_price": round(target, 2),
            "risk_points": round(risk, 2),
            "result_points": round(result_points, 2),
            "result_rr": round(result_points / risk, 4),
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "fill_bars": 1,
            "mfe_points": round(mfe, 2),
            "mae_points": round(mae, 2),
            "signal_high": round(float(s["high"]), 2),
            "signal_low": round(float(s["low"]), 2),
            "signal_close": round(float(s["close"]), 2),
            "range_expansion": round(float(s["range_expansion"]), 4) if s["range_expansion"] is not None else None,
            "rel_volume": round(float(s["rel_volume"]), 4) if s["rel_volume"] is not None else None,
        })

    trades_df = pl.DataFrame(trades)

    if trades_df.height == 0:
        print("No candidate trades found.")
        return

    trades_df = trades_df.sort("entry_time")

    # Equity curve.
    equity_df = (
        trades_df
        .with_columns(
            pl.col("result_points").cum_sum().alias("equity_points")
        )
        .with_columns(
            pl.col("equity_points").cum_max().alias("peak_points")
        )
        .with_columns(
            (pl.col("equity_points") - pl.col("peak_points")).alias("drawdown_points")
        )
    )

    # Summaries.
    all_summary = summarize(trades_df.with_columns(pl.lit("ALL").alias("group")), "group")
    all_summary = add_max_dd_by_group(trades_df.with_columns(pl.lit("ALL").alias("group")), all_summary, "group")

    yearly = summarize(trades_df.rename({"year": "group"}), "group")
    yearly = add_max_dd_by_group(trades_df.rename({"year": "group"}), yearly, "group")

    monthly = summarize(trades_df.rename({"month": "group"}), "group")
    monthly = add_max_dd_by_group(trades_df.rename({"month": "group"}), monthly, "group")

    tod = summarize(trades_df.rename({"hour": "group"}), "group")
    tod = add_max_dd_by_group(trades_df.rename({"hour": "group"}), tod, "group")

    daily = (
        trades_df
        .group_by("trade_date")
        .agg(
            pl.len().alias("trades"),
            pl.col("result_points").sum().round(2).alias("net_points"),
            (pl.col("result_points") > 0).sum().alias("wins"),
            (pl.col("result_points") <= 0).sum().alias("losses"),
        )
        .sort("trade_date")
        .with_columns(
            (100 * pl.col("wins") / pl.col("trades")).round(2).alias("win_rate")
        )
    )

    # Variant comparison.
    variants = []
    variant_defs = {
        "8_CT_only": [8],
        "9_CT_only": [9],
        "10_CT_only": [10],
        "8_10_CT": [8, 10],
        "8_9_10_CT": [8, 9, 10],
    }

    for name, hours in variant_defs.items():
        sub = trades_df.filter(pl.col("hour").is_in(hours)).with_columns(pl.lit(name).alias("variant"))
        if sub.height == 0:
            continue
        s = summarize(sub.rename({"variant": "group"}), "group")
        s = add_max_dd_by_group(sub.rename({"variant": "group"}), s, "group")
        variants.append(s)

    variants_df = pl.concat(variants, how="vertical") if variants else pl.DataFrame()

    # Save.
    trades_df.write_csv(TRADE_CSV)
    all_summary.write_csv(SUMMARY_CSV)
    yearly.write_csv(YEARLY_CSV)
    monthly.write_csv(MONTHLY_CSV)
    tod.write_csv(TIME_CSV)
    daily.write_csv(DAILY_CSV)
    equity_df.write_csv(EQUITY_CSV)
    if variants_df.height:
        variants_df.write_csv(VARIANT_CSV)

    print("\nSUMMARY")
    print(all_summary)

    print("\nYEARLY")
    print(yearly)

    print("\nTIME OF DAY")
    print(tod)

    print("\nVARIANTS")
    print(variants_df)

    print("\nDAILY SAMPLE")
    print(daily.tail(20))

    print(f"\nSaved trades: {TRADE_CSV}")
    print(f"Saved summary: {SUMMARY_CSV}")
    print(f"Saved yearly: {YEARLY_CSV}")
    print(f"Saved monthly: {MONTHLY_CSV}")
    print(f"Saved time-of-day: {TIME_CSV}")
    print(f"Saved variants: {VARIANT_CSV}")
    print(f"Saved daily: {DAILY_CSV}")
    print(f"Saved equity curve: {EQUITY_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
