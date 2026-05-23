from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\event_sequence_v2b"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "event_sequence_v2b_trades.csv"
SUMMARY_OUT = OUTDIR / "event_sequence_v2b_summary.csv"
YEARLY_OUT = OUTDIR / "event_sequence_v2b_yearly.csv"

R_TARGET = 2.0
MAX_HOLD_BARS = 90


def norm(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def boolcol(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def build_signal(df):

    sweep_cols = [
        "pdh_sweep_reclaim_now",
        "pmh_sweep_reclaim_now",
        "asia_high_sweep_reclaim_now",
    ]

    sweep_count = sum([
        boolcol(df, c).astype(int)
        for c in sweep_cols
    ])

    strong_sweep = sweep_count >= 1

    below_vwap = df["close"] < df["vwap"]

    reclaim_failure = below_vwap

    bear_candle = df["close"] < df["open"]

    body_abs = pd.to_numeric(
        df.get("body_abs", 0),
        errors="coerce"
    )

    atr = pd.to_numeric(
        df.get("atr_14", 0),
        errors="coerce"
    )

    strong_body = body_abs > (atr * 0.7)

    close_near_low = (
        ((df["close"] - df["low"]) /
         (df["high"] - df["low"]).replace(0, np.nan))
        < 0.35
    )

    rejection = (
        bear_candle &
        strong_body &
        close_near_low
    )

    if "bear_fvg_size" in df.columns:

        fvg = pd.to_numeric(
            df["bear_fvg_size"],
            errors="coerce"
        )

        healthy_fvg = (
            (fvg >= 2) &
            (fvg <= 20)
        )

    else:
        healthy_fvg = pd.Series(
            True,
            index=df.index
        )

    not_extended = (
        (df["close"] - df["ema_20"]).abs()
        < (df["atr_14"] * 3)
    )

    rel_vol = pd.to_numeric(
        df.get("rel_vol_20", 0),
        errors="coerce"
    )

    vol_ok = rel_vol > 1.0

    session_ok = (
        (df["minute_of_day_ct"] >= (8 * 60 + 30)) &
        (df["minute_of_day_ct"] <= (11 * 60))
    )

    signal = (
        strong_sweep &
        reclaim_failure &
        rejection &
        healthy_fvg &
        not_extended &
        vol_ok &
        session_ok
    )

    return signal.fillna(False)


def run_backtest(df):

    trades = []

    signal = build_signal(df)

    idxs = np.where(signal.values)[0]

    print(f"Signals found: {len(idxs)}")

    for idx in idxs:

        if idx + 3 >= len(df):
            continue

        sig = df.iloc[idx]

        entry_zone = (
            sig["close"] +
            ((sig["high"] - sig["close"]) * 0.4)
        )

        filled = False

        for retrace_idx in range(
            idx + 1,
            min(idx + 5, len(df))
        ):

            rbar = df.iloc[retrace_idx]

            if rbar["high"] >= entry_zone:

                entry = float(entry_zone)

                filled = True

                entry_idx = retrace_idx

                break

        if not filled:
            continue

        stop = float(sig["high"]) + 1.0

        risk = stop - entry

        if risk <= 0:
            continue

        if risk > 80:
            continue

        target = entry - (risk * R_TARGET)

        outcome = "open"

        mfe = 0
        mae = 0

        for fwd in range(
            entry_idx,
            min(entry_idx + MAX_HOLD_BARS, len(df))
        ):

            bar = df.iloc[fwd]

            high = float(bar["high"])
            low = float(bar["low"])

            mfe = max(mfe, entry - low)
            mae = min(mae, entry - high)

            stop_hit = high >= stop
            target_hit = low <= target

            if stop_hit and target_hit:
                outcome = "loss"
                exit_price = stop
                break

            if stop_hit:
                outcome = "loss"
                exit_price = stop
                break

            if target_hit:
                outcome = "win"
                exit_price = target
                break

        if outcome == "open":

            final_bar = df.iloc[
                min(entry_idx + MAX_HOLD_BARS, len(df)-1)
            ]

            exit_price = float(final_bar["close"])

            pnl = entry - exit_price

            outcome = "win" if pnl > 0 else "loss"

        pnl_points = entry - exit_price

        rr = pnl_points / risk

        trades.append({
            "date": sig["trade_date_ct"],
            "entry_time": df.iloc[entry_idx]["ts_ct"],
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "exit_price": round(exit_price, 2),
            "outcome": outcome,
            "risk_points": round(risk, 2),
            "pnl_points": round(pnl_points, 2),
            "rr": round(rr, 2),
            "mfe": round(mfe, 2),
            "mae": round(mae, 2),
        })

    return pd.DataFrame(trades)


def summary(trades):

    if len(trades) == 0:
        return pd.DataFrame([{"trades": 0}])

    wins = trades["outcome"].eq("win").sum()

    gross_win = trades.loc[
        trades["pnl_points"] > 0,
        "pnl_points"
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["pnl_points"] < 0,
            "pnl_points"
        ].sum()
    )

    pf = gross_win / gross_loss if gross_loss > 0 else np.nan

    return pd.DataFrame([{
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": round((wins / len(trades)) * 100, 2),
        "net_points": round(trades["pnl_points"].sum(), 2),
        "avg_points": round(trades["pnl_points"].mean(), 2),
        "avg_rr": round(trades["rr"].mean(), 2),
        "profit_factor": round(pf, 2),
    }])


def main():

    print(f"Loading: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    df = norm(df)

    df["ts_ct"] = pd.to_datetime(df["ts_ct"])

    df = df.sort_values("ts_ct").reset_index(drop=True)

    print(f"Rows loaded: {len(df):,}")

    print("Running event sequence V2B...")

    trades = run_backtest(df)

    summ = summary(trades)

    trades.to_csv(TRADES_OUT, index=False)
    summ.to_csv(SUMMARY_OUT, index=False)

    if len(trades):

        trades["year"] = pd.to_datetime(
            trades["date"]
        ).dt.year

        yearly = (
            trades.groupby("year")
            .agg(
                trades=("outcome", "count"),
                wins=("outcome", lambda x: (x == "win").sum()),
                net_points=("pnl_points", "sum"),
                avg_rr=("rr", "mean"),
            )
            .reset_index()
        )

        yearly["win_rate"] = round(
            (yearly["wins"] / yearly["trades"]) * 100,
            2
        )

        yearly.to_csv(YEARLY_OUT, index=False)

    else:
        yearly = pd.DataFrame()

    print(f"\nSaved trades: {TRADES_OUT}")
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Saved yearly: {YEARLY_OUT}")

    print("\nSUMMARY")
    print(summ.to_string(index=False))

    if len(yearly):
        print("\nYEARLY")
        print(yearly.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()