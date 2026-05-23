from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\event_sequence_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "event_sequence_v1_trades.csv"
SUMMARY_OUT = OUTDIR / "event_sequence_v1_summary.csv"
YEARLY_OUT = OUTDIR / "event_sequence_v1_yearly.csv"

R_TARGET = 2.0
MAX_HOLD_BARS = 90


def norm(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def boolcol(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def build_events(df):

    # -----------------------------
    # EVENT 1: HIGH SWEEP
    # -----------------------------

    high_sweep = (
        boolcol(df, "pdh_sweep_reclaim_now") |
        boolcol(df, "pmh_sweep_reclaim_now") |
        boolcol(df, "asia_high_sweep_reclaim_now")
    )

    # -----------------------------
    # EVENT 2: REJECTION BELOW VWAP
    # -----------------------------

    rejection = (
        (df["close"] < df["vwap"]) &
        (df["close"] < df["open"])
    )

    # -----------------------------
    # EVENT 3: DISPLACEMENT
    # -----------------------------

    body_abs = pd.to_numeric(
        df.get("body_abs", 0),
        errors="coerce"
    )

    atr = pd.to_numeric(
        df.get("atr_14", 0),
        errors="coerce"
    )

    displacement = (
        body_abs > (atr * 0.7)
    )

    # -----------------------------
    # EVENT 4: SEQUENCE WINDOW
    # -----------------------------

    recent_sweep = (
        high_sweep
        .rolling(8, min_periods=1)
        .max()
        .astype(bool)
    )

    # -----------------------------
    # EVENT 5: SESSION FILTER
    # -----------------------------

    session_ok = (
        (df["minute_of_day_ct"] >= (8 * 60 + 30)) &
        (df["minute_of_day_ct"] <= (11 * 60))
    )

    # -----------------------------
    # FINAL SEQUENCE
    # -----------------------------

    signal = (
        recent_sweep &
        rejection &
        displacement &
        session_ok
    )

    return signal.fillna(False)


def run_backtest(df):

    trades = []

    signal = build_events(df)

    idxs = np.where(signal.values)[0]

    print(f"Signals found: {len(idxs)}")

    for idx in idxs:

        if idx + 2 >= len(df):
            continue

        sig = df.iloc[idx]
        entry_bar = df.iloc[idx + 1]

        entry = float(entry_bar["open"])

        stop = max(
            float(sig["high"]),
            float(entry_bar["high"])
        ) + 1.0

        risk = stop - entry

        if risk <= 0:
            continue

        if risk > 80:
            continue

        target = entry - (risk * R_TARGET)

        outcome = "open"
        exit_price = np.nan

        mfe = 0
        mae = 0

        for fwd in range(
            idx + 1,
            min(idx + 1 + MAX_HOLD_BARS, len(df))
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
                min(idx + MAX_HOLD_BARS, len(df)-1)
            ]

            exit_price = float(final_bar["close"])

            pnl = entry - exit_price

            outcome = "win" if pnl > 0 else "loss"

        pnl_points = entry - exit_price

        rr = pnl_points / risk

        trades.append({
            "date": sig["trade_date_ct"],
            "entry_time": entry_bar["ts_ct"],
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

    print("Running event sequence model...")

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
