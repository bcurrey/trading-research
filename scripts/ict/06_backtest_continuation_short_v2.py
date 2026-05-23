from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\continuation_short_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "continuation_short_v2_trades.csv"
SUMMARY_OUT = OUTDIR / "continuation_short_v2_summary.csv"
YEARLY_OUT = OUTDIR / "continuation_short_v2_yearly.csv"
MONTHLY_OUT = OUTDIR / "continuation_short_v2_monthly.csv"

R_MULTIPLE_TARGET = 2.0
MAX_HOLD_BARS = 60


def norm_cols(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def rolling_all_true(series, window=2):
    return (
        series.astype(int)
        .rolling(window, min_periods=window)
        .sum()
        .eq(window)
    )


def build_signal(df):

    # -----------------------------
    # HIGH QUALITY SWEEP FILTER
    # -----------------------------

    major_sweep_cols = [
        "pdh_sweep_reclaim_now",
        "pmh_sweep_reclaim_now",
        "asia_high_sweep_reclaim_now",
    ]

    existing = [c for c in major_sweep_cols if c in df.columns]

    if existing:
        df["major_high_sweep"] = df[existing].fillna(False).any(axis=1)
        df["major_sweep_count"] = df[existing].fillna(False).sum(axis=1)
    else:
        df["major_high_sweep"] = False
        df["major_sweep_count"] = 0

    # -----------------------------
    # VWAP ACCEPTANCE FILTER
    # -----------------------------

    below_vwap = (df["close"] < df["vwap"]).fillna(False)

    df["two_closes_below_vwap"] = rolling_all_true(below_vwap, 2)

    # -----------------------------
    # DOL FILTER
    # -----------------------------

    below_dol = (df["close"] < df["dol"]).fillna(False)

    # -----------------------------
    # STRONG REJECTION CANDLE
    # -----------------------------

    body_pct = pd.to_numeric(df.get("body_pct", 0), errors="coerce").fillna(0)
    rel_vol = pd.to_numeric(df.get("rel_vol_20", 0), errors="coerce").fillna(0)
    body_abs = pd.to_numeric(df.get("body_abs", 0), errors="coerce").fillna(0)
    atr = pd.to_numeric(df.get("atr_14", 0), errors="coerce").fillna(0)

    bear_candle = (df["close"] < df["open"]).fillna(False)

    close_near_lows = (
        ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan))
        < 0.25
    ).fillna(False)

    large_body = (
        body_abs > (atr * 0.6)
    ).fillna(False)

    strong_body_pct = (
        body_pct > 0.55
    ).fillna(False)

    high_rel_vol = (
        rel_vol > 1.2
    ).fillna(False)

    strong_rejection = (
        bear_candle &
        close_near_lows &
        large_body &
        strong_body_pct &
        high_rel_vol
    )

    # -----------------------------
    # SESSION FILTER
    # 8:30–10:30 CT
    # -----------------------------

    session_ok = (
        (df["minute_of_day_ct"] >= (8 * 60 + 30)) &
        (df["minute_of_day_ct"] <= (10 * 60 + 30))
    )

    # -----------------------------
    # OPTIONAL QUALITY TIER
    # -----------------------------

    df["elite_sweep_confluence"] = (
        df["major_sweep_count"] >= 2
    )

    # -----------------------------
    # FINAL SIGNAL
    # -----------------------------

    signal = (
        df["major_high_sweep"] &
        below_dol &
        df["two_closes_below_vwap"] &
        strong_rejection &
        session_ok
    )

    return signal.fillna(False)


def run_backtest(df):

    trades = []

    signal = build_signal(df)

    signal_idxs = np.where(signal.values)[0]

    print(f"Signals found: {len(signal_idxs)}")

    for idx in signal_idxs:

        if idx + 2 >= len(df):
            continue

        signal_bar = df.iloc[idx]
        entry_bar = df.iloc[idx + 1]

        entry = float(entry_bar["open"])

        stop = max(
            float(signal_bar["high"]),
            float(entry_bar["high"])
        ) + 1.0

        risk = stop - entry

        if risk <= 0:
            continue

        if risk > 80:
            continue

        target = entry - (risk * R_MULTIPLE_TARGET)

        outcome = "open"
        exit_price = np.nan
        exit_time = None

        mfe = 0.0
        mae = 0.0

        for fwd in range(idx + 1, min(idx + 1 + MAX_HOLD_BARS, len(df))):

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
                exit_time = bar["ts_ct"]
                break

            if stop_hit:
                outcome = "loss"
                exit_price = stop
                exit_time = bar["ts_ct"]
                break

            if target_hit:
                outcome = "win"
                exit_price = target
                exit_time = bar["ts_ct"]
                break

        if outcome == "open":

            final_bar = df.iloc[min(idx + MAX_HOLD_BARS, len(df)-1)]

            exit_price = float(final_bar["close"])
            exit_time = final_bar["ts_ct"]

            pnl = entry - exit_price

            outcome = "win" if pnl > 0 else "loss"

        pnl_points = entry - exit_price
        rr = pnl_points / risk if risk > 0 else np.nan

        trades.append({
            "date": signal_bar["trade_date_ct"],
            "entry_time": entry_bar["ts_ct"],
            "side": "short",

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

            "major_sweep_count": int(signal_bar["major_sweep_count"]),
            "elite_sweep_confluence": bool(signal_bar["elite_sweep_confluence"]),

            "below_dol": bool(signal_bar["close"] < signal_bar["dol"]),
            "below_vwap": bool(signal_bar["close"] < signal_bar["vwap"]),
        })

    return pd.DataFrame(trades)


def build_summary(trades):

    if len(trades) == 0:
        return pd.DataFrame([{"trades": 0}])

    wins = trades["outcome"].eq("win").sum()
    losses = trades["outcome"].eq("loss").sum()

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

    summary = pd.DataFrame([{
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round((wins / len(trades)) * 100, 2),
        "net_points": round(trades["pnl_points"].sum(), 2),
        "avg_points": round(trades["pnl_points"].mean(), 2),
        "avg_rr": round(trades["rr"].mean(), 2),
        "profit_factor": round(pf, 2),
        "avg_mfe": round(trades["mfe"].mean(), 2),
        "avg_mae": round(trades["mae"].mean(), 2),
    }])

    return summary


def main():

    print(f"Loading data: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    df = norm_cols(df)

    df["ts_ct"] = pd.to_datetime(df["ts_ct"])

    df = df.sort_values("ts_ct").reset_index(drop=True)

    print(f"Rows loaded: {len(df):,}")

    print("Running continuation short V2...")

    trades = run_backtest(df)

    summary = build_summary(trades)

    if len(trades):

        trades["year"] = pd.to_datetime(trades["date"]).dt.year
        trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M").astype(str)

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

        monthly = (
            trades.groupby("month")
            .agg(
                trades=("outcome", "count"),
                wins=("outcome", lambda x: (x == "win").sum()),
                net_points=("pnl_points", "sum"),
                avg_rr=("rr", "mean"),
            )
            .reset_index()
        )

        monthly["win_rate"] = round(
            (monthly["wins"] / monthly["trades"]) * 100,
            2
        )

    else:
        yearly = pd.DataFrame()
        monthly = pd.DataFrame()

    trades.to_csv(TRADES_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    yearly.to_csv(YEARLY_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)

    print(f"\nSaved trades: {TRADES_OUT}")
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Saved yearly: {YEARLY_OUT}")
    print(f"Saved monthly: {MONTHLY_OUT}")

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    if len(yearly):
        print("\nYEARLY")
        print(yearly.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
