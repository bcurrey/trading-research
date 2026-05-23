from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\quality_model_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "quality_model_v1_trades.csv"
SUMMARY_OUT = OUTDIR / "quality_model_v1_summary.csv"
YEARLY_OUT = OUTDIR / "quality_model_v1_yearly.csv"
MONTHLY_OUT = OUTDIR / "quality_model_v1_monthly.csv"

R_TARGET = 2.0
MAX_HOLD_BARS = 90


def norm(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def boolcol(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def score_short(df):

    score = pd.Series(0, index=df.index)

    # -------------------------
    # liquidity sweep rejection
    # -------------------------

    high_sweep = (
        boolcol(df, "pdh_sweep_reclaim_now") |
        boolcol(df, "pmh_sweep_reclaim_now") |
        boolcol(df, "asia_high_sweep_reclaim_now")
    )

    score += high_sweep.astype(int) * 2

    # -------------------------
    # below DOL / VWAP
    # -------------------------

    below_dol = df["close"] < df["dol"]
    below_vwap = df["close"] < df["vwap"]

    score += below_dol.astype(int) * 2
    score += below_vwap.astype(int) * 2

    # -------------------------
    # rejection displacement
    # -------------------------

    bear_candle = df["close"] < df["open"]

    close_near_low = (
        ((df["close"] - df["low"]) /
        (df["high"] - df["low"]).replace(0, np.nan))
        < 0.30
    )

    body_abs = pd.to_numeric(df.get("body_abs", 0), errors="coerce")
    atr = pd.to_numeric(df.get("atr_14", 0), errors="coerce")

    strong_body = body_abs > (atr * 0.6)

    score += (
        bear_candle &
        close_near_low &
        strong_body
    ).astype(int) * 3

    # -------------------------
    # FVG quality
    # -------------------------

    if "bear_fvg_size" in df.columns:

        fvg = pd.to_numeric(
            df["bear_fvg_size"],
            errors="coerce"
        )

        healthy = (fvg >= 3) & (fvg <= 15)

        score += healthy.astype(int) * 2

    # -------------------------
    # rally before short
    # -------------------------

    prior_drive = (
        df["close"] -
        df["close"].shift(10)
    )

    score += (prior_drive >= 10).astype(int) * 2

    # -------------------------
    # exhaustion penalty
    # -------------------------

    extended = (
        (df["close"] - df["ema_20"]).abs() >
        (df["atr_14"] * 3)
    )

    score -= extended.astype(int) * 3

    return score


def build_signals(df):

    score = score_short(df)

    session_ok = (
        (df["minute_of_day_ct"] >= (8 * 60 + 30)) &
        (df["minute_of_day_ct"] <= (11 * 60))
    )

    signal = (
        (score >= 9) &
        session_ok
    )

    return signal.fillna(False), score


def run_backtest(df):

    trades = []

    signal, score = build_signals(df)

    signal_idxs = np.where(signal.values)[0]

    print(f"Signals found: {len(signal_idxs)}")

    for idx in signal_idxs:

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

        mfe = 0.0
        mae = 0.0

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
            "quality_score": int(score.iloc[idx]),
        })

    return pd.DataFrame(trades)


def summary(trades):

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

    return pd.DataFrame([{
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


def main():

    print(f"Loading: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    df = norm(df)

    df["ts_ct"] = pd.to_datetime(df["ts_ct"])

    df = df.sort_values("ts_ct").reset_index(drop=True)

    print(f"Rows loaded: {len(df):,}")

    print("Running quality model V1...")

    trades = run_backtest(df)

    summ = summary(trades)

    if len(trades):

        trades["year"] = pd.to_datetime(
            trades["date"]
        ).dt.year

        trades["month"] = pd.to_datetime(
            trades["date"]
        ).dt.to_period("M").astype(str)

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
    summ.to_csv(SUMMARY_OUT, index=False)
    yearly.to_csv(YEARLY_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)

    print(f"\nSaved trades: {TRADES_OUT}")
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Saved yearly: {YEARLY_OUT}")
    print(f"Saved monthly: {MONTHLY_OUT}")

    print("\nSUMMARY")
    print(summ.to_string(index=False))

    if len(yearly):
        print("\nYEARLY")
        print(yearly.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
