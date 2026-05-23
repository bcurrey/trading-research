from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\continuation_short_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "continuation_short_v1_trades.csv"
SUMMARY_OUT = OUTDIR / "continuation_short_v1_summary.csv"
YEARLY_OUT = OUTDIR / "continuation_short_v1_yearly.csv"
MONTHLY_OUT = OUTDIR / "continuation_short_v1_monthly.csv"


R_MULTIPLE_TARGET = 2.0
MAX_HOLD_BARS = 60


def norm_cols(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def build_signal(df):
    # High-side sweep
    sweep_cols = [
        "pdh_sweep_reclaim_now",
        "pmh_sweep_reclaim_now",
        "asia_high_sweep_reclaim_now",
        "london_high_sweep_reclaim_now",
    ]

    existing = [c for c in sweep_cols if c in df.columns]

    if existing:
        df["high_sweep_reject"] = df[existing].fillna(False).any(axis=1)
    else:
        df["high_sweep_reject"] = False

    below_dol = df["close"] < df["dol"]
    below_vwap = df["close"] < df["vwap"]

    # Bear displacement proxy
    if "bear_displacement" in df.columns:
        bear_disp = df["bear_displacement"].fillna(False)
    else:
        bear_disp = (
            (df["close"] < df["open"]) &
            (df["body_abs"] > df["atr_14"] * 0.6)
        )

    # Session filter
    session_ok = (
        (df["minute_of_day_ct"] >= (8 * 60 + 30)) &
        (df["minute_of_day_ct"] <= (12 * 60))
    )

    signal = (
        df["high_sweep_reject"] &
        below_dol &
        below_vwap &
        bear_disp &
        session_ok
    )

    return signal.fillna(False)


def run_backtest(df):
    trades = []

    signal = build_signal(df)

    signal_idxs = np.where(signal.values)[0]

    for idx in signal_idxs:

        if idx + 2 >= len(df):
            continue

        row = df.iloc[idx]

        entry_bar = df.iloc[idx + 1]

        entry = float(entry_bar["open"])

        stop = max(
            float(row["high"]),
            float(entry_bar["high"])
        ) + 1.0

        risk = stop - entry

        if risk <= 0:
            continue

        target = entry - (risk * R_MULTIPLE_TARGET)

        outcome = "open"
        exit_price = np.nan
        exit_time = None
        mfe = 0.0
        mae = 0.0

        for fwd in range(idx + 1, min(idx + 1 + MAX_HOLD_BARS, len(df))):

            bar = df.iloc[fwd]

            low = float(bar["low"])
            high = float(bar["high"])

            open_pnl = entry - float(bar["close"])

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

            if pnl > 0:
                outcome = "win"
            else:
                outcome = "loss"

        pnl_points = entry - exit_price
        rr = pnl_points / risk if risk > 0 else np.nan

        trades.append({
            "date": row["trade_date_ct"],
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
            "swept_levels": "|".join([
                x for x in [
                    "PDH" if bool(row.get("pdh_sweep_reclaim_now", False)) else "",
                    "PMH" if bool(row.get("pmh_sweep_reclaim_now", False)) else "",
                    "ASIA_HIGH" if bool(row.get("asia_high_sweep_reclaim_now", False)) else "",
                    "LONDON_HIGH" if bool(row.get("london_high_sweep_reclaim_now", False)) else "",
                ] if x
            ]),
            "below_dol": bool(row["close"] < row["dol"]),
            "below_vwap": bool(row["close"] < row["vwap"]),
        })

    return pd.DataFrame(trades)


def build_summary(trades):

    if len(trades) == 0:
        return pd.DataFrame([{
            "trades": 0
        }])

    wins = trades["outcome"].eq("win").sum()
    losses = trades["outcome"].eq("loss").sum()

    gross_win = trades.loc[trades["pnl_points"] > 0, "pnl_points"].sum()
    gross_loss = abs(trades.loc[trades["pnl_points"] < 0, "pnl_points"].sum())

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

    print("Running continuation short prototype...")

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

        yearly["win_rate"] = round((yearly["wins"] / yearly["trades"]) * 100, 2)

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

        monthly["win_rate"] = round((monthly["wins"] / monthly["trades"]) * 100, 2)

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
