from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\displacement_reclaim_model"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "trades.csv"
SUMMARY_OUT = OUTDIR / "summary.csv"
YEARLY_OUT = OUTDIR / "yearly.csv"

R = 2.5
MAX_HOLD = 120


def norm(df):
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def col(df, name, default=False):
    if name not in df.columns:
        return pd.Series(default, index=df.index)
    return df[name]


def build(df):

    bear_trend = (
        (df["ema_20"] < df["ema_50"]) &
        (df["close"] < df["ema_20"])
    )

    displacement = (
        col(df, "bear_displacement", False)
    )

    sweep_context = (
        col(df, "any_liquidity_sweep_reclaim_last_30m", False)
    )

    expansion = (
        df["range_expansion_20"] > 1.2
    )

    vol = (
        df["rel_vol_20"] > 1.1
    )

    not_chop = (
        df["atr_14"] > df["atr_50"] * 0.8
    )

    session = (
        (df["minute_of_day_ct"] >= 510) &
        (df["minute_of_day_ct"] <= 690)
    )

    signal = (
        bear_trend &
        displacement &
        sweep_context &
        expansion &
        vol &
        not_chop &
        session
    )

    return signal.fillna(False)


def run(df):

    trades = []

    sig = build(df)

    idxs = np.where(sig.values)[0]

    print(f"Signals: {len(idxs)}")

    for idx in idxs:

        if idx + 5 >= len(df):
            continue

        bar = df.iloc[idx]

        entry = float(bar["close"])

        stop = float(bar["high"]) + 2

        risk = stop - entry

        if risk <= 4 or risk > 40:
            continue

        target = entry - (risk * R)

        outcome = "open"

        exit_price = entry

        for fwd in range(
            idx + 1,
            min(idx + MAX_HOLD, len(df))
        ):

            x = df.iloc[fwd]

            high = float(x["high"])
            low = float(x["low"])

            if high >= stop:
                outcome = "loss"
                exit_price = stop
                break

            if low <= target:
                outcome = "win"
                exit_price = target
                break

        if outcome == "open":

            final = df.iloc[
                min(idx + MAX_HOLD, len(df)-1)
            ]

            exit_price = float(final["close"])

            pnl = entry - exit_price

            outcome = "win" if pnl > 0 else "loss"

        pnl = entry - exit_price

        rr = pnl / risk

        trades.append({
            "date": bar["trade_date_ct"],
            "entry_time": bar["ts_ct"],
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "pnl_points": round(pnl, 2),
            "rr": round(rr, 2),
            "outcome": outcome
        })

    return pd.DataFrame(trades)


def summarize(trades):

    if len(trades) == 0:
        return pd.DataFrame([{"trades": 0}]), pd.DataFrame()

    wins = (trades["outcome"] == "win").sum()

    gross_win = trades.loc[
        trades["pnl_points"] > 0,
        "pnl_points"
    ].sum()

    gross_loss = abs(
        trades.loc[
            trades["pnl_points"] < 0,
            "pnl_points"]
        .sum()
    )

    pf = gross_win / gross_loss if gross_loss > 0 else np.nan

    summary = pd.DataFrame([{
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": round(wins / len(trades) * 100, 2),
        "net_points": round(trades["pnl_points"].sum(), 2),
        "avg_points": round(trades["pnl_points"].mean(), 2),
        "avg_rr": round(trades["rr"].mean(), 2),
        "profit_factor": round(pf, 2)
    }])

    trades["year"] = pd.to_datetime(
        trades["date"]
    ).dt.year

    yearly = (
        trades.groupby("year")
        .agg(
            trades=("outcome", "count"),
            wins=("outcome", lambda x: (x == "win").sum()),
            net_points=("pnl_points", "sum"),
            avg_rr=("rr", "mean")
        )
        .reset_index()
    )

    yearly["win_rate"] = round(
        yearly["wins"] / yearly["trades"] * 100,
        2
    )

    return summary, yearly


def main():

    print(f"Loading: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    df = norm(df)

    df["ts_ct"] = pd.to_datetime(df["ts_ct"])

    df = df.sort_values("ts_ct").reset_index(drop=True)

    print(f"Rows: {len(df):,}")

    trades = run(df)

    summary, yearly = summarize(trades)

    trades.to_csv(TRADES_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    yearly.to_csv(YEARLY_OUT, index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    print("\nYEARLY")
    print(yearly.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
