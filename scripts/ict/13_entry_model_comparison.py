from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

R = 2.0
MAX_HOLD = 90


def norm(df):
    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]
    return df


def b(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def build_signal(df):

    bear_trend = (
        (df["ema_20"] < df["ema_50"]) &
        (df["close"] < df["ema_20"])
    )

    displacement = b(
        df["bear_displacement"]
    )

    sweep = b(
        df["any_liquidity_sweep_reclaim_last_30m"]
    )

    expansion = (
        pd.to_numeric(
            df["range_expansion_20"],
            errors="coerce"
        ) > 1.15
    )

    vol = (
        pd.to_numeric(
            df["rel_vol_20"],
            errors="coerce"
        ) > 1.05
    )

    session = (
        (df["minute_of_day_ct"] >= 510) &
        (df["minute_of_day_ct"] <= 690)
    )

    signal = (
        bear_trend &
        displacement &
        sweep &
        expansion &
        vol &
        session
    )

    return signal.fillna(False)


def run_model(df, entry_mode):

    trades = []

    signal = build_signal(df)

    idxs = np.where(signal.values)[0]

    print(f"{entry_mode}: {len(idxs)} signals")

    for idx in idxs:

        if idx + 5 >= len(df):
            continue

        sig = df.iloc[idx]

        sig_high = float(sig["high"])
        sig_low = float(sig["low"])
        sig_close = float(sig["close"])

        signal_range = sig_high - sig_low

        if signal_range <= 0:
            continue

        if entry_mode == "market":

            entry = sig_close

        elif entry_mode == "50pct":

            entry = sig_high - (signal_range * 0.5)

        elif entry_mode == "deep_retrace":

            entry = sig_high - (signal_range * 0.25)

        elif entry_mode == "shallow_retrace":

            entry = sig_high - (signal_range * 0.75)

        else:
            continue

        filled = False

        for retrace_idx in range(
            idx + 1,
            min(idx + 6, len(df))
        ):

            rb = df.iloc[retrace_idx]

            if float(rb["high"]) >= entry:

                filled = True
                entry_idx = retrace_idx
                break

        if not filled:
            continue

        stop = sig_high + 2

        risk = stop - entry

        if risk <= 4:
            continue

        if risk > 40:
            continue

        target = entry - (risk * R)

        outcome = "open"
        exit_price = entry

        for fwd in range(
            entry_idx,
            min(entry_idx + MAX_HOLD, len(df))
        ):

            bar = df.iloc[fwd]

            high = float(bar["high"])
            low = float(bar["low"])

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
                min(entry_idx + MAX_HOLD, len(df)-1)
            ]

            exit_price = float(final["close"])

            pnl = entry - exit_price

            outcome = (
                "win"
                if pnl > 0
                else "loss"
            )

        pnl = entry - exit_price

        rr = pnl / risk

        trades.append({
            "entry_mode": entry_mode,
            "pnl_points": round(pnl, 2),
            "rr": round(rr, 2),
            "outcome": outcome
        })

    return pd.DataFrame(trades)


def summarize(trades, mode):

    if len(trades) == 0:

        return {
            "entry_mode": mode,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0,
            "avg_rr": 0,
            "profit_factor": 0
        }

    wins = (
        trades["outcome"] == "win"
    ).sum()

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

    pf = (
        gross_win / gross_loss
        if gross_loss > 0
        else np.nan
    )

    return {
        "entry_mode": mode,
        "trades": len(trades),
        "wins": wins,
        "losses": len(trades) - wins,
        "win_rate": round(
            wins / len(trades) * 100,
            2
        ),
        "net_points": round(
            trades["pnl_points"].sum(),
            2
        ),
        "avg_rr": round(
            trades["rr"].mean(),
            2
        ),
        "profit_factor": round(
            pf,
            2
        )
    }


def main():

    print(f"Loading: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    df = norm(df)

    results = []

    modes = [
        "market",
        "50pct",
        "deep_retrace",
        "shallow_retrace"
    ]

    for mode in modes:

        trades = run_model(df, mode)

        results.append(
            summarize(trades, mode)
        )

    out = pd.DataFrame(results)

    print("\nENTRY MODEL COMPARISON")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
