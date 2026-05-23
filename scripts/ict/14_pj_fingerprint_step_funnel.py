from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\pj_fingerprint_step_funnel"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "pj_step_funnel_summary.csv"
TRADES_STEP1_OUT = OUTDIR / "step1_displacement_after_liquidity_trades.csv"
TRADES_STEP12_OUT = OUTDIR / "step1_2_deep_retrace_trades.csv"
TRADES_STEP123_OUT = OUTDIR / "step1_2_3_continuation_trades.csv"

LOOKBACK_DAYS = 365
R_TARGET = 2.0
MAX_HOLD_BARS = 90
RETRACE_LOOKAHEAD_BARS = 5


def norm(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def as_bool(s):
    return (
        s.fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def get_bool(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return as_bool(df[col])


def summarize(trades, label):
    if len(trades) == 0:
        return {
            "stage": label,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "net_points": 0,
            "avg_points": 0,
            "avg_rr": 0,
            "profit_factor": 0,
            "avg_risk": 0,
            "avg_mfe": 0,
            "avg_mae": 0,
        }

    wins = trades["outcome"].eq("win").sum()
    losses = trades["outcome"].eq("loss").sum()

    gross_win = trades.loc[trades["pnl_points"] > 0, "pnl_points"].sum()
    gross_loss = abs(trades.loc[trades["pnl_points"] < 0, "pnl_points"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan

    return {
        "stage": label,
        "trades": len(trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": round(wins / len(trades) * 100, 2),
        "net_points": round(trades["pnl_points"].sum(), 2),
        "avg_points": round(trades["pnl_points"].mean(), 2),
        "avg_rr": round(trades["rr"].mean(), 2),
        "profit_factor": round(pf, 2),
        "avg_risk": round(trades["risk_points"].mean(), 2),
        "avg_mfe": round(trades["mfe"].mean(), 2),
        "avg_mae": round(trades["mae"].mean(), 2),
    }


def build_step1_signal(df):
    liquidity_event = get_bool(df, "any_liquidity_sweep_reclaim_last_30m")
    bear_displacement = get_bool(df, "bear_displacement")

    bear_trend_context = (
        (df["close"] < df["ema_20"]) &
        (df["ema_20"] < df["ema_50"])
    )

    session_ok = (
        (df["minute_of_day_ct"] >= 510) &
        (df["minute_of_day_ct"] <= 690)
    )

    expansion_ok = pd.to_numeric(df["range_expansion_20"], errors="coerce") > 1.10
    vol_ok = pd.to_numeric(df["rel_vol_20"], errors="coerce") > 1.00

    signal = (
        liquidity_event &
        bear_displacement &
        bear_trend_context &
        session_ok &
        expansion_ok &
        vol_ok
    )

    return signal.fillna(False)


def evaluate_short_trade(df, entry_idx, entry, stop, max_hold=MAX_HOLD_BARS, r_target=R_TARGET):
    risk = stop - entry

    if risk <= 0:
        return None

    target = entry - (risk * r_target)
    outcome = "open"
    exit_price = entry
    exit_time = None
    mfe = 0.0
    mae = 0.0

    for fwd in range(entry_idx, min(entry_idx + max_hold, len(df))):
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
        final = df.iloc[min(entry_idx + max_hold, len(df) - 1)]
        exit_price = float(final["close"])
        exit_time = final["ts_ct"]
        pnl = entry - exit_price
        outcome = "win" if pnl > 0 else "loss"

    pnl_points = entry - exit_price
    rr = pnl_points / risk if risk > 0 else np.nan

    return {
        "stop": round(stop, 2),
        "target": round(target, 2),
        "exit_price": round(exit_price, 2),
        "exit_time": exit_time,
        "outcome": outcome,
        "risk_points": round(risk, 2),
        "pnl_points": round(pnl_points, 2),
        "rr": round(rr, 2),
        "mfe": round(mfe, 2),
        "mae": round(mae, 2),
    }


def run_step1_market(df, signal):
    trades = []
    idxs = np.where(signal.values)[0]

    for idx in idxs:
        if idx + 1 >= len(df):
            continue

        sig = df.iloc[idx]
        entry_idx = idx + 1
        entry_bar = df.iloc[entry_idx]
        entry = float(entry_bar["open"])
        stop = float(sig["high"]) + 2.0
        risk = stop - entry

        if risk <= 4 or risk > 80:
            continue

        result = evaluate_short_trade(df, entry_idx, entry, stop)
        if result is None:
            continue

        trades.append({
            "stage": "step1_market",
            "signal_time": sig["ts_ct"],
            "entry_time": entry_bar["ts_ct"],
            "date": sig["trade_date_ct"],
            "entry": round(entry, 2),
            "signal_high": round(float(sig["high"]), 2),
            "signal_low": round(float(sig["low"]), 2),
            "signal_close": round(float(sig["close"]), 2),
            **result
        })

    return pd.DataFrame(trades)


def run_step12_deep_retrace(df, signal):
    trades = []
    idxs = np.where(signal.values)[0]

    for idx in idxs:
        if idx + 2 >= len(df):
            continue

        sig = df.iloc[idx]
        sig_high = float(sig["high"])
        sig_low = float(sig["low"])
        rng = sig_high - sig_low

        if rng <= 0:
            continue

        entry = sig_high - (rng * 0.25)
        stop = sig_high + 2.0
        risk = stop - entry

        if risk <= 4 or risk > 80:
            continue

        filled = False
        entry_idx = None

        for j in range(idx + 1, min(idx + 1 + RETRACE_LOOKAHEAD_BARS, len(df))):
            bar = df.iloc[j]
            if float(bar["high"]) >= entry:
                filled = True
                entry_idx = j
                break

        if not filled:
            continue

        result = evaluate_short_trade(df, entry_idx, entry, stop)
        if result is None:
            continue

        trades.append({
            "stage": "step1_2_deep_retrace",
            "signal_time": sig["ts_ct"],
            "entry_time": df.iloc[entry_idx]["ts_ct"],
            "date": sig["trade_date_ct"],
            "entry": round(entry, 2),
            "signal_high": round(sig_high, 2),
            "signal_low": round(sig_low, 2),
            "signal_close": round(float(sig["close"]), 2),
            "retrace_fill_bars": entry_idx - idx,
            **result
        })

    return pd.DataFrame(trades)


def main():
    print(f"Loading: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df = norm(df)
    df["ts_ct"] = pd.to_datetime(df["ts_ct"])
    df = df.sort_values("ts_ct").reset_index(drop=True)

    max_dt = df["ts_ct"].max()
    start_dt = max_dt - pd.Timedelta(days=LOOKBACK_DAYS)

    df = df[df["ts_ct"] >= start_dt].copy().reset_index(drop=True)

    print(f"Using last {LOOKBACK_DAYS} days:")
    print(f"{df['ts_ct'].min()} -> {df['ts_ct'].max()}")
    print(f"Rows: {len(df):,}")

    step1_signal = build_step1_signal(df)
    print(f"Step 1 signals: {int(step1_signal.sum())}")

    step1_trades = run_step1_market(df, step1_signal)
    step12_trades = run_step12_deep_retrace(df, step1_signal)

    if len(step12_trades):
        step123_trades = step12_trades[step12_trades["outcome"].eq("win")].copy()
        step123_trades["stage"] = "step1_2_3_continuation"
    else:
        step123_trades = pd.DataFrame()

    summary = pd.DataFrame([
        summarize(step1_trades, "Step 1 only: liquidity event + bearish displacement"),
        summarize(step12_trades, "Step 1 + 2: deep retrace filled"),
        summarize(step123_trades, "Step 1 + 2 + 3: continued to 2R target"),
    ])

    step1_trades.to_csv(TRADES_STEP1_OUT, index=False)
    step12_trades.to_csv(TRADES_STEP12_OUT, index=False)
    step123_trades.to_csv(TRADES_STEP123_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    print(f"\nSaved summary: {SUMMARY_OUT}")
    print(f"Saved Step 1 trades: {TRADES_STEP1_OUT}")
    print(f"Saved Step 1+2 trades: {TRADES_STEP12_OUT}")
    print(f"Saved Step 1+2+3 trades: {TRADES_STEP123_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
