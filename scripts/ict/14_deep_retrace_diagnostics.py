from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")
DATA_PATH = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"

OUTDIR = ROOT / r"research_outputs\deep_retrace_diagnostics"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = OUTDIR / "deep_retrace_trades.csv"
SUMMARY_OUT = OUTDIR / "deep_retrace_summary.csv"
YEARLY_OUT = OUTDIR / "deep_retrace_yearly.csv"
MONTHLY_OUT = OUTDIR / "deep_retrace_monthly.csv"
TOD_OUT = OUTDIR / "deep_retrace_time_of_day.csv"
RISK_OUT = OUTDIR / "deep_retrace_risk_buckets.csv"
FILL_OUT = OUTDIR / "deep_retrace_fill_bars.csv"

R = 2.0
MAX_HOLD = 90
RETRACE_LOOKAHEAD = 5


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


def build_signal(df):
    bear_trend = (
        (df["ema_20"] < df["ema_50"]) &
        (df["close"] < df["ema_20"])
    )

    displacement = get_bool(df, "bear_displacement")
    sweep = get_bool(df, "any_liquidity_sweep_reclaim_last_30m")

    expansion = pd.to_numeric(df["range_expansion_20"], errors="coerce") > 1.15
    vol = pd.to_numeric(df["rel_vol_20"], errors="coerce") > 1.05

    session = (
        (df["minute_of_day_ct"] >= 510) &
        (df["minute_of_day_ct"] <= 690)
    )

    return (
        bear_trend &
        displacement &
        sweep &
        expansion &
        vol &
        session
    ).fillna(False)


def evaluate_short(df, entry_idx, entry, stop):
    risk = stop - entry

    if risk <= 4 or risk > 40:
        return None

    target = entry - (risk * R)

    outcome = "open"
    exit_price = entry
    exit_time = None
    mfe = 0.0
    mae = 0.0

    for fwd in range(entry_idx, min(entry_idx + MAX_HOLD, len(df))):
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
        final = df.iloc[min(entry_idx + MAX_HOLD, len(df) - 1)]
        exit_price = float(final["close"])
        exit_time = final["ts_ct"]
        pnl = entry - exit_price
        outcome = "win" if pnl > 0 else "loss"

    pnl = entry - exit_price
    rr = pnl / risk

    return {
        "exit_time": exit_time,
        "exit_price": round(exit_price, 2),
        "target": round(target, 2),
        "outcome": outcome,
        "risk_points": round(risk, 2),
        "pnl_points": round(pnl, 2),
        "rr": round(rr, 2),
        "mfe": round(mfe, 2),
        "mae": round(mae, 2),
    }


def run_deep_retrace(df):
    signal = build_signal(df)
    idxs = np.where(signal.values)[0]

    print(f"Signals found: {len(idxs)}")

    trades = []

    for idx in idxs:
        if idx + 2 >= len(df):
            continue

        sig = df.iloc[idx]

        sig_high = float(sig["high"])
        sig_low = float(sig["low"])
        rng = sig_high - sig_low

        if rng <= 0:
            continue

        # Deep short retrace = high-side pullback, same as prior best script 13.
        entry = sig_high - (rng * 0.25)
        stop = sig_high + 2.0

        filled = False
        entry_idx = None

        for j in range(idx + 1, min(idx + 1 + RETRACE_LOOKAHEAD, len(df))):
            rb = df.iloc[j]
            if float(rb["high"]) >= entry:
                filled = True
                entry_idx = j
                break

        if not filled:
            continue

        result = evaluate_short(df, entry_idx, entry, stop)
        if result is None:
            continue

        entry_bar = df.iloc[entry_idx]

        trades.append({
            "date": sig["trade_date_ct"],
            "signal_time": sig["ts_ct"],
            "entry_time": entry_bar["ts_ct"],
            "minute_of_day_ct": int(sig["minute_of_day_ct"]),
            "hour_ct": int(sig["minute_of_day_ct"] // 60),
            "entry_mode": "deep_retrace_25pct_from_high",
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "signal_high": round(sig_high, 2),
            "signal_low": round(sig_low, 2),
            "signal_close": round(float(sig["close"]), 2),
            "signal_range": round(rng, 2),
            "retrace_fill_bars": entry_idx - idx,
            "dist_vwap": round(float(sig.get("dist_vwap", np.nan)), 2) if pd.notna(sig.get("dist_vwap", np.nan)) else np.nan,
            "dist_dol": round(float(sig.get("dist_dol", np.nan)), 2) if pd.notna(sig.get("dist_dol", np.nan)) else np.nan,
            "rel_vol_20": round(float(sig.get("rel_vol_20", np.nan)), 2) if pd.notna(sig.get("rel_vol_20", np.nan)) else np.nan,
            "range_expansion_20": round(float(sig.get("range_expansion_20", np.nan)), 2) if pd.notna(sig.get("range_expansion_20", np.nan)) else np.nan,
            **result,
        })

    return pd.DataFrame(trades)


def summarize_group(g, label):
    if len(g) == 0:
        return {
            "group": label,
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

    wins = g["outcome"].eq("win").sum()
    gross_win = g.loc[g["pnl_points"] > 0, "pnl_points"].sum()
    gross_loss = abs(g.loc[g["pnl_points"] < 0, "pnl_points"].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan

    return {
        "group": label,
        "trades": len(g),
        "wins": int(wins),
        "losses": int(len(g) - wins),
        "win_rate": round(wins / len(g) * 100, 2),
        "net_points": round(g["pnl_points"].sum(), 2),
        "avg_points": round(g["pnl_points"].mean(), 2),
        "avg_rr": round(g["rr"].mean(), 2),
        "profit_factor": round(pf, 2),
        "avg_risk": round(g["risk_points"].mean(), 2),
        "avg_mfe": round(g["mfe"].mean(), 2),
        "avg_mae": round(g["mae"].mean(), 2),
    }


def group_summary(df, group_col):
    rows = []
    for k, g in df.groupby(group_col, dropna=False):
        rows.append(summarize_group(g, str(k)))
    return pd.DataFrame(rows)


def main():
    print(f"Loading: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)
    df = norm(df)

    df["ts_ct"] = pd.to_datetime(df["ts_ct"])
    df = df.sort_values("ts_ct").reset_index(drop=True)

    print(f"Rows loaded: {len(df):,}")

    trades = run_deep_retrace(df)

    if len(trades) == 0:
        print("No trades found.")
        return

    trades["date"] = pd.to_datetime(trades["date"])
    trades["year"] = trades["date"].dt.year
    trades["month"] = trades["date"].dt.to_period("M").astype(str)
    trades["entry_hour"] = pd.to_datetime(trades["entry_time"]).dt.hour
    trades["risk_bucket"] = pd.cut(
        trades["risk_points"],
        bins=[0, 8, 12, 16, 20, 30, 40, 999],
        labels=["0-8", "8-12", "12-16", "16-20", "20-30", "30-40", "40+"],
        include_lowest=True,
    )

    summary = pd.DataFrame([summarize_group(trades, "ALL")])
    yearly = group_summary(trades, "year")
    monthly = group_summary(trades, "month")
    tod = group_summary(trades, "entry_hour")
    risk = group_summary(trades, "risk_bucket")
    fill = group_summary(trades, "retrace_fill_bars")

    trades.to_csv(TRADES_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    yearly.to_csv(YEARLY_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)
    tod.to_csv(TOD_OUT, index=False)
    risk.to_csv(RISK_OUT, index=False)
    fill.to_csv(FILL_OUT, index=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))

    print("\nYEARLY")
    print(yearly.to_string(index=False))

    print("\nTIME OF DAY")
    print(tod.to_string(index=False))

    print("\nRISK BUCKETS")
    print(risk.to_string(index=False))

    print("\nFILL BARS")
    print(fill.to_string(index=False))

    print(f"\nSaved trades: {TRADES_OUT}")
    print(f"Saved summary: {SUMMARY_OUT}")
    print(f"Saved yearly: {YEARLY_OUT}")
    print(f"Saved monthly: {MONTHLY_OUT}")
    print(f"Saved time-of-day: {TOD_OUT}")
    print(f"Saved risk buckets: {RISK_OUT}")
    print(f"Saved fill bars: {FILL_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
