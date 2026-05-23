from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np

ROOT = Path(r"D:\TradingResearch")
TRADES_PATH = ROOT / r"ict_trade_study\outputs\ict_keeper_trades_clean.csv"
OUTDIR = ROOT / r"ict_trade_study\outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_MATCHES = OUTDIR / "ict_trade_match_features.csv"
OUT_REVIEW = OUTDIR / "ict_trade_match_review.csv"

# Candidate NQ datasets. Script will use the first one that exists.
DATA_CANDIDATES = [
    ROOT / r"data_parquet\NQ_feature_factory.parquet",
    ROOT / r"data_parquet\nq_feature_factory.parquet",
    ROOT / r"data_parquet\NQ_master.parquet",
    ROOT / r"data_parquet\nq_master.parquet",
    ROOT / r"data_parquet\nq_1m.parquet",
]

LOOKBACK_MINUTES = 20
LOOKFORWARD_MINUTES = 60
MATCH_WINDOW_MINUTES = 4       # approximate screenshot time tolerance
PRICE_TOLERANCE_POINTS = 12.0  # loose because screenshots/manual prices can be approximate


def find_col(cols, options):
    lower = {c.lower(): c for c in cols}
    for opt in options:
        if opt.lower() in lower:
            return lower[opt.lower()]
    for c in cols:
        cl = c.lower()
        if any(opt.lower() in cl for opt in options):
            return c
    return None


def load_nq():
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Could not find NQ parquet file. Checked:\n" + "\n".join(str(p) for p in DATA_CANDIDATES)
        )

    print(f"Loading NQ data: {data_path}")
    df = pd.read_parquet(data_path)
    df.columns = [str(c).strip() for c in df.columns]

    time_col = find_col(df.columns, ["datetime", "timestamp", "time", "date_time", "ts"])
    date_col = find_col(df.columns, ["date"])
    open_col = find_col(df.columns, ["open"])
    high_col = find_col(df.columns, ["high"])
    low_col = find_col(df.columns, ["low"])
    close_col = find_col(df.columns, ["close"])
    volume_col = find_col(df.columns, ["volume", "vol"])

    required = {"open": open_col, "high": high_col, "low": low_col, "close": close_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}. Available columns: {df.columns.tolist()}")

    if time_col is not None:
        df["dt"] = pd.to_datetime(df[time_col], errors="coerce")
    elif date_col is not None:
        df["dt"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        raise ValueError(f"Could not identify datetime column. Available columns: {df.columns.tolist()}")

    df = df.rename(columns={
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
    })
    if volume_col:
        df = df.rename(columns={volume_col: "volume"})
    else:
        df["volume"] = np.nan

    df = df.dropna(subset=["dt", "open", "high", "low", "close"]).copy()
    df = df.sort_values("dt").reset_index(drop=True)
    df["date"] = df["dt"].dt.date
    df["time"] = df["dt"].dt.time

    # Daily Open Line proxy: first close/open for the regular data day available in dataset.
    day_first = df.groupby("date")["open"].first().rename("dol_proxy")
    df = df.merge(day_first, on="date", how="left")

    # Basic candle features
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["direction"] = np.where(df["close"] >= df["open"], "bull", "bear")
    df["ret_1"] = df["close"].diff()

    # Rolling context
    df["atr20_proxy"] = df["range"].rolling(20, min_periods=5).mean()
    df["body20_avg"] = df["body"].rolling(20, min_periods=5).mean()
    df["vol20_avg"] = df["volume"].rolling(20, min_periods=5).mean()

    # Simple 3-candle FVG detection
    df["bull_fvg"] = df["low"] > df["high"].shift(2)
    df["bear_fvg"] = df["high"] < df["low"].shift(2)
    df["bull_fvg_low"] = df["high"].shift(2)
    df["bull_fvg_high"] = df["low"]
    df["bear_fvg_low"] = df["high"]
    df["bear_fvg_high"] = df["low"].shift(2)

    return df


def parse_trade_dt(row):
    d = pd.to_datetime(row["date"], errors="coerce")
    t = pd.to_datetime(str(row["entry_time"]), errors="coerce")
    if pd.isna(d) or pd.isna(t):
        return pd.NaT
    return pd.Timestamp.combine(d.date(), t.time())


def max_drawup_drawdown(window, entry_price, side):
    if window.empty or pd.isna(entry_price):
        return np.nan, np.nan
    if side == "long":
        mfe = window["high"].max() - entry_price
        mae = entry_price - window["low"].min()
    else:
        mfe = entry_price - window["low"].min()
        mae = window["high"].max() - entry_price
    return float(mfe), float(mae)


def analyze_trade(nq, trade):
    approx_dt = parse_trade_dt(trade)
    side = str(trade.get("side", "")).lower().strip()
    entry_price = pd.to_numeric(trade.get("entry_price"), errors="coerce")

    base = trade.to_dict()
    base["approx_entry_dt"] = approx_dt

    if pd.isna(approx_dt) or pd.isna(entry_price) or side not in ["long", "short"]:
        base.update({"match_status": "bad_trade_inputs"})
        return base

    start = approx_dt - pd.Timedelta(minutes=MATCH_WINDOW_MINUTES)
    end = approx_dt + pd.Timedelta(minutes=MATCH_WINDOW_MINUTES)
    cand = nq[(nq["dt"] >= start) & (nq["dt"] <= end)].copy()

    if cand.empty:
        base.update({"match_status": "no_market_rows_in_time_window"})
        return base

    # Price match: entry should be inside candle range or nearest to range.
    cand["price_inside_bar"] = (cand["low"] <= entry_price) & (cand["high"] >= entry_price)
    cand["price_distance"] = np.where(
        cand["price_inside_bar"],
        0.0,
        np.minimum((cand["low"] - entry_price).abs(), (cand["high"] - entry_price).abs()),
    )
    cand["time_distance_min"] = (cand["dt"] - approx_dt).abs().dt.total_seconds() / 60.0
    cand["match_score"] = cand["price_distance"] * 3.0 + cand["time_distance_min"]

    match = cand.sort_values(["match_score", "time_distance_min"]).iloc[0]
    match_dt = match["dt"]

    pre = nq[(nq["dt"] >= match_dt - pd.Timedelta(minutes=LOOKBACK_MINUTES)) & (nq["dt"] < match_dt)].copy()
    post = nq[(nq["dt"] > match_dt) & (nq["dt"] <= match_dt + pd.Timedelta(minutes=LOOKFORWARD_MINUTES))].copy()
    around = nq[(nq["dt"] >= match_dt - pd.Timedelta(minutes=10)) & (nq["dt"] <= match_dt + pd.Timedelta(minutes=10))].copy()

    mfe_5, mae_5 = max_drawup_drawdown(post.head(5), entry_price, side)
    mfe_10, mae_10 = max_drawup_drawdown(post.head(10), entry_price, side)
    mfe_30, mae_30 = max_drawup_drawdown(post.head(30), entry_price, side)
    mfe_60, mae_60 = max_drawup_drawdown(post.head(60), entry_price, side)

    # V-shape proxy: strong move against intended side into the area, then fast reclaim/expansion.
    pre_10 = pre.tail(10)
    post_10 = post.head(10)
    if side == "long":
        pre_drive = pre_10["close"].iloc[-1] - pre_10["close"].iloc[0] if len(pre_10) >= 2 else np.nan
        post_reclaim = post_10["high"].max() - entry_price if not post_10.empty else np.nan
        v_shape_score = (-pre_drive if pd.notna(pre_drive) and pre_drive < 0 else 0) + (post_reclaim if pd.notna(post_reclaim) else 0)
    else:
        pre_drive = pre_10["close"].iloc[-1] - pre_10["close"].iloc[0] if len(pre_10) >= 2 else np.nan
        post_reclaim = entry_price - post_10["low"].min() if not post_10.empty else np.nan
        v_shape_score = (pre_drive if pd.notna(pre_drive) and pre_drive > 0 else 0) + (post_reclaim if pd.notna(post_reclaim) else 0)

    # Recent FVG context
    recent_fvgs = pre.tail(20)
    bull_recent = recent_fvgs[recent_fvgs["bull_fvg"]]
    bear_recent = recent_fvgs[recent_fvgs["bear_fvg"]]

    def nearest_fvg(fvgs, low_col, high_col):
        if fvgs.empty:
            return np.nan, np.nan, np.nan
        mid = (fvgs[low_col] + fvgs[high_col]) / 2.0
        idx = (mid - entry_price).abs().idxmin()
        return float(fvgs.loc[idx, low_col]), float(fvgs.loc[idx, high_col]), float(abs(mid.loc[idx] - entry_price))

    bull_low, bull_high, bull_dist = nearest_fvg(bull_recent, "bull_fvg_low", "bull_fvg_high")
    bear_low, bear_high, bear_dist = nearest_fvg(bear_recent, "bear_fvg_low", "bear_fvg_high")

    nearest_dir = "none"
    nearest_dist = np.nan
    if pd.notna(bull_dist) and (pd.isna(bear_dist) or bull_dist <= bear_dist):
        nearest_dir, nearest_dist = "bull", bull_dist
    elif pd.notna(bear_dist):
        nearest_dir, nearest_dist = "bear", bear_dist

    # DOL relationship proxy
    dol = match.get("dol_proxy", np.nan)
    dol_distance = entry_price - dol if pd.notna(dol) else np.nan
    if pd.isna(dol_distance):
        dol_relation = "unknown"
    elif abs(dol_distance) <= 5:
        dol_relation = "near_dol"
    elif dol_distance > 0:
        dol_relation = "above_dol"
    else:
        dol_relation = "below_dol"

    # Sweep/reclaim rough proxy over last 10 bars
    pre_high = pre_10["high"].max() if not pre_10.empty else np.nan
    pre_low = pre_10["low"].min() if not pre_10.empty else np.nan
    if side == "long":
        liquidity_sweep_proxy = bool(pd.notna(pre_low) and match["low"] <= pre_low and match["close"] > match["open"])
    else:
        liquidity_sweep_proxy = bool(pd.notna(pre_high) and match["high"] >= pre_high and match["close"] < match["open"])

    # Displacement proxy
    body20 = match.get("body20_avg", np.nan)
    displacement_ratio = match["body"] / body20 if pd.notna(body20) and body20 != 0 else np.nan
    displacement_candle = bool(pd.notna(displacement_ratio) and displacement_ratio >= 1.5)

    base.update({
        "match_status": "matched",
        "matched_dt": match_dt,
        "matched_open": match["open"],
        "matched_high": match["high"],
        "matched_low": match["low"],
        "matched_close": match["close"],
        "price_inside_bar": bool(match["price_inside_bar"]),
        "price_distance": float(match["price_distance"]),
        "time_distance_min": float(match["time_distance_min"]),
        "match_score": float(match["match_score"]),
        "dol_proxy": dol,
        "dol_distance": dol_distance,
        "dol_relation": dol_relation,
        "recent_bull_fvg_low": bull_low,
        "recent_bull_fvg_high": bull_high,
        "recent_bull_fvg_dist": bull_dist,
        "recent_bear_fvg_low": bear_low,
        "recent_bear_fvg_high": bear_high,
        "recent_bear_fvg_dist": bear_dist,
        "nearest_fvg_dir": nearest_dir,
        "nearest_fvg_dist": nearest_dist,
        "liquidity_sweep_proxy": liquidity_sweep_proxy,
        "displacement_ratio": displacement_ratio,
        "displacement_candle": displacement_candle,
        "pre_10m_drive_points": pre_drive,
        "post_10m_reclaim_points": post_reclaim,
        "v_shape_score_proxy": v_shape_score,
        "mfe_5m": mfe_5,
        "mae_5m": mae_5,
        "mfe_10m": mfe_10,
        "mae_10m": mae_10,
        "mfe_30m": mfe_30,
        "mae_30m": mae_30,
        "mfe_60m": mfe_60,
        "mae_60m": mae_60,
    })
    return base


def main():
    print(f"Loading trades: {TRADES_PATH}")
    trades = pd.read_csv(TRADES_PATH)
    print(f"Trades loaded: {len(trades)}")

    # Flag obvious typo for review, but do not auto-correct.
    suspicious = trades[(pd.to_numeric(trades["entry_price"], errors="coerce") < 15000) | (pd.to_numeric(trades["entry_price"], errors="coerce") > 30000)]
    if len(suspicious):
        print("\nSuspicious entry prices:")
        print(suspicious[["trade_id", "date", "entry_time", "entry_price", "side", "result"]].to_string(index=False))

    nq = load_nq()
    print(f"NQ rows loaded: {len(nq):,}")
    print(f"NQ range: {nq['dt'].min()} -> {nq['dt'].max()}")

    rows = []
    for _, trade in trades.iterrows():
        rows.append(analyze_trade(nq, trade))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_MATCHES, index=False)

    review_cols = [
        "trade_id", "date", "entry_time", "entry_price", "side", "result",
        "match_status", "matched_dt", "price_inside_bar", "price_distance", "time_distance_min",
        "dol_relation", "dol_distance", "nearest_fvg_dir", "nearest_fvg_dist",
        "liquidity_sweep_proxy", "displacement_ratio", "displacement_candle",
        "pre_10m_drive_points", "post_10m_reclaim_points", "v_shape_score_proxy",
        "mfe_10m", "mae_10m", "mfe_30m", "mae_30m", "mfe_60m", "mae_60m",
    ]
    existing = [c for c in review_cols if c in out.columns]
    out[existing].to_csv(OUT_REVIEW, index=False)

    print(f"\nSaved full features: {OUT_MATCHES}")
    print(f"Saved review file: {OUT_REVIEW}")

    print("\nMatch summary:")
    print(out["match_status"].value_counts(dropna=False).to_string())

    matched = out[out["match_status"] == "matched"].copy()
    if not matched.empty:
        print("\nFirst-pass feature summary by result:")
        summary = matched.groupby("result", dropna=False).agg(
            trades=("trade_id", "count"),
            avg_mfe_30m=("mfe_30m", "mean"),
            avg_mae_30m=("mae_30m", "mean"),
            avg_v_shape_score=("v_shape_score_proxy", "mean"),
            avg_nearest_fvg_dist=("nearest_fvg_dist", "mean"),
            displacement_rate=("displacement_candle", "mean"),
            sweep_proxy_rate=("liquidity_sweep_proxy", "mean"),
        ).reset_index()
        print(summary.to_string(index=False))

        print("\nReview rows:")
        print(matched[existing].to_string(index=False))


if __name__ == "__main__":
    main()
