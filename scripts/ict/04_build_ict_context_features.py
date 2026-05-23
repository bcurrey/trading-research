from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")
INPUT = ROOT / r"data_parquet\NQ_feature_factory.parquet"
OUTPUT = ROOT / r"data_parquet\NQ_feature_factory_ict_context.parquet"
REPORT = ROOT / r"ict_trade_study\outputs\ict_context_feature_report.csv"

ASIA_START_MIN = 17 * 60      # 17:00 CT
ASIA_END_MIN = 0              # 00:00 CT
LONDON_START_MIN = 0          # 00:00 CT
LONDON_END_MIN = 7 * 60 + 30  # 07:30 CT
NY_START_MIN = 8 * 60 + 30    # 08:30 CT
NY_END_MIN = 15 * 60          # 15:00 CT

SWEEP_LOOKBACK = 30
DOL_LOOKBACK = 10


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def rolling_bool_by_session(df: pd.DataFrame, col: str, window: int) -> pd.Series:
    return (
        df.groupby("ict_session_date", sort=False)[col]
        .rolling(window, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(bool)
    )


def main():
    print(f"Loading: {INPUT}")
    df = pd.read_parquet(INPUT)
    df = norm_cols(df)
    print(f"Rows loaded: {len(df):,}")

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "ts_ct" not in df.columns:
        raise ValueError("Missing ts_ct column. Your current feature factory should include ts_ct.")

    df["ts_ct"] = pd.to_datetime(df["ts_ct"], errors="coerce")
    df = df.dropna(subset=["ts_ct"]).sort_values("ts_ct").reset_index(drop=True)

    # Basic time/session fields
    df["trade_date_ct"] = df["ts_ct"].dt.date
    df["minute_of_day_ct"] = df["ts_ct"].dt.hour * 60 + df["ts_ct"].dt.minute
    df["weekday_ct"] = df["ts_ct"].dt.weekday

    # Futures-style ICT session date: 17:00 CT belongs to next trade day
    base_date = pd.to_datetime(df["trade_date_ct"].astype(str))
    df["ict_session_date"] = np.where(
        df["minute_of_day_ct"] >= ASIA_START_MIN,
        (base_date + pd.Timedelta(days=1)).dt.date,
        base_date.dt.date,
    )

    df["is_asia_session"] = (df["minute_of_day_ct"] >= ASIA_START_MIN) | (df["minute_of_day_ct"] < ASIA_END_MIN)
    df["is_london_session"] = (df["minute_of_day_ct"] >= LONDON_START_MIN) & (df["minute_of_day_ct"] < LONDON_END_MIN)
    df["is_ny_rth_session"] = (df["minute_of_day_ct"] >= NY_START_MIN) & (df["minute_of_day_ct"] <= NY_END_MIN)

    print("Building DOL / PDH / PDL...")
    g = df.groupby("ict_session_date", sort=False)

    # Daily open line / session running extremes
    df["dol"] = g["open"].transform("first")
    df["session_high_so_far"] = g["high"].cummax()
    df["session_low_so_far"] = g["low"].cummin()

    daily = g.agg(
        session_open=("open", "first"),
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_close=("close", "last"),
    ).reset_index()

    daily["pdh"] = daily["session_high"].shift(1)
    daily["pdl"] = daily["session_low"].shift(1)
    daily["pdc"] = daily["session_close"].shift(1)
    daily["prior_session_mid"] = (daily["pdh"] + daily["pdl"]) / 2
    daily["prior_session_range"] = daily["pdh"] - daily["pdl"]

    df = df.merge(
        daily[["ict_session_date", "pdh", "pdl", "pdc", "prior_session_mid", "prior_session_range"]],
        on="ict_session_date",
        how="left",
    )

    print("Building Asia / London levels...")
    asia = (
        df[df["is_asia_session"]]
        .groupby("ict_session_date", sort=False)
        .agg(asia_high=("high", "max"), asia_low=("low", "min"))
        .reset_index()
    )

    london = (
        df[df["is_london_session"]]
        .groupby("ict_session_date", sort=False)
        .agg(london_high=("high", "max"), london_low=("low", "min"))
        .reset_index()
    )

    nyopen = (
        df[df["minute_of_day_ct"] >= NY_START_MIN]
        .groupby("ict_session_date", sort=False)
        .agg(ny_open=("open", "first"))
        .reset_index()
    )

    df = df.merge(asia, on="ict_session_date", how="left")
    df = df.merge(london, on="ict_session_date", how="left")
    df = df.merge(nyopen, on="ict_session_date", how="left")

    print("Building VWAP compatibility...")
    if "vwap_day" in df.columns:
        df["vwap"] = df["vwap_day"]
    elif "vwap" not in df.columns:
        vol = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        pv = typical * vol
        df["vwap"] = (
            pv.groupby(df["ict_session_date"]).cumsum()
            / vol.groupby(df["ict_session_date"]).cumsum().replace(0, np.nan)
        )

    # Distances/context
    for level in ["dol", "pdh", "pdl", "prior_session_mid", "asia_high", "asia_low", "london_high", "london_low", "ny_open", "vwap"]:
        if level in df.columns:
            df[f"dist_{level}"] = df["close"] - df[level]

    df["above_dol"] = df["close"] > df["dol"]
    df["below_dol"] = df["close"] < df["dol"]
    df["above_vwap"] = df["close"] > df["vwap"]
    df["below_vwap"] = df["close"] < df["vwap"]

    df["prior_range_position"] = np.where(
        (df["pdh"] - df["pdl"]) > 0,
        (df["close"] - df["pdl"]) / (df["pdh"] - df["pdl"]),
        np.nan,
    )
    df["premium_vs_prior_range"] = df["prior_range_position"] > 0.5
    df["discount_vs_prior_range"] = df["prior_range_position"] < 0.5

    print("Building sweep/reclaim flags...")
    sweep_specs = [
        ("pdh", "pdh", "high"),
        ("pdl", "pdl", "low"),
        ("premarket_high", "pmh", "high"),
        ("premarket_low", "pml", "low"),
        ("asia_high", "asia_high", "high"),
        ("asia_low", "asia_low", "low"),
        ("london_high", "london_high", "high"),
        ("london_low", "london_low", "low"),
    ]

    sweep_cols = []
    for level_col, prefix, direction in sweep_specs:
        if level_col not in df.columns:
            continue

        if direction == "high":
            now = ((df["high"] > df[level_col]) & (df["close"] < df[level_col])).fillna(False)
        else:
            now = ((df["low"] < df[level_col]) & (df["close"] > df[level_col])).fillna(False)

        now_col = f"{prefix}_sweep_reclaim_now"
        roll_col = f"{prefix}_sweep_reclaim_last_{SWEEP_LOOKBACK}m"

        df[now_col] = now
        df[roll_col] = rolling_bool_by_session(df, now_col, SWEEP_LOOKBACK)
        sweep_cols.append(roll_col)

    if sweep_cols:
        df[f"any_liquidity_sweep_reclaim_last_{SWEEP_LOOKBACK}m"] = df[sweep_cols].any(axis=1)
        df[f"liquidity_sweep_reclaim_count_last_{SWEEP_LOOKBACK}m"] = df[sweep_cols].sum(axis=1)
    else:
        df[f"any_liquidity_sweep_reclaim_last_{SWEEP_LOOKBACK}m"] = False
        df[f"liquidity_sweep_reclaim_count_last_{SWEEP_LOOKBACK}m"] = 0

    print("Building DOL reclaim flags...")
    df["dol_cross_now"] = (
        ((df["close"].shift(1) < df["dol"]) & (df["close"] > df["dol"])) |
        ((df["close"].shift(1) > df["dol"]) & (df["close"] < df["dol"]))
    ).fillna(False)

    df["bull_dol_reclaim_now"] = ((df["low"] < df["dol"]) & (df["close"] > df["dol"])).fillna(False)
    df["bear_dol_reclaim_now"] = ((df["high"] > df["dol"]) & (df["close"] < df["dol"])).fillna(False)

    for col in ["dol_cross_now", "bull_dol_reclaim_now", "bear_dol_reclaim_now"]:
        df[col.replace("_now", f"_last_{DOL_LOOKBACK}m")] = rolling_bool_by_session(df, col, DOL_LOOKBACK)

    print("Building FVG / IFVG proxy compatibility...")
    # Existing feature factory already has bull_fvg/bear_fvg columns.
    if "bull_fvg" in df.columns and "bear_fvg" in df.columns:
        df["inside_bull_fvg_current_bar"] = (
            (df["close"] >= df["bull_fvg_low"]) & (df["close"] <= df["bull_fvg_high"])
        ).fillna(False)
        df["inside_bear_fvg_current_bar"] = (
            (df["close"] >= df["bear_fvg_low"]) & (df["close"] <= df["bear_fvg_high"])
        ).fillna(False)

        # Fast rolling approximation: recent FVG occurred in last 120 bars.
        df["recent_bull_fvg_last_120m"] = rolling_bool_by_session(df, "bull_fvg", 120)
        df["recent_bear_fvg_last_120m"] = rolling_bool_by_session(df, "bear_fvg", 120)

        # Crude IFVG proxy, fast version:
        # A recent opposite FVG exists and current bar closes through/retests direction.
        df["bull_ifvg_proxy_fast"] = (
            df["recent_bear_fvg_last_120m"] & (df["close"] > df["open"])
        ).fillna(False)
        df["bear_ifvg_proxy_fast"] = (
            df["recent_bull_fvg_last_120m"] & (df["close"] < df["open"])
        ).fillna(False)
        df["any_ifvg_proxy_fast"] = df["bull_ifvg_proxy_fast"] | df["bear_ifvg_proxy_fast"]
    else:
        df["recent_bull_fvg_last_120m"] = False
        df["recent_bear_fvg_last_120m"] = False
        df["any_ifvg_proxy_fast"] = False

    print("Building V-shape context...")
    look = 10
    df["prior_10m_high"] = g["high"].rolling(look, min_periods=2).max().reset_index(level=0, drop=True)
    df["prior_10m_low"] = g["low"].rolling(look, min_periods=2).min().reset_index(level=0, drop=True)
    df["prior_10m_open"] = g["open"].shift(look - 1)

    atr_base = df["atr_14"] if "atr_14" in df.columns else df.get("avg_range_20", np.nan)

    df["bull_v_reversal_proxy"] = (
        (df["prior_10m_low"] < df["prior_10m_open"]) &
        ((df["close"] - df["prior_10m_low"]) > (atr_base * 1.0))
    ).fillna(False)

    df["bear_v_reversal_proxy"] = (
        (df["prior_10m_high"] > df["prior_10m_open"]) &
        ((df["prior_10m_high"] - df["close"]) > (atr_base * 1.0))
    ).fillna(False)

    df["any_v_reversal_proxy"] = df["bull_v_reversal_proxy"] | df["bear_v_reversal_proxy"]

    print(f"Saving enhanced parquet: {OUTPUT}")
    df.to_parquet(OUTPUT, index=False)

    report_cols = [
        "dol", "pdh", "pdl", "asia_high", "asia_low", "london_high", "london_low", "vwap",
        f"any_liquidity_sweep_reclaim_last_{SWEEP_LOOKBACK}m",
        f"liquidity_sweep_reclaim_count_last_{SWEEP_LOOKBACK}m",
        f"bull_dol_reclaim_last_{DOL_LOOKBACK}m",
        f"bear_dol_reclaim_last_{DOL_LOOKBACK}m",
        "recent_bull_fvg_last_120m",
        "recent_bear_fvg_last_120m",
        "any_ifvg_proxy_fast",
        "any_v_reversal_proxy",
    ]

    rows = []
    for c in report_cols:
        if c in df.columns:
            if df[c].dtype == bool:
                true_count = int(df[c].sum())
            else:
                true_count = ""
            rows.append({
                "feature": c,
                "non_null_count": int(df[c].notna().sum()),
                "true_count": true_count,
                "coverage_pct": round(float(df[c].notna().mean() * 100), 2),
            })

    report = pd.DataFrame(rows)
    report.to_csv(REPORT, index=False)

    print(f"Saved report: {REPORT}")
    print("\nFeature report:")
    print(report.to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
