from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"ict_trade_study\outputs\ict_trade_enhanced_features.csv"

OUTDIR = ROOT / r"ict_trade_study\outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

FEATURE_RANK_OUT = OUTDIR / "ict_feature_rankings.csv"
WINNER_COMMON_OUT = OUTDIR / "ict_winner_commonality.csv"
THRESHOLD_OUT = OUTDIR / "ict_threshold_candidates.csv"


def normalize_bool(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def safe_numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def feature_separation_score(win_vals, loss_vals):

    if len(win_vals) < 2 or len(loss_vals) < 2:
        return np.nan

    w_mean = np.nanmean(win_vals)
    l_mean = np.nanmean(loss_vals)

    pooled = np.nanstd(
        np.concatenate([win_vals, loss_vals])
    )

    if pooled == 0 or np.isnan(pooled):
        return np.nan

    return abs(w_mean - l_mean) / pooled


def main():

    print(f"Loading: {INPUT}")

    df = pd.read_csv(INPUT)

    print(f"Rows loaded: {len(df)}")

    df = df[
        df["match_status"]
        .astype(str)
        .str.lower()
        .eq("matched")
    ].copy()

    print(f"Matched trades: {len(df)}")

    wins = df[
        df["result"]
        .astype(str)
        .str.lower()
        .eq("win")
    ].copy()

    losses = df[
        df["result"]
        .astype(str)
        .str.lower()
        .eq("loss")
    ].copy()

    print(f"Wins: {len(wins)}")
    print(f"Losses: {len(losses)}")

    # -----------------------------
    # Candidate feature columns
    # -----------------------------

    numeric_features = [
        "dol_distance",
        "vwap_distance",
        "nearest_fvg_dist",
        "nearest_fvg_size",
        "nearest_fvg_age_bars",
        "v_shape_score_atr",
        "displacement_ratio",
        "mfe_10m",
        "mae_10m",
        "mfe_30m",
        "mae_30m",
        "mfe_60m",
        "mae_60m",
        "pre_10m_drive_points",
        "post_10m_reclaim_points",
    ]

    bool_features = [
        "v_shape_candidate",
        "exhaustion_proxy",
        "displacement_candle",
        "entry_inside_nearest_fvg",
        "ifvg_proxy",
        "fvg_alignment_with_side",
        "any_key_level_sweep_last_30m",
        "dol_reclaim_last_10m",
    ]

    rankings = []

    # -----------------------------
    # NUMERIC FEATURE ANALYSIS
    # -----------------------------

    print("\nAnalyzing numeric features...")

    for feat in numeric_features:

        if feat not in df.columns:
            continue

        df[feat] = safe_numeric(df, feat)

        w = wins[feat].dropna()
        l = losses[feat].dropna()

        if len(w) < 2 or len(l) < 2:
            continue

        sep = feature_separation_score(
            w.values,
            l.values
        )

        rankings.append({
            "feature": feat,
            "type": "numeric",
            "winner_mean": round(w.mean(), 4),
            "loser_mean": round(l.mean(), 4),
            "winner_median": round(w.median(), 4),
            "loser_median": round(l.median(), 4),
            "winner_q25": round(w.quantile(0.25), 4),
            "winner_q75": round(w.quantile(0.75), 4),
            "loser_q25": round(l.quantile(0.25), 4),
            "loser_q75": round(l.quantile(0.75), 4),
            "difference": round(w.mean() - l.mean(), 4),
            "separation_score": round(sep, 4)
            if not np.isnan(sep) else np.nan,
        })

    # -----------------------------
    # BOOLEAN FEATURE ANALYSIS
    # -----------------------------

    print("Analyzing boolean features...")

    for feat in bool_features:

        if feat not in df.columns:
            continue

        df[feat] = normalize_bool(df[feat])

        w_rate = wins[feat].mean()
        l_rate = losses[feat].mean()

        rankings.append({
            "feature": feat,
            "type": "boolean",
            "winner_mean": round(w_rate * 100, 2),
            "loser_mean": round(l_rate * 100, 2),
            "winner_median": np.nan,
            "loser_median": np.nan,
            "winner_q25": np.nan,
            "winner_q75": np.nan,
            "loser_q25": np.nan,
            "loser_q75": np.nan,
            "difference": round((w_rate - l_rate) * 100, 2),
            "separation_score": round(abs(w_rate - l_rate), 4),
        })

    rankings_df = pd.DataFrame(rankings)

    rankings_df = rankings_df.sort_values(
        "separation_score",
        ascending=False
    )

    rankings_df.to_csv(FEATURE_RANK_OUT, index=False)

    # -----------------------------
    # WINNER COMMONALITY EXTRACTION
    # -----------------------------

    print("Extracting winner commonality ranges...")

    common_rows = []

    for feat in numeric_features:

        if feat not in wins.columns:
            continue

        vals = pd.to_numeric(
            wins[feat],
            errors="coerce"
        ).dropna()

        if len(vals) < 3:
            continue

        common_rows.append({
            "feature": feat,
            "winner_q10": round(vals.quantile(0.10), 4),
            "winner_q25": round(vals.quantile(0.25), 4),
            "winner_median": round(vals.quantile(0.50), 4),
            "winner_q75": round(vals.quantile(0.75), 4),
            "winner_q90": round(vals.quantile(0.90), 4),
            "sample_size": len(vals),
        })

    common_df = pd.DataFrame(common_rows)

    common_df.to_csv(WINNER_COMMON_OUT, index=False)

    # -----------------------------
    # THRESHOLD DISCOVERY
    # -----------------------------

    print("Building threshold candidates...")

    threshold_rows = []

    for feat in numeric_features:

        if feat not in df.columns:
            continue

        vals = pd.to_numeric(df[feat], errors="coerce")

        if vals.notna().sum() < 8:
            continue

        candidate_thresholds = [
            vals.quantile(0.20),
            vals.quantile(0.40),
            vals.quantile(0.60),
            vals.quantile(0.80),
        ]

        for thresh in candidate_thresholds:

            if np.isnan(thresh):
                continue

            above = df[vals >= thresh]
            below = df[vals < thresh]

            if len(above) < 3 or len(below) < 3:
                continue

            above_wr = (
                above["result"]
                .astype(str)
                .str.lower()
                .eq("win")
                .mean()
            )

            below_wr = (
                below["result"]
                .astype(str)
                .str.lower()
                .eq("win")
                .mean()
            )

            threshold_rows.append({
                "feature": feat,
                "threshold": round(thresh, 4),
                "above_threshold_wr": round(above_wr * 100, 2),
                "below_threshold_wr": round(below_wr * 100, 2),
                "edge_difference": round(
                    (above_wr - below_wr) * 100,
                    2
                ),
                "above_count": len(above),
                "below_count": len(below),
            })

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_df = threshold_df.sort_values(
        "edge_difference",
        ascending=False
    )

    threshold_df.to_csv(THRESHOLD_OUT, index=False)

    # -----------------------------
    # CONSOLE OUTPUT
    # -----------------------------

    print(f"\nSaved feature rankings: {FEATURE_RANK_OUT}")
    print(f"Saved winner commonality: {WINNER_COMMON_OUT}")
    print(f"Saved threshold candidates: {THRESHOLD_OUT}")

    print("\nTOP FEATURE SEPARATION")
    print(
        rankings_df.head(20).to_string(index=False)
    )

    print("\nTOP THRESHOLD CANDIDATES")
    print(
        threshold_df.head(20).to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
