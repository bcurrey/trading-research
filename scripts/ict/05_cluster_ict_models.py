from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")
INPUT = ROOT / r"ict_trade_study\outputs\ict_trade_enhanced_features.csv"
OUTDIR = ROOT / r"ict_trade_study\outputs"

OUT_CLUSTERED = OUTDIR / "ict_trade_model_clusters.csv"
OUT_SUMMARY = OUTDIR / "ict_trade_model_cluster_summary.csv"
OUT_WINLOSS = OUTDIR / "ict_trade_model_win_loss_comparison.csv"


def as_bool(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def clean_num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def contains_level(s, terms):
    txt = s.fillna("").astype(str).str.lower()
    out = pd.Series(False, index=s.index)
    for t in terms:
        out = out | txt.str.contains(t, regex=False)
    return out


def classify_row(r):
    side = str(r.get("side", "")).lower()
    result = str(r.get("result", "")).lower()

    below_dol = r.get("dol_relation") == "below_dol"
    above_dol = r.get("dol_relation") == "above_dol"
    below_vwap = r.get("vwap_relation") == "below_vwap"
    above_vwap = r.get("vwap_relation") == "above_vwap"

    vshape = bool(r.get("v_shape_candidate", False))
    exhaustion = bool(r.get("exhaustion_proxy", False))
    displacement = bool(r.get("displacement_candle", False))
    inside_fvg = bool(r.get("entry_inside_nearest_fvg", False))
    ifvg = bool(r.get("ifvg_proxy", False))
    fvg_align = bool(r.get("fvg_alignment_with_side", False))

    swept = str(r.get("swept_levels_last_30m", "")).lower()
    swept_high = any(x in swept for x in ["high", "pdh", "premarket_high", "asia_high", "london_high"])
    swept_low = any(x in swept for x in ["low", "pdl", "premarket_low", "asia_low", "london_low"])

    mfe30 = r.get("mfe_30m", np.nan)
    mae30 = r.get("mae_30m", np.nan)

    # Primary clusters. Order matters.
    if side == "short" and below_dol and below_vwap and swept_high:
        return "A_continuation_short_below_dol_vwap_after_high_sweep"

    if side == "long" and vshape and swept_low:
        return "B_reversal_long_vshape_after_low_sweep"

    if side == "long" and above_dol and above_vwap and vshape:
        return "C_momentum_long_above_dol_vwap"

    if side == "short" and below_dol and vshape:
        return "D_momentum_short_below_dol"

    if inside_fvg and not fvg_align:
        return "E_counter_fvg_entry_risk"

    if inside_fvg and fvg_align:
        return "F_aligned_fvg_entry"

    if ifvg:
        return "G_ifvg_proxy_trade"

    if exhaustion:
        return "H_exhaustion_risk"

    if displacement:
        return "I_displacement_continuation"

    return "Z_unclassified"


def score_quality(r):
    score = 0

    side = str(r.get("side", "")).lower()

    if bool(r.get("v_shape_candidate", False)):
        score += 1
    if bool(r.get("fvg_alignment_with_side", False)):
        score += 1
    if bool(r.get("entry_inside_nearest_fvg", False)):
        score += 1
    if bool(r.get("ifvg_proxy", False)):
        score += 1

    if side == "short" and r.get("dol_relation") == "below_dol":
        score += 1
    if side == "long" and r.get("dol_relation") == "above_dol":
        score += 1

    if side == "short" and r.get("vwap_relation") == "below_vwap":
        score += 1
    if side == "long" and r.get("vwap_relation") == "above_vwap":
        score += 1

    swept = str(r.get("swept_levels_last_30m", "")).lower()
    if side == "short" and any(x in swept for x in ["high", "pdh", "premarket_high", "asia_high", "london_high"]):
        score += 1
    if side == "long" and any(x in swept for x in ["low", "pdl", "premarket_low", "asia_low", "london_low"]):
        score += 1

    if bool(r.get("exhaustion_proxy", False)):
        score -= 1

    # Penalize poor 30m post-entry behavior if present.
    mfe = r.get("mfe_30m", np.nan)
    mae = r.get("mae_30m", np.nan)
    if pd.notna(mfe) and pd.notna(mae):
        if mfe > abs(mae):
            score += 1
        elif abs(mae) > mfe * 1.5:
            score -= 1

    return score


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    print(f"Loading enhanced features: {INPUT}")
    df = pd.read_csv(INPUT)
    print(f"Rows loaded: {len(df)}")

    # Keep matched trades only.
    df = df[df["match_status"].astype(str).eq("matched")].copy()
    print(f"Matched rows used: {len(df)}")

    # Normalize booleans.
    bool_cols = [
        "v_shape_candidate",
        "exhaustion_proxy",
        "displacement_candle",
        "entry_inside_nearest_fvg",
        "ifvg_proxy",
        "fvg_alignment_with_side",
        "any_key_level_sweep_last_30m",
        "dol_reclaim_last_10m",
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = as_bool(df[c])

    # Numeric cleanup.
    num_cols = [
        "entry_price", "dol_distance", "vwap_distance", "nearest_fvg_dist",
        "nearest_fvg_size", "nearest_fvg_age_bars", "v_shape_score_atr",
        "displacement_ratio", "mfe_10m", "mae_10m", "mfe_30m", "mae_30m",
        "mfe_60m", "mae_60m", "pre_10m_drive_points", "post_10m_reclaim_points",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["model_cluster"] = df.apply(classify_row, axis=1)
    df["quality_score"] = df.apply(score_quality, axis=1)

    # Quality tiers.
    df["quality_tier"] = np.select(
        [
            df["quality_score"] >= 6,
            df["quality_score"].between(4, 5),
            df["quality_score"].between(2, 3),
        ],
        ["A+", "A", "B"],
        default="C"
    )

    # Useful flags.
    df["won"] = df["result"].astype(str).str.lower().eq("win")
    df["lost"] = df["result"].astype(str).str.lower().eq("loss")
    df["rr_proxy_30m"] = np.where(df["mae_30m"].abs() > 0, df["mfe_30m"] / df["mae_30m"].abs(), np.nan)

    # Save clustered detail.
    keep_cols = [
        "trade_id", "date", "entry_time", "matched_dt", "side", "result",
        "entry_price", "model_cluster", "quality_score", "quality_tier",
        "dol_relation", "dol_distance", "vwap_relation", "vwap_distance",
        "swept_levels_last_30m", "any_key_level_sweep_last_30m",
        "nearest_fvg_dir", "nearest_fvg_dist", "nearest_fvg_size", "nearest_fvg_age_bars",
        "entry_inside_nearest_fvg", "ifvg_proxy", "fvg_alignment_with_side",
        "v_shape_candidate", "v_shape_score_atr", "exhaustion_proxy",
        "displacement_candle", "displacement_ratio",
        "mfe_10m", "mae_10m", "mfe_30m", "mae_30m", "mfe_60m", "mae_60m",
        "rr_proxy_30m", "content", "discord_url",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUT_CLUSTERED, index=False)

    # Cluster summary.
    summary_rows = []
    for cluster, g in df.groupby("model_cluster", dropna=False):
        row = {
            "model_cluster": cluster,
            "trades": len(g),
            "wins": int(g["won"].sum()),
            "losses": int(g["lost"].sum()),
            "win_rate": round(float(g["won"].mean() * 100), 2),
            "avg_quality_score": round(float(g["quality_score"].mean()), 2),
            "avg_mfe_30m": round(float(g["mfe_30m"].mean()), 2),
            "avg_mae_30m": round(float(g["mae_30m"].mean()), 2),
            "avg_rr_proxy_30m": round(float(g["rr_proxy_30m"].mean()), 2),
            "avg_fvg_dist": round(float(g["nearest_fvg_dist"].mean()), 2),
            "vshape_rate": round(float(g["v_shape_candidate"].mean() * 100), 2) if "v_shape_candidate" in g else np.nan,
            "ifvg_rate": round(float(g["ifvg_proxy"].mean() * 100), 2) if "ifvg_proxy" in g else np.nan,
            "sweep_rate": round(float(g["any_key_level_sweep_last_30m"].mean() * 100), 2) if "any_key_level_sweep_last_30m" in g else np.nan,
            "inside_fvg_rate": round(float(g["entry_inside_nearest_fvg"].mean() * 100), 2) if "entry_inside_nearest_fvg" in g else np.nan,
            "fvg_align_rate": round(float(g["fvg_alignment_with_side"].mean() * 100), 2) if "fvg_alignment_with_side" in g else np.nan,
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["trades", "win_rate"], ascending=[False, False])
    summary.to_csv(OUT_SUMMARY, index=False)

    # Win/loss comparison by feature.
    comp_rows = []
    features = [
        "quality_score", "dol_distance", "vwap_distance", "nearest_fvg_dist",
        "nearest_fvg_size", "nearest_fvg_age_bars", "v_shape_score_atr",
        "displacement_ratio", "mfe_30m", "mae_30m", "rr_proxy_30m",
    ]
    for f in features:
        if f not in df.columns:
            continue
        wins = df[df["won"]][f].dropna()
        losses = df[df["lost"]][f].dropna()
        comp_rows.append({
            "feature": f,
            "win_avg": wins.mean() if len(wins) else np.nan,
            "loss_avg": losses.mean() if len(losses) else np.nan,
            "win_median": wins.median() if len(wins) else np.nan,
            "loss_median": losses.median() if len(losses) else np.nan,
            "difference_avg_win_minus_loss": (wins.mean() - losses.mean()) if len(wins) and len(losses) else np.nan,
        })

    bool_features = [
        "v_shape_candidate", "exhaustion_proxy", "displacement_candle",
        "entry_inside_nearest_fvg", "ifvg_proxy", "fvg_alignment_with_side",
        "any_key_level_sweep_last_30m", "dol_reclaim_last_10m",
    ]
    for f in bool_features:
        if f not in df.columns:
            continue
        wins = df[df["won"]][f].fillna(False)
        losses = df[df["lost"]][f].fillna(False)
        comp_rows.append({
            "feature": f,
            "win_avg": wins.mean() if len(wins) else np.nan,
            "loss_avg": losses.mean() if len(losses) else np.nan,
            "win_median": wins.median() if len(wins) else np.nan,
            "loss_median": losses.median() if len(losses) else np.nan,
            "difference_avg_win_minus_loss": (wins.mean() - losses.mean()) if len(wins) and len(losses) else np.nan,
        })

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT_WINLOSS, index=False)

    print(f"\nSaved clustered trades: {OUT_CLUSTERED}")
    print(f"Saved cluster summary:  {OUT_SUMMARY}")
    print(f"Saved win/loss compare: {OUT_WINLOSS}")

    print("\nCluster summary:")
    print(summary.to_string(index=False))

    print("\nTop clustered trades:")
    show_cols = [
        "trade_id", "date", "entry_time", "side", "result",
        "model_cluster", "quality_score", "quality_tier",
        "dol_relation", "vwap_relation", "swept_levels_last_30m",
        "nearest_fvg_dir", "nearest_fvg_dist", "v_shape_candidate",
        "mfe_30m", "mae_30m", "rr_proxy_30m",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
