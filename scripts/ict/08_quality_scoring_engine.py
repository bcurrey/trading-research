from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(r"D:\TradingResearch")

INPUT = ROOT / r"ict_trade_study\outputs\ict_trade_enhanced_features.csv"

OUTDIR = ROOT / r"ict_trade_study\outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_SCORED = OUTDIR / "ict_trade_quality_scores.csv"
OUT_SUMMARY = OUTDIR / "ict_trade_quality_summary.csv"
OUT_FEATURES = OUTDIR / "ict_quality_feature_breakdown.csv"


def b(x):
    return (
        pd.Series(x)
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def n(x):
    return pd.to_numeric(x, errors="coerce")


def main():

    print(f"Loading: {INPUT}")

    df = pd.read_csv(INPUT)

    df = df[
        df["match_status"]
        .astype(str)
        .str.lower()
        .eq("matched")
    ].copy()

    print(f"Matched trades: {len(df)}")

    # --------------------------------
    # Normalize
    # --------------------------------

    num_cols = [
        "v_shape_score_atr",
        "pre_10m_drive_points",
        "post_10m_reclaim_points",
        "displacement_ratio",
        "nearest_fvg_size",
        "nearest_fvg_dist",
        "dol_distance",
        "vwap_distance",
        "mfe_10m",
        "mfe_30m",
        "mae_10m",
        "mae_30m",
    ]

    for c in num_cols:
        if c in df.columns:
            df[c] = n(df[c])

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
            df[c] = b(df[c])

    # --------------------------------
    # Quality scoring engine
    # --------------------------------

    scores = []
    reasons = []

    for _, r in df.iterrows():

        score = 0
        tags = []

        side = str(r.get("side", "")).lower()

        # --------------------------------
        # Strong reclaim velocity
        # --------------------------------

        reclaim = r.get("post_10m_reclaim_points", np.nan)

        if pd.notna(reclaim):

            if reclaim >= 20:
                score += 3
                tags.append("strong_reclaim")

            elif reclaim >= 10:
                score += 2
                tags.append("moderate_reclaim")

            elif reclaim <= 0:
                score -= 2
                tags.append("failed_reclaim")

        # --------------------------------
        # V-shape strength
        # --------------------------------

        vscore = r.get("v_shape_score_atr", np.nan)

        if pd.notna(vscore):

            if vscore >= 4:
                score += 3
                tags.append("elite_vshape")

            elif vscore >= 3:
                score += 2
                tags.append("strong_vshape")

            elif vscore < 2:
                score -= 2
                tags.append("weak_vshape")

        # --------------------------------
        # Exhaustion penalty
        # --------------------------------

        if bool(r.get("exhaustion_proxy", False)):
            score -= 3
            tags.append("exhaustion")

        # --------------------------------
        # Displacement quality
        # --------------------------------

        disp = r.get("displacement_ratio", np.nan)

        if pd.notna(disp):

            if disp >= 2:
                score += 3
                tags.append("elite_displacement")

            elif disp >= 1.2:
                score += 1
                tags.append("good_displacement")

            elif disp < 0.8:
                score -= 2
                tags.append("weak_displacement")

        # --------------------------------
        # Directional pre-drive
        # --------------------------------

        pre_drive = r.get("pre_10m_drive_points", np.nan)

        if pd.notna(pre_drive):

            if side == "long":

                if pre_drive <= -10:
                    score += 2
                    tags.append("selloff_before_long")

            elif side == "short":

                if pre_drive >= 10:
                    score += 2
                    tags.append("rally_before_short")

        # --------------------------------
        # FVG quality
        # --------------------------------

        fvg_size = r.get("nearest_fvg_size", np.nan)

        if pd.notna(fvg_size):

            if 3 <= fvg_size <= 15:
                score += 2
                tags.append("healthy_fvg")

            elif fvg_size > 30:
                score -= 2
                tags.append("oversized_fvg")

        # --------------------------------
        # Liquidity interaction
        # --------------------------------

        if bool(r.get("any_key_level_sweep_last_30m", False)):
            score += 1
            tags.append("liquidity_event")

        # --------------------------------
        # DOL / VWAP context
        # --------------------------------

        dol = r.get("dol_distance", np.nan)
        vwap = r.get("vwap_distance", np.nan)

        if pd.notna(dol):

            if side == "long" and dol > 0:
                score += 1
                tags.append("above_dol")

            elif side == "short" and dol < 0:
                score += 1
                tags.append("below_dol")

        if pd.notna(vwap):

            if side == "long" and vwap > 0:
                score += 1
                tags.append("above_vwap")

            elif side == "short" and vwap < 0:
                score += 1
                tags.append("below_vwap")

        scores.append(score)
        reasons.append("|".join(tags))

    df["quality_score_v2"] = scores
    df["quality_tags"] = reasons

    # --------------------------------
    # Tiering
    # --------------------------------

    df["quality_tier"] = np.select(
        [
            df["quality_score_v2"] >= 10,
            df["quality_score_v2"] >= 7,
            df["quality_score_v2"] >= 4,
            df["quality_score_v2"] >= 1,
        ],
        [
            "A+",
            "A",
            "B",
            "C",
        ],
        default="F"
    )

    # --------------------------------
    # Results
    # --------------------------------

    df["won"] = (
        df["result"]
        .astype(str)
        .str.lower()
        .eq("win")
    )

    summary = (
        df.groupby("quality_tier")
        .agg(
            trades=("won", "count"),
            wins=("won", "sum"),
            avg_score=("quality_score_v2", "mean"),
        )
        .reset_index()
    )

    summary["win_rate"] = round(
        (summary["wins"] / summary["trades"]) * 100,
        2
    )

    # --------------------------------
    # Tag effectiveness
    # --------------------------------

    tag_rows = []

    all_tags = []

    for x in df["quality_tags"]:
        if isinstance(x, str):
            all_tags.extend(x.split("|"))

    all_tags = sorted(set([x for x in all_tags if x]))

    for tag in all_tags:

        subset = df[
            df["quality_tags"]
            .astype(str)
            .str.contains(tag, regex=False)
        ]

        if len(subset) < 2:
            continue

        wr = subset["won"].mean()

        tag_rows.append({
            "tag": tag,
            "trades": len(subset),
            "win_rate": round(wr * 100, 2),
            "avg_score": round(subset["quality_score_v2"].mean(), 2),
        })

    tag_df = pd.DataFrame(tag_rows)

    if len(tag_df):
        tag_df = tag_df.sort_values(
            ["win_rate", "trades"],
            ascending=[False, False]
        )

    # --------------------------------
    # Save
    # --------------------------------

    keep_cols = [
        "trade_id",
        "date",
        "entry_time",
        "side",
        "result",
        "quality_score_v2",
        "quality_tier",
        "quality_tags",
        "content",
        "discord_url",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]

    df[keep_cols].to_csv(OUT_SCORED, index=False)

    summary.to_csv(OUT_SUMMARY, index=False)

    tag_df.to_csv(OUT_FEATURES, index=False)

    # --------------------------------
    # Console
    # --------------------------------

    print(f"\nSaved scored trades: {OUT_SCORED}")
    print(f"Saved tier summary: {OUT_SUMMARY}")
    print(f"Saved feature breakdown: {OUT_FEATURES}")

    print("\nQUALITY SUMMARY")
    print(summary.to_string(index=False))

    print("\nTOP FEATURE TAGS")
    print(tag_df.head(25).to_string(index=False))

    print("\nTOP SCORING TRADES")
    print(
        df.sort_values(
            "quality_score_v2",
            ascending=False
        )[
            [
                "trade_id",
                "date",
                "side",
                "result",
                "quality_score_v2",
                "quality_tier",
                "quality_tags",
            ]
        ].head(20).to_string(index=False)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
