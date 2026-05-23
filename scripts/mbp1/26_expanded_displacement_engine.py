import polars as pl
from pathlib import Path
from itertools import product

DATA = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT = Path(r"D:\TradingResearch\research_outputs")

START = "2018-01-01"
END = "2026-05-01"

print("Loading data...")

spread_q40 = (
    pl.scan_parquet(DATA)
    .select(pl.col("avg_spread").quantile(0.40))
    .collect()
    .item()
)

df = (
    pl.scan_parquet(DATA)
    .filter(
        (pl.col("trade_date_ct") >= pl.lit(START).str.to_date()) &
        (pl.col("trade_date_ct") <= pl.lit(END).str.to_date())
    )
    .with_columns([
        pl.col("ts_ct").alias("ts"),
        pl.col("trade_date_ct").alias("date"),
        pl.col("or_low_30m").alias("or_low"),
        pl.col("or_range_30m").alias("or_range"),

        (pl.col("avg_spread") <= spread_q40).alias("tight_spread"),

        (
            (pl.col("avg_spread") <= spread_q40) &
            (pl.col("quote_pressure_delta") > 0)
        ).alias("mbp_confirmed"),
    ])
    .select([
        "ts",
        "date",
        "open",
        "high",
        "low",
        "close",
        "body_abs",
        "body_pct",
        "is_rth",
        "minute_of_day_ct",
        "prior_rth_low",
        "or_low",
        "or_range",
        "premarket_range",
        "dist_vwap_day",
        "bull_fvg",
        "bull_fvg_low",
        "bull_fvg_high",
        "bull_fvg_size",
        "avg_spread",
        "quote_pressure_delta",
        "net_mid_pressure",
        "mbp_confirmed",
    ])
    .collect()
    .sort("ts")
)

print(f"Rows loaded: {df.height:,}")

days = df.partition_by("date", as_dict=True)

DISP_THRESHOLDS = [8, 10, 12]
FVG_MIN_LIST = [2, 3, 4]
TP_LIST = [20, 25]
SL_LIST = [12, 15, 20]
BE_TRIGGER_LIST = [8, 10, 12]
ENTRY_STYLE_LIST = ["mid_fvg", "shallow_retrace"]
TIME_WINDOWS = [
    ("early", 510, 570),
    ("midmorning", 570, 660),
]
PM_FILTERS = [False, True]
MBP_FILTERS = [False, True]

results = []
all_trades = []

combos = list(product(
    DISP_THRESHOLDS,
    FVG_MIN_LIST,
    TP_LIST,
    SL_LIST,
    BE_TRIGGER_LIST,
    ENTRY_STYLE_LIST,
    TIME_WINDOWS,
    PM_FILTERS,
    MBP_FILTERS
))

print(f"Testing combos: {len(combos):,}")

for idx, combo in enumerate(combos, 1):

    (
        disp_thresh,
        fvg_min,
        tp,
        sl,
        be_trigger,
        entry_style,
        time_window,
        require_large_pm,
        require_mbp
    ) = combo

    window_name, start_min, end_min = time_window

    trades = []

    for date_key, day in days.items():

        date = date_key[0] if isinstance(date_key, tuple) else date_key

        setups = day.filter(
            (pl.col("is_rth") == True) &
            (pl.col("minute_of_day_ct") >= start_min) &
            (pl.col("minute_of_day_ct") <= end_min) &
            (pl.col("low") < pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("or_low")) &
            (pl.col("dist_vwap_day") < 0) &
            (pl.col("body_abs") >= disp_thresh) &
            (pl.col("body_pct") >= 0.60) &
            (pl.col("close") > pl.col("open")) &
            (pl.col("bull_fvg") == True) &
            (pl.col("bull_fvg_size") >= fvg_min)
        )

        if require_large_pm:
            setups = setups.filter(
                pl.col("premarket_range") >= 80
            )

        if require_mbp:
            setups = setups.filter(
                pl.col("mbp_confirmed") == True
            )

        if setups.height == 0:
            continue

        s = setups.row(0, named=True)

        if entry_style == "mid_fvg":
            entry = (
                s["bull_fvg_low"] +
                s["bull_fvg_high"]
            ) / 2

        else:
            entry = s["close"] - 5

        entry = float(entry)

        stop = float(entry - sl)
        target = float(entry + tp)

        after = day.filter(pl.col("ts") > s["ts"])

        entered = False
        be_active = False

        result = "NO_ENTRY"
        exit_price = entry

        mfe = 0.0
        mae = 0.0
        bars_held = 0

        for bar in after.iter_rows(named=True):

            bars_held += 1

            if not entered:

                if bar["low"] <= entry:
                    entered = True
                else:
                    continue

            mfe = max(mfe, float(bar["high"] - entry))
            mae = min(mae, float(bar["low"] - entry))

            if (
                not be_active and
                bar["high"] >= entry + be_trigger
            ):
                stop = entry
                be_active = True

            if bar["low"] <= stop:

                if stop == entry:
                    result = "BE"
                else:
                    result = "LOSS"

                exit_price = stop
                break

            if bar["high"] >= target:
                result = "WIN"
                exit_price = target
                break

        if not entered:
            continue

        pnl = float(exit_price - entry)

        trades.append({
            "date": str(date),
            "pnl": pnl,
            "result": result,
            "mfe": float(mfe),
            "mae": float(mae),
            "bars_held": int(bars_held),
        })

    if len(trades) < 15:
        continue

    t = pl.DataFrame(trades)

    wins = t.filter(pl.col("pnl") > 0).height
    losses = t.filter(pl.col("pnl") < 0).height
    bes = t.filter(pl.col("pnl") == 0).height

    gross_win = t.filter(pl.col("pnl") > 0)["pnl"].sum()
    gross_loss = abs(
        t.filter(pl.col("pnl") < 0)["pnl"].sum()
    )

    pf = (
        gross_win / gross_loss
        if gross_loss else 999
    )

    wr = wins / len(trades)
    net = t["pnl"].sum()

    eq = t.with_columns(
        pl.col("pnl").cum_sum().alias("equity")
    )

    dd = (
        eq["equity"] -
        eq["equity"].cum_max()
    ).min()

    if wr >= 0.65 and pf >= 1.5:
        tier = "A+"
    elif wr >= 0.58 and pf >= 1.25:
        tier = "A"
    elif wr >= 0.52:
        tier = "B"
    else:
        tier = "REJECT"

    results.append({
        "tier": tier,
        "disp_thresh": disp_thresh,
        "fvg_min": fvg_min,
        "tp": tp,
        "sl": sl,
        "be_trigger": be_trigger,
        "entry_style": entry_style,
        "time_window": window_name,
        "large_pm": require_large_pm,
        "mbp": require_mbp,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "breakevens": bes,
        "winrate": round(wr, 4),
        "profit_factor": round(float(pf), 3),
        "net_points": round(float(net), 2),
        "avg_trade": round(float(t["pnl"].mean()), 2),
        "max_dd": round(float(dd), 2),
        "avg_mfe": round(float(t["mfe"].mean()), 2),
        "avg_mae": round(float(t["mae"].mean()), 2),
        "avg_bars_held": round(
            float(t["bars_held"].mean()),
            2
        ),
    })

    if idx % 100 == 0:
        print(f"Completed {idx:,}/{len(combos):,}")

summary = (
    pl.DataFrame(results)
    .sort(
        ["tier", "winrate", "profit_factor", "net_points"],
        descending=[False, True, True, True]
    )
)

summary_path = (
    OUT / "26_expanded_displacement_summary.csv"
)

summary.write_csv(summary_path)

print("\nTOP MODELS")
print(
    summary
    .filter(pl.col("tier") != "REJECT")
    .head(100)
)

print("\nTIER COUNTS")
print(
    summary
    .group_by("tier")
    .len()
    .sort("tier")
)

print("\nSaved:")
print(summary_path)