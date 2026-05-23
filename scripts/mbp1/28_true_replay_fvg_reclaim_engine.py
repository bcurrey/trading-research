import polars as pl
from pathlib import Path
from itertools import product

DATA = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT = Path(r"D:\TradingResearch\research_outputs")

START = "2018-01-01"
END = "2026-05-01"

print("Loading data...")

df = (
    pl.scan_parquet(DATA)
    .filter(
        (pl.col("trade_date_ct") >= pl.lit(START).str.to_date()) &
        (pl.col("trade_date_ct") <= pl.lit(END).str.to_date()) &
        (pl.col("is_rth") == True)
    )
    .with_columns([
        pl.col("ts_ct").alias("ts"),
        pl.col("trade_date_ct").alias("date"),
        pl.col("or_low_30m").alias("or_low"),
    ])
    .select([
        "ts", "date", "open", "high", "low", "close",
        "body_abs", "body_pct", "minute_of_day_ct",
        "prior_rth_low", "or_low", "premarket_range",
        "dist_vwap_day", "bull_fvg", "bull_fvg_low",
        "bull_fvg_high", "bull_fvg_size",
        "quote_pressure_delta", "net_mid_pressure"
    ])
    .collect()
    .sort("ts")
)

print(f"Rows loaded: {df.height:,}")

days = df.partition_by("date", as_dict=True)

MODELS = list(product(
    [8, 10, 12],          # displacement body min
    [2, 3, 4],            # FVG min
    [20, 25],             # target
    [12, 15, 20],         # stop
    [8, 10, 12],          # BE trigger
    [False, True],        # large premarket
    [False, True],        # MBP confirm
))

rows = []
trade_rows = []

for i, (disp, fvg_min, tp, sl, be_trigger, large_pm, mbp) in enumerate(MODELS, 1):

    trades = []

    for date_key, day in days.items():
        date = date_key[0] if isinstance(date_key, tuple) else date_key

        setups = day.filter(
            (pl.col("minute_of_day_ct") >= 510) &
            (pl.col("minute_of_day_ct") <= 660) &
            (pl.col("low") < pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("or_low")) &
            (pl.col("dist_vwap_day") < 0) &
            (pl.col("close") > pl.col("open")) &
            (pl.col("body_abs") >= disp) &
            (pl.col("body_pct") >= 0.60) &
            (pl.col("bull_fvg") == True) &
            (pl.col("bull_fvg_size") >= fvg_min)
        )

        if large_pm:
            setups = setups.filter(pl.col("premarket_range") >= 80)

        if mbp:
            setups = setups.filter(
                (pl.col("quote_pressure_delta") > 0) &
                (pl.col("net_mid_pressure") > 0)
            )

        if setups.height == 0:
            continue

        s = setups.row(0, named=True)

        entry = float((s["bull_fvg_low"] + s["bull_fvg_high"]) / 2)
        stop = entry - sl
        target = entry + tp

        after = day.filter(pl.col("ts") > s["ts"]).head(120)

        entered = False
        active_stop = stop
        be_moved = False
        exit_price = None
        result = "NO_EXIT"
        mfe = 0.0
        mae = 0.0
        bars = 0

        for bar in after.iter_rows(named=True):
            bars += 1

            if not entered:
                if bar["low"] <= entry:
                    entered = True
                else:
                    continue

            mfe = max(mfe, float(bar["high"] - entry))
            mae = min(mae, float(bar["low"] - entry))

            if not be_moved and bar["high"] >= entry + be_trigger:
                active_stop = entry
                be_moved = True

            if bar["low"] <= active_stop:
                exit_price = active_stop
                result = "BE" if active_stop == entry else "LOSS"
                break

            if bar["high"] >= target:
                exit_price = target
                result = "WIN"
                break

        if not entered:
            continue

        if exit_price is None:
            last = after.tail(1).row(0, named=True)
            exit_price = float(last["close"])
            result = "TIME_EXIT"

        pnl = float(exit_price - entry)

        trades.append(pnl)

        trade_rows.append({
            "disp": disp,
            "fvg_min": fvg_min,
            "tp": tp,
            "sl": sl,
            "be_trigger": be_trigger,
            "large_pm": large_pm,
            "mbp": mbp,
            "date": str(date),
            "setup_ts": str(s["ts"]),
            "entry": entry,
            "exit": exit_price,
            "pnl": pnl,
            "result": result,
            "mfe": mfe,
            "mae": mae,
            "bars": bars,
        })

    if len(trades) < 15:
        continue

    t = pl.Series(trades)

    wins = (t > 0).sum()
    losses = (t < 0).sum()
    bes = (t == 0).sum()

    gross_win = t.filter(t > 0).sum()
    gross_loss = abs(t.filter(t < 0).sum())
    pf = gross_win / gross_loss if gross_loss else 999

    wr = wins / len(trades)
    net = t.sum()

    tier = "REJECT"
    if wr >= 0.65 and pf >= 1.5:
        tier = "A+"
    elif wr >= 0.58 and pf >= 1.25:
        tier = "A"
    elif wr >= 0.52 and pf >= 1.05:
        tier = "B"

    rows.append({
        "tier": tier,
        "disp": disp,
        "fvg_min": fvg_min,
        "tp": tp,
        "sl": sl,
        "be_trigger": be_trigger,
        "large_pm": large_pm,
        "mbp": mbp,
        "trades": len(trades),
        "wins": int(wins),
        "losses": int(losses),
        "breakevens": int(bes),
        "winrate": round(float(wr), 4),
        "profit_factor": round(float(pf), 3),
        "net_points": round(float(net), 2),
        "avg_trade": round(float(t.mean()), 2),
    })

    print(f"Completed {i}/{len(MODELS)}")

summary = pl.DataFrame(rows).sort(
    ["tier", "winrate", "profit_factor", "net_points"],
    descending=[False, True, True, True]
)

trades = pl.DataFrame(trade_rows)

summary_path = OUT / "28_true_replay_fvg_reclaim_summary.csv"
trades_path = OUT / "28_true_replay_fvg_reclaim_trades.csv"

summary.write_csv(summary_path)
trades.write_csv(trades_path)

print("\nTOP MODELS")
print(summary.filter(pl.col("tier") != "REJECT").head(50))

print("\nTIER COUNTS")
print(summary.group_by("tier").len().sort("tier"))

print("\nSaved:")
print(summary_path)
print(trades_path)
