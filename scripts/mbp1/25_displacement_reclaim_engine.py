import polars as pl
from pathlib import Path

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

        (
            (pl.col("body_abs") >= 12) &
            (pl.col("close") > pl.col("open")) &
            (pl.col("body_pct") >= 0.60)
        ).alias("bull_displacement_strong"),

        (
            (pl.col("bull_fvg") == True) &
            (pl.col("bull_fvg_size") >= 4)
        ).alias("valid_bull_fvg"),
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
        "prior_rth_low",
        "or_low",
        "or_range",
        "premarket_range",
        "dist_vwap_day",
        "tight_spread",
        "mbp_confirmed",
        "bull_displacement_strong",
        "valid_bull_fvg",
        "bull_fvg_low",
        "bull_fvg_high",
        "quote_pressure_delta",
        "net_mid_pressure",
    ])
    .collect()
    .sort("ts")
)

print(f"Rows loaded: {df.height:,}")

days = df.partition_by("date", as_dict=True)

MODELS = [
    {
        "name": "disp_reclaim_fvg",
        "require_fvg": True,
        "require_mbp": False,
        "require_large_pm": False,
        "tp": 20,
        "sl": 15,
    },

    {
        "name": "disp_reclaim_fvg_mbp",
        "require_fvg": True,
        "require_mbp": True,
        "require_large_pm": False,
        "tp": 20,
        "sl": 15,
    },

    {
        "name": "disp_reclaim_largepm",
        "require_fvg": True,
        "require_mbp": False,
        "require_large_pm": True,
        "tp": 20,
        "sl": 15,
    },

    {
        "name": "disp_reclaim_largepm_mbp",
        "require_fvg": True,
        "require_mbp": True,
        "require_large_pm": True,
        "tp": 20,
        "sl": 15,
    },
]

all_trades = []
summary_rows = []

for model in MODELS:

    print(f"\nRunning: {model['name']}")

    trades = []

    for date_key, day in days.items():

        date = date_key[0] if isinstance(date_key, tuple) else date_key

        day = day.filter(pl.col("is_rth") == True)

        setups = day.filter(
            (pl.col("low") < pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("or_low")) &
            (pl.col("dist_vwap_day") < 0) &
            (pl.col("bull_displacement_strong") == True)
        )

        if model["require_fvg"]:
            setups = setups.filter(pl.col("valid_bull_fvg") == True)

        if model["require_mbp"]:
            setups = setups.filter(pl.col("mbp_confirmed") == True)

        if model["require_large_pm"]:
            setups = setups.filter(pl.col("premarket_range") >= 80)

        if setups.height == 0:
            continue

        s = setups.row(0, named=True)

        if s["valid_bull_fvg"]:
            entry = float(
                (s["bull_fvg_low"] + s["bull_fvg_high"]) / 2
            )
        else:
            entry = float(s["close"] - 5)

        stop = float(entry - model["sl"])
        target = float(entry + model["tp"])

        after = day.filter(pl.col("ts") > s["ts"])

        if after.height == 0:
            continue

        result = "EOD"
        exit_price = entry

        mfe = 0.0
        mae = 0.0
        bars_held = 0

        entered = False

        for idx, bar in enumerate(after.iter_rows(named=True), 1):

            bars_held += 1

            if not entered:

                if bar["low"] <= entry:
                    entered = True
                else:
                    continue

            mfe = max(mfe, float(bar["high"] - entry))
            mae = min(mae, float(bar["low"] - entry))

            if bar["low"] <= stop:
                exit_price = stop
                result = "LOSS"
                break

            if bar["high"] >= target:
                exit_price = target
                result = "WIN"
                break

        if not entered:
            continue

        pnl = float(exit_price - entry)

        trades.append({
            "model": model["name"],
            "date": str(date),
            "entry": float(entry),
            "exit": float(exit_price),
            "pnl": pnl,
            "result": result,
            "mfe": float(mfe),
            "mae": float(mae),
            "bars_held": int(bars_held),
            "year": int(str(date)[:4]),
        })

    if not trades:
        continue

    t = pl.DataFrame(trades)

    wins = t.filter(pl.col("pnl") > 0).height
    losses = t.filter(pl.col("pnl") < 0).height

    gross_win = t.filter(pl.col("pnl") > 0)["pnl"].sum()
    gross_loss = abs(t.filter(pl.col("pnl") < 0)["pnl"].sum())

    pf = gross_win / gross_loss if gross_loss else 999
    wr = wins / t.height
    net = t["pnl"].sum()

    eq = t.with_columns(
        pl.col("pnl").cum_sum().alias("equity")
    )

    dd = (
        eq["equity"] -
        eq["equity"].cum_max()
    ).min()

    yearly = (
        t.group_by("year")
        .agg([
            pl.col("pnl").sum().alias("year_net"),
            pl.len().alias("trades"),
        ])
        .sort("year")
    )

    profitable_years = yearly.filter(
        pl.col("year_net") > 0
    ).height

    total_years = yearly.height

    summary_rows.append({
        "model": model["name"],
        "trades": t.height,
        "wins": wins,
        "losses": losses,
        "winrate": round(wr, 4),
        "profit_factor": round(float(pf), 3),
        "net_points": round(float(net), 2),
        "avg_trade": round(float(t["pnl"].mean()), 2),
        "max_drawdown": round(float(dd), 2),
        "avg_mfe": round(float(t["mfe"].mean()), 2),
        "avg_mae": round(float(t["mae"].mean()), 2),
        "avg_bars_held": round(float(t["bars_held"].mean()), 2),
        "profitable_years": profitable_years,
        "years_tested": total_years,
        "profitable_years_pct": round(
            profitable_years / total_years,
            3
        ),
    })

    all_trades.append(t)

summary = (
    pl.DataFrame(summary_rows)
    .sort(
        ["winrate", "profit_factor", "net_points"],
        descending=True
    )
)

trades = pl.concat(all_trades, how="vertical_relaxed")

summary_path = OUT / "25_displacement_reclaim_summary.csv"
trades_path = OUT / "25_displacement_reclaim_trades.csv"

summary.write_csv(summary_path)
trades.write_csv(trades_path)

print("\nSUMMARY")
print(summary)

print("\nSaved:")
print(summary_path)
print(trades_path)