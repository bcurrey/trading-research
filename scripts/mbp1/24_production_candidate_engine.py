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
    ])
    .select([
        "ts",
        "date",
        "open",
        "high",
        "low",
        "close",
        "is_rth",
        "prior_rth_low",
        "or_low",
        "or_range",
        "premarket_range",
        "dist_vwap_day",
        "tight_spread",
        "mbp_confirmed",
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
        "name": "core_vwap_green_10pb",
        "require_below_vwap": True,
        "require_green_bar": True,
        "require_large_pm": False,
        "require_mbp": False,
        "pullback": 10,
        "tp": 20,
        "sl": 20,
    },

    {
        "name": "core_vwap_green_10pb_mbp",
        "require_below_vwap": True,
        "require_green_bar": True,
        "require_large_pm": False,
        "require_mbp": True,
        "pullback": 10,
        "tp": 20,
        "sl": 20,
    },

    {
        "name": "large_pm_vwap_green",
        "require_below_vwap": True,
        "require_green_bar": True,
        "require_large_pm": True,
        "require_mbp": False,
        "pullback": 10,
        "tp": 20,
        "sl": 20,
    },

    {
        "name": "large_pm_vwap_green_mbp",
        "require_below_vwap": True,
        "require_green_bar": True,
        "require_large_pm": True,
        "require_mbp": True,
        "pullback": 10,
        "tp": 20,
        "sl": 20,
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

        if model["require_below_vwap"]:
            day = day.filter(pl.col("dist_vwap_day") < 0)

        if model["require_green_bar"]:
            day = day.filter(pl.col("close") > pl.col("open"))

        if model["require_large_pm"]:
            day = day.filter(pl.col("premarket_range") >= 80)

        if model["require_mbp"]:
            day = day.filter(pl.col("mbp_confirmed") == True)

        setups = day.filter(
            (pl.col("low") < pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("or_low"))
        )

        if setups.height == 0:
            continue

        s = setups.row(0, named=True)

        entry = float(s["close"] - model["pullback"])

        future = day.filter(pl.col("ts") > s["ts"]).head(15)

        entry_rows = future.filter(pl.col("low") <= entry)

        if entry_rows.height == 0:
            continue

        entry_ts = entry_rows.row(0, named=True)["ts"]

        after = day.filter(pl.col("ts") > entry_ts)

        stop = entry - model["sl"]
        target = entry + model["tp"]

        result = "EOD"
        exit_price = entry
        mfe = 0.0
        mae = 0.0
        bars_held = 0
        target_hit_bar = None

        for idx, bar in enumerate(after.iter_rows(named=True), 1):

            bars_held += 1

            mfe = max(mfe, float(bar["high"] - entry))
            mae = min(mae, float(bar["low"] - entry))

            if bar["low"] <= stop:
                exit_price = stop
                result = "LOSS"
                break

            if bar["high"] >= target:
                exit_price = target
                result = "WIN"
                target_hit_bar = idx
                break

        pnl = float(exit_price - entry)

        trades.append({
            "model": model["name"],
            "date": str(date),
            "entry_ts": str(entry_ts),
            "entry": float(entry),
            "exit": float(exit_price),
            "pnl": pnl,
            "result": result,
            "mfe": float(mfe),
            "mae": float(mae),
            "bars_held": int(bars_held),
            "target_hit_bar": target_hit_bar if target_hit_bar else -1,
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

    eq = t.with_columns(pl.col("pnl").cum_sum().alias("equity"))

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

    profitable_years = yearly.filter(pl.col("year_net") > 0).height
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
        "profitable_years_pct": round(profitable_years / total_years, 3),
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

summary_path = OUT / "24_production_candidate_summary.csv"
trades_path = OUT / "24_production_candidate_trades.csv"

summary.write_csv(summary_path)
trades.write_csv(trades_path)

print("\nSUMMARY")
print(summary)

print("\nSaved:")
print(summary_path)
print(trades_path)