import polars as pl
from pathlib import Path

DATA = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT = Path(r"D:\TradingResearch\research_outputs")

MODELS = OUT / "21_clean_A_Aplus_models_for_next_test.csv"

print("Loading models...")
models = pl.read_csv(MODELS)

print("Loading market data...")

spread_q40 = (
    pl.scan_parquet(DATA)
    .select(pl.col("avg_spread").quantile(0.40))
    .collect()
    .item()
)

df = (
    pl.scan_parquet(DATA)
    .filter(
        (pl.col("trade_date_ct") >= pl.date(2018,1,1)) &
        (pl.col("trade_date_ct") <= pl.date(2026,5,1))
    )
    .with_columns([
        pl.col("ts_ct").alias("ts"),
        pl.col("trade_date_ct").alias("date"),
        pl.col("or_high_30m").alias("or_high"),
        pl.col("or_low_30m").alias("or_low"),
        pl.col("or_range_30m").alias("or_range"),

        (pl.col("avg_spread") <= spread_q40).alias("tight_spread_filter"),
        (pl.col("quote_pressure_delta") > 0).alias("quote_pressure_filter"),
        (pl.col("net_mid_pressure") > 0).alias("mid_pressure_filter"),

        (
            (pl.col("avg_spread") <= spread_q40) &
            (pl.col("quote_pressure_delta") > 0)
        ).alias("tight_plus_pressure"),
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
        "tight_spread_filter",
        "quote_pressure_filter",
        "mid_pressure_filter",
        "tight_plus_pressure"
    ])
    .collect()
    .sort("ts")
)

print(f"Rows: {df.height:,}")

days = df.partition_by("date", as_dict=True)

def apply_condition_filter(day, cond):
    if "green_bar" in cond:
        day = day.filter(pl.col("close") > pl.col("open"))

    if "below_vwap" in cond:
        day = day.filter(pl.col("dist_vwap_day") < 0)

    if "large_premarket" in cond:
        day = day.filter(pl.col("premarket_range") >= 80)

    if "low_volatility" in cond:
        day = day.filter(pl.col("or_range") <= 60)

    return day

def apply_mbp(day, mbp):
    if mbp == "NO_MBP_FILTER":
        return day

    if mbp == "tight_spread":
        return day.filter(pl.col("tight_spread_filter") == True)

    if mbp == "positive_quote_pressure":
        return day.filter(pl.col("quote_pressure_filter") == True)

    if mbp == "positive_mid_pressure":
        return day.filter(pl.col("mid_pressure_filter") == True)

    if mbp == "tight_spread_plus_pressure":
        return day.filter(pl.col("tight_plus_pressure") == True)

    return day

results = []

models = models.head(120)

print(f"Testing models: {models.height:,}")

for idx, m in enumerate(models.iter_rows(named=True), 1):

    cond = m["conditions"]
    entry_model = m["entry_model"]
    target_model = m["target_model"]

    if "5pt" in entry_model:
        pb = 5
    elif "10pt" in entry_model:
        pb = 10
    else:
        pb = 15

    if "20pt" in target_model:
        tp = 20
    elif "25pt" in target_model:
        tp = 25
    elif "30pt" in target_model:
        tp = 30
    else:
        tp = 40

    for mbp_name in [
        "NO_MBP_FILTER",
        "tight_spread",
        "positive_quote_pressure",
        "positive_mid_pressure",
        "tight_spread_plus_pressure"
    ]:

        trades = []

        for date_key, day in days.items():

            date = date_key[0] if isinstance(date_key, tuple) else date_key

            day = apply_condition_filter(day, cond)
            day = apply_mbp(day, mbp_name)

            setups = day.filter(
                (pl.col("is_rth") == True) &
                (pl.col("low") < pl.col("prior_rth_low")) &
                (pl.col("close") > pl.col("prior_rth_low")) &
                (pl.col("close") > pl.col("or_low"))
            )

            if setups.height == 0:
                continue

            s = setups.row(0, named=True)

            entry = float(s["close"] - pb)
            stop = float(entry - 20)
            target = float(entry + tp)

            after = day.filter(pl.col("ts") > s["ts"])

            if after.height == 0:
                continue

            result = "EOD"
            pnl = 0.0

            for bar in after.iter_rows(named=True):

                if bar["low"] <= stop:
                    pnl = stop - entry
                    result = "LOSS"
                    break

                if bar["high"] >= target:
                    pnl = target - entry
                    result = "WIN"
                    break

            trades.append(pnl)

        if len(trades) < 8:
            continue

        t = pl.Series(trades)

        wins = (t > 0).sum()
        losses = (t < 0).sum()

        gross_win = t.filter(t > 0).sum()
        gross_loss = abs(t.filter(t < 0).sum())

        pf = gross_win / gross_loss if gross_loss else 999
        wr = wins / len(trades)
        net = t.sum()

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
            "conditions": cond,
            "entry_model": entry_model,
            "target_model": target_model,
            "mbp_overlay": mbp_name,
            "trades": len(trades),
            "winrate": round(wr, 4),
            "profit_factor": round(pf, 3),
            "net_points": round(float(net), 2),
            "avg_trade": round(float(t.mean()), 2),
        })

    if idx % 10 == 0:
        print(f"Completed {idx}/{models.height}")

out = (
    pl.DataFrame(results)
    .sort(
        ["tier", "winrate", "profit_factor", "net_points"],
        descending=[False, True, True, True]
    )
)

print("\nTOP MODELS")
print(out.filter(pl.col("tier") != "REJECT").head(100))

print("\nTIER COUNTS")
print(out.group_by("tier").len().sort("tier"))

out_path = OUT / "23_mbp_retested_models.csv"
out.write_csv(out_path)

print("\nSaved:")
print(out_path)