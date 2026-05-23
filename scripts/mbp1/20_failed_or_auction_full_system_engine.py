import polars as pl
from pathlib import Path
from itertools import product

DATA = Path(r"D:\TradingResearch\data_parquet\master_nq_research_dataset.parquet")
OUT = Path(r"D:\TradingResearch\research_outputs")
OUT.mkdir(parents=True, exist_ok=True)

SCRIPT = "20_failed_or_auction_full_system_engine_FIXED"

TP_LIST = [20, 25, 30, 40]
SL_LIST = [20, 25, 30, 40]
MAX_WAIT_LIST = [5, 10, 15]
PULLBACK_LIST = [5, 10, 15]
BE_TRIGGER_LIST = [12, 15, 20]
MIN_OR_RANGE_LIST = [40, 50, 60, 70]
MBP_MODES = [False, True]

START = "2026-02-01"
END = "2026-05-01"

print("Loading data...")

spread_q40 = (
    pl.scan_parquet(DATA)
    .filter(
        (pl.col("trade_date_ct") >= pl.lit(START).str.to_date()) &
        (pl.col("trade_date_ct") < pl.lit(END).str.to_date())
    )
    .select(pl.col("avg_spread").quantile(0.40))
    .collect()
    .item()
)

df = (
    pl.scan_parquet(DATA)
    .filter(
        (pl.col("trade_date_ct") >= pl.lit(START).str.to_date()) &
        (pl.col("trade_date_ct") < pl.lit(END).str.to_date())
    )
    .with_columns([
        pl.col("ts_ct").alias("ts"),
        pl.col("trade_date_ct").alias("date"),
        pl.col("or_high_30m").alias("or_high"),
        pl.col("or_low_30m").alias("or_low"),
        pl.col("or_range_30m").alias("or_range"),
        (pl.col("avg_spread") <= spread_q40).alias("tight_spread"),
    ])
    .select([
        "ts", "date", "open", "high", "low", "close",
        "prior_rth_low", "or_high", "or_low", "or_range",
        "is_rth", "avg_spread", "tight_spread",
        "quote_pressure_delta", "net_mid_pressure", "update_velocity"
    ])
    .collect()
    .sort("ts")
)

print(f"Rows loaded: {df.height:,}")

days = df.partition_by("date", as_dict=True)

def tier_model(trades, win_rate, pf, net, dd):
    if trades < 8:
        return "REJECT"
    if win_rate >= 0.65 and pf >= 1.50 and net > 0 and dd > -250:
        return "A+"
    if win_rate >= 0.58 and pf >= 1.25 and net > 0 and dd > -400:
        return "A"
    if win_rate >= 0.52 and pf >= 1.05 and net > 0:
        return "B"
    return "REJECT"

def test_model(tp, sl, wait_bars, pullback_pts, be_trigger, min_or_range, mbp):
    trades = []

    for date_key, day in days.items():
        date = date_key[0] if isinstance(date_key, tuple) else date_key
        day = day.sort("ts")

        setups = day.filter(
            (pl.col("is_rth") == True) &
            (pl.col("or_range") >= min_or_range) &
            (pl.col("low") < pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("prior_rth_low")) &
            (pl.col("close") > pl.col("or_low")) &
            (pl.col("close") > pl.col("open"))
        )

        if mbp:
            setups = setups.filter(
                (pl.col("tight_spread") == True) &
                (pl.col("quote_pressure_delta") > 0) &
                (pl.col("net_mid_pressure") > 0)
            )

        if setups.height == 0:
            continue

        setup = setups.row(0, named=True)
        setup_ts = setup["ts"]
        entry_price = float(setup["close"] - pullback_pts)

        future = day.filter(pl.col("ts") > setup_ts).head(wait_bars)
        entry_rows = future.filter(pl.col("low") <= entry_price)

        if entry_rows.height == 0:
            continue

        entry_ts = entry_rows.row(0, named=True)["ts"]
        after = day.filter(pl.col("ts") > entry_ts)

        stop = float(entry_price - sl)
        target = float(entry_price + tp)
        active_stop = stop
        be_moved = False

        result = "EOD"
        exit_price = entry_price
        exit_ts = entry_ts
        mfe = 0.0
        mae = 0.0

        if after.height > 0:
            last = after.tail(1).row(0, named=True)
            exit_price = float(last["close"])
            exit_ts = last["ts"]

        for bar in after.iter_rows(named=True):
            mfe = max(mfe, float(bar["high"] - entry_price))
            mae = min(mae, float(bar["low"] - entry_price))

            if not be_moved and bar["high"] >= entry_price + be_trigger:
                active_stop = entry_price
                be_moved = True

            if bar["low"] <= active_stop:
                exit_price = float(active_stop)
                exit_ts = bar["ts"]
                result = "BE" if active_stop == entry_price else "LOSS"
                break

            if bar["high"] >= target:
                exit_price = float(target)
                exit_ts = bar["ts"]
                result = "WIN"
                break

        trades.append({
            "date": str(date),
            "entry_ts": str(entry_ts),
            "exit_ts": str(exit_ts),
            "entry": float(entry_price),
            "exit": float(exit_price),
            "pnl": float(exit_price - entry_price),
            "result": result,
            "mfe": float(mfe),
            "mae": float(mae),
            "tp": int(tp),
            "sl": int(sl),
            "wait_bars": int(wait_bars),
            "pullback_pts": int(pullback_pts),
            "be_trigger": int(be_trigger),
            "min_or_range": int(min_or_range),
            "mbp_confirmed": bool(mbp),
        })

    if not trades:
        return None, None

    t = pl.DataFrame(trades)

    wins = t.filter(pl.col("pnl") > 0).height
    losses = t.filter(pl.col("pnl") < 0).height
    total = t.height
    gross_win = float(t.filter(pl.col("pnl") > 0)["pnl"].sum() or 0)
    gross_loss = abs(float(t.filter(pl.col("pnl") < 0)["pnl"].sum() or 0))
    pf = gross_win / gross_loss if gross_loss else 999.0
    net = float(t["pnl"].sum())
    wr = wins / total if total else 0.0

    eq = t.with_columns(pl.col("pnl").cum_sum().alias("equity"))
    dd = float((eq["equity"] - eq["equity"].cum_max()).min())

    s = {
        "tier": tier_model(total, wr, pf, net, dd),
        "model": "MBP_CONFIRMED" if mbp else "STRUCTURE_ONLY",
        "trades": int(total),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "net_points": float(net),
        "avg_trade": float(t["pnl"].mean()),
        "max_dd": float(dd),
        "tp": int(tp),
        "sl": int(sl),
        "wait_bars": int(wait_bars),
        "pullback_pts": int(pullback_pts),
        "be_trigger": int(be_trigger),
        "min_or_range": int(min_or_range),
    }

    return s, t

summaries = []
trade_sets = []

combos = list(product(
    TP_LIST,
    SL_LIST,
    MAX_WAIT_LIST,
    PULLBACK_LIST,
    BE_TRIGGER_LIST,
    MIN_OR_RANGE_LIST,
    MBP_MODES
))

print(f"Testing models: {len(combos):,}")

for i, combo in enumerate(combos, 1):
    s, t = test_model(*combo)

    if s is not None:
        summaries.append(s)
        trade_sets.append(t)

    if i % 100 == 0:
        print(f"Completed {i:,}/{len(combos):,}")

if not summaries:
    print("No valid models.")
    raise SystemExit

summary = pl.DataFrame(summaries).sort(
    ["tier", "win_rate", "profit_factor", "net_points"],
    descending=[False, True, True, True]
)

trades = pl.concat(trade_sets, how="vertical_relaxed") if trade_sets else pl.DataFrame()

summary_path = OUT / f"{SCRIPT}_summary.csv"
trades_path = OUT / f"{SCRIPT}_trades.csv"

summary.write_csv(summary_path)
trades.write_csv(trades_path)

print("\nTOP MODELS")
print(summary.filter(pl.col("tier") != "REJECT").head(50))

print("\nTIER COUNTS")
print(summary.group_by("tier").len().sort("tier"))

print("\nDONE")
print(summary_path)
print(trades_path)