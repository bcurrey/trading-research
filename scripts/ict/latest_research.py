from pathlib import Path
from datetime import datetime
import traceback
import sys
import polars as pl

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT = Path(r'D:\TradingResearch')
OUT = ROOT / 'research_outputs'
STUDY = OUT / 'htf_fvg_study'
OUT.mkdir(parents=True, exist_ok=True)
STUDY.mkdir(parents=True, exist_ok=True)
STATUS = OUT / 'run_status.txt'
DATA = ROOT / 'data_parquet' / 'NQ_feature_factory_ict_context.parquet'

POINT_VALUE = 2.0
CONTRACTS = 3
STOP_BUFFER = 1.25
TP1, TP2, TP3 = 25.0, 50.0, 75.0
NY_AM_START = 8 * 60 + 30
NY_AM_END = 11 * 60 + 30
FORCE_EXIT = 14 * 60 + 45
VARIANTS = [('NO_MAX_SL', None), ('MAX_SL_75', 75.0), ('MAX_SL_50', 50.0)]


def write_status(s, msg):
    STATUS.write_text(f'status={s}\ntimestamp={datetime.now()}\nmessage={msg}\n', encoding='utf-8')


def minute(ts):
    return ts.hour * 60 + ts.minute


def is_am(ts):
    m = minute(ts)
    return NY_AM_START <= m < NY_AM_END


def touched(b, lo, hi):
    return b['high'] >= lo and b['low'] <= hi


def build_fvgs(df, label, every):
    htf = (
        df.group_by_dynamic('ts', every=every, period=every, closed='left', label='left')
        .agg([
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.len().alias('minute_rows'),
        ])
        .filter(pl.col('minute_rows') >= 10)
        .sort('ts')
        .with_row_index('idx')
        .with_columns([
            pl.col('high').shift(2).alias('high_2'),
            pl.col('low').shift(2).alias('low_2'),
        ])
    )
    bull = htf.filter(pl.col('low') > pl.col('high_2')).select([
        pl.lit(label).alias('zone_type'), pl.lit('LONG').alias('side'), pl.col('idx').alias('source_idx'),
        pl.col('ts').alias('zone_time'), pl.col('high_2').alias('zone_low'), pl.col('low').alias('zone_high')
    ])
    bear = htf.filter(pl.col('high') < pl.col('low_2')).select([
        pl.lit(label).alias('zone_type'), pl.lit('SHORT').alias('side'), pl.col('idx').alias('source_idx'),
        pl.col('ts').alias('zone_time'), pl.col('high').alias('zone_low'), pl.col('low_2').alias('zone_high')
    ])
    return pl.concat([bull, bear])


def mfe_mae(side, entry, path):
    mf = 0.0
    ma = 0.0
    for b in path:
        if side == 'LONG':
            mf = max(mf, b['high'] - entry)
            ma = max(ma, entry - b['low'])
        else:
            mf = max(mf, entry - b['low'])
            ma = max(ma, b['high'] - entry)
    return round(mf, 2), round(ma, 2)


def pf_expr():
    gross_win = pl.col('pnl_dollars').filter(pl.col('pnl_dollars') > 0).sum()
    gross_loss = pl.col('pnl_dollars').filter(pl.col('pnl_dollars') < 0).sum().abs()
    return pl.when(gross_loss != 0).then(gross_win / gross_loss).otherwise(None).round(3).alias('profit_factor')


def replay(z, bars, variant, max_sl):
    side = z['side']
    lo = float(z['zone_low'])
    hi = float(z['zone_high'])
    zone_time = z['zone_time']
    size = hi - lo
    if side == 'LONG':
        entry = hi
        stop0 = lo - STOP_BUFFER
        risk = entry - stop0
        tp1, tp2, tp3 = entry + TP1, entry + TP2, entry + TP3
    else:
        entry = lo
        stop0 = hi + STOP_BUFFER
        risk = stop0 - entry
        tp1, tp2, tp3 = entry - TP1, entry - TP2, entry - TP3
    if max_sl is not None and risk > max_sl:
        return None

    touch_i = None
    touch_bar = None
    for i, b in enumerate(bars):
        ts = b['ts']
        if ts <= zone_time:
            continue
        if not touched(b, lo, hi):
            continue
        if is_am(ts):
            touch_i = i
            touch_bar = b
            break
        if minute(ts) < NY_AM_START or minute(ts) >= NY_AM_END:
            break
    if touch_i is None:
        return None

    trade_date = touch_bar['ts'].date()
    stop = stop0
    open_ct = CONTRACTS
    points = 0.0
    h1 = h2 = h3 = False
    t1 = t2 = t3 = None
    exit_time = None
    exit_reason = None
    final_price = None
    path = []

    for b in bars[touch_i:]:
        ts = b['ts']
        if ts.date() != trade_date:
            break
        path.append(b)
        if side == 'LONG':
            stop_hit = b['low'] <= stop
            hit1, hit2, hit3 = b['high'] >= tp1, b['high'] >= tp2, b['high'] >= tp3
            force_price = b['close']
        else:
            stop_hit = b['high'] >= stop
            hit1, hit2, hit3 = b['low'] <= tp1, b['low'] <= tp2, b['low'] <= tp3
            force_price = b['close']

        if stop_hit:
            points += ((stop - entry) if side == 'LONG' else (entry - stop)) * open_ct
            exit_time, exit_reason, final_price = ts, 'STOP', stop
            open_ct = 0
            break
        if (not h1) and hit1:
            points += TP1
            open_ct -= 1
            h1, t1 = True, ts
            stop = entry
        if (not h2) and hit2:
            points += TP2
            open_ct -= 1
            h2, t2 = True, ts
            stop = entry + TP1 if side == 'LONG' else entry - TP1
        if (not h3) and hit3:
            points += TP3
            open_ct -= 1
            h3, t3 = True, ts
            exit_time, exit_reason, final_price = ts, 'TP3', tp3
            open_ct = 0
            break
        if minute(ts) >= FORCE_EXIT:
            points += ((force_price - entry) if side == 'LONG' else (entry - force_price)) * open_ct
            exit_time, exit_reason, final_price = ts, 'FORCE_EXIT_1445', force_price
            open_ct = 0
            break

    if exit_time is None:
        last = path[-1] if path else touch_bar
        final_price = last['close']
        points += ((final_price - entry) if side == 'LONG' else (entry - final_price)) * open_ct
        exit_time, exit_reason = last['ts'], 'EOD_FALLBACK'

    mf, ma = mfe_mae(side, entry, path)
    pnl = points * POINT_VALUE
    return {
        'max_stop_variant': variant, 'zone_id': z['zone_id'], 'zone_type': z['zone_type'], 'side': side,
        'zone_time': zone_time, 'zone_year': z['zone_year'], 'trade_date': str(trade_date), 'trade_year': touch_bar['ts'].year,
        'entry_time': touch_bar['ts'], 'exit_time': exit_time, 'exit_reason': exit_reason,
        'zone_low': round(lo, 2), 'zone_high': round(hi, 2), 'zone_size': round(size, 2),
        'entry': round(entry, 2), 'stop_initial': round(stop0, 2), 'final_exit_price': round(final_price, 2),
        'initial_risk_points': round(risk, 2), 'initial_risk_dollars': round(risk * CONTRACTS * POINT_VALUE, 2),
        'tp1_hit': h1, 'tp2_hit': h2, 'tp3_hit': h3, 'tp1_time': t1, 'tp2_time': t2, 'tp3_time': t3,
        'gross_points_3_contracts': round(points, 2), 'pnl_dollars': round(pnl, 2),
        'mfe_points_from_entry': mf, 'mae_points_from_entry': ma,
        'r_multiple': round(points / (risk * CONTRACTS), 4) if risk > 0 else None,
        'win': pnl > 0, 'loss': pnl < 0, 'breakeven': pnl == 0,
    }


def summarize(trades, cols):
    return trades.group_by(cols).agg([
        pl.len().alias('trades'), pl.col('win').sum().alias('wins'), pl.col('loss').sum().alias('losses'),
        pl.col('breakeven').sum().alias('breakevens'), (pl.col('win').mean() * 100).round(2).alias('win_rate'),
        pl.col('pnl_dollars').sum().round(2).alias('net_dollars'),
        pl.col('gross_points_3_contracts').sum().round(2).alias('net_points_3_contracts'),
        pl.col('pnl_dollars').mean().round(2).alias('avg_dollars'), pf_expr(),
        (pl.col('tp1_hit').mean() * 100).round(2).alias('tp1_pct'),
        (pl.col('tp2_hit').mean() * 100).round(2).alias('tp2_pct'),
        (pl.col('tp3_hit').mean() * 100).round(2).alias('tp3_pct'),
        pl.col('initial_risk_points').mean().round(2).alias('avg_initial_risk_points'),
        pl.col('initial_risk_points').max().round(2).alias('max_initial_risk_points'),
        pl.col('pnl_dollars').min().round(2).alias('single_worst_loss_dollars'),
        pl.col('gross_points_3_contracts').min().round(2).alias('single_worst_loss_points_3_contracts'),
        pl.col('zone_size').mean().round(2).alias('avg_zone_size'),
        pl.col('zone_size').max().round(2).alias('max_zone_size'),
        pl.col('mfe_points_from_entry').mean().round(2).alias('avg_mfe_points'),
        pl.col('mae_points_from_entry').mean().round(2).alias('avg_mae_points'),
    ]).sort(cols)


def main():
    print('FVG BY YEAR / MAX STOP / WORST LOSSES')
    print(f'DATA: {DATA}')
    df = (pl.read_parquet(DATA).select([
        pl.col('ts_ct').cast(pl.Datetime).alias('ts'), pl.col('open').cast(pl.Float64),
        pl.col('high').cast(pl.Float64), pl.col('low').cast(pl.Float64), pl.col('close').cast(pl.Float64),
    ]).sort('ts').with_columns([
        pl.col('ts').dt.date().alias('trade_date'), pl.col('ts').dt.year().alias('year'),
        ((pl.col('ts').dt.hour() * 60) + pl.col('ts').dt.minute()).alias('minute_of_day'),
    ]))
    print(f'Rows loaded: {df.height:,}')
    print(f'Date range: {df["ts"].min()} -> {df["ts"].max()}')
    bars = df.to_dicts()
    zones = (pl.concat([build_fvgs(df, '30m_FVG', '30m'), build_fvgs(df, '1h_FVG', '1h'), build_fvgs(df, '4h_FVG', '4h')])
        .with_columns([(pl.col('zone_high') - pl.col('zone_low')).alias('zone_size'), pl.col('zone_time').dt.year().alias('zone_year')])
        .filter(pl.col('zone_size') > 0).sort('zone_time').with_row_index('zone_id'))
    print('Zones built:')
    print(zones.group_by('zone_type').agg(pl.len().alias('zones')).sort('zone_type'))

    rows = []
    zrows = zones.to_dicts()
    for label, mx in VARIANTS:
        print(f'Replaying {label}')
        n = 0
        for z in zrows:
            r = replay(z, bars, label, mx)
            if r is not None:
                rows.append(r); n += 1
        print(f'Trades found for {label}: {n:,}')
    if not rows:
        raise RuntimeError('No trades found')
    trades = pl.DataFrame(rows)

    comparison = summarize(trades, ['max_stop_variant', 'zone_type'])
    by_year = summarize(trades, ['max_stop_variant', 'zone_type', 'trade_year'])
    by_side_year = summarize(trades, ['max_stop_variant', 'zone_type', 'side', 'trade_year'])
    worst10 = (trades.sort(['max_stop_variant', 'zone_type', 'trade_year', 'pnl_dollars'])
        .group_by(['max_stop_variant', 'zone_type', 'trade_year'], maintain_order=True).head(10))
    loss_profile = (trades.filter(pl.col('pnl_dollars') < 0).group_by(['max_stop_variant', 'zone_type']).agg([
        pl.len().alias('loss_count'), pl.col('zone_size').mean().round(2).alias('loss_avg_zone_size'),
        pl.col('zone_size').median().round(2).alias('loss_median_zone_size'), pl.col('zone_size').max().round(2).alias('loss_max_zone_size'),
        pl.col('initial_risk_points').mean().round(2).alias('loss_avg_initial_risk_points'),
        pl.col('initial_risk_points').max().round(2).alias('loss_max_initial_risk_points'),
        pl.col('pnl_dollars').min().round(2).alias('worst_loss_dollars')
    ]))

    outputs = {
        'latest_63_fvg_scaleout_all_trades.csv': trades,
        'latest_63_fvg_scaleout_comparison_by_stop.csv': comparison,
        'latest_63_fvg_scaleout_by_year_by_stop.csv': by_year,
        'latest_63_fvg_scaleout_by_side_year_by_stop.csv': by_side_year,
        'latest_63_fvg_scaleout_worst10_by_year.csv': worst10,
        'latest_63_fvg_scaleout_loss_profile.csv': loss_profile,
    }
    for name, frame in outputs.items():
        p = STUDY / name
        frame.write_csv(p)
        print(f'Saved: {p}')
    print('COMPARISON BY STOP')
    print(comparison)
    print('BY YEAR')
    print(by_year)
    write_status('SUCCESS', 'Completed full FVG by-year max-stop worst-loss study')

try:
    main()
except Exception as e:
    print(traceback.format_exc())
    write_status('FAILED', f'{type(e).__name__}: {e}')
    raise
