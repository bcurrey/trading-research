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
DATA = ROOT / 'data_parquet' / 'NQ_feature_factory_ict_context.parquet'
STATUS = OUT / 'run_status.txt'
OUT.mkdir(parents=True, exist_ok=True)
STUDY.mkdir(parents=True, exist_ok=True)

POINT_VALUE = 2.0
CONTRACTS = 3
STOP_BUFFER = 1.25
TP1, TP2, TP3 = 25.0, 50.0, 75.0
NY_AM_START = 8 * 60 + 30
NY_AM_END = 11 * 60 + 30
FORCE_EXIT = 14 * 60 + 45
VARIANTS = [('NO_MAX_SL', None), ('MAX_SL_75', 75.0), ('MAX_SL_50', 50.0)]


def status(s, msg):
    STATUS.write_text(f'status={s}\ntimestamp={datetime.now()}\nmessage={msg}\n', encoding='utf-8')


def minute(ts):
    return ts.hour * 60 + ts.minute


def is_ny_am(ts):
    m = minute(ts)
    return NY_AM_START <= m < NY_AM_END


def touched(b, lo, hi):
    return b['high'] >= lo and b['low'] <= hi


def pf_expr():
    gw = pl.col('pnl_dollars').filter(pl.col('pnl_dollars') > 0).sum()
    gl = pl.col('pnl_dollars').filter(pl.col('pnl_dollars') < 0).sum().abs()
    return pl.when(gl != 0).then(gw / gl).otherwise(None).round(3).alias('profit_factor')


def build_fvgs(df, label, every):
    htf = (df.group_by_dynamic('ts', every=every, period=every, closed='left', label='left')
        .agg([
            pl.col('open').first().alias('open'), pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'), pl.col('close').last().alias('close'), pl.len().alias('rows')
        ])
        .filter(pl.col('rows') >= 10)
        .sort('ts')
        .with_row_index('source_idx')
        .with_columns([pl.col('high').shift(2).alias('high_2'), pl.col('low').shift(2).alias('low_2')]))
    bull = htf.filter(pl.col('low') > pl.col('high_2')).select([
        pl.lit(label).alias('zone_type'), pl.lit('LONG').alias('side'), 'source_idx',
        pl.col('ts').alias('zone_time'), pl.col('high_2').alias('zone_low'), pl.col('low').alias('zone_high')])
    bear = htf.filter(pl.col('high') < pl.col('low_2')).select([
        pl.lit(label).alias('zone_type'), pl.lit('SHORT').alias('side'), 'source_idx',
        pl.col('ts').alias('zone_time'), pl.col('high').alias('zone_low'), pl.col('low_2').alias('zone_high')])
    return pl.concat([bull, bear])


def bucket_touch_depth(side, lo, hi, touch_bar):
    size = hi - lo
    if size <= 0:
        return 'BAD_SIZE', None
    if side == 'LONG':
        px = min(max(touch_bar['low'], lo), hi)
        pct = ((hi - px) / size) * 100.0
    else:
        px = min(max(touch_bar['high'], lo), hi)
        pct = ((px - lo) / size) * 100.0
    if pct < 25:
        b = 'SHALLOW_0_25'
    elif pct < 50:
        b = 'MID_25_50'
    elif pct < 75:
        b = 'DEEP_50_75'
    else:
        b = 'VERY_DEEP_75_100'
    return b, round(pct, 2)


def bucket_size(x):
    if x <= 10: return '00_10'
    if x <= 20: return '10_20'
    if x <= 35: return '20_35'
    if x <= 50: return '35_50'
    if x <= 75: return '50_75'
    return '75_PLUS'


def mfe_mae(side, entry, path):
    mf = ma = 0.0
    for b in path:
        if side == 'LONG':
            mf = max(mf, b['high'] - entry)
            ma = max(ma, entry - b['low'])
        else:
            mf = max(mf, entry - b['low'])
            ma = max(ma, b['high'] - entry)
    return round(mf, 2), round(ma, 2)


def replay(z, bars_by_date, dates_sorted, variant, max_sl):
    side = z['side']; lo = float(z['zone_low']); hi = float(z['zone_high']); size = hi - lo
    if side == 'LONG':
        entry = hi; stop0 = lo - STOP_BUFFER; risk = entry - stop0; tps = [entry + TP1, entry + TP2, entry + TP3]
    else:
        entry = lo; stop0 = hi + STOP_BUFFER; risk = stop0 - entry; tps = [entry - TP1, entry - TP2, entry - TP3]
    if max_sl is not None and risk > max_sl:
        return None

    zone_date = z['zone_time'].date()
    scan_dates = [d for d in dates_sorted if d >= zone_date]
    touch_date = touch_i = None; touch_bar = None
    for d in scan_dates:
        day = bars_by_date[d]
        for i, b in enumerate(day):
            ts = b['ts']
            if ts <= z['zone_time']:
                continue
            if not is_ny_am(ts):
                continue
            if touched(b, lo, hi):
                touch_date, touch_i, touch_bar = d, i, b
                break
        if touch_bar is not None:
            break
        if len(scan_dates) > 0 and d > zone_date and d.year == 2026 and len(scan_dates) > 60:
            pass
    if touch_bar is None:
        return None

    depth_bucket, depth_pct = bucket_touch_depth(side, lo, hi, touch_bar)
    stop = stop0; open_ct = CONTRACTS; points = 0.0
    hit1 = hit2 = hit3 = False; time1 = time2 = time3 = None
    exit_time = None; exit_reason = None; final_price = None; path = []
    for b in bars_by_date[touch_date][touch_i:]:
        ts = b['ts']; path.append(b)
        if side == 'LONG':
            stop_hit = b['low'] <= stop; h1 = b['high'] >= tps[0]; h2 = b['high'] >= tps[1]; h3 = b['high'] >= tps[2]; force_px = b['close']
        else:
            stop_hit = b['high'] >= stop; h1 = b['low'] <= tps[0]; h2 = b['low'] <= tps[1]; h3 = b['low'] <= tps[2]; force_px = b['close']
        if stop_hit:
            points += ((stop - entry) if side == 'LONG' else (entry - stop)) * open_ct
            exit_time, exit_reason, final_price = ts, 'STOP', stop
            break
        if (not hit1) and h1:
            points += TP1; open_ct -= 1; hit1 = True; time1 = ts; stop = entry
        if (not hit2) and h2:
            points += TP2; open_ct -= 1; hit2 = True; time2 = ts; stop = entry + TP1 if side == 'LONG' else entry - TP1
        if (not hit3) and h3:
            points += TP3; open_ct -= 1; hit3 = True; time3 = ts; exit_time, exit_reason, final_price = ts, 'TP3', tps[2]; break
        if minute(ts) >= FORCE_EXIT:
            points += ((force_px - entry) if side == 'LONG' else (entry - force_px)) * open_ct
            exit_time, exit_reason, final_price = ts, 'FORCE_EXIT_1445', force_px
            break
    if exit_time is None:
        last = path[-1]
        final_price = last['close']
        points += ((final_price - entry) if side == 'LONG' else (entry - final_price)) * open_ct
        exit_time, exit_reason = last['ts'], 'EOD_FALLBACK'
    mf, ma = mfe_mae(side, entry, path)
    pnl = points * POINT_VALUE
    return {
        'variant': variant, 'zone_id': z['zone_id'], 'zone_type': z['zone_type'], 'side': side,
        'zone_time': z['zone_time'], 'zone_size': round(size, 2), 'zone_size_bucket': bucket_size(size),
        'trade_date': str(touch_date), 'entry_time': touch_bar['ts'], 'exit_time': exit_time, 'exit_reason': exit_reason,
        'entry': round(entry, 2), 'stop_initial': round(stop0, 2), 'initial_risk_points': round(risk, 2),
        'touch_depth_pct': depth_pct, 'touch_depth_bucket': depth_bucket,
        'tp1_hit': hit1, 'tp2_hit': hit2, 'tp3_hit': hit3, 'tp1_time': time1, 'tp2_time': time2, 'tp3_time': time3,
        'minutes_to_tp1': round((time1 - touch_bar['ts']).total_seconds()/60, 2) if time1 else None,
        'minutes_to_tp2': round((time2 - touch_bar['ts']).total_seconds()/60, 2) if time2 else None,
        'minutes_to_tp3': round((time3 - touch_bar['ts']).total_seconds()/60, 2) if time3 else None,
        'gross_points_3_contracts': round(points, 2), 'pnl_dollars': round(pnl, 2),
        'mfe_points': mf, 'mae_points': ma, 'win': pnl > 0, 'loss': pnl < 0, 'breakeven': pnl == 0,
    }


def summarize(df, cols):
    return df.group_by(cols).agg([
        pl.len().alias('trades'), pl.col('win').sum().alias('wins'), pl.col('loss').sum().alias('losses'),
        (pl.col('win').mean()*100).round(2).alias('win_rate'), pl.col('pnl_dollars').sum().round(2).alias('net_dollars'),
        pl.col('gross_points_3_contracts').sum().round(2).alias('net_points_3_contracts'), pl.col('pnl_dollars').mean().round(2).alias('avg_dollars'), pf_expr(),
        (pl.col('tp1_hit').mean()*100).round(2).alias('tp1_pct'), (pl.col('tp2_hit').mean()*100).round(2).alias('tp2_pct'),
        (pl.col('tp3_hit').mean()*100).round(2).alias('tp3_pct'), pl.col('initial_risk_points').mean().round(2).alias('avg_risk_pts'),
        pl.col('initial_risk_points').max().round(2).alias('max_risk_pts'), pl.col('pnl_dollars').min().round(2).alias('worst_loss_dollars'),
        pl.col('gross_points_3_contracts').min().round(2).alias('worst_loss_points_3c'), pl.col('mfe_points').mean().round(2).alias('avg_mfe'),
        pl.col('mae_points').mean().round(2).alias('avg_mae'), pl.col('minutes_to_tp1').mean().round(2).alias('avg_min_to_tp1')
    ]).sort(cols)


def main():
    print('FAST 2026 FVG PROFILE: 30m/1h/4h, max SL, touch depth, worst losses')
    df = (pl.read_parquet(DATA).select([
        pl.col('ts_ct').cast(pl.Datetime).alias('ts'), pl.col('open').cast(pl.Float64), pl.col('high').cast(pl.Float64), pl.col('low').cast(pl.Float64), pl.col('close').cast(pl.Float64)
    ]).filter((pl.col('ts') >= pl.datetime(2026,1,1)) & (pl.col('ts') < pl.datetime(2027,1,1))).sort('ts').with_columns(pl.col('ts').dt.date().alias('date')))
    print(f'Rows loaded: {df.height:,}')
    zones = pl.concat([build_fvgs(df,'30m_FVG','30m'), build_fvgs(df,'1h_FVG','1h'), build_fvgs(df,'4h_FVG','4h')]).with_columns((pl.col('zone_high')-pl.col('zone_low')).alias('zone_size')).filter(pl.col('zone_size')>0).sort('zone_time').with_row_index('zone_id')
    print(zones.group_by('zone_type').agg(pl.len().alias('zones')).sort('zone_type'))
    bars_by_date = {r['date']: [] for r in df.select('date').unique().to_dicts()}
    for b in df.to_dicts():
        bars_by_date[b['date']].append(b)
    dates_sorted = sorted(bars_by_date.keys())
    rows = []
    for label, mx in VARIANTS:
        print(f'Replay {label}')
        for z in zones.to_dicts():
            r = replay(z, bars_by_date, dates_sorted, label, mx)
            if r: rows.append(r)
        print(f'Rows so far: {len(rows):,}')
    trades = pl.DataFrame(rows)
    outputs = {
        'latest_fast_2026_fvg_trades.csv': trades,
        'latest_fast_2026_fvg_summary.csv': summarize(trades, ['variant','zone_type']),
        'latest_fast_2026_fvg_by_side.csv': summarize(trades, ['variant','zone_type','side']),
        'latest_fast_2026_fvg_by_size_bucket.csv': summarize(trades, ['variant','zone_type','zone_size_bucket']),
        'latest_fast_2026_fvg_by_touch_depth.csv': summarize(trades, ['variant','zone_type','touch_depth_bucket']),
        'latest_fast_2026_fvg_worst10.csv': trades.sort(['variant','zone_type','pnl_dollars']).group_by(['variant','zone_type'], maintain_order=True).head(10),
    }
    for name, frame in outputs.items():
        p = STUDY / name; frame.write_csv(p); print(f'Saved: {p}')
    print(outputs['latest_fast_2026_fvg_summary.csv'])
    status('SUCCESS', 'Completed fast 2026 FVG profile')

try:
    main()
except Exception as e:
    print(traceback.format_exc())
    status('FAILED', f'{type(e).__name__}: {e}')
    raise
