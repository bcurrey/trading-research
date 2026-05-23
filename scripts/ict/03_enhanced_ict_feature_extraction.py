from pathlib import Path
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

ROOT = Path(r'D:\TradingResearch')
TRADES_PATH = ROOT / r'ict_trade_study\outputs\ict_keeper_trades_clean.csv'
NQ_PATH = ROOT / r'data_parquet\NQ_feature_factory_ict_context.parquet'
OUTDIR = ROOT / r'ict_trade_study\outputs'
OUT_FEATURES = OUTDIR / 'ict_trade_enhanced_features.csv'
OUT_REVIEW = OUTDIR / 'ict_trade_enhanced_review.csv'
OUT_SUMMARY = OUTDIR / 'ict_trade_enhanced_summary.csv'

SEARCH_WINDOW_MINUTES = 6
FVG_LOOKBACK_BARS = 120
DOL_NEAR_POINTS = 15.0

def norm_cols(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df

def find_col(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

def load_trades():
    df = norm_cols(pd.read_csv(TRADES_PATH))
    for c in ['trade_id','date','entry_time','entry_price','side','result']:
        if c not in df.columns:
            raise ValueError(f'Missing trade column {c}. Columns: {df.columns.tolist()}')
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['entry_time'] = df['entry_time'].astype(str).str.strip()
    df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
    df['side'] = df['side'].astype(str).str.lower().str.strip()
    df['result'] = df['result'].astype(str).str.lower().str.strip()
    df['entry_dt'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['entry_time'].astype(str), errors='coerce')
    return df

def load_nq():
    df = norm_cols(pd.read_parquet(NQ_PATH))
    dt_col = find_col(df.columns, ["ts_ct", "datetime", "timestamp", "date_time", "time", "dt", "ts_event"])
    date_col = find_col(df.columns, ['date','trade_date','session_date'])
    time_col = find_col(df.columns, ['bar_time','time'])
    if dt_col:
        df['dt'] = pd.to_datetime(df[dt_col], errors='coerce')
    elif date_col and time_col:
        df['dt'] = pd.to_datetime(df[date_col].astype(str)+' '+df[time_col].astype(str), errors='coerce')
    else:
        raise ValueError('Could not identify datetime column')

    cmap = {
        'open': find_col(df.columns, ['open','o']),
        'high': find_col(df.columns, ['high','h']),
        'low': find_col(df.columns, ['low','l']),
        'close': find_col(df.columns, ['close','c']),
        'volume': find_col(df.columns, ['volume','vol']),
        'vwap': find_col(df.columns, ['vwap','daily_vwap']),
        'atr': find_col(df.columns, ['atr_14','atr','atr14']),
        'dol': find_col(df.columns, ['dol','daily_open','day_open','open_dol']),
        'pdh': find_col(df.columns, ['pdh','prev_day_high','prior_day_high']),
        'pdl': find_col(df.columns, ['pdl','prev_day_low','prior_day_low']),
        'premarket_high': find_col(df.columns, ['premarket_high','pre_market_high','pm_high']),
        'premarket_low': find_col(df.columns, ['premarket_low','pre_market_low','pm_low']),
        'asia_high': find_col(df.columns, ['asia_high','asian_high']),
        'asia_low': find_col(df.columns, ['asia_low','asian_low']),
        'london_high': find_col(df.columns, ['london_high','lon_high']),
        'london_low': find_col(df.columns, ['london_low','lon_low']),
    }
    for k in ['open','high','low','close']:
        if not cmap[k]:
            raise ValueError(f'Missing {k} column')
    for k, src in cmap.items():
        df['_'+k] = pd.to_numeric(df[src], errors='coerce') if src else np.nan
    df = df.dropna(subset=['dt','_open','_high','_low','_close']).sort_values('dt').reset_index(drop=True)
    df['_date'] = df['dt'].dt.date
    df['_range'] = (df['_high']-df['_low']).replace(0, np.nan)
    df['_body'] = (df['_close']-df['_open']).abs()
    df['_body_pct'] = df['_body']/df['_range']
    df['_atr_proxy_20'] = df['_range'].rolling(20, min_periods=5).mean()
    df['_body_avg_20'] = df['_body'].rolling(20, min_periods=5).mean()
    df['_vol_avg_20'] = df['_volume'].rolling(20, min_periods=5).mean()
    df['_bull_fvg'] = df['_low'] > df['_high'].shift(2)
    df['_bear_fvg'] = df['_high'] < df['_low'].shift(2)
    df['_bull_fvg_low'] = np.where(df['_bull_fvg'], df['_high'].shift(2), np.nan)
    df['_bull_fvg_high'] = np.where(df['_bull_fvg'], df['_low'], np.nan)
    df['_bear_fvg_low'] = np.where(df['_bear_fvg'], df['_high'], np.nan)
    df['_bear_fvg_high'] = np.where(df['_bear_fvg'], df['_low'].shift(2), np.nan)
    return df, cmap

def nearest_match(day_df, approx_dt, entry_price):
    lo = approx_dt - pd.Timedelta(minutes=SEARCH_WINDOW_MINUTES)
    hi = approx_dt + pd.Timedelta(minutes=SEARCH_WINDOW_MINUTES)
    w = day_df[(day_df['dt']>=lo)&(day_df['dt']<=hi)].copy()
    if w.empty:
        return None, {'match_status':'no_market_rows_in_time_window'}
    below = entry_price < w['_low']
    above = entry_price > w['_high']
    w['_price_dist'] = np.where(below, w['_low']-entry_price, np.where(above, entry_price-w['_high'], 0.0))
    w['_time_dist'] = (w['dt']-approx_dt).abs().dt.total_seconds()/60
    w['_score'] = w['_price_dist']*10 + w['_time_dist']
    idx = int(w['_score'].idxmin())
    return idx, {
        'match_status':'matched',
        'matched_dt': w.loc[idx,'dt'],
        'price_inside_bar': bool(w.loc[idx,'_low'] <= entry_price <= w.loc[idx,'_high']),
        'price_distance': float(w.loc[idx,'_price_dist']),
        'time_distance_min': float(w.loc[idx,'_time_dist'])
    }

def fvg_features(df, idx, entry_price, side):
    recent = df.iloc[max(0, idx-FVG_LOOKBACK_BARS):idx+1]
    cands = []
    for ridx, r in recent[recent['_bull_fvg']].iterrows():
        lo, hi = float(r['_bull_fvg_low']), float(r['_bull_fvg_high'])
        dist = 0 if lo <= entry_price <= hi else min(abs(entry_price-lo), abs(entry_price-hi))
        inv = (df.iloc[ridx:idx+1]['_close'] < lo).any()
        cands.append(('bull', ridx, lo, hi, hi-lo, dist, idx-ridx, inv))
    for ridx, r in recent[recent['_bear_fvg']].iterrows():
        lo, hi = float(r['_bear_fvg_low']), float(r['_bear_fvg_high'])
        dist = 0 if lo <= entry_price <= hi else min(abs(entry_price-lo), abs(entry_price-hi))
        inv = (df.iloc[ridx:idx+1]['_close'] > hi).any()
        cands.append(('bear', ridx, lo, hi, hi-lo, dist, idx-ridx, inv))
    if not cands:
        return {'nearest_fvg_dir':'none','nearest_fvg_dist':np.nan,'nearest_fvg_size':np.nan,'nearest_fvg_age_bars':np.nan,'entry_inside_nearest_fvg':False,'ifvg_proxy':False,'fvg_alignment_with_side':False}
    d, ridx, lo, hi, size, dist, age, inv = sorted(cands, key=lambda x: (x[5], x[6]))[0]
    align = (side == 'long' and d == 'bull') or (side == 'short' and d == 'bear')
    return {
        'nearest_fvg_dir':d,
        'nearest_fvg_low':lo,
        'nearest_fvg_high':hi,
        'nearest_fvg_mid':(lo+hi)/2,
        'nearest_fvg_dist':dist,
        'nearest_fvg_size':size,
        'nearest_fvg_age_bars':age,
        'entry_inside_nearest_fvg': bool(lo <= entry_price <= hi),
        'ifvg_proxy': bool(inv),
        'fvg_alignment_with_side': bool(align),
    }

def dol_features(df, idx, entry_price, side):
    dol = df.at[idx, '_dol']
    out = {'dol':dol,'dol_distance':np.nan,'dol_relation':'unknown','dol_cross_last_10m':False,'dol_reclaim_last_10m':False,'dol_reject_last_10m':False}
    if pd.isna(dol): return out
    out['dol_distance'] = float(entry_price-dol)
    out['dol_relation'] = 'near_dol' if abs(entry_price-dol)<=DOL_NEAR_POINTS else ('above_dol' if entry_price>dol else 'below_dol')
    look = df.iloc[max(0, idx-10):idx+1]
    if len(look) >= 2:
        s0 = np.sign(look['_close'].iloc[:-1]-dol)
        s1 = np.sign(look['_close'].iloc[1:]-dol)
        out['dol_cross_last_10m'] = bool((s0.values*s1.values < 0).any())
        if side == 'long':
            out['dol_reclaim_last_10m'] = bool(look['_low'].min() < dol and look['_close'].iloc[-1] > dol)
            out['dol_reject_last_10m'] = bool(look['_high'].max() > dol and look['_close'].iloc[-1] < dol)
        elif side == 'short':
            out['dol_reclaim_last_10m'] = bool(look['_high'].max() > dol and look['_close'].iloc[-1] < dol)
            out['dol_reject_last_10m'] = bool(look['_low'].min() < dol and look['_close'].iloc[-1] > dol)
    return out

def sweep_features(df, idx):
    look = df.iloc[max(0, idx-30):idx+1]
    levels = {'pdh':'high','pdl':'low','premarket_high':'high','premarket_low':'low','asia_high':'high','asia_low':'low','london_high':'high','london_low':'low'}
    swept=[]
    for name, typ in levels.items():
        lvl = df.at[idx, '_'+name]
        if pd.isna(lvl): continue
        if typ == 'high' and look['_high'].max() > lvl and look['_close'].iloc[-1] < lvl: swept.append(name)
        if typ == 'low' and look['_low'].min() < lvl and look['_close'].iloc[-1] > lvl: swept.append(name)
    return {'swept_levels_last_30m':'|'.join(swept),'sweep_count_last_30m':len(swept),'any_key_level_sweep_last_30m':len(swept)>0}

def displacement_vshape(df, idx, side):
    look10 = df.iloc[max(0, idx-10):idx+1]
    atr = df.at[idx,'_atr']
    if pd.isna(atr) or atr <= 0: atr = df.at[idx,'_atr_proxy_20']
    body_avg = df.at[idx,'_body_avg_20']
    ratio = float(df.at[idx,'_body']/body_avg) if body_avg and not pd.isna(body_avg) and body_avg>0 else np.nan
    disp = bool(ratio >= 1.75 and df.at[idx,'_body_pct'] >= 0.55)
    c = df.at[idx,'_close']
    first_open = look10['_open'].iloc[0]
    if side == 'long':
        pre_drive = look10['_low'].min() - first_open
        reclaim = c - look10['_low'].min()
        ok = pre_drive < 0 and reclaim > 0
        exhaust = pre_drive < 0 and reclaim < abs(pre_drive)*0.35
    elif side == 'short':
        pre_drive = look10['_high'].max() - first_open
        reclaim = look10['_high'].max() - c
        ok = pre_drive > 0 and reclaim > 0
        exhaust = pre_drive > 0 and reclaim < abs(pre_drive)*0.35
    else:
        pre_drive = reclaim = np.nan; ok = exhaust = False
    raw = abs(pre_drive) + reclaim if ok else reclaim
    atr_score = float(raw/atr) if atr and not pd.isna(atr) and atr>0 else np.nan
    return {'displacement_ratio':ratio,'displacement_candle':disp,'pre_10m_drive_points':float(pre_drive),'post_10m_reclaim_points':float(reclaim),'v_shape_score_points':float(raw),'v_shape_score_atr':atr_score,'v_shape_candidate':bool(ok and atr_score>=1.25),'exhaustion_proxy':bool(exhaust)}

def mfe_mae(df, idx, entry_price, side, mins):
    fwd = df.iloc[idx:min(len(df), idx+mins+1)]
    if side == 'long': return float(fwd['_high'].max()-entry_price), float(fwd['_low'].min()-entry_price)
    if side == 'short': return float(entry_price-fwd['_low'].min()), float(entry_price-fwd['_high'].max())
    return np.nan, np.nan

def context(df, idx, entry_price):
    r = df.iloc[idx]
    look = df.iloc[max(0, idx-30):idx+1]
    vwap = r['_vwap']
    rng = look['_high'].max()-look['_low'].min()
    return {
        'bar_open':r['_open'],'bar_high':r['_high'],'bar_low':r['_low'],'bar_close':r['_close'],'bar_range':r['_range'],'bar_body_pct':r['_body_pct'],
        'atr':r['_atr'],'atr_proxy_20':r['_atr_proxy_20'],'vwap':vwap,
        'vwap_distance':float(entry_price-vwap) if not pd.isna(vwap) else np.nan,
        'vwap_relation':'unknown' if pd.isna(vwap) else ('above_vwap' if entry_price>vwap else 'below_vwap'),
        'range_30m':float(rng) if rng>0 else np.nan,
        'entry_location_in_30m_range':float((entry_price-look['_low'].min())/rng) if rng>0 else np.nan,
    }

def analyze_one(tr, nq):
    out = tr.to_dict()
    if pd.isna(tr['entry_dt']) or pd.isna(tr['entry_price']):
        out['match_status']='bad_trade_input'; return out
    day_df = nq[nq['_date'] == tr['entry_dt'].date()]
    if day_df.empty:
        out['match_status']='no_market_rows_for_date'; return out
    idx, match = nearest_match(day_df, tr['entry_dt'], tr['entry_price'])
    out.update(match)
    if idx is None: return out
    side = tr['side']; px = tr['entry_price']
    out.update(context(nq, idx, px))
    out.update(dol_features(nq, idx, px, side))
    out.update(fvg_features(nq, idx, px, side))
    out.update(sweep_features(nq, idx))
    out.update(displacement_vshape(nq, idx, side))
    for m in [5,10,15,30,60,90]:
        mfe, mae = mfe_mae(nq, idx, px, side, m)
        out[f'mfe_{m}m']=mfe; out[f'mae_{m}m']=mae
    return out

def summarize(feat):
    m = feat[feat['match_status'].eq('matched')].copy()
    rows=[]
    bools=['displacement_candle','v_shape_candidate','exhaustion_proxy','entry_inside_nearest_fvg','ifvg_proxy','fvg_alignment_with_side','dol_cross_last_10m','dol_reclaim_last_10m','any_key_level_sweep_last_30m']
    for result,g in m.groupby('result', dropna=False):
        r={'group':f'result={result}','trades':len(g)}
        for c in ['mfe_30m','mae_30m','mfe_60m','mae_60m','nearest_fvg_dist','v_shape_score_atr','displacement_ratio','dol_distance','vwap_distance']:
            if c in g: r['avg_'+c]=g[c].mean(); r['med_'+c]=g[c].median()
        for c in bools:
            if c in g: r['rate_'+c]=g[c].fillna(False).mean()
        rows.append(r)
    return pd.DataFrame(rows)

def main():
    print(f'Loading trades: {TRADES_PATH}')
    trades = load_trades()
    print(f'Trades loaded: {len(trades)}')
    print(f'Loading NQ data: {NQ_PATH}')
    nq, cmap = load_nq()
    print(f'NQ rows loaded: {len(nq):,}')
    print(f'NQ range: {nq.dt.min()} -> {nq.dt.max()}')
    print('\nColumn mapping:')
    for k,v in cmap.items(): print(f'  {k}: {v}')
    feat = pd.DataFrame([analyze_one(tr, nq) for _,tr in trades.iterrows()])
    feat.to_csv(OUT_FEATURES, index=False)
    review_cols = [c for c in ['trade_id','date','entry_time','entry_price','side','result','match_status','matched_dt','price_inside_bar','price_distance','time_distance_min','dol_relation','dol_distance','dol_cross_last_10m','dol_reclaim_last_10m','vwap_relation','vwap_distance','nearest_fvg_dir','nearest_fvg_dist','nearest_fvg_size','nearest_fvg_age_bars','entry_inside_nearest_fvg','ifvg_proxy','fvg_alignment_with_side','any_key_level_sweep_last_30m','swept_levels_last_30m','displacement_ratio','displacement_candle','pre_10m_drive_points','post_10m_reclaim_points','v_shape_score_atr','v_shape_candidate','exhaustion_proxy','mfe_10m','mae_10m','mfe_30m','mae_30m','mfe_60m','mae_60m'] if c in feat.columns]
    review = feat[review_cols]
    review.to_csv(OUT_REVIEW, index=False)
    summary = summarize(feat)
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f'\nSaved enhanced features: {OUT_FEATURES}')
    print(f'Saved enhanced review:   {OUT_REVIEW}')
    print(f'Saved enhanced summary:  {OUT_SUMMARY}')
    print('\nMatch summary:')
    print(feat['match_status'].value_counts(dropna=False).to_string())
    print('\nEnhanced summary:')
    if len(summary):
        show = [c for c in summary.columns if c in ['group','trades','avg_mfe_30m','avg_mae_30m','rate_v_shape_candidate','rate_entry_inside_nearest_fvg','rate_ifvg_proxy','rate_fvg_alignment_with_side','rate_dol_reclaim_last_10m','rate_any_key_level_sweep_last_30m','rate_displacement_candle','avg_nearest_fvg_dist','avg_v_shape_score_atr']]
        print(summary[show].to_string(index=False))
    print('\nReview rows:')
    print(review.to_string(index=False))

if __name__ == '__main__':
    main()
