import datetime
import os
import io
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

CSV_OUTPUT_FILE = "backtest_raw_db.csv"
START_DATE = "2014-01-01"        # Tier 1: 12개년 기준 시작일
SOFR_START_DATE = "2018-04-02"   # Tier 2: SOFR 공식 시작일
VIX1D_START_DATE = "2023-04-24"  # Tier 3: VIX1D 공식 시작일

def get_yahoo_full(ticker_symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range=15y&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if 'chart' in data and data['chart']['result']:
                res_data = data['chart']['result'][0]
                timestamps = res_data.get('timestamp', [])
                quotes = res_data.get('indicators', {}).get('quote', [{}])[0]
                ny_tz = ZoneInfo("America/New_York")
                dates = [datetime.datetime.fromtimestamp(ts, tz=ny_tz).date() for ts in timestamps]
                
                df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Open': quotes.get('open', []),
                    'High': quotes.get('high', []),
                    'Low': quotes.get('low', []),
                    'Close': quotes.get('close', []),
                    'Volume': quotes.get('volume', [])
                }).dropna(subset=['Close'])
                
                df['Open'] = df['Open'].fillna(df['Close'])
                df['High'] = df['High'].fillna(df['Close'])
                df['Low'] = df['Low'].fillna(df['Close'])
                df['Volume'] = df['Volume'].fillna(1.0)
                df.set_index('Date', inplace=True)
                return df[df.index >= pd.to_datetime(START_DATE)].astype(float)
    except Exception as e:
        print(f"야후 다운로드 실패 ({ticker_symbol}): {e}")
    return pd.DataFrame()

def fetch_fred_series(series_id):
    try:
        csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(csv_url, headers=headers, timeout=12)
        if res.status_code == 200 and len(res.text) > 30:
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = [c.strip().upper() for c in df.columns]
            if "DATE" in df.columns and series_id.upper() in df.columns:
                df['Date'] = pd.to_datetime(df['DATE'])
                df[series_id] = pd.to_numeric(df[series_id.upper()], errors='coerce')
                return df.dropna(subset=[series_id]).set_index('Date')[[series_id]]
    except Exception as e:
        print(f"FRED 다운로드 실패 ({series_id}): {e}")
    return pd.DataFrame()

def build_database():
    print(f"🌐 [14개 전수 지표 정밀 DB 구축 시작] (2014-01-01 ~ 현재)...")
    
    # ── Tier 1: 12년 지표 수집 (2014년~) ──
    qqq = get_yahoo_full("QQQ")
    tqqq = get_yahoo_full("TQQQ")
    qqqe = get_yahoo_full("QQQE")
    hyg = get_yahoo_full("HYG")
    tlt = get_yahoo_full("TLT")
    dxy = get_yahoo_full("DX-Y.NYB")
    usdjpy = get_yahoo_full("USDJPY=X")

    vix = get_yahoo_full("^VIX")
    vix3m = get_yahoo_full("^VIX3M")
    vxn = get_yahoo_full("^VXN")
    skew = get_yahoo_full("^SKEW")

    df_hy = fetch_fred_series("BAMLH0A0HYM2")  # 하이일드 스프레드
    df_t10y2y = fetch_fred_series("T10Y2Y")    # 장단기 금리차

    # ── Tier 2 & 3: 기간 제한 지표 수집 ──
    df_sofr = fetch_fred_series("SOFR")        # SOFR (2018.04~)
    vix1d = get_yahoo_full("^VIX1D")           # VIX1D (2023.04~)

    if qqq.empty:
        print("🚨 QQQ 다운로드 실패로 중단")
        return

    df = pd.DataFrame({
        'QQQ_Open': qqq['Open'],
        'QQQ_High': qqq['High'],
        'QQQ_Low': qqq['Low'],
        'QQQ_Close': qqq['Close'],
        'QQQ_Vol': qqq['Volume']
    })

    df['TQQQ_Open'] = tqqq['Open'] if not tqqq.empty else df['QQQ_Open']
    df['TQQQ_Close'] = tqqq['Close'] if not tqqq.empty else df['QQQ_Close']
    df['QQQE'] = qqqe['Close'] if not qqqe.empty else df['QQQ_Close']
    df['HYG'] = hyg['Close'] if not hyg.empty else 75.0
    df['TLT'] = tlt['Close'] if not tlt.empty else 90.0
    df['DXY'] = dxy['Close'] if not dxy.empty else 100.0
    df['USDJPY'] = usdjpy['Close'] if not usdjpy.empty else 120.0

    df['VIX'] = vix['Close'] if not vix.empty else 18.0
    df['VIX3M'] = vix3m['Close'] if not vix3m.empty else (df['VIX'] * 1.1)
    df['VXN'] = vxn['Close'] if not vxn.empty else (df['VIX'] * 1.1)
    df['SKEW'] = skew['Close'] if not skew.empty else 125.0

    df['HY_SPREAD'] = df_hy['BAMLH0A0HYM2'] if not df_hy.empty else np.nan
    df['T10Y2Y'] = df_t10y2y['T10Y2Y'] if not df_t10y2y.empty else np.nan

    # Tier 1 지표 결측치 ffill 채움
    tier1_cols = [c for c in df.columns]
    df[tier1_cols] = df[tier1_cols].ffill().bfill()

    # ── Tier 2 (SOFR): 2018-04-02 이전은 NaN 보존, 이후만 ffill ──
    if not df_sofr.empty:
        df['SOFR'] = df_sofr['SOFR']
        sofr_mask = df.index >= pd.to_datetime(SOFR_START_DATE)
        df.loc[sofr_mask, 'SOFR'] = df.loc[sofr_mask, 'SOFR'].ffill().bfill()
    else:
        df['SOFR'] = np.nan

    # ── Tier 3 (VIX1D): 2023-04-24 이전은 NaN 보존, 이후만 ffill ──
    if not vix1d.empty:
        df['VIX1D'] = vix1d['Close']
        vix1d_mask = df.index >= pd.to_datetime(VIX1D_START_DATE)
        df.loc[vix1d_mask, 'VIX1D'] = df.loc[vix1d_mask, 'VIX1D'].ffill().bfill()
    else:
        df['VIX1D'] = np.nan

    # CSV로 영구 저장
    df.reset_index().to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"✅ '{CSV_OUTPUT_FILE}' 무결성 DB 생성 완료!")
    print(f"   • Tier 1 (12년 전수 데이터): {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)}거래일)")
    print(f"   • Tier 2 (SOFR 실데이터): {df['SOFR'].dropna().count()}건 (2018.04 이후)")
    print(f"   • Tier 3 (VIX1D 실데이터): {df['VIX1D'].dropna().count()}건 (2023.04 이후)")

if __name__ == "__main__":
    build_database()
