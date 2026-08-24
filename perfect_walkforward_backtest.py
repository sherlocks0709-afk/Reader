import datetime
import os
import io
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

RAW_DB_FILE = "longterm_peak_db.csv"

def send_telegram_result(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("텔레그램 토큰 미설정으로 콘솔에만 출력합니다.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

def get_historical_data(ticker_symbol, start_date="2012-01-01"):
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
                return df[df.index >= pd.to_datetime(start_date)].astype(float)
    except Exception as e:
        print(f"데이터 수집 에러 ({ticker_symbol}): {e}")
    return pd.DataFrame()

def fetch_fred_historical(series_id, api_key="", start_date="2012-01-01"):
    try:
        csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(csv_url, headers=headers, timeout=10)
        if res.status_code == 200 and len(res.text) > 30:
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = [c.strip().upper() for c in df.columns]
            if "DATE" in df.columns and series_id.upper() in df.columns:
                df['Date'] = pd.to_datetime(df['DATE'])
                df[series_id] = pd.to_numeric(df[series_id.upper()], errors='coerce')
                return df.dropna(subset=[series_id]).set_index('Date')[[series_id]]
    except Exception:
        pass
    return pd.DataFrame()

def load_or_build_longterm_db():
    if os.path.exists(RAW_DB_FILE):
        try:
            print(f"📦 [장기 DB 로드] '{RAW_DB_FILE}'에서 14개년 시계열 데이터를 불러옵니다.")
            df = pd.read_csv(RAW_DB_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            if len(df) > 2000:
                return df
        except Exception as e:
            print(f"DB 로드 실패, 재생성: {e}")

    print("🌐 [신규 14년 DB 구축] 2012년 이후 전수 데이터를 다운로드하여 DB를 생성합니다...")
    qqq = get_historical_data("QQQ")
    vix = get_historical_data("^VIX")
    vix3m = get_historical_data("^VIX3M")
    vxn = get_historical_data("^VXN")
    skew = get_historical_data("^SKEW")
    hyg = get_historical_data("HYG")
    tlt = get_historical_data("TLT")
    qqqe = get_historical_data("QQQE")
    dxy = get_historical_data("DX-Y.NYB")
    usdjpy = get_historical_data("USDJPY=X")

    fred_key = os.environ.get("FRED_API_KEY", "")
    df_hy = fetch_fred_historical("BAMLH0A0HYM2", fred_key)

    df = pd.DataFrame({
        'QQQ_Close': qqq['Close'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Open': qqq['Open'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_High': qqq['High'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Low': qqq['Low'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Vol': qqq['Volume'] if not qqq.empty else pd.Series(dtype=float)
    })
    
    df['VIX'] = vix['Close'] if not vix.empty else 18.0
    df['VIX3M'] = vix3m['Close'] if not vix3m.empty else (df['VIX'] * 1.1)
    df['VXN'] = vxn['Close'] if not vxn.empty else (df['VIX'] * 1.1)
    df['SKEW'] = skew['Close'] if not skew.empty else 125.0
    df['HYG'] = hyg['Close'] if not hyg.empty else 75.0
    df['TLT'] = tlt['Close'] if not tlt.empty else 90.0
    df['QQQE'] = qqqe['Close'] if not qqqe.empty else df['QQQ_Close']
    df['DXY'] = dxy['Close'] if not dxy.empty else 100.0
    df['USDJPY'] = usdjpy['Close'] if not usdjpy.empty else 120.0
    df['HY_SPREAD'] = df_hy['BAMLH0A0HYM2'] if not df_hy.empty else 3.5

    df = df.ffill().bfill().dropna()
    df.to_csv(RAW_DB_FILE)
    print(f"✅ '{RAW_DB_FILE}' 생성 완료 ({len(df)}거래일)")
    return df

def analyze_historical_major_peaks():
    df = load_or_build_longterm_db()
    print(f"📊 [검증 시작] 총 {len(df)}거래일 (2012 ~ 2026) 3중 필터 고점 검증 가동...")

    qqq_c = df['QQQ_Close']
    sma5 = qqq_c.rolling(5).mean()
    sma20 = qqq_c.rolling(20).mean()
    sma50 = qqq_c.rolling(50).mean()
    sma200 = qqq_c.rolling(200, min_periods=20).mean()
    disp200 = (qqq_c / sma200) * 100
    disp20 = (qqq_c / sma20) * 100

    ema12 = qqq_c.ewm(span=12, adjust=False).mean()
    ema26 = qqq_c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    vix_ratio = df['VIX'] / df['VIX3M'] # 14년 기간구조 지표

    peaks_detected = []

    for i in range(50, len(df) - 30):
        d_curr = df.index[i]
        curr_price = float(qqq_c.iloc[i])

        # Level 1: 에너지 과열
        l1_score = 0
        if disp200.iloc[i] >= 110.0: l1_score += 10
        elif disp200.iloc[i] >= 106.0: l1_score += 5
        if disp20.iloc[i] >= 106.0: l1_score += 10
        qqq_20d = (curr_price / float(qqq_c.iloc[i-20]) - 1) * 100
        qqqe_20d = (float(df['QQQE'].iloc[i]) / float(df['QQQE'].iloc[i-20]) - 1) * 100
        if qqq_20d > 0 and (qqq_20d - qqqe_20d) >= 3.0: l1_score += 10

        # Level 2: 꼬리위험 / 기간구조 역전 / 크레딧 경색
        l2_score = 0
        if float(df['SKEW'].iloc[i]) >= 142.0: l2_score += 15
        elif float(df['SKEW'].iloc[i]) >= 135.0: l2_score += 7
        if float(vix_ratio.iloc[i]) >= 1.05: l2_score += 15 # 역전 경보
        elif float(vix_ratio.iloc[i]) >= 0.95: l2_score += 7
        if (float(df['HY_SPREAD'].iloc[i]) - float(df['HY_SPREAD'].iloc[i-20])) >= 0.25: l2_score += 10

        # Level 3: 가격 방아쇠
        l3_score = 0
        if curr_price < float(sma5.iloc[i]): l3_score += 10
        if float(macd.iloc[i]) < float(signal.iloc[i]): l3_score += 10
        if curr_price < float(sma20.iloc[i]): l3_score += 10

        is_peak = (l1_score >= 15) and (l2_score >= 20) and (l3_score >= 20)

        if is_peak:
            if not peaks_detected or (d_curr - peaks_detected[-1]['date']).days > 30:
                forward_window = qqq_c.iloc[i:i+60] # 이후 60거래일(3개월) 동안의 최대 낙폭 추적
                max_drawdown = ((forward_window.min() / curr_price) - 1) * 100
                peaks_detected.append({
                    'date': d_curr,
                    'price': curr_price,
                    'max_drop_60d': round(max_drawdown, 2)
                })

    peak_df = pd.DataFrame(peaks_detected)
    drop_over_10 = (peak_df['max_drop_60d'] <= -10.0).sum()
    drop_over_5 = (peak_df['max_drop_60d'] <= -5.0).sum()

    msg = (
        f"🏛️ <b>[14개년(2012~2026) -10% 이상 대형 고점 판독 검증 리포트]</b>\n"
        f"📅 검증 기간: {df.index[50].strftime('%Y-%m-%d')} ~ {df.index[-30].strftime('%Y-%m-%d')} ({len(df)}거래일)\n"
        f"📁 DB 상태: <code>{RAW_DB_FILE}</code>\n"
        f"────────────────\n"
        f"• <b>총 감지된 고점 신호:</b> <b>{len(peak_df)}회</b>\n"
        f"• <b>실제 -10% 이상 대형 폭락 적중:</b> <b>{drop_over_10}회</b>\n"
        f"• <b>실제 -5% 이상 조정 포함 적중:</b> <b>{drop_over_5}회</b>\n"
        f"• <b>대형 폭락 적중률:</b> <b>{(drop_over_10 / len(peak_df) * 100):.1f}%</b>\n"
        f"────────────────\n"
        f"<b>[역사적 주요 감지 시점 로그]</b>\n"
    )
    for _, r in peak_df.iterrows():
        status = "🚨 -10% 이상 대형폭락" if r['max_drop_60d'] <= -10.0 else "⚠️ -5%대 일반조정"
        msg += f"• <b>{r['date'].strftime('%Y-%m-%d')}</b>: ${r['price']:.2f} (최대 낙폭: <b>{r['max_drop_60d']}%</b>) {status}\n"

    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    analyze_historical_major_peaks()
