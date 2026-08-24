import datetime
import os
import io
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

def send_telegram_result(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("텔레그램 토큰 미설정으로 콘솔에만 출력합니다.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML"}
    try:
        res = requests.post(url, data=payload, timeout=15)
        if res.status_code == 200:
            print("✅ 텔레그램 전송 성공!")
        else:
            print(f"🚨 텔레그램 전송 에러 ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"텔레그램 통신 실패: {e}")

def get_historical_data(ticker_symbol):
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
                return df.astype(float)
    except Exception as e:
        print(f"데이터 수집 에러 ({ticker_symbol}): {e}")
    return pd.DataFrame()

def fetch_fred_historical(series_id):
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

def run_analysis():
    print("🌐 [14년 시계열 데이터 수집 중...]")
    qqq = get_historical_data("QQQ")
    vix = get_historical_data("^VIX")
    skew = get_historical_data("^SKEW")
    hyg = get_historical_data("HYG")
    tlt = get_historical_data("TLT")
    df_hy = fetch_fred_historical("BAMLH0A0HYM2")

    if qqq.empty:
        print("🚨 QQQ 데이터 수집 실패로 중단")
        return

    df = pd.DataFrame({
        'QQQ_Close': qqq['Close'], 'QQQ_Open': qqq['Open'],
        'QQQ_High': qqq['High'], 'QQQ_Low': qqq['Low'], 'QQQ_Vol': qqq['Volume']
    })
    
    df['VIX'] = vix['Close'] if not vix.empty else 18.0
    df['SKEW'] = skew['Close'] if not skew.empty else 125.0
    df['HYG'] = hyg['Close'] if not hyg.empty else 75.0
    df['TLT'] = tlt['Close'] if not tlt.empty else 90.0
    df['HY_SPREAD'] = df_hy['BAMLH0A0HYM2'] if not df_hy.empty else 3.5

    df = df.ffill().bfill().dropna()
    print(f"📊 총 {len(df)}거래일 데이터 준비 완료. 다중 경로 고점 감지 시작...")

    qqq_c = df['QQQ_Close']
    qqq_v = df['QQQ_Vol']
    
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

    vol_20avg = qqq_v.rolling(20).mean()
    vol_ratio = qqq_v / vol_20avg

    peaks_detected = []

    for i in range(50, len(df) - 60):
        d_curr = df.index[i]
        curr_p = float(qqq_c.iloc[i])

        # 1. 클라이맥스 버블형
        cond_bubble = (
            (disp200.iloc[i] >= 106.0 or disp20.iloc[i] >= 105.0) and
            (float(df['SKEW'].iloc[i]) >= 135.0 or float(df['VIX'].iloc[i]) >= 18.5) and
            (curr_p < float(sma5.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # 2. 크레딧/스프레드 균열형
        hy_chg_20d = float(df['HY_SPREAD'].iloc[i]) - float(df['HY_SPREAD'].iloc[i-20])
        r_now = float(df['HYG'].iloc[i]) / float(df['TLT'].iloc[i])
        r_prev = float(df['HYG'].iloc[i-20]) / float(df['TLT'].iloc[i-20])
        hyg_tlt_drop = ((r_now / r_prev) - 1) * 100

        cond_credit = (
            (hy_chg_20d >= 0.15 or hyg_tlt_drop <= -1.8) and
            (curr_p < float(sma20.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # 3. 50일선 붕괴 모멘텀 이탈형
        cond_breakdown = (
            (curr_p < float(sma50.iloc[i])) and
            (curr_p < float(sma20.iloc[i])) and
            (vol_ratio.iloc[i] >= 1.10)
        )

        trigger_reason = ""
        if cond_bubble: trigger_reason = "버블과열"
        elif cond_credit: trigger_reason = "크레딧경색"
        elif cond_breakdown: trigger_reason = "50일선붕괴"

        if trigger_reason:
            if not peaks_detected or (d_curr - peaks_detected[-1]['date']).days > 25:
                forward_window = qqq_c.iloc[i:i+60]
                max_drawdown = ((forward_window.min() / curr_p) - 1) * 100
                peaks_detected.append({
                    'date': d_curr,
                    'price': curr_p,
                    'reason': trigger_reason,
                    'max_drop_60d': round(max_drawdown, 2)
                })

    peak_df = pd.DataFrame(peaks_detected)
    drop_over_10 = (peak_df['max_drop_60d'] <= -10.0).sum()
    drop_over_5 = (peak_df['max_drop_60d'] <= -5.0).sum()

    msg = (
        f"🏛️ <b>[14개년 다중 경로 고점 판독 전수 검증]</b>\n"
        f"📅 기간: {df.index[50].strftime('%Y-%m-%d')} ~ {df.index[-60].strftime('%Y-%m-%d')} ({len(df)}거래일)\n"
        f"────────────────\n"
        f"• <b>총 감지된 고점 신호:</b> <b>{len(peak_df)}회</b>\n"
        f"• <b>🚨 -10% 이상 대형 폭락 적중:</b> <b>{drop_over_10}회</b>\n"
        f"• <b>⚠️ -5% 이상 일반 조정 적중:</b> <b>{drop_over_5}회</b>\n"
        f"• <b>신호 후 실제 평균 낙폭:</b> <b>{peak_df['max_drop_60d'].mean():.2f}%</b>\n"
        f"────────────────\n"
        f"<b>[역사적 주요 대형 폭락(-10% 이상) 적중 로그]</b>\n"
    )
    
    # -10% 이상 대형 폭락 건만 압축 출력 (메시지 길이 방어)
    major_hits = peak_df[peak_df['max_drop_60d'] <= -10.0]
    for _, r in major_hits.iterrows():
        msg += f"• <b>{r['date'].strftime('%Y-%m-%d')}</b> (${r['price']:.1f}) [{r['reason']}] ➔ 낙폭: <b>{r['max_drop_60d']}%</b>\n"

    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_analysis()
