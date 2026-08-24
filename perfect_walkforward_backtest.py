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
    
    # 1차: HTML 모드로 시도
    payload = {"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML"}
    try:
        res = requests.post(url, data=payload, timeout=15)
        if res.status_code == 200:
            print("✅ 텔레그램 전송 성공 (HTML)!")
            return
        else:
            print(f"⚠️ HTML 전송 실패 ({res.status_code}): {res.text}, 순수 텍스트로 재전송...")
    except Exception as e:
        print(f"통신 에러: {e}")

    # 2차: HTML 태그 제거 후 순수 텍스트로 안전 전송
    plain_text = text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
    payload_plain = {"chat_id": chat_id, "text": plain_text[:4000]}
    try:
        res2 = requests.post(url, data=payload_plain, timeout=15)
        if res2.status_code == 200:
            print("✅ 텔레그램 전송 성공 (Plain Text)!")
        else:
            print(f"🚨 최종 전송 실패 ({res2.status_code}): {res2.text}")
    except Exception as e:
        print(f"최종 통신 실패: {e}")

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

def load_or_build_db():
    if os.path.exists(RAW_DB_FILE):
        try:
            df = pd.read_csv(RAW_DB_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            if len(df) > 2000:
                print(f"📦 [로컬 DB 로드 완료] 총 {len(df)}거래일")
                return df
        except Exception:
            pass

    print("🌐 [신규 DB 다운로드 중...]")
    qqq = get_historical_data("QQQ")
    vix = get_historical_data("^VIX")
    skew = get_historical_data("^SKEW")
    hyg = get_historical_data("HYG")
    tlt = get_historical_data("TLT")
    df_hy = fetch_fred_historical("BAMLH0A0HYM2")

    df = pd.DataFrame({
        'QQQ_Close': qqq['Close'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Open': qqq['Open'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_High': qqq['High'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Low': qqq['Low'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Vol': qqq['Volume'] if not qqq.empty else pd.Series(dtype=float)
    })
    
    df['VIX'] = vix['Close'] if not vix.empty else 18.0
    df['SKEW'] = skew['Close'] if not skew.empty else 125.0
    df['HYG'] = hyg['Close'] if not hyg.empty else 75.0
    df['TLT'] = tlt['Close'] if not tlt.empty else 90.0
    df['HY_SPREAD'] = df_hy['BAMLH0A0HYM2'] if not df_hy.empty else 3.5

    df = df.ffill().bfill().dropna()
    df.to_csv(RAW_DB_FILE)
    return df

def run_pure_crash_analysis():
    df = load_or_build_db()
    if df.empty or len(df) < 500:
        print("🚨 데이터 부족으로 중단")
        return

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

    peaks_detected = []
    rolling_peak = qqq_c.rolling(60).max()

    for i in range(60, len(df) - 60):
        d_curr = df.index[i]
        curr_p = float(qqq_c.iloc[i])
        peak_60d = float(rolling_peak.iloc[i])
        curr_drawdown = ((curr_p / peak_60d) - 1) * 100

        # 고점 대비 -5.0% 이상 빠진 바닥 투매는 기각
        if curr_drawdown < -5.0:
            continue

        # 경로 A: 대형 버블 정점 균열
        cond_bubble = (
            (disp200.iloc[i] >= 107.0 or disp20.iloc[i] >= 105.0) and
            (float(df['VIX'].iloc[i]) >= 20.5 or float(df['SKEW'].iloc[i]) >= 142.0) and
            (curr_p < float(sma20.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # 경로 B: 매크로/크레딧 신용 발작
        hy_chg_20d = float(df['HY_SPREAD'].iloc[i]) - float(df['HY_SPREAD'].iloc[i-20])
        r_now = float(df['HYG'].iloc[i]) / float(df['TLT'].iloc[i])
        r_prev = float(df['HYG'].iloc[i-20]) / float(df['TLT'].iloc[i-20])
        hyg_tlt_drop = ((r_now / r_prev) - 1) * 100

        cond_credit = (
            (hy_chg_20d >= 0.25 or hyg_tlt_drop <= -2.5) and
            (curr_p < float(sma20.iloc[i])) and
            (curr_p < float(sma5.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # 경로 C: 추세 전환형 50일선 초기 붕괴
        cond_initial = (
            (curr_p < float(sma50.iloc[i])) and
            (curr_p < float(sma20.iloc[i])) and
            (float(df['VIX'].iloc[i]) >= 19.0) and
            (curr_drawdown >= -4.5)
        )

        trigger_reason = ""
        if cond_bubble: trigger_reason = "버블정점균열"
        elif cond_credit: trigger_reason = "크레딧발작"
        elif cond_initial: trigger_reason = "어깨50일선붕괴"

        if trigger_reason:
            if not peaks_detected or (d_curr - peaks_detected[-1]['date']).days > 35:
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
    drop_over_8 = (peak_df['max_drop_60d'] <= -8.0).sum()
    total_signals = len(peak_df)
    
    hit_ratio_10 = (drop_over_10 / total_signals * 100) if total_signals > 0 else 0
    hit_ratio_8 = (drop_over_8 / total_signals * 100) if total_signals > 0 else 0

    msg = (
        f"🎯 <b>[-10% 이상 대형 폭락 전용 초정밀 고점 검증]</b>\n"
        f"📅 기간: {df.index[60].strftime('%Y-%m-%d')} ~ {df.index[-60].strftime('%Y-%m-%d')} ({len(df)}거래일)\n"
        f"────────────────\n"
        f"• <b>총 감지된 고점 신호:</b> <b>{total_signals}회</b>\n"
        f"• <b>🚨 -10% 이상 대형 폭락 적중:</b> <b>{drop_over_10}회</b>\n"
        f"• <b>⚠️ -8% 이상 준대형 폭락 포함:</b> <b>{drop_over_8}회</b>\n"
        f"• <b>대형 폭락(-10%) 순수 적중률:</b> <b>{hit_ratio_10:.1f}%</b>\n"
        f"• <b>실질 방어 성공률(-8%이상):</b> <b>{hit_ratio_8:.1f}%</b>\n"
        f"• <b>신호 발생 후 실제 평균 낙폭:</b> <b>{peak_df['max_drop_60d'].mean():.2f}%</b>\n"
        f"────────────────\n"
        f"<b>[감지된 대형 폭락 고점 전수 로그]</b>\n"
    )
    for _, r in peak_df.iterrows():
        status = "🚨 -10% 이상 폭락" if r['max_drop_60d'] <= -10.0 else ("⚠️ -8%대 준폭락" if r['max_drop_60d'] <= -8.0 else "❌ 노이즈")
        msg += f"• <b>{r['date'].strftime('%Y-%m-%d')}</b> (${r['price']:.1f}) [{r['reason']}] :: 낙폭 {r['max_drop_60d']}% {status}\n"

    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_pure_crash_analysis()
