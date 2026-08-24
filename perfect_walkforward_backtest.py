import datetime
import os
import requests
import pandas as pd
import numpy as np

RAW_DB_FILE = "backtest_raw_db.csv"

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
            print("✅ 텔레그램 전송 성공 (HTML)!")
            return
    except Exception:
        pass

    plain_text = text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
    payload_plain = {"chat_id": chat_id, "text": plain_text[:4000]}
    try:
        requests.post(url, data=payload_plain, timeout=15)
        print("✅ 텔레그램 전송 성공 (Plain Text)!")
    except Exception as e:
        print(f"최종 실패: {e}")

def get_historical_data(ticker_symbol, start_date="2023-04-24"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range=4y&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if 'chart' in data and data['chart']['result']:
                res_data = data['chart']['result'][0]
                timestamps = res_data.get('timestamp', [])
                quotes = res_data.get('indicators', {}).get('quote', [{}])[0]
                dates = [datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).date() for ts in timestamps]
                
                df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Open': quotes.get('open', []),
                    'High': quotes.get('high', []),
                    'Low': quotes.get('low', []),
                    'Close': quotes.get('close', []),
                    'Volume': quotes.get('volume', [])
                }).dropna(subset=['Close'])
                df['Open'] = df['Open'].fillna(df['Close'])
                df.set_index('Date', inplace=True)
                return df[df.index >= pd.to_datetime(start_date)].astype(float)
    except Exception as e:
        print(f"다운로드 에러 ({ticker_symbol}): {e}")
    return pd.DataFrame()

def load_strictly_from_db():
    # 1. 로컬 DB 파일이 존재하면 즉시 로드 (네트워크 통신 0)
    if os.path.exists(RAW_DB_FILE):
        try:
            df = pd.read_csv(RAW_DB_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            if len(df) > 500 and 'VIX1D' in df.columns:
                print(f"📁 [로컬 DB 완전 캐시 로드] '{RAW_DB_FILE}' ({len(df)}거래일)")
                return df
        except Exception as e:
            print(f"DB 로드 실패, 1회 재생성: {e}")

    # 2. DB가 없을 때만 단 1회 생성하여 영구 보관
    print(f"⚠️ '{RAW_DB_FILE}'가 없어 1회 신규 생성합니다...")
    qqq = get_historical_data("QQQ")
    vix = get_historical_data("^VIX")
    vix1d = get_historical_data("^VIX1D")
    skew = get_historical_data("^SKEW")
    qqqe = get_historical_data("QQQE")

    df = pd.DataFrame({
        'QQQ_Close': qqq['Close'], 'QQQ_Open': qqq['Open'],
        'QQQ_High': qqq['High'], 'QQQ_Low': qqq['Low'], 'QQQ_Vol': qqq['Volume']
    })
    df['VIX'] = vix['Close'] if not vix.empty else 18.0
    df['VIX1D'] = vix1d['Close'] if not vix1d.empty else df['VIX']
    df['SKEW'] = skew['Close'] if not skew.empty else 125.0
    df['QQQE'] = qqqe['Close'] if not qqqe.empty else df['QQQ_Close']

    df = df.ffill().bfill().dropna()
    df.to_csv(RAW_DB_FILE)
    print(f"✅ '{RAW_DB_FILE}' 로컬 캐싱 완료 ({len(df)}거래일)")
    return df

def run_db_peak_escape_analysis():
    df = load_strictly_from_db()
    if df.empty or len(df) < 200:
        print("🚨 유효 데이터 부족으로 중단")
        return

    qqq_c = df['QQQ_Close']
    qqq_o = df['QQQ_Open']
    
    sma5 = qqq_c.rolling(5).mean()
    ema12 = qqq_c.ewm(span=12, adjust=False).mean()
    ema26 = qqq_c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    ratio_0dte = df['VIX1D'] / df['VIX']
    rolling_peak_20d = qqq_c.rolling(20).max()

    escape_logs = []

    for i in range(30, len(df) - 40):
        d_curr = df.index[i]
        curr_close = float(qqq_c.iloc[i])
        recent_peak = float(rolling_peak_20d.iloc[i])
        
        # 신호 당일 종가 기준 고점 대비 낙폭
        signal_drawdown = ((curr_close / recent_peak) - 1) * 100

        # 어깨 필터: 직전 20일 고점 대비 -4.0% 이내에서만 탈출 신호 발동
        if signal_drawdown < -4.0:
            continue

        cond_0dte = float(ratio_0dte.iloc[i]) >= 1.25
        cond_skew = float(df['SKEW'].iloc[i]) >= 142.0
        
        qqq_20d = (curr_close / float(qqq_c.iloc[i-20]) - 1) * 100
        qqqe_20d = (float(df['QQQE'].iloc[i]) / float(df['QQQE'].iloc[i-20]) - 1) * 100
        cond_divergence = (qqq_20d - qqqe_20d) >= 3.0

        cond_trigger = (curr_close < float(sma5.iloc[i])) and (float(macd.iloc[i]) < float(signal.iloc[i]))

        # 파생 경보 2개 이상 + 가격 추세 이탈
        if (sum([cond_0dte, cond_skew, cond_divergence]) >= 2) and cond_trigger:
            if not escape_logs or (d_curr - escape_logs[-1]['signal_date']).days > 20:
                # T+1일 시초가 체결(현실성 100% 무결 검증)
                exec_date = df.index[i+1]
                exec_price = float(qqq_o.iloc[i+1])
                
                # 직전 최고점 대비 실제 탈출 낙폭
                escape_drop = ((exec_price / recent_peak) - 1) * 100

                # 탈출 후 향후 40거래일 동안의 최저 바닥가 추적
                forward_window = qqq_c.iloc[i+1:i+41]
                trough_price = float(forward_window.min())
                total_crash_from_peak = ((trough_price / recent_peak) - 1) * 100
                protected_drop = ((trough_price / exec_price) - 1) * 100

                escape_logs.append({
                    'signal_date': d_curr,
                    'exec_date': exec_date,
                    'peak_price': round(recent_peak, 2),
                    'exec_price': round(exec_price, 2),
                    'escape_drop': round(escape_drop, 2),
                    'trough_price': round(trough_price, 2),
                    'total_crash': round(total_crash_from_peak, 2),
                    'protected_drop': round(protected_drop, 2)
                })

    res_df = pd.DataFrame(escape_logs)
    avg_escape_drop = res_df['escape_drop'].mean() if not res_df.empty else 0
    avg_protected = res_df['protected_drop'].mean() if not res_df.empty else 0

    msg = (
        f"📊 <b>[실데이터 DB 기반 고점 탈출 실측 백테스트]</b>\n"
        f"📁 DB 상태: <code>{RAW_DB_FILE}</code> ({len(df)}거래일 고정 DB)\n"
        f"📅 기간: {df.index[30].strftime('%Y-%m-%d')} ~ {df.index[-40].strftime('%Y-%m-%d')}\n"
        f"────────────────\n"
        f"• <b>총 탈출 신호 횟수:</b> <b>{len(res_df)}회</b>\n"
        f"• <b>직전 최고점 대비 평균 탈출 위치:</b> <b>{avg_escape_drop:.2f}%</b> (어깨 탈출)\n"
        f"• <b>탈출 후 바닥까지 추가 방어한 낙폭:</b> <b>평균 {avg_protected:.2f}%</b>\n"
        f"────────────────\n"
        f"<b>[실제 전수 탈출 및 방어 로그]</b>\n"
    )
    for _, r in res_df.iterrows():
        msg += (
            f"• <b>신호 {r['signal_date'].strftime('%m-%d')} ➔ 체결 {r['exec_date'].strftime('%m-%d')}</b>\n"
            f"  - 최고점: ${r['peak_price']} ➔ 체결가: ${r['exec_price']} (<b>{r['escape_drop']}%</b> 지점 탈출)\n"
            f"  - 이후 바닥: ${r['trough_price']} (전체 폭락: <b>{r['total_crash']}%</b>)\n"
            f"  - 🛡️ <b>현금화로 방어한 추가 하락:</b> <b>{r['protected_drop']}%</b>\n\n"
        )

    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_db_peak_escape_analysis()
