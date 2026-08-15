import datetime
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np

def calculate_risk_score():
    # QQQ 데이터 1년치 다운로드
    df = yf.download("QQQ", period="1y", interval="1d", progress=False)
    
    # 200일 이동평균 및 이격도
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    current_close = float(df['Close'].iloc[-1])
    sma200 = float(df['SMA200'].iloc[-1])
    disparity_200 = (current_close / sma200) * 100

    # 14일 RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_14 = float((100 - (100 / (1 + rs))).iloc[-1])

    # 위험도 스코어링 (100점 만점)
    disp_score = np.clip((disparity_200 - 100) * 2.5, 0, 50)
    rsi_score = np.clip((rsi_14 - 50) * (50 / 30), 0, 50)
    total_score = round(float(disp_score + rsi_score), 1)
    
    if total_score >= 80:
        level = "🚨 [극단적 과열 / 고점 경보]"
    elif total_score >= 60:
        level = "⚠️ [과열 주의 구간]"
    elif total_score >= 40:
        level = "⚖️ [중립 구간]"
    else:
        level = "🟢 [안정 / 조정 구간]"

    report = (
        f"📊 [QQQ 모닝 고점 판독 보고서]\n"
        f"📅 기준: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"• QQQ 종가: ${current_close:.2f}\n"
        f"• 200일 이격도: {disparity_200:.2f}%\n"
        f"• 14일 RSI: {rsi_14:.2f}\n"
        f"────────────────\n"
        f"🎯 종합 위험도 점수: {total_score} / 100점\n"
        f"상태: {level}"
    )
    return report

def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("토큰 또는 Chat ID 환경변수가 없습니다.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("텔레그램 전송 성공!")
    else:
        print(f"전송 실패 ({response.status_code}): {response.text}")

if __name__ == "__main__":
    msg = calculate_risk_score()
    print(msg)
    send_telegram_message(msg)
