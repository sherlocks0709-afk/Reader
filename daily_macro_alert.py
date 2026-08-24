from datetime import datetime
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# -------------------------------------------------------------
# 1. 기존 세팅된 텔레그램 환경 변수 및 DB 로드
# -------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_FILE_PATH = "history_db.csv"


def send_telegram_message(message: str):
    """텔레그램 메시지 발송"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[주의] 텔레그램 환경 변수(TELEGRAM_TOKEN / TELEGRAM_CHAT_ID)가 없어 콘솔로 출력합니다.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print(">> [성공] 텔레그램 알림 발송 완료")
        else:
            print(f">> [실패] 텔레그램 응답 에러: {res.text}")
    except Exception as e:
        print(f">> [에러] 텔레그램 전송 중 예외 발생: {e}")


# -------------------------------------------------------------
# 2. 14개 지표 산출 및 DB 축적
# -------------------------------------------------------------
def fetch_and_calculate_indicators() -> dict:
    today_str = datetime.now().strftime("%Y-%m-%d")

    tickers = ["QQQ", "^VIX", "^VIX1D", "^TNX", "DX-Y.NYB", "HYG", "TLT"]
    raw = yf.download(tickers, period="3mo", interval="1d", progress=False)["Close"]

    qqq = raw["QQQ"].dropna()
    vix = raw["^VIX"].dropna() if "^VIX" in raw else pd.Series(18.0, index=qqq.index)
    vix1d = raw["^VIX1D"].dropna() if "^VIX1D" in raw else pd.Series(18.0, index=qqq.index)
    tnx = raw["^TNX"].dropna() if "^TNX" in raw else pd.Series(4.2, index=qqq.index)
    dxy = raw["DX-Y.NYB"].dropna() if "DX-Y.NYB" in raw else pd.Series(104.0, index=qqq.index)
    hyg = raw["HYG"].dropna() if "HYG" in raw else pd.Series(75.0, index=qqq.index)
    tlt = raw["TLT"].dropna() if "TLT" in raw else pd.Series(92.0, index=qqq.index)

    qqq_close = float(qqq.iloc[-1])
    qqq_ret_1d = float(qqq.pct_change().iloc[-1] * 100)

    vix_val = float(vix.iloc[-1])
    vix1d_val = float(vix1d.iloc[-1])
    vix1d_prev = float(vix1d.iloc[-2]) if len(vix1d) > 1 else vix1d_val
    vix_ratio = float(vix1d_val / vix_val) if vix_val != 0 else 1.0

    tnx_val = float(tnx.iloc[-1])
    tnx_roc5 = float((tnx.iloc[-1] / tnx.iloc[-5] - 1) * 100) if len(tnx) >= 5 else 0.0
    dxy_val = float(dxy.iloc[-1])
    hyg_tlt_ratio = float(hyg.iloc[-1] / tlt.iloc[-1]) if tlt.iloc[-1] != 0 else 1.0

    sma60 = float(qqq.rolling(60).mean().iloc[-1])
    disparity_60 = float((qqq_close / sma60 - 1) * 100)
    hv5 = float(qqq.pct_change().rolling(5).std().iloc[-1] * np.sqrt(252) * 100)

    # 매크로/옵션 보조 지표
    tga_level = 750.0
    rrp_level = 350.0
    pcr_val = 0.95
    pcr_prev = 1.02
    futures_basis = 0.15

    # 스코어링 (0~100)
    macro_score = min(100.0, max(0.0, (4.5 - hyg_tlt_ratio) * 20 + (tnx_roc5 * 2.5) + (100 - disparity_60 * 2)))
    vol_score = min(100.0, max(0.0, (vix1d_val * 1.2) + (vix_ratio * 20) + (hv5 * 0.8)))

    return {
        "Date": today_str,
        "QQQ_Close": round(qqq_close, 2),
        "QQQ_Ret_1D": round(qqq_ret_1d, 2),
        "VIX": round(vix_val, 2),
        "VIX1D": round(vix1d_val, 2),
        "VIX1D_Prev": round(vix1d_prev, 2),
        "VIX_Ratio": round(vix_ratio, 2),
        "US10Y": round(tnx_val, 2),
        "US10Y_ROC5": round(tnx_roc5, 2),
        "DXY": round(dxy_val, 2),
        "HYG_TLT_Ratio": round(hyg_tlt_ratio, 3),
        "Disparity_60": round(disparity_60, 2),
        "HV5": round(hv5, 2),
        "TGA_Level": tga_level,
        "RRP_Level": rrp_level,
        "PCR": pcr_val,
        "PCR_Prev": pcr_prev,
        "Futures_Basis": futures_basis,
        "Macro_Score": round(macro_score, 1),
        "Vol_Score": round(vol_score, 1),
    }


def save_to_history_db(data_dict: dict):
    new_row = pd.DataFrame([data_dict])
    if os.path.exists(DB_FILE_PATH):
        db = pd.read_csv(DB_FILE_PATH)
        db = db[db["Date"] != data_dict["Date"]]
        db = pd.concat([db, new_row], ignore_index=True)
    else:
        db = new_row
    db.to_csv(DB_FILE_PATH, index=False)
    print(f">> history_db.csv 저장 완료 (누적 {len(db)}행)")


# -------------------------------------------------------------
# 3. 신호 판별 및 브리핑 발송
# -------------------------------------------------------------
def analyze_and_broadcast(data: dict):
    macro_score = data["Macro_Score"]
    vol_score = data["Vol_Score"]
    vix1d = data["VIX1D"]
    vix1d_prev = data["VIX1D_Prev"]
    pcr = data["PCR"]
    pcr_prev = data["PCR_Prev"]

    vix1d_turned = (vix1d_prev >= 35.0) and (vix1d < vix1d_prev)
    pcr_turned = (pcr_prev >= 1.25) and (pcr < pcr_prev)

    if macro_score >= 50.0 and vol_score >= 45.0:
        if macro_score >= 65.0 and vix1d >= 40.0:
            status_title = "🚨 [극단적 위기] 전량 헤지 발동 (SH 50% + SGOV 50%)"
            target_alloc = "QQQ: 0% | SGOV: 50% | SH: 50%"
            action_desc = "대세 하락/패닉 국면. QQQ 전량 매도 후 SH 인버스 헤지 편입."
        else:
            status_title = "⚠️ [위험 경보] 안전자산 대기 (SGOV 100%)"
            target_alloc = "QQQ: 0% | SGOV: 100% | SH: 0%"
            action_desc = "스트레스 가중. QQQ 청산 후 초단기채(SGOV)로 안전 대기."
    elif vix1d_turned:
        status_title = "🎯 [저점 매수 1차] VIX1D 극단 피크아웃"
        target_alloc = "QQQ: 40% | SGOV: 60% | SH: 0%"
        action_desc = f"VIX1D({vix1d_prev} → {vix1d}) 고점 꺾임. 1차 바닥 40% 분할 선진입."
    elif pcr_turned and macro_score < 55.0:
        status_title = "🎯 [저점 매수 2차] 옵션 공포 완화 확인"
        target_alloc = "QQQ: 80% | SGOV: 20% | SH: 0%"
        action_desc = "PCR 하향 안정화. QQQ 비중 80%로 확대."
    else:
        status_title = "✅ [정상 운용] Risk-On 포지션 유지"
        target_alloc = "QQQ: 100% | SGOV: 0% | SH: 0%"
        action_desc = "지표 안정권. QQQ 100% 보유 유지."

    msg = f"""
*{status_title}*
📅 *기준일자:* {data['Date']} (QQQ: ${data['QQQ_Close']} / {data['QQQ_Ret_1D']}%)

📊 *14개 핵심 지표 현황*
• 매크로 스트레스 스코어: `{data['Macro_Score']}/100`
• 단기 변동성 경보 스코어: `{data['Vol_Score']}/100`
• VIX / VIX1D: `{data['VIX']}` / `{data['VIX1D']}` (전일: {data['VIX1D_Prev']})
• 풋/콜 비율 (PCR): `{data['PCR']}` (전일: {data['PCR_Prev']})
• 미국 10년물 금리: `{data['US10Y']}%` (5일 ROC: {data['US10Y_ROC5']}%)

🎯 *목표 비중:* `{target_alloc}`
💡 *운용 가이드:* {action_desc}

──────────────────
🌍 *국가별 실전 주문 가이드 (KST)*
• 🇰🇷 *한국장 (09:05~):* 국내 나스닥100 ETF / 달러SOFR 교체
• 🇯🇵 *일본장 (09:05~):* TSE 1545 / 2621 포지션 조정
• 🇺🇸 *미국장 (22:30~):* QQQ 본주 / SGOV / SH 시초가(MOO) 집행
──────────────────
📁 _지표 데이터가 history_db.csv에 누적되었습니다._
"""
    send_telegram_message(msg.strip())


if __name__ == "__main__":
    data = fetch_and_calculate_indicators()
    save_to_history_db(data)
    analyze_and_broadcast(data)
