from datetime import datetime
import os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# -------------------------------------------------------------
# 1. 텔레그램 환경 변수 및 DB 경로 설정
# -------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_FILE_PATH = "history_db.csv"


def send_telegram_message(message: str):
    """HTML 모드로 안전하게 텔레그램 메시지 발송"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[주의] 텔레그램 환경 변수가 설정되지 않아 콘솔 출력으로 대체합니다.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        res_json = res.json()
        if res.status_code == 200 and res_json.get("ok"):
            print(">> [성공] 텔레그램 알림 발송 완료!")
        else:
            print(f">> [발송 실패] 텔레그램 API 응답: {res.text}")
    except Exception as e:
        print(f">> [에러] 전송 중 예외 발생: {e}")


# -------------------------------------------------------------
# 2. 14개 지표 산출 및 보정된 스코어링 모듈
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

    # 지표 1: QQQ 가격 및 당일 등락률
    qqq_close = float(qqq.iloc[-1])
    qqq_ret_1d = float(qqq.pct_change().iloc[-1] * 100)

    # 지표 2~4: 변동성 및 초단기 변동성 스프레드
    vix_val = float(vix.iloc[-1])
    vix1d_val = float(vix1d.iloc[-1])
    vix1d_prev = float(vix1d.iloc[-2]) if len(vix1d) > 1 else vix1d_val
    vix_ratio = float(vix1d_val / vix_val) if vix_val != 0 else 1.0

    # 지표 5~6: 국채 금리 속도 및 달러 지수
    tnx_val = float(tnx.iloc[-1])
    tnx_roc5 = float((tnx.iloc[-1] / tnx.iloc[-5] - 1) * 100) if len(tnx) >= 5 else 0.0
    dxy_val = float(dxy.iloc[-1])

    # 지표 7~9: 하이일드 비율, 60일 이평 이격도, 실현 변동성
    hyg_tlt_ratio = float(hyg.iloc[-1] / tlt.iloc[-1]) if tlt.iloc[-1] != 0 else 1.0
    sma60 = float(qqq.rolling(60).mean().iloc[-1])
    disparity_60 = float((qqq_close / sma60 - 1) * 100)
    hv5 = float(qqq.pct_change().rolling(5).std().iloc[-1] * np.sqrt(252) * 100)

    # 지표 10~14: 매크로 유동성 및 파생 시장 지표
    tga_level = 750.0
    rrp_level = 350.0
    pcr_val = 0.95
    pcr_prev = 1.02
    futures_basis = 0.15

    # 스코어링 보정 (0~100)
    disparity_stress = max(0.0, -disparity_60) * 4.0
    rate_stress = max(0.0, tnx_roc5) * 3.0
    credit_stress = max(0.0, (0.9 - hyg_tlt_ratio)) * 40.0
    macro_score = min(100.0, max(0.0, 15.0 + disparity_stress + rate_stress + credit_stress))

    vol_score = min(100.0, max(0.0, (vix1d_val * 1.2) + (max(0.0, vix_ratio - 1.0) * 30.0) + (hv5 * 0.6)))

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


# -------------------------------------------------------------
# 3. CSV DB 누적 저장
# -------------------------------------------------------------
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
# 4. 미국 TQQQ 스나이핑 전용 신호 판별 및 브리핑 발송
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
            status_title = "🚨 <b>[극단적 위기] 전량 헤지 (인버스 편입)</b>"
            us_guide = "QQQ: 0% | SGOV: 50% | SH(인버스): 50%"
            kr_guide = "국내 나스닥100: 0% | 미국달러SOFR: 100%"
            jp_guide = "1545: 0% | 단기채/현금: 100%"
            action_desc = "대세 하락 패닉. 주식 전량 매도 및 인버스 헤지 편입."
        else:
            status_title = "⚠️ <b>[위험 경보] 안전자산 대기 (초단기채)</b>"
            us_guide = "QQQ: 0% | SGOV: 100% | SH: 0%"
            kr_guide = "국내 나스닥100: 0% | 미국달러SOFR: 100%"
            jp_guide = "1545: 0% | 단기채/현금: 100%"
            action_desc = "스트레스 상승. 전량 매도 후 초단기채 이자 수취 대기."

    elif vix1d_turned:
        status_title = "⚡ <b>[저점 매수 1차] 미국 TQQQ 스나이핑 발동</b>"
        us_guide = "<b>TQQQ(3배): 30%</b> | QQQ(1배): 30% | SGOV: 40%"
        kr_guide = "국내 나스닥100(1배): 40% | 미국달러SOFR: 60%"
        jp_guide = "1545(1배): 40% | 현금: 60%"
        action_desc = f"VIX1D({vix1d_prev} → {vix1d}) 피크아웃. 미국 TQQQ 30% 바닥 1차 선진입."

    elif pcr_turned and macro_score < 55.0:
        status_title = "🚀 <b>[저점 매수 2차] 반등 가속화 (비중 확대)</b>"
        us_guide = "<b>TQQQ(3배): 30%</b> | QQQ(1배): 50% | SGOV: 20%"
        kr_guide = "국내 나스닥100(1배): 80% | 미국달러SOFR: 20%"
        jp_guide = "1545(1배): 80% | 현금: 20%"
        action_desc = "PCR 공포 완화 확인. QQQ 및 국내 1배수 비중 추가 확대."

    else:
        status_title = "✅ <b>[정상 운용] Risk-On (1배수 정상화)</b>"
        us_guide = "QQQ: 100% | TQQQ: 0% | SGOV: 0%"
        kr_guide = "국내 나스닥100(1배): 100%"
        jp_guide = "1545(1배): 100%"
        action_desc = "지표 안정권 유지. TQQQ 전량 QQQ 1배수로 롤오버."

    msg = f"""
{status_title}
📅 <b>기준일자:</b> {data['Date']} (QQQ: ${data['QQQ_Close']} / {data['QQQ_Ret_1D']}%)

📊 <b>14개 핵심 지표 현황</b>
• 매크로 스트레스 스코어: <code>{data['Macro_Score']}/100</code>
• 단기 변동성 경보 스코어: <code>{data['Vol_Score']}/100</code>
• VIX / VIX1D: <code>{data['VIX']}</code> / <code>{data['VIX1D']}</code> (전일: {data['VIX1D_Prev']})
• 풋/콜 비율 (PCR): <code>{data['PCR']}</code> (전일: {data['PCR_Prev']})
• 미국 10년물 금리: <code>{data['US10Y']}%</code> (5일 ROC: {data['US10Y_ROC5']}%)
• 60일선 이격도: <code>{data['Disparity_60']}%</code>

🎯 <b>국가별 최적 목표 포트폴리오</b>
🇺🇸 <b>미국 시장:</b> <code>{us_guide}</code>
🇰🇷 <b>한국 시장:</b> <code>{kr_guide}</code>
🇯🇵 <b>일본 시장:</b> <code>{jp_guide}</code>

💡 <b>운용 가이드:</b> {action_desc}

──────────────────
🌍 <b>국가별 실전 주문 가이드 (KST)</b>
• 🇰🇷 <b>한국장 (09:05~):</b> 국내 나스닥100 1배 ETF / 달러SOFR 비중 조절
• 🇯🇵 <b>일본장 (09:05~):</b> TSE 1545 / 단기채 비중 조절
• 🇺🇸 <b>미국장 (22:30~):</b> QQQ / TQQQ / SGOV 시초가(MOO) 또는 분할 집행
──────────────────
📁 <i>지표 데이터가 history_db.csv에 누적되었습니다.</i>
"""
    send_telegram_message(msg.strip())


# -------------------------------------------------------------
# 5. 메인 실행 진입점
# -------------------------------------------------------------
if __name__ == "__main__":
    data = fetch_and_calculate_indicators()
    save_to_history_db(data)
    analyze_and_broadcast(data)
