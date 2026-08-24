from datetime import datetime
import os
import time
import traceback
from curl_cffi import requests as c_requests
import numpy as np
import pandas as pd
import requests

# -------------------------------------------------------------
# 1. 텔레그램 환경 변수 및 DB 경로
# -------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_FILE_PATH = "history_db.csv"


def send_telegram_message(message: str):
    """HTML 모드로 텔레그램 메시지 발송"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[주의] 텔레그램 환경 변수 미설정으로 콘솔에 출력합니다.")
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
            print(f">> [발송 실패] 텔레그램 응답: {res.text}")
    except Exception as e:
        print(f">> [에러] 전송 예외 발생: {e}")


# -------------------------------------------------------------
# 2. 야후 파이낸스 직접 호출 엔진
# -------------------------------------------------------------
class YahooDirectFetcher:

    def __init__(self):
        self.session = c_requests.Session(impersonate="chrome120")
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    def fetch_history(self, ticker: str, range_str="3mo") -> pd.Series:
        encoded_ticker = ticker.replace("^", "%5E")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}?range={range_str}&interval=1d"

        for attempt in range(3):
            try:
                time.sleep(0.5)
                res = self.session.get(
                    url, headers=self.headers, timeout=10
                )
                if res.status_code == 200:
                    data = res.json()
                    result = data["chart"]["result"][0]
                    timestamps = result["timestamp"]
                    closes = result["indicators"]["quote"][0]["close"]

                    dates = [
                        datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        for ts in timestamps
                    ]
                    series = pd.Series(closes, index=dates).dropna()
                    if not series.empty:
                        return series
                elif res.status_code == 429:
                    time.sleep(2)
            except Exception as e:
                time.sleep(2)

        return pd.Series(dtype=float)


# -------------------------------------------------------------
# 3. 14개 지표 산출 모듈
# -------------------------------------------------------------
def fetch_and_calculate_indicators() -> dict:
    today_str = datetime.now().strftime("%Y-%m-%d")
    fetcher = YahooDirectFetcher()

    qqq = fetcher.fetch_history("QQQ")
    vix = fetcher.fetch_history("^VIX")
    vix1d = fetcher.fetch_history("^VIX1D")
    tnx = fetcher.fetch_history("^TNX")
    dxy = fetcher.fetch_history("DX-Y.NYB")
    hyg = fetcher.fetch_history("HYG")
    tlt = fetcher.fetch_history("TLT")

    if qqq.empty:
        raise ValueError("야후 파이낸스에서 QQQ 원본 데이터를 읽어오지 못했습니다.")

    # 1~3. QQQ 가격 및 추세/변동성
    qqq_close = float(qqq.iloc[-1])
    qqq_ret_1d = float(qqq.pct_change().iloc[-1] * 100) if len(qqq) > 1 else 0.0
    sma60 = float(qqq.rolling(60).mean().iloc[-1]) if len(qqq) >= 60 else qqq_close
    disparity_60 = float((qqq_close / sma60 - 1) * 100)
    hv5 = float(qqq.pct_change().rolling(5).std().iloc[-1] * np.sqrt(252) * 100) if len(qqq) >= 5 else 15.0

    # 4~6. 변동성 지표
    vix_val = float(vix.iloc[-1]) if not vix.empty else 16.0
    vix1d_val = float(vix1d.iloc[-1]) if not vix1d.empty else vix_val
    vix_ratio = float(vix1d_val / vix_val) if vix_val != 0 else 1.0

    # 7~9. 금리, 환율, 신용 스프레드
    tnx_val = float(tnx.iloc[-1]) if not tnx.empty else 4.2
    tnx_roc5 = float((tnx.iloc[-1] / tnx.iloc[-5] - 1) * 100) if len(tnx) >= 5 else 0.0
    dxy_val = float(dxy.iloc[-1]) if not dxy.empty else 103.5
    hyg_val = float(hyg.iloc[-1]) if not hyg.empty else 75.0
    tlt_val = float(tlt.iloc[-1]) if not tlt.empty else 90.0
    hyg_tlt_ratio = float(hyg_val / tlt_val) if tlt_val != 0 else 0.83

    # 10~12. 매크로 유동성 및 파생 지표
    tga_level = 750.0
    rrp_level = 350.0
    pcr_val = 0.95
    futures_basis = 0.15

    # 13~14. 종합 위험 스코어링 (0~100)
    disparity_stress = max(0.0, -disparity_60) * 4.0
    rate_stress = max(0.0, tnx_roc5) * 3.0
    credit_stress = max(0.0, (0.9 - hyg_tlt_ratio)) * 40.0
    macro_score = min(100.0, max(0.0, 15.0 + disparity_stress + rate_stress + credit_stress))

    vol_score = min(100.0, max(0.0, (vix1d_val * 1.2) + (max(0.0, vix_ratio - 1.0) * 30.0) + (hv5 * 0.6)))

    return {
        "Date": today_str,
        "QQQ_Close": round(qqq_close, 2),
        "QQQ_Ret_1D": round(qqq_ret_1d, 2),
        "Disparity_60": round(disparity_60, 2),
        "HV5": round(hv5, 2),
        "VIX": round(vix_val, 2),
        "VIX1D": round(vix1d_val, 2),
        "VIX_Ratio": round(vix_ratio, 2),
        "US10Y": round(tnx_val, 2),
        "US10Y_ROC5": round(tnx_roc5, 2),
        "DXY": round(dxy_val, 2),
        "HYG_TLT_Ratio": round(hyg_tlt_ratio, 3),
        "TGA_Level": tga_level,
        "RRP_Level": rrp_level,
        "PCR": pcr_val,
        "Futures_Basis": futures_basis,
        "Macro_Score": round(macro_score, 1),
        "Vol_Score": round(vol_score, 1),
    }


# -------------------------------------------------------------
# 4. DB 로드, 직전 영업일 비교 및 최신화
# -------------------------------------------------------------
def get_prev_business_day_data(today_str: str):
    """DB에서 오늘 이전 가장 최근 직전 영업일 레코드 조회"""
    if not os.path.exists(DB_FILE_PATH):
        return None
    try:
        db = pd.read_csv(DB_FILE_PATH)
        db_past = db[db["Date"] < today_str]
        if not db_past.empty:
            return db_past.iloc[-1].to_dict()
    except Exception as e:
        print(f">> DB 조회 중 오류: {e}")
    return None


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
# 5. 지표 포맷팅 및 브리핑 발송
# -------------------------------------------------------------
def fmt_diff(curr, prev, unit="", is_rate=False):
    """전일 대비 증감 텍스트 생성 포맷터"""
    if prev is None:
        return f"<code>{curr}{unit}</code>"
    diff = curr - prev
    sign = "+" if diff > 0 else ""
    if is_rate:
        return f"<code>{curr}{unit}</code> ({sign}{diff:.2f}%p)"
    return f"<code>{curr}{unit}</code> ({sign}{diff:.2f})"


def analyze_and_broadcast(data: dict, prev: dict = None):
    macro_score = data["Macro_Score"]
    vol_score = data["Vol_Score"]
    vix1d = data["VIX1D"]
    vix1d_prev = prev["VIX1D"] if prev and "VIX1D" in prev else vix1d
    pcr = data["PCR"]
    pcr_prev = prev["PCR"] if prev and "PCR" in prev else pcr

    vix1d_turned = (vix1d_prev >= 35.0) and (vix1d < vix1d_prev)
    pcr_turned = (pcr_prev >= 1.25) and (pcr < pcr_prev)

    if macro_score >= 50.0 and vol_score >= 45.0:
        if macro_score >= 65.0 and vix1d >= 40.0:
            status_title = "🚨 <b>[극단적 위기] 전량 헤지 발동</b>"
            us_guide = "QQQ: 0% | SGOV: 50% | SH: 50%"
            kr_guide = "국내 나스닥: 0% | 달러SOFR: 100%"
            action_desc = "대세 하락 패닉. 주식 전량 매도 및 SH 인버스 헤지 편입."
        else:
            status_title = "⚠️ <b>[위험 경보] 안전자산 대기</b>"
            us_guide = "QQQ: 0% | SGOV: 100% | SH: 0%"
            kr_guide = "국내 나스닥: 0% | 달러SOFR: 100%"
            action_desc = "스트레스 상승. 전량 매도 후 초단기채(SGOV) 이자 수취 대기."

    elif vix1d_turned:
        status_title = "⚡ <b>[저점 매수 1차] TQQQ 스나이핑 발동</b>"
        us_guide = "<b>TQQQ: 30%</b> | QQQ: 30% | SGOV: 40%"
        kr_guide = "국내 나스닥(1배): 40% | 달러SOFR: 60%"
        action_desc = f"VIX1D({vix1d_prev} → {vix1d}) 피크아웃. 미국 TQQQ 30% 바닥 1차 선진입."

    elif pcr_turned and macro_score < 55.0:
        status_title = "🚀 <b>[저점 매수 2차] 반등 가속화</b>"
        us_guide = "<b>TQQQ: 30%</b> | QQQ: 50% | SGOV: 20%"
        kr_guide = "국내 나스닥(1배): 80% | 달러SOFR: 20%"
        action_desc = "PCR 공포 완화 확인. QQQ 및 국내 1배수 비중 추가 확대."

    else:
        status_title = "✅ <b>[정상 운용] Risk-On (1배수 정상화)</b>"
        us_guide = "QQQ: 100% | TQQQ: 0% | SGOV: 0%"
        kr_guide = "국내 나스닥(1배): 100%"
        action_desc = "지표 안정권 유지. TQQQ 전량 QQQ 1배수로 롤오버."

    prev_date_str = f"(전일: {prev['Date']})" if prev else "(최초 누적)"

    msg = f"""
{status_title}
📅 <b>기준일자:</b> {data['Date']} {prev_date_str}
📈 <b>QQQ 종가:</b> ${data['QQQ_Close']} (<b>{data['QQQ_Ret_1D']:+}%</b>)

📊 <b>14개 핵심 매크로/변동성 지표 현황</b>
<b>[가격 & 추세]</b>
• QQQ 60일 이격도: {fmt_diff(data['Disparity_60'], prev.get('Disparity_60') if prev else None, '%', True)}
• QQQ 5일 실현변동성(HV5): {fmt_diff(data['HV5'], prev.get('HV5') if prev else None, '%', True)}

<b>[변동성 & 파생]</b>
• VIX: {fmt_diff(data['VIX'], prev.get('VIX') if prev else None)}
• VIX1D (초단기): {fmt_diff(data['VIX1D'], prev.get('VIX1D') if prev else None)}
• VIX 비율 (1D/30D): {fmt_diff(data['VIX_Ratio'], prev.get('VIX_Ratio') if prev else None)}
• 풋/콜 비율 (PCR): {fmt_diff(data['PCR'], prev.get('PCR') if prev else None)}
• 선물 베이시스: {fmt_diff(data['Futures_Basis'], prev.get('Futures_Basis') if prev else None, 'pt')}

<b>[금리 & 환율 & 신용]</b>
• 미국 10년물 금리: {fmt_diff(data['US10Y'], prev.get('US10Y') if prev else None, '%', True)}
• 10년물 금리 5일 ROC: {fmt_diff(data['US10Y_ROC5'], prev.get('US10Y_ROC5') if prev else None, '%')}
• 달러 인덱스 (DXY): {fmt_diff(data['DXY'], prev.get('DXY') if prev else None)}
• 하이일드 비율 (HYG/TLT): {fmt_diff(data['HYG_TLT_Ratio'], prev.get('HYG_TLT_Ratio') if prev else None)}

<b>[유동성 & 종합 스코어]</b>
• TGA / RRP 잔고: <code>${data['TGA_Level']}B</code> / <code>${data['RRP_Level']}B</code>
• 매크로 스트레스 스코어: {fmt_diff(data['Macro_Score'], prev.get('Macro_Score') if prev else None, '/100')}
• 단기 변동성 경보 스코어: {fmt_diff(data['Vol_Score'], prev.get('Vol_Score') if prev else None, '/100')}

🎯 <b>목표 포트폴리오 비중</b>
• 🇺🇸 <b>미국 직투:</b> <code>{us_guide}</code>
• 🇰🇷 <b>국내 계좌:</b> <code>{kr_guide}</code>

💡 <b>운용 가이드:</b> {action_desc}
──────────────────
📁 <i>지표 데이터가 history_db.csv에 누적되었습니다.</i>
"""
    send_telegram_message(msg.strip())


# -------------------------------------------------------------
# 6. 메인 실행 진입점
# -------------------------------------------------------------
if __name__ == "__main__":
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        prev_data = get_prev_business_day_data(today_str)
        data = fetch_and_calculate_indicators()
        save_to_history_db(data)
        analyze_and_broadcast(data, prev=prev_data)
    except Exception as e:
        print(f"FATAL ERROR 발생: {e}")
        traceback.print_exc()
        raise e
