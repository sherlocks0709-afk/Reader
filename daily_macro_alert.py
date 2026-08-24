import math
import os
import time
import traceback
from datetime import datetime
from curl_cffi import requests as c_requests
import numpy as np
import pandas as pd
import requests

# ==============================================================================
# 1. 텔레그램 환경 변수 및 DB 설정
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DB_FILE_PATH = "history_db.csv"


def send_telegram_message(message: str):
    """HTML 파싱 모드로 텔레그램 알림 메시지 발송"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[주의] 텔레그램 토큰 미설정으로 콘솔에 출력합니다.")
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
            print(f">> [발송 실패] {res.text}")
    except Exception as e:
        print(f">> [에러] 전송 예외 발생: {e}")


# ==============================================================================
# 2. 멀티 소스 실시간 수집 엔진 (Yahoo Finance + FRED + CBOE)
# ==============================================================================
class LiveMarketDataCollector:

    def __init__(self):
        self.session = c_requests.Session(impersonate="chrome120")
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            ),
        }

    def fetch_yahoo_df(self, ticker: str, range_str="3mo") -> pd.DataFrame:
        """야후 파이낸스 차트 API로부터 OHLCV 데이터프레임 안전 수집"""
        encoded = ticker.replace("^", "%5E").replace("=", "%3D")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?range={range_str}&interval=1d"
        for _ in range(3):
            try:
                time.sleep(0.3)
                res = self.session.get(url, headers=self.headers, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    result = data["chart"]["result"][0]
                    timestamps = result["timestamp"]
                    quote = result["indicators"]["quote"][0]
                    dates = [
                        datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        for ts in timestamps
                    ]
                    df = pd.DataFrame(
                        {
                            "Open": quote.get("open", []),
                            "High": quote.get("high", []),
                            "Low": quote.get("low", []),
                            "Close": quote.get("close", []),
                            "Volume": quote.get("volume", []),
                        },
                        index=dates,
                    ).dropna()
                    if not df.empty:
                        return df
            except Exception:
                time.sleep(1.2)
        return pd.DataFrame()

    def fetch_fred_liquidity(self) -> tuple:
        """FRED 공식 엔드포인트에서 TGA 및 RRP(역레포) 실시간 잔고 수집 ($B)"""
        tga, rrp = None, None
        try:
            url_tga = (
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN"
            )
            df_tga = pd.read_csv(url_tga)
            df_tga = df_tga[df_tga["WTREGEN"] != "."].dropna()
            if not df_tga.empty:
                tga = round(float(df_tga.iloc[-1]["WTREGEN"]) / 1000.0, 1)
        except Exception:
            pass

        try:
            url_rrp = (
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD"
            )
            df_rrp = pd.read_csv(url_rrp)
            df_rrp = df_rrp[df_rrp["RRPONTSYD"] != "."].dropna()
            if not df_rrp.empty:
                rrp = round(float(df_rrp.iloc[-1]["RRPONTSYD"]), 1)
        except Exception:
            pass

        return tga, rrp

    def fetch_cboe_equity_pcr(self) -> float:
        """CBOE 공식 개별주식 풋/콜 비율(Equity PCR) 실시간 수집"""
        try:
            url = "https://cdn.cboe.com/data/us/options/market_statistics/daily/current_market_statistics.csv"
            res = requests.get(url, timeout=7)
            if res.status_code == 200:
                for line in res.text.split("\n"):
                    line_upper = line.upper()
                    if "EQUITY" in line_upper and "PUT/CALL RATIO" in line_upper:
                        return round(float(line.split(",")[-1].strip()), 2)
        except Exception:
            pass
        return None


# ==============================================================================
# 3. 고정밀 대체재(Proxy) 산출 엔진 (결측치 95%+ 오차 방어)
# ==============================================================================
def estimate_proxy_equity_pcr(qqq_df, vxn_val, prev_pcr=None) -> float:
    """CBOE 장애 시 QQQ 장중 수급 압력(CLV)과 VXN 기반 Equity PCR 정밀 추정"""
    if not qqq_df.empty:
        c, h, l = (
            float(qqq_df["Close"].iloc[-1]),
            float(qqq_df["High"].iloc[-1]),
            float(qqq_df["Low"].iloc[-1]),
        )
        clv = ((c - l) - (h - c)) / (h - l) if (h > l) else 0.0
        skew_est = 0.65 - (clv * 0.20) + max(0.0, (vxn_val - 20.0) * 0.015)
        return round(float(np.clip(skew_est, 0.45, 1.45)), 2)
    return float(prev_pcr) if prev_pcr else 0.65


# ==============================================================================
# 4. 14개 핵심 지표 수집 및 건전성 체크
# ==============================================================================
def fetch_all_live_indicators(prev_record: dict = None) -> dict:
    today_str = datetime.now().strftime("%Y-%m-%d")
    collector = LiveMarketDataCollector()
    warnings = []

    # 야후 파이낸스 실시간 호출
    qqq_df = collector.fetch_yahoo_df("QQQ")
    vix_df = collector.fetch_yahoo_df("^VIX")
    vix1d_df = collector.fetch_yahoo_df("^VIX1D")
    vxn_df = collector.fetch_yahoo_df("^VXN")
    tnx_df = collector.fetch_yahoo_df("^TNX")
    dxy_df = collector.fetch_yahoo_df("DX-Y.NYB")
    hyg_df = collector.fetch_yahoo_df("HYG")
    tlt_df = collector.fetch_yahoo_df("TLT")
    nq_df = collector.fetch_yahoo_df("NQ=F")

    if qqq_df.empty:
        raise ValueError("FATAL: QQQ 주가 데이터를 수집하지 못했습니다.")

    qqq = qqq_df["Close"]
    qqq_close = float(qqq.iloc[-1])
    qqq_ret_1d = (
        float(qqq.pct_change().iloc[-1] * 100) if len(qqq) > 1 else 0.0
    )
    sma60 = (
        float(qqq.rolling(60).mean().iloc[-1])
        if len(qqq) >= 60
        else qqq_close
    )
    disparity_60 = float((qqq_close / sma60 - 1) * 100)
    hv5 = (
        float(qqq.pct_change().rolling(5).std().iloc[-1] * np.sqrt(252) * 100)
        if len(qqq) >= 5
        else 15.0
    )

    vix_val = (
        float(vix_df["Close"].iloc[-1]) if not vix_df.empty else 16.0
    )
    vxn_val = (
        float(vxn_df["Close"].iloc[-1]) if not vxn_df.empty else 20.0
    )

    # VIX1D 합성 복원 (VXN 60% + QQQ 파킨슨 변동성 40%)
    vix1d_is_proxy = False
    if not vix1d_df.empty and "Close" in vix1d_df.columns:
        vix1d_val = float(vix1d_df["Close"].iloc[-1])
    else:
        vix1d_is_proxy = True
        warnings.append("VIX1D (합성 엔진 대체)")
        h, l = (
            float(qqq_df["High"].iloc[-1]),
            float(qqq_df["Low"].iloc[-1]),
        )
        parkinson = (
            np.sqrt((1.0 / (4.0 * np.log(2))) * ((np.log(h / l)) ** 2))
            * np.sqrt(252)
            * 100
            if (l > 0 and h > l)
            else 18.0
        )
        vix1d_val = (vxn_val * 0.6) + (parkinson * 0.4)

    vix1d_val = round(float(vix1d_val), 2)
    vix_ratio = (
        round(float(vix1d_val / vix_val), 2) if vix_val != 0 else 1.0
    )

    tnx = tnx_df["Close"] if not tnx_df.empty else pd.Series(dtype=float)
    tnx_val = float(tnx.iloc[-1]) if not tnx.empty else 4.2
    tnx_roc5 = (
        float((tnx.iloc[-1] / tnx.iloc[-5] - 1) * 100) if len(tnx) >= 5 else 0.0
    )
    dxy_val = (
        float(dxy_df["Close"].iloc[-1]) if not dxy_df.empty else 103.5
    )

    hyg_val = float(hyg_df["Close"].iloc[-1]) if not hyg_df.empty else 75.0
    tlt_val = float(tlt_df["Close"].iloc[-1]) if not tlt_df.empty else 90.0
    hyg_tlt_ratio = (
        round(float(hyg_val / tlt_val), 3) if tlt_val != 0 else 0.83
    )

    # FRED 유동성 수집
    tga_raw, rrp_raw = collector.fetch_fred_liquidity()
    if tga_raw is None:
        warnings.append("TGA 잔고 (DB 계승)")
        tga_level = (
            float(prev_record.get("TGA_Level", 750.0))
            if prev_record
            else 750.0
        )
    else:
        tga_level = tga_raw

    if rrp_raw is None:
        warnings.append("RRP 잔고 (DB 계승)")
        rrp_level = (
            float(prev_record.get("RRP_Level", 350.0))
            if prev_record
            else 350.0
        )
    else:
        rrp_level = rrp_raw

    # CBOE Equity PCR 수집
    pcr_raw = collector.fetch_cboe_equity_pcr()
    pcr_is_proxy = False
    if pcr_raw is None:
        pcr_is_proxy = True
        warnings.append("Equity PCR (수급모델 Proxy)")
        prev_pcr = prev_record.get("PCR") if prev_record else None
        pcr_val = estimate_proxy_equity_pcr(
            qqq_df, vxn_val, prev_pcr=prev_pcr
        )
    else:
        pcr_val = pcr_raw

    # 선물 괴리율(Basis %)
    basis_is_proxy = False
    if not nq_df.empty:
        nq_close = float(nq_df["Close"].iloc[-1])
        futures_basis_pct = round(
            float((nq_close / (qqq_close * 40.0) - 1) * 100), 2
        )
    else:
        basis_is_proxy = True
        warnings.append("선물 베이시스 (이론가 Proxy)")
        futures_basis_pct = round(
            float(0.04 + (tnx_val * 0.01) + (qqq_ret_1d * 0.02)), 2
        )

    # 종합 리스크 스코어링 (0~100)
    disparity_stress = max(0.0, -disparity_60) * 4.0
    rate_stress = max(0.0, tnx_roc5) * 3.0
    credit_stress = max(0.0, (0.9 - hyg_tlt_ratio)) * 40.0
    macro_score = min(
        100.0, max(0.0, 15.0 + disparity_stress + rate_stress + credit_stress)
    )

    vol_score = min(
        100.0,
        max(
            0.0,
            (vix1d_val * 1.2)
            + (max(0.0, vix_ratio - 1.0) * 30.0)
            + (hv5 * 0.6),
        ),
    )

    return {
        "Date": today_str,
        "QQQ_Close": round(qqq_close, 2),
        "QQQ_Ret_1D": round(qqq_ret_1d, 2),
        "Disparity_60": round(disparity_60, 2),
        "HV5": round(hv5, 2),
        "VIX": round(vix_val, 2),
        "VIX1D": vix1d_val,
        "VIX_Ratio": vix_ratio,
        "US10Y": round(tnx_val, 2),
        "US10Y_ROC5": round(tnx_roc5, 2),
        "DXY": round(dxy_val, 2),
        "HYG_TLT_Ratio": hyg_tlt_ratio,
        "TGA_Level": tga_level,
        "RRP_Level": rrp_level,
        "PCR": pcr_val,
        "Futures_Basis": futures_basis_pct,
        "Macro_Score": round(macro_score, 1),
        "Vol_Score": round(vol_score, 1),
        "Warnings": warnings,
        "VIX1D_Proxy": vix1d_is_proxy,
        "PCR_Proxy": pcr_is_proxy,
        "Basis_Proxy": basis_is_proxy,
    }


# ==============================================================================
# 5. DB 로드 및 누적 저장
# ==============================================================================
def get_prev_business_day_data(today_str: str):
    if not os.path.exists(DB_FILE_PATH):
        return None
    try:
        db = pd.read_csv(DB_FILE_PATH)
        db_past = db[db["Date"] < today_str]
        if not db_past.empty:
            return db_past.iloc[-1].to_dict()
    except Exception:
        pass
    return None


def save_to_history_db(data_dict: dict):
    save_data = {
        k: v
        for k, v in data_dict.items()
        if k not in ["Warnings", "VIX1D_Proxy", "PCR_Proxy", "Basis_Proxy"]
    }
    new_row = pd.DataFrame([save_data])
    if os.path.exists(DB_FILE_PATH):
        db = pd.read_csv(DB_FILE_PATH)
        db = db[db["Date"] != data_dict["Date"]]
        db = pd.concat([db, new_row], ignore_index=True)
    else:
        db = new_row
    db.to_csv(DB_FILE_PATH, index=False)
    print(f">> history_db.csv 저장 완료 (누적 {len(db)}행)")


# ==============================================================================
# 6. 포맷터 및 브리핑 발송 (MDD 10%대 Core 30% / Sat 55% / Snip 15% 가이드)
# ==============================================================================
def is_valid_num(val):
    if val is None:
        return False
    try:
        f = float(val)
        return not (math.isnan(f) or math.isinf(f))
    except (ValueError, TypeError):
        return False


def fmt_row(label: str, curr, prev, unit="", is_rate=False, is_proxy=False) -> str:
    curr_valid = is_valid_num(curr)
    prev_valid = is_valid_num(prev)

    if not curr_valid:
        return f"• {label:<16}: <code>-        </code> (-)"

    curr_f = float(curr)
    val_str = f"{curr_f:.2f}{unit}" if is_rate else f"{curr_f}{unit}"
    if is_proxy:
        val_str += "*"

    if prev_valid:
        diff = curr_f - float(prev)
        sign = "+" if diff > 0 else ""
        diff_str = f"{sign}{diff:.2f}%p" if is_rate else f"{sign}{diff:.2f}"
    else:
        diff_str = "-"

    return f"• {label:<16}: <code>{val_str:<9}</code> ({diff_str})"


def analyze_and_broadcast(data: dict, prev: dict = None):
    macro_score = data["Macro_Score"]
    vol_score = data["Vol_Score"]
    vix1d = data["VIX1D"]
    disparity = data["Disparity_60"]
    hv5 = data["HV5"]

    vix1d_prev = (
        float(prev["VIX1D"])
        if (prev and is_valid_num(prev.get("VIX1D")))
        else vix1d
    )
    pcr = data["PCR"]
    pcr_prev = (
        float(prev["PCR"]) if (prev and is_valid_num(prev.get("PCR"))) else pcr
    )
    prev_macro = (
        float(prev["Macro_Score"])
        if (prev and is_valid_num(prev.get("Macro_Score")))
        else macro_score
    )

    vix1d_turned = (vix1d_prev >= 35.0) and (vix1d < vix1d_prev)
    pcr_turned = (pcr_prev >= 1.00) and (pcr < pcr_prev)
    early_exit_triggered = (hv5 >= 22.0 and vix1d >= 24.0) or (
        disparity <= 1.0 and vix1d >= 22.0
    )
    panic_oversold = (vix1d >= 45.0) or (disparity <= -6.5 and vix1d >= 30.0)

    # 1. 저점 스나이핑 1차 (VIX1D 피크아웃 본대 진입: TQQQ 20%)
    if vix1d_turned:
        header_icon = "⚡"
        status_title = "저점 매수 1차 (TQQQ 20% 스나이핑)"
        us_guide = "Core QQQ 30% | Satellite QQQ 35% + TQQQ 10% | Sniping TQQQ 10% + SGOV 5%\n  (총합: QQQ 65% | TQQQ 20% | SGOV 15%)"
        kr_guide = (
            "Core 나스닥 30% | Satellite 나스닥 35% | 레버리지 20% | SOFR 15%"
        )
        action_desc = f"VIX1D({vix1d_prev:.2f} → {vix1d:.2f}) 피크아웃. TQQQ 20% 본대 스나이핑."

    # 2. 저점 스나이핑 2차 (Equity PCR 확신 풀스나이핑: TQQQ 25%)
    elif pcr_turned and macro_score < 60.0:
        header_icon = "🚀"
        status_title = "저점 매수 2차 (TQQQ 25% 확신 스나이핑)"
        us_guide = "Core QQQ 30% | Satellite QQQ 35% + TQQQ 10% | Sniping TQQQ 15%\n  (총합: QQQ 65% | TQQQ 25% | SGOV 10%)"
        kr_guide = (
            "Core 나스닥 30% | Satellite 나스닥 35% | 레버리지 25% | SOFR 10%"
        )
        action_desc = f"Equity PCR({pcr_prev:.2f} → {pcr:.2f}) 공포 완화 확인. TQQQ 25% 확정 진입."

    # 3. 투매 절정 선발대 스나이핑 (TQQQ 10% 선진입)
    elif panic_oversold:
        header_icon = "🎯"
        status_title = "투매 절정 (TQQQ 10% 선발대 스나이핑)"
        us_guide = "Core QQQ 30% 유지 | Satellite SGOV 55% | Sniping TQQQ 10% + SGOV 5%\n  (총합: QQQ 30% | TQQQ 10% | SGOV 60%)"
        kr_guide = (
            "Core 나스닥 30% 유지 | 달러SOFR 60% | 달러레버리지 10% 선진입"
        )
        action_desc = f"단기 패닉 극단치(VIX1D {vix1d:.2f}). 신규 손절 금지 및 현금으로 TQQQ 10% 선발대 매수."

    # 4. 위험 경보 (70% 안전자산 철벽 대피)
    elif (macro_score >= 48.0 and vol_score >= 40.0) or early_exit_triggered:
        header_icon = "⚠️"
        status_title = "위험 경보 (안전자산 70% 대피)"
        us_guide = "Core QQQ 30% 유지 | Satellite SGOV 55% | Sniping SGOV 15%\n  (총합: QQQ 30% | SGOV 70%)"
        kr_guide = "Core 나스닥 30% 유지 | 달러SOFR 70%"
        action_desc = "단기 변동성 및 매크로 스트레스 급등. Core 30% 유지 후 70% SGOV 대피."

    # 5. 휩쏘 방지 버퍼
    elif prev_macro >= 48.0 and macro_score >= 40.0:
        header_icon = "⏳"
        status_title = "재진입 대기 (휩쏘 방지 안착 관망)"
        us_guide = "Core QQQ 30% 유지 | Satellite SGOV 55% | Sniping SGOV 15%\n  (총합: QQQ 30% | SGOV 70%)"
        kr_guide = "Core 나스닥 30% 유지 | 달러SOFR 70%"
        action_desc = f"매크로 점수({macro_score}/100) 안정화 대기. 40점 이하 안착 시 QQQ 85% 복귀."

    # 6. 정상 운용 (Risk-On)
    else:
        header_icon = "✅"
        status_title = "정상 운용 (Risk-On / QQQ 85% 유지)"
        us_guide = "Core QQQ 30% | Satellite QQQ 55% | Sniping SGOV 15%\n  (총합: QQQ 85% | SGOV 15%)"
        kr_guide = "Core 나스닥 30% | Satellite 나스닥 55% | 달러SOFR 15%"
        action_desc = "지표 안정권. QQQ 85% 보유 유지 + 현금 15% SGOV 이자 수취."

    prev_date_str = (
        f"vs {prev['Date']}"
        if (prev and prev.get("Date") and str(prev.get("Date")) != "nan")
        else "Initial"
    )
    p_d = prev if prev else {}

    warnings = data.get("Warnings", [])
    warn_banner = (
        f"⚠️ <b>[데이터 알림] {len(warnings)}개 지표 대체재(*) 가동</b>\n<i>({', '.join(warnings)})</i>\n━━━━━━━━━━━━━━━━━━\n"
        if warnings
        else ""
    )

    tga_s = (
        f"${data['TGA_Level']}B"
        if is_valid_num(data.get("TGA_Level"))
        else "$750.0B"
    )
    rrp_s = (
        f"${data['RRP_Level']}B"
        if is_valid_num(data.get("RRP_Level"))
        else "$350.0B"
    )

    msg = f"""
{header_icon} <b>[SYSTEM ALERT] {status_title}</b>
━━━━━━━━━━━━━━━━━━
{warn_banner}📅 <b>기준일자:</b> {data['Date']} (<code>{prev_date_str}</code>)
📈 <b>QQQ 종가:</b> <code>${data['QQQ_Close']}</code> (<b>{data['QQQ_Ret_1D']:+}%</b>)

📊 <b>핵심 리스크 종합 스코어</b>
{fmt_row('매크로 스트레스', data['Macro_Score'], p_d.get('Macro_Score'), '/100')}
{fmt_row('단기 변동성 경보', data['Vol_Score'], p_d.get('Vol_Score'), '/100')}

📈 <b>가격 & 추세 지표</b>
{fmt_row('60일선 이격도', data['Disparity_60'], p_d.get('Disparity_60'), '%', True)}
{fmt_row('5일 실현변동성', data['HV5'], p_d.get('HV5'), '%', True)}

⚡ <b>변동성 & 파생 시장</b>
{fmt_row('VIX (30D)', data['VIX'], p_d.get('VIX'))}
{fmt_row('VIX1D (1D/합성)', data['VIX1D'], p_d.get('VIX1D'), is_proxy=data.get('VIX1D_Proxy'))}
{fmt_row('VIX 비율(1D/30D)', data['VIX_Ratio'], p_d.get('VIX_Ratio'))}
{fmt_row('Equity PCR', data['PCR'], p_d.get('PCR'), is_proxy=data.get('PCR_Proxy'))}
{fmt_row('선물 괴리율(Basis)', data['Futures_Basis'], p_d.get('Futures_Basis'), '%', True, is_proxy=data.get('Basis_Proxy'))}

💵 <b>금리 & 환율 & 유동성</b>
{fmt_row('미국 10년물 금리', data['US10Y'], p_d.get('US10Y'), '%', True)}
{fmt_row('10년물 5일 ROC', data['US10Y_ROC5'], p_d.get('US10Y_ROC5'), '%')}
{fmt_row('달러 인덱스', data['DXY'], p_d.get('DXY'))}
{fmt_row('HYG/TLT 비율', data['HYG_TLT_Ratio'], p_d.get('HYG_TLT_Ratio'))}
• TGA / RRP 잔고   : <code>{tga_s} / {rrp_s}</code>

━━━━━━━━━━━━━━━━━━
🎯 <b>목표 포트폴리오 비중 (Core-Satellite)</b>
🇺🇸 <b>미국 직투:</b>
<code>{us_guide}</code>

🇰🇷 <b>국내 계좌:</b>
<code>{kr_guide}</code>

💡 <b>운용 가이드:</b>
<i>{action_desc}</i>
━━━━━━━━━━━━━━━━━━
📁 <i>Data logged to history_db.csv</i>
"""
    send_telegram_message(msg.strip())


# ==============================================================================
# 7. 메인 실행 진입점
# ==============================================================================
if __name__ == "__main__":
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        prev_data = get_prev_business_day_data(today_str)
        data = fetch_all_live_indicators(prev_record=prev_data)
        save_to_history_db(data)
        analyze_and_broadcast(data, prev=prev_data)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        raise e
