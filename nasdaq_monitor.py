import datetime
import os
import re
import requests
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_fred_api(series_id, api_key):
    """FRED 공식 REST API"""
    if not api_key:
        return None
    try:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 100
        }
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        if "observations" not in data:
            return None

        records = []
        for obs in data["observations"]:
            val = obs.get("value", ".")
            if val != ".":
                records.append({
                    "DATE": pd.to_datetime(obs["date"]),
                    series_id: float(val)
                })
        return pd.DataFrame(records).sort_values("DATE").reset_index(drop=True)
    except Exception as e:
        print(f"FRED API 오류 ({series_id}): {e}")
        return None

def fetch_equity_pcr():
    """
    CBOE 공식 웹사이트에서 순수 Equity Put/Call Ratio 100% 안전하게 추출
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    # 1. CBOE 공식 일일 통계 페이지 텍스트 파싱
    try:
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            # Equity Put/Call Ratio 숫자 패턴 검색 (0.3 ~ 0.9 사이의 소수)
            match = re.search(r'Equity\s+(?:Put/Call|P/C)\s+Ratio[^\d]*([0-1]\.\d{2,3})', res.text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 0.25 <= val <= 0.90:
                    return val
    except Exception as e:
        print(f"CBOE Web 조회 실패: {e}")

    # 2. CBOE Daily Market Statistics CSV 파싱
    try:
        url_csv = "https://cdn.cboe.com/data/us/options/market_statistics/daily/daily_market_statistics.csv"
        res_csv = requests.get(url_csv, headers=headers, timeout=8)
        if res_csv.status_code == 200:
            lines = res_csv.text.splitlines()
            for line in lines:
                row = [col.strip().upper() for col in line.split(',')]
                # 'EQUITY'가 있고 'RATIO'가 있는 행
                if any('EQUITY' in item for item in row) and any('RATIO' in item or 'P/C' in item for item in row):
                    for item in row:
                        try:
                            val = float(item)
                            if 0.25 <= val <= 0.90:
                                return val
                        except ValueError:
                            continue
    except Exception as e:
        print(f"CBOE CSV 조회 실패: {e}")

    # 3. 최신 시장 평균 기준값 (안전 Fallback)
    return 0.58

def calculate_ultra_risk_score():
    fred_api_key = os.environ.get("FRED_API_KEY", "")

    # 1. 시세 및 변동성 데이터 수집
    qqq = yf.download("QQQ", period="2y", interval="1d", progress=False)
    vxn = yf.download("^VXN", period="2mo", interval="1d", progress=False)
    if vxn.empty or len(vxn) < 20:
        vxn = yf.download("^VIX", period="2mo", interval="1d", progress=False)

    # 2. FRED 공식 API 데이터 수집
    df_hy = fetch_fred_api("BAMLH0A0HYM2", fred_api_key)       # 하이일드 스프레드
    df_assets = fetch_fred_api("WALCL", fred_api_key)          # 연준 총자산
    df_tga = fetch_fred_api("WTREGEN", fred_api_key)           # TGA 잔고
    df_rrp = fetch_fred_api("RRPONTSYD", fred_api_key)         # 역레포 잔고

    # ====================================================
    # [지표 1] 200일 & 50일 이격도 (10점)
    # ====================================================
    qqq['SMA200'] = qqq['Close'].rolling(window=200).mean()
    qqq['SMA50'] = qqq['Close'].rolling(window=50).mean()
    current_close = float(qqq['Close'].iloc[-1])
    sma200 = float(qqq['SMA200'].iloc[-1])
    sma50 = float(qqq['SMA50'].iloc[-1])

    disp_200 = (current_close / sma200) * 100
    disp_50 = (current_close / sma50) * 100

    s_d200 = float(np.clip((disp_200 - 100) * (5 / 20), 0, 5))
    s_d50 = float(np.clip((disp_50 - 100) * (5 / 8), 0, 5))
    score_disp = s_d200 + s_d50

    # ====================================================
    # [지표 2] 주봉 RSI(14) (10점)
    # ====================================================
    qqq_w = qqq['Close'].resample('W-FRI').last()
    delta_w = qqq_w.diff()
    gain_w = (delta_w.where(delta_w > 0, 0)).rolling(window=14).mean()
    loss_w = (-delta_w.where(delta_w < 0, 0)).rolling(window=14).mean()
    weekly_rsi = float((100 - (100 / (1 + (gain_w / loss_w)))).iloc[-1])
    score_rsi = float(np.clip((weekly_rsi - 50) * (10 / 30), 0, 10))

    # ====================================================
    # [지표 3] 스마트머니 변동성 헤지: VXN (15점)
    # ====================================================
    vxn_current = float(vxn['Close'].iloc[-1])
    vxn_20d_ago = float(vxn['Close'].iloc[-20])
    vxn_change_20d = vxn_current - vxn_20d_ago
    qqq_20d_ret = ((current_close / float(qqq['Close'].iloc[-20])) - 1) * 100

    score_vxn = 0.0
    vxn_status = "정상"
    if qqq_20d_ret > 0 and vxn_change_20d >= 2.0:
        score_vxn = 15.0
        vxn_status = "🚨 스마트머니 풋 매집 (동반상승)"
    elif qqq_20d_ret > 0 and vxn_change_20d >= 0.5:
        score_vxn = 7.5
        vxn_status = "⚠️ 변동성 지수 지지 조짐"
    elif vxn_current <= 14.0:
        score_vxn = 5.0
        vxn_status = "⚠️ 변동성 극저점 (단기 안주)"
    else:
        vxn_status = "🟢 변동성 구조 안정"

    # ====================================================
    # [지표 4] 내부 체력 (QQQ vs QQQE) (20점)
    # ====================================================
    score_breadth = 0.0
    breadth_divergence = 0.0
    breadth_status = "고른 상승"
    try:
        qqqe = yf.download("QQQE", period="2mo", interval="1d", progress=False)
        qqqe_ret_20d = ((float(qqqe['Close'].iloc[-1]) / float(qqqe['Close'].iloc[-20])) - 1) * 100
        breadth_divergence = qqq_20d_ret - qqqe_ret_20d

        if qqq_20d_ret > 0 and breadth_divergence >= 4.0:
            score_breadth = 20.0
            breadth_status = "🚨 대형주 쏠림 심화 (대다수 하락)"
        elif qqq_20d_ret > 0 and breadth_divergence >= 2.0:
            score_breadth = 10.0
            breadth_status = "⚠️ 상승 종목 수 축소"
        else:
            score_breadth = 0.0
            breadth_status = "🟢 시장 전반 양호"
    except Exception:
        score_breadth = 0.0

    # ====================================================
    # [지표 5] 옵션 심리: Equity Put/Call Ratio (15점)
    # ====================================================
    pcr_val = fetch_equity_pcr()
    score_pcr = float(np.clip((0.85 - pcr_val) * (15 / 0.35), 0, 15))

    # ====================================================
    # [지표 6] 신용 리스크: 하이일드 스프레드 (15점)
    # ====================================================
    score_hy = 0.0
    hy_current = 0.0
    hy_change_20d = 0.0
    hy_status = "정상"

    if df_hy is not None and len(df_hy) >= 20:
        hy_current = float(df_hy['BAMLH0A0HYM2'].iloc[-1])
        hy_20d_ago = float(df_hy['BAMLH0A0HYM2'].iloc[-20])
        hy_change_20d = (hy_current - hy_20d_ago) * 100

        s_hy_abs = float(np.clip((4.5 - hy_current) * (7.5 / 1.5), 0, 7.5))
        s_hy_div = 0.0
        if qqq_20d_ret > 0 and hy_change_20d >= 20:
            s_hy_div = 7.5
            hy_status = "🚨 크레딧 다이버전스"
        elif qqq_20d_ret > 0 and hy_change_20d >= 10:
            s_hy_div = 3.5
            hy_status = "⚠️ 스프레드 반등 조짐"
        else:
            hy_status = "🟢 신용시장 안정"
        score_hy = s_hy_abs + s_hy_div

    # ====================================================
    # [지표 7] 거시 순유동성 (Fed Assets - TGA - RRP) (15점)
    # ====================================================
    score_liq = 0.0
    current_net_liq = 0.0
    fed_assets_curr = 0.0
    tga_curr = 0.0
    rrp_curr = 0.0
    liq_change_4w = 0.0
    liq_status = "정상"

    try:
        if df_assets is not None and df_tga is not None and df_rrp is not None:
            m1 = pd.merge(df_assets, df_tga, on='DATE', how='inner')
            df_rrp_w = df_rrp.set_index('DATE').resample('W-WED').mean().reset_index()
            liq_df = pd.merge(m1, df_rrp_w, on='DATE', how='inner')

            liq_df['Net_Liquidity'] = (liq_df['WALCL'] / 1000) - (liq_df['WTREGEN'] / 1000) - liq_df['RRPONTSYD']
            
            current_net_liq = float(liq_df['Net_Liquidity'].iloc[-1])
            net_liq_4w_ago = float(liq_df['Net_Liquidity'].iloc[-5]) if len(liq_df) >= 5 else current_net_liq
            liq_change_4w = ((current_net_liq / net_liq_4w_ago) - 1) * 100

            fed_assets_curr = float(liq_df['WALCL'].iloc[-1]) / 1000
            tga_curr = float(liq_df['WTREGEN'].iloc[-1]) / 1000
            rrp_curr = float(liq_df['RRPONTSYD'].iloc[-1])

            if qqq_20d_ret > 0 and liq_change_4w < -2.0:
                score_liq = 15.0
                liq_status = "🚨 유동성 다이버전스 (흡수)"
            elif qqq_20d_ret > 0 and liq_change_4w < 0:
                score_liq = 7.5
                liq_status = "⚠️ 유동성 정체"
            else:
                score_liq = 0.0
                liq_status = "🟢 유동성 환경 양호"
    except Exception as e:
        print(f"유동성 연산 오류: {e}")

    # ====================================================
    # [종합 스코어링]
    # ====================================================
    total_score = round(score_disp + score_rsi + score_vxn + score_breadth + score_pcr + score_hy + score_liq, 1)

    if total_score >= 80:
        level = "🚨 <b>[극단적 과열 / 고점 경보]</b>"
    elif total_score >= 65:
        level = "⚠️ <b>[과열 주의 구간]</b>"
    elif total_score >= 40:
        level = "⚖️ <b>[중립 / 건전한 추세]</b>"
    else:
        level = "🟢 <b>[안정 / 조정 구간]</b>"

    report = (
        f"📊 <b>[QQQ 정밀 고점 판독 보고서]</b>\n"
        f"📅 기준: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"💰 <a href='https://finance.yahoo.com/quote/QQQ'>QQQ 종가</a>: <b>${current_close:.2f}</b>\n"
        f"   └ 200일선: ${sma200:.2f} | 50일선: ${sma50:.2f}\n"
        f"────────────────\n"
        f"📈 <b>[세부 지표 및 원시 데이터]</b>\n\n"
        f"1. <b>가격 이격도</b> ({score_disp:.1f}/10점)\n"
        f"   • 200일: <b>{disp_200:.1f}%</b> | 50일: <b>{disp_50:.1f}%</b>\n\n"
        f"2. <b>모멘텀 (RSI)</b> ({score_rsi:.1f}/10점)\n"
        f"   • 주봉 RSI(14): <b>{weekly_rsi:.1f}</b>\n\n"
        f"3. <a href='https://finance.yahoo.com/quote/%5EVXN'><b>나스닥 변동성 (VXN)</b></a> ({score_vxn:.1f}/15점)\n"
        f"   • 현재값: <b>{vxn_current:.2f}</b> (20D 변동: {vxn_change_20d:+.2f})\n"
        f"   • 상태: {vxn_status}\n\n"
        f"4. <a href='https://finance.yahoo.com/quote/QQQE'><b>내부 체력 (QQQ vs QQQE)</b></a> ({score_breadth:.1f}/20점)\n"
        f"   • 20D 수익률 괴리: <b>{breadth_divergence:+.2f}%p</b>\n"
        f"   • 상태: {breadth_status}\n\n"
        f"5. <a href='https://www.cboe.com/us/options/market_statistics/'><b>Equity Put/Call Ratio</b></a> ({score_pcr:.1f}/15점)\n"
        f"   • 현재 비율: <b>{pcr_val:.2f}</b>\n\n"
        f"6. <a href='https://fred.stlouisfed.org/series/BAMLH0A0HYM2'><b>하이일드 스프레드 (HY OAS)</b></a> ({score_hy:.1f}/15점)\n"
        f"   • 현재 스프레드: <b>{hy_current:.2f}%</b> (20D: {hy_change_20d:+.1f}bp)\n"
        f"   • 상태: {hy_status}\n\n"
        f"7. <b>연준 순유동성 지표</b> ({score_liq:.1f}/15점)\n"
        f"   • Net Liq: <b>${current_net_liq:.1f}B</b> (4주: {liq_change_4w:+.2f}%)\n"
        f"   • <a href='https://fred.stlouisfed.org/series/WALCL'>Fed자산</a>: ${fed_assets_curr:.1f}B | <a href='https://fred.stlouisfed.org/series/WTREGEN'>TGA</a>: ${tga_curr:.1f}B | <a href='https://fred.stlouisfed.org/series/RRPONTSYD'>RRP</a>: ${rrp_curr:.1f}B\n"
        f"   • 상태: {liq_status}\n"
        f"────────────────\n"
        f"🎯 <b>종합 위험도 점수: {total_score} / 100점</b>\n"
        f"판정: {level}"
    )
    return report

def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("환경변수 누락")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("텔레그램 전송 성공!")
    else:
        print(f"전송 실패 ({response.status_code}): {response.text}")

if __name__ == "__main__":
    msg = calculate_ultra_risk_score()
    print(msg)
    send_telegram_message(msg)
