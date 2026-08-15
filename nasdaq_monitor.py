import datetime
import os
import io
import requests
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_fred_csv(series_id):
    """FRED 공식 CSV 엔드포인트에서 데이터 다운로드"""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(res.text))
        df['DATE'] = pd.to_datetime(df['DATE'])
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        df = df.dropna().sort_values('DATE').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"FRED 데이터 수집 오류 ({series_id}): {e}")
        return None

def calculate_master_risk_score():
    # ----------------------------------------------------
    # 1. QQQ 및 VIX 데이터 수집 (야후 파이낸스)
    # ----------------------------------------------------
    qqq = yf.download("QQQ", period="2y", interval="1d", progress=False)
    vix = yf.download("^VIX", period="1mo", interval="1d", progress=False)
    vix_close = float(vix['Close'].iloc[-1])

    # ----------------------------------------------------
    # 2. 거시 유동성 & 신용 & 풋콜 지표 수집 (FRED)
    # ----------------------------------------------------
    df_hy = fetch_fred_csv("BAMLH0A0HYM2")       # 하이일드 스프레드
    df_assets = fetch_fred_csv("WALCL")          # 연준 총자산 (주간)
    df_tga = fetch_fred_csv("WTREGEN")           # TGA 잔고 (주간)
    df_rrp = fetch_fred_csv("RRPONTSYD")         # 역레포 잔고 (일간)
    df_pcr = fetch_fred_csv("PCEPC")             # CBOE Equity Put/Call Ratio

    # ====================================================
    # [지표 1] QQQ 200일 이격도 (15점)
    # ====================================================
    qqq['SMA200'] = qqq['Close'].rolling(window=200).mean()
    current_close = float(qqq['Close'].iloc[-1])
    sma200 = float(qqq['SMA200'].iloc[-1])
    disparity_200 = (current_close / sma200) * 100
    score_disp = float(np.clip((disparity_200 - 100) * (15 / 20), 0, 15))

    # ====================================================
    # [지표 2] QQQ 주봉 RSI(14) (15점)
    # ====================================================
    qqq_w = qqq['Close'].resample('W-FRI').last()
    delta_w = qqq_w.diff()
    gain_w = (delta_w.where(delta_w > 0, 0)).rolling(window=14).mean()
    loss_w = (-delta_w.where(delta_w < 0, 0)).rolling(window=14).mean()
    weekly_rsi = float((100 - (100 / (1 + (gain_w / loss_w)))).iloc[-1])
    score_rsi = float(np.clip((weekly_rsi - 50) * (15 / 30), 0, 15))

    # ====================================================
    # [지표 3-1] CBOE VIX 지수 (10점)
    # ====================================================
    score_vix = float(np.clip((20 - vix_close) * (10 / 8), 0, 10))

    # ====================================================
    # [지표 3-2] Equity Put/Call Ratio 10D MA (15점)
    # ====================================================
    score_pcr = 0.0
    pcr_val = 0.0
    pcr_10d = 0.0
    if df_pcr is not None and len(df_pcr) >= 15:
        df_pcr['PCR_10MA'] = df_pcr['PCEPC'].rolling(10).mean()
        pcr_val = float(df_pcr['PCEPC'].iloc[-1])
        pcr_10d = float(df_pcr['PCR_10MA'].iloc[-1])
        # 0.85 이상: 0점, 0.50 이하(극단적 낙관): 15점 만점
        score_pcr = float(np.clip((0.85 - pcr_10d) * (15 / 0.35), 0, 15))

    # ====================================================
    # [지표 4] 하이일드 스프레드 & 크레딧 다이버전스 (20점)
    # ====================================================
    score_hy_abs = 0.0
    score_hy_div = 0.0
    hy_current = 0.0
    hy_change_20d = 0.0
    hy_status = "정상"

    if df_hy is not None and len(df_hy) >= 25:
        hy_current = float(df_hy['BAMLH0A0HYM2'].iloc[-1])
        hy_20d_ago = float(df_hy['BAMLH0A0HYM2'].iloc[-20])
        hy_change_20d = (hy_current - hy_20d_ago) * 100

        # 절대치 저점 (10점 만점: 4.5%~3.0%)
        score_hy_abs = float(np.clip((4.5 - hy_current) * (10 / 1.5), 0, 10))

        # 크레딧 다이버전스 (10점 만점)
        qqq_close_20d = float(qqq['Close'].iloc[-20])
        qqq_ret_20d = ((current_close / qqq_close_20d) - 1) * 100

        if qqq_ret_20d > 0 and hy_change_20d >= 20:
            score_hy_div = 10.0
            hy_status = "🚨 크레딧 다이버전스 (스프레드 확대)"
        elif qqq_ret_20d > 0 and hy_change_20d >= 10:
            score_hy_div = 5.0
            hy_status = "⚠️ 스프레드 미세 반등"
        else:
            hy_status = "🟢 신용시장 안정"

    score_hy_total = score_hy_abs + score_hy_div

    # ====================================================
    # [지표 5] 연준 순유동성 (Fed Assets - TGA - RRP) (25점)
    # ====================================================
    score_liq_div = 0.0
    score_liq_trend = 0.0
    current_net_liq = 0.0
    tga_curr = 0.0
    rrp_curr = 0.0
    liq_change_4w = 0.0
    liq_status = "정상"

    try:
        if df_assets is not None and df_tga is not None and df_rrp is not None:
            # 주간 데이터 기준 병합
            m1 = pd.merge(df_assets, df_tga, on='DATE', how='inner')
            # RRP 일간 데이터를 주간 평균으로 리샘플링 후 병합
            df_rrp_w = df_rrp.set_index('DATE').resample('W-WED').mean().reset_index()
            liq_df = pd.merge(m1, df_rrp_w, on='DATE', how='inner')

            # 순유동성 계산 ($ Billions 단위 변환)
            # WALCL(백만$), WTREGEN(백만$), RRPONTSYD(십억$)
            liq_df['Net_Liquidity'] = (liq_df['WALCL'] / 1000) - (liq_df['WTREGEN'] / 1000) - liq_df['RRPONTSYD']
            
            current_net_liq = float(liq_df['Net_Liquidity'].iloc[-1])
            net_liq_4w_ago = float(liq_df['Net_Liquidity'].iloc[-5])
            liq_change_4w = ((current_net_liq / net_liq_4w_ago) - 1) * 100

            tga_curr = float(liq_df['WTREGEN'].iloc[-1]) / 1000  # $B
            rrp_curr = float(liq_df['RRPONTSYD'].iloc[-1])       # $B

            # 5-1. 유동성 다이버전스 (15점): QQQ 4주 상승 vs 유동성 4주 감소
            qqq_close_4w = float(qqq['Close'].iloc[-20])
            qqq_ret_4w = ((current_close / qqq_close_4w) - 1) * 100

            if qqq_ret_4w > 0 and liq_change_4w < -2.0:
                score_liq_div = 15.0
                liq_status = "🚨 유동성 다이버전스 (유동성 흡수 중)"
            elif qqq_ret_4w > 0 and liq_change_4w < 0:
                score_liq_div = 7.5
                liq_status = "⚠️ 유동성 정체 국면"
            else:
                liq_status = "🟢 유동성 환경 양호"

            # 5-2. 4주 유동성 급감 점수 (10점): 0%~ -5% 감소율 반영
            if liq_change_4w < 0:
                score_liq_trend = float(np.clip(abs(liq_change_4w) * (10 / 5.0), 0, 10))

    except Exception as e:
        print(f"유동성 계산 오류: {e}")

    score_liq_total = score_liq_div + score_liq_trend

    # ====================================================
    # [종합 스코어링]
    # ====================================================
    total_score = round(score_disp + score_rsi + score_vix + score_pcr + score_hy_total + score_liq_total, 1)

    if total_score >= 80:
        level = "🚨 [극단적 과열 / 고점 경보] - 분할 익절 및 헤지 권고"
    elif total_score >= 65:
        level = "⚠️ [과열 주의 구간] - 신규 공격 매수 자제"
    elif total_score >= 40:
        level = "⚖️ [중립 / 건전한 추세]"
    else:
        level = "🟢 [안정 / 조정 및 저평가 구간]"

    # 보고서 텍스트 생성
    report = (
        f"📊 [QQQ 정밀 고점 판독 종합 보고서]\n"
        f"📅 기준: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"💰 QQQ 종가: ${current_close:.2f}\n"
        f"────────────────\n"
        f"📈 [5대 계층 세부 지표 분석]\n"
        f"1. 200일선 이격도: {disparity_200:.2f}% ({score_disp:.1f}/15점)\n"
        f"2. 주봉 RSI(14): {weekly_rsi:.2f} ({score_rsi:.1f}/15점)\n"
        f"3. 심리/파생:\n"
        f"   • CBOE VIX: {vix_close:.2f} ({score_vix:.1f}/10점)\n"
        f"   • Equity Put/Call 10D: {pcr_10d:.2f} ({score_pcr:.1f}/15점)\n"
        f"4. 신용 리스크 (BofA HY):\n"
        f"   • 스프레드: {hy_current:.2f}% (20D 변동: {hy_change_20d:+.1f}bp)\n"
        f"   • 상태: {hy_status} ({score_hy_total:.1f}/20점)\n"
        f"5. 거시 순유동성:\n"
        f"   • Net Liquidity: ${current_net_liq:.1f}B (4주: {liq_change_4w:+.2f}%)\n"
        f"   • TGA: ${tga_curr:.1f}B | RRP: ${rrp_curr:.1f}B\n"
        f"   • 상태: {liq_status} ({score_liq_total:.1f}/25점)\n"
        f"────────────────\n"
        f"🎯 종합 위험도 점수: {total_score} / 100점\n"
        f"현재 상태: {level}"
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
        "text": text
    }
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        print("텔레그램 전송 성공!")
    else:
        print(f"전송 실패 ({response.status_code}): {response.text}")

if __name__ == "__main__":
    msg = calculate_master_risk_score()
    print(msg)
    send_telegram_message(msg)
