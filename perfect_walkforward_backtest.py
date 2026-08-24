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
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

def load_or_build_longterm_db():
    if os.path.exists(RAW_DB_FILE):
        df = pd.read_csv(RAW_DB_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        if len(df) > 2000:
            return df
    return pd.DataFrame()

def analyze_historical_major_peaks():
    df = load_or_build_longterm_db()
    if df.empty:
        print("DB 파일 로드 실패")
        return

    print(f"📊 [검증 시작] 총 {len(df)}거래일 (2012 ~ 2026) 다중 경로 고점 검증 가동...")

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

        # ── 경로 1: 클라이맥스 버블형 ──
        # 이격 과열 상태에서 5일선 이탈 + MACD 데드크로스
        cond_bubble = (
            (disp200.iloc[i] >= 106.0 or disp20.iloc[i] >= 105.0) and
            (float(df['SKEW'].iloc[i]) >= 135.0 or float(df['VIX'].iloc[i]) >= 19.0) and
            (curr_p < float(sma5.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # ── 경로 2: 크레딧/자금 균열형 ──
        # 하이일드 스프레드 확대 or HYG/TLT 붕괴 + 20일선 이탈
        hy_chg_20d = float(df['HY_SPREAD'].iloc[i]) - float(df['HY_SPREAD'].iloc[i-20])
        r_now = float(df['HYG'].iloc[i]) / float(df['TLT'].iloc[i])
        r_prev = float(df['HYG'].iloc[i-20]) / float(df['TLT'].iloc[i-20])
        hyg_tlt_drop = ((r_now / r_prev) - 1) * 100

        cond_credit = (
            (hy_chg_20d >= 0.20 or hyg_tlt_drop <= -2.0) and
            (curr_p < float(sma20.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # ── 경로 3: 모멘텀 구조 붕괴형 (블랙스완/체제전환) ──
        # 50일선 종가 이탈 + 거래량 실린 음봉
        cond_breakdown = (
            (curr_p < float(sma50.iloc[i])) and
            (curr_p < float(sma20.iloc[i])) and
            (vol_ratio.iloc[i] >= 1.15)
        )

        trigger_reason = ""
        if cond_bubble: trigger_reason = "버블과열"
        elif cond_credit: trigger_reason = "크레딧경색"
        elif cond_breakdown: trigger_reason = "50일선붕괴"

        if trigger_reason:
            # 30거래일 이내 중복 신호 병합
            if not peaks_detected or (d_curr - peaks_detected[-1]['date']).days > 25:
                forward_window = qqq_c.iloc[i:i+60] # 향후 3개월 최대 낙폭
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
        f"🏛️ <b>[14개년 다중 경로 고점 판독 전수 검증 결과]</b>\n"
        f"📅 기간: {df.index[50].strftime('%Y-%m-%d')} ~ {df.index[-60].strftime('%Y-%m-%d')} ({len(df)}거래일)\n"
        f"────────────────\n"
        f"• <b>총 감지된 고점 신호:</b> <b>{len(peak_df)}회</b>\n"
        f"• <b>실제 -10% 이상 대형 폭락 적중:</b> <b>{drop_over_10}회</b>\n"
        f"• <b>실제 -5% 이상 조정 포함 적중:</b> <b>{drop_over_5}회</b>\n"
        f"• <b>고점 신호 후 평균 낙폭:</b> <b>{peak_df['max_drop_60d'].mean():.2f}%</b>\n"
        f"────────────────\n"
        f"<b>[역사적 주요 감지 로그]</b>\n"
    )
    for _, r in peak_df.iterrows():
        status = "🚨 대형폭락(-10%이상)" if r['max_drop_60d'] <= -10.0 else "⚠️ 일반조정"
        msg += f"• <b>{r['date'].strftime('%Y-%m-%d')}</b>: ${r['price']:.2f} [{r['reason']}] ➔ 낙폭: <b>{r['max_drop_60d']}%</b> {status}\n"

    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    analyze_historical_major_peaks()
