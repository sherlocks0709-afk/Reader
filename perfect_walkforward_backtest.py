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
    payload = {"chat_id": chat_id, "text": text[:4000], "parse_mode": "HTML"}
    try:
        res = requests.post(url, data=payload, timeout=15)
        if res.status_code == 200:
            print("✅ 텔레그램 전송 성공!")
    except Exception as e:
        print(f"텔레그램 통신 실패: {e}")

def run_precision_analysis():
    if not os.path.exists(RAW_DB_FILE):
        print("DB 파일이 없습니다.")
        return

    df = pd.read_csv(RAW_DB_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

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

        # ── 1. 정밀 버블 과열 필터 (가짜 랠리 배제) ──
        # 이격도 극단 과열 + SKEW 극단 헤지 + VIX 19 이상 발작 + 5일선/MACD 동시 붕괴
        cond_bubble = (
            (disp200.iloc[i] >= 108.0 or disp20.iloc[i] >= 106.0) and
            (float(df['SKEW'].iloc[i]) >= 140.0) and
            (float(df['VIX'].iloc[i]) >= 19.5) and
            (curr_p < float(sma5.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # ── 2. 정밀 크레딧 경색 필터 (채권 노이즈 배제) ──
        # 하이일드 스프레드 +0.30%p 이상 급등 AND HYG/TLT -2.5% 이상 붕괴 AND 20일선 이탈
        hy_chg_20d = float(df['HY_SPREAD'].iloc[i]) - float(df['HY_SPREAD'].iloc[i-20])
        r_now = float(df['HYG'].iloc[i]) / float(df['TLT'].iloc[i])
        r_prev = float(df['HYG'].iloc[i-20]) / float(df['TLT'].iloc[i-20])
        hyg_tlt_drop = ((r_now / r_prev) - 1) * 100

        cond_credit = (
            (hy_chg_20d >= 0.30 and hyg_tlt_drop <= -2.5) and
            (curr_p < float(sma20.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # ── 3. 정밀 체제 붕괴 필터 (상승장 속 단순 50일선 터치 배제) ──
        # 50일선 붕괴 + 20일선이 50일선 아래(데드) + 거래량 1.25배 실린 투매
        cond_breakdown = (
            (curr_p < float(sma50.iloc[i])) and
            (float(sma20.iloc[i]) <= float(sma50.iloc[i])) and
            (curr_p < float(sma20.iloc[i])) and
            (vol_ratio.iloc[i] >= 1.25)
        )

        trigger_reason = ""
        if cond_bubble: trigger_reason = "극단버블과열"
        elif cond_credit: trigger_reason = "크레딧위기"
        elif cond_breakdown: trigger_reason = "추세체제붕괴"

        if trigger_reason:
            if not peaks_detected or (d_curr - peaks_detected[-1]['date']).days > 30:
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
    drop_over_5 = (peak_df['max_drop_60d'] <= -5.0).sum()
    total_signals = len(peak_df)
    
    hit_ratio_10 = (drop_over_10 / total_signals * 100) if total_signals > 0 else 0
    hit_ratio_5 = (drop_over_5 / total_signals * 100) if total_signals > 0 else 0

    msg = (
        f"🎯 <b>[14개년 노이즈 제거 초정밀 고점 판독 검증]</b>\n"
        f"📅 기간: {df.index[50].strftime('%Y-%m-%d')} ~ {df.index[-60].strftime('%Y-%m-%d')} ({len(df)}거래일)\n"
        f"────────────────\n"
        f"• <b>총 감지된 고점 신호:</b> <b>{total_signals}회</b> (기존 121회에서 대폭 압축)\n"
        f"• <b>🚨 -10% 이상 대형 폭락 적중:</b> <b>{drop_over_10}회</b>\n"
        f"• <b>⚠️ -5% 이상 조정 포함 적중:</b> <b>{drop_over_5}회</b>\n"
        f"• <b>대형 폭락(-10%) 적중률:</b> <b>{hit_ratio_10:.1f}%</b>\n"
        f"• <b>실질 방어 성공률(-5%이상):</b> <b>{hit_ratio_5:.1f}%</b>\n"
        f"• <b>신호 후 실제 평균 낙폭:</b> <b>{peak_df['max_drop_60d'].mean():.2f}%</b>\n"
        f"────────────────\n"
        f"<b>[정밀 필터 통과 고점 전수 로그]</b>\n"
    )
    for _, r in peak_df.iterrows():
        status = "🚨 대형폭락(-10%이상)" if r['max_drop_60d'] <= -10.0 else "⚠️ 일반조정"
        msg += f"• <b>{r['date'].strftime('%Y-%m-%d')}</b> (${r['price']:.1f}) [{r['reason']}] ➔ 낙폭: <b>{r['max_drop_60d']}%</b> {status}\n"

    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_precision_analysis()
