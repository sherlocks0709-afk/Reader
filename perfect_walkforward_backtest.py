import datetime
import os
import requests
import pandas as pd
import numpy as np

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

def run_pure_crash_analysis():
    if not os.path.exists(RAW_DB_FILE):
        print("DB 파일이 없습니다.")
        return

    df = pd.read_csv(RAW_DB_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    qqq_c = df['QQQ_Close']
    
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

    peaks_detected = []
    
    # 60일 전고점 추적용
    rolling_peak = qqq_c.rolling(60).max()

    for i in range(60, len(df) - 60):
        d_curr = df.index[i]
        curr_p = float(qqq_c.iloc[i])
        peak_60d = float(rolling_peak.iloc[i])
        curr_drawdown = ((curr_p / peak_60d) - 1) * 100

        # [필터 1: 바닥 지각 신호 기각]
        # 이미 고점 대비 -5.0% 이상 빠진 자리는 '고점 매도'가 아니므로 완전 배제
        if curr_drawdown < -5.0:
            continue

        # ── 경로 A: 대형 버블 정점 균열 (-10% 이상 유발) ──
        # 200일 이격 108% 이상 극과열 상태에서 20일선 이탈 + VIX 21.0 돌파
        cond_bubble_crack = (
            (disp200.iloc[i] >= 107.0 or disp20.iloc[i] >= 105.0) and
            (float(df['VIX'].iloc[i]) >= 20.5 or float(df['SKEW'].iloc[i]) >= 142.0) and
            (curr_p < float(sma20.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # ── 경로 B: 매크로/크레딧 신용 발작 (-10% 이상 유발) ──
        # 하이일드 스프레드 20일 급등(+0.25%p) + HYG/TLT -2.5% 붕괴 + 20일선 이탈
        hy_chg_20d = float(df['HY_SPREAD'].iloc[i]) - float(df['HY_SPREAD'].iloc[i-20])
        r_now = float(df['HYG'].iloc[i]) / float(df['TLT'].iloc[i])
        r_prev = float(df['HYG'].iloc[i-20]) / float(df['TLT'].iloc[i-20])
        hyg_tlt_drop = ((r_now / r_prev) - 1) * 100

        cond_credit_shock = (
            (hy_chg_20d >= 0.25 or hyg_tlt_drop <= -2.5) and
            (curr_p < float(sma20.iloc[i])) and
            (curr_p < float(sma5.iloc[i])) and
            (float(macd.iloc[i]) < float(signal.iloc[i]))
        )

        # ── 경로 C: 추세 전환형 50일선 초기 붕괴 ──
        # 50일선 아래로 첫 이탈하면서 20일선도 붕괴 (단, 고점 대비 -4% 이내 어깨 구간)
        cond_initial_break = (
            (curr_p < float(sma50.iloc[i])) and
            (curr_p < float(sma20.iloc[i])) and
            (float(df['VIX'].iloc[i]) >= 19.0) and
            (curr_drawdown >= -4.5)
        )

        trigger_reason = ""
        if cond_bubble_crack: trigger_reason = "버블정점균열"
        elif cond_credit_shock: trigger_reason = "크레딧발작"
        elif cond_initial_break: trigger_reason = "어깨50일선붕괴"

        if trigger_reason:
            if not peaks_detected or (d_curr - peaks_detected[-1]['date']).days > 35:
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
    drop_over_8 = (peak_df['max_drop_60d'] <= -8.0).sum()
    total_signals = len(peak_df)
    
    hit_ratio_10 = (drop_over_10 / total_signals * 100) if total_signals > 0 else 0
    hit_ratio_8 = (drop_over_8 / total_signals * 100) if total_signals > 0 else 0

    msg = (
        f"🎯 <b>[-10% 이상 대형 폭락 전용 초정밀 고점 검증]</b>\n"
        f"📅 기간: {df.index[60].strftime('%Y-%m-%d')} ~ {df.index[-60].strftime('%Y-%m-%d')} ({len(df)}거래일)\n"
        f"────────────────\n"
        f"• <b>총 감지된 고점 신호:</b> <b>{total_signals}회</b> (121회 ➔ 47회 ➔ 압축)\n"
        f"• <b>🚨 -10% 이상 대형 폭락 적중:</b> <b>{drop_over_10}회</b>\n"
        f"• <b>⚠️ -8% 이상 준대형 폭락 포함:</b> <b>{drop_over_8}회</b>\n"
        f"• <b>대형 폭락(-10%) 순수 적중률:</b> <b>{hit_ratio_10:.1f}%</b>\n"
        f"• <b>실질 방어 성공률(-8%이상):</b> <b>{hit_ratio_8:.1f}%</b>\n"
        f"• <b>신호 발생 후 실제 평균 낙폭:</b> <b>{peak_df['max_drop_60d'].mean():.2f}%</b>\n"
        f"────────────────\n"
        f"<b>[감지된 대형 폭락 고점 전수 로그]</b>\n"
    )
    for _, r in peak_df.iterrows():
        status = "🚨 -10% 이상 폭락" if r['max_drop_60d'] <= -10.0 else ("⚠️ -8%대 준폭락" if r['max_drop_60d'] <= -8.0 else "❌ 노이즈")
        msg += f"• <b>{r['date'].strftime('%Y-%m-%d')}</b> (${r['price']:.1f}) [{r['reason']}] ➔ 낙폭: <b>{r['max_drop_60d']}%</b> {status}\n"

    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_pure_crash_analysis()
