import datetime
import os
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

def send_telegram_result(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("텔레그램 토큰이 없어 콘솔에만 출력합니다.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

def get_historical_data(ticker_symbol, start_date="2023-04-20"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range=4y&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            res_data = res.json()['chart']['result'][0]
            timestamps = res_data['timestamp']
            quotes = res_data['indicators']['quote'][0]
            ny_tz = ZoneInfo("America/New_York")
            dates = [datetime.datetime.fromtimestamp(ts, tz=ny_tz).date() for ts in timestamps]
            
            df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Open': quotes.get('open', []),
                'High': quotes.get('high', []),
                'Low': quotes.get('low', []),
                'Close': quotes.get('close', []),
                'Volume': quotes.get('volume', [])
            }).dropna(subset=['Close'])
            
            df['Open'] = df['Open'].fillna(df['Close'])
            df.set_index('Date', inplace=True)
            return df[df.index >= pd.to_datetime(start_date)].astype(float)
    except Exception as e:
        print(f"데이터 수집 에러 ({ticker_symbol}): {e}")
    return pd.DataFrame()

def run_backtest():
    print("⏳ 2023.04 이후 정밀 90:5:5 전략 백테스트 실행 중...")
    qqq = get_historical_data("QQQ")
    tqqq = get_historical_data("TQQQ")
    vix = get_historical_data("^VIX")
    vix1d = get_historical_data("^VIX1D")
    vxn = get_historical_data("^VXN")
    skew = get_historical_data("^SKEW")
    hyg = get_historical_data("HYG")
    tlt = get_historical_data("TLT")

    if qqq.empty or tqqq.empty or vix.empty:
        print("🚨 핵심 데이터 수집 실패로 중단합니다.")
        return

    df = pd.DataFrame({
        'QQQ_Close': qqq['Close'], 'QQQ_Open': qqq['Open'],
        'QQQ_High': qqq['High'], 'QQQ_Low': qqq['Low'], 'QQQ_Vol': qqq['Volume'],
        'TQQQ_Close': tqqq['Close'], 'TQQQ_Open': tqqq['Open']
    })
    df['VIX'] = vix['Close']
    df['VIX1D'] = vix1d['Close'] if not vix1d.empty else vix['Close']
    df['VXN'] = vxn['Close'] if not vxn.empty else vix['Close'] * 1.1
    df['SKEW'] = skew['Close'] if not skew.empty else 125.0
    df['HYG'] = hyg['Close'] if not hyg.empty else 75.0
    df['TLT'] = tlt['Close'] if not tlt.empty else 90.0

    df = df.ffill().bfill().dropna()

    sma5 = df['QQQ_Close'].rolling(5).mean()
    sma20 = df['QQQ_Close'].rolling(20).mean()
    sma50 = df['QQQ_Close'].rolling(50).mean()
    sma200 = df['QQQ_Close'].rolling(200, min_periods=20).mean()
    disp200 = (df['QQQ_Close'] / sma200) * 100
    
    ema12 = df['QQQ_Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['QQQ_Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    init_cash = 10000.0
    cash = 0.0
    daily_sgov_rate = (1 + 0.05) ** (1/252) - 1
    fee_rate = 0.0012

    portfolio_values = []
    benchmark_values = []
    dates_list = []

    # 90:5:5 세팅 (초기 $10,000)
    pos_qqq_qty = (init_cash * 0.90) / df['QQQ_Open'].iloc[20]
    pos_kr2x_qty = (init_cash * 0.05) / df['QQQ_Open'].iloc[20]
    pos_tqqq_qty = (init_cash * 0.05) / df['TQQQ_Open'].iloc[20]

    for i in range(20, len(df) - 1):
        d_next = df.index[i+1]
        c_price = df['QQQ_Close'].iloc[i]

        # 정밀 1단계 매크로 점수
        score = 0.0
        if disp200.iloc[i] > 110: score += 15.0
        elif disp200.iloc[i] > 105: score += 7.5

        if df['VXN'].iloc[i] > 25: score += 15.0
        elif df['VXN'].iloc[i] > 20: score += 7.0

        if df['SKEW'].iloc[i] > 145: score += 15.0
        elif df['SKEW'].iloc[i] > 138: score += 7.0

        ratio_0dte = df['VIX1D'].iloc[i] / df['VIX'].iloc[i]
        if ratio_0dte >= 1.35: score += 15.0
        elif ratio_0dte >= 1.20: score += 7.5

        below_5 = c_price < sma5.iloc[i]
        below_20 = c_price < sma20.iloc[i]
        macd_dead = macd.iloc[i] < signal.iloc[i]

        # 레버리지 타겟 비율 결정 (QQQ 90%는 안정 유지)
        target_lev_ratio = 1.0 # 100% 유지 기준

        if score >= 45.0:
            if below_5 and macd_dead and below_20:
                target_lev_ratio = 0.0 # 전량 탈출 (SGOV 파킹)
            elif below_5 and macd_dead:
                target_lev_ratio = 0.4 # 60% 탈출
            elif below_5:
                target_lev_ratio = 0.7 # 30% 탈출
        elif (df['SKEW'].iloc[i] >= 145 and ratio_0dte >= 1.35) or (c_price / sma20.iloc[i] >= 1.08):
            target_lev_ratio = 0.85 # 15% 선제 익절

        # 익일 시초가 레버리지 리밸런싱 집행
        next_qqq_open = df['QQQ_Open'].iloc[i+1]
        next_tqqq_open = df['TQQQ_Open'].iloc[i+1]

        # 레버리지 총 목표 평가액 산정
        base_lev_val = ((init_cash * 0.05) / df['QQQ_Open'].iloc[20] * next_qqq_open * 2) + ((init_cash * 0.05) / df['TQQQ_Open'].iloc[20] * next_tqqq_open)
        target_kr2x_val = (base_lev_val / 2) * target_lev_ratio
        target_tqqq_val = (base_lev_val / 2) * target_lev_ratio

        curr_kr2x_val = pos_kr2x_qty * next_qqq_open * 2
        curr_tqqq_val = pos_tqqq_qty * next_tqqq_open

        # KR2X 조정
        if abs(curr_kr2x_val - target_kr2x_val) > (base_lev_val * 0.05):
            if curr_kr2x_val > target_kr2x_val:
                diff = curr_kr2x_val - target_kr2x_val
                cash += diff * (1 - fee_rate)
                pos_kr2x_qty = (target_kr2x_val / 2) / next_qqq_open
            elif curr_kr2x_val < target_kr2x_val and cash > 0:
                diff = min(cash, target_kr2x_val - curr_kr2x_val)
                cash -= diff
                pos_kr2x_qty += ((diff / 2) * (1 - fee_rate)) / next_qqq_open

        # TQQQ 조정
        if abs(curr_tqqq_val - target_tqqq_val) > (base_lev_val * 0.05):
            if curr_tqqq_val > target_tqqq_val:
                diff = curr_tqqq_val - target_tqqq_val
                cash += diff * (1 - fee_rate)
                pos_tqqq_qty = target_tqqq_val / next_tqqq_open
            elif curr_tqqq_val < target_tqqq_val and cash > 0:
                diff = min(cash, target_tqqq_val - curr_tqqq_val)
                cash -= diff
                pos_tqqq_qty += (diff * (1 - fee_rate)) / next_tqqq_open

        cash *= (1 + daily_sgov_rate)

        # 익일 종가 평가액
        next_qqq_close = df['QQQ_Close'].iloc[i+1]
        next_tqqq_close = df['TQQQ_Close'].iloc[i+1]

        final_val = (pos_qqq_qty * next_qqq_close) + (pos_kr2x_qty * next_qqq_close * 2) + (pos_tqqq_qty * next_tqqq_close) + cash
        bench_val = init_cash * (next_qqq_close / df['QQQ_Close'].iloc[20])

        portfolio_values.append(final_val)
        benchmark_values.append(bench_val)
        dates_list.append(d_next)

    res_df = pd.DataFrame({'Strategy': portfolio_values, 'Benchmark_QQQ': benchmark_values}, index=pd.to_datetime(dates_list))

    total_days = len(res_df)
    cagr_strat = ((res_df['Strategy'].iloc[-1] / init_cash) ** (252 / total_days) - 1) * 100
    cagr_bench = ((res_df['Benchmark_QQQ'].iloc[-1] / init_cash) ** (252 / total_days) - 1) * 100
    mdd_strat = ((res_df['Strategy'] / res_df['Strategy'].cummax()) - 1).min() * 100
    mdd_bench = ((res_df['Benchmark_QQQ'] / res_df['Benchmark_QQQ'].cummax()) - 1).min() * 100

    msg = (
        f"📊 <b>[정상화 90:5:5 전략 실데이터 백테스트 완료]</b>\n"
        f"📅 기간: {dates_list[0].strftime('%Y-%m-%d')} ~ {dates_list[-1].strftime('%Y-%m-%d')} ({total_days}거래일)\n"
        f"────────────────\n"
        f"• <b>우리 90:5:5 전략:</b> 최종 <b>${res_df['Strategy'].iloc[-1]:,.2f}</b> | CAGR: <b>{cagr_strat:+.2f}%</b> | MDD: <b>{mdd_strat:.2f}%</b>\n"
        f"• <b>QQQ 단순보유:</b> 최종 <b>${res_df['Benchmark_QQQ'].iloc[-1]:,.2f}</b> | CAGR: <b>{cagr_bench:+.2f}%</b> | MDD: <b>{mdd_bench:.2f}%</b>\n"
        f"────────────────\n"
        f"👉 <i>QQQ 90% 코어 복리 보존 + 레버리지 위기 탈출 정상화 결과입니다.</i>"
    )
    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_backtest()
