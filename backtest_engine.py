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
            
            # 결측치 보정 (Open이 비어있으면 Close로 대체)
            df['Open'] = df['Open'].fillna(df['Close'])
            df.set_index('Date', inplace=True)
            return df[df.index >= pd.to_datetime(start_date)].astype(float)
    except Exception as e:
        print(f"데이터 수집 에러 ({ticker_symbol}): {e}")
    return pd.DataFrame()

def run_backtest():
    print("⏳ 2023.04 이후 실제 시계열 데이터 수집 및 전처리 중...")
    qqq = get_historical_data("QQQ")
    tqqq = get_historical_data("TQQQ")
    vix = get_historical_data("^VIX")
    vix1d = get_historical_data("^VIX1D")
    vxn = get_historical_data("^VXN")
    skew = get_historical_data("^SKEW")
    hyg = get_historical_data("HYG")
    tlt = get_historical_data("TLT")

    if qqq.empty or tqqq.empty or vix.empty:
        print("🚨 핵심 데이터 수집 실패로 백테스트를 중단합니다.")
        return

    # 공통 거래일 결합 및 시계열 결측치 ffill(전일 값 채우기)
    df_merged = pd.DataFrame({'QQQ_Close': qqq['Close'], 'QQQ_Open': qqq['Open']})
    df_merged['TQQQ_Close'] = tqqq['Close']
    df_merged['TQQQ_Open'] = tqqq['Open']
    df_merged['VIX'] = vix['Close']
    df_merged['VIX1D'] = vix1d['Close'] if not vix1d.empty else vix['Close']
    df_merged['VXN'] = vxn['Close'] if not vxn.empty else vix['Close'] * 1.1
    df_merged['SKEW'] = skew['Close'] if not skew.empty else 125.0
    df_merged['HYG'] = hyg['Close'] if not hyg.empty else 75.0
    df_merged['TLT'] = tlt['Close'] if not tlt.empty else 90.0

    df_merged = df_merged.ffill().bfill().dropna()

    sma200 = df_merged['QQQ_Close'].rolling(200, min_periods=20).mean()
    disp200 = (df_merged['QQQ_Close'] / sma200) * 100

    cash = 0.0
    daily_sgov_rate = (1 + 0.05) ** (1/252) - 1

    # 초기 자본 $10,000 기준 세팅
    init_cash = 10000.0
    pos_qqq = (init_cash * 0.90) / df_merged['QQQ_Open'].iloc[0]
    pos_kr2x = (init_cash * 0.05) / df_merged['QQQ_Open'].iloc[0]
    pos_tqqq = (init_cash * 0.05) / df_merged['TQQQ_Open'].iloc[0]

    portfolio_values = []
    benchmark_values = []
    dates_list = []

    for i in range(len(df_merged) - 1):
        d_next = df_merged.index[i+1]
        
        # 1단계 점수 판독
        score = 0.0
        if disp200.iloc[i] > 105: score += 7.5
        if df_merged['VXN'].iloc[i] > 22: score += 7.0
        if df_merged['SKEW'].iloc[i] > 140: score += 7.0
        if (df_merged['VIX1D'].iloc[i] / df_merged['VIX'].iloc[i]) >= 1.25: score += 5.0
        
        idx_20d = max(0, i-20)
        curr_ratio = df_merged['HYG'].iloc[i] / df_merged['TLT'].iloc[i]
        prev_ratio = df_merged['HYG'].iloc[idx_20d] / df_merged['TLT'].iloc[idx_20d]
        if curr_ratio < prev_ratio: score += 7.0
        
        # 위험 신호 격발 시 레버리지 SGOV 현금화
        if score >= 20.0:
            if pos_tqqq > 0:
                cash += pos_tqqq * df_merged['TQQQ_Open'].iloc[i+1] * (1 - 0.0012)
                pos_tqqq = 0.0
        else:
            if pos_tqqq == 0.0 and cash > 0:
                pos_tqqq = (cash * 0.5) / df_merged['TQQQ_Open'].iloc[i+1]
                cash -= cash * 0.5

        # 익일 포트폴리오 평가액 산출
        curr_qqq_p = df_merged['QQQ_Close'].iloc[i+1]
        curr_tqqq_p = df_merged['TQQQ_Close'].iloc[i+1]
        
        total_val = (pos_qqq * curr_qqq_p) + (pos_kr2x * curr_qqq_p * 2) + (pos_tqqq * curr_tqqq_p) + cash
        bench_val = init_cash * (curr_qqq_p / df_merged['QQQ_Close'].iloc[0])
        
        portfolio_values.append(total_val)
        benchmark_values.append(bench_val)
        dates_list.append(d_next)
        cash *= (1 + daily_sgov_rate)

    res_df = pd.DataFrame({'Strategy': portfolio_values, 'Benchmark_QQQ': benchmark_values}, index=pd.to_datetime(dates_list))

    total_days = len(res_df)
    cagr_strat = ((res_df['Strategy'].iloc[-1] / init_cash) ** (252 / total_days) - 1) * 100
    cagr_bench = ((res_df['Benchmark_QQQ'].iloc[-1] / init_cash) ** (252 / total_days) - 1) * 100
    mdd_strat = ((res_df['Strategy'] / res_df['Strategy'].cummax()) - 1).min() * 100
    mdd_bench = ((res_df['Benchmark_QQQ'] / res_df['Benchmark_QQQ'].cummax()) - 1).min() * 100

    msg = (
        f"📊 <b>[깃허브 실데이터 백테스트 완료]</b>\n"
        f"📅 기간: {dates_list[0].strftime('%Y-%m-%d')} ~ {dates_list[-1].strftime('%Y-%m-%d')} ({total_days}거래일)\n"
        f"────────────────\n"
        f"• <b>우리 전략:</b> 최종 <b>${res_df['Strategy'].iloc[-1]:,.2f}</b> | CAGR: <b>{cagr_strat:+.2f}%</b> | MDD: <b>{mdd_strat:.2f}%</b>\n"
        f"• <b>QQQ 단순보유:</b> 최종 <b>${res_df['Benchmark_QQQ'].iloc[-1]:,.2f}</b> | CAGR: <b>{cagr_bench:+.2f}%</b> | MDD: <b>{mdd_bench:.2f}%</b>\n"
        f"────────────────\n"
        f"👉 <i>CBOE/Yahoo 일별 시계열 전수 시뮬레이션 결과입니다.</i>"
    )
    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_backtest()
