import datetime
import os
import io
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

def get_historical_data(ticker, start_date="2023-04-20"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=4y&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers).json()
    res_data = res['chart']['result'][0]
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
    }).dropna()
    df.set_index('Date', inplace=True)
    return df[df.index >= pd.to_datetime(start_date)].astype(float)

def run_backtest():
    print("⏳ 2023.04 이후 실제 시계열 데이터 수집 중...")
    qqq = get_historical_data("QQQ")
    tqqq = get_historical_data("TQQQ")
    vix = get_historical_data("^VIX")
    vix1d = get_historical_data("^VIX1D")
    vxn = get_historical_data("^VXN")
    skew = get_historical_data("^SKEW")
    hyg = get_historical_data("HYG")
    tlt = get_historical_data("TLT")

    common_dates = qqq.index.intersection(tqqq.index).intersection(vix.index).intersection(vix1d.index)
    qqq = qqq.loc[common_dates]
    tqqq = tqqq.loc[common_dates]
    vix = vix.loc[common_dates]
    vix1d = vix1d.loc[common_dates]
    vxn = vxn.loc[common_dates]
    skew = skew.loc[common_dates]
    hyg = hyg.loc[common_dates]
    tlt = tlt.loc[common_dates]

    sma200 = qqq['Close'].rolling(200).mean()
    disp200 = (qqq['Close'] / sma200) * 100

    cash = 0.0
    daily_sgov_rate = (1 + 0.05) ** (1/252) - 1

    pos_qqq = 9000.0 / qqq['Open'].iloc[20]
    pos_kr2x = 500.0 / qqq['Open'].iloc[20]
    pos_tqqq = 500.0 / tqqq['Open'].iloc[20]

    portfolio_values = []
    benchmark_values = []
    dates_list = []

    for i in range(20, len(common_dates)-1):
        d_next = common_dates[i+1]
        
        # 1단계 점수 판독
        score = 0.0
        if disp200.iloc[i] > 105: score += 7.5
        if vxn['Close'].iloc[i] > 22: score += 7.0
        if skew['Close'].iloc[i] > 140: score += 7.0
        if (vix1d['Close'].iloc[i] / vix['Close'].iloc[i]) >= 1.25: score += 5.0
        if (hyg['Close'].iloc[i] / tlt['Close'].iloc[i]) < (hyg['Close'].iloc[i-20] / tlt['Close'].iloc[i-20]): score += 7.0
        
        # 위험 감지 시 SGOV 현금화
        if score >= 20.0:
            if pos_tqqq > 0:
                cash += pos_tqqq * tqqq['Open'].iloc[i+1] * (1 - 0.0012)
                pos_tqqq = 0.0
        else:
            if pos_tqqq == 0.0 and cash > 0:
                pos_tqqq = (cash * 0.5) / tqqq['Open'].iloc[i+1]
                cash -= cash * 0.5

        total_val = (pos_qqq * qqq['Close'].iloc[i+1]) + (pos_kr2x * qqq['Close'].iloc[i+1] * 2) + (pos_tqqq * tqqq['Close'].iloc[i+1]) + cash
        bench_val = 10000.0 * (qqq['Close'].iloc[i+1] / qqq['Close'].iloc[20])
        
        portfolio_values.append(total_val)
        benchmark_values.append(bench_val)
        dates_list.append(d_next)
        cash *= (1 + daily_sgov_rate)

    res_df = pd.DataFrame({'Strategy': portfolio_values, 'Benchmark_QQQ': benchmark_values}, index=pd.to_datetime(dates_list))

    cagr_strat = ((res_df['Strategy'].iloc[-1] / 10000.0) ** (252 / len(res_df)) - 1) * 100
    cagr_bench = ((res_df['Benchmark_QQQ'].iloc[-1] / 10000.0) ** (252 / len(res_df)) - 1) * 100
    mdd_strat = ((res_df['Strategy'] / res_df['Strategy'].cummax()) - 1).min() * 100
    mdd_bench = ((res_df['Benchmark_QQQ'] / res_df['Benchmark_QQQ'].cummax()) - 1).min() * 100

    msg = (
        f"📊 <b>[깃허브 백테스트 실데이터 검증 완료]</b>\n"
        f"📅 기간: {dates_list[0].strftime('%Y-%m-%d')} ~ {dates_list[-1].strftime('%Y-%m-%d')} ({len(dates_list)}거래일)\n"
        f"────────────────\n"
        f"• <b>우리 전략:</b> 최종 <b>${res_df['Strategy'].iloc[-1]:,.2f}</b> | CAGR: <b>{cagr_strat:+.2f}%</b> | MDD: <b>{mdd_strat:.2f}%</b>\n"
        f"• <b>QQQ 단순보유:</b> 최종 <b>${res_df['Benchmark_QQQ'].iloc[-1]:,.2f}</b> | CAGR: <b>{cagr_bench:+.2f}%</b> | MDD: <b>{mdd_bench:.2f}%</b>\n"
        f"────────────────\n"
        f"👉 <i>실제 CBOE/Yahoo 일별 시계열 기반 전수 검증 결과입니다.</i>"
    )
    print(msg)
    send_telegram_result(msg)

if __name__ == "__main__":
    run_backtest()
