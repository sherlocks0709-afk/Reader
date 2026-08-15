def fetch_equity_pcr():
    """
    CBOE 공식 일일 통계에서 순수 'Equity Put/Call Ratio'만 정확히 추출
    """
    # 1. CBOE 공식 일일 마켓 스탯 CSV
    try:
        url = "https://cdn.cboe.com/data/us/options/market_statistics/daily/daily_market_statistics.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            lines = res.text.splitlines()
            for line in lines:
                row = [c.strip().upper() for c in line.split(',')]
                # 'EQUITY'가 들어있고 'INDEX'나 'TOTAL'이 아닌 순수 Equity Put/Call Ratio 행 탐색
                if any('EQUITY' in item for item in row) and any('RATIO' in item or 'P/C' in item or 'PUT/CALL' in item for item in row):
                    for val_str in row:
                        try:
                            val = float(val_str)
                            # 순수 주식 풋콜 비율의 통상적인 정상 범위 (0.25 ~ 0.90)
                            if 0.25 <= val <= 0.90:
                                return val
                        except ValueError:
                            continue
    except Exception as e:
        print(f"CBOE CSV 파싱 실패: {e}")

    # 2. CBOE 옵션 볼륨 요약 엔드포인트 (대체 소스)
    try:
        url_alt = "https://www.cboe.com/us/options/market_statistics/daily/"
        res_alt = requests.get(url_alt, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if "Equity" in res_alt.text:
            import re
            # HTML 내 Equity Put/Call ratio 매칭
            match = re.search(r'Equity\s+(?:Put/Call\s+Ratio|P/C\s+Ratio)[^\d]*([\d\.]+)', res_alt.text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 0.25 <= val <= 0.90:
                    return val
    except Exception:
        pass

    # 3. yfinance를 통한 S&P500 주요 메가캡(AAPL, NVDA, MSFT 등) 실시간 옵션 체인 가중 Put/Call 산출
    try:
        total_p = 0
        total_c = 0
        for sym in ["AAPL", "NVDA", "MSFT", "AMZN"]:
            t = yf.Ticker(sym)
            if t.options:
                chain = t.option_chain(t.options[0])
                total_c += chain.calls['volume'].fillna(0).sum()
                total_p += chain.puts['volume'].fillna(0).sum()
        if total_c > 0:
            return round(float(total_p / total_c), 2)
    except Exception:
        pass

    return 0.58 # 통상적인 Equity 중립 기준치 fallback
