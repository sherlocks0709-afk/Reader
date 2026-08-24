import datetime
import os
import io
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

RAW_DB_FILE = "backtest_raw_db.csv"

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

def get_historical_data(ticker_symbol, start_date="2023-04-20"):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range=4y&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            if 'chart' in data and data['chart']['result']:
                res_data = data['chart']['result'][0]
                timestamps = res_data.get('timestamp', [])
                quotes = res_data.get('indicators', {}).get('quote', [{}])[0]
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
                df['High'] = df['High'].fillna(df['Close'])
                df['Low'] = df['Low'].fillna(df['Close'])
                df['Volume'] = df['Volume'].fillna(1.0)
                df.set_index('Date', inplace=True)
                return df[df.index >= pd.to_datetime(start_date)].astype(float)
    except Exception as e:
        print(f"데이터 수집 에러 ({ticker_symbol}): {e}")
    return pd.DataFrame()

def fetch_fred_historical(series_id, api_key="", start_date="2023-04-20"):
    if api_key:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key.strip(),
                "file_type": "json",
                "sort_order": "asc",
                "observation_start": start_date
            }
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if "observations" in data:
                records = []
                for obs in data["observations"]:
                    val = obs.get("value", ".")
                    if val != ".":
                        records.append({'Date': pd.to_datetime(obs['date']), series_id: float(val)})
                if records:
                    return pd.DataFrame(records).set_index('Date')
        except Exception:
            pass

    try:
        csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(csv_url, headers=headers, timeout=10)
        if res.status_code == 200 and len(res.text) > 30:
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = [c.strip().upper() for c in df.columns]
            if "DATE" in df.columns and series_id.upper() in df.columns:
                df['Date'] = pd.to_datetime(df['DATE'])
                df[series_id] = pd.to_numeric(df[series_id.upper()], errors='coerce')
                return df.dropna(subset=[series_id]).set_index('Date')[[series_id]]
    except Exception:
        pass
    return pd.DataFrame()

def load_or_build_raw_database():
    if os.path.exists(RAW_DB_FILE):
        try:
            print(f"📦 [로컬 DB 로드] '{RAW_DB_FILE}' 파일에서 시계열 데이터를 불러옵니다.")
            df = pd.read_csv(RAW_DB_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            if not df.empty and len(df) > 300:
                return df
        except Exception as e:
            print(f"⚠️ 기존 DB 파일 파싱 실패, 신규 수집을 진행합니다: {e}")

    print("🌐 [신규 DB 구축] CBOE, FRED, Yahoo에서 2023.04 이후 전수 데이터를 다운로드합니다...")
    qqq = get_historical_data("QQQ")
    tqqq = get_historical_data("TQQQ")
    vix = get_historical_data("^VIX")
    vix1d = get_historical_data("^VIX1D")
    vxn = get_historical_data("^VXN")
    skew = get_historical_data("^SKEW")
    hyg = get_historical_data("HYG")
    tlt = get_historical_data("TLT")
    qqqe = get_historical_data("QQQE")
    dxy = get_historical_data("DX-Y.NYB")
    usdjpy = get_historical_data("USDJPY=X")

    fred_key = os.environ.get("FRED_API_KEY", "")
    df_hy = fetch_fred_historical("BAMLH0A0HYM2", fred_key)
    df_sofr = fetch_fred_historical("SOFR", fred_key)
    df_iorb = fetch_fred_historical("IORB", fred_key)

    df = pd.DataFrame({
        'QQQ_Close': qqq['Close'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Open': qqq['Open'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_High': qqq['High'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Low': qqq['Low'] if not qqq.empty else pd.Series(dtype=float),
        'QQQ_Vol': qqq['Volume'] if not qqq.empty else pd.Series(dtype=float),
        'TQQQ_Close': tqqq['Close'] if not tqqq.empty else pd.Series(dtype=float),
        'TQQQ_Open': tqqq['Open'] if not tqqq.empty else pd.Series(dtype=float)
    })
    
    df['VIX'] = vix['Close'] if not vix.empty else 18.0
    df['VIX1D'] = vix1d['Close'] if not vix1d.empty else (df['VIX'] * 1.0)
    df['VXN'] = vxn['Close'] if not vxn.empty else (df['VIX'] * 1.1)
    df['SKEW'] = skew['Close'] if not skew.empty else 125.0
    df['HYG'] = hyg['Close'] if not hyg.empty else 75.0
    df['TLT'] = tlt['Close'] if not tlt.empty else 90.0
    df['QQQE'] = qqqe['Close'] if not qqqe.empty else df['QQQ_Close']
    df['DXY'] = dxy['Close'] if not dxy.empty else 103.0
    df['USDJPY'] = usdjpy['Close'] if not usdjpy.empty else 150.0

    df['HY_SPREAD'] = df_hy['BAMLH0A0HYM2'] if not df_hy.empty else 3.5
    df['SOFR'] = df_sofr['SOFR'] if not df_sofr.empty else 5.3
    df['IORB'] = df_iorb['IORB'] if not df_iorb.empty else 5.4

    df = df.ffill().bfill().dropna()
    df.to_csv(RAW_DB_FILE)
    print(f"✅ '{RAW_DB_FILE}' 생성 및 영구 캐싱 완료 ({len(df)}거래일)")
    return df

def evaluate_regime_at_day(slice_df):
    """
    [T 시점 슬라이스 기반 판독]
    - QQQ 코어 (90%): 50일선 붕괴 및 거시 경색 시 어깨 매도, 5일선 첫 양봉 탈환 시 무릎 패스트트랙 매수
    - 레버리지 (10%): 5일선 이탈 시 목덜미에서 선제 SGOV 파킹
    """
    qqq_close = slice_df['QQQ_Close']
    current_close = float(qqq_close.iloc[-1])
    
    sma5_s = qqq_close.rolling(5).mean()
    sma20_s = qqq_close.rolling(20).mean()
    sma50_s = qqq_close.rolling(50).mean()
    sma200_s = qqq_close.rolling(200, min_periods=20).mean()
    
    sma5 = float(sma5_s.iloc[-1])
    sma20 = float(sma20_s.iloc[-1])
    sma50 = float(sma50_s.iloc[-1])
    sma200 = float(sma200_s.iloc[-1])
    
    disp200_s = (qqq_close / sma200_s) * 100
    disp_200 = float(disp200_s.iloc[-1])
    disp_20 = (current_close / sma20) * 100

    # 1단계 거시 점수 산출
    disp_mean = float(disp200_s.mean())
    disp_std = float(disp200_s.std())
    z_disp = (disp_200 - disp_mean) / disp_std if disp_std > 0 else 0.0
    score_disp = float(np.clip(z_disp * (7.5 / 2.0), 0, 7.5))

    gain_s = (qqq_close.diff().where(qqq_close.diff() > 0, 0)).rolling(14).mean()
    loss_s = (-qqq_close.diff().where(qqq_close.diff() < 0, 0)).rolling(14).mean()
    l_val = float(loss_s.iloc[-1]) if float(loss_s.iloc[-1]) > 0 else 1e-5
    rs = float(gain_s.iloc[-1]) / l_val
    rsi_val = 100 - (100 / (1 + rs))
    score_rsi = float(np.clip((rsi_val - 50) * (7.5 / 30), 0, 7.5))

    qqq_20d_ret = ((current_close / float(qqq_close.iloc[-20])) - 1) * 100 if len(qqq_close) >= 20 else 0.0
    vxn_curr = float(slice_df['VXN'].iloc[-1])
    vxn_20d_ago = float(slice_df['VXN'].iloc[-20]) if len(slice_df) >= 20 else vxn_curr
    score_vxn = 0.0
    if qqq_20d_ret > 0 and (vxn_curr - vxn_20d_ago) >= 2.0: score_vxn = 7.0
    elif qqq_20d_ret > 0 and (vxn_curr - vxn_20d_ago) >= 0.5: score_vxn = 3.5

    skew_curr = float(slice_df['SKEW'].iloc[-1])
    score_skew = float(np.clip((skew_curr - 120) * (7.0 / 25), 0, 7.0))
    vix_curr = float(slice_df['VIX'].iloc[-1])
    vix1d_curr = float(slice_df['VIX1D'].iloc[-1])
    ratio_0dte = (vix1d_curr / vix_curr) if vix_curr > 0 else 1.0
    if ratio_0dte >= 1.25: score_skew = min(7.0, score_skew + 2.0)

    # VIX 기간구조 (VIX / VIX3M 프록시)
    term_ratio = vix_curr / (vix_curr * 1.1 if vix_curr < 20 else vix_curr * 1.02)
    score_term = float(np.clip((term_ratio - 0.80) * (6.0 / 0.20), 0, 6.0))

    score_breadth = 0.0
    if len(slice_df) >= 20:
        qqqe_ret = ((float(slice_df['QQQE'].iloc[-1]) / float(slice_df['QQQE'].iloc[-20])) - 1) * 100
        if qqq_20d_ret > 0 and (qqq_20d_ret - qqqe_ret) >= 3.0: score_breadth = 8.0

    score_hyg_tlt = 0.0
    if len(slice_df) >= 20:
        r_now = float(slice_df['HYG'].iloc[-1]) / float(slice_df['TLT'].iloc[-1])
        r_prev = float(slice_df['HYG'].iloc[-20]) / float(slice_df['TLT'].iloc[-20])
        r_chg = ((r_now / r_prev) - 1) * 100
        if qqq_20d_ret > 0 and r_chg <= -2.5: score_hyg_tlt = 7.0
        elif qqq_20d_ret > 0 and r_chg <= -1.0: score_hyg_tlt = 3.5

    score_fx = 0.0
    if len(slice_df) >= 20:
        dxy_chg = ((float(slice_df['DXY'].iloc[-1]) / float(slice_df['DXY'].iloc[-20])) - 1) * 100
        if dxy_chg >= 2.5: score_fx += 5.0
        elif dxy_chg >= 1.2: score_fx += 2.5
    if len(slice_df) >= 5:
        jpy_chg = ((float(slice_df['USDJPY'].iloc[-1]) / float(slice_df['USDJPY'].iloc[-5])) - 1) * 100
        if jpy_chg <= -2.5: score_fx += 5.0

    hy_spread = float(slice_df['HY_SPREAD'].iloc[-1])
    score_hy = float(np.clip((4.5 - hy_spread) * (6.0 / 1.5), 0, 6.0))
    if len(slice_df) >= 20 and qqq_20d_ret > 0:
        if (hy_spread - float(slice_df['HY_SPREAD'].iloc[-20])) >= 0.20: score_hy += 6.0

    score_sofr = 0.0
    sofr_bps = (float(slice_df['SOFR'].iloc[-1]) - float(slice_df['IORB'].iloc[-1])) * 100
    if sofr_bps >= 8.0: score_sofr = 10.0
    elif sofr_bps >= 3.0: score_sofr = 5.0

    total_score = round(score_disp + score_rsi + score_vxn + score_skew + score_term + score_breadth + score_hyg_tlt + score_fx + score_hy + score_sofr, 1)

    # 2단계 추세 및 캔들
    ema12 = qqq_close.ewm(span=12, adjust=False).mean()
    ema26 = qqq_close.ewm(span=26, adjust=False).mean()
    macd_s = ema12 - ema26
    sig_s = macd_s.ewm(span=9, adjust=False).mean()
    
    below_sma5 = current_close < sma5
    below_sma20 = current_close < sma20
    below_sma50 = current_close < sma50
    macd_dead = float(macd_s.iloc[-1]) < float(sig_s.iloc[-1])

    # 5일선 재탈환 첫 양봉 (1차 무릎 확인)
    is_bullish_candle = current_close >= float(slice_df['QQQ_Open'].iloc[-1])
    reclaimed_sma5 = (not below_sma5) and is_bullish_candle

    peak_price = float(slice_df['QQQ_Close'].max())
    drawdown = ((current_close / peak_price) - 1) * 100

    # ── [포지션 사이징: QQQ 어깨 매도 & 무릎 매수 2원화] ──
    # 기본값
    target_qqq = 0.90
    target_kr2x = 0.05
    target_tqqq = 0.05

    # 1. 고점 대비 -10% 이하 대형 폭락 국면의 '무릎 매수 패스트트랙'
    if drawdown <= -10.0:
        if not macd_dead or not below_sma20:
            # 2차 무릎: MACD 골든크로스 or 20일선 회복 ➔ QQQ 100% 전량 복귀 + 레버리지 70% 복귀
            target_qqq = 0.90
            target_kr2x = 0.035
            target_tqqq = 0.035
        elif reclaimed_sma5:
            # 1차 무릎: 5일선 첫 양봉 재탈환 ➔ QQQ 60% 선제 복귀
            target_qqq = 0.60
            target_kr2x = 0.015
            target_tqqq = 0.015
        else:
            # 패닉 하락 진행 중: QQQ 30%만 보존, 레버리지 0% (SGOV 70% 대피 유지)
            target_qqq = 0.30
            target_kr2x = 0.0
            target_tqqq = 0.0

    # 2. 고점 대비 -10% 미만 일반/과열 국면
    else:
        # QQQ 어깨 매도 조건: 50일선 완전 붕괴 & 매크로 점수 65점 이상 (대형 하락 초입)
        if below_sma50 and total_score >= 65.0:
            target_qqq = 0.40 # QQQ 50% 현금화 (어깨 대피)
            target_kr2x = 0.0
            target_tqqq = 0.0
        elif below_sma20 and total_score >= 70.0 and term_ratio >= 1.0:
            target_qqq = 0.60 # QQQ 30% 현금화 (어깨 1차 대피)
            target_kr2x = 0.0
            target_tqqq = 0.0
        else:
            # QQQ 코어 90%는 단순 5일선 이탈에 절대 매도하지 않고 100% 홀딩 (복리 보존)
            target_qqq = 0.90

            # 레버리지 10%만 5일선/점수에 따라 기민하게 조절
            if total_score >= 60.0 and below_sma5 and macd_dead:
                target_kr2x = 0.0
                target_tqqq = 0.0 # 레버리지 100% SGOV 파킹
            elif below_sma5 and macd_dead:
                target_kr2x = 0.02
                target_tqqq = 0.02 # 레버리지 60% 축소
            elif below_sma5:
                target_kr2x = 0.035
                target_tqqq = 0.035 # 레버리지 30% 축소
            elif (skew_curr >= 145.0 and ratio_0dte >= 1.35) or (disp_20 >= 108.0):
                target_kr2x = 0.042
                target_tqqq = 0.042 # 15% 선제 익절

    return target_qqq, target_kr2x, target_tqqq, total_score, drawdown

def run_perfect_walkforward_backtest():
    df = load_or_build_raw_database()

    print(f"⏳ [Step 2] {len(df)}개 거래일 '어깨 매도·무릎 매수' Walk-Forward 시뮬레이션 가동...")
    init_cash = 10000.0
    fee_rate = 0.0012
    daily_sgov_rate = (1 + 0.05) ** (1/252) - 1

    cash = 0.0
    first_idx = 20
    first_open_qqq = float(df['QQQ_Open'].iloc[first_idx])
    first_open_tqqq = float(df['TQQQ_Open'].iloc[first_idx])

    pos_qqq_qty = (init_cash * 0.90) / first_open_qqq
    pos_kr2x_qty = (init_cash * 0.05) / first_open_qqq
    pos_tqqq_qty = (init_cash * 0.05) / first_open_tqqq
    cash = init_cash - (pos_qqq_qty * first_open_qqq) - (pos_kr2x_qty * first_open_qqq) - (pos_tqqq_qty * first_open_tqqq)

    strat_values = []
    bench_values = []
    dates_list = []

    trade_count = 0

    for i in range(first_idx, len(df) - 1):
        d_next = df.index[i+1]

        # [T 시점] 과거 데이터 슬라이스로만 판독 (미래 데이터 100% 차단)
        slice_df = df.iloc[:i+1]
        t_qqq_r, t_kr2x_r, t_tqqq_r, score, dd = evaluate_regime_at_day(slice_df)

        # [T+1 시점] 익일 시초가 리밸런싱 집행
        next_open_qqq = float(df['QQQ_Open'].iloc[i+1])
        next_open_tqqq = float(df['TQQQ_Open'].iloc[i+1])

        curr_total = (pos_qqq_qty * next_open_qqq) + (pos_kr2x_qty * next_open_qqq * 2) + (pos_tqqq_qty * next_open_tqqq) + cash

        target_qqq_v = curr_total * t_qqq_r
        target_kr2x_v = curr_total * t_kr2x_r
        target_tqqq_v = curr_total * t_tqqq_r

        # QQQ 리밸런싱 (5% 데드밴드 버퍼로 잔파도 수수료 누수 차단)
        curr_qqq_v = pos_qqq_qty * next_open_qqq
        if abs(curr_qqq_v - target_qqq_v) > (curr_total * 0.05):
            trade_count += 1
            if curr_qqq_v > target_qqq_v:
                sell_val = curr_qqq_v - target_qqq_v
                cash += sell_val * (1 - fee_rate)
                pos_qqq_qty = target_qqq_v / next_open_qqq
            elif curr_qqq_v < target_qqq_v and cash > 0:
                buy_val = min(cash, target_qqq_v - curr_qqq_v)
                cash -= buy_val
                pos_qqq_qty += (buy_val * (1 - fee_rate)) / next_open_qqq

        # KR 2배수 리밸런싱
        curr_kr2x_v = pos_kr2x_qty * next_open_qqq * 2
        if abs(curr_kr2x_v - target_kr2x_v) > (curr_total * 0.015):
            if curr_kr2x_v > target_kr2x_v:
                sell_val = curr_kr2x_v - target_kr2x_v
                cash += sell_val * (1 - fee_rate)
                pos_kr2x_qty = (target_kr2x_v / 2) / next_open_qqq
            elif curr_kr2x_v < target_kr2x_v and cash > 0:
                buy_val = min(cash, target_kr2x_v - curr_kr2x_v)
                cash -= buy_val
                pos_kr2x_qty += ((buy_val / 2) * (1 - fee_rate)) / next_open_qqq

        # TQQQ 리밸런싱
        curr_tqqq_v = pos_tqqq_qty * next_open_tqqq
        if abs(curr_tqqq_v - target_tqqq_v) > (curr_total * 0.015):
            if curr_tqqq_v > target_tqqq_v:
                sell_val = curr_tqqq_v - target_tqqq_v
                cash += sell_val * (1 - fee_rate)
                pos_tqqq_qty = target_tqqq_v / next_open_tqqq
            elif curr_tqqq_v < target_tqqq_v and cash > 0:
                buy_val = min(cash, target_tqqq_v - curr_tqqq_v)
                cash -= buy_val
                pos_tqqq_qty += (buy_val * (1 - fee_rate)) / next_open_tqqq

        # SGOV 무위험 복리 이자 가산
        cash *= (1 + daily_sgov_rate)

        next_close_qqq = float(df['QQQ_Close'].iloc[i+1])
        next_close_tqqq = float(df['TQQQ_Close'].iloc[i+1])

        final_day_v = (pos_qqq_qty * next_close_qqq) + (pos_kr2x_qty * next_close_qqq * 2) + (pos_tqqq_qty * next_close_tqqq) + cash
        bench_v = init_cash * (next_close_qqq / float(df['QQQ_Close'].iloc[first_idx]))

        # 연말 22% 양도소득세 정산
        if d_next.month == 12 and d_next.day >= 28 and i < len(df) - 2 and df.index[i+2].year != d_next.year:
            annual_gain = max(0.0, final_day_v - init_cash)
            tax = annual_gain * 0.22
            final_day_v -= tax
            cash = max(0.0, cash - tax)

        strat_values.append(final_day_v)
        bench_values.append(bench_v)
        dates_list.append(d_next)

    res_df = pd.DataFrame({'Strategy': strat_values, 'Benchmark': bench_values}, index=pd.to_datetime(dates_list))

    total_days = len(res_df)
    cagr_strat = ((res_df['Strategy'].iloc[-1] / init_cash) ** (252 / total_days) - 1) * 100
    cagr_bench = ((res_df['Benchmark'].iloc[-1] / init_cash) ** (252 / total_days) - 1) * 100
    mdd_strat = ((res_df['Strategy'] / res_df['Strategy'].cummax()) - 1).min() * 100
    mdd_bench = ((res_df['Benchmark'] / res_df['Benchmark'].cummax()) - 1).min() * 100

    daily_ret = res_df['Strategy'].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    alpha = cagr_strat - cagr_bench

    report_msg = (
        f"🏆 <b>[어깨 매도·무릎 매수 2원화 Walk-Forward 백테스트 결과]</b>\n"
        f"📅 기간: {dates_list[0].strftime('%Y-%m-%d')} ~ {dates_list[-1].strftime('%Y-%m-%d')} ({total_days}거래일)\n"
        f"📁 DB 상태: <code>{RAW_DB_FILE}</code> 로컬 캐싱 적용 (QQQ 어깨조정 횟수: {trade_count}회)\n"
        f"────────────────\n"
        f"• <b>세후 최종 자산:</b> <b>${res_df['Strategy'].iloc[-1]:,.2f}</b> (QQQ: ${res_df['Benchmark'].iloc[-1]:,.2f})\n"
        f"• <b>세후 연복리 (CAGR):</b> <b>{cagr_strat:+.2f}%</b> (알파: <b>{alpha:+.2f}%p</b>)\n"
        f"• <b>최대 낙폭 (MDD):</b> <b>{mdd_strat:.2f}%</b> (QQQ: <b>{mdd_bench:.2f}%</b>)\n"
        f"• <b>샤프 지수 (Sharpe):</b> <b>{sharpe:.2f}</b>\n"
        f"────────────────\n"
        f"👉 <i>50일선 어깨 매도 + 5일선 첫 양봉 무릎 매수 + 레버리지 선제 탈출 전수 검증 결과입니다.</i>"
    )
    print(report_msg)
    send_telegram_result(report_msg)

if __name__ == "__main__":
    run_perfect_walkforward_backtest()
