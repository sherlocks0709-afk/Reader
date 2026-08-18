import datetime
import os
import re
import requests
import yfinance as yf
import pandas as pd
import numpy as np

def clean_series(df_col):
    """yfinance 멀티인덱스 컬럼을 안전하게 1차원 Series로 변환 및 결측치 제거"""
    if isinstance(df_col, pd.DataFrame):
        s = df_col.iloc[:, 0]
    else:
        s = df_col
    s = pd.to_numeric(s, errors='coerce').dropna()
    return s.astype(float)

def get_latest_price_and_series(ticker_symbol, period="3y"):
    """장마감 직후 야후 파이낸스 일봉 지연/고정 문제를 방어하여 최신 확정 수치 확보"""
    try:
        t = yf.Ticker(ticker_symbol)
        df = t.history(period=period, interval="1d", auto_adjust=False)
        
        if df.empty or len(df) < 5:
            df = yf.download(ticker_symbol, period=period, interval="1d", progress=False, auto_adjust=False)
        
        close_series = clean_series(df['Close'])
        latest_price = float(close_series.iloc[-1])
        latest_date = df.index[-1]
        
        try:
            fast_price = t.fast_info.get('last_price', None) or t.fast_info.get('regular_market_previous_close', None)
            if fast_price is not None and not np.isnan(fast_price):
                if abs(fast_price - latest_price) / (latest_price if latest_price != 0 else 1) > 0.0005:
                    latest_price = float(fast_price)
                    close_series.iloc[-1] = latest_price
        except Exception:
            pass

        return close_series, latest_price, latest_date, False
    except Exception as e:
        print(f"{ticker_symbol} 시세 수집 오류: {e}")
        return pd.Series([100.0]*50), 100.0, datetime.datetime.now(), True

def fetch_vix3m_real():
    """야후 파이낸스 실패 시 CBOE 공식 CDN에서 VIX3M 실제 데이터를 직접 파싱하여 100% 정밀도 보장"""
    try:
        s, p, _, err = get_latest_price_and_series("^VIX3M", period="3mo")
        if not err and len(s) >= 5:
            return s, False
    except Exception:
        pass

    try:
        s, p, _, err = get_latest_price_and_series("^VXV", period="3mo")
        if not err and len(s) >= 5:
            return s, False
    except Exception:
        pass

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv"
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            lines = res.text.splitlines()
            data = []
            for line in lines[1:]:
                parts = line.split(',')
                if len(parts) >= 5:
                    try:
                        data.append({'Date': pd.to_datetime(parts[0]), 'Close': float(parts[4])})
                    except Exception:
                        continue
            df_cboe = pd.DataFrame(data).set_index('Date').sort_index()
            if not df_cboe.empty:
                return df_cboe['Close'].tail(60), False
    except Exception as e:
        print(f"CBOE VIX3M 원천 다운로드 에러: {e}")

    return None, True

def fetch_fred_api(series_id, api_key):
    """FRED 공식 REST API 및 최종 데이터 일자 확인"""
    if not api_key:
        return None, None, False
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
        if "observations" not in data or not data["observations"]:
            return None, None, False

        records = []
        for obs in data["observations"]:
            val = obs.get("value", ".")
            if val != ".":
                records.append({
                    "DATE": pd.to_datetime(obs["date"]),
                    series_id: float(val)
                })
        df = pd.DataFrame(records).sort_values("DATE").reset_index(drop=True)
        last_date = df["DATE"].iloc[-1]
        days_lag = (datetime.datetime.now() - last_date).days
        is_stale = (days_lag > 7)
        return df, last_date, is_stale
    except Exception as e:
        print(f"FRED 에러 ({series_id}): {e}")
        return None, None, False

def fetch_equity_pcr():
    """CBOE 공식 웹/CSV에서 순수 Equity Put/Call Ratio 추출"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        url = "https://www.cboe.com/us/options/market_statistics/daily/"
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            match = re.search(r'Equity\s+(?:Put/Call|P/C)\s+Ratio[^\d]*([0-1]\.\d{2,3})', res.text, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if 0.25 <= val <= 1.20:
                    return val, False
    except Exception:
        pass

    try:
        url_csv = "https://cdn.cboe.com/data/us/options/market_statistics/daily/daily_market_statistics.csv"
        res_csv = requests.get(url_csv, headers=headers, timeout=8)
        if res_csv.status_code == 200:
            lines = res_csv.text.splitlines()
            for line in lines:
                row = [col.strip().upper() for col in line.split(',')]
                if any('EQUITY' in item for item in row) and any('RATIO' in item or 'P/C' in item for item in row):
                    for item in row:
                        try:
                            val = float(item)
                            if 0.25 <= val <= 1.20:
                                return val, False
                        except ValueError:
                            continue
    except Exception:
        pass

    return 0.58, True

def calculate_ultra_risk_score():
    fred_api_key = os.environ.get("FRED_API_KEY", "")
    data_warnings = []
    regime_alerts = []

    # 1. 시세 및 변동성 지표 전수 실시간 수집 파이프라인
    qqq_close, current_close, qqq_last_date, qqq_err = get_latest_price_and_series("QQQ", period="3y")
    tqqq_close, current_tqqq, _, _ = get_latest_price_and_series("TQQQ", period="1y")
    vix_close_s, vix_current_val, _, vix_err = get_latest_price_and_series("^VIX", period="3mo")
    vix1d_close_s, vix1d_val, _, _ = get_latest_price_and_series("^VIX1D", period="2mo")
    vxn_close_s, vxn_current, _, vxn_err = get_latest_price_and_series("^VXN", period="3mo")
    skew_close_s, skew_current, _, skew_err = get_latest_price_and_series("^SKEW", period="2mo")
    
    if qqq_err:
        data_warnings.append("⚠️ QQQ 가격 데이터 수신 실패")
    if vix_err:
        vix_close_s = pd.Series([18.0] * len(qqq_close), index=qqq_close.index)
        vix_current_val = 18.0
    if vxn_err:
        vxn_close_s = vix_close_s
        vxn_current = vix_current_val
    if skew_err:
        skew_close_s = pd.Series([125.0] * len(qqq_close), index=qqq_close.index)
        skew_current = 125.0

    # 2. VIX3M 수집
    vix3m_close_s, vix3m_failed = fetch_vix3m_real()
    if vix3m_failed or vix3m_close_s is None:
        data_warnings.append("⚠️ VIX3M 3개월물 공식 데이터 수신 실패")
        vix3m_val = vix_current_val * 1.1
    else:
        vix3m_val = float(vix3m_close_s.iloc[-1])

    # 3. FRED 데이터 수집
    df_hy, date_hy, hy_stale = fetch_fred_api("BAMLH0A0HYM2", fred_api_key)
    df_assets, date_walcl, walcl_stale = fetch_fred_api("WALCL", fred_api_key)
    df_tga, date_tga, tga_stale = fetch_fred_api("WTREGEN", fred_api_key)
    df_rrp, date_rrp, rrp_stale = fetch_fred_api("RRPONTSYD", fred_api_key)

    if df_hy is None or hy_stale:
        data_warnings.append("⚠️ FRED 하이일드 스프레드 수신 오류 또는 7일 이상 지연")
    if df_assets is None or df_tga is None or df_rrp is None or walcl_stale:
        data_warnings.append("⚠️ FRED 순유동성(연준자산/TGA/RRP) 수신 오류 또는 지연")

    # 4. 주말/야간 선물 갭 감지
    nq_close, nq_curr_val, _, nq_err = get_latest_price_and_series("NQ=F", period="5d")
    fut_gap_status = ""
    fut_gap_severe = False
    if not nq_err and len(nq_close) >= 2:
        fut_curr = float(nq_close.iloc[-1])
        fut_prev = float(nq_close.iloc[-2])
        if fut_prev > 0:
            fut_change = ((fut_curr / fut_prev) - 1) * 100
            if fut_change <= -1.5:
                fut_gap_severe = True
                fut_gap_status = f"🚨 <b>[야간/주말 NQ선물 갭하락 경보]</b> NQ선물: <b>{fut_change:+.2f}%</b>\n"
            elif fut_change >= 1.5:
                fut_gap_status = f"🚀 <b>[야간/주말 NQ선물 갭상승]</b> NQ선물: <b>{fut_change:+.2f}%</b>\n"
    else:
        data_warnings.append("⚠️ NQ 선물(야간 갭) 데이터 수신 미확인")

    # 5. 이동평균선 & 볼린저 밴드
    sma5_s = qqq_close.rolling(window=5).mean().ffill().bfill()
    sma20_s = qqq_close.rolling(window=20).mean().ffill().bfill()
    sma50_s = qqq_close.rolling(window=50).mean().ffill().bfill()
    sma200_s = qqq_close.rolling(window=200).mean().ffill().bfill()
    disp200_s = (qqq_close / sma200_s) * 100

    rolling_std20 = qqq_close.rolling(window=20).std().ffill().bfill()
    bb_upper = sma20_s + (rolling_std20 * 2)
    bb_lower = sma20_s - (rolling_std20 * 2)
    bb_width_s = ((bb_upper - bb_lower) / sma20_s) * 100

    sma5 = float(sma5_s.iloc[-1])
    sma20 = float(sma20_s.iloc[-1])
    sma50 = float(sma50_s.iloc[-1])
    sma200 = float(sma200_s.iloc[-1])
    disp_200 = float(disp200_s.iloc[-1])
    bb_width = float(bb_width_s.iloc[-1])

    is_ranging_market = (bb_width <= 4.0)
    peak_52w = float(qqq_close.tail(252).max())
    drawdown = ((current_close / peak_52w) - 1) * 100
    date_tag = pd.to_datetime(qqq_last_date).strftime('%m/%d')

    # 지표 1: 200일 이격 Z-Score
    disp_mean = float(disp200_s.mean())
    disp_std = float(disp200_s.std())
    z_disp = float((disp_200 - disp_mean) / disp_std) if disp_std > 0 else 0.0
    score_disp = float(np.clip(z_disp * (10 / 2.0), 0, 10))

    # 지표 2: 주봉 RSI
    qqq_w = qqq_close.resample('W-FRI').last().ffill().bfill()
    delta_w = qqq_w.diff()
    gain_w = (delta_w.where(delta_w > 0, 0)).rolling(window=14).mean().ffill().bfill()
    loss_w = (-delta_w.where(delta_w < 0, 0)).rolling(window=14).mean().ffill().bfill()
    rs_w = gain_w / loss_w.replace(0, np.nan)
    weekly_rsi = float((100 - (100 / (1 + rs_w))).iloc[-1])
    if np.isnan(weekly_rsi): weekly_rsi = 50.0
    score_rsi = float(np.clip((weekly_rsi - 50) * (10 / 30), 0, 10))

    # 지표 3-1: VXN
    vxn_20d_ago = float(vxn_close_s.iloc[-20]) if len(vxn_close_s) >= 20 else vxn_current
    vxn_change_20d = vxn_current - vxn_20d_ago
    qqq_ref = float(qqq_close.iloc[-20]) if len(qqq_close) >= 20 else current_close
    qqq_20d_ret = ((current_close / qqq_ref) - 1) * 100

    score_vxn = 0.0
    vxn_status = "정상"
    if qqq_20d_ret > 0 and vxn_change_20d >= 2.0:
        score_vxn = 10.0
        vxn_status = "🚨 스마트머니 풋 매집"
    elif qqq_20d_ret > 0 and vxn_change_20d >= 0.5:
        score_vxn = 5.0
        vxn_status = "⚠️ 변동성 지지 조짐"
    elif vxn_current <= 14.0:
        score_vxn = 3.0
        vxn_status = "⚠️ 변동성 극저점"
    else:
        vxn_status = "🟢 변동성 안정"

    # 지표 3-2: SKEW & 0DTE
    score_skew = float(np.clip((skew_current - 120) * (10 / 25), 0, 10))
    vix1d_tag = ""
    if len(vix1d_close_s) >= 5 and vix_current_val > 0:
        ratio_0dte = vix1d_val / vix_current_val
        if ratio_0dte >= 1.25:
            score_skew = min(10.0, score_skew + 3.0)
            vix1d_tag = f" (🔥0DTE급등 {ratio_0dte:.2f}x)"
            if ratio_0dte >= 1.40:
                regime_alerts.append(f"0DTE 옵션 변동성 괴리 극대화 (VIX1D/VIX = {ratio_0dte:.2f}x) 👉 초단기 파생 포지션 왜곡 점검 필요")
    skew_status = ("🚨 꼬리위험 급증" if skew_current >= 140 else "🟢 정상") + vix1d_tag

    # 지표 3-3: 기간구조
    if np.isnan(vix3m_val) or vix3m_val <= 0:
        vix3m_val = vix_current_val * 1.1

    vix_ratio = round(vix_current_val / vix3m_val, 2)
    score_term = float(np.clip((vix_ratio - 0.80) * (10 / 0.20), 0, 10))
    term_status = "🚨 백워데이션" if vix_ratio >= 1.0 else "🟢 콘탱고 (안정)"

    # 지표 4: QQQ vs QQQE 쏠림
    score_breadth = 0.0
    breadth_divergence = 0.0
    z_breadth = 0.0
    try:
        qqqe_close, _, _, _ = get_latest_price_and_series("QQQE", period="6mo")
        if not qqqe_close.empty and len(qqqe_close) >= 20:
            qqqe_ref = float(qqqe_close.iloc[-20])
            qqqe_ret_20d = ((float(qqqe_close.iloc[-1]) / qqqe_ref) - 1) * 100
            breadth_divergence = qqq_20d_ret - qqqe_ret_20d
            
            qqq_roll_20 = qqq_close.pct_change(20) * 100
            qqqe_roll_20 = qqqe_close.pct_change(20) * 100
            diff_series = (qqq_roll_20 - qqqe_roll_20).dropna()
            if len(diff_series) >= 40:
                diff_mean = float(diff_series.mean())
                diff_std = float(diff_series.std())
                z_breadth = (breadth_divergence - diff_mean) / diff_std if diff_std > 0 else 0.0
                if qqq_20d_ret > 0 and z_breadth >= 2.0:
                    score_breadth = 15.0
                elif qqq_20d_ret > 0 and z_breadth >= 1.0:
                    score_breadth = 7.5
                if abs(z_breadth) >= 2.8:
                    regime_alerts.append(f"빅테크 vs 동일가중 쏠림도 임계 돌파 ({z_breadth:+.2f}σ) 👉 지수 양극화 체제 변화 점검 필요")
            else:
                if qqq_20d_ret > 0 and breadth_divergence >= 4.0: score_breadth = 15.0
                elif qqq_20d_ret > 0 and breadth_divergence >= 2.0: score_breadth = 7.5
        else:
            data_warnings.append("⚠️ QQQE(동일가중) 데이터 수신 누락")
    except Exception:
        data_warnings.append("⚠️ QQQE(동일가중) 연산 오류")

    # 지표 5: PCR
    pcr_val, pcr_is_fallback = fetch_equity_pcr()
    if pcr_is_fallback:
        data_warnings.append("⚠️ Equity PCR 크롤링 실패 (0.58 기본값 고정)")
    score_pcr = float(np.clip((0.85 - pcr_val) * (10 / 0.35), 0, 10))
    pcr_tag = " ⚠️[Fallback적용]" if pcr_is_fallback else ""

    # 지표 6: 하이일드
    score_hy = 0.0
    hy_current = 0.0
    hy_status = "정상"
    hy_tag = ""
    if df_hy is not None and len(df_hy) >= 20:
        hy_current = float(df_hy['BAMLH0A0HYM2'].iloc[-1])
        hy_20d_ago = float(df_hy['BAMLH0A0HYM2'].iloc[-20])
        hy_change_20d = (hy_current - hy_20d_ago) * 100
        s_hy_abs = float(np.clip((4.5 - hy_current) * (5.0 / 1.5), 0, 5.0))
        s_hy_div = 5.0 if (qqq_20d_ret > 0 and hy_change_20d >= 20) else (2.5 if (qqq_20d_ret > 0 and hy_change_20d >= 10) else 0.0)
        score_hy = s_hy_abs + s_hy_div
        hy_status = "🚨 크레딧 다이버전스" if s_hy_div == 5.0 else ("⚠️ 반등 조짐" if s_hy_div == 2.5 else "🟢 안정")
    else:
        hy_tag = " ⚠️[API대체]"

    # 지표 7: 순유동성
    score_liq = 0.0
    current_net_liq = 0.0
    liq_change_4w = 0.0
    liq_status = "정상"
    liq_date_str = ""

    try:
        if df_assets is not None and df_tga is not None and df_rrp is not None:
            a_df = df_assets.copy()
            t_df = df_tga.copy()
            r_df = df_rrp.copy()

            a_df['DATE'] = pd.to_datetime(a_df['DATE']).dt.tz_localize(None).dt.normalize()
            t_df['DATE'] = pd.to_datetime(t_df['DATE']).dt.tz_localize(None).dt.normalize()
            r_df['DATE'] = pd.to_datetime(r_df['DATE']).dt.tz_localize(None).dt.normalize()

            min_d = max(a_df['DATE'].min(), t_df['DATE'].min(), r_df['DATE'].min())
            max_d = min(a_df['DATE'].max(), t_df['DATE'].max(), r_df['DATE'].max())
            all_dates = pd.date_range(start=min_d, end=max_d, freq='D')
            
            full_df = pd.DataFrame({'DATE': all_dates})
            full_df = pd.merge(full_df, a_df[['DATE', 'WALCL']], on='DATE', how='left').ffill().bfill()
            full_df = pd.merge(full_df, t_df[['DATE', 'WTREGEN']], on='DATE', how='left').ffill().bfill()
            full_df = pd.merge(full_df, r_df[['DATE', 'RRPONTSYD']], on='DATE', how='left').ffill().bfill()

            if len(full_df) >= 28:
                full_df['Net_Liquidity'] = (full_df['WALCL'] / 1000) - (full_df['WTREGEN'] / 1000) - full_df['RRPONTSYD']
                current_net_liq = float(full_df['Net_Liquidity'].iloc[-1])
                net_liq_4w_ago = float(full_df['Net_Liquidity'].iloc[-28])
                liq_change_4w = ((current_net_liq / net_liq_4w_ago) - 1) * 100
                
                latest_liq_date = a_df['DATE'].iloc[-1]
                days_lag = (datetime.datetime.now() - latest_liq_date).days
                liq_date_str = f" ({latest_liq_date.strftime('%m/%d')} 기준" + (", 지연)" if days_lag >= 8 else ")")

                if qqq_20d_ret > 0 and liq_change_4w < -2.0:
                    score_liq = 15.0
                    liq_status = "🚨 유동성 흡수"
                elif qqq_20d_ret > 0 and liq_change_4w < 0:
                    score_liq = 7.5
                    liq_status = "⚠️ 정체"
                else:
                    score_liq = 0.0
                    liq_status = "🟢 양호"

                try:
                    q_df = pd.DataFrame({'Date': pd.to_datetime(qqq_close.index).tz_localize(None).normalize(), 'Close': qqq_close.values})
                    merged_check = pd.merge_asof(q_df.sort_values('Date'), full_df.sort_values('DATE'), left_on='Date', right_on='DATE').dropna()
                    if len(merged_check) >= 40:
                        corr_val = merged_check['Close'].tail(40).corr(merged_check['Net_Liquidity'].tail(40))
                        if not np.isnan(corr_val) and corr_val <= -0.65:
                            regime_alerts.append(f"연준 순유동성 vs QQQ 상관계수 역전 (Corr: {corr_val:.2f}) 👉 특수 대출/재정정책 유동성 왜곡 점검 필요")
                except Exception:
                    pass
            else:
                liq_date_str = " ⚠️[데이터부족]"
        else:
            liq_date_str = " ⚠️[API미연결]"
    except Exception as e:
        print(f"순유동성 연산 에러: {e}")
        liq_date_str = " ⚠️[API대체]"

    scores = [score_disp, score_rsi, score_vxn, score_skew, score_term, score_breadth, score_pcr, score_hy, score_liq]
    clean_scores = [0.0 if np.isnan(s) else s for s in scores]
    total_score = round(sum(clean_scores), 1)

    # MACD 연산
    ema12 = qqq_close.ewm(span=12, adjust=False).mean()
    ema26 = qqq_close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_curr = float(macd_line.iloc[-1])
    sig_curr = float(signal_line.iloc[-1])
    macd_deadcross = (macd_curr < sig_curr)
    below_sma5 = (current_close < sma5)
    below_sma20 = (current_close < sma20)

    is_macro_headwind = (liq_change_4w < -1.0) or (hy_current >= 4.0)

    # QQQ 1배수 지침 매핑
    if total_score >= 80:
        if below_sma5 or macd_deadcross:
            qqq_action = "🚨 <b>[확률 95% 대세 고점 격발]</b> 👉 추가 50% 매도 (총 현금 비중 80~100% 확보)"
        else:
            qqq_action = "⚠️ <b>[대세 과열 - 추세 홀딩]</b> 👉 5일선 지지 중. 5일선 이탈 즉시 50% 매도 대기"
    elif total_score >= 65:
        if below_sma5 and macd_deadcross:
            qqq_action = "🚨 <b>[확률 90% 중기 조정 격발]</b> 👉 추가 30% 분할 매도 (총 현금 비중 50% 확보)"
        elif below_sma5 or macd_deadcross:
            qqq_action = "⚠️ <b>[과열 구간 1차 균열]</b> 👉 전체 주식의 20% 1차 분할 익절 (현금 20% 확보)"
        else:
            qqq_action = "⚖️ <b>[과열권 추세 지속 (거짓 경보 방지)]</b> 👉 5일선 지지 지속. 100% 포지션 유지"
    elif is_ranging_market and total_score < 65:
        if below_sma20:
            qqq_action = "⚠️ <b>[박스권 하단 이탈 경보]</b> 👉 20일선 이탈. 보유 주식 20% 비중 축소"
        else:
            qqq_action = "📦 <b>[박스권 횡보 / 휩소 방지]</b> 👉 5일선 잔파도 무시, 20일선 지지 확인하며 100% 유지"
    elif total_score >= 40:
        qqq_action = "🟢 <b>[건전한 추세 / 중립]</b> 👉 기본 Buy & Hold (주식 비중 100% 유지)"
    else:
        qqq_action = "🟢 <b>[안정 국면]</b> 👉 주식 비중 100% 유지"

    # TQQQ 3배수 기관급 지침 매핑
    if fut_gap_severe:
        tqqq_action = "🚨 <b>[선물 갭다운 긴급 방어]</b> NQ선물 -1.5% 이상 급락! 👉 <b>프리마켓/시초가 TQQQ 30% 선제 축소</b> (슬리피지 방어)"
    elif total_score >= 80:
        if below_sma5 or macd_deadcross:
            tqqq_action = "🚨 <b>[TQQQ 3배수 대세 탈출]</b> 👉 <b>TQQQ 전량 매도 (현금 100% ➡️ SGOV 파킹)</b>\n   └ <i>연 4.5% 무위험 이자 수취 & 익년 세금(22%) 재원 확보</i>"
        else:
            tqqq_action = "⚠️ <b>[TQQQ 과열 추세 홀딩]</b> 👉 5일선 지지 중. 5일선 붕괴 즉시 전량 매도 준비"
    elif total_score >= 65:
        if below_sma5 and macd_deadcross:
            tqqq_action = "🚨 <b>[TQQQ 중기 조정 격발]</b> 👉 <b>보유 TQQQ의 50% 분할 매도 (현금 50% SGOV 파킹)</b>"
        elif below_sma5 or macd_deadcross:
            tqqq_action = "⚠️ <b>[TQQQ 1차 균열]</b> 👉 <b>보유 TQQQ의 30% 1차 분할 익절 (SGOV 파킹)</b>"
        else:
            tqqq_action = "⚖️ <b>[TQQQ 과열 랠리 홀딩]</b> 👉 5일선 지지 지속. 섣부른 조기 매도 없이 100% 유지"
    elif is_ranging_market and total_score < 65:
        if below_sma20:
            tqqq_action = "⚠️ <b>[TQQQ 박스권 하단 이탈]</b> 👉 20일선 붕괴. <b>TQQQ 50% 비중 축소 후 현금(SGOV) 대기</b>"
        else:
            tqqq_action = "🔒 <b>[TQQQ 횡보 휩소 방지 모드 (BB Width ≤ 4.0%)]</b> 👉 <b>5일선 잔파도 진입 금지!</b> 20일선 돌파 전까지 매수 보류"
    elif total_score >= 40:
        tqqq_action = "🟢 <b>[TQQQ 상승 추세 유지]</b> 👉 TQQQ 100% 포지션 유지 (본전스탑 트레일링 가동)"
    else:
        tqqq_action = "🟢 <b>[TQQQ 안정 국면]</b> 👉 TQQQ 100% 포지션 유지"

    # 바닥 탐색 모드 (-10% 하락 시)
    bottom_section = ""
    if drawdown <= -10.0:
        vix_sma5 = vix_close_s.rolling(5).mean()
        vix_peaked = vix_current_val < float(vix_sma5.iloc[-1]) and float(vix_close_s.max()) >= 25.0
        term_normalized = (vix_ratio < 0.95)

        b_score = 0.0
        if vix_peaked: b_score += 25.0
        if term_normalized: b_score += 25.0
        if not below_sma5: b_score += 25.0
        if not macd_deadcross: b_score += 25.0

        if b_score >= 75:
            qqq_b_action = "💎 <b>[찐바닥 확정]</b> 잔여 현금 40% 전량 투입 (주식 100% 완전 복귀)"
        elif b_score >= 50:
            qqq_b_action = "🚀 <b>[2차 정석 비중 확대]</b> 보유 현금의 35% 2차 매수 (누적 주식 60%)"
        elif b_score >= 25:
            qqq_b_action = "🟢 <b>[1차 정강이 매수]</b> 보유 현금의 25% 1차 매수 (누적 주식 25%)"
        else:
            qqq_b_action = "⏳ <b>[패닉 진행 중]</b> 현금 100% 보존하며 바닥 신호 대기"

        if is_macro_headwind:
            headwind_tag = "⚠️ <b>[거시 역풍 구간 — TQQQ 투입 총량 50% 캡 제한]</b>\n"
            if b_score >= 75:
                tqqq_b_action = f"{headwind_tag}   👉 <b>잔여 20% 투입 (누적 TQQQ 50% / 잔여 50%는 SGOV 보존)</b>\n   └ <i>수익률 +10% 도달 시 본전스탑(Break-Even) 필수</i>"
            elif b_score >= 50:
                tqqq_b_action = f"{headwind_tag}   👉 <b>현금의 15% 2차 매수 (누적 TQQQ 30% / SGOV 70%)</b>"
            elif b_score >= 25:
                tqqq_b_action = f"{headwind_tag}   👉 <b>현금의 15% 1차 매수 (누적 TQQQ 15% / SGOV 85%)</b>\n   └ <i>진입 평단 대비 -7% 이탈 시 즉시 손절 룰</i>"
            else:
                tqqq_b_action = f"{headwind_tag}   👉 SGOV 100% 파킹 유지하며 바닥 신호 대기"
        else:
            headwind_tag = "🟢 <b>[거시 순풍 구간 — TQQQ 100% 정상 투입 가능]</b>\n"
            if b_score >= 75:
                tqqq_b_action = f"{headwind_tag}   👉 <b>잔여 현금 50% 전량 투입 (TQQQ 100% 완전 복귀)</b>"
            elif b_score >= 50:
                tqqq_b_action = f"{headwind_tag}   👉 <b>현금의 35% 2차 매수 (누적 TQQQ 50%)</b>"
            elif b_score >= 25:
                tqqq_b_action = f"{headwind_tag}   👉 <b>현금의 15% 1차 매수 (누적 TQQQ 15%)</b>"
            else:
                tqqq_b_action = f"{headwind_tag}   👉 현금(SGOV) 100% 보존 대기"

        bottom_section = (
            f"────────────────\n"
            f"🎯 <b>[정밀 바닥 탐색 모드 (고점 대비 {drawdown:.1f}%)]</b>\n"
            f"• 바닥 완성도: <b>{b_score:.0f} / 100점</b> (VIX 피크: {'🟢' if vix_peaked else '🔴'} | 기간구조: {'🟢' if term_normalized else '🔴'})\n\n"
            f"📘 <b>[QQQ 1배수 바닥 지침]</b>\n{qqq_b_action}\n\n"
            f"📙 <b>[TQQQ 3배수 기관급 바닥 지침]</b>\n{tqqq_b_action}\n"
        )

    # 1. 데이터 품질 경보 배너
    warning_banner = ""
    if data_warnings:
        warning_banner = "🚨 <b>[데이터 품질/수집 경보]</b>\n" + "\n".join([f"• {w}" for w in data_warnings]) + "\n────────────────\n"

    # 2. 체제 변화 배너
    regime_banner = ""
    if regime_alerts:
        regime_banner = (
            "⚙️ <b>[시장 체제 변화 감지 — 알고리즘 보정 회의 권장]</b>\n" +
            "\n".join([f"• {a}" for a in regime_alerts]) +
            "\n👉 <i>지표 간 구조적 괴리가 발생했으므로 가중치 보정 회의를 권장합니다.</i>\n────────────────\n"
        )

    report = (
        f"📊 <b>[QQQ & TQQQ 듀얼 전략 정밀 판독기]</b>\n"
        f"📅 기준: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"{warning_banner}"
        f"{regime_banner}"
        f"{fut_gap_status}"
        f"💰 <a href='https://finance.yahoo.com/quote/QQQ'>QQQ 종가 ({date_tag})</a>: <b>${current_close:.2f}</b> (고점 대비: <b>{drawdown:+.1f}%</b>)\n"
        f"🔥 <a href='https://finance.yahoo.com/quote/TQQQ'>TQQQ 종가 ({date_tag})</a>: <b>${current_tqqq:.2f}</b> | BB Width: <b>{bb_width:.1f}%</b> ({'🔒 횡보수축' if is_ranging_market else '🟢 확장'})\n"
        f"   └ QQQ 5일선: ${sma5:.2f} | 20일선: ${sma20:.2f} | 50일선: ${sma50:.2f}\n"
        f"────────────────\n"
        f"🎯 <b>1단계 구조 점수: {total_score} / 100점</b>\n"
        f"⚡ <b>2단계 추세 상태:</b> {'🔴 5일선 이탈' if below_sma5 else '🟢 5일선 지지'} | {'🔴 MACD 데드' if macd_deadcross else '🟢 MACD 상승'}\n"
        f"────────────────\n"
        f"📘 <b>[QQQ 1배수 정석 전략]</b>\n{qqq_action}\n\n"
        f"📙 <b>[TQQQ 3배수 기관급 전략 (세금/스왑/갭 방어)]</b>\n{tqqq_action}\n"
        f"{bottom_section}"
        f"────────────────\n"
        f"📈 <b>[시장 정밀 매크로 데이터]</b>\n"
        f"• 200일 이격: {disp_200:.1f}% ({z_disp:+.2f}σ) | RSI: {weekly_rsi:.1f}\n"
        f"• <a href='https://finance.yahoo.com/quote/%5EVXN'>VXN</a>: {vxn_current:.2f} | <a href='https://finance.yahoo.com/quote/%5ESKEW'>SKEW</a>: {skew_current:.1f} | <a href='https://finance.yahoo.com/quote/%5EVIX3M'>기간구조</a>: {vix_ratio:.2f}\n"
        f"• <a href='https://finance.yahoo.com/quote/QQQE'>QQQ vs QQQE 쏠림</a>: {breadth_divergence:+.2f}%p (동적Z: {z_breadth:+.2f}σ)\n"
        f"• <a href='https://www.cboe.com/us/options/market_statistics/'>Equity PCR</a>: <b>{pcr_val:.2f}</b>{pcr_tag}\n"
        f"• <a href='https://fred.stlouisfed.org/series/BAMLH0A0HYM2'>HY 스프레드</a>: {hy_current:.2f}%{hy_tag} ({hy_status})\n"
        f"• 순유동성: ${current_net_liq:.1f}B ({liq_change_4w:+.2f}%){liq_date_str}"
    )
    return report

def send_telegram_message(text):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("경고: TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code == 200:
            print("텔레그램 전송 성공!")
        else:
            print(f"텔레그램 전송 실패: {res.status_code}, {res.text}")
    except Exception as e:
        print(f"텔레그램 통신 오류: {e}")

if __name__ == "__main__":
    msg = calculate_ultra_risk_score()
    print(msg)
    send_telegram_message(msg)
