import datetime
import os
import re
import io
import requests
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

DB_FILE = "history_db.csv"

def get_kst_now():
    """한국 표준시(KST) 현재 시각"""
    return datetime.datetime.now(ZoneInfo("Asia/Seoul"))

def get_last_us_trading_date():
    """서머타임 자동 적용 뉴욕 거래소 기준 직전 정규 거래일 날짜 산출"""
    ny_now = datetime.datetime.now(ZoneInfo("America/New_York"))
    if ny_now.hour < 16:
        d = ny_now.date() - datetime.timedelta(days=1)
    else:
        d = ny_now.date()
    while d.weekday() >= 5:
        d -= datetime.timedelta(days=1)
    return d

def get_previous_day_record():
    """자체 DB에서 가장 최근에 저장된 직전 거래일 지표 레코드 조회"""
    if os.path.exists(DB_FILE):
        try:
            db_df = pd.read_csv(DB_FILE)
            if not db_df.empty:
                return db_df.iloc[-1].to_dict(), len(db_df)
        except Exception:
            pass
    return None, 0

def update_and_verify_local_db(record_dict):
    """[자체 DB 엔진] 전체 지표를 CSV에 누적하고 과거치 무결성 검증"""
    warnings = []
    cols = list(record_dict.keys())
    
    if os.path.exists(DB_FILE):
        try:
            db_df = pd.read_csv(DB_FILE)
            db_df['Date'] = db_df['Date'].astype(str)
        except Exception:
            db_df = pd.DataFrame(columns=cols)
    else:
        db_df = pd.DataFrame(columns=cols)

    date_str = str(record_dict['Date'])
    if date_str in db_df['Date'].values:
        past_row = db_df[db_df['Date'] == date_str].iloc[-1]
        if abs(float(past_row['QQQ']) - float(record_dict['QQQ'])) > 0.05:
            warnings.append(f"🚨 [DB 불일치] {date_str} 저장 종가(${past_row['QQQ']}) vs 수집 종가(${record_dict['QQQ']:.2f})")
    else:
        new_row = pd.DataFrame([record_dict])
        db_df = pd.concat([db_df, new_row], ignore_index=True)
        db_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        db_df.sort_values('Date', inplace=True)
        db_df.to_csv(DB_FILE, index=False)

    return warnings, len(db_df)

def fetch_yahoo_v8_chart(ticker_symbol):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}?range=2y&interval=1d"
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            ny_tz = ZoneInfo("America/New_York")
            dates = [datetime.datetime.fromtimestamp(ts, tz=ny_tz).date() for ts in timestamps]
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': quotes.get('open', []),
                'High': quotes.get('high', []),
                'Low': quotes.get('low', []),
                'Close': quotes.get('close', []),
                'Volume': quotes.get('volume', [])
            }).dropna()
            
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.astype(float)
            
            meta_p = result['meta'].get('regularMarketPrice', float(df['Close'].iloc[-1]))
            if not df.empty:
                df.iloc[-1, df.columns.get_loc('Close')] = float(meta_p)
                return df, float(meta_p), df.index[-1].date(), False
    except Exception as e:
        print(f"Yahoo v8 수집 실패 ({ticker_symbol}): {e}")
    return None, 0.0, None, True

def fetch_stooq_csv(ticker_symbol):
    headers = {'User-Agent': 'Mozilla/5.0'}
    symbol_map = {
        "QQQ": "qqq", "TQQQ": "tqqq", "QQQE": "qqqe", "HYG": "hyg", "TLT": "tlt",
        "^VIX": "^vix", "^VXN": "^vxn", "^VIX1D": "^vix1d", "^SKEW": "^skew", "NQ=F": "nq.f",
        "DX-Y.NYB": "usd_i", "USDJPY=X": "usdjpy"
    }
    stooq_sym = symbol_map.get(ticker_symbol, ticker_symbol.lower())
    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200 and len(res.text) > 50 and "No data" not in res.text:
            df = pd.read_csv(io.StringIO(res.text))
            if 'Date' in df.columns and 'Close' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').dropna(subset=['Close']).reset_index(drop=True)
                df.set_index('Date', inplace=True)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df, float(df['Close'].iloc[-1]), df.index[-1].date(), False
    except Exception as e:
        print(f"Stooq 수집 실패 ({ticker_symbol}): {e}")
    return None, 0.0, None, True

def fetch_cross_validated_data(ticker_symbol, expected_trading_date):
    warning_msg = None
    df_y, p_y, d_y, err_y = fetch_yahoo_v8_chart(ticker_symbol)
    df_s, p_s, d_s, err_s = fetch_stooq_csv(ticker_symbol)
    
    if err_y and err_s:
        dummy_df = pd.DataFrame({'Close': [100.0]*50, 'Open': [100.0]*50, 'High': [100.0]*50, 'Low': [100.0]*50, 'Volume': [10000]*50})
        return dummy_df, 100.0, expected_trading_date, f"🚨 {ticker_symbol} 모든 원천 데이터 수집 실패"
    
    if err_s and not err_y:
        chosen_df, chosen_p, chosen_d = df_y, p_y, d_y
    elif err_y and not err_s:
        chosen_df, chosen_p, chosen_d = df_s, p_s, d_s
    else:
        diff_pct = abs(p_y - p_s) / p_y * 100
        if diff_pct > 0.5:
            warning_msg = f"🚨 {ticker_symbol} 소스 간 괴리 ({diff_pct:.2f}%) 발생 [야후: ${p_y:.2f} vs Stooq: ${p_s:.2f}]"
            chosen_df, chosen_p, chosen_d = (df_y, p_y, d_y) if d_y >= d_s else (df_s, p_s, d_s)
        else:
            chosen_df, chosen_p, chosen_d = df_y, p_y, d_y

    return chosen_df, chosen_p, chosen_d, warning_msg

def fetch_overnight_futures_gap():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/NQ=F?range=2d&interval=5m"
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            result = data['chart']['result'][0]
            meta = result['meta']
            regular_close = meta.get('chartPreviousClose', None) or meta.get('previousClose', None)
            current_price = meta.get('regularMarketPrice', None)
            if regular_close and current_price and regular_close > 0:
                overnight_gap_pct = ((current_price / regular_close) - 1) * 100
                return overnight_gap_pct, False
    except Exception as e:
        print(f"선물 야간 갭 수집 실패: {e}")
    return 0.0, True

def fetch_vix3m_real():
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

    df_v, p, d, err = fetch_yahoo_v8_chart("^VIX3M")
    if not err:
        return df_v['Close'], False

    return None, True

def fetch_fred_api(series_id, api_key):
    if api_key:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key.strip(),
                "file_type": "json",
                "sort_order": "desc",
                "limit": 100
            }
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if "observations" in data and data["observations"]:
                records = []
                for obs in data["observations"]:
                    val = obs.get("value", ".")
                    if val != ".":
                        records.append({
                            "DATE": pd.to_datetime(obs["date"]),
                            series_id: float(val)
                        })
                if records:
                    df = pd.DataFrame(records).sort_values("DATE").reset_index(drop=True)
                    last_date = df["DATE"].iloc[-1]
                    return df, last_date, False
        except Exception as e:
            print(f"FRED API 에러 ({series_id}): {e}")

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        csv_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        res = requests.get(csv_url, headers=headers, timeout=10)
        if res.status_code == 200 and len(res.text) > 30:
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = [c.strip().upper() for c in df.columns]
            if "DATE" in df.columns and series_id.upper() in df.columns:
                df["DATE"] = pd.to_datetime(df["DATE"])
                df[series_id] = pd.to_numeric(df[series_id.upper()], errors='coerce')
                df = df.dropna(subset=[series_id]).sort_values("DATE").reset_index(drop=True)
                last_date = df["DATE"].iloc[-1]
                return df[['DATE', series_id]], last_date, False
    except Exception:
        pass

    return None, None, True

def fetch_equity_pcr():
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

def format_delta(current, prev, is_pct=False, unit="", reverse_color=False):
    """전일 대비 변화량(Δ) 포맷팅 헬퍼 함수"""
    if prev is None or np.isnan(prev):
        return ""
    diff = current - prev
    if abs(diff) < 1e-5:
        return " (전일동일)"
    
    if is_pct:
        pct_chg = ((current / prev) - 1) * 100 if prev != 0 else 0.0
        sign = "+" if pct_chg > 0 else ""
        return f" (Δ {sign}{pct_chg:.2f}%)"
    else:
        sign = "+" if diff > 0 else ""
        return f" (Δ {sign}{diff:.2f}{unit})"

def calculate_ultra_risk_score():
    fred_api_key = os.environ.get("FRED_API_KEY", "")
    data_warnings = []
    regime_alerts = []

    expected_trading_date = get_last_us_trading_date()
    prev_db_rec, db_total_days = get_previous_day_record()

    # 1. 시세 및 변동성 수집
    qqq_df, current_close, qqq_d, w_qqq = fetch_cross_validated_data("QQQ", expected_trading_date)
    tqqq_df, current_tqqq, tqqq_d, w_tqqq = fetch_cross_validated_data("TQQQ", expected_trading_date)
    vix_df, vix_current_val, vix_d, w_vix = fetch_cross_validated_data("^VIX", expected_trading_date)
    vix1d_df, vix1d_val, _, _ = fetch_cross_validated_data("^VIX1D", expected_trading_date)
    vxn_df, vxn_current, vxn_d, w_vxn = fetch_cross_validated_data("^VXN", expected_trading_date)
    skew_df, skew_current, _, w_skew = fetch_cross_validated_data("^SKEW", expected_trading_date)

    qqq_close = qqq_df['Close']
    vix_close_s = vix_df['Close']
    vxn_close_s = vxn_df['Close']
    vix1d_close_s = vix1d_df['Close']
    skew_close_s = skew_df['Close']

    if w_qqq: data_warnings.append(w_qqq)
    if w_tqqq: data_warnings.append(w_tqqq)
    if w_vix: data_warnings.append(w_vix)
    if w_vxn: data_warnings.append(w_vxn)
    if w_skew: data_warnings.append(w_skew)

    # 2. VIX3M 수집
    vix3m_close_s, vix3m_failed = fetch_vix3m_real()
    if vix3m_failed or vix3m_close_s is None:
        data_warnings.append("⚠️ VIX3M 3개월물 공식 데이터 수신 실패")
        vix3m_val = vix_current_val * 1.1
    else:
        vix3m_val = float(vix3m_close_s.iloc[-1])

    # 3. FRED 데이터 수집
    df_hy, date_hy, hy_err = fetch_fred_api("BAMLH0A0HYM2", fred_api_key)
    df_assets, date_walcl, walcl_err = fetch_fred_api("WALCL", fred_api_key)
    df_tga, date_tga, tga_err = fetch_fred_api("WTREGEN", fred_api_key)
    df_rrp, date_rrp, rrp_err = fetch_fred_api("RRPONTSYD", fred_api_key)
    df_sofr, _, sofr_err = fetch_fred_api("SOFR", fred_api_key)
    df_iorb, _, iorb_err = fetch_fred_api("IORB", fred_api_key)

    if hy_err: data_warnings.append("⚠️ FRED 하이일드 스프레드 수신 오류")
    if walcl_err or tga_err or rrp_err: data_warnings.append("⚠️ FRED 순유동성 데이터 수신 오류")

    # 4. 순수 야간 선물 갭 감지
    overnight_gap_pct, nq_err = fetch_overnight_futures_gap()
    fut_gap_status = ""
    fut_gap_severe = False
    if not nq_err:
        if overnight_gap_pct <= -1.5:
            fut_gap_severe = True
            fut_gap_status = f"🚨 <b>[장마감 후 야간 NQ선물 추가급락 경보]</b> 야간선물: <b>{overnight_gap_pct:+.2f}%</b>\n"
        elif overnight_gap_pct >= 1.5:
            fut_gap_status = f"🚀 <b>[장마감 후 야간 NQ선물 추가급등]</b> 야간선물: <b>{overnight_gap_pct:+.2f}%</b>\n"

    # ── [1단계] 매크로 & 구조 환경 평가 ──
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
    date_tag = qqq_d.strftime('%m/%d')
    qqq_ref = float(qqq_close.iloc[-20]) if len(qqq_close) >= 20 else current_close
    qqq_20d_ret = ((current_close / qqq_ref) - 1) * 100

    # 지표 1: 200일 이격 Z
    disp_mean = float(disp200_s.mean())
    disp_std = float(disp200_s.std())
    z_disp = float((disp_200 - disp_mean) / disp_std) if disp_std > 0 else 0.0
    score_disp = float(np.clip(z_disp * (7.5 / 2.0), 0, 7.5))

    # 지표 2: 주봉 RSI
    qqq_w = qqq_close.resample('W-FRI').last().ffill().bfill()
    delta_w = qqq_w.diff()
    gain_w = (delta_w.where(delta_w > 0, 0)).rolling(window=14).mean().ffill().bfill()
    loss_w = (-delta_w.where(delta_w < 0, 0)).rolling(window=14).mean().ffill().bfill()
    rs_w = gain_w / loss_w.replace(0, np.nan)
    weekly_rsi = float((100 - (100 / (1 + rs_w))).iloc[-1])
    if np.isnan(weekly_rsi): weekly_rsi = 50.0
    score_rsi = float(np.clip((weekly_rsi - 50) * (7.5 / 30), 0, 7.5))

    # 지표 3: VXN
    vxn_20d_ago = float(vxn_close_s.iloc[-20]) if len(vxn_close_s) >= 20 else vxn_current
    vxn_change_20d = vxn_current - vxn_20d_ago
    score_vxn = 0.0
    vxn_status = "🟢 변동성 안정"
    if qqq_20d_ret > 0 and vxn_change_20d >= 2.0:
        score_vxn = 7.0; vxn_status = "🚨 스마트머니 풋 매집"
    elif qqq_20d_ret > 0 and vxn_change_20d >= 0.5:
        score_vxn = 3.5; vxn_status = "⚠️ 변동성 지지 조짐"
    elif vxn_current <= 14.0:
        score_vxn = 2.0; vxn_status = "⚠️ 변동성 극저점"

    # 지표 4: SKEW
    score_skew = float(np.clip((skew_current - 120) * (7.0 / 25), 0, 7.0))
    vix1d_tag = ""
    if len(vix1d_close_s) >= 5 and vix_current_val > 0:
        ratio_0dte = vix1d_val / vix_current_val
        if ratio_0dte >= 1.25:
            score_skew = min(7.0, score_skew + 2.0)
            vix1d_tag = f" (🔥0DTE {ratio_0dte:.2f}x)"
            if ratio_0dte >= 1.40:
                regime_alerts.append(f"0DTE 변동성 괴리 극대화 (VIX1D/VIX = {ratio_0dte:.2f}x)")
    skew_status = ("🚨 꼬리위험 급증" if skew_current >= 140 else "🟢 정상") + vix1d_tag

    # 지표 5: 기간구조
    if np.isnan(vix3m_val) or vix3m_val <= 0: vix3m_val = vix_current_val * 1.1
    vix_ratio = round(vix_current_val / vix3m_val, 2)
    score_term = float(np.clip((vix_ratio - 0.80) * (6.0 / 0.20), 0, 6.0))
    term_status = "🚨 백워데이션" if vix_ratio >= 1.0 else "🟢 콘탱고 (안정)"

    # 지표 6: QQQ/QQQE 쏠림
    score_breadth = 0.0
    breadth_divergence = 0.0
    z_breadth = 0.0
    try:
        qqqe_df, _, _, _ = fetch_cross_validated_data("QQQE", expected_trading_date)
        if not qqqe_df.empty and len(qqqe_df) >= 20:
            aligned_df = pd.DataFrame({'QQQ': qqq_close, 'QQQE': qqqe_df['Close']}).dropna()
            if len(aligned_df) >= 20:
                qqq_al_ret = ((aligned_df['QQQ'].iloc[-1] / aligned_df['QQQ'].iloc[-20]) - 1) * 100
                qqqe_al_ret = ((aligned_df['QQQE'].iloc[-1] / aligned_df['QQQE'].iloc[-20]) - 1) * 100
                breadth_divergence = qqq_al_ret - qqqe_al_ret
                diff_series = (aligned_df['QQQ'].pct_change(20) - aligned_df['QQQE'].pct_change(20)).dropna() * 100
                if len(diff_series) >= 30:
                    diff_mean = float(diff_series.mean())
                    diff_std = float(diff_series.std())
                    z_breadth = (breadth_divergence - diff_mean) / diff_std if diff_std > 0 else 0.0
                    if qqq_20d_ret > 0 and z_breadth >= 2.0: score_breadth = 8.0
                    elif qqq_20d_ret > 0 and z_breadth >= 1.0: score_breadth = 4.0
    except Exception:
        pass

    # 지표 7: HYG/TLT 크레딧
    score_hyg_tlt = 0.0
    hyg_tlt_status = "🟢 안정"
    hyg_tlt_ratio_val = 0.0
    try:
        hyg_df, _, _, _ = fetch_cross_validated_data("HYG", expected_trading_date)
        tlt_df, _, _, _ = fetch_cross_validated_data("TLT", expected_trading_date)
        if not hyg_df.empty and not tlt_df.empty:
            cr_df = pd.DataFrame({'HYG': hyg_df['Close'], 'TLT': tlt_df['Close']}).dropna()
            if len(cr_df) >= 20:
                cr_df['Ratio'] = cr_df['HYG'] / cr_df['TLT']
                hyg_tlt_ratio_val = float(cr_df['Ratio'].iloc[-1])
                ratio_chg_20d = ((hyg_tlt_ratio_val / float(cr_df['Ratio'].iloc[-20])) - 1) * 100
                if qqq_20d_ret > 0 and ratio_chg_20d <= -2.5:
                    score_hyg_tlt = 7.0; hyg_tlt_status = "🚨 크레딧 붕괴 선행"
                elif qqq_20d_ret > 0 and ratio_chg_20d <= -1.0:
                    score_hyg_tlt = 3.5; hyg_tlt_status = "⚠️ 크레딧 약화 조짐"
    except Exception:
        pass

    # 지표 8: PCR
    pcr_val, pcr_is_fallback = fetch_equity_pcr()
    score_pcr = float(np.clip((0.85 - pcr_val) * (5.0 / 0.35), 0, 5.0))
    pcr_tag = " ⚠️[Fallback적용]" if pcr_is_fallback else ""

    # 지표 9: SOFR-IORB
    score_money_market = 0.0
    sofr_iorb_spread_bps = 0.0
    sofr_status = "🟢 정상"
    try:
        if df_sofr is not None and df_iorb is not None:
            sofr_m = df_sofr.rename(columns={'DATE': 'Date'}).set_index('Date')
            iorb_m = df_iorb.rename(columns={'DATE': 'Date'}).set_index('Date')
            mm_df = pd.concat([sofr_m['SOFR'], iorb_m['IORB']], axis=1).sort_index().dropna()
            if not mm_df.empty:
                sofr_iorb_spread_bps = (float(mm_df['SOFR'].iloc[-1]) - float(mm_df['IORB'].iloc[-1])) * 100
                if sofr_iorb_spread_bps >= 8.0:
                    score_money_market = 10.0; sofr_status = "🚨 단기자금 경색 발작"
                elif sofr_iorb_spread_bps >= 3.0:
                    score_money_market = 5.0; sofr_status = "⚠️ 자금 수요 타이트"
    except Exception:
        pass

    # 지표 10: 환율 (DXY, USDJPY)
    score_fx = 0.0
    dxy_val, usdjpy_val = 0.0, 0.0
    dxy_status, usdjpy_status = "🟢 안정", "🟢 안정"
    try:
        dxy_df, dxy_val, _, _ = fetch_cross_validated_data("DX-Y.NYB", expected_trading_date)
        if not dxy_df.empty and len(dxy_df) >= 20:
            dxy_chg_20d = ((dxy_val / float(dxy_df['Close'].iloc[-20])) - 1) * 100
            if dxy_chg_20d >= 2.5:
                score_fx += 5.0; dxy_status = f"🚨 달러 급등 (+{dxy_chg_20d:.1f}%)"
            elif dxy_chg_20d >= 1.2:
                score_fx += 2.5; dxy_status = f"⚠️ 달러 강세 (+{dxy_chg_20d:.1f}%)"
    except Exception:
        pass

    try:
        jpy_df, usdjpy_val, _, _ = fetch_cross_validated_data("USDJPY=X", expected_trading_date)
        if not jpy_df.empty and len(jpy_df) >= 20:
            jpy_chg_5d = ((usdjpy_val / float(jpy_df['Close'].iloc[-5])) - 1) * 100
            if jpy_chg_5d <= -2.5:
                score_fx += 5.0; usdjpy_status = f"🚨 엔캐리 청산 경보 ({jpy_chg_5d:.1f}%)"
            elif jpy_chg_5d <= -1.2:
                score_fx += 2.5; usdjpy_status = f"⚠️ 엔화 강세 ({jpy_chg_5d:.1f}%)"
    except Exception:
        pass

    # 지표 11: FRED 하이일드
    score_hy = 0.0
    hy_current = 0.0
    hy_status = "🟢 안정"
    hy_tag = ""
    if df_hy is not None and len(df_hy) >= 20:
        hy_current = float(df_hy['BAMLH0A0HYM2'].iloc[-1])
        hy_change_20d = (hy_current - float(df_hy['BAMLH0A0HYM2'].iloc[-20])) * 100
        s_hy_abs = float(np.clip((4.5 - hy_current) * (6.0 / 1.5), 0, 6.0))
        s_hy_div = 6.0 if (qqq_20d_ret > 0 and hy_change_20d >= 20) else (3.0 if (qqq_20d_ret > 0 and hy_change_20d >= 10) else 0.0)
        score_hy = s_hy_abs + s_hy_div
        hy_status = "🚨 크레딧 다이버전스" if s_hy_div == 6.0 else ("⚠️ 반등 조짐" if s_hy_div == 3.0 else "🟢 안정")
    else:
        hy_tag = " ⚠️[수집대체]"

    # 지표 12: 순유동성
    score_liq = 0.0
    current_net_liq = 0.0
    liq_change_4w = 0.0
    liq_status = "🟢 양호"
    liq_date_str = ""
    try:
        if df_assets is not None and df_tga is not None and df_rrp is not None:
            a_df = df_assets.copy().rename(columns={'DATE': 'Date'}).set_index('Date')
            t_df = df_tga.copy().rename(columns={'DATE': 'Date'}).set_index('Date')
            r_df = df_rrp.copy().rename(columns={'DATE': 'Date'}).set_index('Date')
            merged_liq = pd.concat([a_df['WALCL'], t_df['WTREGEN'], r_df['RRPONTSYD']], axis=1).sort_index().ffill().bfill()
            if len(merged_liq) >= 28:
                merged_liq['Net_Liquidity'] = (merged_liq['WALCL'] / 1000) - (merged_liq['WTREGEN'] / 1000) - merged_liq['RRPONTSYD']
                current_net_liq = float(merged_liq['Net_Liquidity'].iloc[-1])
                net_liq_4w_ago = float(merged_liq['Net_Liquidity'].iloc[-28])
                liq_change_4w = ((current_net_liq / net_liq_4w_ago) - 1) * 100
                latest_liq_date = a_df.index[-1]
                liq_date_str = f" ({latest_liq_date.strftime('%m/%d')} 최신 발표치 기준)"
                if qqq_20d_ret > 0 and liq_change_4w < -2.0: score_liq = 13.0; liq_status = "🚨 유동성 흡수"
                elif qqq_20d_ret > 0 and liq_change_4w < 0: score_liq = 6.5; liq_status = "⚠️ 정체"
    except Exception:
        liq_date_str = " ⚠️[수집대체]"

    scores = [
        score_disp, score_rsi, score_vxn, score_skew, score_term,
        score_breadth, score_hyg_tlt, score_pcr,
        score_money_market, score_fx, score_hy, score_liq
    ]
    clean_scores = [0.0 if np.isnan(s) else s for s in scores]
    total_score = round(sum(clean_scores), 1)

    # ── [2단계] 일봉 추세 & MACD ──
    ema12 = qqq_close.ewm(span=12, adjust=False).mean()
    ema26 = qqq_close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_curr = float(macd_line.iloc[-1])
    sig_curr = float(signal_line.iloc[-1])
    macd_deadcross = (macd_curr < sig_curr)
    below_sma5 = (current_close < sma5)
    below_sma20 = (current_close < sma20)

    # ── [3단계] 미시 수급 & 캔들 가격행동 ──
    vol_20d_avg = qqq_df['Volume'].rolling(20).mean().iloc[-1]
    curr_vol = qqq_df['Volume'].iloc[-1]
    vol_ratio = (curr_vol / vol_20d_avg) if vol_20d_avg > 0 else 1.0
    heavy_volume_sell = below_sma5 and (vol_ratio >= 1.25)
    low_volume_pullback = below_sma5 and (vol_ratio < 0.85)

    c_open = qqq_df['Open'].iloc[-1]
    c_high = qqq_df['High'].iloc[-1]
    c_low = qqq_df['Low'].iloc[-1]
    c_close = qqq_df['Close'].iloc[-1]
    candle_range = (c_high - c_low) if (c_high - c_low) > 0 else 1.0
    lower_wick_ratio = (min(c_open, c_close) - c_low) / candle_range
    upper_wick_ratio = (c_high - max(c_open, c_close)) / candle_range

    hammer_reversal = (lower_wick_ratio >= 0.45)
    shooting_star_reversal = (upper_wick_ratio >= 0.45)

    flow_status = "정상"
    if heavy_volume_sell: flow_status = f"🚨 <b>[대량 매도 거래량 실림]</b> 거래량 {vol_ratio:.2f}x"
    elif low_volume_pullback: flow_status = f"⚠️ <b>[거래량 없는 눌림목]</b> 거래량 {vol_ratio:.2f}x"
    elif hammer_reversal: flow_status = f"💎 <b>[밑꼬리 반등 핀바]</b> 저가 매수세 방어"
    else: flow_status = f"🟢 <b>[수급 정상]</b> 거래량 비율: {vol_ratio:.2f}x"

    # ── [4단계] 동적 포지션 사이징 ──
    tr1 = qqq_df['High'] - qqq_df['Low']
    tr2 = (qqq_df['High'] - qqq_df['Close'].shift(1)).abs()
    tr3 = (qqq_df['Low'] - qqq_df['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    atr_pct = (atr14 / current_close) * 100

    vol_cap = 1.0
    vol_cap_tag = ""
    if atr_pct >= 2.5:
        vol_cap = 0.75
        vol_cap_tag = f" (🔥고변동성 장세 ATR {atr_pct:.1f}% ➔ 기본 비중 75% 캡 제한)"

    target_tqqq_pos = 1.0 * vol_cap
    is_macro_headwind = (liq_change_4w < -1.0) or (hy_current >= 4.0) or (sofr_iorb_spread_bps >= 5.0) or (score_fx >= 5.0)

    if fut_gap_severe:
        tqqq_action = "🚨 <b>[선물 갭다운 긴급 방어]</b> 야간선물 -1.5% 급락 👉 <b>프리마켓/시초가 TQQQ 30% 선제 축소</b> (슬리피지 방어)"
        target_tqqq_pos = 0.7 * vol_cap
        qqq_action = "🟢 <b>[안정 국면]</b> 1배수는 노이즈 무시하고 100% 홀딩"
    elif total_score >= 80:
        if below_sma5 or macd_deadcross or shooting_star_reversal:
            tqqq_action = "🚨 <b>[대세 탈출 격발 (4단계 확정)]</b> 👉 <b>TQQQ 전량 매도 (현금 100% ➡️ SGOV 파킹)</b>"
            qqq_action = "🚨 <b>[대세 고점 격발]</b> 👉 <b>QQQ 50% 분할 매도 (현금 50% 확보)</b>"
            target_tqqq_pos = 0.0
        else:
            tqqq_action = "⚠️ <b>[대세 과열 - 추세 홀딩]</b> 5일선 지지 중. 5일선 붕괴 즉시 전량 탈출 대기"
            qqq_action = "⚠️ <b>[대세 과열 - 추세 홀딩]</b> 5일선 지지 중. 5일선 이탈 즉시 50% 매도 대기"
            target_tqqq_pos = 1.0 * vol_cap
    elif total_score >= 65:
        if (below_sma5 and macd_deadcross) or heavy_volume_sell:
            tqqq_action = "🚨 <b>[중기 조정 격발 (거래량 실린 이탈)]</b> 👉 <b>TQQQ 50% 분할 매도 (SGOV 파킹)</b>"
            qqq_action = "🚨 <b>[중기 조정 격발]</b> 👉 <b>QQQ 30% 분할 매도</b>"
            target_tqqq_pos = 0.5 * vol_cap
        elif below_sma5:
            if low_volume_pullback or hammer_reversal:
                tqqq_action = "⚖️ <b>[1차 균열 유예 — 수급 방어 확인]</b> 거래량 적은 이탈/밑꼬리 반등 ➔ <b>TQQQ 100% 유지하며 관망</b>"
                qqq_action = "⚖️ <b>[추세 관망]</b> QQQ 100% 유지"
                target_tqqq_pos = 1.0 * vol_cap
            else:
                tqqq_action = "⚠️ <b>[1차 균열 확정]</b> 👉 <b>TQQQ 30% 1차 분할 익절 (SGOV 파킹)</b>"
                qqq_action = "⚠️ <b>[과열 구간 1차 균열]</b> 👉 <b>QQQ 20% 1차 분할 익절</b>"
                target_tqqq_pos = 0.7 * vol_cap
        else:
            tqqq_action = "⚖️ <b>[과열 랠리 홀딩]</b> 5일선 지지 지속. 섣부른 조기 매도 없이 100% 유지"
            qqq_action = "⚖️ <b>[과열권 추세 지속]</b> 100% 포지션 유지"
            target_tqqq_pos = 1.0 * vol_cap
    elif is_ranging_market and total_score < 65:
        if below_sma20:
            tqqq_action = "⚠️ <b>[박스권 하단 이탈]</b> 20일선 붕괴. 👉 <b>TQQQ 50% 비중 축소 (SGOV 대기)</b>"
            qqq_action = "⚠️ <b>[박스권 하단 이탈]</b> 20일선 이탈. 보유 주식 20% 비중 축소"
            target_tqqq_pos = 0.5 * vol_cap
        else:
            tqqq_action = "🔒 <b>[횡보 휩소 방지 모드 (BB Width ≤ 4.0%)]</b> 5일선 잔파도 진입 금지, 20일선 지지 확인하며 유지"
            qqq_action = "📦 <b>[박스권 횡보]</b> 20일선 지지 확인하며 100% 유지"
            target_tqqq_pos = 1.0 * vol_cap
    elif total_score >= 40:
        tqqq_action = "🟢 <b>[상승 추세 순항]</b> TQQQ 100% 포지션 유지 (본전스탑 트레일링 가동)"
        qqq_action = "🟢 <b>[건전한 추세 / 중립]</b> QQQ 100% 유지"
        target_tqqq_pos = 1.0 * vol_cap
    else:
        tqqq_action = "🟢 <b>[안정 국면]</b> TQQQ 100% 포지션 유지"
        qqq_action = "🟢 <b>[안정 국면]</b> QQQ 100% 유지"
        target_tqqq_pos = 1.0 * vol_cap

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
            tqqq_target_b = 1.0
        elif b_score >= 50:
            qqq_b_action = "🚀 <b>[2차 정석 비중 확대]</b> 보유 현금의 35% 2차 매수 (누적 주식 60%)"
            tqqq_target_b = 0.5
        elif b_score >= 25:
            qqq_b_action = "🟢 <b>[1차 정강이 매수]</b> 보유 현금의 25% 1차 매수 (누적 주식 25%)"
            tqqq_target_b = 0.25
        else:
            qqq_b_action = "⏳ <b>[패닉 진행 중]</b> 현금 100% 보존하며 바닥 신호 대기"
            tqqq_target_b = 0.0

        if is_macro_headwind:
            headwind_tag = "⚠️ <b>[거시/환율 역풍 구간 — TQQQ 투입 총량 50% 캡 제한]</b>\n"
            tqqq_target_b = min(0.5, tqqq_target_b)
            if b_score >= 75:
                tqqq_b_action = f"{headwind_tag}   👉 <b>잔여 20% 투입 (누적 TQQQ 50% / 잔여 50%는 SGOV 보존)</b>\n   └ <i>수익률 +10% 도달 시 본전스탑 필수</i>"
            elif b_score >= 50:
                tqqq_b_action = f"{headwind_tag}   👉 <b>현금의 15% 2차 매수 (누적 TQQQ 30% / SGOV 70%)</b>"
            elif b_score >= 25:
                tqqq_b_action = f"{headwind_tag}   👉 <b>현금의 15% 1차 매수 (누적 TQQQ 15% / SGOV 85%)</b>"
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
        target_tqqq_pos = tqqq_target_b

    # ── 자체 DB 기록 딕셔너리 생성 및 무결성 검증 ──
    current_record = {
        'Date': str(qqq_d),
        'QQQ': round(current_close, 2),
        'TQQQ': round(current_tqqq, 2),
        'Score': round(total_score, 1),
        'VXN': round(vxn_current, 2),
        'SKEW': round(skew_current, 1),
        'Term': round(vix_ratio, 2),
        'Breadth_Z': round(z_breadth, 2),
        'HYG_TLT': round(hyg_tlt_ratio_val, 3),
        'PCR': round(pcr_val, 2),
        'SOFR_BPS': round(sofr_iorb_spread_bps, 1),
        'DXY': round(dxy_val, 2),
        'USDJPY': round(usdjpy_val, 2),
        'HY_Spread': round(hy_current, 2),
        'Net_Liq': round(current_net_liq, 1),
        'Target_Pos': round(target_tqqq_pos, 2)
    }

    db_warnings, db_total_days = update_and_verify_local_db(current_record)
    if db_warnings:
        data_warnings.extend(db_warnings)

    # ── 전일 대비 변화량(Δ) 문자열 생성 ──
    p_rec = prev_db_rec if prev_db_rec else {}
    d_qqq = format_delta(current_close, p_rec.get('QQQ'), is_pct=True)
    d_tqqq = format_delta(current_tqqq, p_rec.get('TQQQ'), is_pct=True)
    d_score = format_delta(total_score, p_rec.get('Score'), unit="pt")
    d_vxn = format_delta(vxn_current, p_rec.get('VXN'), unit="pt")
    d_skew = format_delta(skew_current, p_rec.get('SKEW'), unit="pt")
    d_term = format_delta(vix_ratio, p_rec.get('Term'), unit="x")
    d_hyg_tlt = format_delta(hyg_tlt_ratio_val, p_rec.get('HYG_TLT'), is_pct=True)
    d_pcr = format_delta(pcr_val, p_rec.get('PCR'), unit="")
    d_sofr = format_delta(sofr_iorb_spread_bps, p_rec.get('SOFR_BPS'), unit="bp")
    d_dxy = format_delta(dxy_val, p_rec.get('DXY'), is_pct=True)
    d_jpy = format_delta(usdjpy_val, p_rec.get('USDJPY'), is_pct=True)
    d_hy = format_delta(hy_current, p_rec.get('HY_Spread'), unit="%p")
    d_liq = format_delta(current_net_liq, p_rec.get('Net_Liq'), unit="B")

    warning_banner = ""
    if data_warnings:
        warning_banner = "🚨 <b>[데이터 품질/수집 경보]</b>\n" + "\n".join([f"• {w}" for w in data_warnings]) + "\n────────────────\n"

    regime_banner = ""
    if regime_alerts:
        regime_banner = (
            "⚙️ <b>[시장 체제 변화 감지 — 알고리즘 보정 회의 권장]</b>\n" +
            "\n".join([f"• {a}" for a in regime_alerts]) +
            "\n👉 <i>지표 간 구조적 괴리가 발생했으므로 가중치 보정 회의를 권장합니다.</i>\n────────────────\n"
        )

    kst_now_str = get_kst_now().strftime('%Y-%m-%d %H:%M')

    report = (
        f"📊 <b>[QQQ & TQQQ 4단계 정밀 판독기]</b>\n"
        f"📅 기준(KST): {kst_now_str} (자체DB: {db_total_days}일 누적)\n\n"
        f"{warning_banner}"
        f"{regime_banner}"
        f"{fut_gap_status}"
        f"💰 <a href='https://finance.yahoo.com/quote/QQQ'>QQQ 종가 ({date_tag})</a>: <b>${current_close:.2f}</b>{d_qqq} (고점 대비: <b>{drawdown:+.1f}%</b>)\n"
        f"🔥 <a href='https://finance.yahoo.com/quote/TQQQ'>TQQQ 종가 ({date_tag})</a>: <b>${current_tqqq:.2f}</b>{d_tqqq} | ATR(14): <b>{atr_pct:.2f}%</b>{vol_cap_tag}\n"
        f"   └ QQQ 5일선: ${sma5:.2f} | 20일선: ${sma20:.2f} | 50일선: ${sma50:.2f}\n"
        f"────────────────\n"
        f"🎯 <b>1단계 매크로 점수: {total_score} / 100점</b>{d_score}\n"
        f"⚡ <b>2단계 일봉 추세:</b> {'🔴 5일선 이탈' if below_sma5 else '🟢 5일선 지지'} | {'🔴 MACD 데드' if macd_deadcross else '🟢 MACD 상승'}\n"
        f"🌊 <b>3단계 수급/캔들:</b> {flow_status}\n"
        f"🎯 <b>4단계 목표 비중:</b> <b>TQQQ {target_tqqq_pos*100:.0f}%</b> | <b>SGOV 현금 {(1-target_tqqq_pos)*100:.0f}%</b>\n"
        f"────────────────\n"
        f"📘 <b>[QQQ 1배수 정석 전략]</b>\n{qqq_action}\n\n"
        f"📙 <b>[TQQQ 3배수 기관급 전략 (세금/스왑/갭 방어)]</b>\n{tqqq_action}\n"
        f"{bottom_section}"
        f"────────────────\n"
        f"📈 <b>[시장 정밀 매크로 데이터 & 전일 대비 변화(Δ)]</b>\n"
        f"• 200일 이격: {disp_200:.1f}% ({z_disp:+.2f}σ) | RSI: {weekly_rsi:.1f}\n"
        f"• <a href='https://finance.yahoo.com/quote/%5EVXN'>VXN</a>: {vxn_current:.2f}{d_vxn} | <a href='https://finance.yahoo.com/quote/%5ESKEW'>SKEW</a>: {skew_current:.1f}{d_skew} | <a href='https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv'>기간구조</a>: {vix_ratio:.2f}{d_term}\n"
        f"• <a href='https://finance.yahoo.com/quote/QQQE'>QQQ vs QQQE 쏠림</a>: {breadth_divergence:+.2f}%p (동적Z: {z_breadth:+.2f}σ)\n"
        f"• <a href='https://finance.yahoo.com/quote/HYG'>HYG/TLT 크레딧</a>: {hyg_tlt_ratio_val:.3f}{d_hyg_tlt} ({hyg_tlt_status})\n"
        f"• <a href='https://finance.yahoo.com/quote/DX-Y.NYB'>DXY 달러</a>: {dxy_val:.2f}{d_dxy} ({dxy_status}) | <a href='https://finance.yahoo.com/quote/USDJPY=X'>USD/JPY</a>: {usdjpy_val:.2f}{d_jpy} ({usdjpy_status})\n"
        f"• <a href='https://www.cboe.com/us/options/market_statistics/'>Equity PCR</a>: <b>{pcr_val:.2f}</b>{d_pcr}{pcr_tag}\n"
        f"• <a href='https://fred.stlouisfed.org/series/SOFR'>SOFR-IORB 스프레드</a>: <b>{sofr_iorb_spread_bps:+.1f}bp</b>{d_sofr} ({sofr_status})\n"
        f"• <a href='https://fred.stlouisfed.org/series/BAMLH0A0HYM2'>HY 스프레드</a>: {hy_current:.2f}%{d_hy}{hy_tag} ({hy_status})\n"
        f"• 순유동성: ${current_net_liq:.1f}B{d_liq} ({liq_change_4w:+.2f}%){liq_date_str}"
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
