# 1. 저점 스나이핑 1차 (TQQQ 20%)
    if vix1d_turned:
      header_icon = "⚡"
      status_title = "저점 매수 1차 (TQQQ 20% 스나이핑)"
      us_guide = (
          "Core QQQ 30% | Satellite QQQ 35% + TQQQ 10% | Sniping TQQQ 10% +"
          " SGOV 5%\n  (총합: QQQ 65% | TQQQ 20% | SGOV 15%)"
      )
      kr_guide = (
          "Core 나스닥 30% | Satellite 나스닥 35% | 레버리지 20% | SOFR 15%"
      )
      action_desc = f"VIX1D({vix1d_prev:.2f} → {vix1d:.2f}) 피크아웃. TQQQ 20% 본대 스나이핑."

    # 2. 저점 스나이핑 2차 (TQQQ 25% 풀스나이핑)
    elif pcr_turned and macro_score < 60.0:
      header_icon = "🚀"
      status_title = "저점 매수 2차 (TQQQ 25% 확신 스나이핑)"
      us_guide = (
          "Core QQQ 30% | Satellite QQQ 35% + TQQQ 10% | Sniping TQQQ 15%\n "
          " (총합: QQQ 65% | TQQQ 25% | SGOV 10%)"
      )
      kr_guide = (
          "Core 나스닥 30% | Satellite 나스닥 35% | 레버리지 25% | SOFR 10%"
      )
      action_desc = f"Equity PCR({pcr_prev:.2f} → {pcr:.2f}) 공포 완화 확인. TQQQ 25% 확정 진입."

    # 3. 투매 절정 선발대 스나이핑
    elif panic_oversold:
      header_icon = "🎯"
      status_title = "투매 절정 (TQQQ 10% 선발대 스나이핑)"
      us_guide = (
          "Core QQQ 30% 유지 | Satellite SGOV 55% | Sniping TQQQ 10% + SGOV"
          " 5%\n  (총합: QQQ 30% | TQQQ 10% | SGOV 60%)"
      )
      kr_guide = (
          "Core 나스닥 30% 유지 | 달러SOFR 60% | 달러레버리지 10% 선진입"
      )
      action_desc = f"단기 패닉 극단치(VIX1D {vix1d:.2f}). 신규 손절 금지 및 현금으로 TQQQ 10% 선발대 매수."

    # 4. 위험 경보 (70% 안전자산 철벽 대피)
    elif (macro_score >= 48.0 and vol_score >= 40.0) or early_exit_triggered:
      header_icon = "⚠️"
      status_title = "위험 경보 (안전자산 70% 대피)"
      us_guide = (
          "Core QQQ 30% 유지 | Satellite SGOV 55% | Sniping SGOV 15%\n  (총합:"
          " QQQ 30% | SGOV 70%)"
      )
      kr_guide = "Core 나스닥 30% 유지 | 달러SOFR 70%"
      action_desc = "단기 변동성 및 매크로 스트레스 급등. Core 30% 유지 후 70% SGOV 대피."

    # 5. 휩쏘 방지 버퍼
    elif prev_macro >= 48.0 and macro_score >= 40.0:
      header_icon = "⏳"
      status_title = "재진입 대기 (휩쏘 방지 안착 관망)"
      us_guide = (
          "Core QQQ 30% 유지 | Satellite SGOV 55% | Sniping SGOV 15%\n  (총합:"
          " QQQ 30% | SGOV 70%)"
      )
      kr_guide = "Core 나스닥 30% 유지 | 달러SOFR 70%"
      action_desc = f"매크로 점수({macro_score}/100) 안정화 대기. 40점 이하 안착 시 QQQ 85% 복귀."

    # 6. 정상 운용 (Risk-On)
    else:
      header_icon = "✅"
      status_title = "정상 운용 (Risk-On / QQQ 85% 유지)"
      us_guide = (
          "Core QQQ 30% | Satellite QQQ 55% | Sniping SGOV 15%\n  (총합: QQQ"
          " 85% | SGOV 15%)"
      )
      kr_guide = "Core 나스닥 30% | Satellite 나스닥 55% | 달러SOFR 15%"
      action_desc = "지표 안정권. QQQ 85% 보유 유지 + 현금 15% SGOV 이자 수취."
