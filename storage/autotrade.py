"""
자동매매 저장소 (설정 · 상태 · 감사 로그)
----------------------------------------
자동매매는 "무엇을 왜 샀는지"를 나중에 재구성할 수 있어야 합니다. 그래서 이
모듈은 설정뿐 아니라 **모든 판단과 주문을 append-only 로 기록**합니다.

테이블
    at_config          사용자별 설정(JSON) + 가동 상태
    at_position_state  포지션별 진입가·손절가·목표가·고점·고정 (전략이 기억해야 하는 값)
    at_events          판단 로그 (신호·거부·진입·청산·오류) — append-only
    at_orders          주문 원장 — client_order_id 로 중복 방지
    at_daily           일자별 시작/최고/종료 평가금액 (손실 한도 판정 기준)
    at_curve           순수익률 곡선 표본 (오늘 기점 그래프 — 수수료·세금 차감)

왜 포지션 상태를 따로 저장하는가
    브로커 잔고에는 "얼마에 샀나"만 있고 "손절을 어디에 걸었나"는 없습니다.
    서버를 재시작해도 손절선을 기억해야 하므로 DB 에 둡니다.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

from config import DB_PATH
# 시장별 배분 기본값은 engine/allocation.py 가 정의합니다. 기본값이 두 곳에
# 있으면 한쪽만 고쳐졌을 때 화면과 엔진이 서로 다른 값을 믿게 됩니다.
from engine.allocation import DEFAULTS as _ALLOCATION_DEFAULTS

# ---------------------------------------------------------------------------
# 기본 설정 — 보수적인 값에서 시작합니다 (자동매매의 기본값은 안전이어야 합니다)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # 실행
    "mode": "paper",                  # paper | mock | live
    "interval_sec": 60,               # 전체 회전 주기(초) — 신호 계산 + 신규 진입
    # 보호 회전 주기(초). 전체 회전 사이사이에 **보유 포지션의 손절·트레일링·
    # 시간 청산만** 확인합니다. 지표를 계산하지 않아 종목당 시세 조회 한 번이라
    # 짧게 돌 수 있습니다. 0 이면 사용하지 않습니다(예전 동작).
    #
    # 왜 필요한가: 전체 회전이 60초면 손절선이 뚫려도 최대 60초를 기다립니다.
    # 익절이 늦는 것은 기회 손실이지만 손절이 늦는 것은 손실입니다.
    #
    # 기본 10초인 이유: 보유 종목당 시세 조회 한 번이 듭니다. 3종목이면
    # 10초에 3회(초당 0.3회)로, 무료 시세 제공처가 견디는 범위입니다.
    # 더 짧게 잡으면 반응은 빨라지지만 차단당할 위험이 올라갑니다.
    "guard_interval_sec": 10,
    "universe": ["005930", "000660", "069500"],   # 삼성전자 · SK하이닉스 · KODEX200
    "asset_classes": {"STOCK": True, "ETF": True, "FUTURES": False, "OPTION": False},

    # 신호
    "entry_score": 0.35,              # 이 점수를 넘으면 진입
    "exit_score": 0.05,               # 반대로 이만큼 넘어가면 청산
    "intraday_weight": 0.35,          # 분봉 점수 비중
    "use_news": False,                # 뉴스 감성 반영(느림)
    "news_weight": 0.25,
    "allow_short": False,             # 숏 진입 (파생만 가능)
    "allow_pyramiding": False,        # 보유 종목 추가 진입

    # 과열 진입 게이트 (engine/strategy.overheat_check) — 점수가 임계값을 넘었어도
    # '이미 많이 간 자리'면 신규 진입만 막습니다. 청산은 건드리지 않습니다.
    # 지표 점수는 급등 고점에서 오히려 강한 매수로 나옵니다(추세지표 9개가
    # 동시 포화 + 국면 판정이 되돌림 지표를 ×0.55 로 깎음). 그 편향을 여기서 막습니다.
    "overheat_gate": True,
    "overheat_rsi": 75.0,             # RSI(14) 이 값 이상이면 과열 1표
    "overheat_bb": 0.95,              # 볼린저 %B 이 값 이상이면 과열 1표
    "overheat_ext_atr": 3.0,          # 20일선에서 이 배수(ATR)만큼 벌어지면 과열 1표
    "overheat_min_hits": 2,           # 이 표수부터 차단 (1~3). 1은 정상 추세도 많이 막습니다
    "overheat_ma": 20,                # 이격을 재는 기준 이동평균

    # 눌림목 재진입 (engine/strategy.pullback_ready) — 과열 게이트의 나머지 절반.
    # 게이트만 있으면 고점은 피하지만 그 종목을 영영 못 삽니다. 급등 뒤 첫 정상
    # 눌림(추세는 살아 있고 이격만 줄어든 자리)에서만 진입 문턱을 낮춥니다.
    # 기본 꺼짐 — 검증 하네스가 수익 개선을 기각했습니다. 켜면 진입 문턱이
    # 낮아지므로, 모의계좌 로그로 직접 확인한 뒤 켜세요.
    "pullback_entry": False,
    "pullback_lookback": 10,          # 최근 몇 봉 안의 강세를 볼지
    "pullback_surge_atr": 2.5,        # 그 구간에 20일선 위 이만큼(ATR) 떴어야 '강했다'
    "pullback_max_atr": 1.5,          # 지금은 이 안으로 되돌아왔어야 '눌렸다'
    "pullback_entry_ratio": 0.70,     # 진입 임계값을 이 비율로 낮춤 (0.35 → 0.245)
    "pullback_ma": 20,

    # NNFX 규칙 오버레이 (engine/nnfx.py). 기본은 꺼짐 — 신호 생성 방식을
    # 바꾸는 스위치라, 돌고 있는 계정에서 조용히 켜지면 안 됩니다.
    # off | soft(권장) | veto | hard(실측 기각)
    "nnfx_mode": "off",
    "nnfx_soft_weight": 0.30,         # 소프트 결합 가중 (실측 최적)
    "nnfx_baseline_period": 20,
    "nnfx_bridge_atr_mult": 3.0,      # 기준선에서 이만큼 멀면 추격 금지
    "nnfx_c2_adx_min": 20.0,          # 추세가 있다고 볼 최소 ADX
    "nnfx_volume_ratio_min": 0.7,
    # 앙상블 (engine/ensemble.py) — 시평선 합치·난기류·변동성 배수·비용 게이트.
    # observe(기본)는 계산해서 로그에만 남기고 매매 판단을 바꾸지 않습니다.
    # off | observe | soft(점수·손절폭 조정) | gate(진입 차단까지)
    "algo_mode": "observe",
    "algo_mtf_disagree_penalty": 0.5,   # 일봉·분봉 방향 불일치 시 확신 감산
    "algo_turbulence_pct": 95.0,        # 이 백분위를 넘으면 난기류로 판정
    "algo_turbulence_damping": 0.5,     # 난기류일 때 점수에 곱할 값
    "algo_vol_factor_min": 0.6,         # 변동성 배수 하한 (손절·익절 폭 스케일)
    "algo_vol_factor_max": 2.5,         # 변동성 배수 상한
    "algo_cost_edge_multiple": 3.0,     # 목표 수익 ≥ 왕복비용 × 이 배수 (gate)

    # ML 오버레이 (engine/mlsignal.py — XGBoost·PatchTST 이식).
    # observe(기본)는 GBDT·패치 어텐션 점수를 계산해 신호에 첨부만 하고,
    # soft 는 시간순 검증을 통과한 점수만 가중 결합합니다.
    "ml_mode": "observe",               # off | observe | soft
    "ml_weight": 0.25,                  # soft 결합 가중 (×확신도)
    "ml_horizon_bars": 5,               # 예측 시야 (봉)
    "ml_min_val_accuracy": 0.55,        # GBDT 검증 적중률 하한
    "ml_min_confidence": 0.25,          # 패치 어텐션 확신도 하한

    # 보호장치 (engine/protections.py — freqtrade protections 이식).
    # 최근 매매 이력이 나쁠 때 신규 진입을 잠급니다. 청산은 막지 않습니다.
    "protect_enabled": False,
    "protect_cooldown_min": 30,             # 청산 후 같은 종목 재진입 금지(분)
    "protect_cooldown_loss_min": 240,       # 손실 청산 후는 더 길게(복수 매매 방지)
    "protect_stoploss_count": 3,            # lookback 안 손절 횟수 한도
    "protect_stoploss_lookback_min": 240,
    "protect_stoploss_stop_min": 60,
    "protect_stoploss_per_symbol": False,   # True 면 그 종목만 잠금
    "protect_drawdown_pct": 5.0,            # 실현손익 낙폭 한도 (총자산 대비 %)
    "protect_drawdown_lookback_min": 1440,
    "protect_drawdown_min_trades": 5,
    "protect_drawdown_stop_min": 240,
    "protect_lowprofit_pct": 0.0,           # 종목 성적 기준 (0 = 적자면 잠금)
    "protect_lowprofit_lookback_min": 1440,
    "protect_lowprofit_min_trades": 3,
    "protect_lowprofit_stop_min": 120,
    # 추격 재매수 차단 — 마지막 매도가보다 이만큼 위면 되사지 않습니다(%).
    # 위 protect_* 와 달리 protect_enabled 와 무관하게 동작합니다 (기본 켜짐).
    "protect_chase_pct": 3.0,               # 0 이면 끔
    "protect_chase_lookback_min": 1440,     # 매도가를 기억하는 시간(분)

    # 손실 학습 (engine/lessons.py) — 청산된 매매를 복기해 손실 원인을 추론하고,
    # 같은 컨텍스트(태그)의 다음 신호를 감쇠합니다.
    # off | observe(기록·복기만, 점수 유지) | soft(점수 감쇠까지)
    "learn_mode": "observe",
    "learn_half_life_days": 14.0,     # 최근 매매에 더 큰 가중 — 이 일수마다 절반
    "learn_min_trades": 3,            # 태그당 유효 표본 하한 (우연 몇 번에 반응 금지)
    "learn_scale_pct": 3.0,           # 태그 평균 손실 -3% 에서 감쇠가 최대에 도달
    "learn_max_penalty": 0.10,        # 태그 하나가 점수에서 뺄 수 있는 최대치
    "learn_total_cap": 0.25,          # 감쇠 총합 상한 — 학습이 신호를 통째로 죽이지 않게

    # 유니버스에 없는 종목(= 내가 직접 산 주식)까지 자동으로 관리할지.
    # 기본은 False — 자동매매가 기존 보유분을 마음대로 파는 것을 막습니다.
    "manage_only_universe": True,

    # AI 자동 추적 — 매매 대상을 다중 팩터 추천기가 주기적으로 갱신합니다
    "auto_universe": False,
    "auto_universe_size": 5,
    "auto_universe_refresh_min": 30,
    # 순환 평가 — 갱신 1회에 팩터를 **새로** 계산할 종목 수 (10~300).
    # 한국·미국 **각각**의 창 크기입니다(서로 다른 제공처를 쓰므로 한도를 나눠
    # 가질 이유가 없습니다). 전 종목을 돌며 이만큼씩 계산하고, 하루 안에 계산해
    # 둔 종목은 캐시로 순위에 함께 올립니다 → 몇 회전이면 시장 전체가 순위
    # 대상이 됩니다. 클수록 한 바퀴가 빨리 돌지만 갱신 1회가 길어집니다
    # (100개≈1~2분, 백그라운드라 매매는 계속).
    #   한 바퀴 = 종목수 ÷ 이 값 × 갱신주기.
    #   국내 2,600종목이면 100개·30분에 13시간, 300개·10분에 1.5시간입니다.
    "auto_universe_scan_limit": 100,
    # 코스피·코스닥 **전 종목**(KRX 상장법인목록)을 순환 평가할지.
    # 끄면 예전처럼 KIS 거래량순위 상위 30종목만 봅니다 — 호출은 훨씬 적지만,
    # 순위 밖에서 조용히 추세를 만드는 종목은 영원히 후보가 되지 못합니다.
    # 탐색 범위(auto_universe_pool)를 지정하면 이 설정과 무관하게 그 범위만 봅니다.
    "auto_universe_full_market": True,
    # 훑을 시장 — 코스피 / 코스닥 / 나스닥. 예전 값 "KR" · "US" 도 그대로
    # 동작합니다 (data_sources/universe.py 의 normalize_segments 가 풀어줍니다).
    "auto_universe_markets": ["KOSPI", "KOSDAQ"],
    # 탐색 범위 — 비우면 시장 전체. 지수·ETF·섹터 키를 넣으면 그 안에서만 찾습니다
    # (KOSPI200 · KOSDAQ150 · NASDAQ100 · SP500 · KR_SEMI · US_TECHNOLOGY …)
    #
    # **여러 개를 고를 수 있습니다.** 고른 범위들의 합집합에서 찾습니다
    # (코스피200 + 나스닥100 이면 두 지수의 구성종목 전부).
    # 예전에 문자열 하나로 저장된 설정도 그대로 동작합니다 —
    # data_sources/universe.py 의 normalize_pools 가 목록으로 풀어 줍니다.
    "auto_universe_pool": [],
    "auto_universe_min_score": 0.55,   # 이 점수 미만은 편입하지 않습니다
    "auto_universe_keep_manual": True,  # 손으로 넣은 종목은 유지
    # 사람이 직접 넣은 종목 — AI 갱신에서 지키기 위해 **명시적으로** 기록합니다.
    # (추천 목록과 대조해 추론하면, AI가 한 번 같이 뽑은 순간 '사람이 넣은 것'이라는
    #  사실이 사라져 다음 갱신에 조용히 빠집니다)
    "manual_universe": [],

    # 사이징
    "risk_per_trade_pct": 1.0,        # 1회 손절 시 잃을 총자산 비율
    "position_pct": 20.0,             # 종목당 최대 비중
    # 동시에 들고 갈 종목 수. 기본 1 입니다 — 소액 계좌에서 이 값을 올리면
    # 종목당 돌아가는 돈이 1주 값 밑으로 떨어져 "전부 1주씩"이 됩니다.
    # 시장을 나눠 굴리는 중이라면 시장별 상한(kr_max_positions ·
    # us_max_positions, engine/allocation.py)이 이 값보다 먼저 걸립니다.
    "max_positions": 1,
    "max_order_krw": 3_000_000,
    "min_order_krw": 100_000,
    # 소액 계좌 — 리스크 예산으로 1주 미만이 나와도, 현금·비중 한도 안이면
    # 최소 1주는 사게 허용 (1주 위험이 예산을 넘는다는 건 로그에 남습니다)
    "min_one_unit": False,
    "max_gross_exposure_pct": 100.0,  # 총 노출(명목) 한도

    # 청산
    "atr_stop_mult": 2.0,             # ATR 배수 손절 (0이면 아래 고정 %)
    "stop_loss_pct": 4.0,
    "take_profit_pct": 8.0,
    "trailing_stop_pct": 0.0,         # 0 = 사용 안 함
    "reward_risk": 2.0,
    "max_hold_days": 15,
    # 승자 보유 (오닐 — prism-insight 이식): 50일선 +3% 위 + 진입 후 고점의
    # 97% 이내면 목표가 익절을 보류합니다. 손절·트레일링·시간 청산은 그대로
    # 작동합니다. 기본 꺼짐 — 익절 방식을 바꾸는 스위치라 사람이 켜는 것이
    # 맞습니다.
    #
    # ★ 이것은 "추세가 꺾이면 고점 근처에서 판다"가 **아닙니다.** 이 값이 하는
    #   일은 strategy.check_exit 의 skip_take 하나뿐이라, 게이트가 닫혀도 다시
    #   살아나는 것은 **진입 때 정해진 고정 목표가와 take_profit_pct** 입니다.
    #   고점이 그 목표가를 넘어서지 못한 채 추세가 꺾이면 아무 청산도 걸리지
    #   않고 손실 한도·신호 반전·max_hold_days 까지 흘러갑니다.
    #   즉 단독으로 켜면 "추세가 살아 있는 동안 익절을 지우는" 장치입니다.
    #   고점 되돌림에서 나가게 하려면 trailing_stop_pct 를 함께 켜야 합니다.
    #
    #   실측(2026-08, KOSPI200·KOSDAQ150에서 기계적으로 뽑은 27종목 × 250봉
    #   두 구간): 수익률 차이는 두 구간 모두 0과 구별되지 않았고(t=+1.32 /
    #   +0.30, 합계 이득의 대부분이 한 종목), 최대낙폭은 두 구간 모두 유의하게
    #   나빠졌습니다(-1.64→-1.80 t=-2.59 · -1.47→-1.77 t=-3.23).
    "hold_winners": False,

    # 회전(갈아타기) — 더 좋은 신호가 떴는데 현금·슬롯이 없을 때, 수수료·거래세를
    # 전부 빼고도 순이익인 보유 종목을 목표가 전에 팔아 재원을 만듭니다.
    # 계산법·게이트는 engine/rotation.py 에 있습니다. 기본 꺼짐 — 익절 방식을
    # 바꾸는 스위치라 사람이 켜는 것이 맞습니다 (hold_winners 와 같은 이유).
    "rotation_enabled": False,
    "rotation_cost_edge_multiple": 5.0,  # 새 목표수익 ≥ 왕복비용 × 이 배수
    "rotation_min_gap_pct": 1.0,         # (새 기대 − 남은 기대) 최소 여유 %p
    "rotation_score_margin": 0.15,       # 새 점수가 보유 점수보다 높아야 하는 폭
    "rotation_min_hold_min": 30,         # 산 지 이 시간 안 된 포지션은 안 팝니다
    "rotation_max_per_day": 2,           # 하루 회전 상한
    "rotation_reentry_min": 240,         # 회전으로 판 종목의 재진입 금지(분)

    # 분할 매수·분할 매도 (engine/split.py — 매직스플릿 계열 차수별 독립 관리).
    # 한 종목을 차수로 나눠 사고, 각 차수를 그 차수 매수가 기준으로 따로 팝니다.
    # 기본 꺼짐 — 켜면 손절 방식까지 바뀝니다(차수를 다 담을 때까지 버티는 선으로
    # 손절가가 내려갑니다). 종목당 감당하는 손실 폭이 커지므로 position_pct 를
    # 함께 낮추세요. 자세한 계산은 engine/split.py 머리말에 있습니다.
    "split_enabled": False,
    "split_tranches": 5,                 # 최대 차수 (1차 포함, 2~10)
    "split_buy_drop_pct": 5.0,           # 직전 차수 체결가 대비 이만큼 내리면 다음 차수
    "split_buy_drop_steps": [],          # 차수별 하락률 직접 지정 (예: [10, 20, 30])
    "split_sell_gain_pct": 5.0,          # 그 차수 매수가 대비 이만큼 오르면 그 차수만 매도
    "split_sell_gain_steps": [],         # 차수별 익절률 직접 지정
    "split_size_mode": "equal",          # equal(균등) | pyramid(뒤 차수를 크게)
    "split_pyramid_ratio": 1.5,
    "split_require_oversold": True,      # 추가 매수에 과매도 지표도 요구 (하락률 AND 지표)
    "split_oversold_rsi": 35.0,
    "split_oversold_bb": 0.15,
    "split_overbought_sell": True,       # 과매수면 목표 미달이어도 수익 차수 정리
    "split_overbought_rsi": 70.0,
    "split_overbought_bb": 0.90,
    "split_overbought_min_gain": 1.0,
    "split_cost_multiple": 2.0,          # 차수 수익 ≥ 왕복비용 × 이 배수
    "split_min_gap_min": 30,             # 차수 간 최소 시간 간격(분)
    "split_max_adds_per_day": 3,
    "split_final_stop_pct": 7.0,         # 마지막 차수 예정가에서 이만큼 더 빠지면 전량 손절

    # 주문 집행
    "order_timeout_sec": 180,         # 이 시간 안에 안 체결되면 주문을 취소합니다
    # 조회로 결말을 못 짓는 상태가 이만큼 이어지면 버그로 보고 강제로 거둬들입니다
    # (취소 요청 → 추적 종료). 0 이면 24시간까지 기다리던 예전 동작.
    "order_stuck_min": 100,
    "paper_slippage_bps": 5.0,        # 모의 계좌 체결에 적용할 불리한 스프레드

    # 페니주식 초단타 (engine/scalping.py — 하드 한도가 따로 있습니다)
    "scalp": {},

    # 안전장치
    "kill_switch": False,
    "daily_loss_limit_pct": 3.0,
    "max_drawdown_pct": 10.0,
    "regular_session_only": True,
    # 미국 주문 결제 재원 — "fx" 는 외화 예수금이 모자라면 주문을 내지 않고,
    # "krw" 는 원화(통합증거금)를 믿고 일단 보냅니다.
    #
    # 주의: 이건 **주문 방식 선택이 아닙니다.** KIS 해외주문 API 에는 통화를
    # 고르는 항목 자체가 없습니다(공식 샘플 확인). 앱의 [통합매수] 같은 별도
    # 주문이 API 에는 없어서, 우리가 정할 수 있는 건 "외화가 없을 때 미리
    # 막을 것인가, 보내보고 KIS 판단에 맡길 것인가" 뿐입니다.
    "us_order_funding": "fx",
    # 미국 종목만 프리·애프터마켓 매매 허용 (한국 시간외와 별개 스위치).
    # regular_session_only 를 통째로 끄면 한국 시간외 단일가까지 풀려버리므로
    # 미국 확장 시간만 따로 열 수 있게 분리했습니다.
    "us_extended_hours": False,
    # 미국 주간거래(데이마켓, 한국 낮). 확장시간과 스위치를 나눈 이유는
    # engine/feed.py 의 _day_entry_allowed 주석 참고 — ATS 라 성격이 다릅니다.
    "us_day_session": False,
    "max_quote_age_sec": 120,
    "deriv_min_days_to_expiry": 2,
    "dry_run": False,                 # True 면 판단만 하고 주문을 내지 않습니다

    # 한국·미국 분리 운용 (engine/allocation.py).
    # 시장마다 자기 몫의 돈(총자산 × 배분%)과 자기 종목 상한을 갖습니다.
    # 한 지갑을 선착순으로 쓰던 예전 방식에서는, 먼저 온 미국 종목이 현금을
    # 다 쓰면 그 뒤의 모든 주문이 "주문가능금액" 한도에 걸려 1주씩만 나갔습니다.
    **_ALLOCATION_DEFAULTS,
}

# 상태 기계 — 자본을 움직이는 스위치라 상태를 명시적으로 둡니다
STOPPED = "stopped"        # 꺼짐
RUNNING = "running"        # 정상 가동
HALTED = "halted"          # 안전장치 발동으로 신규 진입 중단 (청산은 계속)

STATE_LABELS = {STOPPED: "정지", RUNNING: "가동 중", HALTED: "안전 중단"}


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_config (
            user_id INTEGER PRIMARY KEY,
            updated_at TEXT NOT NULL,
            config TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'stopped',
            state_reason TEXT DEFAULT ''
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_position_state (
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'paper',   -- paper | mock | live
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            peak_price REAL,
            quantity REAL,
            signal_score REAL,
            opened_at TEXT,
            updated_at TEXT,
            note TEXT,
            strategy TEXT,                        -- auto | scalp (누가 잡은 포지션인가)
            splits TEXT,                          -- 분할 차수 원장 (JSON, engine/split.py)
            pinned INTEGER NOT NULL DEFAULT 0,    -- 고정 (목표가·손절가로만 매도)
            PRIMARY KEY (user_id, mode, symbol)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,          -- info | warn | error | trade
            kind TEXT NOT NULL,           -- signal | entry | exit | reject | halt | tick | error
            symbol TEXT DEFAULT '',
            name TEXT DEFAULT '',
            message TEXT NOT NULL,
            detail TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_events_user "
                     "ON at_events(user_id, id DESC)")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            client_order_id TEXT NOT NULL UNIQUE,
            broker_mode TEXT NOT NULL,
            broker_order_id TEXT DEFAULT '',
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            asset_class TEXT DEFAULT '',
            action TEXT NOT NULL,          -- buy | sell | open_long | open_short | close
            side TEXT DEFAULT '',
            quantity REAL NOT NULL,
            price REAL,
            price_krw REAL,
            fee REAL DEFAULT 0,
            realized_pnl REAL,
            status TEXT NOT NULL,          -- filled | partial | pending | cancelled | rejected
            reason TEXT DEFAULT '',
            detail TEXT,
            -- 체결 추적 (실계좌는 '접수'와 '체결'이 다른 사건입니다)
            intended_price REAL,           -- 주문을 낼 때 보고 있던 가격
            filled_quantity REAL DEFAULT 0,
            avg_fill_price REAL,           -- 실제 평균 체결가
            slippage_bps REAL,             -- (체결가 - 의도가) / 의도가 × 10000
            settled_at TEXT,               -- 체결·취소가 확정된 시각
            poll_count INTEGER DEFAULT 0,  -- 체결 확인을 몇 번 시도했는지
            -- 어느 창구로 접수했는가. '' = 평소 경로, 'US_DAY' = 미국 주간거래.
            -- 취소 API 가 창구별로 달라서, 이걸 안 남기면 주간거래 주문을
            -- 정규장 취소로 보내 "찾지 못함"으로 끝나고 미체결이 원장에 박힙니다.
            venue TEXT DEFAULT ''
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_orders_user "
                     "ON at_orders(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_orders_open "
                     "ON at_orders(user_id, status)")
        _migrate_orders(conn)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_recommend (
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            market TEXT,
            rank INTEGER,
            score REAL,
            price REAL,
            picked INTEGER NOT NULL DEFAULT 0,   -- 실제로 매매 대상에 넣었는가
            regime TEXT,
            reasons TEXT,                        -- 추천 근거 (JSON)
            factors TEXT,                        -- 팩터 원값 (JSON)
            created_at TEXT NOT NULL,
            tradable INTEGER NOT NULL DEFAULT 1, -- 편입 자격 (자금·유동성)
            max_quantity INTEGER,                -- 지금 자금으로 살 수 있는 수량
            warnings TEXT,                       -- 경고문 (JSON)
            PRIMARY KEY (user_id, symbol)
        )""")
        _migrate_recommend(conn)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_daily (
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'paper',
            trade_date TEXT NOT NULL,
            start_value REAL NOT NULL,
            peak_value REAL NOT NULL,
            end_value REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0,
            trade_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, mode, trade_date)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'paper',
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            side TEXT DEFAULT 'long',
            status TEXT NOT NULL DEFAULT 'open',  -- open | closed
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            entry_price REAL,
            exit_price REAL,
            quantity REAL,
            pnl_pct REAL,                 -- NULL = 손익을 모름 (집계에서 제외)
            realized_krw REAL,
            exit_reason TEXT DEFAULT '',
            cause TEXT DEFAULT '',        -- 손실 원인 추론 결론 (이긴 매매는 빈 값)
            tags TEXT DEFAULT '[]',       -- 진입 컨텍스트 태그 (JSON 배열)
            context TEXT                  -- 진입 신호 스냅샷 (JSON)
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_lessons_user "
                     "ON at_lessons(user_id, mode, status, id DESC)")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            mode TEXT NOT NULL,           -- paper | mock | live
            source TEXT NOT NULL,         -- auto | scalp (어느 경로가 평가했나)
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            market TEXT DEFAULT '',
            price REAL,                   -- 호가 통화 기준 현재가 (전방 수익률 계산용)
            price_krw REAL,
            daily_score REAL,
            intraday_score REAL,          -- NULL = 분봉을 못 받음
            news_score REAL,
            nnfx_score REAL,
            ml_score REAL,
            score REAL NOT NULL,          -- 최종 점수
            score_wo_intraday REAL,       -- 분봉 비중 0 이었다면 나왔을 점수
            entry_threshold REAL,         -- 그 회전에 적용된 진입 문턱
            passed INTEGER NOT NULL,      -- 점수가 문턱을 넘었나 (1/0)
            passed_wo_intraday INTEGER,   -- 분봉이 없었다면 넘었을까 (1/0)
            direction TEXT DEFAULT '',    -- long | short | flat (게이트 반영 후)
            gated INTEGER NOT NULL DEFAULT 0,  -- 문턱은 넘었지만 게이트가 막았나
            held INTEGER NOT NULL DEFAULT 0,   -- 이미 보유 중인 종목인가
            regime TEXT DEFAULT '',
            vol_factor REAL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_candidates_user "
                     "ON at_candidates(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_candidates_symbol "
                     "ON at_candidates(user_id, symbol, created_at)")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_account (
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            fingerprint TEXT NOT NULL,    -- 계좌 지문(해시). 계좌번호 원본은 담지 않습니다
            reset_at TEXT NOT NULL,       -- 이 계좌로 성적을 세기 시작한 시각
            PRIMARY KEY (user_id, mode)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_curve (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL,
            scope TEXT NOT NULL,           -- ALL | KR | US
            trade_date TEXT NOT NULL,
            ts TEXT NOT NULL,
            -- 원재료만 저장하고 수익률은 읽을 때 계산합니다. 비율을 그대로
            -- 저장하면 계산식을 고치는 순간 과거 곡선과 현재 곡선이 서로 다른
            -- 정의로 그려집니다 (그래프 한 장에 두 규칙이 섞입니다).
            base_krw REAL NOT NULL DEFAULT 0,       -- 이 시장이 굴리는 돈
            realized_krw REAL NOT NULL DEFAULT 0,   -- 오늘 실현손익 누계(수수료·세금 차감 완료)
            unreal_krw REAL NOT NULL DEFAULT 0,     -- 현재 평가손익
            exit_cost_krw REAL NOT NULL DEFAULT 0,  -- 지금 다 팔면 더 낼 비용(추정)
            equity_krw REAL NOT NULL DEFAULT 0      -- 그 시장 보유 평가금액
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_curve "
                     "ON at_curve(user_id, mode, trade_date, scope, id)")
        _migrate_scope(conn)
        # 모드 분리로 테이블을 새로 만드는 경우가 있어서, 컬럼 추가는 그 뒤입니다
        _migrate_position_state(conn)
        _migrate_daily(conn)
        _migrate_config(conn)


def _migrate_daily(conn):
    """오늘 손익을 입출금과 무관하게 계산하기 위한 컬럼.

    start_unrealized  오늘 첫 회전 시점의 평가손익 (밤새 들고 있던 포지션 몫)
    day_pnl           오늘 손익 확정값 — 누적 집계가 이걸 더합니다
    last_cash         직전 회전의 예수금 (참고·진단용)
    external_flow     오늘 들어오고 나간 현금 합계 (손익이 아님을 명시적으로 분리)
    """
    existing = {r[1] for r in conn.execute("PRAGMA table_info(at_daily)")}
    for column, ddl in (("last_cash", "REAL"),
                        ("external_flow", "REAL NOT NULL DEFAULT 0"),
                        ("start_unrealized", "REAL"),
                        ("day_pnl", "REAL NOT NULL DEFAULT 0"),
                        ("peak_pnl", "REAL NOT NULL DEFAULT 0")):
        if column not in existing:
            conn.execute(f"ALTER TABLE at_daily ADD COLUMN {column} {ddl}")


def _migrate_config(conn):
    """AI 순환 평가가 지금까지 몇 번 돌았는가.

    설정(config JSON)이 아니라 **컬럼**입니다. state · state_reason 과 같은
    성격의 실행 상태라, 설정 저장(save_config)이 통째로 덮어쓰는 자리에 두면
    설정을 한 번 고칠 때마다 세던 수가 사라집니다.

    메모리에 두지 않는 이유: 서버를 재시작하면 0으로 돌아가는데, 화면에는
    "3회전"이라고 적혀 있으니 사람은 순환이 처음부터 다시 시작한 줄 압니다.
    실제로는 팩터 캐시가 남아 있어 이어서 도는 중입니다.
    """
    existing = {r[1] for r in conn.execute("PRAGMA table_info(at_config)")}
    if "scan_rounds" not in existing:
        conn.execute("ALTER TABLE at_config ADD COLUMN "
                     "scan_rounds INTEGER NOT NULL DEFAULT 0")


def _migrate_scope(conn):
    """포지션 상태·일자 기준값을 **계좌(모드)별로** 분리합니다.

    분리 전에는 모드가 달라도 같은 행을 썼습니다. 그래서 모의계좌에서 쌓은
    기준 평가금액(예: 1천만원)이 실전 계좌 판정에 그대로 쓰였고,
    실계좌가 300만원이면 첫 회전에 '일일 손실 -70%'로 오판해 멈췄습니다.
    모의 계좌의 손절선이 실계좌 포지션에 붙는 문제도 같은 뿌리였습니다.

    SQLite 는 기본키를 바꿀 수 없어 테이블을 새로 만들고 옮깁니다.
    기존 데이터는 전부 모의계좌(paper)에서 나온 것이므로 'paper' 로 귀속시킵니다.
    """
    for table, create, columns in (
        ("at_position_state", """
            CREATE TABLE at_position_state (
                user_id INTEGER NOT NULL,
                mode TEXT NOT NULL DEFAULT 'paper',
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL, stop_price REAL, target_price REAL,
                peak_price REAL, quantity REAL, signal_score REAL,
                opened_at TEXT, updated_at TEXT, note TEXT,
                PRIMARY KEY (user_id, mode, symbol)
            )""",
         "user_id, symbol, side, entry_price, stop_price, target_price, "
         "peak_price, quantity, signal_score, opened_at, updated_at, note"),
        ("at_daily", """
            CREATE TABLE at_daily (
                user_id INTEGER NOT NULL,
                mode TEXT NOT NULL DEFAULT 'paper',
                trade_date TEXT NOT NULL,
                start_value REAL NOT NULL, peak_value REAL NOT NULL,
                end_value REAL NOT NULL,
                realized_pnl REAL NOT NULL DEFAULT 0,
                trade_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, mode, trade_date)
            )""",
         "user_id, trade_date, start_value, peak_value, end_value, "
         "realized_pnl, trade_count"),
    ):
        info = list(conn.execute(f"PRAGMA table_info({table})"))
        if not info or any(r["name"] == "mode" for r in info):
            continue
        conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
        conn.execute(create)
        conn.execute(
            f"INSERT INTO {table} (mode, {columns}) "
            f"SELECT 'paper', {columns} FROM {table}_old")
        conn.execute(f"DROP TABLE {table}_old")
        print(f"[autotrade] {table} 를 계좌(모드)별로 분리했습니다.")


def _migrate_orders(conn):
    """체결 추적 컬럼을 나중에 붙입니다.

    자동매매를 이미 돌린 설치본은 at_orders 가 옛 스키마로 만들어져 있고,
    `CREATE TABLE IF NOT EXISTS` 는 기존 테이블을 바꾸지 않습니다.
    컬럼이 없으면 체결 확인 로직이 전부 실패하므로 여기서 채웁니다.
    """
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(at_orders)")}
    additions = {
        "intended_price": "REAL",
        "filled_quantity": "REAL DEFAULT 0",
        "avg_fill_price": "REAL",
        "slippage_bps": "REAL",
        "settled_at": "TEXT",
        "poll_count": "INTEGER DEFAULT 0",
        "venue": "TEXT DEFAULT ''",
    }
    for column, spec in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE at_orders ADD COLUMN {column} {spec}")


def _migrate_recommend(conn):
    """추천 표시용 컬럼 보강 — 화면이 저장본만으로 그리게 되면서 필요해졌습니다.

    [지금 추천 받기]가 백그라운드 계산 + 저장본 폴링으로 바뀌어, '살 수 있는
    수량'·'자금 부족'·경고문도 저장돼 있어야 화면에 나옵니다. 예전 행은
    tradable 기본값 1(예전 화면과 같은 동작), 수량·경고는 비어 있습니다.
    """
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(at_recommend)")}
    if "tradable" not in existing:
        conn.execute("ALTER TABLE at_recommend ADD COLUMN "
                     "tradable INTEGER NOT NULL DEFAULT 1")
    if "max_quantity" not in existing:
        conn.execute("ALTER TABLE at_recommend ADD COLUMN max_quantity INTEGER")
    if "warnings" not in existing:
        conn.execute("ALTER TABLE at_recommend ADD COLUMN warnings TEXT")


def _migrate_position_state(conn):
    """포지션마다 '누가 잡았는가'를 붙입니다.

    페니 초단타와 일반 자동매매는 청산 규칙이 완전히 다릅니다(틱·초 vs %·일).
    누가 잡은 포지션인지 모르면, 초단타가 산 종목을 일반 전략이 며칠씩 들고
    있거나 반대로 일반 종목을 초단타가 3틱에 던집니다. 그래서 진입 시점에
    주인을 박아두고, 청산은 **그 주인의 규칙으로만** 합니다.
    """
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(at_position_state)")}
    if "strategy" not in existing:
        conn.execute("ALTER TABLE at_position_state ADD COLUMN strategy TEXT")
    # 분할 차수 원장 (engine/split.py). 값이 없으면 '분할로 잡은 포지션이 아님'
    # 이라는 뜻이고, 그런 포지션은 예전과 똑같이 전량으로만 매매됩니다 —
    # 분할을 켜기 전에 잡아둔 포지션이 갑자기 차수 매매로 넘어가면 안 됩니다.
    if "splits" not in existing:
        conn.execute("ALTER TABLE at_position_state ADD COLUMN splits TEXT")
    # 고정 — 사용자가 "이 종목은 시간·신호·회전으로 팔지 마라"고 못박은 표시입니다.
    # 기본값 0 이라서, 이 기능을 쓰기 전에 잡아둔 포지션은 예전과 똑같이 움직입니다.
    if "pinned" not in existing:
        conn.execute("ALTER TABLE at_position_state ADD COLUMN pinned "
                     "INTEGER NOT NULL DEFAULT 0")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def get_config(user_id: int) -> dict:
    """저장된 설정 + 기본값 병합. 새 설정 항목이 추가돼도 기존 사용자가 깨지지 않습니다."""
    init()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM at_config WHERE user_id = ?",
                           (user_id,)).fetchone()
    cfg = dict(DEFAULT_CONFIG)
    if row:
        try:
            stored = json.loads(row["config"])
            if isinstance(stored, dict):
                # 중첩 dict(asset_classes)는 통째로 덮어쓰지 않고 병합합니다
                classes = dict(DEFAULT_CONFIG["asset_classes"])
                classes.update(stored.get("asset_classes") or {})
                cfg.update(stored)
                cfg["asset_classes"] = classes
        except json.JSONDecodeError:
            pass
        cfg["enabled"] = bool(row["enabled"])
        cfg["state"] = row["state"]
        cfg["state_reason"] = row["state_reason"] or ""
        cfg["updated_at"] = row["updated_at"]
        cfg["scan_rounds"] = int(row["scan_rounds"] or 0)
    else:
        cfg["enabled"] = False
        cfg["state"] = STOPPED
        cfg["state_reason"] = ""
        cfg["updated_at"] = ""
        cfg["scan_rounds"] = 0
    return cfg


# get_config 가 붙여주는 실행 상태 — 별도 컬럼이라 config JSON 에 넣지 않습니다
_RUNTIME_KEYS = ("enabled", "state", "state_reason", "updated_at", "scan_rounds")


def save_config(user_id: int, patch: dict) -> dict:
    """설정 부분 갱신. 알 수 없는 키는 무시해 오타로 엔진이 망가지는 것을 막습니다."""
    init()
    current = get_config(user_id)
    for key, value in (patch or {}).items():
        if key not in DEFAULT_CONFIG:
            continue
        if key == "asset_classes" and isinstance(value, dict):
            merged = dict(current.get("asset_classes") or {})
            merged.update({k: bool(v) for k, v in value.items()})
            current[key] = merged
        elif key == "scalp" and isinstance(value, dict):
            # 초단타 설정은 병합 + 하드 한도로 강제 축소해서 저장합니다.
            # 저장 단계에서 잘라두면, 이후 어떤 경로로 읽어도 한도 안입니다.
            from engine import scalping
            before = current.get("scalp") or {}
            merged = {**before, **value}
            if "pinned" in value:
                # 지정에서 뺀 종목은 추적 목록에서도 빠져야 합니다.
                # (clamp_config 는 지정 종목을 목록 앞에 넣어줄 뿐, 뺀 것을
                #  알아채지 못합니다 — 이전 값을 아는 여기서 걷어냅니다)
                unpinned = set(before.get("pinned") or []) - set(value.get("pinned") or [])
                merged["universe"] = [s for s in (merged.get("universe") or [])
                                      if s not in unpinned]
            capped = scalping.clamp_config(merged)
            capped.pop("_clamped", None)
            current[key] = capped
        else:
            current[key] = value

    # 저장은 '아는 키만'이 아니라 '읽어온 것 + 아는 키'로 합니다.
    # 아는 키만 남기면, 구버전 코드가 한 번이라도 저장하는 순간 새로 추가된
    # 설정이 조용히 사라집니다 (그 설정을 켠 사람은 왜 안 먹는지 알 수 없습니다).
    storable = {k: v for k, v in current.items() if k not in _RUNTIME_KEYS}
    now = datetime.now().isoformat()
    with _conn() as conn:
        exists = conn.execute("SELECT 1 FROM at_config WHERE user_id = ?",
                              (user_id,)).fetchone()
        if exists:
            conn.execute("UPDATE at_config SET config = ?, updated_at = ? WHERE user_id = ?",
                         (json.dumps(storable, ensure_ascii=False), now, user_id))
        else:
            conn.execute(
                "INSERT INTO at_config (user_id, updated_at, config, enabled, state) "
                "VALUES (?, ?, ?, 0, 'stopped')",
                (user_id, now, json.dumps(storable, ensure_ascii=False)))
    return get_config(user_id)


def bump_scan_round(user_id: int) -> int:
    """AI 순환 평가를 한 번 돌았다고 기록하고, 지금까지의 회전 수를 돌려줍니다.

    자동 갱신뿐 아니라 [지금 추천 받기]도 셉니다 — 손으로 돌린 것도 실제로
    순환을 한 칸 밀어내기 때문입니다. 세지 않으면 화면의 회전 수와 실제
    커버 범위가 어긋납니다.

    **save_config 를 쓰지 않습니다.** 그 함수는 설정 JSON 전체를 다시 직렬화해
    쓰기 때문에, 30분마다 도는 회전이 마침 사용자가 설정을 저장하는 순간과
    겹치면 둘 중 하나가 통째로 덮입니다. 여기서는 컬럼 하나만 올립니다
    (행이 없으면 만들되 config 는 건드리지 않습니다).
    """
    init()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO at_config (user_id, updated_at, config, enabled, state, "
            "scan_rounds) VALUES (?, ?, '{}', 0, ?, 1) "
            "ON CONFLICT(user_id) DO UPDATE SET scan_rounds = scan_rounds + 1",
            (user_id, datetime.now().isoformat(), STOPPED))
        row = conn.execute("SELECT scan_rounds FROM at_config WHERE user_id = ?",
                           (user_id,)).fetchone()
    return int((row["scan_rounds"] if row else 0) or 0)


def set_enabled(user_id: int, enabled: bool, reason: str = "") -> dict:
    init()
    save_config(user_id, {})          # 행이 없으면 만들어 둡니다
    state = RUNNING if enabled else STOPPED
    with _conn() as conn:
        conn.execute(
            "UPDATE at_config SET enabled = ?, state = ?, state_reason = ?, updated_at = ? "
            "WHERE user_id = ?",
            (1 if enabled else 0, state, reason, datetime.now().isoformat(), user_id))
    return get_config(user_id)


def set_state(user_id: int, state: str, reason: str = ""):
    init()
    with _conn() as conn:
        conn.execute(
            "UPDATE at_config SET state = ?, state_reason = ?, updated_at = ? "
            "WHERE user_id = ?",
            (state, reason, datetime.now().isoformat(), user_id))


def enabled_users() -> list[int]:
    """자동매매가 켜진 사용자 목록 (엔진 루프가 매 회전 조회)."""
    init()
    with _conn() as conn:
        return [int(r["user_id"]) for r in conn.execute(
            "SELECT user_id FROM at_config WHERE enabled = 1")]


def configured_users() -> list[int]:
    """자동매매 설정이 있는 모든 사용자 (장 상태 알림 대상).

    꺼져 있어도 알림은 받습니다 — "장이 열렸으니 켤까"를 판단하는 데 필요합니다.
    """
    init()
    with _conn() as conn:
        return [int(r["user_id"]) for r in conn.execute(
            "SELECT user_id FROM at_config")]


# ---------------------------------------------------------------------------
# 포지션 상태
# ---------------------------------------------------------------------------

def get_position_states(user_id: int, mode: str = "paper") -> dict:
    init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM at_position_state WHERE user_id = ? AND mode = ?",
            (user_id, mode))
        return {r["symbol"]: dict(r) for r in rows}


def upsert_position_state(user_id: int, mode: str, symbol: str, **fields):
    init()
    now = datetime.now().isoformat()
    allowed = ("side", "entry_price", "stop_price", "target_price", "peak_price",
               "quantity", "signal_score", "opened_at", "note", "strategy",
               "splits")
    data = {k: v for k, v in fields.items() if k in allowed}
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM at_position_state WHERE user_id = ? AND mode = ? AND symbol = ?",
            (user_id, mode, symbol)).fetchone()
        if row:
            if not data:
                return
            sets = ", ".join(f"{k} = ?" for k in data)
            conn.execute(
                f"UPDATE at_position_state SET {sets}, updated_at = ? "
                "WHERE user_id = ? AND mode = ? AND symbol = ?",
                (*data.values(), now, user_id, mode, symbol))
        else:
            data.setdefault("opened_at", now)
            cols = ", ".join(data)
            marks = ", ".join("?" for _ in data)
            conn.execute(
                f"INSERT INTO at_position_state (user_id, mode, symbol, updated_at, {cols}) "
                f"VALUES (?, ?, ?, ?, {marks})",
                (user_id, mode, symbol, now, *data.values()))


def set_position_pinned(user_id: int, mode: str, symbol: str,
                        pinned: bool) -> bool:
    """포지션 고정 켜기/끄기. 상태 줄이 없으면 False (아무것도 만들지 않습니다).

    **일부러 UPDATE 만 합니다.** 여기서 없는 줄을 새로 만들면, 자동매매가
    관여하지 않던 종목(사용자가 증권사 앱에서 직접 산 주식)이 상태 표에 들어와
    `is_managed` 가 True 로 바뀝니다. 그 줄에는 손절·목표가 없으므로 다음
    회전에서 설정의 고정 손절 %가 대신 걸리고 — 건드리지 않기로 한 주식이
    자동으로 팔립니다. 고정은 자동매매가 이미 관리 중인 포지션에만 붙입니다.
    """
    init()
    with _conn() as conn:
        cur = conn.execute(
            "UPDATE at_position_state SET pinned = ?, updated_at = ? "
            "WHERE user_id = ? AND mode = ? AND symbol = ?",
            (1 if pinned else 0, datetime.now().isoformat(),
             user_id, mode, symbol))
        return cur.rowcount > 0


def clear_position_state(user_id: int, mode: str, symbol: str):
    init()
    with _conn() as conn:
        conn.execute(
            "DELETE FROM at_position_state WHERE user_id = ? AND mode = ? AND symbol = ?",
            (user_id, mode, symbol))


# ---------------------------------------------------------------------------
# 이벤트 로그 (감사 추적)
# ---------------------------------------------------------------------------

def log_event(user_id: int, kind: str, message: str, level: str = "info",
              symbol: str = "", name: str = "", detail: dict = None) -> int:
    init()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO at_events (user_id, created_at, level, kind, symbol, name, "
            "message, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(), level, kind, symbol, name, message,
             json.dumps(detail, ensure_ascii=False, default=str) if detail else None))
        return cur.lastrowid


def get_events(user_id: int, limit: int = 100, kinds: list = None,
               after_id: int = 0) -> list[dict]:
    init()
    sql = "SELECT * FROM at_events WHERE user_id = ?"
    params: list = [user_id]
    if kinds:
        sql += f" AND kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    if after_id:
        sql += " AND id > ?"
        params.append(after_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
    for r in rows:
        if r.get("detail"):
            try:
                r["detail"] = json.loads(r["detail"])
            except json.JSONDecodeError:
                pass
    return rows


def purge_events(user_id: int, keep: int = 2000):
    """로그가 무한히 쌓이지 않게 오래된 것부터 지웁니다."""
    init()
    with _conn() as conn:
        conn.execute(
            "DELETE FROM at_events WHERE user_id = ? AND id NOT IN "
            "(SELECT id FROM at_events WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
            (user_id, user_id, keep))


# ---------------------------------------------------------------------------
# 진입 후보 기록 (at_candidates)
# ---------------------------------------------------------------------------
#
# 왜 이벤트 로그(at_events)로 부족한가
#   ① 이벤트에는 **문턱을 통과한 신호만** 남습니다. 그래서 "분봉 점수가 후보를
#      문턱 아래로 끌어내려 진입을 막은 경우"는 어디에도 기록되지 않았고,
#      분봉의 기여도를 사후에 측정할 방법이 없었습니다.
#   ② purge_events 가 2,000행만 남기고 지웁니다 — 며칠이면 사라집니다.
#
# 그래서 평가한 **모든 후보**를, 판단을 바꿀 수 있는 값(일봉·분봉 점수, 반사실
# 점수, 문턱, 통과 여부)과 함께 별도 표에 남깁니다. 가격을 같이 적는 이유는
# 나중에 "그때 사지 않은 종목이 그 뒤 어떻게 됐나"를 이 표만으로 볼 수 있게
# 하기 위해서입니다.

def record_candidate(user_id: int, row: dict) -> int:
    init()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO at_candidates (user_id, created_at, mode, source, symbol, name, "
            "market, price, price_krw, daily_score, intraday_score, news_score, nnfx_score, "
            "ml_score, score, score_wo_intraday, entry_threshold, passed, passed_wo_intraday, "
            "direction, gated, held, regime, vol_factor) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(), row.get("mode", "paper"),
             row.get("source", "auto"), row.get("symbol", ""), row.get("name", ""),
             row.get("market", ""), row.get("price"), row.get("price_krw"),
             row.get("daily_score"), row.get("intraday_score"), row.get("news_score"),
             row.get("nnfx_score"), row.get("ml_score"), float(row.get("score") or 0),
             row.get("score_wo_intraday"), row.get("entry_threshold"),
             1 if row.get("passed") else 0,
             None if row.get("passed_wo_intraday") is None
             else (1 if row.get("passed_wo_intraday") else 0),
             row.get("direction", ""), 1 if row.get("gated") else 0,
             1 if row.get("held") else 0, row.get("regime", ""), row.get("vol_factor")))
        return cur.lastrowid


def get_candidates(user_id: int, days: int = 7, limit: int = 5000,
                   symbol: str = "", mode: str = "") -> list[dict]:
    init()
    sql = "SELECT * FROM at_candidates WHERE user_id = ?"
    params: list = [user_id]
    if days:
        sql += " AND created_at >= ?"
        params.append((datetime.now() - timedelta(days=days)).isoformat())
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def candidate_summary(user_id: int, days: int = 7, mode: str = "") -> dict:
    """분봉 점수가 실제로 판단을 바꿨는지 — 기록만으로 답하는 집계.

    flipped_in   분봉 때문에 진입 판정이 난 건수 (일봉만이면 문턱 미달)
    flipped_out  분봉 때문에 진입이 막힌 건수 (일봉만이면 통과)
    두 값이 모두 0에 가까우면, 분봉은 판단을 바꾸지 않고 점수만 깎고 있다는 뜻입니다.
    """
    init()
    sql = ("SELECT COUNT(*) n, "
           "SUM(CASE WHEN intraday_score IS NOT NULL THEN 1 ELSE 0 END) with_intraday, "
           "SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) passed, "
           "SUM(CASE WHEN passed = 1 AND passed_wo_intraday = 0 THEN 1 ELSE 0 END) flipped_in, "
           "SUM(CASE WHEN passed = 0 AND passed_wo_intraday = 1 THEN 1 ELSE 0 END) flipped_out, "
           "SUM(CASE WHEN gated = 1 THEN 1 ELSE 0 END) gated, "
           "AVG(ABS(daily_score)) abs_daily, AVG(ABS(intraday_score)) abs_intraday, "
           "AVG(score - score_wo_intraday) mean_shift, "
           "SUM(CASE WHEN ABS(score) < ABS(score_wo_intraday) THEN 1 ELSE 0 END) shrunk "
           "FROM at_candidates WHERE user_id = ?")
    params: list = [user_id]
    if days:
        sql += " AND created_at >= ?"
        params.append((datetime.now() - timedelta(days=days)).isoformat())
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    with _conn() as conn:
        row = dict(conn.execute(sql, params).fetchone() or {})
    row["days"] = days
    return row


def purge_candidates(user_id: int, days: int = 90):
    """분석용 기록이라 이벤트보다 오래 남깁니다 (기본 90일)."""
    init()
    with _conn() as conn:
        conn.execute("DELETE FROM at_candidates WHERE user_id = ? AND created_at < ?",
                     (user_id, (datetime.now() - timedelta(days=days)).isoformat()))


# ---------------------------------------------------------------------------
# 주문 원장
# ---------------------------------------------------------------------------

def order_exists(client_order_id: str) -> bool:
    """이미 낸 주문인가 — 재시작/재시도 시 이중 주문을 막는 마지막 방어선."""
    init()
    with _conn() as conn:
        return conn.execute("SELECT 1 FROM at_orders WHERE client_order_id = ?",
                            (client_order_id,)).fetchone() is not None


def record_order(user_id: int, order: dict) -> int | None:
    init()
    filled = order.get("filled_quantity")
    if filled is None:
        # 즉시 체결된 주문은 체결수량을 따로 주지 않으므로 주문수량으로 채웁니다
        filled = float(order.get("quantity") or 0) if order.get("status") == "filled" else 0.0
    with _conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO at_orders (user_id, created_at, client_order_id, broker_mode, "
                "broker_order_id, symbol, name, asset_class, action, side, quantity, price, "
                "price_krw, fee, realized_pnl, status, reason, detail, intended_price, "
                "filled_quantity, avg_fill_price, slippage_bps, settled_at, venue) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, datetime.now().isoformat(), order.get("client_order_id", ""),
                 order.get("broker_mode", ""), order.get("broker_order_id", ""),
                 order.get("symbol", ""), order.get("name", ""),
                 order.get("asset_class", ""), order.get("action", ""),
                 order.get("side", ""), float(order.get("quantity") or 0),
                 order.get("price"), order.get("price_krw"),
                 float(order.get("fee") or 0), order.get("realized_pnl"),
                 order.get("status", ""), order.get("reason", ""),
                 json.dumps(order.get("detail"), ensure_ascii=False, default=str)
                 if order.get("detail") else None,
                 order.get("intended_price"), float(filled or 0),
                 order.get("avg_fill_price"), order.get("slippage_bps"),
                 datetime.now().isoformat()
                 if order.get("status") in ("filled", "rejected", "cancelled") else None,
                 order.get("venue", "")))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # 같은 client_order_id — 이미 기록된 주문입니다
            return None


# 아직 결과가 확정되지 않은 상태들 (매 회전 브로커에 조회합니다)
OPEN_STATUSES = ("pending", "partial", "cancel_requested")


def open_orders(user_id: int, mode: str = None) -> list[dict]:
    """체결이 확정되지 않은 주문들. 엔진이 매 회전 이 목록부터 정리합니다.

    mode 를 주면 그 계좌의 주문만 봅니다. 모의계좌에 남은 주문이 실전 주문을
    막거나, 실전 주문번호를 모의 브로커에 조회하는 사고를 막기 위해서입니다.
    """
    init()
    marks = ",".join("?" * len(OPEN_STATUSES))
    sql = f"SELECT * FROM at_orders WHERE user_id = ? AND status IN ({marks})"
    params = [user_id, *OPEN_STATUSES]
    if mode:
        sql += " AND broker_mode = ?"
        params.append(mode)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(sql + " ORDER BY id", params)]
    for r in rows:
        if r.get("detail"):
            try:
                r["detail"] = json.loads(r["detail"])
            except json.JSONDecodeError:
                pass
    return rows


def has_open_order(user_id: int, mode: str, symbol: str) -> bool:
    """이 계좌의 이 종목에 아직 처리 중인 주문이 있는가.

    미체결 주문을 무시하고 또 진입하면 같은 종목을 두 번 사게 됩니다.
    """
    init()
    marks = ",".join("?" * len(OPEN_STATUSES))
    with _conn() as conn:
        return conn.execute(
            f"SELECT 1 FROM at_orders WHERE user_id = ? AND broker_mode = ? "
            f"AND symbol = ? AND status IN ({marks}) LIMIT 1",
            (user_id, mode, symbol, *OPEN_STATUSES)).fetchone() is not None


def update_order(order_id: int, **fields):
    """주문 상태 갱신 (체결 확인·취소 결과 반영)."""
    init()
    allowed = ("status", "broker_order_id", "filled_quantity", "avg_fill_price",
               "slippage_bps", "settled_at", "poll_count", "reason", "fee",
               "realized_pnl", "price")
    data = {k: v for k, v in fields.items() if k in allowed}
    if not data:
        return
    sets = ", ".join(f"{k} = ?" for k in data)
    with _conn() as conn:
        conn.execute(f"UPDATE at_orders SET {sets} WHERE id = ?",
                     (*data.values(), order_id))


def bump_poll(order_id: int) -> int:
    init()
    with _conn() as conn:
        conn.execute("UPDATE at_orders SET poll_count = COALESCE(poll_count, 0) + 1 "
                     "WHERE id = ?", (order_id,))
        row = conn.execute("SELECT poll_count FROM at_orders WHERE id = ?",
                           (order_id,)).fetchone()
    return int(row["poll_count"]) if row else 0


def execution_quality(user_id: int, limit: int = 200, mode: str = None) -> dict:
    """체결 품질 요약 — 의도한 가격과 실제 체결가가 얼마나 벌어졌는지.

    이 숫자가 백테스트 가정(슬리피지 5bp)보다 크면 전략이 아니라 집행이
    수익을 깎아먹고 있는 것입니다.
    """
    init()
    scope = " AND broker_mode = ?" if mode else ""
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT slippage_bps, status FROM at_orders "
            f"WHERE user_id = ? AND slippage_bps IS NOT NULL{scope} "
            "ORDER BY id DESC LIMIT ?",
            ((user_id, mode, limit) if mode else (user_id, limit)))]
        counts = {r["status"]: r["n"] for r in conn.execute(
            f"SELECT status, COUNT(*) n FROM at_orders WHERE user_id = ?{scope} "
            "GROUP BY status", ((user_id, mode) if mode else (user_id,)))}
    values = [float(r["slippage_bps"]) for r in rows]
    return {
        "samples": len(values),
        "avg_slippage_bps": round(sum(values) / len(values), 2) if values else 0.0,
        "worst_slippage_bps": round(max(values, key=abs), 2) if values else 0.0,
        "status_counts": counts,
    }


def get_orders(user_id: int, limit: int = 50, mode: str = None) -> list[dict]:
    init()
    sql = "SELECT * FROM at_orders WHERE user_id = ?"
    params: list = [user_id]
    if mode:
        sql += " AND broker_mode = ?"
        params.append(mode)
    params.append(limit)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(sql + " ORDER BY id DESC LIMIT ?", params)]
    for r in rows:
        if r.get("detail"):
            try:
                r["detail"] = json.loads(r["detail"])
            except json.JSONDecodeError:
                pass
    return rows


def fills_for(user_id: int, mode: str, symbol: str, since: str = None,
              limit: int = 200) -> list[dict]:
    """이 종목에서 **실제로 체결된** 내 주문들 — 차트에 산 지점/판 지점을 찍는 용도.

    접수만 되고 체결 안 된 주문은 뺍니다. 차트에 "여기서 샀다"고 찍어놓고
    실제로는 안 샀으면 그건 거짓말입니다.

    가격은 `avg_fill_price`(실제 체결가)를 우선합니다. `price`(주문가)로 찍으면
    지정가 주문에서 마커가 실제 체결 자리와 어긋납니다.
    """
    init()
    since = since or datetime.now().strftime("%Y-%m-%d")
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT created_at, action, side, quantity, filled_quantity, price, "
            "       avg_fill_price, realized_pnl, status, reason "
            "FROM at_orders WHERE user_id = ? AND broker_mode = ? AND symbol = ? "
            "  AND created_at >= ? AND status IN ('filled', 'partial') "
            "ORDER BY created_at ASC LIMIT ?",
            (user_id, mode, str(symbol), since, limit))]

    out = []
    for row in rows:
        price = row.get("avg_fill_price") or row.get("price")
        if not price:
            continue          # 체결가를 모르면 찍을 자리가 없습니다
        out.append({
            "at": row["created_at"],
            "sec_of_day": _sec_of_day_iso(row["created_at"]),
            "action": row["action"],              # buy | sell | close
            "side": row["side"],
            "price": float(price),
            "quantity": float(row.get("filled_quantity") or row.get("quantity") or 0),
            "realized_pnl": row.get("realized_pnl"),
            "status": row["status"],
            "reason": row.get("reason") or "",
        })
    return out


def closed_trades(user_id: int, mode: str, since: str = None,
                  limit: int = 500) -> list[dict]:
    """확정된(손익이 잡힌) 청산 기록 — 보호장치(engine/protections.py)의 판단 재료.

    `realized_pnl` 이 채워진 주문만 봅니다. 접수만 된 주문이나 진입 주문은
    "그 매매가 어떻게 끝났는지"를 말해주지 않기 때문입니다.

    시각은 `settled_at`(체결 확정 시각)을 우선합니다. 접수 시각으로 재면,
    체결까지 오래 걸린 주문이 실제보다 이른 시점의 사건으로 집계돼
    쿨다운이 일찍 풀립니다.
    """
    init()
    sql = ("SELECT symbol, name, action, side, quantity, realized_pnl, reason, "
           "       COALESCE(settled_at, created_at) AS closed_at, "
           "       avg_fill_price, price, status "
           "FROM at_orders WHERE user_id = ? AND broker_mode = ? "
           "  AND realized_pnl IS NOT NULL AND status IN ('filled', 'partial')")
    params: list = [user_id, mode]
    if since:
        sql += " AND COALESCE(settled_at, created_at) >= ?"
        params.append(since)
    params.append(limit)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            sql + " ORDER BY closed_at DESC LIMIT ?", params)]
    return rows


def _sec_of_day_iso(text: str) -> int:
    """ISO 시각 문자열 → 자정으로부터의 초 (차트 x축과 맞추기 위해)."""
    try:
        moment = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return 0
    return moment.hour * 3600 + moment.minute * 60 + moment.second


def realized_today(user_id: int, mode: str, symbols: list = None) -> dict:
    """오늘 확정된 손익 (원). 종목을 주면 그 종목들만.

    페니 초단타의 일일 손실 한도는 **금액**이라 계좌 전체 평가금액 변화가
    아니라 이 값으로 판정해야 합니다. 계좌 전체로 재면, 초단타와 무관한
    보유 종목이 빠졌을 때 초단타가 멈추는 엉뚱한 일이 생깁니다.
    """
    init()
    today = datetime.now().strftime("%Y-%m-%d")
    sql = ("SELECT COALESCE(SUM(realized_pnl), 0) AS pnl, COUNT(*) AS n "
           "FROM at_orders WHERE user_id = ? AND broker_mode = ? "
           "AND realized_pnl IS NOT NULL AND substr(created_at, 1, 10) = ?")
    params: list = [user_id, mode, today]
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        sql += f" AND symbol IN ({placeholders})"
        params.extend(list(symbols))
    with _conn() as conn:
        row = conn.execute(sql, params).fetchone()
    return {"realized_pnl": float(row["pnl"] or 0), "closed_count": int(row["n"] or 0)}


def realized_today_by_symbol(user_id: int, mode: str) -> dict:
    """오늘 확정된 손익을 **종목별로**. 시장별 수익률을 가르는 재료입니다.

    종목이 한국인지 미국인지는 엔진(engine/allocation.scope_of)이 압니다 —
    저장소가 시장을 판단하면 규칙이 두 곳에 생깁니다.
    """
    init()
    today = datetime.now().strftime("%Y-%m-%d")
    with _conn() as conn:
        rows = conn.execute(
            "SELECT symbol, COALESCE(SUM(realized_pnl), 0) AS pnl FROM at_orders "
            "WHERE user_id = ? AND broker_mode = ? AND realized_pnl IS NOT NULL "
            "AND substr(created_at, 1, 10) = ? GROUP BY symbol",
            (user_id, mode, today)).fetchall()
    return {r["symbol"]: float(r["pnl"] or 0) for r in rows}


# ---------------------------------------------------------------------------
# 일자별 기준값 (손실 한도 판정)
# ---------------------------------------------------------------------------

def save_recommendations(user_id: int, recommendations: list, picked: list):
    """AI 추천 결과를 통째로 갈아끼웁니다 (최신 한 벌만 유지).

    화면에서 "왜 이 종목이 매매 대상에 들어왔는지"를 새로고침 후에도
    볼 수 있어야 하므로 DB 에 남깁니다.
    """
    init()
    now = datetime.now().isoformat()
    picked_set = set(picked or [])
    with _conn() as conn:
        conn.execute("DELETE FROM at_recommend WHERE user_id = ?", (user_id,))
        for rec in recommendations:
            data = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
            conn.execute(
                "INSERT OR REPLACE INTO at_recommend (user_id, symbol, name, market, "
                "rank, score, price, picked, regime, reasons, factors, created_at, "
                "tradable, max_quantity, warnings) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, data.get("key", ""), data.get("name", ""),
                 data.get("market", ""), data.get("rank", 0), data.get("score", 0),
                 data.get("price_krw") or data.get("price") or 0,
                 1 if data.get("key") in picked_set else 0,
                 data.get("regime", ""),
                 json.dumps(data.get("reasons") or [], ensure_ascii=False),
                 json.dumps(data.get("factors") or {}, ensure_ascii=False, default=str),
                 now,
                 1 if data.get("tradable", True) else 0,
                 int(data.get("max_quantity") or 0),
                 json.dumps(data.get("warnings") or [], ensure_ascii=False)))


def get_recommendations(user_id: int) -> list[dict]:
    init()
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM at_recommend WHERE user_id = ? ORDER BY rank", (user_id,))]
    for r in rows:
        for field_name in ("reasons", "factors", "warnings"):
            if r.get(field_name):
                try:
                    r[field_name] = json.loads(r[field_name])
                except json.JSONDecodeError:
                    pass
        r["picked"] = bool(r["picked"])
        # 이관 전 행은 컬럼이 None — 예전 화면과 같은 '편입 가능' 취급
        r["tradable"] = bool(r.get("tradable", 1))
    return rows


# ---------------------------------------------------------------------------
# 손실 학습 원장 (engine/lessons.py 가 씁니다)
# ---------------------------------------------------------------------------

def lesson_open(user_id: int, mode: str, symbol: str, name: str = "",
                side: str = "long", entry_price: float = None,
                quantity: float = None, context: dict = None,
                tags: list = None) -> int | None:
    """진입 컨텍스트를 기록합니다. 같은 종목의 열린 교훈이 이미 있으면
    새로 만들지 않습니다 — 추가 매수·분할 차수는 한 매매의 일부입니다."""
    init()
    with _conn() as conn:
        row = conn.execute(
            "SELECT id FROM at_lessons WHERE user_id = ? AND mode = ? "
            "AND symbol = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
            (user_id, mode, symbol)).fetchone()
        if row:
            return int(row["id"])
        cursor = conn.execute(
            "INSERT INTO at_lessons (user_id, mode, symbol, name, side, status, "
            "opened_at, entry_price, quantity, tags, context) "
            "VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)",
            (user_id, mode, symbol, name or "", side or "long",
             datetime.now().isoformat(), entry_price, quantity,
             json.dumps(tags or [], ensure_ascii=False),
             json.dumps(context or {}, ensure_ascii=False)))
        return cursor.lastrowid


def _lesson_row(row) -> dict:
    item = dict(row)
    for key in ("tags", "context"):
        try:
            item[key] = json.loads(item.get(key) or ("[]" if key == "tags" else "{}"))
        except (json.JSONDecodeError, TypeError):
            item[key] = [] if key == "tags" else {}
    return item


def lesson_find_open(user_id: int, mode: str, symbol: str) -> dict | None:
    """이 종목의 열린 교훈 — 청산이 손익을 붙일 대상입니다."""
    init()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM at_lessons WHERE user_id = ? AND mode = ? "
            "AND symbol = ? AND status = 'open' ORDER BY id DESC LIMIT 1",
            (user_id, mode, symbol)).fetchone()
    return _lesson_row(row) if row else None


def lesson_close(lesson_id: int, exit_price: float = None,
                 exit_reason: str = "", realized_krw: float = None,
                 pnl_pct: float = None, cause: str = ""):
    init()
    with _conn() as conn:
        conn.execute(
            "UPDATE at_lessons SET status = 'closed', closed_at = ?, "
            "exit_price = ?, exit_reason = ?, realized_krw = ?, "
            "pnl_pct = ?, cause = ? WHERE id = ?",
            (datetime.now().isoformat(), exit_price, exit_reason or "",
             realized_krw, pnl_pct, cause or "", lesson_id))


def lessons_closed(user_id: int, mode: str, limit: int = 400) -> list[dict]:
    """닫힌 교훈들 (최신순) — 태그별 기대수익 집계의 입력입니다."""
    init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM at_lessons WHERE user_id = ? AND mode = ? "
            "AND status = 'closed' ORDER BY id DESC LIMIT ?",
            (user_id, mode, int(limit))).fetchall()
    return [_lesson_row(r) for r in rows]


def lessons_purge(user_id: int, mode: str = None):
    """이 사용자의 교훈 기록 삭제 — 테스트 정리와 초기화용."""
    init()
    with _conn() as conn:
        if mode:
            conn.execute("DELETE FROM at_lessons WHERE user_id = ? AND mode = ?",
                         (user_id, mode))
        else:
            conn.execute("DELETE FROM at_lessons WHERE user_id = ?", (user_id,))


def touch_daily(user_id: int, mode: str, total_value: float,
                unrealized: float = None, cash: float = None,
                settled: int = 0, trade_date: str = None) -> dict:
    """오늘의 손익·최고 평가금액을 갱신하고 돌려줍니다.

    오늘 손익 = (현재 평가손익 − 오늘 시작 평가손익) + 오늘 실현손익

    **왜 총자산 차이로 재지 않는가 — 입금 때문입니다.**
        예전에는 `총자산 − 오늘 시작 총자산` 으로 쟀습니다. 계좌에 돈을 넣으면
        총자산이 그만큼 뛰므로 입금액이 통째로 수익으로 잡혔습니다
        (실측: 11만원 입금 후 '오늘 손익 +1010%'). 반대로 출금하면 없던 손실이
        생겨 **일일 손실 한도가 잘못 발동해 매매가 멈춥니다.**

        입출금을 탐지해 기준선을 옮기는 방법도 써 봤지만, 탐지가 한 번이라도
        어긋나면 같은 버그가 다시 납니다(실제로 콘솔이 7초마다 예수금 기록을
        지워서 탐지가 통째로 무력화됐습니다).

        지금 식은 **탐지가 필요 없습니다.** 평가손익과 실현손익은 돈을 넣거나
        빼도 변하지 않습니다. 입출금이 계산에 들어올 자리가 아예 없습니다.

    unrealized 를 넘기지 않으면(옛 호출부) 총자산 기준으로 되돌아갑니다.

    **계좌(모드)별로 따로 기록합니다.** 모의계좌 1천만원을 기준으로 실계좌
    300만원을 재면 첫 회전에 '-70% 손실'로 오판합니다.
    """
    init()
    trade_date = trade_date or date.today().isoformat()
    total_value = float(total_value or 0)
    cash = float(cash) if cash is not None else None
    unrealized = float(unrealized) if unrealized is not None else None

    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM at_daily WHERE user_id = ? AND mode = ? AND trade_date = ?",
            (user_id, mode, trade_date)).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO at_daily (user_id, mode, trade_date, start_value, "
                "peak_value, end_value, last_cash, start_unrealized, day_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (user_id, mode, trade_date, total_value, total_value, total_value,
                 cash, unrealized))
            return {"trade_date": trade_date, "mode": mode, "start_value": total_value,
                    "peak_value": total_value, "end_value": total_value,
                    "realized_pnl": 0.0, "trade_count": 0, "external_flow": 0.0,
                    "start_unrealized": unrealized, "unrealized": unrealized,
                    "pnl": 0.0, "pnl_pct": 0.0}

        start = float(row["start_value"])
        realized = float(row["realized_pnl"] or 0)
        external = float(row["external_flow"] or 0)
        start_unrealized = row["start_unrealized"]

        # 오늘 첫 회전에 평가손익을 못 받았다면(옛 호출부) 지금 값을 기준으로
        # 삼습니다. 그 시점부터의 변화만 오늘 손익이 됩니다.
        if start_unrealized is None and unrealized is not None:
            start_unrealized = unrealized

        if unrealized is not None and start_unrealized is not None:
            pnl = (unrealized - float(start_unrealized)) + realized
        else:
            pnl = total_value - start          # 옛 방식 (입출금에 취약)

        peak = max(float(row["peak_value"]), total_value)
        # 낙폭도 **손익 기준**으로 잽니다. 총자산 최고점 대비로 재면 출금할 때
        # 총자산만 떨어져 없던 낙폭이 잡히고 매매가 멈춥니다.
        peak_pnl = max(float(row["peak_pnl"] or 0), pnl)

        conn.execute(
            "UPDATE at_daily SET peak_value = ?, end_value = ?, last_cash = ?, "
            "start_unrealized = ?, day_pnl = ?, peak_pnl = ? "
            "WHERE user_id = ? AND mode = ? AND trade_date = ?",
            (peak, total_value, cash, start_unrealized, pnl, peak_pnl,
             user_id, mode, trade_date))

        base = start if start > 0 else (total_value or 1)
        return {"trade_date": trade_date, "mode": mode,
                "start_value": start, "peak_value": peak, "end_value": total_value,
                "realized_pnl": realized,
                "trade_count": int(row["trade_count"]),
                "external_flow": external,
                "start_unrealized": start_unrealized,
                "unrealized": unrealized,
                "pnl": pnl, "peak_pnl": peak_pnl,
                "pnl_pct": pnl / base * 100 if base else 0.0,
                "drawdown_pct": (pnl - peak_pnl) / base * 100 if base else 0.0}


def record_daily_trade(user_id: int, mode: str, realized_pnl: float = 0.0,
                       trade_date: str = None, count: int = 1):
    """오늘의 실현손익을 누적합니다.

    count=0 은 **부분 체결**용입니다. 손익은 체결된 만큼 그때그때 더해야 하지만
    (안 더하면 손실 한도가 늦게 걸립니다), 매매 건수는 그 매매가 끝났을 때 한 번만
    세야 합니다. 부분 체결 3회를 3건으로 세면 승률과 평균손익이 통째로 어긋납니다.
    """
    init()
    trade_date = trade_date or date.today().isoformat()
    with _conn() as conn:
        conn.execute(
            "UPDATE at_daily SET realized_pnl = realized_pnl + ?, "
            "trade_count = trade_count + ? "
            "WHERE user_id = ? AND mode = ? AND trade_date = ?",
            (float(realized_pnl or 0), int(count), user_id, mode, trade_date))


def get_daily_history(user_id: int, limit: int = 30, mode: str = None) -> list[dict]:
    init()
    sql = "SELECT * FROM at_daily WHERE user_id = ?"
    params: list = [user_id]
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    params.append(limit)
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            sql + " ORDER BY trade_date DESC LIMIT ?", params)]


# ---------------------------------------------------------------------------
# 순수익률 곡선 — 오늘 기점으로 그려 나가는 그래프
# ---------------------------------------------------------------------------
# 표본을 남기는 이유
#     at_daily 는 하루에 한 줄이라 "오늘 몇 시에 얼마였는지"를 복원할 수
#     없습니다. 그래프는 시간축이 필요하므로 회전마다 한 점씩 남깁니다.
#
# 무엇을 남기는가 — **비율이 아니라 원재료**입니다
#     실현손익 · 평가손익 · 청산비용 추정 · 기준 자본. 비율은 읽을 때
#     계산합니다. 비율을 저장하면 계산식을 고치는 순간 옛 점과 새 점이 서로
#     다른 정의로 한 그래프에 섞입니다.
#
# 순수익률의 정의
#     (실현손익 − 오늘 첫 표본의 실현손익)      ← 수수료·거래세 이미 차감
#   + (평가손익 − 오늘 첫 표본의 평가손익)
#   − (청산비용 − 오늘 첫 표본의 청산비용)      ← 지금 다 팔면 더 낼 비용
#     ÷ 오늘 첫 표본의 기준 자본 × 100
#
#     첫 표본에서 반드시 0% 가 됩니다. "오늘 기점"이라는 말 그대로입니다.

_CURVE_KEEP_DAYS = 60


def record_curve(user_id: int, mode: str, samples: list,
                 min_gap_sec: float = 30.0) -> bool:
    """수익률 표본 한 벌(전체·한국·미국)을 남깁니다. 남겼으면 True.

    엔진 회전과 콘솔 새로고침이 둘 다 부릅니다 — 자동매매를 꺼 둔 채 화면만
    보고 있어도 곡선이 이어져야 하기 때문입니다. 대신 `min_gap_sec` 안에
    이미 표본이 있으면 건너뜁니다(초 단위 점을 쌓아봐야 그래프는 같습니다).
    """
    if not samples:
        return False
    init()
    now = datetime.now()
    today = now.date().isoformat()
    with _conn() as conn:
        last = conn.execute(
            "SELECT ts FROM at_curve WHERE user_id = ? AND mode = ? "
            "ORDER BY id DESC LIMIT 1", (user_id, mode)).fetchone()
        if last and min_gap_sec > 0:
            try:
                if (now - datetime.fromisoformat(last["ts"])).total_seconds() < min_gap_sec:
                    return False
            except (TypeError, ValueError):
                pass
        stamp = now.isoformat()
        for row in samples:
            conn.execute(
                "INSERT INTO at_curve (user_id, mode, scope, trade_date, ts, "
                "base_krw, realized_krw, unreal_krw, exit_cost_krw, equity_krw) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, mode, str(row.get("scope") or "ALL"), today, stamp,
                 float(row.get("base_krw") or 0), float(row.get("realized_krw") or 0),
                 float(row.get("unreal_krw") or 0), float(row.get("exit_cost_krw") or 0),
                 float(row.get("equity_krw") or 0)))
        cutoff = (now.date() - timedelta(days=_CURVE_KEEP_DAYS)).isoformat()
        conn.execute("DELETE FROM at_curve WHERE user_id = ? AND trade_date < ?",
                     (user_id, cutoff))
    return True


def get_curve(user_id: int, mode: str, days: int = 1) -> dict:
    """시장별 순수익률 곡선.

    days=1 이면 오늘치만. 그 이상이면 지난 날들의 마감 수익률을 복리로 이어
    붙여(cum_pct) 오늘 점까지 한 줄로 그릴 수 있게 합니다 — 하루치 그래프가
    매일 0% 로 리셋되기만 하면 "이 자동매매가 지금까지 얼마를 벌었나"에
    답할 수 없습니다.
    """
    init()
    days = max(1, min(int(days or 1), _CURVE_KEEP_DAYS))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM at_curve WHERE user_id = ? AND mode = ? AND trade_date >= ? "
            "ORDER BY id ASC", (user_id, mode, since))]

    series: dict[str, list] = {}
    base: dict[str, float] = {}
    # scope → trade_date → 그 날의 첫 표본 (0% 기준선)
    anchors: dict[tuple, dict] = {}
    # scope → 이미 지나간 날들의 누적 배수 (복리)
    carried: dict[str, float] = {}
    seen_days: dict[str, str] = {}
    last_pct: dict[str, float] = {}

    for row in rows:
        scope, day = row["scope"], row["trade_date"]
        anchor = anchors.setdefault((scope, day), row)
        if seen_days.get(scope) not in (None, day):
            # 날이 바뀌었습니다 — 지난 날의 마지막 수익률을 누적에 곱해 넣습니다
            carried[scope] = carried.get(scope, 1.0) * (1 + last_pct.get(scope, 0.0) / 100.0)
        seen_days[scope] = day

        net = ((row["realized_krw"] - anchor["realized_krw"])
               + (row["unreal_krw"] - anchor["unreal_krw"])
               - (row["exit_cost_krw"] - anchor["exit_cost_krw"]))
        ground = anchor["base_krw"] or row["base_krw"] or 0.0
        pct = (net / ground * 100.0) if ground > 0 else 0.0
        last_pct[scope] = pct
        base[scope] = ground
        cum = (carried.get(scope, 1.0) * (1 + pct / 100.0) - 1) * 100.0
        series.setdefault(scope, []).append({
            "ts": row["ts"],
            "sec": _sec_of_day_iso(row["ts"]),
            "date": day,
            "pnl": round(net, 2),
            "pct": round(pct, 4),
            "cum_pct": round(cum, 4),
            "equity": round(row["equity_krw"], 2),
            "base": round(ground, 2),
        })

    return {
        "days": days, "mode": mode, "series": series, "base": base,
        "now": {scope: points[-1] for scope, points in series.items() if points},
    }


# ---------------------------------------------------------------------------
# 계좌 교체 — 성적 초기화
# ---------------------------------------------------------------------------
# 계좌번호를 바꾸면 화면의 '오늘 손익 · 누적 손익'은 **남의 계좌 성적**입니다.
# 그대로 두면 새 계좌의 첫 매매가 이미 -6,646원에서 시작한 것처럼 보이고,
# 더 나쁘게는 일일 손실 한도가 그 남은 손실 때문에 곧바로 걸려 매매가 멈춥니다.
# 그래서 계좌가 바뀐 것을 발견하면 그 자리에서 0원으로 되돌립니다.

def get_account_epoch(user_id: int, mode: str) -> dict | None:
    """이 계좌를 언제부터 세고 있는지. 기록이 없으면 None."""
    init()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM at_account WHERE user_id = ? AND mode = ?",
            (user_id, mode)).fetchone()
    return dict(row) if row else None


def reset_pnl(user_id: int, mode: str, fingerprint: str = None,
              reason: str = "") -> dict:
    """이 계좌의 오늘 손익 · 누적 손익을 0원으로 되돌립니다.

    **무엇을 지우고, 무엇을 남기는가**
        지웁니다   at_daily 의 해당 계좌 줄 전부. 누적 손익은 이 줄들의 합이라
                   (summary 참고) 지우면 0이 되고, 오늘 줄이 사라졌으므로 다음
                   회전에서 지금의 평가손익을 기준선으로 새로 잡습니다 —
                   즉 오늘 손익도 그 시점부터 0에서 시작합니다.
        남깁니다   at_orders(주문 원장)와 at_events(기록). 실제로 나갔던 주문을
                   지우면 무슨 일이 있었는지 확인할 길이 없어집니다. 대신
                   at_account.reset_at 을 남겨서, 주문 수·승률·실현손익 집계가
                   이 시각 **이후**만 세도록 합니다 (summary 의 since 조건).

    reset_at 을 남기는 이유가 하나 더 있습니다: 누적은 0인데 '└ 실현'만 옛
    계좌의 숫자가 남아 있으면, 화면 두 줄이 서로 다른 계좌를 가리키게 됩니다.
    """
    init()
    now = datetime.now().isoformat()
    with _conn() as conn:
        if fingerprint is None:
            row = conn.execute(
                "SELECT fingerprint FROM at_account WHERE user_id = ? AND mode = ?",
                (user_id, mode)).fetchone()
            fingerprint = row["fingerprint"] if row else ""
        cleared = conn.execute(
            "DELETE FROM at_daily WHERE user_id = ? AND mode = ?",
            (user_id, mode)).rowcount
        # 수익률 곡선도 그 계좌의 것입니다. 남겨두면 새 계좌의 첫 화면에
        # 남의 계좌 그래프가 그려집니다.
        conn.execute("DELETE FROM at_curve WHERE user_id = ? AND mode = ?",
                     (user_id, mode))
        conn.execute(
            "INSERT OR REPLACE INTO at_account (user_id, mode, fingerprint, reset_at) "
            "VALUES (?, ?, ?, ?)", (user_id, mode, fingerprint or "", now))

    log_event(user_id, "reset",
              f"[{mode}] 오늘 손익·누적 손익을 0원으로 되돌렸습니다"
              + (f" — {reason}" if reason else ""),
              level="warn", detail={"mode": mode, "days_cleared": cleared})
    return {"user_id": user_id, "mode": mode, "reset_at": now,
            "days_cleared": cleared}


def sync_account(user_id: int, mode: str, fingerprint: str) -> bool:
    """계좌가 바뀌었으면 성적을 0으로 되돌립니다. 되돌렸을 때만 True.

    매 회전 · 매 화면 갱신마다 불립니다. 값싸야 하고(조회 한 번), 무엇보다
    **틀린 쪽으로 기울면 안 됩니다.** 그래서 두 경우는 아무것도 하지 않습니다.

        지문이 빈 값        계좌 미설정이거나 키를 잠깐 못 읽은 상태입니다.
                            "계좌가 바뀌었다"가 아니라 "모른다" 입니다. 여기서
                            지우면 Mongo 가 1분 끊긴 것만으로 성적이 날아갑니다.
        처음 보는 계좌       지금까지의 기록이 바로 이 계좌 것입니다 (이 기능이
                            없던 때부터 쓰던 계좌). 기준만 적어 두고 지나갑니다.
        적힌 지문이 빈 값    손으로 초기화했는데(reset_pnl) 그때 계좌번호를 몰라
                            비워 둔 경우입니다. 역시 "모른다" 이므로, 지금 지문을
                            적어만 두고 넘어갑니다 — 여기서 또 지우면 방금 한
                            초기화가 다음 회전에 한 번 더 일어납니다.
    """
    if not fingerprint:
        return False
    init()
    with _conn() as conn:
        row = conn.execute(
            "SELECT fingerprint, reset_at FROM at_account "
            "WHERE user_id = ? AND mode = ?", (user_id, mode)).fetchone()
        if row is None or not row["fingerprint"]:
            # 처음 보는 계좌라면 reset_at 은 빈 값으로 둡니다 — summary 가 예전
            # 주문까지 그대로 셉니다. 손으로 초기화한 뒤라면 그때 찍힌
            # reset_at 을 그대로 지킵니다.
            conn.execute(
                "INSERT OR REPLACE INTO at_account "
                "(user_id, mode, fingerprint, reset_at) VALUES (?, ?, ?, ?)",
                (user_id, mode, fingerprint,
                 (row["reset_at"] if row else "") or ""))
            return False
        if row["fingerprint"] == fingerprint:
            return False

    reset_pnl(user_id, mode, fingerprint=fingerprint, reason="계좌가 바뀌었습니다")
    return True


def summary(user_id: int, mode: str = None) -> dict:
    """콘솔 상단에 띄울 요약 통계 (계좌별).

    가상 자금 성적과 실제 자금 성적을 한 숫자로 섞으면 아무 의미가 없습니다.
    계좌를 바꾼 뒤라면 **바꾼 시각 이후**만 셉니다 (reset_pnl 참고).
    """
    init()
    scope = " AND broker_mode = ?" if mode else ""
    args = (user_id, mode) if mode else (user_id,)
    with _conn() as conn:
        since = ""
        if mode:
            epoch = conn.execute(
                "SELECT reset_at FROM at_account WHERE user_id = ? AND mode = ?",
                (user_id, mode)).fetchone()
            since = (epoch["reset_at"] if epoch else "") or ""
        if since:
            scope += " AND created_at >= ?"
            args = args + (since,)
        orders = conn.execute(
            "SELECT COUNT(*) c, SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) r "
            f"FROM at_orders WHERE user_id = ?{scope}", args).fetchone()
        pnl = conn.execute(
            "SELECT COUNT(*) n, SUM(realized_pnl) total, "
            "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) wins "
            f"FROM at_orders WHERE user_id = ? AND realized_pnl IS NOT NULL{scope}",
            args).fetchone()
        # 누적 손익 — 날짜별로 확정해 둔 '오늘 손익'을 전부 더합니다.
        # 실현손익(판 것)만 세면, 사서 오르는 중인 종목의 성과가 통째로 빠져
        # 오늘 하루 매매를 시작한 계좌는 영원히 0원으로 보입니다.
        # at_daily 는 계좌를 바꿀 때 통째로 지우므로(reset_pnl) 시각 조건이
        # 필요 없습니다 — 남아 있는 줄은 전부 지금 계좌의 것입니다.
        day_scope = " AND mode = ?" if mode else ""
        day_args = (user_id, mode) if mode else (user_id,)
        cumulative = conn.execute(
            f"SELECT SUM(day_pnl) t FROM at_daily WHERE user_id = ?{day_scope}",
            day_args).fetchone()

    closed = int(pnl["n"] or 0)
    wins = int(pnl["wins"] or 0)
    return {
        "orders": int(orders["c"] or 0),
        "rejected": int(orders["r"] or 0),
        "closed_trades": closed,
        "wins": wins,
        "win_rate": (wins / closed * 100) if closed else 0.0,
        "realized_pnl": float(pnl["total"] or 0),
        # 평가손익까지 포함한 누적 성과 (일자별 '오늘 손익'의 합)
        "cumulative_pnl": float(cumulative["t"] or 0),
    }
