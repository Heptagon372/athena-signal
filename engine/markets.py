"""
시장 상수 (Market Constants) — KRX · NXT · US
---------------------------------------------
매직넘버를 코드 곳곳에 흩뿌리지 않기 위한 단일 출처입니다. 모든 값에 **출처와
시행일** 을 주석으로 남깁니다.

수록 항목
    sell_tax_rate       증권거래세 — **시행일 키 테이블** (과거 백테스트는 과거 세율)
    tick_size           호가가격단위 (2023-01-25 개편 이후)
    price_limit         가격제한폭 ±30%
    round_trip_cost_bps 왕복 비용 추정 (세금 + 수수료 + 최소 스프레드)
    tax_drag_per_year   회전율 → 연간 세금 드래그
    sessions            KRX / NXT 세션 시간표
    short_sale_allowed  공매도 금지 이력 (2020~2025)
    KIS_LIMITS          한국투자증권 API 유량 제한

★ 이 파일에서 가장 중요한 설계 — 세율은 상수가 아니라 시간의 함수입니다
    증권거래세는 **2026-01-01 에 인상되었습니다.** 그래서 세율을 상수 하나로
    두면 두 가지가 동시에 틀립니다.

        · 오늘의 주문 비용을 과소평가한다        (실매매가 손해)
        · 2020년 구간 백테스트에 2026년 세율을 쓴다 (백테스트가 과도하게 비관적)

    두 번째가 특히 조용합니다. 과거를 비관적으로 평가하면 "이 전략은 안 되네"
    라는 결론이 나오는데, 그건 전략이 아니라 세율 때문입니다. 그래서
    `sell_tax_rate(date, board)` 로 조회하게 만들었습니다.

    **이것이 설계도 §5.1 의 bitemporal 원칙을 시장 상수에 적용한 것입니다.**
    가격 데이터만 PIT 여야 하는 게 아니라 규칙도 PIT 여야 합니다.

⚠ 저장소 현황과의 불일치 (2026-08-08 확인)
    이 모듈을 만들기 전 저장소에는 서로 다른 세율 두 개가 있었습니다.

        engine/instruments.py:61   TAX_STOCK_KR = 0.0018   (0.18%)
        engine/scalping.py:49      TAX_RATE     = 0.00150  (0.15%)

    둘 다 2026년 현행 0.20% 보다 낮습니다. 특히 scalping 쪽이 **가장 비용에
    민감한 경로**(페니주 초단타, 회전율 최고)인데 가장 낮게 잡혀 있었습니다.
    본전 틱수 계산이 그만큼 낙관적이 됩니다.

    **이 모듈은 올바른 값을 제공만 하고, 기존 상수를 자동으로 바꾸지 않습니다.**
    실계좌 비용 계산을 조용히 바꾸는 것은 위험하므로 호출부 교체는 명시적으로
    하십시오. `audit_repo_constants()` 가 현재 불일치를 보고합니다.
"""

import bisect
from datetime import date, datetime

# ---------------------------------------------------------------------------
# 증권거래세 — 시행일 테이블
# ---------------------------------------------------------------------------
# 출처: 조세특례제한법·증권거래세법 시행령. 2026-01-01 시행분 인상.
#   KOSPI(유가증권)  = 증권거래세 0.05% + 농어촌특별세 0.15% = 0.20%
#   KOSDAQ           = 증권거래세 0.20%, 농특세 없음        = 0.20%
#   KONEX            = 0.10%
#   ETF / ETN 매도   = 면제
# 매수에는 부과되지 않습니다. **매도 시에만.**
#
# 각 항목은 (시행일, 세율) 이고 시행일 오름차순입니다.
SELL_TAX_SCHEDULE: dict[str, list] = {
    "KOSPI": [
        (date(2019, 6, 3), 0.0025),   # 거래세 0.10% + 농특세 0.15%
        (date(2021, 1, 1), 0.0023),   # 거래세 0.08% + 농특세 0.15%
        (date(2023, 1, 1), 0.0020),   # 거래세 0.05% + 농특세 0.15%
        (date(2024, 1, 1), 0.0018),   # 거래세 0.03% + 농특세 0.15%
        (date(2025, 1, 1), 0.0015),   # 거래세 0.00% + 농특세 0.15%
        (date(2026, 1, 1), 0.0020),   # ★ 인상 — 거래세 0.05% + 농특세 0.15%
    ],
    "KOSDAQ": [
        (date(2019, 6, 3), 0.0025),
        (date(2021, 1, 1), 0.0023),
        (date(2023, 1, 1), 0.0020),
        (date(2024, 1, 1), 0.0018),
        (date(2025, 1, 1), 0.0015),
        (date(2026, 1, 1), 0.0020),   # ★ 인상
    ],
    "KONEX": [
        (date(2019, 6, 3), 0.0010),
    ],
    "ETF": [(date(2000, 1, 1), 0.0)],   # 면제
    "ETN": [(date(2000, 1, 1), 0.0)],
}

# 위탁수수료 (온라인). 증권사별 0.0036%~0.015% — 보수적으로 상단을 씁니다.
# 비용을 낮게 잡는 쪽의 실수가 훨씬 비쌉니다: 통과하면 안 되는 매매가 통과합니다.
COMMISSION_BPS_KR = 1.5          # 0.015%
COMMISSION_BPS_US = 0.0          # 대부분 제로커미션. SEC/TAF 수수료는 극소

PRICE_LIMIT_PCT_KR = 30.0        # ±30%. 도달 시 해당 방향 체결 불가
SETTLEMENT_LAG_KR = 2            # T+2 (2026 하반기 T+1 업무표준 마련 예정)
SETTLEMENT_LAG_US = 1


def _lookup_schedule(schedule: list, when) -> float:
    """시행일 테이블에서 해당 시점의 값을 찾습니다 (가장 최근 시행분)."""
    d = _as_date(when)
    dates = [row[0] for row in schedule]
    i = bisect.bisect_right(dates, d) - 1
    if i < 0:
        return schedule[0][1]
    return schedule[i][1]


def _as_date(when) -> date:
    if when is None:
        return date.today()
    if isinstance(when, datetime):
        return when.date()
    if isinstance(when, date):
        return when
    if isinstance(when, str):
        return datetime.fromisoformat(when[:10]).date()
    raise TypeError(f"날짜로 해석할 수 없습니다: {when!r}")


def sell_tax_rate(when=None, board: str = "KOSPI") -> float:
    """해당 **시점** 의 매도 증권거래세율 (소수, 예: 0.0020).

    board: "KOSPI" | "KOSDAQ" | "KONEX" | "ETF" | "ETN"

    백테스트에서는 반드시 **그 봉의 날짜** 를 넘기세요. 오늘 날짜를 넘기면
    2020년 매매에 2026년 세율이 붙습니다.
    """
    key = (board or "KOSPI").upper()
    schedule = SELL_TAX_SCHEDULE.get(key)
    if schedule is None:
        schedule = SELL_TAX_SCHEDULE["KOSPI"]
    return float(_lookup_schedule(schedule, when))


# ---------------------------------------------------------------------------
# 호가가격단위
# ---------------------------------------------------------------------------
# 출처: KRX 2023-01-25 개편. 코스피·코스닥 통일.
# (가격 상한 미만, 호가단위) — 오름차순
TICK_TABLE_KR = [
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
    (float("inf"), 1_000),
]


def tick_size(price: float, market: str = "KR") -> float:
    """가격대별 호가단위.

    **저가주의 상대 틱이 크다는 점이 전략 설계를 좌우합니다.**
    1,000원 주식의 1원 틱 = 10bp 입니다. 동일가중 백테스트에 저가주가 많으면
    bid-ask bounce 만으로 프리미엄이 구조적으로 과대추정됩니다. 유니버스에
    최소 가격 필터를 넣으세요 (`engine/scalping.py` 의 kr_min_price 가 같은 이유).
    """
    if market.upper() != "KR":
        return 0.01
    p = float(price)
    for upper, unit in TICK_TABLE_KR:
        if p < upper:
            return float(unit)
    return 1_000.0


def half_spread_bps(price: float, market: str = "KR") -> float:
    """최소 슬리피지 하한 = 호가단위의 절반을 bp 로.

    아무리 좋은 체결이어도 이보다 나을 수 없습니다. 시장충격 모형이 0을 내도
    이 값이 남습니다.
    """
    p = float(price)
    if p <= 0:
        return 0.0
    return (tick_size(p, market) / p) * 1e4 / 2.0


# ---------------------------------------------------------------------------
# 왕복 비용과 세금 드래그
# ---------------------------------------------------------------------------

def round_trip_cost_bps(price: float, when=None, board: str = "KOSPI",
                        commission_bps: float = None,
                        include_spread: bool = True) -> dict:
    """왕복(매수 → 매도) 비용 추정 (bp).

    구성: 매수 수수료 + 매도 수수료 + 매도 거래세 + (선택) 왕복 최소 스프레드.
    시장충격은 주문 크기에 의존하므로 여기 없습니다 — `impact_bps()` 를 더하세요.
    """
    comm = COMMISSION_BPS_KR if commission_bps is None else float(commission_bps)
    tax_bps = sell_tax_rate(when, board) * 1e4
    spread = (half_spread_bps(price) * 2.0) if include_spread else 0.0
    total = comm * 2 + tax_bps + spread
    return {
        "total_bps": round(total, 2),
        "commission_bps": round(comm * 2, 3),
        "tax_bps": round(tax_bps, 2),
        "spread_bps": round(spread, 2),
        "board": board,
        "as_of": str(_as_date(when)),
    }


def impact_bps(order_value: float, adv: float, daily_vol: float = 0.02,
               eta: float = 0.7) -> float:
    """참여율 기반 제곱근 시장충격.

        impact_bps = η · σ_daily · sqrt(order_value / ADV) · 1e4

    **η 는 추정치입니다.** 초기값 0.7 은 문헌 범위(0.5~1.0)의 중간이고, 실제
    체결 이력이 쌓이면 자기 데이터로 캘리브레이션하는 것이 원칙입니다.
    플랫폼 기본값을 그대로 두지 마세요.
    """
    if adv is None or adv <= 0 or order_value <= 0:
        return 0.0
    participation = float(order_value) / float(adv)
    return float(eta) * float(daily_vol) * (participation ** 0.5) * 1e4


def max_order_value(adv: float, max_participation: float = 0.10) -> float:
    """ADV 대비 주문 상한. 기본 10%.

    상한이 없으면 백테스트가 시장 전체를 사들이고도 아무 불평을 하지 않습니다.
    """
    return max(float(adv or 0.0), 0.0) * float(max_participation)


def tax_drag_per_year(rebalances_per_year: float, turnover: float,
                      when=None, board: str = "KOSPI") -> dict:
    """회전율 → 연간 세금 드래그. **전략 선택의 1차 필터입니다.**

        연간 세금 = 리밸런싱 횟수 × 회전율 × 세율

    감각을 위한 수치 (2026년 0.20% 기준):

        분기 리밸런싱, 회전율 50%  →  4 × 0.5 × 0.20% = 0.40%/yr   감당 가능
        월간 리밸런싱, 회전율 50%  → 12 × 0.5 × 0.20% = 1.20%/yr   주의
        주간 리밸런싱, 회전율 80%  → 52 × 0.8 × 0.20% = 8.32%/yr   전략 사망

    이 저장소의 실측이 정확히 그 셋째 줄입니다 — AUTOTRADE.md 16장의 주간 반전
    전략에서 누적비용이 한국 55~90% 였고, 시야를 월간으로 늘려 회전율을
    0.31 → 0.025 로 낮추자 누적비용이 5~7% 가 되면서 결과가 뒤집혔습니다.
    **전략을 바꾼 게 아니라 회전율을 바꾼 것이 결정적이었습니다.**
    """
    rate = sell_tax_rate(when, board)
    drag = float(rebalances_per_year) * float(turnover) * rate
    if drag < 0.01:
        verdict = "감당 가능"
    elif drag < 0.03:
        verdict = "주의 — 기대 알파와 직접 비교하세요"
    else:
        verdict = "위험 — 이 회전율에서 세금만으로 대부분의 알파가 사라집니다"
    return {
        "annual_tax_drag": round(drag, 5),
        "annual_tax_drag_pct": round(drag * 100, 3),
        "tax_rate": rate, "board": board,
        "rebalances_per_year": rebalances_per_year, "turnover": turnover,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 세션 — "한국 주식은 09:00~15:30" 은 2025년 이후 틀렸습니다
# ---------------------------------------------------------------------------
# NXT(넥스트레이드)는 2025-03-04 출범한 국내 최초 ATS 입니다.
# 실제 거래 가능 시간은 08:00~20:00 입니다.
SESSIONS_KR = {
    "KRX": {
        "opening_auction": ("08:30", "09:00"),
        "regular": ("09:00", "15:30"),
        "closing_auction": ("15:20", "15:30"),   # + 랜덤엔드 최대 30초
        "after_hours_single": ("16:00", "18:00"),  # 10분 단위, 종가 ±10%
    },
    "NXT": {
        "pre": ("08:00", "08:50"),
        "main": ("09:00:30", "15:20"),
        "after": ("15:30", "20:00"),
    },
}

# NXT 는 메인마켓에서 일반 시장가호가 불가 — IOC/FOK 만 허용됩니다.
NXT_ORDER_TYPES_MAIN = ("limit", "ioc", "fok", "midpoint", "stop_limit")
NXT_FEE_BPS = {"maker": 0.013, "taker": 0.018}   # KRX 는 0.023
KRX_FEE_BPS = 0.023

# ⚠ 데이터 한계 — 유동성 필터에 직접 영향
#   pykrx / FinanceDataReader 가 주는 일봉은 **KRX 기준** 입니다.
#   KOSPI 정규장 NXT 점유율: 2026-01 34.1% → 02 35.6% → 03-03(급락일) 47.7%.
#   즉 변동성이 클수록 NXT 로 이동합니다. KRX 거래대금만으로 계산한 ADV 는
#   **30~48% 과소평가** 이고, 그 위에 얹은 시장충격 추정도 같이 틀립니다.
#   → 유동성 필터 임계값은 보수적으로, 충격 추정은 ADV 를 상향 보정하세요.
NXT_SHARE_ESTIMATE = 0.35        # KOSPI 정규장 기준 보수적 추정


def adv_adjusted_for_nxt(krx_adv: float, nxt_share: float = None) -> float:
    """KRX 기준 ADV 를 전체 시장 기준으로 보정합니다.

    KRX 점유율이 (1 − nxt_share) 이므로 전체 = KRX / (1 − nxt_share).
    **보정하지 않으면 충격 추정이 과대**(주문이 실제보다 큰 참여율로 보임)
    되어 전략이 필요 이상으로 보수적이 됩니다. 반대로 유동성 필터는
    보정하지 **않은** 값으로 거는 쪽이 안전합니다 — 두 용도가 반대 방향입니다.
    """
    share = NXT_SHARE_ESTIMATE if nxt_share is None else float(nxt_share)
    share = min(max(share, 0.0), 0.9)
    return float(krx_adv or 0.0) / (1.0 - share)


# ---------------------------------------------------------------------------
# 가격안정화장치
# ---------------------------------------------------------------------------
# VI 발동 중에는 체결이 멈춥니다. 이것을 "데이터 끊김" 으로 오인해 재접속
# 루프에 빠지거나, 단일가 구간의 **예상체결가** 를 실시간 체결가로 오인하는
# 것이 흔한 버그입니다. 명시적 플래그로 다루세요.
VI_STATIC_PCT = 10.0             # 직전 단일가 대비 ±10% → 2분 단일가
VI_DYNAMIC_PCT = {"KOSPI200": 3.0, "OTHER": 6.0}
CIRCUIT_BREAKER_PCT = (8.0, 15.0, 20.0)   # 하락 시에만
SIDECAR_PCT = 5.0                # KOSPI200 선물 ±5% 1분 지속 → 프로그램매매 5분 정지


# ---------------------------------------------------------------------------
# 공매도 — 개인에게는 사실상 닫혀 있습니다
# ---------------------------------------------------------------------------
# (시작일, 종료일 또는 None, 상태) — 상태: "banned" | "partial" | "allowed"
SHORT_SALE_HISTORY = [
    (date(2020, 3, 16), date(2021, 5, 2), "banned"),
    (date(2021, 5, 3), date(2023, 11, 5), "partial"),   # KOSPI200/KOSDAQ150 만
    (date(2023, 11, 6), date(2025, 3, 30), "banned"),
    (date(2025, 3, 31), None, "allowed"),
]


def short_sale_allowed(when=None, in_major_index: bool = False) -> dict:
    """해당 시점에 공매도가 가능했는가.

    **2020~2025 구간을 포함하는 KR 롱숏 백테스트는 이 이력을 반영하지 않으면
    허구입니다.** 그 5년 중 약 3년이 전면 금지였습니다.

    그리고 제도상 허용되어도 **개인 참여는 사실상 0** 입니다 — 2026년 4~5월
    공매도 37.6조원 중 개인은 30억원(0.008%)이었습니다. 개인 대주는 증권사
    재원 한정에 사전교육 필수이고 종목도 제한적입니다.

    → **개인 계좌 기준으로는 롱온리를 기본값으로 두는 것이 맞습니다.**
      시장중립이 필요하면 KOSPI200 선물이나 인버스 ETF 로 헤지하세요.
    """
    d = _as_date(when)
    state = "allowed"
    for start, end, s in SHORT_SALE_HISTORY:
        if d >= start and (end is None or d <= end):
            state = s
            break
    allowed = (state == "allowed") or (state == "partial" and in_major_index)
    return {
        "date": str(d), "state": state, "allowed_institutionally": allowed,
        "retail_practical": False,
        "note": ("개인 대주는 재원·교육·종목 제한으로 사실상 불가합니다. "
                 "개인 계좌 백테스트는 롱온리로 두세요."),
    }


# ---------------------------------------------------------------------------
# KIS API 유량 제한 — 아키텍처를 결정하는 하드 제약
# ---------------------------------------------------------------------------
KIS_LIMITS = {
    "rest_per_second": 20,          # 실전. 초과 시 EGW00201
    "rest_safe_per_second": 18,     # 실측 안전치
    "rest_safe_per_minute": 900,
    "websocket_symbols_per_session": 41,   # 1 approval key = 1 세션
    "token_issue_per_minute": 1,
    "mock_api_coverage": (43, 336),        # 모의투자 지원 43 / 전체 336
}

# 함의 세 가지 — 이 저장소 구조에 직접 해당합니다
#  1) 리서치와 실행을 분리하세요. 유니버스 스크리닝·팩터 계산을 브로커 API 로
#     하면 반드시 유량에 막힙니다. 일별 배치로 로컬 캐시에 쌓으세요.
#  2) **실시간 스트리밍 상한이 41종목입니다.** 코스피200 전체 실시간 추적은
#     불가능합니다. 실시간은 보유 포지션 + 당일 주문 대상에만 씁니다.
#  3) 모의투자 통과는 실전 검증이 아닙니다 (43/336).


def audit_repo_constants() -> dict:
    """저장소에 흩어진 비용 상수가 이 테이블과 맞는지 점검합니다.

    **자동으로 고치지 않습니다.** 실계좌 비용 계산을 조용히 바꾸면 안 되므로
    보고만 합니다. 교체 여부는 사람이 판단하세요.
    """
    today = date.today()
    expected = sell_tax_rate(today, "KOSPI")
    findings = []
    try:
        from engine import instruments
        actual = float(getattr(instruments, "TAX_STOCK_KR", float("nan")))
        if abs(actual - expected) > 1e-9:
            findings.append({
                "where": "engine/instruments.py:TAX_STOCK_KR",
                "actual": actual, "expected": expected,
                "impact_bps": round((expected - actual) * 1e4, 2),
                "note": "일반 주식 매도 비용이 과소/과대 계상됩니다.",
            })
    except Exception as exc:
        findings.append({"where": "engine/instruments.py", "error": repr(exc)})
    try:
        from engine import scalping
        actual = float(getattr(scalping, "TAX_RATE", float("nan")))
        if abs(actual - expected) > 1e-9:
            findings.append({
                "where": "engine/scalping.py:TAX_RATE",
                "actual": actual, "expected": expected,
                "impact_bps": round((expected - actual) * 1e4, 2),
                "note": ("페니주 초단타는 회전율이 가장 높아 이 오차가 가장 크게 "
                         "누적됩니다. 본전 틱수 계산이 낙관적이 됩니다."),
            })
    except Exception as exc:
        findings.append({"where": "engine/scalping.py", "error": repr(exc)})

    return {
        "as_of": str(today),
        "expected_sell_tax": expected,
        "n_mismatches": len([f for f in findings if "actual" in f]),
        "findings": findings,
        "action": ("불일치는 보고만 합니다. 교체하려면 호출부를 "
                   "markets.sell_tax_rate(bar_date, board) 로 바꾸세요 — "
                   "상수를 새 상수로 바꾸면 과거 백테스트가 다시 틀립니다."),
    }
