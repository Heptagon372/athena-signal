"""
보유 비용 & 제도 제약 (Holding Costs & Regulatory Constraints)
--------------------------------------------------------------

체결(engine/fills.py)이 "그 가격에 살 수 있었나" 라면, 여기는
**"애초에 그 포지션을 들고 있을 수 있었나, 있었다면 얼마가 들었나"** 입니다.

수록한 것 둘
    1. 공매도 금지 이력  — 숏 진입을 **제도적으로 막습니다**
    2. 대주 차입비용     — 숏을 들고 있는 동안 매일 나가는 돈

★ 이 파일의 성격 — **고장을 고친 게 아니라 미래의 고장을 막는 안전장치입니다.**

    솔직하게 써 둡니다. 지금 아테나는 주식 숏을 낼 수 없습니다.
    `instruments.from_symbol` 이 주식·ETF 를 `shortable=False` 로 못박고
    (instruments.py:446), `strategy.evaluate` 가
    `allow_short = cfg.allow_short and inst.shortable` 로 한 번 더 막습니다.
    그래서 **오늘 이 검사를 켜도 바뀌는 숫자는 없습니다.**

    그런데 `markets.SHORT_SALE_HISTORY` 는 진작 있었으면서 아무 데서도
    강제되지 않았고, markets.py 자신이 이렇게 씁니다 —

        "2020~2025 구간을 포함하는 KR 롱숏 백테스트는 이 이력을 반영하지
         않으면 허구입니다. 그 5년 중 약 3년이 전면 금지였습니다."

    누군가 롱숏을 연구하려고 `shortable=True` 한 줄을 바꾸는 날, 그 백테스트는
    **조용히 허구가 됩니다.** 금지 구간이 하필 하락장(2020-03, 2022)이라
    숏 수익이 가장 크게 잡히는 구간이 정확히 불가능했던 구간입니다.
    그때 이 검사가 없으면 아무도 눈치채지 못합니다.

    한 줄을 바꾸는 사람이 두 파일을 다 기억할 거라고 가정하지 않습니다.


일부러 만들지 **않은** 것 둘 — 없어서가 아니라 해당이 없어서입니다
--------------------------------------------------------------

  · 무기한선물 펀딩피
        이 저장소에 암호화폐 인스트루먼트가 없습니다 (STOCK / ETF /
        FUTURES / OPTION 뿐). 펀딩피는 무기한선물의 개념이고 KRX 선물은
        만기가 있어 해당하지 않습니다. 껍데기만 만들어 두면 나중에 누군가
        "펀딩피 반영됨" 으로 읽습니다. 암호화폐를 붙이는 날 같이 만드세요.

  · 선물 롤오버 비용
        이미 모델링돼 있습니다. `strategy.check_exit` 가
        `deriv_min_days_to_expiry` 로 만기 전 강제 청산하고, 재진입은 다음
        신호에서 일어나며 **양쪽 다 수수료를 냅니다.** 여기서 롤 비용을 또
        빼면 이중 계상입니다.

  · ETF 총보수 / 레버리지·인버스 ETF 의 변동성 감쇠
        **이미 가격에 들어 있습니다.** ETF 시장가는 보수를 차감한 NAV 를
        따라가고, 레버리지 상품의 일일 리밸런싱 복리 오차도 그 가격 시계열에
        그대로 나타납니다. 따로 빼면 이중 계상입니다.
        (수정주가가 아니라 NAV 를 직접 주입하는 날이 오면 다시 보세요.)

  · 현금계좌 매수 대금의 기회비용·이자
        아테나는 증거금이 현금을 넘으면 주문을 거부합니다(미수 없음).
        빌린 돈이 없으니 이자도 없습니다.
"""

from datetime import date, datetime

from engine import fills, markets
from engine.instruments import ETF, FUTURES, OPTION, STOCK

# 대주(개인)·대차(기관) 차입 이자 연율. 증권사·종목·시점마다 다르므로 근사치이고,
# 설정(`short_borrow_rate_annual`)으로 덮어쓸 수 있습니다.
#
# 하드코딩하지 않고 기본값만 두는 이유: 실제로는 종목별 대차잔고에 따라
# 2%(대형주)~20%(품절주)까지 벌어집니다. 하나로 고정하면 숏 전략의 비용이
# 체계적으로 과소평가됩니다 — 특히 빌리기 어려운 종목일수록 숏하고 싶어지는
# 종목이라 편향이 한쪽으로 쏠립니다.
DEFAULT_BORROW_RATE_KR = 0.025      # 연 2.5% (대형주 대주 근사)

BORROW_DAYS_PER_YEAR = 365          # 이자는 달력일 기준입니다 (영업일 아님)

# 거부 사유는 engine/fills.py 한 곳에 모아 둡니다 (라벨·집계가 거기서 돕니다)
SHORT_BANNED = fills.SHORT_BANNED


def short_entry_allowed(inst, when=None, cfg: dict = None) -> dict:
    """이 시점에 이 종목을 숏으로 진입할 수 있었는가.

    돌려주는 값
        allowed  bool   — 진입 가능 여부
        reason   str     — 불가 사유 (사람이 읽는 문장)
        state    str     — 'allowed' | 'partial' | 'banned' | 'n/a'

    자산군별로 규칙이 다릅니다.
        선물·옵션   매도 진입은 공매도가 아닙니다. 금지 이력과 무관합니다.
        주식·ETF    `markets.SHORT_SALE_HISTORY` 를 그대로 따릅니다.
        미국        이 이력은 KR 제도라 적용하지 않습니다.

    `assume_retail` (기본 True) 이면 제도상 허용 구간이어도 막습니다.
    markets.short_sale_allowed 의 판단을 그대로 가져온 것입니다 — 개인 대주는
    재원·사전교육·종목 제한으로 사실상 불가하고, 2026년 실적으로 전체 공매도의
    0.008% 였습니다. 기관 전략을 연구할 때만 False 로 내리세요.
    """
    cfg = cfg or {}

    if inst.asset_class in (FUTURES, OPTION):
        return {"allowed": True, "state": "n/a",
                "reason": "선물·옵션 매도 진입은 공매도 규제 대상이 아닙니다."}
    if not inst.is_korean:
        return {"allowed": True, "state": "n/a",
                "reason": "국내 공매도 이력은 해외 종목에 적용하지 않습니다."}
    if inst.asset_class not in (STOCK, ETF):
        return {"allowed": True, "state": "n/a", "reason": ""}

    # KOSPI200/KOSDAQ150 편입 여부를 이 계층에서는 알 수 없습니다.
    # 부분 허용 구간을 "허용" 으로 읽으면 실제로는 불가능했던 종목까지
    # 숏이 열리므로, 모를 때는 **막는 쪽** 으로 둡니다.
    in_major = bool(cfg.get("short_in_major_index", False))
    verdict = markets.short_sale_allowed(when, in_major_index=in_major)
    state = verdict["state"]

    if not verdict["allowed_institutionally"]:
        label = {"banned": "전면 금지", "partial": "부분 허용(주요지수만)"}.get(state, state)
        return {"allowed": False, "state": state,
                "reason": f"{verdict['date']} 시점 공매도 {label} 구간입니다."}

    if cfg.get("assume_retail", True):
        return {"allowed": False, "state": state,
                "reason": "제도상 허용 구간이지만 개인 대주는 사실상 불가합니다 "
                          "(재원·교육·종목 제한). 기관 기준으로 보려면 "
                          "assume_retail=False."}

    return {"allowed": True, "state": state, "reason": ""}


def borrow_rate(inst, cfg: dict = None) -> float:
    """이 종목의 연 차입 이자율. 숏이 아니면 0."""
    cfg = cfg or {}
    if inst.asset_class in (FUTURES, OPTION):
        return 0.0          # 증거금 상품 — 빌리는 것이 아닙니다
    return float(cfg.get("short_borrow_rate_annual", DEFAULT_BORROW_RATE_KR) or 0.0)


def borrow_cost(inst, notional: float, days: float, cfg: dict = None) -> float:
    """숏을 `days` 일 들고 있을 때의 차입비용.

    명목금액 기준입니다 — 빌린 것은 현금이 아니라 **주식** 이므로, 주가가
    오르면 이자도 같이 늘어납니다. 여기서는 진입 명목가로 근사합니다
    (봉마다 정확히 재계산하려면 `accrue` 를 쓰세요).
    """
    rate = borrow_rate(inst, cfg)
    if rate <= 0 or days <= 0 or notional <= 0:
        return 0.0
    return abs(float(notional)) * rate * (float(days) / BORROW_DAYS_PER_YEAR)


def accrue(inst, notional: float, prev_day, today, cfg: dict = None) -> float:
    """봉 하나만큼의 차입비용. 백테스트 루프가 매 봉 호출합니다.

    달력일 차이를 씁니다 — 금요일에서 월요일 사이는 3일치 이자가 붙습니다.
    영업일로 세면 주말 이자가 통째로 사라져서, 오래 들고 가는 숏일수록
    비용이 과소평가됩니다 (연 2.5%면 주말만 28% 누락).
    """
    gap = _days_between(prev_day, today)
    if gap <= 0:
        return 0.0
    return borrow_cost(inst, notional, gap, cfg)


def _days_between(a, b) -> float:
    for value in (a, b):
        if value is None:
            return 0.0
    try:
        d1, d2 = _as_date(a), _as_date(b)
    except (ValueError, TypeError, AttributeError):
        return 0.0
    return max(0.0, (d2 - d1).days)


def _as_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return value.to_pydatetime().date()       # pandas Timestamp
    except AttributeError:
        pass
    return datetime.fromisoformat(str(value)[:10]).date()
