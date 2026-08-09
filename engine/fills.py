"""
체결 현실성 (Fill Realism) — 봉 하나로 "정말 그 가격에 체결됐을까" 를 판정합니다
------------------------------------------------------------------------------

백테스트가 조용히 낙관적이 되는 지점은 대개 알파가 아니라 **체결** 입니다.
전략이 틀린 게 아니라, 실제로는 받을 수 없는 가격을 받았다고 가정한 것입니다.

아래 규칙은 `kernc/backtesting.py` 의 `_Broker._process_orders()` 가 15년 가까이
다듬어 온 판정을 옮긴 것입니다. 라이브러리를 그대로 가져오지 않은 이유는
파일 맨 아래 **'왜 직접 구현했는가'** 를 보세요.

이 모듈이 강제하는 여섯 가지
    1) 의사결정과 체결 사이에 **정확히 한 봉의 지연**
       종가를 보고 판단해서 그 종가에 체결하는 것은 실전에서 불가능합니다.
       판단은 close[i] 로, 체결은 open[i+1] 로 갈라놓습니다.

    2) 손절·익절은 **봉 안(intrabar)** 에서 판정
       종가만 보면, 장중에 손절선을 뚫었다가 회복한 봉이 전부 무손실로
       기록됩니다. 실제로는 그 자리에서 잘렸습니다. low/high 를 봅니다.

    3) 같은 봉에서 손절과 익절이 모두 닿으면 **손절 우선**
       봉 내부 경로는 알 수 없습니다. 모르면 나쁜 쪽으로 가정합니다.
       (이 규칙 때문에 브래킷 전략의 승률은 체계적으로 과소평가됩니다 —
        그게 맞습니다. 반대로 가정하면 없는 승률이 생깁니다.)

    4) 갭으로 손절선을 관통하면 체결가는 **시가**, 손절값이 아님
       손절 90원짜리가 70원에 갭 하락 시작하면 90원에 못 팝니다. 70원입니다.
       이 하나만 빠져도 갭 위험이 백테스트에서 통째로 사라집니다.

    5) 체결가는 **호가 격자 위에**, 반올림은 나에게 불리한 쪽으로
       12,345.7원 같은 체결가는 존재하지 않습니다.

    6) 가격제한폭에 닿은 방향으로는 **체결되지 않음**
       상한가에서는 살 수 없습니다. 파는 사람이 없으니까요.

그리고 체결되지 못한 주문은 `Rejection` 으로 **남깁니다.** 조용히 사라지게
두면 "백테스트는 좋은데 실매매는 안 되는" 원인이 그대로 숨습니다.
"""

import math
from dataclasses import dataclass, field

LONG = "long"
SHORT = "short"

BUY = "buy"
SELL = "sell"

# 체결 사유 — 결과를 읽을 때 "왜 그 가격이었나" 를 되짚을 수 있어야 합니다
OPEN = "open"                     # 시장가: 다음 봉 시가
STOP = "stop"                     # 손절선 도달
TARGET = "target"                 # 목표가 도달
GAP_THROUGH_STOP = "gap_stop"     # 갭으로 손절선 관통 → 시가 체결
GAP_THROUGH_TARGET = "gap_target"  # 갭으로 목표가 관통 → 시가 체결

# 거부 사유
NO_NEXT_BAR = "no_next_bar"       # 마지막 봉의 신호 — 체결할 다음 봉이 없음
INSUFFICIENT_CASH = "insufficient_cash"
ZERO_QUANTITY = "zero_quantity"
LIMIT_UP = "limit_up"
LIMIT_DOWN = "limit_down"
SHORT_BANNED = "short_banned"     # engine/holding.py 가 올립니다 (공매도 금지 구간)

REJECTION_LABELS = {
    NO_NEXT_BAR: "다음 봉 없음(구간 끝)",
    INSUFFICIENT_CASH: "예수금 부족",
    ZERO_QUANTITY: "주문 수량 0",
    LIMIT_UP: "상한가 — 매수 불가",
    LIMIT_DOWN: "하한가 — 매도 불가",
    SHORT_BANNED: "공매도 금지 구간 — 숏 진입 불가",
}


@dataclass
class Fill:
    """체결 한 건. `reason` 이 없으면 체결이 아닙니다."""
    price: float
    reason: str
    bar: int = -1
    date: str = ""

    @property
    def gapped(self) -> bool:
        return self.reason in (GAP_THROUGH_STOP, GAP_THROUGH_TARGET)


@dataclass
class Rejection:
    """체결되지 못한 주문 — **1급 레코드**.

    백테스트에서 가장 위험한 것은 틀린 숫자가 아니라 **없는 숫자** 입니다.
    예수금이 모자라 진입이 통째로 건너뛰어져도, 거부를 기록하지 않으면
    매매 횟수가 조금 줄 뿐이라 아무도 눈치채지 못합니다. 그리고 그 전략은
    "거래를 적게 하는 안전한 전략" 으로 잘못 읽힙니다.
    """
    bar: int
    date: str
    reason: str
    side: str = ""
    detail: str = ""
    quantity: float = 0.0

    @property
    def label(self) -> str:
        return REJECTION_LABELS.get(self.reason, self.reason)

    def to_dict(self) -> dict:
        return {"bar": self.bar, "date": self.date, "reason": self.reason,
                "label": self.label, "side": self.side, "detail": self.detail,
                "quantity": self.quantity}


# ---------------------------------------------------------------------------
# 호가 격자
# ---------------------------------------------------------------------------

def snap_adverse(inst, price: float, side: str) -> float:
    """호가 단위에 맞추되 **나에게 불리한 쪽으로** 올림/버림.

    `Instrument.round_price` 는 가장 가까운 호가로 반올림합니다 — 지정가를
    낼 때는 그게 맞습니다. 하지만 백테스트에서 반올림을 쓰면 절반은 유리한
    쪽으로 떨어져서, 수천 번 반복되는 동안 체계적으로 성과를 부풀립니다.
    매수는 올림, 매도는 버림 — 한 틱씩 손해 보는 쪽으로만 맞춥니다.
    """
    if price is None or price <= 0:
        return price
    unit = inst.tick_size(price)
    if not unit or unit <= 0:
        return float(price)
    steps = float(price) / unit
    # 부동소수점 여유 — 100.0 이 100.00000000001 로 들어와 한 틱 밀리는 것 방지
    steps = math.ceil(steps - 1e-9) if side == BUY else math.floor(steps + 1e-9)
    snapped = steps * unit
    if unit >= 1:
        return float(int(round(snapped)))
    digits = max(0, -int(math.floor(math.log10(unit)))) + 2
    return round(snapped, digits)


# ---------------------------------------------------------------------------
# 가격제한폭
# ---------------------------------------------------------------------------

def limit_state(inst, prev_close: float, price: float) -> str:
    """이 가격이 가격제한폭에 닿아 있는가 — `LIMIT_UP` / `LIMIT_DOWN` / `""`.

    상한가에서는 매수 체결이 사실상 불가능합니다(파는 사람이 없음). 하한가는
    반대입니다. 이걸 빼면 백테스트에서 상한가 종목을 자유롭게 사고, 하한가에서
    깔끔하게 손절합니다 — 실전에서 가장 크게 깨지는 두 상황입니다.

    국내 주식/ETF 에만 적용합니다. 파생은 제한폭 단계가 다르고(1~3단계 확대),
    미국은 일일 제한폭 대신 변동성완화장치(LULD)라 같은 규칙이 아닙니다.
    """
    from engine import markets
    from engine.instruments import ETF, STOCK

    if inst.asset_class not in (STOCK, ETF) or not inst.is_korean:
        return ""
    if not prev_close or prev_close <= 0 or not price or price <= 0:
        return ""

    band = markets.PRICE_LIMIT_PCT_KR / 100.0
    # 제한가는 호가 단위로 내림/올림되므로 정확히 ±30.00% 가 아닙니다.
    # 한 틱 여유를 둬서 "상한가 근처" 를 놓치지 않게 합니다.
    tolerance = inst.tick_size(price) * 0.5
    if price >= prev_close * (1 + band) - tolerance:
        return LIMIT_UP
    if price <= prev_close * (1 - band) + tolerance:
        return LIMIT_DOWN
    return ""


def blocks(state: str, side: str) -> bool:
    """이 제한폭 상태가 그 방향의 체결을 막는가."""
    return (state == LIMIT_UP and side == BUY) or (state == LIMIT_DOWN and side == SELL)


# ---------------------------------------------------------------------------
# 시장가 체결 — 다음 봉 시가
# ---------------------------------------------------------------------------

def market_fill(inst, bar, side: str, slippage: float = 0.0,
                prev_close: float = 0.0, bar_index: int = -1,
                bar_date: str = "") -> tuple[Fill | None, str]:
    """봉 `bar` 의 **시가** 에 시장가 체결. 실패하면 `(None, 거부사유)`.

    "다음 봉" 을 넘기는 것은 호출자 책임입니다 — 이 함수는 받은 봉의 시가만
    씁니다. 한 봉 지연은 루프 구조로 강제하는 편이 안전합니다. 여기서
    `i + 1` 을 계산하기 시작하면 언젠가 누군가 그 +1 을 지웁니다.
    """
    state = limit_state(inst, prev_close, float(bar["open"]))
    if blocks(state, side):
        return None, state

    price = float(bar["open"])
    price *= (1 + slippage) if side == BUY else (1 - slippage)
    return Fill(snap_adverse(inst, price, side), OPEN, bar_index, bar_date), ""


# ---------------------------------------------------------------------------
# 보호 주문 — 봉 안에서의 손절/익절
# ---------------------------------------------------------------------------

def protective_fill(inst, bar, position_side: str, stop: float | None,
                    target: float | None, slippage: float = 0.0,
                    prev_close: float = 0.0, bar_index: int = -1,
                    bar_date: str = "") -> Fill | None:
    """이 봉 안에서 손절 또는 익절이 체결됐는가. 아니면 `None`.

    판정 순서가 전부입니다 (아래 순서를 바꾸면 성과가 낙관적으로 틀립니다).

      1. **갭 관통 먼저.** 시가가 이미 손절선 너머면 손절값에 못 팝니다.
         체결가는 시가입니다. 이걸 3번보다 뒤에 두면, 70원에 갭 하락한 봉을
         90원 손절로 기록하게 됩니다 — 실전에서 절대 못 받는 가격입니다.
      2. **갭으로 목표가 관통** 은 반대로 유리하게 열립니다. 시가에 체결.
      3. 봉 안 도달: `low <= 손절` / `high >= 목표` (숏은 반대).
      4. **둘 다 닿았으면 손절.** 봉 내부에서 어느 쪽이 먼저였는지는 일봉으로
         알 수 없습니다. 모를 때는 나쁜 쪽으로 갑니다.

    슬리피지는 손절 체결에도 붙입니다. 손절은 원래 미끄러지는 주문이고,
    정확히 손절값에 체결된다고 보는 것 자체가 낙관입니다.
    """
    exit_side = SELL if position_side == LONG else BUY
    o = float(bar["open"])
    h = float(bar["high"])
    lo = float(bar["low"])

    if blocks(limit_state(inst, prev_close, o), exit_side):
        return None       # 하한가에 갇혀 못 파는 상황 — 다음 봉으로 넘어갑니다

    stop = float(stop) if stop else None
    target = float(target) if target else None

    def done(raw: float, reason: str) -> Fill:
        price = raw * ((1 - slippage) if exit_side == SELL else (1 + slippage))
        return Fill(snap_adverse(inst, price, exit_side), reason, bar_index, bar_date)

    if position_side == LONG:
        if stop is not None and o <= stop:
            return done(o, GAP_THROUGH_STOP)
        if target is not None and o >= target:
            return done(o, GAP_THROUGH_TARGET)
        if stop is not None and lo <= stop:
            return done(stop, STOP)          # ★ 익절보다 먼저 — 비관적 판정
        if target is not None and h >= target:
            return done(target, TARGET)
        return None

    if stop is not None and o >= stop:
        return done(o, GAP_THROUGH_STOP)
    if target is not None and o <= target:
        return done(o, GAP_THROUGH_TARGET)
    if stop is not None and h >= stop:
        return done(stop, STOP)
    if target is not None and lo <= target:
        return done(target, TARGET)
    return None


# ---------------------------------------------------------------------------
# 거부 장부
# ---------------------------------------------------------------------------

@dataclass
class RejectionLog:
    """거부를 모아 요약합니다. 백테스트 결과에 그대로 실어 보냅니다."""
    items: list = field(default_factory=list)

    def add(self, bar: int, date: str, reason: str, side: str = "",
            detail: str = "", quantity: float = 0.0) -> None:
        self.items.append(Rejection(bar, date, reason, side, detail, quantity))

    def __len__(self) -> int:
        return len(self.items)

    def summary(self) -> dict:
        """사유별 건수 + 최근 표본. 건수가 0 이 아니면 리포트 상단에 띄우세요."""
        counts: dict[str, int] = {}
        for r in self.items:
            counts[r.reason] = counts.get(r.reason, 0) + 1
        return {
            "total": len(self.items),
            "by_reason": {k: {"count": v, "label": REJECTION_LABELS.get(k, k)}
                          for k, v in sorted(counts.items(),
                                             key=lambda kv: -kv[1])},
            "recent": [r.to_dict() for r in self.items[-10:]],
        }


# ---------------------------------------------------------------------------
# 왜 직접 구현했는가 (backtesting.py 를 붙이지 않은 이유)
# ---------------------------------------------------------------------------
#
# 위 규칙의 출처인 kernc/backtesting.py 를 의존성으로 넣는 안을 검토했고,
# 넣지 않기로 했습니다. 근거 넷:
#
#   1) 라이선스 — AGPL-3.0 입니다. ATHENA 는 네트워크로 서비스되므로
#      결합 저작물로 판단되면 소스 공개 의무가 발생할 수 있습니다. 별도
#      프로세스로 격리하는 회피책이 알려져 있지만 해석의 여지가 있고,
#      얻는 것에 비해 감당할 리스크가 아닙니다.
#
#   2) 비용 모형이 우리보다 약합니다. 그쪽 `commission` 콜백은 진입·청산에
#      같은 부호의 size 를 넘겨서 **매도 전용 세금을 표현할 수 없습니다.**
#      우리 `Instrument.costs(side, notional, when=)` 은 side 를 알고,
#      게다가 시행일별 세율까지 조회합니다. 붙이면 오히려 후퇴합니다.
#
#   3) 단일 자산 전용 + 정수 수량 강제(`int(size)`). 우리는 승수·증거금이
#      다른 파생까지 다루므로 어차피 어댑터를 두껍게 짜야 합니다.
#
#   4) 우리에게 필요한 것은 라이브러리가 아니라 **판정 규칙** 이었습니다.
#      그건 이 파일 300줄이면 끝나고, 여기 있는 편이 읽고 고치기 쉽습니다.
#
# 검증 대조가 필요해지면 그때 별도 프로세스로 붙이면 됩니다. 위 규칙과
# 동치이므로 결과는 맞아떨어져야 하고, 안 맞으면 그게 신호입니다.
