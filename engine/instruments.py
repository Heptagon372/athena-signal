"""
인스트루먼트 추상화 계층 (Instrument Abstraction Layer)
------------------------------------------------------
자동매매의 상위 로직(신호·리스크·주문)이 "이게 주식인지 선물인지"를 몰라도
동작하도록, 자산군별 차이를 이 모듈 하나에 가둡니다.

자산군별로 실제로 다른 것
    주식/ETF   1주 = 1주(승수 1), 매도 시 증권거래세, 공매도 불가(개인 계좌 기준)
    선물       계약 승수(코스피200 = 25만원/pt), 증거금 거래, 만기·롤오버, 매도 진입 가능
    옵션       계약 승수 + 프리미엄 호가, 매수는 프리미엄 지불 / 매도는 증거금
    ETF        주식과 같은 방식으로 거래되지만 **매도 거래세가 면제**됩니다

이 차이를 위쪽에서 매번 if 문으로 처리하면 반드시 사고가 납니다.
(선물 1계약을 주식 1주처럼 계산하면 실제 노출이 25만배가 됩니다)

파생상품 코드 규칙 (KRX 단축코드 8자리)
    1번째   1=선물, 2=콜옵션, 3=풋옵션
    2~3번째 기초자산 (01=코스피200, 05=미니코스피200, 06=코스닥150 …)
    4번째   결제월 (F=1월 … Z=12월)
    5번째   결제연도 끝자리
    6~8번째 선물은 000, 옵션은 행사가 코드

    예) 101H6000 = 코스피200 선물 2026년 3월물
        201H6300 = 코스피200 콜옵션 2026년 3월물 행사가 300

계약 명세(승수·호가단위·증거금률)는 거래소가 바꿀 수 있으므로 상수가 아니라
**설정 데이터**로 둡니다(DERIV_SPECS). api_keys.json 의 "DERIV_SPECS" 키로
덮어쓸 수 있습니다.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import date

from models import MARKET_US, ResolvedSymbol, SymbolNotFoundError

# ---------------------------------------------------------------------------
# 자산군
# ---------------------------------------------------------------------------

STOCK = "STOCK"
ETF = "ETF"
FUTURES = "FUTURES"
OPTION = "OPTION"

ASSET_LABELS = {
    STOCK: "주식",
    ETF: "ETF",
    FUTURES: "선물",
    OPTION: "옵션",
}

MARKET_DERIV = "KRX_DERIV"          # 파생상품 시장 (models.MARKET_* 와 구분)

# ---------------------------------------------------------------------------
# 비용 모형 (국내 기준 근사 — storage/paper.py 와 일치시킵니다)
# ---------------------------------------------------------------------------

FEE_STOCK_KR = 0.00015        # 위탁수수료 0.015%
TAX_STOCK_KR = 0.0018         # 증권거래세 0.18% (매도)
FEE_STOCK_US = 0.0025         # 미국 0.25% (환전 포함 근사)
FEE_FUTURES = 0.00003         # 선물 위탁수수료 0.003% (명목금액 기준)
FEE_OPTION = 0.003            # 옵션 위탁수수료 0.3% (프리미엄 기준)

# ETF 는 매도 시 증권거래세가 면제됩니다. 이걸 빼먹으면 백테스트가 실제보다
# 0.18%씩 나쁘게 나와서, 짧은 회전 전략을 전부 "수익 안 남"으로 잘못 판정합니다.
TAX_ETF_KR = 0.0

# ---------------------------------------------------------------------------
# 파생상품 계약 명세
# ---------------------------------------------------------------------------

MONTH_CODES = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# 기초자산 코드 -> 명세
#   multiplier   : 계약 승수 (1pt 당 원화)
#   tick         : 호가 단위(pt)
#   margin_rate  : 위탁증거금률 (명목금액 대비) — 증권사·시장상황에 따라 변동
#   option_ticks : 옵션 프리미엄 구간별 호가단위 [(상한, 단위), …]
DERIV_SPECS = {
    "01": {
        "name": "코스피200",
        "multiplier": 250_000,
        "tick": 0.05,
        "margin_rate": 0.15,
        "option_multiplier": 250_000,
        "option_ticks": [(10.0, 0.01), (float("inf"), 0.05)],
        "option_margin_rate": 0.20,
    },
    "05": {
        "name": "미니 코스피200",
        "multiplier": 50_000,
        "tick": 0.02,
        "margin_rate": 0.15,
        "option_multiplier": 50_000,
        "option_ticks": [(10.0, 0.01), (float("inf"), 0.05)],
        "option_margin_rate": 0.20,
    },
    "06": {
        "name": "코스닥150",
        "multiplier": 10_000,
        "tick": 0.10,
        "margin_rate": 0.18,
        "option_multiplier": 10_000,
        "option_ticks": [(10.0, 0.01), (float("inf"), 0.05)],
        "option_margin_rate": 0.22,
    },
}

# 기초자산 코드를 모를 때 쓰는 보수적 기본값 (증거금을 넉넉히 잡습니다)
DEFAULT_DERIV_SPEC = {
    "name": "파생상품",
    "multiplier": 250_000,
    "tick": 0.05,
    "margin_rate": 0.20,
    "option_multiplier": 250_000,
    "option_ticks": [(10.0, 0.01), (float("inf"), 0.05)],
    "option_margin_rate": 0.25,
}

_DERIV_CODE = re.compile(r"^([123])(\d{2})([FGHJKMNQUVXZ])(\d)(\d{3})$")

# 국내 주식 호가 단위 (2023-01 개편). storage/paper.py 와 동일 표.
KR_TICK_TABLE = [
    (2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
    (200_000, 100), (500_000, 500), (float("inf"), 1_000),
]

# 국내 ETF/ETN 브랜드 — KRX 상장법인 목록에 없는 6자리 코드가 이 이름을 달고 있으면
# ETF 로 봅니다. (상장법인 목록은 '회사'만 담고 있어 ETF 는 애초에 들어있지 않습니다)
_ETF_BRANDS = (
    "KODEX", "TIGER", "KBSTAR", "ARIRANG", "ACE", "SOL", "PLUS", "RISE",
    "HANARO", "KOSEF", "TIMEFOLIO", "SMART", "FOCUS", "HK", "WOORI",
    "마이다스", "히어로즈", "BNK", "UNICORN", "ETF", "ETN",
)


def _spec_for(underlying_code: str) -> dict:
    """기초자산 코드에 해당하는 계약 명세 (사용자 설정으로 덮어쓸 수 있음)."""
    try:
        from data_sources import credentials
        override = credentials.get_json("DERIV_SPECS", {}) or {}
    except Exception:
        override = {}
    base = dict(DERIV_SPECS.get(underlying_code, DEFAULT_DERIV_SPEC))
    if isinstance(override, dict) and underlying_code in override:
        base.update(override[underlying_code])
    return base


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

@dataclass
class Instrument:
    """거래 가능한 개별 상품 하나. 상위 로직이 보는 유일한 종목 표현입니다."""

    key: str                       # 내부 식별자 (005930 / AAPL / 101H6000)
    name: str
    asset_class: str               # STOCK | ETF | FUTURES | OPTION
    market: str                    # KOSPI / KOSDAQ / KONEX / US / KRX_DERIV
    currency: str = "KRW"
    multiplier: float = 1.0        # 계약 승수 (주식은 1)
    margin_rate: float = 1.0       # 명목금액 대비 필요 증거금 (주식 현금거래는 1.0)
    shortable: bool = False        # 매도 진입(숏) 가능 여부
    expiry: date | None = None     # 만기 (파생만)
    underlying: str = ""           # 기초자산 이름
    strike: float | None = None    # 행사가 (옵션만)
    right: str = ""                # CALL | PUT (옵션만)
    is_inverse: bool = False       # 인버스 ETF
    leverage: float = 1.0          # 레버리지 배율
    verified_by: str = ""
    symbol: ResolvedSymbol | None = field(default=None, repr=False)  # 원본(주식/ETF)
    _option_ticks: list = field(default_factory=list, repr=False)
    _tick: float = 0.0             # 파생 호가단위

    # -- 분류 -------------------------------------------------------------
    @property
    def is_derivative(self) -> bool:
        return self.asset_class in (FUTURES, OPTION)

    @property
    def is_korean(self) -> bool:
        return self.market != MARKET_US

    @property
    def asset_label(self) -> str:
        return ASSET_LABELS.get(self.asset_class, self.asset_class)

    @property
    def display(self) -> str:
        return f"{self.name} ({self.key} · {self.asset_label})"

    # -- 가격/수량 --------------------------------------------------------
    def tick_size(self, price: float) -> float:
        """해당 가격대의 호가 단위."""
        if self.asset_class == OPTION:
            for upper, unit in (self._option_ticks or DEFAULT_DERIV_SPEC["option_ticks"]):
                if price < upper:
                    return unit
            return 0.05
        if self.asset_class == FUTURES:
            return self._tick or 0.05
        if self.market == MARKET_US:
            return 0.01
        for upper, unit in KR_TICK_TABLE:
            if price < upper:
                return unit
        return 1_000

    def round_price(self, price: float) -> float:
        """지정가를 호가 단위에 맞춥니다. 안 맞추면 거래소가 주문을 거부합니다."""
        if price is None or price <= 0:
            return price
        unit = self.tick_size(price)
        snapped = round(price / unit) * unit
        if unit >= 1:
            return float(int(round(snapped)))
        # 부동소수점 오차 제거 (0.05 단위인데 3.0000000000000004 가 나오는 것 방지)
        digits = max(0, -int(math.floor(math.log10(unit)))) + 2
        return round(snapped, digits)

    def round_quantity(self, qty: float) -> float:
        """수량은 항상 정수 — KIS 는 해외주식도 소수점 주문을 받지 않습니다.

        소수점을 남겨두면 주문 직전 int() 에서 0주가 되어 "주문 수량이 0"
        오류가 매 회전 반복됩니다. 모의계좌도 실계좌와 같은 규칙을 씁니다
        (모의만 소수점이 되면 모의와 실전 결과가 달라집니다).
        """
        return float(int(qty))

    def notional(self, price: float, quantity: float) -> float:
        """명목금액(호가 통화 기준) = 가격 × 수량 × 승수."""
        return float(price) * float(quantity) * self.multiplier

    def margin_required(self, price: float, quantity: float, side: str = "buy") -> float:
        """이 포지션을 열 때 묶이는 금액.

        주식/ETF        전액 (현금거래)
        선물            명목금액 × 증거금률 (매수·매도 동일)
        옵션 매수       프리미엄 전액 (최대손실이 프리미엄이므로 증거금 불필요)
        옵션 매도       수취 프리미엄 + 기초자산 명목금액 × 증거금률
                        (손실이 무한하므로 보수적으로 잡습니다. 실제 거래소
                         증거금 산식은 이보다 복잡한 SPAN 방식입니다.)
        """
        gross = self.notional(price, quantity)
        if self.asset_class == OPTION:
            if side == "buy":
                return gross
            underlying_notional = (self.strike or price) * self.multiplier * float(quantity)
            return gross + underlying_notional * self.margin_rate
        if self.asset_class == FUTURES:
            return gross * self.margin_rate
        return gross

    def costs(self, side: str, notional: float) -> tuple[float, float]:
        """(수수료, 세금). 자산군마다 다릅니다."""
        notional = abs(float(notional))
        if self.asset_class == FUTURES:
            return round(notional * FEE_FUTURES, 2), 0.0
        if self.asset_class == OPTION:
            return round(notional * FEE_OPTION, 2), 0.0
        if self.market == MARKET_US:
            return round(notional * FEE_STOCK_US, 2), 0.0
        fee = round(notional * FEE_STOCK_KR, 2)
        if side != "sell":
            return fee, 0.0
        tax_rate = TAX_ETF_KR if self.asset_class == ETF else TAX_STOCK_KR
        return fee, round(notional * tax_rate, 2)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "asset_class": self.asset_class,
            "asset_label": self.asset_label,
            "market": self.market,
            "currency": self.currency,
            "multiplier": self.multiplier,
            "margin_rate": self.margin_rate,
            "shortable": self.shortable,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "underlying": self.underlying,
            "strike": self.strike,
            "right": self.right,
            "is_inverse": self.is_inverse,
            "leverage": self.leverage,
            "verified_by": self.verified_by,
        }


# ---------------------------------------------------------------------------
# 파생상품 코드 해석
# ---------------------------------------------------------------------------

def parse_derivative(code: str) -> Instrument | None:
    """KRX 파생상품 단축코드를 Instrument 로. 형식이 아니면 None."""
    raw = (code or "").strip().upper()
    m = _DERIV_CODE.match(raw)
    if not m:
        return None

    kind, underlying_code, month_code, year_digit, tail = m.groups()
    spec = _spec_for(underlying_code)
    month = MONTH_CODES[month_code]

    # 연도 끝자리 한 자리만 주어지므로 '지금으로부터 가장 가까운 미래'로 해석합니다.
    today = date.today()
    year = (today.year // 10) * 10 + int(year_digit)
    if year < today.year - 1:
        year += 10
    expiry = _second_thursday(year, month)

    if kind == "1":
        return Instrument(
            key=raw,
            name=f"{spec['name']} 선물 {year % 100:02d}{month:02d}",
            asset_class=FUTURES,
            market=MARKET_DERIV,
            currency="KRW",
            multiplier=float(spec["multiplier"]),
            margin_rate=float(spec["margin_rate"]),
            shortable=True,
            expiry=expiry,
            underlying=spec["name"],
            verified_by="KRX 파생상품 단축코드",
            _tick=float(spec["tick"]),
        )

    right = "CALL" if kind == "2" else "PUT"
    strike = _strike_from_code(tail)
    return Instrument(
        key=raw,
        name=(f"{spec['name']} {'콜' if right == 'CALL' else '풋'}옵션 "
              f"{year % 100:02d}{month:02d} {strike:g}"),
        asset_class=OPTION,
        market=MARKET_DERIV,
        currency="KRW",
        multiplier=float(spec["option_multiplier"]),
        margin_rate=float(spec["option_margin_rate"]),
        shortable=True,
        expiry=expiry,
        underlying=spec["name"],
        strike=strike,
        right=right,
        verified_by="KRX 파생상품 단축코드",
        _option_ticks=list(spec["option_ticks"]),
    )


def _strike_from_code(tail: str) -> float:
    """행사가 코드 3자리 -> 행사가.

    코스피200 옵션은 행사가가 2.5pt 간격이라 소수점이 필요합니다.
    코드 '300' 은 300.0, '3025' 같은 형태는 쓰지 않으므로 마지막 자리를
    0.5 단위로 해석해야 하는 구간(예: 337 -> 337.5)이 존재합니다.
    거래소 코드표 없이 정확히 복원할 수 없어, 여기서는 정수 pt 로 해석하고
    실제 주문 시에는 브로커가 돌려준 종목명을 신뢰합니다.
    """
    try:
        return float(int(tail))
    except ValueError:
        return 0.0


def _second_thursday(year: int, month: int) -> date:
    """국내 지수 파생 만기 = 결제월 두 번째 목요일."""
    d = date(year, month, 1)
    # weekday(): 월0 … 목3
    offset = (3 - d.weekday()) % 7
    return date(year, month, 1 + offset + 7)


# ---------------------------------------------------------------------------
# 주식 / ETF
# ---------------------------------------------------------------------------

def classify_equity(symbol: ResolvedSymbol) -> str:
    """확정된 종목이 주식인지 ETF 인지 판별."""
    quote_type = (getattr(symbol, "quote_type", "") or "").upper()
    if quote_type in ("ETF", "MUTUALFUND"):
        return ETF
    if quote_type == "EQUITY":
        return STOCK

    name = f"{symbol.name} {symbol.long_name}".upper()
    if symbol.market == MARKET_US:
        # 미국은 Yahoo 의 quoteType 이 없을 때만 이름으로 추정합니다.
        if any(w in name for w in ("ETF", "TRUST", "INDEX FUND", "ISHARES", "SPDR",
                                   "VANGUARD", "PROSHARES", "DIREXION")):
            return ETF
        return STOCK

    if any(brand in name for brand in _ETF_BRANDS):
        return ETF

    # KRX 상장법인 목록(회사만 수록)에 없는 6자리 코드는 ETF/ETN 일 가능성이 큽니다.
    try:
        from data_sources import symbol_registry
        if symbol.key not in symbol_registry.get_krx_master()["by_code"]:
            return ETF
    except Exception:
        pass
    return STOCK


def from_symbol(symbol: ResolvedSymbol) -> Instrument:
    """symbol_registry 가 확정한 종목을 Instrument 로 감쌉니다."""
    asset = classify_equity(symbol)
    return Instrument(
        key=symbol.key,
        name=symbol.name,
        asset_class=asset,
        market=symbol.market,
        currency=symbol.currency,
        multiplier=1.0,
        margin_rate=1.0,
        shortable=False,          # 개인 계좌 현물 공매도는 지원하지 않습니다
        is_inverse=symbol.is_inverse,
        leverage=symbol.leverage,
        verified_by=symbol.verified_by,
        symbol=symbol,
    )


# 해석 결과 캐시 — 자동매매 루프가 같은 종목을 초당 여러 번 해석합니다.
# 미국 티커는 해석할 때마다 Yahoo 검색 API 를 때리므로 캐시가 없으면 차단당합니다.
_RESOLVE_TTL = 3600.0
_resolved: dict[str, tuple[float, Instrument]] = {}


def resolve(query: str) -> Instrument:
    """사용자 입력 -> Instrument. 실존하지 않으면 SymbolNotFoundError.

    파생상품 코드를 먼저 보는 이유: '101H6000' 을 주식 티커로 넘기면
    symbol_registry 가 미국 종목으로 오인해 엉뚱한 조회를 합니다.
    """
    import time

    raw = (query or "").strip()
    if not raw:
        raise SymbolNotFoundError("", reason="종목명이나 종목코드를 입력해 주세요.")

    hit = _resolved.get(raw)
    if hit and time.time() - hit[0] < _RESOLVE_TTL:
        return hit[1]

    deriv = parse_derivative(raw)
    if deriv:
        _resolved[raw] = (time.time(), deriv)
        return deriv

    from data_sources import symbol_registry
    inst = from_symbol(symbol_registry.resolve(raw))
    _resolved[raw] = (time.time(), inst)
    return inst


def try_resolve(query: str) -> Instrument | None:
    try:
        return resolve(query)
    except SymbolNotFoundError:
        return None
    except Exception:
        return None
