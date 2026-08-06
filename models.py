from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# 종목 식별
# ---------------------------------------------------------------------------

MARKET_KOSPI = "KOSPI"
MARKET_KOSDAQ = "KOSDAQ"
MARKET_KONEX = "KONEX"
MARKET_US = "US"

MARKET_LABELS = {
    MARKET_KOSPI: "코스피",
    MARKET_KOSDAQ: "코스닥",
    MARKET_KONEX: "코넥스",
    MARKET_US: "미국",
}


@dataclass
class ResolvedSymbol:
    """사용자가 입력한 문자열("삼성전자", "005930", "AAPL")을 확정된 종목으로 변환한 결과.

    key         : 시스템 내부 식별자 (한국주식=6자리 코드, 미국주식=티커)
    name        : 표시용 종목명
    market      : MARKET_KOSPI / MARKET_KOSDAQ / MARKET_KONEX / MARKET_US
    yahoo_symbol: Yahoo Finance 조회용 심볼 (005930.KS 등)
    verified_by : 어느 소스로 실존을 확인했는지 (감사 추적용)
    """
    key: str
    name: str
    market: str
    yahoo_symbol: str
    currency: str = "KRW"
    verified_by: str = ""
    long_name: str = ""
    # 레버리지/인버스 ETF 여부 — 뉴스 해석 방향이 뒤집히므로 반드시 구분해야 합니다
    is_inverse: bool = False       # 인버스(베어) 상품: 기초자산이 오르면 하락
    leverage: float = 1.0          # 2x, 3x 등 배율
    # 상품 유형 (Yahoo quoteType: EQUITY / ETF …). 자동매매의 자산군 판별에 씁니다 —
    # ETF 는 매도 거래세가 면제라 비용 모형이 주식과 다릅니다.
    quote_type: str = ""

    @property
    def is_korean(self) -> bool:
        return self.market in (MARKET_KOSPI, MARKET_KOSDAQ, MARKET_KONEX)

    @property
    def is_leveraged_etf(self) -> bool:
        return self.is_inverse or self.leverage > 1.0

    @property
    def market_label(self) -> str:
        return MARKET_LABELS.get(self.market, self.market)


class SymbolNotFoundError(Exception):
    """입력한 종목이 실제로 상장되어 있지 않을 때 발생.

    suggestions 에는 오타로 추정되는 유사 종목 후보를 담아 사용자에게 되돌려줍니다.
    """

    def __init__(self, query: str, suggestions: list | None = None, reason: str = ""):
        self.query = query
        self.suggestions = suggestions or []
        self.reason = reason or f"'{query}' 은(는) 코스피·코스닥·미국 시장에서 찾을 수 없는 종목입니다."
        super().__init__(self.reason)


# ---------------------------------------------------------------------------
# 시세
# ---------------------------------------------------------------------------

@dataclass
class PricePoint:
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class InvestorFlow:
    """개인/외국인/기관 순매매 동향.

    한국 종목은 네이버 금융의 투자자별 매매동향에서 '순매수 수량'을 받아오고,
    금액 환산이 가능한 경우 *_net_value 에 종가 기준 추정 금액을 함께 채웁니다.
    미국 종목은 동일한 공개 데이터가 없어 available=False 로 남습니다.
    """
    individual_net: Optional[float] = None       # 개인 순매수(+)/순매도(-) 수량(주)
    foreign_net: Optional[float] = None          # 외국인 순매수/순매도 수량(주)
    institution_net: Optional[float] = None      # 기관 순매수/순매도 수량(주)
    individual_net_value: Optional[float] = None   # 종가 기준 추정 금액(원)
    foreign_net_value: Optional[float] = None
    institution_net_value: Optional[float] = None
    foreign_hold_ratio: Optional[float] = None   # 외국인 지분율(%)
    trade_date: Optional[str] = None             # 기준일 (YYYY-MM-DD)
    unit: str = "shares"
    source: str = ""
    available: bool = False


@dataclass
class PriceSnapshot:
    ticker: str
    timestamp: datetime
    current_price: float
    open: float
    high: float
    low: float
    prev_close: float
    volume: int
    trading_value: float       # 거래대금
    change_rate: float         # 등락률(%) = (현재가-전일종가)/전일종가*100
    name: str = ""
    market: str = ""
    currency: str = "KRW"
    market_status: str = ""    # "OPEN" | "CLOSE" 등 (제공되는 경우)
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    market_cap_text: Optional[str] = None
    per: Optional[str] = None
    pbr: Optional[str] = None
    source: str = ""           # 어느 공급자에서 받은 값인지
    investor_flow: InvestorFlow = field(default_factory=InvestorFlow)


# ---------------------------------------------------------------------------
# 기술적 지표
# ---------------------------------------------------------------------------

@dataclass
class IndicatorResult:
    """개별 기술적 지표 하나의 계산 결과 + 그 판단의 근거.

    score  : -1.0(강한 하락 신호) ~ +1.0(강한 상승 신호)
    weight : config.INDICATOR_WEIGHTS 의 상대 비중
    reason : 실제 계산된 수치를 인용한 사람이 읽는 근거 문장
    formula: 어떤 식으로 점수를 냈는지 (감사/디버깅용)
    """
    key: str
    label: str
    value_text: str
    score: float
    weight: float
    verdict: str          # "bullish" | "bearish" | "neutral"
    reason: str
    formula: str = ""
    values: dict = field(default_factory=dict)
    family: str = ""      # "trend" | "meanrev" | "confirm" | "info"

    @property
    def contribution(self) -> float:
        return round(self.score * self.weight, 4)


@dataclass
class TechnicalAnalysis:
    score: float                              # -1 ~ 1 종합 기술점수
    indicators: list = field(default_factory=list)   # list[IndicatorResult]
    bars_used: int = 0
    sufficient_data: bool = False
    note: str = ""
    regime: dict = field(default_factory=dict)   # detect_regime() 결과


# ---------------------------------------------------------------------------
# 뉴스 / 커뮤니티
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    ticker: str
    title: str
    source: str
    published_at: datetime
    sentiment_score: float  # -1.0 (매우 부정) ~ +1.0 (매우 긍정)
    url: str = ""
    language: str = "ko"
    matched_keywords: list = field(default_factory=list)  # 감성 판정 근거가 된 단어


@dataclass
class CommunityPost:
    """실시간 커뮤니티 피드에 표시할 개별 게시글"""
    title: str
    stance: str          # "bullish" | "bearish" | "neutral"
    posted_at: datetime
    source: str = "네이버 종목토론방"
    url: str = ""
    matched_keywords: list = field(default_factory=list)


@dataclass
class CommunitySentiment:
    ticker: str
    collected_at: datetime
    bullish_ratio: float  # 0.0 ~ 1.0, 표본 확신도가 반영된 매수(롱) 의견 비율
    post_count: int
    raw_bullish_ratio: float = 0.5  # 보정 전 원본 비율
    confidence: float = 0.0         # 표본 확신도 (분류된 글 수 / 10, 최대 1.0)
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    sample_titles: list = field(default_factory=list)
    bullish_titles: list = field(default_factory=list)
    bearish_titles: list = field(default_factory=list)
    recent_posts: list = field(default_factory=list)  # list[CommunityPost]
    sources: list = field(default_factory=list)       # 실제로 수집에 성공한 소스명
    source_counts: dict = field(default_factory=dict)  # 소스별 수집 건수
    is_demo: bool = False   # 실데이터 수집 실패로 예시 데이터를 쓴 경우에만 True


@dataclass
class ReasoningItem:
    """상승/하락 근거 한 줄 - 무엇이 영향을 줬는지 추적"""
    text: str
    source_type: str     # "news" | "community" | "technical"
    influence: float     # -1.0 ~ 1.0 (하락 방향/상승 방향 기여도)


@dataclass
class Prediction:
    ticker: str
    horizon_label: str          # "open", "5min", "30min", "close"
    predicted_at: datetime
    direction: str               # "up" or "down"
    probability: float           # 0.0 ~ 100.0 (해당 방향일 확률 %)
    technical_score: float
    news_score: float
    community_score: float
    weights_used: dict
    market: str = ""
    name: str = ""
    actual_result: Optional[str] = None   # backtest 이후 채워짐: "up" / "down"
    correct: Optional[bool] = None        # backtest 이후 채워짐
