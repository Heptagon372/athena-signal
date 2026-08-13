"""
종목 추천 엔진 (Multi-Factor Recommender)
========================================
"뭘 살까"를 사람 대신 정하는 곳입니다. 자동매매의 **매매 대상(유니버스)** 을
AI가 계속 갱신하도록 하는 것이 목적입니다.

전문가들이 실제로 쓰는 순서를 그대로 따릅니다. 한 방에 좋은 종목을 찍는 게
아니라, **깔때기(funnel)** 로 걸러 내려갑니다.

    1단계  유니버스 구성   시장 전체에서 거래가 실제로 되는 종목을 받아옴
    2단계  품질·유동성 심사 못 빠져나올 종목, 이상 급등락, 가격대 이탈 제거
    3단계  팩터 점수화     종목마다 6개 팩터를 계산
    4단계  횡단면 표준화   같은 날 후보들끼리 z-score 로 비교 (절대값 비교는 무의미)
    5단계  국면 가중       지금이 추세장인지 평균회귀장인지에 따라 팩터 비중을 바꿈
    6단계  포트폴리오 구성 상관·중복을 피하고 자금으로 실행 가능한 것만 남김

왜 z-score 인가
    "모멘텀 +12%가 좋은 건가?"는 그 자체로 답할 수 없습니다. 같은 날 다른
    종목들이 +30%씩 올랐으면 +12%는 나쁜 것입니다. 팩터는 **항상 횡단면(cross
    -section)에서 상대 비교**해야 합니다. 절대 임계값으로 자르면 시장 전체가
    오르는 날엔 전부 통과하고, 빠지는 날엔 전부 탈락합니다.

왜 국면에 따라 비중을 바꾸는가
    추세장에서는 모멘텀이 돈을 벌고, 평균회귀장에서는 모멘텀이 돈을 잃습니다.
    같은 팩터 비중을 사철 쓰면 반은 틀립니다. 이 프로젝트에는 이미 국면 판별기
    (Hurst / 분산비율)가 있으므로 그것을 팩터 비중에 연결합니다.

쓰는 팩터 (전부 이 프로젝트 안에 이미 있는 계산입니다)
    momentum       12-1 모멘텀 — 최근 1개월을 제외한 중기 수익률(학계 표준)
    trend_quality  추세 회귀 기울기 × R² — '얼마나 꾸준히' 올랐는가
    risk_adjusted  Yang-Zhang 변동성으로 나눈 수익 — 같은 수익이면 덜 흔들린 쪽
    regime_fit     Hurst·분산비율 — 이 종목이 추세를 내는 성질인가
    technical      아테나 스코어(지표 19종 국면 가중) — 지금 시점의 방향
    flow           외국인·기관 순매수 (KIS 가 있을 때만) — 수급
    liquidity      거래대금 — 클수록 좋음(못 빠져나오는 위험 회피)
"""

import json
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import CACHE_DIR
from engine import econophysics, feed, indicators, instruments, quant
from engine.instruments import ETF, STOCK, Instrument

# 팩터 기본 비중 (국면에 따라 아래에서 조정됩니다)
BASE_WEIGHTS = {
    "momentum": 0.12,
    # 상대강도(IBD식) — 3·6·9·12개월 수익률 가중합, 최근 3개월에 2배 비중.
    # 실전 스크리너 커뮤니티가 IBD RS 등급의 근사식으로 널리 쓰는 공식입니다.
    "rel_strength": 0.10,
    # 트렌드 템플릿(미너비니) — 이평 정배열·52주 위치 8개 체크 통과율.
    # "추세가 이미 증명된 종목만 산다"는 SEPA 방법론의 1차 관문입니다.
    "trend_template": 0.10,
    "trend_quality": 0.10,
    "risk_adjusted": 0.10,
    "regime_fit": 0.05,
    "technical": 0.13,
    # 스마트머니 — CCI가 올라가는데 개인은 팔고 기관이 사는 패턴 (수급 역행 포착)
    "smart_money": 0.10,
    # U/D 거래량비(오닐) — 50일 상승일/하락일 거래량. 기관 매집의 고전적 증거.
    "ud_volume": 0.07,
    # 경제물리학 — Hill 꼬리지수(급락 위험) + RMT 시장 결합도(독자적 알파인가)
    "econophysics": 0.06,
    "liquidity": 0.03,
    # 스크랩 화제성 — 크롬 확장이 수집해 온 커뮤니티·뉴스 언급량
    "buzz": 0.04,
}

# 국면별 비중 배수 — 추세장이면 모멘텀·추세를 키우고, 평균회귀장이면 줄입니다
REGIME_TILT = {
    "추세 국면": {"momentum": 1.4, "rel_strength": 1.4, "trend_template": 1.3,
                "trend_quality": 1.4, "technical": 1.1, "risk_adjusted": 0.9},
    "평균회귀 국면": {"momentum": 0.5, "rel_strength": 0.6, "trend_template": 0.7,
                  "trend_quality": 0.6, "technical": 1.2, "risk_adjusted": 1.3,
                  "ud_volume": 1.2},
    "방향성 불분명": {"momentum": 0.8, "rel_strength": 0.9, "trend_quality": 0.9,
                  "risk_adjusted": 1.2},
}

MIN_BARS = 80              # 팩터 계산에 필요한 최소 일봉

# 편입 자격 하한 — 20일 평균 거래대금 (원 / 달러).
#
# 시장 **전체**를 순환하면서 거래정지·정리매매·사실상 거래가 없는 종목까지
# 후보가 됩니다. 예전에는 거래량순위 상위 30만 봤기 때문에 이런 종목이 애초에
# 들어올 수 없었습니다. 문제는 이들이 점수까지 낮게 나오지는 않는다는 것입니다
# — 가격이 안 움직이니 변동성이 0에 가까워 위험조정수익이 오히려 좋게 나옵니다.
#
# 그래서 점수와 별개로 **편입 자격**을 막습니다(tradable=False). 순위표에는
# 남겨서 "왜 안 사는지"가 보이게 합니다. 값은 '사는 순간 호가가 밀리는' 수준의
# 하한이며, _warn 의 경고선(국내 10억)보다 훨씬 낮습니다 — 경고는 사람이 보고
# 판단할 일이고, 여기서 막는 것은 판단의 여지가 없는 구간입니다.
MIN_TRADING_VALUE = {"KR": 1e8, "US": 1e5}
FACTOR_CACHE_TTL = 900.0   # 15분 — 일봉 기반이라 자주 바뀌지 않습니다

# RMT 는 관측일(T)이 자산 수(N)의 2배는 돼야 유효합니다 (_rmt_coupling 이 강제).
# 120일 수익률로는 N≈60이 한계라, 순환 평가로 후보가 수천이 되면 전부 무효
# 판정이 납니다. 시장 모드는 어차피 유동성 상위가 정의하므로 상위 50으로만
# 추정하고 나머지는 중립(1.0)으로 둡니다.
RMT_MAX_ASSETS = 50

# 수익률(_returns)을 디스크에 남길 종목 수 — **시장별** 거래대금 상위 기준.
# RMT 가 실제로 쓰는 것은 RMT_MAX_ASSETS 종목뿐이라, 여유 배수만 남깁니다.
# (넉넉히 잡는 이유: 회전마다 후보 구성이 조금씩 달라서, 상위 50이 매번 같은
#  종목은 아닙니다. 재시작 직후 첫 회전이 캐시만으로 RMT 를 세울 수 있어야 합니다)
RETURNS_KEEP_PER_MARKET = RMT_MAX_ASSETS * 4

_factor_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 팩터 캐시 디스크 저장 — 순환 평가의 기반
# ---------------------------------------------------------------------------
# 메모리 캐시(15분)는 "같은 회전 안에서 두 번 계산하지 않기" 용이고, 디스크
# 캐시(24시간)는 **순환 평가** 용입니다 — 갱신마다 일부만 새로 계산해도 하루
# 안에 계산해 둔 종목 전부를 순위에 올릴 수 있고, 서버를 재시작해도 순환이
# 처음부터 다시 돌지 않습니다. 일봉 기반 팩터는 하루 안에서 크게 변하지
# 않지만, 수급·화제성처럼 장중에 변하는 값은 그 종목이 다시 계산 창에 들어올
# 때까지 묵은 값입니다 — 순환 평가가 의도적으로 받아들인 비용입니다.

FACTOR_STALE_TTL = 24 * 3600.0
_FACTOR_DISK = CACHE_DIR / "factor_cache.json"
_disk_loaded = False


def _load_disk_cache():
    """디스크 팩터를 메모리로 (프로세스당 1회, 만료분 제외)."""
    global _disk_loaded
    with _cache_lock:
        if _disk_loaded:
            return
        _disk_loaded = True
    try:
        payload = json.loads(_FACTOR_DISK.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    now = time.time()
    loaded: dict = {}
    for key, entry in (payload or {}).items():
        try:
            ts = float(entry.get("ts") or 0)
            if now - ts >= FACTOR_STALE_TTL:
                continue
            factors = dict(entry.get("factors") or {})
            returns = entry.get("returns") or {}
            if returns.get("val"):
                # RMT 결합도 계산용 수익률 — 날짜 인덱스까지 복원해야
                # _rmt_coupling 의 날짜 정렬(inner join)이 동작합니다
                factors["_returns"] = pd.Series(
                    [float(v) for v in returns["val"]],
                    index=pd.to_datetime(returns["idx"]))
            loaded[key] = (ts, factors)
        except Exception:
            continue                     # 항목 하나가 깨져도 나머지는 살립니다
    with _cache_lock:
        for key, item in loaded.items():
            current = _factor_cache.get(key)
            if current is None or current[0] < item[0]:
                _factor_cache[key] = item


def save_factor_cache():
    """메모리 팩터를 디스크로. rank() 끝에서 호출됩니다 (갱신당 1회 쓰기)."""
    now = time.time()
    with _cache_lock:
        # 만료 항목은 메모리에서도 걷어내 캐시가 무한히 크지 않게 합니다
        for key in [k for k, (ts, _) in _factor_cache.items()
                    if now - ts >= FACTOR_STALE_TTL]:
            _factor_cache.pop(key, None)
        items = list(_factor_cache.items())

    # 수익률은 **시장별 거래대금 상위만** 남깁니다.
    #
    # RMT 결합도는 어차피 유동성 상위 RMT_MAX_ASSETS 종목으로만 추정하는데,
    # 저장은 전 종목에 대해 하고 있었습니다. 순환 평가로 캐시가 수천 종목이
    # 되면 파일의 대부분이 한 번도 읽히지 않는 수익률입니다 (실측: 2,761종목
    # 14.2MB 중 11.5MB). 한국까지 전 종목을 돌면 그대로 두 배가 되고, 그 파일을
    # 갱신마다 통째로 쓰고 기동 때마다 통째로 파싱하게 됩니다.
    # 상위 몇 배수만 남기면 RMT 결과는 그대로면서 파일은 3분의 1이 됩니다.
    by_market: dict[str, list] = {}
    for key, (_, factors) in items:
        if factors.get("_returns") is None:
            continue
        # market 은 이번 버전부터 팩터에 담깁니다. 그 전에 계산해 둔 캐시는
        # 종목코드 모양으로 가릅니다 — 6자리 숫자면 국내입니다. 시장을 섞으면
        # 원화 거래대금이 달러를 언제나 이겨서, 미국 상위 종목이 통째로 밀립니다.
        market = str(factors.get("market") or "")
        if not market:
            code = key[2:]
            market = "KR" if len(code) == 6 and code.isdigit() else "US"
        by_market.setdefault(market, []).append(
            (float(factors.get("trading_value") or 0.0), key))
    keep_returns: set = set()
    for rows in by_market.values():
        rows.sort(reverse=True)
        keep_returns.update(key for _, key in rows[:RETURNS_KEEP_PER_MARKET])

    payload = {}
    for key, (ts, factors) in items:
        entry = {"ts": ts,
                 "factors": {k: v for k, v in factors.items()
                             if v is None or isinstance(v, (bool, int, float, str))}}
        returns = factors.get("_returns")
        if returns is not None and len(returns) and key in keep_returns:
            try:
                entry["returns"] = {
                    "idx": [pd.Timestamp(d).strftime("%Y-%m-%d")
                            for d in returns.index],
                    "val": [float(v) for v in returns.values]}
            except Exception:
                pass                     # 수익률만 빠지고 팩터는 저장됩니다
        payload[key] = entry
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _FACTOR_DISK.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_FACTOR_DISK)        # 쓰다 죽어도 원본이 깨지지 않게
    except OSError:
        pass
    _save_misses()


def cached_factor_keys(max_age: float = None) -> set:
    """이 나이 안쪽의 팩터가 (디스크 포함) 있는 종목 키 집합.

    순환 평가에서 "이번 갱신에 새로 계산할 필요가 없는 종목"을 고르는
    기준입니다 — 여기 없는 종목부터 계산 창(window)에 담습니다.
    """
    _load_disk_cache()
    ttl = FACTOR_STALE_TTL if max_age is None else max_age
    now = time.time()
    with _cache_lock:
        return {key[2:] for key, (ts, _) in _factor_cache.items()
                if key.startswith("f:") and now - ts < ttl}


# ---------------------------------------------------------------------------
# 계산 불가 메모 — 순환이 앞으로 나아가게 하는 장치
# ---------------------------------------------------------------------------
# 계산 창은 '팩터 캐시가 없는 종목'부터 채웁니다. 그런데 신규상장이라 일봉이
# MIN_BARS 미만이거나 제공처가 그 종목만 안 주면 캐시가 **영원히** 생기지
# 않아서, 매 갱신마다 같은 종목이 창의 같은 자리를 차지합니다. 순환이 거기서
# 멈추는 것입니다 — 전 종목을 도는 설계에서는 이게 곧 "시장 전체 커버"가
# 영원히 끝나지 않는다는 뜻입니다. 그래서 실패도 기록합니다.
#
# 영구 제외가 아니라 6시간짜리입니다. 상장 이력은 하루에 하루치씩 쌓이고,
# 제공처 장애는 대개 그 안에 복구됩니다.

FACTOR_MISS_TTL = 6 * 3600.0
_MISS_DISK = CACHE_DIR / "factor_misses.json"
_misses: dict[str, float] = {}
_misses_loaded = False


def _load_misses():
    global _misses_loaded
    with _cache_lock:
        if _misses_loaded:
            return
        _misses_loaded = True
    try:
        payload = json.loads(_MISS_DISK.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    now = time.time()
    with _cache_lock:
        for key, ts in (payload or {}).items():
            try:
                if now - float(ts) < FACTOR_MISS_TTL:
                    _misses.setdefault(str(key), float(ts))
            except (TypeError, ValueError):
                continue


def _note_miss(key: str):
    """이 종목은 지금 팩터를 만들 수 없다 — 다음 계산 창에서는 건너뜁니다."""
    with _cache_lock:
        _misses[str(key)] = time.time()


def _save_misses():
    """만료분을 걷어내고 디스크로. save_factor_cache 와 같이 호출됩니다."""
    now = time.time()
    with _cache_lock:
        for key in [k for k, ts in _misses.items() if now - ts >= FACTOR_MISS_TTL]:
            _misses.pop(key, None)
        payload = dict(_misses)
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _MISS_DISK.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def uncomputable_keys() -> set:
    """최근 계산에 실패해 이번 창에서는 건너뛸 종목 (디스크 포함)."""
    _load_misses()
    now = time.time()
    with _cache_lock:
        return {key for key, ts in _misses.items() if now - ts < FACTOR_MISS_TTL}


@dataclass
class Recommendation:
    key: str
    name: str
    market: str
    price: float
    price_krw: float
    score: float = 0.0                     # 최종 합성 점수 (0~1 로 정규화)
    rank: int = 0
    factors: dict = field(default_factory=dict)      # 원값
    z_scores: dict = field(default_factory=dict)     # 표준화 값
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    tradable: bool = True
    max_quantity: float = 0.0
    order_krw: float = 0.0
    regime: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "market": self.market,
            "price": self.price, "price_krw": self.price_krw,
            "score": round(self.score, 4), "rank": self.rank,
            "factors": {k: (round(v, 4) if isinstance(v, (int, float)) else v)
                        for k, v in self.factors.items()},
            "z_scores": {k: round(v, 3) for k, v in self.z_scores.items()},
            "reasons": self.reasons, "warnings": self.warnings,
            "tradable": self.tradable, "max_quantity": self.max_quantity,
            "order_krw": round(self.order_krw), "regime": self.regime,
        }


# ---------------------------------------------------------------------------
# 3단계: 종목별 팩터 계산
# ---------------------------------------------------------------------------

def compute_factors(inst: Instrument, bars: pd.DataFrame = None,
                    quote: dict = None, max_age: float = None,
                    offline: bool = False) -> dict | None:
    """한 종목의 팩터 원값. 데이터가 모자라면 None.

    캐시를 쓰는 이유: 유니버스 갱신마다 수십 종목의 일봉을 다시 받으면
    데이터 제공처 한도에 걸립니다. 일봉 기반이라 15분 캐시는 안전합니다.

    `offline=True` 는 순환 평가용 — `max_age`(기본 24시간까지 허용) 안쪽의
    캐시가 있으면 그걸 주고, 없으면 **네트워크 없이** None 을 돌려줍니다.
    시장 전체를 순위에 올릴 때 캐시 없는 종목까지 일봉을 받으면 한 번에
    수천 호출이 나가기 때문입니다.
    """
    _load_disk_cache()
    cache_key = f"f:{inst.key}"
    now = time.time()
    ttl = FACTOR_CACHE_TTL if max_age is None else max_age
    with _cache_lock:
        hit = _factor_cache.get(cache_key)
    if hit and now - hit[0] < ttl and bars is None:
        return hit[1]
    if offline:
        return None

    if bars is None:
        bars = feed.bars(inst, "day", count=280)
    if bars is None or len(bars) < MIN_BARS:
        _note_miss(inst.key)
        return None

    close = bars["close"]
    returns = close.pct_change().dropna()
    if len(returns) < 40:
        _note_miss(inst.key)
        return None

    factors: dict = {}

    # --- 모멘텀 (12-1): 최근 1개월을 빼는 이유는 단기 반전 효과 때문입니다.
    #     바로 직전 달까지 넣으면 '오르자마자 사서 되돌림을 맞는' 패턴이 섞입니다.
    lookback = min(252, len(close) - 1)
    skip = 21 if len(close) > 60 else 5
    try:
        factors["momentum"] = float(close.iloc[-1 - skip] / close.iloc[-lookback] - 1) * 100
    except (IndexError, ZeroDivisionError):
        factors["momentum"] = 0.0

    # --- 상대강도 (IBD식): 3·6·9·12개월 수익률 가중합, 최근 3개월 2배 비중.
    #     RS "등급"(백분위)은 rank() 의 횡단면 표준화가 맡습니다 — 여기서는
    #     원값만 만듭니다. 상장 기간이 짧으면 있는 구간까지만 봅니다.
    def _roc(days: int) -> float:
        n = min(days, len(close) - 1)
        base_price = float(close.iloc[-1 - n])
        return (float(close.iloc[-1]) / base_price - 1) * 100 if base_price > 0 else 0.0
    factors["rel_strength"] = (0.4 * _roc(63) + 0.2 * _roc(126)
                               + 0.2 * _roc(189) + 0.2 * _roc(252))

    # --- 트렌드 템플릿 (미너비니 SEPA): 8개 체크 통과율 (0~1).
    #     ① 주가>50일선 ② >150일선 ③ >200일선 ④ 50>150 ⑤ 150>200
    #     ⑥ 200일선이 한 달째 상승 ⑦ 52주 고점의 25% 이내 ⑧ 52주 저점 대비 +30%.
    #     상장이 짧아 이평이 없으면 해당 항목은 불통과 — 검증 안 된 추세를
    #     통과시키는 것보다 낫습니다 (원 방법론도 200일 이력을 요구합니다).
    checks = _trend_template_checks(close)
    factors["trend_template"] = sum(checks) / len(checks)
    factors["trend_checks"] = sum(checks)
    hi52 = float(close.iloc[-252:].max())
    lo52 = float(close.iloc[-252:].min())
    price_now = float(close.iloc[-1])
    factors["pct_from_high"] = (price_now / hi52 - 1) * 100 if hi52 > 0 else 0.0
    factors["pct_above_low"] = (price_now / lo52 - 1) * 100 if lo52 > 0 else 0.0

    # --- U/D 거래량비 (오닐/CANSLIM): 최근 50일 상승일 거래량 ÷ 하락일 거래량.
    #     1.0 초과 = 매수 우위, 2.0 이상 = 기관 매집 신호(IBD 기준).
    #     가격은 그대로인데 U/D 가 높으면 조용한 매집일 가능성이 큽니다.
    factors["ud_ratio"] = _up_down_volume_ratio(bars)
    factors["ud_volume"] = min(factors["ud_ratio"], 5.0)     # 이상치가 순위를 지배하지 않게

    # --- ADR(20)%: 하루 평균 진폭 (쿨라마기식 '에너지' 척도).
    #     순위에는 넣지 않고 경고용으로만 씁니다 — 진폭이 1%대인 종목은
    #     추세가 좋아도 단기 매매로 먹을 폭 자체가 없습니다.
    try:
        adr = ((bars["high"].iloc[-20:] / bars["low"].iloc[-20:]) - 1).mean() * 100
        factors["adr20"] = float(adr) if not math.isnan(float(adr)) else 0.0
    except Exception:
        factors["adr20"] = 0.0

    # --- 추세 품질: 기울기 × R² (꾸준히 오른 것에 가점)
    trend = quant.trend_regression(close, window=min(60, len(close) - 1))
    if trend:
        factors["trend_quality"] = (float(trend.get("period_return_pct") or 0)
                                    * float(trend.get("r_squared") or 0))
        factors["trend_r2"] = float(trend.get("r_squared") or 0)
    else:
        factors["trend_quality"] = 0.0
        factors["trend_r2"] = 0.0

    # --- 위험조정 수익: Yang-Zhang 변동성으로 나눕니다 (OHLC 를 다 쓰는 추정량)
    #     quant.yang_zhang_vol 은 **연율 퍼센트**를 돌려줍니다(예: 45.0 = 45%).
    #     연수익률도 퍼센트로 맞춰 같은 단위끼리 나눕니다 — 단위를 섞으면
    #     위험조정수익이 100배 작게 나와 팩터가 사실상 무시됩니다.
    vol = quant.yang_zhang_vol(bars, window=min(20, len(bars) - 1))
    annual_return = float(returns.iloc[-60:].mean() * 252 * 100) if len(returns) >= 60 \
        else float(returns.mean() * 252 * 100)
    factors["volatility"] = float(vol) if vol else 0.0          # 단위: %
    factors["annual_return"] = annual_return                    # 단위: %
    factors["risk_adjusted"] = (annual_return / float(vol)
                                if vol and vol > 0 else 0.0)

    # --- 국면 적합도: 추세를 내는 성질인가 (Hurst > 0.5, 분산비율 > 1)
    hurst = quant.hurst_exponent(close)
    vr = quant.variance_ratio(close, q=5)
    h = float(hurst.get("hurst")) if hurst else 0.5
    v = float(vr.get("vr")) if vr else 1.0
    factors["hurst"] = h
    factors["variance_ratio"] = v
    factors["regime_fit"] = (h - 0.5) * 2 + (v - 1.0)      # 둘 다 추세면 커집니다
    factors["symbol_regime"] = (hurst or {}).get("regime", "")

    # --- 기술 종합점수 (아테나 스코어)
    try:
        analysis = indicators.analyze(bars)
        factors["technical"] = float(analysis.score)
        factors["regime"] = (analysis.regime or {}).get("label", "")
    except Exception:
        factors["technical"] = 0.0
        factors["regime"] = ""

    # --- 경제물리학 ①: Hill 꼬리지수 — 수익률 분포의 **하락** 꼬리가 얼마나 두꺼운가.
    #     α가 낮을수록(≈2) 급락이 '가끔'이 아니라 '구조적으로' 나오는 종목입니다.
    factors["tail_alpha"], factors["tail_alpha_se"] = _hill_alpha(returns)

    # --- CCI(20) 흐름 — 스마트머니 판정의 한 축
    cci_now, cci_slope = _cci_state(bars)
    factors["cci"] = cci_now
    factors["cci_slope"] = cci_slope

    # --- 수급: 개인/기관을 나눠서 봅니다 (합쳐버리면 역행 패턴이 안 보입니다)
    flow = _investor_flow_detail(inst)
    avg_volume = float(bars["volume"].iloc[-20:].mean()) or 1.0
    indiv_ratio = flow["individual"] / avg_volume
    inst_ratio = (flow["institution"] + flow["foreign"]) / avg_volume
    factors["indiv_net"] = flow["individual"]
    factors["inst_net"] = flow["institution"] + flow["foreign"]
    factors["smart_money"] = inst_ratio - indiv_ratio

    # 포착: CCI 상승 + 개인 순매도 + 기관(외인 포함) 순매수 — 전형적 매집 신호
    hit = (cci_slope > 0 and flow["individual"] < 0
           and (flow["institution"] + flow["foreign"]) > 0)
    factors["smart_money_hit"] = hit
    if hit:
        factors["smart_money"] *= 1.5      # 세 조건이 맞물릴 때 가점

    # --- 스크랩 화제성: 크롬 확장이 모아온 최근 24시간 언급량 (로그 스케일)
    factors["scrap_count"] = _scrap_mentions(inst)
    factors["buzz"] = math.log1p(factors["scrap_count"])

    # --- 유동성: 최근 20일 평균 거래대금 (로그 스케일 — 규모 차이가 너무 큽니다)
    try:
        recent_value = float((close.iloc[-20:] * bars["volume"].iloc[-20:]).mean())
    except Exception:
        recent_value = 0.0
    factors["trading_value"] = recent_value
    factors["liquidity"] = math.log10(recent_value + 1)

    factors["price"] = float(close.iloc[-1])
    factors["bars"] = len(bars)
    # 시장 — 저장 단계에서 "거래대금 상위"를 시장별로 가르는 데 씁니다.
    # 원화와 달러 거래대금을 한 줄로 세우면 국내 종목이 언제나 이깁니다.
    factors["market"] = inst.market
    # RMT 결합도 계산용 수익률 (rank() 에서 쓰고 저장 전에 버립니다).
    # **날짜 인덱스를 유지합니다** — 종목마다 거래일이 다를 수 있어, 인덱스를
    # 버리고 길이로만 자르면 서로 다른 날의 수익률이 짝지어집니다.
    factors["_returns"] = returns.iloc[-120:].astype(float)

    with _cache_lock:
        _factor_cache[cache_key] = (now, factors)
        _misses.pop(inst.key, None)      # 예전에 실패했더라도 이제 됩니다
    return factors


def _hill_alpha(returns: pd.Series) -> tuple[float, float]:
    """Hill 추정량 — 수익률 분포 **좌측(하락) 꼬리**의 멱지수 α와 그 표준오차.

    α ≈ 2 는 분산이 겨우 존재하는 수준의 두꺼운 꼬리(급락 상습),
    α ≥ 4 는 정규분포에 가까운 얌전한 꼬리입니다.

    **좌측만 재는 이유**: 이 팩터가 답해야 하는 질문은 "이 종목이 급락을 자주
    내는가"입니다. 예전 구현은 `returns.abs()` 로 양쪽 꼬리를 합쳐 재서, 상한가를
    자주 치는 급등주가 급락 위험 종목으로 계산됐습니다. 수익률 분포의 좌우 꼬리
    두께는 다릅니다(Cont 2001 정형사실). 계산은 engine/econophysics.hill_tail 에
    있고, k 선택은 Hill plot 평탄 구간 자동탐색을 씁니다.

    표준오차 α/√k 를 함께 돌려줍니다 — 호출부가 추정 불확실성을 반영할 수 있도록.
    표본이 부족하면 (3.0, inf) 로 중립 처리합니다. 3.0 은 주식 수익률 꼬리지수의
    실증값입니다(Gopikrishnan et al. 1999, inverse cubic law).
    """
    res = econophysics.hill_tail(returns.to_numpy(dtype=float), side="left")
    if res is None:
        return 3.0, float("inf")
    return float(min(res["alpha"], 8.0)), float(res["se"])


def _trend_template_checks(close: pd.Series) -> list[bool]:
    """미너비니 트렌드 템플릿 8개 체크. 이평이 없으면(NaN) 그 항목은 불통과."""
    price = float(close.iloc[-1])
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()

    def _last(series) -> float:
        try:
            v = float(series.iloc[-1])
            return v if not math.isnan(v) else 0.0
        except (IndexError, TypeError):
            return 0.0

    m50, m150, m200 = _last(ma50), _last(ma150), _last(ma200)
    # 200일선 상승 판정 — 한 달(21봉) 전과 비교합니다
    m200_ago = 0.0
    if len(ma200) > 21:
        try:
            v = float(ma200.iloc[-22])
            m200_ago = v if not math.isnan(v) else 0.0
        except (IndexError, TypeError):
            pass
    hi52 = float(close.iloc[-252:].max())
    lo52 = float(close.iloc[-252:].min())

    return [
        m50 > 0 and price > m50,
        m150 > 0 and price > m150,
        m200 > 0 and price > m200,
        m50 > 0 and m150 > 0 and m50 > m150,
        m150 > 0 and m200 > 0 and m150 > m200,
        m200 > 0 and m200_ago > 0 and m200 > m200_ago,
        hi52 > 0 and price >= hi52 * 0.75,
        lo52 > 0 and price >= lo52 * 1.30,
    ]


def _up_down_volume_ratio(bars: pd.DataFrame, days: int = 50) -> float:
    """오닐 U/D 거래량비 — 상승일 거래량 합 ÷ 하락일 거래량 합 (최근 50일)."""
    try:
        recent = bars.iloc[-(days + 1):]
        change = recent["close"].diff().iloc[1:]
        volume = recent["volume"].iloc[1:]
        up = float(volume[change > 0].sum())
        down = float(volume[change < 0].sum())
        if down <= 0:
            return 5.0 if up > 0 else 1.0      # 하락일이 없으면 상한, 둘 다 없으면 중립
        return up / down
    except Exception:
        return 1.0


def _cci_state(bars: pd.DataFrame, period: int = 20) -> tuple[float, float]:
    """CCI(20)의 현재값과 최근 5봉 기울기. 지표표와 같은 공식(평균편차)입니다."""
    try:
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3
        tp_ma = typical.rolling(period).mean()
        mean_dev = typical.rolling(period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (typical - tp_ma) / (0.015 * mean_dev.replace(0, np.nan))
        now_value = float(cci.iloc[-1])
        past = float(cci.iloc[-6]) if len(cci) > 6 and not math.isnan(float(cci.iloc[-6])) \
            else now_value
        if math.isnan(now_value):
            return 0.0, 0.0
        return now_value, now_value - past
    except Exception:
        return 0.0, 0.0


def _investor_flow_detail(inst: Instrument) -> dict:
    """개인/외국인/기관 순매수를 나눠서 (스마트머니 판정에 필요).

    KIS 는 장중에 갱신되지만 **장 시작 전에는 오늘자 행이 전부 None** 입니다.
    그때는 네이버의 전일 확정치로 폴백합니다 — 0으로 두면 수급 팩터가
    하루의 절반 동안 죽어 있게 됩니다.
    """
    empty = {"individual": 0.0, "foreign": 0.0, "institution": 0.0}
    if inst.market == "US":
        return empty          # 미국은 동일한 일별 수급 공개 데이터가 없습니다

    from data_sources import kis_client
    if kis_client.is_configured():
        try:
            flow = kis_client.get_investor_flow(inst.key) or {}
        except Exception:
            flow = {}
        if any(flow.get(k) is not None for k in ("individual", "foreign", "institution")):
            return {"individual": float(flow.get("individual") or 0),
                    "foreign": float(flow.get("foreign") or 0),
                    "institution": float(flow.get("institution") or 0)}

    try:
        from data_sources import kr_market
        rows = kr_market.naver_investor_trend(inst.key, days=3) or []
    except Exception:
        return empty
    latest = next((r for r in rows
                   if any(r.get(k) is not None
                          for k in ("individual", "foreign", "institution"))), None)
    if not latest:
        return empty
    return {"individual": float(latest.get("individual") or 0),
            "foreign": float(latest.get("foreign") or 0),
            "institution": float(latest.get("institution") or 0)}


def _scrap_mentions(inst: Instrument, hours: int = 24) -> int:
    """크롬 확장이 스크랩해 온 최근 언급 수 (로컬 DB — 네트워크 없음)."""
    try:
        from data_sources import scrap_store
        return len(scrap_store.get_recent(inst.key, hours=hours, limit=300))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 4단계: 횡단면 표준화
# ---------------------------------------------------------------------------

def _zscore(values: list[float]) -> list[float]:
    """같은 날 후보들끼리 비교. 표준편차가 0이면 전부 0점(구분 불가)."""
    array = np.array([v if v is not None and not math.isnan(v) else 0.0 for v in values],
                     dtype=float)
    if len(array) < 2 or float(array.std()) == 0:
        return [0.0] * len(array)
    z = (array - array.mean()) / array.std()
    # 이상치가 순위를 지배하지 않도록 ±3 에서 자릅니다
    return [float(max(-3.0, min(3.0, v))) for v in z]


TAIL_PRIOR_MEAN = 3.0      # Gopikrishnan et al. (1999) inverse cubic law
TAIL_PRIOR_SD = 0.6        # 이 유니버스에서 실측한 꼬리지수의 횡단면 표준편차


def _shrink_tail(alpha: float, se: float) -> float:
    """추정오차가 클수록 꼬리지수를 중립값(3.0)으로 당깁니다 (정규-정규 사후평균)."""
    if se is None or not math.isfinite(se) or se <= 0:
        return TAIL_PRIOR_MEAN
    w_data = 1.0 / (se ** 2)
    w_prior = 1.0 / (TAIL_PRIOR_SD ** 2)
    return (alpha * w_data + TAIL_PRIOR_MEAN * w_prior) / (w_data + w_prior)


def _rmt_coupling(returns_series: list) -> list[float]:
    """상관행렬 시장 모드 기준 결합도 (평균 = 1.0). 잡음이면 전부 중립.

    Laloux et al. (1999), Plerou et al. (2002). 계산은
    engine/econophysics.rmt_decompose 에 있습니다.

    이전 구현에서 고친 두 가지
      1) **Marchenko-Pastur 판정이 없었습니다.** 최대 고유값을 무조건 시장 모드로
         썼는데, RMT 의 요지는 λ₊ = σ²(1+√(N/T))² 를 넘는 고유값만 정보를
         담는다는 것입니다. 이제 못 넘으면 전부 중립을 돌려줍니다.
      2) **날짜 정렬이 없었습니다.** 수익률을 리스트로 받아 뒤에서 min_len 개를
         잘랐기 때문에, 거래정지·신규상장으로 봉 수가 다른 종목끼리 서로 다른
         날짜의 수익률이 짝지어졌습니다. 이제 pd.Series 를 받아 날짜로
         inner join 합니다.

    최소 자산 수도 3 → 15 로 올렸습니다. 3종목의 첫 고유벡터는 '시장'이 아니라
    그 세 종목의 공통 성분입니다 (원논문은 406·1000 종목을 씁니다).
    """
    n_total = len(returns_series)
    usable = [(i, s) for i, s in enumerate(returns_series)
              if s is not None and len(s) >= 40]
    if len(usable) < econophysics.RMT_MIN_ASSETS:
        return [1.0] * n_total

    # 날짜 인덱스로 정렬 — 결측이 있는 날은 통째로 버립니다
    aligned = pd.concat([s.rename(i) for i, s in usable], axis=1, join="inner").dropna()
    if len(aligned) < len(usable) * 2:
        return [1.0] * n_total

    res = econophysics.rmt_decompose(aligned.to_numpy().T,
                                     min_assets=econophysics.RMT_MIN_ASSETS)
    if res is None or not res["market_mode_valid"]:
        # 최대 고유값이 잡음 벌크 안 — 시장 모드가 없습니다
        return [1.0] * n_total

    couplings = {int(col): float(v) for col, v in zip(aligned.columns, res["couplings"])}
    return [couplings.get(i, 1.0) for i in range(n_total)]


def _weights_for(regime: str) -> dict:
    """국면에 맞게 팩터 비중 조정 후 합이 1이 되도록 정규화."""
    tilt = REGIME_TILT.get(regime, {})
    adjusted = {k: BASE_WEIGHTS[k] * tilt.get(k, 1.0) for k in BASE_WEIGHTS}
    total = sum(adjusted.values()) or 1.0
    return {k: v / total for k, v in adjusted.items()}


# ---------------------------------------------------------------------------
# 전체 파이프라인
# ---------------------------------------------------------------------------

def rank(candidates: list, cfg: dict, account: dict, top_n: int = 5,
         market_regime: str = "", stale_keys: set = None) -> list[Recommendation]:
    """후보 목록 → 순위가 매겨진 추천.

    candidates 는 screener.Candidate 또는 종목코드 문자열 목록입니다.

    `stale_keys` 는 순환 평가에서 온 종목들 — 팩터를 새로 계산하지 않고
    24시간 안쪽 캐시만 씁니다 (캐시가 없으면 조용히 순위에서 빠집니다).
    이게 없으면 시장 전체를 넘기는 순간 종목 수만큼 일봉 호출이 나갑니다.
    """
    resolved = []
    for candidate in candidates:
        inst, base = _to_instrument(candidate)
        if inst is None:
            continue
        if stale_keys and inst.key in stale_keys:
            factors = compute_factors(inst, max_age=FACTOR_STALE_TTL, offline=True)
        else:
            factors = compute_factors(inst)
        if not factors:
            continue
        # 캐시된 dict 를 그대로 쓰면 아래에서 _returns 를 지울 때 캐시가 망가집니다
        resolved.append((inst, base, dict(factors)))

    if not resolved:
        return []

    # 시장 국면 — 후보들의 국면 판정을 다수결로 씁니다
    if not market_regime:
        labels = [f.get("regime") for _, _, f in resolved if f.get("regime")]
        market_regime = max(set(labels), key=labels.count) if labels else "방향성 불분명"
    weights = _weights_for(market_regime)

    # 시장(한국/미국) 버킷 — 표준화·상관 계산은 **같은 시장 안에서만** 합니다.
    #   · 수급(smart_money)은 한국에만 있어서 섞으면 미국 종목이 일괄 감점됩니다
    #   · 거래대금은 통화 단위가 달라 섞으면 환율이 점수가 됩니다
    #   · 한·미 일봉은 날짜가 어긋나 시장 간 상관행렬 자체가 무의미합니다
    buckets: dict[str, list[int]] = {}
    for i, (inst, _, _) in enumerate(resolved):
        buckets.setdefault("US" if inst.market == "US" else "KR", []).append(i)

    def _bucketed(values: list[float]) -> list[float]:
        out = [0.0] * len(values)
        for indices in buckets.values():
            z = _zscore([values[i] for i in indices])
            for j, i in enumerate(indices):
                out[i] = z[j]
        return out

    # 경제물리학 ②: RMT 시장 결합도 — 후보들 수익률 상관행렬의 첫 고유벡터에
    # 얼마나 실려 있는가. 결합도가 높으면 "시장이 오르니까 오르는" 종목이고,
    # 낮으면 독자적인 이유로 움직이는 종목입니다. 분산 효과도 후자가 큽니다.
    couplings = [1.0] * len(resolved)
    rmt_evaluated: set = set()
    market_diag = {}
    for market_key, indices in buckets.items():
        # 순환 평가로 후보가 수천이면 T≥2N 조건이 깨져 RMT 가 통째로 무효가
        # 됩니다 — 유동성 상위 RMT_MAX_ASSETS 종목으로만 시장 모드를 추정하고
        # 나머지는 중립(1.0)으로 둡니다 (시장 모드는 어차피 대형주가 정의합니다).
        rmt_idx = indices
        if len(indices) > RMT_MAX_ASSETS:
            rmt_idx = sorted(indices,
                             key=lambda i: resolved[i][2].get("trading_value") or 0.0,
                             reverse=True)[:RMT_MAX_ASSETS]
        rmt_evaluated.update(rmt_idx)
        sub = _rmt_coupling([resolved[i][2].get("_returns") for i in rmt_idx])
        for j, i in enumerate(rmt_idx):
            couplings[i] = sub[j]
        # 시장 진단 (표시 전용) — 같은 수익률을 재사용하므로 추가 비용이 없습니다.
        # **매매 판단에는 쓰지 않습니다.** 게이트로 쓰면 성과가 나빠진다는 것을
        # 측정했습니다 (AUTOTRADE.md 16장).
        try:
            rets = {resolved[i][0].symbol: resolved[i][2].get("_returns")
                    for i in rmt_idx}
            diag = econophysics.market_diagnostics(rets)
            if diag:
                market_diag[str(market_key)] = diag
        except Exception:
            pass
    for i, (_, _, f) in enumerate(resolved):
        f["rmt_coupling"] = couplings[i]
        f.pop("_returns", None)          # 저장·직렬화 전에 원시 수익률은 버립니다

    # 횡단면 표준화 (시장별)
    columns = {name: _bucketed([f.get(name, 0.0) for _, _, f in resolved])
               for name in BASE_WEIGHTS}
    # econophysics 합성: z(꼬리 얌전함) + z(시장 탈동조) 의 평균
    #
    # 꼬리지수는 **추정오차를 반영해 수축시킨 뒤** 표준화합니다. Hill 추정량의
    # 표준오차는 α/√k 라 k 가 작으면 α 가 ±1 씩 흔들립니다. 점추정을 그대로
    # 횡단면 z 로 만들면 꼬리 위험이 아니라 추정 잡음의 순위를 매기게 됩니다.
    # engine/validation.py 가 성과에 대해 세운 원칙("점추정을 임계값과 직접
    # 비교하지 않는다")을 추정량에도 적용한 것입니다.
    #
    #   사후평균 = (α/se² + μ₀/τ²) / (1/se² + 1/τ²)
    #   μ₀ = 3.0 (주식 꼬리지수 실증값), τ = 0.6 (횡단면 실측 표준편차)
    z_tail = _bucketed([_shrink_tail(f.get("tail_alpha", 3.0),
                                     f.get("tail_alpha_se", float("inf")))
                        for _, _, f in resolved])
    # 탈동조 z 는 **실제로 RMT 를 계산한 종목끼리만** 매깁니다. 중립(1.0)으로
    # 채운 수천 종목과 섞어 표준화하면 분산이 0에 붙어서, 계산된 소수의 z 만
    # ±3 으로 폭주합니다. 후보가 RMT_MAX_ASSETS 이하면 예전과 동일합니다.
    z_decouple = [0.0] * len(resolved)
    for indices in buckets.values():
        idx = [i for i in indices if i in rmt_evaluated]
        z = _zscore([-resolved[i][2].get("rmt_coupling", 1.0) for i in idx])
        for j, i in enumerate(idx):
            z_decouple[i] = z[j]
    columns["econophysics"] = [(a + b) / 2 for a, b in zip(z_tail, z_decouple)]

    total_value = float(account.get("total_value") or 0)
    available = float(account.get("available_cash") or 0)
    position_pct = float(cfg.get("position_pct", 20.0))
    max_order = float(cfg.get("max_order_krw") or 0)
    budget = min(total_value * position_pct / 100.0 or available, available)
    if max_order > 0:
        budget = min(budget, max_order)

    out = []
    for i, (inst, base, factors) in enumerate(resolved):
        z = {name: columns[name][i] for name in BASE_WEIGHTS}
        raw = sum(weights[name] * z[name] for name in BASE_WEIGHTS)
        # -3~+3 z 합을 0~1 로 (시그모이드) — 화면에서 비교하기 쉬운 형태
        score = 1 / (1 + math.exp(-raw * 1.5))

        price = float(base.get("price") or factors.get("price") or 0)
        price_krw = float(base.get("price_krw") or 0)
        if not price_krw:
            # 스크리너를 거치지 않은 후보(직접 입력한 미국 티커 등)는 환산이 필요합니다
            if inst.market == "US":
                from data_sources import fx
                price_krw = fx.to_krw(price, inst.currency or "USD")
            else:
                price_krw = price
        quantity = int(budget // price_krw) if price_krw > 0 else 0
        floor = MIN_TRADING_VALUE["US" if inst.market == "US" else "KR"]
        illiquid = float(factors.get("trading_value") or 0.0) < floor

        rec = Recommendation(
            key=inst.key, name=inst.name or base.get("name") or inst.key,
            market=inst.market, price=price, price_krw=price_krw,
            score=score, factors=factors, z_scores=z,
            regime=factors.get("regime", ""),
            max_quantity=quantity, order_krw=quantity * price_krw,
            tradable=quantity >= 1 and not illiquid,
        )
        rec.reasons = _explain(z, factors, weights)
        if factors.get("smart_money_hit"):
            rec.reasons.insert(0, "◆ 스마트머니 포착 — CCI 상승 중인데 개인은 팔고 "
                                  "기관·외인이 사고 있습니다")
        rec.warnings = _warn(factors, quantity, market=inst.market)
        if illiquid:
            # 편입을 막은 이유는 경고 맨 앞에 둡니다 — 화면이 앞의 두어 줄만
            # 보여주는데, 정작 '왜 못 사는지'가 뒤로 밀리면 안 됩니다.
            rec.warnings.insert(0, (
                f"거래가 사실상 없습니다 (20일 평균 거래대금 "
                f"{factors.get('trading_value', 0):,.0f}"
                f"{'달러' if inst.market == 'US' else '원'}) — 편입 대상에서 제외합니다"))
        out.append(rec)

    out.sort(key=lambda r: (r.tradable, r.score), reverse=True)
    for i, rec in enumerate(out, 1):
        rec.rank = i

    # 시장 진단을 남겨 화면이 읽어갈 수 있게 합니다. 추천 순위에는 영향을 주지
    # 않으며, 이번 회전에 실제로 본 후보들로 계산된 값입니다.
    global _last_market_diagnostics
    with _cache_lock:
        _last_market_diagnostics = market_diag

    # 이번에 새로 계산한 팩터를 디스크에 남깁니다 — 순환 평가가 다음 갱신과
    # 서버 재시작 후에도 이어서 돌 수 있는 근거입니다.
    save_factor_cache()

    return out[:max(top_n, 1)] if top_n else out


_last_market_diagnostics: dict = {}


def last_market_diagnostics() -> dict:
    """가장 최근 rank() 회전의 시장 진단 (RMT 시스템 리스크 · LPPLS 버블).

    **표시 전용입니다.** 이 값을 진입·청산 판단에 연결하지 마세요 —
    게이트로 쓰면 홀드아웃 성과가 나빠진다는 것을 측정했습니다
    (AUTOTRADE.md 16장의 주변효과 표).
    """
    with _cache_lock:
        return dict(_last_market_diagnostics)


def _to_instrument(candidate):
    """screener.Candidate / 문자열 / dict → (Instrument, 기본정보)."""
    if isinstance(candidate, str):
        inst = instruments.try_resolve(candidate)
        return inst, {}
    key = getattr(candidate, "key", None) or (candidate or {}).get("key")
    if not key:
        return None, {}
    base = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate)
    if base.get("market") == "US" and base.get("source"):
        # 스크리너 표에서 온 미국 티커 — Yahoo 재검증 없이 그대로 씁니다.
        # try_resolve 는 티커마다 Yahoo 검색을 호출하는데, 순환 평가는 시장
        # 전체(수천 종목)를 넘기므로 재검증만으로 차단당합니다.
        inst = instruments.from_us_screener(key, str(base.get("name") or ""))
    else:
        inst = instruments.try_resolve(key)
    if inst is None or inst.asset_class not in (STOCK, ETF):
        return None, {}
    return inst, base


def _explain(z: dict, factors: dict, weights: dict) -> list[str]:
    """왜 이 종목을 추천했는지 — 기여도가 큰 순서로.

    점수만 보여주면 사용자는 믿을 근거가 없습니다. **어떤 팩터가 얼마나
    밀어올렸는지**를 그대로 보여줍니다.
    """
    contributions = sorted(
        ((name, weights[name] * z[name]) for name in weights),
        key=lambda x: abs(x[1]), reverse=True)

    labels = {
        "momentum": lambda: f"12-1 모멘텀 {factors.get('momentum', 0):+.1f}%",
        "rel_strength": lambda: f"상대강도(IBD식) {factors.get('rel_strength', 0):+.1f}",
        "trend_template": lambda: (f"트렌드 템플릿 {factors.get('trend_checks', 0):.0f}/8 통과 "
                                   f"(52주 고점 {factors.get('pct_from_high', 0):+.0f}%)"),
        "ud_volume": lambda: f"U/D 거래량비 {factors.get('ud_ratio', 1):.2f} (50일)",
        "trend_quality": lambda: f"추세 꾸준함 R²={factors.get('trend_r2', 0):.2f}",
        "risk_adjusted": lambda: (f"위험조정수익 {factors.get('risk_adjusted', 0):+.2f} "
                                  f"(연 {factors.get('annual_return', 0):+.0f}% / "
                                  f"변동성 {factors.get('volatility', 0):.0f}%)"),
        "regime_fit": lambda: f"Hurst {factors.get('hurst', 0.5):.2f} · VR {factors.get('variance_ratio', 1):.2f}",
        "technical": lambda: f"기술점수 {factors.get('technical', 0):+.2f}",
        "smart_money": lambda: (f"수급 — 개인 {factors.get('indiv_net', 0):+,.0f}주 / "
                                f"기관·외인 {factors.get('inst_net', 0):+,.0f}주 "
                                f"(CCI {factors.get('cci', 0):+.0f}, "
                                f"기울기 {factors.get('cci_slope', 0):+.0f})"),
        "econophysics": lambda: (f"물리학 — 꼬리지수 α={factors.get('tail_alpha', 3):.2f} · "
                                 f"시장결합도 {factors.get('rmt_coupling', 1):.2f}"),
        "liquidity": lambda: f"거래대금 {factors.get('trading_value', 0) / 1e8:,.0f}억",
        "buzz": lambda: f"커뮤니티·뉴스 언급 {factors.get('scrap_count', 0)}건/24h",
    }

    out = []
    for name, contribution in contributions[:4]:
        if abs(contribution) < 0.02:
            continue
        mark = "▲" if contribution > 0 else "▼"
        out.append(f"{mark} {labels[name]()} ({contribution:+.2f})")
    return out


def _warn(factors: dict, quantity: int, market: str = "KR") -> list[str]:
    out = []
    # 거래대금은 호가 통화 기준입니다 — 시장별로 기준을 달리 봐야 합니다
    # (달러 거래대금을 원화 10억과 비교하면 미국 종목 전부에 헛경고가 붙습니다)
    if market == "US":
        if factors.get("trading_value", 0) < 1e6:
            out.append("거래대금이 100만달러 미만입니다 — 빠져나오기 어려울 수 있습니다")
    elif factors.get("trading_value", 0) < 1e9:
        out.append("거래대금이 10억 미만입니다 — 빠져나오기 어려울 수 있습니다")
    if factors.get("volatility", 0) > 60:
        out.append(f"연변동성 {factors['volatility']:.0f}% — 손절이 자주 걸립니다")
    if factors.get("hurst", 0.5) < 0.45:
        out.append("평균회귀 성향이 강해 추세 추종이 잘 안 먹힐 수 있습니다")
    if factors.get("tail_alpha", 3.0) < 2.3:
        out.append(f"꼬리지수 α={factors['tail_alpha']:.1f} — 급락이 구조적으로 잦은 분포입니다")
    if factors.get("trend_checks", 8) <= 3:
        out.append(f"트렌드 템플릿 {factors['trend_checks']:.0f}/8 — 검증된 추세가 아닙니다")
    if factors.get("pct_from_high", 0) < -40:
        out.append(f"52주 고점 대비 {factors['pct_from_high']:.0f}% — 하락 추세 한복판일 수 있습니다")
    if 0 < factors.get("ud_ratio", 1.0) < 0.8:
        out.append(f"U/D 거래량비 {factors['ud_ratio']:.2f} — 하락일에 거래가 몰립니다 (분산 우위)")
    if 0 < factors.get("adr20", 2.0) < 1.2:
        out.append(f"ADR {factors['adr20']:.1f}% — 하루 진폭이 작아 단기에 먹을 폭이 좁습니다")
    if quantity < 1:
        out.append("현재 자금으로는 1주도 살 수 없습니다")
    return out


def clear_cache():
    with _cache_lock:
        _factor_cache.clear()


def describe_factors() -> list[dict]:
    """화면에 팩터 설명을 그대로 띄우기 위한 목록."""
    return [
        {"key": "momentum", "label": "12-1 모멘텀",
         "desc": "최근 1개월을 뺀 12개월 수익률. 단기 반전 효과를 피하려고 최근 달을 제외합니다(학계 표준)."},
        {"key": "rel_strength", "label": "상대강도(IBD식)",
         "desc": "3·6·9·12개월 수익률 가중합(최근 3개월 2배). 같은 시장 후보끼리 상대 순위를 매깁니다."},
        {"key": "trend_template", "label": "트렌드 템플릿",
         "desc": "미너비니 8개 체크 — 이평 정배열(50>150>200), 200일선 상승, 52주 고점 25% 이내, 저점 대비 +30%."},
        {"key": "ud_volume", "label": "U/D 거래량비",
         "desc": "50일 상승일 거래량 ÷ 하락일 거래량(오닐). 2.0 이상이면 기관 매집 신호로 봅니다."},
        {"key": "trend_quality", "label": "추세 품질",
         "desc": "회귀 기울기 × R². 같은 상승이라도 들쭉날쭉한 것보다 꾸준한 쪽에 가점."},
        {"key": "risk_adjusted", "label": "위험조정 수익",
         "desc": "Yang-Zhang 변동성으로 나눈 수익. 같은 수익이면 덜 흔들린 종목."},
        {"key": "regime_fit", "label": "국면 적합도",
         "desc": "Hurst 지수와 분산비율. 이 종목이 추세를 내는 성질인지 판단."},
        {"key": "technical", "label": "기술 점수",
         "desc": "지표 19종을 국면 가중 평균한 아테나 스코어."},
        {"key": "smart_money", "label": "스마트머니",
         "desc": "CCI가 올라가는데 개인은 팔고 기관·외인이 사는 매집 패턴 포착 (KIS 필요)."},
        {"key": "econophysics", "label": "경제물리학",
         "desc": "Hill 꼬리지수(급락 상습성) + RMT 시장결합도(독자적으로 움직이는가)."},
        {"key": "liquidity", "label": "유동성",
         "desc": "평균 거래대금. 클수록 원할 때 빠져나올 수 있습니다."},
        {"key": "buzz", "label": "화제성",
         "desc": "크롬 확장이 스크랩한 최근 24시간 커뮤니티·뉴스 언급량."},
    ]
