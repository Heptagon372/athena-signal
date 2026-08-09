"""
종목 스크리너 (Universe Discovery)
---------------------------------
"뭘 살지"를 사람이 고르는 대신 시장에서 조건에 맞는 종목을 찾아옵니다.
페니주식(저가주) 초단타의 매매 대상을 만드는 것이 주 용도입니다.

**어디에서** 찾을지는 두 축으로 정합니다.
    세부시장(segment)  코스피 / 코스닥 / 나스닥
    탐색 범위(universe) 전체 · 지수(코스피200·코스닥150·나스닥100·S&P500)
                        · 섹터(반도체·바이오·기술·금융 …)
    범위 목록과 구성종목 수집은 `data_sources/universe.py` 가 담당합니다.

수집 경로
    한국  1) KIS 거래량순위 — 가격 구간과 **지수 구분**을 API 가 직접 걸러줍니다
             (FID_INPUT_ISCD: 0001 코스피 / 1001 코스닥 / 2001 코스피200)
          2) 지수 파라미터가 없는 범위(코스닥150·섹터)는 그 지수를 추종하는
             ETF 의 구성종목시세로 후보를 만듭니다
          3) 둘 다 실패하면 네이버 금융 거래상위 목록
    미국  1) 나스닥 스크리너 — 미국 상장 전 종목의 시세·거래량·시총·섹터가
             한 번에 옵니다 (거래소별 조회 가능)
          2) 실패 시 Yahoo Finance 스크리너(most_actives / day_gainers)

왜 '거래량 순위'에서 출발하는가
    저가주는 대부분 유동성이 없습니다. 유동성이 없는 종목을 초단타로 사면
    사는 순간 호가가 밀려서 진입가부터 손실입니다. 그래서 **거래가 실제로
    많이 되는 것 중에서** 가격이 낮은 것을 고르는 순서로 접근합니다.
    (가격이 낮은 것 중에서 거래량을 보는 것이 아닙니다 — 순서가 중요합니다)

이 모듈은 **후보만 만듭니다.** 실제로 살지는 engine/scalping.py 의 안전 한도와
engine/risk.py 의 사전 검증을 통과해야 정해집니다.
"""

import re
import threading
import time
from dataclasses import dataclass, field

from data_sources import http_client, kis_client, universe as universe_mod

# 결과 캐시 — 스크리너는 화면에서 반복 호출되고 KIS 는 초당 한도가 있습니다
_cache: dict[str, tuple[float, list]] = {}
_cache_lock = threading.Lock()
CACHE_TTL = 60.0

KIS_VOLUME_RANK_PATH = "/uapi/domestic-stock/v1/quotations/volume-rank"
KIS_VOLUME_RANK_TR = "FHPST01710000"

# 네이버 순위 API — kind 는 up(상승) / down(하락) / searchTop(관심) / marketValue(시총)
# 만 유효합니다. '거래대금 순위'는 없어서 세 목록을 합쳐 근사합니다.
NAVER_RANK_URL = "https://m.stock.naver.com/api/stocks/{kind}/{market}"
NAVER_HEADERS = {"Referer": "https://m.stock.naver.com/", "Accept": "application/json"}

YAHOO_SCREENER = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"

# 미국 저가주 후보 씨앗 — 스크리너 API 가 막혔을 때 최소한의 대상을 확보하기 위한
# 목록입니다. 거래대금이 큰 저가 종목 위주이며, 실제 편입 여부는 아래 필터가 정합니다.
US_SEED = [
    "SOFI", "PLTR", "F", "NIO", "LCID", "RIVN", "SIRI", "PLUG", "AAL", "CCL",
    "NCLH", "SNAP", "GRAB", "CHPT", "RIG", "AMC", "BBAI", "IONQ", "WULF", "MARA",
    "RIOT", "CLSK", "HOOD", "OPEN", "JBLU", "VALE", "ITUB", "BTG", "KGC", "AG",
]


@dataclass
class Candidate:
    """스크리너가 찾아낸 종목 하나.

    타점 지표 세 개(`vol_increase_pct` · `turnover_pct` · `day_range_pct`)가
    초단타 대상 선정의 기준입니다. **None 은 0 이 아니라 '모름'** 입니다 —
    소스마다 주는 필드가 달라서(미국 경로엔 회전율이 없습니다) 0 으로 채우면
    멀쩡한 종목이 '조건 미달' 로 전부 탈락합니다.
    """
    key: str
    name: str
    market: str                    # KOSPI | KOSDAQ | US  (세금·호가단위 판정용)
    price: float                   # 호가 통화 기준
    price_krw: float
    # 세부시장 — 코스피/코스닥은 market 과 같고, 미국은 NASDAQ/NYSE 로 갈립니다.
    # market 을 나스닥으로 바꾸지 않는 이유: 통화·세금·호가단위를 보는 기존
    # 코드가 전부 `market == "US"` 로 갈라져 있습니다.
    segment: str = ""
    change_rate: float = 0.0       # 등락률(%)
    volume: float = 0.0
    trading_value: float = 0.0     # 거래대금 (원 / 달러)
    trading_value_krw: float = 0.0
    prev_close: float | None = None
    high: float | None = None
    low: float | None = None

    # --- 타점 지표 ---
    vol_increase_pct: float | None = None   # 거래량증가율 (평소 대비 %)
    turnover_pct: float | None = None       # 거래회전율 (상장주식수 대비 %)
    day_range_pct: float | None = None      # 당일 고저 변동폭 (%)
    avg_volume: float | None = None         # 평균 거래량 (증가율의 분모)
    listed_shares: float | None = None      # 상장주식수 (회전율의 분모)

    # --- 탐색 범위 정보 (어디에서 찾았는지) ---
    universe: str = ""                      # 이 후보를 담아온 범위 키
    sector: str = ""                        # 섹터 (미국은 스크리너가 직접 줍니다)
    weight: float | None = None             # 그 범위(ETF) 안에서의 비중 %

    source: str = ""
    flags: list = field(default_factory=list)   # 위험 표식 (관리종목 등)

    def derive(self) -> "Candidate":
        """API 가 안 준 지표를 있는 값으로 계산해 채웁니다.

        KIS 거래량순위는 회전율·증가율을 직접 주지만, 네이버/야후 경로는 안
        줍니다. 그쪽에서도 상장주식수나 고저가가 있으면 직접 계산할 수 있어
        타점 기준이 소스에 따라 통째로 빠지는 일을 막습니다.
        """
        if self.turnover_pct is None and self.listed_shares and self.volume:
            self.turnover_pct = self.volume / self.listed_shares * 100
        if self.vol_increase_pct is None and self.avg_volume and self.volume:
            self.vol_increase_pct = (self.volume / self.avg_volume - 1) * 100
        if self.day_range_pct is None and self.high and self.low and self.low > 0:
            self.day_range_pct = (self.high - self.low) / self.low * 100
        return self

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "market": self.market,
            "segment": self.segment or self.market,
            "universe": self.universe, "sector": self.sector, "weight": self.weight,
            "price": self.price, "price_krw": self.price_krw,
            "change_rate": self.change_rate, "volume": self.volume,
            "trading_value": self.trading_value,
            "trading_value_krw": self.trading_value_krw,
            "prev_close": self.prev_close, "high": self.high, "low": self.low,
            "vol_increase_pct": self.vol_increase_pct,
            "turnover_pct": self.turnover_pct,
            "day_range_pct": self.day_range_pct,
            "avg_volume": self.avg_volume, "listed_shares": self.listed_shares,
            "source": self.source, "flags": self.flags,
        }


def _num(v, default=0.0) -> float:
    try:
        return float(str(v).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return default


def _ratio(v) -> float | None:
    """비율 필드 전용 파서 — 값이 없거나 0 이면 None('모름').

    거래량순위에 오른 종목의 회전율이 정확히 0 일 수는 없습니다. 0 이 왔다면
    그 필드를 안 준 것이고, 그걸 0 으로 믿으면 '조건 미달' 로 탈락시킵니다.
    """
    value = _num(v, None)
    return value if value else None


# ---------------------------------------------------------------------------
# 한국
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """스크리너 결과 + **왜 못 찾았는지**.

    후보가 0개일 때 이유를 안 알려주면 사용자는 "종목이 없다"고 오해합니다.
    실제로는 호출 제한이나 장 마감처럼 고칠 수 있는 이유인 경우가 대부분입니다.
    """
    candidates: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    segments: list = field(default_factory=list)   # 실제로 훑은 세부시장
    universe: str = ""                             # 실제로 적용된 탐색 범위

    def __iter__(self):          # 기존 호출부 호환 (list 처럼 순회)
        return iter(self.candidates)

    def __len__(self):
        return len(self.candidates)


# KIS 거래량순위의 순위 기준 (FID_BLNG_CLS_CODE).
# 이 파라미터 하나가 "무엇을 급증으로 볼 것인가" 를 정합니다. 기본값 0(평균거래량)은
# 원래 거래가 많은 대형주가 늘 상위에 오므로, 초단타 타점으로는 1·2번이 맞습니다.
KIS_RANK_BASIS = {
    "volume": "0",          # 평균거래량 — 늘 큰 종목이 1등
    "vol_increase": "1",    # 거래증가율 — 평소 대비 몇 배로 튀었나
    "turnover": "2",        # 평균거래회전율 — 상장주식수 대비 얼마나 손바뀜 했나
    "value": "3",           # 거래금액순
    "value_turnover": "4",  # 평균거래금액회전율
}


def _kis_volume_rank(min_price: int, max_price: int, limit: int = 50,
                     errors: list = None, rank_basis: str = "volume",
                     capture_raw: list = None, iscd: str = "0000",
                     universe_key: str = "") -> list[Candidate]:
    """KIS 거래량순위 — 가격 구간과 시장 구분을 API 가 직접 걸러줍니다.

    종목을 하나씩 조회하면 수백 번 호출해야 하고 초당 한도에 바로 걸립니다.

    `iscd` 가 이 함수의 탐색 범위입니다 (FID_INPUT_ISCD).
        0000 전체 · 0001 코스피 · 1001 코스닥 · 2001 코스피200 · 4001 KRX100
    지수 안에서 찾을 때 **구성종목을 받아와 교집합을 내는 것보다** 이 파라미터가
    낫습니다. 순위 자체가 지수 안에서 매겨져서 지수 전체가 대상이 되기 때문입니다
    (구성종목 API 는 비중 상위 30종목까지만 옵니다).

    이 응답에는 거래량증가율·거래회전율·상장주식수가 **이미 들어 있습니다.**
    회전율을 구하려고 다른 API 를 붙일 필요가 없습니다 — 예전에는 6개 필드만
    꺼내 쓰고 나머지를 버리고 있었습니다.

    `capture_raw` 에 리스트를 주면 첫 행의 원본을 담아줍니다. 필드명이 문서와
    다를 때 무엇이 왔는지 눈으로 확인하기 위한 통로입니다.
    """
    errors = errors if errors is not None else []
    if not kis_client.is_configured():
        errors.append("KIS 키가 없어 거래량순위를 쓸 수 없습니다 (네이버 경로로 시도).")
        return []
    headers = kis_client.auth_headers(KIS_VOLUME_RANK_TR)
    if not headers:
        errors.append("KIS 접근토큰 발급 실패.")
        return []

    from data_sources import kis_trading

    def fetch():
        kis_client.throttle()
        return http_client.get_full(
            kis_client.base_url() + KIS_VOLUME_RANK_PATH,
            headers=headers, params=params, timeout=12)

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20171",
        "FID_INPUT_ISCD": iscd or "0000",  # 전체 / 코스피 / 코스닥 / 코스피200 …
        "FID_DIV_CLS_CODE": "0",           # 전체
        "FID_BLNG_CLS_CODE": KIS_RANK_BASIS.get(rank_basis, "0"),
        "FID_TRGT_CLS_CODE": "111111111",
        "FID_TRGT_EXLS_CLS_CODE": "000000",
        "FID_INPUT_PRICE_1": str(int(min_price)),
        "FID_INPUT_PRICE_2": str(int(max_price)),
        "FID_VOL_CNT": "",
        "FID_INPUT_DATE_1": "",
    }

    status, data, raw = fetch()
    # 초당 호출 제한은 실패가 아니라 '기다리면 되는 상황'입니다.
    # 여기서 바로 포기하면 폴백(네이버)으로 넘어가 데이터 품질이 떨어집니다.
    if kis_trading._is_rate_limited(data, raw):
        status, data, raw = kis_trading._retry_after_limit(fetch)

    if status != 200 or not isinstance(data, dict) or str(data.get("rt_cd")) != "0":
        reason = (data or {}).get("msg1") if isinstance(data, dict) else ""
        errors.append(f"KIS 거래량순위 실패 (HTTP {status}) — {reason or '응답 없음'}")
        return []

    rows = data.get("output") or []
    if not rows:
        errors.append("KIS 거래량순위가 빈 목록을 돌려줬습니다 (장 시작 전이면 정상).")

    if capture_raw is not None and rows:
        capture_raw.append(dict(rows[0]))

    out, skipped = [], 0
    for row in rows[:limit]:
        code = str(row.get("mksc_shrn_iscd") or "").strip()
        price = _num(row.get("stck_prpr"))
        if len(code) != 6 or price <= 0:
            skipped += 1
            continue
        # 이 응답에는 시장 구분도 업종도 없습니다. 상장 마스터(로컬 캐시)에서
        # 채우고, 마스터에 없으면 방금 조회한 지수의 시장을 씁니다.
        entry = universe_mod.kr_entry(code)
        market = (entry.get("market") if entry.get("market") in ("KOSPI", "KOSDAQ")
                  else ("KOSDAQ" if iscd == "1001" else "KOSPI"))
        out.append(Candidate(
            key=code,
            name=row.get("hts_kor_isnm", code),
            market=market, segment=market, universe=universe_key,
            sector=entry.get("sector") or "",
            price=price, price_krw=price,
            change_rate=_num(row.get("prdy_ctrt")),
            volume=_num(row.get("acml_vol")),
            trading_value=_num(row.get("acml_tr_pbmn")),
            trading_value_krw=_num(row.get("acml_tr_pbmn")),
            # --- 타점 지표: 이미 응답에 들어 있던 것들 ---
            # 없으면 None 으로 둡니다. 0 으로 채우면 '조건 미달' 이 되어
            # 필드명이 하나만 달라져도 후보가 전부 사라집니다.
            vol_increase_pct=_ratio(row.get("vol_inrt")),
            turnover_pct=_ratio(row.get("vol_tnrt")),
            # 분모는 prdy_vol(전일거래량)입니다. avrg_vol 은 이 TR 에서 acml_vol
            # 과 같은 값이 와서 '평균'이 아닙니다 — 그걸 쓰면 증가율이 항상 0%가
            # 됩니다. (실제 응답으로 확인: acml_vol 3,966,613 = avrg_vol)
            avg_volume=_ratio(row.get("prdy_vol")),
            listed_shares=_ratio(row.get("lstn_stcn")),
            source="KIS 거래량순위",
        ).derive())
    if skipped:
        errors.append(f"응답 {len(rows)}건 중 {skipped}건은 코드·가격이 비어 있어 제외했습니다.")
    return out


def _naver_active(market: str = "KOSPI", limit: int = 60) -> list[Candidate]:
    """네이버 금융 순위 폴백 (KIS 키가 없거나 KIS 가 실패했을 때).

    네이버에는 '거래대금 순위' API 가 없어서, 상승(up)·관심(searchTop)·
    시총(marketValue) 세 목록을 합쳐 근사합니다 — 지금 움직이는 종목과
    유동성 큰 종목이 함께 잡힙니다. 비공식 엔드포인트라 언제든 막힐 수
    있으므로 실패해도 조용히 빈 목록을 냅니다.
    """
    out, seen = [], set()
    per_kind = max(limit // 2, 20)
    for kind in ("searchTop", "up", "marketValue"):
        data = http_client.get_json(
            NAVER_RANK_URL.format(kind=kind, market=market),
            headers=NAVER_HEADERS, params={"page": 1, "pageSize": per_kind}, timeout=10)
        for row in ((data or {}).get("stocks") or []):
            code = str(row.get("itemCode") or "").strip()
            # *Raw 필드가 콤마 없는 숫자 문자열입니다 ("6,830" 이 아니라 "6830")
            price = _num(row.get("closePriceRaw") or row.get("closePrice"))
            if not re.fullmatch(r"\d{6}", code) or price <= 0 or code in seen:
                continue
            seen.add(code)
            volume = _num(row.get("accumulatedTradingVolumeRaw")
                          or row.get("accumulatedTradingVolume"))
            value = _num(row.get("accumulatedTradingValueRaw")
                         or row.get("accumulatedTradingValue")) or price * volume
            out.append(Candidate(
                key=code, name=row.get("stockName") or code,
                market=market, segment=market, price=price, price_krw=price,
                change_rate=_num(row.get("fluctuationsRatio")),
                volume=volume, trading_value=value, trading_value_krw=value,
                # 네이버 순위 응답은 고저가를 줄 때도 있고 안 줄 때도 있습니다.
                # 없으면 None 으로 두고 enrich_ranges 가 KIS 로 채웁니다.
                high=_ratio(row.get("highPriceRaw") or row.get("highPrice")),
                low=_ratio(row.get("lowPriceRaw") or row.get("lowPrice")),
                source="네이버금융 순위",
            ).derive())
        if len(out) >= limit:
            break
    # 거래대금 큰 순으로 — '순위 API 를 합친 것'이므로 정렬은 우리가 합니다
    out.sort(key=lambda c: c.trading_value_krw, reverse=True)
    return out[:limit]


# 고저가를 채우려고 개별 조회를 몇 개까지 할 것인가.
# 거래량순위 응답에는 고가·저가가 없어서 '가격 변동폭'만 따로 채워야 합니다.
# 종목당 1회 호출인데 kis_client 는 실전에서도 초당 3회로 조여져 있으므로
# (REAL_MIN_INTERVAL=0.35) 12종목이면 약 4초입니다. 사람이 [종목 찾기] 를
# 눌렀을 때는 감당되지만, 자동 갱신 루프에서는 enrich=False 로 끄고
# 변동폭은 실시간체결 스트림(kis_realtime)이 주는 고가/저가로 채웁니다.
ENRICH_LIMIT = 12


def enrich_ranges(candidates: list[Candidate], limit: int = ENRICH_LIMIT,
                  errors: list = None) -> list[Candidate]:
    """당일 고저 변동폭(과 미국은 평균거래량)을 개별 시세 조회로 채웁니다.

    순위·구성종목 API 는 증가율·회전율은 주지만 고가/저가는 주지 않습니다.
    변동폭은 "먹을 폭이 있는가" 를 보는 지표라 초단타에서 빼놓을 수 없어서,
    **상위 후보에 한해서만** 한 번 더 물어봅니다. 전부 조회하면 초당 한도에
    걸려 스크리너 자체가 실패합니다.
    """
    errors = errors if errors is not None else []

    done = 0
    for candidate in candidates:
        if done >= limit:
            break
        if candidate.market == "US":
            if candidate.day_range_pct is not None and candidate.avg_volume:
                continue
            done += _enrich_us(candidate)
            continue
        if candidate.day_range_pct is not None or not kis_client.is_configured():
            continue
        try:
            quote = kis_client.get_quote(candidate.key)
        except Exception as exc:
            errors.append(f"{candidate.key} 고저가 조회 실패: {exc}")
            break                      # 한 번 막히면 나머지도 막힙니다
        done += 1
        if not quote:
            continue
        candidate.high = quote.get("high") or candidate.high
        candidate.low = quote.get("low") or candidate.low
        candidate.derive()
    return candidates


def _enrich_us(candidate: Candidate) -> int:
    """미국 종목 한 개를 Yahoo 일봉으로 보강. (조회했으면 1)

    나스닥 스크리너 표에는 고가·저가·평균거래량이 없습니다. 그 셋이 없으면
    초단타의 '변동폭'과 '거래량 급증' 기준이 통째로 빠집니다.

    한 달치 일봉을 한 번에 받아 meta 로 당일 고저를, 일봉 거래량으로 평균을
    구합니다 — 두 지표를 각각 조회하면 호출이 두 배가 됩니다.
    """
    data = http_client.get_json(YAHOO_CHART.format(sym=candidate.key),
                                params={"range": "1mo", "interval": "1d"}, timeout=10)
    result = (((data or {}).get("chart") or {}).get("result") or [{}])[0] or {}
    meta = result.get("meta") or {}
    if not meta:
        return 1
    candidate.high = _num(meta.get("regularMarketDayHigh")) or candidate.high
    candidate.low = _num(meta.get("regularMarketDayLow")) or candidate.low
    candidate.prev_close = (_num(meta.get("chartPreviousClose")
                                 or meta.get("previousClose")) or candidate.prev_close)

    # 평균거래량 — **오늘을 뺀** 과거 일봉으로 계산합니다. 오늘을 넣으면
    # 급증한 당일 거래량이 분모에도 들어가 증가율이 눌립니다.
    volumes = (((result.get("indicators") or {}).get("quote") or [{}])[0]
               or {}).get("volume") or []
    past = [float(v) for v in volumes[:-1] if v]
    if past:
        candidate.avg_volume = sum(past) / len(past)
    candidate.derive()
    return 1


def _members_to_candidates(found, min_price: float, max_price: float,
                           universe_key: str, fx_rate: float = 1.0) -> list[Candidate]:
    """유니버스 구성종목(시세 포함) → 후보.

    구성종목 응답에는 고가·저가가 없어 변동폭은 `None`(모름) 으로 둡니다.
    0 으로 채우면 '조건 미달' 로 전부 탈락합니다.
    """
    out = []
    for member in found.members:
        if not (min_price <= member.price <= max_price):
            continue
        is_us = member.market == "US"
        out.append(Candidate(
            key=member.key, name=member.name,
            market=member.market, segment=member.segment or member.market,
            universe=universe_key, sector=member.sector, weight=member.weight,
            price=member.price,
            price_krw=member.price * fx_rate if is_us else member.price,
            change_rate=member.change_rate,
            volume=member.volume,
            trading_value=member.trading_value,
            trading_value_krw=(member.trading_value * fx_rate if is_us
                               else member.trading_value),
            avg_volume=member.prev_volume,
            listed_shares=member.listed_shares,
            source=found.source or "유니버스 구성종목",
        ).derive())
    out.sort(key=lambda c: c.trading_value_krw, reverse=True)
    return out


def korean_candidates(min_price: int, max_price: int, limit: int = 60,
                      errors: list = None, rank_basis: str = "volume",
                      enrich: bool = False, capture_raw: list = None,
                      segments: list[str] = None,
                      universe_key: str = "") -> list[Candidate]:
    """한국 후보 — 고른 세부시장과 탐색 범위 안에서만.

    경로가 셋인 이유
        · 지수 코드가 있는 범위(코스피200·KRX100)와 시장 전체는 순위 API 가
          서버에서 걸러 주므로 한 번의 호출로 끝납니다.
        · 코스닥150·섹터처럼 지수 코드가 없는 범위는 그 지수를 추종하는 ETF 의
          구성종목시세를 씁니다 (역시 한 번의 호출, 시세까지 함께 옵니다).
        · KIS 키가 없으면 네이버 거래상위로 내려갑니다 — 이때는 시장 구분만
          가능하고 지수·섹터 범위는 지원되지 않습니다.
    """
    errors = errors if errors is not None else []
    segments = [s for s in (segments or ["KOSPI", "KOSDAQ"])
                if universe_mod.SEGMENTS.get(s, {}).get("region") == "KR"]
    universe = universe_mod.get(universe_key) if universe_key else None
    if universe is not None and universe.region != "KR":
        errors.append(f"'{universe.label}' 는 미국 범위라 국내 시장에서는 찾지 않습니다.")
        return []

    found: list[Candidate] = []

    # --- 1) 순위 API — 지수 코드가 있으면 지수 **전체**가 대상입니다 ---
    # 시장을 둘 다 켜면 지수 구분이 없는 '전체'(0000) 한 번으로 끝납니다.
    # 하나만 켜면 그 시장 코드로 조회해 다른 시장이 섞이지 않게 합니다.
    if universe is not None and universe.kis_iscd:
        iscd = universe.kis_iscd
    elif universe is None and len(segments) == 1:
        iscd = universe_mod.SEGMENTS[segments[0]]["kis_iscd"]
    elif universe is None:
        iscd = "0000"
    else:
        iscd = ""                       # 지수 코드가 없는 범위 → ETF 경로로

    if iscd:
        found = _kis_volume_rank(min_price, max_price, limit=limit, errors=errors,
                                 rank_basis=rank_basis, capture_raw=capture_raw,
                                 iscd=iscd, universe_key=universe_key)
        # 전체(0000)로 받아온 경우에는 시장 판정 결과로 한 번 더 걸러야 합니다.
        chosen = set(segments)
        if found and chosen and chosen != {"KOSPI", "KOSDAQ"}:
            found = [c for c in found if c.market in chosen]

    # --- 2) ETF 구성종목 — 코스닥150·섹터처럼 지수 코드가 없는 범위 ---
    # 순위 API 가 지수 코드로 빈 손이었을 때의 대비책이기도 합니다.
    if not found and universe is not None and universe.etf:
        members = universe_mod.members(universe.key, segments)
        if members.error:
            errors.append(members.error)
        found = _members_to_candidates(members, min_price, max_price, universe.key)
        if not found and not members.error:
            errors.append(f"{universe.label} 구성종목 중 {min_price:,}~{max_price:,}원 "
                          f"가격대에 해당하는 종목이 없습니다.")

    # --- 3) 폴백: 네이버 거래상위 (KIS 키가 없을 때) ---
    if not found and universe is None:
        pool = []
        for segment in segments:
            pool += _naver_active(segment, limit)
        found = [c for c in pool if min_price <= c.price <= max_price]
        if not found:
            errors.append("네이버 거래상위에서도 해당 가격대 종목을 찾지 못했습니다.")
    elif not found and universe is not None and not kis_client.is_configured():
        errors.append(f"KIS 키가 없어 '{universe.label}' 범위로는 찾을 수 없습니다 "
                      f"(네이버 폴백은 지수·섹터를 구분하지 못합니다).")

    if found and enrich:
        enrich_ranges(found, errors=errors)
    return found[:limit]


# ---------------------------------------------------------------------------
# 미국
# ---------------------------------------------------------------------------

def _yahoo_screener_rows(scr_id: str, count: int = 50) -> list[dict]:
    """예약 스크리너 한 종류의 결과 행. 시세 필드까지 통째로 돌려줍니다.

    스크리너 응답에 현재가·등락률·거래량이 이미 들어 있어서 별도 시세 조회가
    필요 없습니다. (예전에 쓰던 v7 quote API 는 crumb 인증이 걸려 죽었습니다 —
    배치 시세 조회는 더 이상 하지 않습니다.)
    """
    data = http_client.get_json(
        YAHOO_SCREENER, params={"scrIds": scr_id, "count": count, "start": 0}, timeout=12)
    result = (((data or {}).get("finance") or {}).get("result") or [{}])[0] or {}
    return [q for q in (result.get("quotes") or []) if q.get("symbol")]


YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"


def _yahoo_chart_quotes(symbols: list[str]) -> list[Candidate]:
    """씨앗 목록 폴백 — chart API meta 로 종목별 시세 (crumb 없이 동작하는 경로)."""
    out = []
    for sym in symbols:
        data = http_client.get_json(YAHOO_CHART.format(sym=sym),
                                    params={"range": "1d", "interval": "1d"}, timeout=8)
        meta = ((((data or {}).get("chart") or {}).get("result") or [{}])[0]
                or {}).get("meta") or {}
        price = _num(meta.get("regularMarketPrice"))
        if price <= 0:
            continue
        prev = _num(meta.get("chartPreviousClose") or meta.get("previousClose"))
        volume = _num(meta.get("regularMarketVolume"))
        out.append(Candidate(
            key=sym, name=meta.get("shortName") or sym,
            market="US", price=price, price_krw=0.0,
            change_rate=round((price - prev) / prev * 100, 2) if prev > 0 else 0.0,
            volume=volume, trading_value=price * volume,
            prev_close=prev or None,
            high=_num(meta.get("regularMarketDayHigh")) or None,
            low=_num(meta.get("regularMarketDayLow")) or None,
            source="Yahoo chart",
        ))
    return out


def us_candidates(min_price: float, max_price: float, limit: int = 60,
                  errors: list = None, segments: list[str] = None,
                  universe_key: str = "", enrich: bool = False) -> list[Candidate]:
    """미국 후보.

    거래소(나스닥/뉴욕)나 지수·섹터를 지정하면 **나스닥 스크리너**에서 출발합니다 —
    미국 상장 전 종목의 시세·거래량·시총·섹터가 한 번의 호출로 오기 때문에,
    "나스닥100 안에서" 나 "기술주 중에서" 같은 조건을 그 표에서 바로 걸러낼 수
    있습니다. Yahoo 예약 스크리너는 거래소도 섹터도 알려주지 않습니다.

    표에 없는 것이 하나 있습니다 — **평균거래량**. 거래량 급증(vol_increase)은
    그래서 `enrich=True` 일 때 상위 후보에 한해 Yahoo 로 채웁니다.
    """
    errors = errors if errors is not None else []
    segments = [s for s in (segments or ["NASDAQ", "NYSE"])
                if universe_mod.SEGMENTS.get(s, {}).get("region") == "US"]
    universe = universe_mod.get(universe_key) if universe_key else None

    from data_sources import fx
    rate, _ = fx.usd_krw()

    if universe is not None or segments != ["NASDAQ", "NYSE"]:
        found = _nasdaq_candidates(min_price, max_price, limit, errors,
                                   segments, universe, rate)
        if found:
            if enrich:
                enrich_ranges(found, errors=errors)
            return found
        if universe is not None:
            return []          # 범위를 지정했는데 못 찾았으면 전체로 되돌리지 않습니다
        errors.append("나스닥 스크리너가 비어 Yahoo 활발종목으로 대체합니다.")

    rows = []
    for scr in ("most_actives", "day_gainers", "small_cap_gainers"):
        rows.extend(_yahoo_screener_rows(scr, count=50))
        if len(rows) >= limit * 3:
            break

    seen: set = set()
    quotes = []
    for q in rows:
        sym = str(q.get("symbol") or "")
        price = _num(q.get("regularMarketPrice"))
        if not sym or sym in seen or price <= 0:
            continue
        seen.add(sym)
        volume = _num(q.get("regularMarketVolume"))
        quotes.append(Candidate(
            key=sym, name=q.get("shortName") or q.get("longName") or sym,
            market="US", price=price, price_krw=0.0,
            change_rate=_num(q.get("regularMarketChangePercent")),
            volume=volume, trading_value=price * volume,
            prev_close=_num(q.get("regularMarketPreviousClose")) or None,
            high=_num(q.get("regularMarketDayHigh")) or None,
            low=_num(q.get("regularMarketDayLow")) or None,
            # 야후는 평균거래량을 같이 줍니다 — 거래량 급증을 여기서 계산합니다.
            # 상장주식수는 안 주므로 미국 경로에 회전율은 없습니다(None = 심사 제외).
            avg_volume=_ratio(q.get("averageDailyVolume3Month")
                              or q.get("averageDailyVolume10Day")),
            source="Yahoo 스크리너",
        ).derive())

    if not quotes:
        # 스크리너가 통째로 막히면 씨앗 목록으로라도 후보를 만듭니다
        errors.append("Yahoo 스크리너 응답이 비어 씨앗 목록으로 대체했습니다.")
        quotes = _yahoo_chart_quotes(list(US_SEED))

    out = []
    for c in quotes:
        if not (min_price <= c.price <= max_price):
            continue
        c.price_krw = c.price * rate
        c.trading_value_krw = c.trading_value * rate
        out.append(c)
    # 거래대금 큰 순 — 유동성 우선 원칙은 한국 경로와 같습니다
    out.sort(key=lambda c: c.trading_value_krw, reverse=True)
    return out[:limit]


def _nasdaq_candidates(min_price: float, max_price: float, limit: int,
                       errors: list, segments: list[str],
                       universe, fx_rate: float) -> list[Candidate]:
    """나스닥 스크리너 표에서 후보 만들기 (지수·섹터·거래소 필터 포함)."""
    if universe is not None and universe.region != "US":
        errors.append(f"'{universe.label}' 는 국내 범위라 미국 시장에서는 찾지 않습니다.")
        return []

    if universe is not None:
        found = universe_mod.members(universe.key, segments)
        if found.error:
            errors.append(found.error)
        return _members_to_candidates(found, min_price, max_price,
                                      universe.key, fx_rate)[:limit]

    # 범위를 지정하지 않았으면 거래소 전체에서 거래대금 상위로 추립니다.
    members = universe_mod.MemberSet(source="나스닥 스크리너")
    for segment in segments:
        part = universe_mod.us_members(
            exchange=universe_mod.SEGMENTS[segment]["nasdaq_exchange"])
        members.members += part.members
        if part.error and not members.members:
            errors.append(part.error)
    if not members.members:
        return []
    return _members_to_candidates(members, min_price, max_price, "", fx_rate)[:limit]


# ---------------------------------------------------------------------------
# 통합 조회
# ---------------------------------------------------------------------------

def scan(markets: list[str], kr_price: tuple[int, int], us_price: tuple[float, float],
         limit: int = 60, use_cache: bool = True, rank_basis: str = "volume",
         enrich: bool = False, capture_raw: list = None,
         universe: str = "") -> ScanResult:
    """고른 시장·범위에서 후보를 모아 하나의 결과로.

    `markets` 는 세부시장 코드입니다 (KOSPI / KOSDAQ / NASDAQ).
    예전 값 "KR" · "US" 도 그대로 받습니다 — 저장된 설정을 깨뜨리지 않으려고
    `universe.normalize_segments` 가 코스피+코스닥 / 나스닥+뉴욕으로 풀어줍니다.

    `universe` 는 탐색 범위 키입니다 (KOSPI200 / KOSDAQ150 / NASDAQ100 / SP500 /
    KR_SEMI / US_TECHNOLOGY …). 비워두면 시장 전체를 훑습니다.

    캐시를 쓰는 이유: 화면이 실시간으로 갱신되는데 매번 시장 전체를 훑으면
    데이터 제공처에서 차단당합니다.

    **기본값은 일반 AI 추천기가 쓰던 그대로입니다** (평균거래량 순, 보강 없음).
    페니 초단타만 `rank_basis="vol_increase"` 나 `"turnover"` 로 타점 기준을
    바꾸고 `enrich=True` 로 고저가를 보강합니다 — 여기 기본값을 초단타 쪽에
    맞추면 일반 추천기의 후보 선정까지 같이 바뀝니다.
    """
    segments = universe_mod.normalize_segments(markets)
    universe = str(universe or "").strip().upper()
    picked = universe_mod.get(universe) if universe else None

    key = (f"{segments}:{universe}:{kr_price}:{us_price}:"
           f"{limit}:{rank_basis}:{enrich}")
    now = time.time()
    if use_cache:
        with _cache_lock:
            hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1]

    result = ScanResult()
    result.segments = segments
    result.universe = universe
    if universe and picked is None:
        result.errors.append(f"알 수 없는 탐색 범위 '{universe}' — 시장 전체에서 찾습니다.")
        universe = ""

    kr_segments = [s for s in segments if universe_mod.SEGMENTS[s]["region"] == "KR"]
    us_segments = [s for s in segments if universe_mod.SEGMENTS[s]["region"] == "US"]

    # 범위가 한쪽 시장 전용이면 반대쪽은 아예 훑지 않습니다 — '코스피200'을
    # 고른 채로 나스닥까지 켜져 있으면 미국 종목이 섞여 들어옵니다.
    if picked is not None:
        if picked.region == "KR":
            us_segments = []
        else:
            kr_segments = []
        if not kr_segments and not us_segments:
            # 범위와 시장이 완전히 어긋난 상태. 0건만 보여주면 사용자는 "종목이
            # 없다"고 읽지만, 실제로는 시장 하나만 켜면 바로 나옵니다.
            need = " · ".join(universe_mod.segment_label(s) for s in picked.segments)
            result.errors.append(
                f"'{picked.label}' 범위는 {need} 시장에 있습니다 — "
                f"지금 켜 놓은 시장과 겹치지 않아 찾을 수 없습니다.")
            return result

    if kr_segments:
        try:
            result.candidates += korean_candidates(
                kr_price[0], kr_price[1], limit, errors=result.errors,
                rank_basis=rank_basis, enrich=enrich, capture_raw=capture_raw,
                segments=kr_segments, universe_key=universe)
        except Exception as exc:
            result.errors.append(f"한국 스캔 실패: {type(exc).__name__}: {exc}")
    if us_segments:
        try:
            found = us_candidates(us_price[0], us_price[1], limit,
                                  errors=result.errors, segments=us_segments,
                                  universe_key=universe, enrich=enrich)
            if not found:
                result.errors.append(
                    "미국 스크리너에서 해당 가격대 종목을 받지 못했습니다 "
                    "(스크리너가 막혔거나 장 마감).")
            result.candidates += found
        except Exception as exc:
            result.errors.append(f"미국 스캔 실패: {type(exc).__name__}: {exc}")

    result.sources = sorted({c.source for c in result.candidates})
    with _cache_lock:
        _cache[key] = (now, result)
    return result


def clear_cache():
    with _cache_lock:
        _cache.clear()
