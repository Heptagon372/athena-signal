"""
종목 레지스트리 (Symbol Registry)
--------------------------------
"삼성전자", "005930", "005930.KS", "에코프로비엠", "AAPL" 같은 사용자 입력을
**실제로 상장되어 있는 종목**으로 확정하는 곳입니다.

핵심 원칙: 존재하지 않는 종목은 절대 조용히 통과시키지 않습니다.
어디에서도 확인되지 않으면 SymbolNotFoundError 를 던지고, 오타로 보이는
유사 종목 후보(suggestions)를 함께 돌려줍니다.

실존 확인 경로
    한국(코스피/코스닥/코넥스)
      1) KRX(KIND) 상장법인 목록 -> 로컬 캐시(.cache/krx_symbols.json, 3일 TTL)
      2) 캐시에 없으면 토스증권 종목정보 API / 네이버 금융 API 로 직접 조회
    미국
      Yahoo Finance 심볼 검색 API 에서 **정확히 일치하는** 심볼이 있을 때만 인정
"""

import json
import re
import time
import unicodedata
from datetime import datetime, timedelta

from bs4 import BeautifulSoup

from config import CACHE_DIR, SYMBOL_CACHE_TTL_DAYS
from models import (
    ResolvedSymbol, SymbolNotFoundError,
    MARKET_KOSPI, MARKET_KOSDAQ, MARKET_KONEX, MARKET_US,
)
from data_sources import http_client

KIND_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do"
TOSS_INFO_URL = "https://wts-info-api.tossinvest.com/api/v2/stock-infos/A{code}"
NAVER_BASIC_URL = "https://m.stock.naver.com/api/stock/{code}/basic"
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

NAVER_HEADERS = {"Referer": "https://m.stock.naver.com/", "Accept": "application/json"}
TOSS_HEADERS = {"Referer": "https://tossinvest.com/", "Accept": "application/json"}

CACHE_FILE = CACHE_DIR / "krx_symbols.json"

_MARKET_BY_KOREAN = {
    "유가증권시장": MARKET_KOSPI,
    "코스피": MARKET_KOSPI,
    "코스닥": MARKET_KOSDAQ,
    "코스닥시장": MARKET_KOSDAQ,
    "코넥스": MARKET_KONEX,
    "코넥스시장": MARKET_KONEX,
}

# 토스/네이버가 돌려주는 시장 코드
_MARKET_BY_CODE = {
    "KSP": MARKET_KOSPI, "KOSPI": MARKET_KOSPI, "0": MARKET_KOSPI,
    "KSQ": MARKET_KOSDAQ, "KOSDAQ": MARKET_KOSDAQ, "1": MARKET_KOSDAQ,
    "KNX": MARKET_KONEX, "KONEX": MARKET_KONEX,
}

_YAHOO_SUFFIX = {
    MARKET_KOSPI: ".KS",
    MARKET_KOSDAQ: ".KQ",
    MARKET_KONEX: ".KQ",   # 코넥스는 Yahoo 커버리지가 불안정 (국내 소스 우선 사용)
}

# 프로세스 내 메모리 캐시
_master_cache: dict | None = None
_master_loaded_at: float = 0.0
_MEMORY_TTL_SEC = 600


# ---------------------------------------------------------------------------
# KRX 상장종목 마스터
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """비교용 정규화: 공백/특수문자 제거 + 소문자 + 유니코드 NFKC"""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"[\s\-_.·,()]", "", text).lower()


def _download_kind_market(market_param: str, market: str) -> list[dict]:
    """KIND(한국거래소 전자공시) 상장법인 목록을 내려받아 파싱.

    응답은 euc-kr 로 인코딩된 HTML <table> 이며,
    컬럼 순서는 회사명 / 시장구분 / 종목코드 / 업종 / ... 입니다.
    """
    html = http_client.get_text(
        KIND_URL,
        params={"method": "download", "marketType": market_param},
        headers={"Referer": "https://kind.krx.co.kr/"},
        encoding="euc-kr",
        timeout=30,
    )
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("tr"):
        cells = [td.get_text(strip=True) for td in tr.select("td")]
        if len(cells) < 3:
            continue
        name, market_text, code = cells[0], cells[1], cells[2]
        code = re.sub(r"\D", "", code)
        if len(code) != 6 or not name:
            continue
        rows.append({
            "code": code,
            "name": name,
            "market": _MARKET_BY_KOREAN.get(market_text, market),
            "sector": cells[3] if len(cells) > 3 else "",
        })
    return rows


def _load_cache() -> dict | None:
    if not CACHE_FILE.exists():
        return None
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    fetched_at = datetime.fromisoformat(payload.get("fetched_at", "1970-01-01"))
    if datetime.utcnow() - fetched_at > timedelta(days=SYMBOL_CACHE_TTL_DAYS):
        payload["stale"] = True
    return payload


def _save_cache(entries: list[dict]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"fetched_at": datetime.utcnow().isoformat(), "entries": entries},
                   ensure_ascii=False),
        encoding="utf-8",
    )


def get_krx_master(force_refresh: bool = False) -> dict:
    """{"by_code": {...}, "by_name": {...}, "entries": [...]} 형태의 상장종목 마스터.

    네트워크가 죽어 있어도 캐시가 있으면 그걸로 계속 동작합니다(만료돼도 사용).
    """
    global _master_cache, _master_loaded_at

    if (not force_refresh and _master_cache
            and time.time() - _master_loaded_at < _MEMORY_TTL_SEC):
        return _master_cache

    cached = _load_cache()
    entries = None
    if cached and not cached.get("stale") and not force_refresh:
        entries = cached.get("entries")

    if entries is None:
        downloaded = (_download_kind_market("stockMkt", MARKET_KOSPI)
                      + _download_kind_market("kosdaqMkt", MARKET_KOSDAQ)
                      + _download_kind_market("konexMkt", MARKET_KONEX))
        if downloaded:
            entries = downloaded
            _save_cache(entries)
        elif cached:
            # 다운로드 실패 -> 만료된 캐시라도 사용 (오프라인 대응)
            entries = cached.get("entries", [])
        else:
            entries = []

    by_code, by_name = {}, {}
    for e in entries:
        by_code[e["code"]] = e
        by_name.setdefault(_normalize(e["name"]), e)

    _master_cache = {"by_code": by_code, "by_name": by_name, "entries": entries}
    _master_loaded_at = time.time()
    return _master_cache


# ---------------------------------------------------------------------------
# 실시간 실존 확인 (마스터에 없는 코드용)
# ---------------------------------------------------------------------------

def _verify_korean_code_live(code: str) -> tuple[str, str, str] | None:
    """토스 -> 네이버 순으로 조회. (종목명, 시장, 확인소스) 또는 None."""
    data = http_client.get_json(TOSS_INFO_URL.format(code=code), headers=TOSS_HEADERS)
    if data:
        result = data.get("result")
        # 없는 종목이면 토스는 result: null 을 돌려줍니다
        if result and result.get("name"):
            market_code = (result.get("market") or {}).get("code", "")
            market = _MARKET_BY_CODE.get(market_code.upper(), MARKET_KOSPI)
            return result["name"], market, "토스증권"

    data = http_client.get_json(NAVER_BASIC_URL.format(code=code), headers=NAVER_HEADERS)
    if data and data.get("stockName"):
        exchange = (data.get("stockExchangeName") or "").upper()
        market = _MARKET_BY_CODE.get(exchange) or _MARKET_BY_CODE.get(str(data.get("sosok", "")))
        return data["stockName"], market or MARKET_KOSPI, "네이버금융"

    return None


# 레버리지/인버스 ETF 판별 패턴
# Yahoo 의 shortname 은 30자에서 잘려서("Direxion Daily Semiconductor Be") 판별이 안 되므로
# 반드시 longname 을 함께 봐야 합니다.
# 한글은 \b 단어경계가 먹지 않습니다("선물인버스2X"는 전부 \w 라 경계가 없음).
# 그래서 영문 용어에만 \b 를 걸고 한글 용어는 부분문자열로 찾습니다.
_INVERSE_PATTERNS = re.compile(
    r"\b(?:bear|inverse|short|ultrashort)\b"      # 영문
    r"|-\s*[1-9](?:\.\d)?\s*x\b"                  # -1x, -3x
    r"|인버스|곱버스|리버스",                        # 한글 (경계 없이)
    re.IGNORECASE,
)
# 배율: 앞 글자가 영숫자/점/하이픈이 아니면 허용 -> 한글 뒤에 붙은 "인버스2X" 도 인식
_LEVERAGE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.\-])([1-9](?:\.\d)?)\s*x\b|\b(ultra|ultrapro)\b|레버리지",
    re.IGNORECASE,
)


def classify_etf(name: str, long_name: str = "") -> tuple[bool, float]:
    """상품명에서 (인버스 여부, 레버리지 배율)을 추정.

    예) "Direxion Daily Semiconductor Bear 3X Shares" -> (True, 3.0)
        "Direxion Daily Semiconductor Bull 3X Shares" -> (False, 3.0)
        "KODEX 200선물인버스2X"                        -> (True, 2.0)
    """
    text = f"{long_name} {name}".strip()
    if not text:
        return False, 1.0

    is_inverse = bool(_INVERSE_PATTERNS.search(text))

    leverage = 1.0
    match = _LEVERAGE_PATTERN.search(text)
    if match:
        if match.group(1):
            try:
                leverage = float(match.group(1))
            except ValueError:
                leverage = 1.0
        else:
            leverage = 2.0      # Ultra/UltraPro 계열은 통상 2x 이상
    return is_inverse, leverage


def _yahoo_lookup(query: str) -> list[dict]:
    data = http_client.get_json(
        YAHOO_SEARCH_URL,
        params={"q": query, "quotesCount": "8", "newsCount": "0"},
    )
    if not data:
        return []
    out = []
    for q in data.get("quotes", []):
        symbol = q.get("symbol")
        if not symbol:
            continue
        out.append({
            "symbol": symbol,
            "name": q.get("shortname") or q.get("longname") or symbol,
            "long_name": q.get("longname") or "",
            "exchange": q.get("exchange", ""),
            "quote_type": q.get("quoteType", ""),
        })
    return out


# ---------------------------------------------------------------------------
# 검색 / 해석
# ---------------------------------------------------------------------------

def search(query: str, limit: int = 8) -> list[dict]:
    """자동완성용 후보 목록. 확정이 아니라 '보여주기용' 검색입니다."""
    query = (query or "").strip()
    if not query:
        return []

    norm = _normalize(query)
    master = get_krx_master()
    results, seen = [], set()

    def push(entry, score):
        if entry["code"] in seen:
            return
        seen.add(entry["code"])
        results.append((score, {
            "key": entry["code"],
            "name": entry["name"],
            "market": entry["market"],
            "sector": entry.get("sector", ""),
        }))

    for entry in master["entries"]:
        name_norm = _normalize(entry["name"])
        if name_norm == norm:
            push(entry, 0)
        elif name_norm.startswith(norm):
            push(entry, 1)
        elif norm and norm in name_norm:
            push(entry, 2)
        elif entry["code"].startswith(re.sub(r"\D", "", query)) and query.strip().isdigit():
            push(entry, 3)

    results.sort(key=lambda x: (x[0], len(x[1]["name"])))
    out = [r[1] for r in results[:limit]]

    # 영문 입력이면 미국 종목 후보도 함께
    if re.fullmatch(r"[A-Za-z.\-]{1,12}", query.strip()):
        for q in _yahoo_lookup(query)[: limit - len(out)]:
            if q["quote_type"] not in ("EQUITY", "ETF"):
                continue
            out.append({
                "key": q["symbol"],
                "name": q["name"],
                "market": MARKET_US,
                "sector": q["exchange"],
            })
    return out[:limit]


def _build_korean(code: str, name: str, market: str, verified_by: str) -> ResolvedSymbol:
    is_inverse, leverage = classify_etf(name)
    return ResolvedSymbol(
        key=code,
        name=name,
        market=market,
        yahoo_symbol=f"{code}{_YAHOO_SUFFIX.get(market, '.KS')}",
        currency="KRW",
        verified_by=verified_by,
        is_inverse=is_inverse,
        leverage=leverage,
    )


def resolve(query: str) -> ResolvedSymbol:
    """사용자 입력 -> 확정 종목. 실존하지 않으면 SymbolNotFoundError.

    허용 입력:
        005930 / 005930.KS / 005930.KQ / A005930  -> 한국 종목코드
        삼성전자 / 에코프로비엠                    -> 한국 종목명
        AAPL / TSLA / BRK-B                        -> 미국 티커
    """
    raw = (query or "").strip()
    if not raw:
        raise SymbolNotFoundError("", reason="종목명이나 종목코드를 입력해 주세요.")

    master = get_krx_master()

    # --- 1) 한국 종목코드 형태인가 (005930, 005930.KS, A005930) ---
    code_match = re.fullmatch(r"[Aa]?(\d{6})(\.(?:KS|KQ|ks|kq))?", raw)
    if code_match:
        code = code_match.group(1)
        entry = master["by_code"].get(code)
        if entry:
            return _build_korean(code, entry["name"], entry["market"], "KRX 상장법인목록")

        live = _verify_korean_code_live(code)
        if live:
            name, market, source = live
            return _build_korean(code, name, market, source)

        raise SymbolNotFoundError(
            raw,
            suggestions=search(code, limit=5),
            reason=f"종목코드 '{code}' 는 코스피·코스닥·코넥스 어디에도 상장되어 있지 않습니다. "
                   f"(상장폐지되었거나 존재하지 않는 코드입니다)",
        )

    # --- 2) 한글이 섞여 있으면 한국 종목명으로 취급 ---
    if re.search(r"[가-힣]", raw):
        norm = _normalize(raw)
        entry = master["by_name"].get(norm)
        if entry:
            return _build_korean(entry["code"], entry["name"], entry["market"], "KRX 상장법인목록")

        candidates = search(raw, limit=5)
        exact = [c for c in candidates if _normalize(c["name"]) == norm]
        if exact:
            c = exact[0]
            return _build_korean(c["key"], c["name"], c["market"], "KRX 상장법인목록")

        raise SymbolNotFoundError(
            raw,
            suggestions=candidates,
            reason=f"'{raw}' 라는 이름의 상장 종목을 찾을 수 없습니다."
                   + ("  혹시 아래 종목을 찾으셨나요?" if candidates else ""),
        )

    # --- 3) 영문/숫자 조합 -> 미국 티커 ---
    ticker = raw.upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise SymbolNotFoundError(
            raw, reason=f"'{raw}' 는 올바른 종목코드/티커 형식이 아닙니다."
        )

    quotes = _yahoo_lookup(ticker)
    exact = next((q for q in quotes if q["symbol"].upper() == ticker), None)
    if exact:
        is_inverse, leverage = classify_etf(exact["name"], exact.get("long_name", ""))
        return ResolvedSymbol(
            key=ticker,
            name=exact["name"],
            market=MARKET_US,
            yahoo_symbol=ticker,
            currency="USD",
            verified_by=f"Yahoo Finance ({exact['exchange']})",
            long_name=exact.get("long_name", ""),
            is_inverse=is_inverse,
            leverage=leverage,
            quote_type=exact.get("quote_type", ""),
        )

    suggestions = [
        {"key": q["symbol"], "name": q["name"], "market": MARKET_US, "sector": q["exchange"]}
        for q in quotes[:5]
    ]
    raise SymbolNotFoundError(
        raw,
        suggestions=suggestions,
        reason=f"티커 '{ticker}' 로 상장된 종목을 찾을 수 없습니다."
               + ("  비슷한 티커를 확인해 보세요." if suggestions else ""),
    )


def try_resolve(query: str) -> ResolvedSymbol | None:
    """예외 대신 None 을 원할 때 쓰는 래퍼 (CLI/배치용)."""
    try:
        return resolve(query)
    except SymbolNotFoundError:
        return None
