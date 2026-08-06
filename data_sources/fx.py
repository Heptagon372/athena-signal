"""
환율 (FX)
---------
모의투자 계좌는 원화 단일 통화로 운영합니다. 미국 종목은 달러로 호가되므로
매매·평가 시점에 원화로 환산해야 합니다.

이 환산이 없으면 "100만원으로 AAPL 3,229주(약 14억원어치) 매수" 같은 일이
벌어집니다. (실제로 이 버그가 있었습니다)

우선순위
    1) 토스증권 Open API `/api/v1/exchange-rate` (키가 있을 때)
    2) Yahoo `KRW=X` (키 불필요)
"""

import time

from data_sources import http_client

_cache: tuple[float, float] | None = None
_TTL = 600          # 10분

# 조회 실패 시 사용할 보수적 기본값 (화면에 '추정치'로 표시됩니다)
FALLBACK_USD_KRW = 1350.0


def usd_krw() -> tuple[float, str]:
    """(환율, 출처). 실패해도 예외를 던지지 않고 기본값을 돌려줍니다."""
    global _cache
    if _cache and time.time() - _cache[0] < _TTL:
        return _cache[1], _cache[2] if len(_cache) > 2 else "캐시"

    rate, source = None, ""

    # 1) 토스 공식 API
    try:
        from data_sources import toss_api
        if toss_api.is_configured():
            headers = toss_api._auth_headers()
            if headers:
                data = http_client.get_json(
                    toss_api.BASE + "/api/v1/exchange-rate",
                    headers=headers, params={"currency": "USD"}, timeout=10)
                result = (data or {}).get("result")
                if isinstance(result, dict):
                    for key in ("rate", "basePrice", "value"):
                        if result.get(key):
                            rate = float(str(result[key]).replace(",", ""))
                            source = "토스증권 Open API"
                            break
    except Exception:
        rate = None

    # 2) Yahoo
    if not rate:
        data = http_client.get_json(
            "https://query1.finance.yahoo.com/v8/finance/chart/KRW=X",
            params={"range": "1d", "interval": "1d"}, timeout=10)
        meta = (((data or {}).get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
        price = meta.get("regularMarketPrice")
        if price:
            rate, source = float(price), "Yahoo Finance"

    if not rate:
        rate, source = FALLBACK_USD_KRW, "기본값(조회 실패)"

    _cache = (time.time(), rate, source)
    return rate, source


def to_krw(price: float, currency: str) -> float:
    """호가 통화를 원화로 환산."""
    if price is None:
        return None
    if (currency or "KRW").upper() == "KRW":
        return float(price)
    rate, _ = usd_krw()
    return float(price) * rate
