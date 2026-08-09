"""
토스증권 공식 Open API 클라이언트
--------------------------------
https://openapi.tossinvest.com — OpenAPI 3.0 스펙 기준으로 구현했습니다.
(스펙 원본: https://openapi.tossinvest.com/openapi-docs/latest/openapi.json)

이 모듈이 기존 비공식 경로(wts-info-api.tossinvest.com)보다 나은 점
    · 공식 API라 약관상 안전하고 버전 관리가 됩니다
    · **장 운영 캘린더**를 제공해 공휴일까지 정확히 반영됩니다
      (기존에는 시간을 하드코딩해 공휴일에 "장중"으로 잘못 표시됐습니다)
    · 종목 마스터에 상장일·상장폐지일·거래정지 여부가 들어 있어 실존 검증이 정확합니다
    · 최대 200종목을 한 번에 조회할 수 있어 감시목록 갱신이 빨라집니다

키 발급
    https://developers.tossinvest.com 에서 앱 등록 후 client_id / client_secret 발급
        set TOSS_CLIENT_ID=...
        set TOSS_CLIENT_SECRET=...
    또는 아테나.bat → [7] API 키

키가 없으면 이 모듈은 조용히 비활성 상태로 남고, 기존 공개 경로가 그대로 동작합니다.

주의: 실제 발급 키로 응답을 검증하지는 못했습니다. 키를 넣은 뒤
      `python -m data_sources.toss_api` 로 자체 점검을 먼저 돌려보세요.
"""

import threading
import time
from datetime import datetime

import pandas as pd

from data_sources import credentials, http_client

BASE = "https://openapi.tossinvest.com"

_token_lock = threading.Lock()
_token: str | None = None
_token_expires_at: float = 0.0

# 캘린더는 하루에 몇 번만 받아도 충분합니다
_calendar_cache: dict[str, tuple[float, dict]] = {}
_CALENDAR_TTL = 1800


def is_configured() -> bool:
    return credentials.is_configured("toss")


def _get_token() -> str | None:
    """OAuth2 client_credentials 토큰 (기본 24시간 유효 -> 캐시 필수)."""
    global _token, _token_expires_at

    if not is_configured():
        return None

    with _token_lock:
        if _token and time.time() < _token_expires_at - 300:
            return _token

        res = http_client.post_form(
            BASE + "/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": credentials.get("TOSS_CLIENT_ID"),
                "client_secret": credentials.get("TOSS_CLIENT_SECRET"),
            },
            timeout=15,
        )
        if not res or not res.get("access_token"):
            return None

        _token = res["access_token"]
        _token_expires_at = time.time() + float(res.get("expires_in") or 86400)
        return _token


def _auth_headers() -> dict | None:
    token = _get_token()
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"} if token else None


def _num(v):
    """토스 API는 수치를 문자열로 돌려줍니다."""
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 시세
# ---------------------------------------------------------------------------

def get_prices(symbols: list[str]) -> dict:
    """현재가 조회 — 최대 200종목 배치. {symbol: {...}} 반환."""
    headers = _auth_headers()
    if not headers or not symbols:
        return {}

    data = http_client.get_json(
        BASE + "/api/v1/prices", headers=headers,
        params={"symbols": ",".join(symbols[:200])}, timeout=12,
    )
    out = {}
    for row in ((data or {}).get("result") or []):
        sym = row.get("symbol")
        if not sym:
            continue
        out[sym] = {
            "price": _num(row.get("lastPrice")),
            "currency": row.get("currency", "KRW"),
            "timestamp": row.get("timestamp"),
        }
    return out


def get_price(symbol: str) -> dict | None:
    return get_prices([symbol]).get(symbol)


def get_candles(symbol: str, interval: str = "1d", count: int = 200,
                adjusted: bool = True) -> pd.DataFrame:
    """캔들 조회.

    공식 API가 지원하는 interval 은 **1m(1분)과 1d(일봉) 두 가지뿐**입니다.
    주봉·월봉·년봉은 이 API에 없으므로 상위 계층이 네이버 경로를 씁니다.

    한 번에 최대 200봉이라, 더 필요하면 nextBefore 로 페이지를 이어받습니다.
    """
    headers = _auth_headers()
    if not headers or interval not in ("1m", "1d"):
        return pd.DataFrame()

    rows, before, guard = [], None, 0
    while len(rows) < count and guard < 10:
        guard += 1
        params = {"symbol": symbol, "interval": interval,
                  "count": str(min(200, count - len(rows))),
                  "adjusted": "true" if adjusted else "false"}
        if before:
            params["before"] = before

        data = http_client.get_json(BASE + "/api/v1/candles", headers=headers,
                                    params=params, timeout=20)
        result = (data or {}).get("result") or {}
        candles = result.get("candles") or []
        if not candles:
            break

        for c in candles:
            close = _num(c.get("closePrice"))
            if close is None:
                continue
            try:
                ts = datetime.fromisoformat(str(c["timestamp"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            rows.append({
                "date": ts.replace(tzinfo=None),
                "open": _num(c.get("openPrice")) or close,
                "high": _num(c.get("highPrice")) or close,
                "low": _num(c.get("lowPrice")) or close,
                "close": close,
                "volume": _num(c.get("volume")) or 0.0,
            })

        before = result.get("nextBefore")
        if not before:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="last")]


# ---------------------------------------------------------------------------
# 종목 정보
# ---------------------------------------------------------------------------

def get_stock_info(symbol: str) -> dict | None:
    """종목 기본 정보. 상장폐지·거래정지 여부까지 확인할 수 있습니다."""
    headers = _auth_headers()
    if not headers:
        return None

    data = http_client.get_json(BASE + "/api/v1/stocks", headers=headers,
                                params={"symbols": symbol}, timeout=12)
    rows = (data or {}).get("result") or []
    if not rows:
        return None

    r = rows[0]
    kr = r.get("koreanMarketDetail") or {}
    return {
        "symbol": r.get("symbol"),
        "name": r.get("name"),
        "english_name": r.get("englishName"),
        "market": r.get("market"),           # KOSPI / KOSDAQ / NASDAQ ...
        "security_type": r.get("securityType"),
        "status": r.get("status"),           # ACTIVE / DELISTED ...
        "currency": r.get("currency"),
        "list_date": r.get("listDate"),
        "delist_date": r.get("delistDate"),
        "shares_outstanding": _num(r.get("sharesOutstanding")),
        "leverage_factor": _num(r.get("leverageFactor")),
        "trading_suspended": bool(kr.get("krxTradingSuspended")),
        "liquidation_trading": bool(kr.get("liquidationTrading")),
    }


# ---------------------------------------------------------------------------
# 장 운영 캘린더  ★ 공휴일 문제 해결
# ---------------------------------------------------------------------------

def get_market_calendar(market: str = "KR") -> dict | None:
    """오늘의 실제 장 운영 시간 (프리마켓 / 정규장 / 애프터마켓).

    하드코딩된 시간표와 달리 **공휴일과 임시 휴장을 반영**합니다.
    """
    market = "US" if str(market).upper() == "US" else "KR"
    cached = _calendar_cache.get(market)
    if cached and time.time() - cached[0] < _CALENDAR_TTL:
        return cached[1]

    headers = _auth_headers()
    if not headers:
        return None

    data = http_client.get_json(BASE + f"/api/v1/market-calendar/{market}",
                                headers=headers, timeout=12)
    today = ((data or {}).get("result") or {}).get("today")
    if not today:
        return None

    sessions = today.get("integrated") or today.get("krx") or {}

    def window(key):
        w = sessions.get(key) or {}
        return {"start": w.get("startTime"), "end": w.get("endTime"),
                "auction": w.get("singlePriceAuctionStartTime")} if w else None

    result = {
        "date": today.get("date"),
        "market": market,
        "pre": window("preMarket"),
        "regular": window("regularMarket"),
        "after": window("afterMarket"),
        # 정규장 구간 자체가 없으면 휴장일입니다
        "is_holiday": not (sessions.get("regularMarket") or {}).get("startTime"),
    }
    _calendar_cache[market] = (time.time(), result)
    return result


def get_orderbook(symbol: str) -> dict | None:
    """호가 조회 (매수/매도 잔량 — 단기 수급 참고용)."""
    headers = _auth_headers()
    if not headers:
        return None
    data = http_client.get_json(BASE + "/api/v1/orderbook", headers=headers,
                                params={"symbol": symbol}, timeout=12)
    return (data or {}).get("result")


# ---------------------------------------------------------------------------
# 자체 점검
# ---------------------------------------------------------------------------

def self_check():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  토스증권 Open API 자체 점검")
    print("=" * 60)

    if not is_configured():
        print("\n[미설정] TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 이 없습니다.")
        print("\n  발급: https://developers.tossinvest.com")
        print("  설정: 아테나.bat → [7] API 키, 또는")
        print("        set TOSS_CLIENT_ID=...")
        print("        set TOSS_CLIENT_SECRET=...")
        print("\n  ※ 설정하지 않아도 기존 공개 경로로 정상 동작합니다.\n")
        return

    print("\n[1/5] 액세스 토큰 발급...", end=" ")
    token = _get_token()
    print("성공" if token else "실패 — client_id/secret 을 확인하세요")
    if not token:
        return

    print("[2/5] 종목 정보 (삼성전자)...", end=" ")
    info = get_stock_info("005930")
    print(f"{info['name']} / {info['market']} / {info['status']}" if info else "실패")

    print("[3/5] 현재가...", end=" ")
    p = get_price("005930")
    print(f"{p['price']:,.0f} {p['currency']}" if p else "실패")

    print("[4/5] 일봉 60개...", end=" ")
    df = get_candles("005930", "1d", 60)
    print(f"{len(df)}봉 ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d})" if not df.empty else "실패")

    print("[5/5] 장 운영 캘린더...", end=" ")
    cal = get_market_calendar("KR")
    if cal:
        reg = cal.get("regular") or {}
        print(f"{cal['date']} 정규장 {reg.get('start','?')} ~ {reg.get('end','?')}"
              + (" (휴장)" if cal["is_holiday"] else ""))
    else:
        print("실패")

    print("\n점검 완료 — 서버를 재시작하면 토스 공식 API가 우선 사용됩니다.\n")


if __name__ == "__main__":
    self_check()
