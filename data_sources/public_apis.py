"""
공개 API 클라이언트 모음 (Reddit / 네이버 검색 / 공공데이터포털 / KRX)
--------------------------------------------------------------------
모두 무료 키로 쓸 수 있는 공식 API이며, 키가 없으면 각 함수는 빈 결과를
돌려주고 상위 계층이 기존 공개 경로를 그대로 씁니다.

    reddit_search       Reddit OAuth2 — RSS의 429 제한을 해소
    naver_news_search   네이버 검색 API — 종목 관련 기사 수집량 증가
    datago_stock_price  공공데이터포털 금융위 주식시세 (KRX 공식 일별 시세)
    krx_daily           KRX Data Marketplace

키 발급처는 data_sources/credentials.py 의 PROVIDERS 를 참고하세요.

주의: 실제 발급 키로 응답을 검증하지 못했습니다. 키를 넣은 뒤
      `python -m data_sources.public_apis` 로 자체 점검을 돌려보세요.
"""

import threading
import time
from datetime import datetime, timedelta

from data_sources import credentials, http_client

# ---------------------------------------------------------------------------
# Reddit (OAuth2 client_credentials)
# ---------------------------------------------------------------------------

REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"
REDDIT_UA = "athena-signal/0.5 (personal research dashboard)"

_reddit_lock = threading.Lock()
_reddit_token: str | None = None
_reddit_expires: float = 0.0


def reddit_enabled() -> bool:
    return credentials.is_configured("reddit")


def _reddit_token_get() -> str | None:
    """Reddit은 HTTP Basic(client_id:secret) + client_credentials 로 토큰을 줍니다."""
    global _reddit_token, _reddit_expires

    if not reddit_enabled():
        return None

    with _reddit_lock:
        if _reddit_token and time.time() < _reddit_expires - 60:
            return _reddit_token

        import base64
        cid = credentials.get("REDDIT_CLIENT_ID")
        secret = credentials.get("REDDIT_CLIENT_SECRET")
        basic = base64.b64encode(f"{cid}:{secret}".encode()).decode()

        res = http_client.post_form(
            REDDIT_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            headers={"Authorization": f"Basic {basic}", "User-Agent": REDDIT_UA},
            timeout=15,
        )
        if not res or not res.get("access_token"):
            return None

        _reddit_token = res["access_token"]
        _reddit_expires = time.time() + float(res.get("expires_in") or 3600)
        return _reddit_token


def reddit_search(query: str, subreddits: list[str] | None = None,
                  limit: int = 25) -> list[dict]:
    """서브레딧에서 종목 언급 글 검색. [{title, url, created, score, subreddit}]"""
    token = _reddit_token_get()
    if not token:
        return []

    subs = "+".join(subreddits or ["stocks", "wallstreetbets", "investing", "StockMarket"])
    data = http_client.get_json(
        f"{REDDIT_API}/r/{subs}/search",
        headers={"Authorization": f"Bearer {token}", "User-Agent": REDDIT_UA},
        params={"q": query, "restrict_sr": "true", "sort": "new",
                "limit": str(min(limit, 100)), "t": "week"},
        timeout=15,
    )
    out = []
    for child in ((data or {}).get("data") or {}).get("children", []):
        d = child.get("data") or {}
        title = (d.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "url": "https://www.reddit.com" + (d.get("permalink") or ""),
            "created": datetime.utcfromtimestamp(d.get("created_utc") or 0),
            "score": d.get("score") or 0,
            "subreddit": d.get("subreddit") or "",
            "num_comments": d.get("num_comments") or 0,
        })
    return out


# ---------------------------------------------------------------------------
# 네이버 검색 API (뉴스)
# ---------------------------------------------------------------------------

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"


def naver_enabled() -> bool:
    return credentials.is_configured("naver")


def naver_news_search(query: str, display: int = 30, sort: str = "date") -> list[dict]:
    """네이버 뉴스 검색. sort: date(최신순) | sim(정확도순)"""
    if not naver_enabled():
        return []

    data = http_client.get_json(
        NAVER_NEWS_URL,
        headers={
            "X-Naver-Client-Id": credentials.get("NAVER_CLIENT_ID"),
            "X-Naver-Client-Secret": credentials.get("NAVER_CLIENT_SECRET"),
        },
        params={"query": query, "display": str(min(display, 100)), "sort": sort},
        timeout=15,
    )
    out = []
    for item in ((data or {}).get("items") or []):
        # 네이버는 검색어를 <b> 태그로 감싸 돌려줍니다
        title = (item.get("title") or "").replace("<b>", "").replace("</b>", "")
        if not title:
            continue
        try:
            published = datetime.strptime(item.get("pubDate", ""),
                                          "%a, %d %b %Y %H:%M:%S %z").replace(tzinfo=None)
        except (ValueError, TypeError):
            published = datetime.now()
        out.append({
            "title": title,
            "url": item.get("originallink") or item.get("link") or "",
            "published_at": published,
        })
    return out


# ---------------------------------------------------------------------------
# 공공데이터포털 — 금융위원회 주식시세정보
# ---------------------------------------------------------------------------

DATAGO_URL = ("https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService"
              "/getStockPriceInfo")


def datago_enabled() -> bool:
    return credentials.is_configured("datago")


def datago_stock_price(code: str, days: int = 30) -> list[dict]:
    """KRX 일별 시세 (금융위 공식 배포).

    네이버 경로가 막혔을 때의 공식 대체 경로입니다.
    """
    if not datago_enabled():
        return []

    end = datetime.now()
    start = end - timedelta(days=int(days * 1.8) + 10)
    data = http_client.get_json(
        DATAGO_URL,
        params={
            "serviceKey": credentials.get("DATAGO_SERVICE_KEY"),
            "resultType": "json",
            "numOfRows": str(days),
            "pageNo": "1",
            "beginBasDt": start.strftime("%Y%m%d"),
            "endBasDt": end.strftime("%Y%m%d"),
            "likeSrtnCd": code,
        },
        timeout=20,
    )
    items = (((data or {}).get("response") or {}).get("body") or {}).get("items") or {}
    rows = items.get("item") or []
    if isinstance(rows, dict):
        rows = [rows]

    def num(v):
        try:
            return float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            return None

    out = []
    for r in rows:
        basd = str(r.get("basDt") or "")
        if len(basd) != 8:
            continue
        out.append({
            "date": datetime.strptime(basd, "%Y%m%d"),
            "open": num(r.get("mkp")), "high": num(r.get("hipr")),
            "low": num(r.get("lopr")), "close": num(r.get("clpr")),
            "volume": num(r.get("trqu")), "value": num(r.get("trPrc")),
            "market_cap": num(r.get("mrktTotAmt")),
            "name": r.get("itmsNm"),
        })
    return sorted(out, key=lambda x: x["date"])


# ---------------------------------------------------------------------------
# KRX Data Marketplace
# ---------------------------------------------------------------------------

KRX_BASE = "http://data-dbg.krx.co.kr/svc/apis"


def krx_enabled() -> bool:
    return credentials.is_configured("krx")


def krx_daily(code: str, date: str | None = None) -> list[dict]:
    """KRX 유가증권 일별 시세.

    KRX Data Marketplace 는 서비스별로 경로가 달라, 대표 경로 하나만 구현했습니다.
    발급 시 안내받은 경로가 다르면 KRX_BASE 와 아래 path 를 맞춰 수정하세요.
    """
    if not krx_enabled():
        return []

    target = date or datetime.now().strftime("%Y%m%d")
    data = http_client.get_json(
        f"{KRX_BASE}/sto/stk_bydd_trd",
        headers={"AUTH_KEY": credentials.get("KRX_AUTH_KEY")},
        params={"basDd": target},
        timeout=20,
    )
    rows = (data or {}).get("OutBlock_1") or []
    return [r for r in rows if str(r.get("ISU_SRT_CD", "")).strip() == code]


# ---------------------------------------------------------------------------
# 자체 점검
# ---------------------------------------------------------------------------

def self_check():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("  공개 API 자체 점검")
    print("=" * 60)

    checks = [
        ("Reddit", reddit_enabled, lambda: reddit_search("NVDA", limit=5),
         lambda r: f"{len(r)}건" + (f" | 예: {r[0]['title'][:48]}" if r else "")),
        ("네이버 검색", naver_enabled, lambda: naver_news_search("삼성전자 주가", display=5),
         lambda r: f"{len(r)}건" + (f" | 예: {r[0]['title'][:48]}" if r else "")),
        ("공공데이터포털", datago_enabled, lambda: datago_stock_price("005930", days=5),
         lambda r: f"{len(r)}건" + (f" | 최근 종가 {r[-1]['close']:,.0f}" if r else "")),
        ("KRX Marketplace", krx_enabled, lambda: krx_daily("005930"),
         lambda r: f"{len(r)}건"),
    ]

    for name, enabled, call, fmt in checks:
        if not enabled():
            print(f"\n  [미설정] {name}")
            continue
        print(f"\n  [설정됨] {name} ... ", end="")
        try:
            result = call()
            print(fmt(result) if result else "응답 없음 — 키 또는 파라미터를 확인하세요")
        except Exception as e:
            print(f"오류 {type(e).__name__}: {str(e)[:60]}")

    print("\n  키 설정: 아테나.bat → [7] API 키, 또는 python -m data_sources.credentials\n")


if __name__ == "__main__":
    self_check()
