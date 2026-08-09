"""
세이브(SAVE) — saveticker.com 실시간 속보 크롤러
------------------------------------------------
'오선의 미국 증시 라이브'가 운영하는 세이브티커(saveticker.com)의 뉴스 피드입니다.
로이터·Financial Juice 등 해외 터미널급 속보를 한국어로 번역해 올려주므로,
미국 종목의 실시간 뉴스 소스로 국내 RSS보다 훨씬 빠릅니다.

연동 방식 (2026-08 실측):
    GET https://saveticker.com/api/news/list
        ?page=1&page_size=20&tickers=NVDA
        &label_group=<그룹id>&label_name=<이름id>&sources=reuters&search=검색어
    GET https://saveticker.com/api/news/top-stories        (오늘 주요뉴스 큐레이션)
    - 로그인 없이 200 OK (웹 앱의 공개 API). 두 엔드포인트 모두 같은 스키마.
    - 응답: {"news_list": [...], "total_count", "page", "page_size", ...}
    - 항목: id, title(한국어), source(reuters/블룸버그 등), created_at(UTC ISO8601),
            tickers([{symbol, name}]), content_labels([{id, name}]) 등

    카테고리 필터는 **숫자 id** 를 받습니다 (/api/content-labels 에서 실측):
        label_group   1=전체  2=뉴스(편집)  3=로이터  6=속보(실시간 헤드라인)
        label_name    그룹 안의 name_id (뉴스 그룹: 3=속보, 12=실적 등)
    한글 이름을 그대로 넣으면 422가 납니다.

    리포트(/api/report/list)만 로그인(401)이 필요합니다. 세이브 앱 구독자가
    브라우저에서 로그인 세션 쿠키를 복사해 환경변수 SAVETICKER_COOKIE 에 넣어두면
    get_reports() 가 그 쿠키로 시도하고, 없으면 조용히 빈 리스트를 돌려줍니다.

주의:
    - 서버는 **모르는 티커를 조용히 무시**하고 전체 피드를 돌려줍니다.
      (한국 종목코드 "005930" 등도 마찬가지) 그래서 get_news() 는 응답 항목의
      tickers 태그에 요청 심볼이 실제로 있는지 클라이언트에서 재검증합니다.
    - 공개 API라도 과도한 호출은 삼가세요. 이 모듈은 페이지당 1회 GET만 보내고,
      api.py 쪽에서 카테고리별 캐시를 한 겹 더 둡니다.

이 크롤러는 news_crawler.py 와 동일한 인터페이스(get_news -> list[NewsItem])를
따르므로 engine 쪽 코드는 수정 없이 그대로 사용할 수 있습니다.
"""

import os
import re
from datetime import datetime

from models import NewsItem
from data_sources import http_client
from data_sources.news_crawler import score_sentiment_ko, score_sentiment_en

SOURCE_NAME = "세이브(SAVE)"  # reasoning.py 가 "SAVE" 포함 여부로 속보 태그를 판단

SITE = "https://saveticker.com"
BASE_URL = f"{SITE}/api/news/list"
TOP_STORIES_URL = f"{SITE}/api/news/top-stories"
REPORT_URL = f"{SITE}/api/report/list"
REPORT_DETAIL_URL = f"{SITE}/api/report/detail"     # ?id=... (경로 아님)
ARTICLE_URL = f"{SITE}/news/{{id}}"
HEADERS = {"Accept": "application/json", "Referer": "https://saveticker.com/news"}
FEED_TIMEOUT = 8

# label_group 숫자 id (모듈 docstring 참고)
GROUP_NEWS, GROUP_REUTERS, GROUP_BREAKING = 2, 3, 6

# 대시보드/API 에서 쓰는 카테고리 이름 -> 목록 API 파라미터
CATEGORY_PARAMS = {
    "breaking": {"label_group": GROUP_BREAKING},   # 실시간 헤드라인 (financial-juice)
    "reuters":  {"label_group": GROUP_REUTERS},    # 로이터 전문
    "news":     {"label_group": GROUP_NEWS},       # 편집 뉴스 (마감 리포트 포함)
}


def _parse_created_at(raw: str) -> datetime:
    """UTC ISO8601("...Z") -> 로컬(KST) naive datetime.

    이 저장소의 NewsItem.published_at 은 naive 로컬 시각 관례입니다
    (네이버 뉴스가 KST naive). UTC 그대로 두면 최신순 정렬에서 9시간 밀립니다.
    """
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone().replace(tzinfo=None)
    except (ValueError, AttributeError):
        return datetime.now()


def fetch_raw_items(ticker: str | None = None, page: int = 1,
                    page_size: int = 20, label_group: int | None = None,
                    label_name: int | None = None, sources: str | None = None,
                    search: str | None = None) -> list[dict]:
    """세이브티커 뉴스 목록 API 호출. 실패하면 빈 리스트."""
    params: dict = {"page": page, "page_size": page_size}
    if ticker:
        params["tickers"] = ticker
    if label_group is not None:
        params["label_group"] = label_group
    if label_name is not None:
        params["label_name"] = label_name
    if sources:
        params["sources"] = sources
    if search:
        params["search"] = search
    data = http_client.get_json(BASE_URL, params=params, headers=HEADERS,
                                timeout=FEED_TIMEOUT)
    if not isinstance(data, dict):
        return []
    return data.get("news_list") or []


def _to_news_item(entry: dict, ticker: str) -> NewsItem | None:
    title = (entry.get("title") or "").strip()
    if not title or entry.get("is_deleted"):
        return None
    korean = bool(re.search(r"[가-힣]", title))
    score, matched = score_sentiment_ko(title) if korean else score_sentiment_en(title)
    origin = (entry.get("source") or "").strip()  # reuters, financial-juice 등
    entry_id = entry.get("id")
    return NewsItem(
        ticker=ticker,
        title=title,
        source=f"{SOURCE_NAME}·{origin}" if origin else SOURCE_NAME,
        published_at=_parse_created_at(entry.get("created_at") or ""),
        sentiment_score=score,
        url=ARTICLE_URL.format(id=entry_id) if entry_id else "",
        language="ko" if korean else "en",
        matched_keywords=matched,
    )


def get_news(ticker: str, limit: int = 15) -> list[NewsItem]:
    """해당 티커가 태깅된 속보만 반환 (미국 심볼 전용).

    서버가 모르는 티커(한국 종목코드 등)는 전체 피드가 그대로 내려오므로,
    tickers 태그에 요청 심볼이 없는 항목은 버립니다. 결과적으로 태깅이 안 되는
    종목은 빈 리스트가 되고, 호출부는 기존 소스로 자연스럽게 넘어갑니다.
    """
    want = (ticker or "").upper()
    if not want:
        return []
    items = []
    for entry in fetch_raw_items(ticker=want, page_size=max(limit, 20)):
        tagged = {(t.get("symbol") or "").upper() for t in entry.get("tickers") or []}
        if want not in tagged:
            continue
        item = _to_news_item(entry, want)
        if item:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def get_market_feed(limit: int = 20) -> list[NewsItem]:
    """티커 무관 전체 속보 피드 (시장 전반 분위기 파악용)."""
    items = []
    for entry in fetch_raw_items(page_size=max(limit, 20)):
        item = _to_news_item(entry, "MARKET")
        if item:
            items.append(item)
        if len(items) >= limit:
            break
    return items


# ---------------------------------------------------------------------------
# 카테고리 피드 — 아테나 시그널 대시보드의 세이브 패널용
# ---------------------------------------------------------------------------

def _fetch_top_stories() -> list[dict]:
    """오늘 주요뉴스 (세이브 큐레이션). 목록 API와 같은 스키마."""
    data = http_client.get_json(TOP_STORIES_URL, headers=HEADERS,
                                timeout=FEED_TIMEOUT)
    if not isinstance(data, dict):
        return []
    return data.get("news_list") or []


def _auth_headers() -> dict | None:
    """로그인 세션 쿠키(SAVETICKER_COOKIE)가 있으면 인증 헤더, 없으면 None."""
    cookie = os.environ.get("SAVETICKER_COOKIE", "").strip()
    if not cookie:
        return None
    return {**HEADERS, "Cookie": cookie,
            "Referer": "https://saveticker.com/report"}


def _fetch_reports() -> list[dict]:
    """리포트 목록 — 로그인 필요. 쿠키가 없거나 만료(401)면 빈 리스트.

    응답: {"reports": [{id, title, created_at, author_name, has_pdf, view_count,
                        content(빈 문자열), tag_names}], "total_count", ...}
    """
    headers = _auth_headers()
    if headers is None:
        return []
    data = http_client.get_json(REPORT_URL, headers=headers, timeout=FEED_TIMEOUT)
    if not isinstance(data, dict):
        return []
    for key in ("reports", "report_list", "news_list", "items", "results"):
        if isinstance(data.get(key), list):
            return data[key]
    return []


def report_pdf_url(report_id: str) -> str | None:
    """리포트 상세에서 PDF 경로를 얻어 절대 URL로. 없으면 None.

    본문(content)은 목록·상세 모두 비어 있고 실제 내용은 PDF에만 있습니다.
    상세 조회는 리포트당 1회이므로 목록 렌더링 때가 아니라 **열람 시점**에만
    호출합니다 (목록 20건마다 20번 두드리지 않기 위해).
    """
    headers = _auth_headers()
    if headers is None or not report_id:
        return None
    data = http_client.get_json(REPORT_DETAIL_URL, params={"id": report_id},
                                headers=headers, timeout=FEED_TIMEOUT)
    if not isinstance(data, dict):
        return None
    path = (data.get("report") or {}).get("pdf_url")
    if not path:
        return None
    return path if path.startswith("http") else f"{SITE}{path}"


def fetch_report_pdf(report_id: str) -> tuple[bytes, str] | None:
    """리포트 PDF 원본 (bytes, 파일명). 실패하면 None."""
    url = report_pdf_url(report_id)
    headers = _auth_headers()
    if not url or headers is None:
        return None
    res = http_client.get(url, headers={"Cookie": headers["Cookie"]}, timeout=30)
    if res is None or res.status_code != 200:
        return None
    if not res.content.startswith(b"%PDF-"):
        return None
    return res.content, f"SAVE_report_{report_id}.pdf"


def _report_payload(entry: dict) -> dict | None:
    """리포트 항목 -> 대시보드 표시용 dict.

    제목이 날짜뿐("2026년 08월 07일 (금)")이라 감성 점수를 매기지 않습니다
    (0.0 고정). 실제 내용은 PDF 안에 있고, 링크로 열어보게 합니다.
    """
    title = (entry.get("title") or "").strip()
    entry_id = entry.get("id")
    if not title or not entry_id:
        return None
    return {
        "title": f"SAVE 마감 리포트 — {title}",
        "source": (entry.get("author_name") or "오선").strip(),
        "sentiment": 0.0,
        "keywords": [],
        "url": f"/api/save/report/{entry_id}/pdf" if entry.get("has_pdf") else "",
        "published_at": _parse_created_at(entry.get("created_at") or "").isoformat(),
        "labels": [t for t in entry.get("tag_names") or []] or ["리포트"],
        "tickers": [],
        "badge": "블룸버그 정보",
        "view_count": entry.get("view_count"),
    }


def _entry_payload(entry: dict) -> dict | None:
    """목록 항목 -> 대시보드 표시용 dict (감성점수·라벨·티커 포함)."""
    title = (entry.get("title") or "").strip()
    if not title or entry.get("is_deleted"):
        return None
    korean = bool(re.search(r"[가-힣]", title))
    score, matched = score_sentiment_ko(title) if korean else score_sentiment_en(title)
    entry_id = entry.get("id")
    return {
        "title": title,
        "source": (entry.get("source") or "").strip() or "SAVE",
        "sentiment": score,
        "keywords": matched,
        "url": ARTICLE_URL.format(id=entry_id) if entry_id else "",
        "published_at": _parse_created_at(entry.get("created_at") or "").isoformat(),
        "labels": [l.get("name") for l in entry.get("content_labels") or [] if l.get("name")],
        "tickers": [t.get("symbol") for t in entry.get("tickers") or [] if t.get("symbol")],
        "badge": "블룸버그 정보",
    }


def feed_payload(category: str = "top", limit: int = 20) -> list[dict]:
    """카테고리별 피드를 JSON 직렬화 가능한 dict 목록으로.

    category: top(오늘 주요뉴스) | breaking(속보) | reuters(로이터)
              | news(편집 뉴스) | report(리포트, 쿠키 필요)
    """
    if category == "report":
        # 리포트는 스키마가 뉴스와 달라 전용 파서를 씁니다
        return [p for p in (_report_payload(e) for e in _fetch_reports()) if p][:limit]
    if category == "top":
        raw = _fetch_top_stories()
    elif category in CATEGORY_PARAMS:
        raw = fetch_raw_items(page_size=max(limit, 20), **CATEGORY_PARAMS[category])
    else:
        return []
    items = []
    for entry in raw:
        payload = _entry_payload(entry)
        if payload:
            items.append(payload)
        if len(items) >= limit:
            break
    return items
