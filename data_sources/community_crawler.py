"""
커뮤니티 여론 크롤러
--------------------
한국 종목은 **네이버 금융 종목토론방** 게시글 제목을 실제로 수집합니다.
(로그인 없이 열람 가능한 공개 게시판이며, 목록 페이지의 제목만 읽습니다.)

토스증권 커뮤니티는 조회에 인증 토큰이 필요해(비로그인 요청은 401) 자동 수집
대상에서 제외했습니다. 토스 세션을 직접 확보한 경우 TOSS_SESSION_COOKIE 환경변수를
채우면 같은 파이프라인에 합류합니다 — 아래 _toss_community() 참고.

여론 점수 = 매수(롱) 성향 글 / (매수 성향 + 매도 성향 글)
게시글마다 어떤 단어로 그렇게 분류했는지 matched_keywords 에 남깁니다.

주의: 종목토론방은 노이즈와 의도적 선동이 많은 채널입니다. 그래서 아테나 공식에서
      커뮤니티 가중치를 가장 낮게(기본 20%) 두었고, backtest 가 실제 적중률을 보고
      이 비중을 자동으로 깎을 수 있게 설계돼 있습니다.
"""

import html
import os
import re
import time
from datetime import datetime

import feedparser
from bs4 import BeautifulSoup

from models import CommunitySentiment, CommunityPost, MARKET_US
from data_sources import http_client, public_apis, scrap_store

# Reddit은 요청 제한이 심해 캐시로 호출을 아낍니다
REDDIT_SUBS = ["stocks", "wallstreetbets", "investing"]
REDDIT_CACHE_TTL = 900          # 15분
_reddit_cache: dict[str, tuple[float, list]] = {}

# 국내 종목의 영문명 (해외 커뮤니티 검색용) — 종목당 한 번만 조회
_english_name_cache: dict[str, str] = {}

# 여론 비율을 100% 신뢰하기 위해 필요한 "방향성이 분류된" 게시글 수
TAGGED_POSTS_FOR_FULL_CONFIDENCE = 10

NAVER_BOARD_URL = "https://finance.naver.com/item/board.naver"
NAVER_BOARD_HEADERS = {"Referer": "https://finance.naver.com/"}
TOSS_COMMUNITY_URL = "https://wts-cert-api.tossinvest.com/api/v3/comments"
STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

# StockTwits 태그가 없는 영어 게시글용 (태그가 있으면 그걸 우선 사용)
EN_BULLISH = ["buy", "long", "bullish", "moon", "calls", "breakout", "rally",
              "undervalued", "accumulate", "hold", "hodl", "dip buy", "load up"]
EN_BEARISH = ["sell", "short", "bearish", "puts", "crash", "dump", "overvalued",
              "bubble", "stop loss", "cut losses", "exit", "collapse"]

# 강도 가중치를 둔 여론 사전
BULLISH_KEYWORDS = {
    "매수": 1.0, "사자": 0.8, "롱": 0.8, "풀매수": 1.0, "물타기": 0.4,
    "존버": 0.7, "홀딩": 0.6, "가즈아": 1.0, "떡상": 1.0, "간다": 0.5,
    "상승": 0.7, "반등": 0.7, "저점": 0.5, "바닥": 0.5, "기회": 0.5,
    "우상향": 0.9, "신고가": 0.8, "익절": 0.3, "수익": 0.5, "호재": 0.9,
    "저평가": 0.7, "탄탄": 0.5, "기대": 0.5, "추매": 0.8, "적립": 0.5,
}
BEARISH_KEYWORDS = {
    "매도": 1.0, "팔자": 0.8, "숏": 0.8, "전량매도": 1.0, "손절": 1.0,
    "탈출": 0.9, "떡락": 1.0, "폭락": 1.0, "하락": 0.7, "급락": 1.0,
    "물렸": 0.8, "물림": 0.8, "나락": 0.9, "지옥": 0.8, "악재": 0.9,
    "고평가": 0.7, "거품": 0.8, "위험": 0.6, "불안": 0.6, "실망": 0.7,
    "쓰레기": 0.8, "사기": 0.9, "먹튀": 0.9, "고점": 0.5, "정리": 0.5,
}


def classify_post(title: str) -> tuple[str, list]:
    """게시글 제목 -> ("bullish"|"bearish"|"neutral", 근거 키워드)"""
    if not title:
        return "neutral", []
    bull = [(w, s) for w, s in BULLISH_KEYWORDS.items() if w in title]
    bear = [(w, s) for w, s in BEARISH_KEYWORDS.items() if w in title]
    bull_score = sum(s for _, s in bull)
    bear_score = sum(s for _, s in bear)

    if bull_score > bear_score:
        return "bullish", [f"+{w}" for w, _ in bull][:4]
    if bear_score > bull_score:
        return "bearish", [f"-{w}" for w, _ in bear][:4]
    return "neutral", [f"+{w}" for w, _ in bull][:2] + [f"-{w}" for w, _ in bear][:2]


def _parse_board_datetime(text: str) -> datetime:
    """'2026.07.27 14:03' -> datetime. 실패하면 현재 시각."""
    text = (text or "").strip()
    for fmt in ("%Y.%m.%d %H:%M", "%Y.%m.%d", "%m.%d %H:%M"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%m.%d %H:%M":
                parsed = parsed.replace(year=datetime.now().year)
            return parsed
        except ValueError:
            continue
    return datetime.now()


def _naver_board(code: str, pages: int = 2, limit: int = 40) -> list[CommunityPost]:
    """네이버 종목토론방 게시글 제목 수집."""
    posts: list[CommunityPost] = []
    for page in range(1, pages + 1):
        html = http_client.get_text(
            NAVER_BOARD_URL,
            params={"code": code, "page": str(page)},
            headers=NAVER_BOARD_HEADERS,
            encoding="utf-8",
        )
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("table.type2 tr")
        for tr in rows:
            link = tr.select_one("td.title a")
            if not link:
                continue
            title = (link.get("title") or link.get_text(strip=True) or "").strip()
            if not title:
                continue
            cells = tr.select("td")
            posted_text = cells[0].get_text(strip=True) if cells else ""
            href = link.get("href", "")
            stance, matched = classify_post(title)
            posts.append(CommunityPost(
                title=title,
                stance=stance,
                posted_at=_parse_board_datetime(posted_text),
                source="네이버 종목토론방",
                url=f"https://finance.naver.com{href}" if href.startswith("/") else href,
                matched_keywords=matched,
            ))
        if len(posts) >= limit:
            break

    return posts[:limit]


def _toss_community(code: str, limit: int = 20) -> list[CommunityPost]:
    """토스증권 커뮤니티 (인증 필요).

    비로그인 요청은 401 이라 기본적으로 건너뜁니다. 본인 계정 세션 쿠키를
    환경변수 TOSS_SESSION_COOKIE 에 넣어두면 여기서 함께 수집합니다.

        set TOSS_SESSION_COOKIE=<브라우저 개발자도구에서 복사한 쿠키 문자열>

    토스증권 이용약관을 직접 확인하고 개인 학습 목적으로만 사용하세요.
    """
    cookie = os.environ.get("TOSS_SESSION_COOKIE")
    if not cookie:
        return []

    data = http_client.get_json(
        TOSS_COMMUNITY_URL,
        params={"subjectId": f"A{code}", "subjectType": "STOCK", "size": str(limit)},
        headers={"Cookie": cookie, "Referer": "https://tossinvest.com/",
                 "Accept": "application/json"},
    )
    if not data:
        return []

    posts = []
    for item in (data.get("result") or {}).get("comments", []) or []:
        message = re.sub(r"\s+", " ", str(item.get("message") or "")).strip()
        if not message:
            continue
        stance, matched = classify_post(message)
        posts.append(CommunityPost(
            title=message[:120], stance=stance, posted_at=datetime.now(),
            source="토스증권 커뮤니티", matched_keywords=matched,
        ))
    return posts


def _stocktwits(symbol, limit: int = 30) -> list[CommunityPost]:
    """StockTwits — 미국 종목 커뮤니티.

    다른 소스와 결정적으로 다른 점: 사용자가 글을 쓸 때 **직접 Bullish/Bearish 태그를
    답니다.** 키워드로 추측할 필요 없이 작성자가 선언한 방향을 그대로 씁니다.
    태그가 없는 글만 키워드 분류로 넘깁니다.
    """
    data = http_client.get_json(
        STOCKTWITS_URL.format(symbol=symbol.key),
        headers={"Accept": "application/json"},
    )
    if not data:
        return []

    posts = []
    for m in (data.get("messages") or [])[:limit]:
        # StockTwits 본문은 &#39; 같은 HTML 엔티티가 그대로 들어옵니다
        body = html.unescape(str(m.get("body") or ""))
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            continue

        tagged = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        if tagged == "Bullish":
            stance, matched = "bullish", ["작성자 Bullish 태그"]
        elif tagged == "Bearish":
            stance, matched = "bearish", ["작성자 Bearish 태그"]
        else:
            stance, matched = classify_post_en(body)

        try:
            posted = datetime.strptime(m.get("created_at", ""), "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            posted = datetime.now()

        posts.append(CommunityPost(
            title=body[:150], stance=stance, posted_at=posted,
            source="StockTwits", matched_keywords=matched,
            url=f"https://stocktwits.com/message/{m.get('id')}" if m.get("id") else "",
        ))
    return posts


def classify_post_en(text: str) -> tuple[str, list]:
    """영어 게시글 분류 (StockTwits 태그가 없는 글용)."""
    lowered = (text or "").lower()
    bull = [w for w in EN_BULLISH if w in lowered]
    bear = [w for w in EN_BEARISH if w in lowered]
    if len(bull) > len(bear):
        return "bullish", [f"+{w}" for w in bull][:4]
    if len(bear) > len(bull):
        return "bearish", [f"-{w}" for w in bear][:4]
    return "neutral", []


def _reddit(symbol, limit: int = 25) -> list[CommunityPost]:
    """Reddit 커뮤니티.

    키가 있으면 공식 OAuth2 API를 씁니다(429 없음). 없으면 RSS 폴백인데,
    `.json`은 403이고 `.rss`도 429가 잦아 실패를 정상 경로로 취급합니다.
    """
    key = symbol.key.upper()
    cached = _reddit_cache.get(key)
    if cached and time.time() - cached[0] < REDDIT_CACHE_TTL:
        return cached[1]

    # --- 공식 API 경로 ---
    if public_apis.reddit_enabled():
        posts = []
        for item in public_apis.reddit_search(key, limit=limit):
            stance, matched = classify_post_en(item["title"])
            # 추천수가 높은 글은 여론을 더 많이 대표한다고 보고 근거에 표시
            if item["score"] >= 50:
                matched = matched + [f"추천 {item['score']}"]
            posts.append(CommunityPost(
                title=item["title"][:200], stance=stance,
                posted_at=item["created"], source=f"Reddit r/{item['subreddit']}",
                url=item["url"], matched_keywords=matched,
            ))
        _reddit_cache[key] = (time.time(), posts)
        return posts

    # --- RSS 폴백 ---
    posts = []
    for sub in REDDIT_SUBS:
        text = http_client.get_text(
            f"https://www.reddit.com/r/{sub}/search.rss",
            params={"q": key, "restrict_sr": "1", "sort": "new", "limit": "15"},
            timeout=12,
        )
        if not text:
            continue
        try:
            feed = feedparser.parse(text)
        except Exception:
            continue
        for entry in feed.entries[:limit]:
            title = html.unescape(str(entry.get("title") or "")).strip()
            if not title:
                continue
            stance, matched = classify_post_en(title)
            published = entry.get("updated_parsed") or entry.get("published_parsed")
            posted = datetime(*published[:6]) if published else datetime.now()
            posts.append(CommunityPost(
                title=title[:200], stance=stance, posted_at=posted,
                source=f"Reddit r/{sub}", url=entry.get("link", ""),
                matched_keywords=matched,
            ))
        if posts:
            break        # 한 곳에서 얻으면 추가 요청을 아껴 429를 피함

    _reddit_cache[key] = (time.time(), posts)
    return posts


def _english_name(symbol) -> str:
    """해외 커뮤니티 검색에 쓸 영문 회사명.

    국내 종목명은 한글이라 Hacker News 같은 영어권 소스에서 검색되지 않습니다.
    Yahoo 심볼 검색이 `005930.KS -> "Samsung Electronics Co., Ltd."` 를 주므로
    이를 한 번 받아 캐시합니다.
    """
    if not symbol.is_korean:
        return symbol.name

    cached = _english_name_cache.get(symbol.key)
    if cached is not None:
        return cached

    data = http_client.get_json(
        "https://query2.finance.yahoo.com/v1/finance/search",
        params={"q": symbol.yahoo_symbol, "quotesCount": "3", "newsCount": "0"},
        timeout=10,
    )
    name = ""
    for q in ((data or {}).get("quotes") or []):
        if q.get("symbol") == symbol.yahoo_symbol:
            name = q.get("longname") or q.get("shortname") or ""
            break
    # "Samsung Electronics Co., Ltd." -> "Samsung Electronics" (검색어로 과하지 않게)
    name = re.split(r",|\bCo\.|\bInc\.|\bCorp", name)[0].strip()
    _english_name_cache[symbol.key] = name
    return name


def _is_relevant(text: str, symbol, english: str) -> bool:
    """이 글이 정말 이 종목 이야기인가.

    Hacker News 검색은 다어절 질의를 OR로 처리해서, "NVIDIA Corporation" 으로 찾으면
    'Corporation' 만 걸린 무관한 글까지 올라옵니다. 제목에 티커나 회사명(첫 단어)이
    실제로 들어 있는 글만 남깁니다.
    """
    lowered = (text or "").lower()
    if symbol.key.lower() in lowered:
        return True
    for candidate in (english, symbol.name):
        head = (candidate or "").split()
        if head and len(head[0]) >= 3 and head[0].lower() in lowered:
            return True
    return False


def _hackernews(symbol, limit: int = 20) -> list[CommunityPost]:
    """Hacker News — 기술·투자 담론.

    Algolia 검색 API가 키 없이 열려 있습니다. 반도체·AI 등 기술주 관련 논의가
    다른 커뮤니티보다 이르게 올라오는 편이라 보조 신호로 씁니다.
    """
    query = _english_name(symbol)
    if not query or len(query) < 2:
        return []

    data = http_client.get_json(
        "https://hn.algolia.com/api/v1/search_by_date",
        params={"query": query, "tags": "story", "hitsPerPage": str(limit)},
        timeout=12,
    )
    posts = []
    for hit in ((data or {}).get("hits") or []):
        title = (hit.get("title") or "").strip()
        if not title:
            continue
        # 제목에 회사명/티커가 실제로 없으면 버립니다 (OR 매칭으로 딸려온 무관한 글)
        if not _is_relevant(title, symbol, query):
            continue
        stance, matched = classify_post_en(title)
        # 관심도가 높은 글은 근거에 표시 (여론 대표성이 크다고 보아)
        points = hit.get("points") or 0
        if points >= 30:
            matched = matched + [f"{points}pt"]
        try:
            posted = datetime.strptime(hit.get("created_at", ""), "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            posted = datetime.now()
        posts.append(CommunityPost(
            title=title[:200], stance=stance, posted_at=posted, source="Hacker News",
            url=f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
            matched_keywords=matched,
        ))
    return posts


def _tradingview(symbol, limit: int = 20) -> list[CommunityPost]:
    """TradingView 아이디어 — 트레이더가 직접 올린 매매 아이디어.

    국내 종목도 `KRX:005930` 형태로 커버됩니다(영문 게시글). 다만 대형주 위주라
    소형주는 결과가 없을 수 있고, 그때는 조용히 빈 목록을 돌려줍니다.
    """
    if symbol.is_korean:
        candidates = [f"KRX:{symbol.key}"]
    else:
        candidates = [f"NASDAQ:{symbol.key}", f"NYSE:{symbol.key}", f"AMEX:{symbol.key}"]

    results = []
    for tv_symbol in candidates:
        data = http_client.get_json(
            "https://www.tradingview.com/api/v1/ideas/",
            params={"symbol": tv_symbol, "sort": "recent"}, timeout=15,
        )
        results = (data or {}).get("results") or []
        if results:
            break

    posts = []
    for idea in results[:limit]:
        title = re.sub(r"\s+", " ", str(idea.get("name") or "")).strip()
        if not title:
            continue
        # 제목만으로 방향이 안 잡히면 본문 앞부분까지 함께 봅니다
        body = re.sub(r"<[^>]+>", " ", str(idea.get("description") or ""))[:200]
        stance, matched = classify_post_en(title)
        if stance == "neutral" and body:
            stance, matched = classify_post_en(f"{title} {body}")

        likes = idea.get("likes_count") or 0
        if likes >= 10:
            matched = matched + [f"좋아요 {likes}"]

        try:
            posted = datetime.fromisoformat(
                str(idea.get("created_at")).replace("Z", "+00:00")).replace(tzinfo=None)
        except (ValueError, TypeError):
            posted = datetime.now()

        posts.append(CommunityPost(
            title=title[:200], stance=stance, posted_at=posted,
            source="TradingView", matched_keywords=matched,
            url=f"https://www.tradingview.com{idea.get('chart_url', '')}",
        ))
    return posts


def _scraped(symbol) -> list[CommunityPost]:
    """확장프로그램이 브라우저에서 긁어 보낸 게시글.

    서버가 직접 못 가는 토스/팍스넷/카카오페이증권 등이 이 경로로 들어옵니다.
    """
    posts = []
    for row in scrap_store.get_recent(symbol.key):
        title = row["title"]
        if symbol.is_korean:
            stance, matched = classify_post(title)
        else:
            stance, matched = classify_post_en(title)

        try:
            collected = datetime.fromisoformat(row["collected_at"])
        except (ValueError, TypeError):
            collected = datetime.now()

        posts.append(CommunityPost(
            title=title[:200], stance=stance, posted_at=collected,
            source=scrap_store.ALLOWED_SOURCES.get(row["source"], row["source"]),
            url=row.get("url") or "", matched_keywords=matched,
        ))
    return posts


def _source_counts(posts: list) -> dict:
    counts = {}
    for p in posts:
        counts[p.source] = counts.get(p.source, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _balanced_sample(posts: list, limit: int) -> list:
    """화면에 보여줄 글을 소스별로 고르게 뽑습니다.

    단순 최신순으로 자르면 글이 자주 올라오는 종목토론방이 목록을 독차지해서,
    TradingView·Hacker News 같은 해외 소스가 화면에서 사라집니다
    (여론 계산에는 반영되는데 눈에는 안 보이는 상태). 그래서 소스별로 한 건씩
    돌아가며 채운 뒤, 남는 자리를 최신순으로 메웁니다.
    """
    by_source: dict[str, list] = {}
    for p in sorted(posts, key=lambda x: x.posted_at, reverse=True):
        by_source.setdefault(p.source, []).append(p)

    picked, seen = [], set()
    while len(picked) < limit:
        added = False
        for source in list(by_source):
            bucket = by_source[source]
            if not bucket:
                continue
            item = bucket.pop(0)
            if id(item) in seen:
                continue
            seen.add(id(item))
            picked.append(item)
            added = True
            if len(picked) >= limit:
                break
        if not added:
            break

    picked.sort(key=lambda x: x.posted_at, reverse=True)
    return picked


def get_community_sentiment(symbol) -> CommunitySentiment:
    """ResolvedSymbol -> 커뮤니티 여론.

    수집 경로
        국내 종목  네이버 종목토론방 + 확장프로그램 스크랩(토스/팍스넷/카카오페이)
        미국 종목  StockTwits(작성자 태그) + Reddit + 스크랩

    수집에 실패하면 중립(0.5)으로 반환하고 없는 데이터를 지어내지 않습니다.
    """
    now = datetime.now()

    if symbol.market == MARKET_US:
        posts = _stocktwits(symbol) + _reddit(symbol)
    else:
        posts = _naver_board(symbol.key) + _toss_community(symbol.key)

    # 해외 커뮤니티 — 국내 종목에도 국제 시각을 더합니다
    # (삼성전자·SK하이닉스처럼 해외에서도 활발히 논의되는 종목에 특히 유효)
    posts += _tradingview(symbol) + _hackernews(symbol)

    # 브라우저 확장이 보내온 스크랩은 시장과 무관하게 합류
    posts += _scraped(symbol)
    posts.sort(key=lambda p: p.posted_at, reverse=True)

    if not posts:
        return CommunitySentiment(
            ticker=symbol.key, collected_at=now, bullish_ratio=0.5, post_count=0,
            sources=[], is_demo=False,
        )

    bullish = [p for p in posts if p.stance == "bullish"]
    bearish = [p for p in posts if p.stance == "bearish"]
    neutral = [p for p in posts if p.stance == "neutral"]

    tagged = len(bullish) + len(bearish)
    raw_ratio = len(bullish) / tagged if tagged else 0.5

    # 표본 확신도 보정: 방향성이 잡힌 글이 TAGGED_POSTS_FOR_FULL_CONFIDENCE 건에
    # 못 미치면 중립(0.5) 쪽으로 끌어당깁니다.
    # 40건 중 3건만 매수, 1건만 매도로 분류됐다면 그건 "여론이 75% 매수"가 아니라
    # 표본이 4건뿐인 것에 가깝습니다. 보정을 안 하면 게시글 몇 건이 확률을 크게 흔듭니다.
    confidence = min(1.0, tagged / TAGGED_POSTS_FOR_FULL_CONFIDENCE)
    ratio = 0.5 + (raw_ratio - 0.5) * confidence

    return CommunitySentiment(
        ticker=symbol.key,
        collected_at=now,
        bullish_ratio=round(ratio, 3),
        raw_bullish_ratio=round(raw_ratio, 3),
        confidence=round(confidence, 3),
        post_count=len(posts),
        bullish_count=len(bullish),
        bearish_count=len(bearish),
        neutral_count=len(neutral),
        sample_titles=[p.title for p in posts[:5]],
        bullish_titles=[p.title for p in bullish[:5]],
        bearish_titles=[p.title for p in bearish[:5]],
        recent_posts=_balanced_sample(posts, 18),
        sources=sorted({p.source for p in posts}),
        source_counts=_source_counts(posts),
        is_demo=False,
    )
