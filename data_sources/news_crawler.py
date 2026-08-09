"""
뉴스 크롤러 (한국 + 미국)
-------------------------
종목 시장에 따라 소스를 자동으로 고릅니다.

    코스피 / 코스닥
        1) 네이버금융 종목뉴스 API  — 해당 종목에 직접 태깅된 국내 기사
        2) 구글뉴스 RSS (한국어)     — "종목명 주가" 검색 결과로 보강
        -> 한국어 감성사전으로 점수화
    미국
        Yahoo Finance RSS -> 영어 감성사전으로 점수화

감성 점수(-1 ~ +1)는 제목에 등장한 긍정/부정 표현의 가중합입니다.
어떤 단어 때문에 그 점수가 나왔는지 NewsItem.matched_keywords 에 남겨서
대시보드에서 근거로 표시할 수 있게 했습니다.

한계: 사전 기반이라 반어법·복합 문맥은 잡지 못합니다. 더 정확한 분석이 필요하면
score_sentiment_ko() / score_sentiment_en() 만 LLM 호출로 바꾸면 나머지는 그대로 씁니다.
"""

import html
import math
import re
from datetime import datetime
from urllib.parse import quote

import feedparser

from models import NewsItem, MARKET_US
from data_sources import http_client, public_apis

NAVER_NEWS_URL = "https://m.stock.naver.com/api/news/stock/{code}"
NAVER_HEADERS = {"Referer": "https://m.stock.naver.com/", "Accept": "application/json"}
GOOGLE_NEWS_RSS = ("https://news.google.com/rss/search"
                   "?q={query}&hl=ko&gl=KR&ceid=KR:ko")
YAHOO_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"


# ---------------------------------------------------------------------------
# ★ 가격서술 표현 — 감성이 아닙니다 (「특징주」 문제)
# ---------------------------------------------------------------------------
# 국내 매체는 「특징주」 XX 급등…」, 「XX 상한가」, 「XX 강세」 형태의 기사를
# 매일 대량 발행합니다.
#
#   **이건 뉴스가 아니라 이미 일어난 가격 움직임의 서술이고, 수 분~수 시간
#   뒤에 발행됩니다.**
#
# 이걸 감성 점수로 쓰면 두 가지 중 하나가 됩니다.
#   · 동시점 스코어링  → 화려하고 완전히 가짜인 IC. 백테스트 샤프를 만들고
#                        실전에서 잃는 가장 빠른 경로입니다.
#   · 엄격한 지연 적용 → 뉴스 옷을 입은 순수 모멘텀 피처. 기술적 신호와 이중계산.
#
# 어느 쪽도 원하는 게 아니므로 **감성에서 제외** 하고 주목도(attention)와
# 이벤트 플래그로만 보존합니다. 값은 강도이지 방향이 아닙니다.
#
# 검증법: engine/causality.leakage_shift_test() 를 국내 슬리브에 돌리세요.
# 이 표현들이 감성에 들어가 있으면 +1일 시프트 후에도 IC 가 남습니다.
PRICE_MOVE_KO = {
    "급등": 1.0, "상한가": 1.0, "신고가": 0.9, "최고가": 0.8, "돌파": 0.7,
    "강세": 0.7, "반등": 0.6, "상승": 0.6, "오름": 0.5,
    "급락": 1.0, "하한가": 1.0, "폭락": 1.0, "신저가": 0.9, "최저가": 0.8,
    "약세": 0.7, "하락": 0.6, "내림": 0.5, "미끄러": 0.6,
}

# 제목 **앞부분** 에 이게 있으면 그 기사는 통째로 가격서술로 봅니다.
HEADLINE_BLOCKLIST_KO = ("특징주", "이 시각", "장중", "시황", "마감", "개장")

# 반전 동사는 단일 토큰이고 실제로 최고 신호입니다 — 감성이 아니라
# **MWE 이벤트 피처** 로 다룹니다. 가격서술과 달리 이건 사건입니다.
MWE_EVENT_KO = ("흑자전환", "적자전환", "어닝서프라이즈", "어닝쇼크",
                "감자", "유상증자", "무상증자", "상장폐지", "거래정지")

# 이 스위치를 False 로 두면 예전 동작(가격서술을 감성에 포함)으로 즉시 되돌아갑니다.
BLOCK_PRICE_DESCRIPTIVE = True

# ---------------------------------------------------------------------------
# 감성 사전 — 사건과 전망만 남깁니다
# ---------------------------------------------------------------------------
# 값은 표현의 강도(0~1). 국내 증권기사에서 실제로 자주 쓰이는 표현 위주로 구성했습니다.
# **가격 움직임 서술은 위 PRICE_MOVE_KO 로 옮겼습니다.**
KO_POSITIVE = {
    "회복": 0.5,
    "호실적": 0.9, "실적 개선": 0.8, "어닝서프라이즈": 1.0, "흑자": 0.8,
    "흑자전환": 0.9, "사상 최대": 0.9, "최대 실적": 0.9, "수주": 0.7,
    "계약 체결": 0.7, "공급계약": 0.7, "수출": 0.4, "증설": 0.5,
    "목표주가 상향": 1.0, "상향": 0.6, "매수 의견": 0.8, "매수의견": 0.8,
    "비중확대": 0.7, "투자의견 상향": 0.9, "저평가": 0.5,
    "수혜": 0.7, "기대감": 0.5, "성장": 0.5, "확대": 0.4, "개선": 0.5,
    "승인": 0.6, "허가": 0.6, "특허": 0.5, "인수": 0.3, "협력": 0.4,
    "자사주 매입": 0.7, "배당 확대": 0.6, "순매수": 0.5, "외국인 매수": 0.7,
    "신기록": 0.8, "역대 최대": 0.9, "낙관": 0.5, "긍정": 0.5,
}

KO_NEGATIVE = {
    "부진": 0.7,
    "어닝쇼크": 1.0, "적자": 0.8, "적자전환": 0.9, "손실": 0.7, "실적 악화": 0.9,
    "감익": 0.7, "역성장": 0.8, "매출 감소": 0.7,
    "목표주가 하향": 1.0, "하향": 0.6, "매도 의견": 0.8, "매도의견": 0.8,
    "비중축소": 0.7, "투자의견 하향": 0.9, "고평가": 0.5,
    "리콜": 0.8, "소송": 0.7, "제재": 0.8, "과징금": 0.8, "횡령": 1.0,
    "배임": 1.0, "분식회계": 1.0, "상장폐지": 1.0, "관리종목": 0.9,
    "거래정지": 0.9, "감사의견 거절": 1.0, "불성실공시": 0.8,
    "유상증자": 0.6, "전환사채": 0.4, "블록딜": 0.7, "오버행": 0.7,
    "매도": 0.4, "순매도": 0.5, "외국인 매도": 0.7, "차익실현": 0.5,
    "우려": 0.6, "경고": 0.6, "위기": 0.7, "리스크": 0.5, "충격": 0.7,
    "철회": 0.6, "중단": 0.6, "무산": 0.8, "해지": 0.6, "지연": 0.5,
}

# 부정어가 앞뒤에 붙으면 극성을 뒤집습니다 ("우려 해소", "적자 탈출")
KO_NEGATORS = ["아니", "없", "해소", "벗어", "탈출", "불구", "무색"]

EN_POSITIVE = {
    "surge": 1.0, "soar": 1.0, "jump": 0.8, "rally": 0.8, "beat": 0.9,
    "record": 0.8, "upgrade": 0.9, "outperform": 0.8, "strong": 0.6,
    "gain": 0.6, "bullish": 0.8, "breakthrough": 0.8, "profit": 0.6,
    "growth": 0.6, "raise": 0.6, "top": 0.5, "approval": 0.7, "win": 0.6,
    "buyback": 0.7, "dividend hike": 0.7,
}
EN_NEGATIVE = {
    "plunge": 1.0, "crash": 1.0, "tumble": 0.9, "slump": 0.8, "miss": 0.9,
    "downgrade": 0.9, "warning": 0.7, "loss": 0.7, "bearish": 0.8,
    "recall": 0.8, "lawsuit": 0.7, "cut": 0.6, "weak": 0.6, "decline": 0.6,
    "fraud": 1.0, "probe": 0.7, "halt": 0.7, "layoff": 0.7, "bankruptcy": 1.0,
}


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


# RSS 요청 타임아웃 (초)
FEED_TIMEOUT = 8


def _parse_feed(url: str, params: dict | None = None):
    """RSS를 타임아웃과 함께 읽어 파싱합니다.

    feedparser.parse(url) 은 **자체 타임아웃이 없어** 응답이 느린 서버를 만나면
    무한정 매달립니다. 피드를 6개씩 순회하는 지금 구조에서는 이게 곧
    /api/predict 전체가 멈추는 것으로 이어지므로, 반드시 타임아웃이 있는
    http_client 로 먼저 받아온 뒤 문자열을 파싱합니다.
    """
    text = http_client.get_text(url, params=params, timeout=FEED_TIMEOUT)
    if not text:
        return None
    try:
        return feedparser.parse(text)
    except Exception:
        return None


# 영어 어형변화 대응 정규식 캐시
# 단순 부분문자열 매칭은 "surge"로 "Surging"을, "plunge"로 "Plunged"를 놓칩니다.
# 그렇다고 접두 매칭(surg\w*)을 하면 "top"이 "topic"에 걸리므로,
# 어간 + 허용 어미만 정확히 매칭합니다.
_en_pattern_cache: dict[str, re.Pattern] = {}


def _en_pattern(word: str) -> re.Pattern:
    cached = _en_pattern_cache.get(word)
    if cached:
        return cached

    if " " in word:                       # "dividend hike" 같은 구는 그대로
        pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    else:
        stem = word[:-1] if word.endswith("e") else word     # surge -> surg
        pattern = re.compile(r"\b" + re.escape(stem) + r"(?:e|es|ed|ing|s)?\b",
                             re.IGNORECASE)
    _en_pattern_cache[word] = pattern
    return pattern


# 점수 스케일 — 가중합을 이 값으로 나눈 뒤 tanh 에 넣습니다.
# 강한 표현 1개(1.0) -> tanh(0.5)=0.46, 3개(3.0) -> tanh(1.5)=0.91 로 자연스럽게 포화됩니다.
_SENTIMENT_SCALE = 2.0


def _score(text: str, positive: dict, negative: dict,
           negators: list | None = None, use_regex: bool = False) -> tuple[float, list]:
    """사전 매칭 -> (-1~1 점수, 근거 키워드 목록)

    점수는 (긍정합 - 부정합)을 tanh 로 눌러 만듭니다.
    예전처럼 (pos-neg)/(pos+neg) 로 정규화하면 **한쪽만 걸리면 무조건 ±1.0**이 되어
    'Top Pick'(약한 단어 1개)과 'Record Surge Beat Upgrade'(강한 단어 4개)가
    똑같이 만점을 받았습니다.
    """
    if not text:
        return 0.0, []
    lowered = text.lower()
    pos_total = neg_total = 0.0
    matched = []

    def hit(word: str) -> bool:
        if use_regex:
            return bool(_en_pattern(word).search(lowered))
        return word.lower() in lowered

    for word, weight in positive.items():
        if hit(word):
            if _is_negated(lowered, word.lower(), negators):
                neg_total += weight
                matched.append(f"-{word}(부정어 반전)")
            else:
                pos_total += weight
                matched.append(f"+{word}")

    for word, weight in negative.items():
        if hit(word):
            if _is_negated(lowered, word.lower(), negators):
                pos_total += weight
                matched.append(f"+{word}(부정어 반전)")
            else:
                neg_total += weight
                matched.append(f"-{word}")

    if pos_total == neg_total == 0:
        return 0.0, []

    score = math.tanh((pos_total - neg_total) / _SENTIMENT_SCALE)
    return round(score, 3), matched[:6]


def _is_negated(lowered: str, key: str, negators: list | None) -> bool:
    """키워드 바로 뒤 6글자 안에 부정어가 있으면 극성 반전 ('우려 해소')"""
    if not negators:
        return False
    idx = lowered.find(key)
    if idx < 0:
        return False
    tail = lowered[idx + len(key): idx + len(key) + 6]
    return any(n in tail for n in negators)


def is_price_descriptive(title: str) -> bool:
    """제목이 「특징주」류 — 이미 일어난 가격 움직임의 서술인가.

    제목 **앞부분** 만 봅니다. 「특징주」는 관례적으로 맨 앞에 붙고, 본문 중간의
    '급등' 은 전망일 수 있기 때문입니다("수출 급등 전망").
    """
    t = (title or "").strip()
    if not t:
        return False
    head = t[:14]
    if any(pat in head for pat in HEADLINE_BLOCKLIST_KO):
        return True
    # 제목 전체가 사실상 "종목명 + 가격서술" 뿐인 경우
    hits = [w for w in PRICE_MOVE_KO if w in t]
    if not hits:
        return False
    has_event = any(w in t for w in KO_POSITIVE) or any(w in t for w in KO_NEGATIVE)
    return not has_event


def price_move_strength(title: str) -> float:
    """가격서술 표현의 강도(0~1). **부호가 없습니다** — 주목도 용도입니다."""
    t = title or ""
    hits = [v for w, v in PRICE_MOVE_KO.items() if w in t]
    return float(max(hits)) if hits else 0.0


def event_tags(title: str) -> list:
    """MWE 이벤트 태그. 감성과 별개로 운반합니다."""
    t = title or ""
    return [w for w in MWE_EVENT_KO if w in t]


def score_sentiment_ko(title: str) -> tuple[float, list]:
    """한국어 제목의 감성 점수.

    ★ 가격서술 기사는 **감성 0** 을 돌려줍니다 (BLOCK_PRICE_DESCRIPTIVE=True 일 때).
    근거는 PRICE_MOVE_KO 주석에 있습니다. 태그는 남겨서 화면에서 왜 0 인지
    보이게 합니다.
    """
    if BLOCK_PRICE_DESCRIPTIVE and is_price_descriptive(title):
        tags = event_tags(title)
        marks = ["※가격서술 기사 — 감성 제외(주목도로만 반영)"]
        if tags:
            # 「특징주」 기사라도 진짜 사건이 함께 언급되면 그것은 남깁니다
            score, matched = _score(title, KO_POSITIVE, KO_NEGATIVE, KO_NEGATORS)
            return score, (matched or []) + marks + [f"이벤트:{','.join(tags)}"]
        return 0.0, marks
    return _score(title, KO_POSITIVE, KO_NEGATIVE, KO_NEGATORS)


def score_sentiment_en(title: str) -> tuple[float, list]:
    return _score(title, EN_POSITIVE, EN_NEGATIVE, use_regex=True)


def score_sentiment(title: str) -> float:
    """하위호환용 단일 값 반환 (한글이 있으면 한국어 사전 사용)."""
    if re.search(r"[가-힣]", title or ""):
        return score_sentiment_ko(title)[0]
    return score_sentiment_en(title)[0]


# ---------------------------------------------------------------------------
# 수집기
# ---------------------------------------------------------------------------

def _naver_stock_news(symbol, limit: int) -> list[NewsItem]:
    data = http_client.get_json(
        NAVER_NEWS_URL.format(code=symbol.key),
        params={"pageSize": str(limit), "page": "1"},
        headers=NAVER_HEADERS,
    )
    if not isinstance(data, list):
        return []

    items = []
    for group in data:
        for entry in (group.get("items") or []):
            title = _strip_html(entry.get("titleFull") or entry.get("title"))
            if not title:
                continue
            raw_dt = str(entry.get("datetime") or "")
            try:
                published = datetime.strptime(raw_dt, "%Y%m%d%H%M")
            except ValueError:
                published = datetime.now()
            score, matched = score_sentiment_ko(title)
            items.append(NewsItem(
                ticker=symbol.key, title=title,
                source=f"네이버 종목뉴스 · {entry.get('officeName', '')}".strip(" ·"),
                published_at=published, sentiment_score=score,
                url=entry.get("mobileNewsUrl", ""), language="ko",
                matched_keywords=matched,
            ))
    return items[:limit]


def _naver_search_api(symbol, limit: int) -> list[NewsItem]:
    """네이버 검색 API — 공식 키 기반. 종목 전용 피드보다 넓게 훑습니다."""
    items = []
    for row in public_apis.naver_news_search(f"{symbol.name} 주가", display=limit * 2):
        title = _strip_html(row["title"])
        if not title:
            continue
        score, matched = score_sentiment_ko(title)
        items.append(NewsItem(
            ticker=symbol.key, title=title, source="네이버 검색 API",
            published_at=row["published_at"], sentiment_score=score,
            url=row["url"], language="ko", matched_keywords=matched,
        ))
        if len(items) >= limit:
            break
    return items


def _google_news_ko(symbol, limit: int) -> list[NewsItem]:
    query = quote(f"{symbol.name} 주가")
    parsed = _parse_feed(GOOGLE_NEWS_RSS.format(query=query))
    if not parsed:
        return []
    items = []
    for entry in parsed.entries[:limit]:
        title = _strip_html(entry.get("title", ""))
        if not title:
            continue
        # 구글뉴스 제목은 "제목 - 언론사" 형태
        outlet = ""
        if " - " in title:
            title, outlet = title.rsplit(" - ", 1)
        published = entry.get("published_parsed")
        pub_dt = datetime(*published[:6]) if published else datetime.now()
        score, matched = score_sentiment_ko(title)
        items.append(NewsItem(
            ticker=symbol.key, title=title,
            source=f"구글뉴스 · {outlet}".strip(" ·") or "구글뉴스",
            published_at=pub_dt, sentiment_score=score,
            url=entry.get("link", ""), language="ko", matched_keywords=matched,
        ))
    return items


def _yahoo_news_en(symbol, limit: int) -> list[NewsItem]:
    parsed = _parse_feed(YAHOO_RSS.format(ticker=symbol.yahoo_symbol))
    if not parsed:
        return []
    items = []
    for entry in parsed.entries[:limit]:
        title = _strip_html(entry.get("title", ""))
        if not title:
            continue
        published = entry.get("published_parsed")
        pub_dt = datetime(*published[:6]) if published else datetime.now()
        score, matched = score_sentiment_en(title)
        items.append(NewsItem(
            ticker=symbol.key, title=title, source="Yahoo Finance RSS",
            published_at=pub_dt, sentiment_score=score,
            url=entry.get("link", ""), language="en", matched_keywords=matched,
        ))
    return items


# 종목 단위 피드가 아닌 '시장 전체' RSS 소스.
# 개별 종목 API만 쓰면 기사 수가 적어 감성 표본이 부족해지므로, 시장 뉴스에서
# 해당 종목/회사명이 언급된 기사만 골라 보강합니다.
KR_MARKET_FEEDS = [
    ("한국경제", "https://www.hankyung.com/feed/finance"),
    ("매일경제", "https://www.mk.co.kr/rss/50200011/"),
    ("연합뉴스", "https://www.yna.co.kr/rss/economy.xml"),
    ("머니투데이", "https://rss.mt.co.kr/mt_news_stock.xml"),
    ("아시아경제", "https://www.asiae.co.kr/rss/stock.htm"),
]
US_MARKET_FEEDS = [
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml"
             "?partnerId=wrss01&id=20910258"),
    ("Investing.com", "https://www.investing.com/rss/news_25.rss"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
    ("Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?category=Stocks"),
]


def _market_feed_news(symbol, feeds: list, korean: bool, limit: int) -> list[NewsItem]:
    """시장 전체 RSS에서 이 종목이 언급된 기사만 추립니다."""
    name = (symbol.name or "").strip()
    # 회사명 앞 두 어절 + 티커를 매칭 키로 사용
    keys = {symbol.key.lower()}
    if name:
        keys.add(name.lower())
        head = " ".join(name.split()[:2]).lower()
        if len(head) >= 2:
            keys.add(head)

    items = []
    for outlet, url in feeds:
        parsed = _parse_feed(url)
        if not parsed:
            continue
        for entry in parsed.entries[:60]:
            title = _strip_html(entry.get("title", ""))
            if not title:
                continue
            lowered = title.lower()
            if not any(k in lowered for k in keys if len(k) >= 2):
                continue

            published = entry.get("published_parsed")
            pub_dt = datetime(*published[:6]) if published else datetime.now()
            score, matched = (score_sentiment_ko(title) if korean
                              else score_sentiment_en(title))
            items.append(NewsItem(
                ticker=symbol.key, title=title, source=f"{outlet} RSS",
                published_at=pub_dt, sentiment_score=score,
                url=entry.get("link", ""), language="ko" if korean else "en",
                matched_keywords=matched,
            ))
            if len(items) >= limit:
                return items
    return items


def get_news(symbol, limit: int = 20, include_save: bool = True) -> list[NewsItem]:
    """ResolvedSymbol -> 뉴스 목록 (최신순, 중복 제거).

    소스를 여러 곳에서 모으는 이유: 종목 전용 피드만 쓰면 감성이 잡히는 기사가
    몇 건뿐이라 표본 확신도 감쇠에 걸려 뉴스 점수가 거의 0이 됩니다.

    include_save=False 는 자동매매 경로 전용입니다 — 세이브(SAVE) 속보는 아직
    분석 화면에서만 검증 중이라, 주문 판단에는 반영하지 않습니다.
    """
    if symbol.market == MARKET_US:
        items = []
        if include_save:
            # 세이브(SAVE) 속보가 가장 빠르므로 1순위. 태깅이 없는 종목은 빈 리스트가
            # 돌아와 아래 소스들이 그대로 채웁니다. (함수 안 import 는 순환참조 회피)
            from data_sources import oceansave_crawler
            items = oceansave_crawler.get_news(symbol.key, limit)
        if len(items) < limit:
            items += _yahoo_news_en(symbol, limit - len(items))
        if len(items) < limit:
            items += _market_feed_news(symbol, US_MARKET_FEEDS, False, limit - len(items))
    else:
        items = _naver_stock_news(symbol, limit)
        # 네이버 검색 API 키가 있으면 종목명으로 직접 검색해 표본을 늘립니다
        if len(items) < limit:
            items += _naver_search_api(symbol, limit - len(items))
        if len(items) < limit:
            items += _google_news_ko(symbol, limit - len(items))
        if len(items) < limit:
            items += _market_feed_news(symbol, KR_MARKET_FEEDS, True, limit - len(items))

    seen, unique = set(), []
    for item in sorted(items, key=lambda n: n.published_at, reverse=True):
        fingerprint = re.sub(r"\W", "", item.title)[:40]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(item)

    # 관련성 가중 + 인버스 상품 부호 보정
    return apply_symbol_context(unique[:limit], symbol)


SIGNAL_ARTICLES_FOR_FULL_CONFIDENCE = 5

# 종목을 직접 언급하지 않은 기사(섹터 일반론 등)의 가중치
OFF_TOPIC_WEIGHT = 0.4


def _mentions_symbol(item: NewsItem, symbol) -> bool:
    """기사 제목이 이 종목을 직접 언급하는지."""
    title = item.title.lower()
    if symbol.key.lower() in title:
        return True
    name = (symbol.name or "").lower()
    # 회사명 앞 두 어절 정도만 비교 (ETF 정식명은 지나치게 길어 그대로는 안 맞음)
    head = " ".join(name.split()[:2]).strip()
    return bool(head) and head in title


def apply_symbol_context(news_items: list[NewsItem], symbol) -> list[NewsItem]:
    """종목 성격에 맞게 기사 감성을 보정합니다.

    1) 관련성 — 제목에 종목/회사명이 없는 기사는 가중치를 낮춥니다.
       (SOXS 피드에 'Dell Is Up 6%' 같은 무관한 기사가 섞여 만점을 만들던 문제)

    2) 인버스 상품 — 베어/인버스 ETF는 기초자산이 오르면 내립니다.
       따라서 **종목을 직접 언급하지 않은 섹터 뉴스**의 부호를 뒤집습니다.
       종목명이 직접 나온 기사('SOXS Jumps 11%')는 그 상품 자체의 이야기이므로
       뒤집지 않습니다.
    """
    for item in news_items:
        on_topic = _mentions_symbol(item, symbol)
        score = item.sentiment_score

        if not on_topic:
            score *= OFF_TOPIC_WEIGHT
            if item.matched_keywords:
                item.matched_keywords = item.matched_keywords + ["※종목 직접언급 없음"]

        if symbol.is_inverse and not on_topic and score != 0:
            score = -score
            item.matched_keywords = (item.matched_keywords or []) + ["※인버스 상품이라 부호 반전"]

        item.sentiment_score = round(score, 3)
    return news_items


def aggregate_news_score(news_items: list[NewsItem]) -> float:
    """여러 뉴스의 감성 점수를 하나의 -1~1 스코어로 합산.

    두 단계로 계산합니다.
      1) 방향  : 감성이 잡힌 기사만 모아 최신순 가중평균 (최신 기사 가중치 ↑)
      2) 확신도: 감성이 잡힌 기사 수 / 5 (최대 1.0) 를 곱해 감쇠

    2단계가 필요한 이유 — 기사 12건 중 1건만 부정 키워드에 걸렸다면 그것은
    "뉴스 흐름이 부정적"이라기보다 표본이 적은 것에 가깝습니다. 감쇠를 걸지 않으면
    기사 한 건이 -1.0 을 그대로 끌고 와서 확률을 과하게 흔듭니다.
    """
    if not news_items:
        return 0.0
    scored = [n for n in news_items if n.sentiment_score != 0]
    if not scored:
        return 0.0

    ordered = sorted(scored, key=lambda n: n.published_at, reverse=True)
    weighted_sum = total_weight = 0.0
    for rank, item in enumerate(ordered):
        weight = 1.0 / (rank + 1)          # 최신 기사일수록 큰 가중치
        weighted_sum += item.sentiment_score * weight
        total_weight += weight

    direction = weighted_sum / total_weight if total_weight else 0.0
    confidence = min(1.0, len(scored) / SIGNAL_ARTICLES_FOR_FULL_CONFIDENCE)
    return round(direction * confidence, 4)


def news_attention(news_items: list[NewsItem]) -> dict:
    """주목도 — 가격서술 기사를 **버리지 않고** 여기로 보냅니다.

    「특징주」 기사는 수익률 방향에 대해서는 아무것도 말해주지 않지만,
    **2차 모멘트(거래량·변동성)에 대해서는 말해줍니다.** 뉴스가 안정적으로
    예측하는 것은 방향이 아니라 분산입니다 — 이건 모든 연구에서 일관됩니다.

    그래서 용도가 다릅니다.
        sentiment  → 방향 판단 (사건 기사만)
        attention  → **사이징 축소** 판단 (전체 기사량)

    주목도가 평소보다 높으면 무조건부 변동성으로 산정한 포지션이 체계적으로
    과대합니다. 수익률 예측력이 전혀 없어도 리스크 게이트로는 정당합니다.
    """
    if not news_items:
        return {"total": 0, "price_descriptive": 0, "attention_score": 0.0}

    desc = [n for n in news_items if is_price_descriptive(n.title or "")]
    strengths = [price_move_strength(n.title or "") for n in news_items]
    tags: list = []
    for n in news_items:
        tags.extend(event_tags(n.title or ""))

    return {
        "total": len(news_items),
        "price_descriptive": len(desc),
        "price_descriptive_share": round(len(desc) / len(news_items), 3),
        # 가격서술의 강도 평균 — 부호 없음. 클수록 "지금 움직이고 있다"
        "attention_score": round(float(sum(strengths) / len(news_items)), 4),
        "max_move_strength": round(float(max(strengths)) if strengths else 0.0, 3),
        "event_tags": sorted(set(tags)),
        "note": ("가격서술 기사는 감성에서 제외되고 여기에만 잡힙니다. "
                 "방향이 아니라 사이징에 쓰세요."),
    }


def summarize_news(news_items: list[NewsItem]) -> dict:
    """대시보드 표시용 집계 (긍정/부정/중립 건수)."""
    positive = sum(1 for n in news_items if n.sentiment_score > 0.1)
    negative = sum(1 for n in news_items if n.sentiment_score < -0.1)
    out = {
        "total": len(news_items),
        "positive": positive,
        "negative": negative,
        "neutral": len(news_items) - positive - negative,
        "sources": sorted({n.source.split(" · ")[0] for n in news_items}),
    }
    out.update({"attention": news_attention(news_items)})
    return out
