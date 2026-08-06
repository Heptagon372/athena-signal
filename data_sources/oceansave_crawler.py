"""
오션세이브 ('세이브(SAVE)' 앱 - '오선의 미국 증시 라이브' 큐레이션) 속보 크롤러
------------------------------------------------------------------------
'세이브' 앱은 공식 공개 API를 제공하지 않는 별도 서비스입니다. 실제 연동하려면:

1. 앱(iOS/Android) 또는 웹 버전 사용 중 프록시 툴(mitmproxy, Charles 등)로
   네트워크 요청 캡처 - 로그인 세션이 필요할 수 있음
2. 속보 목록을 응답으로 주는 API 확인 (제목, 발행시각, 원문 링크 등)
3. 아래 fetch_raw_items() 를 그 응답 구조에 맞게 구현
4. 반드시 '세이브' 앱 이용약관을 직접 확인 후 개인 학습 목적으로만 사용

이 크롤러는 news_crawler.py 와 동일한 인터페이스(get_news(ticker) -> list[NewsItem])를
따르므로, 완성되면 engine 쪽 코드는 수정 없이 그대로 사용할 수 있습니다.
"""

from datetime import datetime
from models import NewsItem
from data_sources.news_crawler import score_sentiment

SOURCE_NAME = "오션세이브(SAVE)"


def fetch_raw_items(ticker: str) -> list[dict]:
    """
    [직접 구현 필요]
    실제 '세이브' 앱/웹의 속보 API를 호출해
    [{"title": ..., "published_at": datetime, "url": ...}, ...] 형태로 반환하도록 채우세요.
    지금은 파이프라인 연결 확인용으로 빈 리스트를 반환합니다.
    """
    # TODO: requests.get(실제_엔드포인트, headers=...) 로 교체
    return []


def get_news(ticker: str, limit: int = 15) -> list[NewsItem]:
    raw_items = fetch_raw_items(ticker)
    items = []
    for entry in raw_items[:limit]:
        title = entry.get("title", "")
        published = entry.get("published_at", datetime.utcnow())
        items.append(
            NewsItem(
                ticker=ticker,
                title=title,
                source=SOURCE_NAME,
                published_at=published,
                sentiment_score=score_sentiment(title),
            )
        )
    return items
