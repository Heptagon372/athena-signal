"""
근거 생성기 (Reasoning Generator)
--------------------------------
예측 방향(direction)과 같은 쪽으로 힘을 실은 뉴스/커뮤니티 신호를 골라
사람이 읽을 수 있는 한 줄 근거로 바꿉니다. "왜 하락으로 보는가"에 대한
설명 가능성(explainability)을 위한 부분이며, 그 자체가 추가 예측 신호는 아닙니다.
"""

from models import NewsItem, CommunitySentiment, ReasoningItem


def _source_type(news_item: NewsItem) -> str:
    return "oceansave" if "오션세이브" in news_item.source or "SAVE" in news_item.source else "news"


def generate_reasoning(news_items: list[NewsItem], community: CommunitySentiment,
                        direction: str, max_items: int = 5) -> list[ReasoningItem]:
    reasoning: list[ReasoningItem] = []

    # 방향과 같은 쪽으로 감성이 쏠린 뉴스만 선택 (up->긍정 뉴스, down->부정 뉴스)
    relevant_news = [
        n for n in news_items
        if (direction == "up" and n.sentiment_score > 0.15)
        or (direction == "down" and n.sentiment_score < -0.15)
    ]
    relevant_news.sort(key=lambda n: abs(n.sentiment_score), reverse=True)

    for n in relevant_news[:3]:
        tag = "블룸버그 정보" if _source_type(n) == "oceansave" else n.source
        reasoning.append(ReasoningItem(
            text=f"[{tag}] \"{n.title}\" — {'긍정적' if n.sentiment_score > 0 else '부정적'} 신호로 반영됨",
            source_type=_source_type(n),
            influence=round(n.sentiment_score, 3),
        ))

    # 커뮤니티 여론 근거
    if community and community.post_count > 0:
        bear_ratio = 1 - community.bullish_ratio
        if direction == "down" and bear_ratio > 0.55:
            reasoning.append(ReasoningItem(
                text=f"[토스 커뮤니티] 최근 게시글 중 {bear_ratio:.0%}가 매도(숏) 의견 — 하락 심리 우세",
                source_type="community",
                influence=-round(bear_ratio, 3),
            ))
        elif direction == "up" and community.bullish_ratio > 0.55:
            reasoning.append(ReasoningItem(
                text=f"[토스 커뮤니티] 최근 게시글 중 {community.bullish_ratio:.0%}가 매수(롱) 의견 — 상승 심리 우세",
                source_type="community",
                influence=round(community.bullish_ratio, 3),
            ))
        else:
            reasoning.append(ReasoningItem(
                text=f"[토스 커뮤니티] 매수 {community.bullish_ratio:.0%} / 매도 {bear_ratio:.0%} — 뚜렷한 쏠림 없음",
                source_type="community",
                influence=0.0,
            ))
    else:
        reasoning.append(ReasoningItem(
            text="[토스 커뮤니티] 아직 커뮤니티 데이터가 연결되지 않아 반영되지 않음",
            source_type="community",
            influence=0.0,
        ))

    if not relevant_news:
        reasoning.insert(0, ReasoningItem(
            text=f"뚜렷한 방향성 뉴스가 없어 주로 기술적 지표 흐름에 근거함",
            source_type="technical",
            influence=0.0,
        ))

    return reasoning[:max_items]
