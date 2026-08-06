"""
아테나 스코어 공식 (Athena Score Formula)
----------------------------------------
세 가지 요소를 -1.0(강한 하락) ~ +1.0(강한 상승) 스코어로 각각 계산한 뒤,
가중합 -> 시그모이드 변환으로 0~100% 확률을 만듭니다.

    raw_score = w_tech * 기술점수      (engine/indicators.py 의 11개 지표 가중평균)
              + w_news * 뉴스점수      (국내/해외 기사 감성 가중평균 × 표본 확신도)
              + w_comm * 커뮤니티점수  (종목토론방 매수·매도 여론 비율)

    probability(%) = 100 / (1 + e^(-SIGMOID_SENSITIVITY * raw_score))

이 공식은 검증된 금융 예측 모델이 아니라 "실험적 휴리스틱 스코어"입니다.
실제 가치는 매일 예측을 기록하고 결과와 비교해 가중치를 조정해나가는
backtest.py 의 자가검증 루프에 있습니다. 이 자체를 투자 판단의
유일한 근거로 삼지 마세요.
"""

import math

import pandas as pd

from config import OPEN_PREDICTION, PREDICTION_HORIZONS, SIGMOID_SENSITIVITY
from engine import indicators
from models import TechnicalAnalysis


def analyze_technical(daily_df: pd.DataFrame) -> TechnicalAnalysis:
    """일봉 -> 지표별 근거가 붙은 기술 분석 결과."""
    return indicators.analyze(daily_df)


def compute_technical_score(daily_df: pd.DataFrame) -> float:
    """종합 기술점수만 필요할 때 쓰는 축약형 (-1 ~ 1)."""
    return analyze_technical(daily_df).score


def combine_scores(technical_score: float, news_score: float,
                   community_bullish_ratio: float, weights: dict) -> dict:
    """세 점수를 가중합하여 최종 확률(%)과 방향을 산출"""
    # 커뮤니티는 0~1 비율이므로 -1~1로 변환
    community_score = (community_bullish_ratio - 0.5) * 2

    raw_score = (
        weights["technical"] * technical_score
        + weights["news_sentiment"] * news_score
        + weights["community_sentiment"] * community_score
    )

    probability_up = 100 / (1 + math.exp(-SIGMOID_SENSITIVITY * raw_score))
    direction = "up" if probability_up >= 50 else "down"
    # 최종 확률은 "예측한 방향"이 맞을 확률로 표시 (up이면 그대로, down이면 100-p)
    probability = probability_up if direction == "up" else 100 - probability_up

    return {
        "direction": direction,
        "probability": round(probability, 1),
        "probability_up": round(probability_up, 1),
        "raw_score": round(raw_score, 4),
        "technical_score": technical_score,
        "news_score": news_score,
        "community_score": round(community_score, 4),
        "contributions": {
            "technical": round(weights["technical"] * technical_score, 4),
            "news_sentiment": round(weights["news_sentiment"] * news_score, 4),
            "community_sentiment": round(weights["community_sentiment"] * community_score, 4),
        },
    }


def combine_for_horizon(horizon: dict, daily_score: float, intraday_score: float | None,
                        news_score: float, community_bullish_ratio: float,
                        weights: dict) -> dict:
    """시평선 하나에 대한 확률.

    시평선마다 다른 값이 나오는 이유는 두 가지입니다.
      1) 기술점수 배합 — 짧은 시평선일수록 분봉 기반 점수를, 긴 시평선일수록
         일봉 기반 점수를 더 많이 씁니다 (config.PREDICTION_HORIZONS.intraday).
      2) 신뢰도 감쇠 — 그 시평선에서 신호를 얼마나 믿을지(conf)를 곱해,
         믿기 어려운 구간일수록 확률이 50%에 가까워집니다.

    분봉 데이터를 못 받으면 intraday 배합 없이 일봉 점수만 씁니다.
    """
    share = horizon["intraday"] if intraday_score is not None else 0.0
    tech = (1 - share) * daily_score + share * (intraday_score or 0.0)

    community_score = (community_bullish_ratio - 0.5) * 2
    raw = (weights["technical"] * tech
           + weights["news_sentiment"] * news_score
           + weights["community_sentiment"] * community_score)

    raw *= horizon["conf"]        # 신뢰도가 낮을수록 50%로 수렴

    probability_up = 100 / (1 + math.exp(-SIGMOID_SENSITIVITY * raw))
    direction = "up" if probability_up >= 50 else "down"
    probability = probability_up if direction == "up" else 100 - probability_up

    return {
        "horizon": horizon["key"],
        "label": horizon["label"],
        "minutes": horizon["minutes"],
        "direction": direction,
        "probability": round(probability, 1),
        "probability_up": round(probability_up, 1),
        "raw_score": round(raw, 4),
        "technical_blended": round(tech, 4),
        "intraday_share": share,
        "confidence": horizon["conf"],
        "technical_score": tech,
        "news_score": news_score,
        "community_score": round(community_score, 4),
    }


def build_horizon_predictions(daily_score: float, intraday_score: float | None,
                              news_score: float, community_bullish_ratio: float,
                              weights: dict, market: str = "") -> list[dict]:
    """설정된 모든 시평선에 대한 예측 + 목표 시각."""
    from datetime import datetime, timedelta
    from data_sources import market_clock

    now = market_clock.now_kst()
    status = market_clock.status_for(market) if market else {"is_open": False}

    out = []
    for h in PREDICTION_HORIZONS:
        result = combine_for_horizon(h, daily_score, intraday_score,
                                     news_score, community_bullish_ratio, weights)
        target = now + timedelta(minutes=h["minutes"])
        result["target_at"] = target.isoformat()
        result["target_text"] = target.strftime("%m/%d %H:%M")
        # 장이 닫혀 있으면 짧은 시평선은 다음 개장 이후에야 의미가 생깁니다
        result["resolves_after_open"] = (not status.get("is_open")) and h["minutes"] <= 720
        out.append(result)
    return out


def build_open_prediction(daily_score: float, news_score: float,
                          community_bullish_ratio: float, weights: dict,
                          market: str = "") -> dict:
    """장 시작 전 예측 — 다음 정규장 개장 시점의 방향.

    시평선 예측(10분~1일)과 별도로 계산하는 이유는 시가 갭의 동인이 다르기 때문입니다.
      · 밤사이 뉴스가 갭을 주로 만듭니다        -> 뉴스 기여도를 1.35배
      · 장이 닫힌 동안 분봉 신호는 무의미합니다  -> 분봉 점수를 쓰지 않음
      · 일봉 기술신호는 갭보다 세션 중 흐름을 설명 -> 0.85배로 낮춤

    반환값에 각 항의 실제 수치를 담아, 화면에서 공식을 그대로 펼쳐 보일 수 있게 했습니다.
    """
    from data_sources import market_clock

    community_score = (community_bullish_ratio - 0.5) * 2

    tech_term = weights["technical"] * daily_score * OPEN_PREDICTION["technical_damping"]
    news_term = weights["news_sentiment"] * news_score * OPEN_PREDICTION["news_amplifier"]
    comm_term = weights["community_sentiment"] * community_score * OPEN_PREDICTION["community_damping"]

    raw = tech_term + news_term + comm_term
    probability_up = 100 / (1 + math.exp(-SIGMOID_SENSITIVITY * raw))
    direction = "up" if probability_up >= 50 else "down"
    probability = probability_up if direction == "up" else 100 - probability_up

    status = market_clock.status_for(market) if market else {}
    target = market_clock.next_regular_open(market) if market else None
    already_open = status.get("session") == "REGULAR"

    return {
        "direction": direction,
        "probability": round(probability, 1),
        "probability_up": round(probability_up, 1),
        "raw_score": round(raw, 4),
        "already_open": already_open,
        "target_at": target.isoformat() if target else None,
        "target_text": target.strftime("%m/%d %H:%M") if target else None,
        "session_label": status.get("label", ""),
        # 화면에서 공식을 그대로 보여주기 위한 항목별 수치
        "terms": {
            "technical": {
                "weight": round(weights["technical"], 4),
                "score": round(daily_score, 4),
                "multiplier": OPEN_PREDICTION["technical_damping"],
                "contribution": round(tech_term, 4),
            },
            "news": {
                "weight": round(weights["news_sentiment"], 4),
                "score": round(news_score, 4),
                "multiplier": OPEN_PREDICTION["news_amplifier"],
                "contribution": round(news_term, 4),
            },
            "community": {
                "weight": round(weights["community_sentiment"], 4),
                "score": round(community_score, 4),
                "multiplier": OPEN_PREDICTION["community_damping"],
                "contribution": round(comm_term, 4),
            },
        },
        "sigmoid_k": SIGMOID_SENSITIVITY,
    }


def build_rationale(technical: TechnicalAnalysis, news_items: list,
                    community, weights: dict) -> list[dict]:
    """
    "왜 오르는지/내리는지"에 대한 사람이 읽을 수 있는 근거 목록.

    각 항목은 {text, source_type, influence, detail} 이며 influence 의 절댓값이
    클수록 최종 확률에 크게 기여한 요소입니다.

    기술적 근거의 influence 계산
        전체 기술 가중치(w_tech)를 지표별 기여도 비율로 나눠 배분합니다.
            지표 influence = w_tech × (지표점수 × 지표가중치) / Σ지표가중치
        이렇게 하면 화면에 표시된 기술 근거들의 influence 합이
        combine_scores 의 technical 기여도와 정확히 일치합니다.
    """
    items = []

    # --- 기술적 지표: 방향성이 있는(중립이 아닌) 지표만 근거로 노출 ---
    total_indicator_weight = sum(i.weight for i in technical.indicators) or 1.0
    for ind in technical.indicators:
        if ind.weight == 0 or abs(ind.score) < 0.05:
            continue
        influence = weights["technical"] * ind.score * ind.weight / total_indicator_weight
        items.append({
            "text": ind.reason,
            "source_type": "technical",
            "source_name": f"{ind.label} = {ind.value_text}",
            "influence": round(influence, 4),
            "detail": {
                "key": ind.key,
                "score": ind.score,
                "weight": ind.weight,
                "formula": ind.formula,
                "values": ind.values,
            },
        })

    if not technical.sufficient_data:
        items.append({
            "text": technical.note,
            "source_type": "technical",
            "source_name": "데이터 부족",
            "influence": 0.0,
        })

    # --- 뉴스: 감성이 뚜렷한 상위 기사 ---
    scored_news = sorted(
        [n for n in news_items if abs(n.sentiment_score) >= 0.2],
        key=lambda n: (abs(n.sentiment_score), n.published_at), reverse=True,
    )
    per_article = weights["news_sentiment"] / max(len(scored_news), 1)
    for n in scored_news[:4]:
        keywords = ", ".join(n.matched_keywords) if n.matched_keywords else "감성 키워드 없음"
        items.append({
            "text": n.title,
            "source_type": "news",
            "source_name": f"{n.source} · 감성 {n.sentiment_score:+.2f} ({keywords})",
            "influence": round(per_article * n.sentiment_score, 4),
            "url": n.url,
        })

    # --- 커뮤니티: 다수 의견 쪽 대표 게시글 ---
    if community and community.post_count > 0:
        comm_score = (community.bullish_ratio - 0.5) * 2
        tagged = community.bullish_count + community.bearish_count
        damped = ""
        if community.confidence < 1.0:
            damped = (f" 표본이 10건에 못 미쳐 확신도 {community.confidence:.0%}를 적용해 "
                      f"원본 {community.raw_bullish_ratio:.0%} → 반영값 {community.bullish_ratio:.0%}로 "
                      f"중립 쪽으로 보정했습니다.")
        items.append({
            "text": (f"종목토론방 최근 글 {community.post_count}건 중 방향성이 있는 "
                     f"{tagged}건을 분류한 결과 매수 의견 {community.bullish_count}건 / "
                     f"매도 의견 {community.bearish_count}건입니다.{damped}"),
            "source_type": "community",
            "source_name": " · ".join(community.sources) or "커뮤니티",
            "influence": round(weights["community_sentiment"] * comm_score, 4),
        })

        rep_titles = community.bullish_titles if comm_score > 0 else community.bearish_titles
        for title in rep_titles[:2]:
            items.append({
                "text": title,
                "source_type": "community",
                "source_name": "대표 게시글",
                "influence": 0.0,
            })
    else:
        items.append({
            "text": "커뮤니티에서 수집된 게시글이 없어 여론 점수는 중립(50%)으로 처리했습니다.",
            "source_type": "community",
            "source_name": "수집 결과 없음",
            "influence": 0.0,
        })

    items.sort(key=lambda x: abs(x["influence"]), reverse=True)
    return items
