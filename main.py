"""
아테나 시그널 - 백엔드 엔진 진입점

사용법:
    python main.py predict 삼성전자        # 종목명으로 예측 (코스피/코스닥/미국 자동 판별)
    python main.py predict 005930          # 종목코드로도 가능
    python main.py predict AAPL            # 미국 종목
    python main.py search 에코프로          # 종목 검색 (자동완성 확인용)
    python main.py close 삼성전자 카카오    # 장 마감 후 채점 + 가중치 재조정
    python main.py stats                   # 누적 적중률 확인

존재하지 않는 종목을 넣으면 분석하지 않고 "없는 종목"이라고 알려줍니다.
"""

import sys
from datetime import datetime, timedelta

from config import INITIAL_WEIGHTS
from data_sources import community_crawler, news_crawler, symbol_registry
from data_sources.price_provider import get_provider
from engine import scoring
from models import Prediction, SymbolNotFoundError
from storage import db


def _print_not_found(exc: SymbolNotFoundError):
    print(f"\n❌ {exc.reason}")
    if exc.suggestions:
        print("\n   혹시 이 종목인가요?")
        for s in exc.suggestions:
            from models import MARKET_LABELS
            label = MARKET_LABELS.get(s["market"], s["market"])
            print(f"     · {s['name']} ({s['key']}, {label})")
    print()


def run_prediction(query: str):
    db.init_db()
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        _print_not_found(exc)
        return

    weights = db.get_latest_weights() or INITIAL_WEIGHTS
    provider = get_provider(symbol)

    daily_history = provider.get_daily_history(symbol, days=120)
    technical = scoring.analyze_technical(daily_history)
    snapshot = provider.get_snapshot(symbol)

    news_items = news_crawler.get_news(symbol, limit=20)
    news_score = news_crawler.aggregate_news_score(news_items)
    news_summary = news_crawler.summarize_news(news_items)

    community = community_crawler.get_community_sentiment(symbol)

    unit = "원" if symbol.is_korean else "$"
    print(f"\n=== {symbol.name} ({symbol.key} · {symbol.market_label}) "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
    print(f"실존 확인: {symbol.verified_by}")
    print(f"현재가 {snapshot.current_price:,.0f}{unit}  "
          f"({snapshot.change_rate:+.2f}%)  거래량 {snapshot.volume:,}주")
    print(f"시세 출처: {snapshot.source}")

    flow = snapshot.investor_flow
    if flow.available:
        print(f"[{flow.trade_date} 순매매] 개인 {flow.individual_net:+,.0f}주 / "
              f"외국인 {flow.foreign_net:+,.0f}주 / 기관 {flow.institution_net:+,.0f}주")

    print(f"\n--- 기술적 지표 ({technical.note}) ---")
    for ind in technical.indicators:
        mark = {"bullish": "▲", "bearish": "▼", "neutral": "―"}[ind.verdict]
        print(f" {mark} {ind.label:<22} {ind.value_text:>10}  "
              f"점수 {ind.score:+.2f} × 비중 {ind.weight}")
        print(f"      └ {ind.reason}")

    print(f"\n기술 종합점수: {technical.score:+.3f}")
    print(f"뉴스 감성점수: {news_score:+.3f}  "
          f"(총 {news_summary['total']}건 · 긍정 {news_summary['positive']} / "
          f"부정 {news_summary['negative']} / 중립 {news_summary['neutral']})")
    print(f"커뮤니티 매수비율: {community.bullish_ratio:.0%}  "
          f"(글 {community.post_count}건 · 매수 {community.bullish_count} / 매도 {community.bearish_count})")
    print(f"사용 가중치: {weights}\n")

    # 분봉 기반 단기 점수 (짧은 시평선에 사용)
    intraday_bars = provider.get_history(symbol, "minute", count=240)
    intraday_score = (scoring.analyze_technical(intraday_bars).score
                      if intraday_bars is not None and len(intraday_bars) >= 30 else None)
    if intraday_score is not None:
        print(f"분봉 단기점수: {intraday_score:+.3f}  (분봉 {len(intraday_bars)}개)")

    horizon_preds = scoring.build_horizon_predictions(
        technical.score, intraday_score, news_score,
        community.bullish_ratio, weights, symbol.market,
    )
    print()
    now = datetime.now()
    for hp in horizon_preds:
        pred = Prediction(
            ticker=symbol.key, horizon_label=hp["horizon"], predicted_at=now,
            direction=hp["direction"], probability=hp["probability"],
            technical_score=hp["technical_score"], news_score=news_score,
            community_score=hp["community_score"], weights_used=weights,
            market=symbol.market, name=symbol.name,
        )
        pred_id = db.save_prediction(
            pred, target_at=now + timedelta(minutes=hp["minutes"]),
            base_price=snapshot.current_price or None,
        )
        arrow = "▲ 상승" if pred.direction == "up" else "▼ 하락"
        note = " (다음 개장 후 판정)" if hp["resolves_after_open"] else ""
        print(f"[{hp['label']:>7}] {arrow}  확률 {pred.probability:5.1f}%   "
              f"목표 {hp['target_text']}  분봉비중 {hp['intraday_share']:.0%}{note}")

    print("\n--- 근거 (영향력 순) ---")
    rationale = scoring.build_rationale(technical, news_items, community, weights)
    for item in rationale[:8]:
        tag = {"technical": "기술", "news": "뉴스", "community": "커뮤"}.get(item["source_type"], "기타")
        print(f" [{tag}] {item['influence']:+.4f}  {item['text'][:90]}")

    print("\n※ 이 확률은 검증된 투자 자문이 아니라 실험적 휴리스틱 점수입니다.\n")


def run_search(query: str):
    results = symbol_registry.search(query, limit=10)
    if not results:
        print(f"\n'{query}' 로 찾은 종목이 없습니다.\n")
        return
    from models import MARKET_LABELS
    print(f"\n'{query}' 검색 결과:")
    for r in results:
        label = MARKET_LABELS.get(r["market"], r["market"])
        print(f"  · {r['name']:<20} {r['key']:<10} {label:<6} {r.get('sector', '')}")
    print()


def run_close(queries: list[str]):
    from engine import backtest
    db.init_db()

    symbols = []
    for q in queries:
        try:
            symbols.append(symbol_registry.resolve(q))
        except SymbolNotFoundError as exc:
            _print_not_found(exc)

    if not symbols:
        print("채점할 종목이 없습니다.")
        return
    backtest.run_daily_close_routine(symbols)


def show_stats():
    db.init_db()
    print(db.get_accuracy_stats())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "predict" and len(sys.argv) >= 3:
        run_prediction(" ".join(sys.argv[2:]))
    elif cmd == "search" and len(sys.argv) >= 3:
        run_search(" ".join(sys.argv[2:]))
    elif cmd == "close" and len(sys.argv) >= 3:
        run_close(sys.argv[2:])
    elif cmd == "stats":
        show_stats()
    else:
        print(__doc__)
