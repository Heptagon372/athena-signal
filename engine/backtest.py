"""
자가검증(Backtest) & 자가학습 루프
--------------------------------
1. 장 마감 후, 그날 저장해둔 예측들을 실제 종가 움직임과 비교해서
   맞았는지(correct) 여부를 기록합니다.
2. 개별 요소(기술적 지표 / 뉴스 / 커뮤니티) 각각이 "혼자 예측했다면" 맞았을
   비율을 계산해서, 더 잘 맞은 요소에 가중치를 조금씩 더 실어줍니다.

주의: 표본이 적을 때(수십 건 이하)는 가중치가 노이즈에 과적합될 수 있으므로
      WEIGHT_LEARNING_RATE 를 낮게 유지하고, 최소 표본 수 이하일 때는
      가중치를 바꾸지 않도록 방어 로직을 넣었습니다.
"""

import sys
from datetime import datetime

from config import INITIAL_WEIGHTS, WEIGHT_LEARNING_RATE
from data_sources.price_provider import get_provider
from storage import db

MIN_SAMPLES_TO_ADJUST = 20

# 만기가 지났는데 그 구간에 거래가 전혀 없었던 예측을 '무효'로 정리하기까지의 유예
# (장 마감 후에 낸 10분 예측 등 — 시간이 더 지나도 채점될 수 없습니다)
VOID_AFTER_HOURS = 6

# 호가 단위보다 작은 가격 차이는 실제 움직임이 아니라 시세 잡음입니다.
#
# 야후는 가격을 float32 로 내려주기 때문에 $304.61 이 304.6099853515625 로 들어옵니다.
# 이 차이(상대오차 5e-8)를 상승으로 읽으면, 아무것도 움직이지 않은 예측이 전부
# '적중'으로 기록되어 적중률이 부풀려집니다. 무변동 편향(final >= base) 을 고쳤을 때와
# 같은 종류의 문제이며, 정확히 같은 규모로 통계를 오염시킵니다.
#
# 1e-6 은 어떤 시장의 호가 단위보다도 작습니다:
#   AAPL $304.61 -> 허용 0.0003 (호가 0.01)
#   삼성전자 239,500원 -> 허용 0.24원 (호가 100원)
FLAT_REL_TOL = 1e-6


def price_moved(base: float | None, final: float | None) -> bool:
    """두 가격이 실제로 다른가 — 부동소수점 잡음은 같은 것으로 봅니다."""
    if not base or final is None:
        return False
    return abs(final - base) > abs(base) * FLAT_REL_TOL


def _price_at(df, when: datetime) -> float | None:
    """해당 시각 이하의 마지막 봉 종가. 없으면 None."""
    if df is None or df.empty:
        return None
    try:
        sliced = df[df.index <= when]
    except TypeError:
        return None
    if sliced.empty:
        return None
    return float(sliced["close"].iloc[-1])


def resolve_matured_predictions(symbol, now: datetime = None):
    """**만기가 지난** 예측만, **각자의 시평선 기간**으로 채점합니다.

    예전 구현의 두 가지 문제를 고친 함수입니다.
      1) 10분 예측을 하루 등락으로 채점했습니다 — 시평선이 6종이 된 뒤로는
         완전히 다른 것을 재는 셈이라 통계가 무의미했습니다.
      2) 방금 낸 '1일 뒤' 예측까지 즉시 채점했습니다 (만기 확인 없음).

    이제 각 예측의 target_at 이 지난 것만 골라, 예측 시점 가격과 만기 시점 가격을
    비교합니다. 하루 미만 시평선은 분봉을, 하루 이상은 일봉을 씁니다.
    """
    now = now or datetime.now()
    matured = db.get_matured_predictions(symbol.key, now)
    if not matured:
        print(f"[backtest] {symbol.name}({symbol.key}): 만기 도래한 예측이 없습니다.")
        return 0

    provider = get_provider(symbol)
    daily = provider.get_daily_history(symbol, days=30)

    # 하루 미만 시평선이 있으면 분봉도 받아둡니다 (여러 날치를 넉넉히)
    needs_intraday = any(
        (p["_target_dt"] - datetime.fromisoformat(p["predicted_at"])).total_seconds() < 86400
        for p in matured
    )
    intraday = provider.get_history(symbol, "minute", count=2000) if needs_intraday else None

    scored = flat = pending = voided = 0
    for pred in matured:
        try:
            made_at = datetime.fromisoformat(pred["predicted_at"])
        except (ValueError, KeyError):
            continue
        target_dt = pred["_target_dt"]
        span_sec = (target_dt - made_at).total_seconds()

        source = intraday if (span_sec < 86400 and intraday is not None
                              and not intraday.empty) else daily

        base = pred.get("base_price") or _price_at(source, made_at)
        final = _price_at(source, target_dt)

        # 만기 시점 가격이 아직 데이터에 없으면(휴장/지연) 다음 기회로 미룹니다
        if base is None or final is None or base <= 0:
            pending += 1
            continue

        # 가격이 사실상 같으면 채점하지 않습니다.
        # 대개 두 시각이 같은 봉에 걸려 새 정보가 없다는 뜻인데, 이때
        # `final >= base` 로 판정하면 전부 '상승 적중'이 되어 통계가 상승 쪽으로
        # 편향됩니다. (실제로 이 버그로 적중률이 부풀려진 적이 있습니다)
        # `final == base` 로 비교하면 float32 잡음 때문에 이 방어가 뚫립니다.
        if not price_moved(base, final):
            # 장 마감 후에 낸 단기 예측처럼 **만기 구간에 거래 자체가 없었던** 경우는
            # 시간이 더 지나도 영원히 채점되지 않습니다. 대기열에 쌓이기만 하므로
            # 유예 시간이 지나면 '무효'로 표시해 정리합니다 (적중률에는 미포함).
            if (now - target_dt).total_seconds() > VOID_AFTER_HOURS * 3600:
                db.void_prediction(pred["id"], "만기 구간에 거래 없음(휴장·장외)")
                voided += 1
            else:
                flat += 1
            continue

        actual = "up" if final > base else "down"
        correct = (pred["direction"] == actual)
        change = (final / base - 1) * 100

        db.resolve_prediction(pred["id"], actual, correct, resolved_price=final)
        scored += 1
        print(f"           #{pred['id']} ({pred['horizon_label']:>5}) "
              f"{base:,.0f} → {final:,.0f} ({change:+.2f}%) "
              f"예측={pred['direction']} 실제={actual} -> {'적중' if correct else '오답'}")

    note = []
    if flat:
        note.append(f"가격 변동 없음 {flat}건 보류")
    if voided:
        note.append(f"거래 없는 구간 {voided}건 무효 처리")
    if pending:
        note.append(f"만기 시세 미확보 {pending}건 보류")
    print(f"[backtest] {symbol.name}({symbol.key}): {scored}/{len(matured)}건 채점"
          + (f" ({', '.join(note)})" if note else ""))
    return scored


# 기존 이름으로 호출하는 코드 호환
def resolve_todays_predictions(symbol):
    return resolve_matured_predictions(symbol)


def adjust_weights():
    """개별 요소별 단독 적중률을 계산해 가중치를 서서히 재조정"""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT technical_score, news_score, community_score, actual_result "
            "FROM predictions WHERE actual_result IS NOT NULL"
        ).fetchall()

    if len(rows) < MIN_SAMPLES_TO_ADJUST:
        print(f"[backtest] 표본 {len(rows)}건 - 최소 {MIN_SAMPLES_TO_ADJUST}건 미만이라 가중치 유지")
        return db.get_latest_weights() or INITIAL_WEIGHTS

    def factor_accuracy(score_key):
        correct, total = 0, 0
        for r in rows:
            score = r[score_key]
            if score is None or score == 0:
                continue
            guess = "up" if score > 0 else "down"
            total += 1
            if guess == r["actual_result"]:
                correct += 1
        return (correct / total) if total else 0.5, total

    acc_tech, n_tech = factor_accuracy("technical_score")
    acc_news, n_news = factor_accuracy("news_score")
    acc_comm, n_comm = factor_accuracy("community_score")

    # "랜덤(50%)보다 얼마나 더 잘 맞았는가"만 신호로 사용 (edge = accuracy - 0.5, 음수는 0으로 clip)
    # 즉 동전던지기 수준인 요소는 target weight가 0에 가까워지고,
    # 확실히 더 잘 맞춘 요소만 가중치를 가져갑니다.
    edge_tech = max(0.0, acc_tech - 0.5)
    edge_news = max(0.0, acc_news - 0.5)
    edge_comm = max(0.0, acc_comm - 0.5)
    total_edge = edge_tech + edge_news + edge_comm

    if total_edge < 1e-6:
        print("[backtest] 모든 요소가 랜덤 수준 적중률 - 가중치 유지")
        return db.get_latest_weights() or INITIAL_WEIGHTS

    target = {
        "technical": edge_tech / total_edge,
        "news_sentiment": edge_news / total_edge,
        "community_sentiment": edge_comm / total_edge,
    }

    current = db.get_latest_weights() or INITIAL_WEIGHTS
    new_weights = {
        k: round(current[k] * (1 - WEIGHT_LEARNING_RATE) + target[k] * WEIGHT_LEARNING_RATE, 4)
        for k in current
    }

    overall_accuracy = db.get_accuracy_stats()["accuracy"]
    db.save_weight_snapshot(new_weights, overall_accuracy)

    print(f"[backtest] 요소별 단독 적중률 - "
          f"기술:{acc_tech:.0%}(n={n_tech}) 뉴스:{acc_news:.0%}(n={n_news}) 커뮤니티:{acc_comm:.0%}(n={n_comm})")
    print(f"[backtest] 새 가중치: {new_weights}")
    return new_weights


def tracked_tickers() -> list[str]:
    """미채점 예측이 있는 종목 목록 — 자동 채점 대상."""
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM predictions WHERE actual_result IS NULL"
        ).fetchall()
    return [r["ticker"] for r in rows]


def auto_resolve_all(verbose: bool = False) -> dict:
    """만기가 지난 예측을 전 종목에 대해 자동 채점합니다.

    서버가 주기적으로 호출하므로, 사용자가 `main.py close` 를 잊어도
    성적표가 저절로 쌓입니다. 만기 도래분만 다루니 반복 실행해도 안전합니다.
    """
    from data_sources import symbol_registry
    from models import SymbolNotFoundError

    scored, failed = 0, []
    for ticker in tracked_tickers():
        try:
            symbol = symbol_registry.resolve(ticker)
        except SymbolNotFoundError:
            failed.append(ticker)
            continue
        try:
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf if not verbose else sys.stdout):
                scored += resolve_matured_predictions(symbol) or 0
        except Exception as exc:      # 한 종목 실패가 전체를 막지 않도록
            failed.append(f"{ticker}({type(exc).__name__})")

    return {"scored": scored, "failed": failed,
            "checked_at": datetime.now().isoformat()}


def run_daily_close_routine(symbols: list):
    """채점 -> 가중치 재조정. symbols 는 ResolvedSymbol 목록입니다.

    만기가 지난 예측만 채점하므로 하루에 여러 번 돌려도 안전합니다.
    """
    total = 0
    for symbol in symbols:
        total += resolve_matured_predictions(symbol) or 0

    new_weights = adjust_weights()
    stats = db.get_accuracy_stats()

    print(f"\n[backtest] 이번 실행에서 {total}건 채점")
    print(f"[backtest] 누적 적중률: {stats['correct']}/{stats['total']} "
          f"({stats['accuracy']}%)" if stats["total"] else "[backtest] 채점된 예측 없음")
    for h in stats.get("by_horizon", []):
        tag = " (구버전)" if h["legacy"] else ""
        print(f"           {h['label']:>8}{tag}: {h['correct']}/{h['total']} ({h['accuracy']}%)")
    return new_weights
