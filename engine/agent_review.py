"""
AI 에이전트 성적표와 검증 게이트
================================
"에이전트 판단으로 매매하면 돈을 버는가"에 답하는 층입니다.

왜 과거 백테스트로는 답할 수 없는가 (중요)
    LLM 판단을 과거로 되돌려 돌릴 수 없습니다. 에이전트가 보는 증거 팩에는
    뉴스와 커뮤니티 여론이 들어가는데, 크롤러는 **지금** 시점의 글을 줍니다.
    2026년 3월로 되감아 분석하면 그 자리에 2026년 8월의 뉴스가 실립니다.
    그건 백테스트가 아니라 미래를 보고 찍는 것입니다 (look-ahead bias).
    게다가 판단 한 건에 실제 요금이 나가므로, 250일 × 12종목을 되돌리면
    호출 3,000회분 요금이 그대로 청구됩니다.

    그래서 이 저장소의 검증은 두 갈래로 나눕니다.
      룰 층        engine/autotrade.simulate + _run_gates 로 지금까지처럼
                  과거 검증을 합니다 (복권 검정, DSR, CPCV). 에이전트를 끈
                  상태의 성적이 바닥이라면 그 위에 무엇을 얹어도 소용없습니다.
      에이전트 층  **전진 검증**만 인정합니다. 판단을 낼 때 그 시점 가격과
                  채점 만기를 박아 두고, 만기가 지난 뒤 그때의 실제 가격으로만
                  채점합니다. 사후에 이야기를 고쳐 쓸 여지가 없습니다.

게이트가 하는 일
    전진 검증 표본이 기준을 넘기 전까지 에이전트가 주문에 영향을 주지 못하게
    막습니다 (engine/agent_trader.influence_allowed). 판단은 계속 쌓입니다 —
    쌓여야 게이트를 넘습니다. 이 기준은 의도적으로 **약합니다**. 룰 층에 쓰는
    복권 검정이나 DSR 만큼 엄격하게 잡으면 표본이 모이기 전에 영원히 잠깁니다.
    "동전 던지기보다 낫다는 최소한의 증거"를 요구할 뿐, 수익을 보장하지 않습니다.
"""

import math
from datetime import datetime

# 게이트 기준 — 무엇을 요구하는지 한곳에 모아 둡니다.
MIN_SAMPLES = 30            # 방향성 판단(매수·매도) 최소 건수
MIN_T_STAT = 1.5            # 평균 수익률이 0 이라는 가정을 얼마나 밀어내는가
HOLD_BAND_PCT = 2.0         # 보유 판단은 이 폭 안에 머물렀으면 맞은 것으로 봅니다


# ---------------------------------------------------------------------------
# 채점 — 만기가 지난 판단을 그때의 실제 가격으로
# ---------------------------------------------------------------------------

def resolve_due(now: datetime = None, limit: int = 200) -> dict:
    """만기가 지난 판단을 채점합니다. 반복 실행해도 안전합니다.

    engine/backtest.resolve_matured_predictions 와 같은 규칙을 씁니다 —
    가격이 사실상 안 움직였으면 채점하지 않고, 오래 지나도록 거래가 없었으면
    무효 처리해 대기열에서 뺍니다. 무변동을 '상승 적중'으로 세면 적중률이
    통째로 부풀어 오릅니다 (이 저장소에서 실제로 있었던 버그입니다).
    """
    from data_sources import price_provider, symbol_registry
    from engine import backtest
    from models import SymbolNotFoundError
    from storage import agents as store

    now = now or datetime.now()
    rows = store.due_for_scoring(now=now, limit=limit)
    out = {"scored": 0, "pending": 0, "voided": 0, "failed": []}
    if not rows:
        return out

    # 종목별로 묶어 시세를 한 번만 받습니다 (같은 종목 판단 10건에 10번 조회하면
    # 증권사 호출 간격 규칙에 그대로 걸립니다)
    by_symbol: dict[str, list] = {}
    for row in rows:
        by_symbol.setdefault(row["symbol"], []).append(row)

    for key, group in by_symbol.items():
        try:
            symbol = symbol_registry.resolve(key)
            provider = price_provider.get_provider(symbol)
            daily = provider.get_daily_history(symbol, days=60)
        except (SymbolNotFoundError, Exception) as exc:          # noqa: BLE001
            out["failed"].append(f"{key}({type(exc).__name__})")
            continue

        for row in group:
            try:
                target_dt = datetime.fromisoformat(row["target_at"])
            except (ValueError, TypeError):
                store.void(row["id"], "만기 시각을 읽을 수 없음")
                out["voided"] += 1
                continue

            base = row["base_price"]
            final = backtest._price_at(daily, target_dt)
            if base is None or final is None or base <= 0:
                out["pending"] += 1
                continue

            if not backtest.price_moved(base, final):
                # 만기 구간에 거래가 없었던 경우는 시간이 더 지나도 채점되지
                # 않습니다. 유예가 지나면 무효로 정리합니다.
                if (now - target_dt).total_seconds() > backtest.VOID_AFTER_HOURS * 3600:
                    store.void(row["id"], "만기 구간에 거래 없음")
                    out["voided"] += 1
                else:
                    out["pending"] += 1
                continue

            change_pct = (final / base - 1) * 100
            store.resolve(row["id"], final, _correct(row["decision"], change_pct),
                          round(change_pct, 4), now=now)
            out["scored"] += 1
    return out


def _correct(decision: str, change_pct: float) -> bool:
    """판단이 맞았는가.

    보유(hold)는 "크게 움직이지 않는다"는 주장으로 해석합니다. 이 해석이
    없으면 보유 판단은 영원히 채점할 수 없고, 에이전트가 보유만 반복해도
    성적표가 깨끗하게 유지되는 구멍이 생깁니다.
    """
    if decision == "buy":
        return change_pct > 0
    if decision == "sell":
        return change_pct < 0
    return abs(change_pct) <= HOLD_BAND_PCT


def directional_return(decision: str, change_pct: float) -> float | None:
    """그 판단대로 했다면 얻었을 수익률(%). 보유는 대상이 아니라 None."""
    if change_pct is None:
        return None
    if decision == "buy":
        return change_pct
    if decision == "sell":
        return -change_pct
    return None


# ---------------------------------------------------------------------------
# 성적표
# ---------------------------------------------------------------------------

def scorecard(user_id: int, limit: int = 500) -> dict:
    """채점이 끝난 판단으로 성적을 냅니다. 수수료와 세금은 빠져 있습니다.

    빠져 있다는 사실을 화면에 같이 적어야 합니다 — 왕복 비용이 0.2% 안팎이라,
    평균 수익률이 그보다 작으면 종이 위에서만 버는 전략입니다.
    """
    from storage import agents as store

    rows = store.scored_rows(user_id, limit=limit)
    total = len(rows)
    scored = [r for r in rows if r["correct"] is not None]
    hits = sum(1 for r in scored if r["correct"])

    returns = []
    by_decision: dict[str, dict] = {}
    for r in rows:
        bucket = by_decision.setdefault(
            r["decision"], {"n": 0, "hit": 0, "sum": 0.0, "returns": []})
        bucket["n"] += 1
        bucket["hit"] += 1 if r["correct"] else 0
        ret = directional_return(r["decision"], r["change_pct"])
        if ret is not None:
            returns.append(ret)
            bucket["sum"] += ret
            bucket["returns"].append(ret)

    stats = _mean_t(returns)
    equity = []
    running = 0.0
    for r in sorted(rows, key=lambda x: x["id"]):
        ret = directional_return(r["decision"], r["change_pct"])
        if ret is None:
            continue
        running += ret
        equity.append({"id": r["id"], "at": r["resolved_at"],
                       "symbol": r["symbol"], "cum_pct": round(running, 3)})

    return {
        "total": total,
        "scored": len(scored),
        "hits": hits,
        "hit_rate": round(hits / len(scored) * 100, 1) if scored else None,
        "directional": stats["n"],
        "avg_return_pct": stats["mean"],
        "stdev_pct": stats["stdev"],
        "t_stat": stats["t"],
        "sum_return_pct": round(sum(returns), 3) if returns else 0.0,
        "by_decision": {
            key: {"n": b["n"], "hit": b["hit"],
                  "hit_rate": round(b["hit"] / b["n"] * 100, 1) if b["n"] else None,
                  "avg_return_pct": (round(b["sum"] / len(b["returns"]), 3)
                                     if b["returns"] else None)}
            for key, b in by_decision.items()
        },
        "equity": equity,
        "note": "수수료와 세금은 빠져 있습니다. 왕복 비용은 국내 주식 기준 "
                "0.2% 안팎입니다.",
    }


def _mean_t(values: list) -> dict:
    """평균과 t 통계. 표본이 2건 미만이면 통계를 내지 않습니다."""
    n = len(values)
    if n < 2:
        return {"n": n, "mean": round(values[0], 3) if n else None,
                "stdev": None, "t": None}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    stdev = math.sqrt(var)
    t = (mean / (stdev / math.sqrt(n))) if stdev > 0 else None
    return {"n": n, "mean": round(mean, 3), "stdev": round(stdev, 3),
            "t": round(t, 2) if t is not None else None}


# ---------------------------------------------------------------------------
# 게이트
# ---------------------------------------------------------------------------

def gate(user_id: int) -> dict:
    """에이전트가 매매에 영향을 줘도 되는가.

    통과 조건 세 가지를 모두 만족해야 합니다.
      1. 방향성 판단(매수·매도)이 MIN_SAMPLES 건 이상 채점되었다
      2. 그 판단대로 했을 때의 평균 수익률이 0보다 크다
      3. t 통계가 MIN_T_STAT 이상이다 (우연이라 보기 어렵다)

    이 기준은 수익을 보장하지 않습니다. "동전 던지기보다 낫다는 최소한의
    증거"를 요구할 뿐입니다. 룰 층의 복권 검정·DSR 만큼 엄격하지 않다는
    사실을 화면에도 같이 적어야 합니다.
    """
    card = scorecard(user_id)
    n = card["directional"]
    mean = card["avg_return_pct"]
    t = card["t_stat"]

    if n < MIN_SAMPLES:
        return {"passed": False, "stage": "표본 부족",
                "reason": f"채점된 방향성 판단이 {n}건입니다. "
                          f"{MIN_SAMPLES}건을 넘어야 판정할 수 있습니다.",
                "scorecard": card}
    if mean is None or mean <= 0:
        return {"passed": False, "stage": "기대값 미달",
                "reason": f"판단대로 했을 때 평균 수익률이 {mean}% 입니다. "
                          f"0보다 커야 합니다.",
                "scorecard": card}
    if t is None or t < MIN_T_STAT:
        return {"passed": False, "stage": "유의성 미달",
                "reason": f"t 통계가 {t} 입니다. {MIN_T_STAT} 이상이어야 "
                          f"우연이 아니라고 볼 수 있습니다.",
                "scorecard": card}
    return {"passed": True, "stage": "통과",
            "reason": f"방향성 판단 {n}건, 평균 {mean}%, t {t}. "
                      f"수수료와 세금은 빠진 값입니다.",
            "scorecard": card}


# ---------------------------------------------------------------------------
# 룰 층 검증 — 기존 하네스를 그대로 씁니다
# ---------------------------------------------------------------------------

def rule_backtest(query: str, cfg: dict = None, days: int = 250) -> dict:
    """에이전트를 끈 상태의 룰 전략을 과거로 돌립니다.

    engine/autotrade.simulate 를 그대로 부릅니다 — 복권 검정, DSR, CPCV 가
    이미 그 안에 붙어 있습니다. 이 함수의 존재 이유는 화면에서 두 층을 나란히
    보여주기 위해서입니다. 룰 층 성적이 게이트를 통과하지 못하는데 에이전트
    오버레이가 그것을 구해줄 것이라 기대할 근거는 없습니다.
    """
    from engine import autotrade

    base = dict(cfg or {})
    base["agent"] = {"exec_mode": "observe"}     # 과거 재현에 LLM 을 섞지 않습니다
    return autotrade.simulate(query, base, days=days)
