# -*- coding: utf-8 -*-
"""
AI 에이전트 분석, 집행 오버레이, 전진 검증 게이트
--------------------------------------------------
네트워크도 API 키도 없이 돕니다. LLM 호출부(engine/agents._ask) 하나만
가짜로 바꾸면 파이프라인 전체가 그대로 돌아가도록 설계했기 때문입니다.

    engine/agents          멀티에이전트 파이프라인, 요금 계산, 설정 클램프
    engine/agent_trader    신호 오버레이, 매도 거부권, 예산 상한
    engine/agent_review    채점, 성적표, 검증 게이트
    storage/agents         분석 저장, 채점 대기열

    python tests/test_agents.py
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from engine import agent_review, agent_trader, agents, strategy
from storage import agents as store

results = []


def check(name: str, passed: bool, detail: str = ""):
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


# ---------------------------------------------------------------------------
# 임시 DB — 사용자의 athena.db 를 절대 건드리지 않습니다
# ---------------------------------------------------------------------------
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
store.DB_PATH = _tmp.name
store.init()

USER = 4242


# ---------------------------------------------------------------------------
# 가짜 LLM — 호출부 하나만 바꿉니다
# ---------------------------------------------------------------------------
class _FakeStub:
    """_client() 자리에 들어갑니다. _ask 를 갈아끼우므로 쓰이지 않습니다."""


def _fake_ask(script):
    """script(prompt) -> dict 를 돌려주는 _ask 대체품. 사용량은 고정값."""
    calls = []

    def _ask(client, model, system, prompt, schema, effort):
        calls.append({"prompt": prompt, "effort": effort, "system": system})
        usage = {"input_tokens": 1000, "output_tokens": 400,
                 "cache_write_tokens": 0, "cache_read_tokens": 2000}
        return script(prompt, schema), usage
    _ask.calls = calls
    return _ask


def _script(prompt, schema):
    """스키마를 보고 그 모양에 맞는 답을 지어냅니다."""
    props = set(schema.get("properties", {}))
    if "decision" in props:
        return {"decision": "buy", "confidence": 0.72, "horizon": "2주에서 1개월",
                "summary": "종합하면 매수", "reasons": ["추세 양호"],
                "risks": ["변동성"], "invalidation": "60일선 이탈"}
    if "winner" in props:
        return {"winner": "bull", "stance": "buy", "confidence": 0.66,
                "summary": "강세 우세", "plan": ["분할 진입"]}
    if "stance" in props:
        return {"stance": "bullish", "confidence": 0.6, "summary": "긍정적",
                "points": ["추세"], "watch": ["거래량 감소"]}
    return {"summary": "주장", "points": ["근거"], "counter": ["반론"],
            "confidence": 0.55}


def _fake_evidence(query, cfg):
    return {"symbol": "005930", "name": "삼성전자", "market": "KR",
            "collected_at": datetime.now().isoformat(timespec="seconds"),
            "missing": [], "quote": {"price": 70000, "currency": "KRW",
                                     "change_pct": 1.2, "source": "test"},
            "technical": {"score": 0.31, "bars_used": 120, "sufficient_data": True,
                          "note": "", "regime": {"label": "추세"},
                          "indicators": [{"label": "RSI", "value": "58",
                                          "score": 0.2, "verdict": "중립",
                                          "reason": "", "family": "meanrev"}]}}


# ---------------------------------------------------------------------------
# 설정 클램프
# ---------------------------------------------------------------------------

def test_clamp():
    cfg = agents.clamp_config({"model": "gpt-없는-모델", "effort": "무한",
                               "debate_rounds": 99, "max_news": -5,
                               "알수없는키": 1})
    check("모르는 모델은 기본값으로", cfg["model"] == agents.DEFAULT_MODEL, cfg["model"])
    check("모르는 노력 수준은 기본값으로", cfg["effort"] == "medium")
    check("토론 라운드 상한 3", cfg["debate_rounds"] == 3)
    check("음수 뉴스 수는 0", cfg["max_news"] == 0)
    check("모르는 키는 버립니다", "알수없는키" not in cfg)

    ex = agent_trader.clamp_config({"exec_mode": "무엇이든", "overlay_weight": 9,
                                    "daily_cost_cap_usd": -3})
    check("모르는 집행 모드는 관찰로", ex["exec_mode"] == agent_trader.OBSERVE)
    check("오버레이 무게 상한 1.0", ex["overlay_weight"] == 1.0)
    check("요금 상한은 음수가 될 수 없음", ex["daily_cost_cap_usd"] == 0.0)


def test_cost():
    """캐시 읽기는 1/10, 캐시 쓰기는 1.25배로 계산됩니다."""
    usd = agents.cost_of("claude-opus-5", {"input_tokens": 1_000_000})
    check("입력 1M = 5달러", abs(usd - 5.0) < 1e-9, f"{usd}")
    usd = agents.cost_of("claude-opus-5", {"cache_read_tokens": 1_000_000})
    check("캐시 읽기 1M = 0.5달러", abs(usd - 0.5) < 1e-9, f"{usd}")
    usd = agents.cost_of("claude-opus-5", {"cache_write_tokens": 1_000_000})
    check("캐시 쓰기 1M = 6.25달러", abs(usd - 6.25) < 1e-9, f"{usd}")
    usd = agents.cost_of("claude-opus-5", {"output_tokens": 1_000_000})
    check("출력 1M = 25달러", abs(usd - 25.0) < 1e-9, f"{usd}")
    check("모르는 모델은 비용을 지어내지 않음",
          agents.cost_of("없는모델", {"output_tokens": 999999}) == 0.0)


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def test_pipeline():
    orig_ask, orig_client, orig_ev = agents._ask, agents._client, agents.collect_evidence
    fake = _fake_ask(_script)
    agents._ask, agents._client = fake, lambda: _FakeStub()
    agents.collect_evidence = _fake_evidence
    try:
        seen = []
        out = agents.run_analysis("삼성전자", {"debate_rounds": 1},
                                  progress=lambda d, t, l: seen.append((d, t, l)))
    finally:
        agents._ask, agents._client = orig_ask, orig_client
        agents.collect_evidence = orig_ev

    stages = out["report"]["stages"]
    # 애널리스트 4 + 토론 2 + 리서치 1 + 리스크 3 + 최종 1 = 11
    check("호출 11회 (토론 1라운드)", len(stages) == 11, f"{len(stages)}회")
    check("진행률 보고가 단계마다 옴", len(seen) == 11, f"{len(seen)}회")
    check("총 단계 수가 진행률과 일치", seen[-1][1] == 11 if seen else False)

    groups = [s["group"] for s in stages]
    check("애널리스트 4명", groups.count("analyst") == 4)
    check("토론 2회", groups.count("debate") == 2)
    check("리스크 심사 3 + 최종 1", groups.count("risk") == 4)

    order = [s["key"] for s in stages[:4]]
    check("애널리스트 순서가 정의 순서로 고정",
          order == [k for k, _, _ in agents.ANALYSTS], str(order))

    check("최종 판단이 결과 최상단에", out["decision"] == "buy", out["decision"])
    check("확신도가 최종 단계에서 옴", out["confidence"] == 0.72)
    check("증거 팩이 리포트에 함께 저장", out["report"]["evidence"]["symbol"] == "005930")

    # 사용량 합계 = 11회 × 각 호출
    check("입력 토큰 합계", out["usage"]["input_tokens"] == 11 * 1000,
          str(out["usage"]))
    check("캐시 읽기 합계", out["usage"]["cache_read_tokens"] == 11 * 2000)
    per_call = agents.cost_of("claude-opus-5",
                              {"input_tokens": 1000, "output_tokens": 400,
                               "cache_read_tokens": 2000})
    check("비용 = 단계 비용의 합",
          abs(out["cost_usd"] - per_call * 11) < 1e-9, f"{out['cost_usd']:.6f}")

    check("최종 판단만 노력 수준이 다름",
          fake.calls[-1]["effort"] == "high" and fake.calls[0]["effort"] == "medium",
          f"{fake.calls[0]['effort']} / {fake.calls[-1]['effort']}")
    check("모든 호출이 같은 시스템 프롬프트를 씀 (캐시 조건)",
          len({c["system"] for c in fake.calls}) == 1)
    return out


def test_evidence_text():
    """증거 팩에 없는 자료는 '없다'고 명시돼야 합니다."""
    ev = dict(_fake_evidence("x", {}))
    ev["missing"] = ["뉴스 (Timeout)"]
    text = agents.evidence_text(ev)
    check("빠진 자료를 프롬프트에 적음", "수집하지 못한 자료" in text and "뉴스" in text)
    check("지어내지 말라는 지시가 있음", "있는 것처럼 말하지 마세요" in text)
    check("시스템 프리앰블에 투자자문 아님 고지",
          "투자자문이 아닙니다" in agents.SYSTEM_PREAMBLE)


# ---------------------------------------------------------------------------
# 저장과 채점
# ---------------------------------------------------------------------------

def test_storage(result):
    analysis_id = store.save_analysis(USER, result, horizon_days=5)
    check("저장 후 행 번호를 돌려줌", analysis_id > 0)

    row = store.get_analysis(USER, analysis_id)
    check("리포트가 그대로 돌아옴", len(row["report"]["stages"]) == 11)
    check("판단 시각 가격이 박혀 있음", row["base_price"] == 70000)
    check("채점 만기가 잡혀 있음", bool(row["target_at"]))
    check("아직 미채점", row["correct"] is None)

    check("남의 판단은 열리지 않음", store.get_analysis(USER + 1, analysis_id) is None)

    listed = store.get_analyses(USER, limit=10)
    check("목록 조회", len(listed) == 1 and listed[0]["id"] == analysis_id)
    check("목록에는 리포트를 싣지 않음", "report" not in listed[0])

    cost = store.cost_summary(USER, days=30)
    check("비용 집계", cost["count"] == 1 and cost["cost_usd"] > 0)
    return analysis_id


def test_due_queue(analysis_id):
    """만기 전에는 채점 대기열에 들어가지 않아야 합니다."""
    now = datetime.now()
    check("만기 전에는 대기열에 없음",
          not any(r["id"] == analysis_id for r in store.due_for_scoring(now=now)))
    later = now + timedelta(days=6)
    check("만기 후에는 대기열에 들어옴",
          any(r["id"] == analysis_id for r in store.due_for_scoring(now=later)))


def test_correctness():
    check("매수는 오르면 맞음", agent_review._correct("buy", 3.0))
    check("매수는 내리면 틀림", not agent_review._correct("buy", -3.0))
    check("매도는 내리면 맞음", agent_review._correct("sell", -3.0))
    check("보유는 조용하면 맞음", agent_review._correct("hold", 1.0))
    check("보유는 크게 움직이면 틀림", not agent_review._correct("hold", 9.0))
    check("매도의 방향성 수익은 부호가 뒤집힘",
          agent_review.directional_return("sell", -4.0) == 4.0)
    check("보유는 방향성 수익 계산에서 제외",
          agent_review.directional_return("hold", 4.0) is None)


def _seed_scored(user_id: int, n: int, ret: float, spread: float = 0.4):
    """채점이 끝난 판단을 n건 심습니다 (성적표와 게이트 검사용)."""
    base = {"symbol": "TEST", "name": "테스트", "market": "KR", "decision": "buy",
            "confidence": 0.7, "horizon": "1주", "summary": "", "report": {},
            "model": "claude-opus-5", "calls": 11, "usage": {}, "cost_usd": 0.1,
            "cost_krw": 140.0, "elapsed_ms": 1000, "price": 100.0}
    for i in range(n):
        rid = store.save_analysis(user_id, base, horizon_days=1)
        change = ret + (spread if i % 2 else -spread)
        store.resolve(rid, 100 * (1 + change / 100),
                      agent_review._correct("buy", change), change)


def test_scorecard_and_gate():
    user = 5150
    verdict = agent_review.gate(user)
    check("표본이 없으면 게이트가 잠김", not verdict["passed"], verdict["stage"])
    check("잠긴 이유가 표본 부족", verdict["stage"] == "표본 부족")

    _seed_scored(user, 10, ret=1.0)
    verdict = agent_review.gate(user)
    check("표본이 모자라면 여전히 잠김", not verdict["passed"])

    _seed_scored(user, 30, ret=1.0)
    card = agent_review.scorecard(user)
    check("방향성 판단 40건", card["directional"] == 40, str(card["directional"]))
    check("평균 수익률이 양수", card["avg_return_pct"] > 0, str(card["avg_return_pct"]))
    check("적중률 100%", card["hit_rate"] == 100.0, str(card["hit_rate"]))
    check("누적 곡선이 건수만큼", len(card["equity"]) == 40)
    check("수수료 미반영 고지가 붙음", "수수료" in card["note"])

    verdict = agent_review.gate(user)
    check("성적이 좋으면 게이트 통과", verdict["passed"], verdict["reason"])

    loser = 5151
    _seed_scored(loser, 40, ret=-1.0)
    verdict = agent_review.gate(loser)
    check("지는 성적은 통과 못 함", not verdict["passed"], verdict["stage"])
    check("막힌 단계가 기대값", verdict["stage"] == "기대값 미달", verdict["stage"])


# ---------------------------------------------------------------------------
# 집행 오버레이
# ---------------------------------------------------------------------------
class _Inst:
    def __init__(self, key):
        self.key = key
        self.name = key


def _sig(score=0.20, threshold=0.35):
    s = strategy.Signal(key="TEST", ok=True, score=score, price=100.0,
                        price_krw=100.0, direction=strategy.FLAT)
    s.entry_threshold = threshold
    return s


def _put_decision(user_id, symbol, decision, confidence, minutes_ago=0):
    row = {"symbol": symbol, "name": symbol, "market": "KR", "decision": decision,
           "confidence": confidence, "horizon": "", "summary": "", "report": {},
           "model": "claude-opus-5", "calls": 11, "usage": {}, "cost_usd": 0.1,
           "cost_krw": None, "elapsed_ms": 1, "price": 100.0}
    rid = store.save_analysis(user_id, row)
    if minutes_ago:
        stamp = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
        with store._conn() as conn:
            conn.execute("UPDATE at_agent_analysis SET created_at = ? WHERE id = ?",
                         (stamp, rid))
    return rid


def test_overlay():
    user = 6000
    inst = _Inst("TESTSYM")
    _put_decision(user, "TESTSYM", "buy", 0.9)

    base = {"agent": {"exec_mode": "observe", "require_gate": False}}
    sig = agent_trader.apply_overlay(user, base, inst, _sig())
    check("관찰 모드는 점수를 건드리지 않음", sig.score == 0.20 and sig.direction == strategy.FLAT)

    cfg = {"agent": {"exec_mode": "overlay", "require_gate": False,
                     "overlay_weight": 0.35, "min_confidence": 0.55}}
    sig = agent_trader.apply_overlay(user, cfg, inst, _sig())
    check("오버레이가 점수를 올림", abs(sig.score - (0.20 + 0.9 * 0.35)) < 1e-9,
          f"{sig.score:.4f}")
    check("문턱을 넘어 진입 방향이 섬", sig.direction == strategy.LONG)
    check("계산 과정이 기록에 남음",
          any(s["key"] == "agent" for s in sig.stages))
    check("사람이 읽을 사유가 붙음", any("에이전트" in r for r in sig.reasons))

    # 매도 판단은 점수와 무관하게 신규 진입을 막습니다
    user2 = 6001
    _put_decision(user2, "TESTSYM", "sell", 0.8)
    cfg2 = {"agent": {"exec_mode": "overlay", "require_gate": False,
                      "veto_on_sell": True}}
    sig = agent_trader.apply_overlay(user2, cfg2, inst, _sig(score=0.9))
    check("매도 판단이면 좋은 점수여도 진입 안 함", sig.direction == strategy.FLAT)

    # 낡은 판단은 없는 셈
    user3 = 6002
    _put_decision(user3, "TESTSYM", "buy", 0.9, minutes_ago=60 * 30)
    cfg3 = {"agent": {"exec_mode": "overlay", "require_gate": False,
                      "max_age_hours": 12}}
    sig = agent_trader.apply_overlay(user3, cfg3, inst, _sig())
    check("낡은 판단은 무시", sig.score == 0.20)

    # 확신도가 낮으면 판단으로 치지 않음
    user4 = 6003
    _put_decision(user4, "TESTSYM", "buy", 0.30)
    cfg4 = {"agent": {"exec_mode": "overlay", "require_gate": False,
                      "min_confidence": 0.55}}
    sig = agent_trader.apply_overlay(user4, cfg4, inst, _sig())
    check("확신도가 낮으면 반영 안 함", sig.score == 0.20)


def test_gate_blocks_overlay():
    """게이트를 켜면, 성적이 쌓이기 전에는 매매에 영향을 주지 못합니다."""
    user = 6100
    inst = _Inst("TESTSYM")
    _put_decision(user, "TESTSYM", "buy", 0.9)
    cfg = {"agent": {"exec_mode": "overlay", "require_gate": True}}
    sig = agent_trader.apply_overlay(user, cfg, inst, _sig())
    check("게이트가 잠겨 있으면 점수 그대로", sig.score == 0.20)
    check("왜 안 먹었는지 기록에 남음",
          any(s["key"] == "agent" and "표본" in (s.get("detail") or "")
              for s in sig.stages), str(sig.stages))


def test_exit_vote():
    user = 6200
    _put_decision(user, "TESTSYM", "sell", 0.85)
    off = {"agent": {"exec_mode": "overlay", "require_gate": False,
                     "exit_on_sell": False}}
    check("청산 연동이 꺼져 있으면 표를 안 냄",
          agent_trader.exit_vote(user, off, "TESTSYM") == "")
    on = {"agent": {"exec_mode": "overlay", "require_gate": False,
                    "exit_on_sell": True}}
    check("켜면 매도 사유를 돌려줌",
          "매도 판단" in agent_trader.exit_vote(user, on, "TESTSYM"))


def test_budget_cap():
    user = 6300
    acfg = agent_trader.clamp_config({"daily_cost_cap_usd": 1.0})
    check("처음에는 예산이 남아 있음",
          abs(agent_trader.budget_left(user, acfg) - 1.0) < 1e-9)
    row = {"symbol": "X", "name": "X", "market": "KR", "decision": "hold",
           "confidence": 0.5, "horizon": "", "summary": "", "report": {},
           "model": "claude-opus-5", "calls": 11, "usage": {}, "cost_usd": 0.75,
           "cost_krw": None, "elapsed_ms": 1, "price": 1.0}
    store.save_analysis(user, row)
    check("쓴 만큼 줄어듦",
          abs(agent_trader.budget_left(user, acfg) - 0.25) < 1e-9,
          str(agent_trader.budget_left(user, acfg)))
    store.save_analysis(user, {**row, "cost_usd": 0.5})
    check("상한을 넘으면 0", agent_trader.budget_left(user, acfg) == 0.0)


def test_config_roundtrip():
    user = 6400
    cfg = store.save_config(user, {"exec_mode": "lead", "debate_rounds": 7,
                                   "model": "claude-sonnet-5"})
    check("집행 설정이 저장됨", cfg["exec_mode"] == "lead")
    check("범위를 넘는 값은 저장 단계에서 잘림", cfg["debate_rounds"] == 3)
    check("분석 설정도 같은 행에", cfg["model"] == "claude-sonnet-5")
    again = store.get_config(user)
    check("다시 읽어도 같음", again["exec_mode"] == "lead" and again["debate_rounds"] == 3)


def test_config_comes_from_own_table():
    """자동매매 회전은 **자기** 설정 dict 를 넘깁니다.

    에이전트 설정은 별도 테이블(at_agent_config)에 있으므로, 넘겨받은 dict 에서
    exec_mode 를 찾으면 영원히 없습니다 — 화면에는 "점수 보정"으로 저장돼 있는데
    실제로는 관찰만 하는 상태가 됩니다. 이 검사가 그 회귀를 막습니다.
    """
    user = 6500
    inst = _Inst("TESTSYM")
    _put_decision(user, "TESTSYM", "buy", 0.9)
    store.save_config(user, {"exec_mode": "overlay", "require_gate": False,
                             "overlay_weight": 0.35, "min_confidence": 0.55})
    agent_trader.invalidate_config(user)

    # 자동매매가 넘기는 것과 같은 모양 — "agent" 키가 없습니다
    autotrade_cfg = {"mode": "paper", "enabled": True, "entry_score": 0.35}
    sig = agent_trader.apply_overlay(user, autotrade_cfg, inst, _sig())
    check("저장된 설정으로 오버레이가 켜짐", sig.score > 0.20, f"{sig.score:.4f}")
    check("방향까지 반영됨", sig.direction == strategy.LONG)

    # cfg 에 agent 키가 있으면 그쪽이 우선 (과거 검증이 에이전트를 끄는 통로)
    off = {**autotrade_cfg, "agent": {"exec_mode": "observe"}}
    sig = agent_trader.apply_overlay(user, off, inst, _sig())
    check("cfg 의 agent 키가 저장된 설정을 이깁니다", sig.score == 0.20)

    # 저장 직후 캐시가 즉시 비워지는가
    store.save_config(user, {"exec_mode": "observe"})
    agent_trader.invalidate_config(user)
    sig = agent_trader.apply_overlay(user, autotrade_cfg, inst, _sig())
    check("설정을 끄면 곧바로 반영됨", sig.score == 0.20)


def test_no_order_path():
    """이 기능이 주문 함수를 직접 부르지 않는지 소스로 확인합니다.

    새 주문 경로가 생기면 자동매매의 방어 열 겹을 통째로 우회하게 됩니다.
    이 검사가 깨지면 그 설계가 무너졌다는 뜻입니다.
    """
    import io
    banned = ("brk.submit", "brk.close", "place_stock_order", "place_overseas_order",
              "place_deriv_order", "get_broker")
    for path in ("engine/agents.py", "engine/agent_trader.py",
                 "engine/agent_review.py", "storage/agents.py"):
        src = io.open(Path(__file__).resolve().parent.parent / path,
                      encoding="utf-8").read()
        hits = [b for b in banned if b in src]
        check(f"{path} 가 주문을 직접 내지 않음", not hits, ", ".join(hits))


def report() -> bool:
    print("\n" + "=" * 60)
    failed = [r for r in results if not r[1]]
    print(f"  총계: {len(results) - len(failed)}/{len(results)} 통과")
    for name, _, detail in failed:
        print(f"  FAIL: {name} {detail}")
    return not failed


if __name__ == "__main__":
    print("\n설정 클램프")
    test_clamp()
    test_cost()
    print("\n파이프라인")
    out = test_pipeline()
    test_evidence_text()
    print("\n저장과 채점")
    aid = test_storage(out)
    test_due_queue(aid)
    test_correctness()
    test_scorecard_and_gate()
    print("\n집행 오버레이")
    test_overlay()
    test_gate_blocks_overlay()
    test_exit_vote()
    test_budget_cap()
    test_config_roundtrip()
    test_config_comes_from_own_table()
    print("\n설계 불변식")
    test_no_order_path()
    try:
        Path(_tmp.name).unlink()
    except OSError:
        pass
    sys.exit(0 if report() else 1)
