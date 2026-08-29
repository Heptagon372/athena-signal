"""
AI 에이전트 집행 정책 (판단 -> 주문)
====================================
engine/agents.py 가 낸 판단을 **실제 매매에 반영**하는 층입니다.

설계의 핵심: 새 주문 경로를 만들지 않습니다
    자동매매 진입 경로(engine/autotrade._handle_entry)에는 방어가 열 겹 넘게
    있습니다 — 시장 배분 게이트, 장 운영 시간, 종목 잠금, 미체결 주문 확인,
    증권사 거부 쿨다운, 회전 재진입 차단, 추격 재매수 차단, 수량 계획,
    리스크 게이트, 킬 스위치. 에이전트용 주문 함수를 따로 만들면 이 열 겹을
    통째로 우회하게 됩니다. 실제로 이 저장소에는 그렇게 뚫린 경로가 이미
    하나 있고(청산 API), 그게 사고 후보 1번입니다.

    그래서 이 모듈은 **신호를 고칠 뿐** 주문을 내지 않습니다. 에이전트의 판단은
    strategy.evaluate() 가 낸 점수 위에 오버레이로 얹히고, 그 뒤의 모든 방어는
    지금까지와 똑같이 지나갑니다. NNFX, 앙상블, ML 오버레이가 이미 쓰는 방식과
    같은 자리입니다.

집행 강도 (exec_mode)
    observe   판단을 기록만 합니다. 매매에 영향 없음 (기본값)
    overlay   에이전트 판단을 점수에 가중 합산합니다. 룰이 주도, 에이전트가 보정
    lead      에이전트가 방향을 정하고, 룰 점수는 거부권만 갖습니다

검증 게이트 (require_gate)
    켜져 있으면, **전진 검증 성적이 기준을 넘기 전까지** 에이전트가 매매에
    영향을 주지 못합니다 (판단은 계속 쌓입니다). LLM 판단은 문장이 매끄러워
    맞는 것처럼 보이기 때문에, 성적표 없이 집행을 붙이면 검증되지 않은
    직관에 돈을 겁니다. 기준은 engine/agent_review.gate() 가 정합니다.
    끄고 싶으면 설정에서 끄면 됩니다 — 다만 그 선택은 명시적이어야 합니다.

비용 방어
    자동 분석(auto_scan)은 회전마다 LLM 을 호출합니다. 하루 상한
    (daily_cost_cap_usd)을 넘으면 그날은 더 부르지 않습니다. 상한이 없으면
    종목 수 × 회전 수만큼 요금이 곱해집니다.
"""

import time
from datetime import datetime, timedelta

# 문자열이 코드 곳곳에 흩어지면 오타 하나로 조용히 꺼집니다
OBSERVE = "observe"
OVERLAY = "overlay"
LEAD = "lead"
EXEC_MODES = (OBSERVE, OVERLAY, LEAD)
EXEC_LABELS = {OBSERVE: "관찰만", OVERLAY: "점수 보정", LEAD: "에이전트 주도"}

DEFAULT_CONFIG = {
    "exec_mode": OBSERVE,
    # 판단이 점수에 실리는 무게. 0.35 면 확신도 1.0 의 매수 판단이 점수를
    # +0.35 올립니다 (점수 범위는 -1 ~ +1, 기본 진입 문턱은 0.35 근처).
    "overlay_weight": 0.35,
    "max_age_hours": 12,        # 이보다 오래된 판단은 없는 셈 칩니다
    "min_confidence": 0.55,     # 이 아래 확신도는 판단으로 치지 않습니다
    "veto_on_sell": True,       # 에이전트가 팔라 하면 신규 진입을 막습니다
    "exit_on_sell": False,      # 보유 종목도 정리할지 (기본 꺼짐)
    "auto_scan": False,         # 유니버스를 주기적으로 스스로 분석
    "scan_interval_min": 180,
    "scan_limit": 3,            # 회전당 분석할 종목 수 (비용에 직결)
    "daily_cost_cap_usd": 5.0,  # 하루 LLM 요금 상한
    "require_gate": True,       # 전진 검증을 통과해야 매매에 영향
}
_INT_KEYS = ("max_age_hours", "scan_interval_min", "scan_limit")
_BOOL_KEYS = ("veto_on_sell", "exit_on_sell", "auto_scan", "require_gate")
_FLOAT_KEYS = ("overlay_weight", "min_confidence", "daily_cost_cap_usd")


def clamp_config(cfg: dict) -> dict:
    """집행 설정을 허용 범위로 강제. 여기서 자르면 이후 어느 경로로 읽어도 안전."""
    out = {k: (cfg or {}).get(k, v) for k, v in DEFAULT_CONFIG.items()}
    if out["exec_mode"] not in EXEC_MODES:
        out["exec_mode"] = OBSERVE
    for key in _BOOL_KEYS:
        out[key] = bool(out[key])
    for key in _INT_KEYS:
        try:
            out[key] = int(out[key])
        except (TypeError, ValueError):
            out[key] = DEFAULT_CONFIG[key]
    for key in _FLOAT_KEYS:
        try:
            out[key] = float(out[key])
        except (TypeError, ValueError):
            out[key] = DEFAULT_CONFIG[key]
    out["overlay_weight"] = max(0.0, min(1.0, out["overlay_weight"]))
    out["min_confidence"] = max(0.0, min(1.0, out["min_confidence"]))
    out["max_age_hours"] = max(1, min(168, out["max_age_hours"]))
    out["scan_interval_min"] = max(30, min(1440, out["scan_interval_min"]))
    out["scan_limit"] = max(1, min(20, out["scan_limit"]))
    out["daily_cost_cap_usd"] = max(0.0, min(200.0, out["daily_cost_cap_usd"]))
    return out


# ---------------------------------------------------------------------------
# 설정 해석 — 이 층이 실제로 켜지는가를 정하는 자리
# ---------------------------------------------------------------------------
# 에이전트 설정은 자동매매 설정(at_config)이 아니라 **별도 테이블**
# (at_agent_config)에 있습니다. 그런데 이 모듈을 부르는 쪽(자동매매 회전)은
# 자기 설정 dict 를 넘깁니다. 거기서 exec_mode 를 찾으면 영원히 없으므로,
# 오버레이가 조용히 꺼진 채로 도는 상태가 됩니다 — 화면에서는 "점수 보정"으로
# 저장돼 있는데 실제로는 아무 일도 일어나지 않습니다.
#
# 그래서 설정은 **여기서 직접** 읽습니다. cfg 에 "agent" 키가 들어 있으면 그것을
# 우선합니다 (테스트와 과거 검증이 에이전트를 끄고 돌릴 때 쓰는 통로입니다).
_cfg_cache: dict[int, tuple[float, dict]] = {}
_CFG_TTL = 20.0                 # 회전 하나 안에서 종목마다 DB 를 왕복하지 않도록


def agent_config(user_id: int, cfg: dict = None) -> dict:
    """이 사용자의 에이전트 설정 (20초 캐시)."""
    if cfg and isinstance(cfg.get("agent"), dict):
        return clamp_config(cfg["agent"])
    now = time.time()
    hit = _cfg_cache.get(user_id)
    if hit and now - hit[0] < _CFG_TTL:
        return hit[1]
    from storage import agents as store

    resolved = clamp_config(store.get_config(user_id))
    _cfg_cache[user_id] = (now, resolved)
    return resolved


def invalidate_config(user_id: int):
    """설정을 저장한 직후 부릅니다. 20초를 기다리게 하면 '안 먹는다'가 됩니다."""
    _cfg_cache.pop(user_id, None)


# ---------------------------------------------------------------------------
# 최근 판단 조회
# ---------------------------------------------------------------------------

def fresh_decision(user_id: int, symbol: str, cfg: dict,
                   now: datetime = None) -> dict | None:
    """이 종목의 **아직 유효한** 최신 판단. 없으면 None.

    오래된 판단을 쓰지 않는 이유는 분명합니다. 어제 아침의 "매수"는 오늘의
    갭 하락을 모릅니다. 신선하지 않으면 판단이 없는 것과 같게 취급합니다.
    """
    from storage import agents as store

    now = now or datetime.now()
    rows = store.get_analyses(user_id, limit=1, symbol=symbol)
    if not rows:
        return None
    row = rows[0]
    try:
        created = datetime.fromisoformat(row["created_at"])
    except (ValueError, TypeError):
        return None
    if now - created > timedelta(hours=cfg["max_age_hours"]):
        return None
    if (row.get("confidence") or 0) < cfg["min_confidence"]:
        return None
    return row


def _signed(decision: str, confidence: float) -> float:
    """판단을 -1 ~ +1 의 부호 있는 값으로. 보유(hold)는 0 입니다."""
    conf = max(0.0, min(1.0, float(confidence or 0)))
    if decision == "buy":
        return conf
    if decision == "sell":
        return -conf
    return 0.0


# ---------------------------------------------------------------------------
# 오버레이 — 자동매매 진입 경로에서 불립니다
# ---------------------------------------------------------------------------

def apply_overlay(user_id: int, cfg: dict, inst, sig, now: datetime = None):
    """룰 신호 위에 에이전트 판단을 얹습니다. 주문은 내지 않습니다.

    sig 를 그 자리에서 고치고 그대로 돌려줍니다 (호출부가 계속 쓰던 객체입니다).
    무엇을 왜 바꿨는지는 sig.stages 와 sig.reasons 에 남깁니다 — 화면의
    "계산 과정"이 엔진과 어긋나지 않게 하는 이 저장소의 규칙입니다.
    """
    from engine import strategy

    acfg = agent_config(user_id, cfg)
    if acfg["exec_mode"] == OBSERVE:
        return sig

    allowed, why = influence_allowed(user_id, acfg)
    if not allowed:
        sig.stage("agent", "에이전트 오버레이", None, why)
        return sig

    row = fresh_decision(user_id, inst.key, acfg, now=now)
    if not row:
        return sig

    before = sig.score
    signed = _signed(row["decision"], row["confidence"])
    threshold = sig.entry_threshold or 0.0

    if acfg["exec_mode"] == LEAD:
        # 에이전트가 방향을 정합니다. 룰 점수는 **거부권만** 갖습니다 —
        # 룰이 반대 방향으로 문턱을 넘고 있으면 진입하지 않습니다.
        if signed > 0 and sig.score > -threshold:
            sig.direction = strategy.LONG
            sig.score = max(sig.score, signed)
        elif signed < 0:
            sig.direction = strategy.FLAT       # 이 모듈은 숏을 열지 않습니다
        else:
            sig.direction = strategy.FLAT
    else:
        # 점수 보정. 문턱을 다시 적용해 방향을 새로 정합니다.
        sig.score = max(-1.0, min(1.0, sig.score + signed * acfg["overlay_weight"]))
        if sig.score >= threshold:
            sig.direction = strategy.LONG
        elif sig.score <= -threshold:
            sig.direction = strategy.SHORT if cfg.get("allow_short") else strategy.FLAT
        else:
            sig.direction = strategy.FLAT

    # 매도 판단은 방향과 무관하게 신규 진입을 막습니다. 점수가 아무리 좋아도
    # "지금 사지 말라"는 판단이 있는데 사는 것은 오버레이를 켠 뜻이 아닙니다.
    if acfg["veto_on_sell"] and row["decision"] == "sell":
        sig.direction = strategy.FLAT
        sig.reasons.append(f"AI 에이전트 매도 판단으로 신규 진입 보류 "
                           f"(확신도 {row['confidence']:.2f})")

    label = {"buy": "매수", "sell": "매도", "hold": "보유"}.get(row["decision"], "")
    sig.stage("agent", "에이전트 오버레이", sig.score - before,
              f"{EXEC_LABELS[acfg['exec_mode']]}, 판단 {label} "
              f"확신도 {row['confidence']:.2f}, {row['created_at'][:16]}")
    sig.reasons.append(f"AI 에이전트 {label} 판단을 반영 "
                       f"(점수 {before:+.3f} 에서 {sig.score:+.3f})")
    return sig


def exit_vote(user_id: int, cfg: dict, symbol: str,
              now: datetime = None) -> str:
    """보유 종목을 지금 정리하라는 에이전트 판단이 있으면 그 사유를 돌려줍니다.

    빈 문자열이면 "에이전트는 청산에 관여하지 않는다"는 뜻입니다. 손절과 익절은
    지금까지처럼 engine/risk.py 가 정합니다 — 에이전트는 그 위에 사유를 하나
    더할 뿐, 기존 청산 규칙을 대체하지 않습니다.
    """
    acfg = agent_config(user_id, cfg)
    if acfg["exec_mode"] == OBSERVE or not acfg["exit_on_sell"]:
        return ""
    allowed, _why = influence_allowed(user_id, acfg)
    if not allowed:
        return ""
    row = fresh_decision(user_id, symbol, acfg, now=now)
    if not row or row["decision"] != "sell":
        return ""
    return (f"AI 에이전트 매도 판단 (확신도 {row['confidence']:.2f}, "
            f"{row['created_at'][:16]})")


# ---------------------------------------------------------------------------
# 검증 게이트
# ---------------------------------------------------------------------------

def influence_allowed(user_id: int, acfg: dict) -> tuple[bool, str]:
    """에이전트가 지금 매매에 영향을 줘도 되는가.

    게이트를 끈 계정은 언제나 통과합니다. 켠 계정은 전진 검증 성적이 기준을
    넘어야 통과합니다. 통과하지 못해도 분석과 기록은 계속됩니다 — 성적을
    쌓아야 게이트를 넘을 수 있기 때문입니다.
    """
    if not acfg.get("require_gate"):
        return True, "검증 게이트 꺼짐"
    from engine import agent_review

    verdict = agent_review.gate(user_id)
    # 단계 이름을 앞에 붙입니다. 신호 기록에는 이 한 줄만 남는데, 사유만 있으면
    # "무엇에 막혔는지"(표본 부족인지 기대값 미달인지)를 화면에서 알 수 없습니다.
    return verdict["passed"], f"{verdict['stage']}, {verdict['reason']}"


# ---------------------------------------------------------------------------
# 자동 분석 — 에이전트가 스스로 종목을 훑습니다
# ---------------------------------------------------------------------------
# 사람이 버튼을 누를 때만 도는 상태에서는 "에이전트가 매매한다"고 할 수 없습니다.
# 다만 회전마다 전 종목을 분석하면 요금이 폭발하므로, 세 가지로 조입니다.
#   1. 주기        scan_interval_min 마다 한 번
#   2. 종목 수     회전당 scan_limit 개, **판단이 가장 오래된 종목부터**
#   3. 하루 상한   daily_cost_cap_usd 를 넘으면 그날은 멈춤

_last_scan: dict[int, datetime] = {}


def scan_due(user_id: int, acfg: dict, now: datetime = None) -> bool:
    if not acfg.get("auto_scan"):
        return False
    now = now or datetime.now()
    last = _last_scan.get(user_id)
    if last and now - last < timedelta(minutes=acfg["scan_interval_min"]):
        return False
    return True


def budget_left(user_id: int, acfg: dict) -> float:
    """오늘 남은 LLM 예산 (USD). 0 이면 오늘은 더 부르지 않습니다."""
    from storage import agents as store

    cap = acfg.get("daily_cost_cap_usd") or 0.0
    if cap <= 0:
        return 0.0
    spent = store.cost_summary(user_id, days=1)["cost_usd"]
    return max(0.0, cap - spent)


def pick_targets(user_id: int, universe: list, acfg: dict,
                 now: datetime = None) -> list[str]:
    """이번 회전에 분석할 종목. 판단이 없거나 가장 오래된 것부터 고릅니다."""
    from storage import agents as store

    now = now or datetime.now()
    latest = {r["symbol"]: r["created_at"] for r in store.latest_by_symbol(user_id, 200)}
    stale_cutoff = now - timedelta(hours=acfg["max_age_hours"])

    def _age_key(symbol: str):
        stamp = latest.get(symbol)
        if not stamp:
            return ""                    # 판단이 아예 없는 종목이 가장 급합니다
        return stamp

    candidates = []
    for symbol in universe:
        stamp = latest.get(symbol)
        if stamp:
            try:
                if datetime.fromisoformat(stamp) > stale_cutoff:
                    continue             # 아직 신선합니다
            except (ValueError, TypeError):
                pass
        candidates.append(symbol)
    candidates.sort(key=_age_key)
    return candidates[:acfg["scan_limit"]]


def run_scan(user_id: int, universe: list, cfg: dict,
             now: datetime = None) -> dict:
    """자동 분석 한 회전. 자동매매 회전에서 불립니다.

    분석은 백그라운드 작업으로 띄웁니다 — 자동매매 회전이 LLM 응답을 기다리며
    수 분씩 멈추면, 그동안 손절도 못 나갑니다.
    """
    from engine import agents
    from storage import agents as store

    acfg = agent_config(user_id, cfg)
    out = {"started": [], "skipped": ""}
    if not scan_due(user_id, acfg, now):
        out["skipped"] = "주기 안"
        return out
    if not agents.readiness()["ready"]:
        out["skipped"] = agents.readiness()["reason"]
        return out
    left = budget_left(user_id, acfg)
    if left <= 0:
        out["skipped"] = "오늘 LLM 예산을 다 썼습니다"
        return out

    targets = pick_targets(user_id, universe, acfg, now)
    if not targets:
        out["skipped"] = "새로 분석할 종목이 없습니다"
        return out

    _last_scan[user_id] = now or datetime.now()
    acfg_full = {**store.get_config(user_id)}
    for symbol in targets:
        agents.start_job(user_id, symbol, acfg_full,
                         on_done=lambda r, uid=user_id: store.save_analysis(uid, r))
        out["started"].append(symbol)
    return out
