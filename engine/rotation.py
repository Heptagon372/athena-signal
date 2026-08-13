# -*- coding: utf-8 -*-
"""회전(갈아타기) — 더 좋은 신호가 떴는데 살 재원이 없을 때의 판단.

새 후보가 진입 게이트를 전부 통과했는데 현금·슬롯이 없어 거부됐다면,
보유 종목 중 하나를 목표가 전에 팔아 자리를 만들지 검토합니다.

왜 별도 모듈인가
    파는 이유가 "이 종목이 나빠서"(strategy.check_exit)가 아니라 "**다른**
    종목이 더 좋아서"입니다. 판단 재료도 다릅니다 — 신호 하나가 아니라
    (새 후보, 보유 전체)를 함께 봅니다. check_exit 에 섞으면 그 함수의
    전제(한 종목씩 검사)가 깨지므로 따로 둡니다.

계산법 — 왜 이 부등식인가
    보유 V 를 계속 들면 얻는 것(시스템 추정): 남은기대% = (목표가−현재가)/현재가
        V 의 청산 수수료·거래세는 지금 팔든 목표가에 팔든 내므로 비교에서
        상쇄됩니다. 빼면 안 되는 것은 **새 자리의 왕복비용**입니다 —
        회전하지 않으면 아예 발생하지 않는 비용이기 때문입니다.
    V 를 팔고 새 후보 N 에 들어가면: 새기대% = N 목표수익% − N 왕복비용%
    따라서 회전이 이득일 조건은

        새기대% − 남은기대% ≥ rotation_min_gap_pct

    여기에 "기대수익 추정은 부정확하다"는 보정을 세 겹 붙입니다.
    스크리너 1등은 수십 후보 중 가장 좋아 **보이는** 종목이라 통계적으로
    과대평가되어 있고(승자의 저주), 이 시스템의 신호 우위도 적중률 55%
    안팎입니다. 추정을 믿는 대신 큰 여유를 요구합니다.

    ① 새 후보의 목표수익이 왕복비용의 rotation_cost_edge_multiple 배 이상
       (진입 게이트의 3배보다 높게 — 회전은 매매를 하나 **더** 만드는 결정)
    ② 점수 히스테리시스 — N 점수가 V 점수 + rotation_score_margin 이상
       (엇비슷한 점수로 갈아타면 수수료만 냅니다)
    ③ 팔리는 쪽은 수수료·거래세를 전부 빼고도 **순이익**일 것.
       지는 포지션의 정리는 손절의 일입니다 — 회전은 이긴 포지션의
       재배치만 합니다. 손실 종목까지 회전으로 팔기 시작하면 손절 규율이
       "더 좋은 게 나왔으니까"라는 핑계로 무너집니다.

이 모듈은 판단만 합니다. 주문 제출·기록은 engine/autotrade.py 가 합니다.
"""

from dataclasses import dataclass, field
from datetime import datetime

from engine import ensemble

# 기본값 — 전부 "회전을 어렵게" 하는 방향입니다. 회전은 안 해도 잃지 않지만
# (기회 손실뿐), 잘못 하면 수수료 + 승자 조기 매도로 두 번 잃습니다.
DEFAULTS = {
    "rotation_enabled": False,          # 사람이 켜야 합니다 (hold_winners 와 같은 이유)
    "rotation_cost_edge_multiple": 5.0, # 새 목표수익 ≥ 왕복비용 × 이 배수
    "rotation_min_gap_pct": 1.0,        # 새기대 − 남은기대 최소 여유 (%p)
    "rotation_score_margin": 0.15,      # 새 점수가 보유 점수보다 높아야 하는 폭
    "rotation_min_hold_min": 30,        # 산 지 이만큼 안 된 포지션은 안 팝니다
    "rotation_max_per_day": 2,          # 하루 회전 상한 (회전 과열 = 수수료 출혈)
    "rotation_reentry_min": 240,        # 회전으로 판 종목의 재진입 금지 시간
}


def params(cfg: dict) -> dict:
    """설정 병합 + 하한 강제. 0이나 음수로 풀어 무제한 회전이 되는 것을 막습니다."""
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        if cfg.get(key) is not None:
            out[key] = cfg[key]
    out["rotation_cost_edge_multiple"] = max(2.0, float(out["rotation_cost_edge_multiple"]))
    out["rotation_min_gap_pct"] = max(0.2, float(out["rotation_min_gap_pct"]))
    out["rotation_score_margin"] = max(0.05, float(out["rotation_score_margin"]))
    out["rotation_min_hold_min"] = max(5, int(out["rotation_min_hold_min"]))
    out["rotation_max_per_day"] = max(0, int(out["rotation_max_per_day"]))
    out["rotation_reentry_min"] = max(30, int(out["rotation_reentry_min"]))
    return out


@dataclass
class Decision:
    ok: bool = False
    victim: dict | None = None     # 팔기로 고른 보유 항목 (호출부가 준 dict 그대로)
    reason: str = ""               # 사람이 읽을 한 줄 (주문 사유·로그에 그대로 씁니다)
    rejects: list = field(default_factory=list)   # 왜 안 되는가 (후보별 사유 포함)
    detail: dict = field(default_factory=dict)    # 판단에 쓴 숫자 전부 (감사용)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "rejects": self.rejects,
                "detail": self.detail,
                "victim": self.victim.get("key") if self.victim else None}


def _held_minutes(opened_at) -> float | None:
    if not opened_at:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(str(opened_at))).total_seconds() / 60.0
    except (TypeError, ValueError):
        return None


def evaluate(cfg: dict, new: dict, holdings: list[dict],
             realized_fn, today_count: int) -> Decision:
    """회전할 것인가, 한다면 무엇을 팔 것인가.

    new       새 후보: inst · key · name · score · direction(±1) · price · target_price
    holdings  보유 목록. 항목마다:
                  inst · key · name · side · quantity · entry_price · price ·
                  target_price · peak_price · score · ma50 · opened_at
              (score/ma50 는 이번 회전의 청산 검사가 계산한 신선한 값이 원칙,
               없으면 호출부가 진입 시점 점수로 채웁니다)
    realized_fn(inst, side, entry, exit, qty) → 왕복 수수료·거래세를 전부 뺀
              순손익(원). 계산 불능이면 None (engine/autotrade._exit_realized_krw).
    today_count  오늘 이미 실행한 회전 횟수.
    """
    d = Decision()
    p = params(cfg)

    if not p["rotation_enabled"]:
        d.rejects.append("회전이 꺼져 있습니다 (rotation_enabled)")
        return d
    if today_count >= p["rotation_max_per_day"]:
        d.rejects.append(f"오늘 회전 한도 도달 ({today_count}/{p['rotation_max_per_day']}회)")
        return d

    inst_n = new.get("inst")
    price_n = float(new.get("price") or 0)
    target_n = float(new.get("target_price") or 0)
    score_n = float(new.get("score") or 0) * float(new.get("direction") or 1)
    if inst_n is None or price_n <= 0 or target_n <= 0:
        d.rejects.append("새 후보의 진입가·목표가가 없어 기대수익을 잴 수 없습니다")
        return d

    # -- 새 자리의 순기대 — 왕복비용(수수료·거래세·슬리피지)을 뺀 값 ---------
    edge = ensemble.cost_edge(inst_n, price_n, target_n, cfg)
    cost_pct = float(edge.get("cost_pct") or 0)
    target_pct = float(edge.get("target_pct") or 0)
    ratio = edge.get("ratio")
    if not target_pct or ratio is None:
        d.rejects.append("새 후보의 비용 대비 기대이익을 계산할 수 없습니다")
        return d
    if ratio < p["rotation_cost_edge_multiple"]:
        d.rejects.append(
            f"새 후보 기대이익 부족 — 목표 {target_pct:.2f}% 는 왕복비용 "
            f"{cost_pct:.2f}% 의 {ratio:.1f}배 (회전에는 "
            f"{p['rotation_cost_edge_multiple']:g}배 필요)")
        return d
    net_new_pct = target_pct - cost_pct

    # -- 보유 종목 하나하나: 팔아도 되는가 → 팔면 이득인가 -------------------
    eligible = []
    for h in holdings:
        key = h.get("key") or "?"
        inst_v = h.get("inst")
        qty = float(h.get("quantity") or 0)
        entry = float(h.get("entry_price") or 0)
        price = float(h.get("price") or 0)
        if inst_v is None or qty <= 0 or entry <= 0 or price <= 0:
            d.rejects.append(f"{key}: 진입가·현재가를 몰라 순손익을 잴 수 없습니다")
            continue

        # 통화가 다르면 판 돈이 새 후보의 재원이 되지 못합니다 (원화 매도대금으로
        # 미국 주식을 사려면 환전이 필요한데, 그건 이 엔진의 일이 아닙니다)
        if getattr(inst_v, "currency", None) != getattr(inst_n, "currency", None):
            d.rejects.append(f"{key}: 통화가 달라 매도대금이 재원이 되지 않습니다")
            continue

        held_min = _held_minutes(h.get("opened_at"))
        if held_min is not None and held_min < p["rotation_min_hold_min"]:
            d.rejects.append(f"{key}: 보유 {held_min:.0f}분 — "
                             f"{p['rotation_min_hold_min']}분 전에는 팔지 않습니다")
            continue

        direction = -1 if str(h.get("side") or "").strip() == "short" else 1

        # ③ 수수료를 전부 빼고도 순이익인가 — 사용자가 요구한 핵심 조건.
        net_krw = realized_fn(inst_v, h.get("side"), entry, price, qty)
        if net_krw is None:
            d.rejects.append(f"{key}: 순손익 계산 불능 (진입가 미상) — 건드리지 않습니다")
            continue
        if net_krw < 0:
            d.rejects.append(f"{key}: 지금 팔면 순손실 {net_krw:,.0f}원 — "
                             "손실 정리는 회전이 아니라 손절의 일입니다")
            continue

        # ② 점수 히스테리시스 — 새 후보가 확실히 더 좋아야 합니다
        score_v = float(h.get("score") or 0) * direction
        if score_n < score_v + p["rotation_score_margin"]:
            d.rejects.append(f"{key}: 점수 우위 부족 (보유 {score_v:+.2f} vs "
                             f"새 후보 {score_n:+.2f}, 필요 마진 "
                             f"{p['rotation_score_margin']:g})")
            continue

        # 승자 보유(hold_winners)가 켜져 있으면 추세가 살아 있는 승자는 회전도
        # 팔지 못합니다 — 목표가 익절도 보류하는 종목을 회전이 팔면 모순입니다.
        if cfg.get("hold_winners") and direction > 0:
            ma50 = h.get("ma50")
            peak = h.get("peak_price")
            if not ma50:
                d.rejects.append(f"{key}: 50일선을 몰라 승자 보유 판정 불능 — "
                                 "안전하게 건너뜁니다")
                continue
            if (price >= float(ma50) * 1.03
                    and peak and price >= float(peak) * 0.97):
                d.rejects.append(f"{key}: 승자 보유 중 (50일선 위 + 고점 부근) — "
                                 "회전 금지")
                continue

        # 남은 기대 — 목표가가 없으면 0 (기다려서 얻을 것으로 정의된 몫이 없음)
        target_v = float(h.get("target_price") or 0)
        remaining_pct = 0.0
        if target_v > 0:
            remaining_pct = max(0.0, (target_v - price) / price * 100.0 * direction)

        gap = net_new_pct - remaining_pct
        if gap < p["rotation_min_gap_pct"]:
            d.rejects.append(f"{key}: 여유 부족 — 새기대 {net_new_pct:.2f}% − "
                             f"남은기대 {remaining_pct:.2f}% = {gap:.2f}%p "
                             f"(필요 {p['rotation_min_gap_pct']:g}%p)")
            continue

        eligible.append({**h, "_net_krw": float(net_krw), "_score_eff": score_v,
                         "_remaining_pct": remaining_pct, "_gap": gap})

    if not eligible:
        return d

    # 잃을 것이 가장 적은 종목부터 — 남은 기대가 작은 순, 같으면 점수 낮은 순
    eligible.sort(key=lambda h: (h["_remaining_pct"], h["_score_eff"]))
    victim = eligible[0]

    d.ok = True
    d.victim = victim
    d.reason = (f"회전 매도 — {new.get('name') or new.get('key')} 진입 재원 확보: "
                f"순익 {victim['_net_krw']:+,.0f}원 확정, "
                f"남은기대 {victim['_remaining_pct']:.2f}% < "
                f"새기대 {net_new_pct:.2f}% (여유 {victim['_gap']:.2f}%p)")
    d.detail = {
        "new_key": new.get("key"), "new_score": round(score_n, 4),
        "new_target_pct": round(target_pct, 4), "new_cost_pct": round(cost_pct, 4),
        "new_net_pct": round(net_new_pct, 4), "cost_edge_ratio": ratio,
        "victim_key": victim.get("key"), "victim_score": round(victim["_score_eff"], 4),
        "victim_net_krw": round(victim["_net_krw"]),
        "victim_remaining_pct": round(victim["_remaining_pct"], 4),
        "gap_pct": round(victim["_gap"], 4),
        "params": {k: p[k] for k in DEFAULTS},
    }
    return d
