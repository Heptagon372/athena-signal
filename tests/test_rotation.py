# -*- coding: utf-8 -*-
"""
회전(갈아타기) 판단 QA
----------------------
회전은 **팔지 않아도 되는 것을 파는** 기능이라, "돌아간다"보다
"팔면 안 되는 상황에서 안 판다"를 많이 검사합니다.

    python tests/test_rotation.py     # 네트워크·서버 없이 판단 로직만
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from engine import rotation
from engine.instruments import STOCK, Instrument
from models import ResolvedSymbol

results = []


def check(name, passed, detail=""):
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


def section(title):
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def fake_stock(key="TEST01", name="테스트종목", market="KOSPI",
               currency="KRW") -> Instrument:
    return Instrument(
        key=key, name=name, asset_class=STOCK, market=market, currency=currency,
        multiplier=1.0, margin_rate=1.0, shortable=False,
        symbol=ResolvedSymbol(key=key, name=name, market=market,
                              yahoo_symbol=key, currency=currency))


def realized_krw(inst, side, entry, exit_, qty):
    """왕복 수수료·거래세를 전부 뺀 순손익 — engine/autotrade._exit_realized_krw
    와 같은 규약 (환산 없이 원화 종목만 씁니다)."""
    direction = -1 if str(side or "") == "short" else 1
    fee_in, tax_in = inst.costs("buy" if direction > 0 else "sell", entry * qty)
    fee_out, tax_out = inst.costs("sell" if direction > 0 else "buy", exit_ * qty)
    return (exit_ - entry) * qty * direction - fee_in - tax_in - fee_out - tax_out


HOURS_AGO = (datetime.now() - timedelta(hours=2)).isoformat()

# 새 후보 — 진입 10,000 / 목표 11,000 (10%). 한국 주식 왕복비용 ~0.31% 라
# 비용 대비 30배가 넘어 회전 게이트(5배)를 여유 있게 통과합니다.
NEW = {"inst": fake_stock("NEW"), "key": "NEW", "name": "새후보",
       "score": 0.80, "direction": 1, "price": 10_000.0, "target_price": 11_000.0}

CFG = {"rotation_enabled": True, "paper_slippage_bps": 5.0}


def holding(key="V", entry=10_000.0, price=10_800.0, target=11_000.0,
            score=0.30, opened_at=HOURS_AGO, currency="KRW", **extra):
    """기본값: +8% 진행, 목표까지 1.85% 남은 순익 포지션 — 회전 적격."""
    return {"inst": fake_stock(key, currency=currency), "key": key, "name": key,
            "side": "long", "quantity": 10.0, "entry_price": entry,
            "price": price, "target_price": target, "peak_price": price,
            "score": score, "ma50": None, "opened_at": opened_at, **extra}


def run(cfg=None, new=None, holdings=None, today=0):
    return rotation.evaluate(cfg or CFG, new or NEW,
                             holdings if holdings is not None else [holding()],
                             realized_krw, today)


section("켜짐/꺼짐 · 한도")

d = run(cfg={"rotation_enabled": False})
check("기본은 꺼짐 — 아무것도 팔지 않는다", not d.ok and "rotation_enabled" in d.rejects[0])

d = run(today=2)
check("하루 회전 한도(기본 2회) 도달이면 거부", not d.ok and "한도" in d.rejects[0])

d = run()
check("정상 케이스 — 순익 + 남은기대 작음 → 회전 승인", d.ok and d.victim["key"] == "V")
check("판단 근거 숫자가 detail 에 남는다",
      d.detail.get("victim_net_krw", 0) > 0 and d.detail.get("gap_pct", 0) >= 1.0)

section("팔리는 쪽 보호 — 사용자 핵심 조건: 수수료 빼고도 순이익")

d = run(holdings=[holding(price=10_010.0)])   # +0.1% — 왕복비용(~0.31%)에 못 미침
check("명목 이익이라도 수수료 빼면 손실이면 거부",
      not d.ok and any("순손실" in r for r in d.rejects))

d = run(holdings=[holding(price=9_500.0)])
check("평가손실 포지션은 회전으로 못 판다 (손절의 일)",
      not d.ok and any("순손실" in r for r in d.rejects))

d = run(holdings=[holding(entry=0)])
check("진입가를 모르면 건드리지 않는다", not d.ok)

d = run(holdings=[holding(opened_at=datetime.now().isoformat())])
check("산 지 30분 안 된 포지션은 안 판다", not d.ok and any("분" in r for r in d.rejects))

d = run(holdings=[holding(currency="USD")])
check("통화가 다르면 매도대금이 재원이 안 되므로 거부",
      not d.ok and any("통화" in r for r in d.rejects))

section("기대수익 비교 — 부정확한 추정에 요구하는 여유")

d = run(holdings=[holding(score=0.75)])       # 새 0.80 vs 보유 0.75 — 마진 0.15 미달
check("점수 우위가 마진(0.15) 미만이면 거부",
      not d.ok and any("점수 우위" in r for r in d.rejects))

d = run(holdings=[holding(price=10_100.0, target=12_000.0)])  # 남은기대 18.8%
check("남은 기대가 새 기대보다 크면 거부 (승자를 팔지 않는다)",
      not d.ok and any("여유 부족" in r for r in d.rejects))

d = run(new={**NEW, "target_price": 10_025.0})   # 목표 0.25% < 왕복비용
check("새 후보 목표가 왕복비용 5배 미만이면 회전 자체를 안 한다",
      not d.ok and any("기대이익 부족" in r for r in d.rejects))

d = run(new={**NEW, "target_price": 0})
check("새 후보 목표가 없으면 잴 수 없으므로 거부", not d.ok)

section("승자 보유(hold_winners)와의 공존")

cfg_hw = {**CFG, "hold_winners": True}
winner = holding(price=10_800.0, ma50=10_000.0, peak_price=10_900.0)  # 50일선 +8%, 고점 99%
d = run(cfg=cfg_hw, holdings=[winner])
check("추세가 살아 있는 승자는 회전도 못 판다",
      not d.ok and any("승자 보유" in r for r in d.rejects))

d = run(cfg=cfg_hw, holdings=[holding(ma50=None)])
check("50일선을 모르면 안전하게 건너뛴다",
      not d.ok and any("판정 불능" in r for r in d.rejects))

d = run(cfg=cfg_hw, holdings=[holding(ma50=11_500.0)])  # 50일선 아래 — 추세 꺾임
check("추세 꺾인 종목은 hold_winners 켜져 있어도 회전 가능", d.ok)

section("피해자 선택 — 잃을 것이 가장 적은 종목")

two = [holding("A", price=10_500.0, target=11_500.0),   # 남은기대 9.5% → 부적격(여유<1%p)
       holding("B", price=10_800.0, target=11_000.0)]   # 남은기대 1.85% → 적격
d = run(holdings=two)
check("남은 기대가 작은 쪽만 뽑힌다", d.ok and d.victim["key"] == "B")

three = [holding("C", price=10_800.0, target=11_000.0),          # 남은기대 1.85%
         holding("D", price=10_950.0, target=11_000.0)]          # 남은기대 0.46%
d = run(holdings=three)
check("적격이 여럿이면 남은 기대 최소를 고른다", d.ok and d.victim["key"] == "D")

section("설정 하한 강제 — 0으로 풀어 무제한 회전이 되지 않게")

p = rotation.params({"rotation_min_gap_pct": 0, "rotation_score_margin": -1,
                     "rotation_cost_edge_multiple": 0.1, "rotation_min_hold_min": 0,
                     "rotation_reentry_min": 0})
check("게이트 하한이 강제된다",
      p["rotation_min_gap_pct"] >= 0.2 and p["rotation_score_margin"] >= 0.05
      and p["rotation_cost_edge_multiple"] >= 2.0 and p["rotation_min_hold_min"] >= 5
      and p["rotation_reentry_min"] >= 30)

failed = [(n, det) for n, ok, det in results if not ok]
print(f"\n{'-' * 68}\n  {len(results)}건 중 실패 {len(failed)}건")
for n, det in failed:
    print(f"    FAIL: {n} {det}")
sys.exit(1 if failed else 0)
