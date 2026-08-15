# -*- coding: utf-8 -*-
"""
분할 매수·분할 매도 QA (engine/split.py)
---------------------------------------
이 기능에서 가장 나쁜 실패는 두 가지입니다.

    1) 손실 난 차수를 "분할 매도"라는 이름으로 파는 것
       — 손절 규율이 "더 좋은 자리가 있으니까"로 무너집니다.
    2) 떨어진다는 사실만으로 차수를 계속 담는 것
       — 추세 하락에서 5차수가 하루에 소진되고 그대로 물립니다.

그래서 **거부 조건**을 집중적으로 봅니다. 네트워크를 쓰지 않습니다.

    python tests/test_split.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from engine import instruments, split, strategy

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = ""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


STOCK = instruments.Instrument(key="005930", name="삼성전자",
                               asset_class=instruments.STOCK, market="KOSPI",
                               currency="KRW")


def cfg(**over) -> dict:
    base = {"split_enabled": True, "split_min_gap_min": 0,
            "split_require_oversold": False}
    base.update(over)
    return base


def ledger_of(p: dict, tranches: list, base_price: float = 50_000,
              total_qty: float = 10.0) -> split.Ledger:
    """(차수번호, 수량, 가격, 몇 분 전) 목록으로 원장을 만듭니다."""
    led = split.Ledger(plan=split.plan_for(p, base_price, total_qty))
    now = datetime.now()
    for n, qty, price, ago_min in tranches:
        led.tranches.append(split.Tranche(
            n=n, quantity=qty, price=price,
            at=(now - timedelta(minutes=ago_min)).isoformat()))
    return led


# ---------------------------------------------------------------------------
# 설정 한도
# ---------------------------------------------------------------------------

def test_params():
    """0·음수로 풀어서 무한 분할이 되지 않는가."""
    print("\n[설정 한도]")
    p = split.params({"split_tranches": 0, "split_buy_drop_pct": 0,
                      "split_sell_gain_pct": -5, "split_pyramid_ratio": 99})
    check("차수 하한 2", p["split_tranches"] == 2, str(p["split_tranches"]))
    check("하락률 하한", p["split_buy_drop_pct"] >= 0.5)
    check("익절률 하한", p["split_sell_gain_pct"] >= 0.5)
    check("피라미드 배수 상한", p["split_pyramid_ratio"] <= 3.0)
    check("차수 상한 10", split.params({"split_tranches": 50})["split_tranches"] == 10)
    check("기본은 꺼짐", split.params({})["split_enabled"] is False)


def test_plan():
    """차수 계획 — 수량 배분과 손절선."""
    print("\n[차수 계획]")
    p = split.params(cfg(split_tranches=5, split_buy_drop_pct=5.0))
    plan = split.plan_for(p, 50_000, 10)
    check("차수 수만큼 수량이 나온다", len(plan["qty"]) == 5, str(plan["qty"]))
    check("균등 분배", plan["qty"] == [2, 2, 2, 2, 2], str(plan["qty"]))
    check("하락률은 차수-1개", len(plan["drop"]) == 4, str(plan["drop"]))

    # 5%씩 4번은 20%가 아니라 18.5% (곱셈 누적)
    check("마지막 차수가는 곱셈 누적",
          abs(plan["last_price"] - 50_000 * 0.95 ** 4) < 1e-6,
          f"{plan['last_price']:.1f}")
    check("손절가는 마지막 차수 아래",
          plan["stop"] < plan["last_price"] < 50_000,
          f"{plan['stop']:.0f} < {plan['last_price']:.0f}")
    check("손절폭이 24% 안팎", 23 < split.total_drawdown_pct(p, 50_000) < 26,
          f"{split.total_drawdown_pct(p, 50_000):.1f}%")

    pyr = split.plan_for(split.params(cfg(split_size_mode="pyramid",
                                          split_pyramid_ratio=1.5)), 50_000, 100)
    check("피라미드는 뒤 차수가 크다", pyr["qty"][-1] > pyr["qty"][0], str(pyr["qty"]))

    steps = split.plan_for(split.params(cfg(split_buy_drop_steps=[10, 20])), 50_000, 10)
    check("차수별 하락률 지정이 먹는다", steps["drop"][:2] == [10.0, 20.0], str(steps["drop"]))
    check("모자란 값은 마지막으로 채움", steps["drop"][-1] == 20.0, str(steps["drop"]))

    check("수량 0인 차수는 1주로 올림",
          all(q >= 1 for q in split.plan_for(p, 50_000, 2)["qty"]),
          str(split.plan_for(p, 50_000, 2)["qty"]))


# ---------------------------------------------------------------------------
# 원장
# ---------------------------------------------------------------------------

def test_ledger():
    print("\n[차수 원장]")
    p = split.params(cfg())
    led = ledger_of(p, [(1, 2, 50_000, 100), (2, 2, 47_500, 50)])
    check("보유 수량 합", led.quantity == 4)
    check("평단", abs(led.avg_price - 48_750) < 1e-6, f"{led.avg_price}")
    check("마지막 차수는 번호가 큰 것", led.last.n == 2)
    check("다음 차수 번호", led.next_n() == 3)

    # 2차를 팔면 그 자리가 비고, 다시 떨어지면 2차를 재매수합니다 —
    # 박스권에서 같은 자리를 반복해서 먹는 것이 이 방식의 핵심입니다.
    split.remove_tranche(led, 2)
    check("차수를 빼면 그 번호가 다시 비어난다", led.next_n() == 2, str(led.next_n()))
    check("남은 수량", led.quantity == 2)

    round_trip = split.load(ledger_of(p, [(1, 2, 50_000, 10)]).to_json())
    check("JSON 왕복", round_trip.count == 1 and round_trip.tranches[0].price == 50_000)
    check("깨진 원장은 빈 원장", split.load("{not json").count == 0)
    check("빈 값도 빈 원장", split.load(None).count == 0 and split.load("").count == 0)

    partial = ledger_of(p, [(1, 5, 50_000, 10)])
    split.remove_tranche(partial, 1, 2)
    check("부분 체결이면 남은 수량만 남는다", partial.quantity == 3, str(partial.quantity))

    today = datetime.now().strftime("%Y-%m-%d")
    check("오늘 담은 차수 수", ledger_of(p, [(1, 2, 50_000, 10),
                                        (2, 2, 47_500, 5)]).adds_on(today) == 2)


# ---------------------------------------------------------------------------
# 분할 매수
# ---------------------------------------------------------------------------

def test_buy():
    print("\n[분할 매수]")
    p = split.params(cfg())
    led = ledger_of(p, [(1, 2, 50_000, 60)])

    d = split.plan_buy(cfg(), STOCK, led, 49_000)
    check("아직 안 떨어졌으면 거부", not d.ok,
          (d.rejects or [""])[0])
    check("얼마나 더 빠져야 하는지 말한다",
          any("47,500" in r for r in d.rejects), " / ".join(d.rejects))

    check("내린 것을 내렸다고 쓴다 (부호)", "-2.00%" in " ".join(d.rejects),
          " / ".join(d.rejects))

    d = split.plan_buy(cfg(), STOCK, led, 47_500)
    check("-5% 도달하면 2차 매수", d.ok and d.tranche == 2, d.reason or str(d.rejects))
    check("계획된 수량", d.quantity == 2, str(d.quantity))
    check("매수 사유도 음수로", "-5.00%" in d.reason, d.reason)

    # 과매도 조건 — 켜면 지표가 없을 때 담지 않습니다
    d = split.plan_buy(cfg(split_require_oversold=True), STOCK, led, 47_500)
    check("과매도 조건이 켜져 있고 지표가 없으면 거부", not d.ok,
          (d.rejects or [""])[0])
    d = split.plan_buy(cfg(split_require_oversold=True), STOCK, led, 47_500, rsi=60)
    check("RSI 60 이면 거부", not d.ok)
    d = split.plan_buy(cfg(split_require_oversold=True), STOCK, led, 47_500, rsi=28)
    check("RSI 28 이면 매수", d.ok, d.reason or str(d.rejects))
    d = split.plan_buy(cfg(split_require_oversold=True), STOCK, led, 47_500, bb_pct=0.05)
    check("볼린저 하단이어도 매수", d.ok, d.reason or str(d.rejects))

    # 시간 간격 — 하루 급락에 차수가 한꺼번에 들어가는 것을 막습니다
    fresh = ledger_of(p, [(1, 2, 50_000, 5)])
    d = split.plan_buy(cfg(split_min_gap_min=30), STOCK, fresh, 47_500)
    check("직전 차수 직후면 거부", not d.ok, (d.rejects or [""])[0])

    # 하루 한도
    many = ledger_of(p, [(1, 2, 50_000, 300), (2, 2, 47_500, 200), (3, 2, 45_125, 100)])
    d = split.plan_buy(cfg(split_max_adds_per_day=2), STOCK, many, 42_868)
    check("하루 추가 매수 한도", not d.ok and "한도" in " ".join(d.rejects),
          " / ".join(d.rejects))

    # 차수 소진
    full = ledger_of(p, [(n, 2, 50_000 * 0.95 ** (n - 1), 300) for n in range(1, 6)])
    d = split.plan_buy(cfg(), STOCK, full, 1_000)
    check("차수를 다 쓰면 거부", not d.ok and "다 썼" in " ".join(d.rejects),
          " / ".join(d.rejects))

    check("원장이 없으면 거부", not split.plan_buy(cfg(), STOCK, split.Ledger(), 100).ok)
    check("꺼져 있으면 거부",
          not split.plan_buy({"split_enabled": False}, STOCK, led, 40_000).ok)


# ---------------------------------------------------------------------------
# 분할 매도
# ---------------------------------------------------------------------------

def test_sell():
    print("\n[분할 매도]")
    p = split.params(cfg())
    # 1차 50,000 / 2차 47,500 — 현재가 49,900 이면 2차만 +5.05%
    led = ledger_of(p, [(1, 2, 50_000, 300), (2, 2, 47_500, 100)])

    d = split.plan_sell(cfg(), STOCK, led, 49_900)
    check("목표 도달한 차수만 매도", d.ok and d.tranche == 2,
          d.reason or " / ".join(d.rejects))
    check("그 차수 수량만", d.quantity == 2, str(d.quantity))
    check("1차는 건드리지 않는다", "1차" not in (d.reason or ""), d.reason)

    d = split.plan_sell(cfg(), STOCK, led, 48_000)
    check("목표 미달이면 매도 안 함", not d.ok, " / ".join(d.rejects))
    check("손실 차수는 이유를 밝힌다",
          any("손실 차수" in r for r in d.rejects), " / ".join(d.rejects))

    # 손실 차수는 어떤 경우에도 팔지 않습니다 (과매수여도)
    d = split.plan_sell(cfg(), STOCK, led, 45_000, rsi=95, bb_pct=1.2)
    check("과매수여도 손실 차수는 안 판다", not d.ok, d.reason or " / ".join(d.rejects))

    # 과매수 조기 정리 — 목표(+5%)에 못 미쳐도 수익 난 차수는 정리
    d = split.plan_sell(cfg(), STOCK, led, 48_500, rsi=75)
    check("과매수면 목표 미달이어도 수익 차수 정리", d.ok and d.tranche == 2,
          d.reason or " / ".join(d.rejects))
    d = split.plan_sell(cfg(split_overbought_sell=False), STOCK, led, 48_500, rsi=75)
    check("과매수 매도를 끄면 안 판다", not d.ok)

    # 왕복비용 — 국내 주식은 왕복 0.2%대. 그 2배를 못 넘으면 팔지 않습니다.
    d = split.plan_sell(cfg(), STOCK, led, 47_600, rsi=75)
    check("수수료도 못 건지는 수익이면 거부", not d.ok,
          " / ".join(r for r in d.rejects if "왕복비용" in r) or " / ".join(d.rejects))

    # 여러 차수가 동시에 조건을 만족하면 늦은 차수(싸게 산 쪽)부터
    both = ledger_of(p, [(1, 2, 40_000, 300), (2, 2, 38_000, 100)])
    d = split.plan_sell(cfg(), STOCK, both, 45_000)
    check("여러 차수가 되면 늦은 차수부터", d.ok and d.tranche == 2, str(d.tranche))

    # 실제 보유가 원장보다 적으면 보유분까지만
    d = split.plan_sell(cfg(), STOCK, led, 49_900, held_quantity=1)
    check("보유 수량을 넘겨 팔지 않는다", d.ok and d.quantity == 1, str(d.quantity))

    check("원장이 없으면 거부", not split.plan_sell(cfg(), STOCK, split.Ledger(), 100).ok)


def test_ledger_drift():
    """원장과 실제 보유가 어긋나도 진입가를 잃지 않는가.

    밖에서 직접 더 사거나 일부를 판 계정에서 생깁니다. 원장이 비었다고
    평단·수량을 NULL 로 덮으면, 남은 실제 보유분이 진입가를 잃고 손익·손절이
    통째로 무너집니다.
    """
    print("\n[원장·보유 불일치]")
    from engine import autotrade
    from storage import autotrade as store
    from test_autotrade import TEST_USER

    key = "DRIFT01"
    store.upsert_position_state(TEST_USER, "paper", key, side="long",
                                entry_price=50_000, quantity=10, stop_price=40_000)
    autotrade._save_ledger(TEST_USER, "paper", key, split.Ledger())
    state = store.get_position_states(TEST_USER, "paper").get(key, {})
    check("원장이 비어도 진입가는 남는다", state.get("entry_price") == 50_000,
          str(state.get("entry_price")))
    check("수량도 남는다", state.get("quantity") == 10, str(state.get("quantity")))
    check("분할 관리에서만 빠진다", not (state.get("splits") or "").strip(),
          str(state.get("splits")))
    check("손절선도 그대로", state.get("stop_price") == 40_000)
    store.clear_position_state(TEST_USER, "paper", key)


def test_adopt():
    """원장 없는 보유 포지션(재동기화 복원 등)이 분할 관리로 편입되는가.

    실계좌에서 재동기화로 복원된 종목은 원장이 없어, 분할을 켜도 매 회전
    "분할로 잡은 포지션이 아닙니다"로 빠졌습니다. 편입되면 보유 전량이
    1차가 되고, 손절선이 분할 규칙(마지막 차수 아래)으로 다시 잡혀야 합니다.
    """
    print("\n[원장 없는 포지션 편입]")
    from engine import autotrade
    from storage import autotrade as store
    from test_autotrade import TEST_USER

    key = "ADOPT01"

    class Pos:
        pass
    pos = Pos()
    pos.key, pos.name, pos.side = key, "편입테스트", "long"
    pos.quantity, pos.avg_price = 4.0, 50_000.0

    conf = cfg(split_tranches=5, split_buy_drop_pct=5.0, split_final_stop_pct=7.0)
    saved_resolve = instruments.try_resolve
    instruments.try_resolve = lambda q: STOCK
    try:
        # 재동기화로 복원된 모양 그대로 — splits 는 NULL
        store.clear_position_state(TEST_USER, "paper", key)
        store.upsert_position_state(TEST_USER, "paper", key, side="long",
                                    entry_price=50_000, quantity=4,
                                    stop_price=48_000, target_price=55_000)
        state = store.get_position_states(TEST_USER, "paper").get(key, {})
        result = {"reconciled": []}
        autotrade._adopt_split(TEST_USER, conf, "paper", pos, state, result)

        state = store.get_position_states(TEST_USER, "paper").get(key, {})
        led = split.load(state.get("splits"))
        check("원장이 생긴다", led.count == 1, f"{led.count}차")
        check("보유 전량이 1차가 된다",
              led.count == 1 and led.tranches[0].quantity == 4.0
              and led.tranches[0].price == 50_000.0,
              str([t.to_dict() for t in led.tranches]))
        check("나머지 차수도 같은 크기로 계획된다",
              led.plan.get("qty") == [4.0] * 5, str(led.plan.get("qty")))
        check("손절선이 마지막 차수 아래로 내려간다",
              float(state.get("stop_price") or 0) < 50_000 * 0.95 ** 4,
              str(state.get("stop_price")))
        check("전량 목표가는 비운다 (차수별 익절이 대신)",
              not state.get("target_price"), str(state.get("target_price")))
        check("편입이 기록된다",
              any(r.get("action") == "split_adopted" for r in result["reconciled"]),
              str(result["reconciled"]))

        # 이미 원장이 있으면 다시 편입하지 않습니다 (계획이 덮이면 안 됩니다)
        before = state.get("splits")
        autotrade._adopt_split(TEST_USER, conf, "paper", pos, state, result)
        state = store.get_position_states(TEST_USER, "paper").get(key, {})
        check("이미 편입된 포지션은 건드리지 않는다", state.get("splits") == before)

        # 빈 문자열("")은 _save_ledger 가 일부러 분할에서 뺀 흔적 — 재편입 금지
        autotrade._save_ledger(TEST_USER, "paper", key, split.Ledger())
        state = store.get_position_states(TEST_USER, "paper").get(key, {})
        autotrade._adopt_split(TEST_USER, conf, "paper", pos, state,
                               {"reconciled": []})
        state = store.get_position_states(TEST_USER, "paper").get(key, {})
        check("일부러 뺀 포지션은 재편입하지 않는다",
              state.get("splits") == "", repr(state.get("splits")))

        # 분할이 꺼져 있으면 아무것도 하지 않습니다
        store.clear_position_state(TEST_USER, "paper", key)
        store.upsert_position_state(TEST_USER, "paper", key, side="long",
                                    entry_price=50_000, quantity=4)
        state = store.get_position_states(TEST_USER, "paper").get(key, {})
        autotrade._adopt_split(TEST_USER, cfg(split_enabled=False), "paper",
                               pos, state, {"reconciled": []})
        state = store.get_position_states(TEST_USER, "paper").get(key, {})
        check("분할이 꺼져 있으면 편입하지 않는다", state.get("splits") is None,
              repr(state.get("splits")))
    finally:
        instruments.try_resolve = saved_resolve
        store.clear_position_state(TEST_USER, "paper", key)


# ---------------------------------------------------------------------------
# 과매수·과매도 판정
# ---------------------------------------------------------------------------

def test_zones():
    print("\n[과매수·과매도 판정]")
    p = split.params(cfg())
    check("RSI 30 은 과매도", split.oversold(p, rsi=30)[0])
    check("RSI 40 은 아님", not split.oversold(p, rsi=40)[0])
    check("%B 0.1 은 과매도", split.oversold(p, bb_pct=0.1)[0])
    check("지표가 없으면 '모름'은 False", not split.oversold(p)[0])
    check("RSI 75 는 과매수", split.overbought(p, rsi=75)[0])
    check("RSI 60 은 아님", not split.overbought(p, rsi=60)[0])
    check("판정 근거를 문장으로 준다", "RSI" in split.oversold(p, rsi=30)[1])


# ---------------------------------------------------------------------------
# 청산 규칙과의 관계
# ---------------------------------------------------------------------------

class _Pos:
    def __init__(self, price, qty=4):
        self.side, self.quantity = "long", qty
        self.avg_price = self.current_price = price


def test_exit_interaction():
    """분할 포지션에서 익절은 넘기고 손절은 그대로 도는가.

    여기가 뒤집히면 두 가지가 동시에 깨집니다 — 전량 익절이 먼저 걸려 분할이
    무의미해지거나, 손절이 안 걸려 무한정 물립니다.
    """
    print("\n[청산 규칙 연동]")
    sig = strategy.Signal(key="005930", ok=True, price=54_000, price_krw=54_000)
    state = {"entry_price": 50_000, "stop_price": 37_875, "target_price": 54_000,
             "opened_at": datetime.now().isoformat()}
    conf = {"take_profit_pct": 8.0, "stop_loss_pct": 0, "max_hold_days": 0}

    normal = strategy.check_exit(STOCK, _Pos(54_000), sig, conf, state)
    check("분할이 아니면 목표가에서 전량 익절", normal.should_exit, normal.reason)

    deferred = strategy.check_exit(STOCK, _Pos(54_000), sig, conf, state,
                                   defer_take_profit=True)
    check("분할이면 전량 익절을 넘긴다", not deferred.should_exit, deferred.reason)

    low = strategy.Signal(key="005930", ok=True, price=37_000, price_krw=37_000)
    stopped = strategy.check_exit(STOCK, _Pos(37_000), low, conf, state,
                                  defer_take_profit=True)
    check("분할이어도 손절은 그대로", stopped.should_exit, stopped.reason)
    check("손절은 급한 청산", stopped.urgency == "urgent")

    # 신호 반전은 전량입니다 — 분할은 '언제 파는가'를 바꾸는 것이지
    # '나빠진 종목을 계속 들고 있는가'를 바꾸는 장치가 아닙니다.
    bad = strategy.Signal(key="005930", ok=True, price=51_000, price_krw=51_000,
                          score=-0.5)
    flipped = strategy.check_exit(STOCK, _Pos(51_000), bad, {"exit_score": 0.05},
                                  state, defer_take_profit=True)
    check("신호 반전은 분할이어도 전량", flipped.should_exit, flipped.reason)


def test_describe():
    print("\n[화면 요약]")
    p = split.params(cfg())
    led = ledger_of(p, [(1, 2, 50_000, 300), (2, 2, 47_500, 100)])
    info = split.describe(led, 49_000)
    check("차수 수", info["count"] == 2 and info["total"] == 5)
    check("차수별 목표가가 있다",
          all(r["target_price"] for r in info["tranches"]), str(info["tranches"]))
    check("다음 차수 트리거를 알려준다", info["next_trigger"] is not None,
          str(info["next_trigger"]))
    check("손절가를 알려준다", info["stop"] is not None)


# ---------------------------------------------------------------------------
# 엔진 통합 — 모의계좌로 실제 주문 경로를 태웁니다
# ---------------------------------------------------------------------------

def test_engine():
    """1차 진입 → 하락 → 2차 매수 → 상승 → 2차만 매도.

    판단 함수가 맞아도 주문 경로가 틀리면 아무 소용이 없습니다. 특히
    **부분 매도**는 이 프로젝트에 없던 동작이라(청산은 늘 전량이었습니다),
    남은 차수가 손절선을 달고 제대로 남는지를 여기서 확인합니다.
    """
    print("\n[엔진 통합 — 모의계좌]")
    from datetime import datetime as dt

    import pandas as pd
    from engine import autotrade, feed
    from storage import autotrade as store
    from storage import paper
    from test_autotrade import FakeFeed, TEST_USER, fake_stock, trend_bars

    paper.ensure_account(TEST_USER)
    paper.reset(TEST_USER, 10_000_000)
    for symbol in list(store.get_position_states(TEST_USER, "paper")):
        store.clear_position_state(TEST_USER, "paper", symbol)
    with store._conn() as conn:
        conn.execute("DELETE FROM at_daily WHERE user_id = ?", (TEST_USER,))

    # 주문번호가 분 단위라 같은 분에 두 번 돌리면 두 번째가 무시됩니다
    key = f"SPL{dt.now():%H%M%S}"
    inst = fake_stock(key=key, name="분할테스트")
    bars = trend_bars(n=140, drift=0.006)
    price = float(bars["close"].iloc[-1])

    saved_bars, saved_resolve = feed.bars, autotrade._resolve_universe_item
    feed.bars = lambda i, tf="day", count=120: (bars if tf == "day" else pd.DataFrame())
    autotrade._resolve_universe_item = lambda q: inst

    store.save_config(TEST_USER, {
        "mode": "paper", "universe": [key], "entry_score": 0.2,
        "intraday_weight": 0.0, "use_news": False, "dry_run": False,
        "regular_session_only": False, "min_order_krw": 0, "max_positions": 3,
        "position_pct": 50.0, "risk_per_trade_pct": 5.0,
        "asset_classes": {"STOCK": True},
        # 분할 — 지표 조건은 끄고 가격 조건만 봅니다 (합성 봉의 RSI 는
        # 이 테스트가 검증하려는 대상이 아닙니다. 지표 조건은 test_buy 가 봅니다)
        "split_enabled": True, "split_tranches": 4, "split_buy_drop_pct": 5.0,
        "split_sell_gain_pct": 5.0, "split_require_oversold": False,
        "split_min_gap_min": 0, "split_max_adds_per_day": 5,
    })

    try:
        with FakeFeed({key: price}, {key: inst}) as fake:
            r = autotrade.run_once(TEST_USER, force=True)
            entries = r.get("entries") or []
            check("1차 진입", len(entries) == 1,
                  str([(e["symbol"], e["quantity"]) for e in entries])
                  or str(r.get("rejects")))

            state = store.get_position_states(TEST_USER, "paper").get(key, {})
            led = split.load(state.get("splits"))
            check("차수 원장이 생긴다", led.count == 1, f"{led.count}차")
            check("1차 수량은 전체의 1/4",
                  len(led.plan.get("qty") or []) == 4
                  and led.tranches[0].quantity == led.plan["qty"][0],
                  f"계획 {led.plan.get('qty')} / 체결 {led.tranches[0].quantity}")
            check("손절선이 마지막 차수 아래로 내려간다",
                  state.get("stop_price")
                  and float(state["stop_price"]) < price * 0.85,
                  f"진입 {price:,.0f} → 손절 {float(state.get('stop_price') or 0):,.0f}")
            check("전량 목표가는 비어 있다 (차수별 익절이 대신)",
                  not state.get("target_price"), str(state.get("target_price")))

            # -5% 아래로 → 2차
            fake.prices[key] = led.tranches[0].price * 0.94
            r2 = autotrade.run_once(TEST_USER, force=True)
            state = store.get_position_states(TEST_USER, "paper").get(key, {})
            led = split.load(state.get("splits"))
            check("하락하면 2차를 담는다", led.count == 2,
                  f"{led.count}차 · {[e.get('split_tranche') for e in r2.get('entries') or []]}"
                  + str(r2.get("rejects") or ""))
            check("평단이 내려간다", led.avg_price < price, f"{led.avg_price:,.0f}")

            # 2차 매수가 대비 +6% → 2차만 매도 (1차는 아직 손실)
            fake.prices[key] = led.tranches[1].price * 1.06
            r3 = autotrade.run_once(TEST_USER, force=True)
            exits = r3.get("exits") or []
            check("2차만 분할 매도", len(exits) == 1
                  and exits[0].get("split_tranche") == 2,
                  str([(e.get("split_tranche"), e.get("quantity")) for e in exits]))

            state = store.get_position_states(TEST_USER, "paper").get(key, {})
            check("포지션 상태가 살아 있다 (전량 청산이 아님)", bool(state),
                  "" if state else "상태가 지워졌습니다 — 남은 차수가 손절선을 잃습니다")
            led = split.load(state.get("splits"))
            check("원장에 1차만 남는다", led.count == 1 and led.tranches[0].n == 1,
                  f"{[t.n for t in led.tranches]}")
            check("손절선은 그대로", state.get("stop_price"),
                  str(state.get("stop_price")))

            positions = {p.key: p for p in
                         autotrade.broker_module.get_broker(
                             TEST_USER, "paper", store.get_config(TEST_USER)).positions()}
            check("계좌에도 일부만 남는다",
                  key in positions and positions[key].quantity == led.quantity,
                  f"계좌 {positions.get(key).quantity if key in positions else 0} / "
                  f"원장 {led.quantity}")

            # 마지막 차수가 목표에 닿으면 전량 청산으로 나가야 합니다.
            # 이 분기가 없으면 남은 한 차수는 익절이 영원히 없습니다
            # (분할 포지션은 전량 목표가를 비워 두기 때문입니다).
            #
            # 진입 문턱을 올려두고 확인합니다 — 이 신호는 계속 양수라, 청산한
            # 바로 그 회전에서 1차를 다시 잡습니다(정상 동작). 그러면 "상태가
            # 정리됐는가"를 볼 수 없습니다. 재진입 자체는 test_autotrade 가 봅니다.
            store.save_config(TEST_USER, {"entry_score": 0.99})
            fake.prices[key] = led.tranches[0].price * 1.07
            r4 = autotrade.run_once(TEST_USER, force=True)
            exits4 = r4.get("exits") or []
            check("마지막 차수는 전량 청산으로 나간다", len(exits4) == 1,
                  str([(e.get("reason"), e.get("quantity")) for e in exits4]))
            check("청산 사유에 차수가 보인다",
                  any("마지막 차수" in (e.get("reason") or "") for e in exits4),
                  str([e.get("reason") for e in exits4]))
            check("전량 청산이면 상태가 정리된다",
                  key not in store.get_position_states(TEST_USER, "paper"))
    finally:
        feed.bars, autotrade._resolve_universe_item = saved_bars, saved_resolve
        store.save_config(TEST_USER, {"split_enabled": False})


def main():
    print("=" * 70)
    print("분할 매수·분할 매도 QA (engine/split.py)")
    print("=" * 70)
    test_params()
    test_plan()
    test_ledger()
    test_buy()
    test_sell()
    test_zones()
    test_exit_interaction()
    test_describe()
    if "--offline" not in sys.argv:
        test_ledger_drift()   # 로컬 DB 만 씁니다
        test_adopt()          # 로컬 DB 만 씁니다
        test_engine()         # 모의계좌(로컬 DB)만 씁니다 — 네트워크는 없습니다

    print("\n" + "=" * 70)
    print(f"통과 {len(PASS)} · 실패 {len(FAIL)}")
    if FAIL:
        for name in FAIL:
            print(f"  ✗ {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
