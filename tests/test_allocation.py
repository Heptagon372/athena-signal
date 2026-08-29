# -*- coding: utf-8 -*-
"""
시장 분리 운용 QA (한국 · 미국)
--------------------------------
배분은 **돈을 나누는** 기능이라, "나뉜다"보다 "한쪽이 남의 몫까지 쓰지 않는다"를
많이 검사합니다. 실제로 고치려던 사고가 이것이었습니다.

    총자산 26만원 계좌에서 먼저 신호가 뜬 미국 종목이 현금을 다 쓰고,
    뒤에 온 종목은 전부 "주문가능금액" 한도에 걸려 1주씩만 샀습니다.

    python tests/test_allocation.py     # 네트워크·서버 없이 판단 로직만
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from engine import allocation, strategy
from engine.broker import Position
from engine.instruments import STOCK, Instrument
from models import ResolvedSymbol
from storage import autotrade as store

results = []


def check(name, passed, detail=""):
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


def section(title):
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def fake_stock(key, name="테스트", market="KOSPI", currency="KRW") -> Instrument:
    return Instrument(
        key=key, name=name, asset_class=STOCK, market=market, currency=currency,
        multiplier=1.0, margin_rate=1.0, shortable=False,
        symbol=ResolvedSymbol(key=key, name=name, market=market,
                              yahoo_symbol=key, currency=currency))


def pos(key, value_krw, pnl=0.0):
    return Position(key=key, name=key, asset_class=STOCK, side="long",
                    quantity=1, avg_price=value_krw, current_price=value_krw,
                    market_value=value_krw, unrealized_pnl=pnl)


CFG = {"market_split": True, "kr_alloc_pct": 40, "us_alloc_pct": 40,
       "kr_max_positions": 1, "us_max_positions": 1}
ACC = {"total_value": 1_000_000.0, "available_cash": 900_000.0}


section("종목을 보고 시장을 가른다")

check("한국은 6자리 숫자", allocation.scope_of("005930") == allocation.KR)
check("미국은 영문 티커", allocation.scope_of("AAPL") == allocation.US)
check("파생 단축코드는 한국", allocation.scope_of("101S3000") == allocation.KR)
check("Instrument 는 market 을 그대로 믿는다",
      allocation.scope_of(fake_stock("TSLA", market="US")) == allocation.US)
check("포지션도 코드로 갈린다", allocation.scope_of(pos("068270", 10_000)) == allocation.KR)


section("배분 — 총자산을 나눠 가진다")

kr = allocation.budget(CFG, allocation.KR, ACC, [])
check("배분한 만큼이 예산", kr["budget_krw"] == 400_000, f"{kr['budget_krw']:,.0f}원")
check("남은 20%는 어느 쪽도 쓰지 않는다",
      sum(allocation.alloc_pcts(CFG).values()) == 80)

over = allocation.alloc_pcts({"kr_alloc_pct": 60, "us_alloc_pct": 60})
check("합이 100%를 넘으면 비율을 지켜서 축소",
      abs(over["KR"] - 50) < 1e-9 and abs(over["US"] - 50) < 1e-9,
      f"{over['KR']:g} / {over['US']:g}")

off = allocation.budget({**CFG, "us_enabled": False}, allocation.KR, ACC, [])
check("꺼진 시장의 몫이 켜진 쪽으로 넘어가지 않는다", off["budget_krw"] == 400_000,
      "한국은 여전히 40%")

used = allocation.budget(CFG, allocation.US, ACC, [pos("AAPL", 300_000)])
check("이미 산 만큼 예산에서 빠진다", used["room_krw"] == 100_000,
      f"여유 {used['room_krw']:,.0f}원")
check("주문가능현금이 예산보다 크면 예산이 이긴다", used["cash_krw"] == 100_000)

thin = allocation.budget(CFG, allocation.US, {"total_value": 1_000_000,
                                              "available_cash": 30_000}, [])
check("주문가능현금이 예산보다 작으면 현금이 이긴다", thin["cash_krw"] == 30_000,
      "통합증거금 계좌는 결제일까지 현금이 묶입니다")


section("게이트 — 신규 진입을 막을 이유")

check("꺼진 시장은 못 산다",
      "꺼져" in allocation.gate({**CFG, "us_enabled": False}, allocation.US, ACC, []))
check("상한에 닿으면 못 산다",
      "상한" in allocation.gate(CFG, allocation.KR, ACC, [pos("005930", 100_000)]))
check("같은 종목을 다시 볼 때는 상한이 걸리지 않는다",
      not allocation.gate(CFG, allocation.KR, ACC, [pos("005930", 100_000)],
                          key="005930"))
# 종목 상한이 아니라 **예산**이 막는 경우 — 상한을 넉넉히 두고 봅니다
roomy = {**CFG, "us_max_positions": 5}
check("예산을 다 쓰면 못 산다",
      "예산" in allocation.gate(roomy, allocation.US, ACC, [pos("AAPL", 400_000)]))
check("한쪽이 막혀도 다른 쪽은 산다",
      not allocation.gate(CFG, allocation.US, ACC, [pos("005930", 400_000)]),
      "한국이 예산을 다 써도 미국은 자기 몫이 있습니다")
check("분리를 끄면 게이트가 없다",
      not allocation.gate({**CFG, "market_split": False}, allocation.KR, ACC,
                          [pos("005930", 999_000)]))


section("전체 종목 상한은 두 시장 상한의 합")

# 이게 어긋나면 미국을 한 종목 들고 있는 동안 한국이 통째로 막힙니다
# (리스크 엔진의 상한은 계좌 전체 보유 수를 셉니다).
check("합으로 넓혀준다",
      allocation.global_cfg({**CFG, "max_positions": 1})["max_positions"] == 2)
check("분리를 끄면 손대지 않는다",
      allocation.global_cfg({**CFG, "market_split": False,
                             "max_positions": 1})["max_positions"] == 1)
check("시장 설정 사본에도 반영된다",
      allocation.market_cfg({**CFG, "max_positions": 1},
                            allocation.KR)["max_positions"] == 2)


section("시장별 설정 덮어쓰기")

over_cfg = {**CFG, "risk_per_trade_pct": 1.0, "kr_risk_per_trade_pct": 3.0,
            "position_pct": 20.0}
check("정한 값만 덮는다",
      allocation.market_cfg(over_cfg, allocation.KR)["risk_per_trade_pct"] == 3.0)
check("0은 '안 정했다'로 읽는다",
      allocation.market_cfg(over_cfg, allocation.US)["risk_per_trade_pct"] == 1.0)
check("공통값은 그대로 남는다",
      allocation.market_cfg(over_cfg, allocation.KR)["position_pct"] == 20.0)
check("원본 설정을 건드리지 않는다", over_cfg["risk_per_trade_pct"] == 1.0)


section("수량 — 배분이 실제로 주수를 정한다")

# 이 검사가 이 기능의 존재 이유입니다. 같은 종목·같은 신호인데 배분만 바꾸면
# 살 수 있는 주수가 달라져야 합니다.
inst = fake_stock("005930", "삼성전자")
sig = strategy.Signal(key="005930", ok=True, direction=strategy.LONG,
                      price=10_000.0, price_krw=10_000.0, atr=200.0)
# 종목당 비중이 수량을 정하는 조건으로 잡습니다 — 배분이 총자산을 갈아끼우면
# 비중 한도도 그만큼 줄어드는지가 이 검사의 요지입니다.
size_cfg = {"risk_per_trade_pct": 5.0, "position_pct": 20.0, "atr_stop_mult": 2.0,
            "reward_risk": 2.0, "min_order_krw": 0}

full = strategy.plan_entry(inst, sig, size_cfg, ACC)
half = strategy.plan_entry(
    inst, sig, size_cfg,
    allocation.scoped_account({**CFG, "kr_alloc_pct": 50, "us_alloc_pct": 50},
                              allocation.KR, ACC, []))
check("배분 50%면 수량도 절반",
      full.quantity == 20 and half.quantity == 10,
      f"전액 {full.quantity:g}주 → 절반 배분 {half.quantity:g}주")
check("무엇이 수량을 정했는지 남는다",
      half.sizing.get("bound_by") == "종목 비중 상한"
      and half.sizing["total_value_krw"] == 500_000,
      f"{half.sizing.get('bound_by')} · 기준 총자산 {half.sizing['total_value_krw']:,.0f}원")

# 미국이 현금을 다 쓴 상황 — 예전에는 한국이 1주도 못 샀습니다
squeezed = {"total_value": 1_000_000.0, "available_cash": 20_000.0}
before = strategy.plan_entry(inst, sig, size_cfg, squeezed)
check("현금이 마르면 배분과 무관하게 그만큼만 산다",
      before.quantity == 1, f"{before.quantity:g}주")


section("기본값")

check("최대 보유 종목 기본값은 1", store.DEFAULT_CONFIG["max_positions"] == 1)
check("시장별 상한 기본값도 1",
      store.DEFAULT_CONFIG["kr_max_positions"] == 1
      and store.DEFAULT_CONFIG["us_max_positions"] == 1)
check("분리 운용이 기본으로 켜져 있다", store.DEFAULT_CONFIG["market_split"] is True)
check("기본값은 한 곳에서만 정의된다",
      store.DEFAULT_CONFIG["kr_alloc_pct"] == allocation.DEFAULTS["kr_alloc_pct"])


section("청산비용 추정 — 브로커마다 다른 통화에 속지 않는가")

# 실측 사고: 856,348원짜리 미국 포지션의 청산 비용이 2,964,634원으로 잡혔습니다.
# PaperBroker 는 current_price 에 **원화**를 넣는데(feed.price_krw), 그걸 달러로
# 보고 환율을 한 번 더 곱했기 때문입니다. 두 브로커가 같은 단위로 채우는 값은
# market_value(원화) 뿐이라, 반드시 그것으로 계산해야 합니다.
from engine import autotrade as at

us = Position(key="AAPL", name="Apple", asset_class=STOCK, side="long",
              quantity=2, avg_price=429_213, current_price=428_143,
              market_value=856_348, unrealized_pnl=-2_079)
cost_us = at._exit_cost_krw(us)
check("미국 종목 청산비용이 평가금액보다 크지 않다",
      0 < cost_us < 856_348 * 0.01, f"{cost_us:,.0f}원 (평가 856,348원)")

kr = Position(key="005930", name="삼성전자", asset_class=STOCK, side="long",
              quantity=4, avg_price=262_039, current_price=262_000,
              market_value=1_048_000, unrealized_pnl=-157)
cost_kr = at._exit_cost_krw(kr)
check("한국 종목은 매도 거래세까지 포함한다",
      cost_kr > 1_048_000 * 0.0015, f"{cost_kr:,.0f}원 (수수료 + 거래세)")

check("평가금액을 모르면 0 (어림하지 않는다)",
      at._exit_cost_krw(Position(key="AAPL", name="Apple", asset_class=STOCK,
                                 side="long", quantity=2, avg_price=1,
                                 market_value=0)) == 0.0)


section("수익률 곡선 — 오늘 기점, 수수료 차감")

USER = 999_901
store.init()
with store._conn() as conn:
    conn.execute("DELETE FROM at_curve WHERE user_id = ?", (USER,))

store.record_curve(USER, "paper", [
    {"scope": "ALL", "base_krw": 1_000_000, "realized_krw": 0, "unreal_krw": 0},
    {"scope": "KR", "base_krw": 400_000, "realized_krw": 0, "unreal_krw": 0},
])
store.record_curve(USER, "paper", [
    {"scope": "ALL", "base_krw": 1_000_000, "realized_krw": 4_000,
     "unreal_krw": 8_000, "exit_cost_krw": 2_000},
    {"scope": "KR", "base_krw": 400_000, "realized_krw": 4_000,
     "unreal_krw": 0, "exit_cost_krw": 0},
], min_gap_sec=0)

curve = store.get_curve(USER, "paper", days=1)
first = curve["series"]["ALL"][0]
last = curve["series"]["ALL"][-1]
check("첫 표본은 반드시 0%", first["pct"] == 0 and first["pnl"] == 0)
check("실현 + 평가 − 청산비용",
      last["pnl"] == 10_000, f"{last['pnl']:,.0f}원 (4,000 + 8,000 − 2,000)")
check("기준 자본으로 나눈다", abs(last["pct"] - 1.0) < 1e-9, f"{last['pct']}%")
check("시장마다 자기 기준 자본을 쓴다",
      abs(curve["series"]["KR"][-1]["pct"] - 1.0) < 1e-9,
      "한국은 4,000원 / 40만원 = 1%")

blocked = store.record_curve(USER, "paper", [{"scope": "ALL", "base_krw": 1}],
                             min_gap_sec=600)
check("너무 자주 남기지 않는다", blocked is False)

with store._conn() as conn:
    conn.execute("DELETE FROM at_curve WHERE user_id = ?", (USER,))

section("탐색 범위 다중 선택 · 순환 회전 수")

from data_sources import universe as universe_mod

check("빈 값은 범위 지정 없음", universe_mod.normalize_pools("") == []
      and universe_mod.normalize_pools(None) == [])
check("예전 설정(문자열 하나)도 그대로 동작",
      universe_mod.normalize_pools("KOSPI200") == ["KOSPI200"])
check("콤마로 이어붙인 값도 푼다",
      universe_mod.normalize_pools("kospi200, sp500") == ["KOSPI200", "SP500"])
check("목록은 중복만 걷어낸다",
      universe_mod.normalize_pools(["KOSPI200", "KOSDAQ150", "KOSPI200"])
      == ["KOSPI200", "KOSDAQ150"])
# 오타를 조용히 지우면 왜 시장 전체가 나오는지 알 수 없습니다 — 남겨서
# screener.scan 이 "알 수 없는 탐색 범위" 로 경고하게 둡니다.
check("모르는 키도 버리지 않는다",
      universe_mod.normalize_pools(["KOSPI200", "없는범위"]) == ["KOSPI200", "없는범위"])
check("설명은 아는 것만 모은다",
      [d["key"] for d in universe_mod.describe_many(["KOSPI200", "없는범위", "SP500"])]
      == ["KOSPI200", "SP500"])
check("기본값은 목록", store.DEFAULT_CONFIG["auto_universe_pool"] == [])

ROUND_USER = 999_950
with store._conn() as conn:
    conn.execute("DELETE FROM at_config WHERE user_id = ?", (ROUND_USER,))
check("행이 없어도 1부터 센다", store.bump_scan_round(ROUND_USER) == 1)
check("계속 올라간다", store.bump_scan_round(ROUND_USER) == 2)
check("설정에 실려 나온다", store.get_config(ROUND_USER)["scan_rounds"] == 2)
# 회전 수는 컬럼입니다. 설정 JSON 에 새면 save_config 가 덮어써 사라집니다.
store.save_config(ROUND_USER, {"auto_universe_size": 7})
after = store.get_config(ROUND_USER)
check("설정을 저장해도 회전 수가 지워지지 않는다",
      after["scan_rounds"] == 2 and after["auto_universe_size"] == 7)
with store._conn() as conn:
    stored = conn.execute("SELECT config FROM at_config WHERE user_id = ?",
                          (ROUND_USER,)).fetchone()["config"]
    check("회전 수가 설정 JSON 으로 새지 않는다", "scan_rounds" not in stored)
    conn.execute("DELETE FROM at_config WHERE user_id = ?", (ROUND_USER,))


section("한 회전 안에서 시장별 상한이 지켜지는가")

# 실계좌에서 실제로 벌어진 일입니다 — 미국 상한 4종목 · 이미 3종목 보유인데
# 한 회전에 후보 4개가 9초 사이에 전부 통과해 7종목이 됐습니다.
# 넷 다 "아직 3종목이니 한 자리 남았다"는 같은 스냅샷을 봤기 때문입니다.
from engine import feed as _feed, risk as _risk


class _FixedQuote:
    """시세 조회를 고정합니다 — 장 상태·신선도 때문에 검사가 흔들리면 안 됩니다."""

    def __enter__(self):
        self._orig = _risk.feed.entry_allowed_now
        _risk.feed.entry_allowed_now = lambda inst, cfg: (True, "", "regular")
        return self

    def __exit__(self, *a):
        _risk.feed.entry_allowed_now = self._orig


def us_stock(key):
    return Instrument(key=key, name=key, asset_class=STOCK, market="US",
                      currency="USD", multiplier=1.0, margin_rate=1.0,
                      shortable=False,
                      symbol=ResolvedSymbol(key=key, name=key, market="US",
                                            yahoo_symbol=key, currency="USD"))


tick_cfg = {**CFG, "us_alloc_pct": 30, "kr_alloc_pct": 70, "us_max_positions": 4,
            "kr_max_positions": 10, "position_pct": 50, "min_order_krw": 0,
            "max_order_krw": 0, "max_gross_exposure_pct": 100.0,
            "asset_classes": {"STOCK": True}, "max_quote_age_sec": 999_999,
            "regular_session_only": False, "daily_loss_limit_pct": 0,
            "max_drawdown_pct": 0, "allow_pyramiding": False}
tick_cfg = allocation.global_cfg(tick_cfg)
tick_acc = {"total_value": 1_000_000.0, "available_cash": 500_000.0}
held3 = [pos("WBD", 40_000), pos("NWS", 49_000), pos("NWSA", 43_000)]

with _FixedQuote():
    engine = _risk.RiskEngine(tick_cfg, tick_acc, list(held3), {})
    passed = []
    for code in ("PYPL", "FOXA", "FOX", "APA"):
        inst = us_stock(code)
        blocked = allocation.gate(tick_cfg, allocation.US, engine.account,
                                  engine.positions, key=code)
        if blocked:
            continue
        sig2 = strategy.Signal(key=code, ok=True, score=0.9,
                               direction=strategy.LONG,
                               price=60.0, price_krw=84_000.0)
        plan2 = strategy.EntryPlan(ok=True, quantity=1, price=60.0,
                                   stop_price=45.0)
        v = engine.check_entry(inst, "buy", 1, sig2, plan2,
                               {"price": 60.0, "price_krw": 84_000.0, "age_sec": 0})
        if v.approved:
            passed.append(code)

check("이미 3종목 · 상한 4종목이면 한 회전에 1종목만 통과",
      len(passed) == 1, f"통과 {passed}")
check("상한을 넘긴 시점부터는 시장 게이트가 막는다",
      "상한" in allocation.gate(tick_cfg, allocation.US, engine.account,
                                engine.positions, key="TSLA"),
      allocation.gate(tick_cfg, allocation.US, engine.account,
                      engine.positions, key="TSLA"))
check("승인한 만큼 주문가능금액이 줄어든다",
      engine.account["available_cash"] < tick_acc["available_cash"],
      f"{tick_acc['available_cash']:,.0f}원 → {engine.account['available_cash']:,.0f}원")
# 원본 계좌 dict 는 건드리지 않습니다 — 회전 결과·화면이 같이 들고 있습니다
check("원본 계좌 스냅샷은 그대로", tick_acc["available_cash"] == 500_000.0)


section("AI 추적과 초단타는 서로의 타이머를 덮지 않는다")

# 실측 사고: 둘이 같은 dict 를 써서, 페니 초단타가 15초마다 시각을 덮어쓰면
# AI 추적은 '지금 − 마지막' 이 갱신 주기(기본 30분)를 영원히 넘지 못했습니다.
# 추적은 켜져 있는데 한 번도 돌지 않았습니다.
import time as _time

at._last_universe_refresh.pop(7, None)
at._last_scalp_refresh.pop(7, None)
at._last_universe_refresh[7] = _time.time() - 4000        # 30분 훨씬 넘게 지남
at._last_scalp_refresh[7] = _time.time()                  # 초단타는 방금 돌았음
check("초단타가 방금 돌아도 AI 추적 타이머는 그대로",
      _time.time() - at._last_universe_refresh[7] > 1800,
      "AI 추적은 다음 sweep 에서 돕니다")

at.reset_universe_timer(7)
check("추적 시작은 AI 타이머만 지운다",
      7 not in at._last_universe_refresh and 7 in at._last_scalp_refresh)
at._last_scalp_refresh.pop(7, None)


failed = [(n, det) for n, ok, det in results if not ok]
print(f"\n{'-' * 68}\n  {len(results)}건 중 실패 {len(failed)}건")
for n, det in failed:
    print(f"    FAIL: {n} {det}")
sys.exit(1 if failed else 0)
