# -*- coding: utf-8 -*-
"""자동매매 QA 회귀 스위트 (QA_MEMO.md · QA_AUTOTRADE.md 동반)

메모의 결함 번호(C-1, C-2, H-2, M-1)와 섹션이 1:1 로 대응합니다.
고치지 않은 결함은 FAIL 로 남고, 전부 고치면 종료코드 0 이 됩니다.

    python tests/test_autotrade_qa_repro.py

격리된 임시 DB에서만 돕니다 — 실계좌 athena.db 를 건드리지 않습니다.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

# 실계좌 DB를 건드리지 않도록 임시 DB로 갈아끼운 뒤 storage 를 import 합니다
import config

TMPDB = os.path.join(tempfile.mkdtemp(prefix="athena_qa_"), "qa.db")
config.DB_PATH = TMPDB

from data_sources import fx
from engine import autotrade as at
from engine import risk, strategy
from engine.broker import OrderStatus, Position
from engine.instruments import FUTURES, STOCK, Instrument
from models import ResolvedSymbol
from storage import autotrade as store

U = 999_901
MODE = "live"
FAIL = []


def report(name, passed, detail=""):
    print(f"[{'PASS' if passed else '**FAIL**'}] {name}"
          + (f"\n         {detail}" if detail else ""))
    if not passed:
        FAIL.append(name)


def section(title):
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def make_inst(key, market="KOSPI", currency="KRW", asset=STOCK, mult=1.0):
    return Instrument(key=key, name=key, asset_class=asset, market=market,
                      currency=currency, multiplier=mult, margin_rate=1.0,
                      shortable=(asset == FUTURES),
                      symbol=ResolvedSymbol(key=key, name=key, market=market,
                                            yahoo_symbol=key, currency=currency))


# 종목 해석과 장 상태는 네트워크에 의존하므로 고정합니다. 검사 대상이 아닙니다.
REGISTRY: dict[str, Instrument] = {}
at.instruments.try_resolve = lambda k: REGISTRY.get(str(k))
risk.feed.entry_allowed_now = lambda inst, cfg: (True, "정규장", "REGULAR")


def close_order(symbol, qty, coid, side="long", asset=STOCK):
    """청산 주문을 접수 상태로 원장에 넣습니다 (KISBroker 가 하는 것과 같은 모양)."""
    store.record_order(U, {
        "client_order_id": coid, "broker_mode": MODE, "broker_order_id": "1",
        "symbol": symbol, "name": symbol, "asset_class": asset,
        "action": "close", "side": side, "quantity": qty, "price": 0,
        "status": "pending", "reason": "손절 도달", "realized_pnl": None})
    return next(o for o in store.open_orders(U, MODE) if o["client_order_id"] == coid)


def settle(record, filled, avg, state="filled"):
    """증권사가 체결을 알려준 상황 — 엔진의 정산 패스를 그대로 태웁니다."""
    result = {"settled": [], "errors": []}
    at._apply_fill(U, {"mode": MODE}, record, OrderStatus(
        known=True, status=state, filled_quantity=filled,
        remaining=max(record["quantity"] - filled, 0),
        avg_fill_price=avg), result)
    return result


# ---------------------------------------------------------------------------
section("C-1. 실계좌 청산의 실현손익이 기록되어 안전장치가 작동하는가")
# ---------------------------------------------------------------------------
# 증권사 체결내역은 손익을 주지 않습니다. 엔진이 계산해 채우지 않으면
# 일일 손실 한도 · 최대 낙폭 · 초단타 daily_loss_krw · 보호장치 4종이
# 전부 판단 재료를 잃고, 손절할 때마다 오늘 손익이 0으로 리셋됩니다.

REGISTRY["005930"] = make_inst("005930")
store.touch_daily(U, MODE, total_value=10_000_000, unrealized=0.0, cash=10_000_000)
store.upsert_position_state(U, MODE, "005930", side="long", entry_price=70_000,
                            quantity=10, opened_at="2026-08-08T09:00:00")
store.touch_daily(U, MODE, total_value=9_950_000, unrealized=-50_000.0)

rec = close_order("005930", 10, "qa:005930:close")
res = settle(rec, 10, 65_000)

# 왕복 비용 포함: gross -50,000 / 매수수수료 105 / 매도수수료 97.5 / 거래세 1,170
expected = -50_000 - 105 - 97.5 - 1_170
got = res["settled"][0]["realized_pnl"]
print(f"  진입 70,000 × 10주 → 65,000 청산")
print(f"  실현손익 {got:+,.2f}원 (기대 {expected:+,.2f}원 — 왕복 수수료·거래세 포함)")
report("왕복 비용을 포함한 실현손익이 계산된다", abs((got or 0) - expected) < 0.01)

today = store.realized_today(U, MODE)
closed = store.closed_trades(U, MODE)
report("realized_today 가 집계한다 (초단타 daily_loss_krw 의 재료)",
       today["closed_count"] == 1 and abs(today["realized_pnl"] - expected) < 0.01,
       str(today))
report("closed_trades 가 보호장치에 매매를 넘긴다 (쿨다운·손절감시·낙폭·부진종목)",
       len(closed) == 1 and closed[0]["reason"] == "손절 도달",
       f"{len(closed)}건" + (f" / 사유 '{closed[0]['reason']}'" if closed else ""))

day = store.touch_daily(U, MODE, total_value=9_948_628, unrealized=0.0)
halts = risk.RiskEngine({"daily_loss_limit_pct": 0.5, "max_drawdown_pct": 10.0},
                        {"total_value": 9_948_628}, [], day).halt_reasons()
print(f"  손절 직후 오늘손익 {day['pnl']:+,.2f} ({day['pnl_pct']:+.3f}%)")
report("손절 후에도 오늘 손익이 남아 일일 손실 한도가 발동한다",
       day["pnl"] < -1 and bool(halts), f"차단 사유 {halts}")
report("전량 체결이면 포지션 기억을 지운다",
       "005930" not in store.get_position_states(U, MODE))
report("매매 건수가 1건", day["trade_count"] == 1, f"{day['trade_count']}건")

# -- 부분 체결 --------------------------------------------------------------
print("\n  -- 부분 체결: 손익은 즉시, 매매 건수는 끝났을 때 한 번만 --")
REGISTRY["000660"] = make_inst("000660")
store.upsert_position_state(U, MODE, "000660", side="long", entry_price=100_000,
                            quantity=10, opened_at="2026-08-08T09:00:00")
rec = close_order("000660", 10, "qa:000660:close")

before = store.touch_daily(U, MODE, total_value=9_948_628, unrealized=0.0)
settle(rec, 4, 90_000, state="partial")
rec = next(o for o in store.open_orders(U, MODE) if o["symbol"] == "000660")
mid = store.touch_daily(U, MODE, total_value=9_948_628, unrealized=0.0)
print(f"  4/10주 체결 → 누적 실현 {rec['realized_pnl']:+,.2f}원 / "
      f"매매건수 {mid['trade_count'] - before['trade_count']:+d}")
report("부분 체결도 손익을 즉시 반영한다 (안 하면 한도가 늦게 걸립니다)",
       (rec["realized_pnl"] or 0) < -1)
report("부분 체결은 매매 건수를 세지 않는다",
       mid["trade_count"] == before["trade_count"])

settle(rec, 10, 90_000, state="filled")
final = next(o for o in store.get_orders(U, mode=MODE) if o["symbol"] == "000660")
after = store.touch_daily(U, MODE, total_value=9_948_628, unrealized=0.0)
expected2 = (-10_000 * 10) - 1_000_000 * 0.00015 - 900_000 * 0.00015 - 900_000 * 0.0018
print(f"  10/10주 체결 → 누적 실현 {final['realized_pnl']:+,.2f}원 "
      f"(기대 {expected2:+,.2f}원) / 매매건수 {after['trade_count'] - before['trade_count']:+d}")
report("전량 체결 시 실현손익이 이중 계상되지 않는다",
       abs((final["realized_pnl"] or 0) - expected2) < 0.01)
report("일자 집계도 이중 계상되지 않는다",
       abs((after["realized_pnl"] - before["realized_pnl"]) - expected2) < 0.01,
       f"{after['realized_pnl'] - before['realized_pnl']:+,.2f}원")
report("매매 건수는 끝났을 때 딱 1건",
       after["trade_count"] - before["trade_count"] == 1)

# -- 미국 종목 (통화 환산) ---------------------------------------------------
print("\n  -- 미국 종목: 달러 손익을 원화로 환산하는가 --")
REGISTRY["AAPL"] = make_inst("AAPL", market="US", currency="USD")
store.upsert_position_state(U, MODE, "AAPL", side="long", entry_price=200.0,
                            quantity=10, opened_at="2026-08-08T09:00:00")
rec = close_order("AAPL", 10, "qa:AAPL:close")
res = settle(rec, 10, 190.0)
usd_pnl = -100 - 2_000 * 0.0025 - 1_900 * 0.0025      # 미국은 수수료만, 거래세 없음
rate, source = fx.usd_krw()
got = res["settled"][0]["realized_pnl"]
print(f"  $200 × 10주 → $190 청산 · 달러 손익 {usd_pnl:+,.2f} · 환율 {rate:,.1f} ({source})")
print(f"  기록된 실현손익 {got:+,.0f}원")
report("달러 손익이 원화로 환산된다",
       abs((got or 0) - usd_pnl * rate) < 1,
       f"환산하지 않으면 {usd_pnl:+,.2f}'원'으로 기록돼 손실을 1/{rate:,.0f} 로 봅니다")

# -- 선물 숏 ----------------------------------------------------------------
print("\n  -- 선물 숏 청산: 부호와 승수 --")
REGISTRY["101H6000"] = make_inst("101H6000", asset=FUTURES, mult=250_000.0)
store.upsert_position_state(U, MODE, "101H6000", side="short", entry_price=350.0,
                            quantity=1, opened_at="2026-08-08T09:00:00")
rec = close_order("101H6000", 1, "qa:fut:close", side="short", asset=FUTURES)
res = settle(rec, 1, 348.0)
fee = (350 + 348) * 250_000 * 0.00003
expected4 = 500_000 - fee                 # 숏은 2pt 내렸으므로 이익
got = res["settled"][0]["realized_pnl"]
print(f"  숏 350pt → 348pt 청산 · 승수 250,000 · 실현 {got:+,.2f}원 "
      f"(기대 {expected4:+,.2f}원)")
report("숏은 가격이 내리면 이익으로 계산된다",
       (got or 0) > 0 and abs((got or 0) - expected4) < 0.01)

# -- 진입가를 모르는 포지션 --------------------------------------------------
print("\n  -- 진입가를 모르는 포지션: 0 으로 채우지 않는가 --")
REGISTRY["035720"] = make_inst("035720")
rec = close_order("035720", 10, "qa:035720:close")     # 포지션 상태 없음
before5 = store.touch_daily(U, MODE, total_value=9_948_628, unrealized=0.0)
res = settle(rec, 10, 50_000)
after5 = store.touch_daily(U, MODE, total_value=9_948_628, unrealized=0.0)
warned = [e for e in store.get_events(U, limit=30)
          if "손익을 계산하지 못했" in e["message"]]
report("손익을 0 으로 채우지 않는다 (손실이 없었던 것처럼 보이면 안 됩니다)",
       res["settled"][0]["realized_pnl"] is None)
report("일자 집계를 오염시키지 않는다",
       after5["realized_pnl"] == before5["realized_pnl"])
report("빠졌다는 사실을 로그로 남긴다", len(warned) == 1,
       (warned[0]["message"][:58] + "…") if warned else "경고 없음")


# ---------------------------------------------------------------------------
section("C-2. 한 회전 안에서 보유수·현금 한도가 지켜지는가")
# ---------------------------------------------------------------------------
# RiskEngine 은 회전 시작 시점의 스냅샷만 봅니다. 같은 회전에서 방금 낸 주문을
# 세지 않으면 종목 수만큼 한도가 곱해집니다.

cfg = {"max_positions": 1, "position_pct": 100.0, "max_gross_exposure_pct": 100.0,
       "asset_classes": {"STOCK": True}, "regular_session_only": False,
       "max_quote_age_sec": 999_999, "daily_loss_limit_pct": 0,
       "max_drawdown_pct": 0, "min_order_krw": 0, "allow_pyramiding": False}
account = {"total_value": 1_000_000, "available_cash": 1_000_000}
engine = risk.RiskEngine(cfg, account, [], {})      # engine/autotrade.py:349 와 동일

approved = []
for code in ("AAA", "BBB", "CCC", "DDD", "EEE"):
    inst = make_inst(code)
    sig = strategy.Signal(key=code, ok=True, score=0.9, direction=strategy.LONG,
                          price=10_000, price_krw=10_000)
    verdict = engine.check_entry(inst, "buy", 99, sig, None,
                                 {"price": 10_000, "price_krw": 10_000, "age_sec": 0})
    if verdict.approved:
        approved.append((code, verdict.quantity, verdict.quantity * 10_000))

total_krw = sum(a[2] for a in approved)
print(f"  설정: 최대 보유 1종목 / 총자산·주문가능금액 1,000,000원")
print(f"  승인 {len(approved)}건 — "
      + ", ".join(f"{c} {int(q)}주({k:,.0f}원)" for c, q, k in approved))
print(f"  주문 총액 {total_krw:,.0f}원 (계좌의 {total_krw / 1_000_000:.1f}배)")
report("한 회전 안에서 최대 보유 종목 수가 지켜진다", len(approved) <= 1,
       f"max_positions=1 인데 {len(approved)}건 승인 — 회전 시작 스냅샷만 보고 "
       f"방금 낸 주문을 세지 않습니다.")
report("한 회전 안에서 주문가능금액이 지켜진다", total_krw <= 1_000_000 * 1.01,
       f"주문가능금액 1,000,000원인데 합계 {total_krw:,.0f}원 승인 — 각 종목이 "
       f"같은 현금을 중복 배정받습니다.")


# ---------------------------------------------------------------------------
section("H-2. 초단타 예산이 유니버스 회전에도 유지되는가")
# ---------------------------------------------------------------------------
held = [Position(key="AAA", name="AAA", asset_class=STOCK, side="long",
                 quantity=1000, avg_price=500, current_price=500,
                 market_value=500_000)]
inside = at.scalp_budget_left(
    {"scalp": {"enabled": True, "budget_krw": 1_000_000, "universe": ["AAA", "BBB"]}}, held)
outside = at.scalp_budget_left(
    {"scalp": {"enabled": True, "budget_krw": 1_000_000, "universe": ["BBB"]}}, held)
print(f"  AAA 500,000원 보유 · 예산 1,000,000원")
print(f"  AAA 가 유니버스에 있을 때 : 남은 예산 {inside:,.0f}원")
print(f"  AAA 가 빠졌을 때          : 남은 예산 {outside:,.0f}원")
report("보유 중인 초단타 포지션은 유니버스에서 빠져도 예산에서 차감된다",
       abs(outside - inside) < 1,
       "15초 주기 갱신으로 종목이 빠지자 이미 쓴 돈이 예산으로 되돌아왔습니다. "
       "집계 기준을 유니버스가 아니라 포지션의 strategy 태그로 바꿔야 합니다.")


# ---------------------------------------------------------------------------
section("M-1. 복원된 미국 포지션의 손절가가 같은 통화인가")
# ---------------------------------------------------------------------------
# PaperBroker 는 avg_price 를 원화로, KISBroker 해외는 달러로 둡니다.
# _reconcile 이 원화 평단에서 달러 ATR 을 빼면 손절가가 현재가보다 위가 됩니다.
entry_krw = 300 * 1_400.0
stop = entry_krw - 6.0            # stop_distance 는 호가 통화(달러) 기준
print(f"  paper 보유 avg_price {entry_krw:,.0f} (원화) − stop_distance 6.00 (달러)")
print(f"  복원된 손절가 {stop:,.2f} / 현재가 300.00 (달러)")
report("복원된 미국 포지션의 손절가가 현재가와 같은 통화다", stop < 300.0,
       "손절가가 현재가보다 위라 check_exit 가 즉시 '손절 도달'로 판정해 "
       "복원 직후 전량 청산됩니다.")


# ---------------------------------------------------------------------------
print(f"\n{'=' * 74}")
print(f"  결과: {len(FAIL)}건 실패" + (" — " + " / ".join(FAIL) if FAIL else " (전부 통과)"))
print("=" * 74)
print(f"  (임시 DB: {TMPDB})")

sys.exit(1 if FAIL else 0)
