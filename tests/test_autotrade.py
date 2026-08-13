# -*- coding: utf-8 -*-
"""
자동매매 QA 스위트
------------------
자동매매는 돈이 나가는 기능이라, "돌아간다"가 아니라 "잘못된 상황에서 멈춘다"를
확인해야 합니다. 그래서 정상 경로만큼 **거부 경로**를 많이 검사합니다.

    python tests/test_autotrade.py            # 전체 (서버가 떠 있으면 API 까지)
    python tests/test_autotrade.py --offline  # 네트워크·서버 없이 계산 로직만

offline 검사는 실제 시세를 쓰지 않습니다. 가짜 시세를 주입해 결정 로직만 봅니다
(시세가 흔들려서 테스트가 깜빡이면 아무도 안 믿게 됩니다).
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from engine import broker, feed, instruments, risk, strategy
from engine.instruments import ETF, FUTURES, OPTION, STOCK, Instrument
from models import ResolvedSymbol
from storage import autotrade as store
from storage import paper

# 서버 주소는 --base 로 바꿀 수 있습니다 (다른 포트에서 띄워둔 경우)
BASE = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--base=")),
            "http://localhost:8000")
TEST_USER = 999_900          # 실제 계정과 겹치지 않는 테스트 전용 id
results = []


def check(category, name, passed, detail=""):
    results.append((category, name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


def section(title):
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


# ---------------------------------------------------------------------------
# 테스트용 가짜 데이터
# ---------------------------------------------------------------------------

def fake_stock(key="TEST01", name="테스트종목", market="KOSPI", asset=STOCK) -> Instrument:
    return Instrument(
        key=key, name=name, asset_class=asset, market=market, currency="KRW",
        multiplier=1.0, margin_rate=1.0, shortable=False,
        symbol=ResolvedSymbol(key=key, name=name, market=market,
                              yahoo_symbol=key, currency="KRW"),
    )


def trend_bars(n=140, start=10_000.0, drift=0.006, sigma=0.012, seed=7) -> pd.DataFrame:
    """추세가 있는 합성 일봉 (시드 고정 랜덤워크). drift 부호로 상승/하락.

    규칙적인 톱니 시계열을 쓰면 안 됩니다. 국면 판별기(variance ratio / Hurst)가
    주기성을 '평균회귀'로 읽어, 아무리 올라가도 추세 신호가 나오지 않습니다.
    실제 시장에 가까운 랜덤워크 + 드리프트를 쓰되, 시드를 고정해 결과를
    재현 가능하게 만듭니다.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(rng.normal(drift, sigma, n)))
    rows = []
    for i, close in enumerate(closes):
        high = close * (1 + abs(rng.normal(0, sigma / 2)))
        low = close * (1 - abs(rng.normal(0, sigma / 2)))
        rows.append({
            "date": datetime(2025, 1, 1) + timedelta(days=i),
            "open": (high + low) / 2, "high": max(high, close),
            "low": min(low, close), "close": close, "volume": 1_000_000 + i * 1000,
        })
    return pd.DataFrame(rows).set_index("date")


def fake_quote(price):
    return {"price": price, "price_krw": price, "age_sec": 0.0,
            "market_open": True, "session": "테스트", "prev_close": price}


class FakeFeed:
    """시세·종목 해석을 가짜로 바꿔 네트워크를 끊습니다.

    feed 는 모든 모듈이 같은 객체를 참조하므로 속성만 갈아끼우면 전역에 적용됩니다.
    """

    def __init__(self, prices: dict, catalog: dict = None):
        self.prices = prices
        self.catalog = catalog or {}
        self._saved = None

    def __enter__(self):
        self._saved = (feed.quote, feed.is_tradable_now, instruments.try_resolve,
                       feed.entry_allowed_now)
        feed.quote = lambda inst, max_age=20.0: (
            fake_quote(self.prices[inst.key]) if inst.key in self.prices else None)
        feed.is_tradable_now = lambda inst, regular_only=True: (True, "테스트 개장")
        # 진입 경로는 별도 게이트를 씁니다 (프리마켓·정규장만). 청산용
        # is_tradable_now 만 가짜로 열면 진입 검사가 전부 '개장 대기'로 빠집니다.
        feed.entry_allowed_now = lambda inst, cfg=None: (True, "테스트 개장", "REGULAR")
        if self.catalog:
            original = self._saved[2]
            instruments.try_resolve = lambda q: self.catalog.get(str(q), original(q))
            broker.instruments.try_resolve = instruments.try_resolve
        return self

    def __exit__(self, *exc):
        (feed.quote, feed.is_tradable_now, instruments.try_resolve,
         feed.entry_allowed_now) = self._saved
        broker.instruments.try_resolve = self._saved[2]


# ---------------------------------------------------------------------------
# 1. 인스트루먼트 (자산군 추상화)
# ---------------------------------------------------------------------------

def test_instruments():
    section("1. 인스트루먼트 — 주식 / ETF / 선물 / 옵션")

    fut = instruments.parse_derivative("101H6000")
    check("계약", "코스피200 선물 코드 해석",
          fut and fut.asset_class == FUTURES and fut.multiplier == 250_000,
          f"{fut.name} 승수 {fut.multiplier:,.0f}" if fut else "해석 실패")
    check("계약", "선물 만기 = 결제월 두 번째 목요일",
          fut.expiry == date(2026, 3, 12), str(fut.expiry))
    check("계약", "선물 명목금액 = 지수 × 승수 × 계약수",
          fut.notional(350, 2) == 350 * 250_000 * 2, f"{fut.notional(350, 2):,.0f}원")
    check("계약", "선물 증거금은 명목의 일부",
          0 < fut.margin_required(350, 1, "buy") < fut.notional(350, 1),
          f"{fut.margin_required(350, 1, 'buy'):,.0f}원 (명목 {fut.notional(350, 1):,.0f}원)")
    check("계약", "선물 호가단위 0.05", fut.round_price(350.037) == 350.05,
          str(fut.round_price(350.037)))

    call = instruments.parse_derivative("201H6320")
    put = instruments.parse_derivative("301H6320")
    check("계약", "콜/풋 옵션 구분",
          call.right == "CALL" and put.right == "PUT" and call.asset_class == OPTION,
          f"{call.name} / {put.name}")
    check("계약", "옵션 프리미엄 구간별 호가단위",
          call.tick_size(3.2) == 0.01 and call.tick_size(12.0) == 0.05)
    check("계약", "옵션 매수 필요자금 = 프리미엄 전액",
          call.margin_required(4.5, 2, "buy") == 4.5 * 250_000 * 2,
          f"{call.margin_required(4.5, 2, 'buy'):,.0f}원")
    check("계약", "옵션 매도는 증거금이 훨씬 크다",
          call.margin_required(4.5, 1, "sell") > call.margin_required(4.5, 1, "buy") * 5,
          f"{call.margin_required(4.5, 1, 'sell'):,.0f}원")
    check("계약", "파생은 숏 진입 가능 / 현물은 불가",
          fut.shortable and call.shortable and not fake_stock().shortable)
    check("계약", "주식코드는 파생으로 해석되지 않음",
          instruments.parse_derivative("005930") is None)

    stock = fake_stock()
    etf = fake_stock(key="069500", name="KODEX 200", asset=ETF)
    fee_s, tax_s = stock.costs("sell", 1_000_000)
    fee_e, tax_e = etf.costs("sell", 1_000_000)
    check("비용", "주식 매도에는 증권거래세", tax_s > 0, f"{tax_s:,.0f}원")
    check("비용", "ETF 매도는 거래세 면제", tax_e == 0 and fee_e > 0,
          f"수수료 {fee_e:,.0f}원 / 세금 {tax_e:,.0f}원")
    check("비용", "선물 수수료는 명목금액 기준으로 매우 낮다",
          fut.costs("buy", 87_500_000)[0] < 87_500_000 * 0.0001,
          f"{fut.costs('buy', 87_500_000)[0]:,.0f}원")
    check("계약", "국내 주식 호가단위 계단", stock.round_price(74_321) == 74_300,
          str(stock.round_price(74_321)))
    check("계약", "국내는 소수점 수량 불가", stock.round_quantity(3.7) == 3.0)


# ---------------------------------------------------------------------------
# 2. 신호 · 사이징
# ---------------------------------------------------------------------------

def test_strategy():
    section("2. 전략 — 신호 · 포지션 사이징")

    cfg = dict(store.DEFAULT_CONFIG, intraday_weight=0.0, use_news=False, entry_score=0.3)
    up, down = trend_bars(drift=0.006), trend_bars(drift=-0.006)
    inst = fake_stock()

    sig_up = strategy.evaluate(inst, cfg, bars_daily=up,
                               quote=fake_quote(float(up["close"].iloc[-1])),
                               allow_fetch=False)
    sig_dn = strategy.evaluate(inst, cfg, bars_daily=down,
                               quote=fake_quote(float(down["close"].iloc[-1])),
                               allow_fetch=False)
    check("신호", "상승 추세 → 롱 신호",
          sig_up.ok and sig_up.direction == strategy.LONG, f"점수 {sig_up.score:+.3f}")
    check("신호", "하락 추세 → 진입 안 함 (현물은 숏 불가)",
          sig_dn.ok and sig_dn.direction == strategy.FLAT, f"점수 {sig_dn.score:+.3f}")

    fut = instruments.parse_derivative("101H6000")
    sig_short = strategy.evaluate(fut, dict(cfg, allow_short=True), bars_daily=down,
                                  quote=fake_quote(float(down["close"].iloc[-1])),
                                  allow_fetch=False)
    check("신호", "파생 + 숏 허용 → 숏 신호",
          sig_short.direction == strategy.SHORT, f"점수 {sig_short.score:+.3f}")

    short_bars = trend_bars(n=10)
    check("신호", "봉이 부족하면 신호를 만들지 않음",
          not strategy.evaluate(inst, cfg, bars_daily=short_bars,
                                quote=fake_quote(10_000), allow_fetch=False).ok)
    check("신호", "현재가가 없으면 신호 없음",
          not strategy.evaluate(inst, cfg, bars_daily=up, quote=None,
                                allow_fetch=False).ok)
    check("신호", "ATR 이 계산됨", sig_up.atr > 0, f"ATR {sig_up.atr:,.0f} ({sig_up.atr_pct:.2f}%)")

    # --- 사이징 ---
    account = {"total_value": 10_000_000, "available_cash": 10_000_000}
    plan = strategy.plan_entry(inst, sig_up, dict(cfg, risk_per_trade_pct=1.0,
                                                  position_pct=100, max_order_krw=0,
                                                  min_order_krw=0), account)
    risk_amount = abs(sig_up.price - plan.stop_price) * plan.quantity
    check("사이징", "위험 예산(1%)에 맞춰 수량 산출",
          plan.ok and abs(risk_amount - 100_000) <= sig_up.price * 1.0,
          f"{plan.quantity:g}주 · 손절까지 위험 {risk_amount:,.0f}원 (예산 100,000원)")
    check("사이징", "손절가는 진입가 아래", plan.stop_price < sig_up.price,
          f"{plan.stop_price:,.0f} < {sig_up.price:,.0f}")
    check("사이징", "목표가는 진입가 위", plan.target_price > sig_up.price)

    capped = strategy.plan_entry(inst, sig_up, dict(cfg, position_pct=5,
                                                    min_order_krw=0), account)
    check("사이징", "종목 비중 상한이 수량을 줄인다",
          capped.quantity < plan.quantity and capped.notional_krw <= 500_000 * 1.01,
          f"{capped.quantity:g}주 / {capped.notional_krw:,.0f}원 (상한 500,000원)")

    broke = strategy.plan_entry(inst, sig_up, cfg,
                                {"total_value": 10_000, "available_cash": 10_000})
    check("사이징", "잔고가 부족하면 진입 계획 실패", not broke.ok, broke.reason)

    # --- 소액 계좌: 리스크 예산 < 1주 손절 위험 (min_one_unit) ---
    dist = sig_up.price - plan.stop_price
    cfg_small = dict(cfg, risk_per_trade_pct=1.0, position_pct=100,
                     max_order_krw=0, min_order_krw=0)
    tiny = {"total_value": dist * 50, "available_cash": sig_up.price * 1.5}
    no_flag = strategy.plan_entry(inst, sig_up, cfg_small, tiny)
    check("사이징", "소액 계좌 — 예산 미달 사유에 해법 안내",
          not no_flag.ok and "위험 예산" in no_flag.reason, no_flag.reason)
    one = strategy.plan_entry(inst, sig_up, dict(cfg_small, min_one_unit=True), tiny)
    check("사이징", "min_one_unit — 현금이 되면 최소 1주 허용",
          one.ok and one.quantity == 1,
          f"{one.quantity:g}주 · 실제 위험 {one.risk_krw:,.0f}원 (예산 {dist * 0.5:,.0f}원 초과 허용)")
    poor = strategy.plan_entry(inst, sig_up, dict(cfg_small, min_one_unit=True),
                               {"total_value": dist * 50,
                                "available_cash": sig_up.price * 0.5})
    check("사이징", "min_one_unit — 현금이 안 되면 여전히 거부", not poor.ok, poor.reason)
    check("계약", "미국 주식도 정수 수량 (KIS 는 소수점 주문 불가)",
          fake_stock(key="USQTY", market="US").round_quantity(2.9) == 2.0)


# ---------------------------------------------------------------------------
# 3. 청산 규칙
# ---------------------------------------------------------------------------

def test_exits():
    section("3. 청산 규칙 — 손절 · 익절 · 트레일링 · 만기 · 시간")

    inst = fake_stock()
    cfg = dict(store.DEFAULT_CONFIG, exit_score=0.05, trailing_stop_pct=5.0,
               max_hold_days=10, stop_loss_pct=0, take_profit_pct=0)
    position = type("P", (), {"side": "long", "avg_price": 10_000,
                              "current_price": 9_000, "quantity": 10})()
    base = {"entry_price": 10_000, "stop_price": 9_500, "target_price": 12_000,
            "peak_price": 10_000, "opened_at": datetime.now().isoformat()}
    flat = strategy.Signal(key=inst.key, ok=True, score=0.2, price=9_400)

    d = strategy.check_exit(inst, position, flat, cfg, base)
    check("청산", "손절가 도달 → 즉시 청산(urgent)",
          d.should_exit and d.urgency == "urgent", d.reason)

    d = strategy.check_exit(inst, position, strategy.Signal(key="x", ok=True, score=0.2,
                                                            price=12_500), cfg, base)
    check("청산", "목표가 도달 → 익절", d.should_exit, d.reason)

    d = strategy.check_exit(inst, position, strategy.Signal(key="x", ok=True, score=0.2,
                                                            price=11_000),
                            cfg, {**base, "peak_price": 12_000})
    check("청산", "고점 대비 되돌림 → 트레일링 스톱", d.should_exit, d.reason)

    d = strategy.check_exit(inst, position, strategy.Signal(key="x", ok=True, score=-0.4,
                                                            price=10_500), cfg, base)
    check("청산", "신호 반전 → 청산", d.should_exit, d.reason)

    old = (datetime.now() - timedelta(days=20)).isoformat()
    d = strategy.check_exit(inst, position, strategy.Signal(key="x", ok=True, score=0.5,
                                                            price=10_500),
                            cfg, {**base, "opened_at": old, "target_price": 99_999})
    check("청산", "보유 기간 초과 → 시간 청산", d.should_exit, d.reason)

    d = strategy.check_exit(inst, position, strategy.Signal(key="x", ok=True, score=0.5,
                                                            price=10_500),
                            cfg, {**base, "target_price": 99_999})
    check("청산", "조건 없으면 계속 보유", not d.should_exit)

    # 만기 임박 파생
    near = instruments.parse_derivative("101H6000")
    near.expiry = date.today() + timedelta(days=1)
    d = strategy.check_exit(near, position, strategy.Signal(key="x", ok=True, score=0.9,
                                                            price=350), cfg,
                            {"entry_price": 349, "opened_at": datetime.now().isoformat()})
    check("청산", "만기 임박 파생은 신호와 무관하게 청산",
          d.should_exit and d.urgency == "urgent", d.reason)


# ---------------------------------------------------------------------------
# 4. 리스크 게이트
# ---------------------------------------------------------------------------

def test_risk():
    section("4. 리스크 게이트 — 거부되어야 할 것들")

    inst = fake_stock()
    sig = strategy.Signal(key=inst.key, ok=True, score=0.6, direction=strategy.LONG,
                          price=10_000, price_krw=10_000)
    plan = strategy.EntryPlan(ok=True, quantity=10, price=10_000, stop_price=9_500)
    account = {"total_value": 10_000_000, "available_cash": 10_000_000}
    quote = fake_quote(10_000)
    cfg = dict(store.DEFAULT_CONFIG, min_order_krw=0, max_order_krw=0,
               position_pct=100, max_gross_exposure_pct=0)

    with FakeFeed({inst.key: 10_000}):
        engine = risk.RiskEngine(cfg, account, [], {})
        v = engine.check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "정상 조건은 통과", v.approved, f"{v.quantity:g}주")

        v = risk.RiskEngine(dict(cfg, kill_switch=True), account, [], {}) \
            .check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "킬 스위치 → 거부 + halt", not v.approved and v.halt, str(v.rejects))

        day = {"start_value": 10_000_000, "peak_value": 10_000_000}
        v = risk.RiskEngine(dict(cfg, daily_loss_limit_pct=3),
                            {"total_value": 9_600_000, "available_cash": 1_000_000},
                            [], day).check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "일일 손실 한도 초과 → 거부", not v.approved, str(v.rejects))

        v = risk.RiskEngine(dict(cfg, max_drawdown_pct=10),
                            {"total_value": 8_800_000, "available_cash": 1_000_000},
                            [], {"peak_value": 10_000_000}).check_entry(
                                inst, "buy", 10, sig, plan, quote)
        check("리스크", "최대 낙폭 초과 → 거부", not v.approved, str(v.rejects))

        v = engine.check_entry(inst, "buy", 10, sig, plan,
                               {**quote, "age_sec": 999})
        check("리스크", "오래된 시세 → 거부 (0원 주문 방지)", not v.approved, str(v.rejects))

        v = risk.RiskEngine(dict(cfg, asset_classes={"STOCK": False}), account, [], {}) \
            .check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "꺼둔 자산군 → 거부", not v.approved, str(v.rejects))

        held = [broker.Position(key="AAA", name="A", asset_class=STOCK, side="long",
                                quantity=1, avg_price=1, market_value=1_000_000)]
        v = risk.RiskEngine(dict(cfg, max_positions=1), account, held, {}) \
            .check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "보유 종목 수 상한 → 거부", not v.approved, str(v.rejects))

        same = [broker.Position(key=inst.key, name="T", asset_class=STOCK, side="long",
                                quantity=1, avg_price=1, market_value=10_000)]
        v = risk.RiskEngine(cfg, account, same, {}).check_entry(
            inst, "buy", 10, sig, plan, quote)
        check("리스크", "보유 종목 재진입(물타기) → 거부", not v.approved, str(v.rejects))

        v = risk.RiskEngine(dict(cfg, max_order_krw=30_000), account, [], {}) \
            .check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "주문 상한 초과 → 거부 대신 수량 축소",
              v.approved and v.quantity == 3, f"10주 → {v.quantity:g}주 · {v.reasons}")

        v = risk.RiskEngine(cfg, {"total_value": 10_000_000, "available_cash": 25_000},
                            [], {}).check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "가용 현금만큼만 축소", v.approved and v.quantity == 2,
              f"{v.quantity:g}주")

        v = risk.RiskEngine(dict(cfg, min_order_krw=1_000_000),
                            {"total_value": 10_000_000, "available_cash": 50_000},
                            [], {}).check_entry(inst, "buy", 10, sig, plan, quote)
        check("리스크", "최소 주문금액 미달 → 거부", not v.approved, str(v.rejects))

        v = risk.RiskEngine(cfg, account, [], {}).check_entry(
            inst, "sell", 10, sig, plan, quote)
        check("리스크", "현물 숏 시도 → 거부", not v.approved, str(v.rejects))

    # 장 마감은 진짜 시계로 판단합니다 (여기서는 함수 계약만 확인)
    engine = risk.RiskEngine(cfg, account, [], {})
    v = engine.check_exit(inst, None)
    check("리스크", "현재가 없으면 청산도 보류", not v.approved, str(v.rejects))


# ---------------------------------------------------------------------------
# 5. 브로커 (모의 계좌) — 실제 체결·정산
# ---------------------------------------------------------------------------

def test_paper_broker():
    section("5. 모의 계좌 — 주식 · 선물 · 옵션 체결과 정산")

    paper.ensure_account(TEST_USER)
    paper.reset(TEST_USER, 50_000_000)

    stock = fake_stock(key="TESTEQ", name="테스트주식")
    fut = instruments.parse_derivative("101H6000")
    call = instruments.parse_derivative("201H6320")
    prices = {"TESTEQ": 10_000, "101H6000": 350.0, "201H6320": 4.5}
    catalog = {"TESTEQ": stock, "101H6000": fut, "201H6320": call}

    with FakeFeed(prices, catalog):
        # 회계 공식을 정확히 검증하기 위해 슬리피지는 끕니다
        # (슬리피지 자체는 아래 '체결가 현실화'에서 따로 확인합니다)
        brk = broker.PaperBroker(TEST_USER, slippage_bps=0)
        start_cash = brk.account()["cash"]

        r = brk.submit(stock, "buy", 100, client_order_id="qa-eq-1")
        check("체결", "주식 매수 체결", r.ok and r.quantity == 100,
              f"{r.quantity:g}주 @ {r.price:,.0f} 수수료 {r.fee:,.0f}원")

        again = brk.submit(stock, "buy", 100, client_order_id="qa-eq-1")
        check("체결", "같은 주문번호는 재전송되지 않음 (멱등)",
              again.quantity == r.quantity and brk.position_for("TESTEQ").quantity == 100,
              f"보유 {brk.position_for('TESTEQ').quantity:g}주")

        prices["TESTEQ"] = 11_000
        pos = brk.position_for("TESTEQ")
        check("평가", "평가손익이 현재가를 반영", abs(pos.unrealized_pnl - 100_000) < 2_000,
              f"{pos.unrealized_pnl:,.0f}원")

        c = brk.close(stock, client_order_id="qa-eq-2")
        expected = 100_000 - (10_000 * 100 * 0.00015) - (11_000 * 100 * (0.00015 + 0.0018))
        check("체결", "주식 청산 실현손익 = 차익 - 수수료 - 거래세",
              c.ok and abs(c.realized_pnl - expected) < 500,
              f"{c.realized_pnl:,.0f}원 (기대 {expected:,.0f}원)")

        # --- 선물 숏 ---
        r = brk.submit(fut, "sell", 1, client_order_id="qa-fu-1")
        acc = brk.account()
        check("파생", "선물 숏 진입 — 증거금이 묶임",
              r.ok and acc["margin_locked"] > 0,
              f"증거금 {acc['margin_locked']:,.0f}원 / 현금 {acc['cash']:,.0f}원")
        check("파생", "묶인 증거금은 명목금액보다 작다",
              acc["margin_locked"] < fut.notional(350, 1),
              f"{acc['margin_locked']:,.0f} < {fut.notional(350, 1):,.0f}")

        prices["101H6000"] = 345.0
        pos = brk.position_for("101H6000")
        check("파생", "숏은 가격이 내리면 이익",
              pos.side == "short" and abs(pos.unrealized_pnl - 1_250_000) < 1,
              f"{pos.unrealized_pnl:,.0f}원 (5pt × 25만원)")

        blocked = brk.submit(fut, "buy", 1, client_order_id="qa-fu-x")
        check("파생", "반대 방향 동시 보유 차단", not blocked.ok, blocked.error)

        c = brk.close(fut, client_order_id="qa-fu-2")
        check("파생", "선물 청산 — 증거금 환입 + 손익 반영",
              c.ok and c.realized_pnl > 1_200_000 and brk.account()["margin_locked"] == 0,
              f"실현 {c.realized_pnl:,.0f}원")

        # --- 옵션 ---
        r = brk.submit(call, "buy", 2, client_order_id="qa-op-1")
        check("파생", "옵션 매수 — 프리미엄 전액 지불",
              r.ok and abs(brk.account()["margin_locked"] - 4.5 * 250_000 * 2) < 1,
              f"{brk.account()['margin_locked']:,.0f}원")
        prices["201H6320"] = 6.0
        c = brk.close(call, client_order_id="qa-op-2")
        check("파생", "옵션 청산 손익 = (청산가-진입가) × 승수 × 계약수 - 수수료",
              c.ok and abs(c.realized_pnl - (750_000 - 9_000)) < 1,
              f"{c.realized_pnl:,.0f}원")

        end = brk.account()
        check("정산", "모든 포지션 정리 후 증거금 0",
              end["margin_locked"] == 0 and not brk.positions(),
              f"현금 {end['cash']:,.0f}원 (시작 {start_cash:,.0f}원)")

        poor = broker.PaperBroker(TEST_USER, slippage_bps=0)
        paper.reset(TEST_USER, 1_000_000)
        r = poor.submit(fut, "buy", 1, client_order_id="qa-fu-3")
        check("정산", "증거금이 모자라면 진입 거부", not r.ok, r.error)

        # --- 체결가 현실화 (슬리피지) ---
        paper.reset(TEST_USER, 10_000_000)
        slippy = broker.PaperBroker(TEST_USER, slippage_bps=20)
        prices["TESTEQ"] = 10_000
        buy = slippy.submit(stock, "buy", 10, client_order_id="qa-slip-1")
        check("체결", "매수는 현재가보다 비싸게 체결 (불리한 방향)",
              buy.ok and buy.avg_fill_price > 10_000,
              f"{buy.avg_fill_price:,.0f}원 (슬리피지 {buy.slippage_bps:+.1f}bp)")
        sell = slippy.close(stock, client_order_id="qa-slip-2")
        check("체결", "매도는 현재가보다 싸게 체결",
              sell.ok and sell.avg_fill_price < 10_000,
              f"{sell.avg_fill_price:,.0f}원 (슬리피지 {sell.slippage_bps:+.1f}bp)")
        check("체결", "슬리피지는 항상 손해 방향이 양수",
              (buy.slippage_bps or 0) > 0 and (sell.slippage_bps or 0) > 0)


# ---------------------------------------------------------------------------
# 6. 설정 저장소
# ---------------------------------------------------------------------------

def test_store():
    section("6. 설정 · 원장")

    store.init()
    cfg = store.save_config(TEST_USER, {"entry_score": 0.42, "unknown_key": 1,
                                        "asset_classes": {"FUTURES": True}})
    check("설정", "알려진 키만 저장", cfg["entry_score"] == 0.42 and "unknown_key" not in cfg)
    check("설정", "자산군 설정은 병합",
          cfg["asset_classes"]["FUTURES"] is True and cfg["asset_classes"]["STOCK"] is True,
          str(cfg["asset_classes"]))
    check("설정", "기본값은 안전한 쪽",
          store.DEFAULT_CONFIG["mode"] == "paper"
          and store.DEFAULT_CONFIG["kill_switch"] is False
          and store.DEFAULT_CONFIG["asset_classes"]["FUTURES"] is False)

    store.set_enabled(TEST_USER, True)
    check("설정", "켜면 enabled_users 에 나타남", TEST_USER in store.enabled_users())
    store.set_enabled(TEST_USER, False)
    check("설정", "끄면 목록에서 빠짐", TEST_USER not in store.enabled_users())

    coid = f"qa-{datetime.now().timestamp()}"
    first = store.record_order(TEST_USER, {"client_order_id": coid, "symbol": "X",
                                           "action": "buy", "quantity": 1,
                                           "status": "filled"})
    second = store.record_order(TEST_USER, {"client_order_id": coid, "symbol": "X",
                                            "action": "buy", "quantity": 1,
                                            "status": "filled"})
    check("원장", "같은 client_order_id 는 한 번만 기록 (DB 제약)",
          first and second is None and store.order_exists(coid))

    store.upsert_position_state(TEST_USER, "paper", "X", side="long", entry_price=100,
                                stop_price=95)
    store.upsert_position_state(TEST_USER, "paper", "X", stop_price=97)
    state = store.get_position_states(TEST_USER, "paper")["X"]
    check("원장", "포지션 상태는 부분 갱신 가능",
          state["entry_price"] == 100 and state["stop_price"] == 97)
    store.clear_position_state(TEST_USER, "paper", "X")
    check("원장", "청산 시 상태 삭제",
          "X" not in store.get_position_states(TEST_USER, "paper"))

    day = store.touch_daily(TEST_USER, "paper", 10_000_000)
    store.touch_daily(TEST_USER, "paper", 11_000_000)
    day = store.touch_daily(TEST_USER, "paper", 9_000_000)
    check("원장", "일자 기준값은 시작·최고를 유지",
          day["start_value"] == 10_000_000 and day["peak_value"] == 11_000_000
          and day["end_value"] == 9_000_000,
          f"시작 {day['start_value']:,.0f} 최고 {day['peak_value']:,.0f}")


# ---------------------------------------------------------------------------
# 7. 백테스트
# ---------------------------------------------------------------------------

def test_backtest():
    section("7. 백테스트 시뮬레이터")

    from engine import autotrade

    inst = fake_stock(key="BTTEST", name="백테스트종목")
    bars = trend_bars(n=220, drift=0.005)
    saved_bars, saved_resolve = feed.bars, autotrade._resolve_universe_item
    feed.bars = lambda i, tf="day", count=120: bars
    autotrade.feed.bars = feed.bars
    autotrade._resolve_universe_item = lambda q: inst
    try:
        r = autotrade.simulate("BTTEST", {"entry_score": 0.2}, days=200)
        check("백테스트", "시뮬레이션 실행", r.get("ok"),
              f"매매 {r.get('trade_count')}회 · 수익률 {r.get('total_return_pct')}% · "
              f"MDD {r.get('max_drawdown_pct')}%")
        check("백테스트", "상승 추세에서 이익", r.get("total_return_pct", 0) > 0)
        check("백테스트", "자산 곡선 생성", len(r.get("curve", [])) > 50,
              f"{len(r.get('curve', []))}포인트")
        check("백테스트", "각 매매에 청산 사유 기록",
              all(t.get("reason") for t in r.get("trades", [])),
              str([t["reason"][:20] for t in r.get("trades", [])[:2]]))
        check("백테스트", "분봉·뉴스를 끄고 일봉만 사용 (미래 참조 차단)",
              r["config"].get("entry_score") == 0.2)
    finally:
        feed.bars = saved_bars
        autotrade.feed.bars = saved_bars
        autotrade._resolve_universe_item = saved_resolve


# ---------------------------------------------------------------------------
# 8. 엔진 루프 (전 구간 통합)
# ---------------------------------------------------------------------------

def test_engine():
    section("8. 엔진 — 진입부터 청산까지 한 바퀴")

    from engine import autotrade

    paper.reset(TEST_USER, 10_000_000)
    for symbol in list(store.get_position_states(TEST_USER, "paper")):
        store.clear_position_state(TEST_USER, "paper", symbol)
    # 앞선 테스트가 남긴 '오늘의 최고 평가금액'을 지웁니다.
    # 안 지우면 계좌를 1천만원으로 되돌린 순간 낙폭 한도가 걸려(정상 동작)
    # 진입 자체가 차단되고, 그건 이 테스트가 보려던 것이 아닙니다.
    with store._conn() as conn:
        conn.execute("DELETE FROM at_daily WHERE user_id = ?", (TEST_USER,))

    # 종목 코드에 실행 시각을 넣습니다. 엔진의 중복 주문 방지는 '분 단위 주문번호'라,
    # 같은 분에 테스트를 두 번 돌리면 두 번째 진입이 (정상적으로) 무시됩니다.
    key = f"ENG{datetime.now():%H%M%S}"
    inst = fake_stock(key=key, name="엔진테스트")
    bars = trend_bars(n=140, drift=0.006)
    price = float(bars["close"].iloc[-1])

    saved_bars = feed.bars
    saved_resolve = autotrade._resolve_universe_item
    feed.bars = lambda i, tf="day", count=120: (bars if tf == "day" else pd.DataFrame())
    autotrade._resolve_universe_item = lambda q: inst

    store.save_config(TEST_USER, {
        "mode": "paper", "universe": [key], "entry_score": 0.2,
        "intraday_weight": 0.0, "use_news": False, "dry_run": False,
        "regular_session_only": False, "min_order_krw": 0, "max_positions": 3,
        "asset_classes": {"STOCK": True},
    })

    try:
        with FakeFeed({key: price}, {key: inst}) as fake:
            r = autotrade.run_once(TEST_USER, force=True)
            check("엔진", "회전 성공", r.get("ok"), f"{r.get('elapsed_sec')}초")
            check("엔진", "신호 → 진입 주문 체결", len(r.get("entries", [])) == 1,
                  str([(e["symbol"], e["quantity"]) for e in r.get("entries", [])])
                  or str(r.get("rejects")))

            state = store.get_position_states(TEST_USER, "paper").get(key, {})
            check("엔진", "손절·목표가가 기록됨",
                  bool(state.get("stop_price") and state.get("target_price")),
                  f"손절 {state.get('stop_price') or 0:,.0f} / 목표 {state.get('target_price') or 0:,.0f}")

            r2 = autotrade.run_once(TEST_USER, force=True)
            check("엔진", "다음 회전에서 중복 진입하지 않음",
                  not r2.get("entries"), str(r2.get("entries")))

            # 손절 폭 아래로 가격을 떨어뜨립니다
            if state.get("stop_price"):
                fake.prices[key] = float(state["stop_price"]) * 0.98
                r3 = autotrade.run_once(TEST_USER, force=True)
                check("엔진", "손절선 이탈 → 자동 청산", len(r3.get("exits", [])) == 1,
                      str([(e["symbol"], round(e.get("realized_pnl") or 0))
                           for e in r3.get("exits", [])]))
                check("엔진", "청산 후 포지션 상태 삭제",
                      key not in store.get_position_states(TEST_USER, "paper"))

            orders = store.get_orders(TEST_USER, 5)
            check("감사", "주문 원장에 진입·청산이 남음",
                  len([o for o in orders if o["symbol"] == key]) >= 2,
                  str([(o["action"], o["status"]) for o in orders[:3]]))
            events = store.get_events(TEST_USER, 20)
            check("감사", "이벤트 로그에 사유가 남음",
                  any(e["kind"] == "exit" for e in events),
                  next((e["message"][:60] for e in events if e["kind"] == "exit"), ""))

            store.save_config(TEST_USER, {"kill_switch": True})
            r4 = autotrade.run_once(TEST_USER, force=True)
            check("엔진", "킬 스위치 상태에서 진입 차단",
                  r4.get("halted") and not r4.get("entries"), str(r4.get("halt_reasons")))
            store.save_config(TEST_USER, {"kill_switch": False})

            store.save_config(TEST_USER, {"dry_run": True})
            r5 = autotrade.run_once(TEST_USER, force=True)
            check("엔진", "dry_run 은 판단만 하고 주문하지 않음",
                  all(e.get("dry_run") for e in r5.get("entries", [])),
                  f"{len(r5.get('entries', []))}건 판단")
    finally:
        feed.bars = saved_bars
        autotrade._resolve_universe_item = saved_resolve
        store.save_config(TEST_USER, {"dry_run": False})


# ---------------------------------------------------------------------------
# 9. 주문 생애주기 — 접수 → 체결 확인 → 포지션 확정
# ---------------------------------------------------------------------------

class StubBroker(broker.Broker):
    """실계좌처럼 '접수만 되고 나중에 체결되는' 브로커를 흉내 냅니다.

    PaperBroker 는 즉시 체결이라 부분체결·미체결 타임아웃 경로를 검증할 수 없습니다.
    """

    mode = "mock"

    def __init__(self, positions=None):
        self._positions = positions or []
        self.submitted = []
        self.cancelled = []
        self.status_map = {}          # broker_order_id -> OrderStatus
        self.cash = 10_000_000

    def health(self):
        return {"mode": self.mode, "label": "테스트 브로커", "ready": True, "detail": ""}

    def account(self):
        return {"mode": self.mode, "label": "테스트 브로커", "cash": self.cash,
                "available_cash": self.cash, "reserved_cash": 0,
                "initial_cash": 10_000_000, "equity_value": 0, "margin_locked": 0,
                "derivative_pnl": 0, "notional_exposure": 0,
                "total_value": self.cash}

    def positions(self):
        return list(self._positions)

    def submit(self, inst, side, quantity, order_type="market", price=None,
               note="", client_order_id=""):
        oid = f"B{len(self.submitted) + 1}"
        self.submitted.append((inst.key, side, quantity))
        return broker.OrderResult(
            ok=True, action=side, quantity=quantity, price=price or 10_000,
            price_krw=price or 10_000, status="pending", broker_order_id=oid,
            client_order_id=client_order_id, intended_price=price or 10_000)

    def close(self, inst, quantity=None, note="", client_order_id=""):
        oid = f"C{len(self.submitted) + 1}"
        self.submitted.append((inst.key, "close", quantity))
        return broker.OrderResult(
            ok=True, action="close", quantity=quantity or 1, price=10_000,
            price_krw=10_000, status="pending", broker_order_id=oid,
            client_order_id=client_order_id, intended_price=10_000)

    def order_status(self, record):
        return self.status_map.get(record.get("broker_order_id"),
                                   broker.OrderStatus(known=False, detail="미등록"))

    def cancel(self, record):
        self.cancelled.append(record.get("broker_order_id"))
        return broker.OrderResult(ok=True, action="cancel", status="cancelled",
                                  broker_order_id=record.get("broker_order_id", ""))


def _new_order(user_id, symbol, action="buy", quantity=10, price=10_000,
               broker_order_id="B1", created_at=None, mode="mock"):
    coid = f"qa-{symbol}-{datetime.now().timestamp()}"
    order_id = store.record_order(user_id, {
        "client_order_id": coid, "broker_mode": mode,
        "broker_order_id": broker_order_id, "symbol": symbol, "name": symbol,
        "asset_class": STOCK, "action": action, "side": "long",
        "quantity": quantity, "price": price, "price_krw": price,
        "status": "pending", "intended_price": price,
        "detail": {"plan": {"price": price, "stop_price": price * 0.95,
                            "target_price": price * 1.10}},
    })
    if created_at:
        with store._conn() as conn:
            conn.execute("UPDATE at_orders SET created_at = ? WHERE id = ?",
                         (created_at, order_id))
    return order_id


def test_order_lifecycle():
    section("9. 주문 생애주기 — 접수 → 체결 확인 → 포지션 확정")

    from engine import autotrade

    user = TEST_USER + 3
    paper.ensure_account(user)
    with store._conn() as conn:
        conn.execute("DELETE FROM at_orders WHERE user_id = ?", (user,))
        conn.execute("DELETE FROM at_position_state WHERE user_id = ?", (user,))
    cfg = dict(store.DEFAULT_CONFIG, mode="mock", order_timeout_sec=120)

    # --- 부분 체결 ---
    brk = StubBroker()
    order_id = _new_order(user, "PART01", quantity=10, broker_order_id="B1")
    brk.status_map["B1"] = broker.OrderStatus(
        known=True, status="partial", filled_quantity=4, remaining=6,
        avg_fill_price=10_050)
    result = {"settled": [], "reconciled": [], "errors": []}
    autotrade._settle_open_orders(user, cfg, brk, result)

    record = next(o for o in store.get_orders(user, 20) if o["id"] == order_id)
    check("체결", "부분 체결이 원장에 반영됨",
          record["status"] == "partial" and record["filled_quantity"] == 4,
          f"{record['status']} {record['filled_quantity']:g}/{record['quantity']:g}")
    check("체결", "부분 체결 주문은 계속 열려 있음",
          any(o["id"] == order_id for o in store.open_orders(user)))
    state = store.get_position_states(user, "mock").get("PART01", {})
    check("체결", "실제 체결가 기준으로 손절선이 잡힘",
          state.get("entry_price") == 10_050
          and abs(state.get("stop_price", 0) - 9_550) < 1,
          f"진입 {state.get('entry_price')} / 손절 {state.get('stop_price')}")
    check("체결", "슬리피지가 계산됨 (의도 10,000 → 체결 10,050)",
          abs((record["slippage_bps"] or 0) - 50) < 0.5, f"{record['slippage_bps']}bp")

    # --- 잔량까지 전량 체결 ---
    brk.status_map["B1"] = broker.OrderStatus(
        known=True, status="filled", filled_quantity=10, remaining=0,
        avg_fill_price=10_050)
    autotrade._settle_open_orders(user, cfg, brk, result)
    record = next(o for o in store.get_orders(user, 20) if o["id"] == order_id)
    check("체결", "전량 체결되면 주문이 닫힘",
          record["status"] == "filled" and not store.open_orders(user),
          f"{record['status']} {record['filled_quantity']:g}")

    # --- 조회 실패는 아무것도 하지 않는다 (fail-closed) ---
    unknown_id = _new_order(user, "UNK01", broker_order_id="ZZZ")
    autotrade._settle_open_orders(user, cfg, StubBroker(), result)
    record = next(o for o in store.get_orders(user, 20) if o["id"] == unknown_id)
    check("체결", "조회 실패 시 상태를 함부로 바꾸지 않음",
          record["status"] == "pending" and record["filled_quantity"] == 0,
          "모르면 그대로 둡니다")

    # --- 미체결 타임아웃 → 자동 취소 ---
    old = (datetime.now() - timedelta(seconds=600)).isoformat()
    stale_id = _new_order(user, "OLD01", broker_order_id="B9", created_at=old)
    brk2 = StubBroker()
    brk2.status_map["B9"] = broker.OrderStatus(known=True, status="pending",
                                               filled_quantity=0, remaining=10)
    autotrade._settle_open_orders(user, cfg, brk2, result)
    record = next(o for o in store.get_orders(user, 20) if o["id"] == stale_id)
    check("체결", "타임아웃 미체결 주문은 자동 취소 요청",
          "B9" in brk2.cancelled and record["status"] == "cancel_requested",
          record["reason"])

    # --- 브로커가 취소를 확정하면 정리 ---
    brk2.status_map["B9"] = broker.OrderStatus(known=True, status="cancelled",
                                               filled_quantity=0, remaining=0)
    autotrade._settle_open_orders(user, cfg, brk2, result)
    record = next(o for o in store.get_orders(user, 20) if o["id"] == stale_id)
    check("체결", "취소 확정되면 주문이 닫힘", record["status"] == "cancelled")

    # --- 미체결 주문이 있으면 같은 종목에 또 주문하지 않는다 ---
    _new_order(user, "DUP01", broker_order_id="B7")
    check("체결", "미체결 주문이 있으면 같은 종목 재주문 차단",
          store.has_open_order(user, "mock", "DUP01")
          and not store.has_open_order(user, "mock", "OTHER"))

    quality = store.execution_quality(user)
    check("체결", "체결 품질(슬리피지) 집계", quality["samples"] >= 1,
          f"평균 {quality['avg_slippage_bps']}bp / {quality['samples']}건")


def test_reconcile():
    section("10. 재동기화 — 실계좌와 내부 상태 대조")

    from engine import autotrade

    user = TEST_USER + 4
    paper.ensure_account(user)
    with store._conn() as conn:
        conn.execute("DELETE FROM at_orders WHERE user_id = ?", (user,))
        conn.execute("DELETE FROM at_position_state WHERE user_id = ?", (user,))
    cfg = dict(store.DEFAULT_CONFIG, mode="paper", atr_stop_mult=0, stop_loss_pct=5,
               manage_only_universe=False)

    # (1) 계좌에는 있는데 엔진은 모르는 포지션 → 복원
    orphan = broker.Position(key="ORPH1", name="고아종목", asset_class=STOCK,
                             side="long", quantity=7, avg_price=20_000,
                             current_price=21_000, market_value=147_000)
    result = {"reconciled": [], "errors": []}
    with FakeFeed({"ORPH1": 21_000}, {"ORPH1": fake_stock(key="ORPH1")}):
        autotrade._reconcile(user, cfg, [orphan], result)
    state = store.get_position_states(user, "paper").get("ORPH1", {})
    check("동기화", "계좌에만 있는 포지션을 관리 대상으로 복원",
          bool(state) and state.get("entry_price") == 20_000,
          f"진입 {state.get('entry_price')} / 손절 {state.get('stop_price')} / 수량 {state.get('quantity')}")
    check("동기화", "복원 사실이 로그에 남음",
          any(e["kind"] == "reconcile" for e in store.get_events(user, 10)),
          next((e["message"][:60] for e in store.get_events(user, 10)
                if e["kind"] == "reconcile"), ""))

    # (2) 수량이 어긋나면 실계좌 기준으로 보정
    result = {"reconciled": [], "errors": []}
    orphan.quantity = 3
    with FakeFeed({"ORPH1": 21_000}, {"ORPH1": fake_stock(key="ORPH1")}):
        autotrade._reconcile(user, cfg, [orphan], result)
    state = store.get_position_states(user, "paper").get("ORPH1", {})
    check("동기화", "수량 불일치를 실계좌 기준으로 보정",
          state.get("quantity") == 3,
          str([r["action"] for r in result["reconciled"]]))

    # (3) 엔진은 들고 있다고 생각하는데 계좌에 없음 → 정리
    result = {"reconciled": [], "errors": []}
    with FakeFeed({}, {}):
        autotrade._reconcile(user, cfg, [], result)
    check("동기화", "계좌에 없는 포지션 기억은 삭제",
          "ORPH1" not in store.get_position_states(user, "paper"),
          str([r["action"] for r in result["reconciled"]]))

    # (4) 다만 체결 대기 주문이 있으면 성급히 지우지 않는다
    store.upsert_position_state(user, "paper", "WAIT1", side="long", entry_price=100,
                                stop_price=95, quantity=1)
    _new_order(user, "WAIT1", broker_order_id="BW1", mode="paper")
    result = {"reconciled": [], "errors": []}
    with FakeFeed({}, {}):
        autotrade._reconcile(user, cfg, [], result)
    check("동기화", "체결 대기 중인 종목의 상태는 유지",
          "WAIT1" in store.get_position_states(user, "paper"))

    # 총평가금액에 현금이 두 번 들어가면 안 됩니다. KIS 의 tot_evlu_amt 는
    # 예수금을 포함하므로, 그걸 유가증권 평가액으로 쓰면 자산이 부풀려집니다.
    from data_sources import kis_trading as KT
    check("잔고", "유가증권 평가액은 예수금을 빼고 계산",
          KT._securities_value({"scts_evlu_amt": "7588", "tot_evlu_amt": "17588",
                                "dnca_tot_amt": "10000"}) == 7588)
    check("잔고", "보유 종목이 없으면 유가증권 평가액은 0 (총평가=예수금)",
          KT._securities_value({"tot_evlu_amt": "10000",
                                "prvs_rcdl_excc_amt": "10000"}) == 0)
    check("잔고", "scts_evlu_amt 가 없으면 총평가−정산예수금으로 보완",
          KT._securities_value({"tot_evlu_amt": "17588",
                                "prvs_rcdl_excc_amt": "10000"}) == 7588)
    check("잔고", "필드 의미가 달라 음수가 나오면 0 (자산 부풀리기 금지)",
          KT._securities_value({"tot_evlu_amt": "5000", "dnca_tot_amt": "10000"}) == 0)

    # 계좌 요약은 우리가 다시 계산하지 않고 증권사 값을 받아 적습니다.
    # 아래 응답은 실계좌 실측값입니다 (2026-08-07, 위탁계좌 1000****-01).
    # 증권사 앱 화면: 총자산 139,518 · 예수금 132,226 · 출금가능 3,325.
    saved_get, saved_account = KT._get, KT.account
    saved_is_mock = KT.kis_client.is_mock
    saved_cache_snap = dict(KT._balance_cache)
    try:
        KT._balance_cache.clear()
        KT.account = lambda kind="stock": ("10001807", "01")
        KT.kis_client.is_mock = lambda: False
        replies = {
            KT.ACCOUNT_ASSETS_PATH: {"ok": True, "output2": {
                "tot_asst_amt": "139518", "tot_dncl_amt": "132226",
                "pchs_amt_smtl": "7605", "evlu_amt_smtl": "7292",
                "evlu_pfls_amt_smtl": "-313", "nass_tot_amt": "7292",
                "ovrs_stck_evlu_amt1": "7292.000000"}},
            KT.OVERSEAS_PRESENT_PATH: {"ok": True, "raw": {
                "output1": [{"bass_exrt": "1418.80000000"}],
                "output3": {"tot_asst_amt": "138963", "tot_dncl_amt": "132226",
                            "wdrw_psbl_tot_amt": "3325", "pchs_amt_smtl": "129379",
                            "evlu_amt_smtl": "129194", "evlu_pfls_amt_smtl": "-185",
                            "ustl_buy_amt_smtl": "129692",
                            "ustl_sll_amt_smtl": "7235"}}},
            KT.BUYABLE_PATH: {"ok": True, "output": {"ord_psbl_cash": "3325",
                                                     "max_buy_amt": "9361"}},
        }
        KT._get = lambda op, path, params, timeout=15: replies[path]
        snap = KT.account_snapshot()
        check("잔고", "총자산은 증권사 '자산현황' 값 그대로",
              snap["ok"] and snap["total_asset"] == 139_518, f"{snap['total_asset']:,.0f}원")
        check("잔고", "주문가능현금은 예수금이 아님 (미결제 매수대금 제외)",
              snap["available_cash"] == 3_325 and snap["deposit"] == 132_226,
              f"예수금 {snap['deposit']:,.0f} / 주문가능 {snap['available_cash']:,.0f}")
        check("잔고", "평가금액은 체결기준 — 결제 전 보유분도 셈",
              snap["eval_amount"] == 129_194 and snap["settled_eval_amount"] == 7_292,
              f"체결 {snap['eval_amount']:,.0f} / 결제 {snap['settled_eval_amount']:,.0f}")
        check("잔고", "환율은 증권사 기준환율을 씀",
              snap["fx_rate"] == 1418.8, f"{snap['fx_rate']}")

        # 한 조각이라도 못 읽으면 반쪽 요약을 정상인 척 돌려주지 않습니다
        KT._balance_cache.clear()
        replies[KT.BUYABLE_PATH] = {"ok": False, "error": "조회 실패"}
        broken = KT.account_snapshot()
        check("잔고", "주문가능현금을 못 읽으면 요약 전체가 실패",
              not broken["ok"] and broken["errors"], str(broken["errors"]))
    finally:
        KT._get, KT.account = saved_get, saved_account
        KT.kis_client.is_mock = saved_is_mock
        KT._balance_cache.clear()
        KT._balance_cache.update(saved_cache_snap)

    # KIS 는 거래소 코드를 넣어도 보유분 전체를 돌려줍니다. 세 거래소를 합치면
    # 같은 종목이 여러 번 잡혀 포지션이 2~3개로 보이고, 평가손익도 그만큼 부풉니다.
    saved_ob = KT.overseas_balance
    saved_cache = dict(KT._balance_cache)
    try:
        KT._balance_cache.clear()
        KT.overseas_balance = lambda code="NASD", currency="USD": (
            {"ok": True, "positions": [{"code": "AMC", "quantity": 1.0, "pnl": -0.02}]}
            if code in ("NASD", "NYSE") else {"ok": True, "positions": []})
        merged = KT.overseas_balance_all()
        check("잔고", "여러 거래소에 겹쳐 나온 같은 종목은 하나로",
              len(merged["positions"]) == 1,
              f"{len(merged['positions'])}개 · {[p['code'] for p in merged['positions']]}")
    finally:
        KT.overseas_balance = saved_ob
        KT._balance_cache.clear()
        KT._balance_cache.update(saved_cache)

    # 통합증거금 계좌는 해외 매수대금이 결제일까지 예수금에 남습니다.
    # 예수금 + 포지션 평가액을 더하면 매수대금이 두 번 잡혀 총자산이 부풉니다.
    import engine.broker as BK
    saved = (BK.kis_trading.stock_balance, BK.kis_trading.deriv_balance,
             BK.kis_trading.overseas_balance_all, BK.kis_trading.account_snapshot,
             BK.kis_trading.overseas_present)
    try:
        # 증권사 요약을 못 받는 상황 — 이때만 아래 재구성 식이 쓰입니다
        BK.kis_trading.account_snapshot = lambda: {"ok": False, "errors": []}
        BK.kis_trading.overseas_present = lambda: {"ok": False}
        BK.kis_trading.stock_balance = lambda: {
            "ok": True, "positions": [], "cash": 10_000.0,
            "available_cash": 10_000.0, "eval_amount": 0.0}
        BK.kis_trading.deriv_balance = lambda: {"ok": True, "positions": [],
                                                "margin_used": 0.0}
        # 7,612원짜리 미국 주식을 사서 32원 평가손실 중인 상태
        BK.kis_trading.overseas_balance_all = lambda: {
            "ok": True, "positions": [{"code": "AMC", "name": "AMC", "quantity": 2,
                                       "avg_price": 2.665, "current_price": 2.654,
                                       "pnl": -0.022, "purchase_amount": 5.33, "eval_amount": 5.308}]}
        acc = BK.KISBroker(user, BK.LIVE, {}).account()
        total = float(acc["total_value"])
        check("잔고", "해외 매수대금이 총자산에 두 번 잡히지 않음",
              9_000 < total < 10_100, f"총평가 {total:,.0f}원 (예수금 10,000 + 평가손익)")
        # 여섯 항목이 서로 맞아떨어져야 합니다: 총자산 = 총예수금 + (평가금액 − 매입금액)
        check("잔고", "평가손익 = 평가금액 − 매입금액",
              abs(acc["unrealized_pnl"]
                  - (acc["equity_value"] - acc["purchase_amount"])) < 1,
              f"손익 {acc['unrealized_pnl']:,.0f} = 평가 {acc['equity_value']:,.0f}"
              f" − 매입 {acc['purchase_amount']:,.0f}")
        check("잔고", "총자산 = 총예수금 + 평가손익 (결제 전 기준)",
              abs(total - (acc["cash"] + acc["unrealized_pnl"])) < 1,
              f"{total:,.0f} = {acc['cash']:,.0f} + {acc['unrealized_pnl']:,.0f}")
        check("잔고", "포지션이 있으면 평가금액이 0이 아님",
              acc["equity_value"] > 0, f"{acc['equity_value']:,.0f}원")
        check("잔고", "증권사 요약을 못 받으면 출처가 '재구성'으로 표시",
              acc["account_source"] == "재구성", acc["account_source"])

        # 증권사가 준 요약이 있으면 재구성 값을 **버리고** 그쪽을 씁니다.
        # 실계좌 실측값(2026-08-07)을 그대로 넣어, 예수금을 주문가능액으로
        # 쓰는 옛 동작이 되살아나면 바로 잡히게 합니다.
        BK.kis_trading.account_snapshot = lambda: {
            "ok": True, "total_asset": 139_518.0, "settled_asset": 138_963.0,
            "deposit": 132_226.0, "available_cash": 3_325.0, "withdrawable": 3_325.0,
            "purchase_amount": 129_379.0, "eval_amount": 129_194.0,
            "unrealized_pnl": -185.0, "unsettled_buy": 129_692.0,
            "unsettled_sell": 7_235.0, "fx_rate": 1_418.8,
            "settled_eval_amount": 7_292.0, "sources": ["자산현황"], "errors": []}
        live = BK.KISBroker(user, BK.LIVE, {}).account()
        check("잔고", "총자산은 증권사가 계산한 값을 그대로",
              live["total_value"] == 139_518.0, f"{live['total_value']:,.0f}원")
        check("잔고", "주문가능 현금은 예수금이 아니라 주문가능현금",
              live["available_cash"] == 3_325.0 and live["cash"] == 132_226.0,
              f"예수금 {live['cash']:,.0f} / 주문가능 {live['available_cash']:,.0f}")
        check("잔고", "예수금에 묶인 미결제 매수대금이 드러남",
              live["reserved_cash"] == 128_901.0 and live["unsettled_buy"] == 129_692.0,
              f"묶임 {live['reserved_cash']:,.0f} / 미결제매수 {live['unsettled_buy']:,.0f}")
        check("잔고", "평가금액은 체결기준 (결제 안 된 보유분도 노출로 계산)",
              live["equity_value"] == 129_194.0 and live["notional_exposure"] == 129_194.0,
              f"{live['equity_value']:,.0f}원")
    finally:
        (BK.kis_trading.stock_balance, BK.kis_trading.deriv_balance,
         BK.kis_trading.overseas_balance_all, BK.kis_trading.account_snapshot,
         BK.kis_trading.overseas_present) = saved

    # 미국 매수 사전 자금 확인 — 못 살 주문을 실제로 내보고 거부당하지 않게.
    import engine.broker as B
    us = fake_stock(key="AMC", name="AMC", market="US")
    original = kis_trading_stub = getattr(B.kis_trading, "overseas_buyable", None)

    def stub(result):
        B.kis_trading.overseas_buyable = lambda *a, **k: result

    try:
        stub({"ok": True, "cash_usd": 0.0, "after_fx_usd": 0.0})
        msg = B._overseas_funds_block(us, "NYSE", 2.68, 1)
        check("사전확인", "외화 0이면 환전을 안내",
              "환전" in msg and "외화" in msg, msg[:70])

        # 통합증거금이면 KIS 가 결제일에 환전해 주므로, 환전 후 금액으로 덮이면
        # 막지 않고 보냅니다 (실제 가부는 KIS 가 판단).
        stub({"ok": True, "cash_usd": 0.0, "after_fx_usd": 7.0})
        check("사전확인", "환전 후 금액으로 커버되면 주문을 막지 않음",
              B._overseas_funds_block(us, "NYSE", 2.68, 1) == "")

        stub({"ok": True, "cash_usd": 7.0, "after_fx_usd": 7.0})
        check("사전확인", "자금이 충분하면 통과",
              B._overseas_funds_block(us, "NYSE", 2.68, 1) == "")
        msg = B._overseas_funds_block(us, "NYSE", 2.68, 5)
        check("사전확인", "수량이 자금을 넘으면 필요·가능 금액을 함께 표시",
              "13.40" in msg and "7.00" in msg, msg[:70])

        stub({"ok": True, "cash_usd": 0.0, "after_fx_usd": 1.0})
        msg = B._overseas_funds_block(us, "NYSE", 2.68, 1)
        check("사전확인", "외화가 0이고 부족하면 환전·통합증거금을 함께 안내",
              "환전" in msg and "통합증거금" in msg, msg[-60:])

        stub({"ok": False, "error": "KIS 점검 중"})
        check("사전확인", "조회 실패는 매매를 막지 않음 (점검에 전체 정지 방지)",
              B._overseas_funds_block(us, "NYSE", 2.68, 1) == "")

        # 결제 재원 설정 — 'krw' 면 사전 차단을 건너뛰고 KIS 판단에 맡깁니다
        stub({"ok": True, "cash_usd": 0.0, "after_fx_usd": 0.0})
        brk_fx = B.KISBroker(user, B.LIVE, {"us_order_funding": "fx"})
        brk_krw = B.KISBroker(user, B.LIVE, {"us_order_funding": "krw"})
        check("사전확인", "설정이 브로커까지 전달됨",
              brk_fx.cfg.get("us_order_funding") == "fx"
              and brk_krw.cfg.get("us_order_funding") == "krw")
        check("사전확인", "기본값은 외화 전용 (모르는 값도 안전한 쪽)",
              B.KISBroker(user, B.LIVE).cfg.get("us_order_funding", "fx") == "fx")
    finally:
        if original:
            B.kis_trading.overseas_buyable = original

    check("사전확인", "사전 안내가 있으면 중복 안내를 붙이지 않음",
          autotrade._reject_hint(us, "AMC 주문가능금액이 0달러입니다 — 확인하세요") == "")

    # 수동 편입 종목은 AI 가 같이 뽑은 뒤에도 지켜져야 합니다.
    # (예전에는 '추천에 없는 것 = 수동'으로 추정해서, AI 가 한 번 뽑으면
    #  다음 갱신에 사람이 넣은 종목이 조용히 빠졌습니다)
    keep_cfg = dict(store.DEFAULT_CONFIG, auto_universe_keep_manual=True,
                    universe=["MYPICK", "AIPICK"], manual_universe=["MYPICK"])
    store.save_recommendations(user, [], ["AIPICK", "MYPICK"])   # 둘 다 AI 도 뽑음
    kept = [s for s in (keep_cfg.get("manual_universe") or [])
            if s in keep_cfg["universe"]]
    check("수동편입", "AI 가 같이 뽑아도 수동 종목은 기록에 남음",
          kept == ["MYPICK"], str(kept))
    dropped = dict(keep_cfg, universe=["AIPICK"])
    check("수동편입", "화면에서 빼면 수동 기록에서도 사라짐",
          [s for s in (dropped.get("manual_universe") or [])
           if s in dropped["universe"]] == [])

    # 증권사 거부 후 재시도 대기 — 외화 부족처럼 저절로 안 풀리는 사유에
    # 60초마다 같은 주문을 다시 내면 실계좌 원장이 거부로 가득 찹니다.
    autotrade.clear_broker_reject(user)
    check("거부대기", "거부 전에는 대기 없음",
          autotrade._reject_cooldown_left(user, "AMC")[0] == 0)
    autotrade.note_broker_reject(user, "AMC", "주문가능금액을 초과 했습니다")
    left, why = autotrade._reject_cooldown_left(user, "AMC")
    check("거부대기", "거부하면 재시도를 멈추고 사유를 기억", left > 0 and "주문가능금액" in why,
          f"{left / 60:.0f}분 대기 · {why}")
    check("거부대기", "다른 종목은 영향 없음",
          autotrade._reject_cooldown_left(user, "JBLU")[0] == 0)
    autotrade.clear_broker_reject(user)
    check("거부대기", "설정을 고치면 대기가 즉시 풀림",
          autotrade._reject_cooldown_left(user, "AMC")[0] == 0)
    us = fake_stock(key="AMC", market="US")
    check("거부대기", "미국 주문 거부에 해결 방법을 붙임",
          "환전" in autotrade._reject_hint(us, "주문가능금액을 초과 했습니다"),
          autotrade._reject_hint(us, "주문가능금액을 초과 했습니다")[:60])
    check("거부대기", "국내 종목에는 달러 안내를 붙이지 않음",
          autotrade._reject_hint(fake_stock(), "주문가능금액을 초과 했습니다") == "")

    # (5) 잔고 조회가 일부 실패했으면 '안 보인다'를 청산으로 단정하지 않는다.
    #     (해외 잔고 조회 한 번 실패에 손절선을 지우면 포지션이 무방비가 됩니다)
    store.upsert_position_state(user, "paper", "USPOS1", side="long", entry_price=5.0,
                                stop_price=4.5, quantity=2)
    result = {"reconciled": [], "errors": []}
    with FakeFeed({}, {}):
        autotrade._reconcile(user, cfg, [], result, partial=True)
    check("동기화", "잔고 조회 실패 시 포지션 기억을 지우지 않음",
          "USPOS1" in store.get_position_states(user, "paper"),
          "partial=True 면 대조를 다음 회전으로 미룹니다")
    result = {"reconciled": [], "errors": []}
    with FakeFeed({}, {}):
        autotrade._reconcile(user, cfg, [], result, partial=False)
    check("동기화", "정상 조회로 확인되면 그때 정리",
          "USPOS1" not in store.get_position_states(user, "paper"))


# ---------------------------------------------------------------------------
# 11. 계좌 분리 — 가상 자금과 실제 자금이 섞이지 않는가
# ---------------------------------------------------------------------------

def test_account_isolation():
    section("11. 계좌 분리 — 가상 자금과 실제 자금이 섞이지 않는가")

    from engine import autotrade

    user = TEST_USER + 5
    with store._conn() as conn:
        for table in ("at_orders", "at_position_state", "at_daily"):
            conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user,))

    # 같은 종목을 두 계좌에서 각각 관리
    store.upsert_position_state(user, "paper", "005930", side="long",
                                entry_price=10_000, stop_price=9_500, quantity=10)
    store.upsert_position_state(user, "live", "005930", side="long",
                                entry_price=70_000, stop_price=66_500, quantity=3)
    paper_state = store.get_position_states(user, "paper")["005930"]
    live_state = store.get_position_states(user, "live")["005930"]
    check("분리", "같은 종목이라도 계좌별로 손절선이 따로 관리됨",
          paper_state["entry_price"] == 10_000 and live_state["entry_price"] == 70_000,
          f"모의 {paper_state['stop_price']:,.0f} / 실전 {live_state['stop_price']:,.0f}")

    store.clear_position_state(user, "paper", "005930")
    check("분리", "모의계좌 상태를 지워도 실전 상태는 남음",
          "005930" not in store.get_position_states(user, "paper")
          and "005930" in store.get_position_states(user, "live"))

    # 일자 기준 평가금액 — 이게 섞이면 손실 한도가 오작동합니다
    store.touch_daily(user, "paper", 10_000_000)
    live_day = store.touch_daily(user, "live", 3_000_000)
    paper_day = store.touch_daily(user, "paper", 10_000_000)
    check("분리", "기준 평가금액이 계좌별로 따로 잡힘",
          live_day["start_value"] == 3_000_000 and paper_day["start_value"] == 10_000_000,
          f"실전 {live_day['start_value']:,.0f} / 모의 {paper_day['start_value']:,.0f}")

    # 이 분리가 없으면: 모의 1천만 기준으로 실계좌 300만을 재서 -70% 로 오판
    limits = dict(store.DEFAULT_CONFIG, daily_loss_limit_pct=3)
    correct = risk.RiskEngine(limits, {"total_value": 3_000_000}, [], live_day)
    check("분리", "실계좌가 자기 기준으로 판정됨 (오판 없음)",
          not correct.halt_reasons(), str(correct.halt_reasons()))
    # (대조) 손익 값이 없던 시절의 계산 — 총자산과 기준선의 차이로 재면,
    # 남의 계좌 기준선을 대는 순간 -70% 로 오판했습니다. 지금은 손익(pnl)을
    # 직접 넘기므로 이 경로를 타지 않습니다.
    legacy_day = {k: v for k, v in paper_day.items() if k not in ("pnl", "drawdown_pct")}
    wrong = risk.RiskEngine(limits, {"total_value": 3_000_000}, [], legacy_day)
    check("분리", "(대조) 옛 총자산 기준 계산은 남의 기준선에 오판함",
          bool(wrong.halt_reasons()), str(wrong.halt_reasons()))
    # 새 계산식은 같은 상황에서도 오판하지 않습니다 (손익은 계좌 간 섞이지 않음)
    safe = risk.RiskEngine(limits, {"total_value": 3_000_000}, [], paper_day)
    check("분리", "손익 기준 계산은 기준선이 섞여도 오판 없음",
          not safe.halt_reasons(), str(safe.halt_reasons()))

    # 주문 원장
    _new_order(user, "AAA", broker_order_id="P1", mode="paper")
    _new_order(user, "AAA", broker_order_id="L1", mode="live")
    check("분리", "미체결 주문도 계좌별로 조회됨",
          len(store.open_orders(user, "paper")) == 1
          and len(store.open_orders(user, "live")) == 1
          and len(store.open_orders(user)) == 2)
    check("분리", "모의 미체결이 실전 주문을 막지 않음",
          store.has_open_order(user, "live", "AAA")
          and not store.has_open_order(user, "live", "BBB"))

    # 주문번호에 계좌가 들어가야 UNIQUE 충돌이 안 납니다
    stamp = datetime(2026, 8, 5, 10, 30)
    paper_id = broker.make_client_order_id(user, "005930", "buy", stamp, mode="paper")
    live_id = broker.make_client_order_id(user, "005930", "buy", stamp, mode="live")
    check("분리", "같은 시각·같은 종목이어도 계좌가 다르면 다른 주문번호",
          paper_id != live_id, f"{paper_id} vs {live_id}")

    # 유니버스 밖 종목 보호
    cfg = dict(store.DEFAULT_CONFIG, universe=["005930"], manage_only_universe=True)
    check("보호", "유니버스 종목은 관리 대상",
          autotrade.is_managed(cfg, "005930", {}))
    check("보호", "계좌에만 있는 다른 종목은 건드리지 않음",
          not autotrade.is_managed(cfg, "379800", {}),
          "사용자가 직접 산 주식을 자동매매가 팔면 안 됩니다")
    check("보호", "엔진이 직접 잡은 포지션은 유니버스에서 빼도 계속 관리",
          autotrade.is_managed(cfg, "379800", {"379800": {"side": "long"}}))
    check("보호", "옵션을 끄면 계좌 전체를 관리",
          autotrade.is_managed(dict(cfg, manage_only_universe=False), "379800", {}))


# ---------------------------------------------------------------------------
# 11.5 계좌 교체 — 남의 계좌 성적을 물려받지 않는가
# ---------------------------------------------------------------------------
# 계좌번호를 바꿨는데 '오늘 손익 -6,646원'이 남아 있으면, 새 계좌가 이미 손실을
# 안고 시작한 것처럼 보일 뿐 아니라 일일 손실 한도가 그 손실로 곧바로 걸립니다.

def test_account_switch_reset():
    section("11.5 계좌 교체 — 성적이 0원에서 다시 시작하는가")

    user = TEST_USER + 6
    store.init()                       # at_account 가 없는 DB 에서도 아래 정리가 돌게
    with store._conn() as conn:
        for table in ("at_orders", "at_daily", "at_account", "at_events"):
            conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user,))

    old_acc, new_acc = "aaaa1111aaaa1111", "bbbb2222bbbb2222"

    # 옛 계좌에서 이틀치 성적을 쌓습니다
    store.sync_account(user, "live", old_acc)
    store.touch_daily(user, "live", 1_000_000, unrealized=0, cash=1_000_000,
                      trade_date="2026-08-10")
    store.touch_daily(user, "live", 1_050_000, unrealized=50_000, cash=1_000_000,
                      trade_date="2026-08-10")
    store.touch_daily(user, "live", 1_050_000, unrealized=0, cash=1_050_000)
    store.record_daily_trade(user, "live", realized_pnl=-30_000)
    store.touch_daily(user, "live", 1_020_000, unrealized=0, cash=1_020_000)
    store.record_order(user, {
        "client_order_id": f"{user}:live:TEST:sell:switch", "broker_mode": "live",
        "symbol": "005930", "action": "sell", "quantity": 1, "price": 70_000,
        "status": "filled", "realized_pnl": -30_000})

    before = store.summary(user, mode="live")
    check("교체", "(준비) 옛 계좌에 성적이 쌓여 있음",
          before["cumulative_pnl"] != 0 and before["orders"] == 1,
          f"누적 {before['cumulative_pnl']:+,.0f}원 · 주문 {before['orders']}건")

    # 같은 계좌로 다시 확인 — 아무것도 지워지면 안 됩니다
    check("교체", "계좌가 그대로면 성적을 건드리지 않음",
          store.sync_account(user, "live", old_acc) is False
          and store.summary(user, mode="live")["cumulative_pnl"]
              == before["cumulative_pnl"])

    # 지문을 못 읽은 상태(키 조회 실패)도 초기화 사유가 아닙니다
    check("교체", "계좌를 못 읽었을 때는 성적을 지우지 않음",
          store.sync_account(user, "live", "") is False
          and store.summary(user, mode="live")["cumulative_pnl"]
              == before["cumulative_pnl"],
          "Mongo 가 잠깐 끊긴 것과 계좌를 바꾼 것은 다릅니다")

    # 계좌 교체
    check("교체", "계좌가 바뀌면 초기화가 일어남",
          store.sync_account(user, "live", new_acc) is True)

    after = store.summary(user, mode="live")
    check("교체", "누적 손익이 0원", after["cumulative_pnl"] == 0,
          f"{after['cumulative_pnl']:+,.0f}원")
    check("교체", "옛 계좌의 주문·실현손익이 집계에서 빠짐",
          after["orders"] == 0 and after["realized_pnl"] == 0,
          f"주문 {after['orders']}건 · 실현 {after['realized_pnl']:+,.0f}원")
    check("교체", "주문 원장 자체는 남아 있음 (확인할 수 있어야 합니다)",
          len([o for o in store.get_orders(user, limit=10)
               if o["broker_mode"] == "live"]) == 1)

    # 새 계좌의 첫 회전 — 지금 평가손익이 기준선이 되어 오늘 손익은 0
    day = store.touch_daily(user, "live", 3_000_000, unrealized=120_000,
                            cash=2_880_000)
    check("교체", "새 계좌의 오늘 손익이 0원에서 시작",
          day["pnl"] == 0,
          f"평가손익 +120,000원을 물려받아도 오늘 손익 {day['pnl']:+,.0f}원")

    day2 = store.touch_daily(user, "live", 3_020_000, unrealized=140_000,
                             cash=2_880_000)
    check("교체", "그 뒤의 변화는 정상적으로 잡힘", day2["pnl"] == 20_000,
          f"{day2['pnl']:+,.0f}원")

    # 계좌를 바꿔도 다른 모드(내부 모의계좌)는 그대로여야 합니다
    store.sync_account(user, "paper", "paper")
    store.touch_daily(user, "paper", 10_000_000, unrealized=0, cash=10_000_000)
    store.touch_daily(user, "paper", 10_100_000, unrealized=100_000, cash=10_000_000)
    store.sync_account(user, "live", "cccc3333cccc3333")
    check("교체", "KIS 계좌를 바꿔도 내부 모의계좌 성적은 그대로",
          store.summary(user, mode="paper")["cumulative_pnl"] == 100_000,
          f"{store.summary(user, mode='paper')['cumulative_pnl']:+,.0f}원")

    check("교체", "초기화가 기록으로 남음",
          any(e["kind"] == "reset" for e in store.get_events(user, limit=20)),
          "숫자가 갑자기 0이 된 이유를 화면에서 찾을 수 있어야 합니다")


# ---------------------------------------------------------------------------
# 12. 페니주식 초단타 — 안전 한도가 설정으로 뚫리는가
# ---------------------------------------------------------------------------

def _candidate(key="900001", price=800, value=30e8, change=3.0, market="KOSDAQ",
               vol_increase=400.0, turnover=3.0, day_range=8.0):
    """타점 세 지표를 기본으로 '통과하는' 값으로 채운 후보.

    지표를 비워두면 심사에서 '모름'으로 빠져나가, 정작 그 조건이 도는지를
    확인하지 못합니다. 그래서 통과값을 기본으로 두고, 개별 테스트가 필요한
    항목만 미달로 덮어씁니다.
    """
    from data_sources.screener import Candidate
    return Candidate(key=key, name=f"테스트{key}", market=market, price=price,
                     price_krw=price, change_rate=change, volume=value / max(price, 1),
                     trading_value=value, trading_value_krw=value,
                     vol_increase_pct=vol_increase, turnover_pct=turnover,
                     day_range_pct=day_range, source="테스트")


def test_penny_scalping():
    section("12. 페니주식 초단타 — 틱 매매 · 예산 통제")

    from data_sources import kis_realtime
    from engine import scalping

    # --- 호가단위: 모든 틱 계산의 출발점 ---
    check("틱", "호가단위가 가격대별로 맞음",
          [scalping.tick_size(p) for p in (100, 1_999, 2_000, 4_999, 5_000, 20_000)]
          == [1, 1, 5, 5, 10, 50],
          str([scalping.tick_size(p) for p in (100, 1_999, 2_000, 5_000)]))
    check("틱", "가격대 경계를 넘으면 호가단위도 바뀜",
          scalping.ticks_to_price(1_998, 3) == 2_005,
          f"1,998 +3틱 → {scalping.ticks_to_price(1_998, 3):,.0f} (단순 덧셈이면 2,001)")
    check("틱", "아래로도 경계를 제대로 넘음",
          scalping.ticks_to_price(2_000, -1) == 1_999 and scalping.ticks_to_price(2_005, -1) == 2_000,
          f"2,000 -1틱 → {scalping.ticks_to_price(2_000, -1):,.0f}")

    # --- 왕복비용: 몇 틱을 먹어야 본전인가 ---
    econ_100 = scalping.tick_economics(100, {})
    econ_1000 = scalping.tick_economics(1_000, {})
    check("비용", "체결가 기준 손익분기 틱을 계산",
          econ_100["breakeven_ticks"] >= 1 and econ_1000["breakeven_ticks"] >= 1,
          f"100원 {econ_100['breakeven_ticks']}틱 / 1,000원 {econ_1000['breakeven_ticks']}틱")
    check("비용", "틱 대비 비용은 저가일수록 유리",
          econ_100["net_win_ticks"] > econ_1000["net_win_ticks"],
          f"실질이익 100원 {econ_100['net_win_ticks']}틱 vs 1,000원 {econ_1000['net_win_ticks']}틱")
    check("비용", "필요 승률을 숨기지 않고 계산",
          econ_100["required_win_rate"] == 60.0 and econ_1000["required_win_rate"] == 80.0,
          f"100원 {econ_100['required_win_rate']}% / 1,000원 {econ_1000['required_win_rate']}%")
    thin_edge = scalping.tick_economics(1_000, {"take_profit_ticks": 2, "min_net_ticks": 1})
    check("비용", "비용을 빼고 남는 게 없으면 진입 불가로 판정",
          not thin_edge["viable"],
          f"익절 2틱 − 비용 {thin_edge['breakeven_ticks']}틱 = {thin_edge['net_win_ticks']}틱")

    # --- 하드 한도: 사용자가 크게 넣어도 잘려야 합니다 ---
    wild = scalping.clamp_config({
        "stop_loss_ticks": 99, "max_hold_sec": 86_400,
        "budget_krw": 999_999_999, "daily_loss_krw": 5_000_000_000,
        "reentry_cooldown_sec": 0, "universe_refresh_sec": 1,
        "kr_price_range": [10, 100_000], "us_price_range": [0.01, 900],
    })
    limits = scalping.HARD_LIMITS
    check("한도", "손절 틱이 한도로 잘림",
          wild["stop_loss_ticks"] == limits["max_stop_loss_ticks"],
          f"99틱 → {wild['stop_loss_ticks']}틱")
    check("한도", "보유 시간이 한도로 잘림",
          wild["max_hold_sec"] == limits["max_hold_sec"],
          f"86,400초 → {wild['max_hold_sec']}초")
    check("한도", "투자금액이 한도로 잘림",
          wild["budget_krw"] == limits["max_budget_krw"],
          f"→ {wild['budget_krw']:,.0f}원")
    check("한도", "일일 손실 한도는 투자금액을 못 넘음",
          wild["daily_loss_krw"] <= wild["budget_krw"],
          f"{wild['daily_loss_krw']:,.0f}원 ≤ {wild['budget_krw']:,.0f}원")
    check("한도", "재진입 쿨다운·갱신주기는 최소값으로 올려짐",
          wild["reentry_cooldown_sec"] == limits["min_reentry_cooldown_sec"]
          and wild["universe_refresh_sec"] == limits["min_universe_refresh_sec"],
          f"쿨다운 {wild['reentry_cooldown_sec']}초 / 갱신 {wild['universe_refresh_sec']}초")
    check("한도", "가격대가 한도 안으로 좁혀짐",
          wild["kr_price_range"] == [limits["kr_min_price"], limits["kr_max_price"]]
          and wild["us_price_range"] == [limits["us_min_price"], limits["us_max_price"]],
          f"{wild['kr_price_range']} / {wild['us_price_range']}")
    check("한도", "무엇이 잘렸는지 사용자에게 알려줌",
          len(wild["_clamped"]) >= 6, f"{len(wild['_clamped'])}건 안내")
    check("한도", "100원짜리가 대상 하한 안에 들어옴",
          limits["kr_min_price"] <= 100, f"하한 {limits['kr_min_price']}원")
    check("한도", "동시 추적 수는 시세 구독 한도 이하",
          scalping.max_tracked() <= kis_realtime.MAX_SUBSCRIPTIONS,
          f"{scalping.max_tracked()}종목 ≤ 구독 {kis_realtime.MAX_SUBSCRIPTIONS}건")

    # --- 종목 자격 심사 ---
    cfg = scalping.clamp_config({})
    ok = scalping.screen_candidate(_candidate(price=800, value=30e8, change=3), cfg)
    check("심사", "정상 후보는 통과", ok.ok, str(ok.reasons))

    thin = scalping.screen_candidate(_candidate(value=2e8), cfg)
    check("심사", "거래대금 부족 → 탈락 (못 빠져나옴)", not thin.ok, thin.reasons[0][:60])

    limit_up = scalping.screen_candidate(_candidate(change=29), cfg)
    check("심사", "상한가 근처 → 탈락 (거래가 멈춰 못 팜)", not limit_up.ok,
          limit_up.reasons[0][:50])
    check("심사", "변동폭이 큰 것 자체는 막지 않음 (타점이므로)",
          scalping.screen_candidate(_candidate(change=18, day_range=25), cfg).ok,
          "+18% / 변동폭 25% 통과")

    # --- 타점 세 지표 ---
    check("타점", "거래량 증가율 미달 → 탈락",
          not scalping.screen_candidate(_candidate(vol_increase=30), cfg).ok)
    check("타점", "회전율 미달 → 탈락",
          not scalping.screen_candidate(_candidate(turnover=0.2), cfg).ok)
    check("타점", "변동폭 미달 → 탈락 (먹을 폭이 없음)",
          not scalping.screen_candidate(_candidate(day_range=0.8), cfg).ok)
    unknown = _candidate(vol_increase=None, turnover=None, day_range=None)
    check("타점", "지표를 못 받으면 0 이 아니라 '모름'으로 통과",
          scalping.screen_candidate(unknown, cfg).ok,
          "None 을 0 으로 읽으면 멀쩡한 종목이 전부 탈락합니다")

    flagged = _candidate()
    flagged.flags = ["관리종목"]
    check("심사", "위험 표식 종목 → 탈락",
          not scalping.screen_candidate(flagged, cfg).ok)

    many = scalping.screen_candidate(
        _candidate(price=90_000, value=1e8, vol_increase=10, turnover=0.1, day_range=0.5), cfg)
    check("심사", "탈락 사유를 모두 모아서 보여줌", len(many.reasons) >= 4,
          f"{len(many.reasons)}건")

    # --- 예산 기반 추천 (비율이 아님) ---
    pool = [_candidate("900001", price=800, value=50e8),
            _candidate("900002", price=2_500, value=20e8),
            _candidate("900003", price=150, value=1e8)]      # 거래대금 부적합
    budget_cfg = scalping.clamp_config({"budget_krw": 100_000})
    ranked = scalping.recommend(pool, available_cash=300_000, total_value=1_000_000,
                                cfg=budget_cfg)
    check("추천", "적합·매수가능 종목이 위로", ranked[0]["eligible"] and ranked[0]["affordable"],
          f"1위 {ranked[0]['key']} {ranked[0]['max_quantity']}주 = {ranked[0]['order_krw']:,}원")
    check("추천", "부적합 종목은 아래로", not ranked[-1]["eligible"], ranked[-1]["key"])
    check("추천", "주문금액이 투자금액을 넘지 않음",
          all(c["order_krw"] <= 100_000 for c in ranked),
          f"최대 {max(c['order_krw'] for c in ranked):,}원 ≤ 100,000원")
    check("추천", "계좌 현금이 투자금액보다 적으면 현금이 상한",
          scalping.recommend([_candidate(price=800)], available_cash=5_000,
                             total_value=1_000_000, cfg=budget_cfg)[0]["budget_cap"] == 5_000)

    # 소액 계좌 — 예전 비율 기반에서는 여기서 0주가 나왔습니다
    tiny = scalping.recommend([_candidate("900004", price=100, value=50e8)],
                              available_cash=1_000, total_value=1_000,
                              cfg=scalping.clamp_config({"budget_krw": 1_000}))
    check("추천", "총자산 1,000원으로도 100원짜리를 살 수 있음",
          tiny[0]["affordable"] and tiny[0]["max_quantity"] == 10,
          f"{tiny[0]['max_quantity']}주 = {tiny[0]['order_krw']:,}원 "
          f"(비율 기반이었다면 5% = 50원 → 0주)")

    check("추천", "내 주문이 거래대금 대비 얼마나 큰지 계산",
          "impact_pct" in ranked[0], f"{ranked[0]['impact_pct']}%")

    # --- 주문 계획: 틱 → 실제 가격·수량 ---
    inst_p = fake_stock(key="900001", name="테스트동전주", market="KOSDAQ")
    quote_p = {**fake_quote(100), "bid": 100, "ask": 101, "trading_value": 50e8}
    plan = scalping.plan_order(inst_p, quote_p, budget_cfg, budget_left=10_000,
                               trading_value_krw=50e8)
    check("주문", "진입은 지정가(매수호가)가 기본",
          plan.order_type == "limit" and plan.entry_price == 100,
          f"{plan.order_type} @ {plan.entry_price:,.0f} (시장가면 101에 체결 = 1틱 손실)")
    check("주문", "수량은 남은 예산으로만 결정",
          plan.quantity == 100, f"10,000원 / 100원 = {plan.quantity}주")
    check("주문", "목표·손절가가 호가단위에 정렬",
          plan.target_price == 103 and plan.stop_price == 98,
          f"목표 {plan.target_price:,.0f} / 손절 {plan.stop_price:,.0f}")

    broke = scalping.plan_order(inst_p, quote_p, budget_cfg, budget_left=50)
    check("주문", "예산이 1주 값보다 적으면 주문하지 않음", not broke.ok, broke.reasons[0][:60])

    huge = scalping.plan_order(inst_p, quote_p, budget_cfg, budget_left=1_000_000,
                               trading_value_krw=1e8)
    check("주문", "거래대금 대비 주문이 크면 수량을 줄임",
          huge.ok and huge.order_krw <= 1e8 * limits["max_order_impact_pct"] / 100 + 100,
          f"{huge.quantity:,}주 = {huge.order_krw:,.0f}원")

    wide = scalping.plan_order(inst_p, {**quote_p, "bid": 100, "ask": 106},
                               budget_cfg, budget_left=10_000)
    spread = scalping.spread_ticks({"bid": 100, "ask": 106})
    check("주문", "스프레드를 틱으로 잴 수 있음", spread == 6, f"{spread}틱")
    check("주문", "지정가 진입이면 넓은 스프레드에도 계획은 성립 (게이트는 신호에서)",
          wide.ok, f"{wide.quantity}주")

    # --- 초단타 신호 ---
    # 페니 가격대(수백 원)로 만듭니다. 만 원대 봉을 쓰면 호가단위가 10원이라
    # 익절 3틱이 왕복비용도 못 넘어서, 신호를 보기 전에 경제성 게이트에서
    # 먼저 걸립니다 (그게 정상 동작이므로 테스트 입력을 맞춥니다).
    bars = trend_bars(n=120, start=500.0, drift=0.0015, sigma=0.004, seed=3)
    inst = fake_stock(key="900001", name="테스트동전주", market="KOSDAQ")

    def penny_quote(price):
        """호가가 1틱만 벌어진 정상 상태 (스프레드 게이트를 통과시킵니다)."""
        price = round(float(price))
        return {**fake_quote(price), "bid": price, "ask": price + 1}

    quiet = scalping.evaluate(inst, cfg, bars=bars,
                              quote=penny_quote(bars["close"].iloc[-1]),
                              allow_fetch=False)
    check("신호", "분봉 신호 생성", quiet.ok and quiet.vwap is not None,
          f"점수 {quiet.score:+.2f} · VWAP {quiet.vwap:,.0f} · RVOL {quiet.rvol:.2f}")
    check("신호", "거래량이 안 터지면 진입 안 함",
          quiet.direction == scalping.FLAT or quiet.rvol >= 2.0,
          f"RVOL {quiet.rvol:.2f} / 방향 {quiet.direction}")

    hot = bars.copy()
    hot.iloc[-5:, hot.columns.get_loc("volume")] *= 8      # 거래량 급증
    hot.iloc[-5:, hot.columns.get_loc("close")] *= 1.02    # 가격도 상승
    hot.iloc[-5:, hot.columns.get_loc("high")] *= 1.03
    fired = scalping.evaluate(inst, cfg, bars=hot,
                              quote=penny_quote(hot["close"].iloc[-1]),
                              allow_fetch=False)
    check("신호", "거래량 급증 + 상승이면 점수 상승",
          fired.score > quiet.score, f"{quiet.score:+.2f} → {fired.score:+.2f}")
    check("신호", "판단 근거가 남음", len(fired.reasons) >= 1, str(fired.reasons[:3]))

    check("신호", "진입 임계값은 0.45 아래로 못 내림",
          scalping.evaluate(inst, dict(cfg, entry_score=0.01), bars=bars,
                            quote=penny_quote(500), allow_fetch=False).direction
          == scalping.FLAT or quiet.score >= 0.45)

    # --- 스프레드 게이트: 아무리 좋은 신호도 호가가 나쁘면 안 들어감 ---
    wide_quote = {**fake_quote(100), "bid": 100, "ask": 106}
    blocked = scalping.evaluate(inst, cfg, bars=hot, quote=wide_quote, allow_fetch=False)
    check("신호", "스프레드가 한도를 넘으면 진입 금지",
          blocked.direction == scalping.FLAT and blocked.ok,
          str(blocked.reasons[:1]))
    check("신호", "손익 구조를 신호에 같이 담아 보여줌",
          "required_win_rate" in (blocked.economics or {}),
          f"필요 승률 {blocked.economics.get('required_win_rate')}%")

    # --- 경고 문구 ---
    warnings = scalping.risk_warnings({"stop_loss_ticks": 99})
    check("경고", "위험 경고가 항상 나옴", len(warnings) >= 5, f"{len(warnings)}줄")
    check("경고", "한도로 잘린 사실을 경고에 포함",
          any("제한" in w for w in warnings))
    check("경고", "매매 횟수 제한이 없다는 사실을 알림",
          any("횟수 제한이 없" in w for w in warnings))
    check("경고", "한도 표를 화면용으로 제공",
          len(scalping.describe_hard_limits()) >= 10)

    # --- 저장 단계에서도 잘리는가 ---
    saved = store.save_config(TEST_USER, {
        "scalp": {"stop_loss_ticks": 50, "budget_krw": 99_999_999}})
    check("한도", "DB 저장 시점에 이미 잘려서 들어감",
          saved["scalp"]["stop_loss_ticks"] == limits["max_stop_loss_ticks"]
          and saved["scalp"]["budget_krw"] == limits["max_budget_krw"],
          f"손절 {saved['scalp']['stop_loss_ticks']}틱 / "
          f"투자금액 {saved['scalp']['budget_krw']:,.0f}원")

    # --- 자산군 제한 · 대상 아님 사유 ---
    fut = instruments.parse_derivative("101H6000")
    check("한도", "파생상품은 초단타 대상이 아님",
          not scalping.is_penny_target(fut, 350, cfg))
    check("한도", "가격대 밖 주식도 대상 아님",
          not scalping.is_penny_target(fake_stock(), 50_000, cfg)
          and scalping.is_penny_target(fake_stock(), 800, cfg))
    check("진단", "대상이 아닌 이유를 말해줌 (조용히 넘어가지 않음)",
          "상한" in scalping.why_not_target(fake_stock(), 50_000, cfg)
          and "주식" in scalping.why_not_target(fut, 350, cfg),
          scalping.why_not_target(fake_stock(), 50_000, cfg))

    # --- 예산 통제: 이미 들어간 돈만큼 줄어드는가 ---
    from engine.autotrade import scalp_budget_left
    from engine.broker import Position

    # 예산은 **초단타 목록에 들어간 돈만** 셉니다. 자동매매 콘솔의 universe 는
    # 다른 전략의 것이라, 여기에 섞으면 초단타가 한 주도 못 삽니다.
    held_pos = [Position(key="900001", name="보유중", asset_class=STOCK, side="long",
                         quantity=50, avg_price=800, market_value=40_000)]
    left_cfg = {"scalp": {"budget_krw": 100_000, "universe": ["900001"]}}
    check("예산", "이미 들어간 평가금액만큼 남은 예산이 줄어듦",
          scalp_budget_left(left_cfg, held_pos) == 60_000,
          f"100,000 − 40,000 = {scalp_budget_left(left_cfg, held_pos):,.0f}원")
    check("예산", "예산을 넘게 들고 있으면 0 (음수로 안 감)",
          scalp_budget_left({"scalp": {"budget_krw": 30_000,
                                       "universe": ["900001"]}}, held_pos) == 0)
    check("예산", "초단타 목록 밖 종목은 이 전략의 예산에 안 잡힘",
          scalp_budget_left({"scalp": {"budget_krw": 100_000,
                                       "universe": ["999999"]}}, held_pos) == 100_000,
          "다른 전략이 들고 있는 종목까지 세면 초단타가 굶습니다")
    check("분리", "자동매매 universe 는 초단타 예산에 영향을 주지 않음",
          scalp_budget_left({"universe": ["900001"],
                             "scalp": {"budget_krw": 100_000}}, held_pos) == 100_000,
          "콘솔이 산 종목이 초단타 예산을 먹으면 안 됩니다")

    # --- 오늘 확정 손익 (일일 손실 한도의 근거) ---
    today = store.realized_today(TEST_USER, "paper", symbols=["900001"])
    check("한도", "종목별 오늘 확정손익을 금액으로 집계",
          "realized_pnl" in today and "closed_count" in today,
          f"{today['realized_pnl']:+,.0f}원 / {today['closed_count']}건")


# ---------------------------------------------------------------------------
# 7-b. 백테스트 검증 — 좋은 숫자를 의심하는 도구
# ---------------------------------------------------------------------------
# 백테스트에서 좋은 숫자가 나오는 경로는 세 가지고, 그중 둘은 알파가 아닙니다.
# (1) 진짜 우위 (2) 그냥 시장이 올랐다 (3) 여러 번 돌려 최고를 골랐다.
# 여기서 보는 것은 우리 도구가 (2)와 (3)을 실제로 걸러내는가입니다.

def test_validation():
    section("7-b. 백테스트 검증 — 초과수익 · 위조검증 · 선택편향")

    import numpy as np

    from engine import validation as V

    rng = np.random.default_rng(11)
    bench_r = rng.normal(0.0003, 0.010, 400)
    curve = lambda r: [{"equity": float(e)} for e in 100 * np.cumprod(1 + r)]

    # --- 시장이 올랐을 뿐인 전략을 가려내는가 ---
    follower = curve(bench_r - 0.0004)            # 시장 그대로 + 비용만 지출
    winner = curve(bench_r + rng.normal(0.0006, 0.003, 400))
    m_follow = V.active_metrics(follower, curve(bench_r))
    m_win = V.active_metrics(winner, curve(bench_r))
    check("초과수익", "시장을 따라가기만 하면 초과수익이 음수",
          m_follow["excess_return_pct"] < 0 and not m_follow["beat_benchmark"],
          f"총수익 {m_follow['cagr_pct']}% 인데 초과 {m_follow['excess_return_pct']}%")
    check("초과수익", "진짜로 이기면 액티브 샤프가 양수",
          m_win["active_sharpe"] > 0 and m_win["beat_benchmark"],
          f"액티브샤프 {m_win['active_sharpe']}")
    check("초과수익", "총수익률만으로는 구분되지 않음 (그래서 초과수익이 필요)",
          m_follow["cagr_pct"] > 0,
          f"따라가기 전략도 총수익 {m_follow['cagr_pct']}% — 숫자만 보면 좋아 보입니다")

    # --- 블록 부트스트랩: 값은 보존, 순서만 파괴 ---
    r = np.array([0.01, -0.02, 0.03, -0.01, 0.005, 0.02])
    boot = V.block_bootstrap(r, block=3, rng=np.random.default_rng(0))
    check("위조", "블록 부트스트랩이 길이와 값 집합을 보존",
          len(boot) == len(r) and set(np.round(boot, 6)) <= set(np.round(r, 6)))

    # --- 위조검증 p-value ---
    nulls = list(rng.normal(0.0, 1.0, 20))
    buried = V.falsification_audit(0.1, nulls)
    stands = V.falsification_audit(9.0, nulls)
    check("위조", "귀무분포에 묻히면 FAIL", buried["passed"] is False,
          f"p={buried['p_value']}")
    check("위조", "귀무분포 밖으로 튀면 PASS", stands["passed"] is True,
          f"p={stands['p_value']}")
    check("위조", "시행이 적으면 통과 자체가 불가능 (최소 p=1/(n+1))",
          V.falsification_audit(9.0, list(rng.normal(0, 1, 5)))["passed"] is False,
          "5회로는 아무리 튀어도 p=0.167 — 표본 부족을 유의성으로 포장하지 않습니다")
    check("위조", "귀무 표본이 없으면 판정하지 않음",
          V.falsification_audit(1.0, [])["passed"] is None)

    # --- DSR: 시행이 늘수록 깎인다 ---
    trials = [0.9, 1.2, 0.7, 1.0]
    d10 = V.deflated_sharpe_ratio(1.2, trials, n_obs=250, n_trials=10)
    d100 = V.deflated_sharpe_ratio(1.2, trials, n_obs=250, n_trials=100)
    check("선택편향", "시행 횟수가 늘면 DSR이 낮아짐",
          d100["dsr"] < d10["dsr"],
          f"10회 {d10['dsr']} → 100회 {d100['dsr']}")
    check("선택편향", "귀무 하 기대 최대 샤프가 시행과 함께 커짐",
          d100["expected_max_sr_under_null"] > d10["expected_max_sr_under_null"],
          f"{d10['expected_max_sr_under_null']} → {d100['expected_max_sr_under_null']}")
    check("선택편향", "연율화 샤프를 그대로 넣어도 포화되지 않음 (√주기 보정)",
          0.0 < d10["dsr"] < 1.0,
          "보정을 빼먹으면 z가 √252배 부풀어 무엇이든 유의하게 나옵니다")
    check("선택편향", "약한 성과는 20회 시행 뒤 유의하지 않음",
          V.deflated_sharpe_ratio(0.5, trials, n_obs=250,
                                  n_trials=20)["significant_at_95"] is False)

    # --- t통계 헬스 게이트 ---
    check("헬스", "표본이 모자라면 판정하지 않음 (워밍업)",
          V.health_from_tstat([-1] * 5)["warming_up"] is True)
    noisy = V.health_from_tstat(list(rng.normal(-0.05, 3.0, 40)))
    broken = V.health_from_tstat(list(rng.normal(-1.5, 1.0, 40)))
    check("헬스", "평균이 음수여도 노이즈면 중단하지 않음",
          noisy["state"] != V.HEALTH_SUSPEND,
          f"t={noisy['t_stat']} → {noisy['label']}")
    check("헬스", "유의하게 나쁘면 중단",
          broken["state"] == V.HEALTH_SUSPEND, f"t={broken['t_stat']}")


# ---------------------------------------------------------------------------
# 2-b. NNFX 규칙 오버레이
# ---------------------------------------------------------------------------
# 원본 NNFX(전 슬롯 AND)는 주식에서 굶습니다. 여기서 지키려는 것은
# "소프트는 신호를 밀어주기만 하고, 게이트는 진입만 막는다"는 경계입니다.

def test_nnfx():
    section("2-b. NNFX 오버레이 — 소프트 결합 · 게이트")

    from engine import nnfx

    up = trend_bars(n=220, drift=0.006, seed=7)
    down = trend_bars(n=220, drift=-0.006, seed=7)
    inst = fake_stock(key="NNFXT")
    quote = lambda b: {"price": float(b["close"].iloc[-1]), "age_sec": 0}

    s_up, s_down = nnfx.compute(up, {}), nnfx.compute(down, {})
    check("NNFX", "상승 추세에서 점수가 양수", s_up.ok and s_up.score > 0,
          f"score={s_up.score:+.2f} slots={s_up.slots}")
    check("NNFX", "하락 추세에서 점수가 음수", s_down.score < 0,
          f"score={s_down.score:+.2f}")
    check("NNFX", "기준선 아래면 거부권 발동", s_down.veto_ok is False,
          str(s_down.reasons[:1]))
    check("NNFX", "일봉이 모자라면 판정하지 않음",
          nnfx.compute(up.iloc[:20], {}).ok is False)

    # 같은 데이터에 모드만 바꿔 넣습니다 (임계값을 낮춰 기본 신호를 롱으로 만든 뒤)
    base = {"entry_score": 0.15, "intraday_weight": 0.0}
    sig = lambda mode, bars: strategy.evaluate(
        inst, {**base, "nnfx_mode": mode}, bars_daily=bars,
        quote=quote(bars), allow_fetch=False)

    off, soft, veto, hard = (sig(m, up) for m in ("off", "soft", "veto", "hard"))
    check("NNFX", "off 는 점수를 건드리지 않음", off.score == veto.score == hard.score,
          f"{off.score:+.3f}")
    check("NNFX", "soft 는 점수를 섞는다 (더하지 않음)",
          soft.score != off.score and abs(soft.score) <= 1.0,
          f"{off.score:+.3f} → {soft.score:+.3f}")
    check("NNFX", "soft 는 신호를 죽이지 않음", soft.direction == strategy.LONG)
    check("NNFX", "hard(원본 AND)는 슬롯 하나만 어긋나도 진입을 막음",
          hard.direction == strategy.FLAT and off.direction == strategy.LONG,
          "실측에서 통과율 10%로 굶었던 바로 그 동작입니다")
    check("NNFX", "차단 사유가 신호에 남는다",
          any("NNFX" in r for r in hard.reasons), str(hard.reasons[:1]))
    check("NNFX", "소프트 결합 사유도 남는다",
          any("NNFX" in r for r in soft.reasons), str(soft.reasons[:1]))
    check("NNFX", "오버레이 상태를 신호에 실어 보냄",
          (soft.to_dict().get("nnfx") or {}).get("slots") is not None
          and off.to_dict().get("nnfx") is None,
          "off 일 때는 붙이지 않습니다")

    # 가중치 0이면 off 와 같아야 합니다 (경계값)
    zero = strategy.evaluate(inst, {**base, "nnfx_mode": "soft", "nnfx_soft_weight": 0.0},
                             bars_daily=up, quote=quote(up), allow_fetch=False)
    check("NNFX", "가중 0 은 off 와 같은 점수", abs(zero.score - off.score) < 1e-9)


# ---------------------------------------------------------------------------
# 12-c. 초단타 ↔ 자동매매 분리
# ---------------------------------------------------------------------------
# 둘은 같은 계좌를 쓰지만 매매 대상도 청산 규칙도 다릅니다. 섞이면 초단타
# 스크리너가 콘솔의 매매 대상을 밀어내고, 콘솔 종목이 3틱에 던져집니다.
# 여기서 보는 것은 "각자 자기 목록만 본다"는 경계 하나입니다.

def test_strategy_separation():
    section("12-c. 초단타와 자동매매 분리 — 각자 자기 목록만")

    from engine import autotrade as engine
    from engine.autotrade import OWNER_AUTO, OWNER_SCALP

    user = TEST_USER + 7
    cfg = {"universe": ["005930"], "manual_universe": ["005930"],
           "scalp": {"enabled": True, "universe": ["900001"], "pinned": ["900001"]}}

    check("분리", "초단타 목록은 콘솔 universe 와 별개로 읽힘",
          engine.scalp_universe(cfg) == ["900001"]
          and "005930" not in engine.scalp_universe(cfg),
          f"초단타 {engine.scalp_universe(cfg)} / 콘솔 {cfg['universe']}")

    check("분리", "콘솔 종목은 초단타 대상이 아님",
          "005930" not in set(engine.scalp_universe(cfg)),
          "여기서 섞이면 삼성전자가 초단타 판단을 탑니다")

    check("분리", "두 목록 모두 자동매매 관리 대상 (계좌의 남의 주식은 제외)",
          engine.is_managed(cfg, "005930", {}) and engine.is_managed(cfg, "900001", {})
          and not engine.is_managed(cfg, "068270", {}))

    # --- 포지션의 주인 ---
    check("주인", "진입 때 박아둔 주인이 우선",
          engine.position_owner(cfg, "005930", {"strategy": OWNER_SCALP}) == OWNER_SCALP
          and engine.position_owner(cfg, "900001", {"strategy": OWNER_AUTO}) == OWNER_AUTO,
          "목록이 15초마다 바뀌어도 들고 있는 포지션의 주인은 안 바뀝니다")
    check("주인", "기록이 없으면 현재 목록으로 추정",
          engine.position_owner(cfg, "900001", {}) == OWNER_SCALP
          and engine.position_owner(cfg, "005930", {}) == OWNER_AUTO)
    check("주인", "초단타 목록에서 빠져도 주인은 유지",
          engine.position_owner({"universe": ["005930"], "scalp": {}},
                                "900001", {"strategy": OWNER_SCALP}) == OWNER_SCALP,
          "여기서 주인이 바뀌면 3틱짜리 포지션에 며칠짜리 손절선이 붙습니다")

    # --- 자동 갱신이 콘솔 목록을 건드리지 않는가 ---
    store.save_config(user, {"universe": ["005930"], "manual_universe": ["005930"],
                             "scalp": {"enabled": True, "auto_universe": False,
                                       "pinned": ["900001", "900002"]}})
    saved = store.get_config(user)
    check("저장", "지정 종목이 초단타 목록에 들어감",
          set(saved["scalp"]["universe"]) == {"900001", "900002"},
          str(saved["scalp"]["universe"]))
    check("저장", "초단타를 저장해도 콘솔 매매 대상은 그대로",
          saved["universe"] == ["005930"], str(saved["universe"]))

    store.save_config(user, {"scalp": {"pinned": ["900001"]}})
    saved = store.get_config(user)
    check("저장", "지정에서 빼면 초단타 목록에서도 빠짐",
          saved["scalp"]["universe"] == ["900001"], str(saved["scalp"]["universe"]))

    # --- 진입 경로: 콘솔 종목이 초단타 판단을 타지 않는가 (사고의 핵심) ---
    calls = []

    def fake_scalp_entry(uid, c, inst, quote, engine_risk, result):
        calls.append(inst.key)
        return c, None, None

    class _Risk:
        account, positions = {"available_cash": 0}, []

    live_cfg = {"mode": "paper", "universe": ["TEST01"],
                "scalp": {"enabled": True, "universe": ["TEST02"]}}
    originals = (engine._scalp_entry, engine.feed.quote, engine.strategy.evaluate,
                 engine._resolve_universe_item, engine.feed.entry_allowed_now)
    try:
        engine._scalp_entry = fake_scalp_entry
        engine.feed.quote = lambda inst, **kw: {"price": 500, "age_sec": 0}
        # 이 검사는 '어느 경로로 가는가'만 봅니다 — 장 시간 게이트는 열어 둡니다
        engine.feed.entry_allowed_now = lambda inst, cfg=None: (True, "테스트 개장",
                                                                "REGULAR")
        engine.strategy.evaluate = lambda inst, cfg, **kw: strategy.Signal(
            key=inst.key, ok=True, direction=strategy.FLAT, price=500)
        # 가짜 코드는 종목 해석이 안 되므로, 여기서 실물 대신 넣어줍니다.
        # (없으면 두 경로 모두 '종목을 찾을 수 없습니다'로 빠져 검사가 헛돕니다)
        engine._resolve_universe_item = lambda q: fake_stock(key=str(q), name=str(q))

        engine._handle_entry(user, live_cfg, None, _Risk(), "TEST01", set(),
                             {"rejects": [], "signals": [], "errors": []})
        check("경로", "콘솔 종목은 초단타 판단을 타지 않음", calls == [],
              "초단타를 켜면 콘솔 종목이 '초단타 대상이 아닙니다'로 거부되던 문제")

        engine._handle_entry(user, live_cfg, None, _Risk(), "TEST02", set(),
                             {"rejects": [], "signals": [], "errors": []},
                             scalp=True)
        check("경로", "초단타 종목만 틱 경로로 감", calls == ["TEST02"], str(calls))
    finally:
        (engine._scalp_entry, engine.feed.quote, engine.strategy.evaluate,
         engine._resolve_universe_item, engine.feed.entry_allowed_now) = originals

    # --- 옛 설정 갈라주기 ---
    legacy_user = TEST_USER + 8
    store.save_config(legacy_user, {
        "universe": ["005930", "900001", "900002"],   # 스크리너가 밀어넣은 상태
        "manual_universe": ["005930"],
        "scalp": {"enabled": True, "auto_universe": True, "pinned": ["900001"]}})
    _forget_scalp_universe(legacy_user)      # '분리 전' 저장 상태 재현
    split = engine.split_legacy_universe(legacy_user, store.get_config(legacy_user))
    check("이전", "옛 설정에서 스크리너가 넣은 종목을 콘솔 대상에서 뺌",
          split["universe"] == ["005930"], str(split["universe"]))
    check("이전", "사람이 넣은 종목은 콘솔에 남음",
          "005930" in split["universe"])
    check("이전", "지정 종목은 초단타 목록으로 옮겨감",
          split["scalp"]["universe"] == ["900001"], str(split["scalp"]["universe"]))
    check("이전", "두 번 불러도 다시 건드리지 않음 (멱등)",
          engine.split_legacy_universe(legacy_user, split)["universe"] == ["005930"])


def _forget_scalp_universe(user_id):
    """'분리 전' 저장 상태를 재현합니다.

    save_config 는 이제 scalp.universe 를 항상 채우므로, 옛 설정을 만들려면
    저장된 JSON 에서 그 키를 직접 지워야 합니다.
    """
    import json as _json

    cfg = store.get_config(user_id)
    stored = {k: v for k, v in cfg.items()
              if k not in ("enabled", "state", "state_reason", "updated_at")}
    stored["scalp"] = {k: v for k, v in (cfg.get("scalp") or {}).items()
                       if k != "universe"}
    with store._conn() as conn:
        conn.execute("UPDATE at_config SET config = ? WHERE user_id = ?",
                     (_json.dumps(stored, ensure_ascii=False), user_id))


# ---------------------------------------------------------------------------
# 12-b. 초봉 차트 — 체결을 모아 만드는 봉 · 매수/매도 지점
# ---------------------------------------------------------------------------

def _tick_frame(code, price, volume, hhmmss):
    """H0STCNT0 프레임 하나 (46필드)."""
    from data_sources import kis_realtime as rt
    row = ["0"] * 46
    row[rt.F_CODE] = code
    row[rt.F_TIME] = hhmmss
    row[rt.F_PRICE] = str(price)
    row[rt.F_ASK] = str(price + 1)
    row[rt.F_BID] = str(price)
    row[rt.F_TICK_VOL] = str(volume)
    row[rt.F_ACML_VOL] = "500000"
    row[rt.F_STRENGTH] = "110"
    row[rt.F_VOL_RATIO] = "400"
    row[rt.F_HIGH] = str(price + 5)
    row[rt.F_LOW] = str(price - 5)
    return "0|H0STCNT0|001|" + "^".join(row)


class _NullWS:
    def send(self, message):
        pass


def test_scalp_chart():
    section("12-b. 초봉 차트 — 체결 집계와 매매 지점")

    from data_sources import kis_realtime as rt
    from engine import scalping

    stream = rt.TickStream()
    ws = _NullWS()

    # 09:00:00 에 3체결 / 09:00:01 에 2체결 / 09:00:03 에 1체결 (02초는 체결 없음)
    for price, vol in [(100, 10), (103, 5), (101, 8)]:
        stream._handle(ws, _tick_frame("123456", price, vol, "090000"))
    for price, vol in [(102, 7), (99, 3)]:
        stream._handle(ws, _tick_frame("123456", price, vol, "090001"))
    stream._handle(ws, _tick_frame("123456", 105, 20, "090003"))

    one = stream.bars("123456", 1)
    check("초봉", "체결을 1초봉으로 접음", len(one) == 3, f"{len(one)}봉")
    check("초봉", "봉 안에서 OHLC 가 맞음",
          one[0]["o"] == 100 and one[0]["h"] == 103 and one[0]["l"] == 100 and one[0]["c"] == 101,
          f"O{one[0]['o']:.0f} H{one[0]['h']:.0f} L{one[0]['l']:.0f} C{one[0]['c']:.0f}")
    check("초봉", "거래량·체결건수를 함께 셈",
          one[0]["v"] == 23 and one[0]["n"] == 3, f"{one[0]['v']:.0f}주 / {one[0]['n']}건")
    check("초봉", "체결이 없는 초는 봉을 만들지 않음",
          [b["t"] % 60 for b in one] == [0, 1, 3],
          "빈 초를 채우면 초단타 차트가 대부분 공백이 됩니다")

    five = stream.bars("123456", 5)
    check("초봉", "N초봉으로 묶임",
          len(five) == 1 and five[0]["h"] == 105 and five[0]["l"] == 99
          and five[0]["v"] == 53 and five[0]["n"] == 6,
          f"5초봉 H{five[0]['h']:.0f} L{five[0]['l']:.0f} V{five[0]['v']:.0f} {five[0]['n']}건")
    check("초봉", "묶어도 원본 1초봉이 망가지지 않음",
          len(stream.bars("123456", 1)) == 3)

    stream._handle(ws, _tick_frame("123456", 999, 1, "085959"))
    check("초봉", "시각이 거꾸로 온 틱은 버림",
          stream.bars("123456", 1)[-1]["c"] == 105,
          "받아들이면 봉 순서가 깨져 차트가 뒤엉킵니다")

    check("초봉", "구독 안 한 종목은 빈 목록 (0 을 지어내지 않음)",
          stream.bars("999999", 1) == [])

    # --- 미국 종목: 다른 TR · 다른 필드 배치 ---
    check("해외", "국내는 H0STCNT0, 해외는 HDFSCNT0 으로 갈라짐",
          rt.tr_for("032680") == rt.TR_TICK and rt.tr_for("CISS") == rt.TR_TICK_US)
    check("해외", "구독 키가 거래소별로 만들어짐",
          rt.tr_key_for("032680") == "032680"
          and rt.tr_key_for("CISS", "NCM") == "DNASCISS"
          and rt.tr_key_for("BAC", "NYQ") == "DNYSBAC"
          and rt.tr_key_for("XYZ", "ASE") == "DAMSXYZ",
          "D + 거래소(3) + 심볼")
    check("해외", "거래소를 모르면 나스닥으로 가정",
          rt.tr_key_for("CISS") == "DNAS" + "CISS")

    now = datetime.now()
    fresh = f"{now.hour:02d}{now.minute:02d}{now.second:02d}"
    us_row = ["0"] * 26
    us_row[rt.U_RSYM] = "DNASCISS"
    us_row[rt.U_SYMB] = "CISS"
    us_row[rt.U_KHMS] = fresh
    us_row[rt.U_OPEN], us_row[rt.U_HIGH], us_row[rt.U_LOW] = "1.90", "2.20", "1.85"
    us_row[rt.U_PRICE], us_row[rt.U_CHANGE_RATE] = "2.05", "7.5"
    us_row[rt.U_BID], us_row[rt.U_ASK] = "2.04", "2.06"
    us_row[rt.U_TICK_VOL], us_row[rt.U_ACML_VOL] = "500", "3200000"
    us_row[rt.U_ACML_VAL], us_row[rt.U_STRENGTH] = "6100000", "121.4"

    us_stream = rt.TickStream()
    us_stream._handle(ws, "0|HDFSCNT0|001|" + "^".join(us_row))
    us_tick = us_stream.last("CISS")
    check("해외", "해외 체결을 제 필드로 파싱",
          us_tick and us_tick["price"] == 2.05 and us_tick["bid"] == 2.04
          and us_tick["ask"] == 2.06 and us_tick["market"] == "US",
          f"{us_tick['price']} bid {us_tick['bid']} ask {us_tick['ask']}" if us_tick else "없음")
    check("해외", "국내 필드 배치를 해외에 쓰지 않음 (가격 자리가 다름)",
          rt.U_PRICE != rt.F_PRICE and rt.U_BID != rt.F_BID,
          f"국내 현재가 [{rt.F_PRICE}] vs 해외 [{rt.U_PRICE}]")
    check("해외", "해외 틱도 초봉으로 쌓임", len(us_stream.bars("CISS", 1)) == 1)

    # 지연 시세 판정 — 몇 틱 승부의 전제가 무너지는 지점
    stale = list(us_row)
    stale_sec = (now.hour * 3600 + now.minute * 60 + now.second - 900) % 86_400
    stale[rt.U_KHMS] = f"{stale_sec // 3600:02d}{stale_sec % 3600 // 60:02d}{stale_sec % 60:02d}"
    us_stream._handle(ws, "0|HDFSCNT0|001|" + "^".join(stale))
    delayed = us_stream.last("CISS")
    check("해외", "지연된 시세를 초 단위로 잡아냄",
          delayed["delay_sec"] >= 800, f"{delayed['delay_sec']:.0f}초 지연")
    check("해외", "지연 허용치가 하드 한도로 박혀 있음",
          scalping.HARD_LIMITS["max_quote_delay_sec"] <= 10,
          f"{scalping.HARD_LIMITS['max_quote_delay_sec']}초 — 설정으로 못 늘립니다")

    # --- 지표 시계열 ---
    bars = [{"t": 32400 + i, "o": 100 + i % 3, "h": 102 + i % 3, "l": 99 + i % 3,
             "c": 100 + (i % 5), "v": 100 + i, "n": 3} for i in range(40)]
    chart = scalping.chart_series(bars, {"budget_krw": 10_000})
    series = chart["series"]
    check("지표", "초봉 지표를 시계열로 펼침",
          all(k in series for k in ("vwap", "ema_fast", "ema_slow",
                                    "bb_upper", "bb_lower", "rsi", "trades")),
          ", ".join(sorted(series)))
    check("지표", "워밍업 구간은 None (없는 값을 그리지 않음)",
          sum(1 for v in series["ema_fast"] if v is None) == scalping.CHART_EMA_FAST - 1
          and sum(1 for v in series["bb_upper"] if v is None) == scalping.CHART_BB_PERIOD - 1,
          f"EMA {scalping.CHART_EMA_FAST - 1}개 / BB {scalping.CHART_BB_PERIOD - 1}개")
    check("지표", "목표·손절선을 차트와 같은 계산으로 제공",
          chart["levels"]["target"] == chart["economics"]["target_price"]
          and chart["levels"]["stop"] == chart["economics"]["stop_price"],
          f"목표 {chart['levels']['target']} / 손절 {chart['levels']['stop']}")

    zero_volume = [{**b, "v": 0} for b in bars]
    vwap = scalping.chart_series(zero_volume)["series"]["vwap"]
    check("지표", "거래량이 0인 봉에서도 VWAP 이 얼지 않음",
          len(set(vwap)) > 1, f"{vwap[0]} → {vwap[-1]} (체결 건수로 가중)")
    check("지표", "봉이 없어도 죽지 않음", scalping.chart_series([])["bars"] == 0)

    # --- 내 매매 지점 ---
    fills = store.fills_for(TEST_USER, "paper", "123456")
    check("지점", "체결 내역 조회가 동작", isinstance(fills, list), f"{len(fills)}건")
    check("지점", "차트 x축에 맞는 초 단위 시각을 함께 줌",
          all("sec_of_day" in f and "price" in f for f in fills) if fills else True)

    from api import _nearest_bar_index
    times = [32400, 32405, 32410, 32415]
    check("지점", "체결 시각을 가장 가까운 봉에 붙임",
          _nearest_bar_index(times, 32409) == 2 and _nearest_bar_index(times, 32400) == 0,
          "봉이 인덱스로 그려지므로 마커도 인덱스로 줘야 자리가 맞습니다")
    check("지점", "봉이 없으면 마커를 만들지 않음",
          _nearest_bar_index([], 32400) is None)


# ---------------------------------------------------------------------------
# 13. AI 추천 — 물리학 팩터 · 스마트머니 · 장 상태 알림
# ---------------------------------------------------------------------------

def test_recommender_factors():
    section("13. AI 추천 — 물리학 팩터 · 스마트머니 · 장 상태 알림")

    import numpy as np
    from engine import recommender as R
    from engine.autotrade import session_message

    rng = np.random.default_rng(7)

    # Hill 꼬리지수 — 두꺼운 꼬리(파레토)가 정규분포보다 낮아야 합니다.
    # 이제 (alpha, se) 튜플을 돌려주고 **좌측 꼬리만** 잽니다.
    normal = pd.Series(rng.normal(0, 0.02, 800))
    heavy = pd.Series(rng.pareto(1.6, 800) * 0.01 * rng.choice([-1, 1], 800))
    (a_normal, se_n), (a_heavy, se_h) = R._hill_alpha(normal), R._hill_alpha(heavy)
    check("물리학", "Hill 꼬리지수 — 급락 상습 분포를 낮게 판정",
          a_normal > a_heavy, f"정규 {a_normal:.2f} > 파레토 {a_heavy:.2f}")
    check("물리학", "표준오차를 함께 돌려줌 (점추정만 쓰지 않도록)",
          np.isfinite(se_n) and se_n > 0, f"se={se_n:.3f}")
    check("물리학", "표본 부족 시 중립값 + 무한 표준오차",
          R._hill_alpha(pd.Series([0.01] * 5)) == (3.0, float("inf")))

    # 좌우 꼬리를 구분하는가 — 예전 구현은 abs() 로 합쳐 급등주를 급락위험으로 읽었습니다
    left_heavy = pd.Series(np.where(rng.random(800) < 0.5,
                                    -rng.pareto(1.5, 800) * 0.01,
                                    rng.normal(0, 0.005, 800)))
    right_heavy = pd.Series(-left_heavy.to_numpy())
    check("물리학", "상승 급등을 급락 위험으로 읽지 않음 (좌우 꼬리 구분)",
          R._hill_alpha(left_heavy)[0] < R._hill_alpha(right_heavy)[0],
          f"좌꼬리형 {R._hill_alpha(left_heavy)[0]:.2f} < "
          f"우꼬리형 {R._hill_alpha(right_heavy)[0]:.2f}")

    # 추정오차가 크면 중립(3.0)으로 수축되는가
    check("물리학", "표준오차가 크면 꼬리지수를 중립으로 수축",
          abs(R._shrink_tail(1.5, 2.0) - 3.0) < abs(R._shrink_tail(1.5, 0.1) - 3.0),
          f"se=2.0 -> {R._shrink_tail(1.5, 2.0):.2f} / se=0.1 -> {R._shrink_tail(1.5, 0.1):.2f}")

    # RMT 결합도 — 시장 추종 종목이 독립 종목보다 높아야 합니다.
    # 시장 모드를 정의하려면 종목이 충분해야 하므로 15종목 이상을 씁니다.
    dates = pd.date_range("2024-01-01", periods=250, freq="B")
    market = rng.normal(0, 0.01, 250)
    follows = [pd.Series(market + rng.normal(0, 0.003, 250), index=dates)
               for _ in range(15)]
    independent = [pd.Series(rng.normal(0, 0.01, 250), index=dates) for _ in range(5)]
    couplings = R._rmt_coupling(follows + independent)
    check("물리학", "RMT — 시장 추종은 결합도↑, 독립은 결합도↓",
          np.mean(couplings[15:]) < np.mean(couplings[:15]),
          f"추종 {np.mean(couplings[:15]):.2f} vs 독립 {np.mean(couplings[15:]):.2f}")

    # Marchenko-Pastur 게이트 — 순수 잡음이면 시장 모드가 없다고 답해야 합니다
    pure_noise = [pd.Series(rng.normal(0, 0.01, 250), index=dates) for _ in range(20)]
    check("물리학", "MP 경계 — 상관 없는 종목뿐이면 전부 중립",
          all(abs(c - 1.0) < 1e-9 for c in R._rmt_coupling(pure_noise)),
          "최대 고유값이 잡음 벌크를 못 넘으면 '시장 모드'가 아닙니다")

    # 날짜 정렬 — 거래정지로 봉 수가 다른 종목이 섞여도 같은 날끼리 짝지어야 합니다
    halted = follows[0].drop(follows[0].index[50:80])      # 30일 거래정지
    mixed = [halted] + follows[1:] + independent
    aligned_c = R._rmt_coupling(mixed)
    check("물리학", "거래정지로 봉 수가 달라도 날짜로 정렬",
          np.isfinite(aligned_c[0]) and aligned_c[0] > 1.0,
          f"정지 종목 결합도 {aligned_c[0]:.2f} — 여전히 시장 추종으로 판정")
    check("물리학", "후보 2개 이하면 중립 (노이즈 고유벡터 방지)",
          R._rmt_coupling([follows[0], independent]) == [1.0, 1.0])

    # CCI 기울기 — **돌파 직후**에 양수가 잡혀야 합니다.
    # (돌파 여러 봉 뒤에는 평균편차가 커져 CCI가 내려오는 것이 정상 동작이라,
    #  비교 기준점 -6봉이 아직 횡보 구간에 있도록 급등을 3봉으로 둡니다)
    flat_then_surge = np.concatenate([np.full(57, 100.0) + rng.normal(0, 0.2, 57),
                                      [103.0, 106.0, 109.0]])
    surge_bars = pd.DataFrame({"high": flat_then_surge * 1.01,
                               "low": flat_then_surge * 0.99,
                               "close": flat_then_surge, "volume": [1e6] * 60})
    _, slope = R._cci_state(surge_bars)
    check("스마트머니", "CCI 기울기 — 돌파 직후 양수", slope > 0, f"{slope:+.1f}")

    # 팩터 비중이 합쳐서 1 (정규화 확인)
    for regime in ("추세 국면", "평균회귀 국면", "방향성 불분명", ""):
        total = sum(R._weights_for(regime).values())
        if abs(total - 1.0) > 1e-9:
            check("팩터", f"비중 정규화 ({regime or '기본'})", False, f"합 {total}")
            break
    else:
        check("팩터", "국면별 팩터 비중 합 = 1", True)
    check("팩터", "새 팩터가 비중표에 포함",
          all(k in R.BASE_WEIGHTS for k in ("smart_money", "econophysics", "buzz")))
    check("팩터", "팩터 설명이 화면용으로 제공",
          len(R.describe_factors()) >= 9)

    # 장 상태 알림 문구
    check("장상태", "한국 개장/마감 문구",
          "열렸습니다" in session_message("KR", "REGULAR")
          and "마감되었습니다" in session_message("KR", "CLOSED"))
    check("장상태", "미국 개장 문구", "미국장이 열렸습니다" in session_message("US", "REGULAR"))
    check("장상태", "모르는 세션도 문구는 나옴 (침묵 금지)",
          "???" in session_message("KR", "???"))

    # 스크랩 화제성 — 저장소에 넣은 만큼 집계되는가
    from data_sources import scrap_store
    scrap_store.init()
    scrap_store.save_batch("QATEST99", "test", [
        {"title": f"화제성 테스트 {i}"} for i in range(7)], kind="community")
    mentions = R._scrap_mentions(fake_stock(key="QATEST99"))
    check("화제성", "스크랩 저장소의 최근 언급을 집계", mentions >= 7, f"{mentions}건")

    # -- 커뮤니티 기법 팩터 (IBD RS · 미너비니 템플릿 · 오닐 U/D) ----------
    up = pd.Series(np.linspace(100, 200, 300) + rng.normal(0, 0.5, 300))
    down = pd.Series(np.linspace(200, 100, 300) + rng.normal(0, 0.5, 300))
    up_pass, down_pass = sum(R._trend_template_checks(up)), sum(R._trend_template_checks(down))
    check("커뮤니티기법", "트렌드 템플릿 — 정배열 상승은 통과, 하락 추세는 탈락",
          up_pass >= 7 and down_pass <= 2, f"상승 {up_pass}/8 vs 하락 {down_pass}/8")

    # U/D 거래량비 — 상승일에 거래가 몰리면(매집) 높고, 하락일에 몰리면(분산) 낮다
    closes = [100 + (i % 2) for i in range(61)]          # +1/-1 이 번갈아 나오는 가격
    accum = pd.DataFrame({"close": closes,
                          "volume": [3e6 if i % 2 else 1e6 for i in range(61)]})
    distrib = pd.DataFrame({"close": closes,
                            "volume": [1e6 if i % 2 else 3e6 for i in range(61)]})
    r_accum, r_dist = R._up_down_volume_ratio(accum), R._up_down_volume_ratio(distrib)
    check("커뮤니티기법", "U/D 거래량비 — 매집 > 2, 분산 < 0.5",
          r_accum > 2.0 and r_dist < 0.5, f"매집 {r_accum:.2f} vs 분산 {r_dist:.2f}")
    check("커뮤니티기법", "U/D — 하락일이 없으면 상한값 (0 나누기 없음)",
          R._up_down_volume_ratio(pd.DataFrame(
              {"close": list(range(100, 161)), "volume": [1e6] * 61})) == 5.0)

    # compute_factors 통합 — 미국 종목(수급 조회 없음)으로 네트워크 없이 검증
    us = fake_stock(key="USQATEST", name="US테스트", market="US")
    trend_bars = pd.DataFrame({
        "open": up.values, "high": up.values * 1.01, "low": up.values * 0.99,
        "close": up.values, "volume": np.full(300, 2e6)})
    f = R.compute_factors(us, bars=trend_bars)
    check("커뮤니티기법", "상승 종목 — 상대강도 양수 · 템플릿 상위 · ADR 계산",
          bool(f) and f["rel_strength"] > 0 and f["trend_template"] >= 0.8
          and f.get("adr20", 0) > 0,
          f"RS {f['rel_strength']:+.0f} · 템플릿 {f['trend_checks']}/8 · "
          f"ADR {f['adr20']:.1f}%" if f else "팩터 계산 실패")
    check("커뮤니티기법", "52주 위치 팩터 존재",
          bool(f) and "pct_from_high" in f and "pct_above_low" in f)


# ---------------------------------------------------------------------------
# 14. 실행 중인 서버 (선택)
# ---------------------------------------------------------------------------

def http(path, method="GET", body=None, token=""):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            raw = res.read().decode()
            try:
                return res.status, json.loads(raw or "{}")
            except json.JSONDecodeError:
                return res.status, {"html": raw[:80]}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw or "{}")
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:150]}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, {"error": str(e)}


def test_api():
    section("14. API (서버가 떠 있을 때만)")

    status, _ = http("/api/status")
    if status == 0:
        print("  서버가 없어 건너뜁니다 (아테나.bat → [1] 시작, 또는 python run_server.py 실행 후 재시도)")
        return

    account = {"username": "at_qa_user", "password": "qatest1234"}
    st, data = http("/api/auth/login", "POST", account)
    if st != 200:
        st, data = http("/api/auth/register", "POST", {**account, "display_name": "QA"})
    token = data.get("token", "")
    if not check("API", "테스트 계정 로그인", bool(token), f"status={st}"):
        return

    # 앞선 실행이 계좌를 바꿔놨을 수 있으므로 기준 상태로 되돌리고 시작합니다
    http("/api/autotrade/config", "POST", token=token, body={"config": {"mode": "paper"}})

    st, snap = http("/api/autotrade", token=token)
    check("API", "상태 조회", st == 200 and "config" in snap,
          f"state={snap.get('state')} / mode={snap.get('config', {}).get('mode')}")
    check("API", "브로커 모드 3종", len(snap.get("modes", [])) == 3,
          ", ".join(f"{m['mode']}{'○' if m['available'] else '×'}" for m in snap.get("modes", [])))
    check("API", "기본은 모의 계좌", snap.get("config", {}).get("mode") == "paper")

    st, data = http("/api/autotrade/config", "POST", token=token, body={"config": {
        "universe": ["005930"], "dry_run": True, "regular_session_only": False}})
    check("API", "설정 저장", st == 200 and data.get("config", {}).get("dry_run") is True)

    st, data = http("/api/autotrade/instrument/FUT_FRONT", token=token)
    inst = data.get("instrument", {})
    check("API", "선물 최근월물 자동 계산",
          st == 200 and inst.get("asset_class") == FUTURES,
          f"{inst.get('key')} 만기 {inst.get('expiry')} ({data.get('days_to_expiry')}일)")

    st, data = http("/api/autotrade/instrument/" + urllib.parse.quote("없는종목zzz"),
                    token=token)
    check("API", "없는 종목 404", st == 404)

    st, data = http("/api/autotrade/run", "POST", token=token)
    result = data.get("result", {})
    check("API", "수동 1회전", st == 200 and result.get("ok"),
          f"신호 {len(result.get('signals', []))} · 거부 {len(result.get('rejects', []))}")

    st, data = http("/api/autotrade/kill", "POST", token=token)
    check("API", "킬 스위치", st == 200 and data["config"]["kill_switch"] is True)
    st, data = http("/api/autotrade/run", "POST", token=token)
    check("API", "킬 상태에서 진입 차단", data.get("result", {}).get("halted") is True)
    http("/api/autotrade/resume", "POST", token=token)

    # 실거래 잠금 — 키 유무에 따라 두 관문 중 하나가 반드시 걸려야 합니다
    st, data = http("/api/autotrade/config", "POST", token=token,
                    body={"config": {"mode": "live"}})
    if st == 400:
        check("API", "KIS 키 없이 실거래 전환 거부", True, str(data.get("error", ""))[:60])
    else:
        st2, data2 = http("/api/autotrade/enable", "POST", token=token,
                          body={"enabled": True})          # 확인 문구 없이 시도
        check("API", "확인 문구 없이는 실거래를 켤 수 없음",
              st2 == 400 and data2.get("needs_confirm"),
              str(data2.get("error", ""))[:60])
        http("/api/autotrade/config", "POST", token=token,
             body={"config": {"mode": "paper"}})           # 테스트 계정 원상복구

    st, data = http("/api/autotrade/screener", "POST", token=token,
                    body={"markets": ["KR"], "limit": 5})
    check("API", "페니 스크리너 동작", st == 200 and "candidates" in data,
          f"{data.get('scanned', 0)}종목 훑음 · {', '.join(data.get('sources', []))}")
    check("API", "위험 경고가 응답에 포함", len(data.get("warnings", [])) >= 5)
    check("API", "하드 한도가 응답에 포함", len(data.get("hard_limits", [])) >= 10)

    st, data = http("/api/autotrade/screener/limits", token=token)
    check("API", "한도 조회", st == 200 and "hard_limits" in data)

    st, data = http("/api/autotrade/events?limit=5", token=token)
    check("API", "이벤트 조회", st == 200 and isinstance(data.get("events"), list))

    st, _ = http("/autotrade")
    check("API", "자동매매 콘솔 페이지", st == 200)

    st, _ = http("/api/autotrade")
    check("API", "미인증 접근 차단", st == 401)


# ---------------------------------------------------------------------------

def main():
    offline = "--offline" in sys.argv
    print("=" * 68)
    print("  아테나 시그널 — 자동매매 QA")
    print("=" * 68)

    test_instruments()
    test_strategy()
    test_nnfx()
    test_exits()
    test_risk()
    test_paper_broker()
    test_store()
    test_backtest()
    test_validation()
    test_engine()
    test_order_lifecycle()
    test_reconcile()
    test_account_isolation()
    test_account_switch_reset()
    test_penny_scalping()
    test_strategy_separation()
    test_scalp_chart()
    test_recommender_factors()
    if not offline:
        test_api()

    total = len(results)
    passed = sum(1 for r in results if r[2])
    print("\n" + "=" * 68)
    print(f"  결과: {passed}/{total} 통과")
    failures = [r for r in results if not r[2]]
    if failures:
        print("\n  실패 항목:")
        for cat, name, _, detail in failures:
            print(f"    · [{cat}] {name} — {detail}")
    print("=" * 68)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
