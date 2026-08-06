"""
자동매매 전략 (Signal → Decision)
--------------------------------
"살까 팔까"를 정하는 곳입니다. 실제 주문·리스크 통제는 여기서 하지 않습니다
(engine/risk.py, engine/autotrade.py 담당).

신호 구성
    일봉 기술점수   engine/indicators.analyze — 18개 지표의 국면 가중 평균
    분봉 기술점수   같은 지표를 분봉에 적용 (단기 흐름)
    뉴스 감성       선택 (느려서 기본 off — 켜면 30분 캐시로 사용)

    score = (1-w) × 일봉점수 + w × 분봉점수,  이후 뉴스 점수를 가중 혼합
    score 는 -1(강한 하락) ~ +1(강한 상승)

왜 확률이 아니라 점수인가
    예측 화면은 시그모이드로 확률을 만들지만, 매매는 "임계값을 넘었나"만
    필요합니다. 확률로 바꾸면 임계값 해석이 한 겹 더 꼬입니다.

손절 폭은 ATR 로 잡습니다
    고정 3% 손절은 삼성전자와 코스닥 소형주에 같은 잣대를 대는 것입니다.
    ATR(평균 진폭)의 배수로 잡으면 종목 변동성에 맞춰 자동으로 넓어지고 좁아집니다.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from engine import ensemble, feed, indicators, nnfx
from engine.instruments import FUTURES, OPTION, Instrument

# 지표 계산에 필요한 최소 봉 개수 — 이보다 적으면 신호를 만들지 않습니다
MIN_DAILY_BARS = 30
MIN_INTRADAY_BARS = 40

LONG = "long"
SHORT = "short"
FLAT = "flat"


@dataclass
class Signal:
    key: str
    ok: bool = False
    score: float = 0.0                 # -1 ~ +1
    direction: str = FLAT              # long | short | flat
    confidence: float = 0.0            # 0 ~ 1 (임계값 대비 초과분)
    price: float = 0.0                 # 호가 통화 기준 현재가
    price_krw: float = 0.0
    atr: float = 0.0                   # 절대 진폭 (호가 통화)
    atr_pct: float = 0.0               # 가격 대비 %
    daily_score: float = 0.0
    intraday_score: float | None = None
    news_score: float | None = None
    regime: str = ""
    bars_used: int = 0
    quote_age: float = 0.0
    nnfx: dict | None = None           # NNFX 오버레이 (nnfx_mode 가 off 가 아닐 때만)
    ensemble: dict | None = None       # 앙상블 진단 (algo_mode 가 off 가 아닐 때만)
    vol_factor: float = 1.0            # 변동성 배수 — 손절·익절 폭 스케일에 사용
    reasons: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key, "ok": self.ok, "score": round(self.score, 4),
            "direction": self.direction, "confidence": round(self.confidence, 3),
            "price": self.price, "price_krw": self.price_krw,
            "atr": round(self.atr, 4), "atr_pct": round(self.atr_pct, 3),
            "daily_score": round(self.daily_score, 4),
            "intraday_score": (round(self.intraday_score, 4)
                               if self.intraday_score is not None else None),
            "news_score": (round(self.news_score, 4)
                           if self.news_score is not None else None),
            "regime": self.regime, "bars_used": self.bars_used,
            "quote_age": self.quote_age, "nnfx": self.nnfx,
            "ensemble": self.ensemble,
            "vol_factor": round(self.vol_factor, 3),
            "reasons": self.reasons[:6], "error": self.error,
        }


# ---------------------------------------------------------------------------
# 뉴스 점수 캐시 (선택 기능)
# ---------------------------------------------------------------------------

_news_cache: dict[str, tuple[float, float]] = {}
NEWS_TTL = 1800.0          # 30분 — 뉴스는 초 단위로 변하지 않습니다


def _news_score(inst: Instrument) -> float | None:
    """뉴스 감성 점수. 크롤링이 느려서 30분 캐시를 씁니다."""
    import time

    if inst.is_derivative or inst.symbol is None:
        return None            # 파생상품은 개별 뉴스가 없습니다

    hit = _news_cache.get(inst.key)
    if hit and time.time() - hit[0] < NEWS_TTL:
        return hit[1]
    try:
        from data_sources import news_crawler
        items = news_crawler.get_news(inst.symbol, limit=15)
        score = news_crawler.aggregate_news_score(items)
    except Exception:
        return None
    _news_cache[inst.key] = (time.time(), float(score))
    return float(score)


# ---------------------------------------------------------------------------
# 신호 생성
# ---------------------------------------------------------------------------

def evaluate(inst: Instrument, cfg: dict, bars_daily: pd.DataFrame = None,
             bars_intraday: pd.DataFrame = None, quote: dict = None,
             allow_fetch: bool = True) -> Signal:
    """한 종목에 대한 매매 신호.

    bars/quote 를 주입할 수 있게 한 이유는 **백테스트와 실매매가 같은 함수를
    쓰게 하기 위해서**입니다. 시뮬레이터는 과거 봉을 잘라서 넣고, 실매매는
    None 을 넘겨 실시간 데이터를 받습니다.

    allow_fetch=False 면 주어진 데이터만 씁니다. 백테스트가 실수로 '오늘의
    실시간 시세'를 끌어와 미래를 훔쳐보는 것을 구조적으로 막습니다.
    """
    sig = Signal(key=inst.key)

    if quote is None and allow_fetch:
        quote = feed.quote(inst)
    if not quote or not quote.get("price"):
        sig.error = "현재가를 조회할 수 없습니다."
        return sig
    sig.price = float(quote["price"])
    sig.price_krw = float(quote.get("price_krw") or quote["price"])
    sig.quote_age = float(quote.get("age_sec") or 0)

    if bars_daily is None and allow_fetch:
        bars_daily = feed.bars(inst, "day", count=180)
    if bars_daily is None or len(bars_daily) < MIN_DAILY_BARS:
        sig.error = f"일봉이 부족합니다 ({0 if bars_daily is None else len(bars_daily)}개)"
        return sig

    daily = indicators.analyze(bars_daily)
    sig.daily_score = float(daily.score)
    sig.bars_used = daily.bars_used
    sig.regime = (daily.regime or {}).get("label", "")

    intraday_weight = float(cfg.get("intraday_weight", 0.35))
    if intraday_weight > 0:
        if bars_intraday is None and allow_fetch:
            bars_intraday = feed.bars(inst, "minute", count=240)
        if bars_intraday is not None and len(bars_intraday) >= MIN_INTRADAY_BARS:
            sig.intraday_score = float(indicators.analyze(bars_intraday).score)

    # 앙상블 (engine/ensemble.py) — 시평선 합치·난기류·변동성 배수.
    # observe 모드는 계산·기록만 하고 점수를 바꾸지 않습니다.
    ens = ensemble.compute(bars_daily, sig.daily_score, sig.intraday_score, cfg)
    ens_mode = ensemble.mode_of(cfg)
    if ens.ok:
        sig.ensemble = ens.to_dict()
        sig.vol_factor = ens.vol_factor

    if ens.ok and ens_mode in (ensemble.SOFT, ensemble.GATE):
        blended = ens.score          # 시평선 불일치 감산·난기류 감쇠가 반영된 점수
    elif sig.intraday_score is None:
        blended = sig.daily_score
    else:
        blended = ((1 - intraday_weight) * sig.daily_score
                   + intraday_weight * sig.intraday_score)

    if cfg.get("use_news"):
        news = _news_score(inst)
        if news is not None:
            sig.news_score = news
            weight = float(cfg.get("news_weight", 0.25))
            blended = (1 - weight) * blended + weight * news

    # 인버스 ETF 는 기초자산이 내릴 때 오릅니다. 지표는 그 ETF 자체의 가격으로
    # 계산되므로 뒤집을 필요가 없지만, 뉴스 점수는 기초자산 기준이라 뒤집습니다.
    if inst.is_inverse and sig.news_score is not None:
        blended -= 2 * float(cfg.get("news_weight", 0.25)) * sig.news_score

    # NNFX 규칙 오버레이 — 켜져 있을 때만. 점수에 **더하지 않고 섞습니다**
    # (슬롯이 보는 추세·모멘텀은 위 지표 점수에 이미 들어 있어, 더하면 이중 계상).
    nnfx_mode = nnfx.mode_of(cfg)
    nnfx_state, nnfx_notes = None, []
    if nnfx_mode != nnfx.OFF:
        nnfx_state = nnfx.compute(bars_daily, cfg)
        sig.nnfx = nnfx_state.to_dict()
        if nnfx_mode == nnfx.SOFT:
            blended, note = nnfx.apply_to_score(blended, nnfx_state, cfg)
            if note:
                nnfx_notes.append(note)

    sig.score = max(-1.0, min(1.0, blended))
    sig.atr, sig.atr_pct = _atr(bars_daily, sig.price)

    entry = float(cfg.get("entry_score", 0.35))
    allow_short = bool(cfg.get("allow_short")) and inst.shortable

    if sig.score >= entry:
        sig.direction = LONG
    elif sig.score <= -entry and allow_short:
        sig.direction = SHORT
    else:
        sig.direction = FLAT

    # 게이트 모드는 방향이 정해진 뒤에 막습니다. 신규 진입만 막고 청산은
    # 건드리지 않습니다 — 못 들어가는 것은 기회 손실이지만, 못 나오는 것은 손실입니다.
    if nnfx_state is not None and nnfx_state.ok and sig.direction == LONG:
        blocked = (nnfx_mode == nnfx.VETO and not nnfx_state.veto_ok) or \
                  (nnfx_mode == nnfx.HARD and not nnfx_state.hard_ok)
        if blocked:
            sig.direction = FLAT
            nnfx_notes.insert(0, "NNFX 게이트 차단 — "
                              + (nnfx_state.reasons[0] if nnfx_state.reasons
                                 else "슬롯 조건 미충족"))

    # 앙상블 게이트 — 난기류 극단 등. NNFX 와 같은 원칙(신규 진입만)입니다.
    if ens.ok and ens.block and sig.direction != FLAT:
        sig.direction = FLAT
        nnfx_notes.insert(0, ens.blocked_by[0] if ens.blocked_by else "앙상블 게이트 차단")

    sig.confidence = min(1.0, abs(sig.score) / entry) if entry > 0 else 0.0
    # _reasons 가 목록을 새로 만들므로, NNFX·앙상블 사유는 **그 뒤에** 붙여야 남습니다.
    # (앞에 붙이면 조용히 덮어써져 "왜 막혔는지 모르는" 상태가 됩니다)
    ens_notes = ens.notes[:2] if ens.ok else []
    sig.reasons = nnfx_notes + ens_notes + _reasons(daily, sig)
    sig.ok = True
    return sig


def _atr(df: pd.DataFrame, price: float, period: int = 14) -> tuple[float, float]:
    """평균 진폭. 손절 폭 산정에 씁니다."""
    try:
        high, low, close = df["high"], df["low"], df["close"]
        prev = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
                       axis=1).max(axis=1)
        value = float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0, 0.0
    if not value or value != value:          # NaN 방어
        return 0.0, 0.0
    return value, (value / price * 100) if price else 0.0


def _reasons(analysis, sig: Signal) -> list[str]:
    """왜 이 방향인지 — 감사 로그에 남길 사람이 읽는 근거."""
    out = []
    ranked = sorted(
        [i for i in analysis.indicators if i.weight > 0 and abs(i.score) >= 0.15],
        key=lambda i: abs(i.score * i.weight), reverse=True)
    for ind in ranked[:3]:
        out.append(f"{ind.label} {ind.value_text} ({ind.score:+.2f})")
    if sig.intraday_score is not None:
        out.append(f"분봉 {sig.intraday_score:+.2f}")
    if sig.news_score is not None:
        out.append(f"뉴스 {sig.news_score:+.2f}")
    if sig.regime:
        out.append(f"국면 {sig.regime}")
    return out


# ---------------------------------------------------------------------------
# 진입 계획 (포지션 사이징)
# ---------------------------------------------------------------------------

@dataclass
class EntryPlan:
    ok: bool = False
    quantity: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    notional_krw: float = 0.0
    margin_krw: float = 0.0
    risk_krw: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "quantity": self.quantity, "price": self.price,
                "stop_price": round(self.stop_price, 4),
                "target_price": round(self.target_price, 4),
                "notional_krw": round(self.notional_krw), "margin_krw": round(self.margin_krw),
                "risk_krw": round(self.risk_krw), "reason": self.reason}


def stop_distance(inst: Instrument, sig: Signal, cfg: dict) -> float:
    """손절까지의 가격 폭 (호가 통화 기준).

    ATR 배수를 우선 쓰고, ATR 을 못 구하면 고정 퍼센트로 떨어집니다.
    """
    atr_mult = float(cfg.get("atr_stop_mult", 2.0))
    if atr_mult > 0 and sig.atr > 0:
        return sig.atr * atr_mult
    return sig.price * float(cfg.get("stop_loss_pct", 3.0)) / 100.0


def plan_entry(inst: Instrument, sig: Signal, cfg: dict, account: dict) -> EntryPlan:
    """얼마나 살 것인가.

    1) 리스크 기반 수량: 손절까지 갔을 때 잃는 금액이 총자산의 risk_per_trade_pct
       가 되도록 수량을 정합니다. (변동성이 큰 종목은 자동으로 적게 삽니다)
    2) 비중 상한: 한 종목이 총자산의 position_pct 를 넘지 않게 자릅니다.
    3) 주문 금액 상한 / 가용 현금(파생은 증거금)으로 한 번 더 자릅니다.
    """
    plan = EntryPlan(price=sig.price)

    # 변동성 배수 (engine/ensemble.py) — soft/gate 모드에서만 고정 % 계열의
    # 손절·익절·트레일링 폭을 스케일합니다. ATR 손절은 이미 변동성이라 제외.
    # 저장된 설정이 아니라 이번 판단의 사본에만 적용됩니다.
    ens_mode = ensemble.mode_of(cfg)
    if ens_mode in (ensemble.SOFT, ensemble.GATE):
        scaled = ensemble.scale_barriers(cfg, getattr(sig, "vol_factor", 1.0))
        if scaled:
            cfg = {**cfg, **scaled}

    total_value = float(account.get("total_value") or 0)
    available = float(account.get("available_cash") or 0)
    if total_value <= 0:
        plan.reason = "계좌 평가금액이 0입니다."
        return plan

    distance = stop_distance(inst, sig, cfg)
    if distance <= 0:
        plan.reason = "손절 폭을 계산할 수 없습니다."
        return plan

    direction = 1 if sig.direction == LONG else -1
    plan.stop_price = sig.price - direction * distance
    take_pct = float(cfg.get("take_profit_pct", 0) or 0)
    if take_pct > 0:
        plan.target_price = sig.price * (1 + direction * take_pct / 100.0)
    else:
        rr = float(cfg.get("reward_risk", 2.0))
        plan.target_price = sig.price + direction * distance * rr

    # 비용 게이트 (gate 모드만) — 목표 수익이 왕복 수수료·거래세·슬리피지를
    # 충분히 넘지 못하면 이길 수 없는 매매입니다. 여기서 걸러야 수량 계산과
    # 리스크 게이트를 헛돌지 않습니다.
    if ens_mode == ensemble.GATE:
        edge = ensemble.cost_edge(inst, sig.price, plan.target_price, cfg)
        if not edge.get("ok", True):
            plan.reason = "기대이익이 거래비용에 못 미칩니다 — " + str(edge.get("reason", ""))
            return plan

    # 1) 리스크 기반
    risk_budget = total_value * float(cfg.get("risk_per_trade_pct", 1.0)) / 100.0
    krw_per_point = inst.multiplier * (sig.price_krw / sig.price if sig.price else 1)
    loss_per_unit = distance * krw_per_point
    if loss_per_unit <= 0:
        plan.reason = "단위당 손실을 계산할 수 없습니다."
        return plan
    qty = risk_budget / loss_per_unit

    # 2) 비중 상한 (파생은 증거금 기준, 현물은 매수금액 기준)
    cap_krw = total_value * float(cfg.get("position_pct", 20.0)) / 100.0
    max_order = float(cfg.get("max_order_krw", 0) or 0)
    if max_order > 0:
        cap_krw = min(cap_krw, max_order)

    unit_cost = _unit_cost_krw(inst, sig)
    if unit_cost <= 0:
        plan.reason = "단위 비용을 계산할 수 없습니다."
        return plan
    qty = min(qty, cap_krw / unit_cost)

    # 3) 가용 현금 (수수료 여유 1% 남김)
    qty = min(qty, available * 0.99 / unit_cost)

    qty = inst.round_quantity(qty)
    if qty <= 0:
        affordable = unit_cost <= min(cap_krw, available * 0.99)
        if cfg.get("min_one_unit") and not inst.is_derivative and affordable:
            # 소액 계좌 — 리스크 예산으로는 1주 미만이지만 현금·비중 한도 안이면
            # 최소 1주는 허용합니다. 이때 실제 위험이 예산을 넘는다는 사실은
            # plan.risk_krw 에 그대로 남습니다 (숨기지 않습니다).
            qty = 1.0
        elif loss_per_unit > risk_budget and affordable:
            plan.reason = (f"리스크 예산 {risk_budget:,.0f}원 < 1주 손절 위험 "
                           f"{loss_per_unit:,.0f}원 — 1회 위험 예산(%)을 올리거나 "
                           f"'최소 1주 허용'을 켜세요")
            return plan
        else:
            plan.reason = (f"주문 가능 수량이 0입니다 "
                           f"(가용 {available:,.0f}원 / 1단위 {unit_cost:,.0f}원)")
            return plan

    side = "buy" if sig.direction == LONG else "sell"
    plan.quantity = qty
    plan.notional_krw = inst.notional(sig.price_krw, qty)
    plan.margin_krw = inst.margin_required(sig.price_krw, qty, side)
    plan.risk_krw = loss_per_unit * qty
    plan.ok = True

    min_order = float(cfg.get("min_order_krw", 0) or 0)
    basis = plan.margin_krw if inst.is_derivative else plan.notional_krw
    if min_order > 0 and basis < min_order:
        plan.ok = False
        plan.reason = f"주문 금액이 최소 {min_order:,.0f}원에 못 미칩니다 ({basis:,.0f}원)"
    return plan


def _unit_cost_krw(inst: Instrument, sig: Signal) -> float:
    """1주(1계약)를 잡는 데 실제로 필요한 원화."""
    side = "buy" if sig.direction != SHORT else "sell"
    return inst.margin_required(sig.price_krw, 1, side)


# ---------------------------------------------------------------------------
# 청산 판단
# ---------------------------------------------------------------------------

@dataclass
class ExitDecision:
    should_exit: bool = False
    reason: str = ""
    urgency: str = "normal"        # normal | urgent (urgent 는 장 상태와 무관하게 시도)


def check_exit(inst: Instrument, position, sig: Signal, cfg: dict,
               state: dict, now: datetime = None) -> ExitDecision:
    """보유 포지션을 정리해야 하는가.

    확인 순서가 중요합니다 — **손실을 막는 조건을 먼저** 봅니다.
    (익절보다 손절이 먼저, 신호 반전보다 만기가 먼저)

    state 는 이 포지션에 대해 엔진이 기억하고 있는 값입니다.
        entry_price, stop_price, target_price, peak_price, opened_at
    """
    now = now or datetime.now()
    direction = 1 if position.side == LONG else -1
    price = sig.price if sig.ok and sig.price else (position.current_price or 0)
    if not price:
        return ExitDecision(False, "현재가 없음")

    entry = float(state.get("entry_price") or position.avg_price or price)
    pnl_pct = (price - entry) / entry * 100 * direction if entry else 0.0

    # 1) 만기 (파생) — 만기일에 물려 있으면 강제 청산됩니다
    if inst.asset_class in (FUTURES, OPTION):
        days = feed.days_to_expiry(inst)
        min_days = int(cfg.get("deriv_min_days_to_expiry", 2))
        if days is not None and days <= min_days:
            return ExitDecision(True, f"만기 {days}일 전 — 롤오버/청산", "urgent")

    # 2) 손절
    stop = state.get("stop_price")
    if stop:
        hit = price <= float(stop) if direction > 0 else price >= float(stop)
        if hit:
            return ExitDecision(True, f"손절 도달 ({price:g} vs {float(stop):g}, {pnl_pct:+.2f}%)",
                                "urgent")

    hard_stop = float(cfg.get("stop_loss_pct", 0) or 0)
    if hard_stop > 0 and pnl_pct <= -hard_stop:
        return ExitDecision(True, f"손실 한도 {hard_stop:g}% 초과 ({pnl_pct:+.2f}%)", "urgent")

    # 3) 트레일링 스톱 — 고점 대비 되돌림
    trail = float(cfg.get("trailing_stop_pct", 0) or 0)
    peak = state.get("peak_price")
    if trail > 0 and peak:
        drop = (float(peak) - price) / float(peak) * 100 * direction
        if drop >= trail:
            return ExitDecision(True, f"고점 대비 {drop:.2f}% 되돌림 (트레일링 {trail:g}%)")

    # 4) 익절
    target = state.get("target_price")
    if target:
        hit = price >= float(target) if direction > 0 else price <= float(target)
        if hit:
            return ExitDecision(True, f"목표가 도달 ({price:g}, {pnl_pct:+.2f}%)")

    take = float(cfg.get("take_profit_pct", 0) or 0)
    if take > 0 and pnl_pct >= take:
        return ExitDecision(True, f"목표 수익 {take:g}% 달성 ({pnl_pct:+.2f}%)")

    # 5) 신호 반전 / 소멸
    if sig.ok:
        exit_score = float(cfg.get("exit_score", 0.05))
        if direction > 0 and sig.score <= -abs(exit_score):
            return ExitDecision(True, f"신호 반전 (점수 {sig.score:+.2f})")
        if direction < 0 and sig.score >= abs(exit_score):
            return ExitDecision(True, f"신호 반전 (점수 {sig.score:+.2f})")

    # 6) 보유 기간 초과 — 신호가 죽었는데 계속 들고 있는 것을 막습니다
    opened = state.get("opened_at")

    # 초단타는 **초 단위**로 끊습니다. max_hold_minutes 는 분 정수라 90초 같은
    # 값을 표현할 수 없어서, 페니 초단타는 이쪽을 씁니다. 둘 다 설정돼 있으면
    # 짧은 쪽이 이깁니다 — 청산 조건은 항상 보수적인 쪽으로 붙어야 합니다.
    max_sec = int(cfg.get("max_hold_sec", 0) or 0)
    if max_sec > 0 and opened:
        try:
            held_sec = (now - datetime.fromisoformat(str(opened))).total_seconds()
            if held_sec >= max_sec:
                return ExitDecision(True, f"보유 {held_sec:.0f}초 (최대 {max_sec}초) — 시간 청산",
                                    "urgent")
        except ValueError:
            pass

    max_minutes = int(cfg.get("max_hold_minutes", 0) or 0)
    if max_minutes > 0 and opened:
        try:
            held_min = (now - datetime.fromisoformat(str(opened))).total_seconds() / 60
            if held_min >= max_minutes:
                return ExitDecision(True, f"보유 {held_min:.0f}분 (최대 {max_minutes}분) — 시간 청산",
                                    "urgent")
        except ValueError:
            pass

    max_days = int(cfg.get("max_hold_days", 0) or 0)
    if max_days > 0 and opened:
        try:
            held = (now - datetime.fromisoformat(str(opened))).days
            if held >= max_days:
                return ExitDecision(True, f"보유 {held}일 (최대 {max_days}일) — 시간 청산")
        except ValueError:
            pass

    return ExitDecision(False, "")


def update_peak(state: dict, price: float, side: str) -> dict:
    """트레일링 스톱용 최고점(숏이면 최저점) 갱신."""
    peak = state.get("peak_price")
    if peak is None:
        state["peak_price"] = price
    elif side == LONG:
        state["peak_price"] = max(float(peak), price)
    else:
        state["peak_price"] = min(float(peak), price)
    return state
