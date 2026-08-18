# -*- coding: utf-8 -*-
"""분할 매수 · 분할 매도 (차수별 독립 관리)
==========================================
한 종목을 한 번에 사고 한 번에 파는 대신, **차수(tranche)** 로 나눠 사고
**차수별로** 팝니다. 안정모 채널에서 소개된 매직스플릿 계열(세븐스플릿 포함)
방식을 이 엔진의 구조에 맞춰 옮긴 것입니다.

핵심은 "차수별 독립"입니다
    3차를 자기 매수가보다 5% 위에서 팔 때, 1·2차는 그대로 둡니다. 전체 평단을
    기준으로 판단하면 이게 불가능합니다 — 평단은 3차를 산 순간 이미 내려가
있고, 평단 대비 익절은 **모든 차수를 한꺼번에** 정리하기 때문입니다.
    차수마다 자기 매수가를 기억하기 때문에, 같은 종목이 오르내리는 동안
    3차 매수 → 3차 매도 → 다시 하락 → 3차 재매수가 반복됩니다. 박스권에서
    수익이 나는 이유가 이것입니다.

이 모듈은 **판단만** 합니다
    주문 제출·체결 기록·원장 저장은 engine/autotrade.py 가 합니다.
    engine/rotation.py 와 같은 규약입니다 — 판단 함수는 부작용이 없어야
    테스트할 수 있고, 실패했을 때 "무엇을 보고 그렇게 판단했는지"(detail)를
    그대로 남길 수 있습니다.

손절과의 관계 — 여기가 제일 중요합니다
    기본 손절은 -4%(stop_loss_pct)인데 차수 간격이 -5%면 **2차를 살 기회가
    영원히 오지 않습니다.** 손절이 먼저 걸리기 때문입니다. 그래서 분할이
    켜진 포지션은 1차 진입 때 손절선을 다시 계산합니다:

        손절가 = 마지막 차수 예정가 × (1 - split_final_stop_pct/100)

    "마지막 차수까지 다 사고도 더 떨어지면 그때는 포기한다"는 선입니다.
    이 선은 차수를 추가해도 움직이지 않습니다 — 물타기가 손절선을 계속
    끌어내리면 그건 손절이 아니라 그냥 버티기입니다.

    대신 **한 종목이 감당하는 손실 폭이 훨씬 커집니다.** 5차수 · 간격 5% ·
    마지막 여유 7%면 1차 진입가 대비 약 -24%까지 들고 갑니다. 종목당 비중
    (position_pct)을 그만큼 낮게 잡아야 계좌 전체가 견딥니다.

지지 않는 쪽으로 기울인 기본값
    · 기본 꺼짐. 사람이 켜야 합니다.
    · 추가 매수에 과매도 조건을 **함께** 요구합니다(하락률 AND 지표).
      떨어진다는 사실만으로 사면 추세 하락에서 차수가 순식간에 소진됩니다.
    · 손실 난 차수는 절대 팔지 않습니다. 손실 정리는 손절의 일입니다.
    · 차수 간 최소 시간 간격을 둡니다. 하루 급락에 5차수가 다 들어가면
      분할이 아니라 그냥 한 번에 산 것입니다.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime

# 기본값 — 전부 "분할을 신중하게" 하는 방향입니다.
DEFAULTS = {
    "split_enabled": False,          # 사람이 켜야 합니다
    "split_tranches": 5,             # 최대 차수 (1차 포함)
    "split_buy_drop_pct": 5.0,       # 직전 차수 체결가 대비 이만큼 내리면 다음 차수
    "split_buy_drop_steps": [],      # 차수별 하락률 직접 지정 (비우면 위 값 균등)
    "split_sell_gain_pct": 5.0,      # 그 차수 매수가 대비 이만큼 오르면 그 차수만 매도
    "split_sell_gain_steps": [],     # 차수별 익절률 직접 지정
    "split_size_mode": "equal",      # equal(균등) | pyramid(뒤 차수를 크게)
    "split_pyramid_ratio": 1.5,      # pyramid 일 때 차수마다 곱하는 배수
    # 추가 매수에 과매도 조건도 요구할지 (하락률 AND 지표)
    "split_require_oversold": True,
    "split_oversold_rsi": 35.0,      # RSI(14) 이하면 과매도
    "split_oversold_bb": 0.15,       # 볼린저 %B 이하면 과매도
    # 과매수면 목표 익절률에 못 미쳐도 수익 난 차수를 정리할지
    "split_overbought_sell": True,
    "split_overbought_rsi": 70.0,
    "split_overbought_bb": 0.90,
    "split_overbought_min_gain": 1.0,   # 그래도 이만큼은 수익이어야 팝니다
    # 익절률이 왕복비용의 몇 배는 되어야 하는가 (수수료만 내는 매매 방지)
    "split_cost_multiple": 2.0,
    "split_min_gap_min": 30,         # 차수 간 최소 시간 간격(분)
    "split_max_adds_per_day": 3,     # 하루 추가 매수 상한
    # 마지막 차수 예정가에서 이만큼 더 빠지면 전량 손절
    "split_final_stop_pct": 7.0,
}

EQUAL, PYRAMID = "equal", "pyramid"


def params(cfg: dict) -> dict:
    """설정 병합 + 한도 강제.

    0이나 음수로 풀어서 "무한 물타기 / 무한 분할"이 되는 것을 막습니다.
    설정 화면이 아니라 여기서 막는 이유: 설정은 API·파일·예전 버전 등 여러
    경로로 들어오는데, 판단 직전에 한 번 조이면 모든 경로가 덮입니다.
    """
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        if cfg.get(key) is not None:
            out[key] = cfg[key]

    out["split_enabled"] = bool(out["split_enabled"])
    # 차수 상한 10 — 그 이상은 차수당 금액이 너무 작아져 수수료·최소주문에 걸립니다
    out["split_tranches"] = max(2, min(int(out["split_tranches"]), 10))
    out["split_buy_drop_pct"] = max(0.5, float(out["split_buy_drop_pct"]))
    out["split_sell_gain_pct"] = max(0.5, float(out["split_sell_gain_pct"]))
    out["split_size_mode"] = (PYRAMID if str(out["split_size_mode"]) == PYRAMID
                              else EQUAL)
    out["split_pyramid_ratio"] = max(1.0, min(float(out["split_pyramid_ratio"]), 3.0))
    out["split_oversold_rsi"] = max(5.0, min(float(out["split_oversold_rsi"]), 50.0))
    out["split_oversold_bb"] = max(0.0, min(float(out["split_oversold_bb"]), 0.5))
    out["split_overbought_rsi"] = max(50.0, min(float(out["split_overbought_rsi"]), 95.0))
    out["split_overbought_bb"] = max(0.5, min(float(out["split_overbought_bb"]), 1.5))
    out["split_overbought_min_gain"] = max(0.0, float(out["split_overbought_min_gain"]))
    out["split_cost_multiple"] = max(1.0, float(out["split_cost_multiple"]))
    out["split_min_gap_min"] = max(0, int(out["split_min_gap_min"]))
    out["split_max_adds_per_day"] = max(0, int(out["split_max_adds_per_day"]))
    out["split_final_stop_pct"] = max(1.0, float(out["split_final_stop_pct"]))
    return out


def enabled(cfg: dict) -> bool:
    return bool(params(cfg)["split_enabled"])


# ---------------------------------------------------------------------------
# 차수 계획 — 1차 진입 때 한 번 정하고 원장에 박아둡니다
# ---------------------------------------------------------------------------

def _steps(raw, count: int, fallback: float) -> list[float]:
    """차수별 값 목록을 `count` 개로 맞춥니다.

    설정이 짧으면 마지막 값으로, 아예 없으면 fallback 으로 채웁니다.
    (예: [10, 20] · 5차수 → [10, 20, 20, 20, 20])
    """
    values = []
    for item in (raw or []):
        try:
            v = float(item)
        except (TypeError, ValueError):
            continue
        if v > 0:
            values.append(v)
    if not values:
        values = [float(fallback)]
    while len(values) < count:
        values.append(values[-1])
    return values[:count]


def size_weights(p: dict) -> list[float]:
    """차수별 수량 비중 (합이 1). equal 이면 균등, pyramid 면 뒤로 갈수록 큽니다.

    pyramid 를 두는 이유: 더 싸게 사는 뒤 차수에 더 많이 담으면 평단이 빨리
    내려옵니다. 대신 하락이 계속될 때 손실도 그만큼 빨리 커집니다 — 그래서
    기본은 균등입니다.
    """
    n = p["split_tranches"]
    if p["split_size_mode"] == PYRAMID:
        ratio = p["split_pyramid_ratio"]
        raw = [ratio ** i for i in range(n)]
    else:
        raw = [1.0] * n
    total = sum(raw) or 1.0
    return [v / total for v in raw]


def plan_for(p: dict, base_price: float, total_quantity: float,
             round_quantity=None) -> dict:
    """1차 체결 직후 만드는 **차수 계획**.

    `total_quantity` 는 분할하지 않았다면 한 번에 샀을 전체 수량입니다
    (strategy.plan_entry 가 계산한 값). 그걸 차수별로 쪼갭니다.

    수량이 0인 차수가 생기면(소액 계좌 + 고가주) 그 차수는 계획에서 빼지 않고
    1주로 올립니다 — 계획과 실제가 어긋나면 원장을 읽는 쪽이 전부 틀립니다.
    대신 그만큼 전체 금액이 계획을 넘으므로, 호출부가 자금을 다시 확인합니다.
    """
    n = p["split_tranches"]
    weights = size_weights(p)
    rounder = round_quantity or (lambda q: float(int(q)))

    quantities = []
    for w in weights:
        q = rounder(total_quantity * w)
        quantities.append(max(float(q), 1.0))

    drops = _steps(p["split_buy_drop_steps"], n - 1, p["split_buy_drop_pct"])
    gains = _steps(p["split_sell_gain_steps"], n, p["split_sell_gain_pct"])

    # 마지막 차수 예정가 — 손절선의 기준입니다. 직전 차수 대비 하락률이므로
    # 곱해서 누적합니다 (5%씩 4번은 20%가 아니라 18.5%입니다).
    last_price = float(base_price)
    for drop in drops:
        last_price *= (1 - drop / 100.0)

    return {
        "qty": quantities,
        "drop": drops,
        "gain": gains,
        "base_price": float(base_price),
        "last_price": last_price,
        "stop": last_price * (1 - p["split_final_stop_pct"] / 100.0),
    }


def stop_price_for(p: dict, base_price: float) -> float:
    """1차 진입가 기준 전량 손절가 (계획 없이 값만 필요할 때)."""
    return plan_for(p, base_price, 1.0)["stop"]


def total_drawdown_pct(p: dict, base_price: float = 100.0) -> float:
    """1차 진입가 대비 손절까지의 폭(%). 화면에 "얼마까지 버티는가"를 보여줍니다."""
    return (1 - stop_price_for(p, base_price) / base_price) * 100.0


# ---------------------------------------------------------------------------
# 원장 (at_position_state.splits 에 JSON 으로 저장)
# ---------------------------------------------------------------------------

@dataclass
class Tranche:
    n: int                 # 차수 번호 (1부터)
    quantity: float
    price: float           # 이 차수의 실제 체결가
    at: str = ""           # 매수 시각 (ISO)

    def to_dict(self) -> dict:
        return {"n": self.n, "qty": self.quantity, "price": self.price, "at": self.at}

    def gain_pct(self, price: float) -> float:
        return (price - self.price) / self.price * 100.0 if self.price else 0.0


@dataclass
class Ledger:
    """이 종목의 차수 원장. `plan` 은 1차 때 정해지고 바뀌지 않습니다."""
    plan: dict = field(default_factory=dict)
    tranches: list = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.tranches)

    @property
    def quantity(self) -> float:
        return sum(t.quantity for t in self.tranches)

    @property
    def avg_price(self) -> float:
        qty = self.quantity
        if qty <= 0:
            return 0.0
        return sum(t.price * t.quantity for t in self.tranches) / qty

    @property
    def last(self) -> Tranche | None:
        return max(self.tranches, key=lambda t: t.n) if self.tranches else None

    def next_n(self) -> int:
        """다음 차수 번호. 중간 차수가 팔려 비어 있으면 **그 자리를 다시 씁니다** —
        3차를 팔고 다시 떨어지면 3차를 재매수하는 것이 이 방식의 핵심입니다."""
        used = {t.n for t in self.tranches}
        n = 1
        while n in used:
            n += 1
        return n

    def adds_on(self, day: str) -> int:
        """그 날짜(YYYY-MM-DD)에 담은 차수 수 — 하루 추가 매수 상한의 기준."""
        return len([t for t in self.tranches if str(t.at)[:10] == day])

    def to_json(self) -> str:
        return json.dumps({"plan": self.plan,
                           "tranches": [t.to_dict() for t in self.tranches]},
                          ensure_ascii=False)


def load(raw) -> Ledger:
    """저장된 원장 → Ledger. 깨져 있으면 빈 원장(= 분할 관리 대상 아님).

    깨진 원장으로 '차수가 하나도 없는 분할 포지션'을 만들면, 다음 회전이
    그 포지션을 1차부터 다시 담기 시작합니다. 그래서 실패는 조용히 빈
    원장으로 두고, 호출부가 '분할 대상 아님'으로 처리하게 합니다.
    """
    if isinstance(raw, Ledger):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except (ValueError, AttributeError):
            return Ledger()
    if not isinstance(raw, dict):
        return Ledger()

    tranches = []
    for row in (raw.get("tranches") or []):
        try:
            qty, price = float(row.get("qty") or 0), float(row.get("price") or 0)
            if qty <= 0 or price <= 0:
                continue
            tranches.append(Tranche(n=int(row.get("n") or 0), quantity=qty,
                                    price=price, at=str(row.get("at") or "")))
        except (TypeError, ValueError):
            continue
    return Ledger(plan=dict(raw.get("plan") or {}), tranches=tranches)


def bootstrap(p: dict, price: float, quantity: float, total_quantity: float,
              round_quantity=None, now: datetime = None) -> Ledger:
    """1차 체결 → 새 원장."""
    now = now or datetime.now()
    plan = plan_for(p, price, total_quantity, round_quantity)
    return Ledger(plan=plan,
                  tranches=[Tranche(n=1, quantity=float(quantity),
                                    price=float(price), at=now.isoformat())])


# ---------------------------------------------------------------------------
# 과매수 / 과매도 판정
# ---------------------------------------------------------------------------

def oversold(p: dict, rsi=None, bb_pct=None) -> tuple[bool, str]:
    """지금이 과매도 구간인가. (판정, 설명)

    RSI 와 볼린저 %B 중 **하나라도** 걸리면 과매도로 봅니다 — 둘 다 요구하면
    실제 바닥에서도 조건이 안 맞아 차수가 들어가지 못하는 경우가 많습니다.
    둘 다 없으면(지표 계산 실패) '모름'이므로 False 입니다. 모르는 것을
    통과시키면 지표 조건을 켜 둔 의미가 없습니다.
    """
    notes = []
    if rsi is not None and float(rsi) <= p["split_oversold_rsi"]:
        notes.append(f"RSI {float(rsi):.1f} ≤ {p['split_oversold_rsi']:g}")
    if bb_pct is not None and float(bb_pct) <= p["split_oversold_bb"]:
        notes.append(f"볼린저 %B {float(bb_pct):.2f} ≤ {p['split_oversold_bb']:g}")
    return bool(notes), " · ".join(notes)


def overbought(p: dict, rsi=None, bb_pct=None) -> tuple[bool, str]:
    """지금이 과매수 구간인가. (판정, 설명)"""
    notes = []
    if rsi is not None and float(rsi) >= p["split_overbought_rsi"]:
        notes.append(f"RSI {float(rsi):.1f} ≥ {p['split_overbought_rsi']:g}")
    if bb_pct is not None and float(bb_pct) >= p["split_overbought_bb"]:
        notes.append(f"볼린저 %B {float(bb_pct):.2f} ≥ {p['split_overbought_bb']:g}")
    return bool(notes), " · ".join(notes)


# ---------------------------------------------------------------------------
# 판단 결과
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    ok: bool = False
    quantity: float = 0.0
    tranche: int = 0                              # 대상 차수 번호
    reason: str = ""                              # 주문 사유·로그에 그대로 씁니다
    rejects: list = field(default_factory=list)   # 왜 안 하는가
    # 거부 사유의 **종류**. 사람이 읽는 rejects 문구에는 RSI 값처럼 회전마다
    # 달라지는 숫자가 섞여 있습니다. 로그를 "사유가 바뀔 때만" 남기려면
    # 숫자가 빠진 안정적인 키가 따로 있어야 합니다.
    code: str = ""
    detail: dict = field(default_factory=dict)    # 판단에 쓴 숫자 전부 (감사용)

    def reject(self, code: str, text: str, **detail) -> "Decision":
        self.code = code
        self.rejects.append(text)
        if detail:
            self.detail.update(detail)
        return self

    def to_dict(self) -> dict:
        return {"ok": self.ok, "quantity": self.quantity, "tranche": self.tranche,
                "reason": self.reason, "rejects": self.rejects, "code": self.code,
                "detail": self.detail}


def _minutes_since(stamp) -> float | None:
    if not stamp:
        return None
    try:
        return (datetime.now() - datetime.fromisoformat(str(stamp))).total_seconds() / 60.0
    except (TypeError, ValueError):
        return None


def _round_trip_cost_pct(inst, price: float) -> float:
    """왕복 수수료·거래세를 가격 대비 %로. 계산 불능이면 0.

    ETF 는 매도 거래세가 면제라 값이 다릅니다 — 상수로 두면 ETF 쪽이 과하게
    보수적이 됩니다. instruments.costs 가 자산군을 이미 알고 있으므로 그걸 씁니다.
    """
    try:
        notional = float(price) * inst.multiplier
        if notional <= 0:
            return 0.0
        buy_fee, _ = inst.costs("buy", notional)
        sell_fee, sell_tax = inst.costs("sell", notional)
        return (buy_fee + sell_fee + sell_tax) / notional * 100.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 분할 매수 — 과매도 구간에서 다음 차수를 담을 것인가
# ---------------------------------------------------------------------------

def plan_buy(cfg: dict, inst, ledger: Ledger, price: float,
             rsi=None, bb_pct=None, now: datetime = None) -> Decision:
    """다음 차수를 살 것인가, 산다면 몇 주인가.

    조건 (전부 만족해야 합니다)
        ① 아직 남은 차수가 있다
        ② 현재가가 **직전 차수 체결가** 대비 계획된 하락률 아래다
        ③ (켜져 있으면) 과매도 지표가 확인된다
        ④ 직전 차수로부터 최소 시간이 지났다
        ⑤ 오늘 추가 매수 한도를 넘지 않았다

    ②의 기준이 '직전 차수 체결가'인 이유: 계획을 1차 가격으로 미리 못 박아
    두면 슬리피지가 누적되어 3차·4차의 실제 간격이 계획과 달라집니다.
    직전 체결가에서 재면 차수 간격이 항상 설정한 값 그대로입니다.
    """
    d = Decision()
    p = params(cfg)
    now = now or datetime.now()

    if not p["split_enabled"]:
        return d.reject("disabled", "분할 매매가 꺼져 있습니다 (split_enabled)")
    if not ledger.tranches:
        return d.reject("no_ledger",
                        "차수 원장이 없습니다 — 분할로 잡은 포지션이 아닙니다")
    if price <= 0:
        return d.reject("no_price", "현재가를 몰라 차수 판단을 할 수 없습니다")

    plan = ledger.plan or {}
    total = int(len(plan.get("qty") or []) or p["split_tranches"])

    # ① 남은 차수
    if ledger.count >= total:
        return d.reject("exhausted", f"차수를 다 썼습니다 ({ledger.count}/{total}차)")

    last = ledger.last
    n = ledger.next_n()
    if n > total:
        return d.reject("plan_overflow",
                        f"다음 차수 번호가 계획을 넘습니다 ({n} > {total})")

    # ② 하락률 — 직전 차수 체결가 대비
    drops = plan.get("drop") or _steps(p["split_buy_drop_steps"], total - 1,
                                       p["split_buy_drop_pct"])
    drop_pct = float(drops[min(ledger.count - 1, len(drops) - 1)]) if drops else p["split_buy_drop_pct"]
    trigger = last.price * (1 - drop_pct / 100.0)
    # 부호는 **가격 변화 기준**입니다 (내렸으면 음수). '하락률 +2%' 처럼 적으면
    # 로그를 읽는 사람이 오른 것으로 읽습니다.
    change_pct = (price - last.price) / last.price * 100.0
    if price > trigger:
        return d.reject(
            "price",
            f"{n}차 매수가 아직입니다 — 직전 차수 {last.price:,.0f} 대비 "
            f"{change_pct:+.2f}% (필요 -{drop_pct:g}%, 기준가 {trigger:,.0f})",
            tranche=n, price=price, trigger=round(trigger, 4),
            prev_price=last.price, change_pct=round(change_pct, 4))

    # ③ 과매도 지표
    is_oversold, note = oversold(p, rsi, bb_pct)
    if p["split_require_oversold"] and not is_oversold:
        have = []
        if rsi is not None:
            have.append(f"RSI {float(rsi):.1f}")
        if bb_pct is not None:
            have.append(f"%B {float(bb_pct):.2f}")
        return d.reject(
            "oversold",
            f"{n}차 가격 조건은 맞지만 과매도가 아닙니다 "
            f"({' · '.join(have) or '지표 없음'}) — 떨어진다는 사실만으로는 담지 않습니다",
            tranche=n, price=price, trigger=round(trigger, 4),
            rsi=(round(float(rsi), 2) if rsi is not None else None),
            bb_pct=(round(float(bb_pct), 3) if bb_pct is not None else None),
            need_rsi=p["split_oversold_rsi"], need_bb=p["split_oversold_bb"])

    # ④ 차수 간 최소 간격
    gap = _minutes_since(last.at)
    if gap is not None and gap < p["split_min_gap_min"]:
        return d.reject("gap", f"직전 차수로부터 {gap:.0f}분 — "
                        f"{p['split_min_gap_min']}분은 지나야 다음 차수를 담습니다",
                        tranche=n, gap_min=round(gap, 1))

    # ⑤ 하루 상한
    today = now.strftime("%Y-%m-%d")
    adds = ledger.adds_on(today)
    if adds >= p["split_max_adds_per_day"]:
        return d.reject("daily_cap",
                        f"오늘 추가 매수 한도 도달 "
                        f"({adds}/{p['split_max_adds_per_day']}회)",
                        tranche=n, adds_today=adds)

    quantities = plan.get("qty") or []
    quantity = float(quantities[n - 1]) if len(quantities) >= n else float(ledger.tranches[0].quantity)
    if quantity <= 0:
        return d.reject("qty", f"{n}차 계획 수량이 0입니다", tranche=n)

    d.ok = True
    d.quantity = quantity
    d.tranche = n
    d.reason = (f"{n}차 분할 매수 — 직전 차수 {last.price:,.0f} 대비 {change_pct:+.2f}% "
                f"(기준 -{drop_pct:g}%)" + (f" · 과매도 {note}" if note else ""))
    d.detail = {"tranche": n, "of": total, "quantity": quantity,
                "price": price, "trigger": round(trigger, 4),
                "prev_price": last.price, "change_pct": round(change_pct, 4),
                "drop_pct": drop_pct, "oversold": is_oversold, "oversold_note": note,
                "adds_today": adds, "gap_min": round(gap, 1) if gap is not None else None}
    return d


# ---------------------------------------------------------------------------
# 분할 매도 — 수익 난 차수만 골라 정리
# ---------------------------------------------------------------------------

def plan_sell(cfg: dict, inst, ledger: Ledger, price: float,
              rsi=None, bb_pct=None, held_quantity: float = None) -> Decision:
    """어느 차수를 팔 것인가.

    두 가지 매도 조건
        ① 그 차수 매수가 대비 계획된 익절률 도달
        ② (켜져 있으면) 과매수 구간 — 목표에 못 미쳐도 수익 난 차수를 정리

    두 경우 모두 **왕복 비용을 넘는 수익**이어야 합니다. 수수료·거래세를 빼면
    마이너스인 매도는 "이겼다"고 기록되지만 계좌는 줄어듭니다.

    손실 난 차수는 어떤 경우에도 팔지 않습니다. 이 시스템에서 손실 정리는
    전량 손절(check_exit)의 일이고, 분할 매도가 그걸 대신하기 시작하면
    "더 좋은 자리가 있으니까"라는 이유로 손절 규율이 무너집니다.

    한 회전에 **한 차수만** 팝니다. 여러 차수가 동시에 조건을 만족해도
    주문을 몰아 내지 않습니다 — 부분 체결·수량 어긋남이 겹치면 원장과 실제
    보유가 갈라지고, 그 상태는 사람이 손으로 풀어야 합니다.
    """
    d = Decision()
    p = params(cfg)

    if not p["split_enabled"]:
        d.rejects.append("분할 매매가 꺼져 있습니다 (split_enabled)")
        return d
    if not ledger.tranches:
        d.rejects.append("차수 원장이 없습니다 — 분할로 잡은 포지션이 아닙니다")
        return d
    if price <= 0:
        d.rejects.append("현재가를 몰라 차수 판단을 할 수 없습니다")
        return d

    cost_pct = _round_trip_cost_pct(inst, price)
    floor = cost_pct * p["split_cost_multiple"]
    is_overbought, ob_note = overbought(p, rsi, bb_pct)
    plan = ledger.plan or {}
    gains = plan.get("gain") or []

    # 마지막 한 차수는 남기지 않습니다 — 전량이 되면 포지션이 사라지고,
    # 그 청산은 원장 정리·실현손익 기록이 다른 경로(_submit_close)의 일입니다.
    # 여기서는 '부분 매도'만 하고, 전량 매도는 호출부가 판단하게 둡니다.
    sellable = sorted(ledger.tranches, key=lambda t: t.n, reverse=True)

    best = None
    for t in sellable:
        gain = t.gain_pct(price)
        target = float(gains[t.n - 1]) if len(gains) >= t.n else p["split_sell_gain_pct"]

        if gain >= target and gain > floor:
            best = (t, gain, target, f"{t.n}차 목표 +{target:g}% 도달")
            break
        if (p["split_overbought_sell"] and is_overbought
                and gain >= p["split_overbought_min_gain"] and gain > floor):
            best = (t, gain, target, f"과매수 정리 ({ob_note})")
            break

        if gain <= 0:
            d.rejects.append(f"{t.n}차: 수익이 없습니다 ({gain:+.2f}%) — "
                             "손실 차수는 분할 매도로 팔지 않습니다")
        elif gain <= floor:
            d.rejects.append(f"{t.n}차: 수익 {gain:+.2f}% 가 왕복비용 {cost_pct:.2f}% 의 "
                             f"{p['split_cost_multiple']:g}배({floor:.2f}%)에 못 미칩니다")
        else:
            d.rejects.append(f"{t.n}차: 수익 {gain:+.2f}% < 목표 {target:g}%"
                             + ("" if is_overbought else " (과매수도 아님)"))

    if best is None:
        return d

    tranche, gain, target, why = best
    quantity = tranche.quantity
    if held_quantity is not None:
        # 실제 보유가 원장보다 적으면(밖에서 일부 팔았거나 부분 체결) 보유분까지만
        # 팝니다. 원장 수량을 그대로 내면 증권사가 주문을 통째로 거부합니다.
        quantity = min(quantity, float(held_quantity))
    if quantity <= 0:
        d.rejects.append(f"{tranche.n}차: 팔 수량이 없습니다 (보유 {held_quantity})")
        return d

    d.ok = True
    d.quantity = quantity
    d.tranche = tranche.n
    d.reason = (f"{tranche.n}차 분할 매도 — {why} "
                f"(매수 {tranche.price:,.0f} → {price:,.0f}, {gain:+.2f}%)")
    d.detail = {"tranche": tranche.n, "quantity": quantity, "price": price,
                "entry": tranche.price, "gain_pct": round(gain, 4),
                "target_pct": target, "cost_pct": round(cost_pct, 4),
                "cost_floor_pct": round(floor, 4),
                "overbought": is_overbought, "overbought_note": ob_note,
                "remaining": ledger.count - 1}
    return d


# ---------------------------------------------------------------------------
# 원장 갱신 (호출부가 체결을 확인한 뒤에 부릅니다)
# ---------------------------------------------------------------------------

def add_tranche(ledger: Ledger, n: int, quantity: float, price: float,
                now: datetime = None) -> Ledger:
    now = now or datetime.now()
    ledger.tranches.append(Tranche(n=int(n), quantity=float(quantity),
                                   price=float(price), at=now.isoformat()))
    ledger.tranches.sort(key=lambda t: t.n)
    return ledger


def remove_tranche(ledger: Ledger, n: int, quantity: float = None) -> Ledger:
    """차수를 원장에서 뺍니다. 일부만 체결됐으면 남은 수량만큼 남겨둡니다."""
    for t in list(ledger.tranches):
        if t.n != int(n):
            continue
        if quantity is None or float(quantity) >= t.quantity - 1e-9:
            ledger.tranches.remove(t)
        else:
            t.quantity -= float(quantity)
        break
    return ledger


def describe(ledger: Ledger, price: float = 0.0) -> dict:
    """화면에 "지금 몇 차까지 담겨 있고 각 차수가 어떤 상태인지"를 보여주기 위한 요약."""
    plan = ledger.plan or {}
    total = int(len(plan.get("qty") or []) or 0)
    gains = plan.get("gain") or []
    rows = []
    for t in sorted(ledger.tranches, key=lambda x: x.n):
        target = float(gains[t.n - 1]) if len(gains) >= t.n else None
        rows.append({
            "n": t.n, "quantity": t.quantity, "price": t.price, "at": t.at,
            "gain_pct": round(t.gain_pct(price), 3) if price else None,
            "target_pct": target,
            "target_price": round(t.price * (1 + target / 100.0), 4) if target else None,
        })
    return {
        "count": ledger.count, "total": total,
        "quantity": ledger.quantity, "avg_price": round(ledger.avg_price, 4),
        "stop": plan.get("stop"),
        "next_trigger": (round(ledger.last.price
                               * (1 - float((plan.get("drop") or [0])[
                                   min(ledger.count - 1,
                                       len(plan.get("drop") or [0]) - 1)]) / 100.0), 4)
                         if ledger.tranches and (plan.get("drop") or []) and ledger.count < total
                         else None),
        "tranches": rows,
    }
