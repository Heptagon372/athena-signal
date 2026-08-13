"""
리스크 엔진 (Pre-Trade Gate)
---------------------------
모든 주문은 집행 전에 이 게이트를 통과해야 합니다. 우회 경로는 없습니다.

핵심 원칙 — fail-closed
    판단에 필요한 정보가 없으면 "일단 통과"가 아니라 **거부**합니다.
    현재가가 오래됐는지 모르겠으면 주문하지 않습니다. 계좌 평가금액을 못 읽으면
    주문하지 않습니다. 자동매매에서 모르는 채로 나가는 주문이 계좌를 없앱니다.

통제 계층
    1) 계좌 차단 (halt)   킬 스위치 · 일일 손실 한도 · 최대 낙폭
       → 신규 진입 전면 금지. 청산은 계속 허용합니다(위험을 줄이는 방향).
    2) 주문 검증 (gate)   장 상태 · 시세 신선도 · 종목 수 · 비중 · 증거금 · 만기
    3) 수량 축소 (reduce) 한도를 넘으면 거부 대신 **줄여서** 통과시킵니다.
       (거부만 하면 자본이 커질수록 아무 주문도 못 내는 상태가 됩니다)

청산 주문은 왜 검증이 느슨한가
    포지션을 줄이는 주문은 리스크를 낮춥니다. 손절을 리스크 게이트가 막으면
    그게 바로 사고입니다. 그래서 청산은 장 상태와 시세 유효성만 확인합니다.
"""

from dataclasses import dataclass, field

from engine import feed
from engine.instruments import ETF, FUTURES, OPTION, STOCK, Instrument


@dataclass
class RiskVerdict:
    approved: bool = False
    quantity: float = 0.0
    original_quantity: float = 0.0
    reasons: list = field(default_factory=list)     # 축소·조정 내역
    rejects: list = field(default_factory=list)     # 거부 사유
    halt: bool = False                              # 계좌 전체 중단 상태인가

    @property
    def reduced(self) -> bool:
        return self.approved and self.quantity < self.original_quantity

    def to_dict(self) -> dict:
        return {"approved": self.approved, "quantity": self.quantity,
                "original_quantity": self.original_quantity,
                "reduced": self.reduced, "reasons": self.reasons,
                "rejects": self.rejects, "halt": self.halt}


class RiskEngine:
    """한 번의 자동매매 회전 동안 쓰이는 리스크 판단 컨텍스트.

    계좌·포지션 스냅샷을 생성 시점에 고정합니다. 회전 중간에 값이 바뀌면
    "같은 회전에서 앞 종목은 통과, 뒤 종목은 거부" 같은 비결정적 동작이 나옵니다.
    """

    def __init__(self, cfg: dict, account: dict, positions: list, day: dict = None):
        self.cfg = cfg or {}
        self.account = account or {}
        self.positions = positions or []
        self.day = day or {}

    # -- 계좌 차단 --------------------------------------------------------
    def halt_reasons(self) -> list[str]:
        """지금 신규 진입을 전면 금지해야 하는 이유들 (없으면 빈 리스트)."""
        out = []
        cfg, acc, day = self.cfg, self.account, self.day

        if cfg.get("kill_switch"):
            out.append("킬 스위치가 켜져 있습니다.")

        total = float(acc.get("total_value") or 0)
        if total <= 0:
            out.append("계좌 평가금액을 읽을 수 없습니다 (안전을 위해 중단).")
            return out

        # 일일 손실은 **평가손익 기준**으로 잽니다 (storage/autotrade.touch_daily).
        # 총자산 차이로 재면 출금 한 번에 없던 손실이 잡혀 매매가 멈춥니다 —
        # 11만원 계좌에서 5만원을 빼면 -41% 로 즉시 한도 발동이었습니다.
        start = float(day.get("start_value") or 0)
        limit = float(cfg.get("daily_loss_limit_pct", 0) or 0)
        if start > 0 and limit > 0:
            pnl = day.get("pnl")
            change = ((float(pnl) / start * 100) if pnl is not None
                      else (total - start) / start * 100)
            if change <= -limit:
                out.append(f"일일 손실 한도 초과 ({change:+.2f}% / 한도 -{limit:g}%)")

        # 낙폭도 손익 기준 — 출금으로 총자산이 줄어든 것을 낙폭으로 읽으면 안 됩니다
        peak = float(day.get("peak_value") or 0)
        max_dd = float(cfg.get("max_drawdown_pct", 0) or 0)
        if max_dd > 0:
            dd = day.get("drawdown_pct")
            if dd is None:
                dd = (total - peak) / peak * 100 if peak > 0 else 0.0
            if float(dd) <= -max_dd:
                out.append(f"최대 낙폭 초과 ({float(dd):+.2f}% / 한도 -{max_dd:g}%)")

        return out

    # -- 진입 검증 --------------------------------------------------------
    def check_entry(self, inst: Instrument, side: str, quantity: float,
                    sig, plan, quote: dict, adding: bool = False) -> RiskVerdict:
        """`adding=True` 는 **계획된 추가 매수**입니다 (분할 매매의 다음 차수).

        추가 매수 금지(allow_pyramiding)만 면제하고, 나머지 한도는 전부 그대로
        받습니다 — 비중·현금·노출·자산군·장 상태. 분할이라는 이유로 한도를
        비켜가면 "차수를 나눴을 뿐인데 계좌 전체가 한 종목이 되는" 일이 생깁니다.

        면제하는 이유: 그 금지는 '신호가 또 떴다고 같은 종목을 덧사는 것'을
        막는 장치입니다. 분할은 살 양을 미리 정해 두고 나눠서 넣는 것이라
        성격이 반대입니다 — 오히려 한 번에 다 사지 않으려고 나눈 것입니다.
        """
        verdict = RiskVerdict(quantity=quantity, original_quantity=quantity)
        cfg = self.cfg

        halts = self.halt_reasons()
        if halts:
            verdict.rejects.extend(halts)
            verdict.halt = True
            return verdict

        # 1) 자산군 허용 여부
        allowed = cfg.get("asset_classes") or {}
        if not allowed.get(inst.asset_class, inst.asset_class in (STOCK, ETF)):
            verdict.rejects.append(f"{inst.asset_label} 자동매매가 꺼져 있습니다.")
            return verdict

        if side == "sell" and not inst.shortable:
            verdict.rejects.append("이 상품은 매도 진입(숏)이 불가능합니다.")
            return verdict
        if side == "sell" and not cfg.get("allow_short"):
            verdict.rejects.append("숏 진입이 설정에서 꺼져 있습니다.")
            return verdict

        # 2) 장 상태 — 신규 진입은 **프리마켓·정규장에서만** 냅니다.
        # 닫힌 장에 낸 주문은 다음 개장 때 원하지 않는 가격에 체결되고,
        # 애프터마켓·시간외 단일가는 호가가 얇아 스프레드부터 지고 시작합니다.
        tradable, why, _session = feed.entry_allowed_now(inst, cfg)
        if not tradable:
            verdict.rejects.append(why)
            return verdict

        # 3) 시세 신선도 — 오래된 가격으로 내는 주문이 가장 위험합니다
        max_age = float(cfg.get("max_quote_age_sec", 120))
        age = float((quote or {}).get("age_sec") or 0)
        if age > max_age:
            verdict.rejects.append(f"시세가 {age:.0f}초 지났습니다 (허용 {max_age:.0f}초)")
            return verdict

        # 4) 만기 임박 파생 — 진입하자마자 청산해야 하는 포지션은 만들지 않습니다
        if inst.asset_class in (FUTURES, OPTION):
            days = feed.days_to_expiry(inst)
            min_days = int(cfg.get("deriv_min_days_to_expiry", 2))
            if days is not None and days <= min_days:
                verdict.rejects.append(f"만기까지 {days}일 — 신규 진입 금지")
                return verdict

        # 5) 보유 종목 수
        max_positions = int(cfg.get("max_positions", 5))
        holding_keys = {p.key for p in self.positions}
        if inst.key not in holding_keys and len(holding_keys) >= max_positions:
            verdict.rejects.append(f"보유 종목 수 상한 {max_positions}개에 도달했습니다.")
            return verdict

        # 6) 같은 종목 추가 매수 (기본 금지 — 물타기는 손실을 키웁니다)
        if inst.key in holding_keys and not adding and not cfg.get("allow_pyramiding"):
            verdict.rejects.append("이미 보유 중인 종목입니다 (추가 진입 꺼짐).")
            return verdict

        # 7) 수량·금액 검증 + 필요 시 축소
        total = float(self.account.get("total_value") or 0)
        available = float(self.account.get("available_cash") or 0)
        price_krw = float(getattr(sig, "price_krw", 0) or quote.get("price_krw") or 0)
        if price_krw <= 0:
            verdict.rejects.append("원화 환산 가격을 계산할 수 없습니다.")
            return verdict

        unit_cost = inst.margin_required(price_krw, 1, side)
        if unit_cost <= 0:
            verdict.rejects.append("단위 비용을 계산할 수 없습니다.")
            return verdict

        quantity = self._cap(verdict, quantity, unit_cost,
                             cap=total * float(cfg.get("position_pct", 20.0)) / 100.0,
                             label=f"종목 비중 상한 {cfg.get('position_pct', 20)}%")

        max_order = float(cfg.get("max_order_krw", 0) or 0)
        if max_order > 0:
            quantity = self._cap(verdict, quantity, unit_cost, cap=max_order,
                                 label=f"1회 주문 상한 {max_order:,.0f}원")

        # 총 노출 상한 (레버리지 통제) — 파생은 명목금액이 커서 이게 없으면
        # 증거금만 보고 계좌의 10배짜리 포지션을 들 수 있습니다.
        gross_cap_pct = float(cfg.get("max_gross_exposure_pct", 0) or 0)
        if gross_cap_pct > 0 and total > 0:
            current = sum(abs(p.market_value or 0) for p in self.positions)
            room = total * gross_cap_pct / 100.0 - current
            notional_per_unit = inst.notional(price_krw, 1)
            if room <= 0:
                verdict.rejects.append(
                    f"총 노출 한도 초과 (현재 {current / total * 100:.0f}% / "
                    f"한도 {gross_cap_pct:g}%)")
                return verdict
            quantity = self._cap(verdict, quantity, notional_per_unit, cap=room,
                                 label=f"총 노출 한도 {gross_cap_pct:g}%")

        # 가용 현금 (수수료 여유 1%)
        quantity = self._cap(verdict, quantity, unit_cost, cap=available * 0.99,
                             label=f"주문가능금액 {available:,.0f}원")

        quantity = inst.round_quantity(quantity)
        if quantity <= 0:
            # **어느 한도가 잘랐는지** 함께 남깁니다. 예전에는 "한도를 적용하니
            # 0이 되었습니다"로만 찍혀서, 종목 비중·총 노출·현금 중 무엇이
            # 막았는지 알 수 없었습니다 (이미 거의 다 투자된 계좌에서는 총 노출
            # 한도가 범인인데, 화면만 봐서는 현금 부족으로 오해합니다).
            detail = " / ".join(verdict.reasons) if verdict.reasons else ""
            verdict.rejects.append(
                "한도를 적용하니 주문 수량이 0이 되었습니다."
                + (f" — {detail}" if detail else ""))
            return verdict

        min_order = float(cfg.get("min_order_krw", 0) or 0)
        basis = unit_cost * quantity
        if min_order > 0 and basis < min_order:
            verdict.rejects.append(
                f"주문 금액 {basis:,.0f}원이 최소 {min_order:,.0f}원 미만입니다.")
            return verdict

        verdict.quantity = quantity
        verdict.approved = True
        return verdict

    def _cap(self, verdict: RiskVerdict, quantity: float, unit_cost: float,
             cap: float, label: str) -> float:
        """한도를 넘으면 수량을 줄이고 그 사실을 기록합니다."""
        if unit_cost <= 0 or cap <= 0:
            return quantity
        allowed = cap / unit_cost
        if allowed < quantity:
            verdict.reasons.append(
                f"{label} 적용: {quantity:g} → {max(allowed, 0):.4g}")
            return max(allowed, 0.0)
        return quantity

    def _regular_only(self, inst: Instrument) -> bool:
        """이 종목에 '정규장만' 제한을 적용할 것인가.

        미국 종목은 us_extended_hours 로 프리·애프터마켓을 따로 열 수 있습니다.
        한국 시간외와 스위치를 분리한 이유: 한국 시간외 단일가(10분 단위 ±10%)와
        미국 확장 시간(연속 체결)은 성격이 달라 한 스위치로 묶으면 안 됩니다.
        """
        regular_only = bool(self.cfg.get("regular_session_only", True))
        if regular_only and inst.market == "US" and self.cfg.get("us_extended_hours"):
            return False
        return regular_only

    # -- 청산 검증 --------------------------------------------------------
    def check_exit(self, inst: Instrument, quote: dict, urgent: bool = False) -> RiskVerdict:
        """청산은 리스크를 줄이는 방향이라 통과를 기본으로 합니다."""
        verdict = RiskVerdict(approved=True)

        if not quote or not quote.get("price"):
            verdict.approved = False
            verdict.rejects.append("현재가를 조회할 수 없어 청산을 보류합니다.")
            return verdict

        # 장이 닫혀 있으면 청산도 나갈 수 없습니다(주문 자체가 불가).
        # 다만 '정규장만' 제한은 긴급 청산에는 적용하지 않습니다.
        regular_only = self._regular_only(inst) and not urgent
        tradable, why = feed.is_tradable_now(inst, regular_only=regular_only)
        if not tradable:
            verdict.approved = False
            verdict.rejects.append(why)
        return verdict


def describe_limits(cfg: dict) -> list[dict]:
    """현재 설정된 한도를 화면에 그대로 보여주기 위한 요약."""
    return [
        {"key": "max_positions", "label": "최대 보유 종목",
         "value": f"{cfg.get('max_positions', 5)}개"},
        {"key": "position_pct", "label": "종목당 최대 비중",
         "value": f"{cfg.get('position_pct', 20)}%"},
        {"key": "risk_per_trade_pct", "label": "1회 위험 예산",
         "value": f"총자산의 {cfg.get('risk_per_trade_pct', 1)}%"},
        {"key": "daily_loss_limit_pct", "label": "일일 손실 한도",
         "value": f"-{cfg.get('daily_loss_limit_pct', 3)}%"},
        {"key": "max_drawdown_pct", "label": "최대 낙폭 한도",
         "value": f"-{cfg.get('max_drawdown_pct', 10)}%"},
        {"key": "max_gross_exposure_pct", "label": "총 노출 한도",
         "value": (f"{cfg.get('max_gross_exposure_pct')}%"
                   if cfg.get("max_gross_exposure_pct") else "제한 없음")},
        {"key": "max_order_krw", "label": "1회 주문 상한",
         "value": (f"{float(cfg.get('max_order_krw', 0)):,.0f}원"
                   if cfg.get("max_order_krw") else "제한 없음")},
        {"key": "regular_session_only", "label": "정규장에만 매매",
         "value": "예" if cfg.get("regular_session_only", True) else "아니오"},
        {"key": "us_extended_hours", "label": "미국 프리·애프터마켓",
         "value": "허용" if cfg.get("us_extended_hours") else "차단"},
        {"key": "min_one_unit", "label": "최소 1주 허용 (소액 계좌)",
         "value": "예" if cfg.get("min_one_unit") else "아니오"},
        {"key": "us_order_funding", "label": "미국 주문 재원",
         "value": ("원화도 시도(통합증거금)"
                   if cfg.get("us_order_funding") == "krw" else "외화(달러)만")},
    ]
