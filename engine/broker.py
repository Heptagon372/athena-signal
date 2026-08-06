"""
브로커 포트 & 어댑터 (Execution Adapter)
---------------------------------------
자동매매 엔진은 이 인터페이스 하나만 알고, "가상 계좌인지 실제 증권사인지"는
모릅니다. 모드 전환이 설정 한 줄로 끝나는 이유가 이것입니다.

    paper   내부 가상 계좌 (athena.db) — 키 없이 즉시 동작, 기본값
    mock    한국투자증권 **모의투자 서버**로 실제 주문 전송 (가상 자금)
    live    한국투자증권 **실전 서버**로 실제 주문 전송 (실제 자금)

멱등성 (같은 주문이 두 번 나가지 않게)
    엔진이 만든 client_order_id 를 주문마다 부여하고, 이미 처리된 id 는
    다시 전송하지 않습니다. 네트워크 타임아웃 후 재시도할 때 이중 주문이
    나가는 것이 자동매매에서 가장 비싼 실수입니다.

fail-closed 원칙
    · 현재가를 못 구하면 주문하지 않습니다 (0원 매수 방지)
    · live 모드인데 실주문 스위치가 꺼져 있으면 거부합니다
    · 파생 주문인데 KIS 가 설정돼 있지 않으면 거부합니다
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from data_sources import fx, kis_trading
from engine import feed, instruments
from engine.instruments import FUTURES, OPTION, STOCK, Instrument
from models import MARKET_US
from storage import derivatives, paper

PAPER = "paper"
MOCK = "mock"
LIVE = "live"

MODE_LABELS = {
    PAPER: "모의(내부 계좌)",
    MOCK: "KIS 모의투자 서버",
    LIVE: "KIS 실전 (실제 자금)",
}


@dataclass
class OrderResult:
    ok: bool
    action: str = ""                 # buy | sell | open_long | open_short | close
    quantity: float = 0.0            # 주문 수량
    price: float = 0.0               # 체결(또는 지정)가 — 호가 통화 기준
    price_krw: float = 0.0
    fee: float = 0.0
    tax: float = 0.0
    realized_pnl: float | None = None
    status: str = "rejected"         # filled | partial | pending | cancelled | rejected
    client_order_id: str = ""
    broker_order_id: str = ""
    error: str = ""
    message: str = ""
    # 체결 추적 — 접수와 체결은 다른 사건입니다
    intended_price: float = 0.0      # 주문을 낼 때 보고 있던 가격
    filled_quantity: float = 0.0
    avg_fill_price: float | None = None
    raw: dict = field(default_factory=dict)

    @property
    def slippage_bps(self) -> float | None:
        """의도한 가격 대비 실제 체결가의 차이(bp). 매도는 부호를 뒤집어
        '나에게 불리한 방향'이 항상 양수가 되게 합니다."""
        if not self.intended_price or not self.avg_fill_price:
            return None
        raw = (self.avg_fill_price - self.intended_price) / self.intended_price * 10_000
        selling = self.action in ("sell", "close", "open_short")
        return round(-raw if selling else raw, 2)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "action": self.action, "quantity": self.quantity,
            "price": self.price, "price_krw": self.price_krw,
            "fee": self.fee, "tax": self.tax, "realized_pnl": self.realized_pnl,
            "status": self.status, "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "intended_price": self.intended_price,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "slippage_bps": self.slippage_bps,
            "error": self.error, "message": self.message,
        }


@dataclass
class OrderStatus:
    """브로커에 조회한 주문의 현재 상태."""
    known: bool = False              # 브로커에서 찾았는가 (못 찾으면 판단 보류)
    status: str = "pending"          # filled | partial | pending | cancelled
    filled_quantity: float = 0.0
    remaining: float = 0.0
    avg_fill_price: float | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {"known": self.known, "status": self.status,
                "filled_quantity": self.filled_quantity, "remaining": self.remaining,
                "avg_fill_price": self.avg_fill_price, "detail": self.detail}


@dataclass
class Position:
    key: str
    name: str
    asset_class: str
    side: str                # long | short
    quantity: float
    avg_price: float
    multiplier: float = 1.0
    current_price: float | None = None
    market_value: float = 0.0        # 원화 평가금액 (파생은 명목금액)
    unrealized_pnl: float = 0.0
    margin: float = 0.0
    opened_at: str = ""

    @property
    def is_derivative(self) -> bool:
        return self.asset_class in (FUTURES, OPTION)

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name, "asset_class": self.asset_class,
            "asset_label": instruments.ASSET_LABELS.get(self.asset_class, self.asset_class),
            "side": self.side, "quantity": self.quantity, "avg_price": self.avg_price,
            "multiplier": self.multiplier, "current_price": self.current_price,
            "market_value": self.market_value, "unrealized_pnl": self.unrealized_pnl,
            "margin": self.margin, "opened_at": self.opened_at,
        }


# ---------------------------------------------------------------------------
# 포트 (인터페이스)
# ---------------------------------------------------------------------------

class Broker(ABC):
    mode = PAPER

    @property
    def label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)

    @abstractmethod
    def account(self) -> dict:
        """{cash, available_cash, equity_value, margin_locked, total_value, ...}"""

    @abstractmethod
    def positions(self) -> list[Position]:
        ...

    @abstractmethod
    def submit(self, inst: Instrument, side: str, quantity: float,
               order_type: str = "market", price: float = None,
               note: str = "", client_order_id: str = "") -> OrderResult:
        """진입/증가 주문. side: buy(롱) | sell(숏 또는 매도)"""

    @abstractmethod
    def close(self, inst: Instrument, quantity: float = None,
              note: str = "", client_order_id: str = "") -> OrderResult:
        """보유 포지션 청산 (수량 생략 시 전량)."""

    def order_status(self, record: dict) -> OrderStatus:
        """접수한 주문이 지금 어떤 상태인지 브로커에 물어봅니다.

        **못 찾으면 known=False** 로 돌려야 합니다. 조회 실패를 '체결됨'이나
        '취소됨'으로 넘겨짚으면 계좌와 내부 상태가 어긋납니다.
        """
        return OrderStatus(known=False, detail="이 브로커는 주문 조회를 지원하지 않습니다.")

    def cancel(self, record: dict) -> OrderResult:
        """미체결 주문 취소."""
        return OrderResult(ok=False, error="이 브로커는 주문 취소를 지원하지 않습니다.")

    def position_for(self, key: str) -> Position | None:
        return next((p for p in self.positions() if p.key == key), None)

    def health(self) -> dict:
        """주문을 낼 수 있는 상태인지. 콘솔에 그대로 표시합니다."""
        return {"mode": self.mode, "label": self.label, "ready": True, "detail": ""}


# ---------------------------------------------------------------------------
# 멱등 처리 — 이미 보낸 주문은 다시 보내지 않습니다
# ---------------------------------------------------------------------------

_sent_lock = threading.Lock()
_sent_orders: dict[str, OrderResult] = {}
_SENT_MAX = 2000


def _remember_order(client_order_id: str, result: OrderResult):
    if not client_order_id:
        return
    with _sent_lock:
        if len(_sent_orders) > _SENT_MAX:
            _sent_orders.clear()
        _sent_orders[client_order_id] = result


def _already_sent(client_order_id: str) -> OrderResult | None:
    if not client_order_id:
        return None
    with _sent_lock:
        return _sent_orders.get(client_order_id)


# ---------------------------------------------------------------------------
# 모의 계좌 브로커 (기본)
# ---------------------------------------------------------------------------

class PaperBroker(Broker):
    """athena.db 안의 가상 계좌. 실제 주문은 절대 나가지 않습니다.

    주식/ETF 는 storage/paper.py, 선물/옵션은 storage/derivatives.py 를 씁니다.
    예수금은 두 쪽이 공유합니다(실제 증권계좌와 같은 구조).
    """

    mode = PAPER

    def __init__(self, user_id: int, slippage_bps: float = 5.0):
        self.user_id = user_id
        # 조회 시점 현재가에 그대로 체결시키면 실제보다 좋은 성과가 나옵니다.
        # 스프레드·시장충격을 한 방향으로만 불리하게 적용해 현실에 가깝게 만듭니다.
        self.slippage_bps = max(float(slippage_bps or 0), 0.0)
        paper.ensure_account(user_id)
        derivatives.init()

    def _fill_price(self, inst: Instrument, price: float, side: str) -> float:
        """체결가 = 현재가 ± 슬리피지 (항상 나에게 불리한 방향)."""
        if not price or self.slippage_bps <= 0:
            return price
        drift = price * self.slippage_bps / 10_000
        filled = price + drift if side == "buy" else price - drift
        return inst.round_price(filled) or filled

    # -- 조회 -------------------------------------------------------------
    def account(self) -> dict:
        """계좌 요약 — positions() 한 번의 시세 조회 결과에서 전부 계산합니다.

        예전에는 account 와 positions 가 각자 보유 종목 시세를 조회해서,
        상태 화면 한 번에 같은 시세를 **두 번씩** 네트워크로 받았습니다.
        보유가 5종목이면 호출 10번 — 화면이 느린 주범이었습니다.
        """
        acc = paper.get_account(self.user_id)
        positions = self.positions()
        equity_value = sum(p.market_value for p in positions if not p.is_derivative)
        margin_locked = sum(p.margin for p in positions if p.is_derivative)
        deriv_pnl = sum(p.unrealized_pnl for p in positions if p.is_derivative)
        return {
            "mode": self.mode,
            "label": self.label,
            "cash": acc["cash"],
            "available_cash": acc["available_cash"],
            "reserved_cash": acc["reserved_cash"],
            "initial_cash": acc["initial_cash"],
            "equity_value": equity_value,
            "domestic_equity": equity_value,
            # 모의계좌는 매수 즉시 현금이 빠지므로(결제 지연이 없습니다)
            # 매입금액을 총자산에서 빼지 않습니다 — 실계좌와 뜻이 다른 값입니다.
            "purchase_amount": sum(p.avg_price * p.quantity for p in positions
                                   if not p.is_derivative),
            "unrealized_pnl": sum(p.unrealized_pnl for p in positions
                                  if not p.is_derivative),
            "margin_locked": margin_locked,
            "derivative_pnl": deriv_pnl,
            "notional_exposure": sum(abs(p.market_value) for p in positions),
            "total_value": acc["cash"] + equity_value + margin_locked + deriv_pnl,
        }

    def _deriv_price(self, symbol_key: str) -> float | None:
        inst = instruments.parse_derivative(symbol_key)
        if not inst:
            return None
        q = feed.quote(inst)
        return q["price"] if q else None

    def positions(self) -> list[Position]:
        out = []
        for h in paper.get_holdings(self.user_id):
            inst = instruments.try_resolve(h["ticker"])
            price = feed.price_krw(inst) if inst else None
            qty, avg = float(h["quantity"]), float(h["avg_price"])
            mark = price if price is not None else avg
            out.append(Position(
                key=h["ticker"], name=h["name"] or h["ticker"],
                asset_class=(inst.asset_class if inst else STOCK),
                side="long", quantity=qty, avg_price=avg,
                current_price=price, market_value=mark * qty,
                unrealized_pnl=(mark - avg) * qty,
                opened_at=h.get("updated_at", ""),
            ))

        for p in derivatives.get_positions(self.user_id):
            price = self._deriv_price(p["symbol"])
            entry, qty, mult = (float(p["avg_price"]), float(p["quantity"]),
                                float(p["multiplier"]))
            mark = price if price is not None else entry
            direction = 1 if p["side"] == "long" else -1
            out.append(Position(
                key=p["symbol"], name=p["name"] or p["symbol"],
                asset_class=p["asset_class"], side=p["side"],
                quantity=qty, avg_price=entry, multiplier=mult,
                current_price=price, market_value=mark * qty * mult,
                unrealized_pnl=(mark - entry) * mult * qty * direction,
                margin=float(p["margin"]), opened_at=p["opened_at"],
            ))
        return out

    # -- 주문 -------------------------------------------------------------
    def submit(self, inst: Instrument, side: str, quantity: float,
               order_type: str = "market", price: float = None,
               note: str = "", client_order_id: str = "") -> OrderResult:
        prior = _already_sent(client_order_id)
        if prior:
            return prior

        quote = feed.quote(inst)
        if not quote:
            return OrderResult(ok=False, error="현재가를 조회할 수 없어 주문하지 않았습니다.",
                               client_order_id=client_order_id)

        intended = float(price) if (order_type == "limit" and price) else quote["price"]
        exec_price = self._fill_price(inst, intended, side)
        exec_krw = (exec_price if inst.currency == "KRW"
                    else fx.to_krw(exec_price, inst.currency))

        if inst.is_derivative:
            pos_side = "long" if side == "buy" else "short"
            res = derivatives.open_position(self.user_id, inst, pos_side,
                                            quantity, exec_price, note=note)
            result = OrderResult(
                ok=bool(res.get("ok")),
                action=f"open_{pos_side}",
                quantity=res.get("quantity", 0), price=exec_price, price_krw=exec_krw,
                fee=res.get("fee", 0), status="filled" if res.get("ok") else "rejected",
                error=res.get("error", ""), client_order_id=client_order_id,
                intended_price=intended,
                filled_quantity=res.get("quantity", 0) if res.get("ok") else 0.0,
                avg_fill_price=exec_price if res.get("ok") else None,
                message=f"증거금 {res.get('margin', 0):,.0f}원" if res.get("ok") else "",
            )
            _remember_order(client_order_id, result)
            return result

        # 주식/ETF — 숏은 지원하지 않습니다 (개인 현물 계좌)
        if side == "sell":
            return self.close(inst, quantity=quantity, note=note,
                              client_order_id=client_order_id)

        res = paper.buy(self.user_id, inst, exec_krw, quantity=quantity, note=note)
        result = OrderResult(
            ok=bool(res.get("ok")), action="buy",
            quantity=res.get("quantity", 0), price=exec_price, price_krw=exec_krw,
            fee=res.get("fee", 0), tax=res.get("tax", 0),
            status="filled" if res.get("ok") else "rejected",
            error=res.get("error", ""), client_order_id=client_order_id,
            intended_price=intended,
            filled_quantity=res.get("quantity", 0) if res.get("ok") else 0.0,
            avg_fill_price=exec_price if res.get("ok") else None,
        )
        _remember_order(client_order_id, result)
        return result

    def close(self, inst: Instrument, quantity: float = None,
              note: str = "", client_order_id: str = "") -> OrderResult:
        prior = _already_sent(client_order_id)
        if prior:
            return prior

        quote = feed.quote(inst)
        if not quote:
            return OrderResult(ok=False, error="현재가를 조회할 수 없어 청산하지 않았습니다.",
                               client_order_id=client_order_id)
        intended = quote["price"]
        position = self.position_for(inst.key)
        # 롱 청산은 매도, 숏 청산은 매수 — 슬리피지 방향이 반대입니다
        exit_side = "buy" if (position and position.side == "short") else "sell"
        price = self._fill_price(inst, intended, exit_side)
        price_krw = (price if inst.currency == "KRW" else fx.to_krw(price, inst.currency))

        if inst.is_derivative:
            res = derivatives.close_position(self.user_id, inst, quantity=quantity,
                                             price=price, note=note)
            result = OrderResult(
                ok=bool(res.get("ok")), action="close",
                quantity=res.get("quantity", 0), price=price, price_krw=price_krw,
                fee=res.get("fee", 0), realized_pnl=res.get("realized_pnl"),
                status="filled" if res.get("ok") else "rejected",
                error=res.get("error", ""), client_order_id=client_order_id,
                intended_price=intended,
                filled_quantity=res.get("quantity", 0) if res.get("ok") else 0.0,
                avg_fill_price=price if res.get("ok") else None,
            )
        else:
            res = paper.sell(self.user_id, inst, price_krw, quantity=quantity, note=note)
            result = OrderResult(
                ok=bool(res.get("ok")), action="sell",
                quantity=res.get("quantity", 0), price=price, price_krw=price_krw,
                fee=res.get("fee", 0), tax=res.get("tax", 0),
                realized_pnl=res.get("realized_pnl"),
                status="filled" if res.get("ok") else "rejected",
                error=res.get("error", ""), client_order_id=client_order_id,
                intended_price=intended,
                filled_quantity=res.get("quantity", 0) if res.get("ok") else 0.0,
                avg_fill_price=price if res.get("ok") else None,
            )
        _remember_order(client_order_id, result)
        return result

    def order_status(self, record: dict) -> OrderStatus:
        """모의 계좌는 즉시 체결이라 원장에 적힌 상태가 곧 최종 상태입니다."""
        filled = float(record.get("filled_quantity") or record.get("quantity") or 0)
        return OrderStatus(
            known=True,
            status=record.get("status") or "filled",
            filled_quantity=filled,
            remaining=max(float(record.get("quantity") or 0) - filled, 0.0),
            avg_fill_price=record.get("avg_fill_price") or record.get("price"),
            detail="모의 계좌 즉시 체결",
        )

    def cancel(self, record: dict) -> OrderResult:
        return OrderResult(ok=False, status="filled",
                           error="모의 계좌 주문은 즉시 체결되어 취소할 것이 없습니다.")


# ---------------------------------------------------------------------------
# 한국투자증권 브로커 (모의투자 서버 / 실전)
# ---------------------------------------------------------------------------

def _overseas_funds_block(inst: Instrument, exchange: str, price: float,
                          quantity: float) -> str:
    """미국 매수를 내기 전 자금을 확인합니다. 막아야 하면 그 사유, 아니면 "".

    조회에 실패하면 **막지 않습니다.** KIS 점검이나 일시 오류로 매매가 통째로
    멈추는 것이, 어쩌다 한 번 거부당하는 것보다 나쁩니다.
    """
    check = kis_trading.overseas_buyable(inst.key, exchange, price)
    if not check.get("ok"):
        return ""

    need = price * quantity
    cash = float(check.get("cash_usd") or 0)
    after_fx = float(check.get("after_fx_usd") or 0)

    # 통합증거금이 켜져 있으면 KIS 가 원화를 환전해 주므로 after_fx 로도 주문됩니다.
    # 켜져 있는지는 이 조회로 알 수 없으니 넉넉한 쪽으로 판단하고, 실제 가부는
    # KIS 에 맡깁니다 — 확실히 안 되는 경우(둘 다 부족)만 미리 막습니다.
    usable = max(cash, after_fx)
    if usable >= need:
        return ""

    # 호출부가 종목명을 앞에 붙이므로 여기서는 사유만 씁니다
    if usable <= 0:
        # OpenAPI 주문은 **외화 예수금 기준**으로 나갑니다. 앱의 '통합매수'는
        # 원화를 쓰지만 API 주문에는 그게 적용되지 않아, 앱에서는 살 수 있는데
        # 자동매매만 거부되는 상황이 생깁니다. 해법은 환전입니다.
        return ("외화(달러) 예수금이 0입니다 — 자동매매 주문은 원화가 아니라 외화로 "
                "나갑니다. 증권사 앱에서 원화를 달러로 환전하면 바로 주문됩니다. "
                "(앱의 '통합매수'는 원화로 사지지만 API 주문에는 적용되지 않습니다)")

    hint = ""
    if cash <= 0 < after_fx:
        hint = " (외화 예수금은 0 — 환전하거나 통합증거금을 신청하면 원화를 씁니다)"
    return (f"{quantity:g}주에 ${need:,.2f} 필요한데 "
            f"주문가능금액은 ${usable:,.2f} 입니다.{hint}")


class KISBroker(Broker):
    """실제 증권사로 주문을 보냅니다.

    mode='mock' 이면 모의투자 서버(가상 자금), 'live' 면 실전 서버(실제 자금)입니다.
    두 경우의 차이는 **서버 주소와 tr_id 뿐**이며 코드 경로는 같습니다
    (같은 코드가 검증되어야 실전에서 처음 도는 코드가 없습니다).
    """

    def __init__(self, user_id: int, mode: str = MOCK, cfg: dict = None):
        self.user_id = user_id
        self.mode = mode
        self.cfg = cfg or {}

    def health(self) -> dict:
        st = kis_trading.status()
        problems = []
        if not st["keys_configured"]:
            problems.append("KIS APP KEY/SECRET 미설정")
        if not st["account_configured"]:
            problems.append("KIS_ACCOUNT(계좌번호) 미설정")
        if self.mode == LIVE and st["mock"]:
            problems.append("KIS_MOCK=1 상태에서는 실전 주문을 낼 수 없습니다")
        if self.mode == LIVE and not st["live_enabled"]:
            problems.append("실주문 스위치(KIS_LIVE_TRADING=1)가 꺼져 있습니다")
        if self.mode == MOCK and not st["mock"]:
            problems.append("모의투자 모드는 KIS_MOCK=1 이 필요합니다")
        return {
            "mode": self.mode, "label": self.label,
            "ready": not problems,
            "detail": " / ".join(problems),
            "server": st["server"], "account": st["account_masked"],
        }

    # -- 조회 -------------------------------------------------------------
    def account(self) -> dict:
        stock = kis_trading.stock_balance()
        deriv = kis_trading.deriv_balance()
        overseas = kis_trading.overseas_balance_all()
        cash = stock.get("cash", 0.0) if stock.get("ok") else 0.0
        available = stock.get("available_cash", cash) if stock.get("ok") else 0.0
        equity_value = stock.get("eval_amount", 0.0) if stock.get("ok") else 0.0
        margin = deriv.get("margin_used", 0.0) if deriv.get("ok") else 0.0

        # 미국 보유분 — 매입금액과 평가금액을 따로 잡습니다.
        #
        # 통합증거금 계좌는 해외 매수대금이 결제일까지 원화 예수금에 그대로
        # 남아 있습니다(산 직후에도 예수금이 안 줄어듭니다). 그래서 총자산은
        #     총예수금 − 매입금액(아직 예수금에 남아 있는 빚) + 평가금액
        # 이고, 정리하면 총예수금 + 평가손익 입니다. 예수금에 평가금액을 그냥
        # 더하면 매수대금이 두 번 잡혀 총자산이 부풀고 주문 수량·손실 한도가
        # 전부 어긋납니다.
        overseas_purchase = overseas_eval = 0.0
        if overseas.get("ok") and overseas.get("positions"):
            from data_sources import fx
            rate, _ = fx.usd_krw()
            overseas_purchase = sum(p.get("purchase_amount") or 0
                                    for p in overseas["positions"]) * rate
            overseas_eval = sum(p.get("eval_amount") or 0
                                for p in overseas["positions"]) * rate
        overseas_pnl = overseas_eval - overseas_purchase
        return {
            "mode": self.mode,
            "label": self.label,
            "cash": cash,
            "available_cash": available,
            "reserved_cash": max(cash - available, 0.0),
            "initial_cash": cash + equity_value,     # 실계좌는 초기자본 개념이 없습니다
            # equity_value = 화면의 '평가금액' — 국내 유가증권 + 해외 평가금액.
            # 총자산 계산에는 이걸 그대로 더하지 않습니다(아래 total_value 참고).
            "equity_value": equity_value + overseas_eval,
            "domestic_equity": equity_value,
            "purchase_amount": overseas_purchase,    # 매입금액 (아직 예수금에 남아 있음)
            "unrealized_pnl": overseas_pnl,          # 평가손익 = 평가금액 − 매입금액
            "margin_locked": margin,
            "derivative_pnl": sum(p.get("pnl", 0) for p in deriv.get("positions", [])),
            "notional_exposure": equity_value + overseas_eval,
            "total_value": cash + equity_value + margin + overseas_pnl,
            "overseas_pnl": overseas_pnl,
            "errors": [x.get("error") for x in (stock, deriv, overseas)
                       if not x.get("ok")],
        }

    def positions(self) -> list[Position]:
        out = []
        # 조회에 실패한 계좌가 있으면 이 목록은 '보유분 전체'가 아닙니다.
        # 호출부는 이걸 보고 "포지션이 사라졌다"는 판단을 미뤄야 합니다.
        self.position_errors: list[str] = []
        stock = kis_trading.stock_balance()
        if not stock.get("ok"):
            self.position_errors.append(f"국내 잔고: {stock.get('error') or '조회 실패'}")
        if stock.get("ok"):
            for p in stock["positions"]:
                inst = instruments.try_resolve(p["code"])
                out.append(Position(
                    key=p["code"], name=p["name"],
                    asset_class=(inst.asset_class if inst else STOCK),
                    side="long", quantity=p["quantity"], avg_price=p["avg_price"],
                    current_price=p["current_price"] or None,
                    market_value=p["eval_amount"], unrealized_pnl=p["pnl"],
                ))

        # 미국 보유분 — 이걸 빼면 엔진이 산 종목을 '없는 포지션'으로 보고
        # 손절·익절을 아예 검사하지 않습니다 (실계좌에서 이건 사고입니다).
        overseas = kis_trading.overseas_balance_all()
        if not overseas.get("ok"):
            self.position_errors.append(overseas.get("error") or "해외 잔고 조회 실패")
        if overseas.get("ok"):
            from data_sources import fx
            rate, _ = fx.usd_krw()
            for p in overseas["positions"]:
                inst = instruments.try_resolve(p["code"])
                # 단가는 달러 그대로 둡니다 — 신호·손절선이 달러 기준이라
                # 여기서 원화로 바꾸면 청산 판단이 통째로 어긋납니다.
                mark = p["current_price"] or p["avg_price"]
                out.append(Position(
                    key=p["code"], name=p["name"] or p["code"],
                    asset_class=(inst.asset_class if inst else STOCK),
                    side="long", quantity=p["quantity"], avg_price=p["avg_price"],
                    current_price=p["current_price"] or None,
                    # 노출 한도는 원화 기준이라 평가금액만 환산합니다
                    market_value=mark * p["quantity"] * rate,
                    unrealized_pnl=p["pnl"] * rate,
                ))
        deriv = kis_trading.deriv_balance()
        if not deriv.get("ok"):
            self.position_errors.append(f"파생 잔고: {deriv.get('error') or '조회 실패'}")
        if deriv.get("ok"):
            for p in deriv["positions"]:
                inst = instruments.parse_derivative(p["code"])
                mult = inst.multiplier if inst else 1.0
                out.append(Position(
                    key=p["code"], name=p["name"],
                    asset_class=(inst.asset_class if inst else FUTURES),
                    side=p["side"], quantity=p["quantity"], avg_price=p["avg_price"],
                    multiplier=mult, current_price=p["current_price"] or None,
                    market_value=(p["current_price"] or p["avg_price"]) * p["quantity"] * mult,
                    unrealized_pnl=p["pnl"],
                ))
        return out

    # -- 주문 -------------------------------------------------------------
    def submit(self, inst: Instrument, side: str, quantity: float,
               order_type: str = "market", price: float = None,
               note: str = "", client_order_id: str = "") -> OrderResult:
        prior = _already_sent(client_order_id)
        if prior:
            return prior

        health = self.health()
        if not health["ready"]:
            return OrderResult(ok=False, error=health["detail"],
                               client_order_id=client_order_id)

        quantity = int(inst.round_quantity(quantity))
        if quantity <= 0:
            return OrderResult(ok=False, error="주문 수량이 0입니다.",
                               client_order_id=client_order_id)

        quote = feed.quote(inst)
        if not quote:
            return OrderResult(ok=False, error="현재가를 조회할 수 없어 주문하지 않았습니다.",
                               client_order_id=client_order_id)
        ref_price = float(price) if price else quote["price"]

        if inst.is_derivative:
            # 파생은 시장가 주문이 제한적이라 지정가로 냅니다.
            # 체결 확률을 위해 매수는 한 틱 위, 매도는 한 틱 아래로 겁니다.
            tick = inst.tick_size(ref_price)
            limit = ref_price + tick if side == "buy" else ref_price - tick
            res = kis_trading.place_deriv_order(inst.key, side, quantity,
                                                price=inst.round_price(limit),
                                                order_type="limit")
        elif inst.market == MARKET_US:
            limit = round(ref_price * (1.002 if side == "buy" else 0.998), 2)
            exchange = kis_trading.overseas_exchange_code(quote.get("exchange"))
            if side == "buy" and self.cfg.get("us_order_funding", "fx") != "krw":
                blocked = _overseas_funds_block(inst, exchange, limit, quantity)
                if blocked:
                    return OrderResult(ok=False, error=blocked,
                                       client_order_id=client_order_id)
            res = kis_trading.place_overseas_order(
                inst.key, side, quantity, price=limit, exchange=exchange)
        else:
            res = kis_trading.place_stock_order(
                inst.key, side, quantity,
                price=inst.round_price(ref_price) if order_type == "limit" else 0,
                order_type=order_type)

        result = OrderResult(
            ok=bool(res.get("ok")),
            action=("open_long" if side == "buy" else "open_short") if inst.is_derivative else side,
            quantity=quantity, price=ref_price,
            price_krw=quote["price_krw"],
            # 실제 체결가·수수료는 브로커가 정합니다. 주문 '접수'까지가 이 계층의
            # 책임이고, 체결 여부는 엔진의 정산 패스가 order_status() 로 확인합니다.
            status="pending" if res.get("ok") else "rejected",
            broker_order_id=res.get("broker_order_id", ""),
            error=res.get("error", ""), message=res.get("message", ""),
            client_order_id=client_order_id, intended_price=ref_price,
            raw=res.get("raw") or {},
        )
        _remember_order(client_order_id, result)
        return result

    # -- 체결 확인 · 취소 --------------------------------------------------
    def order_status(self, record: dict) -> OrderStatus:
        broker_order_id = str(record.get("broker_order_id") or "").lstrip("0")
        if not broker_order_id:
            return OrderStatus(known=False, detail="브로커 주문번호가 없습니다.")

        is_deriv = record.get("asset_class") in (FUTURES, OPTION)
        # 해외 주문은 국내 체결내역에 없습니다. 여기서 갈라주지 않으면 미국 주문이
        # 영원히 '확인 불가'로 남아 그 종목이 미결제로 묶입니다.
        inst = instruments.try_resolve(record.get("symbol") or "")
        is_us = bool(inst and inst.market == MARKET_US)
        if is_deriv:
            res = kis_trading.deriv_executions()
        elif is_us:
            res = kis_trading.overseas_executions()
        else:
            res = kis_trading.stock_executions()
        if not res.get("ok"):
            return OrderStatus(known=False, detail=res.get("error", "조회 실패"))

        row = next((o for o in res["orders"]
                    if str(o["broker_order_id"]).lstrip("0") == broker_order_id), None)
        if row is None:
            return OrderStatus(known=False, detail="주문 내역에서 찾지 못했습니다.")

        ordered = float(record.get("quantity") or row["order_qty"] or 0)
        filled = float(row["filled_qty"] or 0)
        remaining = float(row["remain_qty"] or max(ordered - filled, 0))

        if row["cancelled"] and filled <= 0:
            status = "cancelled"
        elif filled <= 0:
            status = "pending"
        elif remaining > 0:
            status = "partial"
        else:
            status = "filled"

        return OrderStatus(
            known=True, status=status, filled_quantity=filled, remaining=remaining,
            avg_fill_price=row["avg_price"] or None,
            detail=f"체결 {filled:g}/{ordered:g}" + (" (취소)" if row["cancelled"] else ""),
        )

    def cancel(self, record: dict) -> OrderResult:
        broker_order_id = str(record.get("broker_order_id") or "")
        if not broker_order_id:
            return OrderResult(ok=False, error="브로커 주문번호가 없어 취소할 수 없습니다.")

        if record.get("asset_class") in (FUTURES, OPTION):
            res = kis_trading.cancel_deriv_order(broker_order_id)
        else:
            org_no = ""
            open_orders = kis_trading.stock_open_orders()
            if open_orders.get("ok"):
                match = next((o for o in open_orders["orders"]
                              if str(o["broker_order_id"]).lstrip("0")
                              == broker_order_id.lstrip("0")), None)
                if match is None:
                    # 미체결 목록에 없다 = 이미 체결되었거나 취소된 주문
                    return OrderResult(ok=False, status="filled",
                                       error="미체결 목록에 없습니다 (이미 처리된 주문).")
                org_no = match.get("org_no", "")
            res = kis_trading.cancel_stock_order(broker_order_id, org_no=org_no)

        return OrderResult(ok=bool(res.get("ok")), action="cancel",
                           status="cancelled" if res.get("ok") else "pending",
                           broker_order_id=broker_order_id,
                           error=res.get("error", ""), message=res.get("message", ""),
                           raw=res.get("raw") or {})

    def open_orders(self) -> list[dict]:
        res = kis_trading.stock_open_orders()
        return res.get("orders", []) if res.get("ok") else []

    def close(self, inst: Instrument, quantity: float = None,
              note: str = "", client_order_id: str = "") -> OrderResult:
        position = self.position_for(inst.key)
        if not position:
            return OrderResult(ok=False, error="청산할 포지션이 없습니다.",
                               client_order_id=client_order_id)

        qty = quantity if quantity is not None else position.quantity
        # 롱은 매도로, 숏은 매수로 청산합니다.
        side = "sell" if position.side == "long" else "buy"
        result = self.submit(inst, side, qty, order_type="market",
                             note=note, client_order_id=client_order_id)
        result.action = "close"
        return result


# ---------------------------------------------------------------------------
# 팩토리
# ---------------------------------------------------------------------------

def get_broker(user_id: int, mode: str = PAPER, cfg: dict = None) -> Broker:
    """모드에 맞는 브로커. 알 수 없는 모드는 가장 안전한 paper 로 떨어집니다."""
    if mode == LIVE:
        return KISBroker(user_id, LIVE, cfg)
    if mode == MOCK:
        return KISBroker(user_id, MOCK, cfg)
    return PaperBroker(user_id, slippage_bps=(cfg or {}).get("paper_slippage_bps", 5.0))


def available_modes() -> list[dict]:
    """콘솔에 보여줄 모드 목록 + 각 모드가 지금 쓸 수 있는지."""
    st = kis_trading.status()
    return [
        {"mode": PAPER, "label": MODE_LABELS[PAPER], "available": True,
         "detail": "키 없이 즉시 사용 가능. 실제 주문이 나가지 않습니다."},
        {"mode": MOCK, "label": MODE_LABELS[MOCK], "available": st["mock"] and st["ready"],
         "detail": ("사용 가능" if st["mock"] and st["ready"]
                    else "KIS 키 + 계좌번호 + KIS_MOCK=1 이 필요합니다.")},
        {"mode": LIVE, "label": MODE_LABELS[LIVE],
         "available": (not st["mock"]) and st["keys_configured"]
                      and st["account_configured"] and st["live_enabled"],
         "detail": ("사용 가능 — 실제 자금이 움직입니다."
                    if st["live_enabled"] and not st["mock"]
                    else "KIS 키 + 계좌번호 + KIS_LIVE_TRADING=1 (그리고 KIS_MOCK 해제) 필요")},
    ]


def make_client_order_id(user_id: int, key: str, action: str,
                         stamp: datetime = None, mode: str = PAPER) -> str:
    """같은 분 안의 같은 종목·같은 동작은 한 번만 나갑니다.

    루프가 30초 주기로 돌 때 신호가 계속 켜져 있으면 매 회전 주문이 나갑니다.
    분 단위로 id 를 묶어 그것을 막습니다(중복 진입 방지의 마지막 방어선).

    모드를 넣는 이유: 모의계좌에서 낸 주문번호가 실전 주문을 막으면 안 됩니다.
    계좌가 다르면 완전히 다른 주문입니다.
    """
    stamp = stamp or datetime.now()
    return f"{user_id}:{mode}:{key}:{action}:{stamp.strftime('%Y%m%d%H%M')}"


__all__ = ["Broker", "PaperBroker", "KISBroker", "OrderResult", "OrderStatus",
           "Position", "get_broker", "available_modes", "make_client_order_id",
           "PAPER", "MOCK", "LIVE", "MODE_LABELS"]
