"""
모의 파생상품 계좌 (선물·옵션 포지션)
------------------------------------
주식은 `storage/paper.py` 가 담당합니다. 선물·옵션은 회계가 달라서 분리했습니다.

무엇이 다른가
    · 매도(숏)로 **진입**할 수 있습니다. 주식처럼 "보유분을 판다"가 아닙니다.
    · 1계약이 1주가 아닙니다. 코스피200 선물 1계약의 명목금액은 지수 × 25만원이라,
      지수 350pt 면 계약당 8,750만원입니다. 승수를 빼먹으면 노출이 25만배 틀립니다.
    · 전액이 아니라 **증거금**만 묶입니다.

현금 처리 방식 (단순화 모형)
    진입: 필요 증거금을 예수금에서 차감 + 수수료 차감
    청산: 증거금 환입 + 실현손익 반영 - 수수료
    즉 **일일정산(mark-to-market)은 하지 않고 청산 시점에 한 번에 반영**합니다.
    실제 선물 계좌는 매일 정산되어 증거금이 부족하면 마진콜이 오지만, 여기서는
    평가손익을 화면에 보여주고 리스크 엔진이 손실 한도로 통제합니다.

    옵션 매도 증거금은 '수취 프리미엄 + 기초자산 명목 × 증거금률'로 보수적으로
    잡습니다(실제 거래소는 SPAN 방식). 프리미엄은 청산 시점에 손익으로 반영됩니다.

예수금은 `paper_account` 를 **공유**합니다. 한 계좌 안에서 주식과 파생을 함께
운용하는 것이 실제 증권계좌와 같은 구조이기 때문입니다.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH

LONG = "long"
SHORT = "short"


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_deriv_positions (
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            asset_class TEXT NOT NULL,        -- FUTURES | OPTION
            side TEXT NOT NULL,               -- long | short
            quantity REAL NOT NULL,           -- 계약 수
            avg_price REAL NOT NULL,          -- 평균 진입가 (지수 pt 또는 프리미엄)
            multiplier REAL NOT NULL,
            margin REAL NOT NULL DEFAULT 0,   -- 묶여 있는 증거금(원)
            opened_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, symbol, side)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_deriv_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            traded_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            asset_class TEXT,
            action TEXT NOT NULL,             -- open | close
            side TEXT NOT NULL,               -- long | short (포지션 방향)
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            notional REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            margin_delta REAL NOT NULL DEFAULT 0,
            realized_pnl REAL,
            cash_after REAL,
            note TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deriv_trades_user "
                     "ON paper_deriv_trades(user_id, traded_at)")


def _cash(conn, user_id: int) -> float:
    row = conn.execute("SELECT cash FROM paper_account WHERE user_id = ?",
                       (user_id,)).fetchone()
    return float(row["cash"]) if row else 0.0


def margin_locked(user_id: int) -> float:
    init()
    with _conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(margin), 0) m FROM paper_deriv_positions "
            "WHERE user_id = ?", (user_id,)).fetchone()
    return float(row["m"] or 0)


def get_positions(user_id: int) -> list[dict]:
    init()
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM paper_deriv_positions WHERE user_id = ? AND quantity > 0 "
            "ORDER BY symbol", (user_id,))]


def get_position(user_id: int, symbol: str, side: str = None) -> dict | None:
    init()
    with _conn() as conn:
        if side:
            row = conn.execute(
                "SELECT * FROM paper_deriv_positions "
                "WHERE user_id = ? AND symbol = ? AND side = ? AND quantity > 0",
                (user_id, symbol, side)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM paper_deriv_positions "
                "WHERE user_id = ? AND symbol = ? AND quantity > 0",
                (user_id, symbol)).fetchone()
    return dict(row) if row else None


def get_trades(user_id: int, limit: int = 100) -> list[dict]:
    init()
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM paper_deriv_trades WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit))]


# ---------------------------------------------------------------------------
# 진입 / 청산
# ---------------------------------------------------------------------------

def open_position(user_id: int, inst, side: str, quantity: float,
                  price: float, note: str = "") -> dict:
    """신규 진입 (또는 같은 방향 추가). side 는 long / short."""
    from storage import paper

    paper.ensure_account(user_id)
    init()

    if side not in (LONG, SHORT):
        return {"ok": False, "error": "포지션 방향이 올바르지 않습니다."}
    if not price or price <= 0:
        return {"ok": False, "error": "현재가를 확인할 수 없어 진입할 수 없습니다."}

    quantity = float(int(quantity))
    if quantity <= 0:
        return {"ok": False, "error": "계약 수가 0입니다."}

    order_side = "buy" if side == LONG else "sell"
    notional = inst.notional(price, quantity)
    margin = inst.margin_required(price, quantity, order_side)
    fee, _ = inst.costs(order_side, notional)

    # 반대 방향 포지션이 열려 있으면 새로 열지 않고 청산부터 해야 합니다.
    # (같은 종목 롱/숏 동시 보유는 증거금만 두 배로 묶는 무의미한 상태입니다)
    opposite = get_position(user_id, inst.key, SHORT if side == LONG else LONG)
    if opposite:
        return {"ok": False,
                "error": f"반대 방향 포지션이 열려 있습니다 "
                         f"({'매도' if side == LONG else '매수'} {opposite['quantity']:g}계약). "
                         f"먼저 청산하세요."}

    with _conn() as conn:
        cash = _cash(conn, user_id)
        need = margin + fee
        if need > cash + 1e-6:
            return {"ok": False,
                    "error": f"증거금 부족 — 필요 {need:,.0f}원 / 예수금 {cash:,.0f}원"}

        existing = conn.execute(
            "SELECT * FROM paper_deriv_positions "
            "WHERE user_id = ? AND symbol = ? AND side = ?",
            (user_id, inst.key, side)).fetchone()

        now = datetime.now().isoformat()
        if existing and float(existing["quantity"]) > 0:
            old_qty = float(existing["quantity"])
            new_qty = old_qty + quantity
            new_avg = (float(existing["avg_price"]) * old_qty + price * quantity) / new_qty
            conn.execute(
                "UPDATE paper_deriv_positions SET quantity = ?, avg_price = ?, "
                "margin = margin + ?, updated_at = ? "
                "WHERE user_id = ? AND symbol = ? AND side = ?",
                (new_qty, new_avg, margin, now, user_id, inst.key, side))
        else:
            conn.execute(
                "INSERT OR REPLACE INTO paper_deriv_positions "
                "(user_id, symbol, name, asset_class, side, quantity, avg_price, "
                " multiplier, margin, opened_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, inst.key, inst.name, inst.asset_class, side, quantity,
                 price, inst.multiplier, margin, now, now))

        new_cash = cash - need
        conn.execute("UPDATE paper_account SET cash = ? WHERE user_id = ?",
                     (new_cash, user_id))
        conn.execute(
            "INSERT INTO paper_deriv_trades (user_id, traded_at, symbol, name, "
            "asset_class, action, side, quantity, price, notional, fee, margin_delta, "
            "cash_after, note) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, now, inst.key, inst.name, inst.asset_class, side, quantity,
             price, notional, fee, margin, new_cash, note))

    return {"ok": True, "action": "open", "side": side, "quantity": quantity,
            "price": price, "notional": notional, "margin": margin, "fee": fee,
            "cash": new_cash}


def close_position(user_id: int, inst, side: str = None, quantity: float = None,
                   price: float = 0, note: str = "") -> dict:
    """청산. side 를 생략하면 열려 있는 방향을, quantity 를 생략하면 전량."""
    from storage import paper

    paper.ensure_account(user_id)
    init()

    if not price or price <= 0:
        return {"ok": False, "error": "현재가를 확인할 수 없어 청산할 수 없습니다."}

    position = get_position(user_id, inst.key, side)
    if not position:
        return {"ok": False, "error": "청산할 포지션이 없습니다."}

    side = position["side"]
    held = float(position["quantity"])
    quantity = held if quantity is None else float(int(quantity))
    if quantity <= 0:
        return {"ok": False, "error": "청산 수량이 올바르지 않습니다."}
    if quantity > held + 1e-9:
        return {"ok": False,
                "error": f"보유 계약 부족 — 요청 {quantity:g} / 보유 {held:g}"}

    entry = float(position["avg_price"])
    multiplier = float(position["multiplier"])
    notional = price * quantity * multiplier
    order_side = "sell" if side == LONG else "buy"
    fee, _ = inst.costs(order_side, notional)

    direction = 1 if side == LONG else -1
    realized = (price - entry) * multiplier * quantity * direction - fee

    # 부분청산이면 증거금도 비율만큼 돌려줍니다
    release = float(position["margin"]) * (quantity / held)

    with _conn() as conn:
        cash = _cash(conn, user_id)
        now = datetime.now().isoformat()
        remaining = held - quantity

        if remaining <= 1e-9:
            conn.execute(
                "DELETE FROM paper_deriv_positions "
                "WHERE user_id = ? AND symbol = ? AND side = ?",
                (user_id, inst.key, side))
        else:
            conn.execute(
                "UPDATE paper_deriv_positions SET quantity = ?, margin = ?, updated_at = ? "
                "WHERE user_id = ? AND symbol = ? AND side = ?",
                (remaining, float(position["margin"]) - release, now,
                 user_id, inst.key, side))

        new_cash = cash + release + realized
        conn.execute("UPDATE paper_account SET cash = ? WHERE user_id = ?",
                     (new_cash, user_id))
        conn.execute(
            "INSERT INTO paper_deriv_trades (user_id, traded_at, symbol, name, "
            "asset_class, action, side, quantity, price, notional, fee, margin_delta, "
            "realized_pnl, cash_after, note) "
            "VALUES (?, ?, ?, ?, ?, 'close', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, now, inst.key, inst.name, inst.asset_class, side, quantity,
             price, notional, fee, -release, realized, new_cash, note))

    return {"ok": True, "action": "close", "side": side, "quantity": quantity,
            "price": price, "entry_price": entry, "realized_pnl": realized,
            "fee": fee, "margin_released": release, "cash": new_cash}


def portfolio(user_id: int, price_lookup) -> dict:
    """평가 결과. price_lookup(symbol) -> 현재가(포인트/프리미엄) 또는 None."""
    rows = get_positions(user_id)
    positions, unrealized, margin_total = [], 0.0, 0.0

    for p in rows:
        current = price_lookup(p["symbol"])
        entry = float(p["avg_price"])
        qty = float(p["quantity"])
        mult = float(p["multiplier"])
        mark = current if current is not None else entry
        direction = 1 if p["side"] == LONG else -1
        pnl = (mark - entry) * mult * qty * direction
        unrealized += pnl
        margin_total += float(p["margin"])
        notional = mark * mult * qty
        positions.append({
            "symbol": p["symbol"], "name": p["name"],
            "asset_class": p["asset_class"], "side": p["side"],
            "quantity": qty, "avg_price": entry, "current_price": current,
            "multiplier": mult, "notional": notional,
            "margin": float(p["margin"]), "pnl": pnl,
            "pnl_pct": (pnl / float(p["margin"]) * 100) if p["margin"] else 0.0,
            "price_available": current is not None,
            "opened_at": p["opened_at"],
        })

    realized = sum(t["realized_pnl"] or 0 for t in get_trades(user_id, limit=100000))
    return {
        "positions": positions,
        "position_count": len(positions),
        "margin_locked": margin_total,
        "unrealized_pnl": unrealized,
        "realized_pnl": realized,
        "notional_exposure": sum(abs(p["notional"]) for p in positions),
    }


def reset(user_id: int):
    """계좌 초기화 시 파생 포지션·기록도 함께 지웁니다."""
    init()
    with _conn() as conn:
        conn.execute("DELETE FROM paper_deriv_positions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM paper_deriv_trades WHERE user_id = ?", (user_id,))
