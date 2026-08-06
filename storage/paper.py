"""
모의투자 (Paper Trading) — 사용자별 계좌
---------------------------------------
가상 자금으로 실제처럼 매매해보는 기능입니다. 실제 주문은 절대 나가지 않으며,
모든 체결은 조회 시점의 현재가로 이 DB 안에서만 이뤄집니다.

왜 필요한가
    적중률(%)만으로는 "그래서 돈을 벌었나"를 알 수 없습니다. 방향을 60% 맞춰도
    틀릴 때 크게 잃으면 손실입니다. 모의투자는 신호를 실제로 따라갔을 때의
    손익을 수수료·세금까지 포함해 보여줍니다.

계정 분리
    모든 테이블이 user_id 로 나뉘어, 계정마다 독립된 계좌를 갖습니다.

통화
    계좌는 원화 단일 통화입니다. 미국 종목은 상위 계층(api.py)이 실시간 환율로
    원화 환산한 가격을 넘겨주므로, 이 모듈은 통화를 신경 쓰지 않습니다.

주문 유형
    시장가 — 조회 시점의 현재가로 즉시 체결됩니다.
    지정가 — 미체결 주문으로 남아 있다가, 가격이 조건에 닿으면 체결됩니다.
             매수는 현재가 <= 지정가, 매도는 현재가 >= 지정가 일 때.

예수금·수량 구속 (증거금)
    지정가 주문을 걸어두면 그만큼 **예수금이 묶입니다**. 묶지 않으면 100만원으로
    100만원짜리 지정가 매수를 열 번 걸 수 있고, 그게 다 체결되면 잔고가 음수가 됩니다.
    매도도 같은 이유로 보유 수량을 묶습니다.
        주문가능 현금 = cash - reserved_cash
        주문가능 수량 = quantity - reserved_qty

비용 모형 (국내 기준 근사)
    매수: 위탁수수료 0.015%
    매도: 위탁수수료 0.015% + 증권거래세 0.18%
    미국: 수수료 0.25% (환전·거래 비용 근사), 거래세 없음
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH

INITIAL_CASH = 10_000_000          # 초기 가상 자금 1천만원

FEE_RATE_KR = 0.00015              # 위탁수수료 0.015%
TAX_RATE_KR = 0.0018               # 증권거래세 0.18% (매도 시)
FEE_RATE_US = 0.0025               # 미국 0.25% (환전 포함 근사)

# 국내 주식 호가 단위 (2023-01 개편 기준). (상한가 미만, 단위)
# 지정가를 아무 숫자나 받으면 실제로는 낼 수 없는 주문이 됩니다.
KR_TICK_TABLE = [
    (2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50),
    (200_000, 100), (500_000, 500), (float("inf"), 1_000),
]


def tick_size(price: float, market: str) -> float:
    """해당 가격대의 호가 단위."""
    if market == "US":
        return 0.01
    for upper, unit in KR_TICK_TABLE:
        if price < upper:
            return unit
    return 1_000


def snap_to_tick(price: float, market: str) -> float:
    """지정가를 호가 단위에 맞춰 내림 정렬."""
    if price is None or price <= 0:
        return price
    unit = tick_size(price, market)
    snapped = round(price / unit) * unit
    return round(snapped, 2) if market == "US" else float(int(snapped))


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
        CREATE TABLE IF NOT EXISTS paper_account (
            user_id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            initial_cash REAL NOT NULL,
            cash REAL NOT NULL,
            auto_trade INTEGER NOT NULL DEFAULT 0
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_holdings (
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            name TEXT, market TEXT, currency TEXT,
            quantity REAL NOT NULL,
            avg_price REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, ticker)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            traded_at TEXT NOT NULL,
            ticker TEXT NOT NULL, name TEXT, market TEXT, currency TEXT,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            fee REAL NOT NULL DEFAULT 0,
            tax REAL NOT NULL DEFAULT 0,
            realized_pnl REAL,
            cash_after REAL,
            note TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            ticker TEXT NOT NULL, name TEXT, market TEXT, currency TEXT,
            side TEXT NOT NULL,                -- buy | sell
            order_type TEXT NOT NULL,          -- market | limit
            limit_price REAL,                  -- 지정가 (해당 종목의 호가 통화 기준)
            quantity REAL NOT NULL,
            reserved_cash REAL NOT NULL DEFAULT 0,   -- 매수 시 묶어둔 원화
            status TEXT NOT NULL,              -- pending | filled | cancelled
            filled_at TEXT, filled_price REAL, trade_id INTEGER,
            message TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_user "
                     "ON paper_orders(user_id, status, id)")
        _migrate(conn)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_user "
                     "ON paper_trades(user_id, traded_at)")


def _migrate(conn):
    """로그인 도입 이전 스키마(단일 계좌)를 사용자별 스키마로 올립니다.

    `CREATE TABLE IF NOT EXISTS` 는 이미 있는 테이블을 바꾸지 않으므로,
    기존 설치에는 user_id 가 없고 `paper_account` 에는 `CHECK (id = 1)` 제약이
    남아 있습니다. SQLite 는 제약을 떼어낼 수 없어서, 옛 스키마가 감지되면
    테이블을 새로 만들고 데이터를 옮깁니다.

    기존 데이터는 '이전 기록' 계정으로 귀속시켜 잃지 않게 합니다.
    """
    from storage import users

    def columns(table):
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}

    acc_cols = columns("paper_account")
    # 옛 스키마의 표식은 `id` 컬럼입니다. user_id 만 보고 판단하면, 중단된
    # 마이그레이션이 user_id 를 이미 붙여둔 경우 CHECK 제약이 남은 채 통과합니다.
    if acc_cols and "id" in acc_cols:
        legacy_id = users.legacy_user_id()

        # 1) 단일 계좌 -> 사용자별 계좌
        old = conn.execute("SELECT * FROM paper_account LIMIT 1").fetchone()
        conn.execute("ALTER TABLE paper_account RENAME TO paper_account_old")
        conn.execute("""
        CREATE TABLE paper_account (
            user_id INTEGER PRIMARY KEY,
            created_at TEXT NOT NULL,
            initial_cash REAL NOT NULL,
            cash REAL NOT NULL,
            auto_trade INTEGER NOT NULL DEFAULT 0
        )""")
        if old:
            keys = old.keys()
            owner = old["user_id"] if "user_id" in keys and old["user_id"] else legacy_id
            conn.execute(
                "INSERT INTO paper_account (user_id, created_at, initial_cash, cash, auto_trade) "
                "VALUES (?, ?, ?, ?, ?)",
                (owner, old["created_at"], old["initial_cash"], old["cash"],
                 old["auto_trade"] if "auto_trade" in keys else 0))
        conn.execute("DROP TABLE paper_account_old")

        # 2) 보유·거래는 컬럼만 덧붙이면 됩니다
        for table in ("paper_holdings", "paper_trades"):
            cols = columns(table)
            if cols and "user_id" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")
                conn.execute(f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL",
                             (legacy_id,))
        print("[paper] 단일 계좌 스키마를 사용자별 계좌로 마이그레이션했습니다.")

    _migrate_holdings_pk(conn)


def _migrate_holdings_pk(conn):
    """`paper_holdings` 의 기본키를 (user_id, ticker) 로 바로잡습니다.

    옛 스키마는 `ticker TEXT PRIMARY KEY` 였습니다. user_id 컬럼만 덧붙이면
    컬럼은 생기지만 **기본키는 여전히 ticker 단독**이라, 서로 다른 사용자가
    같은 종목을 보유하는 순간 UNIQUE 제약에 걸려 매수가 실패합니다.
    (자동매매처럼 여러 계정이 같은 종목을 다루면 바로 터집니다)

    SQLite 는 기본키를 바꿀 수 없어 테이블을 새로 만들고 데이터를 옮깁니다.
    """
    info = list(conn.execute("PRAGMA table_info(paper_holdings)"))
    if not info:
        return
    pk_columns = {r["name"] for r in info if r["pk"]}
    if pk_columns == {"user_id", "ticker"}:
        return          # 이미 올바른 스키마

    from storage import users
    legacy_id = users.legacy_user_id()

    conn.execute("ALTER TABLE paper_holdings RENAME TO paper_holdings_old")
    conn.execute("""
    CREATE TABLE paper_holdings (
        user_id INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        name TEXT, market TEXT, currency TEXT,
        quantity REAL NOT NULL,
        avg_price REAL NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, ticker)
    )""")
    conn.execute(
        "INSERT INTO paper_holdings (user_id, ticker, name, market, currency, "
        "quantity, avg_price, updated_at) "
        "SELECT COALESCE(user_id, ?), ticker, name, market, currency, "
        "quantity, avg_price, updated_at FROM paper_holdings_old",
        (legacy_id,))
    conn.execute("DROP TABLE paper_holdings_old")
    print("[paper] paper_holdings 기본키를 (user_id, ticker) 로 재구성했습니다.")


def ensure_account(user_id: int):
    """계정마다 계좌를 처음 쓸 때 자동 개설합니다."""
    init()
    with _conn() as conn:
        row = conn.execute("SELECT user_id FROM paper_account WHERE user_id = ?",
                           (user_id,)).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO paper_account (user_id, created_at, initial_cash, cash, auto_trade) "
                "VALUES (?, ?, ?, ?, 0)",
                (user_id, datetime.now().isoformat(), INITIAL_CASH, INITIAL_CASH))


def costs(market: str, side: str, amount: float,
          asset_class: str = None) -> tuple[float, float]:
    """(수수료, 세금) — 매도에만 거래세가 붙습니다.

    ETF 는 매도해도 증권거래세가 면제됩니다. 이걸 반영하지 않으면 ETF 회전
    전략의 비용이 실제보다 0.18%p 크게 잡혀, 되는 전략도 안 되는 것으로 보입니다.
    """
    if market == "US":
        return round(amount * FEE_RATE_US, 2), 0.0
    fee = round(amount * FEE_RATE_KR, 2)
    taxable = side == "sell" and asset_class != "ETF"
    tax = round(amount * TAX_RATE_KR, 2) if taxable else 0.0
    return fee, tax


def _reserved_cash(conn, user_id: int) -> float:
    """미체결 매수 주문에 묶인 원화 합계."""
    row = conn.execute(
        "SELECT COALESCE(SUM(reserved_cash), 0) r FROM paper_orders "
        "WHERE user_id = ? AND status = 'pending' AND side = 'buy'",
        (user_id,)).fetchone()
    return float(row["r"] or 0)


def _reserved_qty(conn, user_id: int, ticker: str) -> float:
    """미체결 매도 주문에 묶인 수량."""
    row = conn.execute(
        "SELECT COALESCE(SUM(quantity), 0) q FROM paper_orders "
        "WHERE user_id = ? AND status = 'pending' AND side = 'sell' AND ticker = ?",
        (user_id, ticker)).fetchone()
    return float(row["q"] or 0)


def available_cash(user_id: int) -> float:
    """주문에 쓸 수 있는 현금 (묶인 금액 제외)."""
    ensure_account(user_id)
    with _conn() as conn:
        acc = conn.execute("SELECT cash FROM paper_account WHERE user_id = ?",
                           (user_id,)).fetchone()
        return float(acc["cash"]) - _reserved_cash(conn, user_id)


def available_quantity(user_id: int, ticker: str) -> float:
    """매도할 수 있는 수량 (미체결 매도에 묶인 것 제외)."""
    ensure_account(user_id)
    with _conn() as conn:
        held = conn.execute(
            "SELECT quantity FROM paper_holdings WHERE user_id = ? AND ticker = ?",
            (user_id, ticker)).fetchone()
        if not held:
            return 0.0
        return max(float(held["quantity"]) - _reserved_qty(conn, user_id, ticker), 0.0)


def get_account(user_id: int) -> dict:
    ensure_account(user_id)
    with _conn() as conn:
        acc = conn.execute("SELECT * FROM paper_account WHERE user_id = ?",
                           (user_id,)).fetchone()
        reserved = _reserved_cash(conn, user_id)
    return {
        "created_at": acc["created_at"],
        "initial_cash": acc["initial_cash"],
        "cash": acc["cash"],
        "reserved_cash": reserved,
        "available_cash": float(acc["cash"]) - reserved,
        "auto_trade": bool(acc["auto_trade"]),
    }


def set_auto_trade(user_id: int, enabled: bool):
    ensure_account(user_id)
    with _conn() as conn:
        conn.execute("UPDATE paper_account SET auto_trade = ? WHERE user_id = ?",
                     (1 if enabled else 0, user_id))


def get_holdings(user_id: int) -> list[dict]:
    ensure_account(user_id)
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM paper_holdings WHERE user_id = ? AND quantity > 0 "
            "ORDER BY ticker", (user_id,))]


def get_trades(user_id: int, limit: int = 100) -> list[dict]:
    ensure_account(user_id)
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM paper_trades WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit))]


def buy(user_id: int, symbol, price: float, quantity: float = None,
        amount: float = None, note: str = "") -> dict:
    """매수. quantity(주) 또는 amount(금액) 중 하나를 지정합니다.

    금액을 주면 수수료를 감안해 살 수 있는 최대 수량을 계산합니다.
    국내 종목은 소수점 매매가 안 되므로 정수 주로 내림합니다.
    """
    ensure_account(user_id)
    if price is None or price <= 0:
        return {"ok": False, "error": "현재가를 확인할 수 없어 매수할 수 없습니다."}

    is_kr = symbol.market != "US"
    with _conn() as conn:
        acc = conn.execute("SELECT * FROM paper_account WHERE user_id = ?",
                           (user_id,)).fetchone()
        cash = float(acc["cash"])
        # 미체결 지정가 매수에 묶인 돈은 쓸 수 없습니다.
        avail = cash - _reserved_cash(conn, user_id)

        if quantity is None:
            budget = min(float(amount or 0), avail)
            if budget <= 0:
                return {"ok": False, "error": "매수 금액을 입력해 주세요."}
            fee_rate = FEE_RATE_KR if is_kr else FEE_RATE_US
            raw_qty = budget / (price * (1 + fee_rate))
            quantity = float(int(raw_qty)) if is_kr else round(raw_qty, 4)

        if quantity <= 0:
            return {"ok": False, "error": "잔액으로 살 수 있는 수량이 없습니다."}

        gross = price * quantity
        fee, tax = costs(symbol.market, "buy", gross, getattr(symbol, "asset_class", None))
        total = gross + fee + tax
        if total > avail + 1e-6:
            reserved = cash - avail
            extra = f" (미체결 주문에 {reserved:,.0f}원 묶임)" if reserved > 0 else ""
            return {"ok": False,
                    "error": f"주문가능 금액 부족 — 필요 {total:,.0f}원 / "
                             f"가능 {avail:,.0f}원{extra}"}

        held = conn.execute(
            "SELECT * FROM paper_holdings WHERE user_id = ? AND ticker = ?",
            (user_id, symbol.key)).fetchone()

        if held:
            new_qty = float(held["quantity"]) + quantity
            # 평균단가는 수수료까지 포함한 실제 취득원가 기준
            new_avg = ((float(held["avg_price"]) * float(held["quantity"]))
                       + gross + fee) / new_qty
            conn.execute(
                "UPDATE paper_holdings SET quantity = ?, avg_price = ?, updated_at = ? "
                "WHERE user_id = ? AND ticker = ?",
                (new_qty, new_avg, datetime.now().isoformat(), user_id, symbol.key))
        else:
            conn.execute(
                "INSERT INTO paper_holdings (user_id, ticker, name, market, currency, "
                "quantity, avg_price, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, symbol.key, symbol.name, symbol.market, symbol.currency,
                 quantity, (gross + fee) / quantity, datetime.now().isoformat()))

        new_cash = cash - total
        conn.execute("UPDATE paper_account SET cash = ? WHERE user_id = ?",
                     (new_cash, user_id))
        conn.execute(
            "INSERT INTO paper_trades (user_id, traded_at, ticker, name, market, currency, "
            "side, quantity, price, amount, fee, tax, cash_after, note) "
            "VALUES (?, ?, ?, ?, ?, ?, 'buy', ?, ?, ?, ?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(), symbol.key, symbol.name, symbol.market,
             symbol.currency, quantity, price, gross, fee, tax, new_cash, note))

    return {"ok": True, "side": "buy", "quantity": quantity, "price": price,
            "amount": gross, "fee": fee, "tax": tax, "cash": new_cash}


def sell(user_id: int, symbol, price: float, quantity: float = None,
         note: str = "") -> dict:
    """매도. quantity 를 생략하면 전량 매도."""
    ensure_account(user_id)
    if price is None or price <= 0:
        return {"ok": False, "error": "현재가를 확인할 수 없어 매도할 수 없습니다."}

    with _conn() as conn:
        held = conn.execute(
            "SELECT * FROM paper_holdings WHERE user_id = ? AND ticker = ?",
            (user_id, symbol.key)).fetchone()
        if not held or float(held["quantity"]) <= 0:
            return {"ok": False, "error": "보유 수량이 없습니다."}

        # 미체결 지정가 매도에 묶인 수량은 다시 팔 수 없습니다.
        holding_qty = float(held["quantity"]) - _reserved_qty(conn, user_id, symbol.key)
        if holding_qty <= 0:
            return {"ok": False,
                    "error": "주문가능 수량이 없습니다 (전량이 미체결 매도 주문에 묶여 있습니다)."}

        if quantity is None:
            quantity = holding_qty          # 수량을 비우면 주문가능 전량
        else:
            quantity = float(quantity)
            if quantity <= 0:
                return {"ok": False, "error": "매도 수량이 올바르지 않습니다."}
            # 요청보다 적게 파는 것은 주문을 조용히 바꾸는 것입니다.
            # 5주를 팔라고 했는데 3주만 나가면 그건 다른 주문입니다.
            if quantity > holding_qty + 1e-9:
                owned = float(held["quantity"])
                locked = owned - holding_qty
                extra = f" (보유 {owned:g}주 중 {locked:g}주가 미체결 주문에 묶임)" if locked > 0 else ""
                return {"ok": False,
                        "error": f"주문가능 수량 부족 — 요청 {quantity:g}주 / "
                                 f"가능 {holding_qty:g}주{extra}"}
        if quantity <= 0:
            return {"ok": False, "error": "매도할 수량이 없습니다."}

        gross = price * quantity
        fee, tax = costs(symbol.market, "sell", gross, getattr(symbol, "asset_class", None))
        proceeds = gross - fee - tax
        cost_basis = float(held["avg_price"]) * quantity
        realized = proceeds - cost_basis

        remaining = holding_qty - quantity
        if remaining <= 1e-9:
            conn.execute("DELETE FROM paper_holdings WHERE user_id = ? AND ticker = ?",
                         (user_id, symbol.key))
        else:
            conn.execute(
                "UPDATE paper_holdings SET quantity = ?, updated_at = ? "
                "WHERE user_id = ? AND ticker = ?",
                (remaining, datetime.now().isoformat(), user_id, symbol.key))

        acc = conn.execute("SELECT cash FROM paper_account WHERE user_id = ?",
                           (user_id,)).fetchone()
        new_cash = float(acc["cash"]) + proceeds
        conn.execute("UPDATE paper_account SET cash = ? WHERE user_id = ?",
                     (new_cash, user_id))
        conn.execute(
            "INSERT INTO paper_trades (user_id, traded_at, ticker, name, market, currency, "
            "side, quantity, price, amount, fee, tax, realized_pnl, cash_after, note) "
            "VALUES (?, ?, ?, ?, ?, ?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(), symbol.key, symbol.name, symbol.market,
             symbol.currency, quantity, price, gross, fee, tax, realized, new_cash, note))

    return {"ok": True, "side": "sell", "quantity": quantity, "price": price,
            "amount": gross, "fee": fee, "tax": tax,
            "realized_pnl": realized, "cash": new_cash}


# ---------------------------------------------------------------------------
# 지정가 주문 (미체결 관리)
# ---------------------------------------------------------------------------

def place_limit_order(user_id: int, symbol, side: str, quantity: float,
                      limit_price: float, krw_rate: float = 1.0) -> dict:
    """지정가 주문 접수. 체결이 아니라 **대기열 등록**입니다.

    krw_rate — 해당 종목 호가 통화 1단위의 원화 가치. 국내는 1.0,
               미국은 환율입니다. 묶어둘 예수금을 원화로 계산하는 데 씁니다.
    """
    ensure_account(user_id)
    if side not in ("buy", "sell"):
        return {"ok": False, "error": "주문 구분이 올바르지 않습니다."}
    if not limit_price or limit_price <= 0:
        return {"ok": False, "error": "지정가를 입력해 주세요."}

    is_kr = symbol.market != "US"
    quantity = float(int(quantity)) if is_kr else round(float(quantity), 4)
    if quantity <= 0:
        return {"ok": False, "error": "주문 수량을 입력해 주세요."}

    limit_price = snap_to_tick(float(limit_price), symbol.market)

    with _conn() as conn:
        if side == "buy":
            gross_krw = limit_price * krw_rate * quantity
            fee, _ = costs(symbol.market, "buy", gross_krw,
                           getattr(symbol, "asset_class", None))
            need = gross_krw + fee
            acc = conn.execute("SELECT cash FROM paper_account WHERE user_id = ?",
                               (user_id,)).fetchone()
            avail = float(acc["cash"]) - _reserved_cash(conn, user_id)
            if need > avail + 1e-6:
                return {"ok": False,
                        "error": f"주문가능 금액 부족 — 필요 {need:,.0f}원 / 가능 {avail:,.0f}원"}
            reserved = need
        else:
            held = conn.execute(
                "SELECT quantity FROM paper_holdings WHERE user_id = ? AND ticker = ?",
                (user_id, symbol.key)).fetchone()
            owned = float(held["quantity"]) if held else 0.0
            free = owned - _reserved_qty(conn, user_id, symbol.key)
            if quantity > free + 1e-9:
                return {"ok": False,
                        "error": f"주문가능 수량 부족 — 요청 {quantity:g}주 / 가능 {free:g}주"}
            reserved = 0.0

        cur = conn.execute(
            "INSERT INTO paper_orders (user_id, created_at, ticker, name, market, currency, "
            "side, order_type, limit_price, quantity, reserved_cash, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'limit', ?, ?, ?, 'pending')",
            (user_id, datetime.now().isoformat(), symbol.key, symbol.name, symbol.market,
             symbol.currency, side, limit_price, quantity, reserved))
        order_id = cur.lastrowid

    return {"ok": True, "order_id": order_id, "side": side, "order_type": "limit",
            "quantity": quantity, "limit_price": limit_price,
            "reserved_cash": reserved, "status": "pending"}


def get_orders(user_id: int, status: str = None, limit: int = 100) -> list[dict]:
    ensure_account(user_id)
    sql = "SELECT * FROM paper_orders WHERE user_id = ?"
    params = [user_id]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def cancel_order(user_id: int, order_id: int) -> dict:
    """미체결 주문 취소 — 묶여 있던 예수금/수량이 즉시 풀립니다."""
    ensure_account(user_id)
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM paper_orders WHERE id = ? AND user_id = ?",
            (order_id, user_id)).fetchone()
        if not row:
            return {"ok": False, "error": "주문을 찾을 수 없습니다."}
        if row["status"] != "pending":
            return {"ok": False, "error": f"이미 {row['status']} 상태인 주문입니다."}
        # reserved_cash 를 0 으로 지워야 취소 후에도 돈이 묶여 있는 일이 없습니다.
        conn.execute(
            "UPDATE paper_orders SET status = 'cancelled', reserved_cash = 0, "
            "message = ? WHERE id = ?", ("사용자 취소", order_id))
    return {"ok": True, "order_id": order_id}


def order_fills_at(order: dict, price: float) -> bool:
    """이 가격이면 체결되는가.

    매수는 지정가 이하로 내려와야, 매도는 지정가 이상으로 올라가야 체결됩니다.
    """
    if price is None or price <= 0 or not order.get("limit_price"):
        return False
    if order["side"] == "buy":
        return price <= float(order["limit_price"])
    return price >= float(order["limit_price"])


def fill_order(user_id: int, order_id: int, symbol, exec_price_krw: float) -> dict:
    """대기 중인 지정가 주문을 체결 처리합니다.

    구속을 먼저 푼 뒤(status 변경) 실제 매매를 돌립니다. 매매가 실패하면
    다시 pending 으로 되돌려, 돈만 풀리고 주문은 사라지는 상태를 막습니다.
    """
    ensure_account(user_id)
    with _conn() as conn:
        row = conn.execute("SELECT * FROM paper_orders WHERE id = ? AND user_id = ?",
                           (order_id, user_id)).fetchone()
        if not row or row["status"] != "pending":
            return {"ok": False, "error": "체결할 수 있는 주문이 아닙니다."}
        order = dict(row)
        conn.execute("UPDATE paper_orders SET status = 'filling' WHERE id = ?", (order_id,))

    note = f"지정가 {order['limit_price']:g} 체결"
    if order["side"] == "buy":
        result = buy(user_id, symbol, exec_price_krw,
                     quantity=float(order["quantity"]), note=note)
    else:
        result = sell(user_id, symbol, exec_price_krw,
                      quantity=float(order["quantity"]), note=note)

    with _conn() as conn:
        if result.get("ok"):
            conn.execute(
                "UPDATE paper_orders SET status = 'filled', filled_at = ?, "
                "filled_price = ?, reserved_cash = 0, message = ? WHERE id = ?",
                (datetime.now().isoformat(), exec_price_krw, note, order_id))
        else:
            # 되돌립니다. 구속도 그대로 유지되어야 다음 시도에서 잔고가 맞습니다.
            conn.execute("UPDATE paper_orders SET status = 'pending', message = ? WHERE id = ?",
                         (result.get("error", "체결 실패"), order_id))
    return {**result, "order_id": order_id}


def reset(user_id: int, initial_cash: float = None) -> dict:
    """계좌 초기화 — 이 사용자의 보유·거래내역만 지웁니다."""
    from storage import derivatives

    ensure_account(user_id)
    cash = float(initial_cash or INITIAL_CASH)
    # 파생 포지션을 남기면 증거금이 묶인 채로 초기화되어 잔고가 맞지 않습니다.
    derivatives.reset(user_id)
    with _conn() as conn:
        conn.execute("DELETE FROM paper_holdings WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM paper_trades WHERE user_id = ?", (user_id,))
        # 미체결 주문을 남기면 초기화 직후부터 예수금이 묶여 있게 됩니다.
        conn.execute("DELETE FROM paper_orders WHERE user_id = ?", (user_id,))
        conn.execute(
            "UPDATE paper_account SET created_at = ?, initial_cash = ?, cash = ? "
            "WHERE user_id = ?",
            (datetime.now().isoformat(), cash, cash, user_id))
    return {"ok": True, "cash": cash}


def portfolio(user_id: int, price_lookup) -> dict:
    """현재 평가액 기준 포트폴리오.

    price_lookup(ticker) -> 현재가(원화, float) 또는 None 을 넘겨받아 평가합니다.
    (시세 조회 책임을 상위 계층에 두어 이 모듈이 네트워크를 몰라도 되게 했습니다)
    """
    acc = get_account(user_id)
    rows = get_holdings(user_id)

    with _conn() as conn:
        locked = {r["ticker"]: float(r["q"]) for r in conn.execute(
            "SELECT ticker, SUM(quantity) q FROM paper_orders "
            "WHERE user_id = ? AND status = 'pending' AND side = 'sell' GROUP BY ticker",
            (user_id,))}

    positions, equity = [], 0.0
    for h in rows:
        current = price_lookup(h["ticker"])
        qty = float(h["quantity"])
        avg = float(h["avg_price"])
        cost = avg * qty
        value = (current if current is not None else avg) * qty
        equity += value
        pnl = value - cost
        reserved = locked.get(h["ticker"], 0.0)
        positions.append({
            "ticker": h["ticker"], "name": h["name"], "market": h["market"],
            "currency": h["currency"], "quantity": qty, "avg_price": avg,
            "current_price": current, "cost": cost, "value": value,
            "pnl": pnl, "pnl_pct": (pnl / cost * 100) if cost else 0.0,
            "price_available": current is not None,
            "reserved_quantity": reserved,
            "available_quantity": max(qty - reserved, 0.0),
        })

    positions.sort(key=lambda p: -abs(p["value"]))
    total = acc["cash"] + equity
    initial = acc["initial_cash"]
    realized = sum(t["realized_pnl"] or 0 for t in get_trades(user_id, limit=100000)
                   if t["side"] == "sell")

    return {
        "cash": acc["cash"],
        "reserved_cash": acc["reserved_cash"],
        "available_cash": acc["available_cash"],
        "initial_cash": initial,
        "equity": equity,
        "total_value": total,
        "total_pnl": total - initial,
        "total_pnl_pct": ((total - initial) / initial * 100) if initial else 0.0,
        "unrealized_pnl": sum(p["pnl"] for p in positions),
        "realized_pnl": realized,
        "positions": positions,
        "position_count": len(positions),
        "auto_trade": acc["auto_trade"],
        "created_at": acc["created_at"],
    }
