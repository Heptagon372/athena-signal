# -*- coding: utf-8 -*-
"""상태 확인이 안 되는 주문의 강제 종료 규칙 (engine/autotrade._force_close_stuck).

    python tests/test_stuck_orders.py

격리된 임시 DB에서만 돕니다 — 실계좌 athena.db 를 건드리지 않습니다.
"""
import os, sys, sqlite3, tempfile
from datetime import datetime, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
import config
config.DB_PATH = os.path.join(tempfile.mkdtemp(prefix="athena_stuck_"), "qa.db")
from engine import autotrade as at
from engine.broker import OrderResult, OrderStatus
from storage import autotrade as store

U, MODE = 999_902, "live"
CFG = dict(store.DEFAULT_CONFIG, mode=MODE, order_timeout_sec=180)
FAIL = []

def ok(name, cond, detail=""):
    print(f"[{'PASS' if cond else '**FAIL**'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond: FAIL.append(name)

class Fake:
    def __init__(self, status, cancel):
        self.status, self.cancel_res, self.cancels = status, cancel, []
    def order_status(self, rec): return self.status
    def cancel(self, rec):
        self.cancels.append(rec["id"]); return self.cancel_res
    def account(self): return {}

def mk(sym, age_min):
    oid = store.record_order(U, {"client_order_id": f"{U}:{MODE}:{sym}",
        "broker_mode": MODE, "broker_order_id": "X1", "symbol": sym, "name": sym,
        "asset_class": "STOCK", "action": "buy", "side": "long", "quantity": 1,
        "price": 10.0, "price_krw": 14000.0, "status": "pending"})
    with sqlite3.connect(config.DB_PATH) as c:
        c.execute("UPDATE at_orders SET created_at=? WHERE id=?",
                  ((datetime.now() - timedelta(minutes=age_min)).isoformat(), oid))
    return oid

def run(brk):
    res = {"errors": [], "settled": [], "reconciled": [], "signals": [],
           "entries": [], "exits": [], "rejects": []}
    at._settle_open_orders(U, CFG, brk, res)
    return res

def status_of(oid):
    with sqlite3.connect(config.DB_PATH) as c:
        return c.execute("SELECT status, reason FROM at_orders WHERE id=?", (oid,)).fetchone()

UNKNOWN = OrderStatus(known=False, searched=True, detail="주문 내역에서 찾지 못했습니다.")

# 1) 99분 — 아직 건드리지 않는다
a = mk("AAA", 99)
run(Fake(UNKNOWN, OrderResult(ok=False, error="없음")))
ok("99분은 그대로 pending", status_of(a)[0] == "pending", status_of(a)[0])

# 2) 101분 + 취소 성공 → cancelled
b = mk("BBB", 101)
f = Fake(UNKNOWN, OrderResult(ok=True))
run(f)
st = status_of(b)
ok("101분 + 취소 성공 → cancelled", st[0] == "cancelled" and b in f.cancels, f"{st}")

# 3) 101분 + 취소 거부 → lost
c_ = mk("CCC", 101)
f2 = Fake(UNKNOWN, OrderResult(ok=False, error="주문내역이 존재하지 않습니다"))
run(f2)
st = status_of(c_)
ok("101분 + 취소 거부 → lost", st[0] == "lost", f"{st}")

# 4) has_open_order 가 풀린다
ok("강제 종료 후 신규 주문 차단 해제", not store.has_open_order(U, "CCC", MODE))

# 5) 조회가 되는 미체결 주문은 강제 종료 대상이 아니다
d = mk("DDD", 500)
f3 = Fake(OrderStatus(known=True, status="pending", filled_quantity=0, remaining=1),
          OrderResult(ok=True))
run(f3)
ok("조회되는 미체결은 강제 종료 안 함", status_of(d)[0] in ("pending", "cancel_requested"),
   status_of(d)[0])

# 6) order_stuck_min=0 이면 예전 동작(24시간)
e = mk("EEE", 300)
CFG["order_stuck_min"] = 0
run(Fake(UNKNOWN, OrderResult(ok=True)))
ok("order_stuck_min=0 이면 강제 종료 안 함", status_of(e)[0] == "pending", status_of(e)[0])

print("\n결과:", "전부 통과" if not FAIL else f"{len(FAIL)}건 실패 — {FAIL}")
sys.exit(1 if FAIL else 0)
