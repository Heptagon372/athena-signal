# -*- coding: utf-8 -*-
"""잔고 캐시의 계좌 격리 — 한 프로세스에서 여러 사용자의 엔진이 돌 때.

배경: 사용자별 KIS 키는 ContextVar 오버레이로, 접근토큰은 앱키별 파일로
격리했는데 잔고 캐시(_balance_cache)만 함수 이름을 키로 써서, TTL 20초 안에
늦게 조회한 사용자가 앞 사람 계좌의 잔고를 그대로 받았습니다. 콘솔에 남의
계좌가 표시되는 것을 넘어 **주문 수량이 남의 현금으로 계산**되는 문제라
회귀하면 안 됩니다 (2026-08-15 실계좌에서 실제 발생).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_sources import credentials
from data_sources import kis_trading as kt

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail and not ok else ""))


current = {"acct": "11112222-01"}
calls = []
_real_get = credentials.get


def _fake_get(name, default=""):
    if name == "KIS_ACCOUNT":
        return current["acct"]
    if name == "KIS_DERIV_ACCOUNT":
        return ""
    return _real_get(name, default)


def _fetch(tag, cash):
    def fetch():
        calls.append(tag)
        return {"ok": True, "available_cash": cash}
    return fetch


credentials.get = _fake_get
try:
    kt.clear_balance_cache()

    a1 = kt._balance_cached("orderable_cash", _fetch("A", 111))
    check("계정 A 첫 조회는 fetch 를 부른다", calls == ["A"] and a1["available_cash"] == 111)

    current["acct"] = "33334444-01"
    b1 = kt._balance_cached("orderable_cash", _fetch("B", 999))
    check("계정을 바꾸면 A 의 캐시를 받지 않고 새로 조회한다",
          b1["available_cash"] == 999 and calls == ["A", "B"],
          f"받은 값 {b1['available_cash']} — 111 이면 남의 잔고로 주문 수량을 계산합니다")

    current["acct"] = "11112222-01"
    a2 = kt._balance_cached("orderable_cash", _fetch("A2", -1))
    check("되돌아오면 A 의 캐시가 TTL 안에서 그대로 산다",
          a2["available_cash"] == 111 and calls == ["A", "B"])

    current["acct"] = ""          # 계좌 미설정 (키를 못 읽은 사용자)
    n1 = kt._balance_cached("orderable_cash", _fetch("N", 5))
    check("계좌 미설정 사용자도 남의 캐시를 받지 않는다",
          n1["available_cash"] == 5 and calls == ["A", "B", "N"])

    kt.clear_balance_cache()
    current["acct"] = "11112222-01"
    a3 = kt._balance_cached("orderable_cash", _fetch("A3", 222))
    check("clear_balance_cache 는 모든 계정의 캐시를 비운다",
          a3["available_cash"] == 222 and calls[-1] == "A3")
finally:
    credentials.get = _real_get
    kt.clear_balance_cache()

print()
print(f"통과 {len(PASS)} · 실패 {len(FAIL)}")
if FAIL:
    for name in FAIL:
        print(f"  FAIL  {name}")
sys.exit(1 if FAIL else 0)
