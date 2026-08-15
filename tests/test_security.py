# -*- coding: utf-8 -*-
"""공개 배포 보안 QA
-------------------
    python tests/test_security.py

네트워크도 MongoDB 도 필요 없습니다. 이 파일이 지키는 것:

    · 레이트 리밋   한도 안은 통과, 초과는 기다릴 초를 돌려주는가.
                   창이 지나면 다시 통과하는가. over_limit 은 기록하지 않는가
                   (성공한 로그인이 실패 카운터를 채우면 안 됩니다)
    · 클라이언트 IP X-Forwarded-For 를 **프록시 뒤에서만** 믿는가 — 아니면
                   헤더 위조로 레이트 리밋이 우회됩니다
    · 공개 모드     ATHENA_PUBLIC_ORIGIN 설정 시 로컬 계정도 KIS 주문에
                   자기 키가 필수가 되는가. Mongo 가 죽어도 **차단 쪽으로**
                   실패하는가 (fail-closed)

설계 근거는 security.py 모듈 docstring 과 DEPLOY.md 에 있습니다.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


from data_sources import credentials
import security

# ---------------------------------------------------------------------------
# 준비: credentials 를 가로채 스위치를 마음대로 켜고 끕니다
# ---------------------------------------------------------------------------

_real_get = credentials.get
_server = {"ATHENA_PUBLIC_ORIGIN": "", "ATHENA_BEHIND_PROXY": ""}


def _patched_get(name, default=""):
    over = credentials._from_overlay(name)
    if over is not None:
        return over
    if name in _server:
        return _server[name]
    return _real_get(name, default)


credentials.get = _patched_get


class FakeClient:
    host = "203.0.113.9"


class FakeRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.client = FakeClient()


try:
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  1. 레이트 리밋")
    print("=" * 70)
    security.reset()

    allowed = [security.throttle("t:a", 3, 60) for _ in range(3)]
    check("한도 안 3회는 전부 통과", all(r == 0 for r in allowed), str(allowed))
    retry = security.throttle("t:a", 3, 60)
    check("4회째는 막히고 기다릴 초(>=1)를 준다", retry >= 1, f"retry={retry}")
    check("다른 키는 영향 없음", security.throttle("t:b", 3, 60) == 0)

    security.reset()
    security.throttle("t:w", 2, 0.2)
    security.throttle("t:w", 2, 0.2)
    check("창 안에서는 막힘", security.throttle("t:w", 2, 0.2) >= 1)
    time.sleep(0.25)
    check("창이 지나면 다시 통과", security.throttle("t:w", 2, 0.2) == 0)

    security.reset()
    for _ in range(5):
        security.over_limit("t:o", 3, 60)
    check("over_limit 은 기록하지 않는다 (성공 로그인 ≠ 실패)",
          security.over_limit("t:o", 3, 60) == 0)
    for _ in range(3):
        security.record("t:o")
    check("record 로 채우면 over_limit 이 막는다",
          security.over_limit("t:o", 3, 60) >= 1)

    # 막힌 시도는 기록되지 않아 차단이 스스로 연장되지 않는다
    security.reset()
    security.throttle("t:x", 1, 0.2)
    for _ in range(10):
        security.throttle("t:x", 1, 0.2)      # 전부 막히는 시도
    time.sleep(0.25)
    check("막힌 시도가 차단을 연장하지 않는다",
          security.throttle("t:x", 1, 0.2) == 0)

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  2. 클라이언트 IP — 위조 방어")
    print("=" * 70)

    spoofed = FakeRequest({"x-forwarded-for": "1.2.3.4"})

    _server["ATHENA_BEHIND_PROXY"] = ""
    check("프록시 없음: X-Forwarded-For 를 무시하고 실제 접속 IP",
          security.client_ip(spoofed) == "203.0.113.9",
          "헤더를 믿으면 위조로 레이트 리밋이 우회됩니다")

    _server["ATHENA_BEHIND_PROXY"] = "1"
    check("프록시 뒤: nginx 가 덮어쓴 X-Forwarded-For 를 사용",
          security.client_ip(spoofed) == "1.2.3.4")
    check("프록시 뒤인데 헤더가 없으면 접속 IP 로 폴백",
          security.client_ip(FakeRequest()) == "203.0.113.9")
    _server["ATHENA_BEHIND_PROXY"] = ""

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  3. 공개 모드 — 로컬 계정도 자기 키 필수")
    print("=" * 70)

    from storage import accounts, user_credentials as uc

    GOOGLE = accounts.USER_ID_OFFSET + 800_000_101   # 검사 대역 (진짜 계정 불가)
    LOCAL = 3

    _server["ATHENA_PUBLIC_ORIGIN"] = ""
    check("로컬 서버: 공개 모드 꺼짐", security.public_mode() is False)
    check("로컬 서버: 로컬 계정은 서버 키 허용 (기존 동작)",
          uc.must_use_own_keys(LOCAL) is False)
    check("로컬 서버: 구글 계정은 항상 자기 키 필수",
          uc.must_use_own_keys(GOOGLE) is True)

    _server["ATHENA_PUBLIC_ORIGIN"] = "https://athena.example.com"
    check("공개 서버: 공개 모드 켜짐", security.public_mode() is True)
    check("공개 서버: 로컬 계정도 자기 키 필수",
          uc.must_use_own_keys(LOCAL) is True,
          "가입이 열려 있으므로 서버 키 폴백 = 서버 주인 계좌로 주문")

    # Mongo 를 죽여도(키 조회 불가) 게이트는 차단 쪽으로 실패해야 합니다
    _real_db = accounts._db

    def _dead_db():
        raise accounts.AccountsUnavailable("검사용: Mongo 없음")

    accounts._db = _dead_db
    try:
        from engine import autotrade

        check("공개 서버 + Mongo 다운: 로컬 계정 KIS 주문 차단 (fail-closed)",
              autotrade._own_keys_gate(LOCAL, "live") != "")
        check("공개 서버라도 가상 모의투자(paper)는 키 없이 허용",
              autotrade._own_keys_gate(LOCAL, "paper") == "")

        _server["ATHENA_PUBLIC_ORIGIN"] = ""
        check("로컬 서버 + Mongo 다운: 로컬 계정은 기존 동작 (서버 키)",
              autotrade._own_keys_gate(LOCAL, "live") == "")
    finally:
        accounts._db = _real_db

    # 오버레이 안전 기본값이 공개 모드의 로컬 계정에도 적용되는가
    _server["ATHENA_PUBLIC_ORIGIN"] = "https://athena.example.com"
    accounts._db = _dead_db
    try:
        uc._invalidate(LOCAL)
        overlay = uc.overlay_for(LOCAL)
        check("공개 서버: 로컬 계정도 서버 계좌·실거래 스위치를 물려받지 않는다",
              overlay.get("KIS_ACCOUNT") == "" and overlay.get("KIS_LIVE_TRADING") == "0"
              and overlay.get("KIS_MOCK") == "1", str(overlay))
    finally:
        accounts._db = _real_db
        uc._invalidate(LOCAL)
        _server["ATHENA_PUBLIC_ORIGIN"] = ""

finally:
    credentials.get = _real_get
    security.reset()

print("\n" + "=" * 70)
print(f"  통과 {len(PASS)}  실패 {len(FAIL)}")
if FAIL:
    print("  실패 목록:")
    for name in FAIL:
        print(f"    - {name}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
