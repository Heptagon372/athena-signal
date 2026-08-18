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
    · 토큰 지문     세션 토큰이 원문이 아니라 SHA-256 지문으로 저장되는가.
                   지문 도입 전에 발급된 토큰도 계속 통하는가 (자동 이전)
    · 외부 URL      javascript: 링크가 걸러지는가. 스킴 사이에 제어문자를 끼운
                   우회도 막히는가 — <a href> 는 esc() 로 막히지 않습니다
    · 비밀번호      8자 미만·흔한 값·같은 글자 반복이 거부되는가

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


# ---------------------------------------------------------------------------
# 4. 세션 토큰 지문
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  4. 세션 토큰 지문 — DB 가 새도 토큰 원문은 복원되지 않아야")
print("=" * 70)

_tok = "abcdefghijklmnopqrstuvwxyz0123456789"
_dig = security.token_digest(_tok)
check("지문은 64자 hex", len(_dig) == 64 and all(c in "0123456789abcdef" for c in _dig))
check("같은 토큰은 같은 지문", _dig == security.token_digest(_tok))
check("다른 토큰은 다른 지문", _dig != security.token_digest(_tok + "x"))
check("지문에서 원문이 보이지 않는다", _tok not in _dig)
check("조회 키는 [지문, 원문] 두 개",
      security.token_lookup_keys(_tok) == [_dig, _tok],
      "원문도 찾아봐야 지문 도입 전 세션이 로그아웃되지 않습니다")
check("빈 토큰은 조회 키 없음", security.token_lookup_keys("") == [])

# 실제 저장·조회 왕복 (SQLite)
from storage import users as _users

_uname = "__qa_digest_user__"


def _wipe_qa_user():
    with _users._conn() as conn:
        conn.execute("DELETE FROM sessions WHERE user_id IN "
                     "(SELECT id FROM users WHERE username = ?)", (_uname,))
        conn.execute("DELETE FROM users WHERE username = ?", (_uname,))


try:
    _wipe_qa_user()
    _users.register(_uname, "qa-password-1234")
    _login = _users.login(_uname, "qa-password-1234")
    _session = _login["token"]
    _uid = _login["user"]["id"]

    with _users._conn() as _c:
        _stored = [r["token"] for r in _c.execute(
            "SELECT token FROM sessions WHERE user_id = ?", (_uid,))]

    check("로그인 토큰으로 사용자가 복원된다",
          (_users.user_from_token(_session) or {}).get("username") == _uname)
    check("DB 에는 토큰 원문이 없다", _session not in _stored)
    check("DB 에 있는 것은 지문", security.token_digest(_session) in _stored)

    # 지문 도입 전에 발급된 세션 흉내 — 원문으로 한 줄 심어 둡니다
    _legacy = "legacy-plaintext-token-0001"
    with _users._conn() as _c:
        _c.execute("INSERT INTO sessions (token, user_id, created_at, expires_at) "
                   "VALUES (?, ?, ?, ?)",
                   (_legacy, _uid, "2026-01-01T00:00:00", "2099-01-01T00:00:00"))
    check("원문으로 저장된 옛 세션도 계속 통한다",
          (_users.user_from_token(_legacy) or {}).get("username") == _uname,
          "지문 도입이 이미 로그인해 둔 사람을 쫓아내면 안 됩니다")

    with _users._conn() as _c:
        _after = [r["token"] for r in _c.execute(
            "SELECT token FROM sessions WHERE user_id = ?", (_uid,))]
    check("한 번 쓰이면 그 자리에서 지문으로 바뀐다",
          _legacy not in _after and security.token_digest(_legacy) in _after)

    _users.logout(_session)
    check("로그아웃하면 세션이 사라진다", _users.user_from_token(_session) is None)
finally:
    _wipe_qa_user()


# ---------------------------------------------------------------------------
# 5. 외부 URL — javascript: 링크 차단
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  5. 외부 URL — <a href> 에 들어가도 되는 값만 통과")
print("=" * 70)

for _good in ("https://finance.naver.com/item/board.naver?code=005930",
              "http://example.com/a?b=c#d"):
    check(f"통과: {_good[:44]}", security.safe_external_url(_good) == _good)

_BAD_URLS = {
    "javascript:alert(1)": "javascript: — 클릭 한 번이 스크립트 실행",
    "JaVaScRiPt:alert(1)": "대소문자 섞기",
    "java\tscript:alert(1)": "스킴 사이 탭",
    "java\nscript:alert(1)": "스킴 사이 개행",
    "  javascript:alert(1)": "앞 공백",
    "data:text/html,<script>alert(1)</script>": "data: 문서",
    "vbscript:msgbox(1)": "vbscript:",
    "file:///etc/passwd": "로컬 파일",
    "//evil.example.com": "프로토콜 상대 URL",
}
for _u, _why in _BAD_URLS.items():
    check(f"차단: {_why}", security.safe_external_url(_u) == "", repr(_u))

check("빈 값은 빈 값", security.safe_external_url("") == "")
check("길이 상한이 걸린다",
      len(security.safe_external_url("https://x.com/" + "a" * 900)) == 500)

_DIRTY_NAME = 'a"b\nc.pdf'
check("파일명에서 따옴표·개행이 제거된다",
      '"' not in security.safe_filename(_DIRTY_NAME)
      and "\n" not in security.safe_filename(_DIRTY_NAME),
      "Content-Disposition 헤더가 쪼개지면 안 됩니다")
check("파일명이 비면 기본값", security.safe_filename("///") == "download")


# ---------------------------------------------------------------------------
# 6. 비밀번호 정책
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  6. 비밀번호 — 가입·변경 때만 검사 (기존 계정은 그대로 로그인)")
print("=" * 70)

check("7자는 거부", _users.validate_password("abc1234") is not None)
check("8자는 통과", _users.validate_password("nabi8391") is None)
check("흔한 값은 8자여도 거부", _users.validate_password("abcd1234") is not None,
      "길이만 채운 사전 단어는 남이 첫 번째로 찍어봅니다")
check("흔한 비밀번호는 거부", _users.validate_password("password") is not None)
check("대소문자가 달라도 흔한 값은 거부", _users.validate_password("PassWord") is not None)
check("같은 글자 반복은 거부", _users.validate_password("aaaaaaaa") is not None)
check("두 글자 반복도 거부", _users.validate_password("12121212") is not None)
check("긴 문장은 통과", _users.validate_password("부엉이가 밤에 시세를 본다") is None,
      "복잡도 규칙 대신 길이를 봅니다")
check("200자 초과는 거부", _users.validate_password("a1b2" * 60) is not None)

print("\n" + "=" * 70)
print(f"  통과 {len(PASS)}  실패 {len(FAIL)}")
if FAIL:
    print("  실패 목록:")
    for name in FAIL:
        print(f"    - {name}")
print("=" * 70)
sys.exit(1 if FAIL else 0)
