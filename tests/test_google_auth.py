# -*- coding: utf-8 -*-
"""구글 로그인 · MongoDB 계정 저장소 QA
------------------------------------
    python tests/test_google_auth.py

네트워크도 MongoDB도 필요 없습니다. 검증 로직·URL 조립·1회용 소비 규칙과,
**설정이 없을 때 기존 로그인이 멀쩡한지**(회귀 방지의 핵심)를 봅니다.

설계 근거는 ACCOUNTS.md, 검증 항목은 그 문서 8장에 대응합니다.
"""
import base64
import hashlib
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


import api
from data_sources import credentials, google_oauth
from storage import accounts, users

# 이 검사는 실제 클라이언트 ID 없이 돌아야 합니다.
#
# credentials.get() 은 환경변수 → api_keys.json → 레지스트리 순으로 봅니다.
# 파일 캐시만 바꾸면 **환경변수가 이겨서** 이 PC 에 실제 키가 설정된 뒤에는
# 검사가 깨집니다(aud 비교가 실제 클라이언트 ID 와 어긋남). 그래서 조회 함수
# 자체를 가로채 이 프로세스 안에서만 값을 정합니다 — 어느 PC 에서도 같은 결과.
TEST_CLIENT_ID = "test-client-id.apps.googleusercontent.com"

_real_cred_get = credentials.get
_overrides: dict[str, str] = {}


def _patched_get(name, default=""):
    if name in _overrides:
        return _overrides[name]
    return _real_cred_get(name, default)


credentials.get = _patched_get


def _configure():
    _overrides.clear()
    _overrides.update({
        "GOOGLE_CLIENT_ID": TEST_CLIENT_ID,
        "GOOGLE_CLIENT_SECRET": "test-secret",
        "GOOGLE_REDIRECT_URI": "",          # 기본값(8000번)을 쓰게 둡니다
        "ATHENA_PUBLIC_ORIGIN": "",         # 외부 오리진 허용 목록을 비웁니다
        "MONGODB_URI": "",
    })


def _unconfigure():
    _overrides.clear()
    _overrides.update({
        "GOOGLE_CLIENT_ID": "", "GOOGLE_CLIENT_SECRET": "",
        "MONGODB_URI": "", "ATHENA_PUBLIC_ORIGIN": "",
    })


# ---------------------------------------------------------------------------
# 1. authorize URL
# ---------------------------------------------------------------------------
print("=" * 70)
print("  1. 동의 화면 URL 조립")
print("=" * 70)

_configure()
verifier, challenge = google_oauth.new_pkce()
url = google_oauth.build_authorize_url("STATE123", challenge, "NONCE456")
parsed = urlparse(url)
query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

check("구글 authorize 엔드포인트로 향한다",
      f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == google_oauth.AUTH_ENDPOINT)
check("response_type=code (암묵적 흐름이 아님)", query.get("response_type") == "code")
check("state 가 실린다", query.get("state") == "STATE123")
check("nonce 가 실린다", query.get("nonce") == "NONCE456")
check("code_challenge_method=S256", query.get("code_challenge_method") == "S256",
      "plain 이면 PKCE 의 의미가 없습니다")
check("scope 는 openid email profile 뿐",
      sorted(query.get("scope", "").split()) == ["email", "openid", "profile"],
      "필요 이상 받으면 유출면만 넓어집니다")
check("redirect_uri 가 8000번 API 서버",
      query.get("redirect_uri") == google_oauth.redirect_uri(),
      google_oauth.redirect_uri())
check("prompt=select_account (계정 선택 가능)", query.get("prompt") == "select_account")
check("client_secret 은 URL 에 없다", "client_secret" not in query,
      "동의 화면 URL 은 브라우저 히스토리에 남습니다")

# ---------------------------------------------------------------------------
# 2. PKCE (RFC 7636)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  2. PKCE 변환")
print("=" * 70)

check("verifier 길이가 43~128", 43 <= len(verifier) <= 128, f"{len(verifier)}자")
expected = base64.urlsafe_b64encode(
    hashlib.sha256(verifier.encode("ascii")).digest()).decode().rstrip("=")
check("challenge = base64url(sha256(verifier))", challenge == expected)
check("패딩(=)이 없다", "=" not in challenge, "RFC 7636 은 패딩을 금지합니다")

# RFC 7636 부록 B 의 공식 예시로 구현 자체를 검증합니다
RFC_VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
RFC_CHALLENGE = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
check("RFC 7636 예시와 일치",
      google_oauth._b64url(hashlib.sha256(RFC_VERIFIER.encode()).digest()) == RFC_CHALLENGE)

# ---------------------------------------------------------------------------
# 3. id_token 클레임 검증
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  3. id_token 검증 — 거부해야 할 것을 거부하는가")
print("=" * 70)

NOW = time.time()


def base_claims(**over):
    payload = {
        "iss": "https://accounts.google.com",
        "aud": TEST_CLIENT_ID,
        "sub": "1174test",
        "exp": NOW + 3600,
        "email": "someone@gmail.com",
        "email_verified": True,
        "name": "홍길동",
        "nonce": "NONCE456",
    }
    payload.update(over)
    return payload


def rejects(label, claims, nonce="NONCE456", expect_code=None):
    try:
        google_oauth.validate_claims(claims, expected_nonce=nonce, now=NOW)
    except google_oauth.GoogleAuthError as exc:
        check(label, expect_code is None or exc.code == expect_code, f"code={exc.code}")
        return
    check(label, False, "통과시켜 버렸습니다")


ok_claims = google_oauth.validate_claims(base_claims(), expected_nonce="NONCE456", now=NOW)
check("정상 클레임은 통과", ok_claims["sub"] == "1174test")
check("iss 표기 두 가지 모두 허용",
      google_oauth.validate_claims(base_claims(iss="accounts.google.com"),
                                   expected_nonce="NONCE456", now=NOW) is not None)

rejects("다른 앱에 발급된 토큰(aud 불일치) 거부",
        base_claims(aud="someone-else.apps.googleusercontent.com"), expect_code="bad_audience")
rejects("발급자가 구글이 아니면 거부",
        base_claims(iss="https://evil.example.com"), expect_code="bad_issuer")
rejects("만료된 토큰 거부", base_claims(exp=NOW - 3600), expect_code="expired")
rejects("nonce 불일치 거부", base_claims(nonce="OTHER"), expect_code="bad_nonce")
rejects("미인증 이메일 거부", base_claims(email_verified=False), expect_code="email_unverified")
rejects("sub 없으면 거부", base_claims(sub=""), expect_code="no_sub")
rejects("email 없으면 거부", base_claims(email=""), expect_code="no_email")

# 시계 오차 허용치는 만료를 무한정 봐주지 않아야 합니다
try:
    google_oauth.validate_claims(
        base_claims(exp=NOW - google_oauth.CLOCK_SKEW_SECONDS + 10),
        expected_nonce="NONCE456", now=NOW)
    check("시계 오차 허용치 안쪽은 통과", True)
except google_oauth.GoogleAuthError:
    check("시계 오차 허용치 안쪽은 통과", False)

# id_token 페이로드 디코딩 (패딩이 필요한 길이로 만들어 확인)
def make_jwt(payload_dict):
    import json
    body = google_oauth._b64url(json.dumps(payload_dict).encode())
    return f"header.{body}.signature"


decoded = google_oauth.decode_id_token(make_jwt(base_claims()))
check("id_token 페이로드 디코딩 (base64 패딩 복원)", decoded["sub"] == "1174test")
rejects_bad = False
try:
    google_oauth.decode_id_token("not-a-jwt")
except google_oauth.GoogleAuthError:
    rejects_bad = True
check("형식이 틀린 id_token 거부", rejects_bad)

# ---------------------------------------------------------------------------
# 4. 오픈 리다이렉트 차단
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  4. 되돌아갈 주소 검증")
print("=" * 70)

DEFAULT = api.FRONTEND_ORIGIN
check("localhost 는 그대로 유지",
      api._safe_frontend_origin("http://localhost:3000") == "http://localhost:3000")
check("127.0.0.1 도 그대로 유지 (토큰이 딴 오리진에 저장되면 로그인이 풀립니다)",
      api._safe_frontend_origin("http://127.0.0.1:3000") == "http://127.0.0.1:3000")
check("포트가 달라도 로컬호스트면 허용",
      api._safe_frontend_origin("http://localhost:4000") == "http://localhost:4000")
check("외부 도메인은 기본값으로 대체",
      api._safe_frontend_origin("http://evil.com") == DEFAULT)
check("javascript: 스킴 거부",
      api._safe_frontend_origin("javascript:alert(1)") == DEFAULT)
check("빈 값은 기본값", api._safe_frontend_origin("") == DEFAULT)
check("끝의 / 는 정리", api._safe_frontend_origin("http://localhost:3000/") == "http://localhost:3000")

check("정상 경로는 유지", api._safe_next_path("/paper") == "/paper")
check("프로토콜 상대 URL(//evil.com) 거부", api._safe_next_path("//evil.com") == "/")
check("절대 URL 거부", api._safe_next_path("http://evil.com") == "/")
check("상대 경로 거부", api._safe_next_path("paper") == "/")

# ---------------------------------------------------------------------------
# 5. 1회용 소비 규칙 (가짜 Mongo)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  5. state · 핸드오프의 1회용 규칙")
print("=" * 70)


class FakeCollection:
    """Mongo 컬렉션 흉내 — 이 검사에 필요한 연산만 있습니다.

    핵심은 find_one_and_delete 의 **원자적 1회용** 성질과 $inc 카운터입니다.
    그 두 가지가 state 재생 차단과 user_id 발급의 근거이기 때문입니다.
    """

    _next_id = 1

    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        # pymongo 는 넣은 문서에 _id 를 붙여줍니다. 그걸 흉내내지 않으면
        # 실제로는 없는 KeyError 가 검사에서만 납니다.
        stored = dict(doc)
        stored.setdefault("_id", f"fake-oid-{FakeCollection._next_id}")
        FakeCollection._next_id += 1
        self.docs.append(stored)

    def find_one(self, spec, projection=None):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in spec.items()):
                return doc
        return None

    def find_one_and_delete(self, spec):
        found = self.find_one(spec)
        if found:
            self.docs.remove(found)
        return found

    def find_one_and_update(self, spec, update, upsert=False, return_document=True):
        found = self.find_one(spec)
        if not found and upsert:
            found = dict(spec)
            self.docs.append(found)
        if found is None:
            return None
        for key, delta in update.get("$inc", {}).items():
            found[key] = found.get(key, 0) + delta
        found.update(update.get("$set", {}))
        return found

    def update_one(self, spec, update):
        found = self.find_one(spec)
        if found is None:
            return None
        found.update(update.get("$set", {}))
        for key, delta in update.get("$inc", {}).items():
            found[key] = found.get(key, 0) + delta
        return None

    def delete_one(self, spec):
        found = self.find_one(spec)
        if found:
            self.docs.remove(found)

        class Result:
            deleted_count = 1 if found else 0
        return Result()

    def create_index(self, *a, **kw):
        pass


# 검사용 user_id 대역.
#
# 이게 왜 필요한가: 가짜 카운터를 0 에서 시작하면 첫 계정이 user_id=100001 을 받는데,
# 그건 **진짜 첫 구글 계정이 받는 번호와 똑같습니다.** 그 상태로 아래 정리 단계가
# `DELETE FROM users WHERE id = 100001` 을 돌면 남의 계정을 지웁니다. 실제로 한 번
# 그렇게 지웠습니다(모의계좌·세션까지). 검사는 진짜 데이터가 절대 닿을 수 없는
# 번호대에서만 놀아야 합니다. (tests/test_firebase_auth.py 도 같은 대역을 씁니다.)
TEST_USER_SEQ = 900_000_000
TEST_ID_FLOOR = accounts.USER_ID_OFFSET + TEST_USER_SEQ


class FakeDB:
    def __init__(self):
        self.oauth_state = FakeCollection()
        self.handoffs = FakeCollection()
        self.sessions = FakeCollection()
        self.accounts = FakeCollection()
        self.counters = FakeCollection()
        self.counters.insert_one({"_id": "user_id", "seq": TEST_USER_SEQ})


fake = FakeDB()
_real_db = accounts._db
accounts._db = lambda: fake
try:
    accounts.save_oauth_state("S1", "verifier-1", "http://localhost:3000", "/paper", "N1")
    first = accounts.consume_oauth_state("S1")
    second = accounts.consume_oauth_state("S1")
    check("state 를 한 번은 쓸 수 있다", first is not None and first["code_verifier"] == "verifier-1")
    check("같은 state 두 번째는 실패 (재생 공격 차단)", second is None)
    check("nonce 와 돌아갈 경로가 함께 보관된다",
          first["nonce"] == "N1" and first["next"] == "/paper")

    code = accounts.create_handoff("session-token-abc")
    check("핸드오프 코드가 세션 토큰과 다르다", code != "session-token-abc",
          "URL 에 실리는 값이 세션 토큰이면 안 됩니다")
    check("핸드오프 1회 교환 성공", accounts.consume_handoff(code) == "session-token-abc")
    check("같은 핸드오프 두 번째는 실패", accounts.consume_handoff(code) is None)

    # 만료된 핸드오프는 문서가 남아 있어도 거부해야 합니다
    # (TTL 인덱스는 최대 60초 늦게 도는 백그라운드 작업입니다)
    from datetime import datetime, timedelta, timezone
    fake.handoffs.insert_one({
        "code": "STALE", "token": "old-token",
        "expires_at": datetime.now(timezone.utc) - timedelta(seconds=5),
    })
    check("만료된 핸드오프 거부 (TTL 지연 구간 방어)",
          accounts.consume_handoff("STALE") is None)
finally:
    accounts._db = _real_db

# ---------------------------------------------------------------------------
# 6. 콜백 전체 흐름 (가짜 Mongo + 구글 응답 스텁)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  6. 로그인 → 계정 저장 → 재로그인 → 계정 불러오기")
print("=" * 70)

from starlette.requests import Request as StarletteRequest


def make_request(query: str = ""):
    """라이브 서버 없이 엔드포인트를 직접 부르기 위한 최소 Request.

    실제 서버를 띄우지 않는 이유: startup 이벤트가 자동매매 루프를 시작합니다.
    이 PC 는 실계좌에 붙어 있어서, 검사용으로 두 번째 인스턴스를 띄우면 매매
    루프가 하나 더 도는 위험이 있습니다. 절대 하면 안 됩니다.
    """
    return StarletteRequest({
        "type": "http", "method": "GET", "path": "/", "headers": [],
        "query_string": query.encode(), "scheme": "http",
        "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 1234),
        "root_path": "", "app": api.app,
    })


PROFILE = {
    "sub": "google-sub-qa-0001",
    "email": "qa-google-user@example.com",
    "email_verified": True,
    "name": "QA 구글 사용자",
    "picture": "https://example.com/p.png",
    "locale": "ko",
}

fake2 = FakeDB()
_real_db = accounts._db
_real_fetch = google_oauth.fetch_identity
accounts._db = lambda: fake2
google_oauth.fetch_identity = lambda code, verifier, nonce="": dict(PROFILE)
_configure()
_overrides["MONGODB_URI"] = "mongodb://localhost:27017"     # 가짜 _db 를 쓰므로 접속은 안 합니다

created_ids = []
try:
    # --- 첫 로그인 ---
    accounts.save_oauth_state("S-FLOW", "verifier-flow", "http://127.0.0.1:3000", "/paper", "N-FLOW")
    response = api.auth_google_callback(make_request(), code="fake-code", state="S-FLOW")

    location = response.headers.get("location", "")
    check("콜백이 302 로 프론트엔드로 되돌린다", response.status_code == 302, location)
    check("시작한 오리진(127.0.0.1)을 지킨다", location.startswith("http://127.0.0.1:3000/auth/callback"),
          "localhost 로 바뀌면 토큰이 딴 오리진에 저장돼 로그인이 풀립니다")

    returned = {k: v[0] for k, v in parse_qs(urlparse(location).query).items()}
    check("URL 에 handoff 만 실린다 (세션 토큰 아님)",
          "handoff" in returned and "token" not in returned)
    check("돌아갈 경로가 유지된다", returned.get("next") == "/paper")

    session_token = fake2.sessions.docs[0]["token"]
    check("핸드오프 코드는 세션 토큰과 다른 값", returned["handoff"] != session_token)

    account_doc = fake2.accounts.docs[0]
    created_ids.append(account_doc["user_id"])
    check("계정이 MongoDB 에 저장됐다", account_doc["email"] == PROFILE["email"])
    check("google_sub 로 저장된다", account_doc["google_sub"] == PROFILE["sub"])
    check("user_id 가 오프셋 위에서 발급된다", account_doc["user_id"] > accounts.USER_ID_OFFSET,
          f"user_id={account_doc['user_id']}")
    check("검사가 진짜 계정 번호대를 건드리지 않는다",
          account_doc["user_id"] >= TEST_ID_FLOOR,
          "겹치면 아래 정리 단계가 진짜 사용자의 행을 지웁니다")
    check("SQLite 에도 같은 id 의 행이 생겼다 (FK 대상)",
          users.find_by_username(f"google:{PROFILE['sub']}")["id"] == account_doc["user_id"])

    # --- 핸드오프 교환 ---
    exchanged = api.auth_google_exchange(api.GoogleExchange(handoff=returned["handoff"]))
    check("핸드오프 교환으로 토큰과 사용자를 받는다",
          exchanged["token"] == session_token and exchanged["user"]["id"] == account_doc["user_id"])
    check("사용자 dict 가 기존 코드가 쓰는 모양 (id/username/display_name)",
          all(k in exchanged["user"] for k in ("id", "username", "display_name")))
    check("표시 이름이 구글 이름", exchanged["user"]["display_name"] == PROFILE["name"])

    reused = api.auth_google_exchange(api.GoogleExchange(handoff=returned["handoff"]))
    check("같은 핸드오프 재사용은 400", getattr(reused, "status_code", 200) == 400)

    # --- 세션으로 사용자 복원 ---
    check("Mongo 세션 토큰으로 사용자 복원",
          accounts.user_from_token(session_token)["id"] == account_doc["user_id"])

    # --- 재로그인: 계정을 새로 만들지 않고 불러와야 합니다 ---
    accounts.save_oauth_state("S-FLOW2", "verifier-flow2", "http://127.0.0.1:3000", "/", "N2")
    api.auth_google_callback(make_request(), code="fake-code-2", state="S-FLOW2")
    check("재로그인해도 계정은 하나 (google_sub 로 찾음)", len(fake2.accounts.docs) == 1,
          f"{len(fake2.accounts.docs)}개")
    check("재로그인 시 같은 user_id (기록이 이어진다)",
          fake2.accounts.docs[0]["user_id"] == account_doc["user_id"])
    check("login_count 가 올라간다", fake2.accounts.docs[0].get("login_count") == 2,
          f"login_count={fake2.accounts.docs[0].get('login_count')}")

    # --- 실패 경로 ---
    denied = api.auth_google_callback(make_request(), error="access_denied")
    check("사용자가 취소하면 JSON 이 아니라 화면으로 되돌린다",
          denied.status_code == 302 and "error=access_denied" in denied.headers.get("location", ""))

    stale = api.auth_google_callback(make_request(), code="c", state="NEVER-ISSUED")
    check("발급하지 않은 state 는 거부",
          stale.status_code == 302 and "error=bad_state" in stale.headers.get("location", ""))

    # 같은 state 로 두 번 (인가 코드 재생) — 두 번째는 반드시 막혀야 합니다
    accounts.save_oauth_state("S-REPLAY", "v", "http://localhost:3000", "/", "N")
    api.auth_google_callback(make_request(), code="c1", state="S-REPLAY")
    replayed = api.auth_google_callback(make_request(), code="c1", state="S-REPLAY")
    check("같은 state 재사용 거부 (재생 공격)",
          "error=bad_state" in replayed.headers.get("location", ""))

    # --- 로그아웃 ---
    check("Mongo 세션 로그아웃", accounts.delete_session(session_token) is True)
    check("로그아웃 후 세션 무효", accounts.user_from_token(session_token) is None)
finally:
    accounts._db = _real_db
    google_oauth.fetch_identity = _real_fetch
    # 검사가 SQLite 에 만든 그림자 행·모의계좌 정리.
    # id 를 그대로 믿고 지우면, 검사용 번호가 진짜 계정과 겹쳤을 때 남의 계정을
    # 지웁니다. 검사 대역 밖은 무슨 일이 있어도 건드리지 않습니다.
    with users._conn() as conn:
        for uid in created_ids:
            if uid < TEST_ID_FLOOR:
                check(f"정리 대상 user_id={uid} 가 검사 대역 안", False,
                      "진짜 계정일 수 있어 지우지 않았습니다")
                continue
            conn.execute("DELETE FROM users WHERE id = ?", (uid,))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM paper_account WHERE user_id = ?", (uid,))

# ---------------------------------------------------------------------------
# 7. 미설정 폴백 — 기존 로그인이 멀쩡해야 합니다
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  7. 설정이 없을 때 (회귀 방지)")
print("=" * 70)

_unconfigure()
accounts._client = None

check("구글 미설정이면 configured=false", api._google_status()["configured"] is False)
check("사유가 사람이 읽을 수 있다", bool(api._google_status().get("reason")))
check("Mongo 미설정이면 configured=false", accounts.status()["configured"] is False)

# Mongo 를 못 써도 예외가 아니라 None 이어야 합니다 — 여기서 터지면
# current_user() 가 죽고 앱 전체가 로그인 불가가 됩니다
try:
    check("Mongo 없이 user_from_token 은 None (예외 아님)",
          accounts.user_from_token("whatever") is None)
except Exception as exc:
    check("Mongo 없이 user_from_token 은 None (예외 아님)", False, f"{type(exc).__name__}: {exc}")

try:
    check("Mongo 없이 delete_session 은 False (예외 아님)",
          accounts.delete_session("whatever") is False)
except Exception as exc:
    check("Mongo 없이 delete_session 은 False (예외 아님)", False, f"{type(exc).__name__}: {exc}")

# 아이디/비번 로그인이 그대로 되는지 — 이게 깨지면 이번 작업은 실패입니다
users.init()
probe_name = f"qa_probe_{int(time.time())}"
registered = users.register(probe_name, "probe-password-123", "QA 검사용")
check("아이디/비번 회원가입 정상", registered.get("ok") is True, registered.get("error", ""))

if registered.get("ok"):
    logged_in = users.login(probe_name, "probe-password-123")
    check("아이디/비번 로그인 정상", logged_in.get("ok") is True)
    if logged_in.get("ok"):
        token = logged_in["token"]
        check("SQLite 세션으로 사용자 복원", users.user_from_token(token) is not None)
        users.logout(token)
        check("로그아웃 후 세션 무효", users.user_from_token(token) is None)

    wrong = users.login(probe_name, "wrong-password")
    check("틀린 비밀번호 거부", wrong.get("ok") is False)

# 구글 전용 계정(비밀번호 없는 계정)은 비밀번호로 로그인할 수 없어야 합니다
shadow_id = 999_001
users.ensure_external_user(shadow_id, "google:qa-shadow-sub", "그림자 계정")
found = users.find_by_username("google:qa-shadow-sub")
check("외부 계정 행이 지정한 id 로 생성된다", found is not None and found["id"] == shadow_id,
      f"id={found['id'] if found else None}")
check("같은 id 로 두 번 호출해도 안전 (INSERT OR IGNORE)",
      users.ensure_external_user(shadow_id, "google:qa-shadow-sub", "그림자 계정") is None)
for attempt in ("", "!", "password", "-"):
    result = users.login("google:qa-shadow-sub", attempt)
    if result.get("ok"):
        check("구글 전용 계정은 비밀번호 로그인 불가", False, f"'{attempt}' 로 뚫렸습니다")
        break
else:
    check("구글 전용 계정은 비밀번호 로그인 불가", True)

check("user_id 오프셋이 로컬 id 와 겹치지 않는다",
      accounts.USER_ID_OFFSET >= 100_000, f"offset={accounts.USER_ID_OFFSET}")

# 검사용으로 만든 행 정리
with users._conn() as conn:
    conn.execute("DELETE FROM users WHERE username = ? OR id = ?",
                 (probe_name, shadow_id))
    conn.execute("DELETE FROM sessions WHERE user_id = ?", (shadow_id,))

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"  결과: 통과 {len(PASS)} / 실패 {len(FAIL)}")
print("=" * 70)
if FAIL:
    for name in FAIL:
        print(f"    - {name}")
    sys.exit(1)
print("  모두 통과했습니다.\n")
