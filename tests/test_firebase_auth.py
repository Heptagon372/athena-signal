# -*- coding: utf-8 -*-
"""Firebase 구글 로그인 QA
------------------------
    python tests/test_firebase_auth.py

네트워크도 MongoDB도 Firebase 프로젝트도 필요 없습니다. 검증 로직·거부 규칙과,
**설정이 없을 때 기존 로그인이 멀쩡한지**(회귀 방지의 핵심)를 봅니다.

여기서 가장 중요한 검사는 두 가지입니다.
  · 서명이 틀린 토큰이 반드시 거부되는가 — 이게 뚫리면 로그인 자체가 무의미합니다
  · Firebase 로 바꿔도 예전 OAuth 계정의 user_id 가 유지되는가 — 안 되면
    사용자 눈에는 예측 기록·모의투자 계좌가 통째로 사라진 것으로 보입니다

설계 근거는 ACCOUNTS.md 0장에 있습니다.
"""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


def skip(name, why=""):
    SKIP.append(name)
    print(f"  [건너뜀] {name}  {why}")


def raises(fn, code):
    """fn() 이 FirebaseAuthError 를 code 와 함께 던지는지."""
    try:
        fn()
    except firebase_auth.FirebaseAuthError as exc:
        return exc.code == code, f"실제 code={exc.code} ({exc})"
    except Exception as exc:                                       # noqa: BLE001
        return False, f"다른 예외: {type(exc).__name__}: {exc}"
    return False, "예외가 나지 않았습니다"


import api
from data_sources import credentials, firebase_auth, http_client
from storage import accounts, users

# 이 검사는 실제 Firebase 프로젝트 없이 돌아야 합니다.
#
# credentials.get() 은 환경변수 → api_keys.json → 레지스트리 순으로 봅니다.
# 파일 캐시만 바꾸면 **환경변수가 이겨서** 이 PC 에 실제 키가 설정된 뒤에는
# 검사가 깨집니다(aud 비교가 실제 프로젝트 ID 와 어긋남). 그래서 조회 함수
# 자체를 가로채 이 프로세스 안에서만 값을 정합니다 — 어느 PC 에서도 같은 결과.
TEST_PROJECT = "athena-qa-project"
TEST_API_KEY = "AIzaSyQA-test-key"

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
        "FIREBASE_PROJECT_ID": TEST_PROJECT,
        "FIREBASE_API_KEY": TEST_API_KEY,
        "FIREBASE_AUTH_DOMAIN": "",     # 기본값(<projectId>.firebaseapp.com)을 쓰게 둡니다
        "MONGODB_URI": "",
        "GOOGLE_CLIENT_ID": "", "GOOGLE_CLIENT_SECRET": "",
    })


def _unconfigure():
    _overrides.clear()
    _overrides.update({
        "FIREBASE_PROJECT_ID": "", "FIREBASE_API_KEY": "", "FIREBASE_AUTH_DOMAIN": "",
        "GOOGLE_CLIENT_ID": "", "GOOGLE_CLIENT_SECRET": "", "MONGODB_URI": "",
    })


# ---------------------------------------------------------------------------
# 토큰 조립 헬퍼 (서명은 가짜 — 서명 검증은 5장에서 진짜로 합니다)
# ---------------------------------------------------------------------------

GOOGLE_SUB = "117400000000000000001"
FIREBASE_UID = "qa-firebase-uid-0001"
EMAIL = "qa-firebase-user@example.com"


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def payload(**overrides) -> dict:
    now = int(time.time())
    base = {
        "iss": f"https://securetoken.google.com/{TEST_PROJECT}",
        "aud": TEST_PROJECT,
        "auth_time": now - 10,
        "user_id": FIREBASE_UID,
        "sub": FIREBASE_UID,
        "iat": now - 10,
        "exp": now + 3600,
        "email": EMAIL,
        "email_verified": True,
        "name": "QA 파이어베이스 사용자",
        "picture": "https://example.com/p.png",
        "firebase": {
            "identities": {"google.com": [GOOGLE_SUB], "email": [EMAIL]},
            "sign_in_provider": "google.com",
        },
    }
    base.update(overrides)
    return base


def token(body: dict | None = None, header: dict | None = None,
          signature: str = "ZmFrZS1zaWduYXR1cmU") -> str:
    head = header if header is not None else {"alg": "RS256", "kid": "qa-kid"}
    return ".".join([
        b64url(json.dumps(head).encode()),
        b64url(json.dumps(body if body is not None else payload()).encode()),
        signature,
    ])


# ---------------------------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------------------------
print("=" * 70)
print("  1. 설정 · 공개 웹 설정")
print("=" * 70)

_unconfigure()
check("미설정이면 configured=false", firebase_auth.status()["configured"] is False)
check("미설정 사유가 어떤 키인지 알려준다",
      "FIREBASE_PROJECT_ID" in firebase_auth.status()["reason"])

_overrides["FIREBASE_PROJECT_ID"] = TEST_PROJECT
check("API 키만 빠져도 configured=false",
      firebase_auth.status()["configured"] is False
      and "FIREBASE_API_KEY" in firebase_auth.status()["reason"])

_configure()
web = firebase_auth.web_config()
check("웹 설정 3종을 내려준다",
      web == {"apiKey": TEST_API_KEY, "authDomain": f"{TEST_PROJECT}.firebaseapp.com",
              "projectId": TEST_PROJECT}, str(web))
check("authDomain 기본값이 <projectId>.firebaseapp.com",
      firebase_auth.auth_domain() == f"{TEST_PROJECT}.firebaseapp.com")
_overrides["FIREBASE_AUTH_DOMAIN"] = "login.example.com"
check("authDomain 을 직접 지정하면 그 값을 쓴다",
      firebase_auth.auth_domain() == "login.example.com")
_configure()
check("iss 는 프로젝트별 securetoken 주소",
      firebase_auth.issuer() == f"https://securetoken.google.com/{TEST_PROJECT}")

# 공개 설정에 시크릿이 섞이면 안 됩니다 (프론트엔드로 그대로 나가는 값)
check("웹 설정에 비밀 항목이 없다",
      not any(k.lower().endswith("secret") for k in web),
      "이 dict 는 브라우저로 그대로 나갑니다")

# ---------------------------------------------------------------------------
# 2. JWT 파싱
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  2. 토큰 형식")
print("=" * 70)

check("정상 토큰의 페이로드를 읽는다",
      firebase_auth.decode_payload(token())["sub"] == FIREBASE_UID)
check("헤더를 읽는다", firebase_auth.decode_header(token())["kid"] == "qa-kid")

for label, bad in (("점이 2개가 아님", "a.b"),
                   ("빈 세그먼트", "a..c"),
                   ("빈 문자열", ""),
                   ("아예 JWT 가 아님", "not-a-token")):
    ok, detail = raises(lambda t=bad: firebase_auth.decode_payload(t), "bad_id_token")
    check(f"거부: {label}", ok, detail)

ok, detail = raises(
    lambda: firebase_auth.decode_payload(f"{b64url(b'{}')}.{b64url(b'not json')}.sig"),
    "bad_id_token")
check("거부: 페이로드가 JSON 이 아님", ok, detail)

ok, detail = raises(
    lambda: firebase_auth.decode_payload(f"{b64url(b'{}')}.{b64url(b'[1,2]')}.sig"),
    "bad_id_token")
check("거부: 페이로드가 객체가 아님 (배열)", ok, detail)

# ---------------------------------------------------------------------------
# 3. 클레임 검증
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  3. 클레임 검증 (iss · aud · exp · email)")
print("=" * 70)

check("정상 클레임 통과", firebase_auth.validate_claims(payload())["sub"] == FIREBASE_UID)

cases = [
    ("다른 프로젝트의 iss", {"iss": "https://securetoken.google.com/someone-else"}, "bad_issuer"),
    ("iss 가 구글이 아님", {"iss": "https://evil.example.com/" + TEST_PROJECT}, "bad_issuer"),
    ("다른 프로젝트의 aud", {"aud": "someone-else"}, "bad_audience"),
    ("만료된 exp", {"exp": int(time.time()) - 600}, "expired"),
    ("미래에 발급된 iat", {"iat": int(time.time()) + 600}, "bad_issued_at"),
    ("sub 없음", {"sub": ""}, "no_sub"),
    ("email 없음", {"email": ""}, "no_email"),
    ("미인증 이메일", {"email_verified": False}, "email_unverified"),
]
for label, over, code in cases:
    ok, detail = raises(lambda o=over: firebase_auth.validate_claims(payload(**o)), code)
    check(f"거부: {label}", ok, detail)

# 시계 오차 여유 — 방금 만료된 토큰까지 자르면 정상 로그인이 랜덤하게 실패합니다
just_expired = payload(exp=int(time.time()) - 30)
check("만료 직후 30초는 허용 (시계 오차 여유)",
      firebase_auth.validate_claims(just_expired)["sub"] == FIREBASE_UID,
      f"CLOCK_SKEW_SECONDS={firebase_auth.CLOCK_SKEW_SECONDS}")

# ---------------------------------------------------------------------------
# 4. 구글 계정 식별자 추출 — 계정이 갈라지지 않게 하는 핵심
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  4. 계정 키는 Firebase UID 가 아니라 구글 sub")
print("=" * 70)

check("firebase.identities['google.com'] 에서 구글 sub 를 꺼낸다",
      firebase_auth.google_sub(payload()) == GOOGLE_SUB)
check("Firebase UID 를 계정 키로 쓰지 않는다",
      firebase_auth.google_sub(payload()) != FIREBASE_UID,
      "이걸 쓰면 예전 계정과 이어지지 않습니다 (ACCOUNTS.md 0-4)")

not_google = [
    ("비밀번호 로그인", {"firebase": {"identities": {"email": [EMAIL]},
                                     "sign_in_provider": "password"}}),
    ("identities 가 비었음", {"firebase": {"identities": {}, "sign_in_provider": "google.com"}}),
    ("firebase 클레임 없음", {"firebase": {}}),
]
for label, over in not_google:
    ok, detail = raises(lambda o=over: firebase_auth.google_sub(payload(**o)), "not_google")
    check(f"거부: {label}", ok, detail)

# ---------------------------------------------------------------------------
# 5. 서명 검증 — 로컬 (cryptography)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  5. RS256 서명 검증 (이 PC 에서 직접)")
print("=" * 70)

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from cryptography.x509.oid import NameOID
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False

if not HAVE_CRYPTO:
    skip("서명 검증 (로컬)", "cryptography 미설치 — 구글 위임 경로는 6장에서 검사합니다")
else:
    from datetime import datetime, timedelta, timezone

    # 구글 흉내: 키 한 쌍과 자체 서명 인증서를 만들어 fetch_certs 를 갈아끼웁니다
    _key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "qa-securetoken")])
    _cert = (x509.CertificateBuilder()
             .subject_name(_name).issuer_name(_name)
             .public_key(_key.public_key())
             .serial_number(x509.random_serial_number())
             .not_valid_before(datetime.now(timezone.utc) - timedelta(days=1))
             .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
             .sign(_key, hashes.SHA256()))
    _pem = _cert.public_bytes(serialization.Encoding.PEM).decode("ascii")

    def signed_token(body: dict | None = None, kid: str = "qa-kid",
                     alg: str = "RS256") -> str:
        head = {"alg": alg}
        if kid:
            head["kid"] = kid
        parts = [b64url(json.dumps(head).encode()),
                 b64url(json.dumps(body if body is not None else payload()).encode())]
        signature = _key.sign(".".join(parts).encode("ascii"),
                              padding.PKCS1v15(), hashes.SHA256())
        return ".".join(parts + [b64url(signature)])

    _real_fetch_certs = firebase_auth.fetch_certs
    firebase_auth.fetch_certs = lambda force=False: {"qa-kid": _pem}
    try:
        check("올바르게 서명된 토큰을 통과시킨다",
              firebase_auth._verify_signature_local(signed_token()) is True)

        # 페이로드만 갈아끼운 토큰 — 이걸 통과시키면 아무 계정이나 사칭할 수 있습니다
        good = signed_token()
        head_seg, _, sig_seg = good.split(".")
        forged_body = b64url(json.dumps(payload(email="victim@example.com")).encode())
        ok, detail = raises(
            lambda: firebase_auth._verify_signature_local(f"{head_seg}.{forged_body}.{sig_seg}"),
            "bad_signature")
        check("거부: 페이로드를 바꿔치기한 토큰", ok, detail)

        ok, detail = raises(
            lambda: firebase_auth._verify_signature_local(token()), "bad_signature")
        check("거부: 아무 서명이나 붙인 토큰", ok, detail)

        ok, detail = raises(
            lambda: firebase_auth._verify_signature_local(signed_token(alg="none")),
            "bad_algorithm")
        check("거부: alg=none 으로 바꾼 토큰 (JWT 의 고전적 구멍)", ok, detail)

        ok, detail = raises(
            lambda: firebase_auth._verify_signature_local(signed_token(alg="HS256")),
            "bad_algorithm")
        check("거부: alg=HS256 으로 바꾼 토큰", ok, detail)

        ok, detail = raises(
            lambda: firebase_auth._verify_signature_local(signed_token(kid="")), "no_kid")
        check("거부: kid 없는 토큰", ok, detail)

        ok, detail = raises(
            lambda: firebase_auth._verify_signature_local(signed_token(kid="다른-kid")),
            "unknown_kid")
        check("거부: 구글 공개키에 없는 kid", ok, detail)
    finally:
        firebase_auth.fetch_certs = _real_fetch_certs

# 인증서 캐시 수명 계산 (네트워크 없이 헤더 파싱만)
check("Cache-Control 의 max-age 를 읽는다",
      firebase_auth._parse_max_age("public, max-age=19008, must-revalidate") == 19008)
check("max-age 가 없으면 0 (기본 수명으로 폴백)",
      firebase_auth._parse_max_age("no-cache") == 0)

# ---------------------------------------------------------------------------
# 6. 서명 검증 — 구글에 위임 (cryptography 가 없을 때의 경로)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  6. 서명 검증 위임 (Identity Toolkit)")
print("=" * 70)

_real_post_full = http_client.post_full
_lookup_calls = []


def stub_lookup(result):
    """(status, json, raw) 를 돌려주는 가짜 accounts:lookup."""
    def _post(url, *, json_body=None, headers=None, timeout=None):
        _lookup_calls.append({"url": url, "body": json_body})
        return result
    return _post


try:
    http_client.post_full = stub_lookup(
        (200, {"users": [{"localId": FIREBASE_UID, "email": EMAIL}]}, ""))
    record = firebase_auth._verify_via_google(token(), expected_sub=FIREBASE_UID)
    check("구글이 사용자 레코드를 주면 통과", record["localId"] == FIREBASE_UID)
    check("웹 API 키를 쿼리로 붙여 부른다", TEST_API_KEY in _lookup_calls[-1]["url"])
    check("토큰을 본문으로 보낸다 (URL 에 싣지 않음)",
          _lookup_calls[-1]["body"] == {"idToken": token()}
          and "idToken" not in _lookup_calls[-1]["url"])

    http_client.post_full = stub_lookup(
        (200, {"users": [{"localId": "somebody-else"}]}, ""))
    ok, detail = raises(
        lambda: firebase_auth._verify_via_google(token(), expected_sub=FIREBASE_UID),
        "sub_mismatch")
    check("거부: 구글이 다른 사람을 가리킴", ok, detail)

    http_client.post_full = stub_lookup((200, {"users": []}, ""))
    ok, detail = raises(
        lambda: firebase_auth._verify_via_google(token(), expected_sub=FIREBASE_UID),
        "no_user")
    check("거부: 사용자를 찾지 못함", ok, detail)

    http_client.post_full = stub_lookup(
        (400, {"error": {"message": "INVALID_ID_TOKEN"}}, ""))
    ok, detail = raises(
        lambda: firebase_auth._verify_via_google(token(), expected_sub=FIREBASE_UID),
        "bad_id_token")
    check("거부: 구글이 위조·만료로 판정", ok, detail)

    http_client.post_full = stub_lookup(
        (403, {"error": {"message": "Requests from referer <empty> are blocked."}}, ""))
    ok, detail = raises(
        lambda: firebase_auth._verify_via_google(token(), expected_sub=FIREBASE_UID),
        "api_key_restricted")
    check("403 은 원인을 짚어준다 (API 키 리퍼러 제한)", ok, detail)
    try:
        firebase_auth._verify_via_google(token(), expected_sub=FIREBASE_UID)
    except firebase_auth.FirebaseAuthError as exc:
        check("403 메시지가 해법까지 알려준다",
              "cryptography" in str(exc) and "제한" in str(exc))

    http_client.post_full = stub_lookup((0, None, "네트워크 연결 실패"))
    ok, detail = raises(
        lambda: firebase_auth._verify_via_google(token(), expected_sub=FIREBASE_UID),
        "network")
    check("네트워크 실패를 통과로 오해하지 않는다", ok, detail)
finally:
    http_client.post_full = _real_post_full

# ---------------------------------------------------------------------------
# 7. verify_id_token 전체
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  7. verify_id_token — 프로필 추출과 순서")
print("=" * 70)

_real_local = firebase_auth._verify_signature_local

_unconfigure()
ok, detail = raises(lambda: firebase_auth.verify_id_token(token()), "not_configured")
check("미설정이면 검증 자체를 하지 않는다", ok, detail)

_configure()
ok, detail = raises(lambda: firebase_auth.verify_id_token(""), "bad_id_token")
check("빈 토큰 거부", ok, detail)

firebase_auth._verify_signature_local = lambda t: True      # 서명은 5장에서 검사했습니다
try:
    profile = firebase_auth.verify_id_token(token())
    check("sub 은 구글 계정 sub", profile["sub"] == GOOGLE_SUB)
    check("firebase_uid 를 따로 담는다", profile["firebase_uid"] == FIREBASE_UID)
    check("이메일·이름·사진이 실린다",
          profile["email"] == EMAIL and profile["name"] and profile["picture"])
    check("email_verified 가 불리언", profile["email_verified"] is True)
    check("upsert_google_account 가 받는 키를 모두 갖췄다",
          all(k in profile for k in ("sub", "email", "email_verified", "name",
                                     "picture", "locale")))

    ok, detail = raises(
        lambda: firebase_auth.verify_id_token(token(payload(aud="someone-else"))),
        "bad_audience")
    # 서명 키는 프로젝트 공용입니다 — 프로젝트를 가르는 건 aud/iss 뿐입니다
    check("서명이 맞아도 다른 프로젝트 토큰은 거부", ok, detail)
finally:
    firebase_auth._verify_signature_local = _real_local

# 서명 검증이 클레임 검증보다 먼저 돌아야 합니다. 순서가 반대면 위조 토큰의
# 내용으로 오류 메시지가 만들어져 공격자에게 우리 설정을 알려주게 됩니다.
firebase_auth._verify_signature_local = lambda t: False
_real_via_google = firebase_auth._verify_via_google


def _reject(t, expected_sub):
    raise firebase_auth.FirebaseAuthError("서명 실패", "bad_signature")


firebase_auth._verify_via_google = _reject
try:
    ok, detail = raises(
        lambda: firebase_auth.verify_id_token(token(payload(aud="someone-else"))),
        "bad_signature")
    # 순서가 반대면 오류 메시지가 우리 설정을 흘립니다
    check("서명 검증이 클레임 검증보다 먼저 돈다", ok, detail)

    # cryptography 도 없고 구글도 못 부르면 통과시키면 안 됩니다
    ok, detail = raises(lambda: firebase_auth.verify_id_token(token()), "bad_signature")
    check("검증할 방법이 없으면 로그인을 막는다 (건너뛰는 경로 없음)", ok, detail)
finally:
    firebase_auth._verify_signature_local = _real_local
    firebase_auth._verify_via_google = _real_via_google

# ---------------------------------------------------------------------------
# 8. 엔드포인트 — 로그인 → 계정 저장 → 재로그인
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  8. POST /api/auth/firebase/session")
print("=" * 70)

from starlette.requests import Request as StarletteRequest


class FakeCollection:
    """Mongo 컬렉션 흉내 — 이 검사에 필요한 연산만 있습니다."""

    _next_id = 1

    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
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
# 그건 **진짜 첫 구글 계정이 받는 번호와 똑같습니다.** 그 상태로 정리 단계가
# `DELETE FROM users WHERE id = 100001` 을 돌면 남의 계정을 지웁니다. 실제로 한 번
# 그렇게 지웠습니다(모의계좌·세션까지). 검사는 진짜 데이터가 절대 닿을 수 없는
# 번호대에서만 놀아야 합니다.
TEST_USER_SEQ = 800_000_000
TEST_ID_FLOOR = accounts.USER_ID_OFFSET + TEST_USER_SEQ


class FakeDB:
    def __init__(self):
        self.oauth_state = FakeCollection()
        self.handoffs = FakeCollection()
        self.sessions = FakeCollection()
        self.accounts = FakeCollection()
        self.counters = FakeCollection()
        self.counters.insert_one({"_id": "user_id", "seq": TEST_USER_SEQ})


def make_request():
    """라이브 서버 없이 엔드포인트를 직접 부르기 위한 최소 Request.

    실제 서버를 띄우지 않는 이유: startup 이벤트가 자동매매 루프를 시작합니다.
    이 PC 는 실계좌에 붙어 있어서, 검사용으로 두 번째 인스턴스를 띄우면 매매
    루프가 하나 더 도는 위험이 있습니다. 절대 하면 안 됩니다.
    """
    return StarletteRequest({
        "type": "http", "method": "POST", "path": "/", "headers": [],
        "query_string": b"", "scheme": "http",
        "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 1234),
        "root_path": "", "app": api.app,
    })


fake = FakeDB()
_real_db = accounts._db
accounts._db = lambda: fake
firebase_auth._verify_signature_local = lambda t: True
_configure()
_overrides["MONGODB_URI"] = "mongodb://localhost:27017"   # 가짜 _db 라 접속은 안 합니다

# 이 검사는 진짜 SQLite(athena.db)에 그림자 행을 만듭니다. 지우지 않고 끝내면
# 같은 user_id 를 쓰는 tests/test_google_auth.py 가 뒤이어 돌 때 INSERT OR IGNORE
# 에 막혀 엉뚱한 곳에서 실패합니다. 만든 것은 반드시 되돌립니다.
created_ids: list[int] = []

try:
    result = api.auth_firebase_session(api.FirebaseSession(id_token=token()), make_request())
    check("세션 토큰과 사용자를 바로 돌려준다 (핸드오프 단계 없음)",
          isinstance(result, dict) and result.get("token") and result.get("user"))
    check("첫 로그인은 created=true", result.get("created") is True)

    account_doc = fake.accounts.docs[0]
    created_ids.append(account_doc["user_id"])
    check("계정이 MongoDB 에 저장됐다", account_doc["email"] == EMAIL)
    check("계정 키는 google_sub (Firebase UID 아님)",
          account_doc["google_sub"] == GOOGLE_SUB)
    check("firebase_uid 를 참고값으로 함께 저장", account_doc.get("firebase_uid") == FIREBASE_UID)
    check("user_id 가 오프셋 위에서 발급된다",
          account_doc["user_id"] > accounts.USER_ID_OFFSET, f"user_id={account_doc['user_id']}")
    check("검사가 진짜 계정 번호대를 건드리지 않는다",
          account_doc["user_id"] >= TEST_ID_FLOOR,
          "겹치면 정리 단계가 진짜 사용자의 행을 지웁니다")
    check("SQLite 에도 같은 id 의 행이 생겼다 (FK 대상)",
          users.find_by_username(f"google:{GOOGLE_SUB}")["id"] == account_doc["user_id"])
    check("세션 토큰으로 사용자를 복원할 수 있다",
          accounts.user_from_token(result["token"])["id"] == account_doc["user_id"])
    check("세션 토큰이 응답 본문에만 있다 (URL·리다이렉트 없음)",
          not hasattr(result, "headers"))

    # --- 재로그인 ---
    again = api.auth_firebase_session(api.FirebaseSession(id_token=token()), make_request())
    check("재로그인해도 계정은 하나", len(fake.accounts.docs) == 1, f"{len(fake.accounts.docs)}개")
    check("재로그인 시 같은 user_id (기록이 이어진다)",
          again["user"]["id"] == account_doc["user_id"])
    check("재로그인은 created=false", again.get("created") is False)
    check("login_count 가 올라간다", fake.accounts.docs[0].get("login_count") == 2,
          f"login_count={fake.accounts.docs[0].get('login_count')}")
    check("로그인마다 세션이 새로 생긴다", len(fake.sessions.docs) == 2)

    # --- 실패 경로 ---
    firebase_auth._verify_signature_local = lambda t: (_ for _ in ()).throw(
        firebase_auth.FirebaseAuthError("토큰 서명이 올바르지 않습니다.", "bad_signature"))
    denied = api.auth_firebase_session(api.FirebaseSession(id_token=token()), make_request())
    check("위조 토큰은 401", getattr(denied, "status_code", 200) == 401)
    check("계정이 늘어나지 않는다", len(fake.accounts.docs) == 1)
    firebase_auth._verify_signature_local = lambda t: True

    # 우리 쪽 문제는 503 이어야 합니다. 401 이면 사용자가 자기 계정을 의심하며
    # 계속 다시 눌러보게 됩니다.
    for code, label in (("network", "구글에 연결 실패"),
                        ("api_key_restricted", "API 키 리퍼러 제한"),
                        ("certs_failed", "공개키를 못 받음")):
        firebase_auth._verify_signature_local = (
            lambda t, c=code: (_ for _ in ()).throw(
                firebase_auth.FirebaseAuthError("서버 쪽 문제", c)))
        broken = api.auth_firebase_session(
            api.FirebaseSession(id_token=token()), make_request())
        check(f"{label} 은 503 (401 이 아님)", getattr(broken, "status_code", 200) == 503)
    firebase_auth._verify_signature_local = lambda t: True

    _unconfigure()
    off = api.auth_firebase_session(api.FirebaseSession(id_token=token()), make_request())
    check("미설정이면 503 (401 이 아님 — 사용자 잘못이 아닙니다)",
          getattr(off, "status_code", 200) == 503)
finally:
    accounts._db = _real_db
    firebase_auth._verify_signature_local = _real_local

# ---------------------------------------------------------------------------
# 9. 예전 OAuth 계정 이어받기 — 로그인 방식을 바꿔도 데이터가 남아야 합니다
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  9. 예전 OAuth 로 만든 계정을 Firebase 로 이어받는다")
print("=" * 70)

legacy = FakeDB()
accounts._db = lambda: legacy
firebase_auth._verify_signature_local = lambda t: True
_configure()
_overrides["MONGODB_URI"] = "mongodb://localhost:27017"

try:
    # 예전 흐름(google_oauth.fetch_identity)이 만들어 두었을 계정
    before = accounts.upsert_google_account({
        "sub": GOOGLE_SUB, "email": EMAIL, "email_verified": True,
        "name": "예전 이름", "picture": "", "locale": "ko",
    })
    old_id = before["user"]["id"]
    created_ids.append(old_id)

    # 같은 사람이 이번엔 Firebase 로 들어옵니다 (Firebase UID 는 처음 보는 값)
    after = api.auth_firebase_session(api.FirebaseSession(id_token=token()), make_request())

    check("계정이 새로 생기지 않는다", len(legacy.accounts.docs) == 1,
          f"{len(legacy.accounts.docs)}개 — 늘어나면 사용자 기록이 사라져 보입니다")
    check("user_id 가 그대로다 (예측 기록·모의투자 계좌 유지)",
          after["user"]["id"] == old_id, f"{old_id} → {after['user']['id']}")
    check("프로필은 최신 값으로 갱신된다",
          legacy.accounts.docs[0]["display_name"] == "QA 파이어베이스 사용자")
    check("firebase_uid 가 나중에 붙는다",
          legacy.accounts.docs[0].get("firebase_uid") == FIREBASE_UID)
finally:
    accounts._db = _real_db
    firebase_auth._verify_signature_local = _real_local

# ---------------------------------------------------------------------------
# 10. 미설정 폴백 — 회귀 방지의 핵심
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  10. 설정이 없어도 기존 로그인이 멀쩡한가")
print("=" * 70)

_unconfigure()
providers = api.auth_providers()
check("/auth/providers 가 두 방식을 모두 보고한다",
      set(providers) == {"firebase", "google"}, str(list(providers)))
check("둘 다 미설정이면 버튼을 그리지 않게 한다",
      providers["firebase"]["configured"] is False
      and providers["google"]["configured"] is False)

_configure()
check("Firebase 만 설정되면 Mongo 가 없어 아직 꺼져 있다",
      api.auth_providers()["firebase"]["configured"] is False,
      "구글 계정을 저장할 곳이 없으면 로그인은 켜지지 않습니다 (ACCOUNTS.md 2-5)")

_overrides["MONGODB_URI"] = "mongodb://localhost:27017"
ready = api.auth_providers()["firebase"]
check("Firebase + Mongo 가 모두 있으면 켜진다", ready["configured"] is True)
check("프론트가 쓸 웹 설정이 함께 온다",
      ready.get("projectId") == TEST_PROJECT and ready.get("apiKey") == TEST_API_KEY)

_unconfigure()
check("Firebase 미설정이어도 Mongo 세션 조회가 예외 없이 None",
      accounts.user_from_token("아무-토큰") is None)

# 아이디/비번 로그인은 Firebase 와 무관하게 동작해야 합니다
import secrets as _secrets

local_name = f"qa-fb-{_secrets.token_hex(4)}"
registered = users.register(local_name, "test-password-123", "QA 로컬")
check("아이디/비번 가입이 정상", registered.get("ok") is True, str(registered)[:80])
signed_in = users.login(local_name, "test-password-123")
check("아이디/비번 로그인이 정상", signed_in.get("ok") is True, str(signed_in)[:80])
check("발급된 세션으로 사용자 복원",
      users.user_from_token(signed_in["token"])["username"] == local_name)
check("틀린 비밀번호는 거부", users.login(local_name, "wrong").get("ok") is not True)

# ---------------------------------------------------------------------------
# 11. MongoDB 접속 문자열 정규화 (두 로그인 흐름이 함께 쓰는 저장소)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  11. 비밀번호에 특수문자가 있어도 붙는가")
print("=" * 70)

norm = accounts.normalize_mongo_uri
HOST = "cluster0.abcde.mongodb.net"

check("특수문자 없는 URI 는 그대로 둔다",
      norm(f"mongodb+srv://ath:pw123@{HOST}/") == f"mongodb+srv://ath:pw123@{HOST}/")
check("아이디·비밀번호가 없는 로컬 URI 도 그대로",
      norm("mongodb://localhost:27017") == "mongodb://localhost:27017")

# 사람이 Atlas 문자열에 비밀번호를 그대로 붙여 넣는 경우들
check("비밀번호의 @ 를 %40 으로 바꾼다",
      norm(f"mongodb+srv://ath:p@ss@{HOST}/") == f"mongodb+srv://ath:p%40ss@{HOST}/")
check("호스트 경계는 마지막 @ (비밀번호에 @ 가 여러 개여도)",
      norm(f"mongodb+srv://ath:a@b@c@{HOST}/") == f"mongodb+srv://ath:a%40b%40c@{HOST}/")
check("비밀번호의 # 를 %23 으로 바꾼다",
      norm(f"mongodb+srv://ath:p#w@{HOST}/") == f"mongodb+srv://ath:p%23w@{HOST}/")
check("비밀번호의 / 와 : 도 처리한다",
      norm(f"mongodb+srv://ath:a/b:c@{HOST}/") == f"mongodb+srv://ath:a%2Fb%3Ac@{HOST}/")
check("아이디 쪽 특수문자도 처리한다",
      norm(f"mongodb+srv://a@th:pw@{HOST}/") == f"mongodb+srv://a%40th:pw@{HOST}/")
check("한글 비밀번호는 UTF-8 로 인코딩",
      norm(f"mongodb+srv://ath:비밀@{HOST}/")
      == f"mongodb+srv://ath:%EB%B9%84%EB%B0%80@{HOST}/")

# 이미 인코딩해 둔 사람의 비밀번호를 망가뜨리면 안 됩니다 (%40 → %2540)
already = f"mongodb+srv://ath:p%40ss@{HOST}/"
check("이미 %XX 로 적힌 것은 건드리지 않는다", norm(already) == already,
      "두 번 걸면 %2540 이 되어 조용히 인증 실패합니다")
check("두 번 걸어도 결과가 같다 (멱등)",
      norm(norm(f"mongodb+srv://ath:p@ss@{HOST}/"))
      == norm(f"mongodb+srv://ath:p@ss@{HOST}/"))
check("%XX 가 아닌 % 는 인코딩한다",
      norm(f"mongodb+srv://ath:100%pw@{HOST}/") == f"mongodb+srv://ath:100%25pw@{HOST}/")

# 뒤에 붙는 옵션·경로가 보존돼야 합니다 (retryWrites 등)
opts = "?retryWrites=true&w=majority&appName=Cluster0"
check("쿼리 옵션이 보존된다",
      norm(f"mongodb+srv://ath:p@ss@{HOST}/{opts}")
      == f"mongodb+srv://ath:p%40ss@{HOST}/{opts}")
check("경로 없이 쿼리만 있어도 호스트를 정확히 자른다",
      norm(f"mongodb+srv://ath:p@ss@{HOST}{opts}")
      == f"mongodb+srv://ath:p%40ss@{HOST}{opts}")

check("빈 값은 빈 값", norm("") == "")
check("://  가 없으면 그대로 둔다 (판단하지 않음)", norm("그냥-문자열") == "그냥-문자열")

# --- 접속 실패 안내 --------------------------------------------------------
# pymongo 의 실패 원문은 세 노드 상태를 통째로 붙여 2000자가 넘고, 정작 필요한
# "Atlas 에 내 IP 를 등록하세요" 는 어디에도 없습니다. 그 번역이 맞는지 봅니다.
explain = accounts.explain_mongo_error

ATLAS_TLS = ("SSL handshake failed: ac-x.mongodb.net:27017: [SSL: "
             "TLSV1_ALERT_INTERNAL_ERROR] tlsv1 alert internal error (_ssl.c:1081), "
             "Topology Description: <TopologyDescription id: 6a78, servers: "
             "[<ServerDescription ('ac-x.mongodb.net', 27017) server_type: Unknown>]>")

msg = explain(Exception(ATLAS_TLS), "접속 실패")
check("TLS 거부는 Network Access 를 짚어준다", "Network Access" in msg, msg[:70])
check("자리표시자 안내가 잘못 나오지 않는다", "자리표시자" not in msg,
      "원문의 <ServerDescription> 에 걸리면 엉뚱한 곳을 고치게 됩니다")
check("원문 전체를 쏟아내지 않는다", len(msg) < 400, f"{len(msg)}자 (원문 {len(ATLAS_TLS)}자)")

placeholder = explain(Exception("bad URI: mongodb+srv://a:<db_password>@h/"), "접속 실패")
check("자리표시자가 남았을 때는 그걸 짚어준다", "자리표시자" in placeholder, placeholder[:70])

authfail = explain(Exception("bad auth : Authentication failed."), "접속 실패")
check("인증 실패는 Database Access 를 짚어준다", "Database Access" in authfail, authfail[:70])

escaped = explain(Exception("Username and password must be escaped, use quote_plus"), "x")
check("이스케이프 오류는 비밀번호를 짚어준다", "비밀번호" in escaped, escaped[:70])

unknown = explain(Exception("무슨 일인지 모를 오류"), "접속 실패")
check("모르는 오류는 원문을 그대로 보여준다", "무슨 일인지 모를 오류" in unknown, unknown[:70])

# 실제로 pymongo 가 받아들이는지 — 이 검사의 목적입니다.
# mongodb+srv:// 는 파싱만 해도 **DNS SRV 조회를 실제로 날립니다.** 오프라인에서
# 돌아야 하는 검사라 평범한 mongodb:// 로 확인합니다 (정규화 규칙은 같습니다).
try:
    from pymongo.uri_parser import parse_uri

    PASSWORD = "p@ss:w#rd/1"        # 사람이 실제로 쓸 법한, 그리고 URI 를 깨뜨리는 조합
    raw = f"mongodb://ath:{PASSWORD}@{HOST}:27017/{opts}"

    parsed = parse_uri(norm(raw))
    check("pymongo 가 정규화된 URI 를 파싱한다", True)
    check("비밀번호가 원래 글자 그대로 복원된다", parsed["password"] == PASSWORD,
          "여기가 어긋나면 인증이 조용히 실패합니다")
    check("아이디도 그대로", parsed["username"] == "ath")

    # 정규화가 없으면 어떤 입력이 막히는지 — 사용자가 본 그 오류입니다.
    # (pymongo 는 `#` 은 눈감아 주지만 `@` 와 `:` 는 거절합니다.)
    for label, pw in (("비밀번호에 @", "p@ss"), ("비밀번호에 :", "p:ss")):
        bad = f"mongodb://ath:{pw}@{HOST}:27017/"
        try:
            parse_uri(bad)
            check(f"정규화 없이 {label} 는 pymongo 가 거절한다", False,
                  "통과해 버렸습니다 — 이 검사가 의미를 잃었습니다")
        except Exception as exc:                                   # noqa: BLE001
            check(f"정규화 없이 {label} 는 pymongo 가 거절한다",
                  "escaped" in str(exc) or "quote_plus" in str(exc), str(exc)[:60])
        check(f"정규화하면 {label} 도 통과한다",
              parse_uri(norm(bad))["password"] == pw)
except ImportError:
    skip("pymongo 파싱 확인", "pymongo 미설치")
except Exception as exc:                                           # noqa: BLE001
    check("pymongo 가 정규화된 URI 를 파싱한다", False, f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# 검사가 SQLite 에 만든 행 정리 (위 8장 주석 참고)
# ---------------------------------------------------------------------------
skipped_ids = []
with users._conn() as conn:
    for uid in sorted(set(created_ids)):
        # id 를 그대로 믿고 지우면, 검사용 번호가 진짜 계정과 겹쳤을 때 남의
        # 계정을 지웁니다. 검사 대역 밖은 무슨 일이 있어도 건드리지 않습니다.
        if uid < TEST_ID_FLOOR:
            skipped_ids.append(uid)
            continue
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
        conn.execute("DELETE FROM paper_account WHERE user_id = ?", (uid,))

    # 아이디/비번 검사 계정은 AUTOINCREMENT 라 대역이 없습니다 — 이름으로 지웁니다
    conn.execute("DELETE FROM sessions WHERE user_id = ?", (registered["user"]["id"],))
    conn.execute("DELETE FROM paper_account WHERE user_id = ?", (registered["user"]["id"],))
    conn.execute("DELETE FROM users WHERE username = ?", (local_name,))

check("정리 대상이 모두 검사 대역 안이었다", not skipped_ids,
      f"대역 밖이라 건너뜀: {skipped_ids}" if skipped_ids else "")
check("검사가 남긴 그림자 행을 모두 지웠다",
      users.find_by_username(f"google:{GOOGLE_SUB}") is None)
check("아이디/비번 검사 계정도 지웠다", users.find_by_username(local_name) is None)

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"  통과 {len(PASS)} · 실패 {len(FAIL)}"
      + (f" · 건너뜀 {len(SKIP)}" if SKIP else ""))
print("=" * 70)
if FAIL:
    for name in FAIL:
        print(f"  FAIL  {name}")
sys.exit(1 if FAIL else 0)
