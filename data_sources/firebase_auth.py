"""
구글 로그인 (Firebase Authentication)
------------------------------------
브라우저가 Firebase SDK 로 구글 계정을 고르고, 그 결과로 받은 **Firebase ID
토큰**을 우리 서버로 보냅니다. 이 모듈은 그 토큰이 진짜인지 확인하고 프로필을
꺼내는 일만 합니다.

왜 흐름을 바꿨나 (기존 data_sources/google_oauth.py 와의 차이)
    OAuth 흐름은 우리가 구글 콘솔에 클라이언트 시크릿과 리디렉션 URI 를 등록하고,
    state·PKCE·nonce 를 직접 관리해야 했습니다. 리디렉션 URI 가 글자 하나만 달라도
    redirect_uri_mismatch 로 막히는 게 설정에서 가장 오래 걸리는 지점이었습니다.
    Firebase 는 계정 선택 창을 SDK 가 띄우고, 우리는 돌아온 토큰만 검증합니다.
    등록할 것은 '승인된 도메인'(localhost 는 기본 포함) 하나뿐이고, 시크릿이
    없습니다. 기존 흐름은 지웠던 게 아니라 그대로 남아 있어, Firebase 설정이
    없으면 예전 방식으로 자동 폴백합니다 (api.py 의 로그인 설정 응답 참고).

서명 검증을 반드시 하는 이유
    google_oauth.py 는 id_token 을 **우리가 연 TLS 연결로** 구글에서 직접 받기
    때문에 서명 검증을 생략할 수 있었습니다. 여기서는 다릅니다 — 토큰이
    브라우저를 거쳐 옵니다. 누구나 아무 JSON 이나 base64 로 엮어 POST 할 수
    있으므로, 서명을 확인하지 않으면 로그인 자체가 무의미해집니다.
    (google_oauth.py 모듈 docstring 이 예고한 바로 그 경우입니다.)

    검증 경로는 둘이고, 앞의 것이 되면 앞의 것을 씁니다.
      1) 로컬 검증  구글이 공개한 X.509 인증서로 RS256 서명을 직접 확인합니다.
                    `cryptography` 가 있어야 합니다. 네트워크는 인증서를 받을
                    때만(캐시 유효 기간 동안 1회) 씁니다.
      2) 구글에 위임 `cryptography` 가 없으면 Identity Toolkit 의 accounts:lookup
                    에 토큰을 넘겨 구글이 판정하게 합니다. 우리가 직접 연 TLS
                    연결이라 응답의 출처는 보장되고, 구글은 자기가 서명하지 않은
                    토큰과 다른 프로젝트의 토큰을 거부합니다.

    둘 다 불가능하면 **로그인을 통과시키지 않습니다.** 검증을 건너뛰는 경로는
    이 파일에 없습니다.
"""

import base64
import json
import time

from data_sources import credentials, http_client

# Firebase ID 토큰 서명에 쓰인 공개키. kid 로 골라 씁니다.
CERTS_ENDPOINT = ("https://www.googleapis.com/robot/v1/metadata/x509/"
                  "securetoken@system.gserviceaccount.com")

# 구글에 검증을 위임할 때 쓰는 엔드포인트 (웹 API 키 필요)
LOOKUP_ENDPOINT = "https://identitytoolkit.googleapis.com/v1/accounts:lookup"

# 우리 시계가 조금 빠를 수 있으니 exp/iat 검사에 여유를 둡니다
CLOCK_SKEW_SECONDS = 120

# 인증서 캐시 기본 수명. 응답의 Cache-Control: max-age 가 있으면 그 값을 씁니다.
CERTS_TTL_SECONDS = 3600

# Firebase 는 ID 토큰을 이 조합으로만 발급합니다
EXPECTED_ALG = "RS256"

# 우리가 받아들이는 로그인 수단. 화면에는 구글 버튼만 있으므로 구글만 허용합니다.
EXPECTED_SIGN_IN_PROVIDER = "google.com"


class FirebaseAuthError(Exception):
    """로그인 실패 — 사용자에게 보여줄 만한 사유를 담습니다."""

    def __init__(self, message: str, code: str = "firebase_auth_failed"):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def project_id() -> str:
    return credentials.get("FIREBASE_PROJECT_ID", "")


def api_key() -> str:
    """웹 API 키.

    Firebase 웹 API 키는 **비밀이 아닙니다** — 프론트엔드 번들에 그대로 실리는
    값이라 브라우저에서 누구나 볼 수 있습니다. 접근 제어는 이 키가 아니라 보안
    규칙과 '승인된 도메인'이 합니다. 그래서 설정 화면에서도 가리지 않습니다.
    """
    return credentials.get("FIREBASE_API_KEY", "")


def auth_domain() -> str:
    """계정 선택 창을 띄우는 도메인. 기본값은 프로젝트의 firebaseapp.com 입니다."""
    explicit = credentials.get("FIREBASE_AUTH_DOMAIN", "")
    if explicit:
        return explicit
    pid = project_id()
    return f"{pid}.firebaseapp.com" if pid else ""


def configured() -> bool:
    return bool(project_id() and api_key())


def web_config() -> dict:
    """프론트엔드가 initializeApp() 에 그대로 넣는 값.

    셋 다 공개값입니다. 이걸 서버가 내려주는 이유는 보안이 아니라 설정 창구를
    하나로 두기 위해서입니다 — 사용자가 api_keys.json 한 곳만 채우면 되고,
    Next 빌드를 다시 하지 않아도 됩니다.
    """
    return {
        "apiKey": api_key(),
        "authDomain": auth_domain(),
        "projectId": project_id(),
    }


def status() -> dict:
    if not project_id():
        return {"configured": False, "reason": "FIREBASE_PROJECT_ID 가 설정되지 않았습니다."}
    if not api_key():
        return {"configured": False, "reason": "FIREBASE_API_KEY 가 설정되지 않았습니다."}
    return {"configured": True, "reason": "", **web_config()}


def issuer() -> str:
    return f"https://securetoken.google.com/{project_id()}"


# ---------------------------------------------------------------------------
# JWT 파싱
# ---------------------------------------------------------------------------

def _b64url_decode(segment: str) -> bytes:
    """패딩이 빠진 base64url 을 복원해 디코딩합니다."""
    segment += "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment.encode("ascii"))


def _split(id_token: str) -> tuple[str, str, str]:
    parts = (id_token or "").split(".")
    if len(parts) != 3 or not all(parts):
        raise FirebaseAuthError("ID 토큰 형식이 올바르지 않습니다.", "bad_id_token")
    return parts[0], parts[1], parts[2]


def _decode_json_segment(segment: str, label: str) -> dict:
    try:
        value = json.loads(_b64url_decode(segment))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FirebaseAuthError(f"ID 토큰의 {label} 를 해석할 수 없습니다.",
                                "bad_id_token") from exc
    if not isinstance(value, dict):
        raise FirebaseAuthError(f"ID 토큰의 {label} 가 객체가 아닙니다.", "bad_id_token")
    return value


def decode_header(id_token: str) -> dict:
    return _decode_json_segment(_split(id_token)[0], "헤더")


def decode_payload(id_token: str) -> dict:
    """서명 검증 없이 페이로드만 꺼냅니다. **단독으로 신뢰하면 안 됩니다.**"""
    return _decode_json_segment(_split(id_token)[1], "페이로드")


# ---------------------------------------------------------------------------
# 클레임 검증
# ---------------------------------------------------------------------------

def validate_claims(payload: dict, now: float | None = None) -> dict:
    """iss / aud / exp / iat / sub / email 검증. 통과하면 payload 를 돌려줍니다.

    서명이 맞아도 이 검사는 따로 필요합니다. 다른 Firebase 프로젝트에서 발급된
    토큰도 서명은 같은 키로 유효하기 때문입니다 — 프로젝트를 가르는 건 aud/iss 뿐.
    """
    now = time.time() if now is None else now
    pid = project_id()

    if payload.get("iss") != issuer():
        raise FirebaseAuthError("발급자(iss)가 이 Firebase 프로젝트가 아닙니다.", "bad_issuer")

    if payload.get("aud") != pid:
        # 다른 프로젝트에 발급된 토큰을 우리 서버에 들이미는 경우 — 반드시 막아야 합니다
        raise FirebaseAuthError("이 프로젝트에 발급된 토큰이 아닙니다(aud 불일치).",
                                "bad_audience")

    try:
        expires = float(payload.get("exp", 0))
    except (TypeError, ValueError):
        expires = 0
    if expires + CLOCK_SKEW_SECONDS < now:
        raise FirebaseAuthError("로그인 정보가 만료됐습니다. 다시 로그인해 주세요.", "expired")

    try:
        issued = float(payload.get("iat", 0))
    except (TypeError, ValueError):
        issued = 0
    if issued - CLOCK_SKEW_SECONDS > now:
        # 미래에 발급된 토큰 — 정상 동작에서는 나올 수 없습니다
        raise FirebaseAuthError("토큰 발급 시각이 올바르지 않습니다.", "bad_issued_at")

    if not payload.get("sub"):
        raise FirebaseAuthError("Firebase 가 사용자 식별자(sub)를 주지 않았습니다.", "no_sub")

    if not payload.get("email"):
        raise FirebaseAuthError("구글 계정에서 이메일을 받지 못했습니다.", "no_email")

    # 미인증 이메일을 받아들이면, 남의 이메일을 적어둔 계정과 잘못 연결될 수 있습니다
    if not payload.get("email_verified"):
        raise FirebaseAuthError("이메일이 확인되지 않은 계정입니다.", "email_unverified")

    return payload


# ---------------------------------------------------------------------------
# 서명 검증 (1) 로컬 — cryptography 가 있을 때
# ---------------------------------------------------------------------------

_certs_cache: dict = {"expires_at": 0.0, "keys": {}}


def _parse_max_age(header: str) -> int:
    """Cache-Control 의 max-age 초. 없으면 0."""
    for part in (header or "").split(","):
        key, _, value = part.strip().partition("=")
        if key.lower() == "max-age":
            try:
                return max(0, int(value))
            except ValueError:
                return 0
    return 0


def fetch_certs(force: bool = False) -> dict:
    """kid → PEM 인증서. 캐시가 살아 있으면 네트워크를 쓰지 않습니다."""
    now = time.time()
    if not force and _certs_cache["keys"] and _certs_cache["expires_at"] > now:
        return _certs_cache["keys"]

    res = http_client.get(CERTS_ENDPOINT, timeout=10)
    if res is None:
        raise FirebaseAuthError("구글 공개키 서버에 연결할 수 없습니다.", "network")
    if res.status_code != 200:
        raise FirebaseAuthError(
            f"구글 공개키를 받지 못했습니다 (HTTP {res.status_code}).", "certs_failed")
    try:
        keys = res.json()
    except ValueError as exc:
        raise FirebaseAuthError("구글 공개키 응답을 해석할 수 없습니다.", "certs_failed") from exc
    if not isinstance(keys, dict) or not keys:
        raise FirebaseAuthError("구글 공개키가 비어 있습니다.", "certs_failed")

    ttl = _parse_max_age(res.headers.get("cache-control", "")) or CERTS_TTL_SECONDS
    _certs_cache["keys"] = keys
    _certs_cache["expires_at"] = now + ttl
    return keys


def _verify_signature_local(id_token: str) -> bool:
    """RS256 서명을 직접 확인합니다.

    반환: True = 확인함. cryptography 가 없으면 False (호출부가 폴백합니다).
    서명이 틀리면 예외 — 폴백하지 않습니다.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.hashes import SHA256
        from cryptography.x509 import load_pem_x509_certificate
    except ImportError:
        return False

    header_seg, payload_seg, signature_seg = _split(id_token)
    header = _decode_json_segment(header_seg, "헤더")

    if header.get("alg") != EXPECTED_ALG:
        # alg=none 이나 HS256 으로 바꿔치기하는 고전적인 공격을 여기서 막습니다
        raise FirebaseAuthError(
            f"서명 알고리즘이 {EXPECTED_ALG} 가 아닙니다.", "bad_algorithm")

    kid = header.get("kid", "")
    if not kid:
        raise FirebaseAuthError("토큰에 서명 키 식별자(kid)가 없습니다.", "no_kid")

    certs = fetch_certs()
    if kid not in certs:
        # 구글이 키를 주기적으로 교체합니다. 캐시가 오래됐을 뿐일 수 있으니 한 번 더.
        certs = fetch_certs(force=True)
    if kid not in certs:
        raise FirebaseAuthError("토큰의 서명 키를 구글 공개키에서 찾을 수 없습니다.",
                                "unknown_kid")

    try:
        certificate = load_pem_x509_certificate(certs[kid].encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise FirebaseAuthError("구글 공개키를 읽을 수 없습니다.", "certs_failed") from exc

    signed = f"{header_seg}.{payload_seg}".encode("ascii")
    try:
        certificate.public_key().verify(
            _b64url_decode(signature_seg), signed, padding.PKCS1v15(), SHA256())
    except InvalidSignature as exc:
        raise FirebaseAuthError("토큰 서명이 올바르지 않습니다.", "bad_signature") from exc
    except (ValueError, TypeError) as exc:
        raise FirebaseAuthError("토큰 서명을 검사할 수 없습니다.", "bad_signature") from exc

    return True


# ---------------------------------------------------------------------------
# 서명 검증 (2) 구글에 위임 — cryptography 가 없을 때
# ---------------------------------------------------------------------------

def _verify_via_google(id_token: str, expected_sub: str) -> dict:
    """accounts:lookup 에 토큰을 넘겨 구글이 판정하게 합니다.

    구글이 사용자 레코드를 돌려주면 서명·프로젝트·만료가 모두 통과한 것입니다.
    돌아온 localId 가 토큰의 sub 와 다르면 두 응답이 서로 다른 사람을 말하는
    것이므로 진행하지 않습니다.
    """
    key = api_key()
    if not key:
        raise FirebaseAuthError("FIREBASE_API_KEY 가 없어 토큰을 검증할 수 없습니다.",
                                "not_configured")

    code, data, raw = http_client.post_full(
        f"{LOOKUP_ENDPOINT}?key={key}",
        json_body={"idToken": id_token},
        timeout=15,
    )

    if code == 0:
        raise FirebaseAuthError("구글 서버에 연결할 수 없습니다.", "network")

    if code == 403:
        # 웹 API 키에 HTTP 리퍼러 제한을 걸면 서버에서 부르는 이 요청이 막힙니다.
        # 원인을 모르면 몇 시간을 헤매는 자리라 그대로 알려줍니다.
        raise FirebaseAuthError(
            "웹 API 키가 서버 요청을 거부했습니다. 구글 클라우드 콘솔에서 이 키의 "
            "'애플리케이션 제한'을 없음으로 두거나, python -m pip install cryptography "
            "로 로컬 검증을 켜 주세요.",
            "api_key_restricted")

    if code != 200 or not isinstance(data, dict):
        reason = ""
        if isinstance(data, dict):
            reason = str((data.get("error") or {}).get("message", ""))
        if "INVALID_ID_TOKEN" in reason or "TOKEN_EXPIRED" in reason:
            raise FirebaseAuthError("로그인 정보가 만료됐거나 올바르지 않습니다. "
                                    "다시 로그인해 주세요.", "bad_id_token")
        raise FirebaseAuthError(
            f"토큰 검증에 실패했습니다: {reason or code} {raw[:120]}".strip(),
            "verify_failed")

    users = data.get("users") or []
    if not users:
        raise FirebaseAuthError("구글이 이 토큰의 사용자를 찾지 못했습니다.", "no_user")

    record = users[0]
    if record.get("localId") and record["localId"] != expected_sub:
        raise FirebaseAuthError("구글 응답이 일관되지 않습니다.", "sub_mismatch")
    return record


# ---------------------------------------------------------------------------
# 프로필 추출
# ---------------------------------------------------------------------------

def google_sub(payload: dict) -> str:
    """토큰 안의 **구글 계정 sub** 를 꺼냅니다.

    이 값이 왜 중요한가: Firebase 의 sub 는 Firebase 가 새로 만든 UID 라, 예전
    OAuth 흐름으로 로그인했던 사람에게는 처음 보는 값입니다. 그걸 계정 키로 쓰면
    같은 사람이 새 계정을 받아 예측 기록·모의투자 계좌가 통째로 사라져 보입니다.
    Firebase 토큰은 원래 구글 sub 를 firebase.identities["google.com"] 에 함께
    담아 주므로, 계정 키는 예전과 똑같이 그 값을 씁니다.
    """
    firebase_claim = payload.get("firebase") or {}
    if not isinstance(firebase_claim, dict):
        raise FirebaseAuthError("토큰에 firebase 클레임이 없습니다.", "bad_id_token")

    provider = firebase_claim.get("sign_in_provider", "")
    identities = firebase_claim.get("identities") or {}
    ids = identities.get(EXPECTED_SIGN_IN_PROVIDER) or []

    if provider != EXPECTED_SIGN_IN_PROVIDER or not ids:
        raise FirebaseAuthError("구글 계정으로 로그인해 주세요.", "not_google")

    sub = str(ids[0] or "").strip()
    if not sub:
        raise FirebaseAuthError("구글 계정 식별자를 받지 못했습니다.", "no_sub")
    return sub


def _profile(payload: dict) -> dict:
    """storage.accounts.upsert_google_account() 이 그대로 받는 모양."""
    return {
        "sub": google_sub(payload),
        "firebase_uid": str(payload.get("sub", "")),
        "email": payload.get("email", ""),
        "email_verified": bool(payload.get("email_verified")),
        "name": payload.get("name", ""),
        "picture": payload.get("picture", ""),
        "locale": "",          # Firebase ID 토큰에는 locale 이 없습니다
    }


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def verify_id_token(id_token: str) -> dict:
    """브라우저가 보낸 Firebase ID 토큰 → 검증된 사용자 프로필.

    반환: {sub, firebase_uid, email, email_verified, name, picture, locale}
    실패는 모두 FirebaseAuthError 로 나갑니다.
    """
    if not configured():
        raise FirebaseAuthError("Firebase 로그인이 설정되지 않았습니다.", "not_configured")
    if not id_token:
        raise FirebaseAuthError("로그인 토큰이 비어 있습니다.", "bad_id_token")

    payload = decode_payload(id_token)

    # 서명부터 확인합니다. 순서를 바꾸면 위조 토큰의 내용으로 오류 메시지가
    # 만들어져 (예: aud 불일치) 공격자에게 설정을 알려주게 됩니다.
    if not _verify_signature_local(id_token):
        _verify_via_google(id_token, expected_sub=str(payload.get("sub", "")))

    validate_claims(payload)
    return _profile(payload)
