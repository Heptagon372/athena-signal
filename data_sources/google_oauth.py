"""
구글 로그인 (OAuth 2.0 Authorization Code + PKCE)
------------------------------------------------
왜 One Tap(Google Identity Services)이 아니라 리다이렉트 흐름인가
    One Tap 은 서드파티 쿠키와 iframe 에 의존합니다. 파이어폭스는 Total Cookie
    Protection 이 기본값이라 서드파티 쿠키를 오리진별로 격리하고, 그래서 One Tap 은
    파폭에서 안 뜨거나 조용히 실패합니다. 반면 전체 페이지를 accounts.google.com
    으로 보내는 표준 코드 흐름은 그 페이지에서 구글이 1st-party 라 서드파티 쿠키를
    쓰지 않습니다. 파폭·크롬·사파리에서 똑같이 동작합니다.

보안 요소
    state   서버(Mongo)에 저장하고 콜백에서 1회용으로 소비 — CSRF 차단
    PKCE    code_verifier 를 서버가 보관, S256 challenge 만 구글에 전달 —
            인가 코드가 유출돼도 verifier 없이는 토큰으로 바꿀 수 없습니다
    nonce   id_token 안에 되돌아오는 값을 대조 — id_token 재사용 차단

id_token 서명을 검증하지 않는 이유 (그리고 언제부터 검증해야 하는가)
    id_token 을 브라우저에서 받는 게 아니라, **우리가 직접 연 TLS 연결로 구글
    토큰 엔드포인트에서** 받습니다. 채널 자체가 출처를 보장하므로 서명 검증은
    필수가 아닙니다 (OpenID Connect Core §3.1.3.7 주석 2).
    그래도 iss/aud/exp/nonce/email_verified 는 검증하고 userinfo 로 교차 확인합니다.
    ※ 만약 나중에 id_token 을 프론트엔드나 제3자에게서 받는 구조로 바꾸면,
      그때는 JWKS 로 **서명을 반드시 검증해야 합니다.** 그 경우 cryptography
      의존성이 필요합니다.
"""

import base64
import hashlib
import json
import secrets
import time
from urllib.parse import urlencode

from data_sources import credentials, http_client

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
USERINFO_ENDPOINT = "https://openidconnect.googleapis.com/v1/userinfo"

# 로그인 확인에 필요한 최소 범위. 그 이상 받아두면 유출면만 넓어집니다.
SCOPES = "openid email profile"

# 구글이 id_token 의 iss 로 쓰는 두 가지 표기 — 둘 다 정상입니다
VALID_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

# 우리 시계가 조금 빠를 수 있으니 exp 검사에 여유를 둡니다
CLOCK_SKEW_SECONDS = 120

DEFAULT_REDIRECT_URI = "http://localhost:8000/api/auth/google/callback"


class GoogleAuthError(Exception):
    """로그인 실패 — 사용자에게 보여줄 만한 사유를 담습니다."""

    def __init__(self, message: str, code: str = "google_auth_failed"):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def client_id() -> str:
    return credentials.get("GOOGLE_CLIENT_ID", "")


def client_secret() -> str:
    return credentials.get("GOOGLE_CLIENT_SECRET", "")


def redirect_uri() -> str:
    """구글 콘솔에 등록한 값과 **글자 단위로 같아야** 합니다.

    기본값이 3000(프론트)이 아니라 8000(API)인 이유: Next 의 /api 프록시는
    응답에서 content-type·cache-control 만 되돌려주고 Location 을 버립니다.
    콜백을 프록시로 보내면 구글의 리다이렉트가 거기서 사라집니다.
    """
    return credentials.get("GOOGLE_REDIRECT_URI", "") or DEFAULT_REDIRECT_URI


def configured() -> bool:
    return bool(client_id() and client_secret())


def status() -> dict:
    if not client_id():
        return {"configured": False, "reason": "GOOGLE_CLIENT_ID 가 설정되지 않았습니다."}
    if not client_secret():
        return {"configured": False, "reason": "GOOGLE_CLIENT_SECRET 이 설정되지 않았습니다."}
    return {"configured": True, "reason": "", "redirect_uri": redirect_uri()}


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------

def _b64url(raw: bytes) -> str:
    """RFC 7636 이 요구하는 base64url — 패딩(=)을 붙이지 않습니다."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def new_pkce() -> tuple[str, str]:
    """(code_verifier, code_challenge) — challenge 는 S256."""
    verifier = _b64url(secrets.token_bytes(32))          # 43자, RFC 7636 허용 범위
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(state: str, code_challenge: str, nonce: str,
                        login_hint: str = "") -> str:
    params = {
        "client_id": client_id(),
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        # 계정이 여러 개인 사람이 원하는 계정을 고를 수 있게 합니다.
        # 이게 없으면 브라우저에 로그인된 계정으로 조용히 넘어가 버립니다.
        "prompt": "select_account",
    }
    if login_hint:
        params["login_hint"] = login_hint
    return f"{AUTH_ENDPOINT}?{urlencode(params)}"


# ---------------------------------------------------------------------------
# id_token
# ---------------------------------------------------------------------------

def decode_id_token(id_token: str) -> dict:
    """서명 검증 없이 페이로드만 꺼냅니다 (모듈 docstring 의 근거 참고)."""
    parts = (id_token or "").split(".")
    if len(parts) != 3:
        raise GoogleAuthError("id_token 형식이 올바르지 않습니다.", "bad_id_token")
    body = parts[1]
    body += "=" * (-len(body) % 4)          # base64 패딩 복원
    try:
        return json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
    except (ValueError, json.JSONDecodeError) as exc:
        raise GoogleAuthError("id_token 을 해석할 수 없습니다.", "bad_id_token") from exc


def validate_claims(payload: dict, expected_nonce: str = "",
                    now: float | None = None) -> dict:
    """iss / aud / exp / nonce / email_verified 검증. 통과하면 payload 를 돌려줍니다."""
    now = time.time() if now is None else now

    if payload.get("iss") not in VALID_ISSUERS:
        raise GoogleAuthError("발급자(iss)가 구글이 아닙니다.", "bad_issuer")

    if payload.get("aud") != client_id():
        # 다른 앱에 발급된 토큰을 우리 앱에 들이미는 경우 — 반드시 막아야 합니다
        raise GoogleAuthError("이 앱에 발급된 토큰이 아닙니다(aud 불일치).", "bad_audience")

    try:
        expires = float(payload.get("exp", 0))
    except (TypeError, ValueError):
        expires = 0
    if expires + CLOCK_SKEW_SECONDS < now:
        raise GoogleAuthError("구글 인증이 만료됐습니다. 다시 로그인해 주세요.", "expired")

    if expected_nonce and payload.get("nonce") != expected_nonce:
        raise GoogleAuthError("인증 요청과 응답이 짝이 맞지 않습니다(nonce).", "bad_nonce")

    if not payload.get("sub"):
        raise GoogleAuthError("구글이 사용자 식별자(sub)를 주지 않았습니다.", "no_sub")

    if not payload.get("email"):
        raise GoogleAuthError("구글 계정에서 이메일을 받지 못했습니다.", "no_email")

    # 미인증 이메일을 받아들이면, 남의 이메일을 적어둔 계정과 잘못 연결될 수 있습니다
    if not payload.get("email_verified"):
        raise GoogleAuthError("이메일이 확인되지 않은 구글 계정입니다.", "email_unverified")

    return payload


# ---------------------------------------------------------------------------
# 코드 교환
# ---------------------------------------------------------------------------

def exchange_code(code: str, code_verifier: str) -> dict:
    """인가 코드를 토큰으로 교환합니다."""
    if not configured():
        raise GoogleAuthError("구글 로그인이 설정되지 않았습니다.", "not_configured")

    status_code, data, raw = http_client.post_form_full(
        TOKEN_ENDPOINT,
        data={
            "code": code,
            "client_id": client_id(),
            "client_secret": client_secret(),
            "redirect_uri": redirect_uri(),
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        },
        timeout=15,
    )

    if status_code == 0:
        raise GoogleAuthError("구글 서버에 연결할 수 없습니다.", "network")

    if status_code != 200 or not data:
        reason = (data or {}).get("error", "") if isinstance(data, dict) else ""
        detail = (data or {}).get("error_description", "") if isinstance(data, dict) else ""
        # 설정 실수 두 가지는 원인이 명확하니 그대로 알려줍니다 — 여기서 헤매는
        # 시간이 가장 깁니다 (콘솔 등록값과 글자 하나만 달라도 이 오류가 납니다).
        if reason == "redirect_uri_mismatch":
            raise GoogleAuthError(
                f"구글 콘솔의 리디렉션 URI 가 다릅니다. 이 값을 등록해 주세요: {redirect_uri()}",
                "redirect_uri_mismatch")
        if reason == "invalid_client":
            raise GoogleAuthError(
                "클라이언트 ID/시크릿이 올바르지 않습니다.", "invalid_client")
        if reason == "invalid_grant":
            raise GoogleAuthError(
                "인가 코드가 만료되었거나 이미 사용됐습니다. 다시 로그인해 주세요.",
                "invalid_grant")
        raise GoogleAuthError(
            f"토큰 교환 실패: {reason or status_code} {detail or raw[:120]}".strip(),
            "token_exchange_failed")

    if not data.get("id_token"):
        raise GoogleAuthError("구글이 id_token 을 주지 않았습니다.", "no_id_token")

    return data


def fetch_userinfo(access_token: str) -> dict:
    """userinfo 로 프로필을 한 번 더 확인합니다 (교차 확인).

    실패해도 로그인을 막지 않습니다 — id_token 만으로도 신원 확인은 끝났고,
    이건 프로필 사진·이름을 보강하고 sub 를 대조하기 위한 보조 경로입니다.
    """
    if not access_token:
        return {}
    return http_client.get_json(
        USERINFO_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    ) or {}


def fetch_identity(code: str, code_verifier: str, nonce: str = "") -> dict:
    """인가 코드 → 검증된 사용자 프로필.

    반환: {sub, email, email_verified, name, picture, locale}
    storage.accounts.upsert_google_account() 이 그대로 받는 모양입니다.
    """
    tokens = exchange_code(code, code_verifier)
    claims = validate_claims(decode_id_token(tokens["id_token"]), expected_nonce=nonce)

    profile = {
        "sub": claims["sub"],
        "email": claims.get("email", ""),
        "email_verified": bool(claims.get("email_verified")),
        "name": claims.get("name", ""),
        "picture": claims.get("picture", ""),
        "locale": claims.get("locale", ""),
    }

    info = fetch_userinfo(tokens.get("access_token", ""))
    if info:
        # sub 이 다르면 두 응답이 서로 다른 사람을 말하고 있다는 뜻입니다.
        # 정상 동작에서는 일어날 수 없으니 진행하지 않습니다.
        if info.get("sub") and info["sub"] != profile["sub"]:
            raise GoogleAuthError("구글 응답이 일관되지 않습니다.", "sub_mismatch")
        for key in ("name", "picture", "locale", "email"):
            if not profile.get(key) and info.get(key):
                profile[key] = info[key]

    return profile
