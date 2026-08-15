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
"""

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
