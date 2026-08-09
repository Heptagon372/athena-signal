"""
외부 요청 공통 클라이언트
------------------------
모든 크롤러/시세 공급자가 같은 세션·타임아웃·User-Agent를 쓰도록 한 곳에 모았습니다.
네트워크가 끊겨도 앱 전체가 죽지 않도록, 실패는 예외 대신 None/빈 값으로 흡수하는
헬퍼(get_json / get_text)를 함께 제공합니다.
"""

import json
import requests

from config import HTTP_TIMEOUT, USER_AGENT

_session = requests.Session()
_session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
})


def get(url: str, *, headers: dict | None = None, params: dict | None = None,
        timeout: int | None = None) -> requests.Response | None:
    try:
        return _session.get(
            url, headers=headers or {}, params=params,
            timeout=timeout or HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return None


def get_json(url: str, *, headers: dict | None = None, params: dict | None = None,
             timeout: int | None = None):
    """JSON 응답을 파싱해 반환. 실패(네트워크/비200/비JSON)하면 None."""
    res = get(url, headers=headers, params=params, timeout=timeout)
    if res is None or res.status_code != 200:
        return None
    try:
        return res.json()
    except (ValueError, json.JSONDecodeError):
        return None


def get_full(url: str, *, headers: dict | None = None, params: dict | None = None,
             timeout: int | None = None) -> tuple[int, dict | None, str]:
    """(상태코드, JSON, 원문) — **오류 본문까지 필요할 때** 씁니다.

    get_json 은 비200을 None 으로 삼켜서 "응답 없음"밖에 알 수 없습니다.
    증권사 API는 거부 사유를 비200 응답 본문에 담아 보내므로, 진단에는
    그 본문이 반드시 필요합니다.
    """
    res = get(url, headers=headers, params=params, timeout=timeout)
    if res is None:
        return 0, None, "네트워크 연결 실패"
    try:
        return res.status_code, res.json(), res.text[:500]
    except (ValueError, json.JSONDecodeError):
        return res.status_code, None, res.text[:500]


def post_full(url: str, *, json_body: dict | None = None, headers: dict | None = None,
              timeout: int | None = None) -> tuple[int, dict | None, str]:
    """(상태코드, JSON, 원문) — POST 판."""
    try:
        res = _session.post(url, json=json_body, headers=headers or {},
                            timeout=timeout or HTTP_TIMEOUT)
    except requests.RequestException as exc:
        return 0, None, f"네트워크 연결 실패: {exc}"
    try:
        return res.status_code, res.json(), res.text[:500]
    except (ValueError, json.JSONDecodeError):
        return res.status_code, None, res.text[:500]


def post_form(url: str, *, data: dict | None = None, headers: dict | None = None,
              timeout: int | None = None):
    """application/x-www-form-urlencoded POST (OAuth2 토큰 발급 등)."""
    try:
        res = _session.post(url, data=data, headers=headers or {},
                            timeout=timeout or HTTP_TIMEOUT)
    except requests.RequestException:
        return None
    if res.status_code != 200:
        return None
    try:
        return res.json()
    except (ValueError, json.JSONDecodeError):
        return None


def post_form_full(url: str, *, data: dict | None = None, headers: dict | None = None,
                   timeout: int | None = None) -> tuple[int, dict | None, str]:
    """(상태코드, JSON, 원문) — form POST 판.

    post_form 은 비200을 None 으로 삼켜서 "실패했다"밖에 알 수 없습니다. OAuth2
    토큰 엔드포인트는 거부 사유를 400 본문의 `error` 에 담아 보냅니다
    (invalid_grant / redirect_uri_mismatch / invalid_client …). 이걸 못 읽으면
    설정이 틀렸는지 코드가 만료된 건지 구분할 수 없어 진단이 불가능합니다.
    """
    try:
        res = _session.post(url, data=data, headers=headers or {},
                            timeout=timeout or HTTP_TIMEOUT)
    except requests.RequestException as exc:
        return 0, None, f"네트워크 연결 실패: {exc}"
    try:
        return res.status_code, res.json(), res.text[:500]
    except (ValueError, json.JSONDecodeError):
        return res.status_code, None, res.text[:500]


def post_json(url: str, *, json_body: dict | None = None, headers: dict | None = None,
              timeout: int | None = None):
    """POST 후 JSON 응답을 반환. 실패하면 None. (KIS 토큰 발급 등에 사용)"""
    try:
        res = _session.post(
            url, json=json_body, headers=headers or {},
            timeout=timeout or HTTP_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if res.status_code != 200:
        return None
    try:
        return res.json()
    except (ValueError, json.JSONDecodeError):
        return None


def get_text(url: str, *, headers: dict | None = None, params: dict | None = None,
             encoding: str | None = None, timeout: int | None = None) -> str | None:
    """HTML/텍스트 응답을 반환. encoding을 지정하면 그 인코딩으로 디코드합니다."""
    res = get(url, headers=headers, params=params, timeout=timeout)
    if res is None or res.status_code != 200:
        return None
    if encoding:
        return res.content.decode(encoding, errors="replace")
    return res.text
