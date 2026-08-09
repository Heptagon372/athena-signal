"""
API 자격증명 통합 관리
---------------------
여러 공식 API 키를 한 곳에서 읽고, 어떤 것이 설정됐는지 알려줍니다.

읽는 순서
    1) 환경변수
    2) 프로젝트 폴더의 `api_keys.json` (setup_api.bat 이 만듭니다)

`api_keys.json` 은 사용자 PC에만 존재하며 서버 밖으로 나가지 않습니다.
실수로 공유되지 않도록 .gitignore 에 넣어두세요.

지원 API
    toss    토스증권 Open API      https://developers.tossinvest.com
    kis     한국투자증권 KIS       https://apiportal.koreainvestment.com
    reddit  Reddit API             https://www.reddit.com/prefs/apps
    naver   네이버 검색 API        https://developers.naver.com
    krx     KRX Data Marketplace   http://openapi.krx.co.kr
    datago  공공데이터포털         https://www.data.go.kr
"""

import json
import os
from pathlib import Path

from config import BASE_DIR

KEY_FILE = BASE_DIR / "api_keys.json"

# 각 API가 필요로 하는 항목과 발급처
PROVIDERS = {
    "toss": {
        "label": "토스증권 Open API",
        "fields": ["TOSS_CLIENT_ID", "TOSS_CLIENT_SECRET"],
        "portal": "https://developers.tossinvest.com",
        "gives": "현재가 · 캔들(1분/일) · 종목마스터 · 장 운영 캘린더 · 호가",
    },
    "kis": {
        "label": "한국투자증권 KIS",
        "fields": ["KIS_APP_KEY", "KIS_APP_SECRET"],
        "portal": "https://apiportal.koreainvestment.com",
        "gives": "현재가 · 일봉 · 종목별 투자자 매매동향(장중)",
    },
    "reddit": {
        "label": "Reddit API",
        "fields": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"],
        "portal": "https://www.reddit.com/prefs/apps (script 타입으로 생성)",
        "gives": "해외 커뮤니티 여론 (RSS의 429 제한 해소)",
    },
    "naver": {
        "label": "네이버 검색 API",
        "fields": ["NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET"],
        "portal": "https://developers.naver.com/apps/#/register",
        "gives": "뉴스 검색 (종목 관련 기사 수집량 증가)",
    },
    "krx": {
        "label": "KRX Data Marketplace",
        "fields": ["KRX_AUTH_KEY"],
        "portal": "http://openapi.krx.co.kr (API 인증키 신청)",
        "gives": "거래소 공식 통계 데이터",
    },
    "datago": {
        "label": "공공데이터포털",
        "fields": ["DATAGO_SERVICE_KEY"],
        "portal": "https://www.data.go.kr (금융위원회_주식시세정보 활용신청)",
        "gives": "금융위 주식시세 (KRX 일별 시세 공식 경로)",
    },
    "google": {
        "label": "구글 로그인 (OAuth)",
        "fields": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
        "portal": "https://console.cloud.google.com (사용자 인증 정보 → OAuth 클라이언트 ID, "
                  "리디렉션 URI: http://localhost:8000/api/auth/google/callback)",
        "gives": "구글 계정으로 로그인 (아이디/비번 없이 바로 시작)",
    },
    "mongo": {
        "label": "MongoDB 계정 저장소",
        "fields": ["MONGODB_URI"],
        "portal": "https://cloud.mongodb.com (무료 M0) 또는 로컬 mongodb://localhost:27017",
        "gives": "구글 계정 저장 · 세션 (여러 PC에서 같은 계정으로 접속)",
    },
}

_file_cache: dict | None = None
_user_env_cache: dict | None = None


def _user_env() -> dict:
    """Windows 사용자 환경변수 원본 (HKCU\\Environment).

    setup_kis.bat 은 setx 로 키를 저장하는데, setx 는 **그 뒤에 새로 열리는**
    프로세스에만 반영됩니다. 그래서 이미 떠 있던 창(에디터 터미널·이전 콘솔)에서
    서버를 재시작하면 키가 하나도 없는 서버가 조용히 뜨고, 화면에는 "KIS 미설정"
    으로만 보입니다 — 계좌 정보가 통째로 0원이 되는 가장 흔한 원인입니다.
    환경변수에도 api_keys.json 에도 없을 때 저장된 원본을 직접 읽습니다.
    """
    global _user_env_cache
    if _user_env_cache is not None:
        return _user_env_cache

    _user_env_cache = {}
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                for i in range(winreg.QueryInfoKey(key)[1]):
                    name, value, _ = winreg.EnumValue(key, i)
                    if isinstance(value, str):
                        _user_env_cache[name] = value
        except (OSError, ImportError):
            pass        # 레지스트리를 못 읽어도 조회는 계속돼야 합니다
    return _user_env_cache


def _from_file() -> dict:
    global _file_cache
    if _file_cache is not None:
        return _file_cache
    if not KEY_FILE.exists():
        _file_cache = {}
        return _file_cache
    try:
        _file_cache = json.loads(KEY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        _file_cache = {}
    return _file_cache


def get(name: str, default: str = "") -> str:
    """환경변수 → api_keys.json → Windows 사용자 환경변수 순으로 조회.

    api_keys.json 에 항목이 있으면 (빈 값이더라도) 거기서 끝냅니다. 파일에
    적어 둔 '꺼짐'을 레지스트리의 옛 값이 되살리면 안 되기 때문입니다 —
    실거래 스위치가 그렇게 켜지는 일은 없어야 합니다.
    """
    value = os.environ.get(name)
    if value:
        return value.strip()
    saved = _from_file()
    if name in saved:
        return str(saved.get(name) or "").strip()
    return str(_user_env().get(name, default) or "").strip()


def get_json(name: str, default=None):
    """설정값이 dict/list 인 항목 조회 (환경변수는 JSON 문자열로 받습니다).

    자동매매의 계약 명세·tr_id 재정의처럼 "코드 수정 없이 바꿔야 하는 표"를
    설정으로 빼기 위한 통로입니다 (config as data).
    """
    raw = os.environ.get(name)
    if not raw:
        value = _from_file().get(name)
        if value is not None:
            return value
        raw = _user_env().get(name)      # get() 과 같은 순서 (환경 → 파일 → 저장된 환경)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return default


def get_bool(name: str, default: bool = False) -> bool:
    """'1' / 'true' / 'yes' / 'on' 을 참으로 봅니다. 그 외는 모두 거짓입니다.

    실거래 스위치처럼 **애매하면 꺼진 것으로 취급해야 하는** 설정에 씁니다.
    """
    raw = get(name, "")
    if not raw:
        file_value = _from_file().get(name)
        if isinstance(file_value, bool):
            return file_value
        raw = str(file_value or "")
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def is_configured(provider: str) -> bool:
    spec = PROVIDERS.get(provider)
    if not spec:
        return False
    return all(get(f) for f in spec["fields"])


def status() -> dict:
    """어떤 API가 설정됐는지 한눈에 보여줍니다 (키 값 자체는 노출하지 않음)."""
    out = {}
    for key, spec in PROVIDERS.items():
        configured = is_configured(key)
        missing = [f for f in spec["fields"] if not get(f)]
        out[key] = {
            "label": spec["label"],
            "configured": configured,
            "portal": spec["portal"],
            "gives": spec["gives"],
            "missing": [] if configured else missing,
        }
    return out


def save(values: dict) -> Path:
    """setup 도구가 호출 — 키를 api_keys.json 에 저장."""
    global _file_cache
    current = dict(_from_file())
    current.update({k: v for k, v in values.items() if v})
    KEY_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    _file_cache = current
    return KEY_FILE


def print_status():
    print("=" * 62)
    print("  아테나 시그널 — API 키 설정 현황")
    print("=" * 62)
    st = status()
    for key, info in st.items():
        mark = "OK " if info["configured"] else "미설정"
        print(f"\n  [{mark}] {info['label']}")
        print(f"        제공: {info['gives']}")
        if not info["configured"]:
            print(f"        필요: {', '.join(info['missing'])}")
            print(f"        발급: {info['portal']}")

    configured = sum(1 for i in st.values() if i["configured"])
    print(f"\n  설정됨 {configured}/{len(st)}")
    print(f"\n  키 파일: {KEY_FILE}")
    print("  설정 방법: setup_api.bat 실행 또는 환경변수 지정")
    print("  ※ 키를 하나도 넣지 않아도 기존 공개 경로로 정상 동작합니다.\n")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print_status()
