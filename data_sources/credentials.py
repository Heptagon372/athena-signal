"""
API 자격증명 통합 관리
---------------------
여러 공식 API 키를 한 곳에서 읽고, 어떤 것이 설정됐는지 알려줍니다.

읽는 순서
    1) 환경변수
    2) 프로젝트 폴더의 `api_keys.json` (아테나.bat → [7] API 키 가 만듭니다)

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

import contextlib
import contextvars
import json
import os
from pathlib import Path

from config import BASE_DIR

KEY_FILE = BASE_DIR / "api_keys.json"


def _harden(path) -> None:
    """비밀이 든 파일을 소유자만 읽게 (chmod 600).

    security.harden_file() 과 같은 일을 합니다. 그 모듈을 import 하지 않는 이유는
    security 가 이 모듈을 import 하기 때문입니다 (순환). 세 줄짜리라 여기 둡니다.
    """
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# 사용자 오버레이
# ---------------------------------------------------------------------------
# 로그인한 사용자가 자기 계정에 저장해 둔 키(storage/user_credentials.py)를
# 요청·자동매매 회전 동안만 서버 키 **위에** 겹쳐 씁니다. get() 호출부 48곳을
# 고치지 않기 위한 장치입니다 — 호출부는 지금까지처럼 get() 만 부릅니다.
#
# ContextVar 인 이유
#     전역 dict 로 하면 동시 요청끼리 키가 섞입니다 (A 의 요청이 B 의 KIS 키로
#     주문을 내는 사고). ContextVar 는 요청(스레드풀 실행 컨텍스트)마다 복사본을
#     갖기 때문에, 요청이 끝나면 그 컨텍스트째 버려집니다 — 리셋을 잊어도
#     다음 요청으로 새지 않습니다. 자동매매 루프처럼 오래 사는 스레드에서는
#     use_user() 컨텍스트 매니저가 명시적으로 되돌립니다.
#
# 여기서도 화이트리스트를 다시 검사하는 이유
#     저장 쪽(user_credentials.save_keys)이 이미 거르지만, 옛 코드로 저장된
#     문서나 다른 경로로 심어진 값이 있을 수 있습니다. 사용자 값이 MONGODB_URI
#     같은 서버 설정을 갈아끼우는 일은 **읽는 쪽에서도** 막아야 합니다.
_user_overlay: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "athena_user_credentials", default=None)


def _overlay_allowed(name: str) -> bool:
    # storage.user_credentials 를 import 하면 순환(storage → credentials → storage)
    # 이라, 허용 목록을 여기서 직접 계산합니다. 기준은 같습니다: 서버 부트스트랩
    # (mongo/firebase/google/공개 오리진)을 뺀 제공자 필드 + KIS 거래 설정.
    if name in ("KIS_ACCOUNT", "KIS_DERIV_ACCOUNT", "KIS_MOCK", "KIS_LIVE_TRADING"):
        return True
    for provider in ("toss", "kis", "reddit", "naver", "krx", "datago"):
        if name in PROVIDERS[provider]["fields"]:
            return True
    return False


@contextlib.contextmanager
def use_user(values: dict | None):
    """이 블록 안에서 get()/get_bool()/get_json() 이 사용자 키를 먼저 봅니다."""
    token = _user_overlay.set(dict(values) if values else None)
    try:
        yield
    finally:
        _user_overlay.reset(token)


def attach_user(values: dict | None):
    """현재 컨텍스트에 오버레이를 장착합니다 (요청 처리용 — 리셋 없음).

    FastAPI 의 동기 엔드포인트는 요청마다 **복사된 컨텍스트**의 스레드풀에서
    돌기 때문에, 여기서 set 한 값은 요청이 끝나면 컨텍스트째 사라집니다.
    오래 사는 스레드(자동매매 루프 등)에서는 이걸 쓰면 안 되고 use_user() 를
    써야 합니다 — 리셋이 없어 다음 회전까지 남기 때문입니다.
    """
    _user_overlay.set(dict(values) if values else None)


def _from_overlay(name: str) -> str | None:
    """오버레이에 값이 있으면 그 값 (화이트리스트 통과 시). 없으면 None."""
    overlay = _user_overlay.get()
    if not overlay or name not in overlay:
        return None
    if not _overlay_allowed(name):
        return None
    return str(overlay[name] or "").strip()

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
    "firebase": {
        "label": "구글 로그인 (Firebase)",
        "fields": ["FIREBASE_PROJECT_ID", "FIREBASE_API_KEY"],
        "portal": "https://console.firebase.google.com (Authentication → Google 사용 설정 → "
                  "프로젝트 설정 → 내 앱 → 웹 앱 추가 후 firebaseConfig 확인)",
        "gives": "구글 계정 클릭 한 번으로 로그인 (시크릿·리디렉션 URI 등록 불필요)",
    },
    "google": {
        "label": "구글 로그인 (OAuth · 구버전)",
        "fields": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"],
        "portal": "https://console.cloud.google.com (사용자 인증 정보 → OAuth 클라이언트 ID, "
                  "리디렉션 URI: http://localhost:8000/api/auth/google/callback)",
        "gives": "Firebase 를 쓰지 않을 때의 예전 로그인 경로 (폴백)",
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

    예전 setup_kis.bat 은 setx 로 키를 저장했는데, setx 는 **그 뒤에 새로 열리는**
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
    """사용자 오버레이 → 환경변수 → api_keys.json → Windows 사용자 환경변수.

    오버레이에 항목이 있으면 (빈 값이더라도) 거기서 끝냅니다. 빈 값으로도 끝내는
    이유: 사용자 컨텍스트에서 "이 값 없음"을 서버 값이 되살리면 안 되는 키가
    있습니다 — 구글 계정의 KIS_LIVE_TRADING 이 서버의 '켜짐'을 물려받으면
    남의 실거래 스위치가 켜집니다. (엔진이 안전 기본값을 명시적으로 채웁니다.)

    api_keys.json 도 같은 이유로 항목이 있으면 (빈 값이더라도) 거기서 끝냅니다.
    파일에 적어 둔 '꺼짐'을 레지스트리의 옛 값이 되살리면 안 됩니다.
    """
    from_user = _from_overlay(name)
    if from_user is not None:
        return from_user
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
    raw = _from_overlay(name)
    if not raw:
        raw = os.environ.get(name)
    if not raw:
        value = _from_file().get(name)
        if value is not None:
            return value
        raw = _user_env().get(name)      # get() 과 같은 순서 (오버레이 → 환경 → 파일 → 저장된 환경)
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
    from_user = _from_overlay(name)
    if from_user is not None:
        # 오버레이가 빈 값이어도 여기서 끝냅니다(거짓). 아래 파일 폴백으로
        # 흘려보내면, 사용자 컨텍스트의 '꺼짐'을 서버 파일의 '켜짐'이
        # 되살립니다 — 실거래 스위치에서 절대 있어선 안 되는 일입니다.
        return from_user.strip().lower() in ("1", "true", "yes", "on", "y")
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
    """setup 도구가 호출 — 키를 api_keys.json 에 저장.

    저장 후 소유자만 읽도록 권한을 좁힙니다. 이 파일에는 MONGODB_URI 와
    ATHENA_CRED_KEY 가 들어가는데, 후자는 **모든 사용자의 KIS 키를 푸는
    복호화 키**입니다. 기본 umask(022)로 만들어지면 같은 리눅스 서버의 다른
    계정이 그냥 읽을 수 있고, 그 순간 Atlas 에 암호문으로 보낸 의미가
    사라집니다. (윈도우에서는 개념이 달라 아무 일도 하지 않습니다)
    """
    global _file_cache
    current = dict(_from_file())
    current.update({k: v for k, v in values.items() if v})
    KEY_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    _harden(KEY_FILE)
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
    print("  설정 방법: 아테나.bat → [7] API 키, 또는 환경변수 지정")
    print("  ※ 키를 하나도 넣지 않아도 기존 공개 경로로 정상 동작합니다.\n")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print_status()
