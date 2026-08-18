"""
사용자별 API 키 저장소 (MongoDB + 암호화)
----------------------------------------
지금까지 API 키는 서버 PC 의 `api_keys.json` 하나였습니다 — 누가 로그인하든
같은 키. 이 모듈은 **계정마다 자기 키**를 MongoDB 에 저장하고, 로그인한 요청에서
그 키를 불러와 서버 키 위에 겹쳐 씁니다 (data_sources/credentials.py 의 오버레이).

무엇이 여기로 오고, 무엇이 못 오는가
    옵니다      토스 · KIS · 레딧 · 네이버 · KRX · 공공데이터 키와
                KIS 계좌·모의/실전 스위치 — "사용자의 것"인 값들
    못 옵니다   MONGODB_URI (이 값이 있어야 여길 읽습니다 — 닭과 달걀),
                FIREBASE_* / GOOGLE_* (로그인 자체의 부품),
                ATHENA_PUBLIC_ORIGIN (서버 동작을 바꾸는 값)
    이 구분은 ALLOWED_KEYS 화이트리스트로 강제합니다. 저장할 때 거르고,
    읽는 쪽(credentials._user_overlay)에서도 한 번 더 거릅니다 — 어느 한쪽이
    실수해도 사용자가 서버 인프라 설정을 갈아끼울 수 없어야 합니다.

왜 암호화하는가, 왜 이 방식인가
    KIS 키는 **실계좌 주문 권한**입니다. Atlas 는 남의 컴퓨터라, 평문으로 두면
    Atlas 쪽 사고(권한 실수·유출)가 곧바로 증권 계좌 사고가 됩니다. Fernet
    (AES-128-CBC + HMAC) 으로 값만 암호화하고, 복호화 키는 이 PC 의
    `api_keys.json` 에만 둡니다. Atlas 에는 암호문만 갑니다.

    대가: 복호화 키가 있는 서버만 키를 읽을 수 있습니다. 다른 PC 에서 같은
    Atlas 를 바라보는 서버를 새로 띄우면, 그 서버는 사용자 키를 못 읽고
    "다시 입력해 주세요" 상태가 됩니다. 계정(신원)은 따라가지만 키는 서버에
    묶입니다 — 키를 평문으로 클라우드에 두는 것보다 나은 어색함입니다.

캐시
    로그인한 요청마다 Mongo 를 왕복하면 모든 API 가 느려집니다. user_id 별로
    60초 캐시하고, 저장/삭제 시 그 자리에서 비웁니다. 잘못된 키를 고쳤는데
    1분 동안 옛 키로 도는 상황은 없어야 합니다.
"""

import base64
import threading
import time

from data_sources import credentials
from storage import accounts

# ---------------------------------------------------------------------------
# 화이트리스트
# ---------------------------------------------------------------------------

# 사용자가 자기 계정에 저장할 수 있는 키. credentials.PROVIDERS 의 필드 중
# "서버 부트스트랩"(mongo/firebase/google)을 뺀 전부 + KIS 거래 설정.
USER_PROVIDERS = ("toss", "kis", "reddit", "naver", "krx", "datago")

# PROVIDERS 필드 밖에 있지만 사용자별이어야 말이 되는 값들.
# KIS_ACCOUNT     주문이 나가는 계좌 — 계정마다 달라야 하는 값의 대표
# KIS_DERIV_ACCOUNT 파생 계좌
# KIS_MOCK        모의(1)/실전(0) 서버 선택
# KIS_LIVE_TRADING 실거래 스위치 — 자기 키·자기 계좌에 한해 자기 책임
_EXTRA_KEYS = ("KIS_ACCOUNT", "KIS_DERIV_ACCOUNT", "KIS_MOCK", "KIS_LIVE_TRADING")


def allowed_keys() -> frozenset[str]:
    fields = []
    for provider in USER_PROVIDERS:
        fields.extend(credentials.PROVIDERS[provider]["fields"])
    return frozenset(fields) | frozenset(_EXTRA_KEYS)


ALLOWED_KEYS = allowed_keys()

# 값을 화면에 돌려줄 때 보여주는 꼬리 글자 수 (키 확인용 — 전체는 절대 안 나감)
_MASK_TAIL = 4

# 저장 요청 한 번의 상한. 화이트리스트가 이미 "어떤 키"를 막고 있으니, 여기서는
# "얼마나 큰가"를 막습니다 — 토큰을 훔친 쪽이 Atlas 를 저장소로 쓰는 것을 막는
# 용도라, 정상 사용에는 닿지 않는 넉넉한 값입니다.
_MAX_FIELDS = 40
_MAX_VALUE_LENGTH = 512

CACHE_TTL_SECONDS = 60

_cache: dict[int, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 암호화
# ---------------------------------------------------------------------------

_KEY_NAME = "ATHENA_CRED_KEY"


class EncryptionUnavailable(RuntimeError):
    """cryptography 미설치 등 — 평문 저장으로 조용히 격하하지 않습니다."""


def _fernet():
    """Fernet 인스턴스. 복호화 키가 없으면 만들어 api_keys.json 에 저장합니다.

    키를 자동 생성하는 이유: "암호화 키를 먼저 만드세요" 라는 절차를 사용자에게
    시키면 그 절차가 빠진 채 평문 저장으로 흐르기 쉽습니다. 처음 저장하는 순간
    만들어 두는 쪽이 안전한 기본값입니다.
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise EncryptionUnavailable(
            "cryptography 가 설치되지 않아 API 키를 암호화할 수 없습니다. "
            "python -m pip install cryptography") from exc

    key = credentials.get(_KEY_NAME, "")
    if not key:
        key = Fernet.generate_key().decode("ascii")
        credentials.save({_KEY_NAME: key})
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, TypeError) as exc:
        raise EncryptionUnavailable(
            f"{_KEY_NAME} 가 올바른 Fernet 키가 아닙니다. api_keys.json 을 확인해 주세요.") from exc


def _encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt(token: str) -> str | None:
    """복호화. 키가 바뀌었거나 손상됐으면 None — 예외로 요청을 죽이지 않습니다."""
    try:
        from cryptography.fernet import InvalidToken
    except ImportError:
        return None
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except (EncryptionUnavailable, InvalidToken, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# 저장 / 조회
# ---------------------------------------------------------------------------

def _collection():
    return accounts._db().user_credentials


def _invalidate(user_id: int):
    with _cache_lock:
        _cache.pop(user_id, None)


def save_keys(user_id: int, values: dict) -> dict:
    """사용자 키 저장. 반환: {"ok": bool, "saved": [...], "rejected": [...]}

    빈 문자열은 "그 키 지우기" 로 취급합니다 — 입력창을 비우고 저장하는 것이
    삭제의 자연스러운 표현이기 때문입니다.

    하나라도 거부되면 **아무것도 저장하지 않습니다.** 반쯤 저장하면 사용자는
    오류를 보고 "저장이 안 됐구나" 라고 이해하는데 실제로는 일부가 바뀐,
    화면과 실제가 어긋난 상태가 됩니다.
    """
    values = values or {}
    # 항목 수 상한 — ALLOWED_KEYS 보다 많이 보내는 정상 요청은 없습니다.
    # 훔친 토큰으로 거대한 문서를 밀어 넣어 Atlas 용량을 채우는 경로를 막습니다.
    if len(values) > _MAX_FIELDS:
        return {"ok": False, "saved": [], "removed": [],
                "rejected": [f"항목이 너무 많습니다 (최대 {_MAX_FIELDS}개)"]}

    saved, rejected, removed = [], [], []
    sets, unsets = {}, {}

    for name, value in values.items():
        name = str(name).strip().upper()[:64]
        if name not in ALLOWED_KEYS:
            # 서버 설정 키(MONGODB_URI 등)를 계정 경로로 밀어넣는 시도 포함.
            # 조용히 버리지 않고 알려줍니다 — 오타(KIS_APPKEY)도 이 길로 옵니다.
            rejected.append(name)
            continue
        value = str(value or "").strip()
        if len(value) > _MAX_VALUE_LENGTH:
            # 실제 API 키는 길어야 200자 남짓입니다. 그보다 긴 값은 키가 아닙니다.
            rejected.append(f"{name} (값이 너무 깁니다)")
            continue
        if not value:
            unsets[f"values.{name}"] = ""
            removed.append(name)
        else:
            sets[f"values.{name}"] = _encrypt(value)
            saved.append(name)

    if rejected:
        return {"ok": False, "saved": [], "removed": [], "rejected": rejected}

    if sets or unsets:
        update: dict = {"$set": {**sets, "updated_at": accounts._now()}}
        if unsets:
            update["$unset"] = unsets
        _collection().update_one({"user_id": user_id}, update, upsert=True)
        _invalidate(user_id)

    return {"ok": not rejected, "saved": saved, "removed": removed,
            "rejected": rejected}


def delete_provider(user_id: int, provider: str) -> dict:
    """한 서비스의 키 전부 삭제 (예: kis → KIS_* 필드 일괄)."""
    if provider not in USER_PROVIDERS:
        return {"ok": False, "error": f"알 수 없는 서비스: {provider}"}
    fields = list(credentials.PROVIDERS[provider]["fields"])
    if provider == "kis":
        fields += list(_EXTRA_KEYS)
    _collection().update_one(
        {"user_id": user_id},
        {"$unset": {f"values.{f}": "" for f in fields},
         "$set": {"updated_at": accounts._now()}})
    _invalidate(user_id)
    return {"ok": True, "removed": fields}


def load_keys(user_id: int) -> dict:
    """복호화된 {키: 값}. 실패(미저장·Mongo 다운·키 불일치)는 빈 dict.

    빈 dict 이 안전한 이유: 오버레이가 비면 credentials.get() 이 서버 키로
    폴백할 뿐입니다 — 지금까지의 동작 그대로. 예외를 올려보내면 키 조회
    하나 때문에 시세·예측 요청까지 죽습니다.
    """
    now = time.time()
    with _cache_lock:
        hit = _cache.get(user_id)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            return dict(hit[1])

    try:
        doc = _collection().find_one({"user_id": user_id}) or {}
    except accounts.AccountsUnavailable:
        return {}
    except Exception:                                              # noqa: BLE001
        return {}

    out = {}
    for name, token in (doc.get("values") or {}).items():
        if name not in ALLOWED_KEYS:
            continue                       # 옛 화이트리스트로 저장된 잔재 방어
        plain = _decrypt(token)
        if plain is not None:
            out[name] = plain

    with _cache_lock:
        _cache[user_id] = (now, dict(out))
    return out


def must_use_own_keys(user_id: int) -> bool:
    """KIS 주문에 자기 키·자기 계좌를 반드시 써야 하는 계정인가.

    구글 계정(user_id >= USER_ID_OFFSET)은 항상 그렇습니다 — 외부 사용자일 수
    있습니다. 공개 서버 모드(ATHENA_PUBLIC_ORIGIN 설정)에서는 **로컬 계정도**
    그렇습니다: 아이디/비번 가입이 누구에게나 열려 있어서, 서버 키 폴백을 두면
    방금 가입한 외부인의 자동매매가 서버 주인의 계좌로 주문을 냅니다.

    로컬 전용 서버(공개 오리진 미설정)의 로컬 계정만 지금까지처럼 서버 키를
    씁니다 — 데스크톱 사용자의 기존 동작을 깨지 않습니다.
    """
    if user_id >= accounts.USER_ID_OFFSET:
        return True
    import security
    return security.public_mode()


def overlay_for(user_id: int) -> dict:
    """credentials.use_user() / attach_user() 에 넣을 오버레이를 조립합니다.

    서버 키를 써도 되는 계정 (must_use_own_keys() 거짓 — 로컬 전용 서버의
    로컬 계정)
        저장한 키만 겹칩니다. 나머지는 서버 키로 폴백 — 기존 동작 그대로.

    자기 키가 필수인 계정 (구글 계정, 공개 서버의 모든 계정)
        거래 설정 세 가지에 **안전 기본값을 명시적으로** 채웁니다. 오버레이에
        없으면 credentials.get() 이 서버 값으로 폴백하는데, 서버가 실전
        (KIS_LIVE_TRADING=1)이면 그 스위치를 사용자가 물려받게 됩니다 —
        자기가 켠 적 없는 실거래가 켜지는 일은 없어야 합니다.

        시세용 키(KIS_APP_KEY 등)는 일부러 채우지 않습니다. 공개 시세를 서버
        키로 읽는 것은 무해하고, 주문은 아래 has_own_kis_keys() 게이트가
        따로 막습니다.
    """
    values = load_keys(user_id)
    if must_use_own_keys(user_id):
        values.setdefault("KIS_MOCK", "1")            # 기본은 모의 서버
        values.setdefault("KIS_LIVE_TRADING", "0")    # 실거래는 명시적으로만
        # 파생 계좌가 서버 것으로 폴백하면 안 됩니다. 빈 값이면 kis_trading 이
        # KIS_ACCOUNT(자기 것)로 내려갑니다.
        values.setdefault("KIS_DERIV_ACCOUNT", "")
        values.setdefault("KIS_ACCOUNT", "")          # 서버 계좌번호 차단
    return values


def has_own_kis_keys(user_id: int) -> bool:
    """이 계정이 자기 KIS 키·계좌를 저장해 뒀는가 — 실거래 게이트의 근거.

    must_use_own_keys() 가 참인 계정(구글 계정, 공개 서버의 모든 계정)은 이게
    참일 때만 KIS 주문을 낼 수 있습니다. 서버 키로 폴백시키면 아무 사용자의
    자동매매가 서버 주인의 실계좌로 주문을 내게 됩니다.
    """
    keys = load_keys(user_id)
    return bool(keys.get("KIS_APP_KEY") and keys.get("KIS_APP_SECRET")
                and keys.get("KIS_ACCOUNT"))


# ---------------------------------------------------------------------------
# 화면용 (비밀값 없이)
# ---------------------------------------------------------------------------

def _mask(value: str) -> str:
    if len(value) <= _MASK_TAIL:
        return "····"
    return f"····{value[-_MASK_TAIL:]}"


def overview(user_id: int) -> dict:
    """설정 화면이 그리는 자료. 값 전체는 절대 담지 않습니다 (마스킹만).

    provider 마다: 내 키가 있는지 / 서버 키로 폴백 중인지 / 각 필드의 마스킹.
    서버 폴백 여부를 함께 주는 이유 — "키를 안 넣었는데 시세가 나오는" 상황이
    버그가 아니라 서버 공용 키 덕분임을 화면이 설명할 수 있어야 합니다.
    """
    mine = load_keys(user_id)
    providers = {}
    for provider in USER_PROVIDERS:
        spec = credentials.PROVIDERS[provider]
        fields = list(spec["fields"])
        if provider == "kis":
            fields += list(_EXTRA_KEYS)
        field_rows = []
        for field in spec["fields"]:
            own = mine.get(field, "")
            field_rows.append({
                "name": field,
                "set": bool(own),
                "masked": _mask(own) if own else "",
                "server_fallback": (not own) and bool(credentials.get(field)),
            })
        extra_rows = []
        if provider == "kis":
            for field in _EXTRA_KEYS:
                own = mine.get(field, "")
                extra_rows.append({
                    "name": field, "set": bool(own),
                    "masked": _mask(own) if own else "",
                    "server_fallback": False,     # 거래 설정은 폴백 개념이 없습니다
                })
        providers[provider] = {
            "label": spec["label"],
            "portal": spec["portal"],
            "gives": spec["gives"],
            "fields": field_rows,
            "extra": extra_rows,
            "mine": all(r["set"] for r in field_rows),
        }
    return {"providers": providers,
            "kis_trade_ready": has_own_kis_keys(user_id)}
