"""
계정 저장소 (MongoDB)
---------------------
구글 로그인으로 들어온 계정의 **신원**을 MongoDB 에 보관합니다.
예측 기록·모의투자 계좌·자동매매는 그대로 SQLite(`athena.db`)에 남습니다.
설계 배경은 ACCOUNTS.md 2장에 정리돼 있습니다. 요약하면:

    Mongo   = 구글 프로필 · 세션 · OAuth 임시 상태   (키: google_sub)
    SQLite  = 앱 데이터                              (키: 정수 user_id)

앱 데이터 전체가 **정수 user_id** 를 FK 로 쓰고 있어서, 그 정수를 ObjectId 로
갈아타는 대신 Mongo 문서가 "이 구글 계정은 user_id=100003 이다" 를 들고 있게 했습니다.

user_id 를 SQLite AUTOINCREMENT 가 아니라 Mongo 카운터로 발급하는 이유
    여러 사람이 각자 PC에서 앱을 띄우고 같은 Atlas 클러스터를 바라볼 때,
    SQLite 로 번호를 매기면 같은 구글 계정이 PC마다 다른 user_id 를 받습니다.
    계정이 따라오지 않습니다. 클러스터를 공유하는 모두가 같은 번호를 보게
    하려면 발급처가 Mongo 여야 합니다. 기존 로컬 계정의 작은 id 와 부딪히지
    않도록 100000 에서 시작합니다.

pymongo 를 최상단에서 import 하지 않는 이유
    이 앱은 지금까지 외부 설정 없이도 돌아갔습니다. 라이브러리가 없는 PC에서
    `import api` 가 터져 서버가 아예 안 뜨는 일은 없어야 합니다. 그래서 지연
    import 하고, 없으면 구글 로그인만 조용히 꺼집니다.
"""

import secrets
from datetime import datetime, timedelta, timezone

from data_sources import credentials
from storage import users as local_users

DEFAULT_DB = "athena"

SESSION_DAYS = 30
STATE_TTL_MINUTES = 10          # 구글 동의 화면에서 머무는 시간 + 여유
HANDOFF_TTL_SECONDS = 60        # 콜백 → 프론트 인계. 사람이 개입할 구간이 아니라 짧게

# 로컬 계정(SQLite AUTOINCREMENT)의 id 와 절대 겹치지 않게 하는 시작점
USER_ID_OFFSET = 100_000

# 클러스터가 죽었을 때 요청이 30초(기본값) 매달려 있으면 화면이 멈춘 것처럼 보입니다.
SERVER_SELECTION_TIMEOUT_MS = 5_000

_client = None
_indexes_ready = False


class AccountsUnavailable(RuntimeError):
    """Mongo 를 쓸 수 없는 상태 — 호출부는 구글 로그인만 막고 나머지는 계속 돕니다."""


# ---------------------------------------------------------------------------
# 연결
# ---------------------------------------------------------------------------

# URI 의 사용자정보(아이디:비밀번호) 안에서 그대로 쓰면 주소를 깨뜨리는 글자들.
# RFC 3986 이 이 자리에 허용하지 않는 문자입니다.
_MUST_ESCAPE = ":/?#[]@"


def _looks_percent_encoded(text: str, index: int) -> bool:
    """text[index] 의 % 가 %XX 삼중자의 시작인지."""
    return (index + 2 < len(text)
            and all(c in "0123456789abcdefABCDEF" for c in text[index + 1:index + 3]))


def _escape_userinfo(part: str) -> str:
    """아이디/비밀번호 한 조각을 URI 에 실을 수 있는 형태로 만듭니다.

    **이미 %XX 로 적혀 있는 것은 건드리지 않습니다.** 그래서 두 번 걸어도 결과가
    같습니다 — 이미 인코딩해 둔 사람의 비밀번호를 %2540 으로 망가뜨리지 않습니다.
    (RFC 3986 상 `%40` 은 언제나 `@` 를 뜻하므로, 이 해석이 규격에 맞습니다.)
    """
    from urllib.parse import quote

    out = []
    i = 0
    while i < len(part):
        ch = part[i]
        if ch == "%" and _looks_percent_encoded(part, i):
            out.append(part[i:i + 3])
            i += 3
            continue
        if ch == "%" or ch in _MUST_ESCAPE or ord(ch) > 127:
            out.append(quote(ch, safe=""))      # 한글 비밀번호도 UTF-8 로 처리됩니다
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def normalize_mongo_uri(uri: str) -> str:
    """비밀번호에 `@`·`#` 같은 글자가 있어도 붙게 만듭니다.

    Atlas 가 주는 접속 문자열에 비밀번호를 **그대로** 붙여 넣는 것이 자연스러운
    동작인데, pymongo 는 거기서

        Username and password must be escaped according to RFC 3986,
        use urllib.parse.quote_plus

    로 거절합니다. 영문 오류 하나를 보고 비밀번호를 손으로 `%40` 으로 바꿔 적으라고
    요구하는 대신 여기서 처리합니다. 사용자가 만질 것은 자기 비밀번호뿐이어야 합니다.

    경계를 찾는 순서가 중요합니다. 경로·쿼리(`/` `?` `#`)를 **먼저** 잘라내면
    `p#w` 같은 비밀번호에서 엉뚱한 자리가 잘립니다. 그래서 `@` 를 먼저 봅니다:
    호스트에는 `@` 가 올 수 없으므로 **마지막** `@` 가 사용자정보의 끝이고,
    경로·쿼리는 그 뒤에서 찾습니다. 비밀번호 안에 `@ # / ?` 가 몇 개 있든
    경계가 흔들리지 않습니다.
    """
    uri = (uri or "").strip()
    scheme, sep, rest = uri.partition("://")
    if not sep:
        return uri

    userinfo, at, hostpart = rest.rpartition("@")
    if not at:
        return uri                                  # 아이디·비밀번호가 없는 URI

    # 호스트가 끝나는 자리 — 경로·쿼리·프래그먼트 앞
    end = len(hostpart)
    for mark in ("/", "?", "#"):
        found = hostpart.find(mark)
        if found != -1:
            end = min(end, found)
    host, tail = hostpart[:end], hostpart[end:]
    if not host:
        return uri                                  # 호스트가 없다 = 우리가 손댈 URI 가 아님

    user, colon, password = userinfo.partition(":")     # 첫 : 가 구분자
    fixed = _escape_userinfo(user)
    if colon:
        fixed += ":" + _escape_userinfo(password)

    return f"{scheme}://{fixed}@{host}{tail}"


def mongo_uri() -> str:
    return normalize_mongo_uri(credentials.get("MONGODB_URI", ""))


# 접속 실패의 원문 → 무엇을 해야 하는지. 위에서부터 먼저 맞는 것을 씁니다.
#
# 이 표가 필요한 이유: pymongo 의 접속 실패 메시지는 영어인 데다, 세 노드의
# ServerDescription 을 통째로 붙여 2000자가 넘습니다. 그중 정작 필요한 한 줄
# ("Atlas 에 내 IP 를 등록하세요")은 어디에도 없습니다. 여기서 막히면 진도가
# 통째로 멈추는 자리라, 원인을 짚어주는 것이 그대로 두는 것보다 훨씬 낫습니다.
_ERROR_HINTS = (
    (("escaped", "quote_plus"),
     "비밀번호에 URI 에 못 쓰는 글자가 있습니다. 보통은 자동으로 처리되니, "
     "이 오류가 계속 나면 비밀번호를 영문·숫자로만 바꿔 보세요."),
    # 자리표시자 이름을 그대로 찾습니다. `<`/`>` 만 보면 실패 원문에 섞여 오는
    # `<ServerDescription ...>` 에 걸려 엉뚱한 안내가 나갑니다.
    (("<db_password>", "<password>", "<username>", "<db_username>"),
     "접속 문자열에 <db_password> 같은 자리표시자가 그대로 남아 있습니다. "
     "Atlas 에서 정한 실제 비밀번호로 바꿔 주세요."),
    (("Invalid URI scheme", "invalid uri scheme"),
     "주소는 mongodb:// 또는 mongodb+srv:// 로 시작해야 합니다. "
     "Atlas 라면 Connect → Drivers → Python 의 문자열을 그대로 쓰세요."),
    (("bad auth", "Authentication failed", "AuthenticationFailed"),
     "아이디 또는 비밀번호가 다릅니다. Atlas → Database Access 에서 "
     "그 사용자의 비밀번호를 다시 지정해 보세요."),
    (("TLSV1_ALERT_INTERNAL_ERROR", "SSL handshake failed", "handshake"),
     "Atlas 가 접속을 거부했습니다. 대개 **내 IP 가 등록되지 않은 것**입니다 — "
     "Atlas → Network Access → Add IP Address → Add Current IP Address. "
     "(클러스터가 일시중지(paused)된 경우에도 같은 오류가 납니다.)"),
    (("No servers found", "SRV", "resolution lifetime", "DNS operation timed out"),
     "클러스터 주소를 DNS 에서 찾지 못했습니다. 주소에 오타가 없는지, "
     "인터넷이 되는지 확인해 주세요."),
    (("Connection refused", "10061", "actively refused", "timed out", "Timeout"),
     "클러스터에 닿지 못했습니다. Atlas 라면 Network Access 의 IP 등록과 "
     "클러스터가 실행 중인지(일시중지 아님)를, 로컬이라면 MongoDB 서비스가 "
     "켜져 있는지 확인해 주세요."),
)

# 원문은 진단에 쓸 만큼만 남깁니다. 전체는 세 노드 상태를 다 붙여 2000자가 넘습니다.
_ERROR_TEXT_LIMIT = 180


def explain_mongo_error(exc: Exception, prefix: str) -> str:
    """pymongo 예외를 사람이 읽고 **행동할 수 있는** 한 문장으로 바꿉니다."""
    text = str(exc)
    for needles, hint in _ERROR_HINTS:
        if any(needle in text for needle in needles):
            short = text.split(",")[0][:_ERROR_TEXT_LIMIT]
            return f"{prefix}: {hint}\n    (원문: {short})"
    return f"{prefix}: {text[:_ERROR_TEXT_LIMIT]}"


def db_name() -> str:
    return credentials.get("MONGODB_DB", "") or DEFAULT_DB


def status() -> dict:
    """설정·연결 상태. 화면에서 구글 버튼을 그릴지 판단하는 근거입니다."""
    uri = mongo_uri()
    if not uri:
        return {"configured": False, "reason": "MONGODB_URI 가 설정되지 않았습니다."}
    try:
        import pymongo                                  # noqa: F401
    except ImportError:
        return {"configured": False,
                "reason": 'pymongo 가 설치되지 않았습니다. python -m pip install "pymongo[srv]"'}
    return {"configured": True, "reason": "", "db": db_name()}


def _db():
    """연결된 Database 객체. 못 쓰는 상태면 AccountsUnavailable."""
    global _client

    uri = mongo_uri()
    if not uri:
        raise AccountsUnavailable("MONGODB_URI 가 설정되지 않았습니다.")

    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise AccountsUnavailable(
            'pymongo 가 설치되지 않았습니다. python -m pip install "pymongo[srv]"') from exc

    if _client is None:
        try:
            _client = MongoClient(
                uri,
                # tz_aware: 꺼두면 naive UTC 가 돌아와 aware datetime 과 비교할 때
                # TypeError 가 납니다. 만료 검사에서 바로 터지는 자리입니다.
                tz_aware=True,
                serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
                appname="athena-signal",
            )
        except Exception as exc:      # 잘못된 URI 형식 등
            raise AccountsUnavailable(explain_mongo_error(exc, "MongoDB 연결 설정이 잘못됐습니다")) from exc

    database = _client[db_name()]
    _ensure_indexes(database)
    return database


def _ensure_indexes(database):
    """인덱스 보장 — 프로세스당 한 번만 시도합니다.

    TTL 인덱스(expires_at)를 걸어두면 만료된 세션·상태를 우리가 지우지 않아도
    Mongo 가 치웁니다. 로그인 때마다 청소 쿼리를 날릴 필요가 없어집니다.
    """
    global _indexes_ready
    if _indexes_ready:
        return
    try:
        database.accounts.create_index("google_sub", unique=True)
        database.accounts.create_index("user_id", unique=True)
        database.accounts.create_index("email")

        database.sessions.create_index("token", unique=True)
        database.sessions.create_index("expires_at", expireAfterSeconds=0)
        database.sessions.create_index("user_id")

        database.oauth_state.create_index("state", unique=True)
        database.oauth_state.create_index("expires_at", expireAfterSeconds=0)

        database.handoffs.create_index("code", unique=True)
        database.handoffs.create_index("expires_at", expireAfterSeconds=0)
    except Exception as exc:
        # 인덱스를 못 만들어도(권한 제한 등) 읽기·쓰기는 됩니다. 연결 자체가
        # 죽은 경우라면 바로 뒤의 실제 쿼리에서 어차피 드러납니다.
        raise AccountsUnavailable(
            explain_mongo_error(exc, "MongoDB 에 접속할 수 없습니다")) from exc
    _indexes_ready = True


def ping() -> dict:
    """실제로 연결되는지 확인 (설정 화면·진단용)."""
    try:
        _db().command("ping")
    except AccountsUnavailable as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "db": db_name()}


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# user_id 발급
# ---------------------------------------------------------------------------

def _next_user_id(database) -> int:
    """클러스터 전체에서 유일한 정수 id 를 원자적으로 발급합니다."""
    from pymongo import ReturnDocument

    doc = database.counters.find_one_and_update(
        {"_id": "user_id"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,     # 증가시킨 뒤의 값을 받아야 합니다
    )
    seq = int(doc.get("seq", 1))
    return USER_ID_OFFSET + seq


# ---------------------------------------------------------------------------
# 계정
# ---------------------------------------------------------------------------

def _as_user(doc: dict) -> dict:
    """기존 코드가 기대하는 사용자 dict 모양으로 변환.

    api.py 전반이 user["id"] 를 쓰므로 그 키 이름을 지킵니다.
    email/picture/provider 는 화면용으로 덧붙인 것이라 없어도 무해합니다.
    """
    return {
        "id": doc["user_id"],
        "username": doc.get("email") or f"google:{doc.get('google_sub', '')}",
        "display_name": doc.get("display_name") or doc.get("email") or "구글 사용자",
        "email": doc.get("email", ""),
        "picture": doc.get("picture", ""),
        "provider": "google",
    }


def find_by_google_sub(sub: str) -> dict | None:
    if not sub:
        return None
    doc = _db().accounts.find_one({"provider": "google", "google_sub": sub})
    return _as_user(doc) if doc else None


def upsert_google_account(profile: dict) -> dict:
    """구글 프로필로 계정을 찾거나 만듭니다.

    profile 은 data_sources.google_oauth.fetch_identity() 또는
    data_sources.firebase_auth.verify_id_token() 이 돌려준 dict 입니다. 두 흐름이
    같은 함수를 쓰는 이유는 `sub` 가 양쪽 모두 **구글 계정의 sub** 이기 때문입니다
    (Firebase 는 firebase.identities["google.com"] 에서 꺼냅니다). 덕분에 예전
    OAuth 로 만든 계정이 Firebase 로 바꾼 뒤에도 그대로 이어집니다.
    반환값: {"user": {...}, "created": bool}

    이메일이 아니라 google_sub 로 찾습니다. 구글에서 이메일을 바꿀 수 있는데,
    이메일로 찾으면 그때 계정이 새로 생겨 그 사람 기록이 통째로 사라져 보입니다.
    """
    sub = (profile.get("sub") or "").strip()
    if not sub:
        raise AccountsUnavailable("구글이 사용자 식별자(sub)를 주지 않았습니다.")

    database = _db()
    email = (profile.get("email") or "").strip()
    display_name = (profile.get("name") or "").strip() or email or "구글 사용자"

    fresh = {
        "email": email,
        "email_verified": bool(profile.get("email_verified")),
        "display_name": display_name,
        "picture": profile.get("picture", ""),
        "locale": profile.get("locale", ""),
    }

    # Firebase 로 들어온 경우에만 붙습니다. 계정을 찾는 키는 아니고(위 docstring),
    # 나중에 Firebase 콘솔의 사용자와 대조할 때 쓰는 참고값입니다.
    if profile.get("firebase_uid"):
        fresh["firebase_uid"] = profile["firebase_uid"]

    existing = database.accounts.find_one({"provider": "google", "google_sub": sub})
    if existing:
        # 재로그인 — 저장된 계정을 불러오고 프로필만 최신으로 맞춥니다
        database.accounts.update_one(
            {"_id": existing["_id"]},
            {"$set": {**fresh, "last_login_at": _now()},
             "$inc": {"login_count": 1}},
        )
        user = _as_user({**existing, **fresh, "user_id": existing["user_id"]})
        local_users.ensure_external_user(user["id"], f"google:{sub}", display_name)
        return {"user": user, "created": False}

    # 첫 로그인 — 이메일이 같은 로컬 계정이 이미 있으면 그 기록을 물려받습니다
    linked = local_users.find_by_username(email) if email else None
    if linked:
        user_id = linked["id"]
    else:
        user_id = _next_user_id(database)

    doc = {
        "provider": "google",
        "google_sub": sub,
        "user_id": user_id,
        **fresh,
        "created_at": _now(),
        "last_login_at": _now(),
        "login_count": 1,
        "linked_local_user_id": linked["id"] if linked else None,
    }

    try:
        database.accounts.insert_one(doc)
    except Exception:
        # 같은 계정으로 동시에 두 번 로그인했을 때 (google_sub 유니크 인덱스 충돌).
        # 이미 만들어진 쪽을 읽어 쓰는 것이 맞습니다.
        again = database.accounts.find_one({"provider": "google", "google_sub": sub})
        if not again:
            raise
        local_users.ensure_external_user(again["user_id"], f"google:{sub}", display_name)
        return {"user": _as_user(again), "created": False}

    # SQLite 쪽에 같은 id 의 행을 만들어 둡니다 — FK 대상이 실제로 있어야 하고
    # paper.ensure_account(user_id) 도 이 행을 전제합니다.
    if not linked:
        local_users.ensure_external_user(user_id, f"google:{sub}", display_name)

    return {"user": _as_user(doc), "created": True}


# ---------------------------------------------------------------------------
# 세션
# ---------------------------------------------------------------------------

def create_session(user_id: int, user_agent: str = "") -> str:
    token = secrets.token_urlsafe(32)
    _db().sessions.insert_one({
        "token": token,
        "user_id": user_id,
        "created_at": _now(),
        "expires_at": _now() + timedelta(days=SESSION_DAYS),
        "user_agent": (user_agent or "")[:300],
    })
    return token


def user_from_token(token: str) -> dict | None:
    """유효한 Mongo 세션이면 사용자 dict, 아니면 None.

    Mongo 를 못 쓰는 상태여도 None 만 돌려줍니다 — 호출부(api.current_user)가
    이어서 SQLite 세션을 확인하므로, 로컬 로그인이 함께 죽으면 안 됩니다.
    """
    if not token:
        return None
    try:
        database = _db()
        session = database.sessions.find_one({"token": token})
        if not session:
            return None

        expires = session.get("expires_at")
        # TTL 인덱스는 최대 60초 늦게 도는 백그라운드 작업이라, 만료 직후의
        # 토큰이 살아 있는 창이 생깁니다. 여기서 한 번 더 봅니다.
        if isinstance(expires, datetime):
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if expires < _now():
                database.sessions.delete_one({"token": token})
                return None

        doc = database.accounts.find_one({"user_id": session["user_id"]})
        return _as_user(doc) if doc else None
    except AccountsUnavailable:
        return None
    except Exception:
        return None


def delete_session(token: str) -> bool:
    """로그아웃. Mongo 세션이었으면 True."""
    if not token:
        return False
    try:
        return _db().sessions.delete_one({"token": token}).deleted_count > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OAuth 임시 상태 (CSRF + PKCE)
# ---------------------------------------------------------------------------

def save_oauth_state(state: str, code_verifier: str, redirect_origin: str,
                     next_path: str = "/", nonce: str = ""):
    _db().oauth_state.insert_one({
        "state": state,
        "code_verifier": code_verifier,
        "nonce": nonce,
        "redirect_origin": redirect_origin,
        "next": next_path or "/",
        "created_at": _now(),
        "expires_at": _now() + timedelta(minutes=STATE_TTL_MINUTES),
    })


def consume_oauth_state(state: str) -> dict | None:
    """state 를 **1회용으로** 소비합니다.

    find_one_and_delete 라서 같은 state 로 두 번 들어오면 두 번째는 None 입니다.
    인가 코드 재사용·재생 공격이 여기서 막힙니다.
    """
    if not state:
        return None
    doc = _db().oauth_state.find_one_and_delete({"state": state})
    if not doc:
        return None

    expires = doc.get("expires_at")
    if isinstance(expires, datetime):
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < _now():
            return None
    return doc


# ---------------------------------------------------------------------------
# 토큰 인계 (핸드오프)
# ---------------------------------------------------------------------------

def create_handoff(token: str) -> str:
    """세션 토큰을 URL 에 싣지 않기 위한 60초·1회용 교환 코드."""
    code = secrets.token_urlsafe(24)
    _db().handoffs.insert_one({
        "code": code,
        "token": token,
        "created_at": _now(),
        "expires_at": _now() + timedelta(seconds=HANDOFF_TTL_SECONDS),
    })
    return code


def consume_handoff(code: str) -> str | None:
    """코드를 소비하고 세션 토큰을 돌려줍니다. 두 번째 호출은 None."""
    if not code:
        return None
    doc = _db().handoffs.find_one_and_delete({"code": code})
    if not doc:
        return None

    expires = doc.get("expires_at")
    if isinstance(expires, datetime):
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires < _now():
            return None
    return doc.get("token")


# ---------------------------------------------------------------------------
# 진단
# ---------------------------------------------------------------------------

def list_accounts(limit: int = 100) -> list[dict]:
    """등록된 구글 계정 목록 (프로필만, 토큰류는 제외)."""
    try:
        rows = _db().accounts.find(
            {"provider": "google"},
            {"_id": 0, "google_sub": 0},
        ).sort("created_at", 1).limit(limit)
        return [dict(r) for r in rows]
    except AccountsUnavailable:
        return []


# 사용자 한 명의 데이터가 흩어져 있는 자리. 전부 정수 user_id 를 FK 로 씁니다 (2장).
_USER_TABLES = (
    ("예측", "predictions"),
    ("관심종목", "watchlist"),
    ("모의보유", "paper_holdings"),
    ("모의거래", "paper_trades"),
    ("자동매매주문", "at_orders"),
    ("세션", "sessions"),
)


def mongo_overview() -> dict:
    """Mongo 쪽에 무엇이 몇 건 들어 있는지. 못 쓰는 상태면 {"ok": False}."""
    try:
        database = _db()
        counts = {name: database[name].count_documents({})
                  for name in ("accounts", "sessions", "handoffs", "oauth_state")}
        counter = database.counters.find_one({"_id": "user_id"}) or {}
        return {
            "ok": True,
            "db": db_name(),
            "counts": counts,
            # 다음 사람이 받게 될 번호 — 카운터가 실제로 도는지 확인용
            "next_user_id": USER_ID_OFFSET + int(counter.get("seq", 0)) + 1,
            "accounts": list_accounts(),
        }
    except AccountsUnavailable as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:                                       # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def sqlite_overview() -> list[dict]:
    """SQLite 쪽 사용자별 데이터 건수 — "내 기록이 어디 있나" 에 답하는 표.

    Mongo 가 죽어 있어도 이건 나옵니다. 앱 데이터는 Mongo 와 무관하게
    `athena.db` 에 있다는 사실 자체가 2장 설계의 요점입니다.
    """
    local_users.init()
    rows = []
    with local_users._conn() as conn:
        for user in conn.execute(
                "SELECT id, username, display_name, password_hash, created_at "
                "FROM users ORDER BY id").fetchall():
            # 비밀번호 해시 자리에 표식만 있는 계정은 두 종류입니다. 구글 계정과,
            # 계정 개념이 생기기 전의 기록을 담아둔 `__local__`. 둘을 같은 줄로
            # 묶으면 "구글로 로그인한 적 없는데 구글 계정이 있다" 로 읽힙니다.
            if user["username"] == local_users.LEGACY_USER:
                origin = "이전기록"
            elif local_users._is_password_hash(user["password_hash"]):
                origin = "로컬"
            else:
                origin = "구글"

            entry = {
                "id": user["id"],
                "username": user["username"],
                "display_name": user["display_name"],
                "origin": origin,
                "external": origin == "구글",
                "created_at": user["created_at"],
                "counts": {},
                "cash": None,
            }
            for label, table in _USER_TABLES:
                try:
                    entry["counts"][label] = conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE user_id = ?",
                        (user["id"],)).fetchone()[0]
                except Exception:                                  # noqa: BLE001
                    entry["counts"][label] = 0      # 오래된 DB 에 없는 테이블
            try:
                found = conn.execute("SELECT cash FROM paper_account WHERE user_id = ?",
                                     (user["id"],)).fetchone()
                if found:
                    entry["cash"] = float(found["cash"])
            except Exception:                                      # noqa: BLE001
                pass
            rows.append(entry)
    return rows


if __name__ == "__main__":
    # python -m storage.accounts  — 설정이 실제로 되는지 확인 ([8] 구글 로그인 안내)
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import unicodedata

    from data_sources import firebase_auth, google_oauth

    def _cells(text: str) -> int:
        """터미널에서 차지하는 칸 수 — 한글·한자는 두 칸입니다.

        len() 으로 맞추면 한글이 섞인 표가 통째로 어긋납니다.
        """
        return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)

    def _pad(text: str, columns: int) -> str:
        return text + " " * max(0, columns - _cells(text))

    def _cut(text: str, columns: int) -> str:
        while _cells(text) > columns:
            text = text[:-1]
        return text

    print("=" * 62)
    print("  아테나 시그널 — 계정 저장소 상태")
    print("=" * 62)

    uri = mongo_uri()
    shown = uri
    if "@" in uri:      # 비밀번호가 URI 에 들어 있으면 가립니다
        head, _, tail = uri.rpartition("@")
        scheme, _, _ = head.partition("//")
        shown = f"{scheme}//***:***@{tail}"
    print(f"\n  MONGODB_URI : {shown or '(설정되지 않음)'}")
    print(f"  MONGODB_DB  : {db_name()}")

    mongo = status()
    if not mongo["configured"]:
        print(f"\n  [미설정] {mongo['reason']}")

    # ---- Mongo 쪽: 신원 -----------------------------------------------------
    print("\n" + "-" * 62)
    print("  MongoDB — 신원 (누구인가)")
    print("-" * 62)

    overview = mongo_overview() if mongo["configured"] else {
        "ok": False, "error": mongo["reason"]}

    if not overview["ok"]:
        print(f"\n  연결 안 됨 — {overview['error']}")
        print("\n  ※ 여기가 비어 있어도 아래 SQLite 의 앱 데이터는 멀쩡합니다.")
    else:
        labels = {"accounts": "구글 계정", "sessions": "로그인 세션",
                  "handoffs": "인계 코드", "oauth_state": "OAuth 임시상태"}
        print()
        for key, label in labels.items():
            print(f"  {_pad(label, 16)}{overview['counts'][key]:>5d}건")
        print(f"  {_pad('다음 user_id', 16)}{overview['next_user_id']:>5d}")

        if overview["accounts"]:
            print("\n  등록된 구글 계정")
            for row in overview["accounts"]:
                print(f"    user_id={row.get('user_id')}  {row.get('email', '')}"
                      f"  ({row.get('display_name', '')})"
                      f"  로그인 {row.get('login_count', 0)}회")
        else:
            print("\n  아직 구글 계정으로 로그인한 사람이 없습니다.")

    # ---- SQLite 쪽: 앱 데이터 ----------------------------------------------
    print("\n" + "-" * 62)
    print("  SQLite (athena.db) — 앱 데이터 (무엇을 했는가)")
    print("-" * 62)

    people = sqlite_overview()
    if not people:
        print("\n  계정이 없습니다.")
    else:
        header = ["user_id", "계정", "출처"] + [label for label, _ in _USER_TABLES]
        widths = [9, 20, 10] + [_cells(label) + 2 for label, _ in _USER_TABLES]
        print()
        print("  " + "".join(_pad(h, w) for h, w in zip(header, widths)))
        for person in people:
            cells = [
                str(person["id"]),
                _cut(person["display_name"] or person["username"], 18),
                person["origin"],
            ] + [str(person["counts"][label]) for label, _ in _USER_TABLES]
            print("  " + "".join(_pad(c, w) for c, w in zip(cells, widths)))

        print("\n  모의투자 잔고")
        for person in people:
            if person["cash"] is not None:
                print(f"    user_id={person['id']:<8d} {person['cash']:>15,.0f}원")

    print("\n  ※ user_id 가 두 저장소를 잇는 유일한 끈입니다 (ACCOUNTS.md 2-2).")
    print("    Mongo 를 통째로 지워도 예측·모의투자 기록은 athena.db 에 남습니다.")

    # ---- 로그인 경로 --------------------------------------------------------
    # Firebase 가 기본이고 OAuth 는 폴백입니다 (ACCOUNTS.md 0장). 둘 중 **하나만**
    # 되면 로그인은 켜지므로, 예전처럼 OAuth 만 보고 "꺼짐" 이라고 하면 안 됩니다.
    print("\n" + "-" * 62)
    print("  로그인 경로")
    print("-" * 62)

    firebase = firebase_auth.status()
    if firebase["configured"]:
        print(f"\n  [OK ] Firebase — projectId = {firebase['projectId']}")
        print(f"        authDomain = {firebase['authDomain']}")
    else:
        print(f"\n  [미설정] Firebase — {firebase['reason']}")

    google = google_oauth.status()
    if google["configured"]:
        print(f"\n  [OK ] 구버전 OAuth (폴백) — redirect_uri = {google['redirect_uri']}")
    else:
        print(f"\n  [미설정] 구버전 OAuth (폴백) — {google['reason']}")

    ready = mongo["configured"] and (firebase["configured"] or google["configured"])
    used = "Firebase" if firebase["configured"] else "구버전 OAuth"
    print(f"\n  구글 로그인: "
          f"{f'사용 가능 ({used})' if ready else '꺼짐 (아이디/비번 로그인은 정상)'}")
    print("  설정: 아테나.bat → [8] 구글 로그인 — 자세한 내용은 ACCOUNTS.md\n")
