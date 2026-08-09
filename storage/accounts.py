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

def mongo_uri() -> str:
    return credentials.get("MONGODB_URI", "")


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
            raise AccountsUnavailable(f"MongoDB 연결 설정이 잘못됐습니다: {exc}") from exc

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
        raise AccountsUnavailable(f"MongoDB 에 접속할 수 없습니다: {exc}") from exc
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

    profile 은 data_sources.google_oauth.fetch_identity() 가 돌려준 dict 입니다.
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


if __name__ == "__main__":
    # python -m storage.accounts  — 설정이 실제로 되는지 확인 (setup_google.bat 안내)
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from data_sources import google_oauth

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
    else:
        result = ping()
        if result["ok"]:
            found = list_accounts()
            print(f"\n  [OK ] MongoDB 연결됨 — 등록된 구글 계정 {len(found)}개")
            for row in found:
                print(f"        user_id={row.get('user_id')}  {row.get('email', '')}"
                      f"  ({row.get('display_name', '')})")
        else:
            print(f"\n  [실패] {result['error']}")

    google = google_oauth.status()
    if google["configured"]:
        print(f"\n  [OK ] 구글 클라이언트 설정됨")
        print(f"        redirect_uri = {google['redirect_uri']}")
        print(f"        ↑ 이 값이 구글 콘솔의 '승인된 리디렉션 URI' 와 같아야 합니다")
    else:
        print(f"\n  [미설정] {google['reason']}")

    ready = mongo["configured"] and google["configured"]
    print(f"\n  구글 로그인: {'사용 가능' if ready else '꺼짐 (아이디/비번 로그인은 정상)'}")
    print("  설정: setup_google.bat 실행 — 자세한 내용은 ACCOUNTS.md\n")
