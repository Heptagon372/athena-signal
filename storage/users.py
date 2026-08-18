"""
사용자 계정 · 세션
-----------------
로컬에서 도는 개인용 앱이라 외부 인증 서비스 없이 표준 라이브러리만으로 구현했습니다.

비밀번호
    PBKDF2-HMAC-SHA256, 20만 회 반복, 계정마다 다른 salt.
    평문은 저장하지도 로그에 남기지도 않습니다.

세션
    로그인하면 무작위 토큰을 발급하고 DB에 만료시각과 함께 저장합니다.
    프론트엔드는 이 토큰을 Authorization: Bearer 로 보냅니다.

    DB 에 들어가는 것은 토큰이 아니라 **SHA-256 지문**입니다. athena.db 는
    백업 파일까지 포함해 이 폴더에 그냥 놓여 있는 파일이라, 원문을 저장하면
    파일 하나가 새는 순간 30일짜리 로그인 세션이 통째로 딸려 나갑니다. 그
    세션은 KIS 주문 권한이기도 합니다. 지문만 있으면 훔쳐도 Authorization
    헤더에 넣을 값을 되돌릴 수 없습니다. (security.token_digest)

    지문 방식 이전에 발급된 세션은 원문으로 저장돼 있습니다. 조회할 때 두 값을
    모두 찾아보고, 원문으로 맞으면 그 자리에서 지문으로 바꿉니다 — 이미
    로그인해 둔 사람을 로그아웃시키지 않고 저절로 이전됩니다.

사용자별 데이터
    예측 기록·모의투자 계좌·관심종목이 모두 user_id 로 분리됩니다.
    같은 종목을 봐도 성적표와 계좌는 계정마다 따로 쌓입니다.
"""

import hashlib
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta

import security
from config import DB_PATH

PBKDF2_ROUNDS = 200_000
SESSION_DAYS = 30


def _digest(token: str) -> str:
    """DB 에 넣을 세션 토큰 지문 (모듈 docstring 의 '세션' 참고)."""
    return security.token_digest(token)

# 로그인하지 않고 쓰던 기존 데이터를 담아둘 계정
LEGACY_USER = "__local__"


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            display_name TEXT,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            name TEXT, market TEXT,
            added_at TEXT NOT NULL,
            PRIMARY KEY (user_id, ticker)
        )""")

        # 로그인 이전에 쌓인 데이터를 담을 계정
        row = conn.execute("SELECT id FROM users WHERE username = ?", (LEGACY_USER,)).fetchone()
        if not row:
            salt = secrets.token_hex(16)
            conn.execute(
                "INSERT INTO users (username, display_name, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (LEGACY_USER, "이전 기록", "-", salt, datetime.now().isoformat()))


def legacy_user_id() -> int:
    init()
    with _conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (LEGACY_USER,)).fetchone()
        return row["id"] if row else 1


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ROUNDS
    ).hex()


# 비밀번호 로그인이 불가능한 계정을 표시하는 값 (구글 전용 계정 등)
NO_PASSWORD = "!"


def _is_password_hash(value: str) -> bool:
    """PBKDF2-SHA256 결과(64자 hex)인지 — 아니면 비밀번호 로그인 불가 계정입니다."""
    return bool(value) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_username(username: str) -> str | None:
    """문제가 있으면 사유 문자열, 없으면 None."""
    if not username or len(username) < 3:
        return "아이디는 3자 이상이어야 합니다."
    if len(username) > 30:
        return "아이디는 30자 이하여야 합니다."
    if not re.fullmatch(r"[A-Za-z0-9_.\-가-힣]+", username):
        return "아이디에는 한글·영문·숫자와 _ . - 만 쓸 수 있습니다."
    if username == LEGACY_USER:
        return "사용할 수 없는 아이디입니다."
    return None


# 실제 유출 목록 상위권에서 한국어권 가입에 실제로 들어오는 것들만 추렸습니다.
# 큰 사전을 들고 오는 대신, "이건 남이 첫 번째로 찍어보는 값"만 걸러냅니다.
_WEAK_PASSWORDS = frozenset({
    "password", "passw0rd", "12345678", "123456789", "1234567890",
    "qwerty123", "qwertyui", "asdfasdf", "abcd1234", "a1234567",
    "iloveyou", "admin123", "administrator", "letmein1", "welcome1",
    "athena12", "signal12", "athenasignal",
})


def validate_password(password: str) -> str | None:
    """가입·변경 때만 검사합니다 (로그인은 검사하지 않음).

    최소 길이가 8자인 이유
        이 계정 하나로 KIS API 키·계좌번호·자동매매 설정에 닿습니다. 로그인
        레이트 리밋(IP 당 5분에 10회, 계정당 15분에 15회 실패)이 온라인 대입을
        늦춰 주지만, 6자는 그 한도 안에서도 흔한 조합이 뚫릴 만큼 짧습니다.

    복잡도 규칙(대문자·특수문자 필수) 대신 길이와 약한 비밀번호 목록을 쓰는
    이유: 복잡도 규칙은 사람을 `Password1!` 로 몰아갑니다. 길이가 실제 강도와
    더 잘 맞고, 진짜 위험한 것은 남이 첫 번째로 찍어보는 값입니다.

    기존 계정은 그대로 로그인됩니다 — 여기를 지나가지 않기 때문입니다.
    비밀번호를 바꿀 때 새 기준을 만나게 됩니다.
    """
    if not password or len(password) < 8:
        return "비밀번호는 8자 이상이어야 합니다."
    if len(password) > 200:
        return "비밀번호가 너무 깁니다."

    lowered = password.strip().lower()
    if lowered in _WEAK_PASSWORDS:
        return "너무 흔한 비밀번호입니다. 다른 값을 써 주세요."
    if len(set(password)) < 4:
        # "aaaaaaaa", "12121212" 처럼 글자 종류가 거의 없는 값
        return "같은 글자만 반복된 비밀번호는 쓸 수 없습니다."
    return None


def register(username: str, password: str, display_name: str = "") -> dict:
    init()
    username = (username or "").strip()

    for problem in (validate_username(username), validate_password(password)):
        if problem:
            return {"ok": False, "error": problem}

    salt = secrets.token_hex(16)
    try:
        with _conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, display_name, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, (display_name or username).strip(),
                 _hash(password, salt), salt, datetime.now().isoformat()))
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "이미 사용 중인 아이디입니다."}

    return {"ok": True, "user": {"id": user_id, "username": username,
                                 "display_name": (display_name or username).strip()}}


def login(username: str, password: str) -> dict:
    init()
    username = (username or "").strip()

    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    # 아이디가 없어도 비밀번호를 한 번 계산해 응답 시간 차이를 줄입니다.
    # 비밀번호로 로그인할 수 없는 계정(LEGACY_USER, 구글 전용 계정)도 여기서 걸러냅니다.
    # 그 계정들의 password_hash 는 PBKDF2 결과(64자 hex)가 아니라 표식 한 글자라,
    # 어떤 비밀번호를 넣어도 일치할 수 없지만 시도 자체를 받지 않는 편이 낫습니다.
    if not row or row["username"] == LEGACY_USER or not _is_password_hash(row["password_hash"]):
        _hash(password or "", "dummy-salt")
        return {"ok": False, "error": "아이디 또는 비밀번호가 올바르지 않습니다."}

    if not secrets.compare_digest(_hash(password, row["salt"]), row["password_hash"]):
        return {"ok": False, "error": "아이디 또는 비밀번호가 올바르지 않습니다."}

    token = secrets.token_urlsafe(32)
    now = datetime.now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (_digest(token), row["id"], now.isoformat(),
             (now + timedelta(days=SESSION_DAYS)).isoformat()))
        conn.execute("UPDATE users SET last_login_at = ? WHERE id = ?",
                     (now.isoformat(), row["id"]))
        # 만료된 세션 정리
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now.isoformat(),))

    return {"ok": True, "token": token,
            "user": {"id": row["id"], "username": row["username"],
                     "display_name": row["display_name"] or row["username"]}}


def logout(token: str):
    if not token:
        return
    # 지문·원문 두 형태 모두 지웁니다 (이전 중인 옛 세션도 확실히 끊기도록)
    keys = security.token_lookup_keys(token)
    with _conn() as conn:
        conn.execute(
            f"DELETE FROM sessions WHERE token IN ({','.join('?' * len(keys))})",
            keys)


def user_from_token(token: str) -> dict | None:
    """유효한 세션이면 사용자 정보, 아니면 None."""
    if not token:
        return None
    init()
    keys = security.token_lookup_keys(token)
    placeholders = ",".join("?" * len(keys))
    with _conn() as conn:
        row = conn.execute(f"""
            SELECT u.id, u.username, u.display_name, s.expires_at, s.token
            FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token IN ({placeholders})""", keys).fetchone()

    if not row:
        return None
    try:
        if datetime.fromisoformat(row["expires_at"]) < datetime.now():
            logout(token)
            return None
    except ValueError:
        return None

    # 원문으로 저장돼 있던 옛 세션이면 지금 지문으로 바꿔 둡니다.
    # 로그인한 채로 쓰다 보면 저절로 전부 이전됩니다.
    if row["token"] != keys[0]:
        with _conn() as conn:
            conn.execute("UPDATE sessions SET token = ? WHERE token = ?",
                         (keys[0], row["token"]))

    return {"id": row["id"], "username": row["username"],
            "display_name": row["display_name"] or row["username"]}


def change_password(user_id: int, current: str, new: str) -> dict:
    problem = validate_password(new)
    if problem:
        return {"ok": False, "error": problem}

    with _conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row or not secrets.compare_digest(_hash(current, row["salt"]), row["password_hash"]):
        return {"ok": False, "error": "현재 비밀번호가 올바르지 않습니다."}

    salt = secrets.token_hex(16)
    with _conn() as conn:
        conn.execute("UPDATE users SET password_hash = ?, salt = ? WHERE id = ?",
                     (_hash(new, salt), salt, user_id))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))   # 재로그인 유도
    return {"ok": True}


def find_by_username(username: str) -> dict | None:
    """아이디(또는 이메일)로 계정 조회.

    구글 로그인이 "이 이메일로 만들어 둔 로컬 계정이 이미 있는가" 를 확인해
    기존 기록을 물려받을 때 씁니다 (ACCOUNTS.md 2-4).
    """
    username = (username or "").strip()
    if not username or username == LEGACY_USER:
        return None
    init()
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, username, display_name FROM users WHERE username = ?",
            (username,)).fetchone()
    return dict(row) if row else None


def ensure_external_user(user_id: int, username: str, display_name: str = ""):
    """외부 인증(구글)으로 들어온 계정의 SQLite 행을 보장합니다.

    id 를 **직접 지정해서** 넣습니다. 예측·모의투자·관심종목 테이블이 이 정수를
    FK 로 쓰고, paper.ensure_account(user_id) 도 이 행이 있다고 전제하기 때문에
    Mongo 에만 있으면 안 됩니다. 번호는 Mongo 카운터가 발급하므로 여기서는
    AUTOINCREMENT 를 쓰지 않습니다 (storage/accounts.py 참고).

    password_hash 는 NO_PASSWORD 표식이라 비밀번호로는 로그인할 수 없습니다.
    """
    init()
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (id, username, display_name, password_hash, salt, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username, (display_name or username).strip(),
             NO_PASSWORD, secrets.token_hex(16), datetime.now().isoformat()))
        # 구글에서 이름을 바꿨으면 따라갑니다 (표시 이름만)
        if display_name:
            conn.execute(
                "UPDATE users SET display_name = ? WHERE id = ? AND password_hash = ?",
                (display_name.strip(), user_id, NO_PASSWORD))


def list_users() -> list[dict]:
    init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, username, display_name, created_at, last_login_at FROM users "
            "WHERE username != ? ORDER BY id", (LEGACY_USER,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 관심종목 (사용자별)
# ---------------------------------------------------------------------------

def get_watchlist(user_id: int) -> list[dict]:
    init()
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT ticker, name, market, added_at FROM watchlist WHERE user_id = ? "
            "ORDER BY added_at DESC", (user_id,))]


def add_watch(user_id: int, ticker: str, name: str = "", market: str = ""):
    init()
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO watchlist (user_id, ticker, name, market, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, ticker, name, market, datetime.now().isoformat()))


def remove_watch(user_id: int, ticker: str):
    with _conn() as conn:
        conn.execute("DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
                     (user_id, ticker))
