"""
스크랩 저장소 (Browser-Assisted Scraping)
----------------------------------------
서버가 직접 접근할 수 없는 커뮤니티를 **사용자 브라우저를 통해** 수집합니다.

왜 필요한가
    토스증권 커뮤니티(401/405), 팍스넷(404), 카카오페이증권(연결 차단),
    한국거래소 데이터플랫폼(LOGOUT) 등은 서버에서 직접 호출이 막혀 있습니다.
    반면 크롬 확장프로그램은 **사용자 브라우저 안에서 사용자 세션으로** 돌기 때문에
    사용자가 이미 볼 수 있는 페이지라면 그대로 읽을 수 있습니다.

    확장프로그램이 그 페이지의 게시글을 긁어 POST /api/scrap 으로 보내면,
    여기 저장했다가 커뮤니티 여론 계산에 합류시킵니다.

설계 원칙
    · 사용자가 이미 열람 중인 페이지의 공개 게시글만 대상으로 합니다.
    · 로그인 자격증명은 절대 다루지 않습니다 (확장프로그램도 쿠키를 읽지 않음).
    · 오래된 스크랩은 자동으로 만료시켜, 옛 여론이 오늘 판단에 섞이지 않게 합니다.
    · 같은 글이 여러 번 올라와도 지문(fingerprint)으로 한 번만 셉니다.
"""

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta

from config import DB_PATH

# 스크랩 유효 기간 — 이보다 오래된 글은 여론 계산에서 제외
SCRAP_TTL_HOURS = 12

# 한 종목당 보관할 최대 건수 (오래된 것부터 정리)
MAX_PER_TICKER = 400

_lock = threading.Lock()

# 확장프로그램이 보내올 수 있는 출처 (화이트리스트 — 임의 문자열 저장 방지)
ALLOWED_SOURCES = {
    "toss": "토스증권 커뮤니티",
    "paxnet": "팍스넷 종목토론방",
    "kakaopay": "카카오페이증권",
    "naver": "네이버 종목토론방",
    "krx": "한국거래소",
    "reddit": "Reddit",
    "other": "기타 커뮤니티",
}


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
        CREATE TABLE IF NOT EXISTS scraped_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'community',
            title TEXT NOT NULL,
            url TEXT,
            posted_at TEXT,
            collected_at TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            UNIQUE(ticker, fingerprint)
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_scrap_ticker "
                     "ON scraped_posts(ticker, collected_at)")


def _fingerprint(title: str) -> str:
    """같은 글의 중복 저장을 막기 위한 지문 (공백/기호 제거 후 앞부분)."""
    return re.sub(r"\W", "", title or "").lower()[:60]


def save_batch(ticker: str, source: str, items: list, kind: str = "community") -> dict:
    """확장프로그램이 보낸 게시글 묶음을 저장.

    items: [{"title": str, "url": str?, "posted_at": str?}, ...]
    반환: {"saved": n, "duplicated": n, "rejected": n}
    """
    if source not in ALLOWED_SOURCES:
        source = "other"

    init()
    now = datetime.now().isoformat()
    saved = dup = rejected = 0

    with _lock, _conn() as conn:
        for item in items or []:
            title = re.sub(r"\s+", " ", str((item or {}).get("title") or "")).strip()
            if not title or len(title) < 2:
                rejected += 1
                continue
            title = title[:300]

            try:
                conn.execute(
                    "INSERT INTO scraped_posts "
                    "(ticker, source, kind, title, url, posted_at, collected_at, fingerprint) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticker, source, kind, title,
                     str(item.get("url") or "")[:500],
                     str(item.get("posted_at") or "")[:40],
                     now, _fingerprint(title)),
                )
                saved += 1
            except sqlite3.IntegrityError:
                # 같은 글을 다시 봤다는 것은 그 글이 아직 게시판에 살아 있다는 뜻입니다.
                # 수집 시각을 갱신하지 않으면, 만료된 행이 지문 충돌 때문에 다시 들어오지도
                # 못하고 조회에서도 빠져 영영 사라집니다.
                conn.execute(
                    "UPDATE scraped_posts SET collected_at = ?, source = ? "
                    "WHERE ticker = ? AND fingerprint = ?",
                    (now, source, ticker, _fingerprint(title)))
                dup += 1

        # 종목당 보관 상한 유지
        conn.execute(
            "DELETE FROM scraped_posts WHERE ticker = ? AND id NOT IN "
            "(SELECT id FROM scraped_posts WHERE ticker = ? ORDER BY id DESC LIMIT ?)",
            (ticker, ticker, MAX_PER_TICKER),
        )

    return {"saved": saved, "duplicated": dup, "rejected": rejected}


def get_recent(ticker: str, hours: int = SCRAP_TTL_HOURS, limit: int = 200) -> list[dict]:
    """유효 기간 내 스크랩 게시글."""
    init()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scraped_posts WHERE ticker = ? AND collected_at >= ? "
            "ORDER BY id DESC LIMIT ?",
            (ticker, cutoff, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def source_summary(ticker: str, hours: int = SCRAP_TTL_HOURS) -> dict:
    """출처별 수집 현황 — 화면에 '어디서 몇 건 들어왔는지' 보여주기 위함."""
    init()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT source, COUNT(*) n, MAX(collected_at) last FROM scraped_posts "
            "WHERE ticker = ? AND collected_at >= ? GROUP BY source",
            (ticker, cutoff),
        ).fetchall()
    return {
        r["source"]: {
            "label": ALLOWED_SOURCES.get(r["source"], r["source"]),
            "count": r["n"],
            "last_collected": r["last"],
        }
        for r in rows
    }


def purge_expired(hours: int = SCRAP_TTL_HOURS * 4) -> int:
    """오래된 스크랩 정리 (서버 시작 시 호출)."""
    init()
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    with _conn() as conn:
        cur = conn.execute("DELETE FROM scraped_posts WHERE collected_at < ?", (cutoff,))
        return cur.rowcount
