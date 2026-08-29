"""
AI 에이전트 분석 저장소
=======================
LLM 멀티에이전트(engine/agents.py)의 판단과 그 판단의 **성적**을 담습니다.

왜 기존 테이블에 얹지 않았나
    `predictions` 는 룰 기반 점수의 예측이고 `at_recommend` 는 팩터 순위입니다.
    둘 다 산식이 재현 가능한 값이라 같은 표에서 비교해도 말이 됩니다. LLM 판단은
    같은 입력에도 다른 답이 나오고, 호출마다 비용이 붙고, 리포트가 통째로 딸려
    옵니다. 한 표에 섞으면 두 성적표가 서로를 오염시킵니다.

성적을 여기서 같이 재는 이유
    LLM 판단은 "그럴듯함"이 성능처럼 보입니다. 근거 문장이 매끄러우면 맞은 것
    같습니다. 그래서 판단을 저장할 때 **판단 시각의 가격과 채점 만기**를 함께
    박아 둡니다. 나중에 그 시각의 실제 가격으로만 채점하면, 사후에 이야기를
    고쳐 쓸 여지가 없습니다 (engine/agent_review.py 가 채점합니다).

테이블
    at_agent_config     사용자별 에이전트 설정 (모델, 노력, 토론 라운드, 집행 모드)
    at_agent_analysis   분석 1건 = 판단 + 리포트(JSON) + 비용 + 채점 결과
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from config import DB_PATH

# 채점 만기 — 판단을 내리고 이만큼 뒤의 가격으로 맞았는지 봅니다.
# 에이전트의 horizon 문장("2주에서 1개월")은 자유 서술이라 채점 기준이 될 수
# 없습니다. 기계가 셀 수 있는 하나의 기준을 따로 둡니다.
DEFAULT_HORIZON_DAYS = 5


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
        CREATE TABLE IF NOT EXISTS at_agent_config (
            user_id INTEGER PRIMARY KEY,
            updated_at TEXT NOT NULL,
            config TEXT NOT NULL
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_agent_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            market TEXT,
            decision TEXT NOT NULL,              -- buy | sell | hold
            confidence REAL,                     -- 0 ~ 1
            horizon TEXT,                        -- 에이전트가 쓴 유효 기간 (자유 서술)
            summary TEXT,                        -- 한 줄 요약
            report TEXT NOT NULL,                -- 단계별 리포트 + 증거 팩 (JSON)
            model TEXT NOT NULL,
            calls INTEGER NOT NULL DEFAULT 0,    -- LLM 호출 횟수
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0,
            cost_krw REAL,
            elapsed_ms INTEGER,
            created_at TEXT NOT NULL,
            -- 채점 (전진 검증). 판단 시각에 박고, 만기가 지나야 채웁니다.
            base_price REAL,                     -- 판단 당시 가격
            target_at TEXT,                      -- 이 시각의 가격으로 채점합니다
            resolved_at TEXT,
            resolved_price REAL,
            change_pct REAL,                     -- 기준가 대비 등락
            correct INTEGER,                     -- 1 맞음 / 0 틀림 / NULL 미채점
            -- 집행 (engine/agent_trader.py 가 채웁니다)
            executed INTEGER NOT NULL DEFAULT 0, -- 이 판단으로 주문이 나갔는가
            exec_mode TEXT,                      -- paper | mock | live
            exec_note TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_agent_user "
                     "ON at_agent_analysis(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_agent_symbol "
                     "ON at_agent_analysis(user_id, symbol, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_agent_open "
                     "ON at_agent_analysis(user_id, correct, target_at)")
        _migrate_analysis(conn)


def _migrate_analysis(conn):
    """이미 만들어진 표에 새 컬럼을 덧붙입니다.

    `CREATE TABLE IF NOT EXISTS` 는 기존 표를 바꾸지 않습니다. 집행과 채점
    컬럼은 이 기능이 자라면서 늘어날 자리라, 처음부터 통로를 열어 둡니다.
    """
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(at_agent_analysis)")}
    additions = {
        "base_price": "REAL",
        "target_at": "TEXT",
        "resolved_at": "TEXT",
        "resolved_price": "REAL",
        "change_pct": "REAL",
        "correct": "INTEGER",
        "executed": "INTEGER NOT NULL DEFAULT 0",
        "exec_mode": "TEXT",
        "exec_note": "TEXT",
        "cache_read_tokens": "INTEGER NOT NULL DEFAULT 0",
        "cache_write_tokens": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, spec in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE at_agent_analysis ADD COLUMN {column} {spec}")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def get_config(user_id: int) -> dict:
    """저장된 설정 + 기본값 병합. 항목이 늘어도 기존 사용자가 깨지지 않습니다."""
    from engine import agents, agent_trader

    init()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM at_agent_config WHERE user_id = ?",
                           (user_id,)).fetchone()
    cfg = {**agents.DEFAULT_CONFIG, **agent_trader.DEFAULT_CONFIG}
    if row:
        try:
            stored = json.loads(row["config"])
            if isinstance(stored, dict):
                cfg.update(stored)
        except json.JSONDecodeError:
            pass
        cfg["updated_at"] = row["updated_at"]
    else:
        cfg["updated_at"] = ""
    return cfg


def save_config(user_id: int, patch: dict) -> dict:
    """설정 부분 갱신. 저장 단계에서 허용 범위로 잘라 둡니다.

    자르는 곳이 저장 단계인 이유는 storage/autotrade.save_config 와 같습니다 —
    한 번 잘라 두면 이후 어느 경로로 읽어도 한도 안입니다.
    """
    from engine import agents, agent_trader

    current = get_config(user_id)
    current.pop("updated_at", None)
    merged = {**current, **{k: v for k, v in (patch or {}).items()}}
    storable = {**agents.clamp_config(merged), **agent_trader.clamp_config(merged)}

    now = datetime.now().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO at_agent_config (user_id, updated_at, config) "
            "VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
            "updated_at = excluded.updated_at, config = excluded.config",
            (user_id, now, json.dumps(storable, ensure_ascii=False)))
    return get_config(user_id)


# ---------------------------------------------------------------------------
# 분석 기록
# ---------------------------------------------------------------------------

def _row(row, with_report: bool = False) -> dict:
    out = {
        "id": row["id"],
        "symbol": row["symbol"],
        "name": row["name"] or row["symbol"],
        "market": row["market"] or "",
        "decision": row["decision"],
        "confidence": row["confidence"],
        "horizon": row["horizon"] or "",
        "summary": row["summary"] or "",
        "model": row["model"],
        "calls": row["calls"],
        "tokens": {
            "input": row["input_tokens"], "output": row["output_tokens"],
            "cache_read": row["cache_read_tokens"],
            "cache_write": row["cache_write_tokens"],
        },
        "cost_usd": row["cost_usd"],
        "cost_krw": row["cost_krw"],
        "elapsed_ms": row["elapsed_ms"],
        "created_at": row["created_at"],
        "base_price": row["base_price"],
        "target_at": row["target_at"],
        "resolved_at": row["resolved_at"],
        "resolved_price": row["resolved_price"],
        "change_pct": row["change_pct"],
        "correct": None if row["correct"] is None else bool(row["correct"]),
        "executed": bool(row["executed"]),
        "exec_mode": row["exec_mode"] or "",
        "exec_note": row["exec_note"] or "",
    }
    if with_report:
        try:
            out["report"] = json.loads(row["report"])
        except (json.JSONDecodeError, TypeError):
            out["report"] = {"stages": [], "evidence": {}}
    return out


def save_analysis(user_id: int, result: dict,
                  horizon_days: int = DEFAULT_HORIZON_DAYS) -> int:
    """분석 결과 저장. 돌려주는 값은 화면이 곧바로 열 수 있는 행 번호입니다."""
    from datetime import timedelta

    init()
    usage = result.get("usage") or {}
    now = datetime.now()
    target_at = (now + timedelta(days=horizon_days)).isoformat()
    with _conn() as conn:
        cur = conn.execute("""
            INSERT INTO at_agent_analysis
            (user_id, symbol, name, market, decision, confidence, horizon,
             summary, report, model, calls, input_tokens, output_tokens,
             cache_read_tokens, cache_write_tokens, cost_usd, cost_krw,
             elapsed_ms, created_at, base_price, target_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, result.get("symbol", ""), result.get("name", ""),
            result.get("market", ""), result.get("decision", "hold"),
            result.get("confidence"), result.get("horizon", ""),
            result.get("summary", ""),
            json.dumps(result.get("report") or {}, ensure_ascii=False, default=str),
            result.get("model", ""), result.get("calls", 0),
            usage.get("input_tokens", 0), usage.get("output_tokens", 0),
            usage.get("cache_read_tokens", 0), usage.get("cache_write_tokens", 0),
            result.get("cost_usd", 0.0), result.get("cost_krw"),
            result.get("elapsed_ms"), now.isoformat(),
            result.get("price"), target_at,
        ))
        return cur.lastrowid


def get_analyses(user_id: int, limit: int = 50, symbol: str = "",
                 decision: str = "") -> list[dict]:
    init()
    sql = "SELECT * FROM at_agent_analysis WHERE user_id = ?"
    params: list = [user_id]
    if symbol:
        sql += " AND symbol = ?"
        params.append(symbol)
    if decision:
        sql += " AND decision = ?"
        params.append(decision)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        return [_row(r) for r in conn.execute(sql, params).fetchall()]


def get_analysis(user_id: int, analysis_id: int) -> dict | None:
    """리포트까지 통째로. 사용자 확인을 SQL 에서 합니다 (남의 판단을 못 엽니다)."""
    init()
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM at_agent_analysis WHERE id = ? AND user_id = ?",
            (analysis_id, user_id)).fetchone()
    return _row(row, with_report=True) if row else None


def latest_by_symbol(user_id: int, limit: int = 30) -> list[dict]:
    """종목마다 가장 최근 판단 하나씩. 화면 첫 화면이 이걸 보여줍니다."""
    init()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT * FROM at_agent_analysis WHERE id IN (
                SELECT MAX(id) FROM at_agent_analysis WHERE user_id = ?
                GROUP BY symbol
            ) ORDER BY id DESC LIMIT ?
        """, (user_id, limit)).fetchall()
    return [_row(r) for r in rows]


def mark_executed(analysis_id: int, mode: str, note: str = ""):
    with _conn() as conn:
        conn.execute(
            "UPDATE at_agent_analysis SET executed = 1, exec_mode = ?, "
            "exec_note = ? WHERE id = ?", (mode, note, analysis_id))


# ---------------------------------------------------------------------------
# 채점 (전진 검증)
# ---------------------------------------------------------------------------

def due_for_scoring(now: datetime = None, limit: int = 200) -> list[dict]:
    """만기가 지났는데 아직 채점하지 않은 판단.

    storage/db.get_matured_predictions 와 같은 역할입니다. 만기 **전**을 채점하면
    "방금 낸 판단이 즉시 맞았다"가 되어 성적표가 무의미해집니다.
    """
    init()
    now = now or datetime.now()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM at_agent_analysis WHERE correct IS NULL "
            "AND target_at IS NOT NULL AND target_at <= ? "
            "AND base_price IS NOT NULL ORDER BY id LIMIT ?",
            (now.isoformat(), limit)).fetchall()
    return [{"id": r["id"], "user_id": r["user_id"], "symbol": r["symbol"],
             "decision": r["decision"], "base_price": r["base_price"],
             "target_at": r["target_at"], "created_at": r["created_at"]}
            for r in rows]


def resolve(analysis_id: int, resolved_price: float, correct: bool,
            change_pct: float, now: datetime = None):
    now = now or datetime.now()
    with _conn() as conn:
        conn.execute(
            "UPDATE at_agent_analysis SET resolved_price = ?, correct = ?, "
            "change_pct = ?, resolved_at = ? WHERE id = ?",
            (resolved_price, int(correct), change_pct, now.isoformat(), analysis_id))


def void(analysis_id: int, note: str = ""):
    """채점 불가로 확정된 판단을 대기열에서 뺍니다.

    correct 는 NULL 로 두지 못합니다(그러면 영원히 대기열에 남습니다). 대신
    resolved_at 을 채우고 correct 를 -1 로 둬서 통계에서 빠지게 합니다.
    """
    with _conn() as conn:
        conn.execute(
            "UPDATE at_agent_analysis SET correct = -1, resolved_at = ?, "
            "exec_note = COALESCE(exec_note, '') || ? WHERE id = ?",
            (datetime.now().isoformat(), f" 채점 불가: {note}", analysis_id))


def scored_rows(user_id: int, limit: int = 500, decision: str = "") -> list[dict]:
    """채점이 끝난 판단. 성적표와 검증 게이트가 이 목록을 먹습니다."""
    init()
    sql = ("SELECT * FROM at_agent_analysis WHERE user_id = ? AND correct >= 0 "
           "AND resolved_at IS NOT NULL")
    params: list = [user_id]
    if decision:
        sql += " AND decision = ?"
        params.append(decision)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        return [_row(r) for r in conn.execute(sql, params).fetchall()]


def cost_summary(user_id: int, days: int = 30) -> dict:
    """비용 집계. 화면 상단에 "이번 달 얼마 썼는가"를 보여주기 위한 값입니다."""
    from datetime import timedelta

    init()
    since = (datetime.now() - timedelta(days=days)).isoformat()
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(cost_usd), 0) usd, "
            "COALESCE(SUM(cost_krw), 0) krw, COALESCE(SUM(calls), 0) calls "
            "FROM at_agent_analysis WHERE user_id = ? AND created_at >= ?",
            (user_id, since)).fetchone()
        total = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(cost_usd), 0) usd "
            "FROM at_agent_analysis WHERE user_id = ?", (user_id,)).fetchone()
    return {
        "days": days,
        "count": row["n"], "cost_usd": row["usd"], "cost_krw": row["krw"],
        "calls": row["calls"],
        "avg_usd": (row["usd"] / row["n"]) if row["n"] else 0.0,
        "total_count": total["n"], "total_usd": total["usd"],
    }


def purge(user_id: int, keep: int = 500):
    """오래된 판단 정리. 리포트 JSON 이 커서 무한히 쌓이면 DB 가 부풀어 오릅니다."""
    with _conn() as conn:
        conn.execute("""
            DELETE FROM at_agent_analysis WHERE user_id = ? AND id NOT IN (
                SELECT id FROM at_agent_analysis WHERE user_id = ?
                ORDER BY id DESC LIMIT ?
            )""", (user_id, user_id, keep))
