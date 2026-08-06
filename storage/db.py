import sqlite3
import json
from datetime import datetime, timedelta
from contextlib import contextmanager

from config import DB_PATH


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            horizon_label TEXT NOT NULL,
            predicted_at TEXT NOT NULL,
            direction TEXT NOT NULL,
            probability REAL NOT NULL,
            technical_score REAL,
            news_score REAL,
            community_score REAL,
            weights_used TEXT,
            actual_result TEXT,
            correct INTEGER
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS weight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            updated_at TEXT NOT NULL,
            technical REAL,
            news_sentiment REAL,
            community_sentiment REAL,
            accuracy_snapshot REAL
        )
        """)
        # 기존 DB에 컬럼을 덧붙임 (이미 있으면 무시)
        #   target_at   : 이 예측이 언제 만기되는가 — 시평선별 채점에 필수
        #   base_price  : 예측 당시 가격 — 채점 시 등락 방향의 기준
        #   resolved_price / resolved_at : 채점 결과 기록
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(predictions)")}
        for column, coltype in (("market", "TEXT"), ("name", "TEXT"),
                                ("target_at", "TEXT"), ("base_price", "REAL"),
                                ("resolved_price", "REAL"), ("resolved_at", "TEXT"),
                                ("user_id", "INTEGER")):
            if column not in existing:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {column} {coltype}")

        # 로그인 도입 전에 쌓인 기록은 '이전 기록' 계정으로 귀속시킵니다
        if "user_id" not in existing:
            from storage import users
            conn.execute("UPDATE predictions SET user_id = ? WHERE user_id IS NULL",
                         (users.legacy_user_id(),))
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_open "
                     "ON predictions(ticker, horizon_label, actual_result)")


# 같은 (종목, 시평선) 예측을 이 시간 안에 다시 내면 새로 쌓지 않고 갱신합니다.
# 대시보드가 4초마다 폴링하고 감시목록도 계속 갱신하기 때문에, 이 장치가 없으면
# "적중률 트랙레코드"가 단순 조회 횟수로 부풀어 통계가 무의미해집니다.
# (실제로 1분에 136건이 쌓여 삼성전자 1일 예측만 129건이 중복 저장된 적이 있습니다)
DEDUPE_MINUTES = 15


def save_prediction(pred, target_at=None, base_price=None,
                    dedupe_minutes: int = DEDUPE_MINUTES, user_id: int = None) -> int:
    """예측 저장.

    같은 종목·시평선의 **미채점** 예측이 dedupe_minutes 안에 있으면
    새로 INSERT 하지 않고 그 행을 최신 값으로 UPDATE 합니다.
    """
    predicted_at = pred.predicted_at
    target_iso = target_at.isoformat() if hasattr(target_at, "isoformat") else target_at

    with get_conn() as conn:
        if dedupe_minutes > 0:
            cutoff = (predicted_at - timedelta(minutes=dedupe_minutes)).isoformat()
            existing = conn.execute(
                "SELECT id FROM predictions "
                "WHERE ticker = ? AND horizon_label = ? AND actual_result IS NULL "
                "AND predicted_at >= ? AND user_id IS ? ORDER BY id DESC LIMIT 1",
                (pred.ticker, pred.horizon_label, cutoff, user_id),
            ).fetchone()

            if existing:
                conn.execute("""
                    UPDATE predictions SET
                      predicted_at = ?, direction = ?, probability = ?,
                      technical_score = ?, news_score = ?, community_score = ?,
                      weights_used = ?, market = ?, name = ?,
                      target_at = ?, base_price = ?
                    WHERE id = ?
                """, (
                    predicted_at.isoformat(), pred.direction, pred.probability,
                    pred.technical_score, pred.news_score, pred.community_score,
                    json.dumps(pred.weights_used),
                    getattr(pred, "market", ""), getattr(pred, "name", ""),
                    target_iso, base_price, existing["id"],
                ))
                return existing["id"]

        cur = conn.execute("""
            INSERT INTO predictions
            (ticker, horizon_label, predicted_at, direction, probability,
             technical_score, news_score, community_score, weights_used,
             market, name, target_at, base_price, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred.ticker, pred.horizon_label, predicted_at.isoformat(),
            pred.direction, pred.probability,
            pred.technical_score, pred.news_score, pred.community_score,
            json.dumps(pred.weights_used),
            getattr(pred, "market", ""), getattr(pred, "name", ""),
            target_iso, base_price, user_id,
        ))
        return cur.lastrowid


def get_unresolved_predictions(before_date: str = None):
    with get_conn() as conn:
        query = "SELECT * FROM predictions WHERE actual_result IS NULL"
        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]


def get_matured_predictions(ticker: str = None, now: datetime = None):
    """**만기가 지난** 미채점 예측만 반환.

    시평선이 6종(10분~1일)이 되면서, 방금 낸 '1일 뒤' 예측까지 즉시 채점하던
    문제를 막기 위한 함수입니다. target_at 이 비어 있는 예전 기록은
    horizon_label 로 만기를 역산합니다.
    """
    from config import PREDICTION_HORIZONS
    now = now or datetime.now()
    minutes_by_key = {h["key"]: h["minutes"] for h in PREDICTION_HORIZONS}
    # 구버전 라벨 호환 (하루 기준으로 취급)
    minutes_by_key.update({"open": 0, "5min": 5, "30min": 30, "close": 390})

    with get_conn() as conn:
        sql = "SELECT * FROM predictions WHERE actual_result IS NULL"
        params = ()
        if ticker:
            sql += " AND ticker = ?"
            params = (ticker,)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    matured = []
    for r in rows:
        target = r.get("target_at")
        if target:
            try:
                target_dt = datetime.fromisoformat(target)
            except ValueError:
                continue
        else:
            try:
                base = datetime.fromisoformat(r["predicted_at"])
            except (ValueError, KeyError):
                continue
            target_dt = base + timedelta(minutes=minutes_by_key.get(r["horizon_label"], 1440))
        if target_dt <= now:
            r["_target_dt"] = target_dt
            matured.append(r)
    return matured


def get_live_predictions(ticker: str = None, user_id: int = None,
                         now: datetime = None, limit: int = 40):
    """**아직 만기가 안 된** 미채점 예측 — 실시간 중간채점용.

    get_matured_predictions 의 반대편입니다. 저쪽은 '이제 채점해도 되는 것',
    이쪽은 '아직 진행 중이라 결과가 확정되지 않은 것'을 돌려줍니다.
    남은 시간과 기준가를 함께 주므로, 현재가만 있으면 화면에서
    "지금 채점하면 맞음/틀림"을 계산할 수 있습니다.
    """
    from config import PREDICTION_HORIZONS
    now = now or datetime.now()
    minutes_by_key = {h["key"]: h["minutes"] for h in PREDICTION_HORIZONS}
    minutes_by_key.update({"open": 0, "5min": 5, "30min": 30, "close": 390})

    sql = "SELECT * FROM predictions WHERE actual_result IS NULL"
    params = []
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker)
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    with get_conn() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    live = []
    for r in rows:
        target = r.get("target_at")
        try:
            if target:
                target_dt = datetime.fromisoformat(target)
            else:
                target_dt = (datetime.fromisoformat(r["predicted_at"])
                             + timedelta(minutes=minutes_by_key.get(r["horizon_label"], 1440)))
        except (ValueError, KeyError):
            continue
        if target_dt <= now:
            continue                      # 만기가 지난 것은 채점 대상이지 진행 중이 아님
        r["target_at"] = target_dt.isoformat()
        r["seconds_left"] = int((target_dt - now).total_seconds())
        live.append(r)

    live.sort(key=lambda r: r["seconds_left"])
    return live[:limit]


def void_prediction(pred_id: int, reason: str = ""):
    """채점 불가로 확정된 예측을 무효 처리.

    actual_result 에 'void' 를 넣어 대기열에서는 빠지되, correct 는 NULL 로 남겨
    적중률 통계에는 들어가지 않게 합니다.
    """
    with get_conn() as conn:
        conn.execute(
            "UPDATE predictions SET actual_result = 'void', correct = NULL, "
            "resolved_at = ? WHERE id = ?",
            (datetime.now().isoformat(), pred_id),
        )


def get_void_count(user_id: int = None) -> int:
    sql = "SELECT COUNT(*) c FROM predictions WHERE actual_result = 'void'"
    params = []
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    with get_conn() as conn:
        return conn.execute(sql, params).fetchone()["c"]


def resolve_prediction(pred_id: int, actual_result: str, correct: bool,
                       resolved_price: float = None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE predictions SET actual_result = ?, correct = ?, "
            "resolved_price = ?, resolved_at = ? WHERE id = ?",
            (actual_result, int(correct), resolved_price,
             datetime.now().isoformat(), pred_id),
        )


def get_accuracy_stats(ticker: str = None, user_id: int = None) -> dict:
    """누적 적중률 + **시평선별** 적중률.

    시평선마다 예측 대상 기간이 다르므로 하나로 뭉뚱그리면 해석이 어렵습니다.
    (10분 예측과 1일 예측의 적중률은 성격이 다릅니다)
    """
    from config import PREDICTION_HORIZONS
    labels = {h["key"]: h["label"] for h in PREDICTION_HORIZONS}
    order = {h["key"]: i for i, h in enumerate(PREDICTION_HORIZONS)}

    with get_conn() as conn:
        query = ("SELECT horizon_label, correct FROM predictions "
                 "WHERE correct IS NOT NULL")
        params = []
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        rows = conn.execute(query, params).fetchall()

    if not rows:
        return {"total": 0, "correct": 0, "accuracy": None, "by_horizon": []}

    total = len(rows)
    correct = sum(r["correct"] for r in rows)

    buckets = {}
    for r in rows:
        key = r["horizon_label"]
        b = buckets.setdefault(key, [0, 0])
        b[0] += 1
        b[1] += r["correct"]

    by_horizon = [
        {"horizon": k, "label": labels.get(k, k), "total": n, "correct": c,
         "accuracy": round(c / n * 100, 1) if n else None,
         "legacy": k not in labels}
        for k, (n, c) in buckets.items()
    ]
    by_horizon.sort(key=lambda x: order.get(x["horizon"], 99))

    return {"total": total, "correct": correct,
            "accuracy": round(correct / total * 100, 1),
            "by_horizon": by_horizon}


def get_scored_results(limit: int = 50, ticker: str = None,
                       user_id: int = None) -> list[dict]:
    """채점이 끝난 예측 목록 — 성적표에 '무엇을 어떻게 맞췄는지' 보여주기 위함."""
    with get_conn() as conn:
        sql = ("SELECT id, ticker, name, market, horizon_label, predicted_at, target_at, "
               "direction, probability, actual_result, correct, base_price, "
               "resolved_price, resolved_at FROM predictions "
               "WHERE correct IS NOT NULL")
        params = []
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker)
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        sql += " ORDER BY resolved_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    for r in rows:
        base, final = r.get("base_price"), r.get("resolved_price")
        r["change_pct"] = round((final / base - 1) * 100, 2) if base and final else None
    return rows


def get_accuracy_by_ticker(limit: int = 20, user_id: int = None) -> list[dict]:
    """종목별 적중률 — 어떤 종목에서 잘 맞추는지 확인용."""
    where = "WHERE correct IS NOT NULL"
    params = []
    if user_id is not None:
        where += " AND user_id = ?"
        params.append(user_id)
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT ticker, MAX(name) name, MAX(market) market,
                   COUNT(*) total, SUM(correct) correct
            FROM predictions {where}
            GROUP BY ticker ORDER BY total DESC LIMIT ?
        """, params).fetchall()
    return [{
        "ticker": r["ticker"], "name": r["name"] or r["ticker"], "market": r["market"],
        "total": r["total"], "correct": r["correct"] or 0,
        "accuracy": round((r["correct"] or 0) / r["total"] * 100, 1) if r["total"] else None,
    } for r in rows]


def get_pending_count(user_id: int = None) -> int:
    sql = "SELECT COUNT(*) c FROM predictions WHERE actual_result IS NULL"
    params = []
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    with get_conn() as conn:
        return conn.execute(sql, params).fetchone()["c"]


def save_weight_snapshot(weights: dict, accuracy: float):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO weight_history (updated_at, technical, news_sentiment, community_sentiment, accuracy_snapshot)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            weights["technical"], weights["news_sentiment"], weights["community_sentiment"],
            accuracy,
        ))


def get_latest_weights() -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM weight_history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return {
            "technical": row["technical"],
            "news_sentiment": row["news_sentiment"],
            "community_sentiment": row["community_sentiment"],
        }
