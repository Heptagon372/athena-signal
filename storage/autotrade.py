"""
자동매매 저장소 (설정 · 상태 · 감사 로그)
----------------------------------------
자동매매는 "무엇을 왜 샀는지"를 나중에 재구성할 수 있어야 합니다. 그래서 이
모듈은 설정뿐 아니라 **모든 판단과 주문을 append-only 로 기록**합니다.

테이블
    at_config          사용자별 설정(JSON) + 가동 상태
    at_position_state  포지션별 진입가·손절가·목표가·고점 (전략이 기억해야 하는 값)
    at_events          판단 로그 (신호·거부·진입·청산·오류) — append-only
    at_orders          주문 원장 — client_order_id 로 중복 방지
    at_daily           일자별 시작/최고/종료 평가금액 (손실 한도 판정 기준)

왜 포지션 상태를 따로 저장하는가
    브로커 잔고에는 "얼마에 샀나"만 있고 "손절을 어디에 걸었나"는 없습니다.
    서버를 재시작해도 손절선을 기억해야 하므로 DB 에 둡니다.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime

from config import DB_PATH

# ---------------------------------------------------------------------------
# 기본 설정 — 보수적인 값에서 시작합니다 (자동매매의 기본값은 안전이어야 합니다)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # 실행
    "mode": "paper",                  # paper | mock | live
    "interval_sec": 60,               # 회전 주기(초)
    "universe": ["005930", "000660", "069500"],   # 삼성전자 · SK하이닉스 · KODEX200
    "asset_classes": {"STOCK": True, "ETF": True, "FUTURES": False, "OPTION": False},

    # 신호
    "entry_score": 0.35,              # 이 점수를 넘으면 진입
    "exit_score": 0.05,               # 반대로 이만큼 넘어가면 청산
    "intraday_weight": 0.35,          # 분봉 점수 비중
    "use_news": False,                # 뉴스 감성 반영(느림)
    "news_weight": 0.25,
    "allow_short": False,             # 숏 진입 (파생만 가능)
    "allow_pyramiding": False,        # 보유 종목 추가 진입

    # NNFX 규칙 오버레이 (engine/nnfx.py). 기본은 꺼짐 — 신호 생성 방식을
    # 바꾸는 스위치라, 돌고 있는 계정에서 조용히 켜지면 안 됩니다.
    # off | soft(권장) | veto | hard(실측 기각)
    "nnfx_mode": "off",
    "nnfx_soft_weight": 0.30,         # 소프트 결합 가중 (실측 최적)
    "nnfx_baseline_period": 20,
    "nnfx_bridge_atr_mult": 3.0,      # 기준선에서 이만큼 멀면 추격 금지
    "nnfx_c2_adx_min": 20.0,          # 추세가 있다고 볼 최소 ADX
    "nnfx_volume_ratio_min": 0.7,
    # 앙상블 (engine/ensemble.py) — 시평선 합치·난기류·변동성 배수·비용 게이트.
    # observe(기본)는 계산해서 로그에만 남기고 매매 판단을 바꾸지 않습니다.
    # off | observe | soft(점수·손절폭 조정) | gate(진입 차단까지)
    "algo_mode": "observe",
    "algo_mtf_disagree_penalty": 0.5,   # 일봉·분봉 방향 불일치 시 확신 감산
    "algo_turbulence_pct": 95.0,        # 이 백분위를 넘으면 난기류로 판정
    "algo_turbulence_damping": 0.5,     # 난기류일 때 점수에 곱할 값
    "algo_vol_factor_min": 0.6,         # 변동성 배수 하한 (손절·익절 폭 스케일)
    "algo_vol_factor_max": 2.5,         # 변동성 배수 상한
    "algo_cost_edge_multiple": 3.0,     # 목표 수익 ≥ 왕복비용 × 이 배수 (gate)

    # ML 오버레이 (engine/mlsignal.py — XGBoost·PatchTST 이식).
    # observe(기본)는 GBDT·패치 어텐션 점수를 계산해 신호에 첨부만 하고,
    # soft 는 시간순 검증을 통과한 점수만 가중 결합합니다.
    "ml_mode": "observe",               # off | observe | soft
    "ml_weight": 0.25,                  # soft 결합 가중 (×확신도)
    "ml_horizon_bars": 5,               # 예측 시야 (봉)
    "ml_min_val_accuracy": 0.55,        # GBDT 검증 적중률 하한
    "ml_min_confidence": 0.25,          # 패치 어텐션 확신도 하한

    # 보호장치 (engine/protections.py — freqtrade protections 이식).
    # 최근 매매 이력이 나쁠 때 신규 진입을 잠급니다. 청산은 막지 않습니다.
    "protect_enabled": False,
    "protect_cooldown_min": 30,             # 청산 후 같은 종목 재진입 금지(분)
    "protect_stoploss_count": 3,            # lookback 안 손절 횟수 한도
    "protect_stoploss_lookback_min": 240,
    "protect_stoploss_stop_min": 60,
    "protect_stoploss_per_symbol": False,   # True 면 그 종목만 잠금
    "protect_drawdown_pct": 5.0,            # 실현손익 낙폭 한도 (총자산 대비 %)
    "protect_drawdown_lookback_min": 1440,
    "protect_drawdown_min_trades": 5,
    "protect_drawdown_stop_min": 240,
    "protect_lowprofit_pct": 0.0,           # 종목 성적 기준 (0 = 적자면 잠금)
    "protect_lowprofit_lookback_min": 1440,
    "protect_lowprofit_min_trades": 3,
    "protect_lowprofit_stop_min": 120,

    # 유니버스에 없는 종목(= 내가 직접 산 주식)까지 자동으로 관리할지.
    # 기본은 False — 자동매매가 기존 보유분을 마음대로 파는 것을 막습니다.
    "manage_only_universe": True,

    # AI 자동 추적 — 매매 대상을 다중 팩터 추천기가 주기적으로 갱신합니다
    "auto_universe": False,
    "auto_universe_size": 5,
    "auto_universe_refresh_min": 30,
    "auto_universe_markets": ["KR"],
    "auto_universe_min_score": 0.55,   # 이 점수 미만은 편입하지 않습니다
    "auto_universe_keep_manual": True,  # 손으로 넣은 종목은 유지
    # 사람이 직접 넣은 종목 — AI 갱신에서 지키기 위해 **명시적으로** 기록합니다.
    # (추천 목록과 대조해 추론하면, AI가 한 번 같이 뽑은 순간 '사람이 넣은 것'이라는
    #  사실이 사라져 다음 갱신에 조용히 빠집니다)
    "manual_universe": [],

    # 사이징
    "risk_per_trade_pct": 1.0,        # 1회 손절 시 잃을 총자산 비율
    "position_pct": 20.0,             # 종목당 최대 비중
    "max_positions": 5,
    "max_order_krw": 3_000_000,
    "min_order_krw": 100_000,
    # 소액 계좌 — 리스크 예산으로 1주 미만이 나와도, 현금·비중 한도 안이면
    # 최소 1주는 사게 허용 (1주 위험이 예산을 넘는다는 건 로그에 남습니다)
    "min_one_unit": False,
    "max_gross_exposure_pct": 100.0,  # 총 노출(명목) 한도

    # 청산
    "atr_stop_mult": 2.0,             # ATR 배수 손절 (0이면 아래 고정 %)
    "stop_loss_pct": 4.0,
    "take_profit_pct": 8.0,
    "trailing_stop_pct": 0.0,         # 0 = 사용 안 함
    "reward_risk": 2.0,
    "max_hold_days": 15,

    # 주문 집행
    "order_timeout_sec": 180,         # 이 시간 안에 안 체결되면 주문을 취소합니다
    "paper_slippage_bps": 5.0,        # 모의 계좌 체결에 적용할 불리한 스프레드

    # 페니주식 초단타 (engine/scalping.py — 하드 한도가 따로 있습니다)
    "scalp": {},

    # 안전장치
    "kill_switch": False,
    "daily_loss_limit_pct": 3.0,
    "max_drawdown_pct": 10.0,
    "regular_session_only": True,
    # 미국 주문 결제 재원 — "fx" 는 외화 예수금이 모자라면 주문을 내지 않고,
    # "krw" 는 원화(통합증거금)를 믿고 일단 보냅니다.
    #
    # 주의: 이건 **주문 방식 선택이 아닙니다.** KIS 해외주문 API 에는 통화를
    # 고르는 항목 자체가 없습니다(공식 샘플 확인). 앱의 [통합매수] 같은 별도
    # 주문이 API 에는 없어서, 우리가 정할 수 있는 건 "외화가 없을 때 미리
    # 막을 것인가, 보내보고 KIS 판단에 맡길 것인가" 뿐입니다.
    "us_order_funding": "fx",
    # 미국 종목만 프리·애프터마켓 매매 허용 (한국 시간외와 별개 스위치).
    # regular_session_only 를 통째로 끄면 한국 시간외 단일가까지 풀려버리므로
    # 미국 확장 시간만 따로 열 수 있게 분리했습니다.
    "us_extended_hours": False,
    "max_quote_age_sec": 120,
    "deriv_min_days_to_expiry": 2,
    "dry_run": False,                 # True 면 판단만 하고 주문을 내지 않습니다
}

# 상태 기계 — 자본을 움직이는 스위치라 상태를 명시적으로 둡니다
STOPPED = "stopped"        # 꺼짐
RUNNING = "running"        # 정상 가동
HALTED = "halted"          # 안전장치 발동으로 신규 진입 중단 (청산은 계속)

STATE_LABELS = {STOPPED: "정지", RUNNING: "가동 중", HALTED: "안전 중단"}


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
        CREATE TABLE IF NOT EXISTS at_config (
            user_id INTEGER PRIMARY KEY,
            updated_at TEXT NOT NULL,
            config TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'stopped',
            state_reason TEXT DEFAULT ''
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_position_state (
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'paper',   -- paper | mock | live
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL,
            stop_price REAL,
            target_price REAL,
            peak_price REAL,
            quantity REAL,
            signal_score REAL,
            opened_at TEXT,
            updated_at TEXT,
            note TEXT,
            strategy TEXT,                        -- auto | scalp (누가 잡은 포지션인가)
            PRIMARY KEY (user_id, mode, symbol)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            level TEXT NOT NULL,          -- info | warn | error | trade
            kind TEXT NOT NULL,           -- signal | entry | exit | reject | halt | tick | error
            symbol TEXT DEFAULT '',
            name TEXT DEFAULT '',
            message TEXT NOT NULL,
            detail TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_events_user "
                     "ON at_events(user_id, id DESC)")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            client_order_id TEXT NOT NULL UNIQUE,
            broker_mode TEXT NOT NULL,
            broker_order_id TEXT DEFAULT '',
            symbol TEXT NOT NULL,
            name TEXT DEFAULT '',
            asset_class TEXT DEFAULT '',
            action TEXT NOT NULL,          -- buy | sell | open_long | open_short | close
            side TEXT DEFAULT '',
            quantity REAL NOT NULL,
            price REAL,
            price_krw REAL,
            fee REAL DEFAULT 0,
            realized_pnl REAL,
            status TEXT NOT NULL,          -- filled | partial | pending | cancelled | rejected
            reason TEXT DEFAULT '',
            detail TEXT,
            -- 체결 추적 (실계좌는 '접수'와 '체결'이 다른 사건입니다)
            intended_price REAL,           -- 주문을 낼 때 보고 있던 가격
            filled_quantity REAL DEFAULT 0,
            avg_fill_price REAL,           -- 실제 평균 체결가
            slippage_bps REAL,             -- (체결가 - 의도가) / 의도가 × 10000
            settled_at TEXT,               -- 체결·취소가 확정된 시각
            poll_count INTEGER DEFAULT 0   -- 체결 확인을 몇 번 시도했는지
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_orders_user "
                     "ON at_orders(user_id, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_at_orders_open "
                     "ON at_orders(user_id, status)")
        _migrate_orders(conn)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_recommend (
            user_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT,
            market TEXT,
            rank INTEGER,
            score REAL,
            price REAL,
            picked INTEGER NOT NULL DEFAULT 0,   -- 실제로 매매 대상에 넣었는가
            regime TEXT,
            reasons TEXT,                        -- 추천 근거 (JSON)
            factors TEXT,                        -- 팩터 원값 (JSON)
            created_at TEXT NOT NULL,
            PRIMARY KEY (user_id, symbol)
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS at_daily (
            user_id INTEGER NOT NULL,
            mode TEXT NOT NULL DEFAULT 'paper',
            trade_date TEXT NOT NULL,
            start_value REAL NOT NULL,
            peak_value REAL NOT NULL,
            end_value REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0,
            trade_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, mode, trade_date)
        )""")
        _migrate_scope(conn)
        # 모드 분리로 테이블을 새로 만드는 경우가 있어서, 컬럼 추가는 그 뒤입니다
        _migrate_position_state(conn)


def _migrate_scope(conn):
    """포지션 상태·일자 기준값을 **계좌(모드)별로** 분리합니다.

    분리 전에는 모드가 달라도 같은 행을 썼습니다. 그래서 모의계좌에서 쌓은
    기준 평가금액(예: 1천만원)이 실전 계좌 판정에 그대로 쓰였고,
    실계좌가 300만원이면 첫 회전에 '일일 손실 -70%'로 오판해 멈췄습니다.
    모의 계좌의 손절선이 실계좌 포지션에 붙는 문제도 같은 뿌리였습니다.

    SQLite 는 기본키를 바꿀 수 없어 테이블을 새로 만들고 옮깁니다.
    기존 데이터는 전부 모의계좌(paper)에서 나온 것이므로 'paper' 로 귀속시킵니다.
    """
    for table, create, columns in (
        ("at_position_state", """
            CREATE TABLE at_position_state (
                user_id INTEGER NOT NULL,
                mode TEXT NOT NULL DEFAULT 'paper',
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL, stop_price REAL, target_price REAL,
                peak_price REAL, quantity REAL, signal_score REAL,
                opened_at TEXT, updated_at TEXT, note TEXT,
                PRIMARY KEY (user_id, mode, symbol)
            )""",
         "user_id, symbol, side, entry_price, stop_price, target_price, "
         "peak_price, quantity, signal_score, opened_at, updated_at, note"),
        ("at_daily", """
            CREATE TABLE at_daily (
                user_id INTEGER NOT NULL,
                mode TEXT NOT NULL DEFAULT 'paper',
                trade_date TEXT NOT NULL,
                start_value REAL NOT NULL, peak_value REAL NOT NULL,
                end_value REAL NOT NULL,
                realized_pnl REAL NOT NULL DEFAULT 0,
                trade_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, mode, trade_date)
            )""",
         "user_id, trade_date, start_value, peak_value, end_value, "
         "realized_pnl, trade_count"),
    ):
        info = list(conn.execute(f"PRAGMA table_info({table})"))
        if not info or any(r["name"] == "mode" for r in info):
            continue
        conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
        conn.execute(create)
        conn.execute(
            f"INSERT INTO {table} (mode, {columns}) "
            f"SELECT 'paper', {columns} FROM {table}_old")
        conn.execute(f"DROP TABLE {table}_old")
        print(f"[autotrade] {table} 를 계좌(모드)별로 분리했습니다.")


def _migrate_orders(conn):
    """체결 추적 컬럼을 나중에 붙입니다.

    자동매매를 이미 돌린 설치본은 at_orders 가 옛 스키마로 만들어져 있고,
    `CREATE TABLE IF NOT EXISTS` 는 기존 테이블을 바꾸지 않습니다.
    컬럼이 없으면 체결 확인 로직이 전부 실패하므로 여기서 채웁니다.
    """
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(at_orders)")}
    additions = {
        "intended_price": "REAL",
        "filled_quantity": "REAL DEFAULT 0",
        "avg_fill_price": "REAL",
        "slippage_bps": "REAL",
        "settled_at": "TEXT",
        "poll_count": "INTEGER DEFAULT 0",
    }
    for column, spec in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE at_orders ADD COLUMN {column} {spec}")


def _migrate_position_state(conn):
    """포지션마다 '누가 잡았는가'를 붙입니다.

    페니 초단타와 일반 자동매매는 청산 규칙이 완전히 다릅니다(틱·초 vs %·일).
    누가 잡은 포지션인지 모르면, 초단타가 산 종목을 일반 전략이 며칠씩 들고
    있거나 반대로 일반 종목을 초단타가 3틱에 던집니다. 그래서 진입 시점에
    주인을 박아두고, 청산은 **그 주인의 규칙으로만** 합니다.
    """
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(at_position_state)")}
    if "strategy" not in existing:
        conn.execute("ALTER TABLE at_position_state ADD COLUMN strategy TEXT")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def get_config(user_id: int) -> dict:
    """저장된 설정 + 기본값 병합. 새 설정 항목이 추가돼도 기존 사용자가 깨지지 않습니다."""
    init()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM at_config WHERE user_id = ?",
                           (user_id,)).fetchone()
    cfg = dict(DEFAULT_CONFIG)
    if row:
        try:
            stored = json.loads(row["config"])
            if isinstance(stored, dict):
                # 중첩 dict(asset_classes)는 통째로 덮어쓰지 않고 병합합니다
                classes = dict(DEFAULT_CONFIG["asset_classes"])
                classes.update(stored.get("asset_classes") or {})
                cfg.update(stored)
                cfg["asset_classes"] = classes
        except json.JSONDecodeError:
            pass
        cfg["enabled"] = bool(row["enabled"])
        cfg["state"] = row["state"]
        cfg["state_reason"] = row["state_reason"] or ""
        cfg["updated_at"] = row["updated_at"]
    else:
        cfg["enabled"] = False
        cfg["state"] = STOPPED
        cfg["state_reason"] = ""
        cfg["updated_at"] = ""
    return cfg


# get_config 가 붙여주는 실행 상태 — 별도 컬럼이라 config JSON 에 넣지 않습니다
_RUNTIME_KEYS = ("enabled", "state", "state_reason", "updated_at")


def save_config(user_id: int, patch: dict) -> dict:
    """설정 부분 갱신. 알 수 없는 키는 무시해 오타로 엔진이 망가지는 것을 막습니다."""
    init()
    current = get_config(user_id)
    for key, value in (patch or {}).items():
        if key not in DEFAULT_CONFIG:
            continue
        if key == "asset_classes" and isinstance(value, dict):
            merged = dict(current.get("asset_classes") or {})
            merged.update({k: bool(v) for k, v in value.items()})
            current[key] = merged
        elif key == "scalp" and isinstance(value, dict):
            # 초단타 설정은 병합 + 하드 한도로 강제 축소해서 저장합니다.
            # 저장 단계에서 잘라두면, 이후 어떤 경로로 읽어도 한도 안입니다.
            from engine import scalping
            before = current.get("scalp") or {}
            merged = {**before, **value}
            if "pinned" in value:
                # 지정에서 뺀 종목은 추적 목록에서도 빠져야 합니다.
                # (clamp_config 는 지정 종목을 목록 앞에 넣어줄 뿐, 뺀 것을
                #  알아채지 못합니다 — 이전 값을 아는 여기서 걷어냅니다)
                unpinned = set(before.get("pinned") or []) - set(value.get("pinned") or [])
                merged["universe"] = [s for s in (merged.get("universe") or [])
                                      if s not in unpinned]
            capped = scalping.clamp_config(merged)
            capped.pop("_clamped", None)
            current[key] = capped
        else:
            current[key] = value

    # 저장은 '아는 키만'이 아니라 '읽어온 것 + 아는 키'로 합니다.
    # 아는 키만 남기면, 구버전 코드가 한 번이라도 저장하는 순간 새로 추가된
    # 설정이 조용히 사라집니다 (그 설정을 켠 사람은 왜 안 먹는지 알 수 없습니다).
    storable = {k: v for k, v in current.items() if k not in _RUNTIME_KEYS}
    now = datetime.now().isoformat()
    with _conn() as conn:
        exists = conn.execute("SELECT 1 FROM at_config WHERE user_id = ?",
                              (user_id,)).fetchone()
        if exists:
            conn.execute("UPDATE at_config SET config = ?, updated_at = ? WHERE user_id = ?",
                         (json.dumps(storable, ensure_ascii=False), now, user_id))
        else:
            conn.execute(
                "INSERT INTO at_config (user_id, updated_at, config, enabled, state) "
                "VALUES (?, ?, ?, 0, 'stopped')",
                (user_id, now, json.dumps(storable, ensure_ascii=False)))
    return get_config(user_id)


def set_enabled(user_id: int, enabled: bool, reason: str = "") -> dict:
    init()
    save_config(user_id, {})          # 행이 없으면 만들어 둡니다
    state = RUNNING if enabled else STOPPED
    with _conn() as conn:
        conn.execute(
            "UPDATE at_config SET enabled = ?, state = ?, state_reason = ?, updated_at = ? "
            "WHERE user_id = ?",
            (1 if enabled else 0, state, reason, datetime.now().isoformat(), user_id))
    return get_config(user_id)


def set_state(user_id: int, state: str, reason: str = ""):
    init()
    with _conn() as conn:
        conn.execute(
            "UPDATE at_config SET state = ?, state_reason = ?, updated_at = ? "
            "WHERE user_id = ?",
            (state, reason, datetime.now().isoformat(), user_id))


def enabled_users() -> list[int]:
    """자동매매가 켜진 사용자 목록 (엔진 루프가 매 회전 조회)."""
    init()
    with _conn() as conn:
        return [int(r["user_id"]) for r in conn.execute(
            "SELECT user_id FROM at_config WHERE enabled = 1")]


def configured_users() -> list[int]:
    """자동매매 설정이 있는 모든 사용자 (장 상태 알림 대상).

    꺼져 있어도 알림은 받습니다 — "장이 열렸으니 켤까"를 판단하는 데 필요합니다.
    """
    init()
    with _conn() as conn:
        return [int(r["user_id"]) for r in conn.execute(
            "SELECT user_id FROM at_config")]


# ---------------------------------------------------------------------------
# 포지션 상태
# ---------------------------------------------------------------------------

def get_position_states(user_id: int, mode: str = "paper") -> dict:
    init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM at_position_state WHERE user_id = ? AND mode = ?",
            (user_id, mode))
        return {r["symbol"]: dict(r) for r in rows}


def upsert_position_state(user_id: int, mode: str, symbol: str, **fields):
    init()
    now = datetime.now().isoformat()
    allowed = ("side", "entry_price", "stop_price", "target_price", "peak_price",
               "quantity", "signal_score", "opened_at", "note", "strategy")
    data = {k: v for k, v in fields.items() if k in allowed}
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM at_position_state WHERE user_id = ? AND mode = ? AND symbol = ?",
            (user_id, mode, symbol)).fetchone()
        if row:
            if not data:
                return
            sets = ", ".join(f"{k} = ?" for k in data)
            conn.execute(
                f"UPDATE at_position_state SET {sets}, updated_at = ? "
                "WHERE user_id = ? AND mode = ? AND symbol = ?",
                (*data.values(), now, user_id, mode, symbol))
        else:
            data.setdefault("opened_at", now)
            cols = ", ".join(data)
            marks = ", ".join("?" for _ in data)
            conn.execute(
                f"INSERT INTO at_position_state (user_id, mode, symbol, updated_at, {cols}) "
                f"VALUES (?, ?, ?, ?, {marks})",
                (user_id, mode, symbol, now, *data.values()))


def clear_position_state(user_id: int, mode: str, symbol: str):
    init()
    with _conn() as conn:
        conn.execute(
            "DELETE FROM at_position_state WHERE user_id = ? AND mode = ? AND symbol = ?",
            (user_id, mode, symbol))


# ---------------------------------------------------------------------------
# 이벤트 로그 (감사 추적)
# ---------------------------------------------------------------------------

def log_event(user_id: int, kind: str, message: str, level: str = "info",
              symbol: str = "", name: str = "", detail: dict = None) -> int:
    init()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO at_events (user_id, created_at, level, kind, symbol, name, "
            "message, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, datetime.now().isoformat(), level, kind, symbol, name, message,
             json.dumps(detail, ensure_ascii=False, default=str) if detail else None))
        return cur.lastrowid


def get_events(user_id: int, limit: int = 100, kinds: list = None,
               after_id: int = 0) -> list[dict]:
    init()
    sql = "SELECT * FROM at_events WHERE user_id = ?"
    params: list = [user_id]
    if kinds:
        sql += f" AND kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    if after_id:
        sql += " AND id > ?"
        params.append(after_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(sql, params)]
    for r in rows:
        if r.get("detail"):
            try:
                r["detail"] = json.loads(r["detail"])
            except json.JSONDecodeError:
                pass
    return rows


def purge_events(user_id: int, keep: int = 2000):
    """로그가 무한히 쌓이지 않게 오래된 것부터 지웁니다."""
    init()
    with _conn() as conn:
        conn.execute(
            "DELETE FROM at_events WHERE user_id = ? AND id NOT IN "
            "(SELECT id FROM at_events WHERE user_id = ? ORDER BY id DESC LIMIT ?)",
            (user_id, user_id, keep))


# ---------------------------------------------------------------------------
# 주문 원장
# ---------------------------------------------------------------------------

def order_exists(client_order_id: str) -> bool:
    """이미 낸 주문인가 — 재시작/재시도 시 이중 주문을 막는 마지막 방어선."""
    init()
    with _conn() as conn:
        return conn.execute("SELECT 1 FROM at_orders WHERE client_order_id = ?",
                            (client_order_id,)).fetchone() is not None


def record_order(user_id: int, order: dict) -> int | None:
    init()
    filled = order.get("filled_quantity")
    if filled is None:
        # 즉시 체결된 주문은 체결수량을 따로 주지 않으므로 주문수량으로 채웁니다
        filled = float(order.get("quantity") or 0) if order.get("status") == "filled" else 0.0
    with _conn() as conn:
        try:
            cur = conn.execute(
                "INSERT INTO at_orders (user_id, created_at, client_order_id, broker_mode, "
                "broker_order_id, symbol, name, asset_class, action, side, quantity, price, "
                "price_krw, fee, realized_pnl, status, reason, detail, intended_price, "
                "filled_quantity, avg_fill_price, slippage_bps, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, datetime.now().isoformat(), order.get("client_order_id", ""),
                 order.get("broker_mode", ""), order.get("broker_order_id", ""),
                 order.get("symbol", ""), order.get("name", ""),
                 order.get("asset_class", ""), order.get("action", ""),
                 order.get("side", ""), float(order.get("quantity") or 0),
                 order.get("price"), order.get("price_krw"),
                 float(order.get("fee") or 0), order.get("realized_pnl"),
                 order.get("status", ""), order.get("reason", ""),
                 json.dumps(order.get("detail"), ensure_ascii=False, default=str)
                 if order.get("detail") else None,
                 order.get("intended_price"), float(filled or 0),
                 order.get("avg_fill_price"), order.get("slippage_bps"),
                 datetime.now().isoformat()
                 if order.get("status") in ("filled", "rejected", "cancelled") else None))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # 같은 client_order_id — 이미 기록된 주문입니다
            return None


# 아직 결과가 확정되지 않은 상태들 (매 회전 브로커에 조회합니다)
OPEN_STATUSES = ("pending", "partial", "cancel_requested")


def open_orders(user_id: int, mode: str = None) -> list[dict]:
    """체결이 확정되지 않은 주문들. 엔진이 매 회전 이 목록부터 정리합니다.

    mode 를 주면 그 계좌의 주문만 봅니다. 모의계좌에 남은 주문이 실전 주문을
    막거나, 실전 주문번호를 모의 브로커에 조회하는 사고를 막기 위해서입니다.
    """
    init()
    marks = ",".join("?" * len(OPEN_STATUSES))
    sql = f"SELECT * FROM at_orders WHERE user_id = ? AND status IN ({marks})"
    params = [user_id, *OPEN_STATUSES]
    if mode:
        sql += " AND broker_mode = ?"
        params.append(mode)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(sql + " ORDER BY id", params)]
    for r in rows:
        if r.get("detail"):
            try:
                r["detail"] = json.loads(r["detail"])
            except json.JSONDecodeError:
                pass
    return rows


def has_open_order(user_id: int, mode: str, symbol: str) -> bool:
    """이 계좌의 이 종목에 아직 처리 중인 주문이 있는가.

    미체결 주문을 무시하고 또 진입하면 같은 종목을 두 번 사게 됩니다.
    """
    init()
    marks = ",".join("?" * len(OPEN_STATUSES))
    with _conn() as conn:
        return conn.execute(
            f"SELECT 1 FROM at_orders WHERE user_id = ? AND broker_mode = ? "
            f"AND symbol = ? AND status IN ({marks}) LIMIT 1",
            (user_id, mode, symbol, *OPEN_STATUSES)).fetchone() is not None


def update_order(order_id: int, **fields):
    """주문 상태 갱신 (체결 확인·취소 결과 반영)."""
    init()
    allowed = ("status", "broker_order_id", "filled_quantity", "avg_fill_price",
               "slippage_bps", "settled_at", "poll_count", "reason", "fee",
               "realized_pnl", "price")
    data = {k: v for k, v in fields.items() if k in allowed}
    if not data:
        return
    sets = ", ".join(f"{k} = ?" for k in data)
    with _conn() as conn:
        conn.execute(f"UPDATE at_orders SET {sets} WHERE id = ?",
                     (*data.values(), order_id))


def bump_poll(order_id: int) -> int:
    init()
    with _conn() as conn:
        conn.execute("UPDATE at_orders SET poll_count = COALESCE(poll_count, 0) + 1 "
                     "WHERE id = ?", (order_id,))
        row = conn.execute("SELECT poll_count FROM at_orders WHERE id = ?",
                           (order_id,)).fetchone()
    return int(row["poll_count"]) if row else 0


def execution_quality(user_id: int, limit: int = 200, mode: str = None) -> dict:
    """체결 품질 요약 — 의도한 가격과 실제 체결가가 얼마나 벌어졌는지.

    이 숫자가 백테스트 가정(슬리피지 5bp)보다 크면 전략이 아니라 집행이
    수익을 깎아먹고 있는 것입니다.
    """
    init()
    scope = " AND broker_mode = ?" if mode else ""
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT slippage_bps, status FROM at_orders "
            f"WHERE user_id = ? AND slippage_bps IS NOT NULL{scope} "
            "ORDER BY id DESC LIMIT ?",
            ((user_id, mode, limit) if mode else (user_id, limit)))]
        counts = {r["status"]: r["n"] for r in conn.execute(
            f"SELECT status, COUNT(*) n FROM at_orders WHERE user_id = ?{scope} "
            "GROUP BY status", ((user_id, mode) if mode else (user_id,)))}
    values = [float(r["slippage_bps"]) for r in rows]
    return {
        "samples": len(values),
        "avg_slippage_bps": round(sum(values) / len(values), 2) if values else 0.0,
        "worst_slippage_bps": round(max(values, key=abs), 2) if values else 0.0,
        "status_counts": counts,
    }


def get_orders(user_id: int, limit: int = 50, mode: str = None) -> list[dict]:
    init()
    sql = "SELECT * FROM at_orders WHERE user_id = ?"
    params: list = [user_id]
    if mode:
        sql += " AND broker_mode = ?"
        params.append(mode)
    params.append(limit)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(sql + " ORDER BY id DESC LIMIT ?", params)]
    for r in rows:
        if r.get("detail"):
            try:
                r["detail"] = json.loads(r["detail"])
            except json.JSONDecodeError:
                pass
    return rows


def fills_for(user_id: int, mode: str, symbol: str, since: str = None,
              limit: int = 200) -> list[dict]:
    """이 종목에서 **실제로 체결된** 내 주문들 — 차트에 산 지점/판 지점을 찍는 용도.

    접수만 되고 체결 안 된 주문은 뺍니다. 차트에 "여기서 샀다"고 찍어놓고
    실제로는 안 샀으면 그건 거짓말입니다.

    가격은 `avg_fill_price`(실제 체결가)를 우선합니다. `price`(주문가)로 찍으면
    지정가 주문에서 마커가 실제 체결 자리와 어긋납니다.
    """
    init()
    since = since or datetime.now().strftime("%Y-%m-%d")
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT created_at, action, side, quantity, filled_quantity, price, "
            "       avg_fill_price, realized_pnl, status, reason "
            "FROM at_orders WHERE user_id = ? AND broker_mode = ? AND symbol = ? "
            "  AND created_at >= ? AND status IN ('filled', 'partial') "
            "ORDER BY created_at ASC LIMIT ?",
            (user_id, mode, str(symbol), since, limit))]

    out = []
    for row in rows:
        price = row.get("avg_fill_price") or row.get("price")
        if not price:
            continue          # 체결가를 모르면 찍을 자리가 없습니다
        out.append({
            "at": row["created_at"],
            "sec_of_day": _sec_of_day_iso(row["created_at"]),
            "action": row["action"],              # buy | sell | close
            "side": row["side"],
            "price": float(price),
            "quantity": float(row.get("filled_quantity") or row.get("quantity") or 0),
            "realized_pnl": row.get("realized_pnl"),
            "status": row["status"],
            "reason": row.get("reason") or "",
        })
    return out


def closed_trades(user_id: int, mode: str, since: str = None,
                  limit: int = 500) -> list[dict]:
    """확정된(손익이 잡힌) 청산 기록 — 보호장치(engine/protections.py)의 판단 재료.

    `realized_pnl` 이 채워진 주문만 봅니다. 접수만 된 주문이나 진입 주문은
    "그 매매가 어떻게 끝났는지"를 말해주지 않기 때문입니다.

    시각은 `settled_at`(체결 확정 시각)을 우선합니다. 접수 시각으로 재면,
    체결까지 오래 걸린 주문이 실제보다 이른 시점의 사건으로 집계돼
    쿨다운이 일찍 풀립니다.
    """
    init()
    sql = ("SELECT symbol, name, action, side, quantity, realized_pnl, reason, "
           "       COALESCE(settled_at, created_at) AS closed_at, "
           "       avg_fill_price, price, status "
           "FROM at_orders WHERE user_id = ? AND broker_mode = ? "
           "  AND realized_pnl IS NOT NULL AND status IN ('filled', 'partial')")
    params: list = [user_id, mode]
    if since:
        sql += " AND COALESCE(settled_at, created_at) >= ?"
        params.append(since)
    params.append(limit)
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            sql + " ORDER BY closed_at DESC LIMIT ?", params)]
    return rows


def _sec_of_day_iso(text: str) -> int:
    """ISO 시각 문자열 → 자정으로부터의 초 (차트 x축과 맞추기 위해)."""
    try:
        moment = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return 0
    return moment.hour * 3600 + moment.minute * 60 + moment.second


def realized_today(user_id: int, mode: str, symbols: list = None) -> dict:
    """오늘 확정된 손익 (원). 종목을 주면 그 종목들만.

    페니 초단타의 일일 손실 한도는 **금액**이라 계좌 전체 평가금액 변화가
    아니라 이 값으로 판정해야 합니다. 계좌 전체로 재면, 초단타와 무관한
    보유 종목이 빠졌을 때 초단타가 멈추는 엉뚱한 일이 생깁니다.
    """
    init()
    today = datetime.now().strftime("%Y-%m-%d")
    sql = ("SELECT COALESCE(SUM(realized_pnl), 0) AS pnl, COUNT(*) AS n "
           "FROM at_orders WHERE user_id = ? AND broker_mode = ? "
           "AND realized_pnl IS NOT NULL AND substr(created_at, 1, 10) = ?")
    params: list = [user_id, mode, today]
    if symbols:
        placeholders = ",".join("?" * len(symbols))
        sql += f" AND symbol IN ({placeholders})"
        params.extend(list(symbols))
    with _conn() as conn:
        row = conn.execute(sql, params).fetchone()
    return {"realized_pnl": float(row["pnl"] or 0), "closed_count": int(row["n"] or 0)}


# ---------------------------------------------------------------------------
# 일자별 기준값 (손실 한도 판정)
# ---------------------------------------------------------------------------

def save_recommendations(user_id: int, recommendations: list, picked: list):
    """AI 추천 결과를 통째로 갈아끼웁니다 (최신 한 벌만 유지).

    화면에서 "왜 이 종목이 매매 대상에 들어왔는지"를 새로고침 후에도
    볼 수 있어야 하므로 DB 에 남깁니다.
    """
    init()
    now = datetime.now().isoformat()
    picked_set = set(picked or [])
    with _conn() as conn:
        conn.execute("DELETE FROM at_recommend WHERE user_id = ?", (user_id,))
        for rec in recommendations:
            data = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
            conn.execute(
                "INSERT OR REPLACE INTO at_recommend (user_id, symbol, name, market, "
                "rank, score, price, picked, regime, reasons, factors, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, data.get("key", ""), data.get("name", ""),
                 data.get("market", ""), data.get("rank", 0), data.get("score", 0),
                 data.get("price_krw") or data.get("price") or 0,
                 1 if data.get("key") in picked_set else 0,
                 data.get("regime", ""),
                 json.dumps(data.get("reasons") or [], ensure_ascii=False),
                 json.dumps(data.get("factors") or {}, ensure_ascii=False, default=str),
                 now))


def get_recommendations(user_id: int) -> list[dict]:
    init()
    with _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM at_recommend WHERE user_id = ? ORDER BY rank", (user_id,))]
    for r in rows:
        for field_name in ("reasons", "factors"):
            if r.get(field_name):
                try:
                    r[field_name] = json.loads(r[field_name])
                except json.JSONDecodeError:
                    pass
        r["picked"] = bool(r["picked"])
    return rows


def touch_daily(user_id: int, mode: str, total_value: float,
                trade_date: str = None) -> dict:
    """오늘의 시작·최고·현재 평가금액을 갱신하고 돌려줍니다.

    일일 손실 한도는 '오늘 시작 대비'로 재는 것이 맞습니다. 누적 손익으로 재면
    어제까지 벌어둔 돈이 오늘의 손실을 가려 한도가 작동하지 않습니다.

    **계좌(모드)별로 따로 기록합니다.** 모의계좌 1천만원을 기준으로 실계좌
    300만원을 재면 첫 회전에 '-70% 손실'로 오판합니다.
    """
    init()
    trade_date = trade_date or date.today().isoformat()
    total_value = float(total_value or 0)
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM at_daily WHERE user_id = ? AND mode = ? AND trade_date = ?",
            (user_id, mode, trade_date)).fetchone()
        if not row:
            conn.execute(
                "INSERT INTO at_daily (user_id, mode, trade_date, start_value, "
                "peak_value, end_value) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, mode, trade_date, total_value, total_value, total_value))
            return {"trade_date": trade_date, "mode": mode, "start_value": total_value,
                    "peak_value": total_value, "end_value": total_value,
                    "realized_pnl": 0.0, "trade_count": 0}

        peak = max(float(row["peak_value"]), total_value)
        conn.execute(
            "UPDATE at_daily SET peak_value = ?, end_value = ? "
            "WHERE user_id = ? AND mode = ? AND trade_date = ?",
            (peak, total_value, user_id, mode, trade_date))
        return {"trade_date": trade_date, "mode": mode,
                "start_value": float(row["start_value"]),
                "peak_value": peak, "end_value": total_value,
                "realized_pnl": float(row["realized_pnl"]),
                "trade_count": int(row["trade_count"])}


def record_daily_trade(user_id: int, mode: str, realized_pnl: float = 0.0,
                       trade_date: str = None):
    init()
    trade_date = trade_date or date.today().isoformat()
    with _conn() as conn:
        conn.execute(
            "UPDATE at_daily SET realized_pnl = realized_pnl + ?, "
            "trade_count = trade_count + 1 "
            "WHERE user_id = ? AND mode = ? AND trade_date = ?",
            (float(realized_pnl or 0), user_id, mode, trade_date))


def get_daily_history(user_id: int, limit: int = 30, mode: str = None) -> list[dict]:
    init()
    sql = "SELECT * FROM at_daily WHERE user_id = ?"
    params: list = [user_id]
    if mode:
        sql += " AND mode = ?"
        params.append(mode)
    params.append(limit)
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            sql + " ORDER BY trade_date DESC LIMIT ?", params)]


def summary(user_id: int, mode: str = None) -> dict:
    """콘솔 상단에 띄울 요약 통계 (계좌별).

    가상 자금 성적과 실제 자금 성적을 한 숫자로 섞으면 아무 의미가 없습니다.
    """
    init()
    scope = " AND broker_mode = ?" if mode else ""
    args = (user_id, mode) if mode else (user_id,)
    with _conn() as conn:
        orders = conn.execute(
            "SELECT COUNT(*) c, SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) r "
            f"FROM at_orders WHERE user_id = ?{scope}", args).fetchone()
        pnl = conn.execute(
            "SELECT COUNT(*) n, SUM(realized_pnl) total, "
            "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) wins "
            f"FROM at_orders WHERE user_id = ? AND realized_pnl IS NOT NULL{scope}",
            args).fetchone()
    closed = int(pnl["n"] or 0)
    wins = int(pnl["wins"] or 0)
    return {
        "orders": int(orders["c"] or 0),
        "rejected": int(orders["r"] or 0),
        "closed_trades": closed,
        "wins": wins,
        "win_rate": (wins / closed * 100) if closed else 0.0,
        "realized_pnl": float(pnl["total"] or 0),
    }
