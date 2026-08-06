"""
보호장치 (Protections)
---------------------
freqtrade 의 plugins/protections 를 아테나의 주문 원장 위로 옮긴 것입니다.
리스크 게이트(engine/risk.py)가 **한 건의 주문**을 보는 반면, 여기는 **최근 매매
이력**을 봅니다. 질문 자체가 다릅니다.

    risk.py        "이 주문이 한도 안인가?"
    protections.py "지금 이 전략(또는 이 종목)이 계속 매매해도 되는 상태인가?"

왜 이게 따로 필요한가
    같은 자리에서 반복해 손절당하는 국면이 있습니다. 지표는 계속 매수를
    가리키는데 시장이 그 지표를 배신하는 구간입니다. 한 건씩 보는 리스크
    게이트는 매번 "한도 안"이라고 통과시키므로, 계좌는 하루 종일 같은 손절을
    반복하며 수수료와 함께 녹습니다. 사람이라면 "오늘 이 종목은 안 되네" 하고
    손을 뗍니다. 그 판단을 규칙으로 옮긴 것이 보호장치입니다.

수록 항목 (freqtrade 원본 이름)
    CooldownPeriod   청산 직후 같은 종목 재진입 금지 — 되사기(revenge buy) 차단
    StoplossGuard    정해진 시간 안에 손절이 N회 → 잠금
    MaxDrawdown      정해진 시간 안의 실현손익 낙폭이 한도 초과 → 전체 잠금
    LowProfitPairs   그 종목의 최근 성적이 기준 미달 → 그 종목만 잠금

**청산은 어떤 잠금으로도 막지 않습니다.** 못 들어가는 것은 기회 손실이지만
못 나오는 것은 손실입니다. 이 모듈이 만드는 잠금은 전부 '신규 진입' 대상입니다.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

GLOBAL, SYMBOL = "global", "symbol"

DEFAULTS = {
    "protect_enabled": False,          # 기본 꺼짐 — 매매를 멈추는 스위치입니다

    # 1) 쿨다운 — 청산 후 이 시간 동안 같은 종목 재진입 금지 (0이면 사용 안 함)
    "protect_cooldown_min": 30,

    # 2) 손절 감시 — lookback 안에 손절이 count 회 이상이면 stop 분간 잠금
    "protect_stoploss_count": 3,
    "protect_stoploss_lookback_min": 240,
    "protect_stoploss_stop_min": 60,
    "protect_stoploss_per_symbol": False,   # True 면 그 종목만, False 면 전체

    # 3) 낙폭 감시 — lookback 안 실현손익 곡선의 낙폭이 자산 대비 한도 초과
    "protect_drawdown_pct": 5.0,
    "protect_drawdown_lookback_min": 1440,
    "protect_drawdown_min_trades": 5,
    "protect_drawdown_stop_min": 240,

    # 4) 부진 종목 — lookback 안 그 종목 누적 손익이 기준 미만이면 그 종목만 잠금
    "protect_lowprofit_pct": 0.0,           # 자산 대비 %. 0 이면 '적자면 잠금'
    "protect_lowprofit_lookback_min": 1440,
    "protect_lowprofit_min_trades": 3,
    "protect_lowprofit_stop_min": 120,
}

# 청산 사유 문자열에서 '손절로 끝난 매매'를 알아보는 표지.
# engine/strategy.check_exit 가 만드는 문장을 그대로 받습니다.
# 트레일링 되돌림도 freqtrade 와 같이 손절로 셉니다 — 이익을 지키려다 나온
# 것이든 손실을 끊으려 나온 것이든, 가격이 나를 밀어낸 것은 같습니다.
_STOP_MARKERS = ("손절", "손실 한도", "되돌림")


@dataclass
class Lock:
    """신규 진입 잠금 한 건."""
    protection: str                # cooldown | stoploss_guard | max_drawdown | low_profit
    scope: str = GLOBAL            # global | symbol
    symbol: str = ""
    until: datetime = None
    reason: str = ""

    def active(self, now: datetime = None) -> bool:
        return self.until is None or (now or datetime.now()) < self.until

    def left_min(self, now: datetime = None) -> float:
        if self.until is None:
            return 0.0
        return max(0.0, (self.until - (now or datetime.now())).total_seconds() / 60)

    def to_dict(self) -> dict:
        return {"protection": self.protection, "scope": self.scope,
                "symbol": self.symbol,
                "until": self.until.isoformat() if self.until else None,
                "left_min": round(self.left_min(), 1), "reason": self.reason}


@dataclass
class LockSet:
    """한 회전 동안 고정해 두고 쓰는 잠금 스냅샷.

    engine/risk.py 의 RiskEngine 과 같은 이유로 회전 시작 시점에 한 번만
    계산합니다. 회전 중간에 값이 바뀌면 "앞 종목은 통과, 뒤 종목은 거부" 같은
    비결정적 동작이 나오고, 나중에 로그만 보고는 재현할 수 없게 됩니다.
    """
    enabled: bool = False
    locks: list = field(default_factory=list)
    checked_at: str = ""
    error: str = ""

    def global_reasons(self, now: datetime = None) -> list[str]:
        return [lock.reason for lock in self.locks
                if lock.scope == GLOBAL and lock.active(now)]

    def for_symbol(self, symbol: str, now: datetime = None) -> Lock | None:
        for lock in self.locks:
            if lock.scope == SYMBOL and lock.symbol == str(symbol) and lock.active(now):
                return lock
        return None

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "checked_at": self.checked_at,
                "locks": [lock.to_dict() for lock in self.locks],
                "error": self.error}


def config_from(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        value = (cfg or {}).get(key)
        if value is not None:
            out[key] = value
    return out


def enabled(cfg: dict) -> bool:
    return bool((cfg or {}).get("protect_enabled"))


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate(user_id: int, mode: str, cfg: dict, equity: float = 0.0,
             now: datetime = None, trades: list = None) -> LockSet:
    """지금 걸려 있어야 할 잠금들을 계산합니다.

    trades 를 직접 넘기면 DB 를 읽지 않습니다 (테스트·백테스트용).
    """
    now = now or datetime.now()
    params = config_from(cfg)
    result = LockSet(enabled=enabled(cfg), checked_at=now.isoformat())
    if not result.enabled:
        return result

    lookbacks = [int(params["protect_cooldown_min"]),
                 int(params["protect_stoploss_lookback_min"]),
                 int(params["protect_drawdown_lookback_min"]),
                 int(params["protect_lowprofit_lookback_min"])]
    horizon = max([v for v in lookbacks if v > 0] or [1440])

    if trades is None:
        try:
            from storage import autotrade as store
            since = (now - timedelta(minutes=horizon)).isoformat()
            trades = store.closed_trades(user_id, mode, since=since)
        except Exception as exc:
            # 이력을 못 읽으면 잠그지 않습니다. 여기서 fail-closed 로 가면
            # DB 오류 한 번에 자동매매가 통째로 멈춥니다 — 보호장치는 어디까지나
            # '최근 성적이 나쁠 때'를 위한 것이지 가용성 통제가 아닙니다.
            result.error = f"매매 이력을 읽지 못했습니다: {type(exc).__name__}"
            return result

    trades = [t for t in (trades or []) if _closed_at(t) is not None]
    trades.sort(key=_closed_at)

    for lock in (_cooldown(trades, params, now)
                 + _stoploss_guard(trades, params, now)
                 + _max_drawdown(trades, params, now, equity)
                 + _low_profit(trades, params, now, equity)):
        if lock.active(now):
            result.locks.append(lock)
    return result


# ---------------------------------------------------------------------------
# 1) 쿨다운 — freqtrade CooldownPeriod
# ---------------------------------------------------------------------------

def _cooldown(trades: list, params: dict, now: datetime) -> list[Lock]:
    """청산 직후 같은 종목에 바로 다시 들어가는 것을 막습니다.

    막는 이유는 신호가 틀려서가 아닙니다. 방금 나온 자리는 **직전 판단이
    이미 반영된 가격**이라, 같은 지표가 같은 신호를 다시 낼 수밖에 없습니다.
    그래서 손절 → 재진입 → 손절이 몇 분 간격으로 반복됩니다.
    """
    minutes = int(params["protect_cooldown_min"])
    if minutes <= 0:
        return []

    latest: dict[str, datetime] = {}
    for trade in trades:
        symbol = str(trade.get("symbol") or "")
        when = _closed_at(trade)
        if symbol and (symbol not in latest or when > latest[symbol]):
            latest[symbol] = when

    out = []
    for symbol, when in latest.items():
        until = when + timedelta(minutes=minutes)
        if until > now:
            out.append(Lock(
                protection="cooldown", scope=SYMBOL, symbol=symbol, until=until,
                reason=f"청산 직후 쿨다운 — {(until - now).total_seconds() / 60:.0f}분 남음 "
                       f"(설정 {minutes}분)"))
    return out


# ---------------------------------------------------------------------------
# 2) 손절 감시 — freqtrade StoplossGuard
# ---------------------------------------------------------------------------

def _stoploss_guard(trades: list, params: dict, now: datetime) -> list[Lock]:
    """짧은 시간에 손절이 몰리면 그 구간 자체를 피합니다.

    손절이 세 번 연속으로 나왔다는 것은 "내 손절 폭보다 시장의 진폭이 크다"는
    뜻입니다. 이때 필요한 것은 다음 종목을 고르는 것이 아니라 잠시 멈추는
    것입니다. 폭을 넓히는 선택지도 있지만, 그건 사람이 근거를 보고 정할 일이지
    엔진이 실시간으로 할 일이 아닙니다.
    """
    limit = int(params["protect_stoploss_count"])
    lookback = int(params["protect_stoploss_lookback_min"])
    stop_min = int(params["protect_stoploss_stop_min"])
    if limit <= 0 or lookback <= 0 or stop_min <= 0:
        return []

    cutoff = now - timedelta(minutes=lookback)
    stops = [t for t in trades if _closed_at(t) >= cutoff and _is_stop_exit(t)]
    if not stops:
        return []

    if params.get("protect_stoploss_per_symbol"):
        out = []
        by_symbol: dict[str, list] = {}
        for trade in stops:
            by_symbol.setdefault(str(trade.get("symbol") or ""), []).append(trade)
        for symbol, hits in by_symbol.items():
            if len(hits) < limit:
                continue
            until = _closed_at(hits[-1]) + timedelta(minutes=stop_min)
            out.append(Lock(
                protection="stoploss_guard", scope=SYMBOL, symbol=symbol, until=until,
                reason=f"{lookback}분 안에 손절 {len(hits)}회 (한도 {limit}회) — "
                       f"{symbol} 신규 진입 {stop_min}분 중단"))
        return out

    if len(stops) < limit:
        return []
    until = _closed_at(stops[-1]) + timedelta(minutes=stop_min)
    symbols = sorted({str(t.get("symbol") or "") for t in stops})
    return [Lock(
        protection="stoploss_guard", scope=GLOBAL, until=until,
        reason=f"{lookback}분 안에 손절 {len(stops)}회 (한도 {limit}회, "
               f"{', '.join(symbols[:4])}) — 신규 진입 {stop_min}분 중단")]


# ---------------------------------------------------------------------------
# 3) 낙폭 감시 — freqtrade MaxDrawdown
# ---------------------------------------------------------------------------

def _max_drawdown(trades: list, params: dict, now: datetime,
                  equity: float) -> list[Lock]:
    """실현손익 곡선의 낙폭을 봅니다.

    risk.py 의 max_drawdown_pct 와 무엇이 다른가
        risk.py 는 **계좌 평가금액**의 낙폭을 봅니다. 보유 종목의 평가손익이
        섞여 있어서, 자동매매가 잘하고 있어도 들고 있는 다른 종목이 빠지면
        같이 멈춥니다. 여기는 **닫힌 매매의 손익만** 누적해서 봅니다.
        "내 매매 자체가 망가지고 있는가"에 대한 답이라 훨씬 직접적입니다.
    """
    limit_pct = float(params["protect_drawdown_pct"])
    lookback = int(params["protect_drawdown_lookback_min"])
    stop_min = int(params["protect_drawdown_stop_min"])
    min_trades = int(params["protect_drawdown_min_trades"])
    if limit_pct <= 0 or lookback <= 0 or stop_min <= 0 or equity <= 0:
        return []

    cutoff = now - timedelta(minutes=lookback)
    window = [t for t in trades if _closed_at(t) >= cutoff]
    if len(window) < max(1, min_trades):
        return []

    cumulative = peak = 0.0
    worst, worst_at = 0.0, None
    for trade in window:
        cumulative += float(trade.get("realized_pnl") or 0)
        peak = max(peak, cumulative)
        drop = peak - cumulative
        if drop > worst:
            worst, worst_at = drop, _closed_at(trade)

    drawdown_pct = worst / equity * 100
    if drawdown_pct < limit_pct or worst_at is None:
        return []

    until = worst_at + timedelta(minutes=stop_min)
    return [Lock(
        protection="max_drawdown", scope=GLOBAL, until=until,
        reason=f"{lookback}분 실현손익 낙폭 {drawdown_pct:.2f}% "
               f"({worst:,.0f}원 / 한도 {limit_pct:g}%) — 신규 진입 {stop_min}분 중단")]


# ---------------------------------------------------------------------------
# 4) 부진 종목 — freqtrade LowProfitPairs
# ---------------------------------------------------------------------------

def _low_profit(trades: list, params: dict, now: datetime,
                equity: float) -> list[Lock]:
    """최근 성적이 기준에 못 미치는 종목만 따로 잠급니다.

    전체를 멈추지 않는 것이 핵심입니다. 한 종목이 계속 지고 있다고 해서 다른
    종목까지 멈추면, 잘 되던 쪽의 기회까지 같이 버립니다.
    """
    required_pct = float(params["protect_lowprofit_pct"])
    lookback = int(params["protect_lowprofit_lookback_min"])
    stop_min = int(params["protect_lowprofit_stop_min"])
    min_trades = int(params["protect_lowprofit_min_trades"])
    if lookback <= 0 or stop_min <= 0 or equity <= 0:
        return []

    cutoff = now - timedelta(minutes=lookback)
    by_symbol: dict[str, list] = {}
    for trade in trades:
        if _closed_at(trade) >= cutoff:
            by_symbol.setdefault(str(trade.get("symbol") or ""), []).append(trade)

    out = []
    for symbol, hits in by_symbol.items():
        if not symbol or len(hits) < max(1, min_trades):
            continue
        total = sum(float(t.get("realized_pnl") or 0) for t in hits)
        profit_pct = total / equity * 100
        if profit_pct >= required_pct:
            continue
        until = _closed_at(hits[-1]) + timedelta(minutes=stop_min)
        out.append(Lock(
            protection="low_profit", scope=SYMBOL, symbol=symbol, until=until,
            reason=f"{lookback}분 성적 {profit_pct:+.2f}% ({total:+,.0f}원, "
                   f"{len(hits)}건 / 기준 {required_pct:+g}%) — "
                   f"{symbol} 신규 진입 {stop_min}분 중단"))
    return out


# ---------------------------------------------------------------------------
# 보조
# ---------------------------------------------------------------------------

def _closed_at(trade: dict) -> datetime | None:
    for key in ("closed_at", "settled_at", "created_at", "at"):
        value = trade.get(key)
        if not value:
            continue
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            continue
    return None


def _is_stop_exit(trade: dict) -> bool:
    """이 매매가 손절(또는 트레일링 되돌림)로 끝났는가.

    사유 문구가 비어 있으면 손익 부호로 판정합니다 — 문구 형식이 바뀌어도
    보호장치가 통째로 무력화되지 않게 하기 위한 이중화입니다.
    """
    reason = str(trade.get("reason") or "")
    if reason:
        return any(marker in reason for marker in _STOP_MARKERS)
    return float(trade.get("realized_pnl") or 0) < 0


def describe(cfg: dict = None) -> list[dict]:
    """화면에 그대로 띄울 수 있는 현재 설정 요약."""
    p = config_from(cfg or {})
    return [
        {"key": "protect_enabled", "label": "보호장치",
         "value": "켜짐" if p["protect_enabled"] else "꺼짐"},
        {"key": "protect_cooldown_min", "label": "청산 후 재진입 금지",
         "value": f"{p['protect_cooldown_min']}분" if p["protect_cooldown_min"] else "없음"},
        {"key": "protect_stoploss_count", "label": "손절 감시",
         "value": f"{p['protect_stoploss_lookback_min']}분 안에 "
                  f"{p['protect_stoploss_count']}회 → {p['protect_stoploss_stop_min']}분 중단"},
        {"key": "protect_drawdown_pct", "label": "실현손익 낙폭 감시",
         "value": f"{p['protect_drawdown_lookback_min']}분 낙폭 "
                  f"{p['protect_drawdown_pct']:g}% → {p['protect_drawdown_stop_min']}분 중단"},
        {"key": "protect_lowprofit_pct", "label": "부진 종목 차단",
         "value": f"{p['protect_lowprofit_lookback_min']}분 성적 "
                  f"{p['protect_lowprofit_pct']:+g}% 미만 → "
                  f"{p['protect_lowprofit_stop_min']}분 중단"},
    ]


__all__ = ["evaluate", "LockSet", "Lock", "DEFAULTS", "config_from", "enabled",
           "describe", "GLOBAL", "SYMBOL"]
