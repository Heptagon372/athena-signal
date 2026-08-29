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
    ChaseGuard       판 가격보다 크게 오른 자리에서의 재매수 금지 (아테나 자체)

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
    # 손실 청산 뒤에는 더 길게 쉽니다 (0이면 위 공통값만 적용).
    # prism-insight 의 실계좌 진단 이식: 하루 안 재매수 왕복 31건이 평균 −5.6%,
    # −10% 로 판 종목을 0.78일 만에 되사는 패턴이 반복됐습니다. 손실 후 재진입
    # (복수 매매)이 문제였고, **이익 실현 후 재진입은 정당한 추세 지속**이라
    # 두 경우를 같은 잣대로 묶으면 안 됩니다.
    "protect_cooldown_loss_min": 240,

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

    # 5) 추격 재매수 차단 — 아래 chase_reason() 참고.
    #    위 1~4 와 달리 **protect_enabled 와 무관하게** 동작합니다. 이건
    #    "매매를 멈추는" 스위치가 아니라 진입 가격 하나를 보는 규칙이라,
    #    보호장치를 통째로 켜야만 걸리는 것은 과합니다. 0 이면 사용 안 함.
    "protect_chase_pct": 3.0,               # 마지막 매도가 대비 이만큼 위면 금지(%)
    "protect_chase_lookback_min": 1440,     # 매도가를 기억하는 시간(분)
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
    # 종목 -> (마지막 청산 시각, 그때의 체결가). 추격 재매수 차단의 기준선입니다.
    # 잠금과 달리 시간만으로는 판정할 수 없어서(가격을 봐야 합니다) 여기에
    # 재료만 담아 두고, 실제 판정은 현재가를 아는 진입 경로에서 합니다.
    exits: dict = field(default_factory=dict)
    chase_pct: float = 0.0

    def global_reasons(self, now: datetime = None) -> list[str]:
        return [lock.reason for lock in self.locks
                if lock.scope == GLOBAL and lock.active(now)]

    def for_symbol(self, symbol: str, now: datetime = None) -> Lock | None:
        for lock in self.locks:
            if lock.scope == SYMBOL and lock.symbol == str(symbol) and lock.active(now):
                return lock
        return None

    def chase_reason(self, symbol: str, price: float) -> str | None:
        """"판 가격보다 너무 올라서 지금은 못 산다"면 그 사유, 아니면 None.

        왜 필요한가
            시간 쿨다운은 **얼마나 지났는지**만 봅니다. 그래서 30분(또는 4시간)만
            지나면, 그 사이에 종목이 얼마를 올랐든 같은 신호로 다시 들어갑니다.
            그런데 매도 뒤 급등한 종목은 추세추종 지표가 전부 켜지는 자리라
            신호는 오히려 더 좋아 보입니다 — 규칙이 **가장 비싼 자리를 가장
            자신 있게** 고르는 구조입니다.

            여기서 막는 것은 신호가 틀려서가 아니라 **기준가가 있어서**입니다.
            같은 종목을 방금 판 가격보다 비싸게 되사려면, 판 판단이 틀렸다는
            뜻입니다. 그 재평가는 하루 지나 기준이 사라진 뒤에 하는 것이 맞고,
            5분 뒤 호가창을 보고 할 일이 아닙니다.

        되돌아오면 다시 살 수 있습니다 — 회전마다 **현재가로** 다시 재므로,
        가격이 문턱 아래로 내려오면 그 순간 풀립니다. 잠금이 아니라 문턱입니다.
        """
        if self.chase_pct <= 0:
            return None
        record = self.exits.get(str(symbol))
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None
        if not record or price <= 0:
            return None
        when, sold = record
        if not sold or sold <= 0:
            return None
        rise = (price - sold) / sold * 100
        if rise < self.chase_pct:
            return None
        # 경과 시간은 이 스냅샷을 만든 시각 기준입니다 (실시간 now 가 아니라).
        # 백테스트·재현 로그에서 "지금"이 다른데도 같은 문장이 나와야 합니다.
        try:
            ref = datetime.fromisoformat(self.checked_at)
        except (TypeError, ValueError):
            ref = datetime.now()
        ago = (ref - when).total_seconds() / 60
        return (f"추격 재매수 차단 — {ago / 60:.1f}시간 전 {sold:,.2f} 에 팔고 "
                f"{rise:+.1f}% 오른 {price:,.2f} (문턱 +{self.chase_pct:g}%)")

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "checked_at": self.checked_at,
                "locks": [lock.to_dict() for lock in self.locks],
                "chase_pct": self.chase_pct,
                "chase_marks": {sym: {"at": when.isoformat(), "price": price}
                                for sym, (when, price) in self.exits.items()},
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

    # 추격 차단은 보호장치 스위치와 별개로 삽니다 (DEFAULTS 5번 주석 참고).
    chase_pct = float(params.get("protect_chase_pct") or 0)
    chase_look = int(params.get("protect_chase_lookback_min") or 0)
    chase_on = chase_pct > 0 and chase_look > 0
    if not result.enabled and not chase_on:
        return result

    lookbacks = [int(params["protect_cooldown_min"]),
                 int(params["protect_stoploss_lookback_min"]),
                 int(params["protect_drawdown_lookback_min"]),
                 int(params["protect_lowprofit_lookback_min"])]         if result.enabled else []
    if chase_on:
        lookbacks.append(chase_look)
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

    if chase_on:
        result.chase_pct = chase_pct
        result.exits = _last_exits(trades, now, chase_look)
    if not result.enabled:
        return result

    for lock in (_cooldown(trades, params, now)
                 + _stoploss_guard(trades, params, now)
                 + _max_drawdown(trades, params, now, equity)
                 + _low_profit(trades, params, now, equity)):
        if lock.active(now):
            result.locks.append(lock)
    return result


def _last_exits(trades: list, now: datetime, lookback_min: int) -> dict:
    """종목별 **마지막** 청산의 (시각, 체결가).

    마지막 것만 봅니다. 기준선은 "가장 최근에 내가 판 가격"이고, 그보다 이전
    가격은 이미 내가 한 번 갱신한 판단이라 문턱이 될 수 없습니다.

    가격은 `avg_fill_price`(실제 체결가) 우선, 없으면 주문가입니다 — 저장소의
    다른 집계(storage/autotrade.closed_trades 계열)와 같은 순서입니다. 둘 다
    없는 매매는 기준선이 될 수 없으니 건너뜁니다.
    """
    cutoff = now - timedelta(minutes=lookback_min)
    out: dict = {}
    for trade in trades:
        when = _closed_at(trade)
        symbol = str(trade.get("symbol") or "")
        if not symbol or when < cutoff:
            continue
        try:
            price = float(trade.get("avg_fill_price") or trade.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        if symbol not in out or when > out[symbol][0]:
            out[symbol] = (when, price)
    return out


# ---------------------------------------------------------------------------
# 1) 쿨다운 — freqtrade CooldownPeriod
# ---------------------------------------------------------------------------

def _cooldown(trades: list, params: dict, now: datetime) -> list[Lock]:
    """청산 직후 같은 종목에 바로 다시 들어가는 것을 막습니다.

    막는 이유는 신호가 틀려서가 아닙니다. 방금 나온 자리는 **직전 판단이
    이미 반영된 가격**이라, 같은 지표가 같은 신호를 다시 낼 수밖에 없습니다.
    그래서 손절 → 재진입 → 손절이 몇 분 간격으로 반복됩니다.

    손실 청산은 더 길게 쉽니다 (prism-insight reentry_cooldown 이식).
    원본의 실계좌 진단: 손실로 판 종목을 하루 안에 되사는 '복수 매매'가
    체계적 손실원이었습니다 (왕복 31건 평균 −5.6%). 반대로 이익 실현 뒤
    재진입은 추세 지속의 정당한 형태라 공통 쿨다운만 적용합니다.
    """
    minutes = int(params["protect_cooldown_min"])
    loss_minutes = int(params.get("protect_cooldown_loss_min") or 0)
    if minutes <= 0 and loss_minutes <= 0:
        return []

    latest: dict[str, datetime] = {}         # 마지막 청산 (손익 무관)
    latest_loss: dict[str, datetime] = {}    # 마지막 손실 청산
    for trade in trades:
        symbol = str(trade.get("symbol") or "")
        if not symbol:
            continue
        when = _closed_at(trade)
        if symbol not in latest or when > latest[symbol]:
            latest[symbol] = when
        if float(trade.get("realized_pnl") or 0) < 0 and \
                (symbol not in latest_loss or when > latest_loss[symbol]):
            latest_loss[symbol] = when

    out = []
    for symbol, when in latest.items():
        until = when + timedelta(minutes=minutes) if minutes > 0 else None
        loss_at = latest_loss.get(symbol)
        loss_until = (loss_at + timedelta(minutes=loss_minutes)
                      if loss_at and loss_minutes > 0 else None)

        if loss_until and (until is None or loss_until > until):
            if loss_until > now:
                out.append(Lock(
                    protection="cooldown", scope=SYMBOL, symbol=symbol,
                    until=loss_until,
                    reason=f"손실 청산 후 쿨다운(복수 매매 방지) — "
                           f"{(loss_until - now).total_seconds() / 60:.0f}분 남음 "
                           f"(설정 {loss_minutes}분)"))
        elif until and until > now:
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
        {"key": "protect_cooldown_loss_min", "label": "손실 청산 후 재진입 금지",
         "value": (f"{p['protect_cooldown_loss_min']}분 (복수 매매 방지)"
                   if p["protect_cooldown_loss_min"] else "공통값만 적용")},
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
        {"key": "protect_chase_pct", "label": "추격 재매수 차단",
         "value": (f"마지막 매도가 +{p['protect_chase_pct']:g}% 위에서는 재매수 금지 "
                   f"({p['protect_chase_lookback_min']}분간, 보호장치 스위치와 무관)"
                   if p["protect_chase_pct"] and p["protect_chase_lookback_min"]
                   else "꺼짐")},
    ]


__all__ = ["evaluate", "LockSet", "Lock", "DEFAULTS", "config_from", "enabled",
           "describe", "GLOBAL", "SYMBOL"]
