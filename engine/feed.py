"""
통합 시세 피드 (Market Data Feed)
--------------------------------
자동매매 엔진이 "주식이면 토스, 파생이면 KIS" 를 신경 쓰지 않도록 시세 조회를
한 곳으로 모읍니다. Instrument 를 주면 알아서 맞는 소스로 갑니다.

    주식/ETF (국내)  price_provider(토스 → KIS → 네이버 → Yahoo)
    주식/ETF (미국)  Yahoo
    선물/옵션        KIS 파생 시세 (다른 무료 경로가 없습니다)

캐시가 필요한 이유
    자동매매 루프는 종목마다 매 회전 시세를 봅니다. 20종목 × 30초 주기면
    하루 2만 회가 넘습니다. 캐시가 없으면 데이터 제공처에서 차단당합니다.

**신선도(staleness)** 를 함께 돌려주는 이유
    가격이 오래된 줄 모르고 주문을 내는 것이 자동매매에서 가장 위험합니다.
    (장 마감 후 마지막 가격으로 계속 매수 신호가 나오는 사고)
    그래서 quote() 는 값과 함께 '몇 초 전 값인지'를 반환하고, 리스크 게이트가
    이를 근거로 주문을 막습니다.
"""

import threading
import time

import pandas as pd

from data_sources import fx, kis_trading, market_clock
from engine import instruments, preprocess
from engine.instruments import Instrument

QUOTE_TTL = 20.0            # 현재가 캐시 (초)
BARS_TTL = 300.0            # 일봉 캐시 (초) — 하루 단위 데이터라 길게 잡아도 안전
INTRADAY_TTL = 120.0        # 분봉 캐시 (초)
PREPROCESS = True           # 봉 전처리 (engine/preprocess.py) 기본 적용

_quote_cache: dict[str, tuple[float, dict | None]] = {}
_bars_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_lock = threading.Lock()


def clear_cache():
    with _lock:
        _quote_cache.clear()
        _bars_cache.clear()


# ---------------------------------------------------------------------------
# 현재가
# ---------------------------------------------------------------------------

def quote(inst: Instrument, max_age: float = QUOTE_TTL) -> dict | None:
    """{price, price_krw, prev_close, change_rate, age_sec, source, market_open}

    실패하면 None (조용히 0 을 돌려주면 상위에서 0원에 매수하는 사고가 납니다).
    """
    key = f"q:{inst.key}"
    now = time.time()
    with _lock:
        hit = _quote_cache.get(key)
    if hit and now - hit[0] < max_age:
        cached = hit[1]
        if cached is None:
            return None
        return {**cached, "age_sec": round(now - hit[0], 1), "cached": True}

    data = _fetch_quote(inst)
    with _lock:
        _quote_cache[key] = (now, data)
    if data is None:
        return None
    return {**data, "age_sec": 0.0, "cached": False}


def _fetch_quote(inst: Instrument) -> dict | None:
    if inst.is_derivative:
        q = kis_trading.deriv_quote(inst.key)
        if not q:
            return None
        status = market_clock.status_for("KOSPI")
        return {
            "price": q["price"],
            "price_krw": q["price"],            # 파생은 원화 호가
            "prev_close": q.get("prev_close"),
            "change_rate": q.get("change_rate"),
            "volume": q.get("volume"),
            "open_interest": q.get("open_interest"),
            "source": q.get("source", "KIS"),
            "market_open": bool(status.get("is_open")),
            "session": status.get("label", ""),
        }

    symbol = inst.symbol
    if symbol is None:
        return None

    from data_sources.price_provider import get_provider
    try:
        live = get_provider(symbol).get_realtime_quote(symbol)
    except Exception:
        return None

    price = live.get("price")
    if not price or price <= 0:
        return None

    status = live.get("market_status")
    if not isinstance(status, dict):
        status = market_clock.status_for(inst.market)

    out = {
        "price": float(price),
        "price_krw": fx.to_krw(float(price), inst.currency),
        "prev_close": live.get("prev_close"),
        "change_rate": live.get("change_rate"),
        "volume": live.get("volume"),
        "exchange": live.get("exchange", ""),
        "source": live.get("source", ""),
        "market_open": bool(status.get("is_open")),
        "session": status.get("label", ""),
        # 이 가격이 **어느 세션의 가격인가**. 주간거래 구간에서 정규장 종가로
        # 주문하는 사고를 리스크 게이트가 잡아낼 수 있게 하는 표식입니다.
        "price_session": "",
    }
    if inst.market == "US" and status.get("session") == "DAY":
        out = _overlay_day_price(inst, out)
    return out


def _overlay_day_price(inst: Instrument, base: dict) -> dict:
    """주간거래 구간이면 가격을 KIS 주간 시세로 갈아끼웁니다.

    Yahoo 의 확장시간 데이터는 ET 04:00~20:00 까지입니다. 주간거래(ET 20:00~03:30)
    는 그 바깥이라, 이 구간에 Yahoo 를 믿으면 **몇 시간 전 정규장 종가**를
    현재가로 씁니다. 신선도 게이트는 '캐시에 담긴 지 몇 초'를 재기 때문에 이걸
    걸러내지 못합니다 — 방금 받아온 낡은 값은 age_sec 이 0 입니다.

    거래소 코드(NASD/NYSE/AMEX)는 Yahoo 응답에서 옵니다. 그래서 Yahoo 를
    건너뛰지 않고 받아온 뒤 가격만 덮어씁니다 (전일종가·거래소도 함께 씁니다).

    실패하면 base 를 그대로 돌려주되 price_session 은 비워 둡니다. 부르는 쪽이
    "주간 시세를 못 받았다"를 알아야 낡은 가격으로 주문하지 않습니다.
    """
    if not kis_trading.is_configured():
        return base
    exchange = kis_trading.overseas_exchange_code(base.get("exchange"))
    try:
        day = kis_trading.overseas_day_quote(inst.key, exchange)
    except Exception:
        return base
    if not day or not day.get("price"):
        return base

    price = float(day["price"])
    return {
        **base,
        "price": price,
        "price_krw": fx.to_krw(price, inst.currency),
        # 전일종가는 주간 시세 것이 있으면 그것을, 없으면 Yahoo 것을 씁니다.
        "prev_close": day.get("prev_close") or base.get("prev_close"),
        "change_rate": day.get("change_rate", base.get("change_rate")),
        "volume": day.get("volume") or base.get("volume"),
        "source": day.get("source", "한국투자증권 KIS (주간거래)"),
        "price_session": "DAY",
        "day_excd": day.get("excd", ""),
    }


def price_krw(inst: Instrument) -> float | None:
    """원화 환산 현재가만 필요할 때."""
    q = quote(inst)
    return q["price_krw"] if q else None


# ---------------------------------------------------------------------------
# 봉 데이터
# ---------------------------------------------------------------------------

def bars(inst: Instrument, timeframe: str = "day", count: int = 120,
         clean: bool = None) -> pd.DataFrame:
    """기술적 지표 계산용 OHLCV. 실패하면 빈 DataFrame.

    받아온 봉은 engine/preprocess.py 를 한 번 거칩니다 — 중복·정합성 위반·
    액면분할처럼 **지표를 조용히 망가뜨리는 결함**을 여기서 걷어내야, 이 함수를
    쓰는 모든 곳(신호·추천·백테스트·차트)이 같은 데이터를 보게 됩니다.
    정제 결과 보고서는 `df.attrs["quality"]` 에 붙습니다.

    캐시에는 **정제된 결과**를 넣습니다. 원본을 캐시하면 같은 봉을 회전마다
    다시 정제하게 되고, 20종목 × 30초 주기에서는 그 비용이 무시할 수 없습니다.
    """
    if clean is None:
        clean = PREPROCESS
    key = f"b:{inst.key}:{timeframe}:{count}:{int(bool(clean))}"
    ttl = BARS_TTL if timeframe == "day" else INTRADAY_TTL
    now = time.time()
    with _lock:
        hit = _bars_cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]

    df = _fetch_bars(inst, timeframe, count)
    if clean and df is not None and not df.empty:
        try:
            df = preprocess.clean_bars(df, timeframe)
        except Exception:
            # 전처리 실패가 시세 조회 실패로 번지면 안 됩니다 — 원본을 씁니다
            pass
    with _lock:
        _bars_cache[key] = (now, df)
    return df


def _fetch_bars(inst: Instrument, timeframe: str, count: int) -> pd.DataFrame:
    try:
        if inst.is_derivative:
            if timeframe != "day":
                # 파생 분봉은 별도 TR 이 필요합니다. 지금은 일봉만 지원하고,
                # 상위(전략)는 분봉이 없으면 일봉만으로 판단합니다.
                return pd.DataFrame()
            return kis_trading.deriv_daily_chart(inst.key, days=count)

        symbol = inst.symbol
        if symbol is None:
            return pd.DataFrame()
        from data_sources.price_provider import get_provider
        provider = get_provider(symbol)
        if timeframe == "day":
            return provider.get_daily_history(symbol, days=count)
        return provider.get_history(symbol, timeframe, count=count)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 장 상태
# ---------------------------------------------------------------------------

def market_status(inst: Instrument) -> dict:
    """이 상품이 지금 거래 가능한 시간인지.

    파생상품은 정규장이 주식보다 15분 늦게(15:45) 끝나지만, 자동매매는 보수적으로
    주식 정규장 시간표를 따릅니다 — 유동성이 얇은 구간에서 자동으로 던지는 것을
    막기 위해서입니다.
    """
    market = "KOSPI" if inst.is_derivative else inst.market
    status = market_clock.status_for(market)
    return {
        "is_open": bool(status.get("is_open")),
        "is_regular": bool(status.get("is_regular")),
        "session": status.get("session", ""),
        "label": status.get("label", ""),
        "next_event": status.get("next_event", ""),
    }


def is_tradable_now(inst: Instrument, regular_only: bool = True) -> tuple[bool, str]:
    """(거래 가능 여부, 사유). 청산 경로가 씁니다 — 진입은 entry_allowed_now."""
    status = market_status(inst)
    if regular_only and not status["is_regular"]:
        return False, f"정규장이 아닙니다 ({status['label']})"
    if not status["is_open"]:
        return False, f"장이 열려 있지 않습니다 ({status['label']})"
    return True, status["label"]


# 신규 진입을 허용하는 세션 — 기본은 **프리마켓과 정규장뿐입니다.**
#
# 애프터마켓(미국)과 시간외 단일가(한국)를 뺀 이유: 호가가 얇아 스프레드가
# 벌어지고, 한국 시간외는 10분 단위 단일가라 낸 가격에 체결된다는 보장이
# 없습니다. 그 구간에 새로 들어가면 시작부터 비용을 지고 갑니다.
# 예외: 미국 애프터마켓은 us_extended_hours 를 **켠 경우에만** 신규 진입을
# 허용합니다 — 설정 라벨("미국 프리·애프터마켓 허용")이 약속하는 동작이고,
# 스프레드 비용을 감수하겠다는 명시적 선택이므로 기본값은 꺼짐입니다.
# 예외 2: 미국 주간거래(DAY, 한국 낮)는 us_day_session 을 켠 경우에만. ATS 라
# 호가가 더 얇아 스위치를 따로 뒀습니다 (_day_entry_allowed 주석 참고).
# 반대로 **청산은 이 제한을 받지 않습니다** — 못 들어가는 것은 기회 손실이지만
# 못 나오는 것은 손실입니다 (check_exit 는 그대로 is_tradable_now 를 씁니다).
ENTRY_SESSIONS = ("REGULAR", "PRE", "PRE_AUCTION")
_PRE_SESSIONS = ("PRE", "PRE_AUCTION")


def _after_entry_allowed(inst: Instrument, cfg: dict, session: str) -> bool:
    """미국 애프터마켓 신규 진입 — 확장 시간 스위치를 켠 경우에만."""
    return (session == "AFTER" and inst.market == "US"
            and bool((cfg or {}).get("us_extended_hours")))


def _day_entry_allowed(inst: Instrument, cfg: dict, session: str) -> bool:
    """미국 주간거래(데이마켓) 신규 진입 — us_day_session 을 켠 경우에만.

    확장시간(프리·애프터)과 **스위치를 분리한 이유**: 프리·애프터는 나스닥·NYSE
    정규 거래소의 연장이고, 주간거래는 FINRA 승인 ATS 라 체결 경로 자체가
    다릅니다. 호가가 훨씬 얇고 미국 현지 참여가 거의 없어 변동성이 큽니다.
    한 스위치로 묶으면 "애프터마켓만 하려던 사람"이 ATS 까지 열게 됩니다.
    """
    return (session == "DAY" and inst.market == "US"
            and bool((cfg or {}).get("us_day_session")))


def entry_allowed_now(inst: Instrument, cfg: dict) -> tuple[bool, str, str]:
    """지금 이 종목에 **신규 진입**을 시도해도 되는가. (가능여부, 사유, 세션키)

    자동매매 루프가 신호를 계산하기 **전에** 이걸 먼저 봅니다. 예전에는 지표를
    다 계산하고 주문 직전 리스크 게이트에서야 "정규장이 아닙니다"로 거부했습니다.
    한국 종목이 유니버스에 있으면 미국장 시간(새벽) 내내 회전마다 지표를
    계산하고 버렸고, 거부 로그만 쌓였습니다.
    """
    status = market_status(inst)
    session = status.get("session", "")
    label = status.get("label", "")

    if (session not in ENTRY_SESSIONS
            and not _after_entry_allowed(inst, cfg, session)
            and not _day_entry_allowed(inst, cfg, session)):
        return False, f"신규 진입 시간이 아닙니다 ({label})", session

    if session in _PRE_SESSIONS and not _pre_market_allowed(inst, cfg):
        return False, f"프리마켓 신규 진입이 꺼져 있습니다 ({label})", session

    if not status["is_open"]:
        return False, f"장이 열려 있지 않습니다 ({label})", session
    return True, label, session


def _pre_market_allowed(inst: Instrument, cfg: dict) -> bool:
    """프리마켓 진입을 허용하는 스위치.

    한국 시간외 단일가와 미국 확장 시간은 성격이 달라 스위치를 분리해 뒀습니다
    (engine/risk.py 의 _regular_only 와 같은 이유).
    """
    cfg = cfg or {}
    if inst.market == "US":
        return bool(cfg.get("us_extended_hours"))
    return not bool(cfg.get("regular_session_only", True))


# ---------------------------------------------------------------------------
# 만기 관리 (파생 전용)
# ---------------------------------------------------------------------------

def days_to_expiry(inst: Instrument) -> int | None:
    if not inst.expiry:
        return None
    from datetime import date
    return (inst.expiry - date.today()).days


def front_month_futures(underlying: str = "01") -> Instrument | None:
    """가장 가까운 만기의 지수선물 Instrument.

    선물은 만기가 지나면 코드 자체가 죽습니다. 유니버스에 '101H6000' 을
    하드코딩해 두면 3월 둘째 목요일 이후 전부 조회 실패가 됩니다.
    그래서 코드를 고정하지 않고 이 함수로 매번 최근월물을 계산합니다.
    """
    from datetime import date

    today = date.today()
    year, month = today.year, today.month
    for _ in range(8):
        # 지수선물·옵션 결제월은 3·6·9·12월
        candidate_month = min((m for m in (3, 6, 9, 12) if m >= month), default=None)
        if candidate_month is None:
            month, year = 3, year + 1
            candidate_month = 3
        code_char = next(k for k, v in instruments.MONTH_CODES.items()
                         if v == candidate_month)
        code = f"1{underlying}{code_char}{year % 10}000"
        inst = instruments.parse_derivative(code)
        if inst and inst.expiry and (inst.expiry - today).days >= 1:
            return inst
        # 이번 만기가 지났으면 다음 분기로
        month = candidate_month + 1
        if month > 12:
            month, year = 1, year + 1
    return None


__all__ = ["quote", "price_krw", "bars", "market_status", "is_tradable_now",
           "entry_allowed_now", "ENTRY_SESSIONS",
           "days_to_expiry", "front_month_futures", "clear_cache", "QUOTE_TTL"]
