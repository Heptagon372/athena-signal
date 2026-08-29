"""
장 운영 시간 · 세션 판별
-----------------------
가격이 안 움직이는 게 "고장"인지 "장 마감"인지 화면에서 구분할 수 있어야 해서
분리한 모듈입니다. 폴링 주기와 예측 목표시각 계산도 이 결과를 씁니다.

세션 구분
    한국(KRX)
        PRE_AUCTION   08:30~09:00  장 시작 동시호가 (08:30~08:40 장전 시간외 종가 병행)
        REGULAR       09:00~15:20  정규장 연속거래
        CLOSE_AUCTION 15:20~15:30  장 마감 동시호가 (종가 결정)
        AFTER_CLOSE   15:40~16:00  장후 시간외 종가 (당일 종가로 체결)
        AFTER         16:00~18:00  시간외 단일가 (10분 단위, ±10%)
    미국(NYSE/NASDAQ, KST 환산)
        PRE          04:00~09:30 ET  프리마켓
        REGULAR      09:30~16:00 ET  정규장
        AFTER        16:00~20:00 ET  애프터마켓
        DAY          20:00~03:30 ET  주간거래(데이마켓) — KST 09:00~16:30 / 표준시 10:00~17:30

공휴일 달력은 포함하지 않았습니다(매년 바뀌고 별도 데이터가 필요). 공휴일에는
"장중"으로 표시될 수 있으므로 실제 체결 시각(traded_at)을 함께 표시해 판단을 돕습니다.
"""

from datetime import date, datetime, time, timedelta, timezone

KST = timezone(timedelta(hours=9))

# 세션 키 -> 사람이 읽는 이름
SESSION_LABELS = {
    "PRE_AUCTION": "장 시작 동시호가",
    "PRE": "개장 전 (프리마켓)",
    "REGULAR": "정규장",
    "CLOSE_AUCTION": "장 마감 동시호가",
    "AFTER_CLOSE": "장후 시간외 종가",
    "AFTER": "애프터마켓 (시간외)",
    "DAY": "주간거래 (데이마켓)",
    "CLOSED": "장 마감",
    "WEEKEND": "주말 휴장",
}

# 한국은 같은 키라도 부르는 이름이 다릅니다 (미국 AFTER=애프터마켓과 구분)
KR_SESSION_LABELS = {
    "AFTER": "시간외 단일가",
}

# 체결이 실제로 일어나는 세션 (가격이 움직이는 구간)
TRADING_SESSIONS = {"PRE_AUCTION", "PRE", "REGULAR", "CLOSE_AUCTION",
                    "AFTER_CLOSE", "AFTER", "DAY"}

KR_SESSIONS = [
    ("PRE_AUCTION",   time(8, 30),  time(9, 0)),
    ("REGULAR",       time(9, 0),   time(15, 20)),
    ("CLOSE_AUCTION", time(15, 20), time(15, 30)),
    ("AFTER_CLOSE",   time(15, 40), time(16, 0)),   # 당일 종가로만 체결
    ("AFTER",         time(16, 0),  time(18, 0)),   # 시간외 단일가 (±10%)
]


def now_kst() -> datetime:
    return datetime.now(KST)


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _next_trading_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def korea_status(now: datetime | None = None) -> dict:
    now = now or now_kst()

    if now.weekday() >= 5:
        return _build("WEEKEND", False, "월요일 09:00 개장", now, "KR")

    cur = _minutes(now.time())
    for key, start, end in KR_SESSIONS:
        if _minutes(start) <= cur < _minutes(end):
            return _build(key, True, f"{end.strftime('%H:%M')} 종료", now, "KR")

    if cur < _minutes(time(8, 30)):
        return _build("CLOSED", False, "08:30 동시호가 시작", now, "KR")
    if _minutes(time(15, 30)) <= cur < _minutes(time(15, 40)):
        return _build("CLOSED", False, "15:40 장후 시간외 종가 시작", now, "KR")
    return _build("CLOSED", False, "다음 거래일 09:00 개장", now, "KR")


# 미국 세션이 **KST 로 무슨 요일에 나타나는가**.
#
# ET 하루는 KST 로 이틀에 걸칩니다. 그래서 "미국장이 열리는 요일"을 한 덩어리로
# 묶으면 틀립니다 — 예전 코드는 weekday()<=4 하나로 판정해서, KST 월요일 새벽
# 03:00(=미국 일요일 14:00)에 "정규장 진행 중", 06:00(=일요일 17:00)에
# "애프터마켓"이라고 답했습니다. 그 시각 미국은 통째로 휴장입니다.
_KST_EVENING = (0, 1, 2, 3, 4)   # KST 월~금 저녁 = ET 월~금 낮   (PRE, REGULAR 전반)
_KST_MORNING = (1, 2, 3, 4, 5)   # KST 화~토 새벽 = ET 월~금 오후 (REGULAR 후반, AFTER)
_KST_DAYTIME = (0, 1, 2, 3, 4)   # KST 월~금 낮  = ET 일~목 밤    (주간거래)


def us_status(now: datetime | None = None) -> dict:
    """미국 시장 상태를 KST 기준으로 판단.

    서머타임(3월 둘째 일요일~11월 첫째 일요일)은 월 기준으로 근사합니다.
    ET->KST 시차: 서머타임 +13시간, 표준시 +14시간.

    주간거래(DAY)
        정규 거래소가 아니라 FINRA 승인 ATS 에서 돕니다. 한국 낮에 미국 주식을
        사고팔 수 있는 구간이고, KIS 는 주문·시세 API 가 정규장과 **따로**입니다
        (data_sources/kis_trading.py 의 place_overseas_day_order 참고).
        ET 20:00~03:30 = KST 09:00~16:30 (서머타임) / 10:00~17:30 (표준시).
    """
    now = now or now_kst()
    dst = 3 <= now.month <= 11
    offset = 13 if dst else 14        # KST = ET + offset

    def kst_min(et_hour: int, et_min: int = 0) -> int:
        return (et_hour * 60 + et_min + offset * 60) % (24 * 60)

    cur = _minutes(now.time())
    weekday = now.weekday()

    # ET 04:00 / 09:30 / 16:00 / 20:00 / 03:30 을 KST 분으로 환산
    pre_start, reg_start = kst_min(4), kst_min(9, 30)
    reg_end, after_end = kst_min(16), kst_min(20)
    day_start, day_end = kst_min(20), kst_min(3, 30)

    def between(start: int, end: int) -> bool:
        # 자정을 넘어가는 구간 처리
        return (start <= cur < end) if start < end else (cur >= start or cur < end)

    # 순서대로 보되, 각 세션은 **자기 요일 집합** 안에서만 열립니다.
    for session, start, end, days, note in (
        ("DAY",     day_start, day_end,   _KST_DAYTIME, "주간거래 종료 대기"),
        ("PRE",     pre_start, reg_start, _KST_EVENING, "정규장 개장 대기"),
        ("REGULAR", reg_start, reg_end,   None,         "정규장 진행 중"),
        ("AFTER",   reg_end,   after_end, _KST_MORNING, "애프터마켓 종료 대기"),
    ):
        if not between(start, end):
            continue
        if days is None:
            # 정규장은 KST 자정을 걸칩니다. 저녁 부분(ET 낮 전반)과 새벽
            # 부분(ET 낮 후반)의 요일 집합이 다릅니다.
            days = _KST_EVENING if cur >= reg_start else _KST_MORNING
        if weekday in days:
            return _build(session, True, note, now, "US")
        break

    if weekday == 5 or weekday == 6:
        return _build("WEEKEND", False, "월요일 주간거래 개장(KST)", now, "US")

    hh, mm = divmod(day_start if cur < day_start else pre_start, 60)
    label = "주간거래" if cur < day_start else "프리마켓"
    return _build("CLOSED", False, f"{hh:02d}:{mm:02d} {label} 개장(KST)", now, "US")


def _build(session: str, is_open: bool, next_event: str,
           now: datetime, market_kind: str) -> dict:
    label = SESSION_LABELS.get(session, session)
    if market_kind == "KR":
        label = KR_SESSION_LABELS.get(session, label)
    return {
        "session": session,
        "state": session,                       # 하위호환 (기존 코드가 state를 봄)
        "label": label,
        "is_open": is_open,
        "is_regular": session == "REGULAR",
        "next_event": next_event,
        "market_kind": market_kind,
        "checked_at": now.isoformat(),
    }


def status_from_calendar(calendar: dict, now: datetime | None = None) -> dict | None:
    """토스 공식 캘린더로 세션을 판정합니다.

    하드코딩된 시간표와 달리 **공휴일·임시휴장까지 반영**됩니다.
    캘린더가 없거나 형식이 다르면 None을 돌려 하드코딩 경로로 넘어갑니다.
    """
    if not calendar:
        return None

    now = now or now_kst()
    kind = calendar.get("market", "KR")

    if calendar.get("is_holiday"):
        return _build("CLOSED", False, "휴장일 (다음 거래일 대기)", now, kind)

    def parse(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    windows = [
        ("PRE", calendar.get("pre")),
        ("REGULAR", calendar.get("regular")),
        ("AFTER", calendar.get("after")),
    ]

    upcoming = []
    for session, w in windows:
        if not w:
            continue
        start, end = parse(w.get("start")), parse(w.get("end"))
        if not start or not end:
            continue
        if start <= now < end:
            return _build(session, True, f"{end:%H:%M} 종료", now, kind)
        if now < start:
            upcoming.append((start, session))

    if upcoming:
        upcoming.sort()
        start, session = upcoming[0]
        label = {"PRE": "프리마켓", "REGULAR": "정규장", "AFTER": "시간외"}.get(session, session)
        return _build("CLOSED", False, f"{start:%H:%M} {label} 개장", now, kind)

    return _build("CLOSED", False, "장 종료 (다음 거래일 대기)", now, kind)


def status_for(market: str) -> dict:
    """장 상태 — 토스 공식 캘린더가 있으면 그걸 쓰고, 없으면 시간표로 판정."""
    from models import MARKET_US
    from data_sources import toss_api

    is_us = market == MARKET_US

    if toss_api.is_configured():
        calendar = toss_api.get_market_calendar("US" if is_us else "KR")
        resolved = status_from_calendar(calendar)
        if resolved:
            # 토스 캘린더에는 pre/regular/after 창만 있고 **주간거래(ATS) 창이
            # 없습니다.** 그대로 두면 캘린더를 쓰는 계정에서는 DAY 세션이 영영
            # 뜨지 않습니다 (캘린더가 KST 낮을 "장 마감"으로 답하므로).
            # 캘린더가 닫혔다고 한 시간대에 한해, 휴장일이 아니면 내장 시간표의
            # 주간거래 판정으로 덮습니다. 휴장일이면 덮지 않습니다 — 세션을
            # 놓치는 것은 기회 손실이지만, 닫힌 장에 주문을 내는 것은 사고입니다.
            if (is_us and not resolved["is_open"]
                    and not (calendar or {}).get("is_holiday")):
                fallback = us_status()
                if fallback["session"] == "DAY":
                    fallback["source"] = "내장 시간표 (주간거래 — 캘린더 미수록)"
                    return fallback
            resolved["source"] = "토스 공식 캘린더"
            return resolved

    status = us_status() if is_us else korea_status()
    status["source"] = "내장 시간표 (공휴일 미반영)"
    return status


def next_regular_open(market: str, now: datetime | None = None) -> datetime:
    """다음 정규장 개장 시각 (예측 목표시각을 장중으로 옮길 때 사용)."""
    from models import MARKET_US
    now = now or now_kst()

    if market == MARKET_US:
        dst = 3 <= now.month <= 11
        offset = 13 if dst else 14
        open_min = (9 * 60 + 30 + offset * 60) % (24 * 60)
        hh, mm = divmod(open_min, 60)
    else:
        hh, mm = 9, 0

    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate
