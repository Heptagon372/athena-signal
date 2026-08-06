"""
KIS 실시간체결 (WebSocket) — 틱 단위 시세
=========================================

왜 이게 필요한가
    REST 폴링으로는 초단타를 할 수 없습니다. engine/feed.py 의 시세 캐시는
    TTL 이 20초이고, KIS REST 는 이 프로그램에서 초당 3회로 조여져 있습니다
    (kis_client.REAL_MIN_INTERVAL). 20초 전 가격으로 3틱 승부를 보는 것은
    의미가 없습니다.

    실시간체결(H0STCNT0)은 **푸시**입니다. 체결이 일어날 때마다 서버가 보내주고,
    REST 초당 한도와 무관합니다. 대신 계정당 구독 종목 수에 한도가 있습니다.

이 스트림이 주는 것 (초단타에 필요한 게 전부 들어 있습니다)
    현재가 · **매도호가1 · 매수호가1** · 체결량 · 누적거래량 · 체결강도
    · 당일 고가/저가 · 전일거래량 대비 비율

    특히 호가(bid/ask)가 틱마다 온다는 게 큽니다. 스프레드가 몇 틱인지를
    실시간으로 알 수 있어야 "지금 들어가면 몇 틱 손해로 시작하는가" 를
    주문 직전에 판단할 수 있습니다.

    당일 고가/저가가 함께 오므로, 추적 중인 종목의 **가격 변동폭**은 REST 를
    한 번도 더 부르지 않고 여기서 얻습니다.

설계상 중요한 것
    · 연결이 끊겨도 엔진이 죽으면 안 됩니다. 실패는 status() 로 보고하고,
      호출부는 REST 폴링으로 계속 돌아갑니다.
    · 구독 변경(종목 교체)은 수신 스레드 안에서 처리합니다. sync WebSocket 을
      여러 스레드에서 동시에 send 하면 프레임이 깨집니다.
    · 체결가(H0STCNT0)는 암호화되지 않습니다. 암호화되는 것은 체결통보
      (H0STCNI0)뿐이라, 여기서는 복호화가 필요 없습니다.
"""

import json
import queue
import threading
import time
from collections import deque

from data_sources import http_client, kis_client

REAL_WS = "ws://ops.koreainvestment.com:21000"
MOCK_WS = "ws://ops.koreainvestment.com:31000"
TICK_PATH = "/tryitout/H0STCNT0"
TR_TICK = "H0STCNT0"         # 국내주식 실시간체결
TR_TICK_US = "HDFSCNT0"      # 해외주식 실시간(지연)체결

# 야후 거래소코드 → KIS 실시간 거래소코드.
# tr_key 는 'D' + 거래소(3) + 심볼 입니다 (예: DNASCISS).
US_EXCHANGE_MAP = {
    "NMS": "NAS", "NCM": "NAS", "NGM": "NAS", "NAS": "NAS", "NASDAQ": "NAS",
    "NYQ": "NYS", "NYS": "NYS", "NYSE": "NYS",
    "ASE": "AMS", "AMX": "AMS", "AMS": "AMS", "PCX": "AMS", "PSE": "AMS",
}
DEFAULT_US_EXCHANGE = "NAS"

# 계정당 실시간 구독 한도. HTS/MTS 에서 쓰는 구독도 같은 한도를 나눠 씁니다.
# 이 값이 "동시에 몇 종목을 추적할 수 있는가" 의 물리적 상한입니다.
MAX_SUBSCRIPTIONS = 41

# H0STCNT0 응답 필드 인덱스 (^ 로 구분된 순서)
F_CODE = 0          # 종목코드
F_TIME = 1          # 체결시각 HHMMSS
F_PRICE = 2         # 현재가
F_CHANGE_RATE = 5   # 전일 대비율
F_OPEN = 7
F_HIGH = 8
F_LOW = 9
F_ASK = 10          # 매도호가1
F_BID = 11          # 매수호가1
F_TICK_VOL = 12     # 체결 거래량
F_ACML_VOL = 13     # 누적 거래량
F_ACML_VAL = 14     # 누적 거래대금
F_STRENGTH = 18     # 체결강도
F_VOL_RATIO = 23    # 전일 거래량 대비 비율(%)

# HDFSCNT0 (해외) 필드 인덱스 — 국내와 **배치가 완전히 다릅니다.**
# 국내 인덱스를 그대로 쓰면 엉뚱한 값을 가격으로 읽어 그 값에 주문이 나갑니다.
U_RSYM = 0          # 실시간종목코드 (DNASCISS)
U_SYMB = 1          # 종목코드 (CISS)
U_KHMS = 7          # 한국시간 HHMMSS  ← 지연 판정의 근거
U_OPEN = 8
U_HIGH = 9
U_LOW = 10
U_PRICE = 11        # 현재가
U_CHANGE_RATE = 14  # 등락율
U_BID = 15          # 매수호가
U_ASK = 16          # 매도호가
U_TICK_VOL = 19     # 체결량
U_ACML_VOL = 20     # 거래량
U_ACML_VAL = 21     # 거래대금
U_STRENGTH = 24     # 체결강도
U_FIELDS = 26       # 이 TR 의 필드 수

RECONNECT_DELAY = 3.0       # 끊겼을 때 재연결 간격(초)
MAX_RECONNECT_DELAY = 30.0
RECV_TIMEOUT = 1.0          # 이 주기로 구독 변경 큐를 확인합니다

# 종목당 보관할 1초봉 개수 (8시간 = 정규장 전체보다 깁니다).
# **틱 원본을 쌓지 않고 곧바로 1초봉으로 접습니다.** 활발한 저가주는 초당
# 수십 체결이 나와서, 원본을 들고 있으면 몇 분 만에 수십만 건이 됩니다.
# 1초봉으로 접으면 종목당 최대 28,800개로 상한이 정해집니다.
BAR_HISTORY = 28_800
# 봉을 들고 있을 종목 수 상한 (대상에서 빠져도 차트는 볼 수 있게 남겨둡니다)
BAR_SYMBOLS = 60


def _sec_of_day(hhmmss, fallback_epoch: float) -> int:
    """체결시각(HHMMSS) → 자정으로부터의 초.

    거래소가 준 시각을 씁니다. 수신 시각을 쓰면 네트워크 지연이 봉 경계를
    흔들어서, 같은 체결이 실행할 때마다 다른 봉에 들어갑니다.
    """
    text = str(hhmmss or "").strip()
    if len(text) == 6 and text.isdigit():
        hour, minute, second = int(text[0:2]), int(text[2:4]), int(text[4:6])
        if hour < 24 and minute < 60 and second < 60:
            return hour * 3600 + minute * 60 + second
    local = time.localtime(fallback_epoch)
    return local.tm_hour * 3600 + local.tm_min * 60 + local.tm_sec


def is_korean_code(code) -> bool:
    text = str(code or "").strip()
    return len(text) == 6 and text.isdigit()


def us_exchange(hint) -> str:
    """야후/스크리너가 준 거래소 표기를 KIS 실시간 거래소코드로."""
    return US_EXCHANGE_MAP.get(str(hint or "").strip().upper(), DEFAULT_US_EXCHANGE)


def tr_for(code) -> str:
    """이 종목을 어떤 TR 로 구독할 것인가."""
    return TR_TICK if is_korean_code(code) else TR_TICK_US


def tr_key_for(code, exchange=None) -> str:
    """구독 키. 국내는 종목코드 그대로, 해외는 D + 거래소(3) + 심볼."""
    text = str(code or "").strip().upper()
    if is_korean_code(text):
        return text
    return "D" + us_exchange(exchange) + text


def _num(v, default=0.0) -> float:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def ws_url() -> str:
    return MOCK_WS if kis_client.is_mock() else REAL_WS


def request_approval_key() -> tuple[str | None, str]:
    """실시간 접속용 approval_key 발급. (키, 오류메시지)

    REST 의 access_token 과 **다른 키**입니다. 그리고 이 엔드포인트만
    본문 필드 이름이 `secretkey` 입니다 (다른 곳은 전부 `appsecret`).
    이걸로 몇 시간씩 헤매기 쉬워서 여기 적어둡니다.
    """
    if not kis_client.is_configured():
        return None, "KIS 키가 설정되지 않았습니다."
    status, body, raw = http_client.post_full(
        kis_client.base_url() + "/oauth2/Approval",
        json_body={
            "grant_type": "client_credentials",
            "appkey": kis_client.app_key(),
            "secretkey": kis_client.app_secret(),
        },
        headers={"content-type": "application/json"},
        timeout=10,
    )
    if status != 200 or not isinstance(body, dict):
        return None, f"approval_key 발급 실패 (HTTP {status}) — {raw[:200]}"
    key = body.get("approval_key")
    if not key:
        return None, f"approval_key 가 응답에 없습니다 — {raw[:200]}"
    return key, ""


class TickStream:
    """실시간 체결 스트림 하나.

    `on_tick(tick: dict)` 콜백은 **수신 스레드에서** 불립니다. 여기서 무거운
    일(주문, DB 쓰기)을 하면 그동안 다른 종목의 틱이 밀립니다. 콜백은 상태만
    갱신하고, 판단은 엔진 루프에서 하세요.
    """

    def __init__(self, on_tick=None):
        self.on_tick = on_tick
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._commands: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

        self._wanted: set[str] = set()      # 구독하고 싶은 종목
        self._active: set[str] = set()      # 실제로 구독된 종목
        self._last: dict[str, dict] = {}    # 종목별 마지막 틱
        self._bars: dict[str, deque] = {}   # 종목별 1초봉 (틱을 접어서 보관)
        self._exchanges: dict[str, str] = {}  # 해외 종목의 거래소 힌트

        self._connected = False
        self._error = ""
        self._connected_at = 0.0
        self._tick_count = 0
        self._last_tick_at = 0.0
        self._dropped: list[str] = []       # 한도 초과로 못 담은 종목
        self._effective_limit = 0           # 서버가 알려준 실제 한도 (0 = 아직 모름)
        self._pending_code = ""             # 방금 등록 요청한 종목 (거부 처리용)

    # -- 수명 주기 ---------------------------------------------------------

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kis-tick",
                                        daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._connected = False

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # -- 구독 --------------------------------------------------------------

    def set_symbols(self, codes: list[str], exchanges: dict = None):
        """추적 종목을 이 목록으로 **교체**합니다. 국내·해외를 함께 받습니다.

        `exchanges` 는 해외 종목의 거래소 힌트({'CISS': 'NCM'})입니다. 없으면
        나스닥으로 가정합니다 — 틀리면 구독은 되지만 데이터가 안 옵니다.

        한도를 넘으면 앞쪽부터 담고 나머지는 버립니다 — 스크리너가
        순위대로 넘겨주므로 앞쪽이 더 좋은 후보입니다. 버린 종목은
        status()['dropped'] 로 보고합니다. 조용히 버리면 "왜 이 종목은
        신호가 안 뜨지" 가 됩니다.
        """
        clean, seen = [], set()
        for code in codes or []:
            code = str(code).strip().upper()
            # 국내는 6자리 숫자, 해외는 영문·숫자 티커
            valid = is_korean_code(code) or (
                1 <= len(code) <= 12 and code.replace(".", "").replace("-", "").isalnum()
                and any(ch.isalpha() for ch in code))
            if valid and code not in seen:
                seen.add(code)
                clean.append(code)

        if exchanges:
            with self._lock:
                for code, hint in exchanges.items():
                    self._exchanges[str(code).strip().upper()] = hint

        with self._lock:
            cap = self._effective_limit or MAX_SUBSCRIPTIONS
            keep, dropped = clean[:cap], clean[cap:]
            self._dropped = dropped
            target = set(keep)
            add = target - self._wanted
            remove = self._wanted - target
            self._wanted = target

            # **해제를 먼저, 등록을 나중에.** 순서를 뒤집으면 종목을 교체하는
            # 순간 (기존 + 신규) 가 동시에 잡혀 한도를 넘고, 서버가
            # "MAX SUBSCRIBE OVER" 로 거부합니다. 자리를 비우고 넣어야 합니다.
            for code in sorted(remove):
                self._commands.put(("2", code))       # tr_type 2 = 해제
                self._last.pop(code, None)
            for code in sorted(add):
                self._commands.put(("1", code))       # tr_type 1 = 등록

    # -- 조회 --------------------------------------------------------------

    def last(self, code: str) -> dict | None:
        """이 종목의 마지막 틱. 없으면 None (0 을 돌려주면 0원에 주문합니다)."""
        with self._lock:
            tick = self._last.get(str(code))
        if not tick:
            return None
        return {**tick, "age_sec": round(time.time() - tick["received_at"], 2)}

    def snapshot(self) -> dict[str, dict]:
        now = time.time()
        with self._lock:
            return {k: {**v, "age_sec": round(now - v["received_at"], 2)}
                    for k, v in self._last.items()}

    # -- 초봉 --------------------------------------------------------------

    def _fold_into_bar(self, tick: dict):
        """틱 하나를 1초봉에 접어 넣습니다. (_lock 을 잡은 상태로 부릅니다)

        초봉은 어떤 공개 API 도 주지 않습니다. 체결을 직접 모으는 것이
        유일한 방법이고, 그래서 이 스트림이 붙어 있는 동안에만 쌓입니다.
        """
        code = tick["code"]
        second = _sec_of_day(tick.get("time"), tick["received_at"])
        price, volume = tick["price"], tick.get("tick_volume") or 0.0

        bars = self._bars.get(code)
        if bars is None:
            if len(self._bars) >= BAR_SYMBOLS:
                # 가장 오래 안 쓰인 종목의 봉을 버립니다 (dict 는 삽입 순서 유지)
                self._bars.pop(next(iter(self._bars)), None)
            bars = self._bars[code] = deque(maxlen=BAR_HISTORY)

        if bars and bars[-1]["t"] == second:
            bar = bars[-1]
            bar["h"] = max(bar["h"], price)
            bar["l"] = min(bar["l"], price)
            bar["c"] = price
            bar["v"] += volume
            bar["n"] += 1
            return
        if bars and bars[-1]["t"] > second:
            return          # 시각이 거꾸로 온 틱 — 봉 순서가 깨지므로 버립니다
        bars.append({"t": second, "o": price, "h": price, "l": price,
                     "c": price, "v": volume, "n": 1})

    def bars(self, code: str, interval_sec: int = 1, count: int = 300) -> list[dict]:
        """N초봉. 1초봉을 묶어서 만듭니다.

        `t` 는 자정으로부터의 초입니다 (화면에서 HH:MM:SS 로 풉니다).
        `n` 은 그 봉에 들어간 체결 건수 — 초단타에서는 거래량만큼 중요합니다.
        같은 거래량이라도 체결 1건과 50건은 전혀 다른 상황입니다.
        """
        interval = max(int(interval_sec or 1), 1)
        count = max(int(count or 300), 1)
        with self._lock:
            raw = list(self._bars.get(str(code)) or [])
        if not raw:
            return []
        if interval == 1:
            return [dict(b) for b in raw[-count:]]

        grouped: dict[int, dict] = {}
        for bar in raw:
            key = bar["t"] // interval * interval
            merged = grouped.get(key)
            if merged is None:
                grouped[key] = {**bar, "t": key}
            else:
                merged["h"] = max(merged["h"], bar["h"])
                merged["l"] = min(merged["l"], bar["l"])
                merged["c"] = bar["c"]
                merged["v"] += bar["v"]
                merged["n"] += bar["n"]
        keys = sorted(grouped)[-count:]
        return [grouped[k] for k in keys]

    def bar_count(self, code: str) -> int:
        with self._lock:
            return len(self._bars.get(str(code)) or [])

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self.is_running(),
                "connected": self._connected,
                "error": self._error,
                "wanted": sorted(self._wanted),
                "active": sorted(self._active),
                "dropped": list(self._dropped),
                "max_subscriptions": self._effective_limit or MAX_SUBSCRIPTIONS,
                "limit_learned": bool(self._effective_limit),
                "tick_count": self._tick_count,
                "last_tick_age": (round(time.time() - self._last_tick_at, 1)
                                  if self._last_tick_at else None),
                "uptime_sec": (round(time.time() - self._connected_at)
                               if self._connected_at else 0),
            }

    # -- 내부 --------------------------------------------------------------

    def _run(self):
        from websockets.sync.client import connect

        delay = RECONNECT_DELAY
        while not self._stop.is_set():
            approval, error = request_approval_key()
            if not approval:
                self._set_error(error)
                if self._stop.wait(delay):
                    break
                delay = min(delay * 2, MAX_RECONNECT_DELAY)
                continue

            try:
                with connect(ws_url() + TICK_PATH, open_timeout=10,
                             close_timeout=3, max_size=None) as ws:
                    self._on_connected()
                    delay = RECONNECT_DELAY
                    # 재연결이면 원하는 종목을 전부 다시 등록해야 합니다.
                    # 서버는 연결이 끊긴 순간 구독을 전부 잊습니다.
                    #
                    # 밀린 명령을 먼저 비우는 이유: start() 직후 set_symbols() 가
                    # 들어오면 그쪽에서도 등록 명령을 넣습니다. 그대로 두면 같은
                    # 종목을 두 번 보내 "ALREADY IN SUBSCRIBE" 거부를 받고,
                    # 실제로는 멀쩡한데 화면에는 오류로 뜹니다.
                    while True:
                        try:
                            self._commands.get_nowait()
                        except queue.Empty:
                            break
                    with self._lock:
                        self._active.clear()
                        pending = sorted(self._wanted)
                    for code in pending:
                        self._commands.put(("1", code))

                    self._pump(ws, approval)
            except Exception as exc:
                self._set_error(f"{type(exc).__name__}: {exc}")

            self._connected = False
            if self._stop.wait(delay):
                break
            delay = min(delay * 2, MAX_RECONNECT_DELAY)

    def _pump(self, ws, approval: str):
        """수신 루프. 구독 변경도 **이 스레드에서만** 보냅니다."""
        while not self._stop.is_set():
            # 1) 밀린 구독 변경 먼저 처리
            while True:
                try:
                    tr_type, code = self._commands.get_nowait()
                except queue.Empty:
                    break
                # 이미 그 상태면 보내지 않습니다 — 중복 등록/해제는 서버가
                # 거부하고, 그 거부가 진짜 문제를 가립니다.
                with self._lock:
                    subscribed = code in self._active
                if (tr_type == "1") == subscribed:
                    continue
                with self._lock:
                    hint = self._exchanges.get(code)
                try:
                    ws.send(self._subscribe_frame(approval, tr_type, code, hint))
                except Exception:
                    self._commands.put((tr_type, code))   # 재연결 후 다시 시도
                    raise
                with self._lock:
                    if tr_type == "1":
                        self._active.add(code)
                        self._pending_code = code
                    else:
                        self._active.discard(code)

            # 2) 수신
            try:
                message = ws.recv(timeout=RECV_TIMEOUT)
            except TimeoutError:
                continue
            if message:
                self._handle(ws, message)

    @staticmethod
    def _subscribe_frame(approval: str, tr_type: str, code: str,
                         exchange: str = None) -> str:
        return json.dumps({
            "header": {
                "approval_key": approval,
                "custtype": "P",              # 개인
                "tr_type": tr_type,           # 1 등록 / 2 해제
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": tr_for(code),
                               "tr_key": tr_key_for(code, exchange)}},
        })

    def _handle(self, ws, message):
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")

        # 제어 메시지(JSON) — 구독 결과, PINGPONG
        if message.startswith("{"):
            try:
                body = json.loads(message)
            except json.JSONDecodeError:
                return
            header = body.get("header") or {}
            if header.get("tr_id") == "PINGPONG":
                # 그대로 되돌려주지 않으면 서버가 연결을 끊습니다
                try:
                    ws.send(message)
                except Exception:
                    pass
                return
            result = body.get("body") or {}
            if str(result.get("rt_cd", "0")) != "0":
                self._on_subscribe_reject(str(result.get("msg_cd") or ""),
                                          str(result.get("msg1") or ""))
            return

        # 실시간 데이터: 암호화여부|TR_ID|건수|데이터^데이터^...
        parts = message.split("|", 3)
        if len(parts) < 4 or parts[1] not in (TR_TICK, TR_TICK_US):
            return
        overseas = parts[1] == TR_TICK_US
        try:
            count = max(int(parts[2]), 1)
        except ValueError:
            count = 1

        values = parts[3].split("^")
        width = len(values) // count
        # 국내와 해외는 필드 배치가 완전히 다릅니다. 폭이 모자란 채로 인덱스를
        # 읽으면 엉뚱한 값을 가격으로 쓰게 되고, 그 값에 주문이 나갑니다.
        need = (U_STRENGTH if overseas else F_VOL_RATIO)
        if width <= need:
            self._set_error(
                f"{'해외' if overseas else '국내'} 체결 필드 수 이상 "
                f"(건수 {count}, 폭 {width})")
            return

        now = time.time()
        for i in range(count):
            row = values[i * width:(i + 1) * width]
            tick = self._parse_us(row, now) if overseas else self._parse(row, now)
            if not tick:
                continue
            with self._lock:
                self._last[tick["code"]] = tick
                self._fold_into_bar(tick)
                self._tick_count += 1
                self._last_tick_at = now
            if self.on_tick:
                try:
                    self.on_tick(tick)
                except Exception:
                    pass          # 콜백 예외로 스트림이 죽으면 안 됩니다

    @staticmethod
    def _parse(row: list, now: float) -> dict | None:
        code = (row[F_CODE] or "").strip()
        price = _num(row[F_PRICE])
        if len(code) != 6 or price <= 0:
            return None
        high, low = _num(row[F_HIGH]), _num(row[F_LOW])
        return {
            "code": code,
            "time": (row[F_TIME] or "").strip(),
            "price": price,
            "bid": _num(row[F_BID]) or None,
            "ask": _num(row[F_ASK]) or None,
            "open": _num(row[F_OPEN]) or None,
            "high": high or None,
            "low": low or None,
            "day_range_pct": ((high - low) / low * 100) if high and low else None,
            "change_rate": _num(row[F_CHANGE_RATE]),
            "tick_volume": _num(row[F_TICK_VOL]),
            "volume": _num(row[F_ACML_VOL]),
            "trading_value": _num(row[F_ACML_VAL]),
            "strength": _num(row[F_STRENGTH]),        # 체결강도 (매수/매도 체결 비)
            "vol_ratio_pct": _num(row[F_VOL_RATIO]),  # 전일 거래량 대비 %
            "received_at": now,
            "source": "KIS 실시간체결",
        }

    @staticmethod
    def _parse_us(row: list, now: float) -> dict | None:
        """해외주식 실시간체결 한 건.

        `delay_sec` 을 함께 담습니다 — KIS 해외 실시간은 계정에 따라 **지연**
        시세일 수 있고, 지연 시세로는 몇 틱 승부가 불가능합니다. 지연을
        모른 채 초봉을 그리면 과거를 현재로 착각하게 됩니다.
        """
        code = (row[U_SYMB] or "").strip().upper()
        price = _num(row[U_PRICE])
        if not code or price <= 0:
            return None
        high, low = _num(row[U_HIGH]), _num(row[U_LOW])

        # 한국시간(KHMS) 기준으로 지연을 잽니다 (현지시간은 시차 계산이 필요)
        korea_sec = _sec_of_day(row[U_KHMS], now)
        local = time.localtime(now)
        now_sec = local.tm_hour * 3600 + local.tm_min * 60 + local.tm_sec
        delay = now_sec - korea_sec
        if delay < -43_200:          # 자정을 넘어간 경우
            delay += 86_400
        elif delay < 0:
            delay = 0

        return {
            "code": code,
            "time": (row[U_KHMS] or "").strip(),
            "price": price,
            "bid": _num(row[U_BID]) or None,
            "ask": _num(row[U_ASK]) or None,
            "open": _num(row[U_OPEN]) or None,
            "high": high or None,
            "low": low or None,
            "day_range_pct": ((high - low) / low * 100) if high and low else None,
            "change_rate": _num(row[U_CHANGE_RATE]),
            "tick_volume": _num(row[U_TICK_VOL]),
            "volume": _num(row[U_ACML_VOL]),
            "trading_value": _num(row[U_ACML_VAL]),
            "strength": _num(row[U_STRENGTH]),
            "vol_ratio_pct": 0.0,           # 해외 TR 은 전일대비 거래량비를 안 줍니다
            "delay_sec": delay,
            "market": "US",
            "received_at": now,
            "source": "KIS 해외주식 실시간체결",
        }

    def _on_subscribe_reject(self, code: str, message: str):
        """구독 거부 처리.

        계정의 실제 구독 한도는 문서값(41)보다 낮을 수 있습니다 — HTS·MTS 나
        다른 프로그램이 같은 계정으로 이미 잡아둔 구독이 한도를 나눠 쓰기
        때문입니다. 그래서 한도 초과를 받으면 **지금 실제로 잡힌 수**를
        이 계정의 한도로 기억하고, 다음부터 그만큼만 요청합니다.
        고정값을 믿고 계속 밀어넣으면 매번 거부만 쌓입니다.
        """
        with self._lock:
            if "OPSP0008" in code or "MAX SUBSCRIBE" in message.upper():
                learned = max(len(self._active) - 1, 1)
                self._active.discard(self._pending_code or "")
                self._effective_limit = learned
                self._error = (
                    f"구독 한도에 걸렸습니다 — 이 계정의 실제 한도를 "
                    f"{learned}종목으로 낮춰 잡습니다 (HTS·MTS 가 같은 한도를 "
                    f"나눠 씁니다)")
                return
            if "ALREADY IN SUBSCRIBE" in message.upper():
                return          # 이미 구독 중 — 문제 아님
            self._error = f"구독 거부 ({code}): {message}"

    def _on_connected(self):
        with self._lock:
            self._connected = True
            self._connected_at = time.time()
            self._error = ""
            # 구독 한도는 연결(세션) 단위입니다. 지난 세션에서 낮게 배운 값을
            # 계속 들고 가면, 일시적으로 붐볐던 것 때문에 영원히 3종목만
            # 추적하게 됩니다. 새 연결에서는 다시 확인합니다.
            self._effective_limit = 0
            self._pending_code = ""

    def _set_error(self, message: str):
        with self._lock:
            self._error = message


# ---------------------------------------------------------------------------
# 프로세스 전역 스트림 하나
# ---------------------------------------------------------------------------
# 구독 한도가 계정 단위라, 스트림을 여러 개 만들면 서로 한도를 뺏습니다.
_stream: TickStream | None = None
_stream_lock = threading.Lock()


def get_stream() -> TickStream:
    global _stream
    with _stream_lock:
        if _stream is None:
            _stream = TickStream()
        return _stream


def start(symbols: list[str] = None, exchanges: dict = None) -> dict:
    stream = get_stream()
    stream.start()
    if symbols is not None:
        stream.set_symbols(symbols, exchanges=exchanges)
    return stream.status()


def stop():
    global _stream
    with _stream_lock:
        if _stream:
            _stream.stop()
            _stream = None


def tick(code: str) -> dict | None:
    return get_stream().last(code)


def bars(code: str, interval_sec: int = 1, count: int = 300) -> list[dict]:
    return get_stream().bars(code, interval_sec, count)


def bar_count(code: str) -> int:
    return get_stream().bar_count(code)


def subscription_room() -> int:
    """지금 실제로 구독 가능한 종목 수.

    서버가 알려준 한도가 있으면 그걸 씁니다. 유니버스를 이보다 크게 잡으면
    남는 종목은 틱 없이 REST 폴링으로 돌아가, 초단타 판단이 20초 지연됩니다.
    """
    return int(get_stream().status().get("max_subscriptions") or MAX_SUBSCRIPTIONS)


def status() -> dict:
    return get_stream().status()


__all__ = ["TickStream", "get_stream", "start", "stop", "tick", "status",
           "bars", "bar_count", "subscription_room", "request_approval_key",
           "MAX_SUBSCRIPTIONS", "TR_TICK"]
