"""
한국투자증권 KIS 주문 어댑터 (실제 주문이 나가는 유일한 경로)
-------------------------------------------------------------
시세 조회는 `kis_client.py`, **주문·잔고·체결**은 이 모듈이 담당합니다.
둘을 나눈 이유는 권한 격리입니다 — 자금을 움직이는 코드는 한 파일에만 있어야
감사와 검토가 가능합니다.

지원 범위
    국내 주식/ETF   현금 매수·매도, 정정·취소, 잔고, 매수가능금액
    국내 선물/옵션  신규·청산 주문, 잔고, 시세, 일봉
    해외 주식/ETF   미국 지정가 매수·매도, 잔고
    미국 주간거래   데이마켓 지정가 매수·매도·취소, 주간 시세 (정규장과 별도 API)

3단 안전장치
    1) 키가 없으면 이 모듈 전체가 비활성 (is_configured() == False)
    2) KIS_MOCK=1 이면 모의투자 서버로만 나갑니다 (실제 자금 X)
    3) 실전 서버 주문은 KIS_LIVE_TRADING=1 을 **따로** 켜야 합니다.
       키만 있고 이 스위치가 없으면 주문은 거부됩니다 (fail-closed).

tr_id 에 대하여
    KIS 는 거래ID(tr_id)를 개편한 이력이 있습니다(예: 현금매수 TTTC0802U →
    TTTC0012U). 어느 쪽이 유효한지는 계정·시점에 따라 다를 수 있어, 이 모듈은
    후보를 **순서대로 시도**하고 성공한 것을 기억합니다. 필요하면
    api_keys.json 의 "KIS_TR_OVERRIDE" 로 코드 수정 없이 교체할 수 있습니다.

        "KIS_TR_OVERRIDE": { "stock_buy_real": ["TTTC0012U"] }

주의: 이 어댑터는 KIS 공식 문서 스펙에 맞춰 작성했으나, 실계좌 키로 응답을
      검증하지는 못했습니다. **반드시 모의투자(KIS_MOCK=1)로 먼저 돌려보고**,
      주문 응답(raw)에 담긴 rt_cd / msg1 을 확인하세요.
"""

import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from data_sources import credentials, http_client, kis_client

# ---------------------------------------------------------------------------
# 거래 ID 표 (config as data — 코드 수정 없이 교체 가능)
# ---------------------------------------------------------------------------

TR_CANDIDATES = {
    # 국내 주식 현금 주문 (신규 코드 → 구 코드 순으로 시도)
    "stock_buy_real":   ["TTTC0012U", "TTTC0802U"],
    "stock_sell_real":  ["TTTC0011U", "TTTC0801U"],
    "stock_buy_mock":   ["VTTC0012U", "VTTC0802U"],
    "stock_sell_mock":  ["VTTC0011U", "VTTC0801U"],
    "stock_cancel_real": ["TTTC0013U", "TTTC0803U"],
    "stock_cancel_mock": ["VTTC0013U", "VTTC0803U"],
    "stock_balance_real": ["TTTC8434R"],
    "stock_balance_mock": ["VTTC8434R"],
    "stock_buyable_real": ["TTTC8908R"],
    "stock_buyable_mock": ["VTTC8908R"],
    # 계좌 요약 — 증권사 앱 '총자산' 화면이 쓰는 TR 입니다.
    # 모의투자에는 없으므로(V- 코드 없음) 실전에서만 호출합니다.
    "account_assets_real": ["CTRP6548R"],
    # 체결 조회 — 실계좌 매매의 핵심입니다. '주문 접수'와 '체결'은 다른 사건이라,
    # 접수만 보고 포지션을 잡았다고 착각하면 계좌와 내부 상태가 어긋납니다.
    "stock_executions_real": ["TTTC8001R", "CTSC9115R"],
    "stock_executions_mock": ["VTTC8001R"],
    "stock_open_orders_real": ["TTTC8036R"],
    "stock_open_orders_mock": ["VTTC8036R"],
    # 국내 선물·옵션
    "deriv_order_real":  ["TTTO1101U"],
    "deriv_order_mock":  ["VTTO1101U"],
    "deriv_cancel_real": ["TTTO1103U"],
    "deriv_cancel_mock": ["VTTO1103U"],
    "deriv_executions_real": ["TTTO5201R"],
    "deriv_executions_mock": ["VTTO5201R"],
    "deriv_balance_real": ["CTFO6118R"],
    "deriv_balance_mock": ["VTFO6118R"],
    # 해외 주식 (미국)
    "overseas_buy_real":  ["TTTT1002U"],
    "overseas_sell_real": ["TTTT1006U"],
    "overseas_buy_mock":  ["VTTT1002U"],
    "overseas_sell_mock": ["VTTT1001U"],
    "overseas_balance_real": ["TTTS3012R"],
    "overseas_balance_mock": ["VTTS3012R"],
    # 체결기준 현재잔고 — 원화 환산·미결제·출금가능을 증권사 계산으로 받습니다
    "overseas_present_real": ["CTRP6504R"],
    "overseas_present_mock": ["VTRP6504R"],
    "overseas_buyable_real": ["TTTS3007R"],
    "overseas_buyable_mock": ["VTTS3007R"],
    "overseas_executions_real": ["TTTS3035R"],
    "overseas_executions_mock": ["VTTS3035R"],
    "overseas_cancel_real": ["TTTT1004U"],
    "overseas_cancel_mock": ["VTTT1004U"],
    # 미국 주간거래 (데이마켓) — 정규장과 API 가 통째로 다릅니다.
    # 모의투자 짝(V로 시작하는 tr_id)이 없습니다 → 모의 모드에서는 거부합니다.
    "overseas_day_buy_real":    ["TTTS6036U"],
    "overseas_day_sell_real":   ["TTTS6037U"],
    "overseas_day_cancel_real": ["TTTS6038U"],
    "overseas_day_quote":       ["HHDFS76200200"],
    # 시세 (주문 아님)
    "deriv_quote": ["FHMIF10000000"],
    "deriv_chart": ["FHKIF03020100"],
}

# 성공한 tr_id 를 기억해 다음부터 첫 시도에 맞춥니다 (불필요한 실패 호출 감소)
_tr_hit: dict[str, str] = {}
_tr_lock = threading.Lock()

# 잔고 조회 캐시 — 화면(7초 폴링)과 엔진(회전당 2~4회)이 같은 잔고를 반복 조회합니다.
# KIS 는 초당 호출 한도가 있어서, 캐시 없이는 잔고 조회만으로 한도를 다 씁니다.
# 주문을 내면 잔고가 바뀌므로 그때는 즉시 비웁니다.
#
# TTL 이 20초인 이유: 계좌 화면 한 번을 다 그리는 데 KIS 호출이 ~10초 걸립니다
# (실측 2026-08-07 — 해외잔고 3회 5.8초 포함 합계 10.4초). TTL 이 그보다 짧으면
# **한 번의 갱신이 끝나기도 전에 캐시가 만료**되어, 같은 화면 안에서 같은 잔고를
# 두 번씩 조회합니다 (positions() 10.5초 뒤 account() 가 또 11.8초 — 실측).
# 잔고는 주문·체결 때만 바뀌고 주문 경로는 clear_balance_cache() 로 즉시 비우므로,
# 20초 묵은 값이 화면에 남는 위험은 사실상 없습니다.
_balance_cache: dict[str, tuple[float, dict]] = {}
_balance_lock = threading.Lock()
BALANCE_TTL = 20.0


def _cache_scope() -> str:
    """잔고 캐시의 계정 구분자 — 사용자 오버레이가 바뀌면 값도 바뀝니다.

    함수 이름만으로 캐시하면 한 프로세스에서 여러 사용자의 엔진·요청이 돌 때
    TTL 안에 늦게 조회한 쪽이 **앞 사람 계좌의 잔고**를 그대로 받습니다.
    화면 표시가 섞이는 것으로 끝나지 않고, 주문 수량이 남의 현금으로
    계산됩니다. 접근토큰을 앱키별로 캐시하는 것(kis_token_<해시>.json)과
    같은 이유로, 잔고도 계좌별로 캐시해야 합니다.
    """
    acc = account()
    server = "mock" if kis_client.is_mock() else "real"
    return f"{server}:{acc[0]}-{acc[1]}" if acc else f"{server}:noacct"


def _balance_cached(key: str, fetch) -> dict:
    key = f"{_cache_scope()}|{key}"
    now = time.time()
    with _balance_lock:
        hit = _balance_cache.get(key)
    if hit and now - hit[0] < BALANCE_TTL:
        return hit[1]
    result = fetch()
    if result.get("ok"):
        with _balance_lock:
            _balance_cache[key] = (now, result)
    return result


def clear_balance_cache():
    """주문·취소 직후 호출 — 방금 바뀐 잔고를 캐시가 가리면 안 됩니다."""
    with _balance_lock:
        _balance_cache.clear()

ORDER_PATH = "/uapi/domestic-stock/v1/trading/order-cash"
CANCEL_PATH = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
EXECUTIONS_PATH = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
OPEN_ORDERS_PATH = "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl"
DERIV_EXECUTIONS_PATH = "/uapi/domestic-futureoption/v1/trading/inquire-ccnl"
BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"
BUYABLE_PATH = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"
ACCOUNT_ASSETS_PATH = "/uapi/domestic-stock/v1/trading/inquire-account-balance"
DERIV_ORDER_PATH = "/uapi/domestic-futureoption/v1/trading/order"
DERIV_CANCEL_PATH = "/uapi/domestic-futureoption/v1/trading/order-rvsecncl"
DERIV_BALANCE_PATH = "/uapi/domestic-futureoption/v1/trading/inquire-balance"
DERIV_QUOTE_PATH = "/uapi/domestic-futureoption/v1/quotations/inquire-price"
DERIV_CHART_PATH = "/uapi/domestic-futureoption/v1/quotations/inquire-daily-fuopchartprice"
OVERSEAS_ORDER_PATH = "/uapi/overseas-stock/v1/trading/order"
OVERSEAS_BALANCE_PATH = "/uapi/overseas-stock/v1/trading/inquire-balance"
OVERSEAS_PRESENT_PATH = "/uapi/overseas-stock/v1/trading/inquire-present-balance"
OVERSEAS_BUYABLE_PATH = "/uapi/overseas-stock/v1/trading/inquire-psamount"
OVERSEAS_EXECUTIONS_PATH = "/uapi/overseas-stock/v1/trading/inquire-ccnl"
OVERSEAS_CANCEL_PATH = "/uapi/overseas-stock/v1/trading/order-rvsecncl"
OVERSEAS_DAY_ORDER_PATH = "/uapi/overseas-stock/v1/trading/daytime-order"
OVERSEAS_DAY_CANCEL_PATH = "/uapi/overseas-stock/v1/trading/daytime-order-rvsecncl"
OVERSEAS_DAY_QUOTE_PATH = "/uapi/overseas-price/v1/quotations/price-detail"


def _tr_list(op: str) -> list[str]:
    override = credentials.get_json("KIS_TR_OVERRIDE", {}) or {}
    if isinstance(override, dict) and override.get(op):
        value = override[op]
        return [value] if isinstance(value, str) else list(value)
    candidates = list(TR_CANDIDATES.get(op, []))
    hit = _tr_hit.get(op)
    if hit and hit in candidates:
        candidates.remove(hit)
        candidates.insert(0, hit)
    return candidates


def _remember(op: str, tr_id: str):
    with _tr_lock:
        _tr_hit[op] = tr_id


def _op(base: str) -> str:
    """실전/모의에 따라 연산 키를 고릅니다."""
    return f"{base}_{'mock' if kis_client.is_mock() else 'real'}"


# ---------------------------------------------------------------------------
# 설정 · 안전장치
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    """키와 계좌번호가 모두 있어야 주문 어댑터가 살아납니다."""
    return bool(kis_client.is_configured() and account())


def account(kind: str = "stock") -> tuple[str, str] | None:
    """(종합계좌번호 8자리, 상품코드 2자리). 형식: 12345678-01

    선물옵션 계좌는 상품코드가 다릅니다(보통 03). KIS_DERIV_ACCOUNT 를 따로
    지정할 수 있고, 없으면 주식 계좌를 그대로 씁니다.
    """
    raw = ""
    if kind == "deriv":
        raw = credentials.get("KIS_DERIV_ACCOUNT")
    raw = raw or credentials.get("KIS_ACCOUNT")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) < 10:
        return None
    return digits[:8], digits[8:10]


def live_enabled() -> bool:
    """실전 서버로 진짜 주문을 낼 수 있는가.

    모의투자 서버(KIS_MOCK=1)는 이 스위치와 무관하게 허용합니다 — 가상 자금이라
    잘못 나가도 손실이 없고, "실제로 주문이 체결되는지" 검증에 꼭 필요합니다.
    """
    if kis_client.is_mock():
        return True
    return credentials.get_bool("KIS_LIVE_TRADING", False)


def status() -> dict:
    """설정 현황 — 화면에 그대로 띄우기 위한 요약 (키 값은 노출하지 않음)."""
    acc = account()
    return {
        "keys_configured": kis_client.is_configured(),
        "account_configured": bool(acc),
        "account_masked": (f"{acc[0][:4]}****-{acc[1]}" if acc else ""),
        "mock": kis_client.is_mock(),
        "live_enabled": live_enabled(),
        "server": kis_client.base_url(),
        "ready": is_configured() and live_enabled(),
    }


class OrderRejected(Exception):
    """브로커가 주문을 받지 않았을 때. 상위에서 사유를 그대로 기록합니다."""


# ---------------------------------------------------------------------------
# 저수준 호출
# ---------------------------------------------------------------------------

def _hashkey(body: dict) -> str | None:
    """주문 본문 위변조 검증용 해시. 실패해도 주문은 진행합니다(선택 헤더)."""
    try:
        res = http_client.post_json(
            kis_client.base_url() + "/uapi/hashkey",
            json_body=body,
            headers={
                "Content-Type": "application/json",
                "appkey": kis_client.app_key(),
                "appsecret": kis_client.app_secret(),
            },
            timeout=10,
        )
        return (res or {}).get("HASH")
    except Exception:
        return None


def _post(op: str, path: str, body: dict, timeout: int = 15) -> dict:
    """주문 계열 POST. tr_id 후보를 순서대로 시도합니다.

    주문·정정·취소는 잔고를 바꾸므로 성공 여부와 무관하게 잔고 캐시를 비웁니다
    (부분 성공·타임아웃 후 체결 같은 애매한 경우까지 안전하게).
    """
    clear_balance_cache()
    if not kis_client.is_configured():
        return {"ok": False, "error": "KIS 키가 설정되지 않았습니다."}

    last = {"ok": False, "error": "요청을 보내지 못했습니다."}
    hashkey = _hashkey(body)

    for tr_id in _tr_list(op):
        extra = {"hashkey": hashkey} if hashkey else None
        headers = kis_client.auth_headers(tr_id, extra)
        if not headers:
            return {"ok": False, "error": "KIS 접근토큰 발급에 실패했습니다."}

        kis_client.throttle()
        status, res, raw = http_client.post_full(
            kis_client.base_url() + path, json_body=body, headers=headers, timeout=timeout)
        if _is_rate_limited(res, raw):
            status, res, raw = _retry_after_limit(
                lambda: http_client.post_full(kis_client.base_url() + path,
                                              json_body=body, headers=headers,
                                              timeout=timeout))
        if res is None or status != 200:
            last = _http_error(status, res, raw, tr_id)
            # 서버가 준 거부 사유가 있으면 다음 tr_id 를 시도할 이유가 없습니다
            if res and not _looks_like_bad_tr(res):
                break
            continue

        if str(res.get("rt_cd")) == "0":
            _remember(op, tr_id)
            return {"ok": True, "tr_id": tr_id, "raw": res,
                    "output": res.get("output") or res.get("output1") or {},
                    "message": res.get("msg1", "")}

        last = {"ok": False, "tr_id": tr_id, "raw": res,
                "code": res.get("msg_cd", ""),
                "error": res.get("msg1") or f"KIS 거부 (rt_cd={res.get('rt_cd')})"}
        # tr_id 자체가 틀린 경우에만 다음 후보로 넘어갑니다.
        # 잔고 부족·장 마감 같은 사유라면 다시 시도해도 결과가 같습니다.
        if not _looks_like_bad_tr(res):
            break

    return last


def _get(op: str, path: str, params: dict, timeout: int = 15) -> dict:
    if not kis_client.is_configured():
        return {"ok": False, "error": "KIS 키가 설정되지 않았습니다."}

    last = {"ok": False, "error": "요청을 보내지 못했습니다."}
    for tr_id in _tr_list(op):
        headers = kis_client.auth_headers(tr_id)
        if not headers:
            return {"ok": False, "error": "KIS 접근토큰 발급에 실패했습니다."}

        kis_client.throttle()
        status, res, raw = http_client.get_full(
            kis_client.base_url() + path, headers=headers, params=params, timeout=timeout)
        if _is_rate_limited(res, raw):
            status, res, raw = _retry_after_limit(
                lambda: http_client.get_full(kis_client.base_url() + path,
                                             headers=headers, params=params,
                                             timeout=timeout))
        if res is None or status != 200:
            last = _http_error(status, res, raw, tr_id)
            if res and not _looks_like_bad_tr(res):
                break
            continue
        if str(res.get("rt_cd")) == "0":
            _remember(op, tr_id)
            return {"ok": True, "tr_id": tr_id, "raw": res,
                    "output": res.get("output"),
                    "output1": res.get("output1"), "output2": res.get("output2"),
                    "message": res.get("msg1", "")}
        last = {"ok": False, "tr_id": tr_id, "raw": res,
                "code": res.get("msg_cd", ""),
                "error": res.get("msg1") or f"KIS 거부 (rt_cd={res.get('rt_cd')})"}
        if not _looks_like_bad_tr(res):
            break
    return last


def _is_rate_limited(body: dict | None, raw: str) -> bool:
    """'초당 거래건수를 초과하였습니다' 계열인가.

    KIS 는 계정당 초당 호출 수를 제한합니다(실전 약 20건, 모의 2건).
    자동매매는 종목마다 조회를 반복하므로 여기에 아주 쉽게 걸립니다.
    """
    text = ""
    if isinstance(body, dict):
        text = f"{body.get('msg1', '')} {body.get('msg_cd', '')}"
    text += " " + (raw or "")
    return any(k in text for k in ("초당", "거래건수", "EGW00201", "rate limit"))


def _retry_after_limit(call, attempts: int = 3):
    """제한에 걸리면 잠깐 쉬었다가 다시 시도합니다.

    한 번 걸렸다고 실패로 처리하면, 종목이 몇 개만 늘어도 자동매매가
    "조회 실패"투성이가 됩니다. 실패가 아니라 **기다리면 되는 상황**입니다.
    """
    delay = max(kis_client.min_interval() * 2, 0.5)
    status, body, raw = 0, None, ""
    for _ in range(attempts):
        time.sleep(delay)
        kis_client.throttle()       # 재시도도 간격 규칙을 지켜야 합니다
        status, body, raw = call()
        if not _is_rate_limited(body, raw):
            return status, body, raw
        delay *= 2
    return status, body, raw


def _http_error(status: int, body: dict | None, raw: str, tr_id: str) -> dict:
    """비200 응답을 사람이 읽을 수 있는 사유로 바꿉니다.

    KIS 는 거부 사유를 본문(msg1)에 담아 보냅니다. 그걸 버리면 화면에는
    "응답 없음"만 남아 원인을 알 수 없습니다.
    """
    message = ""
    if isinstance(body, dict):
        message = body.get("msg1") or body.get("error_description") or body.get("msg") or ""
    if not message:
        message = (raw or "").strip().replace("\n", " ")[:200]

    hint = ""
    mock = kis_client.is_mock()
    if _is_rate_limited(body, raw):
        hint = ("초당 호출 제한입니다. 잠시 후 자동으로 재시도합니다 — "
                "연결이나 권한 문제가 아닙니다.")
    elif status in (401, 403):
        hint = ("앱키/토큰이 이 서버에서 인증되지 않았습니다. "
                "한국투자증권은 **실전용 앱키와 모의투자용 앱키가 다릅니다** — "
                f"지금은 {'모의투자' if mock else '실전'} 서버로 붙는 중입니다.")
    elif status == 500:
        hint = ("서버가 요청을 거부했습니다. 거래ID와 서버가 맞는지"
                f"(현재 tr_id={tr_id}, {'모의투자' if mock else '실전'} 서버), "
                "계좌번호(앞 8자리-뒤 2자리)가 맞는지 확인하세요.")
    elif status == 0:
        hint = "네트워크에 연결하지 못했습니다 (방화벽·인터넷 확인)."

    return {"ok": False, "tr_id": tr_id, "status": status, "raw": body,
            "error": f"HTTP {status} — {message or '본문 없음'}" + (f" · {hint}" if hint else "")}


def _looks_like_bad_tr(res: dict) -> bool:
    """'거래ID가 잘못됐다' 계열 오류인지. 이때만 다음 후보를 시도합니다."""
    text = f"{res.get('msg_cd', '')} {res.get('msg1', '')}"
    return any(k in text for k in ("거래ID", "TR", "tr_id", "유효하지 않은",
                                   "EGW00", "not found", "OPSQ"))


def _num(v, default=0.0) -> float:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 국내 주식 / ETF
# ---------------------------------------------------------------------------

def place_stock_order(code: str, side: str, quantity: int,
                      price: float = 0, order_type: str = "market") -> dict:
    """국내 주식·ETF 주문.

    order_type
        market  시장가 (ORD_DVSN=01, 단가 0)
        limit   지정가 (ORD_DVSN=00, 단가 지정)
    """
    if not live_enabled():
        return {"ok": False, "error": "실전 주문이 잠겨 있습니다 "
                                      "(KIS_LIVE_TRADING=1 필요). 모의투자는 KIS_MOCK=1."}
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT(계좌번호)가 설정되지 않았습니다."}

    quantity = int(quantity)
    if quantity <= 0:
        return {"ok": False, "error": "주문 수량이 0입니다."}

    cano, prdt = acc
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "PDNO": code,
        "ORD_DVSN": "01" if order_type == "market" else "00",
        "ORD_QTY": str(quantity),
        "ORD_UNPR": "0" if order_type == "market" else str(int(price)),
    }
    op = _op("stock_buy") if side == "buy" else _op("stock_sell")
    res = _post(op, ORDER_PATH, body)
    if not res.get("ok"):
        return res

    out = res.get("output") or {}
    return {
        "ok": True,
        "broker_order_id": out.get("ODNO", ""),
        "order_time": out.get("ORD_TMD", ""),
        "krx_fwdg_ord_orgno": out.get("KRX_FWDG_ORD_ORGNO", ""),
        "message": res.get("message", ""),
        "raw": res.get("raw"),
    }


def cancel_stock_order(broker_order_id: str, org_no: str = "",
                       quantity: int = 0, all_remaining: bool = True) -> dict:
    if not live_enabled():
        return {"ok": False, "error": "실전 주문이 잠겨 있습니다."}
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}

    cano, prdt = acc
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "KRX_FWDG_ORD_ORGNO": org_no,
        "ORGN_ODNO": broker_order_id,
        "ORD_DVSN": "00",
        "RVSE_CNCL_DVSN_CD": "02",           # 02 = 취소
        "ORD_QTY": "0" if all_remaining else str(int(quantity)),
        "ORD_UNPR": "0",
        "QTY_ALL_ORD_YN": "Y" if all_remaining else "N",
    }
    return _post(_op("stock_cancel"), CANCEL_PATH, body)


def stock_balance() -> dict:
    """주식 잔고 — 보유종목과 예수금 (5초 캐시)."""
    return _balance_cached("stock", _stock_balance_fetch)


def _stock_balance_fetch() -> dict:
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc
    res = _get(_op("stock_balance"), BALANCE_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
        "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
    })
    if not res.get("ok"):
        return res

    positions = []
    for row in (res.get("output1") or []):
        qty = _num(row.get("hldg_qty"))
        if qty <= 0:
            continue
        positions.append({
            "code": row.get("pdno", ""),
            "name": row.get("prdt_name", ""),
            "quantity": qty,
            "avg_price": _num(row.get("pchs_avg_pric")),
            "current_price": _num(row.get("prpr")),
            "eval_amount": _num(row.get("evlu_amt")),
            "pnl": _num(row.get("evlu_pfls_amt")),
            "pnl_pct": _num(row.get("evlu_pfls_rt")),
        })

    summary = (res.get("output2") or [{}])
    summary = summary[0] if isinstance(summary, list) and summary else {}
    return {
        "ok": True,
        "positions": positions,
        "cash": _num(summary.get("dnca_tot_amt")),           # 예수금 총액
        "available_cash": _num(summary.get("prvs_rcdl_excc_amt")  # 가수도정산금액(D+2 예수금)
                               or summary.get("dnca_tot_amt")),
        # 유가증권 평가액만. tot_evlu_amt(총평가금액)를 쓰면 안 됩니다 —
        # 그 값은 **예수금을 포함**해서, 여기에 현금을 더하면 현금이 두 번 세어집니다
        # (보유 종목이 하나도 없는 계좌의 총평가금액이 예수금과 같은 것이 그 증거입니다).
        "eval_amount": _securities_value(summary),
        "purchase_amount": _num(summary.get("pchs_amt_smtl_amt")),
        "total_eval": _num(summary.get("tot_evlu_amt")),     # 참고용(예수금 포함)
        "total_pnl": _num(summary.get("evlu_pfls_smtl_amt")),
        "raw": res.get("raw"),
    }


def _securities_value(summary: dict) -> float:
    """보유 유가증권의 평가액(예수금 제외).

    KIS 가 scts_evlu_amt(유가증권평가금액)를 주면 그대로 쓰고, 없으면
    총평가금액에서 정산예수금을 빼서 만듭니다. 뺄셈이 음수가 되면 (필드 의미가
    가정과 다르다는 뜻이므로) 0 으로 둡니다 — 자산을 부풀리는 것보다 안전합니다.
    """
    direct = summary.get("scts_evlu_amt")
    if direct not in (None, ""):
        return _num(direct)
    total = _num(summary.get("tot_evlu_amt"))
    deposit = _num(summary.get("prvs_rcdl_excc_amt") or summary.get("dnca_tot_amt"))
    return max(total - deposit, 0.0)


def buyable_cash(code: str, price: float = 0) -> dict:
    """매수 가능 금액 조회 (주문 전 잔고 확인)."""
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc
    res = _get(_op("stock_buyable"), BUYABLE_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt, "PDNO": code,
        "ORD_UNPR": str(int(price or 0)), "ORD_DVSN": "01",
        "CMA_EVLU_AMT_ICLD_YN": "N", "OVRS_ICLD_YN": "N",
    })
    if not res.get("ok"):
        return res
    out = res.get("output") or {}
    return {
        "ok": True,
        "available_cash": _num(out.get("ord_psbl_cash")),
        "max_quantity": _num(out.get("nrcvb_buy_qty") or out.get("max_buy_qty")),
        "raw": res.get("raw"),
    }


# ---------------------------------------------------------------------------
# 계좌 요약 — 증권사 앱 '총자산' 화면과 같은 숫자
# ---------------------------------------------------------------------------
#
# 총자산을 우리가 다시 계산하면 증권사 화면과 반드시 어긋납니다. 예수금·평가금액
# 사이에 미결제 매수대금, 미결제 매도대금, 통합증거금, 환율 기준이 끼어 있는데
# 그 규칙이 계좌 유형마다 다르기 때문입니다. 그래서 아래 함수들은 **계산하지
# 않고 증권사가 계산한 값을 받아 적습니다.**
#
# 실계좌로 확인한 예 (2026-08-07, 위탁계좌 1000****-01)
#     예수금 132,226 · 주문가능 3,325 · 매입 129,379 · 평가 129,194
#     미결제매수 129,692 · 미결제매도 7,235 · 총자산 139,518
# 예수금을 '쓸 수 있는 돈'으로 읽으면 3,325원짜리 계좌로 132,226원어치 주문을
# 냅니다 — 40배입니다. 이게 이 코드가 있는 이유입니다.

def account_assets() -> dict:
    """투자계좌 자산현황 (CTRP6548R) — 증권사 앱 '총자산' 화면의 원본.

    여기서 오는 매입금액·평가금액은 **결제완료분만** 셉니다. 오늘 산 종목은
    아직 미결제라 잡히지 않고, 그 매수대금은 예수금에 그대로 남아 있습니다
    (그래서 총자산 = 예수금 + 결제완료 평가금액 으로 맞아떨어집니다).
    지금 실제로 들고 있는 물량의 평가금액이 필요하면 overseas_present() 쪽을
    보세요 — 그건 체결기준입니다.
    """
    return _balance_cached("account_assets", _account_assets_fetch)


def _account_assets_fetch() -> dict:
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    if kis_client.is_mock():
        return {"ok": False, "error": "모의투자 서버는 자산현황(CTRP6548R)을 제공하지 않습니다."}

    cano, prdt = acc
    res = _get(_op("account_assets"), ACCOUNT_ASSETS_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "INQR_DVSN_1": "", "BSPR_BF_DT_APLY_YN": "",
    })
    if not res.get("ok"):
        return res

    summary = res.get("output2") or {}
    if isinstance(summary, list):
        summary = summary[0] if summary else {}
    if not summary:
        return {"ok": False, "error": "자산현황 응답이 비어 있습니다."}
    return {
        "ok": True,
        "total_asset": _num(summary.get("tot_asst_amt")),      # 총자산 (앱 화면 숫자)
        "deposit": _num(summary.get("tot_dncl_amt") or summary.get("dncl_amt")),
        "purchase_amount": _num(summary.get("pchs_amt_smtl")),  # 결제완료분 매입금액
        "eval_amount": _num(summary.get("evlu_amt_smtl")),      # 결제완료분 평가금액
        "pnl": _num(summary.get("evlu_pfls_amt_smtl")),
        "net_asset": _num(summary.get("nass_tot_amt")),
        "overseas_eval": _num(summary.get("ovrs_stck_evlu_amt1")),
        "raw": res.get("raw"),
    }


def overseas_present() -> dict:
    """해외주식 체결기준 현재잔고 (CTRP6504R) — 원화 환산·미결제·출금가능.

    overseas_balance() 와 다른 점
        · 금액이 **원화로** 옵니다 (증권사 기준환율로 환산된 값 그대로).
          우리가 별도 환율로 곱하면 화면 숫자가 증권사와 미세하게 어긋납니다.
        · 미결제 매수/매도 금액과 출금가능금액이 함께 옵니다. 예수금에서
          아직 빠져나가지 않은 매수대금이 얼마인지는 여기서만 알 수 있습니다.
    """
    return _balance_cached("overseas_present", _overseas_present_fetch)


def _overseas_present_fetch() -> dict:
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc
    res = _get(_op("overseas_present"), OVERSEAS_PRESENT_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "WCRC_FRCR_DVSN_CD": "01",       # 01 = 원화 환산
        "NATN_CD": "000",                # 전체 국가
        "TR_MKET_CD": "00",              # 전체 시장
        "INQR_DVSN_CD": "00",            # 전체 (01 일반 / 02 통합증거금)
    })
    if not res.get("ok"):
        return res

    raw = res.get("raw") or {}
    summary = raw.get("output3") or {}
    if isinstance(summary, list):
        summary = summary[0] if summary else {}
    if not summary:
        return {"ok": False, "error": "체결기준잔고 응답이 비어 있습니다."}

    # 기준환율은 종목 줄에 실려 옵니다. 보유분이 없으면 0 이고, 그때는
    # 호출부가 자체 환율로 넘어가야 합니다 (0 을 곱하면 자산이 사라집니다).
    rate = 0.0
    for row in (raw.get("output1") or []):
        rate = _num(row.get("bass_exrt"))
        if rate > 0:
            break

    # 이 TR 의 tot_asst_amt 는 **해외주식까지만** 셉니다. 실계좌 두 개로 확인한
    # 분해식 (2026-08-29):
    #     tot_asst_amt = tot_frcr_cblc_smtl(총현금) + evlu_amt_smtl(해외평가)
    #                    − ustl_buy_amt_smtl + ustl_sll_amt_smtl
    #     1000****-01  1,007,247 + 42,948 − 23,814,950 + 23,296,454 = 531,699 ✓
    #     1002****-01    430,721 + 235,677 −     59,076 +     72,258 = 679,580 ✓
    # 국내 보유분이 이 식에 없습니다. 국내 종목을 들고 있으면 그만큼 총자산이
    # 비므로, 호출부(engine/broker.py)가 국내 평가금액을 더해야 합니다.
    #
    # tot_frcr_cblc_smtl 은 이름과 달리 **원화 예수금까지 합친 총현금**입니다
    # (해외 보유가 없는 계좌에서 원화 예수금과 정확히 같은 값이 나옵니다).
    # 이 값이 없는 응답을 대비해 원화예수금 + 외화예수금으로도 만들어 둡니다.
    krw_deposit = _num(summary.get("tot_dncl_amt"))
    foreign_cash = _num(summary.get("frcr_evlu_tota")
                        or summary.get("frcr_use_psbl_amt"))
    cash_total = _num(summary.get("tot_frcr_cblc_smtl")) or (krw_deposit + foreign_cash)

    return {
        "ok": True,
        # 결제까지 반영한 순자산. 증권사 앱의 총자산(=자산현황 TR)과 몇백 원
        # 차이가 날 수 있습니다 — 한쪽은 결제기준, 한쪽은 체결기준입니다.
        "total_asset": _num(summary.get("tot_asst_amt")),
        "deposit": krw_deposit,
        "foreign_cash": foreign_cash,      # 외화 예수금 (원화 환산)
        "cash_total": cash_total,          # 원화 + 외화 예수금
        "withdrawable": _num(summary.get("wdrw_psbl_tot_amt")),
        "purchase_amount": _num(summary.get("pchs_amt_smtl")
                                or summary.get("pchs_amt_smtl_amt")),
        "eval_amount": _num(summary.get("evlu_amt_smtl")
                            or summary.get("evlu_amt_smtl_amt")),
        "pnl": _num(summary.get("evlu_pfls_amt_smtl")
                    or summary.get("tot_evlu_pfls_amt")),
        "pnl_pct": _num(summary.get("evlu_erng_rt1")),
        "unsettled_buy": _num(summary.get("ustl_buy_amt_smtl")),
        "unsettled_sell": _num(summary.get("ustl_sll_amt_smtl")),
        "fx_rate": rate,
        "raw": raw,
    }


# 주문가능현금 조회는 종목코드를 요구하지만, ord_psbl_cash 자체는 종목과 무관한
# '계좌에 남은 현금'입니다. 그래서 아무 종목이나 하나 넣어 물어봅니다.
CASH_PROBE_CODE = "005930"      # 삼성전자 — 상장폐지 걱정이 없는 조회용 더미


def orderable_cash() -> dict:
    """지금 실제로 주문에 쓸 수 있는 원화 (주문가능현금).

    예수금과 다릅니다. 미결제 매수대금이 결제일까지 예수금에 남아 있어서,
    예수금을 주문 가능액으로 쓰면 이미 써버린 돈으로 또 주문을 냅니다.
    """
    def fetch():
        res = buyable_cash(CASH_PROBE_CODE)
        if not res.get("ok"):
            return res
        return {"ok": True, "available_cash": res.get("available_cash", 0.0)}

    return _balance_cached("orderable_cash", fetch)


def account_snapshot() -> dict:
    """계좌 요약 한 장 — 화면·엔진이 쓰는 숫자를 전부 증권사 값으로.

    돌려주는 값 (원)
        total_asset       결제기준 총자산 (자산현황 TR — 증권사 앱 '총자산' 화면)
        settled_asset     체결기준 총자산 — **해외분까지만.** 국내 보유분은
                          호출부가 더해야 합니다 (engine/broker.py 참고)
        deposit           원화 예수금 (미결제 매수대금 포함 — 아직 나가지 않은 돈)
        foreign_cash      외화 예수금 (원화 환산)
        cash_total        총현금 = 원화 + 외화 예수금
        settled_deposit   체결기준 예수금 = 총현금 − 미결제매수 + 미결제매도
        available_cash    주문가능현금 — 지금 진짜 쓸 수 있는 돈
        withdrawable      출금가능금액
        purchase_amount   매입금액 (체결기준)
        eval_amount       평가금액 (체결기준)
        unrealized_pnl    평가손익
        unsettled_buy     미결제 매수대금 (예수금에 아직 남아 있는 금액)
        unsettled_sell    미결제 매도대금
        fx_rate           증권사 기준환율 (USD/KRW)
        sources           어느 TR 에서 온 값인지 (진단용)
        errors            실패한 조회 (전부 실패해도 ok=False 로만 알립니다)

    **한 조각이라도 실패하면 ok=False 입니다.** 반쪽짜리 요약을 정상인 척
    돌려주면 호출부가 "예수금 0원"을 진짜로 믿고 매매를 멈추거나, 반대로
    주문가능액을 못 읽은 채 예수금 전액을 쓸 수 있다고 착각합니다.
    """
    assets = account_assets()
    present = overseas_present()
    cash = orderable_cash()

    sources, errors = [], []
    for name, res in (("자산현황", assets), ("체결기준잔고", present),
                      ("주문가능현금", cash)):
        if res.get("ok"):
            sources.append(name)
        else:
            errors.append(f"{name}: {res.get('error') or '조회 실패'}")

    total = assets.get("total_asset") if assets.get("ok") else None
    if not total:
        total = present.get("total_asset") if present.get("ok") else None

    deposit = (assets.get("deposit") if assets.get("ok")
               else present.get("deposit") if present.get("ok") else None)

    # 체결기준 예수금 — 미결제분이 전부 결제되고 나면 남을 현금.
    #
    # 예수금에는 아직 빠져나가지 않은 매수대금이 들어 있고, 이미 판 대금은 아직
    # 들어와 있지 않습니다. 그래서 예수금을 그대로 '총예수금'으로 띄우면
    # 총자산보다 큰 예수금이 화면에 뜹니다 (실측 2026-08-29:
    # 예수금 1,007,247 · 체결기준 총자산 531,699). 미결제 매수를 빼고 미결제
    # 매도를 더하면 체결기준 총자산과 아귀가 맞습니다.
    #
    # **원화 예수금이 아니라 총현금(원화+외화)에서 출발합니다.** 외화 예수금을
    # 빼먹으면 달러를 들고 있는 계좌에서 그만큼 예수금이 비고, 화면의
    # '총자산 = 총예수금 + 평가금액' 이 어긋납니다 (실측 1002****-01:
    # 원화 158,968 · 외화 271,753 — 외화를 빼면 27만원이 사라집니다).
    unsettled_buy = present.get("unsettled_buy", 0.0) if present.get("ok") else 0.0
    unsettled_sell = present.get("unsettled_sell", 0.0) if present.get("ok") else 0.0
    cash_total = present.get("cash_total", 0.0) if present.get("ok") else 0.0
    settled_deposit = (cash_total or deposit or 0.0) - unsettled_buy + unsettled_sell

    return {
        "ok": total is not None and cash.get("ok") and present.get("ok"),
        "total_asset": total or 0.0,
        # 체결기준 순자산 — 결제 전 매매까지 반영한, 지금 실제로 가진 돈
        "settled_asset": present.get("total_asset", 0.0) if present.get("ok") else 0.0,
        "deposit": deposit or 0.0,                  # 원화 예수금 (미결제 포함)
        "foreign_cash": present.get("foreign_cash", 0.0) if present.get("ok") else 0.0,
        "cash_total": cash_total,                   # 원화 + 외화 예수금
        "settled_deposit": settled_deposit,
        "available_cash": cash.get("available_cash", 0.0) if cash.get("ok") else 0.0,
        "withdrawable": present.get("withdrawable", 0.0) if present.get("ok") else 0.0,
        "purchase_amount": present.get("purchase_amount", 0.0) if present.get("ok") else 0.0,
        "eval_amount": present.get("eval_amount", 0.0) if present.get("ok") else 0.0,
        "unrealized_pnl": present.get("pnl", 0.0) if present.get("ok") else 0.0,
        "unsettled_buy": unsettled_buy,
        "unsettled_sell": unsettled_sell,
        "fx_rate": present.get("fx_rate", 0.0) if present.get("ok") else 0.0,
        # 결제완료분만 센 평가금액 (앱 '총자산' 화면의 평가금액과 같은 값)
        "settled_eval_amount": assets.get("eval_amount", 0.0) if assets.get("ok") else 0.0,
        "sources": sources,
        "errors": errors,
    }


def _first(row: dict, *keys, default=None):
    """KIS 응답은 필드명이 TR 마다 조금씩 다릅니다. 후보를 순서대로 찾습니다."""
    for key in keys:
        if row.get(key) not in (None, ""):
            return row[key]
    return default


def stock_executions(order_id: str = "", days_back: int = 0,
                     symbol: str = "") -> dict:
    """주식 일별 주문체결 조회 — **주문이 실제로 얼마나 체결됐는지** 확인합니다.

    접수 응답(ODNO)만 보고 "샀다"고 처리하면 미체결·부분체결이 그대로 누락되어
    내부 포지션과 실계좌가 어긋납니다. 자동매매에서 가장 흔한 사고입니다.

    반환: {ok, orders: [{broker_order_id, code, name, side, order_qty,
                        filled_qty, remain_qty, avg_price, order_price,
                        cancelled, time}]}
    """
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}

    cano, prdt = acc
    day = (datetime.now() - timedelta(days=max(days_back, 0))).strftime("%Y%m%d")
    res = _get(_op("stock_executions"), EXECUTIONS_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "INQR_STRT_DT": day, "INQR_END_DT": datetime.now().strftime("%Y%m%d"),
        "SLL_BUY_DVSN_CD": "00",          # 전체
        "INQR_DVSN": "00",                # 역순
        # 종목으로 좁힙니다 — 한 페이지 분량을 넘기면 방금 낸 주문이 목록
        # 밖으로 밀려 "찾지 못했습니다"가 됩니다 (해외 경로에서 실제로 겪은
        # 사고입니다 — overseas_executions 주석 참고).
        "PDNO": symbol or "",
        "CCLD_DVSN": "00",                # 전체(체결+미체결)
        "ORD_GNO_BRNO": "",
        "ODNO": order_id or "",
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
    })
    if not res.get("ok"):
        return res

    orders = []
    for row in (res.get("output1") or []):
        odno = str(_first(row, "odno", "ODNO", default="")).lstrip("0")
        order_qty = _num(_first(row, "ord_qty", "tot_ord_qty"))
        filled = _num(_first(row, "tot_ccld_qty", "ccld_qty"))
        remain = _num(_first(row, "rmn_qty"), default=max(order_qty - filled, 0.0))
        amount = _num(_first(row, "tot_ccld_amt", "ccld_amt"))
        avg = _num(_first(row, "avg_prvs", "ccld_avg_unpr"))
        if not avg and filled:
            avg = amount / filled if filled else 0.0
        orders.append({
            "broker_order_id": odno,
            "code": _first(row, "pdno", default=""),
            "name": _first(row, "prdt_name", default=""),
            "side": "buy" if str(_first(row, "sll_buy_dvsn_cd", default="")) == "02" else "sell",
            "order_qty": order_qty,
            "filled_qty": filled,
            "remain_qty": remain,
            "avg_price": avg,
            "order_price": _num(_first(row, "ord_unpr")),
            "cancelled": str(_first(row, "cncl_yn", default="N")).upper() == "Y",
            "time": _first(row, "ord_tmd", default=""),
        })
    return {"ok": True, "orders": orders, "raw": res.get("raw")}


def stock_open_orders() -> dict:
    """정정·취소 가능한(=아직 안 끝난) 주문 목록."""
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc
    res = _get(_op("stock_open_orders"), OPEN_ORDERS_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
        "INQR_DVSN_1": "0", "INQR_DVSN_2": "0",
    })
    if not res.get("ok"):
        return res
    orders = []
    for row in (res.get("output") or res.get("output1") or []):
        orders.append({
            "broker_order_id": str(_first(row, "odno", default="")).lstrip("0"),
            "org_no": _first(row, "ord_gno_brno", default=""),
            "code": _first(row, "pdno", default=""),
            "name": _first(row, "prdt_name", default=""),
            "side": "buy" if str(_first(row, "sll_buy_dvsn_cd", default="")) == "02" else "sell",
            "order_qty": _num(_first(row, "ord_qty")),
            "remain_qty": _num(_first(row, "psbl_qty", "rmn_qty")),
            "order_price": _num(_first(row, "ord_unpr")),
            "time": _first(row, "ord_tmd", default=""),
        })
    return {"ok": True, "orders": orders, "raw": res.get("raw")}


def deriv_executions(order_id: str = "") -> dict:
    """선물옵션 주문체결 조회."""
    acc = account("deriv")
    if not acc:
        return {"ok": False, "error": "선물옵션 계좌번호가 설정되지 않았습니다."}
    cano, prdt = acc
    today = datetime.now().strftime("%Y%m%d")
    res = _get(_op("deriv_executions"), DERIV_EXECUTIONS_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "STRT_ORD_DT": today, "END_ORD_DT": today,
        "SLL_BUY_DVSN_CD": "00", "CCLD_NCCS_DVSN": "00",
        "SORT_SQN": "DS", "STRT_ODNO": order_id or "",
        "PDNO": "", "MKET_ID_CD": "",
        "CTX_AREA_FK200": "", "CTX_AREA_NK200": "",
    })
    if not res.get("ok"):
        return res
    orders = []
    for row in (res.get("output1") or []):
        order_qty = _num(_first(row, "ord_qty"))
        filled = _num(_first(row, "tot_ccld_qty", "ccld_qty"))
        orders.append({
            "broker_order_id": str(_first(row, "odno", default="")).lstrip("0"),
            "code": _first(row, "shtn_pdno", "pdno", default=""),
            "name": _first(row, "prdt_name", default=""),
            "side": "buy" if str(_first(row, "sll_buy_dvsn_cd", default="")) == "02" else "sell",
            "order_qty": order_qty,
            "filled_qty": filled,
            "remain_qty": _num(_first(row, "rmn_qty"), default=max(order_qty - filled, 0.0)),
            "avg_price": _num(_first(row, "avg_idx", "ccld_avg_unpr1", "ord_idx")),
            "cancelled": str(_first(row, "cncl_yn", default="N")).upper() == "Y",
            "time": _first(row, "ord_tmd", default=""),
        })
    return {"ok": True, "orders": orders, "raw": res.get("raw")}


def cancel_deriv_order(broker_order_id: str, quantity: int = 0,
                       all_remaining: bool = True) -> dict:
    if not live_enabled():
        return {"ok": False, "error": "실전 주문이 잠겨 있습니다."}
    acc = account("deriv")
    if not acc:
        return {"ok": False, "error": "선물옵션 계좌번호가 설정되지 않았습니다."}
    cano, prdt = acc
    body = {
        "ORD_PRCS_DVSN_CD": "02",
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "RVSE_CNCL_DVSN_CD": "02",           # 02 = 취소
        "ORGN_ODNO": broker_order_id,
        "ORD_QTY": "0" if all_remaining else str(int(quantity)),
        "UNIT_PRICE": "0",
        "NMPR_TYPE_CD": "", "KRX_NMPR_CNDT_CD": "",
        "RMN_QTY_YN": "Y" if all_remaining else "N",
        "ORD_DVSN_CD": "01",
    }
    return _post(_op("deriv_cancel"), DERIV_CANCEL_PATH, body)


# ---------------------------------------------------------------------------
# 국내 선물 / 옵션
# ---------------------------------------------------------------------------

def place_deriv_order(code: str, side: str, quantity: int,
                      price: float = 0, order_type: str = "limit") -> dict:
    """선물·옵션 주문.

    side 는 매수/매도 그대로입니다. '신규/청산'은 계좌 포지션에 따라 브로커가
    상계 처리하므로, 숏 진입도 side='sell' 로 냅니다.

    선물옵션은 시장가 주문이 제한되는 경우가 많아 기본이 지정가입니다.
    """
    if not live_enabled():
        return {"ok": False, "error": "실전 주문이 잠겨 있습니다 (KIS_LIVE_TRADING=1 필요)."}
    acc = account("deriv")
    if not acc:
        return {"ok": False, "error": "선물옵션 계좌번호가 설정되지 않았습니다."}

    quantity = int(quantity)
    if quantity <= 0:
        return {"ok": False, "error": "주문 수량이 0입니다."}

    cano, prdt = acc
    body = {
        "ORD_PRCS_DVSN_CD": "02",                    # 02 = 주문전송
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "SLL_BUY_DVSN_CD": "02" if side == "buy" else "01",   # 01 매도 / 02 매수
        "SHTN_PDNO": code,
        "ORD_QTY": str(quantity),
        "UNIT_PRICE": "0" if order_type == "market" else f"{float(price):g}",
        "NMPR_TYPE_CD": "",
        "KRX_NMPR_CNDT_CD": "",
        "ORD_DVSN_CD": "02" if order_type == "market" else "01",  # 01 지정가 / 02 시장가
    }
    res = _post(_op("deriv_order"), DERIV_ORDER_PATH, body)
    if not res.get("ok"):
        return res
    out = res.get("output") or {}
    return {
        "ok": True,
        "broker_order_id": out.get("ODNO", ""),
        "order_time": out.get("ORD_TMD", ""),
        "message": res.get("message", ""),
        "raw": res.get("raw"),
    }


def deriv_balance() -> dict:
    """선물옵션 잔고 — 미결제약정과 증거금 (5초 캐시)."""
    return _balance_cached("deriv", _deriv_balance_fetch)


def _deriv_balance_fetch() -> dict:
    acc = account("deriv")
    if not acc:
        return {"ok": False, "error": "선물옵션 계좌번호가 설정되지 않았습니다."}
    cano, prdt = acc
    res = _get(_op("deriv_balance"), DERIV_BALANCE_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "MGNA_DVSN": "01", "EXCC_STAT_CD": "1",
        "CTX_AREA_FK200": "", "CTX_AREA_NK200": "",
    })
    if not res.get("ok"):
        return res

    positions = []
    for row in (res.get("output1") or []):
        qty = _num(row.get("cblc_qty") or row.get("hldg_qty"))
        if qty <= 0:
            continue
        sell_buy = str(row.get("sll_buy_dvsn_cd", "02"))
        positions.append({
            "code": row.get("shtn_pdno") or row.get("pdno", ""),
            "name": row.get("prdt_name", ""),
            "side": "long" if sell_buy == "02" else "short",
            "quantity": qty,
            "avg_price": _num(row.get("pchs_avg_pric") or row.get("ccld_avg_unpr1")),
            "current_price": _num(row.get("idx_clpr") or row.get("prpr")),
            "pnl": _num(row.get("evlu_pfls_amt")),
        })

    summary = res.get("output2") or {}
    if isinstance(summary, list):
        summary = summary[0] if summary else {}
    return {
        "ok": True,
        "positions": positions,
        "cash": _num(summary.get("dnca_cash")),
        "available_cash": _num(summary.get("ord_psbl_cash") or summary.get("dnca_cash")),
        "margin_used": _num(summary.get("tot_ccld_amt") or summary.get("mgna_tota")),
        "raw": res.get("raw"),
    }


def deriv_quote(code: str) -> dict | None:
    """선물·옵션 현재가. 파생 시세는 KIS 외에 무료 경로가 없습니다."""
    market_div = "F" if code.startswith("1") else "O"
    res = _get("deriv_quote", DERIV_QUOTE_PATH, {
        "FID_COND_MRKT_DIV_CODE": market_div,
        "FID_INPUT_ISCD": code,
    }, timeout=10)
    if not res.get("ok"):
        return None
    out = res.get("output") or {}
    price = _num(out.get("futs_prpr") or out.get("optn_prpr") or out.get("prpr"), 0.0)
    if price <= 0:
        return None
    prev = _num(out.get("futs_prdy_clpr") or out.get("optn_prdy_clpr")
                or out.get("prdy_clpr"), 0.0) or None
    return {
        "price": price,
        "prev_close": prev,
        "change_rate": _num(out.get("prdy_ctrt"), 0.0),
        "open": _num(out.get("futs_oprc") or out.get("optn_oprc")) or None,
        "high": _num(out.get("futs_hgpr") or out.get("optn_hgpr")) or None,
        "low": _num(out.get("futs_lwpr") or out.get("optn_lwpr")) or None,
        "volume": _num(out.get("acml_vol")),
        "open_interest": _num(out.get("hts_otst_stpl_qty")),
        "source": "한국투자증권 KIS (파생)",
    }


def deriv_daily_chart(code: str, days: int = 120) -> pd.DataFrame:
    """선물·옵션 일봉 (기술적 지표 계산용)."""
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.7) + 20)
    res = _get("deriv_chart", DERIV_CHART_PATH, {
        "FID_COND_MRKT_DIV_CODE": "F" if code.startswith("1") else "O",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
        "FID_INPUT_DATE_2": end.strftime("%Y%m%d"),
        "FID_PERIOD_DIV_CODE": "D",
    })
    rows = res.get("output2") if res.get("ok") else None
    parsed = []
    for r in (rows or []):
        raw_date = str(r.get("stck_bsop_date") or r.get("bsop_date") or "")
        close = _num(r.get("futs_prpr") or r.get("optn_prpr") or r.get("stck_clpr"), 0.0)
        if len(raw_date) != 8 or close <= 0:
            continue
        parsed.append({
            "date": datetime.strptime(raw_date, "%Y%m%d"),
            "open": _num(r.get("futs_oprc") or r.get("optn_oprc") or r.get("stck_oprc"), close),
            "high": _num(r.get("futs_hgpr") or r.get("optn_hgpr") or r.get("stck_hgpr"), close),
            "low": _num(r.get("futs_lwpr") or r.get("optn_lwpr") or r.get("stck_lwpr"), close),
            "close": close,
            "volume": _num(r.get("acml_vol")),
        })
    if not parsed:
        return pd.DataFrame()
    df = pd.DataFrame(parsed).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="last")]


# ---------------------------------------------------------------------------
# 해외 주식 (미국)
# ---------------------------------------------------------------------------

# Yahoo 거래소 코드 → KIS 해외거래소 코드.
# 이 매핑이 틀리면 주문이 "해당종목정보가 없습니다" 로 거부됩니다 —
# KIS 는 티커만으로 거래소를 추론해 주지 않습니다.
_KIS_EXCHANGE_BY_YAHOO = {
    "NMS": "NASD", "NGM": "NASD", "NCM": "NASD", "NAS": "NASD", "NASDAQ": "NASD",
    "NYQ": "NYSE", "NYSE": "NYSE",
    "ASE": "AMEX", "AMEX": "AMEX",
    "PCX": "AMEX",          # NYSE Arca (ETF 다수) 는 KIS 에서 AMEX 로 접수됩니다
    "BTS": "AMEX", "BATS": "AMEX",
}


def overseas_exchange_code(yahoo_exchange: str, default: str = "NASD") -> str:
    """Yahoo 상장 거래소 표기를 KIS 주문용 코드로. 모르면 기본값(나스닥)."""
    return _KIS_EXCHANGE_BY_YAHOO.get(str(yahoo_exchange or "").upper(), default)


def place_overseas_order(ticker: str, side: str, quantity: float,
                         price: float, exchange: str = "NASD") -> dict:
    """미국 주식·ETF 주문 — 해외는 시장가가 없어 **지정가만** 지원합니다.

    exchange: NASD(나스닥) / NYSE / AMEX
    """
    if not live_enabled():
        return {"ok": False, "error": "실전 주문이 잠겨 있습니다 (KIS_LIVE_TRADING=1 필요)."}
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    if not price or price <= 0:
        return {"ok": False, "error": "해외 주식은 지정가가 필요합니다."}

    cano, prdt = acc
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "OVRS_EXCG_CD": exchange,
        "PDNO": ticker,
        "ORD_QTY": str(int(quantity)),
        "OVRS_ORD_UNPR": f"{float(price):.2f}",
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": "00",                     # 지정가
    }
    op = _op("overseas_buy") if side == "buy" else _op("overseas_sell")
    res = _post(op, OVERSEAS_ORDER_PATH, body)
    if not res.get("ok"):
        return res
    out = res.get("output") or {}
    return {"ok": True, "broker_order_id": out.get("ODNO", ""),
            "message": res.get("message", ""), "raw": res.get("raw")}


def cancel_overseas_order(ticker: str, order_id: str, quantity: float,
                          exchange: str = "NASD") -> dict:
    """미국 주식 미체결 주문 취소.

    국내 취소(cancel_stock_order)와 API 가 완전히 다릅니다 — 미국 주문을 국내
    경로로 보내면 국내 미체결 목록에서 못 찾아 "이미 처리된 주문"으로만 끝나고,
    그 주문은 걷을 방법 없이 미결제로 남습니다 (실측: DBGI 청산 주문이 이렇게
    나흘을 pending 으로 버텼습니다).
    """
    if not live_enabled():
        return {"ok": False, "error": "실전 주문이 잠겨 있습니다 (KIS_LIVE_TRADING=1 필요)."}
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "OVRS_EXCG_CD": exchange,
        "PDNO": ticker,
        "ORGN_ODNO": str(order_id),
        "RVSE_CNCL_DVSN_CD": "02",            # 02 = 취소 (01 은 정정)
        "ORD_QTY": str(int(quantity)),
        "OVRS_ORD_UNPR": "0",                 # 취소는 가격이 없습니다
        "ORD_SVR_DVSN_CD": "0",
    }
    res = _post(_op("overseas_cancel"), OVERSEAS_CANCEL_PATH, body)
    if not res.get("ok"):
        return res
    out = res.get("output") or {}
    return {"ok": True, "broker_order_id": out.get("ODNO", ""),
            "message": res.get("message", ""), "raw": res.get("raw")}


# ---------------------------------------------------------------------------
# 미국 주간거래 (데이마켓)
# ---------------------------------------------------------------------------
#
# 한국 낮에 미국 주식을 사고파는 구간입니다. ET 20:00~03:30
# = KST 09:00~16:30(서머타임) / 10:00~17:30(표준시), 월~금.
#
# **정규장과 같은 함수를 쓰면 안 됩니다.** 접수 경로가 완전히 다릅니다:
#   · 주문 경로   /trading/daytime-order        (정규장은 /trading/order)
#   · 정정취소    /trading/daytime-order-rvsecncl
#   · 시세        /quotations/price-detail + 주간 전용 거래소코드
#
# 성격상 알아야 할 것
#   · 정규 거래소가 아니라 FINRA 승인 ATS 에서 SOR 로 체결됩니다. 호가가 얇고
#     화면 가격과 다르게 체결되거나 아예 안 걸릴 수 있습니다.
#   · **지정가만 됩니다** (시장가·예약주문 불가).
#   · 미체결 잔량은 주간거래 종료 시 **증권사가 자동 취소**합니다. 그래서 우리가
#     따로 걷지 않아도 정규장으로 넘어가 엉뚱한 가격에 체결되지는 않습니다.
#     (그래도 종료 전에 우리 손으로 거두는 편이 상태 정합에 좋습니다)
#   · 계좌에 "미국주식 주간거래" 서비스 신청이 되어 있어야 접수됩니다.

# 주문용 거래소 코드와 **시세용 거래소 코드가 다릅니다.**
#   주문 NASD / NYSE / AMEX   (정규장과 동일)
#   시세 BAQ  / BAY  / BAA    (주간거래 전용 — 정규장의 NAS/NYS/AMS 와 또 다름)
# 이 둘을 섞으면 "해당종목정보가 없습니다"로 조용히 실패합니다.
_DAY_QUOTE_EXCD = {"NASD": "BAQ", "NYSE": "BAY", "AMEX": "BAA"}


def overseas_day_quote_excd(exchange: str, default: str = "BAQ") -> str:
    """주문용 거래소코드(NASD/NYSE/AMEX) -> 주간거래 시세용 코드(BAQ/BAY/BAA)."""
    return _DAY_QUOTE_EXCD.get(str(exchange or "").upper(), default)


def _day_trading_blocked() -> str:
    """주간거래를 낼 수 없는 상태면 그 사유를, 낼 수 있으면 빈 문자열."""
    if not live_enabled():
        return "실전 주문이 잠겨 있습니다 (KIS_LIVE_TRADING=1 필요)."
    if kis_client.is_mock():
        # 모의투자 서버에는 주간거래 tr_id 자체가 없습니다. 정규장 코드로
        # 대신 보내면 **모의 계좌에서 정규장 주문이 나갑니다** — 조용히 다른
        # 일을 하느니 여기서 막습니다.
        return "미국 주간거래는 모의투자를 지원하지 않습니다 (실전 계좌 전용)."
    if not account():
        return "KIS_ACCOUNT 가 설정되지 않았습니다."
    return ""


def place_overseas_day_order(ticker: str, side: str, quantity: float,
                             price: float, exchange: str = "NASD") -> dict:
    """미국 주간거래 주문 — 지정가만 가능합니다.

    exchange 는 **주문용 코드**(NASD/NYSE/AMEX)를 그대로 받습니다.
    """
    blocked = _day_trading_blocked()
    if blocked:
        return {"ok": False, "error": blocked}
    if not price or price <= 0:
        return {"ok": False, "error": "주간거래는 지정가가 필요합니다 (시장가 불가)."}
    quantity = int(quantity)
    if quantity <= 0:
        return {"ok": False, "error": "주문 수량이 0입니다."}

    cano, prdt = account()
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "OVRS_EXCG_CD": exchange,
        "PDNO": ticker,
        "ORD_QTY": str(quantity),
        "OVRS_ORD_UNPR": f"{float(price):.2f}",
        "CTAC_TLNO": "",
        "MGCO_APTM_ODNO": "",
        "ORD_SVR_DVSN_CD": "0",
        "ORD_DVSN": "00",                     # 주간거래는 지정가(00)만
    }
    op = "overseas_day_buy_real" if side == "buy" else "overseas_day_sell_real"
    res = _post(op, OVERSEAS_DAY_ORDER_PATH, body)
    if not res.get("ok"):
        return res
    out = res.get("output") or {}
    return {"ok": True, "broker_order_id": out.get("ODNO", ""),
            "order_time": out.get("ORD_TMD", ""),
            "message": res.get("message", ""), "raw": res.get("raw")}


def cancel_overseas_day_order(ticker: str, order_id: str, quantity: float,
                              exchange: str = "NASD") -> dict:
    """주간거래 미체결 취소.

    정규장 취소(cancel_overseas_order)로 보내면 주간거래 주문번호를 찾지 못해
    실패합니다 — 접수 창구가 다르기 때문입니다.
    """
    blocked = _day_trading_blocked()
    if blocked:
        return {"ok": False, "error": blocked}

    cano, prdt = account()
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": prdt,
        "OVRS_EXCG_CD": exchange,
        "PDNO": ticker,
        "ORGN_ODNO": str(order_id),
        "RVSE_CNCL_DVSN_CD": "02",            # 02 = 취소 (01 은 정정)
        "ORD_QTY": str(int(quantity)),
        "OVRS_ORD_UNPR": "0",
        "CTAC_TLNO": "",
        "MGCO_APTM_ODNO": "",
        "ORD_SVR_DVSN_CD": "0",
    }
    res = _post("overseas_day_cancel_real", OVERSEAS_DAY_CANCEL_PATH, body)
    if not res.get("ok"):
        return res
    out = res.get("output") or {}
    return {"ok": True, "broker_order_id": out.get("ODNO", ""),
            "message": res.get("message", ""), "raw": res.get("raw")}


# 주간거래 현재가 응답의 필드명은 KIS 문서 공개분으로 확정하지 못했습니다.
# 후보를 순서대로 보고, 하나도 못 맞히면 **받은 키 목록을 오류에 실어** 한 번의
# 실행으로 이름을 확정할 수 있게 합니다 (조용히 0원을 돌려주지 않습니다).
_DAY_PRICE_KEYS = ("last", "ovrs_prpr", "prpr", "stck_prpr")
_DAY_BASE_KEYS = ("base", "ovrs_prdy_clpr", "prdy_clpr", "sdpr")
_DAY_VOL_KEYS = ("tvol", "acml_vol", "vol")
_DAY_RATE_KEYS = ("rate", "prdy_ctrt", "diff_rate")


def _first_num(out: dict, keys: tuple, default: float = 0.0) -> float:
    for k in keys:
        if out.get(k) not in (None, "", "0.0000"):
            value = _num(out.get(k), 0.0)
            if value:
                return value
    return default


def overseas_day_quote(ticker: str, exchange: str = "NASD") -> dict | None:
    """미국 주간거래 현재가.

    Yahoo 로는 이 구간을 못 받습니다 — Yahoo 의 확장시간(includePrePost)은
    ET 04:00~20:00 까지고, 주간거래는 ET 20:00 이후 ATS 물량이라 아예 안 나옵니다.
    그래서 이 구간만은 KIS 시세를 씁니다.
    """
    excd = overseas_day_quote_excd(exchange)
    res = _get("overseas_day_quote", OVERSEAS_DAY_QUOTE_PATH, {
        "AUTH": "",
        "EXCD": excd,
        "SYMB": str(ticker or "").upper(),
    }, timeout=10)
    if not res.get("ok"):
        return None

    out = res.get("output") or {}
    price = _first_num(out, _DAY_PRICE_KEYS)
    if price <= 0:
        # 거래가 아예 없어 0 인 것과 필드명을 못 맞힌 것을 구분해야 합니다.
        return {"price": None, "excd": excd,
                "error": "주간거래 현재가 필드를 찾지 못했습니다 "
                         f"(응답 키: {sorted(out.keys())[:20]})"}

    prev = _first_num(out, _DAY_BASE_KEYS) or None
    rate = _first_num(out, _DAY_RATE_KEYS, default=None)
    if rate is None and prev:
        rate = (price - prev) / prev * 100
    return {
        "price": price,
        "prev_close": prev,
        "change_rate": round(rate, 2) if rate is not None else None,
        "volume": _first_num(out, _DAY_VOL_KEYS),
        "excd": excd,
        "source": "한국투자증권 KIS (주간거래)",
    }


def overseas_executions(days_back: int = 3, order_id: str = "",
                        symbol: str = "") -> dict:
    """해외주식 주문·체결 내역.

    이게 없으면 미국 주문은 접수 뒤 결말을 확인할 방법이 없습니다. 국내 체결
    내역에는 해외 주문이 안 들어 있어서, 계속 "주문 상태를 확인하지 못했습니다"
    가 반복되고 그 종목은 미결제로 묶여 새 주문도 못 냅니다.

    **symbol 을 꼭 넘기세요.** 이 API 는 한 번에 20건만 돌려주고 연속조회
    (CTX_AREA_NK200) 는 이 계좌에서 rt_cd=7 로 거부됩니다. 하루에 수십 건을
    내는 자동매매에서는 조회 창 안의 주문이 20건을 훌쩍 넘으므로, 종목으로
    좁히지 않으면 방금 낸 주문이 목록 밖으로 밀려나 "주문 내역에서 찾지
    못했습니다" 가 됩니다 — 실제로는 **전량 체결된** 주문인데도요.

    실측 2026-08-20: 8/19 주문 9건이 전부 체결됐는데 20건 창 밖으로 밀려
    16~23시간째 pending 으로 남아 있었습니다. 체결 원장에 기록되지 않았고,
    _ORDER_LOST_SEC(24시간)이 지나면 결말 미상으로 버려질 상태였습니다.

    SORT_SQN 도 "DS"(정순 — 가장 오래된 주문부터)에서 "AS"(역순)로 바꿉니다.
    20건 창은 최신 주문부터 채워야 방금 낸 주문이 항상 들어옵니다.
    """
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc
    start = (datetime.now() - timedelta(days=max(days_back, 0))).strftime("%Y%m%d")
    res = _get(_op("overseas_executions"), OVERSEAS_EXECUTIONS_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "PDNO": symbol or "", "ORD_STRT_DT": start,
        "ORD_END_DT": datetime.now().strftime("%Y%m%d"),
        "SLL_BUY_DVSN": "00",             # 전체
        "CCLD_NCCS_DVSN": "00",           # 체결+미체결
        "OVRS_EXCG_CD": "%",              # 전체 거래소
        "SORT_SQN": "AS", "ORD_DT": "", "ORD_GNO_BRNO": "",
        "ODNO": order_id or "",
        "CTX_AREA_FK200": "", "CTX_AREA_NK200": "",
    })
    if not res.get("ok"):
        return res

    orders = []
    for row in (res.get("output") or res.get("output1") or []):
        odno = str(_first(row, "odno", "ODNO", default="")).lstrip("0")
        order_qty = _num(_first(row, "ft_ord_qty", "ord_qty"))
        filled = _num(_first(row, "ft_ccld_qty", "ccld_qty"))
        remain = _num(_first(row, "nccs_qty"), default=max(order_qty - filled, 0.0))
        avg = _num(_first(row, "ft_ccld_unpr3", "ccld_unpr", "avg_unpr"))
        state = str(_first(row, "prcs_stat_name", "rvse_cncl_dvsn_name", default=""))
        orders.append({
            "broker_order_id": odno,
            "symbol": str(_first(row, "pdno", "ovrs_pdno", default="")),
            "order_qty": order_qty, "filled_qty": filled, "remain_qty": remain,
            "avg_price": avg,
            "cancelled": ("취소" in state),
            "state": state,
        })
    return {"ok": True, "orders": orders, "raw": res.get("raw")}


def overseas_buyable(ticker: str, exchange: str, price: float) -> dict:
    """이 종목을 지금 몇 주까지 살 수 있는지 KIS 에 먼저 물어봅니다.

    이걸 안 물어보면 살 수 없는 주문을 실제로 내보고 거부당한 뒤에야 압니다.
    실계좌 원장에 거부 기록이 쌓이고, 원인(외화 없음·미신청 서비스)은
    KIS 문구만으로는 알기 어렵습니다.

    돌려주는 값
        ok              조회 성공 여부 (실패 시 주문을 막지 않습니다 — 아래 주의)
        cash_usd        환전 없이 지금 쓸 수 있는 외화 (ord_psbl_frcr_amt)
        after_fx_usd    원화까지 환전했을 때 쓸 수 있는 금액 (echm_af_ord_psbl_amt)
        quantity        위 금액으로 살 수 있는 수량 (KIS 계산값)
        exchange_rate   적용 환율

    **조회 실패는 '살 수 없음'이 아닙니다.** 조회가 안 된다고 주문을 막으면
    KIS 점검·일시 오류에 매매가 통째로 멈춥니다. 판단은 호출부에 맡깁니다.
    """
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc

    def fetch():
        res = _get(_op("overseas_buyable"), OVERSEAS_BUYABLE_PATH, {
            "CANO": cano, "ACNT_PRDT_CD": prdt,
            "OVRS_EXCG_CD": exchange, "ITEM_CD": ticker,
            "OVRS_ORD_UNPR": f"{float(price):.2f}",
        })
        if not res.get("ok"):
            return res
        out = res.get("output") or {}
        if isinstance(out, list):
            out = out[0] if out else {}
        return {
            "ok": True,
            "cash_usd": _num(out.get("ord_psbl_frcr_amt")),
            "after_fx_usd": _num(out.get("echm_af_ord_psbl_amt")),
            "quantity": _num(out.get("max_ord_psbl_qty")
                             or out.get("ovrs_max_ord_psbl_qty")),
            "exchange_rate": _num(out.get("exrt")),
            "currency": out.get("tr_crcy_cd") or "USD",
            "raw": out,
        }

    return _balance_cached(f"psamount:{exchange}:{ticker}:{price:.2f}", fetch)


def overseas_balance_all() -> dict:
    """미국 보유분 전체.

    실전 서버의 NASD 코드는 **미국 전체**(나스닥+뉴욕+아멕스)를 돌려줍니다 —
    이 계좌로 실측했습니다(NASD 조회에 NYSE 종목 4개가 그대로 왔습니다).
    그래서 실전은 한 번만 조회합니다. 예전처럼 세 거래소를 다 부르면 이것만
    5.8초였고(호출당 ~1.9초), 계좌 화면이 느린 첫 번째 원인이었습니다.

    모의투자 서버는 거래소를 정확히 지정해야 해서 세 번 그대로 부릅니다.

    **조회가 하나라도 실패하면 ok=False 입니다.** 일부만 성공한 결과를
    "전체 보유분"으로 돌려주면, 엔진이 못 본 포지션을 청산 대상에서
    빼버려 손절 없는 포지션이 방치됩니다.
    """
    def fetch():
        # 같은 종목이 여러 조회에 겹쳐 나오면 한 번만 남깁니다
        merged, seen, failures = [], set(), []

        def collect(code) -> bool:
            res = overseas_balance(code)
            if not res.get("ok"):
                failures.append(f"{code}: {res.get('error') or '조회 실패'}")
                return False
            for p in (res.get("positions") or []):
                key = str(p.get("code") or "")
                if key and key not in seen:
                    seen.add(key)
                    merged.append(p)
            return True

        if kis_client.is_mock():
            exchanges = ("NASD", "NYSE", "AMEX")
        else:
            exchanges = ("NASD",)
        for code in exchanges:
            collect(code)

        # 실전 NASD 가 "보유 없음"이면, 필터가 먹는 계좌일 가능성에 대비해
        # 나머지 거래소로 한 번 더 확인합니다. 진짜 빈 계좌에서만 도는 경로라
        # 느려질 일이 없고, 잘못된 "포지션 없음"은 손절 방치로 이어지므로
        # 이 확인은 생략하면 안 됩니다.
        if not kis_client.is_mock() and not failures and not merged:
            for code in ("NYSE", "AMEX"):
                collect(code)

        if failures:
            return {"ok": False, "positions": [],
                    "error": "해외 잔고 조회 실패 — " + " / ".join(failures)}
        return {"ok": True, "positions": merged}

    return _balance_cached("overseas", fetch)


def overseas_balance(exchange: str = "NASD", currency: str = "USD") -> dict:
    acc = account()
    if not acc:
        return {"ok": False, "error": "KIS_ACCOUNT 가 설정되지 않았습니다."}
    cano, prdt = acc
    res = _get(_op("overseas_balance"), OVERSEAS_BALANCE_PATH, {
        "CANO": cano, "ACNT_PRDT_CD": prdt,
        "OVRS_EXCG_CD": exchange, "TR_CRCY_CD": currency,
        "CTX_AREA_FK200": "", "CTX_AREA_NK200": "",
    })
    if not res.get("ok"):
        return res
    positions = []
    for row in (res.get("output1") or []):
        qty = _num(row.get("ovrs_cblc_qty"))
        if qty <= 0:
            continue
        # 매입금액·평가금액은 KIS 가 주는 값을 그대로 씁니다(수량×단가로 다시
        # 계산하면 반올림 때문에 화면 숫자가 증권사와 미세하게 어긋납니다).
        purchase = _num(row.get("frcr_pchs_amt1"))
        evaluated = _num(row.get("ovrs_stck_evlu_amt"))
        positions.append({
            "code": row.get("ovrs_pdno", ""),
            "name": row.get("ovrs_item_name", ""),
            "quantity": qty,
            "avg_price": _num(row.get("pchs_avg_pric")),
            "current_price": _num(row.get("now_pric2")),
            "purchase_amount": purchase or _num(row.get("pchs_avg_pric")) * qty,
            "eval_amount": evaluated or _num(row.get("now_pric2")) * qty,
            "pnl": _num(row.get("frcr_evlu_pfls_amt")),
            "exchange": row.get("ovrs_excg_cd", ""),
        })
    summary = res.get("output2") or {}
    if isinstance(summary, list):
        summary = summary[0] if summary else {}
    return {"ok": True, "positions": positions,
            # tot_evlu_pfls_amt 는 평가'손익'이지 평가금액이 아닙니다.
            # 평가금액이 필요하면 positions 의 수량×현재가로 계산하세요.
            "total_pnl": _num(summary.get("tot_evlu_pfls_amt")),
            "raw": res.get("raw")}


# ---------------------------------------------------------------------------
# 자체 점검
# ---------------------------------------------------------------------------

def diagnose() -> dict:
    """실제로 호출해 보고 무엇이 되는지 확인합니다 (주문은 내지 않습니다).

    "키는 넣었는데 왜 안 되지"를 화면에서 바로 알 수 있게 하려는 것입니다.
    각 항목은 {ok, detail} 이고, 실패 사유는 KIS 가 준 문구를 그대로 담습니다.
    """
    st = status()
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:200]})

    add("APP KEY / SECRET", st["keys_configured"],
        "설정됨" if st["keys_configured"] else "아테나.bat → [7] API 키 에서 등록하세요")
    add("계좌번호", st["account_configured"],
        st["account_masked"] or "KIS_ACCOUNT=12345678-01 형식으로 설정")

    if not (st["keys_configured"] and st["account_configured"]):
        return {"ok": False, "server": st["server"], "mock": st["mock"],
                "live_enabled": st["live_enabled"], "checks": checks}

    token = kis_client.access_token()
    add("접근토큰 발급", bool(token),
        "OK" if token else (kis_client.token_error() or "APP KEY/SECRET 을 확인하세요"))

    if token:
        quote = kis_client.get_quote("005930")
        add("시세 조회", bool(quote),
            f"삼성전자 {quote['close']:,.0f}원" if quote else "조회 실패")

        balance = stock_balance()
        add("주식 잔고 조회", balance.get("ok"),
            f"예수금 {balance.get('cash', 0):,.0f}원 / 보유 {len(balance.get('positions', []))}종목"
            if balance.get("ok") else balance.get("error", ""))

        # 증권사 앱과 대조할 수 있게 요약 숫자를 그대로 보여줍니다.
        # 이게 실패하면 화면 총자산은 '우리가 다시 계산한 값'으로 떨어집니다.
        snap = account_snapshot()
        add("계좌 요약(총자산)", snap.get("ok"),
            (f"총자산 {snap['total_asset']:,.0f}원 / 주문가능 {snap['available_cash']:,.0f}원"
             f" / 출금가능 {snap['withdrawable']:,.0f}원"
             + (f" · 미결제매수 {snap['unsettled_buy']:,.0f}원" if snap["unsettled_buy"] else ""))
            if snap.get("ok") else " / ".join(snap.get("errors", [])))

        executions = stock_executions()
        add("주문·체결 조회", executions.get("ok"),
            f"오늘 주문 {len(executions.get('orders', []))}건"
            if executions.get("ok") else executions.get("error", ""))

        open_orders = stock_open_orders()
        add("미체결 조회", open_orders.get("ok"),
            f"미체결 {len(open_orders.get('orders', []))}건"
            if open_orders.get("ok") else open_orders.get("error", ""))

        futures = _front_month_futures()
        deriv = deriv_quote(futures)
        if deriv:
            add("선물 시세 조회", True, f"{futures} {deriv['price']:,.2f}")
        else:
            # 왜 안 됐는지를 그대로 보여줍니다 (권한 문제와 호출 제한은 완전히 다릅니다)
            probe = _get("deriv_quote", DERIV_QUOTE_PATH, {
                "FID_COND_MRKT_DIV_CODE": "F", "FID_INPUT_ISCD": futures}, timeout=10)
            add("선물 시세 조회", False,
                f"{futures} — {probe.get('error', '조회 실패')}"
                if not probe.get("ok")
                else f"{futures} 응답은 왔으나 가격이 비어 있습니다 (선물옵션 권한 확인)")

    add("실주문 허용", st["live_enabled"],
        ("모의투자 서버" if st["mock"] else "실전 주문 허용됨")
        if st["live_enabled"] else "KIS_LIVE_TRADING=1 이 필요합니다 (조회만 가능)")

    required = ("접근토큰 발급", "주식 잔고 조회", "주문·체결 조회")
    ready = all(c["ok"] for c in checks if c["name"] in required)
    return {"ok": ready, "server": st["server"], "mock": st["mock"],
            "live_enabled": st["live_enabled"], "checks": checks}


def self_check():
    """python -m data_sources.kis_trading — 주문 없이 연결·권한만 확인합니다."""
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print("=" * 60)
    print("  KIS 주문 어댑터 자체 점검 (주문은 전송하지 않습니다)")
    print("=" * 60)

    st = status()
    print(f"\n  키 설정        : {'OK' if st['keys_configured'] else '없음'}")
    print(f"  계좌번호       : {st['account_masked'] or '없음'}")
    print(f"  서버           : {st['server']}")
    print(f"  모의투자       : {'예' if st['mock'] else '아니오 (실전)'}")
    print(f"  실주문 허용    : {'예' if st['live_enabled'] else '아니오 (잠김)'}")

    if not st["keys_configured"]:
        print("\n  KIS_APP_KEY / KIS_APP_SECRET 을 설정하세요 (아테나.bat → [7] API 키).")
        return
    if not st["account_configured"]:
        print("\n  KIS_ACCOUNT=12345678-01 형식으로 계좌번호를 설정하세요.")
        return

    result = diagnose()
    print()
    for check in result["checks"]:
        print(f"  [{'OK  ' if check['ok'] else '실패'}] {check['name']:<16} {check['detail']}")

    print(f"\n  종합: {'자동매매를 붙일 수 있습니다' if result['ok'] else '위 실패 항목을 먼저 해결하세요'}")
    print("  실제 주문은 자동매매 콘솔에서 모의투자로 먼저 확인하세요.\n")


def _front_month_futures() -> str:
    """가장 가까운 만기의 코스피200 선물 코드 (점검·기본 유니버스용)."""
    from engine.instruments import MONTH_CODES

    now = datetime.now()
    # 지수선물 만기월은 3·6·9·12월. 이번 달이 지났으면 다음 만기월로.
    quarter_months = [3, 6, 9, 12]
    year, month = now.year, now.month
    nxt = next((m for m in quarter_months if m >= month), None)
    if nxt is None:
        nxt, year = 3, year + 1
    code_char = next(k for k, v in MONTH_CODES.items() if v == nxt)
    return f"101{code_char}{year % 10}000"


if __name__ == "__main__":
    self_check()
