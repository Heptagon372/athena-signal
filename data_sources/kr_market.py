"""
한국 시장 데이터 클라이언트 (코스피 / 코스닥)
--------------------------------------------
두 개의 공개 웹 엔드포인트를 씁니다.

  토스증권  wts-info-api.tossinvest.com
      - 종목정보(이름/시장/상태)
      - 실시간 체결가·거래량  (장중이면 실시간, 장 마감 후엔 종가)
  네이버금융 m.stock.naver.com / api.stock.naver.com
      - 일봉 OHLCV (기술적 지표 계산용)
      - 시가/고가/저가/거래대금/52주 고저/PER/PBR
      - 투자자별 매매동향 (개인 / 외국인 / 기관 순매수)

둘 다 로그인 없이 조회되는 공개 엔드포인트지만 **공식 문서화된 API가 아니라**
서비스 내부용이므로, 예고 없이 구조가 바뀔 수 있습니다. 그래서 모든 파서는
필드가 없거나 형식이 바뀌면 예외 대신 None 을 돌려주고, 상위(price_provider)가
다른 소스로 자동 폴백하도록 만들었습니다.

개인 학습·실험 목적으로만 사용하고, 상업적 배포 전에는 각 서비스 이용약관을
직접 확인하세요.
"""

import re
from datetime import datetime, date

import pandas as pd

from data_sources import http_client

TOSS_INFO_URL = "https://wts-info-api.tossinvest.com/api/v2/stock-infos/A{code}"
TOSS_PRICE_URL = "https://wts-info-api.tossinvest.com/api/v2/stock-prices"
NAVER_BASIC_URL = "https://m.stock.naver.com/api/stock/{code}/basic"
NAVER_INTEGRATION_URL = "https://m.stock.naver.com/api/stock/{code}/integration"
NAVER_TREND_URL = "https://m.stock.naver.com/api/stock/{code}/trend"
NAVER_CHART_URL = "https://api.stock.naver.com/chart/domestic/item/{code}/day"

NAVER_HEADERS = {"Referer": "https://m.stock.naver.com/", "Accept": "application/json"}
TOSS_HEADERS = {"Referer": "https://tossinvest.com/", "Accept": "application/json"}


# ---------------------------------------------------------------------------
# 파싱 헬퍼
# ---------------------------------------------------------------------------

def _num(text) -> float | None:
    """'254,000' / '+2,801,281' / '46.65%' -> float. 실패하면 None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = re.sub(r"[^\d.\-+]", "", str(text))
    if cleaned in ("", "-", "+", ".", "-.", "+."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_korean_amount(text: str) -> float | None:
    """'8조 5,199억' -> 8519900000000.0 (거래대금/시총 표기 파싱)"""
    if not text:
        return None
    total, matched = 0.0, False
    for value, unit in re.findall(r"([\d,]+(?:\.\d+)?)\s*(조|억|만)?", str(text)):
        v = _num(value)
        if v is None:
            continue
        multiplier = {"조": 1e12, "억": 1e8, "만": 1e4}.get(unit, 1)
        if unit or not matched:
            total += v * multiplier
            matched = True
    return total if matched else None


# ---------------------------------------------------------------------------
# 토스증권
# ---------------------------------------------------------------------------

def toss_stock_info(code: str) -> dict | None:
    """종목 존재 여부 + 이름/시장. 없는 종목이면 result가 null 이라 None 반환."""
    data = http_client.get_json(TOSS_INFO_URL.format(code=code), headers=TOSS_HEADERS)
    if not data:
        return None
    result = data.get("result")
    if not result or not result.get("name"):
        return None
    return {
        "name": result.get("name"),
        "english_name": result.get("englishName"),
        "market_code": (result.get("market") or {}).get("code"),
        "market_name": (result.get("market") or {}).get("displayName"),
        "status": result.get("status"),
    }


def toss_realtime_price(code: str) -> dict | None:
    """토스증권 실시간 체결가/거래량. 장중이면 실시간, 마감 후면 당일 종가."""
    data = http_client.get_json(
        TOSS_PRICE_URL, params={"codes": f"A{code}"}, headers=TOSS_HEADERS,
    )
    if not data:
        return None
    prices = (data.get("result") or {}).get("prices") or []
    if not prices:
        return None
    p = prices[0]
    close = _num(p.get("close"))
    base = _num(p.get("base"))          # 기준가(전일 종가)
    if close is None:
        return None
    return {
        "close": close,
        "prev_close": base,
        "volume": int(_num(p.get("volume")) or 0),
        "change_type": p.get("changeType"),
        "currency": p.get("currency", "KRW"),
        "trading_end": p.get("tradingEnd"),
        "source": "토스증권",
    }


# ---------------------------------------------------------------------------
# 네이버금융
# ---------------------------------------------------------------------------

def naver_basic(code: str) -> dict | None:
    data = http_client.get_json(NAVER_BASIC_URL.format(code=code), headers=NAVER_HEADERS)
    if not data or not data.get("stockName"):
        return None
    return {
        "name": data.get("stockName"),
        "close": _num(data.get("closePrice")),
        "change": _num(data.get("compareToPreviousClosePrice")),
        "change_rate": _num(data.get("fluctuationsRatio")),
        "direction": (data.get("compareToPreviousPrice") or {}).get("name"),
        "market_status": data.get("marketStatus"),
        "exchange": data.get("stockExchangeName"),
        "traded_at": data.get("localTradedAt"),
        "delay": data.get("delayTimeName"),
        "source": "네이버금융",
    }


def naver_totals(code: str) -> dict | None:
    """시가/고가/저가/거래량/거래대금/52주 고저/PER/PBR 등 종합 지표."""
    data = http_client.get_json(NAVER_INTEGRATION_URL.format(code=code), headers=NAVER_HEADERS)
    if not data:
        return None
    out = {}
    for info in data.get("totalInfos") or []:
        out[info.get("code")] = info.get("value")
    if not out:
        return None
    return {
        "prev_close": _num(out.get("lastClosePrice")),
        "open": _num(out.get("openPrice")),
        "high": _num(out.get("highPrice")),
        "low": _num(out.get("lowPrice")),
        "volume": _num(out.get("accumulatedTradingVolume")),
        "trading_value": _parse_korean_amount(out.get("accumulatedTradingValue")),
        "trading_value_text": out.get("accumulatedTradingValue"),
        "market_cap_text": out.get("marketValue"),
        "high_52w": _num(out.get("highPriceOf52Weeks")),
        "low_52w": _num(out.get("lowPriceOf52Weeks")),
        "per": out.get("per"),
        "pbr": out.get("pbr"),
        "foreign_rate": out.get("foreignRate"),
        "industry_code": data.get("industryCode"),
    }


def naver_investor_trend(code: str, days: int = 5) -> list[dict]:
    """투자자별 매매동향 (개인/외국인/기관 순매수 수량, 최신일 우선)."""
    data = http_client.get_json(
        NAVER_TREND_URL.format(code=code),
        params={"pageSize": str(days), "page": "1"},
        headers=NAVER_HEADERS,
    )
    if not isinstance(data, list):
        return []
    out = []
    for row in data:
        bizdate = str(row.get("bizdate") or "")
        out.append({
            "date": f"{bizdate[:4]}-{bizdate[4:6]}-{bizdate[6:]}" if len(bizdate) == 8 else bizdate,
            "individual": _num(row.get("individualPureBuyQuant")),
            "foreign": _num(row.get("foreignerPureBuyQuant")),
            "institution": _num(row.get("organPureBuyQuant")),
            "foreign_hold_ratio": _num(row.get("foreignerHoldRatio")),
            "close": _num(row.get("closePrice")),
            "volume": _num(row.get("accumulatedTradingVolume")),
        })
    return out


# 네이버 차트가 지원하는 주기와, 요청 구간을 얼마나 뒤로 잡아야 하는지
#   path        : 엔드포인트 경로
#   span_days   : 봉 1개당 대략 며칠 (요청 시작일 역산용)
#   datetime_key: 응답의 시각 필드명 (분봉만 다름)
#   close_key   : 응답의 종가 필드명 (분봉은 currentPrice)
NAVER_TIMEFRAMES = {
    "minute": {"path": "minute", "span_days": 1 / 380, "fmt": "%Y%m%d%H%M%S",
               "datetime_key": "localDateTime", "close_key": "currentPrice"},
    "day":    {"path": "day",    "span_days": 1.6,  "fmt": "%Y%m%d",
               "datetime_key": "localDate", "close_key": "closePrice"},
    "week":   {"path": "week",   "span_days": 7.5,  "fmt": "%Y%m%d",
               "datetime_key": "localDate", "close_key": "closePrice"},
    "month":  {"path": "month",  "span_days": 31,   "fmt": "%Y%m%d",
               "datetime_key": "localDate", "close_key": "closePrice"},
    "year":   {"path": "year",   "span_days": 366,  "fmt": "%Y%m%d",
               "datetime_key": "localDate", "close_key": "closePrice"},
}


def naver_chart(code: str, timeframe: str = "day", count: int = 120) -> pd.DataFrame:
    """주기별 OHLCV DataFrame (columns: open, high, low, close, volume; index=시각).

    timeframe: minute / day / week / month / year
    count    : 받아올 봉 개수 (넉넉히 요청한 뒤 마지막 count개를 반환)
    """
    spec = NAVER_TIMEFRAMES.get(timeframe)
    if spec is None:
        return pd.DataFrame()

    # 휴장일·주말을 감안해 여유 있게 과거로 잡되, 상한을 둡니다.
    # (년봉 200개를 그대로 역산하면 300년이 되어 pandas Timedelta 한계(약 292년)를 넘습니다)
    MAX_LOOKBACK_DAYS = 25_000        # 약 68년 — 국내 상장 종목 이력으로 충분
    lookback_days = min(MAX_LOOKBACK_DAYS,
                        max(2, int(count * spec["span_days"] * 1.5) + 5))
    end = datetime.now()
    start = end - pd.Timedelta(days=lookback_days)

    # 분봉은 시각까지, 나머지는 날짜까지만 지정
    if timeframe == "minute":
        start_s, end_s = start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M")
    else:
        start_s, end_s = start.strftime("%Y%m%d") + "0000", end.strftime("%Y%m%d") + "0000"

    data = http_client.get_json(
        f"https://api.stock.naver.com/chart/domestic/item/{code}/{spec['path']}",
        params={"startDateTime": start_s, "endDateTime": end_s},
        headers=NAVER_HEADERS,
        timeout=25,
    )
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for r in data:
        try:
            stamp = str(r[spec["datetime_key"]])
            close = float(r[spec["close_key"]])
            rows.append({
                "date": datetime.strptime(stamp, spec["fmt"]),
                "open": float(r.get("openPrice") or close),
                "high": float(r.get("highPrice") or close),
                "low": float(r.get("lowPrice") or close),
                "close": close,
                "volume": float(r.get("accumulatedTradingVolume") or 0),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.iloc[-count:] if count else df


def naver_daily_chart(code: str, days: int = 180) -> pd.DataFrame:
    """일봉 OHLCV (기존 호출부 호환용 얇은 래퍼)."""
    return naver_chart(code, "day", count=days)
