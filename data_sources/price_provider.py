"""
시세 데이터 공급자 (Price Provider)
---------------------------------
종목이 어느 시장인지에 따라 자동으로 소스를 고릅니다.

    코스피 / 코스닥 -> KoreanPriceProvider
        실시간 체결가·거래량 : 토스증권 (실패 시 네이버금융)
        일봉 OHLCV          : 네이버금융 (실패 시 Yahoo .KS/.KQ)
        투자자별 매매동향    : 네이버금융 (개인/외국인/기관 순매수)
    미국          -> YahooFinanceProvider

모든 공급자는 ResolvedSymbol(symbol_registry.resolve 결과)을 입력으로 받습니다.
즉 이 계층에 도달한 시점에는 "실존하는 종목"임이 이미 검증돼 있습니다.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from models import PriceSnapshot, InvestorFlow, ResolvedSymbol, MARKET_US
from data_sources import kis_client, kr_market, market_clock, toss_api


# 지원 주기 정의 (UI 라벨 + 기본 봉 개수)
TIMEFRAMES = {
    "second": {"label": "초봉",  "default_count": 120, "live_only": True},
    "minute": {"label": "분봉",  "default_count": 240, "live_only": False},
    "day":    {"label": "일봉",  "default_count": 120, "live_only": False},
    "week":   {"label": "주봉",  "default_count": 120, "live_only": False},
    "month":  {"label": "월봉",  "default_count": 120, "live_only": False},
    "year":   {"label": "년봉",  "default_count": 30,  "live_only": False},
}


class PriceProvider(ABC):
    @abstractmethod
    def get_daily_history(self, symbol: ResolvedSymbol, days: int = 120) -> pd.DataFrame:
        """기술적 지표 계산용 일봉. columns: open, high, low, close, volume"""

    @abstractmethod
    def get_history(self, symbol: ResolvedSymbol, timeframe: str = "day",
                    count: int = 120) -> pd.DataFrame:
        """주기별 OHLCV. timeframe: minute / day / week / month / year"""

    @abstractmethod
    def get_snapshot(self, symbol: ResolvedSymbol) -> PriceSnapshot:
        """현재가/시가/고가/저가/거래량/거래대금/등락률 스냅샷"""

    def get_investor_flow(self, symbol: ResolvedSymbol) -> InvestorFlow:
        return InvestorFlow(available=False)

    @abstractmethod
    def get_realtime_quote(self, symbol: ResolvedSymbol) -> dict:
        """실시간 폴링용 경량 시세. {price, change_rate, volume, source, market_status}"""


# ---------------------------------------------------------------------------
# 한국 (코스피 / 코스닥)
# ---------------------------------------------------------------------------

class KoreanPriceProvider(PriceProvider):
    """코스피·코스닥 전용 공급자.

    소스 우선순위
        1) 한국투자증권 KIS (공식 API — 환경변수로 키가 설정된 경우에만)
        2) 토스증권 실시간 체결가 + 네이버금융 종합지표
        3) Yahoo (.KS/.KQ) 최후 폴백
    """

    def get_daily_history(self, symbol: ResolvedSymbol, days: int = 120) -> pd.DataFrame:
        # 공식 API 우선 (토스 -> KIS), 없으면 공개 경로
        if toss_api.is_configured():
            df = toss_api.get_candles(symbol.key, "1d", count=days)
            if not df.empty:
                return df
        if kis_client.is_configured() and kis_client.use_for_quotes():
            df = kis_client.get_daily_chart(symbol.key, days=days)
            if not df.empty:
                return df

        df = kr_market.naver_daily_chart(symbol.key, days=days)
        if not df.empty:
            return df
        # 네이버가 막히면 Yahoo(.KS/.KQ)로 폴백
        return _yahoo_daily(symbol.yahoo_symbol, days)

    def get_history(self, symbol: ResolvedSymbol, timeframe: str = "day",
                    count: int = 120) -> pd.DataFrame:
        if timeframe == "day":
            return self.get_daily_history(symbol, days=count)

        # 토스 공식 API는 1분/일봉만 지원합니다 (주·월·년봉은 네이버 경로)
        if timeframe == "minute" and toss_api.is_configured():
            df = toss_api.get_candles(symbol.key, "1m", count=count)
            if not df.empty:
                return df

        df = kr_market.naver_chart(symbol.key, timeframe, count=count)
        if not df.empty:
            return df
        return _yahoo_history(symbol.yahoo_symbol, timeframe, count)

    def get_realtime_quote(self, symbol: ResolvedSymbol) -> dict:
        """실시간 폴링 전용 경량 조회 (뉴스/지표 계산 없이 가격만).

        대시보드가 몇 초마다 호출하므로 최소한의 요청만 보냅니다.
        """
        clock = market_clock.status_for(symbol.market)

        if toss_api.is_configured():
            p = toss_api.get_price(symbol.key)
            if p and p.get("price"):
                # 공식 API는 현재가만 주므로 전일종가는 일봉에서 채웁니다
                prev = None
                hist = kr_market.naver_daily_chart(symbol.key, days=3)
                if not hist.empty and len(hist) >= 2:
                    prev = float(hist["close"].iloc[-2])
                rate = ((p["price"] - prev) / prev * 100) if prev else None
                return {
                    "price": p["price"], "prev_close": prev,
                    "change_rate": round(rate, 2) if rate is not None else None,
                    "traded_at": p.get("timestamp"),
                    "source": "토스증권 Open API (공식)",
                    "market_status": clock,
                }

        if kis_client.is_configured() and kis_client.use_for_quotes():
            q = kis_client.get_quote(symbol.key)
            if q and q.get("close"):
                return {
                    "price": q["close"], "prev_close": q.get("prev_close"),
                    "change_rate": q.get("change_rate"),
                    "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                    "volume": q.get("volume"), "trading_value": q.get("trading_value"),
                    "source": q["source"], "market_status": clock,
                }

        toss = kr_market.toss_realtime_price(symbol.key)
        basic = kr_market.naver_basic(symbol.key)
        price = (toss or {}).get("close") or (basic or {}).get("close")
        if not price:
            return {"price": None, "source": "조회 실패", "market_status": clock}

        prev = (toss or {}).get("prev_close")
        rate = (basic or {}).get("change_rate")
        if rate is not None and prev:
            rate = abs(rate) * (1 if price >= prev else -1)
        elif prev:
            rate = (price - prev) / prev * 100

        return {
            "price": price,
            "prev_close": prev,
            "change_rate": round(rate, 2) if rate is not None else None,
            "volume": (toss or {}).get("volume"),
            "traded_at": (basic or {}).get("traded_at"),
            "source": (toss or {}).get("source") or "네이버금융",
            "market_status": clock,
        }

    def get_snapshot(self, symbol: ResolvedSymbol) -> PriceSnapshot:
        code = symbol.key

        # KIS가 설정돼 있으면 한 번의 공식 API 호출로 대부분을 채웁니다
        if kis_client.is_configured() and kis_client.use_for_quotes():
            snap = self._snapshot_from_kis(symbol)
            if snap:
                return snap

        toss = kr_market.toss_realtime_price(code)
        basic = kr_market.naver_basic(code)
        totals = kr_market.naver_totals(code) or {}

        # 현재가는 토스(실시간 체결) 우선, 없으면 네이버
        current = None
        source_parts = []
        if toss and toss.get("close"):
            current = toss["close"]
            source_parts.append("토스증권(실시간 체결)")
        elif basic and basic.get("close"):
            current = basic["close"]
            source_parts.append("네이버금융")

        prev_close = (totals.get("prev_close")
                      or (toss or {}).get("prev_close")
                      or None)
        volume = int((toss or {}).get("volume") or totals.get("volume") or 0)

        if totals:
            source_parts.append("네이버금융(시가·고저·거래대금)")

        # 마지막 안전망: 실시간 소스가 모두 실패하면 일봉 마지막 행 사용
        if current is None:
            hist = self.get_daily_history(symbol, days=10)
            if hist.empty:
                return PriceSnapshot(
                    ticker=code, timestamp=datetime.now(), current_price=0, open=0,
                    high=0, low=0, prev_close=0, volume=0, trading_value=0, change_rate=0,
                    name=symbol.name, market=symbol.market, currency=symbol.currency,
                    source="조회 실패", investor_flow=self.get_investor_flow(symbol),
                )
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else last
            current = float(last["close"])
            prev_close = prev_close or float(prev["close"])
            volume = volume or int(last["volume"])
            totals = {**totals, "open": float(last["open"]),
                      "high": float(last["high"]), "low": float(last["low"])}
            source_parts.append("네이버 일봉(폴백)")

        prev_close = prev_close or current
        change_rate = ((current - prev_close) / prev_close * 100) if prev_close else 0.0
        if basic and basic.get("change_rate") is not None:
            # 네이버가 계산한 공식 등락률이 있으면 부호를 맞춰 그대로 사용
            rate = abs(basic["change_rate"])
            change_rate = rate if current >= prev_close else -rate

        trading_value = totals.get("trading_value") or (current * volume)

        return PriceSnapshot(
            ticker=code,
            timestamp=datetime.now(),
            current_price=current,
            open=totals.get("open") or 0,
            high=totals.get("high") or 0,
            low=totals.get("low") or 0,
            prev_close=prev_close,
            volume=volume,
            trading_value=trading_value,
            change_rate=round(change_rate, 2),
            name=symbol.name,
            market=symbol.market,
            currency="KRW",
            market_status=(basic or {}).get("market_status", ""),
            high_52w=totals.get("high_52w"),
            low_52w=totals.get("low_52w"),
            market_cap_text=totals.get("market_cap_text"),
            per=totals.get("per"),
            pbr=totals.get("pbr"),
            source=" + ".join(source_parts) if source_parts else "알 수 없음",
            investor_flow=self.get_investor_flow(symbol),
        )

    def _snapshot_from_kis(self, symbol: ResolvedSymbol) -> PriceSnapshot | None:
        q = kis_client.get_quote(symbol.key)
        if not q or not q.get("close"):
            return None

        market_cap = q.get("market_cap")
        return PriceSnapshot(
            ticker=symbol.key,
            timestamp=datetime.now(),
            current_price=q["close"],
            open=q.get("open") or 0,
            high=q.get("high") or 0,
            low=q.get("low") or 0,
            prev_close=q.get("prev_close") or q["close"],
            volume=int(q.get("volume") or 0),
            trading_value=q.get("trading_value") or 0,
            change_rate=round(q.get("change_rate") or 0, 2),
            name=symbol.name, market=symbol.market, currency="KRW",
            market_status=market_clock.status_for(symbol.market)["label"],
            high_52w=q.get("high_52w"), low_52w=q.get("low_52w"),
            market_cap_text=(f"{market_cap / 10000:,.0f}조원" if market_cap and market_cap >= 10000
                             else (f"{market_cap:,.0f}억원" if market_cap else None)),
            per=q.get("per"), pbr=q.get("pbr"),
            source="한국투자증권 KIS (공식 API)",
            investor_flow=self.get_investor_flow(symbol),
        )

    def get_investor_flow(self, symbol: ResolvedSymbol) -> InvestorFlow:
        """개인/외국인/기관 순매매.

        KIS가 설정돼 있으면 장중에도 갱신되는 공식 데이터를 쓰고,
        아니면 네이버금융 투자자별 매매동향(전일 확정치)을 사용합니다.
        """
        if kis_client.is_configured() and kis_client.use_for_quotes():
            k = kis_client.get_investor_flow(symbol.key)
            if k:
                return InvestorFlow(
                    individual_net=k.get("individual"),
                    foreign_net=k.get("foreign"),
                    institution_net=k.get("institution"),
                    individual_net_value=k.get("individual_value"),
                    foreign_net_value=k.get("foreign_value"),
                    institution_net_value=k.get("institution_value"),
                    trade_date=k.get("date"),
                    unit="shares",
                    source=k["source"] + " (장중 갱신)",
                    available=any(k.get(x) is not None
                                  for x in ("individual", "foreign", "institution")),
                )

        rows = kr_market.naver_investor_trend(symbol.key, days=3)
        if not rows:
            return InvestorFlow(available=False, source="네이버금융(조회 실패)")

        latest = rows[0]
        close = latest.get("close") or 0

        def to_value(quantity):
            if quantity is None or not close:
                return None
            return quantity * close

        return InvestorFlow(
            individual_net=latest.get("individual"),
            foreign_net=latest.get("foreign"),
            institution_net=latest.get("institution"),
            individual_net_value=to_value(latest.get("individual")),
            foreign_net_value=to_value(latest.get("foreign")),
            institution_net_value=to_value(latest.get("institution")),
            foreign_hold_ratio=latest.get("foreign_hold_ratio"),
            trade_date=latest.get("date"),
            unit="shares",
            source="네이버금융 투자자별 매매동향",
            available=any(latest.get(k) is not None
                          for k in ("individual", "foreign", "institution")),
        )


# ---------------------------------------------------------------------------
# 미국
# ---------------------------------------------------------------------------

def _yahoo_daily(yahoo_symbol: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    try:
        data = yf.download(yahoo_symbol, period=f"{int(days * 1.6) + 20}d",
                           interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]
    data = data.rename(columns=str.lower)
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in data.columns]
    return data[keep].dropna(subset=["close"])


YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

# 주기 -> Yahoo chart API 의 (range, interval)
# Yahoo는 분봉 보관 기간이 짧고(1분봉 최대 7일), 년봉은 아예 없어서 월봉을 묶어 만듭니다.
_YAHOO_TF = {
    "minute": ("5d", "1m"),
    "day":    ("1y", "1d"),
    "week":   ("5y", "1wk"),
    "month":  ("max", "1mo"),
    "year":   ("max", "1mo"),   # 아래에서 연 단위로 재집계
}


def _yahoo_history(yahoo_symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Yahoo chart API 로 주기별 OHLCV 조회 (미국 종목 + 국내 폴백)."""
    from data_sources import http_client

    rng, interval = _YAHOO_TF.get(timeframe, ("1y", "1d"))
    data = http_client.get_json(
        YAHOO_QUOTE_URL.format(sym=yahoo_symbol),
        params={"range": rng, "interval": interval}, timeout=20,
    )
    result = (((data or {}).get("chart") or {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        return pd.DataFrame()

    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    df = pd.DataFrame({
        "date": pd.to_datetime(result["timestamp"], unit="s"),
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "volume": quote.get("volume"),
    }).dropna(subset=["close"]).set_index("date").sort_index()

    if df.empty:
        return df

    if timeframe == "year":
        # 월봉 -> 년봉 재집계 (Yahoo에 연봉 인터벌이 없음)
        df = df.resample("YE").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(subset=["close"])

    return df.iloc[-count:] if count else df


class YahooFinanceProvider(PriceProvider):
    """미국 주식용 공급자 (인증 불필요, 무료)."""

    def get_daily_history(self, symbol: ResolvedSymbol, days: int = 120) -> pd.DataFrame:
        return _yahoo_daily(symbol.yahoo_symbol, days)

    def get_history(self, symbol: ResolvedSymbol, timeframe: str = "day",
                    count: int = 120) -> pd.DataFrame:
        return _yahoo_history(symbol.yahoo_symbol, timeframe, count)

    def get_realtime_quote(self, symbol: ResolvedSymbol) -> dict:
        """Yahoo chart API의 meta 블록에서 실시간 시세만 가볍게 가져옵니다.

        yfinance 를 쓰지 않는 이유: 폴링마다 Ticker 객체를 만들면 느리고 요청이 많아집니다.
        """
        from data_sources import http_client

        clock = market_clock.status_for(symbol.market)
        data = http_client.get_json(
            YAHOO_QUOTE_URL.format(sym=symbol.yahoo_symbol),
            params={"range": "1d", "interval": "1m", "includePrePost": "true"},
            timeout=8,
        )
        result = ((((data or {}).get("chart") or {}).get("result") or [{}])[0]) or {}
        meta = result.get("meta") or {}
        price = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")

        # 프리·애프터마켓 — meta 의 정규장 가격은 마지막 정규장 체결가에 멈춰
        # 있습니다. 이 가격으로 확장 시간에 지정가 주문을 내면 몇 시간 전
        # 가격이 됩니다. includePrePost 분봉의 마지막 체결가가 실제 현재가입니다.
        closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        last_bar = next((c for c in reversed(closes) if c), None)
        if last_bar:
            price = float(last_bar)

        if not price:
            return {"price": None, "source": "조회 실패", "market_status": clock}

        rate = ((price - prev) / prev * 100) if prev else None
        return {
            "price": float(price),
            "prev_close": float(prev) if prev else None,
            "change_rate": round(rate, 2) if rate is not None else None,
            "high": meta.get("regularMarketDayHigh"),
            "low": meta.get("regularMarketDayLow"),
            "volume": meta.get("regularMarketVolume"),
            # 상장 거래소 — 해외주문은 거래소 코드가 맞아야 접수됩니다
            # (NYSE 종목을 나스닥으로 내면 "해당종목정보가 없습니다" 로 거부됩니다)
            "exchange": meta.get("exchangeName") or "",
            "source": "Yahoo Finance",
            "market_status": clock,
        }

    def get_snapshot(self, symbol: ResolvedSymbol) -> PriceSnapshot:
        import yfinance as yf

        current = open_price = high = low = prev_close = 0.0
        volume = 0
        try:
            info = yf.Ticker(symbol.yahoo_symbol).fast_info
            current = float(info.get("last_price") or 0)
            open_price = float(info.get("open") or 0)
            high = float(info.get("day_high") or 0)
            low = float(info.get("day_low") or 0)
            prev_close = float(info.get("previous_close") or 0)
            volume = int(info.get("last_volume") or 0)
        except Exception:
            pass

        if not current:
            hist = self.get_daily_history(symbol, days=10)
            if hist.empty:
                return PriceSnapshot(
                    ticker=symbol.key, timestamp=datetime.now(), current_price=0, open=0,
                    high=0, low=0, prev_close=0, volume=0, trading_value=0, change_rate=0,
                    name=symbol.name, market=symbol.market, currency="USD", source="조회 실패",
                )
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else last
            current = float(last["close"])
            open_price, high, low = float(last["open"]), float(last["high"]), float(last["low"])
            prev_close = float(prev["close"])
            volume = int(last["volume"])

        change_rate = ((current - prev_close) / prev_close * 100) if prev_close else 0.0
        return PriceSnapshot(
            ticker=symbol.key,
            timestamp=datetime.now(),
            current_price=current,
            open=open_price, high=high, low=low, prev_close=prev_close,
            volume=volume, trading_value=current * volume,
            change_rate=round(change_rate, 2),
            name=symbol.name, market=symbol.market, currency="USD",
            source="Yahoo Finance",
            investor_flow=InvestorFlow(
                available=False,
                source="미국 시장은 개인/외국인/기관 일별 순매매 공개 데이터가 없습니다",
            ),
        )


# ---------------------------------------------------------------------------
# 라우터
# ---------------------------------------------------------------------------

_korean = KoreanPriceProvider()
_yahoo = YahooFinanceProvider()


def get_provider(symbol: ResolvedSymbol) -> PriceProvider:
    """종목의 시장에 맞는 공급자를 반환."""
    return _yahoo if symbol.market == MARKET_US else _korean
