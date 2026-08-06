"""
아테나 시그널 API 서버
----------------------
웹사이트와 크롬 확장 프로그램이 공통으로 호출하는 REST API입니다.

실행:
    uvicorn api:app --reload --port 8000

엔드포인트:
    GET  /api/search?q=삼성            - 종목 자동완성 (코스피/코스닥/미국)
    GET  /api/resolve/{query}          - 종목 실존 확인만 수행
    GET  /api/predict/{query}          - 예측 생성 + 저장 후 반환
    GET  /api/community/{query}        - 실시간 커뮤니티 피드 (폴링용)
    GET  /api/news/{query}             - 뉴스 목록 + 감성 근거
    GET  /api/history/{query}          - 해당 종목의 최근 예측 기록
    GET  /api/stats                    - 전체 누적 적중률
    POST /api/close                    - 장마감 채점 + 가중치 재조정

종목이 실제로 상장돼 있지 않으면 모든 종목 엔드포인트가 **HTTP 404** 와 함께
{"error": "SYMBOL_NOT_FOUND", "message": ..., "suggestions": [...]} 를 돌려줍니다.
프론트엔드는 이 응답을 받으면 데모 데이터로 대체하지 않고 "없는 종목" 화면을 띄웁니다.
"""

import dataclasses
import logging
import socket
import sys
import time

# 콘솔이 cp949 면 '—' 같은 문자를 못 찍고 **서버가 기동 중 죽습니다.**
# 화면에 글자 하나 예쁘게 나오자고 서비스가 안 뜨는 일은 없어야 합니다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import INITIAL_WEIGHTS, INTRADAY_BARS, PREDICTION_HORIZONS
from data_sources import (community_crawler, credentials, fx, kis_client, kis_realtime,
                          kis_trading,
                          market_clock, news_crawler, scrap_store, screener,
                          symbol_registry, toss_api)
from data_sources import price_provider
from data_sources.price_provider import get_provider
from engine import (autotrade, backtest, broker, ensemble, feed, indicators,
                    instruments, protections, recommender, risk, scalping, scoring)
from models import Prediction, SymbolNotFoundError
from storage import autotrade as at_store
from storage import db, derivatives, paper, users

# 모의투자 평가용 시세 캐시 (종목당 30초)
_price_cache: dict[str, tuple[float, float | None]] = {}
# 마지막 자동 채점 결과 (성적표에 표시)
_last_auto_resolve: dict | None = None

app = FastAPI(title="Athena Signal API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).parent / "web"


# 자동 채점 주기 (초) — 만기 도래분만 처리하므로 자주 돌아도 부담이 적습니다
AUTO_RESOLVE_INTERVAL = 600


def _auto_resolve_loop():
    """만기가 지난 예측을 주기적으로 자동 채점합니다.

    사용자가 `main.py close` 를 잊어도 성적표가 저절로 쌓이게 하기 위한 루프입니다.
    데몬 스레드라 서버를 끄면 함께 종료됩니다.
    """
    global _last_auto_resolve
    import threading
    time.sleep(20)          # 서버 기동 직후 첫 요청과 겹치지 않게 잠깐 대기
    while True:
        try:
            result = backtest.auto_resolve_all()
            _last_auto_resolve = result
            if result["scored"]:
                print(f"[auto] 자동 채점 {result['scored']}건 완료")
                backtest.adjust_weights()
        except Exception as exc:
            print(f"[auto] 자동 채점 실패: {type(exc).__name__}: {exc}")
        time.sleep(AUTO_RESOLVE_INTERVAL)


# ---------------------------------------------------------------------------
# 인증
# ---------------------------------------------------------------------------

def current_user(request: Request) -> dict | None:
    """Authorization: Bearer <token> 또는 athena_token 쿠키에서 사용자 확인."""
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = request.cookies.get("athena_token", "")
    return users.user_from_token(token)


def require_user(request: Request) -> dict:
    """로그인이 필요한 엔드포인트용. 미인증이면 401."""
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return user


class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/register")
def auth_register(req: RegisterRequest):
    result = users.register(req.username, req.password, req.display_name)
    if not result["ok"]:
        return JSONResponse(status_code=400, content=result)
    # 가입 직후 바로 로그인 처리
    login_result = users.login(req.username, req.password)
    paper.ensure_account(login_result["user"]["id"])
    return login_result


@app.post("/api/auth/login")
def auth_login(req: LoginRequest):
    result = users.login(req.username, req.password)
    if not result["ok"]:
        return JSONResponse(status_code=401, content=result)
    paper.ensure_account(result["user"]["id"])
    return result


@app.post("/api/auth/logout")
def auth_logout(request: Request):
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    users.logout(token or request.cookies.get("athena_token", ""))
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(request: Request):
    user = current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "로그인이 필요합니다."})
    return {"user": user}


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/auth/password")
def auth_change_password(req: PasswordChange, request: Request):
    user = require_user(request)
    result = users.change_password(user["id"], req.current_password, req.new_password)
    if not result["ok"]:
        return JSONResponse(status_code=400, content=result)
    return result


# ---------------------------------------------------------------------------
# 관심종목 (사용자별)
# ---------------------------------------------------------------------------

@app.get("/api/watchlist")
def watchlist_get(request: Request):
    user = require_user(request)
    return {"watchlist": users.get_watchlist(user["id"])}


class WatchItem(BaseModel):
    ticker: str


@app.post("/api/watchlist")
def watchlist_add(item: WatchItem, request: Request):
    user = require_user(request)
    try:
        symbol = symbol_registry.resolve(item.ticker)
    except SymbolNotFoundError as exc:
        return _not_found(exc)
    users.add_watch(user["id"], symbol.key, symbol.name, symbol.market)
    return {"ok": True, "watchlist": users.get_watchlist(user["id"])}


@app.delete("/api/watchlist/{ticker}")
def watchlist_remove(ticker: str, request: Request):
    user = require_user(request)
    users.remove_watch(user["id"], ticker)
    return {"ok": True, "watchlist": users.get_watchlist(user["id"])}


# ---------------------------------------------------------------------------
# 콘솔 정리 — 서버 창은 "지금 괜찮은가"를 한눈에 보는 곳입니다
# ---------------------------------------------------------------------------

# 화면이 7초마다 물어보는 주소들. 이것까지 다 찍으면 하루에 수만 줄이 쌓여
# 정작 봐야 할 오류가 묻힙니다. 나머지 요청은 그대로 남깁니다.
_QUIET_PATHS = (
    "/api/autotrade/events", "/api/autotrade ", "/api/status",
    "/api/paper", "/api/chart/", "/api/quote/",
)


class _QuietAccessLog(logging.Filter):
    """폴링 접속 로그만 걸러내는 필터 (오류·느린 응답은 통과)."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # 4xx·5xx 는 폴링이라도 보여줘야 합니다
        if ' 200 ' not in message and ' 304 ' not in message:
            return True
        return not any(path in message for path in _QUIET_PATHS)


def _tidy_console():
    """색상 코드 제거 + 폴링 로그 억제.

    cmd 창은 ANSI 색상을 해석하지 못해 `[32mINFO[0m` 같은 글자가 그대로
    찍힙니다. uvicorn 이 이미 포매터를 만든 뒤이므로, 여기서 색을 끕니다.
    """
    logging.getLogger("uvicorn.access").addFilter(_QuietAccessLog())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            formatter = getattr(handler, "formatter", None)
            if hasattr(formatter, "use_colors"):
                formatter.use_colors = False


@app.on_event("startup")
def startup():
    _tidy_console()
    db.init_db()
    users.init()
    scrap_store.init()
    paper.init()
    derivatives.init()
    at_store.init()
    removed = scrap_store.purge_expired()

    import threading
    threading.Thread(target=_auto_resolve_loop, daemon=True,
                     name="athena-auto-resolve").start()

    # 자동매매 루프 — 켜둔 사용자가 있으면 서버 재시작 후에도 이어서 돕니다.
    # (자동매매를 켜놓고 서버를 재부팅했는데 포지션만 남고 관리가 멈추면 사고입니다)
    autotrade.loop.start()
    trading = kis_trading.status()

    # 시작 요약 — 창을 흘깃 봤을 때 "지금 무엇이 켜져 있는가"가 보여야 합니다
    live = trading["keys_configured"] and not trading["mock"] and trading["live_enabled"]
    # 문자는 ASCII 로만 그립니다 — 콘솔 코드페이지가 무엇이든 깨지지 않게.
    line = "-" * 58
    print("\n" + line)
    print("  아테나 시그널 - 백엔드 준비 완료")
    print(line)
    print("   주소       http://127.0.0.1:8000   (웹은 3000번)")
    print(f"   자동매매   가동 중인 계정 {len(at_store.enabled_users())}개 / "
          f"{AUTO_RESOLVE_INTERVAL}초마다 자동 채점")
    if trading["keys_configured"]:
        print(f"   증권사     {trading['account_masked'] or '계좌 미설정'} / "
              f"{'실전' if not trading['mock'] else '모의투자'} / "
              f"실주문 {'허용' if trading['live_enabled'] else '잠김'}")
    else:
        print("   증권사     미연결 (모의 계좌로만 동작)")
    if removed:
        print(f"   정리       만료된 스크랩 {removed}건")
    if live:
        print("   [주의] 실계좌 실주문이 켜져 있습니다 - 실제 자금이 움직입니다.")
    print(line)
    # 파이프로 흘릴 때 버퍼에 갇히지 않게 (배너는 바로 보여야 의미가 있습니다)
    print("  아래에는 오류와 주요 요청만 표시됩니다 (화면 폴링 로그는 숨김)\n",
          flush=True)


def _to_dict(obj):
    """dataclass든 아니든 안전하게 dict로 변환 (datetime은 isoformat 문자열로)"""
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
    else:
        d = dict(obj)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def _particle(word: str, with_batchim: str, without_batchim: str) -> str:
    """한글 받침 유무에 맞는 조사를 고릅니다 ('지표는' / '여론은').

    한글 음절은 유니코드 AC00부터 28개 종성 주기로 배열돼 있어,
    (코드 - 0xAC00) % 28 == 0 이면 받침이 없습니다.
    """
    if not word:
        return without_batchim
    last = word.strip()[-1]
    if not ("가" <= last <= "힣"):
        return without_batchim
    return without_batchim if (ord(last) - 0xAC00) % 28 == 0 else with_batchim


def _not_found(exc: SymbolNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": "SYMBOL_NOT_FOUND",
            "query": exc.query,
            "message": exc.reason,
            "suggestions": exc.suggestions,
        },
    )


def _symbol_payload(symbol) -> dict:
    payload = {
        "key": symbol.key,
        "name": symbol.name,
        "long_name": symbol.long_name,
        "market": symbol.market,
        "market_label": symbol.market_label,
        "currency": symbol.currency,
        "verified_by": symbol.verified_by,
        "is_korean": symbol.is_korean,
        "is_inverse": symbol.is_inverse,
        "leverage": symbol.leverage,
        "is_leveraged_etf": symbol.is_leveraged_etf,
    }
    if symbol.is_leveraged_etf:
        parts = []
        if symbol.leverage > 1:
            parts.append(f"{symbol.leverage:g}배 레버리지")
        if symbol.is_inverse:
            parts.append("인버스(기초자산과 반대로 움직임)")
        payload["leverage_warning"] = (
            " · ".join(parts)
            + " 상품입니다. 일일 수익률을 추종하므로 장기 보유 시 기초지수와 크게 어긋날 수 있고"
              " (변동성 끌림), 뉴스 감성 신호의 신뢰도도 일반 종목보다 낮습니다."
        )
    return payload


# ---------------------------------------------------------------------------
# 종목 검색 / 검증
# ---------------------------------------------------------------------------

@app.get("/api/search")
def search_symbols(q: str = "", limit: int = 8):
    """자동완성용. 확정이 아니라 후보 목록만 돌려줍니다."""
    return {"query": q, "results": symbol_registry.search(q, limit=limit)}


@app.get("/api/resolve/{query}")
def resolve_symbol(query: str):
    """종목 실존 여부만 확인 (분석은 수행하지 않음)."""
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)
    return {"found": True, "symbol": _symbol_payload(symbol)}


# ---------------------------------------------------------------------------
# 예측
# ---------------------------------------------------------------------------

@app.get("/api/predict/{query}")
def predict(query: str, request: Request):
    user = current_user(request)
    user_id = user["id"] if user else None
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    weights = db.get_latest_weights() or INITIAL_WEIGHTS
    provider = get_provider(symbol)

    daily_history = provider.get_daily_history(symbol, days=120)
    technical = scoring.analyze_technical(daily_history)
    snapshot = provider.get_snapshot(symbol)

    news_items = news_crawler.get_news(symbol, limit=20)
    news_score = news_crawler.aggregate_news_score(news_items)

    community = community_crawler.get_community_sentiment(symbol)
    rationale = scoring.build_rationale(technical, news_items, community, weights)

    # 단기 시평선용 분봉 기술점수 (실패해도 일봉 점수로 대체되므로 치명적이지 않음)
    intraday_score = _intraday_score(symbol, provider)

    result = scoring.combine_scores(
        technical.score, news_score, community.bullish_ratio, weights,
    )

    # 방금 받아온 무거운 데이터를 실시간 폴링(/api/quote)이 재사용하도록 캐시에 적재
    _store_slow_context(symbol.key, daily_history, news_score,
                        community.bullish_ratio, intraday_score)

    horizon_preds = scoring.build_horizon_predictions(
        technical.score, intraday_score, news_score,
        community.bullish_ratio, weights, symbol.market,
    )
    open_pred = scoring.build_open_prediction(
        technical.score, news_score, community.bullish_ratio, weights, symbol.market,
    )

    now = datetime.now()
    base_price = snapshot.current_price or None

    horizons_out = []
    for hp in horizon_preds:
        pred = Prediction(
            ticker=symbol.key, horizon_label=hp["horizon"], predicted_at=now,
            direction=hp["direction"], probability=hp["probability"],
            technical_score=hp["technical_score"], news_score=news_score,
            community_score=hp["community_score"], weights_used=weights,
            market=symbol.market, name=symbol.name,
        )
        # 만기 시각과 기준가를 함께 저장해야 시평선별로 정확히 채점할 수 있습니다
        pred_id = db.save_prediction(
            pred,
            target_at=now + timedelta(minutes=hp["minutes"]),
            base_price=base_price,
            user_id=user_id,
        )
        horizons_out.append({**hp, "id": pred_id})

    return {
        "ticker": symbol.key,
        "symbol": _symbol_payload(symbol),
        "generated_at": datetime.now().isoformat(),
        "technical_score": technical.score,
        "technical": {
            "score": technical.score,
            "bars_used": technical.bars_used,
            "sufficient_data": technical.sufficient_data,
            "note": technical.note,
            "regime": technical.regime,
            "intraday_score": intraday_score,
            "indicators": [
                {
                    "key": i.key, "label": i.label, "value_text": i.value_text,
                    "score": i.score, "weight": i.weight, "verdict": i.verdict,
                    "reason": i.reason, "formula": i.formula, "values": i.values,
                    "contribution": i.contribution, "family": i.family,
                }
                for i in technical.indicators
            ],
        },
        "news_score": news_score,
        "news_summary": news_crawler.summarize_news(news_items),
        "community_bullish_ratio": community.bullish_ratio,
        "market_snapshot": _to_dict(snapshot),
        "rationale": rationale,
        "score_breakdown": result,
        "news_sample": [
            {"title": n.title, "sentiment": n.sentiment_score, "source": n.source,
             "url": n.url, "keywords": n.matched_keywords,
             "published_at": n.published_at.isoformat()}
            for n in news_items[:8]
        ],
        "community": {
            "bullish_ratio": community.bullish_ratio,
            "post_count": community.post_count,
            "bullish_count": community.bullish_count,
            "bearish_count": community.bearish_count,
            "neutral_count": community.neutral_count,
            "sources": community.sources,
            "source_counts": community.source_counts,
            "recent_posts": [_to_dict(p) for p in community.recent_posts],
            "demo": community.is_demo,
        },
        "weights": weights,
        "open_prediction": open_pred,
        "market_status": market_clock.status_for(symbol.market),
        "data_sources": {
            "kis_enabled": kis_client.is_configured(),
            "price": snapshot.source,
            "investor_flow": snapshot.investor_flow.source,
        },
        "predictions": horizons_out,
    }


@app.get("/api/status")
def service_status():
    """어떤 데이터 소스가 활성화돼 있는지 + 현재 장 상태."""
    creds = credentials.status()
    active = [v["label"] for v in creds.values() if v["configured"]]

    return {
        "apis": creds,
        "active_count": len(active),
        "total_count": len(creds),
        "active_labels": active,
        "hint": ("공식 API 사용 중: " + ", ".join(active) if active
                 else "setup_api.bat 을 실행하면 공식 API를 사용합니다 "
                      "(미설정 시에도 공개 경로로 정상 동작)"),
        # 하위호환
        "kis_enabled": kis_client.is_configured(),
        "toss_enabled": toss_api.is_configured(),
        "korea_market": market_clock.status_for("KOSPI"),
        "us_market": market_clock.status_for("US"),
        "server_time_kst": market_clock.now_kst().isoformat(),
    }


# ---------------------------------------------------------------------------
# 실시간 시세 폴링
# ---------------------------------------------------------------------------
# 대시보드가 몇 초마다 /api/quote 를 호출하므로, 매번 뉴스·커뮤니티·일봉을 새로
# 받아오면 외부 서버에 과부하를 주고 응답도 느려집니다. 그래서 "느리게 변하는 것"만
# TTL 캐시에 담고, 가격만 매번 실시간으로 가져와 확률을 다시 계산합니다.
_slow_cache: dict[str, tuple[float, dict]] = {}
_SLOW_TTL = 180  # 초


def _intraday_score(symbol, provider) -> float | None:
    """분봉 기반 단기 기술점수 — 10분/1시간 같은 짧은 시평선에 씁니다.

    분봉을 못 받으면(신규 상장·해외 장 마감 등) None을 돌려주고,
    호출부는 일봉 점수만으로 계산합니다.
    """
    try:
        bars = provider.get_history(symbol, "minute", count=INTRADAY_BARS)
    except Exception:
        return None
    if bars is None or len(bars) < 30:
        return None
    return scoring.analyze_technical(bars).score


def _store_slow_context(key: str, history, news_score: float, community_ratio: float,
                        intraday_score: float | None = None):
    """/predict 가 이미 받아온 데이터를 캐시에 넣어 둡니다.

    이게 없으면 첫 /quote 호출이 일봉·뉴스·커뮤니티를 처음부터 다시 받느라 10초 이상
    걸려서, 대시보드에 실시간 가격이 한참 동안 뜨지 않습니다.
    """
    _slow_cache[key] = (time.time(), {
        "history": history,
        "news_score": news_score,
        "community_ratio": community_ratio,
        "intraday_score": intraday_score,
    })


def _slow_context(symbol, provider) -> dict:
    """뉴스 점수 / 커뮤니티 점수 / 일봉 / 분봉점수 — 3분 캐시."""
    key = symbol.key
    hit = _slow_cache.get(key)
    if hit and time.time() - hit[0] < _SLOW_TTL:
        return hit[1]

    history = provider.get_daily_history(symbol, days=120)
    news_items = news_crawler.get_news(symbol, limit=20)
    community = community_crawler.get_community_sentiment(symbol)
    _store_slow_context(key, history,
                        news_crawler.aggregate_news_score(news_items),
                        community.bullish_ratio,
                        _intraday_score(symbol, provider))
    return _slow_cache[key][1]


@app.get("/api/quote/{query}")
def quote(query: str):
    """실시간 시세 + 그 가격을 반영한 확률 재계산 (폴링 전용, 가볍게).

    장중에 가격이 움직이면 마지막 일봉의 종가·고가·저가를 실시간 가격으로 갈아끼운 뒤
    기술 지표를 다시 계산합니다. 그래서 확률이 가격 흐름에 맞춰 실제로 변합니다.
    뉴스/커뮤니티 점수는 3분 캐시를 재사용합니다(수초 단위로 변하는 값이 아님).
    """
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    provider = get_provider(symbol)
    live = provider.get_realtime_quote(symbol)

    if not live.get("price"):
        return {"ticker": symbol.key, "symbol": _symbol_payload(symbol),
                "price": None, "market_status": live.get("market_status"),
                "source": live.get("source"), "error": "시세를 가져오지 못했습니다."}

    ctx = _slow_context(symbol, provider)
    history = ctx["history"]

    # 마지막 봉을 실시간 가격으로 갱신 -> 지표가 현재 가격을 반영
    if history is not None and not history.empty:
        history = history.copy()
        last = history.index[-1]
        price = live["price"]
        history.loc[last, "close"] = price
        history.loc[last, "high"] = max(float(history.loc[last, "high"]), price)
        history.loc[last, "low"] = min(float(history.loc[last, "low"]), price)
        if live.get("volume"):
            history.loc[last, "volume"] = live["volume"]

    technical = scoring.analyze_technical(history)
    weights = db.get_latest_weights() or INITIAL_WEIGHTS
    result = scoring.combine_scores(
        technical.score, ctx["news_score"], ctx["community_ratio"], weights,
    )
    # 시평선별 확률 + 장 시작 전 예측도 함께 갱신 (게이지가 살아 움직이도록)
    horizon_preds = scoring.build_horizon_predictions(
        technical.score, ctx.get("intraday_score"), ctx["news_score"],
        ctx["community_ratio"], weights, symbol.market,
    )
    open_pred = scoring.build_open_prediction(
        technical.score, ctx["news_score"], ctx["community_ratio"],
        weights, symbol.market,
    )

    return {
        "ticker": symbol.key,
        "symbol": _symbol_payload(symbol),
        "price": live["price"],
        "prev_close": live.get("prev_close"),
        "change_rate": live.get("change_rate"),
        "high": live.get("high"),
        "low": live.get("low"),
        "volume": live.get("volume"),
        "trading_value": live.get("trading_value"),
        "traded_at": live.get("traded_at"),
        "source": live.get("source"),
        "market_status": live.get("market_status"),
        "technical_score": technical.score,
        "direction": result["direction"],
        "probability": result["probability"],
        "probability_up": result["probability_up"],
        "predictions": horizon_preds,
        "open_prediction": open_pred,
        "regime": technical.regime.get("label") if technical.regime else None,
        "server_time": datetime.now().isoformat(),
    }


@app.get("/api/chart/{query}")
def chart(query: str, days: int = 120, timeframe: str = "day"):
    """캔들차트용 시계열 (OHLCV + 이동평균 + 볼린저 + RSI + MACD).

    timeframe: minute(분봉) / day(일봉) / week(주봉) / month(월봉) / year(년봉)
    초봉(second)은 공개 API가 제공하지 않아 프론트에서 실시간 체결을 모아 그립니다.

    지표는 요청 구간보다 80봉 더 받아 이동평균 워밍업을 채운 뒤 잘라내므로,
    어떤 주기를 골라도 60선이 차트 첫 봉부터 그려집니다.
    """
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    if timeframe not in price_provider.TIMEFRAMES or timeframe == "second":
        timeframe = "day"

    provider = get_provider(symbol)
    history = provider.get_history(symbol, timeframe, count=days + 80)
    series = indicators.build_chart_series(history, days=days, timeframe=timeframe)

    return {
        "ticker": symbol.key,
        "symbol": _symbol_payload(symbol),
        "timeframe": timeframe,
        "timeframe_label": price_provider.TIMEFRAMES[timeframe]["label"],
        **series,
    }


# ---------------------------------------------------------------------------
# 스크랩 수집 (확장프로그램 -> 서버)
# ---------------------------------------------------------------------------

class ScrapItem(BaseModel):
    title: str
    url: str | None = None
    posted_at: str | None = None


class ScrapRequest(BaseModel):
    ticker: str
    source: str = "other"
    kind: str = "community"
    items: list[ScrapItem] = []


@app.post("/api/scrap")
def ingest_scrap(req: ScrapRequest):
    """확장프로그램이 브라우저에서 긁은 게시글을 받습니다.

    서버가 직접 접근할 수 없는 커뮤니티(토스·팍스넷·카카오페이증권 등)를
    사용자 브라우저를 통해 수집하기 위한 경로입니다. 사용자가 이미 열람 중인
    페이지의 공개 게시글만 대상으로 하며, 자격증명은 주고받지 않습니다.
    """
    try:
        symbol = symbol_registry.resolve(req.ticker)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    result = scrap_store.save_batch(
        symbol.key, req.source,
        [i.model_dump() for i in req.items], req.kind,
    )
    # 새 글이 들어왔으면 캐시를 비워 다음 조회에 바로 반영되게 합니다
    if result["saved"]:
        _slow_cache.pop(symbol.key, None)

    return {
        "ticker": symbol.key,
        "symbol": _symbol_payload(symbol),
        "source": req.source,
        **result,
        "sources": scrap_store.source_summary(symbol.key),
    }


@app.get("/api/scrap/{query}")
def scrap_status(query: str):
    """이 종목에 대해 어떤 출처에서 몇 건이 수집됐는지."""
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    return {
        "ticker": symbol.key,
        "sources": scrap_store.source_summary(symbol.key),
        "ttl_hours": scrap_store.SCRAP_TTL_HOURS,
        "recent": scrap_store.get_recent(symbol.key, limit=30),
    }


@app.get("/api/timeframes")
def timeframes():
    """UI가 주기 선택 버튼을 그릴 때 쓰는 목록."""
    return {"timeframes": [{"key": k, **v} for k, v in price_provider.TIMEFRAMES.items()]}


# ---------------------------------------------------------------------------
# 실시간 적중 판정
# ---------------------------------------------------------------------------

@app.get("/api/verify/{query}")
def verify(query: str):
    """오늘 낸 예측이 지금 맞고 있는지 실시간으로 채점하고 이유를 설명합니다.

    장 마감 후 일괄 채점(backtest)과 달리, 장중에 "지금까지는 맞는 중"인지
    바로 확인하기 위한 엔드포인트입니다. 저장된 예측 중 오늘자 가장 오래된 것
    (= 개장 무렵 판단)을 기준으로 삼습니다.
    """
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    provider = get_provider(symbol)
    live = provider.get_realtime_quote(symbol)
    clock = live.get("market_status") or market_clock.status_for(symbol.market)

    today = datetime.now().strftime("%Y-%m-%d")
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE ticker = ? AND predicted_at LIKE ? "
            "ORDER BY id ASC",
            (symbol.key, today + "%"),
        ).fetchall()

    if not rows:
        return {
            "ticker": symbol.key, "symbol": _symbol_payload(symbol),
            "has_prediction": False, "market_status": clock,
            "message": "오늘 저장된 예측이 없습니다. 이 종목을 한 번 조회하면 예측이 기록되고, "
                       "그 다음부터 적중 여부를 추적할 수 있습니다.",
        }

    first = dict(rows[0])
    price = live.get("price")
    prev_close = live.get("prev_close")
    if not price or not prev_close:
        return {
            "ticker": symbol.key, "symbol": _symbol_payload(symbol),
            "has_prediction": True, "market_status": clock,
            "message": "현재 시세를 가져오지 못해 판정할 수 없습니다.",
        }

    change_pct = (price - prev_close) / prev_close * 100
    actual = "up" if price >= prev_close else "down"
    predicted = first["direction"]
    is_hit = (predicted == actual)

    # 요소별로 "혼자 예측했다면" 맞았는지 — 무엇이 맞고 틀렸는지 설명하기 위함
    factor_specs = [
        ("technical", "기술적 지표", first["technical_score"]),
        ("news_sentiment", "뉴스 감성", first["news_score"]),
        ("community_sentiment", "커뮤니티 여론", first["community_score"]),
    ]
    factors = []
    for key, label, score in factor_specs:
        score = score or 0.0
        topic = _particle(label, "은", "는")
        if score == 0:
            factors.append({"key": key, "label": label, "score": 0.0,
                            "said": "중립", "correct": None,
                            "note": f"{label}{topic} 방향 신호를 내지 않았습니다(중립)."})
            continue
        said = "up" if score > 0 else "down"
        correct = (said == actual)
        factors.append({
            "key": key, "label": label, "score": round(score, 4),
            "said": "상승" if said == "up" else "하락",
            "correct": correct,
            "note": (f"{label}{topic} {score:+.3f}로 "
                     f"{'상승' if said == 'up' else '하락'}을 가리켰고, "
                     f"실제는 {'상승' if actual == 'up' else '하락'}이라 "
                     f"{'맞았습니다' if correct else '틀렸습니다'}."),
        })

    hits = [f for f in factors if f["correct"] is True]
    misses = [f for f in factors if f["correct"] is False]

    if is_hit:
        headline = f"예측 적중 중 — {'상승' if actual == 'up' else '하락'} 방향이 맞고 있습니다"
    else:
        headline = f"예측 빗나가는 중 — {'상승' if predicted == 'up' else '하락'}을 예상했지만 실제는 {'상승' if actual == 'up' else '하락'}입니다"

    explanation = (
        f"개장 무렵 {first['probability']}% 확률로 "
        f"{'상승' if predicted == 'up' else '하락'}을 예상했습니다. "
        f"현재가 {price:,.0f}은 전일 종가 {prev_close:,.0f} 대비 {change_pct:+.2f}%로 "
        f"{'상승' if actual == 'up' else '하락'} 중입니다. "
    )
    if hits:
        explanation += "맞춘 요소는 " + ", ".join(f["label"] for f in hits) + "입니다. "
    if misses:
        explanation += "틀린 요소는 " + ", ".join(f["label"] for f in misses) + "입니다. "
    if not clock.get("is_open"):
        explanation += ("다만 지금은 장이 열려 있지 않아 이 판정은 마지막 체결가 기준이며, "
                        "정식 채점은 `python main.py close` 로 이뤄집니다.")

    return {
        "ticker": symbol.key,
        "symbol": _symbol_payload(symbol),
        "has_prediction": True,
        "market_status": clock,
        "is_open": clock.get("is_open", False),
        "predicted_direction": predicted,
        "predicted_probability": first["probability"],
        "predicted_at": first["predicted_at"],
        "actual_direction": actual,
        "current_price": price,
        "prev_close": prev_close,
        "change_pct": round(change_pct, 2),
        "is_hit": is_hit,
        "headline": headline,
        "explanation": explanation,
        "factors": factors,
        "hit_count": len(hits),
        "miss_count": len(misses),
        "tracked_predictions": len(rows),
    }


@app.get("/api/community/{query}")
def community_feed(query: str):
    """실시간(폴링) 커뮤니티 피드 - 프론트에서 몇 초 간격으로 호출합니다."""
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    community = community_crawler.get_community_sentiment(symbol)
    return {
        "ticker": symbol.key,
        "symbol": _symbol_payload(symbol),
        "collected_at": community.collected_at.isoformat(),
        "bullish_ratio": community.bullish_ratio,
        "raw_bullish_ratio": community.raw_bullish_ratio,
        "confidence": community.confidence,
        "post_count": community.post_count,
        "bullish_count": community.bullish_count,
        "bearish_count": community.bearish_count,
        "neutral_count": community.neutral_count,
        "sources": community.sources,
        "source_counts": community.source_counts,
        "scrap_sources": scrap_store.source_summary(symbol.key),
        "recent_posts": [_to_dict(p) for p in community.recent_posts],
    }


@app.get("/api/news/{query}")
def news_feed(query: str, limit: int = 20):
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    items = news_crawler.get_news(symbol, limit=limit)
    return {
        "ticker": symbol.key,
        "symbol": _symbol_payload(symbol),
        "score": news_crawler.aggregate_news_score(items),
        "summary": news_crawler.summarize_news(items),
        "items": [
            {"title": n.title, "source": n.source, "sentiment": n.sentiment_score,
             "keywords": n.matched_keywords, "url": n.url,
             "published_at": n.published_at.isoformat()}
            for n in items
        ],
    }


@app.get("/api/history/{query}")
def history(query: str, limit: int = 20):
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE ticker = ? ORDER BY id DESC LIMIT ?",
            (symbol.key, limit),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/stats")
def stats(ticker: str | None = None):
    return db.get_accuracy_stats(ticker)


# ---------------------------------------------------------------------------
# 성적표 (AI 채점 결과)
# ---------------------------------------------------------------------------

@app.get("/api/scorecard")
def scorecard(request: Request, limit: int = 40):
    """AI가 지금까지 얼마나 맞췄는지 — 전체·시평선별·종목별 + 채점 상세.

    로그인한 경우 그 계정이 만든 예측만 집계합니다.
    """
    user = current_user(request)
    uid = user["id"] if user else None
    overall = db.get_accuracy_stats(user_id=uid)
    return {
        "overall": overall,
        "by_horizon": overall.get("by_horizon", []),
        "by_ticker": db.get_accuracy_by_ticker(limit=15, user_id=uid),
        "recent": db.get_scored_results(limit=limit, user_id=uid),
        "pending": db.get_pending_count(user_id=uid),
        "voided": db.get_void_count(user_id=uid),
        "scoped_to_user": bool(user),
        "last_auto_resolve": _last_auto_resolve,
        "generated_at": datetime.now().isoformat(),
    }


@app.post("/api/scorecard/resolve")
def scorecard_resolve():
    """지금 바로 채점 — 만기가 지난 예측을 전 종목에 대해 처리합니다."""
    global _last_auto_resolve
    result = backtest.auto_resolve_all()
    _last_auto_resolve = result
    # 채점이 끝나면 가중치도 갱신 시도 (표본이 충분할 때만 실제로 바뀝니다)
    if result["scored"]:
        backtest.adjust_weights()
    return {**result, "overall": db.get_accuracy_stats()}


@app.get("/api/scorecard/live")
def scorecard_live(request: Request, ticker: str = "", limit: int = 30):
    """실시간 중간채점 — 아직 만기가 안 된 예측이 **지금 시점에서** 맞고 있는지.

    확정 채점(/scorecard)은 만기가 지나야 나오므로, 10분~1일을 기다리는 동안
    화면이 텅 빕니다. 이 엔드포인트는 진행 중인 예측을 현재가와 대조해
    "지금 채점하면 맞음/틀림"을 알려줍니다.

    주의 — 이건 **잠정치**입니다. 만기까지 가격이 뒤집히면 결과도 뒤집힙니다.
    적중률 통계에는 절대 반영하지 않습니다(확정 채점만 반영).
    """
    user = current_user(request)
    uid = user["id"] if user else None

    symbol = None
    if ticker:
        try:
            symbol = symbol_registry.resolve(ticker)
        except SymbolNotFoundError as exc:
            return _not_found(exc)

    live_price = None
    change_rate = None
    market = None
    if symbol:
        try:
            q = get_provider(symbol).get_realtime_quote(symbol)
            live_price = q.get("price")
            change_rate = q.get("change_rate")
        except Exception:
            pass
        market = market_clock.status_for(symbol.market)

    rows = db.get_live_predictions(
        ticker=symbol.key if symbol else None, user_id=uid, limit=limit)

    hz_label = {h["key"]: h["label"] for h in PREDICTION_HORIZONS}
    hz_label.update({"open": "개장", "5min": "+5분", "30min": "+30분", "close": "마감"})

    items = []
    leading = lagging = unknown = 0
    for r in rows:
        base = r.get("base_price")
        # 다른 종목이 섞여 있으면 그 종목의 현재가를 따로 조회합니다(30초 캐시).
        cur = live_price if (symbol and r["ticker"] == symbol.key) else _raw_price_for(r["ticker"])

        provisional = None
        change_pct = None
        if base and cur:
            change_pct = round((cur - base) / base * 100, 3)
            # 확정 채점과 같은 기준을 씁니다. 여기서 부동소수점 잡음을 상승으로
            # 읽으면, 화면에는 "304.61 → 304.61 인데 맞는 중" 이 뜹니다.
            if backtest.price_moved(base, cur):
                provisional = "up" if cur > base else "down"

        if provisional is None:
            unknown += 1
            hitting = None
        else:
            hitting = provisional == r["direction"]
            leading += 1 if hitting else 0
            lagging += 0 if hitting else 1

        items.append({
            "id": r["id"],
            "ticker": r["ticker"],
            "name": r.get("name") or r["ticker"],
            "horizon_label": r["horizon_label"],
            "label": hz_label.get(r["horizon_label"], r["horizon_label"]),
            "direction": r["direction"],
            "probability": r.get("probability"),
            "base_price": base,
            "current_price": cur,
            "change_pct": change_pct,
            "provisional": provisional,
            "hitting": hitting,
            "predicted_at": r.get("predicted_at"),
            "target_at": r.get("target_at"),
            "seconds_left": r.get("seconds_left"),
        })

    return {
        "symbol": _symbol_payload(symbol) if symbol else None,
        "price": live_price,
        "change_rate": change_rate,
        "market_status": market,
        "server_time": datetime.now().isoformat(),
        "scoped_to_user": bool(user),
        "items": items,
        "summary": {
            "total": len(items),
            "leading": leading,        # 지금 맞고 있는 예측
            "lagging": lagging,        # 지금 틀리고 있는 예측
            "undecided": unknown,      # 보합이라 아직 판정 불가
            "provisional_accuracy": (round(leading / (leading + lagging) * 100, 1)
                                     if (leading + lagging) else None),
        },
    }


@app.get("/api/stats/mine")
def stats_mine(request: Request):
    user = require_user(request)
    return db.get_accuracy_stats(user_id=user["id"])


# ---------------------------------------------------------------------------
# 모의투자 (가상머니)
# ---------------------------------------------------------------------------

_raw_price_cache: dict[str, tuple[float, float | None]] = {}


def _raw_price_for(ticker: str) -> float | None:
    """채점용 현재가 — **환산하지 않은** 원래 통화 그대로입니다.

    예측의 base_price 는 그 종목의 호가 통화로 저장돼 있습니다. 여기에 원화 환산가를
    대면 AAPL 이 240원에서 343,000원으로 뛴 것처럼 보여 전부 '상승 적중'이 됩니다.
    모의투자용 _price_for 와 반드시 구분해서 써야 합니다.
    """
    hit = _raw_price_cache.get(ticker)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    try:
        symbol = symbol_registry.resolve(ticker)
        price = get_provider(symbol).get_realtime_quote(symbol).get("price")
    except Exception:
        price = None
    _raw_price_cache[ticker] = (time.time(), price)
    return price


def _price_for(ticker: str) -> float | None:
    """모의투자 평가용 현재가 — **원화 환산** 값입니다.

    계좌가 원화 단일 통화라, 달러로 호가되는 미국 종목은 환산해야 합니다.
    (환산을 빠뜨리면 100만원으로 AAPL 3,229주를 사는 일이 생깁니다)
    """
    hit = _price_cache.get(ticker)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    try:
        symbol = symbol_registry.resolve(ticker)
        quote = get_provider(symbol).get_realtime_quote(symbol)
        price = fx.to_krw(quote.get("price"), symbol.currency)
    except Exception:
        price = None
    _price_cache[ticker] = (time.time(), price)
    return price


def _deriv_price_for(symbol_key: str) -> float | None:
    """선물·옵션 평가용 현재가 (지수 포인트 / 프리미엄, 환산 없음)."""
    inst = instruments.parse_derivative(symbol_key)
    if not inst:
        return None
    quote = feed.quote(inst)
    return quote["price"] if quote else None


def match_pending_orders(user_id: int) -> list[dict]:
    """미체결 지정가 주문을 현재가와 대조해 체결시킵니다.

    실제 거래소의 체결 엔진 대신 **현재가 도달 여부**만 봅니다. 호가창 잔량이나
    시간 우선 원칙은 재현하지 않으므로, 가격이 스쳐 지나간 경우까지 잡지는 못합니다.
    (조회 시점 사이에 찍힌 고가/저가는 알 수 없습니다)

    지정가는 그 종목의 호가 통화 기준이라 **환산하지 않은 가격**과 비교하고,
    실제 체결은 원화 환산가로 처리합니다.
    """
    filled = []
    for order in paper.get_orders(user_id, status="pending"):
        native = _raw_price_for(order["ticker"])
        if not paper.order_fills_at(order, native):
            continue
        try:
            symbol = symbol_registry.resolve(order["ticker"])
        except SymbolNotFoundError:
            paper.cancel_order(user_id, order["id"])
            continue
        krw = fx.to_krw(native, symbol.currency)
        result = paper.fill_order(user_id, order["id"], symbol, krw)
        if result.get("ok"):
            filled.append({"order_id": order["id"], "ticker": order["ticker"],
                           "name": order["name"], "side": order["side"],
                           "quantity": result.get("quantity"),
                           "price": krw, "limit_price": order["limit_price"]})
    return filled


@app.get("/api/paper")
def paper_portfolio(request: Request):
    """모의투자 계좌 현황 — 잔고·보유종목·미체결·평가손익 (모두 원화 기준)."""
    user = require_user(request)
    # 조회할 때마다 체결 조건을 확인합니다. 화면을 열어두면 지정가가 알아서 체결됩니다.
    just_filled = match_pending_orders(user["id"])
    rate, fx_source = fx.usd_krw()
    return {
        "portfolio": paper.portfolio(user["id"], _price_for),
        # 자동매매가 선물·옵션을 잡으면 여기 담깁니다 (예수금은 주식과 공유).
        "derivatives": derivatives.portfolio(user["id"], _deriv_price_for),
        "orders": paper.get_orders(user["id"], status="pending"),
        "order_history": paper.get_orders(user["id"], limit=20),
        "just_filled": just_filled,
        "trades": paper.get_trades(user["id"], limit=30),
        "fx": {"usd_krw": round(rate, 2), "source": fx_source},
        "fees": {
            "kr_fee_pct": paper.FEE_RATE_KR * 100,
            "kr_tax_pct": paper.TAX_RATE_KR * 100,
            "us_fee_pct": paper.FEE_RATE_US * 100,
        },
    }


@app.get("/api/paper/quote/{query}")
def paper_quote(query: str, request: Request):
    """주문표 전용 시세 — 현재가·호가단위·주문가능 수량/금액을 한 번에.

    주문 화면은 종목을 고르는 즉시 "얼마에 몇 주까지 살 수 있는지"를 보여줘야 해서,
    시세와 계좌 상태를 따로 부르지 않고 여기서 함께 계산합니다.
    """
    user = require_user(request)
    try:
        symbol = symbol_registry.resolve(query)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    live = get_provider(symbol).get_realtime_quote(symbol)
    native = live.get("price")
    rate, fx_source = fx.usd_krw()
    krw = fx.to_krw(native, symbol.currency)

    avail_cash = paper.available_cash(user["id"])
    held = next((h for h in paper.get_holdings(user["id"])
                 if h["ticker"] == symbol.key), None)
    owned = float(held["quantity"]) if held else 0.0
    sellable = paper.available_quantity(user["id"], symbol.key)

    fee_rate = paper.FEE_RATE_US if symbol.market == "US" else paper.FEE_RATE_KR
    max_qty = 0.0
    if krw and krw > 0:
        raw = avail_cash / (krw * (1 + fee_rate))
        max_qty = float(int(raw)) if symbol.market != "US" else round(raw, 4)

    return {
        "symbol": _symbol_payload(symbol),
        "price": native,                      # 호가 통화 기준 (지정가 입력에 쓰는 값)
        "price_krw": krw,                     # 원화 환산 (금액 계산에 쓰는 값)
        "change_rate": live.get("change_rate"),
        "change_amount": live.get("change_amount"),
        "prev_close": live.get("prev_close"),
        "open": live.get("open"), "high": live.get("high"), "low": live.get("low"),
        "volume": live.get("volume"),
        "market_status": live.get("market_status") or market_clock.status_for(symbol.market),
        "tick_size": paper.tick_size(native or 0, symbol.market),
        "fx": {"usd_krw": round(rate, 2), "source": fx_source},
        "account": {
            "available_cash": avail_cash,
            "max_buy_quantity": max_qty,
            "owned_quantity": owned,
            "sellable_quantity": sellable,
            "avg_price": float(held["avg_price"]) if held else None,
        },
    }


class PaperOrder(BaseModel):
    ticker: str
    quantity: float | None = None
    amount: float | None = None
    note: str = ""


@app.post("/api/paper/buy")
def paper_buy(order: PaperOrder, request: Request):
    user = require_user(request)
    try:
        symbol = symbol_registry.resolve(order.ticker)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    price = _price_for(symbol.key)
    result = paper.buy(user["id"], symbol, price, quantity=order.quantity,
                       amount=order.amount, note=order.note or "수동 매수")
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return {**result, "symbol": _symbol_payload(symbol),
            "portfolio": paper.portfolio(user["id"], _price_for)}


@app.post("/api/paper/sell")
def paper_sell(order: PaperOrder, request: Request):
    user = require_user(request)
    try:
        symbol = symbol_registry.resolve(order.ticker)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    price = _price_for(symbol.key)
    result = paper.sell(user["id"], symbol, price, quantity=order.quantity,
                        note=order.note or "수동 매도")
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return {**result, "symbol": _symbol_payload(symbol),
            "portfolio": paper.portfolio(user["id"], _price_for)}


class PaperTicket(BaseModel):
    """증권사 주문표 한 장."""
    ticker: str
    side: str                                  # buy | sell
    order_type: str = "market"                 # market | limit
    quantity: float | None = None
    amount: float | None = None                # 매수 시 금액 지정 (시장가 전용)
    limit_price: float | None = None
    note: str = ""


@app.post("/api/paper/order")
def paper_order(ticket: PaperTicket, request: Request):
    """통합 주문 — 시장가는 즉시 체결, 지정가는 미체결 대기열로."""
    user = require_user(request)
    if ticket.side not in ("buy", "sell"):
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": "주문 구분이 올바르지 않습니다."})
    if ticket.order_type not in ("market", "limit"):
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": "주문 유형이 올바르지 않습니다."})
    try:
        symbol = symbol_registry.resolve(ticket.ticker)
    except SymbolNotFoundError as exc:
        return _not_found(exc)

    label = "매수" if ticket.side == "buy" else "매도"

    if ticket.order_type == "limit":
        rate, _ = fx.usd_krw()
        krw_rate = rate if symbol.currency == "USD" else 1.0
        result = paper.place_limit_order(
            user["id"], symbol, ticket.side,
            quantity=ticket.quantity or 0, limit_price=ticket.limit_price,
            krw_rate=krw_rate)
        if not result.get("ok"):
            return JSONResponse(status_code=400, content=result)
        # 접수 직후 이미 조건을 만족할 수 있습니다 (현재가보다 높게 건 매수 등).
        # 그대로 두면 "왜 안 사지" 가 되므로 즉시 한 번 확인합니다.
        filled = match_pending_orders(user["id"])
        hit = next((f for f in filled if f["order_id"] == result["order_id"]), None)
        return {**result, "filled_now": hit,
                "symbol": _symbol_payload(symbol),
                "portfolio": paper.portfolio(user["id"], _price_for),
                "orders": paper.get_orders(user["id"], status="pending")}

    price = _price_for(symbol.key)
    if ticket.side == "buy":
        result = paper.buy(user["id"], symbol, price, quantity=ticket.quantity,
                           amount=ticket.amount, note=ticket.note or f"시장가 {label}")
    else:
        result = paper.sell(user["id"], symbol, price, quantity=ticket.quantity,
                            note=ticket.note or f"시장가 {label}")
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return {**result, "order_type": "market", "symbol": _symbol_payload(symbol),
            "portfolio": paper.portfolio(user["id"], _price_for),
            "orders": paper.get_orders(user["id"], status="pending")}


@app.get("/api/paper/orders")
def paper_orders(request: Request):
    user = require_user(request)
    filled = match_pending_orders(user["id"])
    return {"orders": paper.get_orders(user["id"], status="pending"),
            "history": paper.get_orders(user["id"], limit=30),
            "just_filled": filled}


@app.delete("/api/paper/orders/{order_id}")
def paper_cancel_order(order_id: int, request: Request):
    user = require_user(request)
    result = paper.cancel_order(user["id"], order_id)
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return {**result, "orders": paper.get_orders(user["id"], status="pending"),
            "portfolio": paper.portfolio(user["id"], _price_for)}


class PaperReset(BaseModel):
    initial_cash: float | None = None


@app.post("/api/paper/reset")
def paper_reset(req: PaperReset, request: Request):
    user = require_user(request)
    result = paper.reset(user["id"], req.initial_cash)
    _price_cache.clear()
    return {**result, "portfolio": paper.portfolio(user["id"], _price_for)}


class CloseRequest(BaseModel):
    tickers: list[str]


@app.post("/api/close")
def close_market(req: CloseRequest):
    resolved, unknown = [], []
    for raw in req.tickers:
        try:
            resolved.append(symbol_registry.resolve(raw))
        except SymbolNotFoundError as exc:
            unknown.append({"query": raw, "message": exc.reason})

    new_weights = backtest.run_daily_close_routine(resolved) if resolved else \
        (db.get_latest_weights() or INITIAL_WEIGHTS)

    return {
        "new_weights": new_weights,
        "resolved": [_symbol_payload(s) for s in resolved],
        "not_found": unknown,
        "stats": db.get_accuracy_stats(),
    }


# ---------------------------------------------------------------------------
# 자동매매 (Auto Trading)
# ---------------------------------------------------------------------------
# 실제로 주문을 내는 기능입니다. 그래서 모든 엔드포인트가 로그인을 요구하고,
# 실거래(live) 전환에는 확인 문구를 따로 받습니다.

class AutoTradeConfig(BaseModel):
    config: dict


class AutoTradeToggle(BaseModel):
    enabled: bool
    confirm: str = ""          # live 모드에서 켤 때 필요한 확인 문구


class BacktestRequest(BaseModel):
    query: str
    days: int = 250
    config: dict | None = None
    initial_cash: float = 10_000_000
    # 위조검증 시행 횟수. 0이면 건너뜁니다 — 한 번에 백테스트를 통째로 다시
    # 돌리므로 시행당 수 초가 걸립니다. 20회 미만은 통과 판정 자체가 불가능해
    # (최소 p = 1/(n+1)) 켤 때는 20 이상을 씁니다.
    falsify: int = 0


def _account_identity(mode: str) -> dict:
    """이 화면의 숫자가 **어느 계좌의 돈인지**를 명시합니다.

    가상 자금과 실제 자금을 같은 화면에 같은 모양으로 띄우면 반드시 헷갈립니다.
    """
    kis = kis_trading.status()
    if mode == broker.LIVE:
        return {"mode": mode, "real_money": True, "label": "KIS 실전",
                "account": kis["account_masked"] or "계좌 미설정",
                "funds": "실제 자금", "server": "한국투자증권 실전 서버"}
    if mode == broker.MOCK:
        return {"mode": mode, "real_money": False, "label": "KIS 모의투자",
                "account": kis["account_masked"] or "계좌 미설정",
                "funds": "가상 자금 (증권사 모의투자)",
                "server": "한국투자증권 모의투자 서버"}
    return {"mode": mode, "real_money": False, "label": "내부 모의계좌",
            "account": "athena.db (이 PC)", "funds": "가상 자금",
            "server": "이 프로그램 안 — 주문이 밖으로 나가지 않습니다"}


# 콘솔(autotrade.html)과의 버전 약속.
# 화면 파일은 디스크에서 항상 최신으로 읽히지만 서버 코드는 재시작해야 바뀝니다.
# 그래서 "버튼은 보이는데 누르면 Not Found"가 반복됩니다 — 화면이 이 숫자를 보고
# 서버가 옛 코드면 스스로 "재시작하세요"를 띄우게 합니다.
# 콘솔이 쓰는 엔드포인트를 추가/변경할 때마다 1씩 올리세요 (autotrade.html 의
# REQUIRED_API 와 짝).
CONSOLE_API_VERSION = 6


# 포지션·계좌 블록 캐시 — 스냅샷의 유일하게 비싼 부분입니다 (시세·잔고 실호출).
# 화면 폴링(7초)마다 새로 조회하면 보유 종목 수만큼 네트워크가 나가고,
# 탭을 여러 개 열면 그 배수가 됩니다. 주문이 나가면 즉시 무효화합니다.
_snap_cache: dict[tuple, tuple[float, dict]] = {}
_SNAP_TTL = 8.0


def _invalidate_snapshot(user_id: int):
    for key in [k for k in _snap_cache if k[0] == user_id]:
        _snap_cache.pop(key, None)


def _positions_block(user_id: int, mode: str, cfg: dict) -> dict:
    """계좌·포지션 조회 (8초 캐시). 실패도 형태를 갖춰 돌려줍니다."""
    key = (user_id, mode)
    now = time.time()
    hit = _snap_cache.get(key)
    if hit and now - hit[0] < _SNAP_TTL:
        return hit[1]

    block: dict = {}
    try:
        brk = broker.get_broker(user_id, mode, cfg)
        brk_positions = brk.positions()
        states = at_store.get_position_states(user_id, mode)
        block["broker_health"] = brk.health()
        block["account"] = brk.account()
        block["position_states"] = states
        block["positions"] = [
            {**p.to_dict(), "managed": autotrade.is_managed(cfg, p.key, states)}
            for p in brk_positions]
        block["day"] = at_store.touch_daily(
            user_id, mode, block["account"].get("total_value") or 0)
    except Exception as exc:
        block["account_error"] = f"{type(exc).__name__}: {exc}"

    _snap_cache[key] = (now, block)
    return block


def _autotrade_snapshot(user_id: int, include_positions: bool = True) -> dict:
    """콘솔이 한 번에 그릴 수 있는 상태 묶음.

    모든 수치는 **현재 선택된 계좌(mode) 범위**로만 집계합니다.
    include_positions=False(lite)는 시세·잔고 조회를 전부 건너뜁니다 —
    포지션을 그리지 않는 페이지(AI·페니·백테스트)가 씁니다.
    """
    # 옛 설정(초단타가 콘솔 대상을 덮어쓰던 시절) 한 번 갈라주기.
    # 엔진뿐 아니라 화면에서도 갈라진 목록이 보여야 해서 여기서도 부릅니다.
    cfg = autotrade.split_legacy_universe(user_id, at_store.get_config(user_id))
    mode = cfg.get("mode", "paper")
    payload = {
        "api_version": CONSOLE_API_VERSION,
        "config": cfg,
        "state": cfg.get("state", at_store.STOPPED),
        "state_label": at_store.STATE_LABELS.get(cfg.get("state"), cfg.get("state")),
        "state_reason": cfg.get("state_reason", ""),
        "enabled": bool(cfg.get("enabled")),
        "modes": broker.available_modes(),
        "limits": risk.describe_limits(cfg),
        # 앙상블·보호장치 (engine/ensemble.py, engine/protections.py).
        # 잠금 조회는 equity=0 으로 부릅니다 — 낙폭·부진 판정(자산 필요)은
        # 엔진 회전이 실제 계좌값으로 하고, 여기는 쿨다운·손절 잠금만 보입니다.
        "algo_modes": ensemble.describe(),
        "protections": protections.describe(cfg),
        "protection_locks": protections.evaluate(user_id, mode, cfg).to_dict(),
        "loop": autotrade.loop.status(),
        "summary": at_store.summary(user_id, mode=mode),
        "kis": kis_trading.status(),
        "defaults": at_store.DEFAULT_CONFIG,
        "identity": _account_identity(mode),
    }

    payload["open_orders"] = at_store.open_orders(user_id, mode=mode)
    payload["execution_quality"] = at_store.execution_quality(user_id, mode=mode)

    # 매매 대상마다 'AI가 넣은 것인지 + 왜인지'를 붙여 화면에서 바로 보이게 합니다
    picks = {r["symbol"]: r for r in at_store.get_recommendations(user_id)}
    payload["universe_detail"] = [
        {"key": key,
         "auto": bool(picks.get(key, {}).get("picked")),
         "score": picks.get(key, {}).get("score"),
         "rank": picks.get(key, {}).get("rank"),
         "name": picks.get(key, {}).get("name") or key,
         "reasons": picks.get(key, {}).get("reasons") or []}
        for key in (cfg.get("universe") or [])
    ]
    payload["auto_tracking"] = autotrade.auto_tracking_enabled(cfg)
    payload["ai_tracking"] = autotrade.tracking_status(user_id, cfg)

    # 초단타는 자기 매매 대상을 따로 들고 있습니다 — 위 universe 와 섞지 않습니다
    payload["scalp_universe"] = autotrade.scalp_universe(cfg)

    if include_positions:
        payload.update(_positions_block(user_id, mode, cfg))
    return payload


@app.get("/api/autotrade")
def autotrade_status(request: Request, lite: int = 0):
    """콘솔 상태. lite=1 이면 시세·잔고 조회를 건너뜁니다 (포지션 없는 페이지용)."""
    user = require_user(request)
    return _autotrade_snapshot(user["id"], include_positions=not lite)


@app.post("/api/autotrade/config")
def autotrade_save_config(req: AutoTradeConfig, request: Request):
    user = require_user(request)
    patch = dict(req.config or {})
    before = at_store.get_config(user["id"])

    # 실거래로 바꾸는 것은 설정 변경이 아니라 '권한 상승'입니다.
    # 여기서는 모드 문자열만 저장하고, 실제 가동은 enable 에서 한 번 더 확인합니다.
    if patch.get("mode") == "live":
        st = kis_trading.status()
        if not (st["keys_configured"] and st["account_configured"]):
            return JSONResponse(status_code=400, content={
                "ok": False,
                "error": "실거래 모드는 KIS APP KEY/SECRET 과 계좌번호가 필요합니다."})

    # 화면에서 매매 대상을 바꾸는 것은 '사람의 결정'입니다. 그대로 기록해 두어야
    # AI 갱신이 그 종목을 지우지 않습니다 (추천 목록과 대조해 추측하면, AI 가
    # 한 번 같이 뽑은 순간 사람이 넣었다는 사실이 사라집니다).
    if "universe" in patch:
        requested = [str(s) for s in (patch.get("universe") or [])]
        was = set(before.get("universe") or [])
        manual = set(before.get("manual_universe") or [])
        manual |= {s for s in requested if s not in was}       # 추가한 것
        manual &= set(requested)                               # 뺀 것은 잊습니다
        patch["manual_universe"] = sorted(manual)

    mode_changed = "mode" in patch and patch["mode"] != before.get("mode")
    cfg = at_store.save_config(user["id"], patch)

    stopped = False
    if mode_changed and before.get("enabled"):
        # 돌아가는 중에 계좌를 갈아끼우면 안 됩니다. 모의계좌 포지션을 들고
        # 실계좌 로직으로 넘어가는 순간 무엇이 진짜인지 알 수 없게 됩니다.
        at_store.set_enabled(user["id"], False, "계좌 변경으로 자동 중지")
        stopped = True
        at_store.log_event(
            user["id"], "stop",
            f"계좌를 {before.get('mode')} → {patch['mode']} 로 바꿔 자동매매를 중지했습니다. "
            f"확인 후 다시 시작하세요.", level="warn")

    # 설정을 고쳤다는 것은 거부 사유를 해결했다는 뜻일 수 있습니다.
    # 재시도 대기를 풀어, 30분을 기다리지 않고 다음 회전에 바로 시도합니다.
    autotrade.clear_broker_reject(user["id"])

    at_store.log_event(user["id"], "config", "설정을 변경했습니다.",
                       detail={"patch": patch})
    return {"ok": True, **_autotrade_snapshot(user["id"], include_positions=False),
            "config": cfg, "mode_changed": mode_changed, "stopped": stopped}


@app.post("/api/autotrade/enable")
def autotrade_enable(req: AutoTradeToggle, request: Request):
    user = require_user(request)
    cfg = at_store.get_config(user["id"])

    if req.enabled:
        mode = cfg.get("mode", "paper")
        brk = broker.get_broker(user["id"], mode, cfg)
        health = brk.health()
        if not health.get("ready"):
            return JSONResponse(status_code=400, content={
                "ok": False, "error": f"브로커를 쓸 수 없습니다 — {health.get('detail')}"})

        # 실제 자금이 움직이는 모드는 확인 문구를 받습니다.
        # 버튼 하나만으로 실계좌 자동매매가 켜지는 일은 없어야 합니다.
        if mode == broker.LIVE and req.confirm.strip().upper() != "LIVE":
            return JSONResponse(status_code=400, content={
                "ok": False, "needs_confirm": True,
                "error": "실거래를 켜려면 확인란에 LIVE 를 입력해야 합니다."})

        at_store.set_enabled(user["id"], True)
        at_store.log_event(user["id"], "start",
                           f"자동매매를 시작했습니다 ({broker.MODE_LABELS.get(mode, mode)})",
                           level="trade", detail={"mode": mode})
        autotrade.loop.start()
    else:
        at_store.set_enabled(user["id"], False)
        at_store.log_event(user["id"], "stop", "자동매매를 중지했습니다.", level="warn")

    _invalidate_snapshot(user["id"])
    return {"ok": True, **_autotrade_snapshot(user["id"])}


@app.post("/api/autotrade/run")
def autotrade_run_now(request: Request):
    """지금 한 회전 실행 — 설정을 바꾼 뒤 바로 확인할 때 씁니다."""
    user = require_user(request)
    result = autotrade.run_once(user["id"], force=True)
    _invalidate_snapshot(user["id"])          # 회전으로 포지션이 바뀌었을 수 있습니다
    return {"result": result, **_autotrade_snapshot(user["id"])}


@app.post("/api/autotrade/kill")
def autotrade_kill(request: Request):
    """킬 스위치 — 신규 진입을 즉시 전면 차단합니다 (청산은 계속 동작)."""
    user = require_user(request)
    at_store.save_config(user["id"], {"kill_switch": True})
    at_store.set_state(user["id"], at_store.HALTED, "킬 스위치 (수동)")
    at_store.log_event(user["id"], "halt", "킬 스위치를 눌렀습니다 — 신규 진입 차단",
                       level="warn")
    return {"ok": True, **_autotrade_snapshot(user["id"], include_positions=False)}


@app.post("/api/autotrade/resume")
def autotrade_resume(request: Request):
    user = require_user(request)
    at_store.save_config(user["id"], {"kill_switch": False})
    cfg = at_store.get_config(user["id"])
    at_store.set_state(user["id"],
                       at_store.RUNNING if cfg.get("enabled") else at_store.STOPPED, "")
    at_store.log_event(user["id"], "resume", "킬 스위치를 해제했습니다.")
    return {"ok": True, **_autotrade_snapshot(user["id"], include_positions=False)}


@app.get("/api/autotrade/events")
def autotrade_events(request: Request, limit: int = 80, after_id: int = 0,
                     kind: str = ""):
    user = require_user(request)
    kinds = [k for k in kind.split(",") if k] or None
    return {"events": at_store.get_events(user["id"], limit=limit,
                                          kinds=kinds, after_id=after_id)}


@app.get("/api/autotrade/orders")
def autotrade_orders(request: Request, limit: int = 50):
    user = require_user(request)
    mode = at_store.get_config(user["id"]).get("mode", "paper")
    # 계좌가 다르면 완전히 다른 원장입니다 (가상 자금 성적과 실제 자금 성적을 섞지 않습니다)
    return {"orders": at_store.get_orders(user["id"], limit=limit, mode=mode),
            "open_orders": at_store.open_orders(user["id"], mode=mode),
            "summary": at_store.summary(user["id"], mode=mode),
            "execution_quality": at_store.execution_quality(user["id"], mode=mode),
            "daily": at_store.get_daily_history(user["id"], limit=14, mode=mode)}


@app.post("/api/autotrade/orders/{order_id}/cancel")
def autotrade_cancel_order(order_id: int, request: Request):
    """미체결 주문 수동 취소 — 자동 취소 시간을 기다리지 않고 지금 거둡니다."""
    user = require_user(request)
    cfg = at_store.get_config(user["id"])
    mode = cfg.get("mode", "paper")
    record = next((o for o in at_store.open_orders(user["id"], mode=mode)
                   if o["id"] == order_id), None)
    if not record:
        return JSONResponse(status_code=404,
                            content={"ok": False, "error": "취소할 미체결 주문이 없습니다."})

    brk = broker.get_broker(user["id"], cfg.get("mode", "paper"), cfg)
    result = brk.cancel(record)
    _invalidate_snapshot(user["id"])
    if result.ok:
        at_store.update_order(order_id, status="cancel_requested", reason="사용자 취소")
        at_store.log_event(user["id"], "reject",
                           f"{record['name'] or record['symbol']} 주문을 수동 취소했습니다.",
                           level="warn", symbol=record["symbol"])
    else:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": result.error})
    return {"ok": True, **_autotrade_snapshot(user["id"], include_positions=False)}


@app.post("/api/autotrade/recommend")
def autotrade_recommend(request: Request):
    """AI 다중 팩터 추천 — 지금 시장에서 무엇을 매매 대상으로 삼을지 계산합니다.

    편입은 하지 않습니다. 결과만 돌려주고, 넣을지는 사용자가 정하거나
    `auto_universe` 가 켜져 있으면 엔진이 주기적으로 자동 편입합니다.
    """
    user = require_user(request)
    cfg = at_store.get_config(user["id"])
    try:
        brk = broker.get_broker(user["id"], cfg.get("mode", "paper"), cfg)
        account = brk.account()
    except Exception as exc:
        account = {"available_cash": 0, "total_value": 0, "error": str(exc)}

    ranked = autotrade.recommend_universe(user["id"], cfg, account)
    threshold = float(cfg.get("auto_universe_min_score", 0.55))
    picked = [r.key for r in ranked if r.tradable and r.score >= threshold][
        :int(cfg.get("auto_universe_size", 5))]
    at_store.save_recommendations(user["id"], ranked, picked)

    session = market_clock.status_for("KOSPI")
    return {
        "recommendations": [r.to_dict() for r in ranked],
        "would_pick": picked,
        "threshold": threshold,
        "factors": recommender.describe_factors(),
        "regime": ranked[0].regime if ranked else "",
        "session": session.get("label", ""),
        "account": {"available_cash": account.get("available_cash"),
                    "total_value": account.get("total_value")},
    }


class TrackToggle(BaseModel):
    enabled: bool


@app.post("/api/autotrade/track")
def autotrade_track(req: TrackToggle, request: Request):
    """AI 자동 추적 시작/중지 — **매매와는 별개**입니다.

    추적은 주기적으로 시장을 훑어 추천을 갱신하고 매매 대상을 갈아끼우는
    것까지만 합니다. 실제 주문은 [시작] 으로 자동매매를 켜야 나갑니다.
    """
    user = require_user(request)
    at_store.save_config(user["id"], {"auto_universe": req.enabled})
    if req.enabled:
        # 다음 sweep(15초 안)에서 바로 첫 추천을 계산하게 합니다
        autotrade.reset_universe_timer(user["id"])
        autotrade.loop.start()
        at_store.log_event(user["id"], "screen",
                           "AI 자동 추적을 시작했습니다 — 곧 첫 추천을 계산합니다 "
                           "(이후 설정한 주기마다 자동 갱신).")
    else:
        at_store.log_event(user["id"], "screen", "AI 자동 추적을 중지했습니다.")
    return {"ok": True, **_autotrade_snapshot(user["id"], include_positions=False)}


@app.get("/api/autotrade/recommend")
def autotrade_recommend_saved(request: Request):
    """마지막 추천 결과 (새로고침 후에도 근거를 볼 수 있게 저장해 둡니다)."""
    user = require_user(request)
    return {"recommendations": at_store.get_recommendations(user["id"]),
            "factors": recommender.describe_factors()}


class ScreenerRequest(BaseModel):
    markets: list[str] | None = None       # ["KR"] | ["US"] | 둘 다
    limit: int = 30
    refresh: bool = False                  # 캐시 무시하고 다시 훑기


@app.post("/api/autotrade/screener")
def autotrade_screener(req: ScreenerRequest, request: Request):
    """페니주식 후보를 시장에서 찾아 **내 자금에 맞게** 순위화합니다.

    "좋은 종목"이 아니라 "이 돈으로 살 수 있고, 되팔 수 있는 종목"을 고릅니다.
    """
    user = require_user(request)
    cfg = at_store.get_config(user["id"])
    scalp = scalping.clamp_config(cfg.get("scalp") or {})
    markets = req.markets or scalp.get("markets") or ["KR"]

    try:
        brk = broker.get_broker(user["id"], cfg.get("mode", "paper"), cfg)
        account = brk.account()
    except Exception as exc:
        account = {"available_cash": 0, "total_value": 0, "error": str(exc)}

    # 화면에서 사람이 누른 경우이므로 고저가 보강(enrich)을 켭니다 —
    # '가격 변동폭' 타점을 보려면 개별 조회가 필요하고, 몇 초 걸립니다.
    # (자동 갱신 루프는 반대로 꺼서, WebSocket 틱이 주는 고저가를 씁니다)
    scan = screener.scan(
        markets=markets,
        kr_price=tuple(scalp["kr_price_range"]),
        us_price=tuple(scalp["us_price_range"]),
        limit=max(10, min(req.limit * 2, 80)),
        use_cache=not req.refresh,
        rank_basis=scalp.get("rank_basis", "vol_increase"),
        enrich=True)

    ranked = scalping.recommend(scan.candidates, account.get("available_cash") or 0,
                                account.get("total_value") or 0, scalp)

    # 장이 닫혀 있으면 거래대금이 0이라 전부 '부적합'으로 나옵니다.
    # 그걸 설명 없이 보여주면 "쓸 종목이 하나도 없다"고 오해합니다.
    session = market_clock.status_for("KOSPI" if "KR" in markets else "US")
    notes = list(scan.errors)
    if not session.get("is_regular"):
        notes.append(f"지금은 {session.get('label', '장 마감')}입니다 — 거래대금이 0으로 집계돼 "
                     f"대부분 '부적합'으로 나옵니다. 정규장 중에 다시 확인하세요.")

    return {
        "candidates": ranked[:req.limit],
        "scanned": len(scan.candidates),
        "errors": notes,
        "session": session.get("label", ""),
        "markets": markets,
        "account": {"available_cash": account.get("available_cash"),
                    "total_value": account.get("total_value")},
        "scalp": {k: v for k, v in scalp.items() if not k.startswith("_")},
        "clamped": scalp.get("_clamped", []),
        "warnings": scalping.risk_warnings(scalp),
        "hard_limits": scalping.describe_hard_limits(),
        "sources": scan.sources or ["조회 실패"],
        "stream": _tick_stream_status(),
    }


def _tick_stream_status() -> dict:
    """실시간 체결 스트림 상태. 실패해도 화면은 떠야 합니다."""
    try:
        from data_sources import kis_realtime
        return kis_realtime.status()
    except Exception as exc:
        return {"running": False, "connected": False, "error": str(exc)}


@app.get("/api/autotrade/screener/limits")
def autotrade_screener_limits(request: Request):
    """설정으로 못 뚫는 한도 + 경고 문구 (화면에 항상 띄웁니다)."""
    user = require_user(request)
    scalp = scalping.clamp_config(at_store.get_config(user["id"]).get("scalp") or {})
    return {"hard_limits": scalping.describe_hard_limits(),
            "warnings": scalping.risk_warnings(scalp),
            "defaults": scalping.DEFAULT_SCALP,
            "scalp": {k: v for k, v in scalp.items() if not k.startswith("_")},
            "stream": _tick_stream_status()}


@app.get("/api/autotrade/scalp/chart")
def autotrade_scalp_chart(symbol: str, request: Request,
                          interval: int = 1, count: int = 300):
    """초단타 차트 — 초봉 + 보조지표 + **내가 산 지점/판 지점**.

    초봉은 어떤 공개 API 도 주지 않습니다. 실시간 체결(WebSocket)을 모아
    직접 만들기 때문에, **스트림이 붙어 있는 동안 쌓인 만큼만** 나옵니다.
    아직 없으면 분봉으로 대신 그리고 그 사실을 `source` 에 표시합니다 —
    초봉인 척 분봉을 그리면 몇 틱 판단이 통째로 어긋납니다.
    """
    user = require_user(request)
    cfg = at_store.get_config(user["id"])
    scalp = scalping.clamp_config(cfg.get("scalp") or {})
    mode = cfg.get("mode", "paper")

    code = str(symbol).strip()
    interval = max(min(int(interval or 1), 300), 1)
    count = max(min(int(count or 300), 1_000), 10)

    # 종목 해석 실패를 조용히 넘기면 "차트가 비어 있는데 왜인지 모르는" 상태가
    # 됩니다. 지정 종목 티커를 잘못 적는 경우가 실제로 가장 흔합니다.
    inst, resolve_error = None, ""
    try:
        inst = instruments.resolve(code)
    except SymbolNotFoundError as exc:
        hint = ", ".join(s.get("key", "") for s in (getattr(exc, "suggestions", None) or [])[:3])
        resolve_error = (f"'{code}' 로 상장된 종목을 찾을 수 없습니다."
                         + (f" 혹시 {hint} 인가요?" if hint else ""))
    except Exception as exc:
        resolve_error = f"'{code}' 종목 해석 실패: {exc}"

    market = "US" if (inst and inst.market == "US") else "KR"

    source, bars, note = "tick", [], resolve_error
    try:
        bars = kis_realtime.bars(code, interval, count)
    except Exception as exc:
        note = note or f"실시간 봉을 읽지 못했습니다: {exc}"

    if not bars:
        # 폴백 — 분봉. 초단타 판단용이 아니라 '모양이라도 보기' 위한 것입니다.
        source = "minute"
        if not note:
            session = "미국 정규장" if market == "US" else "정규장"
            note = (f"아직 쌓인 체결이 없습니다 — 분봉으로 대신 그립니다. "
                    f"{session} 중 초단타를 켜면 초봉이 쌓입니다.")
        if inst is not None:
            frame = feed.bars(inst, "minute", count=count)
            rows = list(frame.iterrows()) if frame is not None and len(frame) else []
            for position, (stamp, row) in enumerate(rows):
                # 인덱스가 시각이 아니면(제공처가 바뀌는 경우) 전부 0초로 뭉쳐서
                # 시간축이 무너집니다. 그때는 순번을 분으로 환산해 순서만 지킵니다.
                if hasattr(stamp, "hour"):
                    second = stamp.hour * 3600 + stamp.minute * 60
                else:
                    second = position * 60
                bars.append({"t": second, "o": float(row["open"]), "h": float(row["high"]),
                             "l": float(row["low"]), "c": float(row["close"]),
                             "v": float(row.get("volume") or 0), "n": 1})

    chart = scalping.chart_series(bars, scalp, market)

    # --- 내 체결 지점을 봉 인덱스에 붙입니다 ---
    # 화면은 봉을 인덱스로 그리므로(체결 없는 초는 봉 자체가 없습니다),
    # 마커도 시각이 아니라 인덱스로 줘야 자리가 맞습니다.
    times = chart["series"]["t"]
    fills = at_store.fills_for(user["id"], mode, code)
    for fill in fills:
        fill["bar_index"] = _nearest_bar_index(times, fill["sec_of_day"])

    # 해외 실시간은 계정에 따라 지연되어 옵니다. 지연된 초봉을 그냥 보여주면
    # 과거를 현재로 착각하므로, 몇 초 늦은 값인지 그대로 표시합니다.
    last_tick = kis_realtime.tick(code) or {}
    delay_sec = float(last_tick.get("delay_sec") or 0)
    if source == "tick" and delay_sec > scalping.HARD_LIMITS["max_quote_delay_sec"]:
        note = (f"시세가 약 {delay_sec:.0f}초 지연됩니다 — 이 지연으로는 틱 매매를 "
                f"하지 않습니다(진입이 자동으로 차단됩니다). 차트는 참고용입니다.")

    state = (at_store.get_position_states(user["id"], mode) or {}).get(code) or {}
    return {
        "symbol": code,
        "name": (inst.name if inst else code),
        "market": market,
        "source": source,                 # tick(초봉) | minute(분봉 폴백)
        "interval_sec": interval,
        "delay_sec": delay_sec,
        "note": note,
        "buffered_bars": kis_realtime.bar_count(code),
        **chart,
        "fills": fills,
        "position": {
            "holding": bool(state),
            "entry_price": state.get("entry_price"),
            "stop_price": state.get("stop_price"),
            "target_price": state.get("target_price"),
            "quantity": state.get("quantity"),
            "opened_at": state.get("opened_at"),
        },
        "stream": _tick_stream_status(),
    }


def _nearest_bar_index(times: list, second: int) -> int | None:
    """체결 시각과 가장 가까운 봉의 인덱스. 봉이 없으면 None."""
    if not times or not second:
        return None
    best, best_gap = None, None
    for i, t in enumerate(times):
        gap = abs(t - second)
        if best_gap is None or gap < best_gap:
            best, best_gap = i, gap
    return best


@app.get("/api/autotrade/scalp/economics")
def autotrade_scalp_economics(price: float, request: Request):
    """이 가격대에서 지금 설정으로 매매하면 산수가 맞는가.

    설정 화면에서 값을 바꿀 때마다 "그래서 몇 % 이겨야 하는가"를 바로
    보여주기 위한 것입니다. 이 숫자를 모르고 틱을 정하면 안 됩니다.
    """
    user = require_user(request)
    scalp = scalping.clamp_config(at_store.get_config(user["id"]).get("scalp") or {})
    return scalping.tick_economics(float(price), scalp)


@app.get("/api/autotrade/scalp/state")
def autotrade_scalp_state(request: Request):
    """페니 초단타가 지금 돌고 있는지 + 종목별로 무엇을 하는 중인지.

    관망중 / 매수 준비중 / 매도 준비중 — 이 셋은 **엔진이 마지막 회전에서
    실제로 내린 판단**입니다. 이 호출은 시세를 다시 조회하지 않습니다.
    화면 때문에 시세를 새로 부르면 몇 틱 승부에 써야 할 REST 한도를 표시용으로
    태우고, 그렇게 만든 숫자는 엔진이 본 것과 달라집니다. 대신 판단이 얼마나
    지났는지를 `age_sec` 으로 같이 줍니다 — 늦은 화면은 괜찮지만, 늦은 줄
    모르는 화면은 안 됩니다.
    """
    user = require_user(request)
    return autotrade.scalp_status(user["id"], at_store.get_config(user["id"]))


@app.get("/api/autotrade/broker/check")
def autotrade_broker_check(request: Request):
    """실계좌 연결 진단 — 실제로 호출해 보고 무엇이 되는지 알려줍니다(주문 없음).

    "키를 넣었는데 왜 안 되는지"를 화면에서 바로 볼 수 있게 하려는 것입니다.
    """
    require_user(request)
    return kis_trading.diagnose()


@app.post("/api/autotrade/close/{symbol}")
def autotrade_close_position(symbol: str, request: Request):
    """수동 청산 — 자동매매가 켜져 있어도 사람이 언제든 끊을 수 있어야 합니다."""
    user = require_user(request)
    cfg = at_store.get_config(user["id"])
    inst = instruments.try_resolve(symbol)
    if inst is None:
        return JSONResponse(status_code=404,
                            content={"ok": False, "error": "종목을 찾을 수 없습니다."})

    brk = broker.get_broker(user["id"], cfg.get("mode", "paper"), cfg)
    coid = broker.make_client_order_id(user["id"], inst.key, "manual_close")
    order = brk.close(inst, note="수동 청산", client_order_id=coid)
    at_store.record_order(user["id"], {
        "client_order_id": coid, "broker_mode": cfg.get("mode", "paper"),
        "broker_order_id": order.broker_order_id, "symbol": inst.key,
        "name": inst.name, "asset_class": inst.asset_class, "action": "close",
        "quantity": order.quantity, "price": order.price, "price_krw": order.price_krw,
        "fee": order.fee, "realized_pnl": order.realized_pnl,
        "status": order.status if order.ok else "rejected",
        "reason": "수동 청산" if order.ok else (order.error or ""),
        "intended_price": order.intended_price or order.price,
        "filled_quantity": order.filled_quantity,
        "avg_fill_price": order.avg_fill_price,
        "slippage_bps": order.slippage_bps,
    })
    # 접수만 된 주문(실계좌)은 아직 포지션이 남아 있으므로 상태를 지우지 않습니다.
    if order.ok and order.status == "filled":
        at_store.clear_position_state(user["id"], cfg.get("mode", "paper"), inst.key)
    if order.ok:
        at_store.log_event(user["id"], "exit", f"{inst.name} 수동 청산", level="trade",
                           symbol=inst.key, name=inst.name, detail=order.to_dict())
    else:
        return JSONResponse(status_code=400, content={"ok": False, "error": order.error})
    _invalidate_snapshot(user["id"])          # 청산으로 포지션이 바뀌었습니다
    return {"ok": True, "order": order.to_dict(),
            **_autotrade_snapshot(user["id"])}


@app.post("/api/autotrade/backtest")
def autotrade_backtest(req: BacktestRequest, request: Request):
    """설정을 켜기 전에 과거 데이터로 검증합니다 (라이브와 같은 신호·청산 규칙).

    `falsify>0` 이면 예측 가능한 순서를 없앤 가짜 가격에 같은 전략을 통과시켜
    "이 성과가 가짜 데이터와 구분되는가"를 함께 돌려줍니다. 백테스트를 시행 수만큼
    다시 돌리므로 수십 초가 걸립니다 — 기본은 꺼져 있습니다.
    """
    user = require_user(request)
    cfg = {**at_store.get_config(user["id"]), **(req.config or {})}
    result = autotrade.simulate(req.query, cfg, days=max(60, min(req.days, 1000)),
                                initial_cash=req.initial_cash,
                                falsify=max(0, min(int(req.falsify or 0), 40)))
    if not result.get("ok"):
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/api/autotrade/instrument/{query}")
def autotrade_instrument(query: str, request: Request):
    """유니버스에 넣기 전 종목 확인 — 자산군·승수·증거금·호가단위를 보여줍니다."""
    require_user(request)
    inst = autotrade._resolve_universe_item(query)
    if inst is None:
        return JSONResponse(status_code=404, content={
            "error": "SYMBOL_NOT_FOUND",
            "message": f"'{query}' 을(를) 찾을 수 없습니다."})
    quote = feed.quote(inst)
    return {
        "instrument": inst.to_dict(),
        "quote": quote,
        "tick_size": inst.tick_size(quote["price"]) if quote else None,
        "unit_cost_krw": (inst.margin_required(quote["price_krw"], 1, "buy")
                          if quote else None),
        "days_to_expiry": feed.days_to_expiry(inst),
        "market": feed.market_status(inst),
    }


# 자동매매 콘솔 — Next.js 없이 이 서버만으로 열리는 독립 화면들입니다.
# 자동매매는 프론트엔드가 꺼져 있어도 조작할 수 있어야 합니다
# (돌고 있는 자동매매를 멈추려는데 화면이 안 열리면 그게 사고입니다).
# 한 페이지에 전부 몰아넣지 않고 역할별로 나눴습니다. 공통 스타일·헤더는
# /static/at_common.{css,js} 로 공유합니다.
AUTOTRADE_PAGES = {
    "": "autotrade.html",           # 메인 — 계좌·포지션·설정·안전장치·연결상태
    "ai": "at_ai.html",             # AI 추천·자동 추적·매매 대상
    "penny": "at_penny.html",       # 페니주식 초단타
    "deriv": "at_deriv.html",       # 선물·옵션 (전용 화면은 준비 중)
    "backtest": "at_backtest.html", # 백테스트·주문 원장·기타
}


def _serve_console_page(name: str):
    filename = AUTOTRADE_PAGES.get(name)
    if not filename:
        return HTMLResponse("<h1>없는 페이지입니다.</h1>", status_code=404)
    page = WEB_DIR / filename
    if page.exists():
        return FileResponse(page)
    return HTMLResponse(f"<h1>{filename} 을 찾을 수 없습니다.</h1>", status_code=404)


@app.get("/autotrade", include_in_schema=False)
def serve_autotrade_console():
    return _serve_console_page("")


@app.get("/autotrade/{page}", include_in_schema=False)
def serve_autotrade_subpage(page: str):
    return _serve_console_page(page)


# ---------------------------------------------------------------------------
# 프론트엔드 안내
# ---------------------------------------------------------------------------
# 실제 화면은 Next.js(3000번)가 그립니다. 이 서버(8000번)는 분석·API 담당입니다.
# 그런데 예전 습관이나 북마크로 8000번에 들어오는 경우가 있어서, 여기로 들어와도
# 막다른 길이 되지 않도록 3000번의 알맞은 경로로 넘겨줍니다.

FRONTEND_PORT = 3000
FRONTEND_ORIGIN = f"http://localhost:{FRONTEND_PORT}"

# 해시(#paper)는 서버로 전송되지 않으므로 브라우저에서 변환해야 합니다.
HASH_ROUTES = {"": "/", "#": "/", "#analysis": "/", "#score": "/score",
               "#paper": "/paper", "#login": "/login"}


def frontend_alive(timeout: float = 0.35) -> bool:
    """Next.js 개발 서버가 떠 있는지 TCP 연결로 확인."""
    try:
        with socket.create_connection(("127.0.0.1", FRONTEND_PORT), timeout):
            return True
    except OSError:
        return False


def _bridge_page(target_path: str = "") -> str:
    """3000번으로 넘겨주는 최소 페이지 (해시 → 경로 변환 포함)."""
    import json
    forced = json.dumps(target_path)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>이동 중… — 아테나 시그널</title>
<style>
 html,body{{height:100%;margin:0;background:#0c0c0d;color:#e8e6e3;
   font-family:system-ui,'Malgun Gothic',sans-serif;display:grid;place-items:center}}
 .b{{text-align:center;line-height:1.9}} .m{{color:#8a8681;font-size:13px}}
 a{{color:#d9a441}}
</style></head><body><div class="b">
 <div style="font-size:15px;letter-spacing:.08em">ATHENA SIGNAL</div>
 <div class="m">화면으로 이동하고 있습니다…</div>
 <div class="m">넘어가지 않으면 <a id="lnk" href="{FRONTEND_ORIGIN}">{FRONTEND_ORIGIN}</a></div>
</div><script>
 var MAP = {json.dumps(HASH_ROUTES, ensure_ascii=False)};
 var forced = {forced};
 var path = forced || MAP[(location.hash || "").toLowerCase()] || "/";
 var url = "{FRONTEND_ORIGIN}" + path + location.search;
 document.getElementById("lnk").href = url;
 document.getElementById("lnk").textContent = url;
 location.replace(url);
</script></body></html>"""


def _frontend_down_page() -> str:
    """3000번이 꺼져 있을 때 — 무한 대기 대신 무엇을 하면 되는지 알려줍니다."""
    legacy = ('<p class="m">급하면 <a href="/legacy">구버전 화면(/legacy)</a>도 있습니다. '
              '기능이 제한적입니다.</p>') if WEB_DIR.exists() else ""
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>웹 화면이 꺼져 있습니다 — 아테나 시그널</title>
<style>
 html,body{{height:100%;margin:0;background:#0c0c0d;color:#e8e6e3;
   font-family:system-ui,'Malgun Gothic',sans-serif;display:grid;place-items:center}}
 .b{{max-width:560px;padding:32px;line-height:1.85}}
 h1{{font-size:19px;margin:0 0 6px}} .m{{color:#8a8681;font-size:13.5px}}
 code{{background:#1a1a1c;padding:3px 8px;border-radius:4px;
   font-family:Consolas,monospace;font-size:12.5px;color:#d9a441}}
 a{{color:#d9a441}}
</style></head><body><div class="b">
 <h1>분석 엔진은 켜져 있는데, 웹 화면이 꺼져 있습니다</h1>
 <p class="m">여기(8000번)는 분석·API 서버입니다. 실제 화면은 3000번에서 그려집니다.</p>
 <p class="m"><code>start.bat</code> 을 실행하면 둘 다 같이 켜집니다.<br>
    이미 실행 중이라면 <b>'Athena Signal - 웹'</b> 창이 닫히지 않았는지 확인해 주세요.</p>
 <p class="m">직접 켜려면 <code>cd frontend</code> 후 <code>npm run dev</code></p>
 <p class="m">켠 다음 <a href="/">이 페이지를 새로고침</a>하면 자동으로 넘어갑니다.</p>
 {legacy}
</div></body></html>"""


def _frontend_response(path: str = ""):
    if frontend_alive():
        return HTMLResponse(_bridge_page(path))
    return HTMLResponse(_frontend_down_page(), status_code=503)


@app.get("/", include_in_schema=False)
def serve_root():
    # 해시는 브리지 페이지의 스크립트가 읽어 알맞은 경로로 바꿉니다.
    return _frontend_response()


@app.get("/score", include_in_schema=False)
def serve_score():
    return _frontend_response("/score")


@app.get("/paper", include_in_schema=False)
def serve_paper():
    return _frontend_response("/paper")


@app.get("/login", include_in_schema=False)
def serve_login():
    return _frontend_response("/login")


# 구버전 단일 페이지 — 프론트엔드를 못 켰을 때의 비상용으로만 남겨둡니다.
if WEB_DIR.exists():
    @app.get("/legacy", include_in_schema=False)
    def serve_legacy():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
