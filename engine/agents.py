"""
LLM 멀티에이전트 종목 분석 (TradingAgents 방식)
==============================================
룰 기반 자동매매(engine/strategy.py)와는 **성격이 다른 판단기**입니다. 저쪽은
지표를 숫자로 합산해 점수를 냅니다. 이쪽은 역할이 다른 LLM 에이전트 여럿이
같은 증거를 보고 서로 다른 결론을 낸 뒤, 토론과 리스크 심사를 거쳐 하나의
판단으로 수렴합니다.

왜 별도 모듈인가
    자동매매 엔진에 끼워 넣으면 두 가지가 섞입니다 — 재현 가능한 산식과
    재현 불가능한 생성 결과. 게다가 이 모듈은 호출 한 번에 실제 돈(API 요금)이
    나갑니다. 자동매매 회전에 딸려 들어가면 사용자가 모르는 사이에 비용이
    쌓입니다. 그래서 **사람이 버튼을 눌렀을 때만** 돕니다.

이 모듈은 주문을 내지 않습니다
    broker / kis_trading / autotrade 를 import 하지 않습니다. 결과는 저장되고
    화면에 표시될 뿐, 어떤 경로로도 주문으로 이어지지 않습니다. 시그널 신뢰도가
    검증되기 전에 표시와 집행을 붙이면, 검증되지 않은 판단이 곧바로 돈이 됩니다.

파이프라인 (한 번 돌 때 LLM 호출 11회, 토론 1라운드 기준)
    1) 애널리스트 4인   기술적, 시장 수급, 뉴스, 심리          (동시 호출)
    2) 연구원 토론      강세 대 약세, 라운드마다 2회           (순차, 서로의 말을 봄)
    3) 리서치 매니저    토론 판정 + 투자 계획                  (1회)
    4) 리스크 심사 3인  공격적, 중립, 보수적                   (동시 호출)
    5) 리스크 매니저    최종 판단 매수/매도/보유 + 확신도      (1회)

증거는 LLM 이 지어내지 않습니다
    collect_evidence() 가 이 프로젝트가 이미 쓰는 경로(가격 공급자, 기술적
    지표, 뉴스 크롤러, 커뮤니티 크롤러)로 실제 데이터를 모아 시스템 프롬프트에
    싣습니다. 에이전트는 그 팩만 보고 말합니다. 시스템 프롬프트는 모든 호출에서
    같으므로 prompt caching 이 걸립니다 (두 번째 호출부터 입력 비용 약 1/10).
"""

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from data_sources import credentials

# ---------------------------------------------------------------------------
# 모델과 요금
# ---------------------------------------------------------------------------
# 화면에 "분석 1회당 비용"을 보여주려면 요금표가 있어야 합니다. 공식 요금은
# 1M 토큰당 USD 입니다. 캐시 쓰기는 1.25배, 캐시 읽기는 0.1배로 계산합니다.
#
# 요금이 바뀌면 이 표만 고치면 됩니다. api_keys.json 의 ANTHROPIC_PRICING 으로
# 덮어쓸 수도 있습니다 (코드 수정 없이 바꿔야 하는 표 — config as data).
MODEL_PRICING = {
    "claude-opus-5":    {"label": "Claude Opus 5",    "input": 5.0, "output": 25.0},
    "claude-sonnet-5":  {"label": "Claude Sonnet 5",  "input": 2.0, "output": 10.0},
    "claude-haiku-4-5": {"label": "Claude Haiku 4.5", "input": 1.0, "output": 5.0},
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

DEFAULT_MODEL = "claude-opus-5"

# 노력 수준은 생각 토큰의 깊이입니다. 애널리스트 4인과 토론은 medium, 최종
# 판단만 high 로 둡니다. 전부 high 로 두면 호출 11회의 생각 토큰이 그대로
# 비용이 되는데, 중간 단계는 증거 요약에 가까워 그만큼이 필요 없습니다.
DEFAULT_CONFIG = {
    "model": DEFAULT_MODEL,
    "effort": "medium",             # low, medium, high, xhigh, max
    "final_effort": "high",         # 최종 판단만 더 깊게
    "debate_rounds": 1,             # 강세 약세 토론 라운드 (1~3)
    "include_news": True,
    "include_community": True,
    "max_news": 12,
    "max_posts": 12,
}
_INT_KEYS = ("debate_rounds", "max_news", "max_posts")
_BOOL_KEYS = ("include_news", "include_community")
_EFFORTS = ("low", "medium", "high", "xhigh", "max")

MAX_TOKENS = 16000              # 생각 토큰이 여기 포함됩니다 — 넉넉히 둡니다

DECISION_LABELS = {"buy": "매수", "sell": "매도", "hold": "보유"}
STANCE_LABELS = {"bullish": "강세", "bearish": "약세", "neutral": "중립"}


def clamp_config(cfg: dict) -> dict:
    """설정을 아는 값과 허용 범위 안으로 강제. 저장 단계에서 잘라 둡니다.

    engine/scalping.clamp_config 와 같은 이유입니다 — 저장할 때 한 번 자르면
    이후 어느 경로로 읽어도 한도 안입니다.
    """
    out = dict(DEFAULT_CONFIG)
    for key, value in (cfg or {}).items():
        if key not in DEFAULT_CONFIG:
            continue                       # 오타로 파이프라인이 망가지지 않게
        out[key] = value
    if out["model"] not in pricing_table():
        out["model"] = DEFAULT_MODEL
    for key in ("effort", "final_effort"):
        if out[key] not in _EFFORTS:
            out[key] = DEFAULT_CONFIG[key]
    for key in _BOOL_KEYS:
        out[key] = bool(out[key])
    for key in _INT_KEYS:
        try:
            out[key] = int(out[key])
        except (TypeError, ValueError):
            out[key] = DEFAULT_CONFIG[key]
    out["debate_rounds"] = max(1, min(3, out["debate_rounds"]))
    out["max_news"] = max(0, min(30, out["max_news"]))
    out["max_posts"] = max(0, min(30, out["max_posts"]))
    return out


def pricing_table() -> dict:
    """요금표. api_keys.json 의 ANTHROPIC_PRICING 으로 덮어쓸 수 있습니다."""
    table = {k: dict(v) for k, v in MODEL_PRICING.items()}
    override = credentials.get_json("ANTHROPIC_PRICING", None)
    if isinstance(override, dict):
        for model, spec in override.items():
            if isinstance(spec, dict):
                table.setdefault(model, {"label": model, "input": 0.0, "output": 0.0})
                table[model].update(spec)
    return table


def cost_of(model: str, usage: dict) -> float:
    """토큰 사용량을 USD 로. 모르는 모델이면 0 (비용을 지어내지 않습니다)."""
    spec = pricing_table().get(model)
    if not spec:
        return 0.0
    million = 1_000_000
    fresh = usage.get("input_tokens", 0) or 0
    written = usage.get("cache_write_tokens", 0) or 0
    read = usage.get("cache_read_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    return (fresh * spec["input"] / million
            + written * spec["input"] * CACHE_WRITE_MULTIPLIER / million
            + read * spec["input"] * CACHE_READ_MULTIPLIER / million
            + out * spec["output"] / million)


# ---------------------------------------------------------------------------
# API 키와 클라이언트
# ---------------------------------------------------------------------------

def is_configured() -> bool:
    return bool(credentials.get("ANTHROPIC_API_KEY"))


def sdk_installed() -> bool:
    try:
        import anthropic            # noqa: F401
    except ImportError:
        return False
    return True


def readiness() -> dict:
    """왜 못 도는지를 화면이 구체적으로 말할 수 있게 상태를 나눠 돌려줍니다.

    "실패했습니다" 한 줄만 보여주면 사용자는 키 문제인지 설치 문제인지
    네트워크 문제인지 구분할 수 없습니다.
    """
    sdk = sdk_installed()
    key = is_configured()
    if not sdk:
        reason = "anthropic 패키지가 없습니다. pip install anthropic 후 서버를 다시 켜세요."
    elif not key:
        reason = ("ANTHROPIC_API_KEY 가 없습니다. api_keys.json 에 넣거나 "
                  "계정 설정의 내 API 키에 저장하세요.")
    else:
        reason = ""
    return {"ready": sdk and key, "sdk_installed": sdk, "key_configured": key,
            "reason": reason}


def _client():
    """anthropic 클라이언트. 준비가 안 됐으면 이유를 담아 RuntimeError."""
    state = readiness()
    if not state["ready"]:
        raise RuntimeError(state["reason"])
    import anthropic
    return anthropic.Anthropic(api_key=credentials.get("ANTHROPIC_API_KEY"))


# 거절 폴백 — 안전 분류기가 요청을 거절하면 같은 호출 안에서 다른 모델이
# 이어받습니다. 이 인자를 모르는 SDK 버전에서는 한 번 거부당한 뒤 끄고 갑니다
# (그 뒤로는 폴백 없이 돕니다. 기능이 통째로 죽는 것보다 낫습니다).
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_fallback_ok = True


# ---------------------------------------------------------------------------
# LLM 호출 — 이 모듈에서 앤트로픽으로 나가는 유일한 지점
# ---------------------------------------------------------------------------
# 테스트는 이 함수 하나만 갈아끼우면 파이프라인 전체를 키 없이 돌 수 있습니다
# (tests/test_agents.py).

def _ask(client, model: str, system: str, prompt: str, schema: dict,
         effort: str) -> tuple[dict, dict]:
    """에이전트 한 명에게 묻고 (구조화된 답, 사용량) 을 돌려줍니다."""
    global _fallback_ok
    import anthropic

    kwargs = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        # 시스템 프롬프트가 모든 에이전트에서 동일합니다 -> 캐시가 걸립니다.
        # 증거 팩이 크기 때문에 이 한 줄이 비용의 큰 부분을 줄입니다.
        system=[{"type": "text", "text": system,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": schema}},
    )

    try:
        if _fallback_ok:
            resp = client.beta.messages.create(
                betas=[_FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            resp = client.messages.create(**kwargs)
    except (TypeError, anthropic.BadRequestError):
        if not _fallback_ok:
            raise                       # 폴백과 무관한 진짜 오류입니다
        _fallback_ok = False
        resp = client.messages.create(**kwargs)

    if getattr(resp, "stop_reason", "") == "refusal":
        detail = getattr(resp, "stop_details", None)
        raise RuntimeError("모델이 응답을 거절했습니다"
                           + (f" ({detail.category})" if detail else ""))

    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # output_config 가 JSON 을 보장하지만, 형식이 깨지더라도 분석 전체가
        # 죽지는 않게 원문을 요약 자리에 넣어 둡니다.
        data = {"summary": (text or "").strip()[:800], "points": []}
    if not isinstance(data, dict):
        data = {"summary": str(data)[:800], "points": []}

    u = resp.usage
    usage = {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }
    return data, usage


# ---------------------------------------------------------------------------
# 증거 수집 — 에이전트가 볼 수 있는 사실의 전부
# ---------------------------------------------------------------------------

def collect_evidence(query: str, cfg: dict) -> dict:
    """이 프로젝트가 이미 쓰는 경로로 실제 데이터를 모읍니다.

    한 조각이 실패해도 분석 자체는 돌아가야 합니다 (뉴스가 막혔다고 기술적
    분석까지 못 할 이유가 없습니다). 실패한 조각은 `missing` 에 이름을 남겨
    에이전트가 "그건 모른다"고 말할 수 있게 합니다.
    """
    from data_sources import (community_crawler, market_clock, news_crawler,
                              price_provider, symbol_registry)
    from engine import scoring

    symbol = symbol_registry.resolve(query)       # 못 찾으면 SymbolNotFoundError
    ev = {
        "symbol": symbol.key,
        "name": symbol.name,
        "market": symbol.market,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "missing": [],
    }

    provider = price_provider.get_provider(symbol)

    try:
        history = provider.get_daily_history(symbol, days=120)
        technical = scoring.analyze_technical(history)
        ev["technical"] = {
            "score": round(technical.score, 4),
            "bars_used": technical.bars_used,
            "sufficient_data": technical.sufficient_data,
            "note": technical.note,
            "regime": technical.regime,
            "indicators": [
                {"label": i.label, "value": i.value_text, "score": round(i.score, 3),
                 "verdict": i.verdict, "reason": i.reason, "family": i.family}
                for i in technical.indicators
            ],
        }
    except Exception as exc:                                     # noqa: BLE001
        ev["missing"].append(f"기술적 지표 ({type(exc).__name__})")

    try:
        snapshot = provider.get_snapshot(symbol)
        ev["quote"] = {
            "price": snapshot.current_price,
            "change_pct": getattr(snapshot, "change_percent", None),
            "volume": getattr(snapshot, "volume", None),
            "currency": getattr(snapshot, "currency", ""),
            "source": snapshot.source,
        }
    except Exception as exc:                                     # noqa: BLE001
        ev["missing"].append(f"현재가 ({type(exc).__name__})")

    if cfg.get("include_news") and cfg.get("max_news"):
        try:
            items = news_crawler.get_news(symbol, limit=cfg["max_news"])
            ev["news"] = {
                "score": news_crawler.aggregate_news_score(items),
                "summary": news_crawler.summarize_news(items),
                "items": [
                    {"title": n.title, "sentiment": round(n.sentiment_score, 3),
                     "source": n.source,
                     "published_at": n.published_at.isoformat(timespec="minutes")}
                    for n in items[:cfg["max_news"]]
                ],
            }
        except Exception as exc:                                 # noqa: BLE001
            ev["missing"].append(f"뉴스 ({type(exc).__name__})")

    if cfg.get("include_community") and cfg.get("max_posts"):
        try:
            sentiment = community_crawler.get_community_sentiment(symbol)
            ev["community"] = {
                "bullish_ratio": sentiment.bullish_ratio,
                "post_count": sentiment.post_count,
                "bullish_count": sentiment.bullish_count,
                "bearish_count": sentiment.bearish_count,
                "neutral_count": sentiment.neutral_count,
                "sources": sentiment.sources,
                "demo": sentiment.is_demo,
                "posts": [
                    {"title": p.title, "kind": getattr(p, "sentiment", ""),
                     "source": getattr(p, "source", "")}
                    for p in (sentiment.recent_posts or [])[:cfg["max_posts"]]
                ],
            }
        except Exception as exc:                                 # noqa: BLE001
            ev["missing"].append(f"커뮤니티 ({type(exc).__name__})")

    try:
        ev["market_status"] = market_clock.status_for(symbol.market)
    except Exception:                                            # noqa: BLE001
        pass

    return ev


def evidence_text(ev: dict) -> str:
    """증거 팩을 사람이 읽는 형태로. 모든 에이전트가 이 문자열 하나를 공유합니다."""
    lines = [
        "# 분석 대상",
        f"종목 {ev.get('name')} ({ev.get('symbol')}), 시장 {ev.get('market')}",
        f"수집 시각 {ev.get('collected_at')}",
    ]
    quote = ev.get("quote") or {}
    if quote:
        lines.append(f"현재가 {quote.get('price')} {quote.get('currency')}, "
                     f"등락 {quote.get('change_pct')}%, 출처 {quote.get('source')}")
    status = ev.get("market_status") or {}
    if status:
        lines.append(f"장 상태 {status.get('label') or status.get('state')}")

    tech = ev.get("technical")
    if tech:
        lines += ["", "# 기술적 지표",
                  f"종합 기술점수 {tech['score']} (-1 매도 에서 +1 매수), "
                  f"사용한 일봉 {tech['bars_used']}개",
                  f"국면 {json.dumps(tech.get('regime'), ensure_ascii=False)}"]
        for i in tech.get("indicators", []):
            lines.append(f"  {i['label']}: {i['value']} (점수 {i['score']}, "
                         f"{i['verdict']}) {i['reason']}")

    news = ev.get("news")
    if news:
        s = news.get("summary") or {}
        lines += ["", "# 뉴스",
                  f"뉴스 감성 종합 {news.get('score')}, 전체 {s.get('total')}건 "
                  f"(긍정 {s.get('positive')} 부정 {s.get('negative')} "
                  f"중립 {s.get('neutral')})"]
        for n in news.get("items", []):
            lines.append(f"  [{n['published_at']}] {n['title']} "
                         f"(감성 {n['sentiment']}, {n['source']})")

    comm = ev.get("community")
    if comm:
        lines += ["", "# 커뮤니티 여론",
                  f"강세 비율 {comm.get('bullish_ratio')}, 표본 {comm.get('post_count')}건 "
                  f"(강세 {comm.get('bullish_count')} 약세 {comm.get('bearish_count')} "
                  f"중립 {comm.get('neutral_count')})"]
        if comm.get("demo"):
            lines.append("  주의: 실제 수집에 실패해 예시 데이터입니다. 근거로 쓰지 마세요.")
        for p in comm.get("posts", []):
            lines.append(f"  {p['title']} ({p['kind']}, {p['source']})")

    if ev.get("missing"):
        lines += ["", "# 수집하지 못한 자료",
                  "다음은 이번 분석에 없습니다. 없는 자료를 있는 것처럼 말하지 마세요.",
                  "  " + ", ".join(ev["missing"])]
    return "\n".join(lines)


SYSTEM_PREAMBLE = """당신은 한국 개인투자자용 분석 시스템 아테나의 에이전트 중 한 명입니다.

규칙
1. 아래 증거 팩에 있는 사실만 씁니다. 팩에 없는 수치, 뉴스, 실적을 지어내지 마세요.
   모르는 것은 모른다고 쓰십시오.
2. 답은 한국어로, 개인투자자가 읽을 수 있는 평이한 문장으로 씁니다.
3. 확신도(confidence)는 0 에서 1 사이의 숫자이며, 증거가 약하면 낮게 잡습니다.
4. 이 분석은 연구 참고용이며 투자자문이 아닙니다. 단정적인 수익 약속을 쓰지 마세요.
5. 요청받은 역할의 관점을 지키십시오. 균형을 잡는 것은 다음 단계의 일입니다.

"""

# --- 응답 형식 (structured outputs) ---------------------------------------
# 형식을 스키마로 강제하면 파싱이 무너지지 않습니다. 화면이 단계마다 같은
# 모양을 기대할 수 있어야 접기 펼치기 UI 를 한 벌로 그릴 수 있습니다.
_OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "points": {"type": "array", "items": {"type": "string"}},
        "watch": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["stance", "confidence", "summary", "points", "watch"],
    "additionalProperties": False,
}

_DEBATE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "points": {"type": "array", "items": {"type": "string"}},
        "counter": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["summary", "points", "counter", "confidence"],
    "additionalProperties": False,
}

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["bull", "bear", "tie"]},
        "stance": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence": {"type": "number"},
        "summary": {"type": "string"},
        "plan": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["winner", "stance", "confidence", "summary", "plan"],
    "additionalProperties": False,
}

_FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence": {"type": "number"},
        "horizon": {"type": "string"},
        "summary": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "invalidation": {"type": "string"},
    },
    "required": ["decision", "confidence", "horizon", "summary", "reasons",
                 "risks", "invalidation"],
    "additionalProperties": False,
}

ANALYSTS = [
    ("analyst.technical", "기술적 분석가",
     "차트와 기술적 지표만 봅니다. 추세, 모멘텀, 과열 침체, 거래량 확인을 근거로 "
     "지금이 진입 구간인지 판단하세요. 뉴스와 여론은 다른 사람이 봅니다."),
    ("analyst.market", "시장 수급 분석가",
     "가격 수준, 변동성, 유동성, 장 상태를 봅니다. 이 종목을 지금 사고팔 때 "
     "체결과 슬리피지 관점에서 무엇이 문제인지, 국면이 무엇을 말하는지 판단하세요."),
    ("analyst.news", "뉴스 분석가",
     "뉴스 헤드라인과 감성 점수만 봅니다. 어떤 사건이 가격을 움직일 만한지, "
     "이미 반영된 소식인지 판단하세요. 뉴스가 없으면 없다고 쓰세요."),
    ("analyst.sentiment", "심리 분석가",
     "커뮤니티 여론과 화제성만 봅니다. 과열된 낙관인지, 항복 국면인지, "
     "표본이 너무 적어 의미가 없는지 판단하세요."),
]

RISK_SEATS = [
    ("risk.aggressive", "공격적 심사역",
     "수익 기회를 놓치는 비용을 강조하는 자리입니다. 이 계획이 지나치게 "
     "소극적인 지점을 지적하세요."),
    ("risk.neutral", "중립 심사역",
     "기대값과 확률을 따지는 자리입니다. 양쪽 주장의 크기를 비교하세요."),
    ("risk.conservative", "보수적 심사역",
     "자본 보존이 우선인 자리입니다. 이 계획이 틀렸을 때 얼마나 잃는지, "
     "어떤 전제가 무너지면 손실이 커지는지 지적하세요."),
]

# 화면이 단계를 묶어 보여줄 때 쓰는 이름입니다 (접기 펼치기 한 덩어리 = 한 그룹).
STAGE_GROUPS = {
    "analyst": "애널리스트 분석",
    "debate": "강세 약세 토론",
    "research": "연구 판정",
    "risk": "리스크 평가",
}


def _group_of(key: str) -> str:
    return key.split(".")[0]


def _brief(stage: dict) -> str:
    """다음 에이전트에게 넘길 앞 단계 요약. 토큰을 아끼려고 요점만 넘깁니다."""
    parts = [f"[{stage.get('role', '')}] {stage.get('summary', '')}"]
    for p in (stage.get("points") or [])[:5]:
        parts.append(f"  - {p}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def run_analysis(query: str, cfg: dict = None, progress=None) -> dict:
    """종목 하나를 끝까지 분석합니다. 주문은 내지 않습니다.

    progress(done, total, label) 을 주면 단계마다 불러 화면에 진행률을 줍니다.
    """
    cfg = clamp_config(cfg or {})
    model = cfg["model"]
    started = time.time()

    ev = collect_evidence(query, cfg)
    system = SYSTEM_PREAMBLE + evidence_text(ev)

    client = _client()
    stages: list[dict] = []
    usage_total = {"input_tokens": 0, "output_tokens": 0,
                   "cache_write_tokens": 0, "cache_read_tokens": 0}
    total_steps = len(ANALYSTS) + cfg["debate_rounds"] * 2 + 1 + len(RISK_SEATS) + 1
    done = 0
    lock = threading.Lock()

    def _step(key: str, role: str, prompt: str, schema: dict, effort: str) -> dict:
        data, usage = _ask(client, model, system, prompt, schema, effort)
        stage = {"key": key, "group": _group_of(key), "role": role,
                 "usage": usage, "cost_usd": cost_of(model, usage), **data}
        with lock:                     # 동시 호출 구간에서 합계가 어긋나지 않게
            for name in usage_total:
                usage_total[name] += usage.get(name, 0)
        return stage

    def _announce(label: str):
        nonlocal done
        with lock:
            done += 1
            current = done
        if progress:
            progress(current, total_steps, label)

    # --- 1) 애널리스트 4인. 서로를 보지 않습니다, 관점 다양성이 목적입니다 ---
    with ThreadPoolExecutor(max_workers=len(ANALYSTS)) as pool:
        futures = []
        for key, role, angle in ANALYSTS:
            prompt = (
                f"당신은 {role}입니다. {angle}\n\n"
                "증거 팩을 근거로 당신 관점의 결론을 내세요. points 에는 근거를 "
                "3개에서 5개, watch 에는 이 판단이 틀렸음을 알려줄 관찰 지점을 "
                "2개에서 3개 적으세요."
            )
            futures.append((role,
                            pool.submit(_step, key, role, prompt,
                                        _OPINION_SCHEMA, cfg["effort"])))
        for role, future in futures:
            stages.append(future.result())
            _announce(role)
    # 동시 실행이라 완료 순서가 뒤섞일 수 있습니다. 화면 순서는 정의 순서로 고정.
    order = {key: i for i, (key, _, _) in enumerate(ANALYSTS)}
    stages.sort(key=lambda s: order.get(s["key"], 99))

    analyst_brief = "\n".join(_brief(s) for s in stages)

    # --- 2) 강세 대 약세 토론. 순차입니다, 상대 말을 받아야 토론입니다 ---
    transcript = ""
    for rnd in range(1, cfg["debate_rounds"] + 1):
        for side, label, angle in (
            ("bull", "강세 연구원", "이 종목을 지금 사야 하는 이유"),
            ("bear", "약세 연구원", "이 종목을 지금 사면 안 되는 이유"),
        ):
            prompt = (
                f"당신은 {label}입니다. {angle}를 주장하세요. "
                f"지금은 {rnd}번째 라운드입니다.\n\n"
                f"# 애널리스트 4인의 의견\n{analyst_brief}\n\n"
                + (f"# 지금까지의 토론\n{transcript}\n\n" if transcript else "")
                + "points 에는 당신 주장의 근거를, counter 에는 상대 주장의 "
                  "약한 고리를 적으세요. 상대가 아직 말하지 않았다면 예상되는 "
                  "반론을 적으세요."
            )
            stage = _step(f"debate.{side}.{rnd}", f"{label} {rnd}라운드",
                          prompt, _DEBATE_SCHEMA, cfg["effort"])
            stages.append(stage)
            transcript += _brief(stage) + "\n"
            _announce(f"{label} {rnd}라운드")

    # --- 3) 리서치 매니저. 토론을 판정하고 계획을 세웁니다 ---
    manager = _step(
        "research.manager", "리서치 매니저",
        "당신은 리서치 매니저입니다. 아래 토론을 읽고 어느 쪽이 더 설득력 "
        "있는지 판정한 뒤, 그 판정에 따른 투자 계획을 세우세요. 어느 한쪽 편을 "
        "들기 어렵다면 tie 로 적고 보유(hold)를 고르십시오.\n\n"
        f"# 애널리스트 의견\n{analyst_brief}\n\n# 토론\n{transcript}\n\n"
        "plan 에는 실행 단계를 3개에서 5개 적으세요.",
        _PLAN_SCHEMA, cfg["effort"])
    stages.append(manager)
    _announce("리서치 매니저")

    plan_brief = _brief({**manager, "points": manager.get("plan", [])})

    # --- 4) 리스크 심사 3인. 동시입니다, 서로 눈치 보지 않게 ---
    with ThreadPoolExecutor(max_workers=len(RISK_SEATS)) as pool:
        futures = []
        for key, role, angle in RISK_SEATS:
            prompt = (
                f"당신은 {role}입니다. {angle}\n\n"
                f"# 리서치 매니저의 계획\n{plan_brief}\n\n"
                f"# 애널리스트 의견\n{analyst_brief}\n\n"
                "points 에는 지적 사항을 3개에서 5개 적으세요."
            )
            futures.append((role,
                            pool.submit(_step, key, role, prompt,
                                        _DEBATE_SCHEMA, cfg["effort"])))
        risk_stages = []
        for role, future in futures:
            risk_stages.append(future.result())
            _announce(role)
    risk_order = {key: i for i, (key, _, _) in enumerate(RISK_SEATS)}
    risk_stages.sort(key=lambda s: risk_order.get(s["key"], 99))
    stages.extend(risk_stages)

    risk_brief = "\n".join(_brief(s) for s in risk_stages)

    # --- 5) 리스크 매니저. 최종 판단 ---
    final = _step(
        "risk.manager", "리스크 매니저",
        "당신은 리스크 매니저이며 이 분석의 최종 판단을 내립니다. 심사역 세 "
        "명의 지적을 반영해 매수 매도 보유 중 하나를 고르세요.\n\n"
        f"# 리서치 매니저의 계획\n{plan_brief}\n\n"
        f"# 리스크 심사\n{risk_brief}\n\n"
        "horizon 에는 이 판단이 유효한 기간을 적고(예: 2주에서 1개월), "
        "invalidation 에는 무엇이 관찰되면 이 판단을 버려야 하는지 한 문장으로 "
        "적으세요. 증거가 엇갈리면 보유(hold)가 정답일 수 있습니다.",
        _FINAL_SCHEMA, cfg["final_effort"])
    stages.append(final)
    _announce("리스크 매니저")

    cost_usd = sum(s["cost_usd"] for s in stages)
    return {
        "symbol": ev["symbol"],
        "name": ev["name"],
        "market": ev["market"],
        "decision": final.get("decision", "hold"),
        "confidence": final.get("confidence"),
        "horizon": final.get("horizon", ""),
        "summary": final.get("summary", ""),
        "price": (ev.get("quote") or {}).get("price"),
        "model": model,
        "config": cfg,
        "usage": usage_total,
        "calls": len(stages),
        "cost_usd": cost_usd,
        "cost_krw": to_krw(cost_usd),
        "elapsed_ms": int((time.time() - started) * 1000),
        "report": {"stages": stages, "evidence": ev},
    }


def to_krw(usd: float) -> float | None:
    """USD 비용을 원화로. 환율을 못 구하면 None (0 으로 속이지 않습니다)."""
    try:
        from data_sources import fx
        rate, _ = fx.usd_krw()
        return round(usd * rate, 1) if rate else None
    except Exception:                                            # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 실행 관리 — 분석 한 번이 수십 초에서 수 분 걸립니다
# ---------------------------------------------------------------------------
# HTTP 요청 안에서 끝내려 하면 프록시와 브라우저가 먼저 끊습니다. 그래서
# 백그라운드 스레드로 돌리고 화면은 상태를 폴링합니다.
#
# 같은 사용자가 같은 종목을 두 번 누르면 두 번 과금됩니다. 그래서 진행 중인
# (사용자, 종목) 조합은 새로 시작하지 않고 기존 작업을 그대로 돌려줍니다.

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOB_KEEP = 40                  # 끝난 작업을 이만큼만 남깁니다 (메모리 보호)


def _prune_jobs():
    finished = [j for j in _jobs.values() if j["state"] in ("done", "failed")]
    if len(finished) <= _JOB_KEEP:
        return
    finished.sort(key=lambda j: j["finished_at"] or "")
    for job in finished[:len(finished) - _JOB_KEEP]:
        _jobs.pop(job["id"], None)


def start_job(user_id: int, query: str, cfg: dict, on_done=None) -> dict:
    """분석을 백그라운드에서 시작하고 작업 표를 돌려줍니다."""
    with _jobs_lock:
        for job in _jobs.values():
            if (job["user_id"] == user_id and job["state"] == "running"
                    and job["query"].lower() == query.lower()):
                return dict(job)             # 중복 클릭은 중복 과금입니다
        job = {
            "id": uuid.uuid4().hex[:12],
            "user_id": user_id,
            "query": query,
            "state": "running",
            "step": 0,
            "total": 0,
            "label": "증거 수집 중",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": "",
            "analysis_id": None,
            "error": "",
        }
        _jobs[job["id"]] = job
        _prune_jobs()

    def _progress(step, total, label):
        with _jobs_lock:
            job.update(step=step, total=total, label=label)

    def _worker():
        try:
            result = run_analysis(query, cfg, progress=_progress)
            analysis_id = on_done(result) if on_done else None
            with _jobs_lock:
                job.update(state="done", label="완료", analysis_id=analysis_id,
                           finished_at=datetime.now().isoformat(timespec="seconds"))
        except Exception as exc:                                 # noqa: BLE001
            with _jobs_lock:
                job.update(state="failed", label="실패",
                           error=f"{type(exc).__name__}: {exc}",
                           finished_at=datetime.now().isoformat(timespec="seconds"))

    threading.Thread(target=_worker, daemon=True,
                     name=f"agents-{job['id']}").start()
    return dict(job)


def jobs_for(user_id: int) -> list[dict]:
    with _jobs_lock:
        rows = [dict(j) for j in _jobs.values() if j["user_id"] == user_id]
    rows.sort(key=lambda j: j["started_at"], reverse=True)
    return rows[:20]
