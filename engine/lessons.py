# -*- coding: utf-8 -*-
"""
손실 학습 (Post-mortem Learning)
--------------------------------
청산된 매매를 전부 기록하고, 손실이 난 매매는 **왜 잃었는지**를 진입 시점의
컨텍스트에서 추론합니다. 그 결론을 컨텍스트 태그별 기대수익으로 집계해
두었다가, 같은 컨텍스트의 다음 신호를 그만큼 감쇠합니다.

동작 순서
    1. 진입 체결 → open_lesson : 신호 스냅샷 + 컨텍스트 태그를 저장
    2. 청산 체결 → close_lesson: 손익 확정, 손실이면 원인 추론(diagnose)
    3. 다음 회전 → inject      : 태그별 감쇠표를 계산해 cfg 사본에 실음
    4. 신호 계산 → apply_to_score: 지금 신호의 태그와 감쇠표를 대조해 점수 감쇠

계산식 (learn_mode=soft 일 때)
    태그 t 의 기대수익  E_t = Σ w_i·pnl_i / Σ w_i,  w_i = 0.5^(경과일/반감기)
    유효 표본           n_t = Σ w_i
    태그 감쇠           pen_t = learn_max_penalty × min(1, |E_t| / learn_scale_pct)
                               (n_t ≥ learn_min_trades 이고 E_t < 0 일 때만)
    점수 적용           |score| 에서 Σ pen_t 를 빼되 learn_total_cap 을 넘지 않게.
                        부호는 그대로 둡니다 — 학습이 신호를 반대로 뒤집지 않습니다.

왜 이긴 매매도 기록하는가
    손실만 모으면 모든 태그가 나빠 보입니다(선택 편향). 태그의 기대수익은
    이긴 매매와 진 매매를 **함께** 넣어야 공정하게 나옵니다. 복기(원인 추론)는
    손실에만 하지만, 기록과 집계는 전 매매가 대상입니다.

왜 점수를 올리지는 않는가
    돈을 잃은 조건을 피하는 것은 보수적이지만, 돈을 번 조건에 더 태우는 것은
    과최적화로 가는 지름길입니다. 감쇠(penalty)만 합니다.

백테스트 안전
    감쇠표는 회전(_tick)이 cfg **사본**에 `_lesson_penalties` 로 실어줄 때만
    동작합니다. 저장되는 설정(JSON)에는 이 키가 없으므로 백테스트가 실수로
    오늘의 학습 상태를 과거 판단에 섞는 일(미래 참조)이 구조적으로 없습니다.
"""

import time
from datetime import datetime

OFF, OBSERVE, SOFT = "off", "observe", "soft"

# 신호 컨텍스트에서 뽑는 태그들이 쓰는 문턱값. 설정으로 열지 않습니다 —
# 태그 정의가 바뀌면 과거에 쌓인 태그와 다른 뜻이 되어 집계가 오염됩니다.
LEAD_MIN = 0.15            # 이보다 커야 "주도 성분"으로 인정
RSI_HOT, RSI_COLD = 70.0, 30.0
DIVERGE_GAP = 0.30         # 일봉·분봉 불일치로 볼 최소 간격
NEWS_HEAVY = 0.30          # 뉴스 점수가 이보다 크고 일봉이 약하면 "뉴스의존"
HIGH_ATR_PCT = 4.0         # 이보다 크면 "고변동성"

# 태그 통계 캐시 — 회전마다 DB 를 다시 읽지 않습니다. 청산이 기록되면
# invalidate() 로 비웁니다 (그 전까지 감쇠표가 몇 분 늦는 것은 허용).
_stats_cache: dict[tuple, tuple[float, dict]] = {}
STATS_TTL = 300.0


def mode_of(cfg: dict) -> str:
    value = str((cfg or {}).get("learn_mode", OBSERVE)).strip().lower()
    return value if value in (OFF, OBSERVE, SOFT) else OBSERVE


# ---------------------------------------------------------------------------
# 컨텍스트 스냅샷과 태그
# ---------------------------------------------------------------------------

_SNAP_KEYS = ("score", "direction", "regime", "daily_score", "intraday_score",
              "news_score", "rsi", "bb_pct", "atr_pct", "vol_factor")


def snapshot(sig_dict: dict) -> dict:
    """Signal.to_dict() 에서 학습에 필요한 값만 남긴 사본."""
    src = sig_dict if isinstance(sig_dict, dict) else {}
    return {k: src.get(k) for k in _SNAP_KEYS}


def tags_of(snap: dict, symbol: str = None) -> list[str]:
    """진입(또는 지금 신호) 컨텍스트를 태그 목록으로 바꿉니다.

    진입 때 저장하는 태그와 신호 계산 때 대조하는 태그가 **같은 함수**여야
    합니다 — 정의가 갈라지면 배운 것과 적용하는 것이 서로 다른 이야기가 됩니다.
    """
    tags = []
    try:
        score = float(snap.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    d = 1.0 if score >= 0 else -1.0

    regime = str(snap.get("regime") or "").strip()
    if regime:
        tags.append(f"국면:{regime}")

    # 점수를 주도한 성분 — 방향과 같은 쪽으로 가장 크게 민 것
    components = {"일봉": snap.get("daily_score"),
                  "분봉": snap.get("intraday_score"),
                  "뉴스": snap.get("news_score")}
    contrib = {}
    for label, value in components.items():
        try:
            if value is not None:
                contrib[label] = float(value) * d
        except (TypeError, ValueError):
            continue
    if contrib:
        lead, value = max(contrib.items(), key=lambda kv: kv[1])
        if value >= LEAD_MIN:
            tags.append(f"주도:{lead}")

    rsi = snap.get("rsi")
    if rsi is not None:
        try:
            rsi = float(rsi)
            if d > 0 and rsi >= RSI_HOT:
                tags.append("과열진입")
            elif d < 0 and rsi <= RSI_COLD:
                tags.append("과매도숏")
        except (TypeError, ValueError):
            pass

    daily, intraday = snap.get("daily_score"), snap.get("intraday_score")
    try:
        if (daily is not None and intraday is not None
                and float(daily) * float(intraday) < 0
                and abs(float(daily) - float(intraday)) >= DIVERGE_GAP):
            tags.append("일봉분봉불일치")
    except (TypeError, ValueError):
        pass

    news = snap.get("news_score")
    try:
        if (news is not None and float(news) * d > NEWS_HEAVY
                and (daily is None or float(daily) * d < LEAD_MIN)):
            tags.append("뉴스의존")
    except (TypeError, ValueError):
        pass

    atr_pct = snap.get("atr_pct")
    try:
        if atr_pct is not None and float(atr_pct) >= HIGH_ATR_PCT:
            tags.append("고변동성")
    except (TypeError, ValueError):
        pass

    if symbol:
        tags.append(f"종목:{symbol}")
    return tags


# ---------------------------------------------------------------------------
# 손실 원인 추론
# ---------------------------------------------------------------------------

def exit_class(reason: str) -> str:
    """청산 사유 문자열(check_exit 가 만든 한국어)을 분류합니다."""
    r = str(reason or "")
    if "손절" in r or "손실 한도" in r:
        return "손절"
    if "되돌림" in r or "트레일링" in r:
        return "트레일링"
    if "시간 청산" in r or "보유 " in r:
        return "시간청산"
    if "신호 반전" in r or "신호 소멸" in r:
        return "신호반전"
    if "목표" in r:
        return "익절"
    if "만기" in r:
        return "만기"
    if "회전" in r:
        return "회전매도"
    if "외부" in r or "대조" in r:
        return "외부청산"
    return "기타"


def diagnose(snap: dict, tags: list, reason: str, pnl_pct: float | None) -> str:
    """손실 매매의 원인 추론 — 사람이 읽는 결론 한 문장.

    확정이 아니라 추정입니다. 태그(정황)와 청산 사유(결말)를 맞대어 가장
    그럴듯한 이야기 하나를 고릅니다. 집계(감쇠표)는 이 문장이 아니라 태그로
    하므로, 문구를 다듬어도 학습 결과는 변하지 않습니다.
    """
    klass = exit_class(reason)
    rsi = snap.get("rsi")
    daily, intraday = snap.get("daily_score"), snap.get("intraday_score")
    atr_pct = snap.get("atr_pct")

    if "일봉분봉불일치" in tags:
        text = (f"일봉({float(daily):+.2f})과 분봉({float(intraday):+.2f})이 "
                f"반대였는데 단기 흐름을 따라 들어갔습니다")
    elif "과열진입" in tags:
        text = f"RSI {float(rsi):.0f} 과열 구간 진입 — 추격 매수로 추정"
    elif "뉴스의존" in tags:
        text = "뉴스 감성이 점수를 끌어올렸지만 가격이 따라오지 않았습니다"
    elif klass == "손절" and "고변동성" in tags:
        text = (f"변동성(ATR {float(atr_pct):.1f}%)이 커서 정상 노이즈에 "
                f"손절이 걸렸을 가능성")
    elif klass == "시간청산":
        text = "보유 기간이 다하도록 방향이 나오지 않았습니다 — 진입 근거가 약했던 것으로 추정"
    elif klass == "신호반전":
        text = "진입 근거가 청산 전에 무너졌습니다 — 신호의 지속력이 짧았습니다"
    else:
        lead = next((t.split(":", 1)[1] for t in tags if t.startswith("주도:")), "")
        text = (f"주도 성분({lead})이 실현되지 않았습니다" if lead
                else "진입 근거가 실현되지 않았습니다")

    regime = str(snap.get("regime") or "").strip()
    suffix = f" · 국면 {regime}" if regime else ""
    pnl_text = f", {pnl_pct:+.2f}%" if pnl_pct is not None else ""
    return f"{text} ({klass}{pnl_text}){suffix}"


# ---------------------------------------------------------------------------
# 태그별 기대수익 집계 → 감쇠표
# ---------------------------------------------------------------------------

def invalidate(user_id: int, mode: str):
    _stats_cache.pop((int(user_id), str(mode)), None)


def _tag_stats(user_id: int, mode: str, cfg: dict) -> dict:
    """닫힌 교훈들에서 태그별 (기대수익 E, 유효 표본 n)을 계산합니다.

    최근일수록 크게 반영합니다 — w = 0.5^(경과일/반감기). 시장은 변하고,
    한 달 전의 실수와 어제의 실수는 같은 무게가 아닙니다.
    """
    key = (int(user_id), str(mode))
    hit = _stats_cache.get(key)
    if hit and time.time() - hit[0] < STATS_TTL:
        return hit[1]

    from storage import autotrade as store
    rows = store.lessons_closed(user_id, mode, limit=400)
    half_life = max(1.0, float(cfg.get("learn_half_life_days", 14.0)))
    now = datetime.now()

    acc: dict[str, dict] = {}
    for row in rows:
        pnl = row.get("pnl_pct")
        if pnl is None:
            continue          # 외부 청산 등 손익을 모르는 매매는 집계에서 뺍니다
        try:
            age_days = max(0.0, (now - datetime.fromisoformat(
                str(row.get("closed_at")))).total_seconds() / 86400.0)
        except (TypeError, ValueError):
            age_days = 0.0
        w = 0.5 ** (age_days / half_life)
        for tag in row.get("tags") or []:
            slot = acc.setdefault(str(tag), {"w": 0.0, "we": 0.0})
            slot["w"] += w
            slot["we"] += w * float(pnl)

    stats = {tag: {"e": slot["we"] / slot["w"], "n": slot["w"]}
             for tag, slot in acc.items() if slot["w"] > 0}
    _stats_cache[key] = (time.time(), stats)
    return stats


def build_penalties(user_id: int, mode: str, cfg: dict) -> dict | None:
    """이번 회전이 쓸 감쇠표. 배울 것이 없으면 None."""
    learn = mode_of(cfg)
    if learn == OFF:
        return None
    try:
        stats = _tag_stats(user_id, mode, cfg)
    except Exception:
        return None           # 학습이 죽어도 매매는 계속되어야 합니다
    min_n = float(cfg.get("learn_min_trades", 3))
    scale = max(0.1, float(cfg.get("learn_scale_pct", 3.0)))
    per_tag_cap = float(cfg.get("learn_max_penalty", 0.10))

    tags = {}
    for tag, st in stats.items():
        # 방금 닫힌 매매도 나이가 0 이 아니라 w 가 1 에 살짝 못 미칩니다.
        # 3건이 2.9999… 로 집계되므로 하한 비교에 여유를 둡니다.
        if st["n"] >= min_n - 0.01 and st["e"] < 0:
            pen = per_tag_cap * min(1.0, abs(st["e"]) / scale)
            if pen >= 0.005:
                tags[tag] = {"e": round(st["e"], 3), "n": round(st["n"], 1),
                             "pen": round(pen, 4)}
    if not tags:
        return None
    return {"mode": learn, "tags": tags}


def inject(user_id: int, mode: str, cfg: dict) -> dict:
    """회전 시작에서 부릅니다 — cfg 사본에 감쇠표를 실어 돌려줍니다."""
    penalties = build_penalties(user_id, mode, cfg)
    if not penalties:
        return cfg
    return {**cfg, "_lesson_penalties": penalties}


def apply_to_score(score: float, sig, cfg: dict) -> tuple[float, str]:
    """지금 신호의 태그를 감쇠표와 대조합니다. (새 점수, 화면용 설명) 반환.

    observe 모드는 점수를 바꾸지 않고 "soft 였다면 얼마를 깎았을지"만
    알려줍니다 — 켜기 전에 학습이 무슨 짓을 할지 보여주기 위해서입니다.
    """
    table = (cfg or {}).get("_lesson_penalties") or {}
    pens = table.get("tags") or {}
    if not pens:
        return score, ""

    snap = {"score": score, "regime": sig.regime, "daily_score": sig.daily_score,
            "intraday_score": sig.intraday_score, "news_score": sig.news_score,
            "rsi": sig.rsi, "bb_pct": sig.bb_pct, "atr_pct": sig.atr_pct}
    matched = [(tag, pens[tag]) for tag in tags_of(snap, symbol=sig.key)
               if tag in pens]
    if not matched:
        return score, ""

    cap_total = float(cfg.get("learn_total_cap", 0.25))
    total = min(cap_total, sum(float(info["pen"]) for _, info in matched))
    parts = ", ".join(f"{tag} 평균 {info['e']:+.1f}%·{info['n']:.0f}회"
                      for tag, info in matched[:3])

    if str(table.get("mode")) != SOFT:
        return score, f"관찰 — 같은 조건에서 잃은 이력 (soft 면 -{total:.2f} 감쇠: {parts})"

    sign = 1.0 if score >= 0 else -1.0
    adjusted = sign * max(0.0, abs(score) - total)
    return adjusted, f"과거 손실 학습 -{total:.2f} ({parts})"


# ---------------------------------------------------------------------------
# 기록 훅 — 진입·청산에서 부릅니다 (실패해도 매매를 막지 않습니다)
# ---------------------------------------------------------------------------

def open_lesson(user_id: int, mode: str, symbol: str, name: str, side: str,
                sig_dict: dict, entry_price: float, quantity: float):
    """진입 체결 직후 — 이 매매의 컨텍스트를 저장합니다."""
    try:
        from storage import autotrade as store
        snap = snapshot(sig_dict)
        store.lesson_open(
            user_id, mode, symbol, name=name or "", side=side or "long",
            entry_price=float(entry_price or 0) or None,
            quantity=float(quantity or 0) or None,
            context=snap, tags=tags_of(snap, symbol=symbol))
    except Exception:
        pass          # 기록 실패가 주문 흐름을 깨면 안 됩니다


def close_lesson(user_id: int, mode: str, symbol: str, name: str = "",
                 exit_reason: str = "", exit_price: float = None,
                 realized_krw: float = None):
    """청산 체결 직후 — 손익을 확정하고 손실이면 원인을 추론해 남깁니다."""
    try:
        from storage import autotrade as store
        row = store.lesson_find_open(user_id, mode, symbol)
        if not row:
            return

        pnl_pct = None
        entry = row.get("entry_price")
        if entry and exit_price:
            direction = -1.0 if str(row.get("side") or "") == "short" else 1.0
            pnl_pct = (float(exit_price) - float(entry)) / float(entry) * 100 * direction

        cause = ""
        lost = ((pnl_pct is not None and pnl_pct < 0)
                or (pnl_pct is None and realized_krw is not None and realized_krw < 0))
        if lost:
            cause = diagnose(row.get("context") or {}, row.get("tags") or [],
                             exit_reason, pnl_pct)

        store.lesson_close(
            row["id"], exit_price=exit_price, exit_reason=str(exit_reason or ""),
            realized_krw=realized_krw, pnl_pct=pnl_pct, cause=cause)
        invalidate(user_id, mode)

        if cause:
            label = name or row.get("name") or symbol
            store.log_event(
                user_id, "lesson", f"{label} 손실 복기 — {cause}",
                level="warn", symbol=symbol, name=label,
                detail={"pnl_pct": pnl_pct, "realized_krw": realized_krw,
                        "exit_reason": exit_reason, "tags": row.get("tags"),
                        "context": row.get("context")})
    except Exception:
        pass
