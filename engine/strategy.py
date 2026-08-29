"""
자동매매 전략 (Signal → Decision)
--------------------------------
"살까 팔까"를 정하는 곳입니다. 실제 주문·리스크 통제는 여기서 하지 않습니다
(engine/risk.py, engine/autotrade.py 담당).

신호 구성
    일봉 기술점수   engine/indicators.analyze — 18개 지표의 국면 가중 평균
    분봉 기술점수   같은 지표를 분봉에 적용 (단기 흐름)
    뉴스 감성       선택 (느려서 기본 off — 켜면 30분 캐시로 사용)

    score = (1-w) × 일봉점수 + w × 분봉점수,  이후 뉴스 점수를 가중 혼합
    score 는 -1(강한 하락) ~ +1(강한 상승)

왜 확률이 아니라 점수인가
    예측 화면은 시그모이드로 확률을 만들지만, 매매는 "임계값을 넘었나"만
    필요합니다. 확률로 바꾸면 임계값 해석이 한 겹 더 꼬입니다.

손절 폭은 ATR 로 잡습니다
    고정 3% 손절은 삼성전자와 코스닥 소형주에 같은 잣대를 대는 것입니다.
    ATR(평균 진폭)의 배수로 잡으면 종목 변동성에 맞춰 자동으로 넓어지고 좁아집니다.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from engine import ensemble, feed, indicators, lessons, mlsignal, nnfx
from engine.instruments import FUTURES, OPTION, Instrument

# 지표 계산에 필요한 최소 봉 개수 — 이보다 적으면 신호를 만들지 않습니다
MIN_DAILY_BARS = 30
MIN_INTRADAY_BARS = 40

LONG = "long"
SHORT = "short"
FLAT = "flat"


@dataclass
class Signal:
    key: str
    ok: bool = False
    score: float = 0.0                 # -1 ~ +1
    direction: str = FLAT              # long | short | flat
    confidence: float = 0.0            # 0 ~ 1 (임계값 대비 초과분)
    price: float = 0.0                 # 호가 통화 기준 현재가
    price_krw: float = 0.0
    atr: float = 0.0                   # 절대 진폭 (호가 통화)
    atr_pct: float = 0.0               # 가격 대비 %
    daily_score: float = 0.0
    intraday_score: float | None = None
    news_score: float | None = None
    # 분봉을 빼고(비중 0) 같은 파이프라인을 태웠다면 나왔을 점수.
    # 사후에 재계산하지 않고 **여기서 같이 굴리는** 이유는, 어떤 오버레이가
    # 켜져 있었고 어떤 감쇠가 곱해졌는지를 이 함수 밖에서는 알 수 없기 때문입니다.
    # 판단에는 쓰지 않습니다 — "분봉이 실제로 결정을 바꿨나"를 로그로 답하기 위한 값입니다.
    score_wo_intraday: float | None = None
    entry_threshold: float = 0.0       # 그 회전에 실제로 적용된 진입 문턱
    regime: str = ""
    bars_used: int = 0
    quote_age: float = 0.0
    nnfx: dict | None = None           # NNFX 오버레이 (nnfx_mode 가 off 가 아닐 때만)
    ensemble: dict | None = None       # 앙상블 진단 (algo_mode 가 off 가 아닐 때만)
    ml: dict | None = None             # ML 오버레이 (ml_mode 가 off 가 아닐 때만)
    overheat: dict | None = None       # 과열 진입 게이트 판정 (방향이 정해졌을 때만)
    pullback: dict | None = None       # 눌림목 재진입 판정 (급등 뒤 첫 눌림인가)
    ma50: float | None = None          # 50일 이동평균 — 승자 보유 판정(check_exit)용
    # 과매수·과매도 판정에 쓰는 원값 (engine/split.py 의 분할 매수·매도).
    # 점수(score)만으로는 "과매도라서 담는다"를 판단할 수 없습니다 — 점수는
    # 19개 지표를 국면 가중으로 합친 값이라, RSI 가 28이어도 추세 지표가 좋으면
    # 양수로 나옵니다. 분할은 **그 지표 자체**를 봐야 하므로 따로 실어 나릅니다.
    rsi: float | None = None           # RSI(14)
    bb_pct: float | None = None        # 볼린저 %B (0=하단, 1=상단)
    vol_factor: float = 1.0            # 변동성 배수 — 손절·익절 폭 스케일에 사용
    # 점수가 만들어진 순서 — 화면에서 계산 과정을 그대로 보여주기 위한 기록입니다.
    # 화면이 이 과정을 따로 재현하면 반드시 엔진과 어긋납니다(어느 오버레이가
    # 켜져 있었는지, 어느 값이 실제로 곱해졌는지는 여기서만 알 수 있습니다).
    stages: list = field(default_factory=list)
    reasons: list = field(default_factory=list)
    error: str = ""

    def stage(self, key: str, label: str, value, detail: str = ""):
        self.stages.append({
            "key": key, "label": label, "detail": detail,
            "value": None if value is None else round(float(value), 4),
        })

    def to_dict(self) -> dict:
        return {
            "key": self.key, "ok": self.ok, "score": round(self.score, 4),
            "direction": self.direction, "confidence": round(self.confidence, 3),
            "price": self.price, "price_krw": self.price_krw,
            "atr": round(self.atr, 4), "atr_pct": round(self.atr_pct, 3),
            "daily_score": round(self.daily_score, 4),
            "intraday_score": (round(self.intraday_score, 4)
                               if self.intraday_score is not None else None),
            "news_score": (round(self.news_score, 4)
                           if self.news_score is not None else None),
            "score_wo_intraday": (round(self.score_wo_intraday, 4)
                                  if self.score_wo_intraday is not None else None),
            "entry_threshold": round(self.entry_threshold, 4),
            "regime": self.regime, "bars_used": self.bars_used,
            "quote_age": self.quote_age, "nnfx": self.nnfx,
            "ensemble": self.ensemble, "ml": self.ml,
            "overheat": self.overheat,
            "pullback": self.pullback,
            "ma50": round(self.ma50, 4) if self.ma50 else None,
            "rsi": round(self.rsi, 2) if self.rsi is not None else None,
            "bb_pct": round(self.bb_pct, 3) if self.bb_pct is not None else None,
            "vol_factor": round(self.vol_factor, 3),
            "stages": self.stages,
            "reasons": self.reasons[:6], "error": self.error,
        }


# ---------------------------------------------------------------------------
# 뉴스 점수 캐시 (선택 기능)
# ---------------------------------------------------------------------------

_news_cache: dict[str, tuple[float, float]] = {}
NEWS_TTL = 1800.0          # 30분 — 뉴스는 초 단위로 변하지 않습니다


def _news_score(inst: Instrument) -> float | None:
    """뉴스 감성 점수. 크롤링이 느려서 30분 캐시를 씁니다."""
    import time

    if inst.is_derivative or inst.symbol is None:
        return None            # 파생상품은 개별 뉴스가 없습니다

    hit = _news_cache.get(inst.key)
    if hit and time.time() - hit[0] < NEWS_TTL:
        return hit[1]
    try:
        from data_sources import news_crawler
        # include_save=False: 세이브(SAVE) 속보는 분석 화면 검증 중 — 주문 판단 제외
        items = news_crawler.get_news(inst.symbol, limit=15, include_save=False)
        score = news_crawler.aggregate_news_score(items)
    except Exception:
        return None
    _news_cache[inst.key] = (time.time(), float(score))
    return float(score)


# ---------------------------------------------------------------------------
# 신호 생성
# ---------------------------------------------------------------------------

def evaluate(inst: Instrument, cfg: dict, bars_daily: pd.DataFrame = None,
             bars_intraday: pd.DataFrame = None, quote: dict = None,
             allow_fetch: bool = True) -> Signal:
    """한 종목에 대한 매매 신호.

    bars/quote 를 주입할 수 있게 한 이유는 **백테스트와 실매매가 같은 함수를
    쓰게 하기 위해서**입니다. 시뮬레이터는 과거 봉을 잘라서 넣고, 실매매는
    None 을 넘겨 실시간 데이터를 받습니다.

    allow_fetch=False 면 주어진 데이터만 씁니다. 백테스트가 실수로 '오늘의
    실시간 시세'를 끌어와 미래를 훔쳐보는 것을 구조적으로 막습니다.
    """
    sig = Signal(key=inst.key)

    if quote is None and allow_fetch:
        quote = feed.quote(inst)
    if not quote or not quote.get("price"):
        sig.error = "현재가를 조회할 수 없습니다."
        return sig
    sig.price = float(quote["price"])
    sig.price_krw = float(quote.get("price_krw") or quote["price"])
    sig.quote_age = float(quote.get("age_sec") or 0)

    if bars_daily is None and allow_fetch:
        # 260개를 요청하는 이유: ML(gbdt)은 워밍업 60봉 + 학습 표본 80봉 + 검증이
        # 필요하고, 앙상블 난기류도 250봉 창을 씁니다. 제공처가 요청보다 적게
        # 주는 일이 있어서(소스 폴백에 따라 다름) 여유 있게 잡습니다.
        bars_daily = feed.bars(inst, "day", count=260)
    if bars_daily is None or len(bars_daily) < MIN_DAILY_BARS:
        sig.error = f"일봉이 부족합니다 ({0 if bars_daily is None else len(bars_daily)}개)"
        return sig

    daily = indicators.analyze(bars_daily)
    sig.daily_score = float(daily.score)
    sig.bars_used = daily.bars_used
    # 과매수·과매도 원값 — 이미 계산된 지표에서 꺼내 옵니다 (추가 계산 없음)
    for item in (daily.indicators or []):
        if item.key == "rsi":
            sig.rsi = (item.values or {}).get("rsi")
        elif item.key == "bollinger":
            sig.bb_pct = (item.values or {}).get("pct_b")
    sig.regime = (daily.regime or {}).get("label", "")
    sig.stage("daily", "일봉 지표 종합", sig.daily_score,
              f"{daily.bars_used}봉 · 지표 {len(daily.indicators)}개를 국면 가중으로 합산"
              + (f" · 국면 {sig.regime}" if sig.regime else ""))

    intraday_weight = float(cfg.get("intraday_weight", 0.35))
    if intraday_weight > 0:
        if bars_intraday is None and allow_fetch:
            bars_intraday = feed.bars(inst, "minute", count=240)
        if bars_intraday is not None and len(bars_intraday) >= MIN_INTRADAY_BARS:
            sig.intraday_score = float(indicators.analyze(bars_intraday).score)
            sig.stage("intraday", "분봉 단기 점수", sig.intraday_score,
                      f"같은 지표를 분봉으로 · 결합 비중 {intraday_weight:.0%}")

    # 앙상블 (engine/ensemble.py) — 시평선 합치·난기류·변동성 배수·시장 방향.
    # observe 모드는 계산·기록만 하고 점수를 바꾸지 않습니다.
    # market 은 실시간 경로에서만 넘깁니다 — 백테스트(allow_fetch=False)가
    # 지금의 지수 상태를 읽으면 과거 판단에 현재 정보가 섞입니다.
    ens = ensemble.compute(bars_daily, sig.daily_score, sig.intraday_score, cfg,
                           market=(inst.market if allow_fetch else ""))
    ens_mode = ensemble.mode_of(cfg)
    if ens.ok:
        sig.ensemble = ens.to_dict()
        sig.vol_factor = ens.vol_factor

    # alt = 분봉 비중을 0 으로 뒀다면 나왔을 점수. 아래 모든 단계에서 blended 와
    # **같은 변환을 그대로** 받습니다 (뉴스·NNFX·ML·학습 감쇠는 점수의 함수라
    # 두 번 적용해도 모델을 다시 계산하지 않습니다).
    if ens.ok and ens_mode in (ensemble.SOFT, ensemble.GATE):
        blended = ens.score          # 시평선 불일치 감산·난기류 감쇠가 반영된 점수
        # 분봉이 없으면 시평선 합치 결과가 일봉 점수 그대로이므로, 그 뒤에 곱해진
        # 감쇠(ens.damping)만 같이 적용하면 반사실 점수가 정확히 나옵니다.
        alt = max(-1.0, min(1.0, sig.daily_score * ens.damping))
        sig.stage("ensemble", "앙상블 결합", blended,
                  "시평선 불일치 감산 · 난기류 감쇠 반영"
                  + (f" · {ens.notes[0]}" if ens.notes else ""))
    elif sig.intraday_score is None:
        blended = sig.daily_score
        alt = blended
        sig.stage("blend", "결합 점수", blended, "분봉이 없어 일봉 점수를 그대로")
    else:
        blended = ((1 - intraday_weight) * sig.daily_score
                   + intraday_weight * sig.intraday_score)
        alt = sig.daily_score
        sig.stage("blend", "일봉·분봉 결합", blended,
                  f"일봉 {1 - intraday_weight:.0%} + 분봉 {intraday_weight:.0%}")

    if cfg.get("use_news"):
        news = _news_score(inst)
        if news is not None:
            sig.news_score = news
            weight = float(cfg.get("news_weight", 0.25))
            blended = (1 - weight) * blended + weight * news
            alt = (1 - weight) * alt + weight * news
            sig.stage("news", "뉴스 감성 반영", blended,
                      f"뉴스 점수 {news:+.2f} 을 {weight:.0%} 비중으로 결합")

    # 인버스 ETF 는 기초자산이 내릴 때 오릅니다. 지표는 그 ETF 자체의 가격으로
    # 계산되므로 뒤집을 필요가 없지만, 뉴스 점수는 기초자산 기준이라 뒤집습니다.
    if inst.is_inverse and sig.news_score is not None:
        blended -= 2 * float(cfg.get("news_weight", 0.25)) * sig.news_score
        alt -= 2 * float(cfg.get("news_weight", 0.25)) * sig.news_score

    # NNFX 규칙 오버레이 — 켜져 있을 때만. 점수에 **더하지 않고 섞습니다**
    # (슬롯이 보는 추세·모멘텀은 위 지표 점수에 이미 들어 있어, 더하면 이중 계상).
    nnfx_mode = nnfx.mode_of(cfg)
    nnfx_state, nnfx_notes = None, []
    if nnfx_mode != nnfx.OFF:
        nnfx_state = nnfx.compute(bars_daily, cfg)
        sig.nnfx = nnfx_state.to_dict()
        if nnfx_mode == nnfx.SOFT:
            blended, note = nnfx.apply_to_score(blended, nnfx_state, cfg)
            alt, _ = nnfx.apply_to_score(alt, nnfx_state, cfg)
            if note:
                nnfx_notes.append(note)
            sig.stage("nnfx", "NNFX 규칙 결합", blended,
                      note or "기준선·모멘텀·추세·거래량 슬롯 결합")

    # ML 오버레이 (engine/mlsignal.py — XGBoost·PatchTST 이식).
    # observe 는 계산·첨부만, soft 는 검증을 통과한 점수만 가중 결합합니다.
    ml_mode = mlsignal.mode_of(cfg)
    if ml_mode != mlsignal.OFF:
        ml_state = mlsignal.compute(inst.key, bars_daily, cfg)
        if ml_state.ok or ml_state.error:
            sig.ml = ml_state.to_dict()
        if ml_mode == mlsignal.SOFT:
            blended, note = mlsignal.apply_to_score(blended, ml_state, cfg)
            alt, _ = mlsignal.apply_to_score(alt, ml_state, cfg)
            if note:
                nnfx_notes.append(note)
            sig.stage("ml", "ML 오버레이", blended,
                      note or "검증을 통과한 모델 점수만 가중 결합")

    sig.score = max(-1.0, min(1.0, blended))
    sig.score_wo_intraday = max(-1.0, min(1.0, alt))
    sig.atr, sig.atr_pct = _atr(bars_daily, sig.price)
    if len(bars_daily) >= 50:
        try:
            sig.ma50 = float(bars_daily["close"].iloc[-50:].mean())
        except (KeyError, TypeError, ValueError):
            sig.ma50 = None

    # 손실 학습 감쇠 (engine/lessons.py) — 과거에 돈을 잃었던 컨텍스트(태그)와
    # 지금 신호의 컨텍스트를 대조해 점수를 깎습니다. 회전(_tick)이 cfg 사본에
    # 실어준 감쇠표가 있을 때만 동작하므로 백테스트에는 닿지 않고(미래 참조
    # 없음), soft 모드에서만 실제로 점수를 바꿉니다 (observe 는 기록만).
    if cfg.get("_lesson_penalties"):
        adjusted, lesson_note = lessons.apply_to_score(sig.score, sig, cfg)
        if lesson_note:
            sig.score = adjusted
            sig.score_wo_intraday, _ = lessons.apply_to_score(
                sig.score_wo_intraday, sig, cfg)
            sig.stage("lessons", "손실 학습", sig.score, lesson_note)

    entry = float(cfg.get("entry_score", 0.35))
    allow_short = bool(cfg.get("allow_short")) and inst.shortable

    # 급등 뒤 첫 정상 눌림이면 **롱 문턱만** 낮춥니다. 숏에 같은 논리를 쓰려면
    # 하락 추세의 되돌림을 봐야 하는데, 그건 반대 방향의 별개 판정입니다.
    pull = pullback_ready(bars_daily, sig.price, sig.atr, cfg)
    sig.pullback = pull
    entry_long = entry
    if pull["ready"]:
        entry_long = entry * pull["params"]["pullback_entry_ratio"]
        sig.stage("pullback", "눌림목 재진입", entry_long, pull["note"])

    if sig.score >= entry_long:
        sig.direction = LONG
    elif sig.score <= -entry and allow_short:
        sig.direction = SHORT
    else:
        sig.direction = FLAT

    # 게이트 모드는 방향이 정해진 뒤에 막습니다. 신규 진입만 막고 청산은
    # 건드리지 않습니다 — 못 들어가는 것은 기회 손실이지만, 못 나오는 것은 손실입니다.
    if nnfx_state is not None and nnfx_state.ok and sig.direction == LONG:
        blocked = (nnfx_mode == nnfx.VETO and not nnfx_state.veto_ok) or \
                  (nnfx_mode == nnfx.HARD and not nnfx_state.hard_ok)
        if blocked:
            sig.direction = FLAT
            nnfx_notes.insert(0, "NNFX 게이트 차단 — "
                              + (nnfx_state.reasons[0] if nnfx_state.reasons
                                 else "슬롯 조건 미충족"))

    # 앙상블 게이트 — 난기류 극단 등. NNFX 와 같은 원칙(신규 진입만)입니다.
    if ens.ok and ens.block and sig.direction != FLAT:
        sig.direction = FLAT
        nnfx_notes.insert(0, ens.blocked_by[0] if ens.blocked_by else "앙상블 게이트 차단")

    # 과열 진입 게이트 — 점수가 임계값을 넘었어도 '이미 많이 간 자리'면 막습니다.
    # 지표 점수 자체가 급등 고점에서 강한 매수로 나오는 구조적 편향을 여기서
    # 차단합니다 (위 overheat_check 주석에 이유를 적어두었습니다).
    if sig.direction != FLAT:
        heat = overheat_check(sig.direction, sig.price, sig.atr, sig.rsi, sig.bb_pct,
                              bars_daily, cfg)
        sig.overheat = heat
        if heat["blocked"]:
            sig.direction = FLAT
            nnfx_notes.insert(0, heat["note"])
        if heat["note"]:
            sig.stage("overheat", "과열 진입 게이트", heat["hits"], heat["note"])

    # 눌림목 완화는 롱에만 적용되므로, 확신도·표시 문턱도 방향에 맞춰 고릅니다.
    # (숏에 완화된 롱 문턱을 쓰면 확신도가 부풀려집니다)
    eased = pull["ready"] and sig.direction != SHORT
    applied = entry_long if eased else entry
    sig.entry_threshold = float(applied)
    sig.stage("final", "최종 점수", sig.score,
              f"진입 임계값 +{applied:g}"
              + (f" (눌림목 완화, 원래 {entry:g})" if eased else "")
              + (f" · 숏 -{entry:g}" if allow_short else "")
              + " · 판정 "
              + {LONG: "매수", SHORT: "매도(숏)"}.get(sig.direction, "관망"))

    sig.confidence = min(1.0, abs(sig.score) / applied) if applied > 0 else 0.0
    # _reasons 가 목록을 새로 만들므로, NNFX·앙상블 사유는 **그 뒤에** 붙여야 남습니다.
    # (앞에 붙이면 조용히 덮어써져 "왜 막혔는지 모르는" 상태가 됩니다)
    ens_notes = ens.notes[:2] if ens.ok else []
    sig.reasons = nnfx_notes + ens_notes + _reasons(daily, sig)
    sig.ok = True
    return sig


# ---------------------------------------------------------------------------
# 과열 진입 게이트 (Overheat Entry Gate)
# ---------------------------------------------------------------------------
#
# 왜 필요한가
#     지표 점수(engine/indicators.analyze)는 급등 고점에서 오히려 **강한 매수**로
#     나옵니다. 구조적인 이유가 셋입니다.
#       ① 추세추종 지표 9개가 사실상 같은 질문("최근에 올랐나")을 하고, 급등하면
#          9개가 동시에 +1.0 으로 포화합니다 — 상관 높은 한 표를 아홉 표로 셉니다.
#       ② 국면 판정이 순환적입니다. 수직 급등이면 ADX·Hurst·분산비율이 전부
#          '추세 국면'을 가리키고, 그러면 되돌림군 가중치가 ×0.55 로 깎입니다.
#          과열 경고가 가장 필요한 순간에 그 경고의 목소리가 가장 작아집니다.
#       ③ ROC·이평이격은 ±10%/±5% 에서 만점입니다. 20% 급등이든 60% 급등이든
#          점수는 똑같은 +1.0 — 더 위험해질수록 페널티가 붙지 않습니다.
#
#     실측: 합성 급등 고점(RSI 99 · %B 0.95 · %K 94)에서 최종 기술점수 +0.65,
#     기본 진입 임계값 0.35 를 넘어 **매수**가 나왔습니다.
#
# 그래서 점수를 고치지 않고 진입만 막습니다
#     점수 산식을 손대면 예측 화면·기존 백테스트 결과가 통째로 바뀝니다. 이
#     게이트는 NNFX·앙상블 게이트와 같은 자리(방향 판정 뒤)에서 **신규 진입만**
#     막습니다. 청산은 건드리지 않습니다 — 못 들어가는 것은 기회 손실이지만,
#     못 나오는 것은 손실입니다.
#
# 세 가지를 각각 세고 min_hits 개 이상 걸릴 때만 막습니다
#     지표 하나로 막으면 정상 추세까지 통째로 죽습니다. 서로 다른 각도
#     (모멘텀 과열 · 밴드 상단 · 이평 이격)에서 동시에 걸릴 때만 추격으로 봅니다.

OVERHEAT_DEFAULTS = {
    "overheat_gate": True,        # 과열 구간 신규 진입 차단
    "overheat_rsi": 75.0,         # RSI(14) 이 값 이상이면 과열 1표 (숏은 100-이 값)
    "overheat_bb": 0.95,          # 볼린저 %B 이 값 이상이면 과열 1표 (숏은 1-이 값)
    "overheat_ext_atr": 3.0,      # MA20 에서 이 배수(ATR)만큼 벌어지면 과열 1표
    "overheat_min_hits": 2,       # 몇 표부터 막을지 (1~3)
    "overheat_ma": 20,            # 이격을 재는 기준 이동평균
}


def overheat_params(cfg: dict) -> dict:
    """설정에서 게이트 파라미터를 꺼내 안전 범위로 조입니다."""
    p = {**OVERHEAT_DEFAULTS}
    for k in OVERHEAT_DEFAULTS:
        if cfg.get(k) is not None:
            p[k] = cfg[k]
    p["overheat_gate"] = bool(p["overheat_gate"])
    p["overheat_rsi"] = max(50.0, min(float(p["overheat_rsi"]), 99.0))
    p["overheat_bb"] = max(0.5, min(float(p["overheat_bb"]), 1.5))
    p["overheat_ext_atr"] = max(0.5, float(p["overheat_ext_atr"]))
    p["overheat_min_hits"] = max(1, min(int(p["overheat_min_hits"]), 3))
    p["overheat_ma"] = max(5, min(int(p["overheat_ma"]), 120))
    return p


def overheat_check(direction: str, price: float, atr: float, rsi, bb_pct,
                   bars_daily: pd.DataFrame, cfg: dict) -> dict:
    """지금 진입하면 추격매수(또는 추격매도)인가.

    반환 {blocked, hits, note, detail}. 판정 재료가 하나도 없으면 막지 않습니다
    (모르는 것을 위험하다고 취급하면 게이트가 아니라 정지 버튼이 됩니다).
    """
    p = overheat_params(cfg)
    out = {"blocked": False, "hits": 0, "note": "", "detail": {}, "params": p}
    if not p["overheat_gate"] or direction not in (LONG, SHORT) or price <= 0:
        return out

    long_side = direction == LONG
    hits, notes = [], []

    # ① 모멘텀 과열 — RSI
    rsi_limit = p["overheat_rsi"] if long_side else 100.0 - p["overheat_rsi"]
    if rsi is not None:
        rsi = float(rsi)
        out["detail"]["rsi"] = round(rsi, 2)
        out["detail"]["rsi_limit"] = round(rsi_limit, 2)
        if (rsi >= rsi_limit) if long_side else (rsi <= rsi_limit):
            hits.append("rsi")
            notes.append(f"RSI {rsi:.1f} {'≥' if long_side else '≤'} {rsi_limit:g}")

    # ② 밴드 이탈 — 볼린저 %B
    bb_limit = p["overheat_bb"] if long_side else 1.0 - p["overheat_bb"]
    if bb_pct is not None:
        bb_pct = float(bb_pct)
        out["detail"]["bb_pct"] = round(bb_pct, 3)
        out["detail"]["bb_limit"] = round(bb_limit, 3)
        if (bb_pct >= bb_limit) if long_side else (bb_pct <= bb_limit):
            hits.append("bollinger")
            notes.append(f"%B {bb_pct:.2f} {'≥' if long_side else '≤'} {bb_limit:g}")

    # ③ 이격도 — 이동평균에서 몇 ATR 떨어져 있는가.
    #    ATR 로 재는 이유: 고정 %는 삼성전자와 코스닥 소형주에 같은 잣대를 댑니다.
    ma_n = p["overheat_ma"]
    if atr and atr > 0 and bars_daily is not None and len(bars_daily) >= ma_n:
        try:
            ma = float(bars_daily["close"].iloc[-ma_n:].mean())
        except (KeyError, TypeError, ValueError):
            ma = 0.0
        if ma > 0:
            ext = (price - ma) / atr          # +면 이평 위, -면 아래
            signed = ext if long_side else -ext
            out["detail"]["ext_atr"] = round(ext, 2)
            out["detail"]["ext_limit"] = p["overheat_ext_atr"]
            out["detail"]["ma"] = round(ma, 4)
            if signed >= p["overheat_ext_atr"]:
                hits.append("extension")
                notes.append(f"{ma_n}일선에서 {abs(ext):.1f} ATR "
                             f"{'위' if long_side else '아래'} (한도 {p['overheat_ext_atr']:g})")

    out["hits"] = len(hits)
    out["detail"]["hit_keys"] = hits
    if len(hits) >= p["overheat_min_hits"]:
        out["blocked"] = True
        out["note"] = (("과열 진입 차단" if long_side else "과냉 진입 차단")
                       + f" — {' · '.join(notes)}"
                       + f" ({len(hits)}/{p['overheat_min_hits']}표). "
                       + ("이미 많이 오른 자리라 추격매수로 봅니다."
                          if long_side else "이미 많이 내린 자리라 추격매도로 봅니다."))
    elif notes:
        out["note"] = (f"과열 신호 {len(hits)}표 ({' · '.join(notes)}) — "
                       f"차단 기준 {p['overheat_min_hits']}표 미만이라 통과")
    return out


# ---------------------------------------------------------------------------
# 눌림목 재진입 (Pullback Re-entry)
# ---------------------------------------------------------------------------
#
# 게이트만 있으면 절반짜리입니다. 고점 추격은 막지만, 그 종목을 **영영 못 삽니다**.
# 강했던 종목이 눌릴 때쯤이면 추세 지표가 식어 점수가 임계값 아래로 내려가 있고,
# 정작 위험이 낮아진 자리에서 진입 신호가 사라집니다. 남는 건 기회 손실뿐입니다.
#
# 그래서 "최근에 강했고 · 지금은 눌렸고 · 추세는 살아 있는" 자리에서만 진입
# 문턱을 낮춥니다. 급등 직후 첫 정상 눌림이 손절폭 대비 보상이 가장 좋다는 것은
# 추세매매의 오래된 관찰입니다 (돌파에 사서 물리면 손절이 멀어 한 번 지면 여러 번
# 이긴 것이 지워집니다).
#
# **봉에서만 계산합니다 — 외부 상태를 쓰지 않습니다.**
#     "언제 무장했는지"를 메모리에 들고 있으면 백테스트와 실매매가 갈라집니다.
#     (재시작하면 사라지고, 과거를 다시 돌리면 재현되지 않습니다.) 최근 강세도
#     지금의 눌림도 전부 과거 봉에서 다시 계산할 수 있으므로, 순수 함수로 둡니다.
#     engine/strategy.evaluate 가 백테스트와 실매매에서 같은 함수인 이유와 같습니다.

PULLBACK_DEFAULTS = {
    # 기본 꺼짐 — 논리는 맞고 합성 시나리오에서 의도대로 동작하지만, 검증
    # 하네스가 수익 개선을 기각했습니다 (config.REGIME_MULTIPLIER 주석 참고).
    # 문턱을 낮추는 장치라 근거 없이 켜두면 진입만 늘어납니다.
    "pullback_entry": False,      # 눌림목 재진입 허용
    "pullback_lookback": 10,      # 최근 몇 봉 안의 강세를 볼지
    "pullback_surge_atr": 2.5,    # 그 구간에 이만큼(ATR) 떠 있었어야 '강했다'
    "pullback_max_atr": 1.5,      # 지금은 이 안으로 들어와 있어야 '눌렸다'
    "pullback_entry_ratio": 0.70, # 진입 임계값을 이 비율로 낮춤 (0.35 -> 0.245)
    "pullback_ma": 20,            # 기준 이동평균
}


def pullback_params(cfg: dict) -> dict:
    p = {**PULLBACK_DEFAULTS}
    for k in PULLBACK_DEFAULTS:
        if cfg.get(k) is not None:
            p[k] = cfg[k]
    p["pullback_entry"] = bool(p["pullback_entry"])
    p["pullback_lookback"] = max(2, min(int(p["pullback_lookback"]), 60))
    p["pullback_surge_atr"] = max(0.5, float(p["pullback_surge_atr"]))
    p["pullback_max_atr"] = max(0.0, float(p["pullback_max_atr"]))
    # 문턱을 0.3 미만으로 낮추는 것은 사실상 임계값을 없애는 것입니다
    p["pullback_entry_ratio"] = max(0.30, min(float(p["pullback_entry_ratio"]), 1.0))
    p["pullback_ma"] = max(5, min(int(p["pullback_ma"]), 120))
    return p


def pullback_ready(bars_daily: pd.DataFrame, price: float, atr: float,
                   cfg: dict) -> dict:
    """지금이 '급등 뒤 첫 정상 눌림' 자리인가.

    세 가지가 **모두** 맞아야 합니다.
      ① 최근 lookback 봉 안에서 이동평균 위로 surge_atr 이상 떠 있던 적이 있다
      ② 지금은 max_atr 안으로 되돌아왔다
      ③ 추세가 살아 있다 — 종가가 이동평균 위이고, 이동평균이 오르고 있다

    ③ 이 없으면 '눌림'과 '추세 붕괴'를 구분하지 못합니다. 떨어지는 칼을 눌림목
    이라고 부르는 것이 이 전략이 죽는 가장 흔한 방식입니다.
    """
    p = pullback_params(cfg)
    out = {"ready": False, "note": "", "detail": {}, "params": p}
    n = p["pullback_ma"]
    if not p["pullback_entry"] or bars_daily is None or price <= 0 or atr <= 0:
        return out
    if len(bars_daily) < n + p["pullback_lookback"]:
        return out

    try:
        close = bars_daily["close"]
        ma = close.rolling(n).mean()
        ext = (close - ma) / atr                       # 이동평균에서 몇 ATR 위인가
        window = ext.iloc[-p["pullback_lookback"]:].dropna()
        if window.empty:
            return out
        peak = float(window.max())
        now_ext = (price - float(ma.iloc[-1])) / atr
        ma_now, ma_prev = float(ma.iloc[-1]), float(ma.iloc[-6])
    except (KeyError, TypeError, ValueError, IndexError):
        return out

    surged = peak >= p["pullback_surge_atr"]
    pulled = now_ext <= p["pullback_max_atr"]
    alive = price > ma_now and ma_now > ma_prev

    out["detail"] = {"peak_atr": round(peak, 2), "now_atr": round(now_ext, 2),
                     "ma": round(ma_now, 4), "ma_rising": ma_now > ma_prev,
                     "above_ma": price > ma_now,
                     "surged": surged, "pulled": pulled, "alive": alive}

    if surged and pulled and alive:
        out["ready"] = True
        out["note"] = (f"눌림목 재진입 — 최근 {p['pullback_lookback']}봉 안에 "
                       f"{n}일선 위 {peak:.1f} ATR 까지 떴다가 지금 {now_ext:+.1f} ATR 로 "
                       f"되돌아왔고, {n}일선은 아직 오르고 있습니다. "
                       f"진입 문턱을 {p['pullback_entry_ratio']:.0%} 로 낮춥니다.")
    return out


def _atr(df: pd.DataFrame, price: float, period: int = 14) -> tuple[float, float]:
    """평균 진폭. 손절 폭 산정에 씁니다."""
    try:
        high, low, close = df["high"], df["low"], df["close"]
        prev = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
                       axis=1).max(axis=1)
        value = float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0, 0.0
    if not value or value != value:          # NaN 방어
        return 0.0, 0.0
    return value, (value / price * 100) if price else 0.0


def _reasons(analysis, sig: Signal) -> list[str]:
    """왜 이 방향인지 — 감사 로그에 남길 사람이 읽는 근거."""
    out = []
    ranked = sorted(
        [i for i in analysis.indicators if i.weight > 0 and abs(i.score) >= 0.15],
        key=lambda i: abs(i.score * i.weight), reverse=True)
    for ind in ranked[:3]:
        out.append(f"{ind.label} {ind.value_text} ({ind.score:+.2f})")
    if sig.intraday_score is not None:
        out.append(f"분봉 {sig.intraday_score:+.2f}")
    if sig.news_score is not None:
        out.append(f"뉴스 {sig.news_score:+.2f}")
    if sig.regime:
        out.append(f"국면 {sig.regime}")
    return out


# ---------------------------------------------------------------------------
# 진입 계획 (포지션 사이징)
# ---------------------------------------------------------------------------

@dataclass
class EntryPlan:
    ok: bool = False
    quantity: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    notional_krw: float = 0.0
    margin_krw: float = 0.0
    risk_krw: float = 0.0
    reason: str = ""
    # 수량이 왜 그 숫자인지 — "왜 1주밖에 안 사지?" 에 답하기 위한 내역.
    # 계산 자체는 예전과 같고, 어느 한도가 잘랐는지를 기록만 합니다.
    sizing: dict = field(default_factory=dict)

    @property
    def over_budget(self) -> bool:
        """최소 1주 허용 때문에 실제 위험이 예산을 넘긴 상태인가."""
        return bool(self.sizing.get("over_budget"))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "quantity": self.quantity, "price": self.price,
                "stop_price": round(self.stop_price, 4),
                "target_price": round(self.target_price, 4),
                "notional_krw": round(self.notional_krw), "margin_krw": round(self.margin_krw),
                "risk_krw": round(self.risk_krw), "reason": self.reason,
                "sizing": self.sizing}


def stop_distance(inst: Instrument, sig: Signal, cfg: dict) -> float:
    """손절까지의 가격 폭 (호가 통화 기준).

    ATR 배수를 우선 쓰고, ATR 을 못 구하면 고정 퍼센트로 떨어집니다.
    """
    atr_mult = float(cfg.get("atr_stop_mult", 2.0))
    if atr_mult > 0 and sig.atr > 0:
        return sig.atr * atr_mult
    return sig.price * float(cfg.get("stop_loss_pct", 3.0)) / 100.0


def plan_entry(inst: Instrument, sig: Signal, cfg: dict, account: dict) -> EntryPlan:
    """얼마나 살 것인가.

    1) 리스크 기반 수량: 손절까지 갔을 때 잃는 금액이 총자산의 risk_per_trade_pct
       가 되도록 수량을 정합니다. (변동성이 큰 종목은 자동으로 적게 삽니다)
    2) 비중 상한: 한 종목이 총자산의 position_pct 를 넘지 않게 자릅니다.
    3) 주문 금액 상한 / 가용 현금(파생은 증거금)으로 한 번 더 자릅니다.
    """
    plan = EntryPlan(price=sig.price)

    # 변동성 배수 (engine/ensemble.py) — soft/gate 모드에서만 고정 % 계열의
    # 손절·익절·트레일링 폭을 스케일합니다. ATR 손절은 이미 변동성이라 제외.
    # 저장된 설정이 아니라 이번 판단의 사본에만 적용됩니다.
    ens_mode = ensemble.mode_of(cfg)
    if ens_mode in (ensemble.SOFT, ensemble.GATE):
        scaled = ensemble.scale_barriers(cfg, getattr(sig, "vol_factor", 1.0))
        if scaled:
            cfg = {**cfg, **scaled}

    total_value = float(account.get("total_value") or 0)
    available = float(account.get("available_cash") or 0)
    if total_value <= 0:
        plan.reason = "계좌 평가금액이 0입니다."
        return plan

    distance = stop_distance(inst, sig, cfg)
    if distance <= 0:
        plan.reason = "손절 폭을 계산할 수 없습니다."
        return plan

    direction = 1 if sig.direction == LONG else -1
    plan.stop_price = sig.price - direction * distance
    take_pct = float(cfg.get("take_profit_pct", 0) or 0)
    if take_pct > 0:
        plan.target_price = sig.price * (1 + direction * take_pct / 100.0)
    else:
        rr = float(cfg.get("reward_risk", 2.0))
        plan.target_price = sig.price + direction * distance * rr

    # 비용 게이트 (gate 모드만) — 목표 수익이 왕복 수수료·거래세·슬리피지를
    # 충분히 넘지 못하면 이길 수 없는 매매입니다. 여기서 걸러야 수량 계산과
    # 리스크 게이트를 헛돌지 않습니다.
    if ens_mode == ensemble.GATE:
        edge = ensemble.cost_edge(inst, sig.price, plan.target_price, cfg)
        if not edge.get("ok", True):
            plan.reason = "기대이익이 거래비용에 못 미칩니다 — " + str(edge.get("reason", ""))
            return plan

    # 1) 리스크 기반
    risk_budget = total_value * float(cfg.get("risk_per_trade_pct", 1.0)) / 100.0
    krw_per_point = inst.multiplier * (sig.price_krw / sig.price if sig.price else 1)
    loss_per_unit = distance * krw_per_point
    if loss_per_unit <= 0:
        plan.reason = "단위당 손실을 계산할 수 없습니다."
        return plan
    qty = risk_budget / loss_per_unit
    qty_by_risk = qty
    bound_by = "1회 위험 예산"

    # 2) 비중 상한 (파생은 증거금 기준, 현물은 매수금액 기준)
    cap_krw = total_value * float(cfg.get("position_pct", 20.0)) / 100.0
    max_order = float(cfg.get("max_order_krw", 0) or 0)
    if max_order > 0:
        cap_krw = min(cap_krw, max_order)

    unit_cost = _unit_cost_krw(inst, sig)
    if unit_cost <= 0:
        plan.reason = "단위 비용을 계산할 수 없습니다."
        return plan
    qty_by_weight = cap_krw / unit_cost
    if qty_by_weight < qty:
        qty, bound_by = qty_by_weight, "종목 비중 상한"

    # 3) 가용 현금 (수수료 여유 1% 남김)
    qty_by_cash = available * 0.99 / unit_cost
    if qty_by_cash < qty:
        qty, bound_by = qty_by_cash, "주문가능금액"

    plan.sizing = {
        "bound_by": bound_by,
        "qty_raw": round(qty, 4),
        "qty_by_risk": round(qty_by_risk, 4),
        "qty_by_weight": round(qty_by_weight, 4),
        "qty_by_cash": round(qty_by_cash, 4),
        "risk_budget_krw": round(risk_budget),
        "loss_per_unit_krw": round(loss_per_unit),
        "unit_cost_krw": round(unit_cost),
        "position_cap_krw": round(cap_krw),
        "total_value_krw": round(total_value),
    }

    qty = inst.round_quantity(qty)
    if qty <= 0:
        affordable = unit_cost <= min(cap_krw, available * 0.99)
        if cfg.get("min_one_unit") and not inst.is_derivative and affordable:
            # 소액 계좌 — 리스크 예산으로는 1주 미만이지만 현금·비중 한도 안이면
            # 최소 1주는 허용합니다. 이때 실제 위험이 예산을 넘는다는 사실은
            # plan.risk_krw 와 sizing 에 그대로 남습니다 (숨기지 않습니다).
            qty = 1.0
            bound_by = "최소 1주 허용"
            plan.sizing["bound_by"] = bound_by
            plan.sizing["forced_one_unit"] = True
        elif loss_per_unit > risk_budget and affordable:
            plan.reason = (f"리스크 예산 {risk_budget:,.0f}원 < 1주 손절 위험 "
                           f"{loss_per_unit:,.0f}원 — 1회 위험 예산(%)을 올리거나 "
                           f"'최소 1주 허용'을 켜세요")
            return plan
        elif unit_cost > cap_krw:
            # 1주 값이 종목 비중 상한보다 비싼 경우. 예전에는 이때도 "가용
            # 현금이 모자랍니다"로 찍혀서, 현금이 충분한데도 왜 안 사는지
            # 알 수 없었습니다 (소액 계좌 + 고가주에서 항상 이쪽입니다).
            plan.reason = (f"1주 {unit_cost:,.0f}원이 종목 비중 상한 "
                           f"{cap_krw:,.0f}원(총자산의 "
                           f"{float(cfg.get('position_pct', 20.0)):g}%)을 넘습니다 — "
                           f"'종목당 최대 비중'을 올리거나 더 싼 종목이어야 합니다")
            return plan
        else:
            plan.reason = (f"주문 가능 수량이 0입니다 — 1주 {unit_cost:,.0f}원, "
                           f"주문가능금액 {available:,.0f}원")
            return plan

    side = "buy" if sig.direction == LONG else "sell"
    plan.quantity = qty
    plan.notional_krw = inst.notional(sig.price_krw, qty)
    plan.margin_krw = inst.margin_required(sig.price_krw, qty, side)
    plan.risk_krw = loss_per_unit * qty
    plan.ok = True

    # 실제 위험이 예산의 몇 %인가 — 최소 1주 허용이 켜져 있으면 이 값이 설정을
    # 넘길 수 있습니다. 넘겼다는 사실을 여기서 계산해 두면, 호출부가 로그에
    # 경고로 남길 수 있습니다 (조용히 넘어가면 리스크 모형이 무너진 줄 모릅니다).
    risk_pct = (plan.risk_krw / total_value * 100) if total_value > 0 else 0.0
    budget_pct = float(cfg.get("risk_per_trade_pct", 1.0))
    plan.sizing.update({
        "quantity": qty,
        "risk_krw": round(plan.risk_krw),
        "risk_pct": round(risk_pct, 2),
        "budget_pct": budget_pct,
        "over_budget": bool(plan.risk_krw > risk_budget * 1.05),
    })
    if not plan.reason:
        plan.reason = (f"{bound_by} 기준 {qty:g}{'주' if not inst.is_derivative else '계약'} "
                       f"(1주 {unit_cost:,.0f}원 · 손절 시 위험 {plan.risk_krw:,.0f}원 "
                       f"= 자산의 {risk_pct:.1f}%)")

    min_order = float(cfg.get("min_order_krw", 0) or 0)
    basis = plan.margin_krw if inst.is_derivative else plan.notional_krw
    if min_order > 0 and basis < min_order:
        plan.ok = False
        plan.reason = f"주문 금액이 최소 {min_order:,.0f}원에 못 미칩니다 ({basis:,.0f}원)"
    return plan


def _unit_cost_krw(inst: Instrument, sig: Signal) -> float:
    """1주(1계약)를 잡는 데 실제로 필요한 원화."""
    side = "buy" if sig.direction != SHORT else "sell"
    return inst.margin_required(sig.price_krw, 1, side)


# ---------------------------------------------------------------------------
# 청산 판단
# ---------------------------------------------------------------------------

@dataclass
class ExitDecision:
    should_exit: bool = False
    reason: str = ""
    urgency: str = "normal"        # normal | urgent (urgent 는 장 상태와 무관하게 시도)


def hold_winner(price: float, sig, cfg: dict, state: dict,
                direction: int = 1, pinned: bool = False) -> bool:
    """승자 보유(hold_winners) 게이트 — 지금 목표가 익절을 보류할 상황인가.

    오닐 ("승자는 추세가 깨질 때까지 보유"). 종목이 50일선을 +3% 이상 상회하고
    진입 후 고점의 97% 이내에 있으면 추세가 아직 살아 있다고 보고 목표가·목표
    수익률 익절을 미룹니다. 손절·트레일링·시간 청산은 여기와 무관합니다.

    **이 함수가 True 를 돌려주는 동안에는 고점 되돌림을 잡는 장치가 없습니다.**
    돌려주는 값은 check_exit 의 skip_take 로만 들어가므로, 나중에 False 로
    바뀌어도 되살아나는 것은 진입 때 정해진 고정 목표가와 take_profit_pct
    뿐입니다. 그 사이 고점이 목표가를 넘지 못했다면 되돌림 구간에서 아무
    청산도 걸리지 않고 손실 한도·신호 반전·max_hold_days 까지 갑니다.
    고점 대비로 나가게 하려면 trailing_stop_pct 가 함께 켜져 있어야 합니다
    (그 검사는 이 게이트보다 **위**에 있어 보류와 무관하게 작동합니다).

    **check_exit 과 백테스트가 반드시 같은 함수를 봐야 합니다.**
    백테스트 루프(engine/autotrade._run_backtest)의 익절은 봉 안 체결
    (engine/fills.protective_fill)이라 check_exit 을 거치지 않습니다. 판정을
    여기로 모으지 않으면 hold_winners 가 백테스트에서만 조용히 꺼져서, 이
    설정을 켠 효과가 과거 데이터에서 항상 0으로 나옵니다.
    """
    if not cfg.get("hold_winners") or pinned or direction <= 0:
        return False
    if not (sig is not None and getattr(sig, "ok", False) and getattr(sig, "ma50", 0)):
        return False
    if not price or price < sig.ma50 * 1.03:
        return False
    peak = state.get("peak_price")
    return bool(peak and price >= float(peak) * 0.97)


def check_exit(inst: Instrument, position, sig: Signal, cfg: dict,
               state: dict, now: datetime = None,
               protective_only: bool = False,
               defer_take_profit: bool = False) -> ExitDecision:
    """보유 포지션을 정리해야 하는가.

    확인 순서가 중요합니다 — **손실을 막는 조건을 먼저** 봅니다.
    (익절보다 손절이 먼저, 신호 반전보다 만기가 먼저)

    state 는 이 포지션에 대해 엔진이 기억하고 있는 값입니다.
        entry_price, stop_price, target_price, peak_price, opened_at, pinned

    고정 (state["pinned"])
        사용자가 "이 종목은 **목표가나 손절가에서만** 팔아라"고 못박은
        포지션입니다. 가격선이 아닌 사유 — 트레일링 되돌림 · 신호 반전 ·
        보유 시간 초과 — 로는 나가지 않습니다. 회전(갈아타기)의 매도
        후보에서도 빠집니다 (engine/autotrade._maybe_rotate).

        그대로 작동하는 것: 손절가 · 손실 한도 % · 목표가 · 목표 수익률 %.
        승자 보유(hold_winners)의 익절 보류도 고정에는 적용하지 않습니다 —
        "목표가에 판다"가 사용자의 명시적 지시라 추세를 이유로 미루면 안 됩니다.

        예외는 파생 만기 하나입니다 (아래 1). 만기는 우리가 안 팔아도
        거래소가 정산합니다 — 고정으로 막으면 가격을 우리가 못 고릅니다.

    protective_only
        **지키는 조건만** 봅니다 — 만기·손절·손실한도·트레일링·보유시간.
        익절과 신호 반전은 건너뜁니다.

        고속 보호 회전(engine/autotrade.run_guard)이 쓰는 모드입니다. 그 회전은
        지표를 계산하지 않아 신호 점수가 없고, 승자 보유(hold_winners) 판정에
        필요한 50일선도 없습니다. 없는 근거로 익절을 결정하면 정규 회전과 다른
        판단이 나오므로, 아예 보지 않고 다음 정규 회전에 넘깁니다.

        방향은 안전한 쪽입니다 — 익절이 몇십 초 늦는 것은 기회 손실이지만,
        손절이 몇십 초 늦는 것은 손실입니다.

    defer_take_profit
        **익절만** 넘깁니다 — 목표가 도달과 목표 수익률 달성을 건너뜁니다.
        분할 매도(engine/split.py)가 켜진 포지션이 쓰는 모드입니다.

        차수별로 나눠 파는 포지션에서 전량 익절이 먼저 걸리면, 1차부터
        마지막 차수까지 한 번에 정리되어 분할 자체가 무의미해집니다.
        지키는 조건(손절·트레일링·만기·시간·신호 반전)은 그대로 둡니다 —
        이건 분할이든 아니든 전량으로 나가야 하는 상황입니다.
    """
    now = now or datetime.now()
    direction = 1 if position.side == LONG else -1
    price = sig.price if sig.ok and sig.price else (position.current_price or 0)
    if not price:
        return ExitDecision(False, "현재가 없음")

    entry = float(state.get("entry_price") or position.avg_price or price)
    pnl_pct = (price - entry) / entry * 100 * direction if entry else 0.0

    # 사용자가 고정한 포지션인가 (위 docstring 의 '고정' 참고)
    pinned = bool(state.get("pinned"))

    # 1) 만기 (파생) — 만기일에 물려 있으면 강제 청산됩니다.
    # 고정도 이것만은 못 막습니다. 만기는 우리 규칙이 아니라 거래소 규칙이라
    # "안 판다"는 선택지가 없고, 남는 것은 **누가 가격을 고르느냐**뿐입니다.
    if inst.asset_class in (FUTURES, OPTION):
        days = feed.days_to_expiry(inst)
        min_days = int(cfg.get("deriv_min_days_to_expiry", 2))
        if days is not None and days <= min_days:
            return ExitDecision(True, f"만기 {days}일 전 — 롤오버/청산", "urgent")

    # 2) 손절
    stop = state.get("stop_price")
    if stop:
        hit = price <= float(stop) if direction > 0 else price >= float(stop)
        if hit:
            return ExitDecision(True, f"손절 도달 ({price:g} vs {float(stop):g}, {pnl_pct:+.2f}%)",
                                "urgent")

    hard_stop = float(cfg.get("stop_loss_pct", 0) or 0)
    if hard_stop > 0 and pnl_pct <= -hard_stop:
        return ExitDecision(True, f"손실 한도 {hard_stop:g}% 초과 ({pnl_pct:+.2f}%)", "urgent")

    # 3) 트레일링 스톱 — 고점 대비 되돌림
    # 고정된 포지션은 보지 않습니다. 트레일링은 "손절가는 아직 멀지만 여기서
    # 끊자"는 판단이고, 고정은 정확히 그 판단을 하지 말라는 지시입니다.
    trail = float(cfg.get("trailing_stop_pct", 0) or 0)
    peak = state.get("peak_price")
    if trail > 0 and peak and not pinned:
        drop = (float(peak) - price) / float(peak) * 100 * direction
        if drop >= trail:
            return ExitDecision(True, f"고점 대비 {drop:.2f}% 되돌림 (트레일링 {trail:g}%)")

    # 4) 익절 — 승자 보유(hold_winners) 게이트가 먼저 봅니다.
    #
    # prism-insight 의 oneil_fallback (오닐: "승자는 추세가 깨질 때까지 보유")
    # 이식. 원본 실측 사례: 목표가 도달 당일 +11% 급등 중이던 주도주를 자동
    # 익절로 전량 청산 (INCY, 2026-07-29). 종목이 50일선을 +3% 이상 상회하고
    # 진입 후 고점의 97% 이내에 있으면 — 추세가 아직 살아 있으면 — 목표가
    # 익절을 보류합니다. 손절·트레일링·시간 청산은 이 게이트와 무관하게
    # 그대로 작동합니다 (지키는 장치는 절대 끄지 않습니다).
    winner_hold = hold_winner(price, sig, cfg, state, direction, pinned)

    skip_take = winner_hold or protective_only or defer_take_profit

    target = state.get("target_price")
    if target and not skip_take:
        hit = price >= float(target) if direction > 0 else price <= float(target)
        if hit:
            return ExitDecision(True, f"목표가 도달 ({price:g}, {pnl_pct:+.2f}%)")

    take = float(cfg.get("take_profit_pct", 0) or 0)
    if take > 0 and pnl_pct >= take and not skip_take:
        return ExitDecision(True, f"목표 수익 {take:g}% 달성 ({pnl_pct:+.2f}%)")

    # 5) 신호 반전 / 소멸 (고정된 포지션은 신호로 팔지 않습니다)
    if sig.ok and not protective_only and not pinned:
        exit_score = float(cfg.get("exit_score", 0.05))
        if direction > 0 and sig.score <= -abs(exit_score):
            return ExitDecision(True, f"신호 반전 (점수 {sig.score:+.2f})")
        if direction < 0 and sig.score >= abs(exit_score):
            return ExitDecision(True, f"신호 반전 (점수 {sig.score:+.2f})")

    # 6) 보유 기간 초과 — 신호가 죽었는데 계속 들고 있는 것을 막습니다.
    #
    # 여기부터 끝까지는 **시간 청산뿐**입니다. 고정된 포지션은 시간으로 팔지
    # 않으므로 여기서 끝냅니다. 시간이 아닌 새 청산 규칙을 추가할 때는 이
    # 줄 **위**에 두세요 — 아래에 두면 고정 포지션에서 조용히 사라집니다.
    opened = state.get("opened_at")
    if pinned or not opened:
        return ExitDecision(False, "")

    # 초단타는 **초 단위**로 끊습니다. max_hold_minutes 는 분 정수라 90초 같은
    # 값을 표현할 수 없어서, 페니 초단타는 이쪽을 씁니다. 둘 다 설정돼 있으면
    # 짧은 쪽이 이깁니다 — 청산 조건은 항상 보수적인 쪽으로 붙어야 합니다.
    max_sec = int(cfg.get("max_hold_sec", 0) or 0)
    if max_sec > 0 and opened:
        try:
            held_sec = (now - datetime.fromisoformat(str(opened))).total_seconds()
            if held_sec >= max_sec:
                return ExitDecision(True, f"보유 {held_sec:.0f}초 (최대 {max_sec}초) — 시간 청산",
                                    "urgent")
        except ValueError:
            pass

    max_minutes = int(cfg.get("max_hold_minutes", 0) or 0)
    if max_minutes > 0 and opened:
        try:
            held_min = (now - datetime.fromisoformat(str(opened))).total_seconds() / 60
            if held_min >= max_minutes:
                return ExitDecision(True, f"보유 {held_min:.0f}분 (최대 {max_minutes}분) — 시간 청산",
                                    "urgent")
        except ValueError:
            pass

    max_days = int(cfg.get("max_hold_days", 0) or 0)
    if max_days > 0 and opened:
        try:
            held = (now - datetime.fromisoformat(str(opened))).days
            if held >= max_days:
                return ExitDecision(True, f"보유 {held}일 (최대 {max_days}일) — 시간 청산")
        except ValueError:
            pass

    return ExitDecision(False, "")


def update_peak(state: dict, price: float, side: str) -> dict:
    """트레일링 스톱용 최고점(숏이면 최저점) 갱신."""
    peak = state.get("peak_price")
    if peak is None:
        state["peak_price"] = price
    elif side == LONG:
        state["peak_price"] = max(float(peak), price)
    else:
        state["peak_price"] = min(float(peak), price)
    return state
