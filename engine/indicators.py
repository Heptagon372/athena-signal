"""
기술적 지표 엔진 (Technical Indicators)
--------------------------------------
지표를 "블랙박스 숫자 하나"로 뭉개지 않고, **지표별로 실제 계산값 + 판단 근거 문장**을
같이 돌려주는 것이 이 모듈의 목적입니다.

각 지표 함수는 IndicatorResult 를 반환합니다:
    score      -1(강한 하락) ~ +1(강한 상승)
    value_text 실제 계산된 수치 (예: "RSI 28.4")
    reason     그 수치가 왜 그 판단으로 이어지는지 한 문장
    formula    점수를 만든 식 (검증/디버깅용)

종합 기술점수 = Σ(가중치 × 지표점수) / Σ(계산 성공한 지표의 가중치)
가중치는 config.INDICATOR_WEIGHTS 에서 관리합니다.

해석 규약 (섞이지 않도록 명시)
    · 추세추종형 (MACD, 이동평균, ROC, OBV, 추세위치)
        -> 상승 추세면 +, 하락 추세면 -
    · 되돌림형 (RSI, 스토캐스틱, 볼린저)
        -> 과매도면 +(반등 기대), 과매수면 -(조정 기대), 중립구간은 0
    · 확인형 (거래량)
        -> 방향 자체는 주가 변화가 정하고, 거래량은 그 신뢰도를 키움
    · 참고용 (ATR)
        -> 방향성이 없으므로 가중치 0, 화면에는 변동성 정보로만 표시

주의: 기술적 지표는 과거 가격의 통계적 요약일 뿐 미래를 보장하지 않습니다.
"""

import numpy as np
import pandas as pd

from config import INDICATOR_WEIGHTS, MIN_BARS_FOR_TECHNICAL
from engine import quant
from models import IndicatorResult, TechnicalAnalysis

# 지표를 성격별로 묶습니다. 국면(regime)에 따라 어느 군을 더 믿을지 달라집니다.
#   trend    추세추종형 — 추세 국면에서 유효
#   meanrev  평균회귀형 — 되돌림 국면에서 유효
#   confirm  확인형(수급) — 방향보다 신뢰도를 보강
#   info     참고용 — 방향성이 없어 점수에 반영하지 않음
INDICATOR_FAMILY = {
    "macd": "trend", "ma_cross": "trend", "ma_trend": "trend", "roc": "trend",
    "obv": "trend", "price_position": "trend", "adx": "trend",
    "aroon": "trend", "trend_reg": "trend",
    "rsi": "meanrev", "bollinger": "meanrev", "stochastic": "meanrev",
    "cci": "meanrev", "williams": "meanrev",
    "volume": "confirm", "mfi": "confirm", "cmf": "confirm",
    "atr": "info", "yz_vol": "info",
}


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _verdict(score: float, threshold: float = 0.1) -> str:
    if score > threshold:
        return "bullish"
    if score < -threshold:
        return "bearish"
    return "neutral"


def _mk(key: str, label: str, value_text: str, score: float,
        reason: str, formula: str = "", values: dict | None = None) -> IndicatorResult:
    score = round(_clip(score), 4)
    return IndicatorResult(
        key=key, label=label, value_text=value_text, score=score,
        weight=INDICATOR_WEIGHTS.get(key, 0.0), verdict=_verdict(score),
        reason=reason, formula=formula, values=values or {},
    )


# ---------------------------------------------------------------------------
# 개별 지표
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> IndicatorResult | None:
    """RSI(14) — Wilder 지수이동평균 방식.

    되돌림형 해석: 35 이하 과매도(+), 65 이상 과매수(-), 그 사이는 중립(0).
    극단 구간에서만 신호를 내므로 추세 구간에서 지표끼리 충돌하지 않습니다.
    """
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    series = 100 - (100 / (1 + rs))
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    value = float(value)

    if value <= 35:
        score = (35 - value) / 35
        reason = (f"RSI가 {value:.1f}로 과매도 기준선(35) 아래입니다. "
                  f"매도세가 과열된 구간이라 기술적 반등 가능성이 있습니다.")
    elif value >= 65:
        score = -(value - 65) / 35
        reason = (f"RSI가 {value:.1f}로 과매수 기준선(65) 위입니다. "
                  f"단기 과열 구간이라 차익실현 압력이 나올 수 있습니다.")
    else:
        score = 0.0
        reason = (f"RSI가 {value:.1f}로 중립 구간(35~65)입니다. "
                  f"과매수·과매도 어느 쪽도 아니어서 방향 신호를 내지 않습니다.")

    return _mk("rsi", "RSI(14)", f"{value:.1f}", score, reason,
               formula="≤35: (35-RSI)/35 / ≥65: -(RSI-65)/35 / 그 외 0",
               values={"rsi": round(value, 2)})


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorResult | None:
    """MACD(12,26,9) — 히스토그램을 주가 대비 비율로 정규화해 점수화.

    추세추종형: 히스토그램(MACD-시그널)이 양수면 상승 모멘텀.
    종목별 가격대가 달라도 비교 가능하도록 종가의 1%를 만점 기준으로 삼습니다.
    """
    if len(close) < slow + signal:
        return None
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    last_price = float(close.iloc[-1])
    h = float(hist.iloc[-1])
    if last_price <= 0:
        return None

    scale = last_price * 0.01           # 종가의 1% 를 만점 기준
    score = _clip(h / scale)

    # 최근 3봉 안에서 시그널선 교차가 있었는지
    cross_note = ""
    recent = hist.iloc[-4:]
    if len(recent) >= 2:
        signs = np.sign(recent.values)
        if len(signs) >= 2 and signs[-1] != 0:
            for i in range(len(signs) - 1, 0, -1):
                if signs[i] != signs[i - 1] and signs[i - 1] != 0:
                    bars_ago = len(signs) - 1 - i
                    kind = "골든크로스(상향 돌파)" if signs[i] > 0 else "데드크로스(하향 이탈)"
                    cross_note = f" 최근 {bars_ago}거래일 전 MACD {kind}가 발생했습니다."
                    score = _clip(score * 1.3)   # 갓 발생한 교차는 신호를 강화
                    break

    direction = "상승" if h > 0 else "하락"
    reason = (f"MACD 히스토그램이 {h:+,.1f}(종가의 {h / last_price * 100:+.2f}%)로 "
              f"{direction} 모멘텀 쪽입니다. MACD선이 시그널선 "
              f"{'위' if h > 0 else '아래'}에 있습니다.{cross_note}")

    return _mk("macd", "MACD(12,26,9)", f"{h:+,.1f}", score, reason,
               formula="히스토그램/(종가×1%), 최근 교차 시 ×1.3",
               values={"macd": round(float(macd_line.iloc[-1]), 3),
                       "signal": round(float(signal_line.iloc[-1]), 3),
                       "histogram": round(h, 3)})


def ma_cross(close: pd.Series, short: int = 5, long: int = 20) -> IndicatorResult | None:
    """단기 이동평균 이격도 (5일선 vs 20일선).

    추세추종형: 이격률 ±5% 를 만점으로 정규화.
    """
    if len(close) < long:
        return None
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long).mean()
    s, l = float(ma_s.iloc[-1]), float(ma_l.iloc[-1])
    if pd.isna(s) or pd.isna(l) or l == 0:
        return None

    gap_pct = (s - l) / l * 100
    score = _clip(gap_pct / 5.0)        # ±5% 이격 = 만점

    cross_note = ""
    diff = (ma_s - ma_l).dropna()
    if len(diff) >= 2:
        signs = np.sign(diff.iloc[-6:].values)
        for i in range(len(signs) - 1, 0, -1):
            if signs[i] != signs[i - 1] and signs[i - 1] != 0 and signs[i] != 0:
                bars_ago = len(signs) - 1 - i
                kind = "골든크로스" if signs[i] > 0 else "데드크로스"
                cross_note = f" {bars_ago}거래일 전 {short}일선이 {long}일선을 {'상향 돌파(골든크로스)' if signs[i] > 0 else '하향 이탈(데드크로스)'}했습니다."
                break

    reason = (f"{short}일선({s:,.0f})이 {long}일선({l:,.0f}) 대비 {gap_pct:+.2f}% "
              f"{'위' if gap_pct > 0 else '아래'}에 있습니다. "
              f"단기 추세가 {'우상향' if gap_pct > 0 else '우하향'}입니다.{cross_note}")

    return _mk("ma_cross", f"이동평균 {short}/{long}", f"{gap_pct:+.2f}%", score, reason,
               formula=f"({short}일선-{long}일선)/{long}일선 ÷ 5%",
               values={f"ma{short}": round(s, 2), f"ma{long}": round(l, 2),
                       "gap_pct": round(gap_pct, 3)})


def ma_trend(close: pd.Series, mid: int = 20, long: int = 60) -> IndicatorResult | None:
    """중기 추세 정배열/역배열 (20일선 vs 60일선). ±10% 이격을 만점으로 정규화."""
    if len(close) < long:
        return None
    ma_m = close.rolling(mid).mean().iloc[-1]
    ma_l = close.rolling(long).mean().iloc[-1]
    if pd.isna(ma_m) or pd.isna(ma_l) or ma_l == 0:
        return None
    ma_m, ma_l = float(ma_m), float(ma_l)

    gap_pct = (ma_m - ma_l) / ma_l * 100
    score = _clip(gap_pct / 10.0)
    arrangement = "정배열" if gap_pct > 0 else "역배열"
    reason = (f"{mid}일선({ma_m:,.0f})이 {long}일선({ma_l:,.0f}) 대비 {gap_pct:+.2f}%로 "
              f"중기 {arrangement} 상태입니다. "
              f"{'중기 상승 추세가 유지되고 있습니다.' if gap_pct > 0 else '중기 하락 추세가 이어지고 있습니다.'}")

    return _mk("ma_trend", f"중기추세 {mid}/{long}", f"{gap_pct:+.2f}%", score, reason,
               formula=f"({mid}일선-{long}일선)/{long}일선 ÷ 10%",
               values={f"ma{mid}": round(ma_m, 2), f"ma{long}": round(ma_l, 2)})


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> IndicatorResult | None:
    """볼린저밴드(20, 2σ) %B — 되돌림형.

    %B = (종가-하단)/(상단-하단). 0.5 중앙 기준으로 밴드 상단에 붙을수록 과열(-).
    밴드폭(스퀴즈) 정보도 함께 제공합니다.
    """
    if len(close) < period:
        return None
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper, lower = ma + num_std * std, ma - num_std * std
    c = float(close.iloc[-1])
    u, l, m = float(upper.iloc[-1]), float(lower.iloc[-1]), float(ma.iloc[-1])
    if pd.isna(u) or pd.isna(l) or u == l:
        return None

    pct_b = (c - l) / (u - l)
    score = _clip((0.5 - pct_b) * 2)

    band_width = (u - l) / m * 100 if m else 0
    width_series = ((upper - lower) / ma * 100).dropna()
    squeeze = ""
    if len(width_series) >= period:
        pctile = float((width_series.iloc[-period:] < band_width).mean())
        if pctile < 0.2:
            squeeze = " 밴드폭이 최근 20일 중 하위 20%로 좁아진 스퀴즈 상태라, 곧 변동성 확대가 나올 수 있습니다."

    if pct_b > 1:
        position = "상단 밴드를 이탈(과열)"
    elif pct_b > 0.8:
        position = "상단 밴드 근처(과열권)"
    elif pct_b > 0.6:
        position = "중심선 위(상단권)"
    elif pct_b < 0:
        position = "하단 밴드를 이탈(과매도)"
    elif pct_b < 0.2:
        position = "하단 밴드 근처(과매도권)"
    elif pct_b < 0.4:
        position = "중심선 아래(하단권)"
    else:
        position = "밴드 중앙 부근"

    reason = (f"%B가 {pct_b:.2f}로 {position}입니다 "
              f"(하단 {l:,.0f} / 중심 {m:,.0f} / 상단 {u:,.0f}, 밴드폭 {band_width:.1f}%)."
              f"{squeeze}")

    return _mk("bollinger", "볼린저밴드 %B(20,2σ)", f"{pct_b:.2f}", score, reason,
               formula="(0.5 - %B) × 2",
               values={"pct_b": round(pct_b, 3), "upper": round(u, 2),
                       "lower": round(l, 2), "band_width_pct": round(band_width, 2)})


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorResult | None:
    """스토캐스틱 슬로우 %K/%D(14,3,3) — 되돌림형.

    20 이하 과매도(+), 80 이상 과매수(-), 그 사이 중립(0).
    """
    if len(df) < k_period + d_period * 2:
        return None
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    fast_k = (df["close"] - low_min) / denom * 100
    slow_k = fast_k.rolling(d_period).mean()
    slow_d = slow_k.rolling(d_period).mean()

    k, d = slow_k.iloc[-1], slow_d.iloc[-1]
    if pd.isna(k) or pd.isna(d):
        return None
    k, d = float(k), float(d)

    if k <= 20:
        score = (20 - k) / 20
        zone = "과매도 구간(20 이하)이라 반등 시도가 나올 수 있습니다"
    elif k >= 80:
        score = -(k - 80) / 20
        zone = "과매수 구간(80 이상)이라 단기 조정 압력이 있습니다"
    else:
        score = 0.0
        zone = "중립 구간(20~80)이라 과열·침체 신호는 없습니다"

    cross = ""
    if len(slow_k) >= 2 and not pd.isna(slow_k.iloc[-2]) and not pd.isna(slow_d.iloc[-2]):
        prev_diff = slow_k.iloc[-2] - slow_d.iloc[-2]
        curr_diff = k - d
        if prev_diff < 0 <= curr_diff:
            cross = " 다만 %K가 %D를 상향 돌파해 단기 매수 신호가 나왔습니다."
            score = _clip(score + 0.25)
        elif prev_diff > 0 >= curr_diff:
            cross = " 다만 %K가 %D를 하향 이탈해 단기 매도 신호가 나왔습니다."
            score = _clip(score - 0.25)

    if not cross and score == 0.0:
        cross = " %K·%D 교차도 없어 방향 신호를 내지 않습니다."

    reason = f"스토캐스틱 %K {k:.1f} / %D {d:.1f}. {zone}.{cross}"

    return _mk("stochastic", "스토캐스틱(14,3,3)", f"%K {k:.1f}", score, reason,
               formula="≤20: (20-%K)/20 / ≥80: -(%K-80)/20, %D 교차 시 ±0.25",
               values={"k": round(k, 2), "d": round(d, 2)})


def volume_signal(df: pd.DataFrame, period: int = 20) -> IndicatorResult | None:
    """거래량 확인 지표 — 확인형.

    거래량 자체는 방향이 없으므로 당일 주가 변화 방향에 신뢰도를 곱합니다.
    (평균 대비 3배 거래량 = 만점)
    """
    if len(df) < period + 1 or df["volume"].iloc[-period:].sum() == 0:
        return None
    today_vol = float(df["volume"].iloc[-1])
    avg_vol = float(df["volume"].iloc[-period - 1:-1].mean())
    if avg_vol <= 0:
        return None

    ratio = today_vol / avg_vol
    change_pct = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-2]) - 1) * 100
    strength = _clip((ratio - 1) / 2, 0, 1)      # 평균의 3배면 1.0
    score = _clip(np.sign(change_pct) * strength)

    if ratio >= 1.5:
        vol_desc = f"평균 대비 {ratio:.1f}배로 거래가 크게 늘었습니다"
    elif ratio <= 0.6:
        vol_desc = f"평균 대비 {ratio:.1f}배로 거래가 말라 있습니다"
    else:
        vol_desc = f"평균 대비 {ratio:.1f}배로 평소 수준입니다"

    reason = (f"최근 거래량 {today_vol:,.0f}주 — {period}일 평균({avg_vol:,.0f}주) {vol_desc}. "
              f"같은 날 주가가 {change_pct:+.2f}% 움직였으므로 "
              f"{'상승' if change_pct > 0 else '하락'}에 거래량이 실린 정도로 반영했습니다.")

    return _mk("volume", f"거래량({period}일 평균 대비)", f"{ratio:.2f}배", score, reason,
               formula="sign(당일 등락) × clip((거래량비율-1)/2, 0, 1)",
               values={"ratio": round(ratio, 3), "today": today_vol,
                       "avg": round(avg_vol, 1), "change_pct": round(change_pct, 3)})


def obv(df: pd.DataFrame, period: int = 20) -> IndicatorResult | None:
    """OBV(On-Balance Volume) 추세 — 추세추종형(수급).

    최근 period일 OBV 기울기를 같은 기간 평균 거래량으로 정규화합니다.
    """
    if len(df) < period + 1:
        return None
    direction = np.sign(df["close"].diff().fillna(0))
    obv_series = (direction * df["volume"]).cumsum()

    window = obv_series.iloc[-period:]
    if len(window) < period:
        return None
    x = np.arange(len(window))
    slope = float(np.polyfit(x, window.values, 1)[0])
    avg_vol = float(df["volume"].iloc[-period:].mean())
    if avg_vol <= 0:
        return None

    normalized = slope / avg_vol           # 하루당 평균거래량의 몇 배씩 누적되는가
    score = _clip(normalized / 0.5)        # 0.5배/일 이면 만점

    reason = (f"OBV(매집·분산 지표)가 최근 {period}거래일 동안 하루 평균 거래량의 "
              f"{normalized:+.2f}배씩 {'쌓이고' if normalized > 0 else '빠지고'} 있습니다. "
              f"{'상승일 거래가 하락일보다 우세해 매집 흐름입니다.' if normalized > 0 else '하락일 거래가 우세해 분산(매도) 흐름입니다.'}")

    return _mk("obv", f"OBV 추세({period}일)", f"{normalized:+.2f}", score, reason,
               formula="OBV 선형회귀 기울기 ÷ 평균거래량 ÷ 0.5",
               values={"slope_per_avg_vol": round(normalized, 4)})


def roc(close: pd.Series, period: int = 10) -> IndicatorResult | None:
    """ROC(10) 모멘텀 — 추세추종형. 10% 변동을 만점으로 정규화."""
    if len(close) < period + 1:
        return None
    now, past = float(close.iloc[-1]), float(close.iloc[-period - 1])
    if past == 0:
        return None
    change = (now / past - 1) * 100
    score = _clip(change / 10.0)
    reason = (f"최근 {period}거래일 동안 주가가 {past:,.0f} → {now:,.0f} 으로 "
              f"{change:+.2f}% 움직였습니다. "
              f"{'상승 모멘텀이 살아 있습니다.' if change > 0 else '하락 모멘텀이 이어지고 있습니다.'}")
    return _mk("roc", f"모멘텀 ROC({period})", f"{change:+.2f}%", score, reason,
               formula=f"{period}일 등락률 ÷ 10%",
               values={"roc_pct": round(change, 3)})


def price_position(df: pd.DataFrame, period: int = 120) -> IndicatorResult | None:
    """기간 내 고저 대비 위치 — 추세추종형(강도).

    신고가 근처면 추세 강도가 강하다고 보고 +, 신저가 근처면 -.
    """
    window = df.iloc[-period:]
    if len(window) < 20:
        return None
    high = float(window["high"].max())
    low = float(window["low"].min())
    close = float(df["close"].iloc[-1])
    if high == low:
        return None

    pos = (close - low) / (high - low)
    score = _clip((pos - 0.5) * 2 * 0.8)   # 최대 ±0.8 (단독으로 과하지 않게)
    bars = len(window)

    if pos >= 0.9:
        desc = "기간 신고가 부근으로 추세가 매우 강합니다"
    elif pos >= 0.6:
        desc = "기간 상단부에 위치해 추세가 우호적입니다"
    elif pos <= 0.1:
        desc = "기간 신저가 부근으로 하락 추세가 강합니다"
    elif pos <= 0.4:
        desc = "기간 하단부에 위치해 추세가 약합니다"
    else:
        desc = "기간 중간대에 위치해 방향성이 뚜렷하지 않습니다"

    reason = (f"최근 {bars}거래일 최저 {low:,.0f} ~ 최고 {high:,.0f} 구간에서 "
              f"현재가 {close:,.0f}는 {pos * 100:.0f}% 지점입니다. {desc}.")

    return _mk("price_position", f"추세위치({period}일)", f"{pos * 100:.0f}%", score, reason,
               formula="((위치비율-0.5)×2)×0.8",
               values={"position_pct": round(pos * 100, 1), "high": high, "low": low})


def adx(df: pd.DataFrame, period: int = 14) -> IndicatorResult | None:
    """ADX / DMI — Wilder (1978) 추세 강도.

    +DI 와 -DI 의 우열이 방향을, ADX 가 그 추세의 '강도'를 나타냅니다.
    ADX < 20 이면 추세가 없다고 보고 방향 점수를 크게 줄입니다.
    """
    if len(df) < period * 3:
        return None

    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)

    alpha = 1 / period
    atr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_s

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()

    adx_v, p_di, m_di = adx_series.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
    if pd.isna(adx_v) or pd.isna(p_di) or pd.isna(m_di):
        return None
    adx_v, p_di, m_di = float(adx_v), float(p_di), float(m_di)

    # 방향은 DI 격차, 강도는 ADX가 결정
    direction = (p_di - m_di) / max(p_di + m_di, 1e-9)
    strength = _clip((adx_v - 20) / 30, 0, 1)
    score = _clip(direction * strength * 1.5)

    if adx_v >= 40:
        desc = "매우 강한 추세"
    elif adx_v >= 25:
        desc = "추세 형성 중"
    elif adx_v >= 20:
        desc = "추세 초입 또는 약한 추세"
    else:
        desc = "추세 없음(횡보)"

    reason = (f"ADX가 {adx_v:.1f}로 {desc}입니다. "
              f"+DI {p_di:.1f} vs -DI {m_di:.1f}로 "
              f"{'상승' if direction > 0 else '하락'} 쪽이 우세합니다."
              + ("" if adx_v >= 20 else " 다만 ADX 20 미만이라 방향 신호를 크게 줄여 반영했습니다."))

    return _mk("adx", "ADX/DMI(14)", f"{adx_v:.1f}", score, reason,
               formula="((+DI - -DI)/(+DI + -DI)) × clip((ADX-20)/30,0,1) × 1.5",
               values={"adx": round(adx_v, 2), "plus_di": round(p_di, 2),
                       "minus_di": round(m_di, 2)})


def aroon(df: pd.DataFrame, period: int = 25) -> IndicatorResult | None:
    """Aroon 오실레이터 — 최근 고점/저점이 얼마나 최근인지로 추세를 봅니다."""
    if len(df) < period + 1:
        return None
    window_high = df["high"].iloc[-(period + 1):]
    window_low = df["low"].iloc[-(period + 1):]

    # 배열을 뒤집어 argmax를 구하면 "몇 봉 전에 고점이 나왔는지"가 바로 나옵니다
    bars_since_high = int(np.argmax(window_high.values[::-1]))
    bars_since_low = int(np.argmin(window_low.values[::-1]))
    aroon_up = (period - bars_since_high) / period * 100
    aroon_down = (period - bars_since_low) / period * 100
    osc = aroon_up - aroon_down

    score = _clip(osc / 100)
    reason = (f"Aroon Up {aroon_up:.0f} / Down {aroon_down:.0f} (오실레이터 {osc:+.0f}). "
              f"최근 {period}봉 중 신고가는 {bars_since_high}봉 전, "
              f"신저가는 {bars_since_low}봉 전입니다. "
              f"{'고점 갱신이 더 최근이라 상승 추세' if osc > 0 else '저점 갱신이 더 최근이라 하락 추세'}입니다.")

    return _mk("aroon", f"Aroon({period})", f"{osc:+.0f}", score, reason,
               formula="(AroonUp - AroonDown)/100",
               values={"up": round(aroon_up, 1), "down": round(aroon_down, 1)})


def cci(df: pd.DataFrame, period: int = 20) -> IndicatorResult | None:
    """CCI (Lambert) — 전형가격이 평균에서 몇 배 벗어났는지. ±100 밖이 과열/침체."""
    if len(df) < period:
        return None
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_s = (tp - sma) / (0.015 * mad.replace(0, np.nan))

    v = cci_s.iloc[-1]
    if pd.isna(v):
        return None
    v = float(v)

    # 되돌림형: ±100 밖에서만 신호
    if v >= 100:
        score = -_clip((v - 100) / 200)
        zone = "과열 구간(+100 초과)이라 되돌림 압력이 있습니다"
    elif v <= -100:
        score = _clip((-100 - v) / 200)
        zone = "침체 구간(-100 미만)이라 반등 여지가 있습니다"
    else:
        score = 0.0
        zone = "정상 범위(±100 이내)라 방향 신호를 내지 않습니다"

    reason = f"CCI가 {v:+.1f}로 {zone}."
    return _mk("cci", f"CCI({period})", f"{v:+.0f}", score, reason,
               formula="≥+100: -(CCI-100)/200 / ≤-100: (-100-CCI)/200 / 그 외 0",
               values={"cci": round(v, 2)})


def williams_r(df: pd.DataFrame, period: int = 14) -> IndicatorResult | None:
    """Williams %R — 스토캐스틱의 역스케일(0 ~ -100). -20 위 과매수, -80 아래 과매도."""
    if len(df) < period:
        return None
    hh = df["high"].rolling(period).max().iloc[-1]
    ll = df["low"].rolling(period).min().iloc[-1]
    c = float(df["close"].iloc[-1])
    if pd.isna(hh) or pd.isna(ll) or hh == ll:
        return None

    wr = (float(hh) - c) / (float(hh) - float(ll)) * -100

    if wr >= -20:
        score = -_clip((wr + 20) / 20)
        zone = "과매수 구간(-20 위)"
    elif wr <= -80:
        score = _clip((-80 - wr) / 20)
        zone = "과매도 구간(-80 아래)"
    else:
        score = 0.0
        zone = "중립 구간"

    reason = (f"Williams %R이 {wr:.1f}로 {zone}입니다 "
              f"(최근 {period}봉 고가 {float(hh):,.0f} / 저가 {float(ll):,.0f} 대비 위치).")
    return _mk("williams", f"Williams %R({period})", f"{wr:.1f}", score, reason,
               formula="≥-20: -(WR+20)/20 / ≤-80: (-80-WR)/20 / 그 외 0",
               values={"williams_r": round(wr, 2)})


def mfi(df: pd.DataFrame, period: int = 14) -> IndicatorResult | None:
    """MFI (Money Flow Index) — 거래량을 반영한 RSI. 자금 유입/유출 강도."""
    if len(df) < period + 1 or df["volume"].iloc[-period:].sum() == 0:
        return None
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_flow = tp * df["volume"]
    direction = np.sign(tp.diff().fillna(0))

    pos = raw_flow.where(direction > 0, 0.0).rolling(period).sum()
    neg = raw_flow.where(direction < 0, 0.0).rolling(period).sum()
    ratio = pos / neg.replace(0, np.nan)
    mfi_s = 100 - (100 / (1 + ratio))

    v = mfi_s.iloc[-1]
    if pd.isna(v):
        return None
    v = float(v)

    if v <= 20:
        score = (20 - v) / 20
        zone = "자금 유출이 과도해 반등 여지가 있습니다"
    elif v >= 80:
        score = -(v - 80) / 20
        zone = "자금 유입이 과열돼 조정 압력이 있습니다"
    else:
        score = 0.0
        zone = "자금 흐름이 중립 범위입니다"

    reason = f"MFI가 {v:.1f}입니다. 거래대금 기준 {zone}."
    return _mk("mfi", f"MFI({period})", f"{v:.1f}", score, reason,
               formula="≤20: (20-MFI)/20 / ≥80: -(MFI-80)/20 / 그 외 0",
               values={"mfi": round(v, 2)})


def cmf(df: pd.DataFrame, period: int = 20) -> IndicatorResult | None:
    """Chaikin Money Flow — 봉 내 종가 위치와 거래량으로 매집/분산을 측정."""
    if len(df) < period or df["volume"].iloc[-period:].sum() == 0:
        return None
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    mf_mult = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mf_vol = mf_mult * df["volume"]

    v = mf_vol.rolling(period).sum().iloc[-1] / df["volume"].rolling(period).sum().iloc[-1]
    if pd.isna(v):
        return None
    v = float(v)

    score = _clip(v * 4)      # ±0.25 를 만점으로
    reason = (f"CMF가 {v:+.3f}입니다. 최근 {period}봉 동안 종가가 봉의 "
              f"{'상단' if v > 0 else '하단'}에서 형성되며 거래가 실려 "
              f"{'매집(자금 유입)' if v > 0 else '분산(자금 유출)'} 흐름입니다.")
    return _mk("cmf", f"CMF({period})", f"{v:+.3f}", score, reason,
               formula="CMF × 4 (±0.25를 만점으로)",
               values={"cmf": round(v, 4)})


def trend_regression_indicator(df: pd.DataFrame, window: int = 20) -> IndicatorResult | None:
    """최소자승 추세선 — 기울기가 방향을, R²가 그 추세의 신뢰도를 담당합니다.

    같은 기울기라도 R²가 낮으면(등락이 심하면) 점수를 줄입니다.
    """
    res = quant.trend_regression(df["close"].values, window=window)
    if res is None:
        return None

    ret = res["period_return_pct"]
    r2 = res["r_squared"]
    score = _clip(ret / 10.0) * r2     # ±10% 를 만점으로 하고 R²로 감쇠

    quality = "매우 뚜렷" if r2 >= 0.7 else "뚜렷한 편" if r2 >= 0.4 else "불분명"
    reason = (f"최근 {window}봉 회귀 추세가 {ret:+.2f}% 기울기이고 결정계수 R²={r2:.2f}로 "
              f"추세의 일관성이 {quality}합니다. "
              f"{'R²가 낮아 방향 점수를 크게 줄였습니다.' if r2 < 0.4 else ''}")

    return _mk("trend_reg", f"회귀추세({window})", f"{ret:+.1f}% R²{r2:.2f}", score, reason,
               formula="clip(기간수익률/10%) × R²",
               values=res)


def yang_zhang_indicator(df: pd.DataFrame, window: int = 20) -> IndicatorResult | None:
    """Yang-Zhang 연율 변동성 — 방향성이 없어 참고용(가중치 0)."""
    yz = quant.yang_zhang_vol(df, window=window)
    if yz is None:
        return None
    rs = quant.rogers_satchell_vol(df, window=window)

    if yz >= 80:
        level = "매우 높음 — 포지션 크기를 줄여야 하는 구간입니다"
    elif yz >= 45:
        level = "높은 편입니다"
    elif yz <= 20:
        level = "낮습니다(저변동 국면)"
    else:
        level = "보통입니다"

    reason = (f"Yang-Zhang 연율 변동성이 {yz:.1f}%로 {level}. "
              f"(종가만 쓰는 방식과 달리 시가·고가·저가와 밤사이 갭까지 반영한 추정량입니다"
              + (f", Rogers-Satchell 기준 {rs:.1f}%)" if rs else ")")
              + " 방향성 지표가 아니라 확률 계산에는 반영하지 않습니다.")

    return _mk("yz_vol", "변동성(Yang-Zhang)", f"{yz:.1f}%", 0.0, reason,
               formula="가중치 0 (방향성 없음)",
               values={"yang_zhang": round(yz, 2),
                       "rogers_satchell": round(rs, 2) if rs else None})


def atr(df: pd.DataFrame, period: int = 14) -> IndicatorResult | None:
    """ATR(14) 변동성 — 방향성이 없어 가중치 0. 리스크 참고용으로만 표시."""
    if len(df) < period + 1:
        return None
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_value = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if pd.isna(atr_value):
        return None

    close = float(df["close"].iloc[-1])
    atr_value = float(atr_value)
    atr_pct = atr_value / close * 100 if close else 0

    if atr_pct >= 5:
        level = "매우 높음 — 손절폭을 넓게 잡아야 하는 구간입니다"
    elif atr_pct >= 3:
        level = "높은 편 — 일중 변동이 큽니다"
    elif atr_pct <= 1.2:
        level = "낮음 — 횡보/저변동 구간입니다"
    else:
        level = "보통 수준입니다"

    reason = (f"ATR({period})이 {atr_value:,.0f}원(주가의 {atr_pct:.2f}%)으로 변동성이 {level}. "
              f"방향성 지표가 아니라 확률 계산에는 반영하지 않고 리스크 참고용으로만 표시합니다.")

    return _mk("atr", "ATR(14) 변동성", f"{atr_pct:.2f}%", 0.0, reason,
               formula="가중치 0 (방향성 없음)",
               values={"atr": round(atr_value, 2), "atr_pct": round(atr_pct, 3)})


# ---------------------------------------------------------------------------
# 종합
# ---------------------------------------------------------------------------

def build_chart_series(df: pd.DataFrame, days: int = 120, timeframe: str = "day") -> dict:
    """차트용 시계열 묶음.

    analyze() 가 '마지막 한 점의 판단'을 만든다면, 이 함수는 그 판단이 나온
    과정을 눈으로 볼 수 있도록 **지표를 시계열 전체로** 펼쳐 돌려줍니다.
    화면에 그려지는 값과 점수 계산에 쓰인 값이 같은 공식에서 나오도록
    위의 개별 지표 함수들과 동일한 파라미터(EMA 방식 RSI, 12/26/9 MACD 등)를 씁니다.

    JSON 직렬화를 위해 NaN 은 None 으로 바꿉니다 (프론트에서 선을 끊어 그림).
    """
    if df is None or df.empty:
        return {"dates": [], "empty": True,
                "note": "일봉 시세를 받아오지 못해 차트를 그릴 수 없습니다."}

    df = df.dropna(subset=["close"]).copy()
    close = df["close"]

    def series(s: pd.Series) -> list:
        return [None if pd.isna(v) else round(float(v), 4) for v in s]

    # 이동평균
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # 볼린저밴드 (20, 2σ)
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20

    # RSI(14) — rsi() 와 동일한 Wilder EMA 방식
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_series = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD(12,26,9) — macd() 와 동일
    macd_line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # --- 선택 보조지표 (화면에서 켜고 끌 수 있는 것들) ---
    # 위의 개별 지표 함수와 같은 파라미터를 씁니다. 화면에 그려지는 선과
    # 점수 계산에 쓰인 값이 다르면 근거를 대조할 수 없기 때문입니다.
    high, low = df["high"], df["low"]
    volume = df["volume"]

    # CCI(20) — cci() 와 동일 (평균편차 기준)
    typical = (high + low + close) / 3
    tp_ma = typical.rolling(20).mean()
    mean_dev = typical.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci_series = (typical - tp_ma) / (0.015 * mean_dev.replace(0, np.nan))

    # 스토캐스틱 슬로우(14,3,3) — stochastic() 과 동일하게 fast %K 를 한 번 더
    # 평활한 slow %K 를 씁니다 (평활을 빼먹으면 화면 값과 지표표 값이 어긋납니다)
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    fast_k = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    stoch_k = fast_k.rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()

    # ATR(14) — Wilder 방식
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                            (low - prev_close).abs()], axis=1).max(axis=1)
    atr_series = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    # OBV — 종가가 오른 날 거래량 +, 내린 날 -
    obv_series = (np.sign(close.diff().fillna(0)) * volume).cumsum()

    # MFI(14) — 거래대금 가중 RSI
    money_flow = typical * volume
    up_flow = money_flow.where(typical.diff() > 0, 0.0).rolling(14).sum()
    down_flow = money_flow.where(typical.diff() < 0, 0.0).rolling(14).sum()
    mfi_series = 100 - (100 / (1 + up_flow / down_flow.replace(0, np.nan)))

    # 요청 구간만 잘라서 반환 (지표는 위에서 전체 구간으로 계산했으므로 앞부분 워밍업이 살아 있음)
    tail = slice(-days, None)
    idx = df.index[tail]

    # 주기에 맞는 축 라벨 — 분/초봉은 시각까지, 월/년봉은 굵직하게
    stamp_fmt = {
        "second": "%m-%d %H:%M:%S",
        "minute": "%m-%d %H:%M",
        "day": "%Y-%m-%d",
        "week": "%Y-%m-%d",
        "month": "%Y-%m",
        "year": "%Y",
    }.get(timeframe, "%Y-%m-%d")

    return {
        "empty": False,
        "timeframe": timeframe,
        "dates": [d.strftime(stamp_fmt) for d in idx],
        "timestamps": [d.isoformat() for d in idx],
        "open": series(df["open"].iloc[tail]),
        "high": series(df["high"].iloc[tail]),
        "low": series(df["low"].iloc[tail]),
        "close": series(close.iloc[tail]),
        "volume": series(df["volume"].iloc[tail]),
        "ma5": series(ma5.iloc[tail]),
        "ma20": series(ma20.iloc[tail]),
        "ma60": series(ma60.iloc[tail]),
        "bb_upper": series(bb_upper.iloc[tail]),
        "bb_lower": series(bb_lower.iloc[tail]),
        "rsi": series(rsi_series.iloc[tail]),
        "macd": series(macd_line.iloc[tail]),
        "macd_signal": series(signal_line.iloc[tail]),
        "macd_hist": series((macd_line - signal_line).iloc[tail]),
        # 선택 보조지표 — 화면에서 켤 때만 그려집니다
        "cci": series(cci_series.iloc[tail]),
        "stoch_k": series(stoch_k.iloc[tail]),
        "stoch_d": series(stoch_d.iloc[tail]),
        "atr": series(atr_series.iloc[tail]),
        "obv": series(obv_series.iloc[tail]),
        "mfi": series(mfi_series.iloc[tail]),
        "bars_total": len(df),
    }


def detect_regime(df: pd.DataFrame) -> dict:
    """지금이 '추세 국면'인지 '평균회귀 국면'인지 통계적으로 판별합니다.

    왜 필요한가
        추세추종 지표(MACD·이동평균)와 평균회귀 지표(RSI·볼린저)는 서로 반대 신호를
        내는 일이 잦습니다. 고정 가중치로 합치면 둘이 상쇄돼 신호가 뭉개집니다.
        그래서 먼저 국면을 판별하고, 그 국면에서 실제로 작동하는 지표군에
        가중치를 몰아줍니다.

    판별 재료 (셋 다 서로 다른 각도에서 같은 질문을 봄)
        · Lo-MacKinlay 분산비율 VR(5) — 수익률에 양/음의 자기상관이 있는가
        · Hurst 지수 (Anis-Lloyd 보정) — 시계열이 지속적인가 반지속적인가
        · ADX — 방향성 추세의 강도가 있는가

    결과
        trend_score  -1(강한 평균회귀) ~ +1(강한 추세)
        multipliers  지표군별 가중치 배수
    """
    close = df["close"].values
    evidence, score_parts = [], []

    vr = quant.variance_ratio(close, q=5)
    if vr:
        # VR 1.0 기준, 0.5~1.5 를 ±1 로 매핑. 유의하지 않으면 절반만 반영
        part = _clip((vr["vr"] - 1.0) * 2)
        if not vr["significant"]:
            part *= 0.5
        score_parts.append(part)
        evidence.append(
            f"분산비율 VR(5)={vr['vr']:.3f}"
            + (f" (z={vr['z']:+.2f}, p={vr['p_value']:.3f}, 랜덤워크 기각)"
               if vr["significant"] else f" (z={vr['z']:+.2f}, 통계적 유의성 없음)")
        )

    hurst = quant.hurst_exponent(close)
    if hurst:
        score_parts.append(_clip((hurst["hurst"] - 0.5) * 4))
        evidence.append(
            f"Hurst={hurst['hurst']:.3f} ("
            + {"persistent": "지속성", "antipersistent": "반지속성",
               "random": "랜덤워크"}[hurst["regime"]] + ")"
        )

    adx_result = adx(df)
    adx_value = None
    if adx_result:
        adx_value = adx_result.values.get("adx")
        if adx_value is not None:
            score_parts.append(_clip((adx_value - 22) / 18))
            evidence.append(f"ADX={adx_value:.1f}")

    hl = quant.ou_half_life(close)
    if hl and hl.get("half_life"):
        evidence.append(f"평균회귀 반감기 {hl['half_life']:.1f}봉")

    trend_score = float(np.mean(score_parts)) if score_parts else 0.0

    if trend_score >= 0.25:
        regime, label = "TREND", "추세 국면"
        multipliers = {"trend": 1.45, "meanrev": 0.55, "confirm": 1.0, "info": 0.0}
        strategy = "추세추종 지표(MACD·이동평균·ADX)에 비중을 싣고, 되돌림 지표는 낮췄습니다."
    elif trend_score <= -0.25:
        regime, label = "MEAN_REVERT", "평균회귀 국면"
        multipliers = {"trend": 0.55, "meanrev": 1.45, "confirm": 1.0, "info": 0.0}
        strategy = "되돌림 지표(RSI·볼린저·CCI)에 비중을 싣고, 추세추종 지표는 낮췄습니다."
    else:
        regime, label = "RANDOM", "방향성 불분명"
        multipliers = {"trend": 0.8, "meanrev": 0.8, "confirm": 1.0, "info": 0.0}
        strategy = ("추세도 되돌림도 통계적으로 뚜렷하지 않아 모든 지표의 비중을 낮추고 "
                    "확률을 50%에 가깝게 유지합니다.")

    return {
        "regime": regime,
        "label": label,
        "trend_score": round(trend_score, 4),
        "multipliers": multipliers,
        "strategy": strategy,
        "evidence": evidence,
        "variance_ratio": vr,
        "hurst": hurst,
        "adx": adx_value,
        "half_life": hl,
    }


def analyze(df: pd.DataFrame) -> TechnicalAnalysis:
    """일봉 DataFrame(open/high/low/close/volume) -> 종합 기술 분석."""
    if df is None or df.empty:
        return TechnicalAnalysis(
            score=0.0, indicators=[], bars_used=0, sufficient_data=False,
            note="일봉 시세를 받아오지 못해 기술적 분석을 수행하지 못했습니다. 기술점수는 0(중립)으로 처리됩니다.",
        )

    df = df.dropna(subset=["close"]).copy()
    bars = len(df)
    if bars < MIN_BARS_FOR_TECHNICAL:
        return TechnicalAnalysis(
            score=0.0, indicators=[], bars_used=bars, sufficient_data=False,
            note=(f"일봉이 {bars}개뿐이라 최소 {MIN_BARS_FOR_TECHNICAL}개 요건을 채우지 못했습니다. "
                  f"신규 상장 종목이거나 거래정지 이력이 있을 수 있어 기술점수는 0(중립)으로 처리됩니다."),
        )

    close = df["close"]
    computed = [
        # 추세추종형
        macd(close), ma_cross(close), ma_trend(close), roc(close),
        obv(df), price_position(df), adx(df), aroon(df),
        trend_regression_indicator(df),
        # 평균회귀형
        rsi(close), bollinger(close), stochastic(df), cci(df), williams_r(df),
        # 확인형(수급)
        volume_signal(df), mfi(df), cmf(df),
        # 참고용
        atr(df), yang_zhang_indicator(df),
    ]
    indicators = [i for i in computed if i is not None]

    # 국면을 먼저 판별하고, 그 국면에서 유효한 지표군에 가중치를 몰아줍니다
    regime = detect_regime(df)
    multipliers = regime["multipliers"]

    for ind in indicators:
        family = INDICATOR_FAMILY.get(ind.key, "confirm")
        ind.family = family
        ind.weight = round(ind.weight * multipliers.get(family, 1.0), 4)

    weighted_sum = sum(i.score * i.weight for i in indicators)
    total_weight = sum(i.weight for i in indicators)
    score = round(weighted_sum / total_weight, 4) if total_weight else 0.0

    skipped = len(computed) - len(indicators)
    note = (f"일봉 {bars}개 기준 {len(indicators)}개 지표를 계산했습니다. "
            f"국면 판정: {regime['label']} (추세점수 {regime['trend_score']:+.2f}). "
            f"{regime['strategy']}")
    if skipped:
        note += f" 데이터 부족으로 {skipped}개 지표는 건너뛰었습니다."

    # 화면에서 영향력이 큰 순으로 보이도록 정렬
    indicators.sort(key=lambda i: abs(i.contribution), reverse=True)

    return TechnicalAnalysis(
        score=score, indicators=indicators, bars_used=bars,
        sufficient_data=True, note=note, regime=regime,
    )
