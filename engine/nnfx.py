"""
NNFX 규칙 오버레이 (No Nonsense Forex — 주식 재보정판)
------------------------------------------------------
NNFX는 FX 스윙 매매용 규칙 묶음입니다. 기준선(baseline) 위에 있고, 서로 다른
계열의 확인지표 두 개가 동의하고, 거래가 실제로 붙어 있고, 가격이 기준선에서
너무 멀지 않을 때만 들어갑니다. 각 슬롯은 "무엇을 볼 것인가"만 정하고 지표는
갈아 끼울 수 있게 설계돼 있습니다.

**이 모듈의 핵심은 NNFX를 그대로 쓰지 않는다는 것입니다.**

원본 NNFX는 모든 슬롯을 AND로 묶습니다(하드 게이트). 합성시장 실측에서 이 방식은
주식 횡단면에서 통과율이 10%까지 떨어져 후보가 고갈되고, 액티브 샤프가 음수로
갔습니다(−0.110). FX는 28쌍을 훑어 하나만 걸리면 되는 구조라 통과율이 낮아도
되지만, 소수 종목을 보는 구조에서는 그대로 굶습니다.

그래서 기본 사용법은 **소프트 결합**입니다. 슬롯 통과 여부를 0/1 관문이 아니라
점수로 바꿔 기존 신호에 가중 평균으로 섞습니다(실측 최적 가중 0.30). 같은 실측에서
소프트 결합은 액티브 샤프를 39% 올렸고, 거부권(veto)만 쓰는 방식이 그다음이었습니다.

모드
    off   쓰지 않음 (기본값 — 켜는 것은 사람의 결정입니다)
    soft  점수 결합. 신호를 죽이지 않고 밀거나 당깁니다  ← 권장
    veto  치명적 위반(기준선 반대 · 기준선에서 너무 멂)일 때만 진입 차단
    hard  원본 NNFX (전 슬롯 AND) — 실측에서 기각됐습니다. 재현용으로만 둡니다

주의: 여기 지표들은 기존 indicators.analyze 가 이미 보는 것과 겹칩니다. 그래서
      **점수에 그냥 더하면 같은 정보를 두 번 세게 됩니다.** 반드시 가중 평균으로
      섞습니다(같은 실측에서 NNFX를 별도 피처로 추가하자 오히려 22% 나빠졌습니다).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 실측으로 정해진 기본값. FX 원본과 다른 곳은 주석에 이유를 적었습니다.
DEFAULTS = {
    "baseline_period": 20,       # 기준선 EMA
    "c1_fast": 12, "c1_slow": 26, "c1_signal": 9,      # 확인1 = MACD 히스토그램
    "c2_period": 14,             # 확인2 = ADX/DI (C1과 다른 계열이어야 의미가 있음)
    "c2_adx_min": 20.0,          # 추세가 있다고 볼 최소 ADX
    "atr_period": 14,
    "bridge_atr_mult": 3.0,      # FX 원본은 1.0. 주식은 갭·변동성이 커서 1.0이면
                                 # 정상 추세도 '너무 멀다'로 걸러집니다 (실측 재보정)
    "volume_ratio_min": 0.7,     # 최근 거래량 / 20일 평균
    "soft_weight": 0.30,         # 실측 최적
}

OFF, SOFT, VETO, HARD = "off", "soft", "veto", "hard"
MODES = (OFF, SOFT, VETO, HARD)

# 슬롯 가중 — 합이 1이 되게 둡니다. 기준선과 확인지표에 무게를 싣고,
# 거래량은 '있으면 가점' 정도로만 봅니다(저유동 종목을 죽이는 것은 별도 필터의 일).
_SLOT_WEIGHTS = {"baseline": 0.30, "c1": 0.25, "c2": 0.25,
                 "volume": 0.10, "bridge": 0.10}


@dataclass
class NNFXState:
    ok: bool = False
    score: float = 0.0            # -1 ~ +1 (소프트 결합에 쓰는 값)
    slots: dict = field(default_factory=dict)     # 슬롯별 통과 여부
    hard_ok: bool = False         # 원본 AND 게이트
    veto_ok: bool = True          # 치명적 위반이 없는가
    baseline: float = 0.0
    atr: float = 0.0
    distance_atr: float = 0.0     # 기준선에서 몇 ATR 떨어져 있는가
    reasons: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "score": round(self.score, 4), "slots": self.slots,
                "hard_ok": self.hard_ok, "veto_ok": self.veto_ok,
                "baseline": round(self.baseline, 4), "atr": round(self.atr, 4),
                "distance_atr": round(self.distance_atr, 2),
                "reasons": self.reasons[:4], "error": self.error}


def config_from(cfg: dict) -> dict:
    """자동매매 설정에서 NNFX 파라미터만 뽑아 기본값과 병합."""
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        value = (cfg or {}).get(f"nnfx_{key}")
        if value is not None:
            out[key] = value
    return out


def mode_of(cfg: dict) -> str:
    mode = str((cfg or {}).get("nnfx_mode") or OFF).lower()
    return mode if mode in MODES else OFF


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=max(int(period), 1), adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> float:
    """평균 진폭 (Wilder). 갭을 포함하므로 종가 변동폭보다 큽니다."""
    high, low, close = df["high"], df["low"], df["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    value = tr.ewm(alpha=1 / max(int(period), 1), adjust=False).mean().iloc[-1]
    return float(value) if np.isfinite(value) else 0.0


def _adx_di(df: pd.DataFrame, period: int) -> tuple[float, float, float]:
    """(ADX, +DI, −DI). 추세의 '세기'와 '방향'을 나눠서 봅니다."""
    high, low, close = df["high"], df["low"], df["close"]
    up, down = high.diff(), -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()],
                   axis=1).max(axis=1)
    alpha = 1 / max(int(period), 1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    # ATR이 0인 구간(거래정지 등)에서 0으로 나누면 inf가 퍼집니다
    safe = atr.replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / safe
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / safe
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    pick = lambda s: float(s.iloc[-1]) if np.isfinite(s.iloc[-1]) else 0.0
    return pick(adx), pick(plus_di), pick(minus_di)


def compute(bars: pd.DataFrame, cfg: dict = None) -> NNFXState:
    """일봉으로 슬롯을 채웁니다.

    반환 `score` 는 롱 관점 −1 ~ +1 입니다. 슬롯이 전부 롱에 동의하면 +1,
    전부 반대하면 −1, 반반이면 0 근처가 됩니다.
    """
    state = NNFXState()
    params = config_from(cfg or {})
    need = max(int(params["baseline_period"]), int(params["c1_slow"]),
               int(params["c2_period"]) * 2, int(params["atr_period"])) + 5
    if bars is None or len(bars) < need:
        state.error = f"NNFX 계산에 일봉이 부족합니다 ({0 if bars is None else len(bars)}/{need})"
        return state

    close = bars["close"].astype(float)
    price = float(close.iloc[-1])
    if price <= 0:
        state.error = "가격이 0 이하입니다."
        return state

    # --- 슬롯 1: 기준선 (추세의 방향) ---
    baseline = float(_ema(close, params["baseline_period"]).iloc[-1])
    base_long = price > baseline

    # --- 슬롯 2: 확인1 = MACD 히스토그램 (모멘텀) ---
    macd_line = _ema(close, params["c1_fast"]) - _ema(close, params["c1_slow"])
    hist = macd_line - _ema(macd_line, params["c1_signal"])
    c1_long = float(hist.iloc[-1]) > 0

    # --- 슬롯 3: 확인2 = ADX/DI (추세의 세기 — C1과 다른 계열) ---
    adx, plus_di, minus_di = _adx_di(bars, params["c2_period"])
    c2_long = adx >= float(params["c2_adx_min"]) and plus_di > minus_di

    # --- 슬롯 4: 거래량 (움직임에 참여자가 있는가) ---
    volume_ok = True
    if "volume" in bars.columns:
        vol = bars["volume"].astype(float)
        avg = float(vol.iloc[-20:].mean()) if len(vol) >= 20 else float(vol.mean())
        volume_ok = bool(avg > 0 and float(vol.iloc[-1]) / avg >= float(params["volume_ratio_min"]))

    # --- 슬롯 5: Bridge Too Far (추격매수 방지) ---
    atr = _atr(bars, params["atr_period"])
    distance = abs(price - baseline) / atr if atr > 0 else 0.0
    bridge_ok = distance <= float(params["bridge_atr_mult"])

    slots = {"baseline": base_long, "c1": c1_long, "c2": c2_long,
             "volume": volume_ok, "bridge": bridge_ok}

    # --- 점수: 통과 슬롯을 가중 평균해 -1~+1 로 ---
    # 방향성이 있는 슬롯(기준선·확인1·확인2)은 통과=+1 / 미통과=−1 로 놓고,
    # 방향이 없는 슬롯(거래량·거리)은 통과=+1 / 미통과=0 으로 둡니다. 거래량이
    # 적다는 것은 '팔아야 한다'가 아니라 '근거가 약하다'이기 때문입니다.
    directional = ("baseline", "c1", "c2")
    score = sum(_SLOT_WEIGHTS[k] * ((1.0 if slots[k] else -1.0) if k in directional
                                    else (1.0 if slots[k] else 0.0))
                for k in _SLOT_WEIGHTS)

    state.ok = True
    state.score = max(-1.0, min(1.0, float(score)))
    state.slots = slots
    state.baseline, state.atr, state.distance_atr = baseline, atr, distance
    # 원본 NNFX = 전 슬롯 AND. 실측에서 기각됐지만 재현을 위해 계산은 해둡니다.
    state.hard_ok = all(slots.values())
    # 거부권은 '치명적 위반'만 — 기준선 반대에 있거나, 너무 멀리 떨어져 있을 때.
    # 확인지표 불일치는 흔한 일이라 여기 넣으면 hard 게이트와 다를 바 없어집니다.
    state.veto_ok = base_long and bridge_ok

    if not base_long:
        state.reasons.append(f"기준선 아래 ({price:,.0f} < {baseline:,.0f})")
    if not bridge_ok:
        state.reasons.append(f"기준선에서 {distance:.1f}ATR — 너무 멀어 추격 위험")
    if base_long and c1_long and c2_long:
        state.reasons.append(f"기준선·모멘텀·추세(ADX {adx:.0f}) 모두 롱")
    elif not c2_long and adx < float(params["c2_adx_min"]):
        state.reasons.append(f"추세 약함 (ADX {adx:.0f} < {params['c2_adx_min']:g})")
    if not volume_ok:
        state.reasons.append("거래량이 평소보다 적습니다")
    return state


def apply_to_score(base_score: float, state: NNFXState, cfg: dict) -> tuple[float, str]:
    """소프트 결합 — 기존 점수와 가중 평균.

    더하지 않고 섞는 이유: NNFX 슬롯이 보는 것(추세·모멘텀·거래량)은 기존 지표
    점수에 이미 들어 있습니다. 더하면 같은 근거를 두 번 세어 점수가 부풀고,
    임계값이 실질적으로 낮아집니다.
    """
    if not state.ok:
        return base_score, ""
    weight = float(config_from(cfg)["soft_weight"])
    weight = max(0.0, min(weight, 1.0))
    blended = (1 - weight) * base_score + weight * state.score
    return blended, f"NNFX {state.score:+.2f} (가중 {weight:.0%})"


def describe() -> list[dict]:
    """화면에 띄울 모드 설명 — 실측 근거를 함께 보여줍니다."""
    return [
        {"mode": OFF, "label": "사용 안 함", "note": "기존 지표 점수만 씁니다."},
        {"mode": SOFT, "label": "소프트 결합 (권장)",
         "note": f"NNFX 점수를 가중 {DEFAULTS['soft_weight']:.0%}로 섞습니다. "
                 "합성시장 실측에서 액티브 샤프 +39%로 가장 좋았습니다."},
        {"mode": VETO, "label": "거부권만",
         "note": "기준선 아래이거나 기준선에서 너무 멀면 진입만 막습니다 (+26%)."},
        {"mode": HARD, "label": "원본 NNFX (권장하지 않음)",
         "note": "전 슬롯 AND. 실측에서 통과율 10%로 굶고 액티브 샤프가 음수(−0.110)였습니다."},
    ]
