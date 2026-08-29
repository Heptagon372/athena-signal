"""
신호 합성 앙상블 (Signal Ensemble)
---------------------------------
지표 점수 하나로 "살까 팔까"를 정하던 자리에, 서로 **다른 질문에 답하는** 네 개의
판단을 겹쳐 놓습니다. 각 판단은 기존 점수를 대체하지 않고, 들어가도 되는지를
다른 각도에서 한 번 더 묻습니다.

    1) 시평선 합치 (Multi-Timeframe Confluence)   freqtrade 의 informative timeframe
       일봉과 분봉이 서로 반대를 가리키면 확신을 깎습니다. 지금까지는 가중평균만
       해서, 일봉이 아주 강하면 분봉이 반대여도 그대로 진입했습니다.

    2) 시장 난기류 (Turbulence)                    FinRL 의 turbulence index
       Kritzman & Li (2010) 의 마할라노비스 거리. "오늘의 (수익률·진폭·거래량)
       조합이 지난 1년의 정상 범위에서 얼마나 벗어났는가"를 잽니다. 지표는
       평상시 분포를 가정하고 만들어졌으므로, 분포가 깨진 날은 지표를 덜 믿어야
       합니다.

    3) 변동성 스케일 배리어                        hummingbot 의 TripleBarrierConfig
       .new_instance_with_adjusted_volatility() 를 옮긴 것입니다. 손절·익절·트레일링
       폭을 현재 변동성 / 평상시 변동성 배수로 늘리고 줄입니다.

    4) 비용 대비 기대이익                          freqtrade 의 수수료 반영 백테스트
       왕복 수수료·거래세·슬리피지를 넘지 못하는 목표가는 이길 수 없는 매매입니다.
       한국 주식은 매도 시 거래세 0.18% 가 붙어서, 0.5% 를 노리는 매매의 3분의 1이
       비용으로 나갑니다.

모드 (algo_mode)
    off      쓰지 않음
    observe  전부 계산해서 신호에 붙이지만 **판단은 바꾸지 않습니다** (기본값)
    soft     점수와 배리어를 조정합니다 (진입 차단은 하지 않음)
    gate     위 조정에 더해, 조건을 못 넘으면 신규 진입을 막습니다

왜 observe 가 기본인가
    매매 방식을 바꾸는 스위치가 돌고 있는 계정에서 조용히 켜지면 안 됩니다.
    observe 는 "이 규칙이 켜져 있었다면 무엇이 달라졌을지"를 로그에 남기므로,
    며칠 지켜본 뒤 근거를 갖고 soft/gate 로 올릴 수 있습니다.

이중 계상 주의
    여기 쓰는 재료(추세·변동성·거래량)는 indicators.analyze 가 이미 보고 있습니다.
    그래서 점수에 **더하지 않고 곱하거나 섞습니다**. 더하면 같은 근거를 두 번 세어
    임계값이 실질적으로 낮아집니다 (engine/nnfx.py 가 같은 이유로 소프트 결합을 씁니다).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import quant

OFF, OBSERVE, SOFT, GATE = "off", "observe", "soft", "gate"
MODES = (OFF, OBSERVE, SOFT, GATE)

# Market Pulse 상태 문자열 (engine/marketpulse.py 와의 계약 — 순환 임포트 없이)
UNDER_PRESSURE_STATE, CORRECTION_STATE = "UNDER_PRESSURE", "CORRECTION"

DEFAULTS = {
    # 1) 시평선 합치
    "mtf_disagree_penalty": 0.5,   # 일봉·분봉이 반대일 때 점수를 얼마나 깎을지
    "mtf_agree_boost": 0.0,        # 같은 방향일 때 얼마나 키울지 (기본 0 — 아래 주석)

    # 2) 난기류
    "turbulence_window": 250,      # 정상 분포를 추정할 구간 (거래일 ≈ 1년)
    "turbulence_pct": 95.0,        # 이 백분위를 넘으면 '난기류'
    "turbulence_damping": 0.5,     # 난기류일 때 점수에 곱할 값

    # 3) 변동성 스케일
    "vol_fast": 10,                # 현재 변동성 창
    "vol_slow": 60,                # 평상시 변동성 창
    "vol_factor_min": 0.6,
    "vol_factor_max": 2.5,

    # 4) 비용
    "cost_edge_multiple": 3.0,     # 목표 수익이 왕복 비용의 몇 배는 돼야 하는가

    # 5) Market Pulse (오닐 시장 방향 — engine/marketpulse.py)
    # 감쇠만 하고 차단하지 않습니다. prism-insight 의 V2 매매 표본 감사가
    # "조정장 전면 매수 중단"을 기각했습니다 — 조정 구간 매수는 손절율 38%
    # 였지만 순손익 +25.3% (폭락 후 반등 대어가 이 구간에 삽니다).
    "pulse_pressure_damping": 0.85,    # UNDER_PRESSURE 일 때 점수에 곱할 값
    "pulse_correction_damping": 0.6,   # CORRECTION 일 때
}


@dataclass
class EnsembleState:
    ok: bool = False
    mode: str = OBSERVE
    base_score: float = 0.0        # 들어온 점수 (일봉·분봉 혼합 전 기준)
    score: float = 0.0             # 조정 후 점수
    vol_factor: float = 1.0
    # 시평선 합치 **뒤에** 점수에 곱해진 감쇠(난기류·시장 방향)의 누적곱.
    # 이걸 따로 들고 있는 이유는 "분봉을 빼면 점수가 얼마였을까"를 나눗셈 없이
    # 정확히 계산하기 위해서입니다 — base_score 가 0 근처면 비율로는 못 구합니다.
    damping: float = 1.0
    block: bool = False            # gate 모드에서 신규 진입을 막아야 하는가
    mtf: dict = field(default_factory=dict)
    turbulence: dict = field(default_factory=dict)
    volatility: dict = field(default_factory=dict)
    pulse: dict = field(default_factory=dict)      # Market Pulse (시장 방향)
    notes: list = field(default_factory=list)      # 사람이 읽는 조정 내역
    blocked_by: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "mode": self.mode,
                "base_score": round(self.base_score, 4),
                "score": round(self.score, 4),
                "vol_factor": round(self.vol_factor, 3),
                "damping": round(self.damping, 4),
                "block": self.block, "mtf": self.mtf,
                "turbulence": self.turbulence, "volatility": self.volatility,
                "pulse": self.pulse,
                "notes": self.notes[:6], "blocked_by": self.blocked_by,
                "error": self.error}


def mode_of(cfg: dict) -> str:
    mode = str((cfg or {}).get("algo_mode") or OBSERVE).lower()
    return mode if mode in MODES else OBSERVE


def config_from(cfg: dict) -> dict:
    """자동매매 설정에서 algo_* 파라미터만 뽑아 기본값과 병합."""
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        value = (cfg or {}).get(f"algo_{key}")
        if value is not None:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# 1) 시평선 합치 (Multi-Timeframe Confluence)
# ---------------------------------------------------------------------------

def blend_timeframes(daily: float, intraday: float | None, weight: float,
                     cfg: dict = None) -> tuple[float, dict]:
    """일봉 점수와 분봉 점수를 합칩니다.

    기존 방식은 단순 가중평균이었습니다.
        score = (1-w)·일봉 + w·분봉

    이 식의 빈틈은 **한쪽이 아주 강할 때** 드러납니다. w=0.35 에서
    일봉 +0.90 / 분봉 −0.20 이면 합이 +0.52 로 임계값(0.35)을 넘어 진입합니다.
    그러나 실제 상황은 "일봉은 상승인데 지금 이 순간은 팔리고 있다"이고,
    바로 그 자리에서 사면 진입 직후 손절에 닿기 쉽습니다.

    그래서 부호가 반대일 때만 추가로 깎습니다. 얼마나 깎을지는 **약한 쪽의
    세기**에 비례합니다 — 분봉이 −0.02 로 살짝 반대인 것과 −0.60 으로 확실히
    반대인 것은 다릅니다.

        불일치도 = min(|일봉|, |분봉|) / max(|일봉|, |분봉|)     (0~1)
        score   = 가중평균 × (1 − penalty × 불일치도)

    같은 방향일 때 키우는 boost 는 기본 0 입니다. 점수를 부풀리면 임계값을
    낮춘 것과 같아서 매매가 늘어나는데, 늘어난 매매가 더 좋다는 근거가 아직
    없습니다. 깎는 쪽(위험 감소)만 기본으로 켭니다.
    """
    params = config_from(cfg or {})
    weight = float(max(0.0, min(1.0, weight)))
    if intraday is None:
        return float(daily), {"used": False, "reason": "분봉 점수 없음"}

    daily, intraday = float(daily), float(intraday)
    base = (1 - weight) * daily + weight * intraday

    strong, weak = max(abs(daily), abs(intraday)), min(abs(daily), abs(intraday))
    conflict = (weak / strong) if strong > 1e-9 else 0.0
    opposed = (daily > 0) != (intraday > 0) and weak > 1e-9

    info = {"used": True, "daily": round(daily, 4), "intraday": round(intraday, 4),
            "weight": round(weight, 3), "raw": round(base, 4),
            "agree": not opposed, "conflict": round(conflict, 3)}

    if opposed:
        penalty = float(max(0.0, min(1.0, params["mtf_disagree_penalty"])))
        adjusted = base * (1 - penalty * conflict)
        info["factor"] = round(1 - penalty * conflict, 3)
        info["note"] = (f"일봉 {daily:+.2f} / 분봉 {intraday:+.2f} 방향 불일치 — "
                        f"확신 {penalty * conflict:.0%} 감산")
    else:
        boost = float(max(0.0, min(1.0, params["mtf_agree_boost"])))
        adjusted = base * (1 + boost * conflict)
        info["factor"] = round(1 + boost * conflict, 3)
        if boost > 0 and conflict > 0:
            info["note"] = f"두 시평선 동의 — 확신 {boost * conflict:.0%} 가산"

    info["score"] = round(float(adjusted), 4)
    return float(max(-1.0, min(1.0, adjusted))), info


# ---------------------------------------------------------------------------
# 2) 난기류 지수 (Turbulence)
# ---------------------------------------------------------------------------

def turbulence(df: pd.DataFrame, cfg: dict = None) -> dict | None:
    """오늘의 시장 상태가 평상시 분포에서 얼마나 벗어나 있는가.

    Kritzman, M., & Li, Y. (2010). "Skulls, Financial Turbulence, and Risk
    Management." Financial Analysts Journal, 66(5).
    FinRL 의 FeatureEngineer.calculate_turbulence 가 쓰는 것과 같은 통계량입니다.

        d_t = (y_t − μ)' Σ⁻¹ (y_t − μ)

    FinRL 원본은 종목 여러 개의 **횡단면** 수익률 벡터를 씁니다(다우 30 종목).
    여기서는 한 종목만 볼 때도 쓸 수 있도록 관측 벡터를 바꿨습니다.

        y_t = [ 로그수익률, 로그(고가/저가), 거래량 로그편차 ]

    세 축이 서로 다른 종류의 이상을 잡습니다 — 가격이 튀는 것, 장중 진폭이
    벌어지는 것, 거래가 몰리는 것. 셋이 동시에 벗어난 날이 진짜 위험한 날입니다.
    (횡단면 원본이 필요하면 아래 market_turbulence 를 쓰세요.)

    왜 마할라노비스인가
        세 값을 각각 z 로 보고 임계값을 걸면, "수익률은 평범한데 거래량만 폭증"
        같은 조합을 놓칩니다. 공분산의 역행렬을 끼우면 **평소 같이 움직이지 않던
        것들이 같이 움직인** 날을 잡아냅니다. 그게 시장이 깨지는 방식입니다.
    """
    params = config_from(cfg or {})
    window = int(params["turbulence_window"])
    features = _turbulence_features(df, window)
    if features is None:
        return None

    y, index = features
    # 오늘을 빼고 평상시 분포를 추정합니다. 오늘을 넣으면 이상치가 스스로
    # 공분산을 키워 자기 거리를 줄입니다 (자기 자신을 정상으로 만드는 편향).
    history = y[:-1]
    if len(history) < 40:
        return None

    mu = history.mean(axis=0)
    cov = np.cov(history, rowvar=False)
    cov = np.atleast_2d(cov)
    # 특이행렬(거래량이 전부 0 인 종목 등)에서도 죽지 않도록 유사역행렬을 씁니다
    inv = np.linalg.pinv(cov)

    dev = y - mu
    distances = np.einsum("ij,jk,ik->i", dev, inv, dev)
    distances = np.where(np.isfinite(distances) & (distances > 0), distances, 0.0)

    today = float(distances[-1])
    past = distances[:-1]
    percentile = float((past <= today).mean() * 100) if len(past) else 50.0

    threshold_pct = float(params["turbulence_pct"])
    if percentile >= 99.0:
        level, label = "extreme", "극단"
    elif percentile >= threshold_pct:
        level, label = "stressed", "난기류"
    elif percentile >= 80.0:
        level, label = "elevated", "불안정"
    else:
        level, label = "calm", "평상"

    return {
        "distance": round(today, 3),
        "percentile": round(percentile, 1),
        "threshold_pct": threshold_pct,
        "level": level, "label": label,
        "elevated": percentile >= threshold_pct,
        "samples": int(len(past)),
        "at": str(index[-1])[:16],
        "features": ["수익률", "장중진폭", "거래량편차"],
    }


def _turbulence_features(df: pd.DataFrame, window: int):
    """(관측행렬, 인덱스). 표본이 모자라거나 분산이 없는 축은 걸러냅니다."""
    if df is None or len(df) < 60:
        return None
    d = df.iloc[-(window + 1):]
    try:
        close = d["close"].astype(float)
        high = d["high"].astype(float)
        low = d["low"].astype(float)
        volume = d["volume"].astype(float)
    except (KeyError, TypeError, ValueError):
        return None

    ret = np.log(close / close.shift(1))
    rng = np.log((high / low.replace(0, np.nan)).clip(lower=1.0))
    # 거래량은 절대 규모가 종목마다 달라 그대로 못 씁니다. 로그를 취한 뒤
    # 20봉 중앙값과의 차이로 바꿔 '평소 대비 얼마나'만 남깁니다.
    log_vol = np.log1p(volume.clip(lower=0))
    vol_dev = log_vol - log_vol.rolling(20, min_periods=5).median()

    frame = pd.concat([ret, rng, vol_dev], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 50:
        return None

    y = frame.to_numpy(dtype=float)
    # 분산이 0 인 축(거래량을 안 주는 제공처)은 빼야 공분산이 덜 병적입니다
    keep = y.std(axis=0) > 1e-12
    if not keep.any():
        return None
    return y[:, keep], frame.index


def market_turbulence(closes: pd.DataFrame, window: int = 252) -> dict | None:
    """FinRL 원본 그대로의 **횡단면** 난기류 — 종목 여러 개의 종가 표가 필요합니다.

    closes : index=날짜, columns=종목, values=종가

    한 종목의 급락과 시장 전체의 동반 급락은 성격이 다릅니다. 전자는 그 종목만
    피하면 되지만, 후자는 어느 종목을 사도 같이 맞습니다. 유니버스 전체를 한
    번에 판정할 때 이쪽을 씁니다.
    """
    if closes is None or closes.shape[0] < 60 or closes.shape[1] < 2:
        return None

    returns = closes.astype(float).pct_change().dropna(how="all")
    returns = returns.iloc[-(window + 1):].dropna(axis=1, how="any")
    if returns.shape[0] < 40 or returns.shape[1] < 2:
        return None

    history = returns.iloc[:-1].to_numpy(dtype=float)
    today = returns.iloc[-1].to_numpy(dtype=float)

    mu = history.mean(axis=0)
    inv = np.linalg.pinv(np.cov(history, rowvar=False))
    dev_today = today - mu
    distance = float(dev_today @ inv @ dev_today)

    dev_hist = history - mu
    past = np.einsum("ij,jk,ik->i", dev_hist, inv, dev_hist)
    percentile = float((past <= distance).mean() * 100)

    return {"distance": round(max(distance, 0.0), 3),
            "percentile": round(percentile, 1),
            "tickers": int(returns.shape[1]),
            "samples": int(len(past)),
            "elevated": percentile >= 95.0}


# ---------------------------------------------------------------------------
# 3) 변동성 스케일 배리어
# ---------------------------------------------------------------------------

def volatility_factor(df: pd.DataFrame, cfg: dict = None) -> dict:
    """현재 변동성 / 평상시 변동성.

    hummingbot 의 TripleBarrierConfig.new_instance_with_adjusted_volatility()
    와 같은 발상입니다 — 손절·익절 폭을 시장 상태에 맞춰 함께 늘리고 줄입니다.
    고정 4% 손절은 조용한 장에서는 너무 멀고, 흔들리는 장에서는 너무 가까워
    정상 등락에 털립니다.

    추정량은 Yang-Zhang 을 씁니다 (engine/quant.py). 종가-종가 방식보다 같은
    표본에서 분산이 훨씬 작고, 갭이 잦은 국내 종목에 특히 유리합니다.

    GARCH(1,1) 보정 (engine/garch.py — bashtage/arch 이식)
        Yang-Zhang 비율은 실현 변동성의 **사후** 비교입니다. GARCH 가 변동성
        군집을 유의하게 찾아내면(우도비 검정 통과), 그 **예측** 비율의 제곱근을
        보정 배수로 곱합니다 — 제곱근으로 눌러 쓰는 이유는 예측 모형 하나가
        배리어를 크게 흔들지 않게 하기 위해서입니다. GARCH 가 자격 미달이면
        기존 Yang-Zhang 값이 그대로 남습니다 (관찰 정보로만 첨부).

    **ATR 손절에는 이 배수를 곱하지 않습니다.** ATR 자체가 이미 변동성이라
    두 번 곱하면 손절 폭이 제곱으로 벌어집니다. 아래 scale_barriers 가 고정
    퍼센트 계열에만 배수를 적용하는 이유입니다.
    """
    params = config_from(cfg or {})
    fast_w, slow_w = int(params["vol_fast"]), int(params["vol_slow"])
    lo, hi = float(params["vol_factor_min"]), float(params["vol_factor_max"])

    fast = quant.yang_zhang_vol(df, window=fast_w)
    slow = quant.yang_zhang_vol(df, window=slow_w)
    if not fast or not slow or slow <= 0:
        return {"ok": False, "factor": 1.0, "reason": "변동성 추정 표본 부족"}

    raw = fast / slow

    # GARCH 예측 보정 — 자격을 통과했을 때만
    garch_info = None
    try:
        from engine import garch as garch_mod
        g = garch_mod.fit_bars(df, cfg)
        if g.ok:
            garch_info = g.to_dict()
        if g.converged and g.vol_ratio > 0:
            adj = float(np.clip(np.sqrt(g.vol_ratio), 0.85, 1.35))
            raw *= adj
            garch_info["applied_adj"] = round(adj, 3)
    except Exception:
        pass                       # 변동성 배수는 GARCH 없이도 성립해야 합니다

    factor = float(max(lo, min(hi, raw)))
    if factor >= 1.3:
        label = "평소보다 큼"
    elif factor <= 0.8:
        label = "평소보다 작음"
    else:
        label = "평상 수준"
    out = {"ok": True, "factor": round(factor, 3), "raw": round(float(raw), 3),
           "fast_vol": round(fast, 2), "slow_vol": round(slow, 2),
           "fast_window": fast_w, "slow_window": slow_w, "label": label}
    if garch_info is not None:
        out["garch"] = garch_info
    return out


def scale_barriers(cfg: dict, factor: float) -> dict:
    """변동성 배수를 손절·익절·트레일링 폭에 반영한 **설정 덮어쓰기 조각**.

    돌려주는 것은 cfg 전체가 아니라 바뀐 키만입니다. 호출부가
    `{**cfg, **scale_barriers(cfg, f)}` 로 합쳐 쓰면, 원본 설정은 그대로 남고
    이번 판단에만 적용됩니다 (저장된 사용자 설정을 건드리지 않습니다).

    건드리지 않는 것
        atr_stop_mult   ATR 이 이미 변동성이라 이중 적용이 됩니다
        max_hold_*      변동성이 크다고 보유 기간이 짧아져야 할 이유는 없습니다
                        (hummingbot 도 time_limit 은 스케일하지 않습니다)
    """
    factor = float(factor)
    if not np.isfinite(factor) or factor <= 0 or abs(factor - 1.0) < 1e-6:
        return {}
    out = {}
    for key in ("stop_loss_pct", "take_profit_pct", "trailing_stop_pct"):
        value = float(cfg.get(key) or 0)
        if value > 0:
            out[key] = round(value * factor, 4)
    return out


# ---------------------------------------------------------------------------
# 4) 비용 대비 기대이익
# ---------------------------------------------------------------------------

def cost_edge(inst, entry_price: float, target_price: float, cfg: dict = None) -> dict:
    """목표 수익이 왕복 거래비용을 이길 만큼 큰가.

    한국 주식 왕복 비용은 대략
        매수 수수료 0.015% + 매도 수수료 0.015% + 증권거래세 0.18% ≒ 0.21%
    여기에 스프레드·슬리피지가 편도 0.05% 씩 붙으면 0.31% 입니다.
    0.5% 를 노리는 매매는 이기더라도 6할이 비용으로 나갑니다. 승률이 웬만큼
    높지 않으면 기댓값이 음수인 매매인데, 점수만 보는 로직은 이걸 못 봅니다.

    ETF 는 매도 거래세가 면제라 같은 목표라도 훨씬 유리합니다 — 그 차이가
    Instrument.costs 에 이미 들어 있어서, 자산군을 나눠 쓰지 않아도 됩니다.
    """
    params = config_from(cfg or {})
    entry, target = float(entry_price or 0), float(target_price or 0)
    if entry <= 0 or target <= 0:
        return {"ok": True, "reason": "가격이 없어 비용 판정을 건너뜁니다."}

    # 비율만 필요하므로 명목금액은 1주 기준으로 잡습니다
    notional_in = inst.notional(entry, 1)
    notional_out = inst.notional(target, 1)
    if notional_in <= 0:
        return {"ok": True, "reason": "명목금액을 계산할 수 없습니다."}

    fee_in, tax_in = inst.costs("buy", notional_in)
    fee_out, tax_out = inst.costs("sell", notional_out)
    slip_bps = float((cfg or {}).get("paper_slippage_bps", 5.0) or 0)
    slippage = (notional_in + notional_out) * slip_bps / 10000.0

    cost = fee_in + tax_in + fee_out + tax_out + slippage
    cost_pct = cost / notional_in * 100
    target_pct = abs(target - entry) / entry * 100
    need = float(params["cost_edge_multiple"])
    ratio = (target_pct / cost_pct) if cost_pct > 0 else float("inf")

    return {
        "ok": ratio >= need,
        "cost_pct": round(cost_pct, 4),
        "target_pct": round(target_pct, 4),
        "ratio": round(float(ratio), 2) if np.isfinite(ratio) else None,
        "need": need,
        "breakeven_pct": round(cost_pct, 4),
        "reason": (f"목표 {target_pct:.2f}% 가 왕복비용 {cost_pct:.2f}% 의 "
                   f"{ratio:.1f}배 (필요 {need:g}배)"),
    }


# ---------------------------------------------------------------------------
# 통합 계산
# ---------------------------------------------------------------------------

def compute(bars_daily: pd.DataFrame, daily_score: float,
            intraday_score: float | None, cfg: dict,
            market: str = "") -> EnsembleState:
    """앙상블 한 번. 신호 생성부(strategy.evaluate)에서 호출합니다.

    market 은 Market Pulse 프록시 선택용입니다 ("KOSPI"/"US" 등). 비우면
    시장 상태 판독을 건너뜁니다 (백테스트가 종목 봉만 주입하는 경로).

    반환값의 score 는 **조정된 점수**지만, 모드가 observe 면 base_score 와 같습니다
    (계산은 다 하되 판단은 바꾸지 않는다는 뜻입니다).
    """
    state = EnsembleState(mode=mode_of(cfg))
    if state.mode == OFF:
        return state

    weight = float((cfg or {}).get("intraday_weight", 0.35))

    # (1) 시평선 합치
    blended, mtf = blend_timeframes(daily_score, intraday_score, weight, cfg)
    state.mtf = mtf
    state.base_score = float(blended)
    score = float(blended)
    if mtf.get("note"):
        state.notes.append(mtf["note"])

    # (2) 난기류
    try:
        turb = turbulence(bars_daily, cfg) or {}
    except Exception as exc:
        turb, state.error = {}, f"난기류 계산 실패: {type(exc).__name__}"
    state.turbulence = turb
    if turb.get("elevated"):
        damping = float(config_from(cfg)["turbulence_damping"])
        state.notes.append(
            f"난기류 {turb['label']} (상위 {100 - turb['percentile']:.0f}%) — "
            f"신호 확신 {1 - damping:.0%} 감산")
        if state.mode in (SOFT, GATE):
            factor = max(0.0, min(1.0, damping))
            score *= factor
            state.damping *= factor
        if state.mode == GATE and turb.get("level") == "extreme":
            state.block = True
            state.blocked_by.append(
                f"난기류 극단 (마할라노비스 {turb['distance']:.1f}, "
                f"상위 {100 - turb['percentile']:.1f}%) — 신규 진입 중단")

    # (3) 변동성 배수
    try:
        vol = volatility_factor(bars_daily, cfg)
    except Exception as exc:
        vol = {"ok": False, "factor": 1.0, "reason": f"{type(exc).__name__}"}
    state.volatility = vol
    state.vol_factor = float(vol.get("factor", 1.0)) if state.mode != OBSERVE else 1.0
    if vol.get("ok") and abs(float(vol["factor"]) - 1.0) >= 0.1:
        state.notes.append(
            f"변동성 {vol['label']} (배수 {vol['factor']:.2f}) — 손절·익절 폭 조정")

    # (4) Market Pulse — 오닐 시장 방향 (engine/marketpulse.py, prism-insight 이식).
    # 지수 프록시를 읽지 못하면 None → 아무 효과 없음 (fail-open).
    # **어떤 모드에서도 차단하지 않습니다** — 원본 V2 감사에서 조정장 매수가
    # 순손익 +25.3% 였습니다. 시장이 나쁠 때는 확신만 줄입니다.
    if market:
        try:
            from engine import marketpulse
            pulse = marketpulse.market_state(market)
        except Exception:
            pulse = None
        if pulse:
            state.pulse = pulse
            params = config_from(cfg)
            damping = {UNDER_PRESSURE_STATE: float(params["pulse_pressure_damping"]),
                       CORRECTION_STATE: float(params["pulse_correction_damping"])
                       }.get(pulse.get("state"))
            if damping is not None:
                state.notes.append(
                    f"시장 {pulse['label']} (분산일 {pulse['distribution_days']}개"
                    + (f", 랠리 {pulse['rally_day']}일차" if pulse.get("in_rally_attempt")
                       else "")
                    + f") — 신호 확신 {1 - damping:.0%} 감산")
                if state.mode in (SOFT, GATE):
                    factor = max(0.0, min(1.0, damping))
                    score *= factor
                    state.damping *= factor

    state.score = float(max(-1.0, min(1.0, score))) if state.mode in (SOFT, GATE) \
        else state.base_score
    state.ok = True
    return state


def describe() -> list[dict]:
    """화면에 띄울 모드 설명."""
    return [
        {"mode": OFF, "label": "사용 안 함",
         "note": "기존 신호 로직만 씁니다."},
        {"mode": OBSERVE, "label": "관찰 (기본)",
         "note": "시평선 합치·난기류·변동성·비용을 계산해 로그에 남기지만 "
                 "매매 판단은 바꾸지 않습니다. 먼저 여기서 며칠 지켜보세요."},
        {"mode": SOFT, "label": "점수 반영",
         "note": "시평선 불일치와 난기류로 점수를 깎고, 변동성 배수로 "
                 "손절·익절 폭을 조정합니다. 진입을 막지는 않습니다."},
        {"mode": GATE, "label": "진입 차단까지",
         "note": "위 조정에 더해 난기류 극단·비용 미달이면 신규 진입을 막습니다. "
                 "청산은 어떤 모드에서도 막지 않습니다."},
    ]


__all__ = ["compute", "EnsembleState", "blend_timeframes", "turbulence",
           "market_turbulence", "volatility_factor", "scale_barriers",
           "cost_edge", "mode_of", "config_from", "describe",
           "OFF", "OBSERVE", "SOFT", "GATE", "MODES", "DEFAULTS"]
