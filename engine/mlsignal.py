"""
ML 오버레이 (XGBoost·LightGBM·PatchTST·Kalman 이식 위원회)
---------------------------------------------------------
네 예측기를 하나의 점수로 묶어 기존 신호에 **소프트 결합**하는 곳입니다.

    engine/gbdt.py "xgb"   XGBoost 이식 — 깊이 우선 트리 (피처 상호작용)
    engine/gbdt.py "lgbm"  LightGBM 이식 — 리프 우선 + GOSS (같은 재료, 다른 성장)
    engine/patchtst.py     PatchTST 이식 — 시계열 모양의 반복 (패치 어텐션)
    engine/kalman.py       filterpy 이식 — 국소 선형 추세 + 적응형 Q (상태공간)

넷은 서로 다른 실패 방식을 갖습니다. 부스팅은 피처에 없는 패턴을 못 보고,
패치 어텐션은 과거에 없던 모양 앞에서 무력하며, 칼만 추세는 횡보에서 뒤집니다.
그래서 합의를 요구하지 않고 **각자의 검증 성적으로 가중**해 섞습니다 —
검증에서 무가치했던 쪽은 그 회전에서 자연히 0 에 가까운 가중치를 받습니다.

품질 게이트 — 예측기는 자격을 증명해야 점수에 들어갑니다
    GBDT 두 변형   시간순 검증 적중률 ≥ 기준(기본 55%) + 다수 클래스 초과(edge>0).
                   각자 따로 심사받습니다 (한쪽만 통과하는 날이 흔합니다).
    Patch          실질 이웃 수·방향 합의 confidence ≥ 기준
    Kalman         인과적 과거 채점 적중률 ≥ 기준 (표본 30개 이상)
    전원 탈락이면 이 회전에서 ML 은 **아무 힘도 갖지 않습니다** (관찰 기록만 남음).

모드 (ml_mode)
    off      계산하지 않음
    observe  계산해서 신호에 첨부만 — **기본값.** 매매 판단 불변
    soft     품질 게이트를 통과한 점수를 가중 평균으로 결합 (기본 25%)

더하지 않고 섞는 이유는 nnfx.py·ensemble.py 와 같습니다 — ML 피처(수익률·
RSI·MACD)는 기존 지표 점수와 재료가 겹쳐서, 더하면 같은 근거를 두 번 셉니다.

캐시
    학습은 봉이 갱신될 때만 다시 합니다. feed.bars 캐시(5분)와 맞물려,
    60초 회전마다 같은 데이터로 같은 모델을 다시 학습하는 낭비를 막습니다.
"""

import threading
from dataclasses import dataclass, field

import pandas as pd

from engine import gbdt, kalman, patchtst

OFF, OBSERVE, SOFT = "off", "observe", "soft"
MODES = (OFF, OBSERVE, SOFT)

DEFAULTS = {
    "weight": 0.25,             # soft 결합 가중 (nnfx soft_weight 와 같은 역할)
    "horizon_bars": 5,          # 두 예측기가 공통으로 보는 예측 시야 (봉)
    "min_val_accuracy": 0.55,   # GBDT 검증 적중률 하한
    "min_confidence": 0.25,     # Patch 확신도 하한
}

_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 256


@dataclass
class MLState:
    ok: bool = False
    mode: str = OBSERVE
    score: float | None = None       # 결합 점수 (품질 미달이면 None)
    confidence: float = 0.0
    gbdt: dict = field(default_factory=dict)          # XGBoost 방식
    lgbm: dict = field(default_factory=dict)          # LightGBM 방식
    patch: dict = field(default_factory=dict)
    kalman: dict = field(default_factory=dict)
    gbdt_used: bool = False
    lgbm_used: bool = False
    patch_used: bool = False
    kalman_used: bool = False
    notes: list = field(default_factory=list)
    error: str = ""

    @property
    def usable(self) -> bool:
        return self.ok and self.score is not None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "mode": self.mode,
                "score": round(self.score, 4) if self.score is not None else None,
                "confidence": round(self.confidence, 3),
                "gbdt": self.gbdt, "lgbm": self.lgbm,
                "patch": self.patch, "kalman": self.kalman,
                "gbdt_used": self.gbdt_used, "lgbm_used": self.lgbm_used,
                "patch_used": self.patch_used, "kalman_used": self.kalman_used,
                "notes": self.notes[:6], "error": self.error}


def mode_of(cfg: dict) -> str:
    mode = str((cfg or {}).get("ml_mode") or OBSERVE).lower()
    return mode if mode in MODES else OBSERVE


def config_from(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        value = (cfg or {}).get(f"ml_{key}")
        if value is not None:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------

def compute(key: str, bars: pd.DataFrame, cfg: dict) -> MLState:
    """한 종목의 ML 점수. 같은 봉에 대해서는 캐시를 씁니다."""
    state = MLState(mode=mode_of(cfg))
    if state.mode == OFF:
        return state
    if bars is None or len(bars) == 0:
        state.error = "봉 데이터가 없습니다."
        return state

    params = config_from(cfg)
    cache_key = _cache_key(key, bars, params)
    if cache_key is not None:
        with _cache_lock:
            hit = _cache.get(cache_key)
        if hit is not None:
            hit.mode = state.mode        # 모드는 캐시와 무관하게 현재 설정을 따름
            return hit

    state = _compute(bars, params, state)

    if cache_key is not None:
        with _cache_lock:
            if len(_cache) >= _CACHE_MAX:
                _cache.clear()           # 단순 전체 비움 — LRU 가 필요할 규모가 아님
            _cache[cache_key] = state
    return state


def _cache_key(key: str, bars: pd.DataFrame, params: dict):
    """봉이 실제로 갱신됐을 때만 바뀌는 키.

    마지막 봉의 (시각, 종가) 를 넣습니다 — 장중에 오늘 봉의 종가가 바뀌면
    다시 학습하고, 같은 5분 캐시 안에서는 재사용합니다.
    """
    try:
        last_ts = str(bars.index[-1])
        last_close = round(float(bars["close"].iloc[-1]), 6)
        return (str(key), len(bars), last_ts, last_close,
                tuple(sorted(params.items())))
    except Exception:
        return None


def _compute(bars: pd.DataFrame, params: dict, state: MLState) -> MLState:
    h = int(params["horizon_bars"])
    min_acc = float(params["min_val_accuracy"])
    min_conf = float(params["min_confidence"])
    parts: list[tuple[float, float]] = []      # (점수, 가중치)

    # 1) GBDT 두 변형 — 같은 피처, 다른 성장 정책. 각자 따로 심사합니다.
    #    가중치 = 검증 적중률의 초과분 (50% 를 넘은 만큼) — backtest.adjust_weights
    #    가 요소 가중치를 정하는 것과 같은 원리입니다.
    for variant, label, used_attr, dict_attr in (
            ("xgb", "GBDT", "gbdt_used", "gbdt"),
            ("lgbm", "LGBM", "lgbm_used", "lgbm")):
        try:
            g = gbdt.direction_score(bars, horizon_bars=h, variant=variant)
        except Exception as exc:
            g = gbdt.GBDTResult(error=f"{type(exc).__name__}: {exc}")
        setattr(state, dict_attr, g.to_dict())
        if g.ok and g.val_accuracy >= min_acc and g.edge > 0:
            parts.append((g.score, max(0.0, g.val_accuracy - 0.5) * 2))
            setattr(state, used_attr, True)
            state.notes.append(
                f"{label} {g.score:+.2f} (검증 {g.val_accuracy:.0%}, n={g.val_n})")
        elif g.ok:
            state.notes.append(
                f"{label} 검증 미달 ({g.val_accuracy:.0%} < {min_acc:.0%} "
                f"또는 edge≤0) — 미반영")
        elif g.error and variant == "xgb":     # 표본 부족 같은 공통 사유는 한 번만
            state.notes.append(f"GBDT 미가동 — {g.error}")

    # 2) 패치 어텐션 (PatchTST 이식) — 예측 시야를 GBDT 와 맞춥니다
    try:
        p = patchtst.forecast(bars, {"patch_horizon": h})
    except Exception as exc:
        p = patchtst.PatchForecast(error=f"{type(exc).__name__}: {exc}")
    state.patch = p.to_dict()

    if p.ok and p.confidence >= min_conf:
        parts.append((p.score, p.confidence))
        state.patch_used = True
        state.notes.append(
            f"패치 {p.score:+.2f} (기대 {p.expected_return_pct:+.1f}%, "
            f"이웃 {p.effective_n:.0f}개)")
    elif p.ok:
        state.notes.append(
            f"패치 확신 미달 ({p.confidence:.2f} < {min_conf:g}) — 미반영")

    # 3) 칼만 추세 (filterpy 이식) — 인과적 적중률로 심사합니다
    try:
        k = kalman.direction_score(bars, horizon_bars=h)
    except Exception as exc:
        k = kalman.KalmanResult(error=f"{type(exc).__name__}: {exc}")
    state.kalman = k.to_dict()

    if k.ok and k.hit_rate >= min_acc and k.n_eval >= 30:
        parts.append((k.score, max(0.0, k.hit_rate - 0.5) * 2))
        state.kalman_used = True
        state.notes.append(
            f"칼만 {k.score:+.2f} (적중 {k.hit_rate:.0%}, n={k.n_eval})")
    elif k.ok:
        state.notes.append(
            f"칼만 적중 미달 ({k.hit_rate:.0%} < {min_acc:.0%}) — 미반영")

    # 4) 검증 성적 가중 결합
    total_weight = sum(w for _, w in parts)
    if total_weight > 1e-9:
        state.score = sum(s * w for s, w in parts) / total_weight
        state.confidence = min(1.0, total_weight)
    else:
        state.score = None               # 전원 자격 미달 — 이 회전에서 ML 무력
    ok_any = (state.gbdt.get("ok") or state.lgbm.get("ok")
              or p.ok or k.ok)
    state.ok = bool(ok_any)
    if not state.ok:
        state.error = state.gbdt.get("error") or p.error or k.error
    return state


# ---------------------------------------------------------------------------
# 결합
# ---------------------------------------------------------------------------

def apply_to_score(base_score: float, state: MLState, cfg: dict) -> tuple[float, str]:
    """soft 결합 — nnfx.apply_to_score 와 같은 형태.

    결합 가중은 (설정 가중 × ML 확신도) 입니다. 확신이 낮은 날은 저절로
    존재감이 줄어듭니다.
    """
    if not state.usable:
        return base_score, ""
    weight = float(config_from(cfg)["weight"]) * float(state.confidence)
    weight = max(0.0, min(weight, 1.0))
    if weight <= 0:
        return base_score, ""
    blended = (1 - weight) * base_score + weight * float(state.score)
    return blended, f"ML {state.score:+.2f} (가중 {weight:.0%})"


def describe() -> list[dict]:
    return [
        {"mode": OFF, "label": "사용 안 함", "note": "ML 예측을 계산하지 않습니다."},
        {"mode": OBSERVE, "label": "관찰 (기본)",
         "note": "GBDT(XGBoost)·LGBM(LightGBM)·패치 어텐션(PatchTST)·칼만 추세"
                 "(filterpy) 네 예측기의 점수를 계산해 신호 로그에 남기지만 "
                 "매매 판단은 바꾸지 않습니다."},
        {"mode": SOFT, "label": "점수 반영",
         "note": f"검증을 통과한 ML 점수를 가중 {DEFAULTS['weight']:.0%}×확신도로 "
                 "기존 점수에 섞습니다. 검증 미달이면 그 회전에서는 반영되지 않습니다."},
    ]


def clear_cache():
    with _cache_lock:
        _cache.clear()


__all__ = ["compute", "MLState", "apply_to_score", "mode_of", "config_from",
           "describe", "clear_cache", "OFF", "OBSERVE", "SOFT", "MODES", "DEFAULTS"]
