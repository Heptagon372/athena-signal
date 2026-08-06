"""
패치 어텐션 예측기 (PatchTST 이식)
---------------------------------
Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
ICLR 2023. (PatchTST/PatchTST 저장소)

원본에서 가져온 세 가지 구조적 장치
    1) RevIN 인스턴스 정규화   창마다 (x − μ) / σ 로 정규화하고, 예측을 다시
       역정규화합니다 (layers/RevIN.py). 분포가 계속 이동하는 금융 시계열에서
       "작년의 저변동 구간"과 "지금의 고변동 구간"을 같은 자로 비교하게 해줍니다.
    2) 패칭                    시계열을 patch_len 크기 조각으로 잘라 토큰으로
       씁니다 (backbone 의 unfold(size=patch_len, step=stride)). 점 하나가
       아니라 **국소 모양**(패치)이 비교 단위가 됩니다.
    3) 채널 독립               변수(종목)마다 따로 처리합니다. 여기서는 종목당
       하나의 수익률 채널만 쓰므로 자연히 성립합니다.

**무엇을 학습하지 않는가 — 정직하게**
    원본은 학습된 선형 투영(W_P)과 멀티헤드 어텐션을 GPU 로 훈련합니다.
    torch 를 이 프로젝트에 들여오면 exe 가 기가바이트 단위가 되고, 일봉
    수백 개로 트랜스포머를 훈련하는 것은 과적합 외의 결과를 주지 않습니다.

    대신 **학습 없는 어텐션**을 씁니다. 투영 없이 정규화된 패치 벡터로
    스케일드 닷프로덕트 어텐션(softmax(q·kᵀ/√d))을 계산하면, 이는 수학적으로
    Nadaraya-Watson 커널 회귀와 같습니다 — "지금과 모양이 닮은 과거 구간들이
    그 뒤에 어떻게 움직였는가"의 유사도 가중 평균입니다. PatchTST 의 표현
    (RevIN + 패치 + 어텐션)은 유지하고, 학습되는 부분만 커널로 대체한 것입니다.

키(key)가 되는 과거 창은 값(value)인 '그 뒤 h봉 수익률'까지 **전부 지나간
구간**에서만 뽑습니다. 마지막 키의 값도 이미 실현된 과거이므로 미래 정보가
섞일 수 없습니다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULTS = {
    "context": 64,          # 질의 창 길이 (원본 336 은 장기 예측용 — 일봉 180개
                            # 환경에서는 키가 부족해져 64 로 줄였습니다)
    "patch_len": 8,         # 원본 기본 16/스트라이드 8 과 같은 ½ 겹침 비율
    "stride": 4,
    "horizon": 5,           # 몇 봉 뒤를 예측하는가
    "temperature": 1.0,     # softmax 온도 — 낮을수록 최근접 이웃에 가까움
    "top_k": 24,            # 상위 k개 키만 사용 (먼 유사도의 잡음 평균화 방지)
    "eps": 1e-5,            # RevIN 원본과 같은 수치 안정 상수
}

MIN_KEYS = 30               # 비교할 과거 창이 이보다 적으면 예측하지 않습니다


@dataclass
class PatchForecast:
    ok: bool = False
    score: float = 0.0                # tanh 정규화 방향 점수 ∈ [−1, +1]
    expected_return_pct: float = 0.0  # h봉 기대수익률 (역정규화 후)
    confidence: float = 0.0           # 0~1 — 유효 이웃 수 × 방향 합의
    agreement: float = 0.0            # 이웃들의 방향 합의도 |Σw·sign(v)|
    effective_n: float = 0.0          # 1/Σw² — 몇 개의 이웃이 실질 기여했나
    keys: int = 0
    horizon: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "score": round(self.score, 4),
                "expected_return_pct": round(self.expected_return_pct, 4),
                "confidence": round(self.confidence, 3),
                "agreement": round(self.agreement, 3),
                "effective_n": round(self.effective_n, 1),
                "keys": self.keys, "horizon": self.horizon, "error": self.error}


def config_from(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        value = (cfg or {}).get(f"patch_{key}")
        if value is not None:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# RevIN + 패칭 (원본 layers/RevIN.py · backbone unfold 대응)
# ---------------------------------------------------------------------------

def _revin_embed(windows: np.ndarray, patch_len: int, stride: int,
                 eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """창들 → (패치 임베딩, 창 평균 μ, 창 표준편차 σ).

    RevIN: 창마다 자기 통계로 정규화합니다. 역정규화에 쓰도록 μ·σ 를 돌려줍니다.
    패칭: 정규화된 창을 unfold 방식(size=patch_len, step=stride)으로 잘라
          이어붙입니다. patch_num = (context − patch_len)/stride + 1.
    """
    mu = windows.mean(axis=1, keepdims=True)
    sigma = np.sqrt(windows.var(axis=1, keepdims=True) + eps)   # 원본과 같은 분모
    normed = (windows - mu) / sigma

    context = windows.shape[1]
    starts = np.arange(0, context - patch_len + 1, stride)
    patches = np.stack([normed[:, s:s + patch_len] for s in starts], axis=1)
    embed = patches.reshape(len(windows), -1)                    # 패치 이어붙임
    return embed, mu[:, 0], sigma[:, 0]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# 예측
# ---------------------------------------------------------------------------

def forecast(bars: pd.DataFrame, cfg: dict = None) -> PatchForecast:
    """일봉 → 다음 h봉 수익률 예측 + 방향 점수."""
    params = config_from(cfg or {})
    context = int(params["context"])
    patch_len = int(params["patch_len"])
    stride = max(1, int(params["stride"]))
    h = max(1, int(params["horizon"]))
    result = PatchForecast(horizon=h)

    if bars is None or "close" not in getattr(bars, "columns", []):
        result.error = "봉 데이터가 없습니다."
        return result
    close = bars["close"].astype(float).to_numpy()
    close = close[np.isfinite(close) & (close > 0)]
    r = np.diff(np.log(close))
    n = len(r)

    need = context + h + MIN_KEYS
    if n < need or context <= patch_len:
        result.error = f"봉이 부족합니다 ({n}/{need})"
        return result

    # 창 구성: 끝 t 의 창 = r[t-context+1 .. t]
    # 키 창의 끝은 (n-1-h) 까지 — 값(그 뒤 h봉 수익률)이 이미 실현된 구간만.
    all_windows = np.lib.stride_tricks.sliding_window_view(r, context)
    # all_windows[i] 는 r[i .. i+context-1], 창 끝 t = i + context - 1
    last_key_end = n - 1 - h
    key_rows = np.arange(0, last_key_end - context + 2)
    if len(key_rows) < MIN_KEYS:
        result.error = f"비교할 과거 창이 부족합니다 ({len(key_rows)}/{MIN_KEYS})"
        return result

    embed, mu, sigma = _revin_embed(
        np.vstack([all_windows[key_rows], all_windows[-1:]]),
        patch_len, stride, float(params["eps"]))
    keys_e, query_e = embed[:-1], embed[-1]
    mu_q, sigma_q = float(mu[-1]), float(sigma[-1])

    # 값 = 키 창 끝 이후 h봉의 평균 수익률, 그 창의 RevIN 통계로 정규화
    ends = key_rows + context - 1
    cum = np.concatenate([[0.0], np.cumsum(r)])
    v_raw = (cum[ends + 1 + h] - cum[ends + 1]) / h
    v_norm = (v_raw - mu[:-1]) / sigma[:-1]

    # 스케일드 닷프로덕트 어텐션 (softmax(q·kᵀ/√d)) — 학습 없는 커널 회귀
    d = len(query_e)
    sim = keys_e @ query_e / (np.sqrt(d) * float(params["temperature"]))

    top_k = int(params["top_k"])
    if 0 < top_k < len(sim):
        keep = np.argpartition(sim, -top_k)[-top_k:]
    else:
        keep = np.arange(len(sim))
    w = _softmax(sim[keep])
    v_kept = v_norm[keep]

    pred_norm = float(w @ v_kept)
    pred_per_bar = pred_norm * sigma_q + mu_q          # RevIN 역정규화
    total_log_ret = pred_per_bar * h
    result.expected_return_pct = float(np.expm1(total_log_ret) * 100)

    # 방향 점수: 예측 수익률을 그 시평선의 기대 변동성으로 나눠 tanh
    # (σ_q 는 1봉 수익률 표준편차 → h봉이면 √h 배)
    scale = sigma_q * np.sqrt(h)
    result.score = float(np.tanh(total_log_ret / scale)) if scale > 0 else 0.0

    # 확신도: 실질 이웃 수(1/Σw²)와 이웃들의 방향 합의
    eff_n = float(1.0 / np.sum(w * w))
    agreement = float(abs(w @ np.sign(v_kept)))
    result.effective_n = eff_n
    result.agreement = agreement
    result.confidence = float(min(1.0, eff_n / 8.0) * agreement)
    result.keys = int(len(keep))
    result.ok = True
    return result


__all__ = ["forecast", "PatchForecast", "config_from", "DEFAULTS", "MIN_KEYS"]
