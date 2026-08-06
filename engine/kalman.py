"""
칼만 추세 필터 (filterpy 이식)
-----------------------------
rlabbe/filterpy 의 KalmanFilter (predict/update 방정식)와, 같은 저자의 교재
rlabbe/Kalman-and-Bayesian-Filters-in-Python 14장(Adaptive Filtering)의
적응형 프로세스 노이즈를 로그가격 추세 추정에 이식한 것입니다.

모델 — 국소 선형 추세 (local linear trend)
    상태  x = [수준(level), 기울기(slope)]     — 로그가격과 봉당 로그수익률
    전이  F = [[1, 1], [0, 1]]                 — 수준은 기울기만큼 이동
    관측  H = [1, 0]                           — 종가(로그)만 관측
    Q     filterpy 의 Q_discrete_white_noise(dim=2) 와 같은
          q·[[¼, ½], [½, 1]]  (dt=1)

이동평균과 무엇이 다른가
    이동평균의 기울기는 "며칠 평균이 며칠 전보다 높다"는 사후 확인이고 지연이
    창 길이에 비례합니다. 칼만 필터는 관측 잡음(R)과 과정 잡음(Q)의 비율로
    지연·민감도를 통제하고, **기울기의 불확실성(P)** 을 함께 줍니다. 그래서
    "기울기가 +다"가 아니라 "기울기가 표준편차의 몇 배로 +다"(SNR)를 물을 수
    있습니다 — 잡음 속에서 방향을 주장하지 않게 만드는 장치입니다.

적응형 Q (교재 14장, Adjustable Process Noise)
    ε = y²/S (정규화 혁신 제곱)가 한계를 넘으면 Q 를 배율로 키우고, 잠잠해지면
    되돌립니다. 급등락·국면 전환에서 필터가 "옛 추세를 고집"하는 지연을
    줄입니다. 평시에는 Q 가 작아 부드럽고, 사건 뒤에는 커져 빨리 따라갑니다.

이 예측기의 정직성 장치
    과거 전 구간을 **인과적으로**(그 시점까지의 데이터만으로) 필터링하면서
    "그때의 기울기 부호가 그 뒤 h봉 방향을 맞혔는가"를 세어 적중률을 냅니다.
    engine/mlsignal.py 는 이 적중률이 기준 미달이면 점수를 쓰지 않습니다 —
    GBDT 의 검증 적중률과 같은 자격 심사입니다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULTS = {
    "q_ratio": 0.01,        # 과정 잡음 / 관측 잡음 비 — 클수록 민감, 작을수록 부드러움
    "eps_max": 4.0,         # ε 한계 (≈2σ). 교재 14장의 임계값 방식
    "q_scale_step": 2.0,    # 한계 초과 시 Q 배율 증감 폭
    "q_scale_cap": 100.0,   # 배율 상한 — 무한히 커지면 필터가 관측 추종기가 됩니다
    "warmup": 30,           # 이 구간의 기울기는 채점·판단에 쓰지 않습니다
    "snr_scale": 2.0,       # score = tanh(SNR / snr_scale)
}

MIN_BARS = 60


@dataclass
class KalmanResult:
    ok: bool = False
    score: float = 0.0            # tanh(기울기 SNR) ∈ [−1, +1]
    slope: float = 0.0            # 봉당 로그수익률 추정
    slope_std: float = 0.0
    snr: float = 0.0              # slope / slope_std
    level: float = 0.0            # 필터가 본 '적정' 로그가격
    stretch: float = 0.0          # (관측 − 수준) / √S — 수준 대비 이격 (혁신 단위)
    hit_rate: float = 0.5         # 인과적 과거 채점: 기울기 부호 → h봉 뒤 방향
    n_eval: int = 0
    adapted: int = 0              # 적응형 Q 가 개입한 횟수
    horizon: int = 0
    error: str = ""

    @property
    def edge(self) -> float:
        return self.hit_rate - 0.5

    def to_dict(self) -> dict:
        return {"ok": self.ok, "score": round(self.score, 4),
                "slope": round(self.slope, 6), "slope_std": round(self.slope_std, 6),
                "snr": round(self.snr, 3), "stretch": round(self.stretch, 3),
                "hit_rate": round(self.hit_rate, 4), "edge": round(self.edge, 4),
                "n_eval": self.n_eval, "adapted": self.adapted,
                "horizon": self.horizon, "error": self.error}


def config_from(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        value = (cfg or {}).get(f"kalman_{key}")
        if value is not None:
            out[key] = value
    return out


def direction_score(bars: pd.DataFrame, horizon_bars: int = 5,
                    cfg: dict = None) -> KalmanResult:
    """일봉 → 추세 방향 점수 + 인과적 적중률."""
    params = config_from(cfg or {})
    h = max(1, int(horizon_bars))
    result = KalmanResult(horizon=h)

    if bars is None or "close" not in getattr(bars, "columns", []):
        result.error = "봉 데이터가 없습니다."
        return result
    close = bars["close"].astype(float).to_numpy()
    close = close[np.isfinite(close) & (close > 0)]
    if len(close) < MIN_BARS:
        result.error = f"봉이 부족합니다 ({len(close)}/{MIN_BARS})"
        return result

    z = np.log(close)
    n = len(z)

    # 관측 잡음 R — 1봉 수익률의 로버스트 분산(MAD 기반)으로 잡습니다.
    # 표본분산을 쓰면 급등락 며칠이 R 을 키워 필터 전체가 무뎌집니다.
    diffs = np.diff(z)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    r_var = max((1.4826 * mad) ** 2, 1e-10)
    q_var = r_var * float(params["q_ratio"])

    # filterpy Q_discrete_white_noise(dim=2, dt=1) 의 형태
    q_base = q_var * np.array([[0.25, 0.5], [0.5, 1.0]])
    fm = np.array([[1.0, 1.0], [0.0, 1.0]])          # F

    # 초기 상태 — 첫 관측과 초반 평균 기울기, 불확실성은 크게
    x = np.array([z[0], float(np.mean(diffs[:10]))])
    pm = np.array([[r_var * 10, 0.0], [0.0, r_var]])

    eps_max = float(params["eps_max"])
    step = float(params["q_scale_step"])
    cap = float(params["q_scale_cap"])
    warmup = int(params["warmup"])
    q_scale = 1.0
    adapted = 0

    slopes = np.zeros(n)
    slope_vars = np.zeros(n)
    levels = np.zeros(n)
    innovations = np.zeros(n)
    s_vars = np.full(n, r_var)

    slopes[0], slope_vars[0], levels[0] = x[1], pm[1, 1], x[0]

    for t in range(1, n):
        # predict (filterpy KalmanFilter.predict)
        x = fm @ x
        pm = fm @ pm @ fm.T + q_base * q_scale

        # update (filterpy KalmanFilter.update) — 관측이 스칼라라 역행렬이 나눗셈
        y = z[t] - x[0]                      # 혁신
        s = pm[0, 0] + r_var                 # 혁신 분산
        k_gain = pm[:, 0] / s                # 칼만 이득
        x = x + k_gain * y
        pm = pm - np.outer(k_gain, pm[0, :])

        # 적응형 Q (교재 14장): ε = y²/S 가 크면 과정 잡음을 키워 빨리 따라감
        eps = y * y / s
        if eps > eps_max:
            if q_scale * step <= cap:
                q_scale *= step
                adapted += 1
        elif eps < 1.0 and q_scale > 1.0:
            q_scale = max(1.0, q_scale / step)

        slopes[t], slope_vars[t], levels[t] = x[1], max(pm[1, 1], 1e-16), x[0]
        innovations[t], s_vars[t] = y, s

    # 인과적 채점 — 시점 t 의 기울기(그때까지 데이터만 사용)로 t+h 방향을 맞혔는가
    hits = total = 0
    for t in range(warmup, n - h):
        move = z[t + h] - z[t]
        if abs(move) < 1e-12:
            continue
        total += 1
        hits += int(np.sign(slopes[t]) == np.sign(move))
    result.hit_rate = hits / total if total else 0.5
    result.n_eval = total

    slope_std = float(np.sqrt(slope_vars[-1]))
    result.slope = float(slopes[-1])
    result.slope_std = slope_std
    result.snr = result.slope / slope_std if slope_std > 0 else 0.0
    result.score = float(np.tanh(result.snr / float(params["snr_scale"])))
    result.level = float(levels[-1])
    result.stretch = float(innovations[-1] / np.sqrt(s_vars[-1])) \
        if s_vars[-1] > 0 else 0.0
    result.adapted = adapted
    result.ok = True
    return result


__all__ = ["direction_score", "KalmanResult", "config_from", "DEFAULTS", "MIN_BARS"]
