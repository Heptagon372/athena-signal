"""
GARCH(1,1) 변동성 예측 (arch 이식)
---------------------------------
bashtage/arch 의 GARCH 분산 재귀와 초기화 방법을 numpy 로 옮긴 것입니다.

    Bollerslev, T. (1986). "Generalized Autoregressive Conditional
    Heteroskedasticity." Journal of Econometrics, 31(3).

무엇을 주는가 — **앞을 보는** 변동성
    지금까지 쓰던 Yang-Zhang 비율(engine/ensemble.volatility_factor)은 실현된
    변동성의 사후 비교입니다. GARCH 는 변동성 군집("큰 움직임 뒤에는 큰
    움직임")을 모형으로 두고 **다음 봉의 분산을 예측**합니다.

        σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}     (arch recursions_python.py 그대로)

    α+β(지속성)가 1 에 가까울수록 충격이 오래 남습니다. k봉 뒤 분산은
    무조건부 분산으로 (α+β)^k 속도로 되돌아가므로, 여러 날짜의 예측을 닫힌
    식으로 낼 수 있습니다.

원본에서 가져온 것
    분산 재귀            recursions_python.garch_recursion 의 σ² 갱신식
    backcast 초기화      지수가중(λ=0.94) 제곱수익률 평균으로 σ²₀ 를 잡음
                         (VolatilityProcess.backcast 와 같은 방법)
    가우시안 로그우도    −½[log 2π + log σ²_t + ε²_t/σ²_t]

원본과 다른 것 — 최적화 방법
    arch 는 scipy SLSQP 로 (ω, α, β) 를 최적화합니다. 이 프로젝트는 scipy 를
    쓰지 않으므로 **분산 타게팅 + 격자 탐색 + 국소 정밀화**로 바꿨습니다.
        · 분산 타게팅: ω = σ̄²(1−α−β) 로 고정해 미지수를 (α, β) 둘로 줄임
          (표본 분산이 무조건부 분산과 일치하도록 — 널리 쓰이는 표준 기법)
        · (α, β) 격자를 전 후보 동시에 벡터화해 우도 계산 → 최고점 주변을
          두 번 좁혀 정밀화. 결정적이고(seed 불필요), scipy 수렴 실패 같은
          비결정 경로가 없습니다.

자격 심사 — 우도비 검정
    상수 분산(백색) 모형 대비 우도가 유의하게 좋아야 '수렴'으로 인정합니다
    (LR = 2(llf − llf₀) > 6, χ²(2) 5% 임계 근처). 변동성 군집이 없는 데이터에
    GARCH 를 강제로 끼우면 잡음에 맞춘 파라미터가 나오기 때문입니다.
"""

import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULTS = {
    "min_bars": 100,          # GARCH 는 짧은 표본에서 불안정합니다
    "lr_min": 6.0,            # 상수분산 대비 우도비 최소값 (수렴 인정 기준)
    "persistence_max": 0.995, # α+β 상한 — 1 을 넘으면 분산이 발산합니다
}

_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 256


@dataclass
class GarchResult:
    ok: bool = False              # 계산 자체가 됐는가
    converged: bool = False       # 자격 심사(우도비)까지 통과했는가
    omega: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    persistence: float = 0.0      # α+β
    half_life: float = 0.0        # 변동성 충격의 반감기 (봉)
    sigma_now_pct: float = 0.0    # 조건부 변동성 (현재, 봉당 %)
    sigma_next_pct: float = 0.0   # 1봉 뒤 예측 (봉당 %)
    sigma_uncond_pct: float = 0.0 # 무조건부 (장기 평균, 봉당 %)
    vol_ratio: float = 1.0        # 예측 / 무조건부 — 1보다 크면 '앞으로 시끄러움'
    loglik: float = 0.0
    lr_stat: float = 0.0          # 상수분산 대비 우도비
    samples: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "converged": self.converged,
                "alpha": round(self.alpha, 4), "beta": round(self.beta, 4),
                "persistence": round(self.persistence, 4),
                "half_life": round(self.half_life, 1),
                "sigma_now_pct": round(self.sigma_now_pct, 3),
                "sigma_next_pct": round(self.sigma_next_pct, 3),
                "sigma_uncond_pct": round(self.sigma_uncond_pct, 3),
                "vol_ratio": round(self.vol_ratio, 3),
                "lr_stat": round(self.lr_stat, 2),
                "samples": self.samples, "error": self.error}


# ---------------------------------------------------------------------------
# 적합
# ---------------------------------------------------------------------------

def fit(returns, cfg: dict = None) -> GarchResult:
    """로그수익률 → GARCH(1,1). 실패해도 예외 대신 error 를 채워 돌려줍니다."""
    params = {**DEFAULTS, **{k[6:]: v for k, v in (cfg or {}).items()
                             if k.startswith("garch_")}}
    result = GarchResult()

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < int(params["min_bars"]):
        result.error = f"표본 부족 ({len(r)}/{params['min_bars']})"
        return result

    r = r - r.mean()                       # arch 의 상수 평균 모형과 동일
    r2 = r * r
    sample_var = float(r2.mean())
    if sample_var <= 0:
        result.error = "분산이 0 입니다 (무변동 시계열)."
        return result
    result.samples = len(r)

    # backcast (arch VolatilityProcess.backcast): λ=0.94 지수가중 제곱수익률
    tau = min(75, len(r))
    w = 0.94 ** np.arange(tau)
    w /= w.sum()
    backcast = float(w @ r2[:tau])

    # 1) 격자 탐색 — 전 후보를 동시에 벡터화
    alphas = np.linspace(0.02, 0.30, 8)
    betas = np.linspace(0.50, 0.97, 10)
    best = _grid_best(r2, sample_var, backcast, alphas, betas,
                      float(params["persistence_max"]))
    if best is None:
        result.error = "유효한 (α, β) 후보가 없습니다."
        return result

    # 2) 국소 정밀화 — 최고점 주변을 두 번 좁힘
    for span in (0.04, 0.012):
        a0, b0 = best[1], best[2]
        alphas = np.clip(np.linspace(a0 - span, a0 + span, 7), 1e-4, 0.5)
        betas = np.clip(np.linspace(b0 - span * 2, b0 + span * 2, 7), 0.0, 0.999)
        refined = _grid_best(r2, sample_var, backcast, alphas, betas,
                             float(params["persistence_max"]))
        if refined is not None and refined[0] > best[0]:
            best = refined

    loglik, alpha, beta = best
    persistence = alpha + beta
    omega = sample_var * (1 - persistence)          # 분산 타게팅

    # 상수분산 모형의 우도 — 자격 심사 기준선
    ll0 = float(-0.5 * (np.log(2 * np.pi) + np.log(sample_var)
                        + r2 / sample_var).sum())
    lr = 2 * (loglik - ll0)

    # 조건부 분산 경로 재구성 (최종 파라미터로 한 번 더)
    sigma2 = _recursion(r2, omega, alpha, beta, backcast)
    sigma2_now = float(sigma2[-1])
    sigma2_next = omega + alpha * float(r2[-1]) + beta * sigma2_now

    result.ok = True
    result.omega, result.alpha, result.beta = float(omega), float(alpha), float(beta)
    result.persistence = float(persistence)
    result.half_life = float(np.log(0.5) / np.log(persistence)) \
        if 0 < persistence < 1 else 0.0
    result.sigma_now_pct = float(np.sqrt(sigma2_now) * 100)
    result.sigma_next_pct = float(np.sqrt(sigma2_next) * 100)
    result.sigma_uncond_pct = float(np.sqrt(sample_var) * 100)
    result.vol_ratio = float(np.sqrt(sigma2_next / sample_var))
    result.loglik, result.lr_stat = float(loglik), float(lr)
    result.converged = bool(
        lr >= float(params["lr_min"])
        and 0 < alpha and persistence < float(params["persistence_max"]))
    if not result.converged and not result.error:
        result.error = (f"변동성 군집 증거 부족 (LR={lr:.1f} < {params['lr_min']:g})"
                        if lr < float(params["lr_min"]) else "파라미터 경계 도달")
    return result


def _grid_best(r2, sample_var, backcast, alphas, betas, persistence_max):
    """(α, β) 격자 전체의 로그우도를 벡터화로 계산해 최고를 돌려줍니다."""
    aa, bb = np.meshgrid(alphas, betas)
    aa, bb = aa.ravel(), bb.ravel()
    keep = (aa + bb) < persistence_max
    aa, bb = aa[keep], bb[keep]
    if len(aa) == 0:
        return None

    omega = sample_var * (1 - aa - bb)              # 분산 타게팅
    sigma2 = np.full(len(aa), backcast)
    loglik = np.zeros(len(aa))
    log2pi = np.log(2 * np.pi)

    # arch garch_recursion 의 시간 루프 — 후보 축은 numpy 로 병렬
    for t in range(len(r2)):
        loglik += -0.5 * (log2pi + np.log(sigma2) + r2[t] / sigma2)
        sigma2 = omega + aa * r2[t] + bb * sigma2
        np.clip(sigma2, sample_var * 1e-4, sample_var * 1e4, out=sigma2)

    k = int(np.argmax(loglik))
    if not np.isfinite(loglik[k]):
        return None
    return float(loglik[k]), float(aa[k]), float(bb[k])


def _recursion(r2, omega, alpha, beta, backcast):
    """최종 파라미터의 조건부 분산 경로.

    σ²₀ = backcast 로 시작해 σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1} 를 굴립니다 —
    격자 탐색(_grid_best)의 우도 계산과 같은 경로여야 최고점 파라미터와
    여기서 낸 σ² 가 서로 맞습니다.
    """
    sigma2 = np.empty(len(r2))
    sigma2[0] = backcast
    for t in range(1, len(r2)):
        sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
    return sigma2


# ---------------------------------------------------------------------------
# 예측
# ---------------------------------------------------------------------------

def forecast_vol_pct(result: GarchResult, horizon: int = 5) -> float | None:
    """앞으로 h봉의 **평균** 봉당 변동성(%) — 배리어 스케일에 쓰는 값.

    k봉 뒤 분산은 무조건부로 기하급수 회귀합니다:
        σ²_{t+k} = σ̄² + (α+β)^{k−1} (σ²_{t+1} − σ̄²)
    """
    if not result.ok:
        return None
    h = max(1, int(horizon))
    uncond = (result.sigma_uncond_pct / 100) ** 2
    next1 = (result.sigma_next_pct / 100) ** 2
    p = result.persistence
    total = sum(uncond + (p ** (k - 1)) * (next1 - uncond) for k in range(1, h + 1))
    return float(np.sqrt(total / h) * 100)


# ---------------------------------------------------------------------------
# 캐시 진입점 — 봉이 갱신될 때만 다시 적합합니다
# ---------------------------------------------------------------------------

def fit_bars(bars: pd.DataFrame, cfg: dict = None) -> GarchResult:
    """종가 DataFrame 용 편의 함수 + 캐시."""
    empty = GarchResult(error="봉 데이터가 없습니다.")
    if bars is None or "close" not in getattr(bars, "columns", []):
        return empty
    close = bars["close"].astype(float).to_numpy()
    close = close[np.isfinite(close) & (close > 0)]
    if len(close) < 3:
        return empty

    key = (len(close), round(float(close[-1]), 6), round(float(close[0]), 6))
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit

    result = fit(np.diff(np.log(close)), cfg)
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = result
    return result


def clear_cache():
    with _cache_lock:
        _cache.clear()


__all__ = ["fit", "fit_bars", "forecast_vol_pct", "GarchResult", "clear_cache",
           "DEFAULTS"]
