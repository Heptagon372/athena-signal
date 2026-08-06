"""
백테스트 검증 (Backtest Validation)
-----------------------------------
"이 백테스트 성과가 진짜인가"를 묻는 통계 도구들입니다. 수익률을 계산하는
코드가 아니라, **계산된 수익률을 의심하는 코드**입니다.

수록 항목
    active_metrics          시장(사서 들고만 있기) 대비 초과수익 · Active Sharpe
    sharpe_ratio            연율화 샤프
    block_bootstrap         블록 부트스트랩 — 자기상관은 남기고 순서만 파괴
    falsification_audit     위조검증 — 알파가 없는 데이터에서도 성과가 나오는가
    deflated_sharpe_ratio   DSR — 여러 번 시도해서 고른 최고값을 깎아냄
    probabilistic_sharpe_ratio  PSR — 표본 수·왜도·첨도를 반영한 샤프 유의성

왜 이것들이 필요한가
    백테스트에서 좋은 숫자가 나오는 경로는 세 가지입니다.
        1) 전략에 진짜 우위가 있다
        2) 그냥 시장이 올랐다              → active_metrics 가 가려냅니다
        3) 설정을 여러 번 바꿔 최고를 골랐다 → deflated_sharpe_ratio 가 깎아냅니다
    2번과 3번을 걸러내지 않은 수익률은 "이 전략이 돈을 번다"의 근거가 못 됩니다.
    그리고 세 경로를 다 통과해도 마지막 질문이 남습니다 — **아무 우위가 없는
    데이터에 이 파이프라인을 통과시키면 무엇이 나오는가**(falsification_audit).
    거기서도 좋은 숫자가 나오면, 좋은 숫자를 만드는 것은 전략이 아니라 절차입니다.

주의: 모든 함수는 표본이 적으면 None 또는 낮은 신뢰도를 그대로 돌려줍니다.
      표본 부족을 좋은 결과로 포장하지 않는 것이 이 모듈의 존재 이유입니다.
"""

import math

import numpy as np

from engine.quant import _norm_cdf

TRADING_DAYS = 252

# 오일러-마스케로니 상수 — DSR의 기대 최대 샤프 계산에 쓰입니다
_EULER_GAMMA = 0.5772156649015329


def _clean(values) -> np.ndarray:
    """None·NaN을 걷어낸 float 배열.

    `values or []` 로 쓰면 numpy 배열이 들어왔을 때 진리값 판정에서 터집니다
    (요소가 2개 이상인 배열은 bool 로 못 바꿉니다). 그래서 None 검사를 명시합니다.
    """
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    return arr[np.isfinite(arr)] if len(arr) else arr


# ---------------------------------------------------------------------------
# 수익률 · 샤프
# ---------------------------------------------------------------------------

def to_returns(curve) -> np.ndarray:
    """자산곡선(list[dict] 또는 list[float])을 일간 수익률로."""
    if not curve:
        return np.array([])
    if isinstance(curve[0], dict):
        values = np.array([float(p.get("equity") or 0) for p in curve], dtype=float)
    else:
        values = np.asarray(curve, dtype=float)
    if len(values) < 2:
        return np.array([])
    prev = values[:-1]
    # 0으로 나누면 inf가 퍼져 전체 지표가 NaN이 됩니다 — 그 구간은 0으로 둡니다
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev > 0, (values[1:] - prev) / prev, 0.0)
    return np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)


def sharpe_ratio(returns, periods_per_year: int = TRADING_DAYS) -> float:
    """연율화 샤프. 표준편차가 0이면 0을 돌려줍니다(무한대 금지)."""
    rets = np.asarray(returns, dtype=float)
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(rets.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(curve) -> float:
    """최대 낙폭 (%, 음수)."""
    if not curve:
        return 0.0
    values = ([float(p.get("equity") or 0) for p in curve]
              if isinstance(curve[0], dict) else [float(v) for v in curve])
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, (v - peak) / peak * 100)
    return worst


def active_metrics(curve, benchmark_curve, periods_per_year: int = TRADING_DAYS) -> dict:
    """시장 대비 초과 성과.

    **이 프로젝트에서 가장 중요한 지표입니다.** 총수익률만 보면 상승장에서는
    아무 전략이나 좋아 보입니다. 같은 기간 그 종목을 그냥 사서 들고 있었을 때와
    비교해야, 매매가 실제로 무엇을 보탰는지 보입니다. 그 차이가 음수면 전략은
    수수료를 내며 시장을 따라간 것입니다.

    Active Sharpe = (전략 수익률 − 벤치마크 수익률)의 샤프.
    """
    strat = to_returns(curve)
    bench = to_returns(benchmark_curve)
    if len(strat) == 0:
        return {}

    n = min(len(strat), len(bench)) if len(bench) else 0
    out = {
        "sharpe": round(sharpe_ratio(strat, periods_per_year), 4),
        "volatility_pct": round(float(strat.std(ddof=1) * math.sqrt(periods_per_year) * 100), 2)
        if len(strat) > 1 else 0.0,
    }

    dd = max_drawdown(curve)
    total = float(np.prod(1 + strat) - 1) if len(strat) else 0.0
    years = len(strat) / periods_per_year if len(strat) else 0.0
    cagr = ((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else 0.0
    out["cagr_pct"] = round(cagr * 100, 2)
    # Calmar = 수익을 낙폭으로 나눈 값. 같은 수익이면 덜 흔들린 쪽이 좋습니다.
    out["calmar"] = round(cagr * 100 / abs(dd), 2) if dd < 0 else 0.0

    if n < 2:
        return out

    active = strat[:n] - bench[:n]
    bench_total = float(np.prod(1 + bench[:n]) - 1)
    out.update({
        "benchmark_return_pct": round(bench_total * 100, 2),
        "excess_return_pct": round((total - bench_total) * 100, 2),
        "active_sharpe": round(sharpe_ratio(active, periods_per_year), 4),
        "beat_benchmark": bool(total > bench_total),
    })
    return out


# ---------------------------------------------------------------------------
# 위조검증 — 알파가 없는 데이터에서도 성과가 나오는가
# ---------------------------------------------------------------------------

def block_bootstrap(returns, block: int = 5, rng=None) -> np.ndarray:
    """블록 부트스트랩 — 변동성 군집은 남기고 **예측 가능한 순서만** 파괴합니다.

    수익률을 한 개씩 섞으면(단순 셔플) 변동성 군집이 사라져서, 만들어진 가격이
    현실의 주가와 너무 달라집니다. 그 위에서 나온 귀무분포는 지나치게 관대합니다.
    길이 `block` 짜리 토막을 통째로 뽑아 이으면, 하루하루의 자기상관은 보존한 채
    "이 시점 다음에 무엇이 오는가"라는 구조만 무너집니다. 전략이 먹고 사는 것이
    바로 그 구조이므로, 이것이 올바른 귀무가설입니다.
    """
    rets = np.asarray(returns, dtype=float)
    n = len(rets)
    if n == 0:
        return rets
    rng = rng or np.random.default_rng()
    block = max(int(block), 1)
    out = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        # 순환 인덱싱 — 끝에서 잘리면 앞으로 돌아갑니다(꼬리 표본 손실 방지)
        out.extend(rets[(start + k) % n] for k in range(block))
    return np.array(out[:n], dtype=float)


def falsification_audit(observed: float, null_values, higher_is_better: bool = True,
                        label: str = "") -> dict:
    """관측값이 귀무분포 안에 묻히는가.

    경험적 p-value = (귀무값이 관측값 이상인 횟수 + 1) / (시행 + 1).
    +1은 보수적 보정입니다 — 시행 수가 적을 때 p=0이 나와 "완벽하게 유의하다"고
    읽히는 것을 막습니다. 20회 시행에서 나올 수 있는 최소 p는 0.048입니다.
    """
    nulls = _clean(null_values)
    if len(nulls) == 0:
        return {"passed": None, "p_value": None, "n_trials": 0,
                "reason": "귀무 표본이 없습니다 — 판정하지 않습니다."}

    hits = int(np.sum(nulls >= observed) if higher_is_better else np.sum(nulls <= observed))
    p = (hits + 1) / (len(nulls) + 1)
    return {
        "label": label,
        "observed": round(float(observed), 4),
        "null_mean": round(float(nulls.mean()), 4),
        "null_std": round(float(nulls.std(ddof=1)), 4) if len(nulls) > 1 else 0.0,
        "null_min": round(float(nulls.min()), 4),
        "null_max": round(float(nulls.max()), 4),
        "n_trials": len(nulls),
        "p_value": round(p, 4),
        "passed": bool(p <= 0.05),
    }


# ---------------------------------------------------------------------------
# 선택편향 보정 — 여러 번 시도해서 고른 최고값 깎아내기
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(observed_sr: float, n_obs: int,
                               benchmark_sr: float = 0.0,
                               skew: float = 0.0, kurtosis: float = 3.0) -> float:
    """PSR — 관측 샤프가 기준 샤프보다 진짜로 큰가 (Bailey & López de Prado).

    표본 수가 적을수록, 수익률이 왼쪽으로 치우칠수록(음의 왜도), 꼬리가 두꺼울수록
    같은 샤프라도 신뢰도가 낮아집니다. 금융 수익률은 셋 다 해당합니다.
    """
    if n_obs < 3:
        return 0.0
    denom = 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr ** 2
    if denom <= 0:
        return 0.0
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(_norm_cdf(z))


def expected_max_sharpe_under_null(sr_trials, n_trials: int = None) -> float:
    """아무 우위가 없어도, N번 시도해서 최고를 고르면 기대되는 샤프.

    시행이 늘수록 이 값이 커집니다. 즉 **아무것도 없는데 20번 돌려 최고를 고르면
    꽤 그럴듯한 샤프가 나옵니다.** 관측된 최고 샤프는 이 값을 넘어야 의미가 있습니다.
    """
    trials = _clean(sr_trials)
    n = int(n_trials or len(trials))
    if n < 2 or len(trials) < 2:
        return 0.0
    var = float(trials.var(ddof=1))
    if var <= 0:
        return 0.0
    # Bailey & López de Prado (2014) 식 — 극단값 분포 근사
    z1 = _norm_ppf(1 - 1.0 / n)
    z2 = _norm_ppf(1 - 1.0 / (n * math.e))
    return float(math.sqrt(var) * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(observed_sr: float, sr_trials, n_obs: int,
                          skew: float = 0.0, kurtosis: float = 3.0,
                          n_trials: int = None, annualized: bool = True,
                          periods_per_year: int = TRADING_DAYS) -> dict:
    """DSR — 시행 횟수를 반영해 깎아낸 샤프의 유의확률.

    쓰는 법: 설정을 20번 바꿔가며 백테스트하고 그중 최고를 채택했다면,
    `sr_trials` 에 20개 샤프를 전부 넣습니다. 최고값만 넣으면 보정이 되지 않습니다.

    **단위 주의**: PSR 공식은 `n_obs` 와 **같은 주기의 샤프**를 받습니다(일간
    관측이면 일간 샤프). 이 모듈의 sharpe_ratio() 는 연율화된 값을 주므로,
    기본값 `annualized=True` 로 두면 여기서 √주기로 나눠 맞춥니다. 이 변환을
    빼먹으면 z가 √252배 부풀어 DSR이 무조건 1.0으로 포화됩니다 — 즉 어떤
    성과든 "유의함"으로 나옵니다.

    판정: DSR > 0.95 여야 "이 성과는 운으로 설명되지 않는다"고 말할 수 있습니다.
    """
    trials = _clean(sr_trials)
    n = int(n_trials or len(trials))
    if n_obs < 3 or n < 2:
        return {"dsr": None, "n_trials": n, "n_obs": n_obs,
                "reason": "시행이 2회 미만이거나 표본이 부족해 보정할 수 없습니다."}

    scale = math.sqrt(periods_per_year) if annualized else 1.0
    sr_star = expected_max_sharpe_under_null(trials / scale, n)
    dsr = probabilistic_sharpe_ratio(observed_sr / scale, n_obs,
                                     benchmark_sr=sr_star,
                                     skew=skew, kurtosis=kurtosis)
    return {
        "dsr": round(dsr, 4),
        "observed_sr": round(float(observed_sr), 4),
        # 화면에는 입력과 같은 단위(연율)로 되돌려 보여줍니다
        "expected_max_sr_under_null": round(sr_star * scale, 4),
        "n_trials": n,
        "n_obs": n_obs,
        "significant_at_95": bool(dsr > 0.95),
    }


def _norm_ppf(p: float) -> float:
    """표준정규 분위수 (Acklam 근사). scipy 없이 씁니다."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ---------------------------------------------------------------------------
# 전략 건강도 — 점추정이 아니라 t통계로 판단
# ---------------------------------------------------------------------------

HEALTH_NORMAL, HEALTH_CAUTION, HEALTH_SUSPEND = "normal", "caution", "suspend"
HEALTH_LABELS = {HEALTH_NORMAL: "정상", HEALTH_CAUTION: "주의", HEALTH_SUSPEND: "중단"}


def health_from_tstat(values, min_obs: int = 20,
                      t_caution: float = -1.0, t_suspend: float = -2.0) -> dict:
    """최근 실적이 통계적으로 유의하게 나쁜가.

    **점추정을 임계값과 직접 비교하면 안 됩니다.** 예를 들어 "최근 평균 손익이
    마이너스면 중단"은 표본이 20건일 때 거의 항상 발동합니다 — 손익의 표준편차가
    평균보다 훨씬 크기 때문입니다. 그러면 가드가 신호가 아니라 노이즈를 읽고,
    멀쩡한 전략을 계속 껐다 켰다 합니다.

    그래서 평균이 아니라 **t = 평균 / (표준편차/√n)** 으로 판단하고, 표본이
    `min_obs` 에 못 미치면 아예 판정하지 않습니다(워밍업).
    """
    vals = _clean(values)
    if len(vals) < max(int(min_obs), 3):
        return {"state": HEALTH_NORMAL, "label": HEALTH_LABELS[HEALTH_NORMAL],
                "t_stat": None, "n": len(vals), "warming_up": True,
                "reason": f"표본 {len(vals)}건 — {min_obs}건이 모일 때까지 판정하지 않습니다."}

    mean, std = float(vals.mean()), float(vals.std(ddof=1))
    if std <= 1e-12:
        t = 0.0
    else:
        t = mean / (std / math.sqrt(len(vals)))

    if t <= t_suspend:
        state = HEALTH_SUSPEND
        reason = f"최근 {len(vals)}건 성과가 유의하게 나쁩니다 (t={t:.2f} ≤ {t_suspend:g})"
    elif t <= t_caution:
        state = HEALTH_CAUTION
        reason = f"최근 {len(vals)}건 성과가 나빠지는 중입니다 (t={t:.2f} ≤ {t_caution:g})"
    else:
        state = HEALTH_NORMAL
        reason = f"최근 {len(vals)}건 기준 이상 징후 없음 (t={t:.2f})"

    return {"state": state, "label": HEALTH_LABELS[state], "t_stat": round(t, 3),
            "mean": round(mean, 2), "n": len(vals), "warming_up": False,
            "reason": reason}
