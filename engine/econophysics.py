"""
경제물리학 추정량 (Econophysics Estimators)
-------------------------------------------
`engine/quant.py` 가 계량금융 추정량을 담는다면, 이 모듈은 **물리학 문헌에서
금융으로 넘어온** 추정량을 담습니다. numpy 만으로 구현했고, 각 함수는 원논문과
계산식을 주석에 남깁니다.

수록 항목
    dfa                 Peng et al. (1994) 추세제거 변동성 분석 — 추세에 강건한 장기기억
    modified_rs         Lo (1991) 수정 R/S — 단기 자기상관을 제거한 장기기억 검정
    hill_tail           Hill (1975) 꼬리지수 — **한쪽 꼬리만**, 표준오차·Hill plot 포함
    marchenko_pastur    Marchenko & Pastur (1967) 잡음 고유값 경계
    rmt_decompose       Laloux et al. (1999), Plerou et al. (2002) 상관행렬 분해·정화
    ledoit_wolf         Ledoit & Wolf (2004) 수축 공분산 — 소표본 RMT 대안
    chow_denning        Chow & Denning (1993) 다중 분산비율 동시검정
    sqrt_impact_cap     Tóth et al. (2011) 제곱근 시장충격 — 참여율 상한

기존 engine/quant.py 와의 관계
    quant.hurst_exponent 는 고전 R/S 입니다. 추세가 있으면 H를 과대추정하고,
    단기 자기상관과 장기기억을 구분하지 못합니다. 이 모듈의 dfa() 와
    modified_rs() 가 각각 그 두 문제를 겨냥합니다. 셋을 **대체가 아니라 병렬로**
    쓰고 서로 어긋날 때는 확신을 낮추는 것이 의도된 사용법입니다.

설계 원칙 — 점추정을 임계값과 직접 비교하지 않습니다
    engine/validation.py 가 전략 성과에 대해 세운 원칙을 추정량에도 적용합니다.
    모든 주요 추정량은 **표준오차를 함께** 돌려주고, 호출부는 t통계
    (추정치 − 귀무값)/SE 로 판단해야 합니다. 표본이 부족하면 None 입니다.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# DFA — Detrended Fluctuation Analysis
# ---------------------------------------------------------------------------


def dfa(series, min_scale: int = 8, max_scale: int = None,
        order: int = 2, n_scales: int = 12) -> dict | None:
    """추세제거 변동성 분석으로 스케일링 지수 α 를 추정합니다.

    Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., &
    Goldberger, A. L. (1994). "Mosaic organization of DNA nucleotides."
    Physical Review E, 49(2), 1685-1689.

    절차
        1) 프로파일  Y(i) = Σ_{k≤i} (x_k − x̄)              ... 누적편차
        2) 길이 s 로 분할 (앞에서/뒤에서 각각 → 2·N_s 구간)
        3) 각 구간에서 m차 다항식 추세를 빼고 잔차 분산을 구함
        4) F(s) = sqrt( 전 구간 잔차 분산의 평균 )
        5) F(s) ∝ s^α  →  log F 를 log s 에 회귀한 기울기가 α

    해석 (R/S 의 H 와 같은 척도)
        α > 0.5  지속성      α = 0.5  랜덤워크      α < 0.5  반지속성

    **order 기본값이 2인 이유 (문헌 기본값은 1입니다)**
        이 저장소에서 실측한 결과입니다. 참값 0.5 인 랜덤워크에 시변 추세
        (진폭이 변하는 사인 + 드리프트)를 얹고 800봉으로 25회 평균했을 때:

            DFA order=1 → 0.648   (크게 부풀음)
            DFA order=2 → 0.529
            고전 R/S    → 0.506

        order=1 이 무너지는 이유는 detrending 차수가 아니라 **스케일 범위**입니다.
        max_scale 이 n/4 까지 가면 그 길이의 구간 안에서 완만한 곡선 추세가
        1차 다항식으로 제거되지 않아 큰 s 의 F(s) 가 부풀고 기울기가 가팔라집니다.
        고전 R/S 가 이 실험에서 좋아 보인 것은 강건해서가 아니라 max_lag 이
        64 로 짧아 같은 함정을 피한 것뿐입니다.

        즉 **"DFA 가 R/S 보다 추세에 강건하다"는 통념은 차수와 스케일 범위를
        맞췄을 때만 참입니다.** 주가처럼 시변 추세가 기본인 시계열에는 2차를
        쓰세요. 검증 코드는 tests/test_econophysics.py 에 있습니다.

    반환에 `se`(기울기 표준오차)를 포함합니다. **α 만 보고 0.5 와 비교하지 말고
    t = (α − 0.5)/se 로 판단하세요.**
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_scale * 4:
        return None

    if max_scale is None:
        max_scale = n // 4
    max_scale = int(min(max_scale, n // 4))
    if max_scale <= min_scale:
        return None

    # 프로파일 — 누적편차
    profile = np.cumsum(x - x.mean())

    # 로그 등간격 스케일 (중복 제거)
    scales = np.unique(np.round(
        np.exp(np.linspace(math.log(min_scale), math.log(max_scale), n_scales))
    ).astype(int))
    scales = scales[(scales >= min_scale) & (scales <= max_scale)]
    if len(scales) < 4:
        return None

    fluct = []
    used = []
    for s in scales:
        n_seg = n // s
        if n_seg < 2 or s <= order + 1:
            continue
        # 앞에서 자른 구간 + 뒤에서 자른 구간 (꼬리 표본 손실 방지 — 원논문 절차)
        head = profile[:n_seg * s].reshape(n_seg, s)
        tail = profile[n - n_seg * s:].reshape(n_seg, s)
        segments = np.vstack([head, tail])

        # 같은 스케일의 모든 구간은 설계행렬이 동일합니다. 잔차 사영행렬을 한 번만
        # 만들어 행렬곱으로 전 구간을 동시에 추세제거합니다 (구간별 polyfit 대비 ~100배).
        t = np.linspace(-1.0, 1.0, s)
        vander = np.vander(t, order + 1)
        resid_proj = np.eye(s) - vander @ np.linalg.pinv(vander)
        residuals = segments @ resid_proj.T

        f = math.sqrt(float(np.mean(residuals ** 2)))
        if f > 0:
            fluct.append(f)
            used.append(int(s))

    if len(used) < 4:
        return None

    log_s = np.log(np.asarray(used, dtype=float))
    log_f = np.log(np.asarray(fluct, dtype=float))

    slope, intercept, se, r2 = _ols_slope_with_se(log_s, log_f)
    if slope is None:
        return None

    alpha = float(slope)
    if alpha > 0.55:
        regime = "persistent"
    elif alpha < 0.45:
        regime = "antipersistent"
    else:
        regime = "random"

    raw_t = (alpha - 0.5) / se if se > 1e-9 else 0.0
    cal = dfa_null_calibration(n, order=order, min_scale=min_scale,
                               max_scale=max_scale, n_scales=n_scales)

    return {
        "alpha": round(alpha, 4),
        "se": round(float(se), 4),
        # ⚠ 이 t 를 정규분포 분위수와 비교하지 마세요. OLS 표준오차는 스케일 간
        #   잔차 상관을 무시해 과소추정됩니다 — 아래 t_calibrated 를 쓰세요.
        "t_stat": round(raw_t, 3),
        # 경험적 귀무분포로 표준화한 t. 랜덤워크에서 평균 0, 표준편차 1 이 되도록
        # 보정했으므로 |t_calibrated| > 2 를 5% 유의로 읽어도 됩니다.
        "t_calibrated": (round((raw_t - cal["mean"]) / cal["sd"], 3)
                         if cal and cal["sd"] > 1e-9 else None),
        "r_squared": round(float(r2), 4),
        "regime": regime,
        "n_scales": len(used),
        "scales": used,
    }


_DFA_NULL_CACHE: dict = {}


def dfa_null_calibration(n: int, order: int = 2, min_scale: int = 8,
                         max_scale: int = None, n_scales: int = 12,
                         n_sim: int = 400, seed: int = 20260807) -> dict | None:
    """DFA t통계의 **경험적 귀무분포** (랜덤워크에서의 평균·표준편차).

    왜 필요한가 — 이 저장소에서 실측한 결과입니다. 창 252봉, 400회 몬테카를로:

        랜덤워크에서 t 의 평균 +1.16, 표준편차 3.34, |t| > 2 인 비율 51%

    정규분포라면 4.6% 여야 합니다. 두 가지가 겹쳐 일어난 일입니다.

      1) **과분산** — `_ols_slope_with_se` 는 log F(s) 를 log s 에 회귀하면서
         잔차가 스케일 간 독립이라고 가정하는데, 실제로는 큰 스케일의 구간이
         작은 스케일의 구간을 **포함**하므로 강하게 상관됩니다. 그래서 표준오차가
         과소추정되고 t 가 약 3.3배 부풉니다.
      2) **유한표본 편향** — 짧은 창에서 α̂ 가 0.5 보다 약간 위로 치우칩니다.
         고전 R/S 가 Anis-Lloyd 보정을 필요로 했던 것과 같은 종류의 문제입니다.

    두 문제 모두 닫힌 형태의 보정식이 알려져 있지 않아, **같은 (n, order, 스케일
    설정) 에서 랜덤워크를 직접 돌려** 평균과 표준편차를 재고 그것으로 표준화합니다.
    결과는 캐시되므로 같은 설정의 두 번째 호출부터는 비용이 없습니다.
    """
    key = (int(n), int(order), int(min_scale),
           int(max_scale) if max_scale else 0, int(n_scales), int(n_sim))
    if key in _DFA_NULL_CACHE:
        return _DFA_NULL_CACHE[key]
    if n < min_scale * 4:
        return None

    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n_sim):
        x = rng.normal(0.0, 1.0, n)
        # 재귀를 피하려고 기울기·표준오차를 직접 계산합니다
        res = _dfa_slope_only(x, min_scale, max_scale, order, n_scales)
        if res is not None:
            slope, se = res
            if se > 1e-9:
                ts.append((slope - 0.5) / se)
    if len(ts) < 30:
        return None

    arr = np.asarray(ts, dtype=float)
    out = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
           "n_sim": len(arr),
           "reject_rate_naive": float(np.mean(np.abs(arr) > 2))}
    _DFA_NULL_CACHE[key] = out
    return out


def _dfa_slope_only(x, min_scale, max_scale, order, n_scales):
    """dfa() 의 계산부만 떼어낸 것 — 귀무분포 시뮬레이션 전용."""
    n = len(x)
    if max_scale is None:
        max_scale = n // 4
    max_scale = int(min(max_scale, n // 4))
    if max_scale <= min_scale:
        return None
    profile = np.cumsum(x - x.mean())
    scales = np.unique(np.round(
        np.exp(np.linspace(math.log(min_scale), math.log(max_scale), n_scales))
    ).astype(int))
    scales = scales[(scales >= min_scale) & (scales <= max_scale)]
    fluct, used = [], []
    for s in scales:
        n_seg = n // s
        if n_seg < 2 or s <= order + 1:
            continue
        head = profile[:n_seg * s].reshape(n_seg, s)
        tail = profile[n - n_seg * s:].reshape(n_seg, s)
        segments = np.vstack([head, tail])
        t = np.linspace(-1.0, 1.0, s)
        vander = np.vander(t, order + 1)
        residuals = segments @ (np.eye(s) - vander @ np.linalg.pinv(vander)).T
        f = math.sqrt(float(np.mean(residuals ** 2)))
        if f > 0:
            fluct.append(f)
            used.append(int(s))
    if len(used) < 4:
        return None
    slope, _, se, _ = _ols_slope_with_se(np.log(np.asarray(used, dtype=float)),
                                         np.log(np.asarray(fluct, dtype=float)))
    return (slope, se) if slope is not None else None


def _ols_slope_with_se(x, y):
    """단순회귀 기울기 + 그 표준오차 + R². (slope, intercept, se, r2)"""
    n = len(x)
    if n < 3:
        return None, None, None, None
    x_mean, y_mean = x.mean(), y.mean()
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 1e-12:
        return None, None, None, None
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = float(y_mean - slope * x_mean)

    resid = y - (slope * x + intercept)
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - y_mean) ** 2))
    # 잔차 자유도 n-2
    se = math.sqrt(sse / (n - 2) / sxx) if n > 2 and sse > 0 else 1e-9
    r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
    return slope, intercept, se, r2


# ---------------------------------------------------------------------------
# Lo (1991) 수정 R/S
# ---------------------------------------------------------------------------


def modified_rs(returns, q: int = None) -> dict | None:
    """단기 자기상관을 제거한 R/S 검정 통계량.

    Lo, A. W. (1991). "Long-Term Memory in Stock Market Prices."
    Econometrica, 59(5), 1279-1313.

    고전 R/S 의 문제: **단기 의존성만 있어도** R/S 가 부풀어 장기기억으로 오독됩니다.
    Lo 는 분모의 표준편차를 Newey-West 형태의 자기공분산 보정 분산으로 바꿔
    이를 제거했습니다.

        σ²(q) = γ₀ + 2 Σ_{j=1}^{q} ω_j(q) · γ_j ,   ω_j(q) = 1 − j/(q+1)
        V(q)  = [ max_k Σ(r−r̄) − min_k Σ(r−r̄) ] / ( √n · σ(q) )

    지연 q 는 Andrews (1991) 자동선택식을 씁니다.

    판정 (Lo 1991 Table II, 점근 임계값)
        V > 1.862  5% 유의수준에서 "단기의존성으로 설명되지 않는 장기기억" 존재
        V < 0.809  5% 유의수준에서 반대쪽 기각
        그 사이     귀무가설(단기의존성만 있음)을 기각하지 못함
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 32:
        return None

    dev = r - r.mean()
    cumdev = np.cumsum(dev)
    rng = float(cumdev.max() - cumdev.min())
    if rng <= 0:
        return None

    gamma0 = float(np.mean(dev ** 2))
    if gamma0 <= 1e-18:
        return None

    if q is None:
        q = _andrews_lag(dev, n)
    q = int(max(0, min(q, n - 2)))

    # Newey-West 가중 자기공분산 합
    s2 = gamma0
    for j in range(1, q + 1):
        gamma_j = float(np.mean(dev[j:] * dev[:-j]))
        weight = 1.0 - j / (q + 1.0)
        s2 += 2.0 * weight * gamma_j
    if s2 <= 1e-18:
        # 보정 분산이 음수/0 — 이 표본에서는 판정하지 않습니다
        return None

    v_stat = rng / (math.sqrt(n) * math.sqrt(s2))

    # 고전 R/S 도 함께 돌려주어 둘의 차이를 볼 수 있게 합니다
    classic = rng / (math.sqrt(n) * math.sqrt(gamma0))

    if v_stat > 1.862:
        verdict = "long_memory"
    elif v_stat < 0.809:
        verdict = "anti_memory"
    else:
        verdict = "short_memory_only"

    return {
        "v_stat": round(float(v_stat), 4),
        "classic_v": round(float(classic), 4),
        "q_lag": q,
        "n": n,
        "verdict": verdict,
        "significant": bool(v_stat > 1.862 or v_stat < 0.809),
    }


def _andrews_lag(dev: np.ndarray, n: int) -> int:
    """Andrews (1991) 자동 지연 선택 — AR(1) 근사.

        rho = 1차 자기상관
        q*  = [ (3n/2)^(1/3) · (2rho/(1−rho²))^(2/3) ]
    """
    if n < 4:
        return 0
    denom = float(np.sum(dev[:-1] ** 2))
    if denom <= 1e-18:
        return 0
    rho = float(np.sum(dev[1:] * dev[:-1]) / denom)
    rho = max(-0.99, min(0.99, rho))
    if abs(rho) < 1e-6:
        return 0
    term = (2.0 * rho / (1.0 - rho ** 2)) ** 2
    q = ((3.0 * n / 2.0) ** (1.0 / 3.0)) * (term ** (1.0 / 3.0))
    return int(max(0, min(math.floor(q), n // 4)))


# ---------------------------------------------------------------------------
# Hill 꼬리지수 — 한쪽 꼬리, 표준오차, Hill plot
# ---------------------------------------------------------------------------


def hill_tail(returns, side: str = "left", k: int = None,
              k_min_frac: float = 0.02, k_max_frac: float = 0.20) -> dict | None:
    """수익률 분포 **한쪽 꼬리**의 멱지수 α 를 Hill 추정량으로 구합니다.

    Hill, B. M. (1975). "A Simple General Approach to Inference About the Tail
    of a Distribution." Annals of Statistics, 3(5), 1163-1174.

        정렬  X_(1) ≥ X_(2) ≥ … ≥ X_(n)
        α̂_k = [ (1/k) Σ_{i=1}^{k} ln( X_(i) / X_(k+1) ) ]^(−1)
        SE(α̂_k) ≈ α̂_k / √k                       ... 점근 표준오차

    **왜 한쪽만 재는가**: 수익률 분포의 좌우 꼬리는 두께가 다릅니다 (gain/loss
    asymmetry — Cont 2001 의 정형사실 중 하나). 절댓값으로 합쳐 재면 급등이 급락
    위험으로 계산됩니다. 하락 위험을 재려면 `side="left"` 로 **음수 수익률만** 씁니다.

    **k 선택**: Hill 추정량은 k 에 극도로 민감합니다(편의-분산 트레이드오프).
    k 를 주지 않으면 Hill plot 의 **평탄 구간**을 찾아 자동 선택합니다
    (Drees, de Haan & Resnick 2000, "How to make a Hill plot").

    실증 기준값: 주식 수익률의 꼬리지수는 α ≈ 3 부근입니다
    (Gopikrishnan, Plerou, Amaral, Meyer & Stanley 1999 — inverse cubic law).
    α 가 2 에 가까우면 분산이 겨우 존재하는 수준, 4 이상이면 정규분포에 가깝습니다.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if side == "left":
        tail_data = -r[r < 0]
    elif side == "right":
        tail_data = r[r > 0]
    else:
        raise ValueError("side 는 'left' 또는 'right' 여야 합니다")

    tail_data = tail_data[tail_data > 0]
    n = len(tail_data)
    if n < 40:
        return None

    ordered = np.sort(tail_data)[::-1]          # 내림차순
    log_ordered = np.log(ordered)

    # 누적합으로 모든 k 의 α̂ 를 한 번에 — Hill plot
    cumsum_log = np.cumsum(log_ordered)
    k_lo = max(int(n * k_min_frac), 10)
    k_hi = max(int(n * k_max_frac), k_lo + 5)
    k_hi = min(k_hi, n - 2)
    if k_hi <= k_lo:
        return None

    ks = np.arange(k_lo, k_hi + 1)
    # mean_{i≤k} ln X_(i) − ln X_(k+1)
    mean_logs = cumsum_log[ks - 1] / ks
    alphas = 1.0 / (mean_logs - log_ordered[ks])
    valid = np.isfinite(alphas) & (alphas > 0)
    if valid.sum() < 5:
        return None
    ks, alphas = ks[valid], alphas[valid]

    if k is None:
        chosen_idx = _hill_plateau(alphas)
        k_used = int(ks[chosen_idx])
        alpha = float(alphas[chosen_idx])
    else:
        k_used = int(max(10, min(k, n - 2)))
        mean_log = float(cumsum_log[k_used - 1] / k_used)
        denom = mean_log - float(log_ordered[k_used])
        if denom <= 0:
            return None
        alpha = 1.0 / denom

    se = alpha / math.sqrt(k_used)

    return {
        "alpha": round(alpha, 4),
        "se": round(se, 4),
        "k": k_used,
        "n_tail": n,
        "side": side,
        # 안정성 진단 — Hill plot 이 평탄한가. 크면 k 선택에 결과가 좌우된다는 뜻
        "plot_dispersion": round(float(np.std(alphas)), 4),
        # α=3 (inverse cubic law) 대비 t통계
        "t_vs_cubic": round((alpha - 3.0) / se, 3) if se > 1e-9 else 0.0,
    }


def _hill_plateau(alphas: np.ndarray, band: int = 7) -> int:
    """Hill plot 에서 가장 평탄한 구간의 중심 인덱스.

    각 위치에서 폭 `band` 창의 표준편차를 재고 그것이 최소인 지점을 고릅니다.
    "α̂ 가 k 에 둔감한 영역이 참값에 가깝다"는 Hill plot 판독 관행의 자동화입니다.
    """
    n = len(alphas)
    if n <= band:
        return n // 2
    half = band // 2
    best_idx, best_disp = half, float("inf")
    for i in range(half, n - half):
        window = alphas[i - half:i + half + 1]
        disp = float(np.std(window)) / max(float(np.mean(window)), 1e-9)
        if disp < best_disp:
            best_disp, best_idx = disp, i
    return best_idx


# ---------------------------------------------------------------------------
# 랜덤행렬이론 — Marchenko-Pastur 경계와 상관행렬 정화
# ---------------------------------------------------------------------------


def marchenko_pastur(n_assets: int, n_obs: int, sigma2: float = 1.0) -> dict | None:
    """순수 잡음 상관행렬 고유값이 놓이는 구간 [λ₋, λ₊].

    Marchenko, V. A., & Pastur, L. A. (1967). "Distribution of eigenvalues for
    some sets of random matrices." Mathematics of the USSR-Sbornik, 1(4).

        Q = T/N  (관측/자산),   λ± = σ² ( 1 ± √(N/T) )²

    **이 경계가 RMT 의 핵심입니다.** 최대 고유값을 뽑는 것이 아니라, 어느 고유값이
    잡음 벌크 **위**에 있는지 판정하는 것이 목적입니다. λ₊ 아래 고유값은 표본
    잡음으로 설명되므로 정보가 없습니다.
    """
    if n_assets < 2 or n_obs < 2:
        return None
    ratio = n_assets / n_obs
    if ratio >= 1.0:
        # T ≤ N 이면 상관행렬이 특이(rank 부족) — 고유값 분해가 의미를 잃습니다
        return None
    sqrt_ratio = math.sqrt(ratio)
    return {
        "lambda_minus": sigma2 * (1.0 - sqrt_ratio) ** 2,
        "lambda_plus": sigma2 * (1.0 + sqrt_ratio) ** 2,
        "q": n_obs / n_assets,
    }


# 시장 모드를 정의하기 위한 최소 자산 수. 이보다 적으면 첫 고유벡터는 '시장'이
# 아니라 그 몇 종목의 공통 성분입니다 (원논문은 406·1000 종목).
RMT_MIN_ASSETS = 15


def rmt_decompose(returns_matrix, min_assets: int = RMT_MIN_ASSETS) -> dict | None:
    """상관행렬을 잡음/신호로 나누고 시장 모드를 추출합니다.

    Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999).
    "Noise Dressing of Financial Correlation Matrices." PRL 83(7), 1467.
    Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T., &
    Stanley, H. E. (2002). "Random matrix approach to cross correlations in
    financial data." Physical Review E, 65, 066126.

    입력은 (n_assets, n_obs) 배열입니다. 각 자산은 **자기 표준편차로 정규화**한 뒤
    상관행렬을 만듭니다 (변동성이 큰 자산이 상관을 끌고 가는 것을 막습니다).

    돌려주는 것
        lambda_max          최대 고유값
        lambda_plus         MP 잡음 상한
        n_signal            λ₊ 를 넘는 고유값 개수 (= 의미 있는 모드 수)
        market_mode_valid   최대 고유값이 잡음 위에 있는가 (False 면 "시장 모드" 없음)
        market_share        λ₁/N — 시장 모드가 설명하는 분산 비율.
                            **위기에 급등합니다** (Plerou et al. 2002)
        couplings           자산별 시장 모드 결합도 (평균 1.0)

    **min_assets 가 있는 이유**: 3~5 종목으로 계산한 첫 고유벡터는 "시장"이 아니라
    그 몇 종목의 공통 성분입니다. 원논문은 각각 406종목·1000종목을 씁니다.
    자산 수가 부족하면 계산하지 않고 None 을 돌려줍니다.
    """
    x = np.asarray(returns_matrix, dtype=float)
    if x.ndim != 2:
        return None
    n_assets, n_obs = x.shape
    if n_assets < min_assets or n_obs < n_assets * 2:
        return None
    if not np.all(np.isfinite(x)):
        return None

    # 자산별 표준화 — 상관행렬을 변동성 편차 없이 만듭니다
    std = x.std(axis=1, ddof=1)
    if np.any(std <= 1e-12):
        return None
    z = (x - x.mean(axis=1, keepdims=True)) / std[:, None]
    corr = (z @ z.T) / (n_obs - 1)
    np.fill_diagonal(corr, 1.0)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
    except np.linalg.LinAlgError:
        return None

    bounds = marchenko_pastur(n_assets, n_obs)
    if bounds is None:
        return None
    lambda_plus = bounds["lambda_plus"]

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    lambda_max = float(eigenvalues[0])
    n_signal = int(np.sum(eigenvalues > lambda_plus))
    market_valid = lambda_max > lambda_plus

    v1 = eigenvectors[:, 0]
    # 단위벡터이므로 Σv₁ⱼ² = 1 → n 을 곱하면 평균 1.0 인 결합도가 됩니다
    couplings = (n_assets * v1 ** 2).astype(float)

    return {
        "lambda_max": round(lambda_max, 4),
        "lambda_plus": round(float(lambda_plus), 4),
        "n_signal": n_signal,
        "n_assets": n_assets,
        "n_obs": n_obs,
        "market_mode_valid": bool(market_valid),
        # 시장 모드가 설명하는 분산 비율 — 시스템 리스크 게이지
        "market_share": round(lambda_max / n_assets, 4),
        "couplings": couplings,
        "eigenvalues": eigenvalues,
    }


def ledoit_wolf(returns_matrix) -> dict | None:
    """Ledoit-Wolf 수축 공분산 — 소표본에서 RMT 정화의 실용적 대안.

    Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for
    large-dimensional covariance matrices." Journal of Multivariate Analysis, 88.

        Σ* = δ·μI + (1−δ)·S
        μ = tr(S)/N,   δ = min( β²/d², 1 )
        d² = ‖S − μI‖²_F / N,   β² = (1/(N·T²)) Σ_t ‖x_t x_tᵀ − S‖²_F

    자산 수가 MP 판정을 쓰기에 부족할 때(N < 15) 이쪽을 씁니다. 표본 공분산의
    최대 고유값 과대추정·최소 고유값 과소추정을 한 번에 눌러줍니다.
    """
    x = np.asarray(returns_matrix, dtype=float)
    if x.ndim != 2:
        return None
    n_assets, n_obs = x.shape
    if n_assets < 2 or n_obs < 10:
        return None

    xc = x - x.mean(axis=1, keepdims=True)
    s = (xc @ xc.T) / n_obs
    mu = float(np.trace(s)) / n_assets
    target = mu * np.eye(n_assets)

    d2 = float(np.sum((s - target) ** 2)) / n_assets

    beta2 = 0.0
    for t in range(n_obs):
        xt = xc[:, t:t + 1]
        beta2 += float(np.sum((xt @ xt.T - s) ** 2))
    beta2 /= (n_assets * n_obs ** 2)

    if d2 <= 1e-18:
        intensity = 1.0
    else:
        intensity = max(0.0, min(beta2 / d2, 1.0))

    shrunk = intensity * target + (1.0 - intensity) * s
    return {
        "covariance": shrunk,
        "shrinkage": round(float(intensity), 4),
        "n_assets": n_assets,
        "n_obs": n_obs,
    }


# ---------------------------------------------------------------------------
# Chow-Denning 다중 분산비율 동시검정
# ---------------------------------------------------------------------------


def chow_denning(prices, qs=(2, 4, 8, 16), alpha: float = 0.05) -> dict | None:
    """여러 q 에서 분산비율을 동시에 검정합니다 (크기 왜곡 보정).

    Chow, K. V., & Denning, K. C. (1993). "A simple multiple variance ratio
    test." Journal of Econometrics, 58(3), 385-401.

    **왜 필요한가**: 단일 q 에서 |z| > 1.96 을 여러 q(또는 여러 종목)에 반복
    적용하면 명목 5% 가 실제로는 훨씬 큰 위양성률이 됩니다. q 를 4개 보면
    랜덤워크인데도 하나쯤 유의하게 나올 확률이 약 19% 입니다.

        MV = max_i | z*(q_i) |
        임계값 = SMM(α, m, ∞) = Φ⁻¹( (1 + (1−α)^(1/m)) / 2 )

    z*(q) 는 engine/quant.variance_ratio 의 이분산 강건 통계량을 그대로 씁니다.
    """
    from engine import quant

    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 40:
        return None

    stats = []
    for q in qs:
        vr = quant.variance_ratio(p, q=int(q))
        if vr is None:
            continue
        stats.append({"q": int(q), "vr": vr["vr"], "z": vr["z"]})
    if len(stats) < 2:
        return None

    m = len(stats)
    mv = max(abs(s["z"]) for s in stats)
    # Studentized Maximum Modulus 임계값 (자유도 ∞ 근사)
    critical = _norm_ppf((1.0 + (1.0 - alpha) ** (1.0 / m)) / 2.0)

    return {
        "mv_stat": round(float(mv), 4),
        "critical": round(float(critical), 4),
        "m_tests": m,
        "alpha": alpha,
        # 단일 q 임계값과의 비교 — 얼마나 관대했는지 보여줍니다
        "single_test_critical": 1.96,
        "reject_random_walk": bool(mv > critical),
        "by_q": stats,
    }


def _norm_ppf(p: float) -> float:
    """표준정규 분위수. engine/validation.py 의 구현을 재사용합니다."""
    from engine.validation import _norm_ppf as ppf
    return ppf(p)


# ---------------------------------------------------------------------------
# LPPLS — 로그주기 멱함수 특이점 (버블 종말 탐지)
# ---------------------------------------------------------------------------

# 소르네트 표준 필터 조건 (Filimonov & Sornette 2013, lppls 패키지 기본값)
LPPLS_M_RANGE = (0.01, 1.2)
LPPLS_W_RANGE = (2.0, 25.0)
LPPLS_DAMPING_MIN = 0.8
LPPLS_OSCILLATION_MIN = 2.5


def lppls_fit(log_prices, tc_max_frac: float = 0.4,
              n_tc: int = 24, n_m: int = 9, n_w: int = 12) -> dict | None:
    """로그가격을 LPPLS 모형에 적합시켜 임계시각 tc 를 추정합니다.

    Johansen, A., Ledoit, O., & Sornette, D. (2000). "Crashes as critical
    points." International Journal of Theoretical and Applied Finance, 3(2).
    Sornette, D. (2003). "Why Stock Markets Crash."
    Filimonov, V., & Sornette, D. (2013). "A stable and robust calibration
    scheme of the log-periodic power law model." Physica A, 392(17).

        ln p(t) = A + (t_c − t)^m · [ B + C₁·cos(ω·ln(t_c−t))
                                        + C₂·sin(ω·ln(t_c−t)) ]

    **발상**: 투기적 버블에서 가격은 단순 지수가 아니라 **초지수적(super-exponential)**
    으로 오르며, 유한시간 특이점 t_c 를 향해 발산합니다. 거기에 얹힌 로그주기 진동은
    거래자 간 모방(herding)의 이산적 스케일 불변성에서 나옵니다. t_c 는 "반드시
    폭락하는 날"이 아니라 **체제가 바뀔 가장 확률 높은 시점**입니다.

    **구현**: 비선형 파라미터는 (t_c, m, ω) 셋뿐이고, 선형 4개 (A, B, C₁, C₂) 는
    주어진 셋에 대해 최소자승 정규방정식으로 종속됩니다. 원 구현(scipy Nelder-Mead
    + 무작위 재시작)과 달리 **결정적 격자 탐색**을 씁니다 — engine/garch.py 가 같은
    이유로 격자 MLE 를 쓰는 것과 같습니다(scipy 불필요, 같은 입력에 같은 출력).

    부호 규약
        B < 0  양(positive)의 버블 — 초지수적 상승 뒤의 하락 위험
        B > 0  음(negative)의 버블 — 초지수적 하락 (반등 위험)
    """
    y = np.asarray(log_prices, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 60:
        return None

    t = np.arange(n, dtype=float)
    t2 = float(n - 1)

    # t_c 는 표본 끝 직후부터 탐색 (이미 지난 t_c 는 버블 종료 후를 뜻해 무의미)
    tc_lo = t2 + 1.0
    tc_hi = t2 + max(tc_max_frac * n, 5.0)
    tc_grid = np.linspace(tc_lo, tc_hi, n_tc)
    m_grid = np.linspace(LPPLS_M_RANGE[0], LPPLS_M_RANGE[1], n_m)
    w_grid = np.linspace(6.0, 13.0, n_w)         # 문헌상 유효 진동수 대역

    best = None
    for tc in tc_grid:
        dt = np.abs(tc - t) + 1e-8
        log_dt = np.log(dt)
        for m in m_grid:
            fi = np.power(dt, m)
            for w in w_grid:
                phase = w * log_dt
                gi = fi * np.cos(phase)
                hi = fi * np.sin(phase)

                # 정규방정식 — 선형 파라미터를 비선형에 종속시킵니다
                design = np.column_stack([np.ones(n), fi, gi, hi])
                gram = design.T @ design
                gram[np.diag_indices_from(gram)] += 1e-8
                try:
                    coef = np.linalg.solve(gram, design.T @ y)
                except np.linalg.LinAlgError:
                    continue
                resid = y - design @ coef
                sse = float(resid @ resid)
                if best is None or sse < best["sse"]:
                    best = {"sse": sse, "tc": float(tc), "m": float(m),
                            "w": float(w), "coef": coef}

    if best is None:
        return None

    a, b, c1, c2 = (float(v) for v in best["coef"])
    c = math.hypot(c1, c2)

    # 감쇠 D = m|B| / (ω|C|)  — 1보다 크면 가격이 발산하지 않고 유계
    damping = (best["m"] * abs(b)) / (best["w"] * c) if c > 1e-12 else float("inf")
    # 진동 횟수 O = (ω/2π)·ln((t_c−t₁)/(t_c−t₂)) — 너무 적으면 로그주기라 할 수 없음
    tc_t1 = best["tc"] - 0.0
    tc_t2 = best["tc"] - t2
    oscillations = ((best["w"] / (2 * math.pi)) * math.log(tc_t1 / tc_t2)
                    if tc_t2 > 1e-9 and tc_t1 > 0 else 0.0)

    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - best["sse"] / sst if sst > 1e-15 else 0.0

    qualified = (
        LPPLS_M_RANGE[0] <= best["m"] <= LPPLS_M_RANGE[1]
        and LPPLS_W_RANGE[0] <= best["w"] <= LPPLS_W_RANGE[1]
        and damping >= LPPLS_DAMPING_MIN
        and oscillations >= LPPLS_OSCILLATION_MIN
    )

    return {
        "tc": round(best["tc"], 2),
        # 표본 끝에서 t_c 까지 남은 봉 수 — 작을수록 임박
        "tc_ahead": round(best["tc"] - t2, 2),
        "m": round(best["m"], 4),
        "omega": round(best["w"], 4),
        "a": round(a, 6), "b": round(b, 6),
        "c1": round(c1, 6), "c2": round(c2, 6),
        "damping": round(float(damping), 4),
        "oscillations": round(float(oscillations), 4),
        "r_squared": round(r2, 4),
        "qualified": bool(qualified),
        # B<0 이 양의 버블(상승 뒤 하락 위험)
        "bubble_sign": "positive" if b < 0 else "negative",
    }


def lppls_confidence(log_prices, window_fracs=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                     **fit_kwargs) -> dict | None:
    """LPPLS 신뢰지표 — 시작점을 바꿔 여러 번 적합했을 때 몇 %가 필터를 통과하는가.

    Sornette 그룹의 표준 사용법입니다. **한 번의 적합은 신뢰할 수 없습니다** —
    LPPLS 는 파라미터가 많아 아무 시계열에나 그럴듯하게 붙습니다. 시작점 t₁ 을
    바꿔가며 적합했을 때 **일관되게** 같은 t_c 로 수렴하고 필터를 통과해야 의미가
    있습니다. 통과 비율이 신뢰지표(confidence indicator)입니다.

    돌려주는 `positive_conf` 가 양의 버블(하락 위험) 신뢰도입니다.
    """
    y = np.asarray(log_prices, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 80:
        return None

    fits, pos, neg = [], 0, 0
    for frac in window_fracs:
        start = int(n * (1.0 - frac))
        seg = y[start:]
        if len(seg) < 60:
            continue
        res = lppls_fit(seg, **fit_kwargs)
        if res is None:
            continue
        fits.append(res)
        if res["qualified"]:
            if res["bubble_sign"] == "positive":
                pos += 1
            else:
                neg += 1

    if not fits:
        return None

    total = len(fits)
    tcs = [f["tc_ahead"] for f in fits if f["qualified"]]
    return {
        "n_fits": total,
        "positive_conf": round(pos / total, 4),
        "negative_conf": round(neg / total, 4),
        # 통과한 적합들의 t_c 일치도 — 흩어져 있으면 신뢰도가 낮습니다
        "tc_ahead_median": round(float(np.median(tcs)), 2) if tcs else None,
        "tc_ahead_spread": round(float(np.std(tcs)), 2) if len(tcs) > 1 else None,
        "mean_r_squared": round(float(np.mean([f["r_squared"] for f in fits])), 4),
    }


# ---------------------------------------------------------------------------
# 시장 진단 — 화면 표시용 통합
# ---------------------------------------------------------------------------

# 시장모드 비중이 이 수준을 넘으면 "종목이 아니라 시장이 움직이는" 국면입니다.
# 이 저장소 실측: KR 평상시 0.20~0.30 · 2020년 3월 0.45 / US 평상시 0.35~0.45 · 0.61
RMT_STRESS_HIGH = 0.45
RMT_STRESS_ELEVATED = 0.35


def market_diagnostics(returns_by_symbol: dict, index_log_prices=None) -> dict | None:
    """시장 전체의 경제물리 진단 — **표시용입니다. 매매 판단에 쓰지 마세요.**

    ⚠ 이 함수의 출력을 신호나 게이트에 연결하면 성과가 **나빠집니다.**
      이 저장소에서 288개 설정으로 측정한 결과, RMT 스트레스를 게이트로 쓰면
      홀드아웃 액티브샤프가 KR +0.529 → +0.246, US +0.285 → −0.177 이었습니다
      (AUTOTRADE.md 16장). 상관이 치솟은 뒤 노출을 줄이면 반등을 놓칩니다.

      그럼에도 남기는 이유: **무슨 일이 일어나는지 아는 것**과 **그것으로 매매하는
      것**은 다릅니다. 급락장에서 "지금 개별 종목 분석이 안 먹히는 국면"이라는
      사실은 사용자에게 알릴 가치가 있습니다. 다만 엔진이 그걸로 손을 대지는
      않습니다.

    인자
        returns_by_symbol  {종목: pd.Series(수익률, 날짜인덱스)} — 날짜로 정렬됩니다
        index_log_prices   지수 프록시 로그가격 (없으면 LPPLS 생략)
    """
    import pandas as pd

    out = {"rmt": None, "lppls": None, "stress_level": "unknown", "messages": []}

    usable = {k: v for k, v in (returns_by_symbol or {}).items()
              if v is not None and len(v) >= 60}
    if len(usable) >= RMT_MIN_ASSETS:
        aligned = pd.concat(usable.values(), axis=1, join="inner").dropna()
        if len(aligned) >= len(usable) * 2:
            res = rmt_decompose(aligned.to_numpy().T)
            if res:
                share = res["market_share"]
                out["rmt"] = {
                    "market_share": share,
                    "lambda_max": res["lambda_max"],
                    "lambda_plus": res["lambda_plus"],
                    "n_signal": res["n_signal"],
                    "market_mode_valid": res["market_mode_valid"],
                    "n_assets": res["n_assets"],
                }
                if not res["market_mode_valid"]:
                    out["stress_level"] = "none"
                    out["messages"].append(
                        "종목들이 서로 독립적으로 움직입니다 — 개별 분석이 유효한 국면입니다.")
                elif share >= RMT_STRESS_HIGH:
                    out["stress_level"] = "high"
                    out["messages"].append(
                        f"시장 전체가 한 방향으로 묶여 있습니다 (시장모드 {share:.0%}). "
                        "지금은 종목 선택보다 시장 방향이 손익을 지배합니다.")
                elif share >= RMT_STRESS_ELEVATED:
                    out["stress_level"] = "elevated"
                    out["messages"].append(
                        f"종목 간 동조화가 평소보다 높습니다 (시장모드 {share:.0%}). "
                        "분산 효과가 줄어든 상태입니다.")
                else:
                    out["stress_level"] = "normal"

    if index_log_prices is not None and len(index_log_prices) >= 80:
        conf = lppls_confidence(index_log_prices)
        if conf:
            out["lppls"] = conf
            if conf["positive_conf"] >= 0.5:
                out["messages"].append(
                    f"지수가 초지수적으로 상승 중입니다 "
                    f"(LPPLS 신뢰도 {conf['positive_conf']:.0%}"
                    + (f", 임계시각 약 {conf['tc_ahead_median']:.0f}봉 뒤"
                       if conf.get("tc_ahead_median") else "") + "). "
                    "버블 종말 패턴이며 시점은 확정이 아닙니다.")
            elif conf["negative_conf"] >= 0.5:
                out["messages"].append(
                    f"지수가 초지수적으로 하락 중입니다 "
                    f"(LPPLS 신뢰도 {conf['negative_conf']:.0%}). 반등 국면 가능성.")

    return out


# ---------------------------------------------------------------------------
# 제곱근 시장충격 법칙
# ---------------------------------------------------------------------------


def sqrt_impact_cap(daily_volume: float, daily_volatility: float,
                    max_impact_bps: float = 10.0, y_factor: float = 1.0) -> dict | None:
    """허용 가능한 시장충격에서 역산한 **주문 수량 상한**.

    Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J.,
    & Bouchaud, J.-P. (2011). "Anomalous price impact and the critical nature
    of liquidity in financial markets." Physical Review X, 1, 021006.

        ΔP/P ≈ Y · σ_daily · √( Q / V )

    Q 는 주문 수량, V 는 일 거래량, σ 는 일간 변동성, Y 는 1 부근의 상수입니다.
    충격이 참여율의 **제곱근**에 비례한다는 것이 이 법칙의 핵심입니다 — 선형이
    아니므로, "거래대금의 X% 이하" 같은 고정 비율 규칙은 큰 주문에서 충격을
    과소평가하고 작은 주문에서 과대평가합니다.

    허용 충격 `max_impact_bps` 를 정하면 참여율 상한이 닫힌 형태로 나옵니다.

        Q/V ≤ ( 목표충격 / (Y·σ) )²

    반환 `max_participation` 을 일 거래량에 곱한 것이 수량 상한입니다.
    """
    if daily_volume is None or daily_volume <= 0:
        return None
    if daily_volatility is None or daily_volatility <= 1e-9:
        return None

    target = max_impact_bps / 10000.0
    participation = (target / (y_factor * daily_volatility)) ** 2
    participation = float(min(participation, 1.0))

    return {
        "max_participation": round(participation, 6),
        "max_quantity": participation * float(daily_volume),
        "max_impact_bps": max_impact_bps,
        "daily_volatility": round(float(daily_volatility), 6),
        # 참고: 흔히 쓰는 고정 0.5% 규칙이 실제로 만들어내는 충격
        "impact_at_half_pct_bps": round(
            y_factor * daily_volatility * math.sqrt(0.005) * 10000.0, 2),
    }


# ---------------------------------------------------------------------------
# GHE — 일반화 허스트 지수 (두꺼운 꼬리에서 가장 강건)
# ---------------------------------------------------------------------------


def ghe(log_prices, q: float = 2.0, tau_max_range=(5, 19),
        n_tau_sets: int = 5) -> dict | None:
    """일반화 허스트 지수 H(q). **두꺼운 꼬리에서 유한표본 성능이 가장 좋습니다.**

    Di Matteo, T., Aste, T., & Dacorogna, M. (2005).
    "Long-term memories of developed and emerging markets."

        K_q(tau) = (1/(T-tau)) * sum_t |X(t+tau) - X(t)|^q ,  K_q(tau) ~ c*tau^{q*H(q)}
        q=2 에서  log K_2(tau) ~ log tau  회귀 기울기 / 2 = H

    **[MUST] 입력은 누적계열(로그가격)입니다.** DFA·R/S·DMA 는 수익률을 받고
    GHE 는 누적계열을 받습니다. 여기에 수익률을 넣으면 H 가 0 근처로 붕괴하고,
    반대로 dfa() 에 로그가격을 넣으면 alpha≈1.5 가 나와 순진한 코드가 이를
    "H=1.5 초지속성" 으로 보고합니다. **가장 흔한 버그입니다.** 그래서 이
    함수는 입력이 수익률처럼 보이면 경고를 함께 돌려줍니다.

    왜 GHE 를 쓰는가 (Barunik & Kristoufek 의 비교 실험)
        두꺼운 꼬리(alpha=1.1, n=512)에서
            GHE 95% CI [0.366, 0.578]   <- 좁다
            DFA 95% CI [0.223, 0.851]   <- 쓸 수 없다
        금융 수익률은 두꺼운 꼬리가 기본이므로 이 차이가 실무에서 결정적입니다.

    tau_max 강건성: tau_max 를 [5, 19] 범위에서 여러 번 잡아 H 의 **분산** 을
    함께 돌려줍니다. 이 값이 크면 스케일 선택에 결과가 의존한다는 뜻이라
    믿으면 안 됩니다.
    """
    x = np.asarray(log_prices, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 64:
        return None

    # 입력이 누적계열인지 대충 확인 — 수익률이면 증분의 분산이 수준의 분산보다 큽니다
    looks_like_returns = bool(float(np.std(np.diff(x))) > float(np.std(x)))

    lo, hi = int(tau_max_range[0]), int(tau_max_range[1])
    hi = min(hi, max(lo + 1, n // 8))
    if hi <= lo:
        return None

    estimates = []
    for tmax in np.unique(np.linspace(lo, hi, int(n_tau_sets)).astype(int)):
        taus, ks = [], []
        for tau in range(1, int(tmax) + 1):
            d = np.abs(x[tau:] - x[:-tau])
            d = d[np.isfinite(d)]
            if len(d) < 8:
                continue
            k = float(np.mean(d ** q))
            if k > 0:
                taus.append(float(tau))
                ks.append(k)
        if len(taus) < 4:
            continue
        slope, _, se, r2 = _ols_slope_with_se(np.log(np.asarray(taus)),
                                              np.log(np.asarray(ks)))
        if slope is None:
            continue
        estimates.append((slope / q, (se / q) if se else None, r2))

    if not estimates:
        return None
    hs = np.asarray([e[0] for e in estimates], dtype=float)
    ses = [e[1] for e in estimates if e[1] is not None]
    r2s = [e[2] for e in estimates if e[2] is not None]

    out = {
        "hurst": round(float(hs.mean()), 4),
        "q": float(q),
        "se": round(float(np.mean(ses)), 4) if ses else None,
        # tau_max 선택에 따른 흔들림 — 크면 스케일 의존적입니다
        "tau_sensitivity": round(float(hs.std(ddof=1)), 4) if len(hs) > 1 else 0.0,
        "r2": round(float(np.mean(r2s)), 4) if r2s else None,
        "n_tau_sets": len(estimates),
    }
    if looks_like_returns:
        out["warning"] = ("입력이 수익률처럼 보입니다. GHE 는 누적계열(로그가격)을 "
                          "받습니다. 수익률을 넣으면 H 가 0 근처로 붕괴합니다.")
    return out


# ---------------------------------------------------------------------------
# Higuchi 프랙탈 차원
# ---------------------------------------------------------------------------


def higuchi_fd(series, k_max: int = None) -> dict | None:
    """Higuchi (1988) 프랙탈 차원. 정확도·진폭 동적범위에서 우수(Esteller et al.).

        L_m(k) = (1/k) * [ sum_i |x(m+ik) - x(m+(i-1)k)| ] * (N-1)/(M*k),
                 M = floor((N-m)/k)
        L(k)   = (1/k) * sum_m L_m(k)
        D      = -slope( log L(k) ~ log k )

    **`(N-1)/(M*k)` 정규화 인자를 빠뜨리는 것이 표준 버그입니다.** 빼면 D 가
    계열 길이에 따라 달라져서 종목 간·기간 간 비교가 무의미해집니다.

    **[MUST] k_max 는 캘리브레이션 대상입니다.** 흔히 인용되는 Wanliss 3차식
    `ln k_max = 0.04235(ln N)^3 - 1.392(ln N)^2 + 15.15 ln N - 47.29` 는
    N=200 에서 k_max≈1.2 를 내놓습니다(무의미). `calibrate_kmax()` 로
    **알려진 H 의 합성 fBm 에 대해 RMSE 를 최소화하는 값** 을 고르세요.

    **[MUST] Katz·Petrosian 은 넣지 않았습니다.** Katz 는 진폭·길이 의존적이라
    국면 간·종목 간 비교 피처로 부적합합니다(쓰려면 수익률 표준화 후).
    그리고 **시계열에 박스카운팅은 금지** 입니다 — 자기아핀 객체라 답이 박스
    종횡비에 의존해 사실상 임의값이 됩니다.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 32:
        return None
    kmax = int(k_max) if k_max else max(5, min(n // 20, 40))
    kmax = max(3, min(kmax, n // 4))

    ks, ls = [], []
    for k in range(1, kmax + 1):
        lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 3:
                continue
            m_count = len(idx) - 1
            dist = float(np.sum(np.abs(np.diff(x[idx]))))
            norm = (n - 1) / (m_count * k)      # 이 인자를 빼면 안 됩니다
            lk.append(dist * norm / k)
        if lk:
            ks.append(float(k))
            ls.append(float(np.mean(lk)))

    if len(ks) < 4:
        return None
    slope, _, se, r2 = _ols_slope_with_se(np.log(np.asarray(ks)),
                                          np.log(np.asarray(ls)))
    if slope is None:
        return None
    return {"fd": round(float(-slope), 4),
            "se": round(float(se), 4) if se else None,
            "r2": round(float(r2), 4) if r2 is not None else None,
            "k_max": kmax, "n_points": len(ks)}


def _synth_fgn(n: int, h: float, rng) -> np.ndarray:
    """Davies-Harte 로 참 H 를 갖는 fGn 증분을 만듭니다."""
    k = np.arange(n)
    g = 0.5 * (np.abs(k - 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h)
               + np.abs(k + 1) ** (2 * h))
    circ = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(circ).real
    lam[lam < 0] = 0
    m = len(circ)
    w = rng.normal(size=m) + 1j * rng.normal(size=m)
    return np.fft.fft(np.sqrt(lam / (2 * m)) * w)[:n].real


def calibrate_kmax(n: int, h_true: float = 0.5, candidates=(5, 8, 10, 15, 20, 30),
                   n_sim: int = 50, seed: int = 20260808) -> dict | None:
    """알려진 H 의 합성 fBm 에서 RMSE 를 최소화하는 k_max 를 고릅니다.

    이론 관계 `D = 2 - H` 는 **자기유사 과정에서만** 성립합니다. fBm 은
    자기유사이므로 캘리브레이션 기준으로 쓸 수 있습니다. 실제 수익률에 그
    관계를 그대로 적용하는 것은 별개 문제입니다 — `selfaffinity_gap()` 참조.
    """
    n = int(n)
    if n < 64:
        return None
    rng = np.random.default_rng(seed)
    target = 2.0 - float(h_true)

    best, results = None, {}
    for km in candidates:
        errs = []
        for _ in range(int(n_sim)):
            res = higuchi_fd(np.cumsum(_synth_fgn(n, float(h_true), rng)),
                             k_max=int(km))
            if res:
                errs.append((res["fd"] - target) ** 2)
        if len(errs) >= 10:
            rmse = math.sqrt(float(np.mean(errs)))
            results[int(km)] = round(rmse, 4)
            if best is None or rmse < results[best]:
                best = int(km)
    if best is None:
        return None
    return {"k_max": best, "rmse": results[best], "target_fd": round(target, 4),
            "n": n, "all": results}


def selfaffinity_gap(returns, h_estimate: float = None,
                     k_max: int = None) -> dict | None:
    """자기아핀성 이탈 지표 `D_hat - (2 - H_hat)`.

    **[MUST] D 와 H 를 둘 다 피처로 넣지 마세요.** 하나에서 다른 하나를
    유도했다면 완전공선성을 주입하는 것입니다. 그리고 `D = 2 - H` 는
    **자기유사 과정에서만** 성립합니다(Gneiting & Schlather). 금융 수익률은
    멀티프랙탈이라 성립하지 않습니다.

    그래서 둘을 함께 넣는 대신 **편차** 를 씁니다. 이건 정당한 정보량입니다 —
    "이 계열이 자기유사 가정에서 얼마나 벗어나 있는가" 를 재니까요.
    fBm 에서는 0 에 가깝고, 실제 수익률에서는 0 이 아니어야 정상입니다.

    입력은 **수익률** 입니다 (내부에서 누적계열로 바꿔 씁니다).
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return None
    profile = np.cumsum(x - x.mean())

    if h_estimate is None:
        g = ghe(profile, q=2.0)
        if not g:
            return None
        h_estimate = g["hurst"]

    d = higuchi_fd(profile, k_max=k_max)
    if not d:
        return None
    gap = float(d["fd"] - (2.0 - float(h_estimate)))
    return {"gap": round(gap, 4), "fd": d["fd"],
            "hurst": round(float(h_estimate), 4),
            "implied_fd": round(2.0 - float(h_estimate), 4),
            "note": ("0 에 가까우면 자기유사(fBm 유사). 크게 벗어나면 멀티프랙탈 — "
                     "D 와 H 를 호환 가능한 것으로 다루면 안 됩니다.")}


# ---------------------------------------------------------------------------
# 엔트로피
# ---------------------------------------------------------------------------


def sample_entropy(series, m: int = 2, r: float = 0.2) -> dict | None:
    """SampEn (Richman & Moorman 2000). **ApEn 이 아닙니다.**

        SampEn(m, r, N) = -ln(A / B)
        A = m+1 길이 템플릿 쌍 중 체비셰프 거리 < r 인 수
        B = m   길이 템플릿 쌍 중 체비셰프 거리 < r 인 수

    ApEn 을 안 쓰는 이유 두 가지:
      · **자기매칭을 포함** 해 규칙성 쪽으로 편향됩니다
      · **N 의존적** 이라 100봉 값과 500봉 값을 비교할 수 없습니다
    SampEn 은 둘 다 없습니다.

    `r` 은 표준편차 배수입니다(기본 0.2*SD). A=0 이면 로그가 정의되지 않으므로
    `r=0.25*SD` 로 한 번 완화해 재시도하고, 그래도 안 되면 None 입니다.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 32:
        return None
    sd = float(x.std(ddof=1))
    if sd <= 1e-12:
        return None

    def _count(mm, tol):
        if n - mm < 4:
            return None
        idx = np.arange(n - mm)
        tmpl = np.stack([x[idx + j] for j in range(mm)], axis=1)
        cnt = 0
        for i in range(len(tmpl) - 1):
            d = np.max(np.abs(tmpl[i + 1:] - tmpl[i]), axis=1)
            cnt += int(np.sum(d < tol))
        return cnt

    for mult in (float(r), 0.25):
        tol = mult * sd
        b = _count(m, tol)
        a = _count(m + 1, tol)
        if a and b:
            return {"sampen": round(float(-math.log(a / b)), 4),
                    "m": int(m), "r": round(mult, 3), "n": n,
                    "relaxed": bool(mult != float(r))}
    return None


def permutation_entropy(series, order: int = 3, delay: int = 1,
                        normalize: bool = True) -> dict | None:
    """순열 엔트로피 (Bandt & Pompe 2002).

    값의 **순서 패턴** 만 보므로 단조변환에 불변입니다 — 정규화를 어떻게 할지
    고민할 필요가 없고, 잡음에 강건하며, SampEn 의 `r` 같은 임의 파라미터가
    없습니다. 실무에서 다루기 가장 편한 복잡도 측도입니다.

    **[MUST] 창 100~500봉에서는 order=3 또는 4만 쓰세요.**
        order=5 -> 120 패턴 (한계)
        order=6 -> 720 패턴 (사용 불가 — 표본보다 패턴이 많습니다)
    패턴 수가 표본에 비해 많으면 엔트로피가 구조가 아니라 **표본 크기의
    함수** 가 됩니다. 그래서 창이 부족하면 값 대신 이유를 돌려줍니다.

    `forbidden` 은 한 번도 나타나지 않은 패턴 수입니다. **부트스트랩 null 과
    비교하세요** — 작은 T 에서는 구조가 없어도 패턴이 그냥 사라집니다.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    order, delay = int(order), int(delay)
    if order < 2 or order > 6 or n < order * delay + 8:
        return None

    n_patterns = math.factorial(order)
    n_windows = n - (order - 1) * delay
    if n_windows < n_patterns:
        return {"pe": None, "order": order, "n_windows": int(n_windows),
                "reason": (f"창 {n_windows}개 < 패턴 {n_patterns}개 — "
                           f"order 를 줄이거나 창을 늘리세요.")}

    windows = np.stack([x[i * delay: i * delay + n_windows] for i in range(order)],
                       axis=1)
    ranks = np.argsort(windows, axis=1, kind="mergesort")
    codes = np.zeros(n_windows, dtype=np.int64)
    for j in range(order):
        codes = codes * order + ranks[:, j]
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = float(-np.sum(p * np.log(p)))
    hmax = math.log(n_patterns)

    return {
        "pe": round(h / hmax if normalize and hmax > 0 else h, 4),
        "pe_raw": round(h, 4), "order": order, "delay": delay,
        "n_windows": int(n_windows),
        "observed_patterns": int(len(counts)),
        "forbidden": int(n_patterns - len(counts)),
        "note": "forbidden 은 부트스트랩 null 과 비교하세요 (engine/nulls.py).",
    }


# ---------------------------------------------------------------------------
# 마할라노비스 효율성 지수
# ---------------------------------------------------------------------------


def efficiency_index(returns, calib: dict = None, estimators: dict = None,
                     window: int = None, n_boot: int = 300) -> dict | None:
    """시장 효율성 지수 — **마할라노비스 형태** 로 구현합니다.

    Kristoufek & Vosvrda 의 원형 `EI = sqrt(sum((M_i - M_i*)/R_i)^2)` 를
    그대로 쓰지 않습니다. 세 가지 이유:

      1) **부호가 없습니다.** 추세와 평균회귀가 같은 EI 로 매핑됩니다.
         국면 탐지기가 원하는 건 정확히 그 방향인데 지워집니다.
      2) **구성요소가 구조적으로 상관** 되어 있습니다(같은 데이터의 Hurst
         추정량 3개 + 프랙탈차원 4개). 유클리드 제곱합은 원소가 많은 계열을
         과대가중합니다 — 정보가 는 게 아니라 같은 정보를 여러 번 센 것입니다.
      3) `R_i` 가 임의적입니다. H 의 이론적 범위 [0,1) 로 나누면 노이즈 대비
         기여가 과소평가됩니다.

    대신::

        EI_M = sqrt( (M - M*)^T * Sigma_null^-1 * (M - M*) )

    Sigma_null 은 `engine.nulls.calibrate_covariance()` 에서 옵니다. 이러면
    자유도 k 의 **카이제곱 통계량으로 직접 해석** 되고 상관 구조를 올바로
    처리합니다. 그리고 **부호 성분(VR-1, VR z)은 별도로 함께 운반합니다.**

    `condition_number` 가 매우 크면 추정량들이 사실상 같은 것을 재고 있다는
    뜻입니다 — 그 경우 구성요소를 줄이세요.
    """
    from engine import nulls as _nulls
    from engine.quant import variance_ratio as _vr

    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 128:
        return None
    window = int(window or min(len(x), 256))

    if estimators is None:
        def _h_ghe(r):
            g = ghe(np.cumsum(r - r.mean()), q=2.0)
            return g["hurst"] if g else None

        def _dfa2(r):
            d = dfa(r, order=2)
            return d.get("alpha") if d else None

        def _pe(r):
            o = permutation_entropy(r, order=3)
            return o.get("pe") if o else None

        estimators = {"ghe": _h_ghe, "dfa": _dfa2, "permen": _pe}

    if calib is None:
        calib = _nulls.calibrate_covariance(estimators, x, window=window,
                                            n_boot=int(n_boot))
    if not calib:
        return None

    seg = x[-window:]
    vals = []
    for nm in calib["names"]:
        v = estimators[nm](seg)
        if v is None or not math.isfinite(float(v)):
            return None
        vals.append(float(v))

    dist = _nulls.mahalanobis(np.asarray(vals), calib)
    if dist is None:
        return None

    vr = _vr(np.exp(np.cumsum(seg)), q=5)
    return {
        "efficiency_index": round(float(dist), 4),
        "df": len(calib["names"]),
        "components": {nm: round(v, 4) for nm, v in zip(calib["names"], vals)},
        "null_mean": {nm: round(float(m), 4)
                      for nm, m in zip(calib["names"], calib["mean"])},
        "condition_number": round(float(calib.get("condition_number") or 0.0), 1),
        # 부호축 — 이것이 방향을 담습니다. EI 는 크기만 잽니다.
        "sign_vr_minus_1": round(float(vr["vr"] - 1.0), 4) if vr else None,
        "sign_vr_z": vr.get("z") if vr else None,
        "note": ("EI 는 카이제곱 통계량의 제곱근입니다 — df=3 이면 랜덤워크에서 "
                 "2.8 을 넘는 일이 약 5%. 방향은 sign_* 필드를 보세요."),
    }
