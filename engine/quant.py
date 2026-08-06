"""
계량금융 추정량 (Quantitative Estimators)
----------------------------------------
학술 문헌에 정의된 통계량들을 numpy만으로 구현했습니다. 각 함수는 논문 출처와
계산식을 주석으로 남겨, 나중에 값이 이상할 때 원 정의와 대조할 수 있게 했습니다.

수록 항목
    variance_ratio      Lo & MacKinlay (1988) 분산비율 검정 — 랜덤워크 귀무가설 검정
    hurst_exponent      R/S 분석 (Hurst 1951, Mandelbrot & Wallis 1969) — 시계열 지속성
    yang_zhang_vol      Yang & Zhang (2000) — OHLC 기반 최소분산 변동성 추정량
    rogers_satchell_vol Rogers & Satchell (1991) — 드리프트에 강건한 변동성
    ou_half_life        Ornstein-Uhlenbeck 평균회귀 반감기 (AR(1) 회귀)
    trend_regression    최소자승 추세 기울기 + 결정계수 R²

왜 이것들을 쓰는가
    기존 RSI/MACD 같은 지표는 "지금 신호가 무엇인가"만 알려줍니다. 위 통계량들은
    "지금 이 종목이 추세를 따라가는 국면인가, 평균으로 되돌아오는 국면인가"를
    판별해 줍니다. 이 판별이 있어야 추세추종 지표와 평균회귀 지표를 언제 믿을지
    정할 수 있습니다 (engine/indicators.py 의 detect_regime 참고).

주의: 모든 추정량은 표본이 적으면 불안정합니다. 각 함수는 표본 부족 시 None을
      반환하며, 호출부에서 반드시 None 처리를 해야 합니다.
"""

import math

import numpy as np


def _norm_cdf(x: float) -> float:
    """표준정규 누적분포 (scipy 없이 erf로 계산)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Lo & MacKinlay (1988) 분산비율 검정
# ---------------------------------------------------------------------------

def variance_ratio(prices, q: int = 5) -> dict | None:
    """분산비율 VR(q)와 이분산 강건 z통계량.

    Lo, A. W., & MacKinlay, A. C. (1988).
    "Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple
     Specification Test." Review of Financial Studies, 1(1), 41-66.

    정의 (log 가격 p, 수익률 r_t = p_t - p_{t-1}, 표본 n개)
        mu       = (p_n - p_0) / n
        sigma_a² = (1/(n-1)) * Σ (r_t - mu)²                       ... 1기 분산
        sigma_c² = (1/m)     * Σ (p_t - p_{t-q} - q*mu)²           ... q기 분산
        m        = q(n-q+1)(1 - q/n)                                ... 불편보정
        VR(q)    = sigma_c² / sigma_a²

    해석
        VR > 1  양의 자기상관 -> 추세 지속(모멘텀) 국면
        VR < 1  음의 자기상관 -> 평균회귀 국면
        VR ≈ 1  랜덤워크 (예측 불가)

    이분산 강건 z통계량 z*(q) 로 유의성을 판정합니다. |z| > 1.96 이면 5% 유의수준에서
    랜덤워크 귀무가설을 기각합니다.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < q * 4 + 2 or q < 2:
        return None

    log_p = np.log(p)
    r = np.diff(log_p)
    n = len(r)
    if n < q * 3:
        return None

    mu = (log_p[-1] - log_p[0]) / n
    resid = r - mu
    sigma_a2 = np.sum(resid ** 2) / (n - 1)
    if sigma_a2 <= 0:
        return None

    # q기 중첩 수익률의 분산
    q_diff = log_p[q:] - log_p[:-q] - q * mu
    m = q * (n - q + 1) * (1 - q / n)
    if m <= 0:
        return None
    sigma_c2 = np.sum(q_diff ** 2) / m

    vr = float(sigma_c2 / sigma_a2)

    # 이분산 강건 분산 theta*(q) = Σ_{j=1}^{q-1} [2(q-j)/q]² * delta_j
    denom = np.sum(resid ** 2) ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum((resid[j:] ** 2) * (resid[:-j] ** 2))
        delta_j = num / denom if denom > 0 else 0.0
        theta += ((2.0 * (q - j) / q) ** 2) * delta_j

    if theta <= 0:
        return {"vr": vr, "q": q, "z": None, "p_value": None, "significant": False}

    z = (vr - 1.0) / math.sqrt(theta)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))

    return {
        "vr": round(vr, 4),
        "q": q,
        "z": round(float(z), 3),
        "p_value": round(float(p_value), 4),
        "significant": bool(abs(z) > 1.96),
    }


# ---------------------------------------------------------------------------
# Hurst 지수 (R/S 분석)
# ---------------------------------------------------------------------------

def _anis_lloyd_expected_rs(n: int) -> float:
    """랜덤워크(H=0.5)일 때 기대되는 R/S 값.

    Anis, A. A., & Lloyd, E. H. (1976). "The expected value of the adjusted
    rescaled Hurst range of independent normal summands." Biometrika, 63(1).

    보정이 필요한 이유: R/S 통계량은 표본이 작을 때 H를 실제보다 높게 추정합니다.
    (순수 랜덤워크 500개로 실측하면 H가 0.60 부근까지 올라감)
    그래서 관측 기울기에서 이 기댓값의 기울기를 빼고 0.5를 더해 보정합니다.
    """
    if n < 2:
        return 0.0
    idx = np.arange(1, n)
    tail = float(np.sum(np.sqrt((n - idx) / idx)))

    if n > 340:
        front = 1.0 / math.sqrt(n * math.pi / 2.0)
    else:
        # gamma((n-1)/2) / (sqrt(pi) * gamma(n/2)) — lgamma로 오버플로 회피
        front = math.exp(math.lgamma((n - 1) / 2.0)
                         - math.lgamma(n / 2.0)) / math.sqrt(math.pi)
    return ((n - 0.5) / n) * front * tail


def hurst_exponent(prices, min_lag: int = 8, max_lag: int = 64) -> dict | None:
    """재조정 범위(R/S) 분석으로 Hurst 지수 H를 추정 (Anis-Lloyd 편향 보정 포함).

    Hurst, H. E. (1951). "Long-term storage capacity of reservoirs."
    Mandelbrot, B. B., & Wallis, J. R. (1969).
    Anis & Lloyd (1976) 소표본 편향 보정 적용.

    각 구간 길이 n에 대해
        Y_k  = Σ_{i=1}^{k} (r_i - r̄)          ... 누적편차
        R(n) = max(Y) - min(Y)                  ... 범위
        S(n) = 표준편차
        E[R/S] ∝ n^H
    log(R/S)를 log(n)에 회귀한 기울기에서, 랜덤워크 기대 기울기를 빼고 0.5를 더합니다.

    해석
        H > 0.5  지속성(추세가 이어짐)
        H = 0.5  랜덤워크
        H < 0.5  반지속성(되돌림이 잦음)
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < max_lag * 2:
        max_lag = max(min_lag * 2, len(p) // 2)
    if len(p) < min_lag * 4:
        return None

    r = np.diff(np.log(p))
    n_total = len(r)
    if n_total < min_lag * 2:
        return None

    lags, rs_values = [], []
    lag = min_lag
    while lag <= min(max_lag, n_total // 2):
        chunks = n_total // lag
        if chunks < 1:
            break
        ratios = []
        for c in range(chunks):
            seg = r[c * lag:(c + 1) * lag]
            if len(seg) < 2:
                continue
            mean = seg.mean()
            cumdev = np.cumsum(seg - mean)
            rng = cumdev.max() - cumdev.min()
            sd = seg.std(ddof=1)
            if sd > 0 and rng > 0:
                ratios.append(rng / sd)
        if ratios:
            lags.append(lag)
            rs_values.append(np.mean(ratios))
        lag = int(lag * 1.5) + 1

    if len(lags) < 3:
        return None

    log_lags = np.log(lags)
    raw_slope = float(np.polyfit(log_lags, np.log(rs_values), 1)[0])

    # Anis-Lloyd 보정: 같은 구간 길이에서 랜덤워크가 낼 기울기를 빼고 0.5 기준으로 재정렬
    expected = [_anis_lloyd_expected_rs(n) for n in lags]
    if all(e > 0 for e in expected):
        expected_slope = float(np.polyfit(log_lags, np.log(expected), 1)[0])
        h = 0.5 + (raw_slope - expected_slope)
    else:
        h = raw_slope

    h = float(max(0.0, min(1.0, h)))

    if h > 0.55:
        regime = "persistent"
    elif h < 0.45:
        regime = "antipersistent"
    else:
        regime = "random"

    return {"hurst": round(h, 4), "raw_hurst": round(raw_slope, 4),
            "regime": regime, "points": len(lags)}


# ---------------------------------------------------------------------------
# 변동성 추정량
# ---------------------------------------------------------------------------

def rogers_satchell_vol(df, window: int = 20, annualize: int = 252) -> float | None:
    """Rogers & Satchell (1991) 변동성 — 드리프트(추세)가 있어도 편향이 없습니다.

        RS = (1/n) Σ [ ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) ]
    """
    if df is None or len(df) < window:
        return None
    d = df.iloc[-window:]
    try:
        ho = np.log(d["high"] / d["open"])
        lo = np.log(d["low"] / d["open"])
        co = np.log(d["close"] / d["open"])
    except (KeyError, TypeError):
        return None

    rs = ho * (ho - co) + lo * (lo - co)
    rs = rs.replace([np.inf, -np.inf], np.nan).dropna()
    if rs.empty or rs.mean() < 0:
        return None
    return float(np.sqrt(rs.mean() * annualize) * 100)


def yang_zhang_vol(df, window: int = 20, annualize: int = 252) -> float | None:
    """Yang & Zhang (2000) 변동성 — OHLC 추정량 중 분산이 가장 작습니다.

    Yang, D., & Zhang, Q. (2000). "Drift-Independent Volatility Estimation
    Based on High, Low, Open, and Close Prices." Journal of Business, 73(3).

        σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_RS
        k     = 0.34 / (1.34 + (n+1)/(n-1))

    종가-종가 방식 대비 같은 표본으로 훨씬 안정적인 추정치를 줍니다.
    (갭 상승/하락이 잦은 국내 종목에서 특히 유리)
    """
    if df is None or len(df) < window + 1:
        return None
    d = df.iloc[-(window + 1):]
    try:
        o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    except KeyError:
        return None

    prev_close = c.shift(1)
    overnight = np.log(o / prev_close).dropna()          # 종가 -> 다음날 시가
    open_close = np.log(c / o).dropna()                  # 시가 -> 종가
    if len(overnight) < 2 or len(open_close) < 2:
        return None

    n = len(overnight)
    var_o = float(overnight.var(ddof=1))
    var_c = float(open_close.iloc[-n:].var(ddof=1))

    ho, lo, co = np.log(h / o), np.log(l / o), np.log(c / o)
    rs_series = (ho * (ho - co) + lo * (lo - co)).replace([np.inf, -np.inf], np.nan).dropna()
    if rs_series.empty:
        return None
    var_rs = float(rs_series.iloc[-n:].mean())

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_yz = var_o + k * var_c + (1 - k) * var_rs
    if not np.isfinite(var_yz) or var_yz <= 0:
        return None
    return float(np.sqrt(var_yz * annualize) * 100)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck 평균회귀
# ---------------------------------------------------------------------------

def ou_half_life(prices, window: int = 60) -> dict | None:
    """평균회귀 반감기 — 가격이 평균에서 벗어난 뒤 절반만큼 돌아오는 데 걸리는 기간.

    OU 과정 dP = λ(μ - P)dt + σdW 를 이산화하면 AR(1) 회귀가 됩니다.
        ΔP_t = α + β·P_{t-1} + ε
        반감기 = -ln(2) / ln(1 + β)

    해석
        반감기가 짧다(수일)   -> 평균회귀가 빠름, 되돌림 전략 유효
        반감기가 길다/음수    -> 추세가 지배적, 평균회귀 전략 부적합
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) < window:
        window = len(p)
    if window < 20:
        return None

    p = p[-window:]
    lagged = p[:-1]
    delta = np.diff(p)

    # 최소자승: delta = alpha + beta * lagged
    x = np.column_stack([np.ones(len(lagged)), lagged])
    try:
        beta_hat, *_ = np.linalg.lstsq(x, delta, rcond=None)
    except np.linalg.LinAlgError:
        return None
    beta = float(beta_hat[1])

    if beta >= 0:
        # 되돌리는 힘이 없음 = 추세/발산
        return {"half_life": None, "beta": round(beta, 6), "mean_reverting": False}

    ratio = 1.0 + beta
    if ratio <= 0:
        return {"half_life": None, "beta": round(beta, 6), "mean_reverting": False}

    hl = -math.log(2) / math.log(ratio)
    if not np.isfinite(hl) or hl <= 0:
        return {"half_life": None, "beta": round(beta, 6), "mean_reverting": False}

    return {"half_life": round(float(hl), 2), "beta": round(beta, 6), "mean_reverting": True}


# ---------------------------------------------------------------------------
# 추세 회귀
# ---------------------------------------------------------------------------

def trend_regression(prices, window: int = 20) -> dict | None:
    """로그가격에 대한 최소자승 추세: 기울기(기간수익률 환산) + 결정계수 R².

    R²는 "추세가 얼마나 깔끔한가"를 나타냅니다. 기울기가 같아도 R²가 낮으면
    등락이 심해 추세를 신뢰하기 어렵습니다. 그래서 방향은 기울기가,
    확신도는 R²가 담당하도록 점수화합니다.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < window or window < 5:
        return None

    y = np.log(p[-window:])
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 기울기는 1봉당 로그수익률 -> 기간 전체 % 로 환산
    period_return = (math.exp(slope * window) - 1) * 100

    return {
        "slope_per_bar": round(float(slope), 8),
        "period_return_pct": round(float(period_return), 3),
        "r_squared": round(float(max(0.0, min(1.0, r2))), 4),
        "window": window,
    }
