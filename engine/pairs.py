"""
페어 트레이딩 · 통계적 차익거래 (Statistical Arbitrage)
--------------------------------------------------------
두 종목의 **가격 차이(스프레드)** 가 평균으로 되돌아오는 성질을 씁니다.
시장 방향과 무관한(market-neutral) 전략이라 지수가 오르든 내리든 손익이
스프레드의 수렴 여부로만 결정됩니다.

**왜 이 모듈이 경제물리 추정량의 제자리인가**
    engine/quant.py 의 Hurst 와 engine/econophysics.py 의 DFA 는 "이 시계열이
    평균회귀하는가"를 재는 도구입니다. 그런데 개별 종목 **가격**은 거의 랜덤워크라
    (이 저장소 실측: DFA α 평균 0.500) 그 질문에 답할 게 없습니다. 정보가 없는
    곳에 정밀한 자를 댄 셈입니다.

    공적분된 두 종목의 **스프레드**는 다릅니다. 이론적으로 정상(stationary)
    시계열이고, 실제로 평균회귀하는지·얼마나 빨리 되돌아오는지가 그대로 손익을
    결정합니다. Hurst·DFA·OU 반감기가 바로 그 질문에 답합니다.

수록 항목
    adf_test         Augmented Dickey-Fuller 단위근 검정
    engle_granger    Engle-Granger 2단계 공적분 검정
    spread_quality   스프레드가 거래할 만한가 (문헌 표준 필터 묶음)
    find_pairs       후보 전체에서 조건을 통과하는 페어 탐색
    zscore_signal    진입·청산 신호

필터 기준은 문헌 관행을 따랐습니다
    ADF p < 0.05 · Hurst 0.20~0.48 · OU 반감기 3~25일 · 평균교차 12회 이상
    (Ramos-Requena et al. 2021; Hudson & Thames arbitragelab 문서)
"""

import math

import numpy as np

from engine import econophysics, quant

# ---------------------------------------------------------------------------
# ADF 임계값 (MacKinnon 1994/2010, 상수항 포함·추세 없음, 대표본)
# ---------------------------------------------------------------------------
ADF_CRIT = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}
# Engle-Granger 잔차 검정은 잔차가 추정된 값이라 임계값이 더 엄격합니다 (변수 2개)
COINT_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}


def adf_test(series, max_lag: int = None) -> dict | None:
    """Augmented Dickey-Fuller 단위근 검정 (상수항 포함).

        Δy_t = α + ρ·y_{t−1} + Σ_{i=1..p} γ_i·Δy_{t−i} + ε_t

    귀무가설 ρ = 0 (단위근 있음 = 비정상 = 평균회귀 안 함).
    t통계량이 임계값보다 **작으면**(더 음수) 귀무가설을 기각하고 정상이라 봅니다.

    지연 p 는 Schwert 규칙 ⌊12·(n/100)^{1/4}⌋ 에서 시작해 AIC 로 줄입니다.
    scipy 없이 최소자승만으로 계산합니다.
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 30:
        return None

    if max_lag is None:
        max_lag = int(min(12 * (n / 100.0) ** 0.25, n // 4))
    max_lag = max(0, int(max_lag))

    dy = np.diff(y)
    best = None
    for p in range(max_lag, -1, -1):
        m = len(dy) - p
        if m < 15:
            continue
        # 설계행렬: [상수, y_{t-1}, Δy_{t-1}, ..., Δy_{t-p}]
        cols = [np.ones(m), y[p:p + m]]
        for i in range(1, p + 1):
            cols.append(dy[p - i:p - i + m])
        X = np.column_stack(cols)
        Y = dy[p:p + m]
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = Y - X @ beta
        sse = float(resid @ resid)
        k = X.shape[1]
        if m - k <= 0 or sse <= 0:
            continue
        aic = m * math.log(sse / m) + 2 * k
        if best is None or aic < best["aic"]:
            sigma2 = sse / (m - k)
            try:
                xtx_inv = np.linalg.inv(X.T @ X)
            except np.linalg.LinAlgError:
                continue
            se_rho = math.sqrt(sigma2 * xtx_inv[1, 1])
            if se_rho <= 1e-15:
                continue
            best = {"aic": aic, "rho": float(beta[1]),
                    "t_stat": float(beta[1] / se_rho), "lag": p, "n": m}

    if best is None:
        return None

    t = best["t_stat"]
    # 임계값 표에서 대략적인 p-value 를 보간합니다 (정확한 분포는 비표준)
    if t <= ADF_CRIT[0.01]:
        p_approx = 0.01
    elif t <= ADF_CRIT[0.05]:
        p_approx = 0.05
    elif t <= ADF_CRIT[0.10]:
        p_approx = 0.10
    else:
        p_approx = 0.50

    return {"t_stat": round(t, 4), "lag": best["lag"], "n": best["n"],
            "p_approx": p_approx,
            "stationary_5pct": bool(t <= ADF_CRIT[0.05]),
            "critical": ADF_CRIT}


def engle_granger(y, x) -> dict | None:
    """Engle-Granger 2단계 공적분 검정.

    Engle, R. F., & Granger, C. W. J. (1987). "Co-integration and Error
    Correction." Econometrica, 55(2), 251-276.

        1단계   y_t = α + β·x_t + u_t          (최소자승)
        2단계   잔차 u_t 에 ADF 검정

    잔차가 정상이면 두 시계열은 공적분되어 있고, **스프레드 u_t 가 거래 대상**입니다.
    β 는 헤지비율(x 를 몇 주 잡아야 y 1주와 균형인가)입니다.

    잔차가 추정된 값이라 임계값이 일반 ADF 보다 엄격합니다(COINT_CRIT).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    if len(y) < 60:
        return None

    X = np.column_stack([np.ones(len(x)), x])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    spread = y - X @ beta
    if np.std(spread) <= 1e-12:
        return None

    adf = adf_test(spread)
    if adf is None:
        return None

    return {
        "alpha": float(beta[0]), "hedge_ratio": float(beta[1]),
        "spread": spread, "adf_t": adf["t_stat"], "adf_lag": adf["lag"],
        "cointegrated_5pct": bool(adf["t_stat"] <= COINT_CRIT[0.05]),
        "cointegrated_10pct": bool(adf["t_stat"] <= COINT_CRIT[0.10]),
        "critical": COINT_CRIT,
    }


# ---------------------------------------------------------------------------
# 스프레드 품질 — 거래할 만한가
# ---------------------------------------------------------------------------

DEFAULT_FILTERS = {
    "hurst_max": 0.48,      # 이보다 크면 평균회귀라 볼 수 없음
    "hurst_min": 0.20,      # 너무 낮으면 노이즈(호가 튐)일 가능성
    "half_life_min": 3.0,   # 3일 미만이면 비용을 감당 못 함
    "half_life_max": 25.0,  # 25일 초과면 자본이 묶여 회전이 안 됨
    "min_crossings": 12,    # 표본 기간 동안 평균선을 이만큼은 넘나들어야
}


def spread_quality(spread, filters: dict = None) -> dict:
    """스프레드가 평균회귀 거래에 적합한지 판정합니다.

    문헌 관행을 그대로 옮긴 필터 묶음입니다. **어느 하나도 단독으로는 충분하지
    않습니다** — ADF 는 정상성만 보고 속도를 못 보며, 반감기는 속도만 보고
    안정성을 못 봅니다. Hurst 는 둘 다와 다른 각도입니다.

    돌려주는 `tradable` 이 True 여야 거래 후보입니다.
    """
    f = dict(DEFAULT_FILTERS)
    if filters:
        f.update(filters)

    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 60:
        return {"tradable": False, "reason": "표본 부족"}

    # --- 지속성 척도: DFA 를 **차분에** 적용합니다 ---------------------------
    #
    # dfa() 는 내부에서 프로파일(cumsum)을 만들므로, 차분을 넣으면 프로파일이
    # 다시 스프레드가 되어 **스프레드 자체의 스케일링 지수**가 나옵니다.
    # 이것이 페어 문헌에서 말하는 "스프레드의 Hurst" 와 같은 척도입니다.
    #
    # quant.hurst_exponent 를 쓰지 않는 이유: 그 함수는 **가격**을 받아 내부에서
    # diff(log(·)) 를 합니다. 스프레드는 0 을 넘나들어 로그를 취하려면 평행이동이
    # 필요한데, 그 비선형 변환이 기억 구조를 왜곡합니다. 실제로 ADF t=−5.05,
    # 반감기 8.1일, 평균교차 79회로 명백히 평균회귀인 스프레드가 R/S 로는
    # 0.512(추세)로 나와 필터에서 탈락한 사례가 있었습니다.
    #
    # 인공 OU 로 실측한 분리력 (25회 평균):
    #     OU 빠름/보통/느림 -> DFA 0.350 / 0.421 / 0.479
    #     랜덤워크          -> DFA 0.517
    #   (같은 표본에서 R/S 는 0.414 / 0.457 / 0.483 vs 0.497 로 훨씬 뭉개집니다)
    dfa = econophysics.dfa(np.diff(s))
    hurst_v = dfa["alpha"] if dfa else None

    # 참고용으로 고전 R/S 도 함께 남깁니다 (둘이 갈리면 그 자체가 정보)
    shifted = s - s.min() + max(np.std(s), 1e-9)
    h_rs = quant.hurst_exponent(shifted)

    # **전체 창을 명시합니다.** ou_half_life 의 기본값 60 은 마지막 60개만 보고,
    # 그 표본에서 AR(1) 계수가 Kendall 소표본 편향으로 과소추정되어 반감기가
    # 절반 이하로 나옵니다 (참 6.9일 -> 1.7일로 측정된 사례가 있습니다).
    # 페어 선정에서는 반감기가 곧 필터이므로 이 편향이 후보를 통째로 날립니다.
    hl = quant.ou_half_life(s, window=len(s))

    mean = float(np.mean(s))
    crossings = int(np.sum(np.diff(np.sign(s - mean)) != 0))

    reasons = []
    if hurst_v is None:
        reasons.append("DFA 계산 불가")
    elif not (f["hurst_min"] <= hurst_v <= f["hurst_max"]):
        reasons.append(f"DFA α {hurst_v:.3f} 가 "
                       f"[{f['hurst_min']}, {f['hurst_max']}] 밖")

    hl_v = hl.get("half_life") if hl else None
    if not hl_v or not np.isfinite(hl_v):
        reasons.append("반감기 계산 불가 (평균회귀 아님)")
    elif not (f["half_life_min"] <= hl_v <= f["half_life_max"]):
        reasons.append(f"반감기 {hl_v:.1f}일 가 "
                       f"[{f['half_life_min']}, {f['half_life_max']}] 밖")

    if crossings < f["min_crossings"]:
        reasons.append(f"평균 교차 {crossings}회 < {f['min_crossings']}회")

    return {
        "tradable": len(reasons) == 0,
        # hurst 는 DFA 기반 값입니다 (위 주석 참고). 이름은 문헌 용어를 따랐습니다.
        "hurst": hurst_v,
        "half_life": hl_v,
        "dfa_alpha": hurst_v,
        "hurst_rs": h_rs["hurst"] if h_rs else None,   # 참고용 고전 R/S
        "crossings": crossings,
        "spread_std": float(np.std(s)),
        "reasons": reasons,
    }


def find_pairs(prices, min_obs: int = 250, filters: dict = None,
               use_log: bool = True, max_pairs: int = None) -> list[dict]:
    """후보 종목들에서 거래 가능한 페어를 찾습니다.

    prices  {종목: np.ndarray} 또는 (n_obs, n_assets) DataFrame 유사 객체.
            **모든 시계열은 같은 날짜로 정렬돼 있어야 합니다.**

    로그가격을 쓰는 것이 기본입니다 — 가격 수준이 크게 다른 종목 쌍에서
    헤지비율이 안정적입니다.
    """
    if hasattr(prices, "columns"):
        cols = list(prices.columns)
        arr = {c: np.asarray(prices[c], dtype=float) for c in cols}
    else:
        arr = {k: np.asarray(v, dtype=float) for k, v in prices.items()}
        cols = list(arr)

    usable = {}
    for c in cols:
        v = arr[c]
        if len(v) >= min_obs and np.all(np.isfinite(v)) and np.all(v > 0):
            usable[c] = np.log(v) if use_log else v
    names = sorted(usable)

    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            eg = engle_granger(usable[a], usable[b])
            if eg is None or not eg["cointegrated_5pct"]:
                continue
            q = spread_quality(eg["spread"], filters)
            if not q["tradable"]:
                continue
            out.append({
                "y": a, "x": b, "hedge_ratio": eg["hedge_ratio"],
                "alpha": eg["alpha"], "adf_t": eg["adf_t"],
                "hurst": q["hurst"], "half_life": q["half_life"],
                "dfa_alpha": q["dfa_alpha"], "crossings": q["crossings"],
                "spread_std": q["spread_std"],
            })

    # 평균회귀가 뚜렷한 순서 (ADF t 가 더 음수일수록 강한 정상성)
    out.sort(key=lambda d: d["adf_t"])
    return out[:max_pairs] if max_pairs else out


# ---------------------------------------------------------------------------
# 신호
# ---------------------------------------------------------------------------

def zscore_signal(spread, window: int = 60, entry: float = 2.0,
                  exit_z: float = 0.5, stop_z: float = 4.0) -> dict | None:
    """스프레드 z-score 기반 진입·청산 신호.

        z = (spread − 이동평균) / 이동표준편차

        z ≤ −entry   스프레드가 과도하게 낮음 → 롱 (y 매수, x 매도)
        z ≥ +entry   과도하게 높음           → 숏
        |z| ≤ exit_z 평균 근처로 복귀        → 청산
        |z| ≥ stop_z 발산                    → 손절 (공적분이 깨진 것)

    **stop_z 가 필요한 이유**: 공적분은 표본 안의 성질이고 밖에서 깨질 수 있습니다.
    깨진 페어에 "언젠가 돌아온다"고 버티면 손실이 무한히 커집니다.
    """
    s = np.asarray(spread, dtype=float)
    if len(s) < window + 5:
        return None
    mu = np.convolve(s, np.ones(window) / window, mode="valid")
    sd = np.array([np.std(s[i:i + window], ddof=1)
                   for i in range(len(s) - window + 1)])
    z = np.full(len(s), np.nan)
    valid = sd > 1e-12
    z[window - 1:][valid] = (s[window - 1:][valid] - mu[valid]) / sd[valid]

    return {"z": z, "entry": entry, "exit": exit_z, "stop": stop_z,
            "last_z": float(z[-1]) if np.isfinite(z[-1]) else None}
