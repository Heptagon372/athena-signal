"""
시장미시구조 (Market Microstructure)
------------------------------------
호가·체결·재고(inventory)를 다루는 모형들입니다. `engine/econophysics.py` 가
가격 시계열의 통계적 성질을 본다면, 이 모듈은 **주문이 실제로 체결되는 층**을 봅니다.

수록 항목
    avellaneda_stoikov  Avellaneda & Stoikov (2008) 최적 마켓메이킹 호가
    calibrate_intensity 체결 강도 λ(δ) = A·exp(−k·δ) 의 A, k 추정
    inventory_limit     재고 위험 기준 최대 보유 수량
    reversal_edge       단기반전 = 유동성 공급 대가 (Nagel 2012) 의 기대 우위 추정

왜 이 모듈이 이 프로젝트에 필요한가
    아테나의 페니주 초단타(AUTOTRADE.md 8장)는 "지정가 매수호가로 진입, 시장가로
    청산"을 씁니다. 이것은 **마켓메이킹의 한쪽 다리**입니다. 그런데 호가를 어디에
    걸지, 재고가 쌓였을 때 얼마나 밀지에 대한 근거가 없었습니다. Avellaneda-Stoikov
    는 그 두 질문에 닫힌 해를 줍니다.

    또한 이 저장소의 실측(tests/) 에서 확인된 유일한 횡단면 신호가 **단기반전**인데,
    단기반전의 경제적 정체가 유동성 공급의 대가입니다(Nagel 2012). 즉 알파와 집행이
    같은 이론에서 나옵니다.

주의: 이 모듈은 **호가·체결 데이터가 있을 때** 의미가 있습니다. 일봉만으로는
      λ(δ) 를 보정할 수 없어 calibrate_intensity() 가 None 을 돌려줍니다.
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Avellaneda-Stoikov 최적 호가
# ---------------------------------------------------------------------------


def avellaneda_stoikov(mid_price: float, inventory: float, volatility: float,
                       time_left: float, gamma: float = 0.1, kappa: float = 1.5,
                       tick_size: float = None) -> dict | None:
    """재고 위험을 반영한 최적 매수·매도 호가.

    Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit
    order book." Quantitative Finance, 8(3), 217-224.

        유보가격   r(s, q, t) = s − q·γ·σ²·(T − t)
        최적 스프레드 δᵃ + δᵇ = γ·σ²·(T − t) + (2/γ)·ln(1 + γ/κ)
        호가       ra = r + (δᵃ+δᵇ)/2 ,   rb = r − (δᵃ+δᵇ)/2

    **유보가격이 핵심입니다.** 중간가가 아니라 "지금 재고를 감안했을 때 내가
    무차별한 가격"을 중심으로 호가를 냅니다. 재고가 롱이면(q>0) 유보가격이 중간가
    아래로 내려가 **양쪽 호가가 함께 내려갑니다** — 팔기는 쉬워지고 사기는 어려워져
    재고가 자연히 줄어듭니다. 재고 한도를 하드코딩하는 것보다 부드럽고, 시장이
    불리하게 움직일 때 더 강하게 작동합니다(σ² 에 비례).

    스프레드의 두 항이 각각 다른 것을 값 매깁니다.
        γ·σ²·(T−t)        재고를 떠안는 위험의 대가 — 변동성과 남은 시간에 비례
        (2/γ)·ln(1+γ/κ)   체결 확률과의 교환 — κ 가 크면(체결이 δ에 민감하면) 좁게

    **참조 구현과의 차이**: fedecaccia/avellaneda-stoikov 와
    deuavt/order-book-simulator 는 스프레드에서 γσ²(T−t) 항을 빼고
    (2/γ)ln(1+γ/κ) 만 씁니다(무한시간 근사). 여기서는 원논문의 유한시간 식을
    그대로 씁니다 — 초단타는 T 가 짧아 그 항이 실제로 지배적입니다.

    인자
        volatility  중간가의 **단위시간당** 표준편차 (σ, 수익률 아닌 가격 단위)
        time_left   남은 시간 (T − t). 보유 상한까지 남은 시간을 1로 정규화해 쓰세요
        gamma       위험회피. 크면 재고를 빨리 털고 스프레드가 넓어집니다
        kappa       주문도착 강도의 가격민감도. calibrate_intensity() 로 구합니다
    """
    if volatility is None or volatility <= 0 or time_left is None or time_left < 0:
        return None
    if gamma <= 0 or kappa <= 0:
        return None

    var = volatility ** 2
    reservation = mid_price - inventory * gamma * var * time_left
    spread = gamma * var * time_left + (2.0 / gamma) * math.log(1.0 + gamma / kappa)

    ask = reservation + spread / 2.0
    bid = reservation - spread / 2.0

    if tick_size and tick_size > 0:
        ask = math.ceil(ask / tick_size) * tick_size
        bid = math.floor(bid / tick_size) * tick_size

    return {
        "reservation_price": reservation,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        # 중간가 대비 유보가격의 밀림 — 재고 압력의 크기
        "inventory_skew": reservation - mid_price,
        "half_spread": spread / 2.0,
    }


def calibrate_intensity(deltas, fill_times) -> dict | None:
    """체결 강도 λ(δ) = A·exp(−κ·δ) 의 A, κ 를 로그선형 회귀로 추정합니다.

    δ 는 중간가로부터의 호가 거리, λ 는 단위시간당 체결 건수입니다.
    ln λ = ln A − κ·δ 이므로 단순 회귀로 풀립니다.

    κ 가 크다  →  조금만 물러서도 체결이 급감 (좁게 걸어야 함)
    κ 가 작다  →  멀리 걸어도 체결됨 (넓게 걸어 마진을 챙길 수 있음)

    호가·체결 이력이 없으면 추정할 수 없습니다. 그 경우 avellaneda_stoikov 의
    kappa 기본값(1.5)은 **문헌 예시값이지 이 시장의 값이 아닙니다** — 실호가
    데이터가 붙기 전까지는 스프레드 절대수준을 신뢰하지 마세요.
    """
    d = np.asarray(deltas, dtype=float)
    lam = np.asarray(fill_times, dtype=float)
    mask = np.isfinite(d) & np.isfinite(lam) & (lam > 0)
    d, lam = d[mask], lam[mask]
    if len(d) < 5 or np.std(d) < 1e-12:
        return None

    log_lam = np.log(lam)
    d_mean, l_mean = d.mean(), log_lam.mean()
    sxx = float(np.sum((d - d_mean) ** 2))
    if sxx <= 1e-15:
        return None
    slope = float(np.sum((d - d_mean) * (log_lam - l_mean)) / sxx)
    intercept = float(l_mean - slope * d_mean)

    kappa = -slope
    if kappa <= 0:
        # 거리가 멀수록 체결이 늘어난다 — 모형과 반대. 보정 실패로 처리합니다.
        return None

    resid = log_lam - (slope * d + intercept)
    sst = float(np.sum((log_lam - l_mean) ** 2))
    r2 = 1.0 - float(resid @ resid) / sst if sst > 1e-15 else 0.0

    return {"A": math.exp(intercept), "kappa": kappa,
            "r_squared": round(r2, 4), "n": len(d)}


def inventory_limit(capital: float, price: float, volatility: float,
                    gamma: float = 0.1, horizon: float = 1.0) -> dict | None:
    """재고 위험이 자본 대비 허용치를 넘지 않는 최대 보유 수량.

    A-S 의 유보가격 밀림 q·γ·σ²·(T−t) 이 가격의 일정 비율을 넘으면, 그 재고는
    호가를 시장에서 밀어내 체결이 한쪽으로만 일어납니다(역선택). 그 지점을
    재고 상한으로 잡습니다.

        q_max = (허용 밀림 비율 × price) / (γ · σ² · T)
    """
    if price <= 0 or volatility is None or volatility <= 0 or gamma <= 0:
        return None
    max_skew = 0.5 * volatility * price      # 밀림이 0.5σ 를 넘지 않도록
    denom = gamma * (volatility ** 2) * max(horizon, 1e-9)
    if denom <= 1e-15:
        return None
    q_max = max_skew / denom
    afford = capital / price if price > 0 else 0.0
    return {
        "q_max_risk": q_max,
        "q_max_capital": afford,
        "q_max": float(min(q_max, afford)),
        "binding": "risk" if q_max < afford else "capital",
    }


# ---------------------------------------------------------------------------
# 단기반전 = 유동성 공급 대가
# ---------------------------------------------------------------------------


def reversal_edge(past_returns, forward_returns, cost_round_trip: float) -> dict | None:
    """단기반전 신호의 기대 우위가 왕복비용을 넘는지 직접 계산합니다.

    Lehmann, B. N. (1990). "Fads, Martingales, and Market Efficiency." QJE.
    Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security Returns." JF.
    Nagel, S. (2012). "Evaporating Liquidity." Review of Financial Studies, 25(7).

    Nagel 의 요지: 단기반전 수익은 이상현상이 아니라 **유동성 공급의 대가**이며,
    변동성이 높을 때(공급자가 물러설 때) 커집니다. 즉 이 우위는 위험 프리미엄이라
    사라지지 않지만, **비용을 못 이기면 남는 게 없습니다.**

    이 함수는 그 마지막 질문만 답합니다 — 신호의 횡단면 스프레드가 왕복비용의
    몇 배인가. 1.0 이하면 그 시장에서 이 전략은 성립하지 않습니다.
    """
    past = np.asarray(past_returns, dtype=float)
    fwd = np.asarray(forward_returns, dtype=float)
    mask = np.isfinite(past) & np.isfinite(fwd)
    past, fwd = past[mask], fwd[mask]
    if len(past) < 20:
        return None

    # 과거 수익률 하위/상위 분위의 이후 수익률 차이 = 반전 스프레드
    lo, hi = np.percentile(past, 20), np.percentile(past, 80)
    losers = fwd[past <= lo]
    winners = fwd[past >= hi]
    if len(losers) < 5 or len(winners) < 5:
        return None

    spread = float(losers.mean() - winners.mean())
    pooled = math.sqrt(losers.var(ddof=1) / len(losers) + winners.var(ddof=1) / len(winners))
    t_stat = spread / pooled if pooled > 1e-15 else 0.0

    return {
        "reversal_spread": round(spread, 6),
        "t_stat": round(float(t_stat), 3),
        "cost_round_trip": cost_round_trip,
        "edge_multiple": round(spread / cost_round_trip, 3) if cost_round_trip > 0 else None,
        "viable": bool(cost_round_trip > 0 and spread > cost_round_trip),
        "n_losers": len(losers), "n_winners": len(winners),
    }


# ---------------------------------------------------------------------------
# OFI — 주문흐름 불균형 (L2 가 있으면 최우선 피처)
# ---------------------------------------------------------------------------


def order_flow_imbalance(bid_prices, bid_sizes, ask_prices, ask_sizes,
                         interval: int = 1) -> dict | None:
    """Cont, Kukanov & Stoikov 의 주문흐름 불균형.

    Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact of order
    book events." Journal of Financial Econometrics, 12(1), 47-88.

    각 호가 이벤트 n 에 대해::

        e_n = 1{P^b_n >= P^b_{n-1}} * q^b_n - 1{P^b_n <= P^b_{n-1}} * q^b_{n-1}
            - 1{P^a_n <= P^a_{n-1}} * q^a_n + 1{P^a_n >= P^a_{n-1}} * q^a_{n-1}

        OFI_k = sum_{n in interval k} e_n,     dP_k = beta * OFI_k + eps_k

    **평균 R^2 이 약 65% 입니다.** 미시구조에서 가장 잘 복제된 관계 중 하나이고,
    큐 임밸런스보다 훨씬 강합니다. beta 는 관측 가능한 호가 깊이에 반비례하며,
    **beta 와 그 안정성 자체가 훌륭한 국면 피처** 입니다.

    왜 이걸 굳이 구현하는가: L1 호가 예측을 시도한 공개 노트북 하나가 README 에서
    OFI 를 인용해 놓고 **실제로는 구현하지 않았습니다**(실질 피처가 큐 임밸런스와
    상대 스프레드 2개뿐이었습니다). 그 상태에서 나온 3-클래스 83.6% 정확도는
    알파가 아니라 텐서 축 버그의 산물이었습니다. 인용과 구현은 다릅니다.

    지표의 부호는 **매수 압력이 양수** 입니다.
    """
    bp = np.asarray(bid_prices, dtype=float)
    bs = np.asarray(bid_sizes, dtype=float)
    ap = np.asarray(ask_prices, dtype=float)
    asz = np.asarray(ask_sizes, dtype=float)
    n = min(len(bp), len(bs), len(ap), len(asz))
    if n < 8:
        return None
    bp, bs, ap, asz = bp[:n], bs[:n], ap[:n], asz[:n]

    # e_n — 벡터화. 등호가 포함된 부등호라 가격이 안 변해도 잔량 변화가 잡힙니다.
    e = (np.where(bp[1:] >= bp[:-1], bs[1:], 0.0)
         - np.where(bp[1:] <= bp[:-1], bs[:-1], 0.0)
         - np.where(ap[1:] <= ap[:-1], asz[1:], 0.0)
         + np.where(ap[1:] >= ap[:-1], asz[:-1], 0.0))

    step = max(int(interval), 1)
    n_bins = len(e) // step
    if n_bins < 4:
        return None
    ofi = e[: n_bins * step].reshape(n_bins, step).sum(axis=1)

    mid = (bp + ap) / 2.0
    mid_binned = mid[: n_bins * step + 1: step]
    if len(mid_binned) < n_bins + 1:
        mid_binned = mid[np.linspace(0, n - 1, n_bins + 1).astype(int)]
    dp = np.diff(mid_binned[: n_bins + 1])

    m = min(len(ofi), len(dp))
    if m < 4:
        return None
    x, y = ofi[:m], dp[:m]
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 1e-12:
        return None
    beta = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    resid = y - (beta * x + (y.mean() - beta * x.mean()))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / sst if sst > 1e-15 else 0.0

    return {
        "ofi": [float(v) for v in ofi[-10:]],
        "ofi_last": float(ofi[-1]),
        "ofi_mean": float(ofi.mean()),
        "beta": beta,
        "r2": round(float(r2), 4),
        "n_events": int(n), "n_bins": int(n_bins),
        "note": ("문헌 수준 R^2 는 약 0.65 입니다. 크게 낮으면 호가 갱신이 "
                 "누락됐거나 interval 이 시장 속도와 안 맞는 것입니다."),
    }


def queue_imbalance(bid_size, ask_size) -> float | None:
    """큐 임밸런스 I = q^b / (q^b + q^a). 0.5 가 중립입니다.

    OFI 보다 약한 신호입니다 — 잔량의 **수준** 만 보고 흐름을 안 봅니다.
    단독으로 쓰지 말고 OFI 와 함께 쓰세요.
    """
    b, a = float(bid_size or 0), float(ask_size or 0)
    tot = b + a
    return float(b / tot) if tot > 0 else None


def microprice(bid_price, ask_price, bid_size, ask_size) -> dict | None:
    """마이크로프라이스 `mp = I*a + (1-I)*b`.

    중간가보다 다음 체결 방향을 잘 맞힙니다. 변위가 정확히
    `mp - mid = (I - 1/2) * spread` 라서, 큐가 매수 쪽으로 기울면 마이크로프라이스가
    중간가 위로 올라갑니다.

    **[MUST] 백테스트에서 체결가를 중간가로 잡지 마세요.** 매수는 ask, 매도는
    bid 로 체결하면 양쪽 다 taker 입니다. 중간가 체결 가정은 스프레드를
    공짜로 얻는 것이라 손익이 통째로 바뀝니다.
    """
    b, a = float(bid_price or 0), float(ask_price or 0)
    if not (b > 0 and a > 0 and a >= b):
        return None
    i = queue_imbalance(bid_size, ask_size)
    if i is None:
        return None
    mid = (a + b) / 2.0
    mp = i * a + (1.0 - i) * b
    return {"microprice": float(mp), "mid": float(mid),
            "imbalance": round(float(i), 4),
            "displacement": float(mp - mid),
            "spread": float(a - b)}


def book_slope(prices, sizes, side: str = "bid", mid: float = None) -> float | None:
    """호가 사다리의 기울기 — 누적잔량을 가격 거리에 회귀한 값.

    깊이가 얕을수록(기울기가 가파를수록) 같은 주문이 더 크게 움직입니다.

    **국내 주식은 10단계 호가만 제공되므로 제한적으로 쓰세요.** 암호화폐처럼
    L2 전체가 열린 시장이 이 지표의 주 무대입니다.
    """
    p = np.asarray(prices, dtype=float)
    s = np.asarray(sizes, dtype=float)
    n = min(len(p), len(s))
    if n < 3:
        return None
    p, s = p[:n], s[:n]
    ref = float(mid) if mid else float(p[0])
    if ref <= 0:
        return None
    dist = np.abs(p - ref) / ref
    cum = np.cumsum(s)
    sxx = float(np.sum((dist - dist.mean()) ** 2))
    if sxx <= 1e-15:
        return None
    return float(np.sum((dist - dist.mean()) * (cum - cum.mean())) / sxx)


# ---------------------------------------------------------------------------
# 유동성·충격
# ---------------------------------------------------------------------------


def amihud_illiquidity(returns, dollar_volumes, log_transform: bool = True) -> dict | None:
    """Amihud (2002) 비유동성 `ILLIQ = mean(|R_d| / DVOL_d)`.

    **[MUST] log 변환이 사실상 필수입니다.** 원시 ILLIQ 는 왜도가 극심해서
    (소수 종목이 평균을 지배) 횡단면 회귀에 그대로 넣으면 그 몇 종목이 계수를
    결정합니다. 그래서 기본값이 log 변환입니다.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(dollar_volumes, dtype=float)
    n = min(len(r), len(v))
    if n < 20:
        return None
    r, v = r[:n], v[:n]
    ok = np.isfinite(r) & np.isfinite(v) & (v > 0)
    if int(ok.sum()) < 20:
        return None
    ratio = np.abs(r[ok]) / v[ok]
    raw = float(np.mean(ratio))
    # log(ILLIQ) 를 그대로 씁니다. `log1p(raw * 상수)` 형태는 통화 단위에
    # 의존합니다 — 원화 거래대금(1e9 규모)에서는 raw 가 1e-11 이라 어떤
    # 고정 상수를 곱해도 0 으로 뭉개지거나 폭발합니다. 문헌 관행도 log(ILLIQ) 입니다.
    return {"illiq": raw,
            "illiq_log": float(math.log(raw)) if (log_transform and raw > 0) else None,
            "n": int(ok.sum()),
            "note": ("횡단면 피처로는 illiq_log 를 쓰세요. 원시값은 왜도가 극심해 "
                     "소수 종목이 회귀계수를 결정합니다. 통화 단위가 다른 시장끼리는 "
                     "log 값도 직접 비교하지 말고 각 시장 안에서 표준화하세요.")}


def kyle_lambda(price_changes, signed_order_flow) -> dict | None:
    """Kyle (1985) 람다 — `dp_t = lambda * OF_t + eps_t`.

    `signed_order_flow` 는 `sum(sign(trade) * size)` 입니다. 틱룰이나 체결
    구분 플래그로 부호를 매기세요.

    lambda 는 **가격 충격의 단위 물량당 크기** 라 그대로 비용 모형에 들어갑니다.
    """
    dp = np.asarray(price_changes, dtype=float)
    of = np.asarray(signed_order_flow, dtype=float)
    n = min(len(dp), len(of))
    if n < 20:
        return None
    dp, of = dp[:n], of[:n]
    ok = np.isfinite(dp) & np.isfinite(of)
    if int(ok.sum()) < 20:
        return None
    x, y = of[ok], dp[ok]
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 1e-15:
        return None
    lam = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    resid = y - (lam * x + (y.mean() - lam * x.mean()))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"kyle_lambda": lam,
            "r2": round(1.0 - float(np.sum(resid ** 2)) / sst, 4) if sst > 1e-15 else None,
            "n": int(ok.sum())}


def corwin_schultz_spread(highs, lows) -> dict | None:
    """Corwin & Schultz (2012) 고가-저가 스프레드 추정량.

    **호가 데이터 없이 일봉만으로** 유효 스프레드를 추정합니다. 국내 일봉
    백테스트에서 스프레드 비용을 넣어야 할 때 쓸 수 있는 몇 안 되는 방법입니다.

        beta  = E[ (ln(H_t/L_t))^2 + (ln(H_{t+1}/L_{t+1}))^2 ]
        gamma = (ln( max(H_t,H_{t+1}) / min(L_t,L_{t+1}) ))^2
        alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma/(3-2*sqrt(2)))
        S     = 2*(exp(alpha) - 1) / (1 + exp(alpha))

    **음수 추정치는 0 으로 자릅니다** — 원논문 권고입니다. 변동성이 낮은 구간에서
    구조적으로 음수가 나오는데, 그걸 그대로 평균내면 스프레드가 과소추정됩니다.

    ⚠ **하향 편향을 전제하고 쓰세요.** 이 추정량은 고가·저가가 실제 체결에서
    나오고 그 체결이 매수/매도호가를 오간다는 것을 가정합니다. 종가 근처에
    거래가 몰리거나 장중 변동성이 스프레드를 압도하면 참값보다 작게 나옵니다.
    저장소 합성 실험에서 참 40bp 를 24bp 로 추정했고 음수 비율이 58% 였습니다.
    **비용 하한으로 쓰고 상한으로 쓰지 마세요** — `negative_share` 가 0.4 를
    넘으면 그 구간에서는 이 추정량 자체를 믿지 않는 것이 맞습니다.
    """
    h = np.asarray(highs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    n = min(len(h), len(lo))
    if n < 20:
        return None
    h, lo = h[:n], lo[:n]
    ok = np.isfinite(h) & np.isfinite(lo) & (lo > 0) & (h >= lo)
    if int(ok.sum()) < 20:
        return None
    h, lo = h[ok], lo[ok]

    hl = np.log(h / lo) ** 2
    beta = hl[:-1] + hl[1:]
    h2 = np.maximum(h[:-1], h[1:])
    l2 = np.minimum(lo[:-1], lo[1:])
    gamma = np.log(h2 / l2) ** 2

    k = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return None
    s_clipped = np.maximum(s, 0.0)
    return {"spread": float(np.mean(s_clipped)),
            "spread_bps": round(float(np.mean(s_clipped)) * 1e4, 2),
            "negative_share": round(float(np.mean(s < 0)), 3),
            "n": int(len(s)),
            "note": "음수 추정치는 0 으로 절단했습니다(원논문 권고)."}


# ---------------------------------------------------------------------------
# 큐 포지션 체결 시뮬레이터
# ---------------------------------------------------------------------------


class RestingOrder:
    """호가에 걸려 있는 지정가 주문 하나.

    `queue_ahead` 가 이 클래스의 존재 이유입니다 — 접수 시점에 그 레벨에
    **먼저 서 있던 잔량** 입니다. 이게 소진되어야 우리 차례가 옵니다.
    """

    __slots__ = ("price", "size", "queue_ahead", "filled", "submitted_at", "side")

    def __init__(self, price: float, size: float, queue_ahead: float,
                 side: str = "buy", submitted_at=None):
        self.price = float(price)
        self.size = float(size)
        self.queue_ahead = max(float(queue_ahead), 0.0)
        self.filled = 0.0
        self.side = side
        self.submitted_at = submitted_at

    @property
    def remaining(self) -> float:
        return max(self.size - self.filled, 0.0)

    def to_dict(self) -> dict:
        return {"price": self.price, "size": self.size,
                "queue_ahead": round(self.queue_ahead, 2),
                "filled": round(self.filled, 4), "side": self.side}


class QueueSimulator:
    """큐 포지션을 추적하는 지정가 체결 시뮬레이터.

    **[MUST] "가격이 닿으면 무조건 체결" 을 금지하기 위해 존재합니다.**

    참조 구현들이 공통으로 저지른 실수가 이것입니다::

        if (price.getAsk() < buyLimitOrder.getLevel()) return true;   // 전량 즉시 체결

    호가가 지정가를 관통하면 **지정가에서, 즉시, 무한 수량** 체결됩니다.
    큐도 부분체결도 거부도 없습니다. DC 전략처럼 주문을 *시장이 곧 뚫고
    지나갈 자리* 에 놓는 전략은 이 가정의 왜곡이 특히 큽니다 — 체결되지
    말았어야 할 틱에서 체결되고, 원하는 틱에서 체결되지 않습니다.

    갱신 규칙
        1) 해당 레벨에서 체결 발생 -> queue_ahead -= min(traded, queue_ahead)
        2) 해당 레벨에서 취소 발생 -> queue_ahead -= 비례배분(추정)
        3) queue_ahead <= 0 이고 체결이 계속되면 -> 우리 주문이 체결되기 시작
        4) ★ 정정(replace) -> queue_ahead 를 **새 레벨의 전체 잔량으로 리셋**

    **규칙 4가 핵심입니다.** 매 틱 무료로 지정가를 재조정하는 구현이 흔한데,
    실제 거래소에서 재조정은 취소/정정이고 **큐 맨 뒤로** 갑니다. 저장소에서
    가장 비현실적인 동작이 이것이었습니다.

    intrinsic-time 과의 시너지: DC 러너가 다음 이벤트 가격을 *미리* 알려주므로
    (`engine/intrinsic_time.py`), 주문을 그 시점에 등록하고 **접수 순간부터
    큐 소진을 누적** 할 수 있습니다.
    """

    def __init__(self, cancel_ratio: float = 0.5):
        # 레벨 취소량 중 우리 앞쪽에서 빠지는 비율 추정치. 0.5 는 중립 가정이고
        # 실제로는 큐 앞쪽이 덜 취소되는 경향이 있어 보수적으로 낮춰도 됩니다.
        self.cancel_ratio = float(cancel_ratio)
        self.orders: list = []

    def submit(self, price: float, size: float, level_size: float,
               side: str = "buy", at=None) -> RestingOrder:
        """지정가 주문 접수. `level_size` 는 **접수 시점 그 레벨의 잔량** 입니다."""
        o = RestingOrder(price, size, level_size, side=side, submitted_at=at)
        self.orders.append(o)
        return o

    def replace(self, order: RestingOrder, new_price: float,
                new_level_size: float) -> RestingOrder:
        """정정 — **큐 맨 뒤로 갑니다.** 이게 공짜가 아니라는 것이 요점입니다."""
        if order in self.orders:
            self.orders.remove(order)
        return self.submit(new_price, order.remaining, new_level_size,
                           side=order.side, at=order.submitted_at)

    def cancel(self, order: RestingOrder) -> None:
        if order in self.orders:
            self.orders.remove(order)

    def on_trade(self, price: float, volume: float) -> list:
        """그 가격에서 체결이 일어났을 때 큐를 갱신합니다.

        Returns
        -------
        list[(order, filled_qty)] — 이번에 체결된 분량.
        """
        fills = []
        for o in list(self.orders):
            if abs(o.price - float(price)) > 1e-9 or o.remaining <= 0:
                continue
            vol = float(volume)
            if o.queue_ahead > 0:
                eaten = min(o.queue_ahead, vol)
                o.queue_ahead -= eaten
                vol -= eaten
            if vol > 0 and o.queue_ahead <= 0:
                got = min(vol, o.remaining)
                o.filled += got
                fills.append((o, got))
                if o.remaining <= 0:
                    self.orders.remove(o)
        return fills

    def on_cancel(self, price: float, volume: float) -> None:
        """그 레벨에서 취소가 일어났을 때 — 일부는 우리 앞쪽에서 빠집니다."""
        for o in self.orders:
            if abs(o.price - float(price)) <= 1e-9 and o.queue_ahead > 0:
                o.queue_ahead = max(o.queue_ahead
                                    - float(volume) * self.cancel_ratio, 0.0)

    def state(self) -> list:
        return [o.to_dict() for o in self.orders]


def conservative_fill(bar_low: float, bar_high: float, limit_price: float,
                      side: str, bar_volume: float = None,
                      level_size: float = None) -> dict:
    """L2 가 없는 시장(국내 일봉, 10단계 호가)에서 쓰는 **보수적 근사**.

        지정가: 그 봉에서 가격이 레벨을 **관통(strictly through)** 해야 체결.
                터치만 하면 체결 확률 p = clip(관통량/호가잔량, 0, 1) 로 부분체결.

    "봉의 저가가 내 매수 지정가 이하면 체결" 은 낙관적입니다 — 저가를 찍은
    그 순간 내 주문이 큐 앞쪽에 있었다는 보장이 없습니다. 관통을 요구하는
    것이 훨씬 현실에 가깝습니다.
    """
    lo, hi, lp = float(bar_low), float(bar_high), float(limit_price)
    if side == "buy":
        through = lp - lo          # 지정가 아래로 얼마나 내려갔나
        touched = lo <= lp
    else:
        through = hi - lp
        touched = hi >= lp

    if not touched:
        return {"filled": False, "fill_ratio": 0.0, "reason": "가격이 닿지 않았습니다."}
    if through > 1e-12:
        return {"filled": True, "fill_ratio": 1.0, "reason": "가격이 지정가를 관통했습니다."}

    ratio = 0.0
    if level_size and level_size > 0 and bar_volume:
        ratio = float(min(max(float(bar_volume) / float(level_size), 0.0), 1.0))
    return {"filled": ratio > 0, "fill_ratio": round(ratio, 4),
            "reason": ("터치만 했습니다 — 큐 잔량 대비 거래량으로 부분체결 "
                       "확률을 잡았습니다. 잔량 정보가 없으면 0 입니다.")}
