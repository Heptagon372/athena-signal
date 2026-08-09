"""
복권 매매 벤치마크 (Lottery Trading Benchmark)
----------------------------------------------
"내 전략이 **같은 횟수로 아무렇게나 매매한 것** 보다 나은가" 를 묻습니다.

수록 항목
    random_timing_positions   거래 횟수·보유기간을 맞춘 무작위 포지션 계열
    strategy_returns          포지션 → 비용 반영 수익률 (관측·귀무 공통 경로)
    lottery_benchmark         무작위 전략 분포 대비 관측 전략의 위치
    permutation_benchmark     횡단면: 종목 순위만 무작위로 치환
    match_profile             관측 포지션에서 거래횟수·보유기간·노출을 추출

왜 이것이 필요한가 — 실패의 원인을 구분할 수 없기 때문입니다
    전략 탐색이 실패했을 때 두 가지 해석이 가능합니다.

        (a) 시장이 효율적이라 찾을 우위가 없다
        (b) 내 탐색기가 나빠서 못 찾았다

    **매칭된 무작위 전략과 비교하지 않으면 이 둘을 구분할 수 없습니다**
    (Chen & Navet). 그리고 성공했을 때도 같은 문제가 있습니다 — 무작위 전략
    1,000개를 뽑아도 그중 최고는 꽤 좋아 보입니다. 내 전략이 그 우측 꼬리
    안에 있다면, 찾은 것은 구조가 아니라 분포의 꼬리입니다.

    "매칭" 이 핵심입니다. 무작위 전략이 관측 전략과 **거래 횟수·평균 보유기간·
    평균 노출** 이 같아야 합니다. 그래야 비교가 성립합니다. 하루 한 번 매매하는
    전략을 연 1회 매매하는 무작위 전략과 비교하면 비용 구조가 달라서 아무 의미가
    없습니다.

정직한 사전 기대
    Sullivan-Timmermann-White 가 기술적 규칙 39,832개를 정지 부트스트랩으로
    검정했을 때, **성숙 시장(DJIA, S&P500)에서는 통계적으로 유의한 규칙이
    존재하지 않았습니다.** 유의했던 곳은 신흥·젊은 시장(NASDAQ, Russell 2000)
    입니다. 즉 이 게이트는 통과하기 어렵도록 설계된 것이 아니라, 실제로 통과가
    어려운 것입니다. 실패는 정보이지 결함이 아닙니다.

이 저장소의 기존 귀무가설과의 관계
    `engine/validation.block_bootstrap` 은 **수익률의 순서** 를 파괴합니다.
    `engine/nulls.stationary_bootstrap` 은 **추정량의** 귀무분포를 만듭니다.
    이 모듈은 **전략의 타이밍** 을 무작위화합니다.

    셋은 서로 다른 질문에 답하고, 어느 것이 맞는지는 전략에 달렸습니다.
    AUTOTRADE.md 16장에 기록된 실측이 그 예입니다 — 횡단면 모멘텀에서 블록
    부트스트랩이 p≈0.5 를 낸 것은 전략이 나빠서가 아니라 **귀무가설이 틀렸기
    때문** 이었습니다. 블록 부트스트랩은 순서만 파괴하고 평균은 보존하는데,
    횡단면 모멘텀의 우위는 타이밍이 아니라 평균에 있습니다. 거기서는
    `permutation_benchmark`(신호 치환)가 옳은 귀무입니다.

    **전략이 무엇을 먹고 사는지 먼저 정하고 귀무를 고르세요.**
        타이밍을 먹는 전략   → lottery_benchmark (이 모듈의 무작위 타이밍)
        횡단면 순위를 먹는 전략 → permutation_benchmark (신호 치환)
"""

import math

import numpy as np

from engine.validation import falsification_audit, sharpe_ratio


# ---------------------------------------------------------------------------
# 관측 전략의 프로필 추출 — "무엇을 맞출 것인가"
# ---------------------------------------------------------------------------

def match_profile(positions) -> dict:
    """관측 포지션 계열에서 무작위 전략이 맞춰야 할 특성을 뽑습니다.

    맞추는 것은 세 가지입니다.
        n_trades      진입 횟수 (0 → 비0 전이)
        mean_holding  평균 보유 봉 수
        exposure      비0 포지션의 시간 비중 × 평균 크기

    노출을 맞추는 이유: 무작위 전략이 시장에 더 오래 머물면 상승장에서 그냥
    베타로 이깁니다. 그러면 검정이 전략이 아니라 노출을 비교하게 됩니다.
    """
    p = np.asarray(positions, dtype=float)
    p = np.nan_to_num(p, nan=0.0)
    n = len(p)
    if n == 0:
        return {"n": 0, "n_trades": 0, "mean_holding": 0.0, "exposure": 0.0}

    active = np.abs(p) > 1e-12
    entries = int(np.sum(active[1:] & ~active[:-1])) + int(active[0])
    n_active = int(active.sum())
    return {
        "n": n,
        "n_trades": entries,
        "mean_holding": float(n_active / entries) if entries else 0.0,
        "exposure": float(np.mean(np.abs(p))),
        "share_active": float(n_active / n),
        # 보유 중일 때의 평균 크기. 무작위 전략은 이 크기로 진입해야 노출이 맞습니다.
        "mean_size": float(np.mean(np.abs(p[active]))) if n_active else 0.0,
        "long_share": float(np.mean(p[active] > 0)) if n_active else 0.5,
    }


# ---------------------------------------------------------------------------
# 무작위 타이밍
# ---------------------------------------------------------------------------

def random_timing_positions(n: int, n_trades: int, mean_holding: float,
                            size: float = 1.0, long_share: float = 1.0,
                            rng=None) -> np.ndarray:
    """거래 횟수와 평균 보유기간을 맞춘 무작위 포지션 계열.

    진입 시점을 균등하게 뽑고, 보유기간은 평균이 `mean_holding` 인 기하분포에서
    뽑습니다. 기하분포를 쓰는 이유는 **무기억** 이기 때문입니다 — 고정 길이로
    두면 무작위 전략에 "정확히 N봉 뒤 청산" 이라는 구조가 생겨서, 그 주기가
    데이터의 주기와 맞으면 귀무분포가 이유 없이 좋아지거나 나빠집니다.

    겹치는 진입은 병합됩니다. 그래서 실현 거래 횟수가 요청보다 적을 수 있고,
    `lottery_benchmark` 는 그 실현값을 보고합니다.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = int(n)
    out = np.zeros(n, dtype=float)
    n_trades = int(max(n_trades, 0))
    if n <= 0 or n_trades <= 0:
        return out

    hold = max(float(mean_holding), 1.0)
    p = min(1.0 / hold, 1.0)
    starts = rng.integers(0, n, size=n_trades)
    lengths = rng.geometric(p, size=n_trades)

    for s, ln in zip(starts, lengths):
        e = min(int(s) + int(ln), n)
        sign = 1.0 if (long_share >= 1.0 or rng.random() < long_share) else -1.0
        out[int(s):e] = sign * float(size)
    return out


def strategy_returns(positions, asset_returns, cost_bps: float = 0.0) -> np.ndarray:
    """포지션 계열 → 비용 반영 수익률.

    **관측 전략과 무작위 전략이 반드시 이 같은 함수를 타야 합니다.** 비용 처리가
    다르면 비교가 무효입니다. 국내 주식은 매도측 거래세가 지배적이라, 비용을
    회전율에 비례시키지 않고 상수로 두면 무작위 전략이 부당하게 유리해집니다.

    비용은 포지션 **변화량** 에 부과합니다 — |Δw| × cost_bps.
    """
    p = np.asarray(positions, dtype=float)
    r = np.asarray(asset_returns, dtype=float)
    n = min(len(p), len(r))
    if n < 2:
        return np.array([], dtype=float)
    p, r = np.nan_to_num(p[:n], nan=0.0), np.nan_to_num(r[:n], nan=0.0)

    # t-1 의 포지션으로 t 의 수익률을 먹습니다 (신호는 봉 종료 후 → 다음 봉 체결)
    gross = np.zeros(n, dtype=float)
    gross[1:] = p[:-1] * r[1:]

    turnover = np.zeros(n, dtype=float)
    turnover[0] = abs(p[0])
    turnover[1:] = np.abs(np.diff(p))
    return gross - turnover * (float(cost_bps) / 1e4)


# ---------------------------------------------------------------------------
# 벤치마크
# ---------------------------------------------------------------------------

def lottery_benchmark(observed_positions, asset_returns, *, cost_bps: float = 0.0,
                      n_random: int = 1000, metric_fn=None,
                      seed: int = 20260808, periods_per_year: int = 252) -> dict:
    """관측 전략이 **매칭된 무작위 전략 분포** 의 어디에 있는가.

    Parameters
    ----------
    observed_positions : 실제 전략의 포지션 계열 (신호 × 크기, [-1, 1] 권장)
    metric_fn : `f(returns: np.ndarray) -> float`. 기본은 연율화 샤프.

    Returns
    -------
    dict — `percentile` 이 관측값의 귀무분포 내 백분위입니다.
        **95 미만이면 통과하지 못한 것입니다.** 그 경우 전략이 찾은 것은
        구조가 아니라 무작위 분포의 우측 꼬리입니다.
    """
    if metric_fn is None:
        def metric_fn(r):
            return sharpe_ratio(r, periods_per_year)

    prof = match_profile(observed_positions)
    if prof["n_trades"] < 1:
        return {"passed": None, "reason": "관측 전략에 거래가 없습니다."}

    obs_ret = strategy_returns(observed_positions, asset_returns, cost_bps)
    if len(obs_ret) < 8:
        return {"passed": None, "reason": "표본이 부족합니다."}
    observed = float(metric_fn(obs_ret))

    rng = np.random.default_rng(seed)
    nulls, realized_trades = [], []
    for _ in range(int(n_random)):
        pos = random_timing_positions(
            prof["n"], prof["n_trades"], prof["mean_holding"],
            size=prof["exposure"] / max(prof["share_active"], 1e-9)
            if prof["share_active"] > 0 else 1.0,
            long_share=prof["long_share"], rng=rng)
        rr = strategy_returns(pos, asset_returns, cost_bps)
        if len(rr) < 8:
            continue
        v = metric_fn(rr)
        if v is not None and math.isfinite(float(v)):
            nulls.append(float(v))
            realized_trades.append(match_profile(pos)["n_trades"])

    if len(nulls) < 50:
        return {"passed": None, "reason": "귀무 표본이 부족합니다."}

    arr = np.asarray(nulls, dtype=float)
    audit = falsification_audit(observed, arr, higher_is_better=True,
                                label="lottery_trading")
    pct = float(np.mean(arr < observed) * 100)

    audit.update({
        "percentile": round(pct, 2),
        "passed": bool(pct >= 95.0),
        "matched_profile": {
            "n_trades_observed": prof["n_trades"],
            "n_trades_random_mean": round(float(np.mean(realized_trades)), 1)
            if realized_trades else None,
            "mean_holding": round(prof["mean_holding"], 2),
            "exposure": round(prof["exposure"], 4),
        },
        "verdict": (
            "통과 — 관측 전략이 매칭된 무작위 전략 상위 5% 밖에 있습니다."
            if pct >= 95.0 else
            f"미통과 — 관측값이 무작위 분포의 {pct:.0f} 백분위입니다. "
            f"찾은 것은 구조가 아니라 분포의 꼬리일 수 있습니다."),
    })
    return audit


def permutation_benchmark(scores_by_date, forward_returns_by_date, *,
                          top_k: int = None, n_random: int = 300,
                          metric_fn=None, seed: int = 20260808,
                          periods_per_year: int = 252) -> dict:
    """횡단면 전략용 — **종목 순위만 무작위로 치환** 합니다.

    노출·비용·회전율을 그대로 두고 "어느 종목을 고르는가" 만 무작위화하므로,
    선택 능력을 정확히 겨냥합니다. 횡단면 모멘텀처럼 우위가 타이밍이 아니라
    평균에 있는 전략에서는 이쪽이 올바른 귀무입니다.

    Parameters
    ----------
    scores_by_date : list[np.ndarray] — 날짜별 종목 점수 (높을수록 매수)
    forward_returns_by_date : list[np.ndarray] — 같은 순서의 다음 기간 수익률
    top_k : 상위 몇 개를 담을 것인가. None 이면 종목 수의 20%.
    """
    if metric_fn is None:
        def metric_fn(r):
            return sharpe_ratio(r, periods_per_year)

    dates = list(zip(scores_by_date or [], forward_returns_by_date or []))
    dates = [(np.asarray(s, dtype=float), np.asarray(r, dtype=float))
             for s, r in dates if len(s) == len(r) and len(s) >= 4]
    if len(dates) < 8:
        return {"passed": None, "reason": "유효한 날짜가 8개 미만입니다."}

    def portfolio_returns(pick_fn) -> np.ndarray:
        out = []
        for s, r in dates:
            ok = np.isfinite(s) & np.isfinite(r)
            if int(ok.sum()) < 4:
                continue
            ss, rr = s[ok], r[ok]
            k = int(top_k or max(1, round(len(ss) * 0.2)))
            k = min(k, len(ss))
            idx = pick_fn(ss, k)
            out.append(float(np.mean(rr[idx])))
        return np.asarray(out, dtype=float)

    observed_ret = portfolio_returns(lambda s, k: np.argsort(s)[::-1][:k])
    if len(observed_ret) < 8:
        return {"passed": None, "reason": "표본이 부족합니다."}
    observed = float(metric_fn(observed_ret))

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(int(n_random)):
        rr = portfolio_returns(lambda s, k: rng.choice(len(s), size=k, replace=False))
        if len(rr) < 8:
            continue
        v = metric_fn(rr)
        if v is not None and math.isfinite(float(v)):
            nulls.append(float(v))

    if len(nulls) < 30:
        return {"passed": None, "reason": "귀무 표본이 부족합니다."}

    arr = np.asarray(nulls, dtype=float)
    audit = falsification_audit(observed, arr, higher_is_better=True,
                                label="signal_permutation")
    pct = float(np.mean(arr < observed) * 100)
    audit.update({"percentile": round(pct, 2),
                  "passed": bool(audit.get("p_value") is not None
                                 and audit["p_value"] <= 0.05),
                  "n_dates": len(observed_ret)})
    return audit
