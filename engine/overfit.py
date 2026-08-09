"""
과최적화 정량화 (Backtest Overfitting)
--------------------------------------
"이 백테스트를 얼마나 깎아서 믿어야 하는가" 를 숫자로 답합니다.

수록 항목
    pbo_cscv              CSCV 로 계산한 과최적화 확률 PBO
    expected_max_z        시행 N 에서 무능력 귀무의 기대 최대 z
    min_backtest_length   시행 N 을 정당화하려면 몇 년치가 필요한가
    max_configs           보유한 데이터 길이로 정직하게 시험 가능한 설정 수
    required_tstat        Harvey & Liu 요구 t통계량
    haircut_sharpe        다중검정 보정 후 샤프와 삭감률
    bhy_fdr               Benjamini-Hochberg-Yekutieli (Bonferroni 아님)

기존 engine/validation.py 와의 분업
    validation.py 는 **한 전략** 을 의심합니다 (PSR, DSR, 위조검증).
    이 모듈은 **탐색 절차 전체** 를 의심합니다. 같은 데이터에서 40개 설정을
    시험했다면 개별 설정의 DSR 만으로는 부족합니다 — 절차 자체가 과적합을
    생산하고 있는지 봐야 합니다. PBO 가 그 질문에 답합니다.

읽기 전에 알아야 할 숫자들
    · 순수 노이즈에 10번 시도하면 최고 인샘플 샤프가 **1.57** 이 나옵니다.
      진짜 샤프가 0인데도요.
    · N=1,000 시행이면 노이즈만으로 기대 최대 샤프가 **3.0 을 넘습니다.**
      → 백테스트 샤프 3.0 이상이면 데이터 누수를 의심하는 것이 통계적으로 정당합니다.
    · 표준 t=2.0 은 **틀렸습니다.** N=10 이면 2.8, N=200 이면 3.66, N=1,000 이면 4.1.
    · **인샘플 1위를 고르는 행위 자체** 가 아웃샘플 성과를 깎습니다. 1위성의
      상당 부분이 운이기 때문입니다. 이것이 PBO 가 재는 것이고, 인샘플을
      공격적으로 최적화할수록 이 손실이 커집니다.
    · López de Prado: **약 20회 반복이면** 표준 유의수준에서 거짓 발견이 나옵니다.
      그 지점부터는 "최적화" 가 아니라 통계적 기준으로 부정행위입니다.
"""

import math
from itertools import combinations

import numpy as np

from engine.validation import _norm_ppf
from engine.quant import _norm_cdf

_EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# PBO — Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------

def pbo_cscv(returns_matrix, n_blocks: int = 16, max_combos: int = 20000,
             seed: int = 20260808) -> dict | None:
    """CSCV 로 과최적화 확률을 계산합니다.

    Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest
    Overfitting* (CSCV).

    Parameters
    ----------
    returns_matrix : (T, N) 배열
        T 개 시점 × N 개 **설정** 의 수익률. 열 하나가 설정 하나입니다.
        같은 데이터에서 시험한 모든 설정을 넣으세요 — 살아남은 것만 넣으면
        그 선별 자체가 이미 과적합이라 PBO 가 0에 가깝게 나옵니다.

    절차
        1) T 를 S 개 블록으로 나눈다
        2) C(S, S/2) 가지 방법으로 절반을 인샘플, 나머지를 아웃샘플로 삼는다
        3) 인샘플 최고 설정 n* 를 고른다
        4) n* 가 아웃샘플에서 **몇 등** 인지 본다
        5) PBO = n* 가 아웃샘플 중앙값 아래로 떨어질 확률

    해석
        PBO > 0.5  → 인샘플 최고를 고르는 행위가 동전던지기보다 나쁩니다
        PBO > 0.05 → 기각 (지시서 기준)
        PBO ≈ 0    → 설정 선택이 실제로 정보를 담고 있습니다

    S=16 이면 C(16,8) = 12,870 조합입니다. 그보다 커지면 무작위로
    `max_combos` 개만 뽑고 그 사실을 보고합니다.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] < 32 or M.shape[1] < 2:
        return None
    T, N = M.shape

    S = int(n_blocks)
    if S % 2 == 1:
        S -= 1
    S = max(min(S, T // 4), 2)
    if S < 2:
        return None

    # 블록별 통계를 미리 접어 둡니다 — 조합마다 원본을 다시 훑지 않기 위해서입니다
    usable = (T // S) * S
    blocks = M[:usable].reshape(S, usable // S, N)
    blk_n = blocks.shape[1]
    blk_sum = blocks.sum(axis=1)                 # (S, N)
    blk_sq = (blocks ** 2).sum(axis=1)           # (S, N)

    combos = list(combinations(range(S), S // 2))
    sampled = False
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combos), size=max_combos, replace=False)
        combos = [combos[i] for i in idx]
        sampled = True

    # (C, S) 지시행렬 → 행렬곱 한 번으로 모든 조합의 IS 통계를 얻습니다
    C = len(combos)
    ind = np.zeros((C, S), dtype=float)
    for i, c in enumerate(combos):
        ind[i, list(c)] = 1.0

    n_is = blk_n * (S // 2)
    is_sum = ind @ blk_sum                        # (C, N)
    is_sq = ind @ blk_sq
    oos_sum = blk_sum.sum(axis=0)[None, :] - is_sum
    oos_sq = blk_sq.sum(axis=0)[None, :] - is_sq

    def _sharpe(s, sq, n):
        mean = s / n
        var = sq / n - mean ** 2
        sd = np.sqrt(np.maximum(var, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sd > 1e-12, mean / sd, 0.0)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    is_perf = _sharpe(is_sum, is_sq, n_is)        # (C, N)
    oos_perf = _sharpe(oos_sum, oos_sq, n_is)

    best = np.argmax(is_perf, axis=1)             # (C,)
    rows = np.arange(C)
    chosen_oos = oos_perf[rows, best]

    # 아웃샘플에서의 상대 순위 ω ∈ (0, 1)
    rank = (oos_perf < chosen_oos[:, None]).sum(axis=1)
    omega = (rank + 1.0) / (N + 1.0)
    omega = np.clip(omega, 1e-6, 1 - 1e-6)
    logits = np.log(omega / (1.0 - omega))

    pbo = float(np.mean(logits <= 0.0))
    return {
        "pbo": round(pbo, 4),
        "passed": bool(pbo <= 0.05),
        "n_blocks": S, "n_combos": C, "n_configs": N, "n_obs": T,
        "combos_sampled": sampled,
        "logit_mean": round(float(np.mean(logits)), 4),
        "logit_median": round(float(np.median(logits)), 4),
        "chosen_oos_mean": round(float(np.mean(chosen_oos)), 4),
        "chosen_oos_negative_share": round(float(np.mean(chosen_oos < 0)), 4),
        "verdict": (
            "통과 — 인샘플 선택이 아웃샘플로 이어집니다."
            if pbo <= 0.05 else
            f"기각 — 인샘플 최고 설정이 아웃샘플 중앙값 아래로 떨어질 확률이 "
            f"{pbo:.0%} 입니다. 설정 선택 절차가 노이즈를 고르고 있습니다."),
    }


# is_oos_degradation() 은 의도적으로 넣지 않았습니다
# ---------------------------------------------------------------------------
# 설정별 인샘플 샤프를 아웃샘플 샤프에 회귀해 "인샘플 우위가 이어지는가" 를
# 보는 진단을 흔히 씁니다. **구현했다가 뺐습니다.** 측정하려는 것을 측정하지
# 못하기 때문입니다.
#
#   순수 노이즈 50개 설정 × 1,000봉에서 이 기울기가 **+0.23** 이 나옵니다.
#   원인은 알파가 아니라 표본입니다 — 한 설정의 IS 절반과 OOS 절반은 같은
#   열에서 나오므로 **그 열의 실현 표본평균을 공유** 합니다. 설정 간에 실현
#   평균이 흩어져 있으면 두 절반이 함께 높거나 함께 낮아 기울기가 양수가 됩니다.
#   진짜 우위를 심어도 +0.25 로, 노이즈와 구분되지 않았습니다.
#
# 즉 "우위가 이어진다" 와 "같은 유한표본을 나눠 썼다" 가 분리되지 않습니다.
# 같은 질문에 대해 pbo_cscv() 는 제대로 답합니다(위 실험에서 노이즈 0.40 vs
# 우위 0.007). 맞는 지표 옆에 미묘하게 틀린 지표를 두면 사람은 편한 쪽을
# 읽으므로, 하나만 둡니다.
#
# ---------------------------------------------------------------------------
# 최소 백테스트 길이
# ---------------------------------------------------------------------------

def expected_max_z(n_trials: int) -> float:
    """무능력 귀무에서 N 번 시도했을 때 기대되는 최대 z.

        E[max z] ≈ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)),  γ = 0.5772

    `validation.expected_max_sharpe_under_null` 은 실제 시행 샤프의 분산을
    쓰지만, 이쪽은 **N 만으로** 계산합니다. 아직 돌려보지 않은 상태에서
    "몇 개까지 시험해도 되는가" 를 정할 때 필요합니다.
    """
    n = int(n_trials)
    if n < 2:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return float((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def min_backtest_length(n_trials: int, target_sharpe: float = 1.0) -> dict:
    """N 개 설정을 시험하려면 최소 몇 년치 데이터가 필요한가.

    Bailey, Borwein, López de Prado & Zhu, *Pseudo-Mathematics and Financial
    Charlatanism* (AMS Notices).

        MinBTL(년) ≈ (E[max z])² / SR_target²

    발표된 기준점과 일치합니다: 5년 → 45개 설정, 2년 → 7개 설정 (SR=1 기준).

    **이 표가 말하는 것**: 2년치 데이터로 20개 설정을 시험했다면, 그중 최고가
    좋아 보이는 것은 당연합니다. 데이터가 그만큼의 탐색을 지탱하지 못합니다.
    """
    sr = float(target_sharpe)
    if sr <= 0:
        return {"years": None, "reason": "target_sharpe 는 양수여야 합니다."}
    e = expected_max_z(n_trials)
    years = (e ** 2) / (sr ** 2)
    return {
        "n_trials": int(n_trials), "target_sharpe": sr,
        "expected_max_z": round(e, 4),
        "min_years": round(years, 2),
        "min_trading_days": int(round(years * 252)),
    }


def max_configs(years: float, target_sharpe: float = 1.0,
                cap: int = 100000) -> dict:
    """보유한 데이터 길이로 **정직하게** 시험 가능한 설정 수.

    `min_backtest_length` 의 역함수를 수치적으로 풉니다. 이 수를 넘겨 탐색했다면
    DSR·PBO 로 깎아도 결론이 남지 않을 가능성이 높습니다.
    """
    y = float(years)
    if y <= 0:
        return {"max_configs": 0}
    target = y * (float(target_sharpe) ** 2)
    lo, hi = 2, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if expected_max_z(mid) ** 2 <= target:
            lo = mid
        else:
            hi = mid - 1
    return {"years": y, "target_sharpe": target_sharpe, "max_configs": int(lo)}


# ---------------------------------------------------------------------------
# 다중검정 보정 — Harvey & Liu
# ---------------------------------------------------------------------------

# Harvey & Liu (2015), *Evaluating Trading Strategies* 의 발표된 기준점.
# 그 사이는 log(N) 선형보간입니다 (원표가 이산 격자로만 주어져 있습니다).
_HL_ANCHORS = [(10, 2.8), (200, 3.66), (1000, 4.1)]


def required_tstat(n_trials: int) -> float:
    """시행 N 에서 유의하다고 말하려면 필요한 t통계량.

    **표준 t=2.0 은 틀렸습니다.** 시행이 늘수록 문턱이 올라갑니다.

        N=10 → 2.8   N=200 → 3.66   N=1000 → 4.1
    """
    n = max(int(n_trials), 1)
    if n <= _HL_ANCHORS[0][0]:
        return _HL_ANCHORS[0][1]
    if n >= _HL_ANCHORS[-1][0]:
        # 바깥은 완만하게 외삽합니다 — 원표 범위 밖이므로 참고치입니다
        (n0, t0), (n1, t1) = _HL_ANCHORS[-2], _HL_ANCHORS[-1]
        slope = (t1 - t0) / (math.log(n1) - math.log(n0))
        return float(t1 + slope * (math.log(n) - math.log(n1)))
    for (n0, t0), (n1, t1) in zip(_HL_ANCHORS, _HL_ANCHORS[1:]):
        if n0 <= n <= n1:
            w = (math.log(n) - math.log(n0)) / (math.log(n1) - math.log(n0))
            return float(t0 + w * (t1 - t0))
    return _HL_ANCHORS[-1][1]


def _sr_to_pvalue(sr: float, n_obs: int, scale: float) -> tuple:
    """(주기 단위 샤프, t통계, 양측 p값)."""
    sr_p = float(sr) / scale
    t = sr_p * math.sqrt(n_obs)
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return sr_p, t, min(max(p, 1e-300), 1.0)


def haircut_sharpe(observed_sr: float, n_trials: int, n_obs: int,
                   periods_per_year: int = 252, annualized: bool = True,
                   method: str = "bonferroni", all_trial_sharpes=None) -> dict:
    """다중검정 보정 후 샤프와 삭감률.

    **삭감은 비선형입니다.** "백테스트 샤프를 반으로 나눠라" 는 통념은 양방향
    모두 틀립니다 — 한계 전략은 100% 까지 깎이고(즉 남는 게 없고), 강한 전략은
    상대적으로 조금만 깎입니다.

    method
        "bonferroni" (기본) — p_adj = p · N. 순위를 몰라도 계산되는 보수적 경계.
        "holm"              — 1위 가정에서는 Bonferroni 와 동일합니다.
        "bhy"               — **`all_trial_sharpes` 를 함께 넘겨야 제대로 됩니다.**

    ⚠ BHY 를 쓸 때의 함정
        BHY(FDR)는 전략들이 서로 상관되어 있을 때 Bonferroni 보다 **관대한** 것이
        장점입니다. 그런데 관측치 하나와 N 만 아는 상태에서는 "내 전략이 N개 중
        1위" 라고 가정할 수밖에 없고, 그 경우 BHY 의 조정 p값은 p·N·c(N) 이 되어
        오히려 Bonferroni 보다 **c(N)배 더 보수적** 이 됩니다. 방향이 뒤집힙니다.

        그래서 시행 전체의 샤프를 `all_trial_sharpes` 로 넘기세요. 그러면 실제
        순위로 BHY 절차를 돌려 본래의 관대함을 얻습니다.
        `engine.registry.TrialRegistry.sharpes()` 가 그 목록을 줍니다.

    삭감률이 100% 로 포화하는 것은 버그가 아닙니다 — 그 시행 수에서는 관측
    t통계가 문턱에 못 미쳐 **남는 것이 없다** 는 뜻입니다. `required_tstat` 과
    함께 읽으세요.
    """
    if n_obs < 3 or n_trials < 1:
        return {"haircut_sr": None, "reason": "표본 또는 시행 수가 부족합니다."}

    scale = math.sqrt(periods_per_year) if annualized else 1.0
    _, t_stat, p_single = _sr_to_pvalue(observed_sr, n_obs, scale)
    n = int(n_trials)
    note = None

    if method == "bhy" and all_trial_sharpes is not None:
        trials = [float(s) for s in all_trial_sharpes
                  if s is not None and math.isfinite(float(s))]
        if len(trials) >= 2:
            pvals = [_sr_to_pvalue(s, n_obs, scale)[2] for s in trials]
            # 관측치가 목록에 없으면 추가해서 함께 순위를 매깁니다
            if not any(abs(s - float(observed_sr)) < 1e-12 for s in trials):
                pvals.append(p_single)
            arr = np.sort(np.asarray(pvals, dtype=float))
            m = len(arr)
            c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
            rank = int(np.searchsorted(arr, p_single, side="left")) + 1
            # BHY step-up 조정 p값 (해당 순위 이상에서의 최솟값)
            j = np.arange(rank, m + 1)
            p_adj = float(np.min(np.minimum(1.0, arr[rank - 1:] * m * c_m / j)))
            note = f"BHY 절차를 실제 순위({rank}/{m})로 적용했습니다."
        else:
            c_n = sum(1.0 / i for i in range(1, n + 1)) if n > 1 else 1.0
            p_adj = min(1.0, p_single * n * c_n)
            note = "시행 목록이 부족해 1위 가정의 보수적 경계를 썼습니다."
    elif method == "bhy":
        c_n = sum(1.0 / i for i in range(1, n + 1)) if n > 1 else 1.0
        p_adj = min(1.0, p_single * n * c_n)
        note = ("all_trial_sharpes 없이 BHY 를 쓰면 1위 가정이라 Bonferroni 보다 "
                "보수적입니다. 시행 목록을 넘기세요.")
    else:  # bonferroni / holm — 1위 가정에서 동일
        p_adj = min(1.0, p_single * n)

    t_adj = max(_norm_ppf(1.0 - p_adj / 2.0), 0.0)
    sr_adj = (t_adj / math.sqrt(n_obs)) * scale
    cut = 1.0 - (sr_adj / observed_sr) if abs(observed_sr) > 1e-12 else 1.0

    out = {
        "observed_sr": round(float(observed_sr), 4),
        "haircut_sr": round(float(sr_adj), 4),
        "haircut_pct": round(float(min(max(cut, 0.0), 1.0)) * 100, 1),
        "t_stat": round(float(t_stat), 3),
        "required_tstat": round(required_tstat(n), 3),
        "passes_required_t": bool(abs(t_stat) >= required_tstat(n)),
        "p_single": float(f"{p_single:.3e}"),
        "p_adjusted": float(f"{p_adj:.3e}"),
        "saturated": bool(p_adj >= 1.0),
        "method": method, "n_trials": n, "n_obs": int(n_obs),
    }
    if note:
        out["note"] = note
    return out


def bhy_fdr(p_values, alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg-Yekutieli — 임의 의존 구조에서의 FDR 통제.

    **Bonferroni 가 아니라 이것을 쓰라는 것이 저자 권고입니다.** 전략들의
    수익률은 서로 강하게 상관되어 있어 Bonferroni 는 지나치게 보수적입니다.
    Yekutieli 보정 c(N) = Σ 1/i 가 그 의존성을 흡수합니다.

    Returns
    -------
    dict — `rejected` 가 유의하다고 판정된 인덱스입니다.
    """
    p = np.asarray([v for v in (p_values or []) if v is not None], dtype=float)
    p = p[np.isfinite(p)]
    n = len(p)
    if n == 0:
        return {"n": 0, "rejected": [], "threshold": None}

    order = np.argsort(p)
    sorted_p = p[order]
    c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    ranks = np.arange(1, n + 1)
    crit = (ranks / (n * c_n)) * float(alpha)

    below = np.flatnonzero(sorted_p <= crit)
    if len(below) == 0:
        return {"n": n, "rejected": [], "threshold": None, "c_n": round(c_n, 4),
                "alpha": alpha, "n_rejected": 0}
    k = int(below[-1])
    thresh = float(sorted_p[k])
    rejected = sorted(int(i) for i in order[: k + 1])
    return {"n": n, "n_rejected": len(rejected), "rejected": rejected,
            "threshold": thresh, "c_n": round(c_n, 4), "alpha": alpha}
