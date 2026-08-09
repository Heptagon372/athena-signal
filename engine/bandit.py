"""
메타컨트롤러와 T5 결정 실험 (Bandits & the T5 Gate)
----------------------------------------------------
"여러 전략 중 지금 무엇을 쓸 것인가" 를 고르는 계층입니다. 그리고 그보다 먼저,
**그 계층을 지을 값어치가 있는지** 를 하루 만에 판정하는 실험을 제공합니다.

수록 항목
    LinTS                 선형 톰슨 샘플링 (기본)
    LinUCB                선형 신뢰상한
    NonContextualUCB      ★ 국면 피처를 안 보는 UCB — 이것이 넘어야 할 벽
    EqualWeight           균등가중 (자주 못 이깁니다)
    BestExPost            사후 최적 단일 팔 (도달 불가 상한)
    DelayedRewardBuffer   t 의 결정에 t+h 의 보상을 귀속
    run_t5_experiment     ★ 컨텍스추얼이 비컨텍스추얼을 이기는가

★ 왜 이 파일의 핵심이 밴딧이 아니라 T5 인가
    설계 문서가 공통으로 지목하는 단 하나의 결정적 실험이 이것입니다.

    > 컨텍스추얼 밴딧이 **비컨텍스추얼 UCB를 못 이기면 경제물리학 계층 전체가
    > 죽은 무게다.** 이 실험을 GA·RL 코드를 한 줄 쓰기 전에 먼저 돌린다.
    > 비용은 하루, 정보 가치는 프로젝트 전체다.

    논리는 단순합니다. 컨텍스추얼 밴딧은 국면 컨텍스트 x_t 를 보고 팔을 고르고,
    비컨텍스추얼 UCB 는 그냥 "평균적으로 제일 좋았던 팔" 을 고릅니다.
    전자가 후자를 못 이긴다면 **x_t 에 정보가 없다는 뜻** 입니다. 그러면
    Hurst·프랙탈·엔트로피·국면 라벨을 아무리 정교하게 만들어도 소용없습니다.

    **이 저장소에는 이미 방증이 있습니다.** AUTOTRADE.md 16장의 288개 설정
    실험에서 경제물리 오버레이의 주변효과가 DFA 0, Hill 소폭 마이너스,
    LPPLS 0, RMT 스트레스 게이트는 크게 해로웠습니다. T5 는 그 결론을
    **밴딧 형태로 다시, 더 싸게** 확인하는 장치입니다.

    T5 실패 시 처방: 경제물리 계층을 버리고 **실현변동성 단일 축** 으로 갑니다.
    나머지 아키텍처는 그대로 유효합니다. 이것은 실패가 아니라 수개월을 아낀 것입니다.

    **게이트의 실측 성질** (합성 데이터 8개 실현, T=1500·K=4·d=2):
        컨텍스트가 최적 팔을 결정하는 경우  → 7/8 통과 (검정력)
        컨텍스트가 순수 노이즈인 경우      → **0/8 통과 (거짓양성 0)**
    비대칭이 의도된 것입니다. 이 게이트의 임무는 "지을 값어치 없는 계층을
    짓지 않게 막는 것" 이므로, 놓치는 쪽보다 **통과시키는 쪽 실수가 훨씬
    비쌉니다.** 보수적으로 기울여 두었습니다.

    ⚠ 구현하며 실제로 겪은 두 함정 — 직접 실험을 짤 때 반복하지 마세요
      1) **보상 스케일.** LinTS 의 v=0.25, UCB 의 c=1.0 은 보상이 O(1) 이라는
         가정입니다. 일간 수익률 O(0.01) 을 그대로 넣으면 탐색항이 신호보다
         25~100배 커져 두 컨트롤러가 사실상 무작위로 팔을 고릅니다. 그 비교는
         데이터가 아니라 난수를 재는 것입니다. → 내부에서 보상을 표준화합니다.
      2) **시드로 짝지은 t검정.** 비컨텍스추얼 UCB 와 균등가중은 결정적이라
         시드 간 분산이 0 입니다. 시드로 대응표본 검정을 하면 차이의 분산이
         LinTS 내부 난수에서만 나와 |t| 가 폭발하고, 데이터 실현이 조금만
         바뀌어도 **부호가 뒤집힙니다.** → 시간 **구간** 으로 짝짓습니다.

[MUST] 제약 — 지키지 않으면 리그렛 바운드가 공허해집니다
    d ≤ 5 : 일봉 5년 = 결정 1,250회. LinUCB 리그렛 Õ(d√T) 에서
            d=10 이면 10·√1250 ≈ 354 로 라운드당 평균 리그렛이 보상 범위의
            ~28% 입니다. 거의 아무 말도 하지 않는 바운드입니다. d=3 이면 ~8.5%.
            **경제물리 측도 10개를 그대로 먹이지 말고 2~3개 팩터로 압축하세요.**
    K ≤ 8 : 팔이 서로 진짜로 달라야 합니다. 상관 높은 팔은 탐색만 낭비합니다.
    종목 풀링: 가장 레버리지 높은 설계 변경입니다. 50종목이 밴딧 하나를
            공유하면 T = 62,500 이 됩니다.
"""

import math
from dataclasses import dataclass, field

import numpy as np


class MetaController:
    """모든 메타컨트롤러의 공통 인터페이스."""

    name = "base"

    def select(self, ctx: np.ndarray, t=None) -> int:
        raise NotImplementedError

    def observe(self, arm: int, reward: float, ctx: np.ndarray = None) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 컨텍스추얼
# ---------------------------------------------------------------------------

class LinTS(MetaController):
    """선형 톰슨 샘플링. **기본값으로 씁니다.**

    각 팔 a 에 대해 사후 `θ̃_a ~ N(A_a⁻¹b_a, v²A_a⁻¹)` 에서 표본을 뽑고
    `argmax θ̃_aᵀx` 를 고릅니다.

    LinUCB 보다 이쪽을 기본으로 두는 이유: **지연·배치 피드백을 더 우아하게
    다룹니다.** 우리 보상은 h기 뒤에 도착하므로 그 사이 여러 결정이 같은
    사후분포에서 이뤄지는데, 톰슨 샘플링은 매번 다른 표본을 뽑아 자연스럽게
    탐색을 유지합니다. UCB 는 사후가 안 바뀌면 같은 팔만 반복합니다.
    """

    name = "LinTS"

    def __init__(self, n_arms: int, dim: int, v: float = 0.25,
                 lam: float = 1.0, seed: int = 20260808):
        if dim > 5:
            raise ValueError(
                f"컨텍스트 차원 {dim} > 5. 리그렛 바운드가 공허해집니다 — "
                f"직교 팩터 2~3개로 압축하세요 (모듈 docstring 참조).")
        self.k, self.d, self.v = int(n_arms), int(dim), float(v)
        self.A = [np.eye(self.d) * float(lam) for _ in range(self.k)]
        self.b = [np.zeros(self.d) for _ in range(self.k)]
        self.rng = np.random.default_rng(seed)

    def select(self, ctx, t=None) -> int:
        x = np.asarray(ctx, dtype=float).ravel()[: self.d]
        best, best_val = 0, -np.inf
        for a in range(self.k):
            inv = np.linalg.inv(self.A[a])
            mu = inv @ self.b[a]
            try:
                theta = self.rng.multivariate_normal(mu, (self.v ** 2) * inv)
            except np.linalg.LinAlgError:
                theta = mu
            val = float(theta @ x)
            if val > best_val:
                best, best_val = a, val
        return best

    def observe(self, arm: int, reward: float, ctx=None) -> None:
        x = np.asarray(ctx, dtype=float).ravel()[: self.d]
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x


class LinUCB(MetaController):
    """선형 신뢰상한. `p_a = θ_aᵀx + α√(xᵀA_a⁻¹x)`."""

    name = "LinUCB"

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0,
                 lam: float = 1.0):
        if dim > 5:
            raise ValueError(f"컨텍스트 차원 {dim} > 5.")
        self.k, self.d, self.alpha = int(n_arms), int(dim), float(alpha)
        self.A = [np.eye(self.d) * float(lam) for _ in range(self.k)]
        self.b = [np.zeros(self.d) for _ in range(self.k)]

    def select(self, ctx, t=None) -> int:
        x = np.asarray(ctx, dtype=float).ravel()[: self.d]
        best, best_val = 0, -np.inf
        for a in range(self.k):
            inv = np.linalg.inv(self.A[a])
            theta = inv @ self.b[a]
            val = float(theta @ x) + self.alpha * math.sqrt(max(x @ inv @ x, 0.0))
            if val > best_val:
                best, best_val = a, val
        return best

    def observe(self, arm: int, reward: float, ctx=None) -> None:
        x = np.asarray(ctx, dtype=float).ravel()[: self.d]
        self.A[arm] += np.outer(x, x)
        self.b[arm] += float(reward) * x


# ---------------------------------------------------------------------------
# ★ 상시 베이스라인 — 모든 리포트에 강제로 포함됩니다
# ---------------------------------------------------------------------------

class NonContextualUCB(MetaController):
    """★ 국면 피처를 **보지 않는** UCB. 이것이 넘어야 할 벽입니다.

    `select()` 가 ctx 를 받긴 하지만 **쓰지 않습니다.** 의도적입니다.
    컨텍스추얼 밴딧이 이걸 못 이기면 국면 컨텍스트에 정보가 없는 것입니다.
    """

    name = "UCB(비컨텍스추얼)"

    def __init__(self, n_arms: int, c: float = 1.0):
        self.k, self.c = int(n_arms), float(c)
        self.counts = np.zeros(self.k)
        self.means = np.zeros(self.k)
        self.t = 0

    def select(self, ctx=None, t=None) -> int:
        self.t += 1
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            return int(untried[0])
        bonus = self.c * np.sqrt(2.0 * math.log(self.t) / self.counts)
        return int(np.argmax(self.means + bonus))

    def observe(self, arm: int, reward: float, ctx=None) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (float(reward) - self.means[arm]) / n


class EqualWeight(MetaController):
    """모든 팔을 균등하게 씁니다. **자주 못 이깁니다** — 그게 요점입니다."""

    name = "균등가중"

    def __init__(self, n_arms: int):
        self.k = int(n_arms)
        self.i = -1

    def select(self, ctx=None, t=None) -> int:
        self.i = (self.i + 1) % self.k
        return self.i

    def observe(self, arm: int, reward: float, ctx=None) -> None:
        pass


class BestExPost:
    """사후 최적 단일 팔. **도달 불가능한 상한** 이므로 컨트롤러가 아닙니다.

    시점마다 고르는 게 아니라 전 구간을 다 보고 가장 좋았던 팔 하나를 씁니다.
    컨텍스추얼 밴딧이 이걸 넘으면 (원리상 가능합니다 — 국면마다 다른 팔을 쓰니까)
    컨텍스트가 진짜로 정보를 담고 있다는 강한 증거입니다.
    """

    name = "사후최적 단일팔"

    @staticmethod
    def evaluate(rewards: np.ndarray) -> tuple:
        """rewards: (T, K). (총보상, 최적 팔 인덱스)"""
        totals = np.nansum(rewards, axis=0)
        best = int(np.argmax(totals))
        return float(totals[best]), best


# ---------------------------------------------------------------------------
# 지연 보상
# ---------------------------------------------------------------------------

@dataclass
class DelayedRewardBuffer:
    """t 의 결정에 대한 보상이 t+h 에 도착하는 상황을 다룹니다.

    **이걸 안 하면 밴딧이 미래를 봅니다.** t 시점에 t 의 보상을 즉시 학습시키면
    그 보상은 [t, t+h] 구간의 수익률이므로 t 시점에 알 수 없는 값입니다.
    백테스트에서 밴딧이 비현실적으로 잘 나오는 가장 흔한 원인입니다.

    그리고 **CPCV purge 길이에 이 h 를 포함해야 합니다**
    (`cpcv.combinatorial_purged_cv(horizon=h)`).
    """
    horizon: int = 5
    _pending: list = field(default_factory=list)

    def push(self, t: int, arm: int, ctx, reward: float) -> None:
        self._pending.append((int(t) + int(self.horizon), arm, ctx, float(reward)))

    def pop_ready(self, now: int) -> list:
        """지금 학습에 써도 되는 (arm, ctx, reward) 목록."""
        ready = [(a, c, r) for due, a, c, r in self._pending if due <= now]
        self._pending = [p for p in self._pending if p[0] > now]
        return ready


# ---------------------------------------------------------------------------
# ★ T5 — 결정적 실험
# ---------------------------------------------------------------------------

def run_t5_experiment(contexts, rewards, *, horizon: int = 5,
                      n_seeds: int = 20, controllers=None,
                      standardize: bool = True, time_index=None) -> dict:
    """★ 컨텍스추얼 밴딧이 비컨텍스추얼 UCB 를 이기는가.

    Parameters
    ----------
    contexts : (T, d) — 시점별 국면 컨텍스트. **d ≤ 5.**
        `regime.context_vector()` 나 실현변동성 분위 + VR z 같은 것.
    rewards : (T, K) — 시점 t 에 팔 a 를 골랐을 때 받는 보상.
        **비용 반영 후 수익률** 이어야 합니다. 총수익률로 돌리면 회전율 높은
        팔이 부당하게 이깁니다.
    horizon : 보상 지연 h. 팔을 고른 뒤 h기 후에 보상을 학습합니다.
    time_index : (T,) 각 행의 **시간 좌표**(날짜 서수 등).
        ★ **종목 풀링 패널이면 반드시 넘기세요.**
        생략하면 지연을 행 인덱스로 셉니다. 그런데 날짜 우선으로 19종목을
        쌓은 패널에서는 5행 = 5/19 ≈ 0.26 거래일이라, **5거래일 선행 수익률을
        같은 날 안에 학습** 하게 됩니다 — 그대로 미래 참조입니다.
        실제로 이 버그로 비컨텍스추얼 UCB 가 *사후 최적 단일팔보다* 높은
        보상을 냈고, 그 모순이 버그를 드러냈습니다. 도달 불가능한 상한을
        넘는 컨트롤러가 나오면 타이밍을 의심하세요.
    n_seeds : 시드를 바꿔 반복. 밴딧은 초기 탐색 운에 민감해서 단일 시드
        결과는 신뢰할 수 없습니다.

    Returns
    -------
    dict — `verdict` 가 판정입니다. `passed=False` 면 **국면 계층을 짓지 마세요.**

    해석 규칙
        LinTS 가 비컨텍스추얼 UCB 를 유의하게 이기지 못하면 → 컨텍스트에 정보 없음
        → 경제물리 계층을 버리고 실현변동성 단일 축으로. 나머지 설계는 그대로 유효.
    """
    X = np.asarray(contexts, dtype=float)
    R = np.asarray(rewards, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if R.ndim != 2 or X.shape[0] != R.shape[0]:
        return {"passed": None, "reason": "contexts 와 rewards 의 길이가 다릅니다."}
    T, K = R.shape
    d = X.shape[1]
    if d > 5:
        return {"passed": None,
                "reason": f"컨텍스트 차원 {d} > 5 — 먼저 압축하세요."}
    if T < 200 or K < 2:
        return {"passed": None, "reason": "표본 또는 팔이 부족합니다."}

    if standardize:
        mu, sd = X.mean(axis=0), X.std(axis=0)
        X = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
    # 절편 항 — 이게 없으면 컨텍스추얼 밴딧이 팔의 기본 성능을 표현하지 못해
    # 비컨텍스추얼 UCB 에게 부당하게 집니다. 공정한 비교의 전제입니다.
    X = np.hstack([X, np.ones((T, 1))])
    dim = X.shape[1]

    # ★ 보상 스케일 정규화 — 이걸 빼먹으면 실험이 통째로 무의미해집니다.
    #   LinTS 의 사후표본 분산은 v², UCB 의 탐색 보너스는 c·√(2ln t/n) 인데,
    #   기본값 v=0.25 · c=1.0 은 **보상이 O(1)** 이라고 가정한 값입니다.
    #   일간 수익률은 O(0.01) 이라 그대로 넣으면 탐색항이 신호보다 25~100배
    #   커져서 **두 컨트롤러 모두 사실상 무작위로 팔을 고릅니다.**
    #   그 상태의 비교 결과는 데이터가 아니라 난수를 재는 것입니다.
    r_scale = float(np.nanstd(R))
    R_n = R / r_scale if r_scale > 1e-12 else R

    # 지연을 재는 시간 좌표. 종목 풀링 패널이면 **날짜** 여야 합니다.
    if time_index is None:
        tkey = np.arange(T)
        pooled_warning = None
    else:
        tkey = np.asarray(time_index)
        if len(tkey) != T:
            return {"passed": None, "reason": "time_index 길이가 다릅니다."}
        uniq = len(np.unique(tkey))
        pooled_warning = (None if uniq == T else
                          f"시간축 {uniq}개 시점에 {T}개 행 "
                          f"(시점당 평균 {T / uniq:.1f}종목) — 풀링 패널로 처리합니다.")

    def _run(make, seed):
        """컨트롤러 하나를 끝까지 돌리고 **시점별 실현 보상** 을 돌려줍니다."""
        ctrl = make(seed)
        buf = DelayedRewardBuffer(horizon=horizon)
        path = np.zeros(T, dtype=float)
        for t in range(T):
            ctx = X[t]
            a = ctrl.select(ctx, t)
            path[t] = float(R_n[t, a])
            # 지연은 **시간 좌표** 기준. 행 인덱스로 세면 풀링 패널에서
            # h 행이 h/종목수 거래일이 되어 미래를 봅니다.
            buf.push(tkey[t], a, ctx, float(R_n[t, a]))
            for arm, c, r in buf.pop_ready(tkey[t]):
                ctrl.observe(arm, r, c)
        return path

    if controllers is None:
        controllers = {
            "LinTS": lambda s: LinTS(K, dim, seed=s),
            "LinUCB": lambda s: LinUCB(K, dim),
            "UCB(비컨텍스추얼)": lambda s: NonContextualUCB(K),
            "균등가중": lambda s: EqualWeight(K),
        }

    # 시드별 경로를 평균내 컨트롤러당 하나의 "평균 보상 경로" 를 만듭니다.
    n_blocks = max(min(10, T // 100), 4)
    results = {}
    for nm, make in controllers.items():
        paths = np.vstack([_run(make, 1000 + s) for s in range(int(n_seeds))])
        mean_path = paths.mean(axis=0)
        blocks = np.array([b.mean() for b in np.array_split(mean_path, n_blocks)])
        results[nm] = {
            "per_step": float(mean_path.mean() * r_scale),
            "seed_sd": float(paths.sum(axis=1).std(ddof=1)) if n_seeds > 1 else 0.0,
            "n_seeds": int(n_seeds),
            "_blocks": blocks,
        }

    ex_post, best_arm = BestExPost.evaluate(R_n)
    ep_blocks = np.array([b.mean() for b in
                          np.array_split(R_n[:, best_arm], n_blocks)])
    results["사후최적 단일팔"] = {
        "per_step": float(ex_post / T * r_scale), "seed_sd": 0.0, "n_seeds": 1,
        "best_arm": best_arm, "_blocks": ep_blocks}

    # ★ LinTS vs 비컨텍스추얼 UCB — **구간별** 대응표본 t검정
    #   시드별로 짝지으면 안 됩니다: UCB 와 균등가중은 결정적(seed 무관)이라
    #   차이의 분산이 LinTS 의 내부 난수에서만 나옵니다. 그러면 사소한 차이도
    #   |t| 가 폭발하고, 데이터 실현이 조금만 바뀌어도 부호가 뒤집힙니다.
    #   우리가 알고 싶은 것은 "여러 기간에 걸쳐 일관되게 이기는가" 이므로
    #   시간 구간으로 짝짓습니다.
    a = results.get("LinTS", {}).get("_blocks")
    b = results.get("UCB(비컨텍스추얼)", {}).get("_blocks")
    verdict, passed, tstat = "판정 불가", None, None
    if a is not None and b is not None and len(a) == len(b) and len(a) > 3:
        diff = a - b
        sd = float(diff.std(ddof=1))
        tstat = float(diff.mean() / (sd / math.sqrt(len(diff)))) if sd > 1e-12 else 0.0
        passed = bool(tstat > 2.0)
        if passed:
            verdict = (f"통과 — 컨텍스추얼(LinTS)이 비컨텍스추얼 UCB를 "
                       f"유의하게 이깁니다 (t={tstat:.2f}). 국면 컨텍스트에 "
                       f"정보가 있습니다. 국면 계층을 지을 근거가 생겼습니다.")
        else:
            verdict = (f"미통과 — 컨텍스추얼이 비컨텍스추얼 UCB를 이기지 "
                       f"못합니다 (t={tstat:.2f}). **국면 컨텍스트에 정보가 "
                       f"없습니다.** 경제물리 계층을 버리고 실현변동성 단일 축으로 "
                       f"가세요. GA·RL 코드는 쓰지 마세요. 나머지 아키텍처는 "
                       f"그대로 유효합니다 — 이것은 실패가 아니라 절약입니다.")

    # 정합성 참고 — 사후 최적 단일팔을 넘는 컨트롤러가 있는가.
    #
    # ⚠ **이건 자동으로 버그를 뜻하지 않습니다.** 사후 최적 단일팔은 "최선의
    #   *고정* 팔" 이지 도달 불가능한 상한이 아닙니다. 보상이 비정상적이면
    #   (금융 시계열은 대체로 그렇습니다) 적응형 정책이 어떤 고정 팔보다
    #   나을 수 있습니다 — 밴딧 문헌의 tracking regret 이 다루는 영역입니다.
    #
    #   다만 **타이밍 버그의 증상이기도** 합니다. 실제로 겪었습니다: 풀링
    #   패널에서 지연을 행 인덱스로 세는 바람에 h행 = h/종목수 거래일이 되어
    #   비컨텍스추얼 UCB 가 미래를 봤고, 이 초과가 그 단서였습니다.
    #
    #   그래서 **경고만 하고 판정을 무효화하지 않습니다.** 무효화하면 정상적인
    #   비정상성에서도 답을 못 얻습니다. 두 원인을 구분하는 방법은 아래 문구에.
    ex_post_ps = results["사후최적 단일팔"]["per_step"]
    violators = [nm for nm, r in results.items()
                 if nm != "사후최적 단일팔" and r["per_step"] > ex_post_ps + 1e-12]
    integrity = None
    if violators:
        integrity = (
            f"참고: {', '.join(violators)} 이(가) 사후 최적 단일팔"
            f"({ex_post_ps:.8f})을 넘었습니다. 두 가지 원인이 가능합니다 — "
            f"(a) 보상이 비정상적이라 적응이 고정 팔을 이긴 것(정상), "
            f"(b) 보상 타이밍 버그. 구분법: 풀링 패널이면 time_index 에 날짜를 "
            f"넘겼는지 확인하고, horizon 을 2배로 늘려도 초과가 유지되는지 보세요. "
            f"버그라면 지연을 늘릴수록 초과가 커집니다.")

    return {
        "passed": passed, "t_stat": round(tstat, 3) if tstat is not None else None,
        "verdict": verdict,
        "integrity_warning": integrity,
        "pooling_note": pooled_warning,
        "n_obs": T, "n_arms": K, "context_dim": d, "horizon": horizon,
        "n_blocks": n_blocks, "reward_scale": round(r_scale, 8),
        "ranking": sorted(((nm, round(r["per_step"], 8))
                           for nm, r in results.items()),
                          key=lambda kv: kv[1], reverse=True),
        "detail": {nm: {k: (round(v, 8) if isinstance(v, (int, float)) else v)
                        for k, v in r.items() if k != "_blocks"}
                   for nm, r in results.items()},
    }
