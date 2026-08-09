"""
조합 교차검증 (Combinatorial Purged Cross-Validation)
-----------------------------------------------------
백테스트에서 **하나의 성과 숫자가 아니라 성과의 분포**를 얻는 분할기입니다.

수록 항목
    CPCVPlan / PurgedSplit  분할 계획과 개별 폴드
    combinatorial_purged_cv 조합 분할 + purge + embargo
    assemble_paths          폴드별 예측을 **경로** 로 재조립
    purged_walk_forward     확장창 워크포워드 (같은 purge 규칙)
    path_report             경로 분포 요약 (중앙값 · IQR · 최악 경로)

왜 단일 홀드아웃으로는 부족한가
    홀드아웃 하나로 얻는 샤프는 **표본 1개** 입니다. 그 값이 좋았다는 것은
    "이 전략이 좋다" 가 아니라 "이 전략이 그 한 경로에서 좋았다" 입니다.
    2019~2026 을 한 번 통과시켜 샤프 1.2 를 얻었다면, 그 숫자의 표준오차를
    모릅니다. 다시 뽑을 수 없으니까요.

    CPCV 는 N 개 그룹 중 k 개를 테스트로 쓰는 **모든** 조합을 돌려서,
    각 그룹이 정확히 한 번씩 등장하는 서로 다른 백테스트 경로
    **C(N−1, k−1) 개** 를 만듭니다. N=8, k=2 면 28개 폴드에서 7개 경로가
    나옵니다. 그러면 샤프의 *분포* 를 보고 IQR 과 최악 경로를 말할 수 있습니다.

    이 저장소의 `engine/validation.py` 가 이미 같은 철학입니다 — 점추정을
    믿지 않고 귀무분포와 비교합니다. CPCV 는 그 분포를 **데이터 분할 쪽에서**
    만드는 장치입니다. 둘은 경쟁하지 않고 겹쳐 씁니다.

purge 와 embargo — 이게 없으면 CPCV 는 그냥 느린 K-fold 입니다
    시계열에서 학습셋과 테스트셋이 인덱스상 겹치지 않아도 **정보는 겹칩니다.**

      purge   : t 시점 관측의 라벨이 [t, t+h] 구간의 수익률이라면, 그 라벨은
                테스트 구간을 이미 들여다본 것입니다. 겹치는 학습 관측을 뺍니다.
                피처 워밍업도 같습니다 — t 의 피처가 [t−w, t] 를 쓰면 그쪽으로도
                겹칩니다. **양쪽 다 잘라야 합니다.** 한쪽만 자르는 구현이 흔한데,
                워밍업이 긴 추정량(DFA·GHE 는 창 256)에서는 그쪽이 더 큽니다.
      embargo : 테스트 직후 구간은 인덱스가 안 겹쳐도 자기상관으로 이어져
                있습니다. 일정 비율을 추가로 학습에서 뺍니다.

    **[중요] purge 길이에는 밴딧의 지연 보상 h 도 들어갑니다.** 메타컨트롤러가
    t 의 결정을 t+h 의 보상으로 학습한다면 그것도 라벨 호라이즌입니다.
    `horizon` 인자에 둘 중 큰 값을 넣으세요.
"""

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class PurgedSplit:
    """폴드 하나. 인덱스는 원본 시계열에 대한 정수 위치입니다."""
    fold_id: int
    test_groups: tuple
    train_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)
    # 이 폴드의 각 테스트 그룹이 어느 경로에 속하는가 {group: path_id}
    path_of_group: dict = field(default_factory=dict)
    n_purged: int = 0
    n_embargoed: int = 0

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


@dataclass
class CPCVPlan:
    n_obs: int
    n_groups: int
    k: int
    horizon: int
    warmup: int
    embargo: int
    group_bounds: list           # [(start, end_exclusive), ...]
    splits: list                 # [PurgedSplit, ...]

    @property
    def n_paths(self) -> int:
        """C(N−1, k−1) — 서로 다른 백테스트 경로의 수."""
        return math.comb(self.n_groups - 1, self.k - 1)

    @property
    def n_folds(self) -> int:
        return len(self.splits)

    def summary(self) -> dict:
        train_sizes = [s.n_train for s in self.splits]
        purged = [s.n_purged + s.n_embargoed for s in self.splits]
        return {
            "n_obs": self.n_obs, "n_groups": self.n_groups, "k": self.k,
            "n_folds": self.n_folds, "n_paths": self.n_paths,
            "horizon": self.horizon, "warmup": self.warmup,
            "embargo": self.embargo,
            "train_size_min": int(min(train_sizes)) if train_sizes else 0,
            "train_size_mean": round(float(np.mean(train_sizes)), 1) if train_sizes else 0,
            "removed_mean": round(float(np.mean(purged)), 1) if purged else 0,
            "removed_pct_mean": round(float(np.mean(purged)) / self.n_obs * 100, 2)
            if self.n_obs else 0.0,
        }


def combinatorial_purged_cv(n_obs: int, n_groups: int = 8, k: int = 2, *,
                            horizon: int = 1, warmup: int = 0,
                            embargo_pct: float = 0.02) -> CPCVPlan | None:
    """López de Prado 식 조합 purged 교차검증 계획을 만듭니다.

    López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 7 & 12.

    Parameters
    ----------
    n_obs : 전체 관측 수
    n_groups : 연속 그룹 수 N
    k : 한 폴드에서 테스트로 쓰는 그룹 수 (1 ≤ k < N)
    horizon : 라벨 호라이즌 h. **밴딧 지연보상이 있으면 그것도 포함한 최대값.**
    warmup : 피처가 뒤를 돌아보는 봉 수. DFA/GHE 처럼 창이 긴 추정량이 있으면 그 창.
    embargo_pct : 테스트 직후 학습에서 뺄 구간의 비율 (전체 길이 대비)

    Returns
    -------
    CPCVPlan | None — 분할이 불가능하면(관측 부족 등) None.
    """
    n_obs = int(n_obs)
    n_groups = int(n_groups)
    k = int(k)
    if n_obs < n_groups * 2 or n_groups < 2 or not (1 <= k < n_groups):
        return None

    horizon = max(int(horizon), 0)
    warmup = max(int(warmup), 0)
    embargo = int(round(max(float(embargo_pct), 0.0) * n_obs))

    # 연속 그룹 경계 — 크기가 최대 1 차이나도록 균등 분할
    edges = np.array_split(np.arange(n_obs), n_groups)
    bounds = [(int(g[0]), int(g[-1]) + 1) for g in edges]

    combos = list(combinations(range(n_groups), k))
    # 각 그룹이 등장하는 조합들에 순서를 매겨 경로 id 로 씁니다.
    # 그룹 g 는 정확히 C(N−1, k−1) 개 조합에 등장하므로, 경로 p 는 모든 그룹을
    # 정확히 한 번씩 갖습니다 — 그래서 경로가 완전한 백테스트 하나가 됩니다.
    seen: dict = {g: 0 for g in range(n_groups)}
    path_map: list = []
    for combo in combos:
        m = {}
        for g in combo:
            m[g] = seen[g]
            seen[g] += 1
        path_map.append(m)

    splits = []
    for fold_id, (combo, pmap) in enumerate(zip(combos, path_map)):
        test_mask = np.zeros(n_obs, dtype=bool)
        purge_mask = np.zeros(n_obs, dtype=bool)
        embargo_mask = np.zeros(n_obs, dtype=bool)

        for g in combo:
            a, b = bounds[g]          # 테스트 구간 [a, b)
            test_mask[a:b] = True

            # purge — 라벨이 앞을 보고(h) 피처가 뒤를 보므로(w) 양쪽으로 번집니다.
            #   i 의 라벨 [i, i+h] 가 테스트와 겹침  → i ∈ [a−h, b)
            #   i 의 피처 [i−w, i] 가 테스트와 겹침  → i ∈ [a, b+w)
            lo = max(a - horizon, 0)
            hi = min(b + warmup, n_obs)
            purge_mask[lo:hi] = True

            # embargo — 테스트 직후 구간
            if embargo > 0:
                embargo_mask[b:min(b + embargo, n_obs)] = True

        removed = purge_mask | embargo_mask | test_mask
        train_idx = np.flatnonzero(~removed)
        test_idx = np.flatnonzero(test_mask)

        # 순수 purge/embargo 로만 빠진 개수 (테스트 자체는 제외하고 셉니다)
        n_purged = int(np.sum(purge_mask & ~test_mask))
        n_emb = int(np.sum(embargo_mask & ~test_mask & ~purge_mask))

        splits.append(PurgedSplit(
            fold_id=fold_id, test_groups=tuple(combo),
            train_idx=train_idx, test_idx=test_idx,
            path_of_group=pmap, n_purged=n_purged, n_embargoed=n_emb,
        ))

    return CPCVPlan(n_obs=n_obs, n_groups=n_groups, k=k, horizon=horizon,
                    warmup=warmup, embargo=embargo, group_bounds=bounds,
                    splits=splits)


def assemble_paths(plan: CPCVPlan, fold_outputs: dict) -> list:
    """폴드별 출력을 **경로** 로 재조립합니다.

    Parameters
    ----------
    fold_outputs : {fold_id: {group_id: array}}
        각 폴드가 각 테스트 그룹 구간에 대해 낸 값(수익률·예측 등).
        길이는 그 그룹의 관측 수와 같아야 합니다.

    Returns
    -------
    list[np.ndarray] — 길이 n_paths. 각 원소가 전체 구간을 덮는 하나의 계열.

    경로 하나가 곧 **완전한 백테스트 한 번** 입니다. 그래서 경로들의 샤프
    분포를 그대로 성과의 분포로 읽을 수 있습니다. 폴드별 성과를 평균내는 것과
    다릅니다 — 평균은 경로 내부의 시간 구조를 뭉개 버립니다.
    """
    if plan is None:
        return []
    paths = [np.full(plan.n_obs, np.nan, dtype=float) for _ in range(plan.n_paths)]
    for split in plan.splits:
        out = fold_outputs.get(split.fold_id)
        if not out:
            continue
        for g, path_id in split.path_of_group.items():
            vals = out.get(g)
            if vals is None:
                continue
            a, b = plan.group_bounds[g]
            arr = np.asarray(vals, dtype=float)
            if len(arr) != b - a:
                continue
            paths[path_id][a:b] = arr
    return paths


def path_report(paths, metric_fn=None, periods_per_year: int = 252) -> dict:
    """경로 분포 요약. **중앙값보다 최악 경로가 더 중요한 숫자입니다.**

    metric_fn 기본값은 연율화 샤프입니다. 다른 지표를 보려면
    `metric_fn(np.ndarray) -> float` 를 넘기세요.
    """
    if metric_fn is None:
        def metric_fn(r):
            r = r[np.isfinite(r)]
            if len(r) < 2:
                return float("nan")
            sd = float(r.std(ddof=1))
            if sd <= 1e-12:
                return 0.0
            return float(r.mean() / sd * math.sqrt(periods_per_year))

    vals = []
    for p in paths or []:
        arr = np.asarray(p, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue
        v = metric_fn(arr)
        if v is not None and math.isfinite(v):
            vals.append(float(v))

    if not vals:
        return {"n_paths": 0, "reason": "유효한 경로가 없습니다."}

    a = np.asarray(vals, dtype=float)
    return {
        "n_paths": len(a),
        "median": round(float(np.median(a)), 4),
        "mean": round(float(a.mean()), 4),
        "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
        "q25": round(float(np.quantile(a, 0.25)), 4),
        "q75": round(float(np.quantile(a, 0.75)), 4),
        "worst": round(float(a.min()), 4),
        "best": round(float(a.max()), 4),
        # 경로의 절반 이상이 음수면 중앙값이 양수여도 믿을 게 못 됩니다
        "share_positive": round(float(np.mean(a > 0)), 3),
    }


def purged_walk_forward(n_obs: int, n_splits: int = 5, *, horizon: int = 1,
                        warmup: int = 0, embargo_pct: float = 0.02,
                        expanding: bool = True, min_train: int = None) -> list:
    """확장창(또는 롤링) 워크포워드 — CPCV 와 **같은 purge 규칙** 을 씁니다.

    CPCV 가 경로 분포를 주는 대신 시간 순서를 여러 번 재사용하는 반면, 이쪽은
    한 번의 정직한 시간 순서를 줍니다. 경로의존적 전략(추적손절·재고 누적)에서는
    이쪽이 더 현실에 가깝습니다. **둘 다 보고하세요** — 어긋나면 전략이
    경로의존적이라는 뜻이고, 그건 그 자체로 알아야 할 정보입니다.
    """
    n_obs = int(n_obs)
    n_splits = int(n_splits)
    if n_obs < n_splits * 2 or n_splits < 1:
        return []
    horizon = max(int(horizon), 0)
    warmup = max(int(warmup), 0)
    embargo = int(round(max(float(embargo_pct), 0.0) * n_obs))
    min_train = int(min_train or max(n_obs // (n_splits + 1), warmup + horizon + 2))

    folds = np.array_split(np.arange(min_train, n_obs), n_splits)
    out = []
    for i, blk in enumerate(folds):
        if len(blk) == 0:
            continue
        a, b = int(blk[0]), int(blk[-1]) + 1
        train_end = max(a - horizon, 0)
        train_start = 0 if expanding else max(train_end - min_train, 0)
        train_idx = np.arange(train_start, train_end)
        # 뒤쪽 purge(warmup)와 embargo 는 확장창에서는 학습이 테스트 이전만
        # 쓰므로 발생하지 않습니다. 다음 폴드의 학습 시작점에서 반영됩니다.
        if embargo > 0 and i > 0:
            pass
        out.append(PurgedSplit(
            fold_id=i, test_groups=(i,), train_idx=train_idx,
            test_idx=np.arange(a, b), path_of_group={i: 0},
            n_purged=int(min(horizon, a)), n_embargoed=0,
        ))
    return out
