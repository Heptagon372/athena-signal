"""
시행 기록부 (Trial Registry)
----------------------------
**평가한 모든 설정을 기록하는** 장부입니다. DSR 의 N 을 정직하게 세기 위한 것이고,
이 모듈의 존재 이유는 하나입니다 — **소급 적용이 불가능하기 때문에 1일차부터
켜 두어야 합니다.**

수록 항목
    TrialSpec           한 시행의 설정 (해시 가능한 정규형)
    TrialRegistry       등록 · 중복제거 · 결과 로깅 · 유효 시행수 N̂
    effective_n         수익률 계열 상관 클러스터링으로 센 독립 시행 수
    RegistryGuard       미등록 백테스트 실행 거부

왜 "몇 번 돌렸는지" 가 성과만큼 중요한가
    아무 우위가 없어도 N 번 시도해서 최고를 고르면 그럴듯한 샤프가 나옵니다.
    `engine/validation.expected_max_sharpe_under_null` 이 그 값을 계산합니다.

        N=1000, σ_SR=0.5 인 무능력 귀무에서 E[max SR] ≈ 1.65

    즉 **1000개 설정을 탐색해서 얻은 백테스트 샤프 1.5 는 순수 노이즈보다
    못합니다.** 이 저장소의 AUTOTRADE.md 16장이 이미 같은 계산을 합니다 —
    주간 반전 전략에서 1,200회 시행의 기대 최대 액티브샤프가 +0.94 였고
    개발구간 최고값 +0.61 이 그 아래였습니다. 그래서 그 전략을 버렸습니다.
    그 판단이 가능했던 이유는 **1,200을 정직하게 셌기 때문** 입니다.

    이 모듈은 그 세기를 손으로 하지 않게 만듭니다.

[중요] N 은 GA 개체수가 아닙니다
    N 은 파이프라인 전체의 시행 수입니다:

        N = (국면 설정 수) × (전략 탐색: 실행 × 개체 × 세대, 중복제거)
            × (메타 하이퍼파라미터 수) × (사이징 변형 수)

    GA 개체수만 세면 수십 배 과소평가합니다. 그래서 `register()` 는 설정
    **전체** 를 해시합니다 — 피처 설정, 국면 설정, 전략 파라미터, 비용 모형,
    분할 시드까지. 하나라도 바뀌면 새 시행입니다.

    다만 시행들이 서로 독립이 아니라는 점도 반영해야 공정합니다. 파라미터를
    1 만큼 바꾼 두 시행은 사실상 같은 시행입니다. 그래서 단순 개수 M 이 아니라
    **수익률 계열을 클러스터링한 클러스터 수** 를 N̂ 으로 씁니다.
"""

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

DEFAULT_PATH = os.path.join("storage", "trial_registry.json")


def _canonical(obj):
    """설정을 해시 가능한 정규형으로. dict 는 키 정렬, float 는 유효자리 고정.

    float 를 반올림하는 이유: 0.1 + 0.2 = 0.30000000000000004 같은 부동소수점
    잔차 때문에 **같은 설정이 다른 해시** 를 받으면 중복제거가 무너지고 N 이
    부풀어 DSR 이 필요 이상으로 보수적이 됩니다.
    """
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return str(obj)
        return round(obj, 10)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 10)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


@dataclass
class TrialSpec:
    """한 시행의 설정.

    `label` 은 사람이 읽는 이름일 뿐 해시에 들어가지 않습니다 — 같은 설정에
    다른 이름을 붙여 두 번 세는 것을 막기 위해서입니다.
    """
    family: str                     # 전략군 (예: "xsec_momentum")
    config: dict                    # 실제로 결과를 바꾸는 모든 것
    data_fingerprint: str = ""      # 유니버스·기간·수정주가 버전
    label: str = ""

    def hash(self) -> str:
        raw = json.dumps(
            {"family": self.family,
             "config": _canonical(self.config),
             "data": self.data_fingerprint},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass
class TrialResult:
    trial_id: str
    sharpe: float | None = None
    returns: list = field(default_factory=list, repr=False)
    metrics: dict = field(default_factory=dict)
    logged_at: float = 0.0


class TrialRegistry:
    """설정 → trial_id 장부. 같은 설정은 항상 같은 id 를 받습니다.

    JSON 파일로 영속화합니다. 저장소의 athena.db 를 쓰지 않는 이유는 이 장부가
    **백테스트 스크립트에서도 켜져야** 하기 때문입니다 — API 서버가 떠 있지
    않은 상태에서 돌린 실험이 안 세어지면 N 이 과소평가됩니다.
    """

    def __init__(self, path: str = DEFAULT_PATH, autosave: bool = True):
        self.path = path
        self.autosave = bool(autosave)
        self._trials: dict = {}      # trial_id -> spec dict
        self._results: dict = {}     # trial_id -> TrialResult
        self.load()

    # -- 영속화 -----------------------------------------------------------

    def load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                blob = json.load(f)
        except Exception:
            return
        self._trials = blob.get("trials", {}) or {}
        for tid, r in (blob.get("results", {}) or {}).items():
            self._results[tid] = TrialResult(
                trial_id=tid, sharpe=r.get("sharpe"),
                returns=r.get("returns") or [], metrics=r.get("metrics") or {},
                logged_at=r.get("logged_at") or 0.0)

    def save(self) -> None:
        if not self.path:
            return
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        blob = {
            "trials": self._trials,
            "results": {tid: {"sharpe": r.sharpe, "returns": r.returns,
                              "metrics": r.metrics, "logged_at": r.logged_at}
                        for tid, r in self._results.items()},
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    # -- 등록 · 로깅 ------------------------------------------------------

    def register(self, spec: TrialSpec) -> str:
        """설정 해시 → trial_id. 이미 있으면 **기존 id 를 그대로** 돌려줍니다.

        같은 설정을 두 번 돌리는 것은 시행 1회입니다. 중복제거를 안 하면
        디버깅하며 100번 재실행한 것이 N=100 이 되어 DSR 이 무의미하게
        보수적이 됩니다.
        """
        tid = spec.hash()
        if tid not in self._trials:
            d = asdict(spec)
            d["registered_at"] = time.time()
            self._trials[tid] = d
            if self.autosave:
                self.save()
        return tid

    def is_registered(self, trial_id: str) -> bool:
        return trial_id in self._trials

    def log_result(self, trial_id: str, returns=None, sharpe: float = None,
                   metrics: dict = None, max_returns: int = 4000) -> None:
        """시행 결과를 기록합니다.

        `returns` 를 보관하는 이유는 N̂ 클러스터링에 필요하기 때문입니다.
        샤프만 저장하면 시행들이 서로 얼마나 겹치는지 알 수 없어 M 을 그대로
        N 으로 쓸 수밖에 없습니다(과도하게 보수적).
        """
        if trial_id not in self._trials:
            raise RegistryError(
                f"미등록 시행입니다: {trial_id}. register() 를 먼저 호출하세요.")
        arr = np.asarray(returns if returns is not None else [], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > max_returns:
            # 너무 길면 앞뒤를 균등 샘플링 — 상관 구조는 보존됩니다
            idx = np.linspace(0, len(arr) - 1, max_returns).astype(int)
            arr = arr[idx]
        if sharpe is None and len(arr) > 2:
            sd = float(arr.std(ddof=1))
            sharpe = float(arr.mean() / sd * math.sqrt(252)) if sd > 1e-12 else 0.0
        self._results[trial_id] = TrialResult(
            trial_id=trial_id, sharpe=sharpe, returns=arr.tolist(),
            metrics=metrics or {}, logged_at=time.time())
        if self.autosave:
            self.save()

    # -- 조회 -------------------------------------------------------------

    @property
    def n_trials(self) -> int:
        """등록된 서로 다른 설정의 수 M (중복제거 후)."""
        return len(self._trials)

    def sharpes(self, family: str = None) -> list:
        out = []
        for tid, r in self._results.items():
            if family and (self._trials.get(tid, {}).get("family") != family):
                continue
            if r.sharpe is not None and math.isfinite(r.sharpe):
                out.append(float(r.sharpe))
        return out

    def return_matrix(self, family: str = None):
        """(시행 × 시점) 수익률 행렬. 길이가 다르면 **가장 짧은 것에 맞춥니다.**"""
        series, ids = [], []
        for tid, r in self._results.items():
            if family and (self._trials.get(tid, {}).get("family") != family):
                continue
            if len(r.returns) > 8:
                series.append(np.asarray(r.returns, dtype=float))
                ids.append(tid)
        if len(series) < 2:
            return None, ids
        n = min(len(s) for s in series)
        return np.vstack([s[-n:] for s in series]), ids

    def effective_n(self, family: str = None, threshold: float = 0.7) -> dict:
        """유효 독립 시행 수 N̂.

        두 가지로 계산해 **둘 다** 돌려줍니다.

          clusters : 수익률 계열 상관행렬에서 |ρ| ≥ threshold 인 시행을 한 덩어리로
                     묶고(단일연결) 덩어리 수를 셉니다. **이쪽을 기본으로 씁니다.**
          analytic : N̂ = M(1−ρ̄)/(1+(M−1)ρ̄). 닫힌 형태지만 M 이 커지면
                     ρ̄ 가 조금만 양수여도 N̂ 이 1 로 붕괴해 퇴화합니다.

        수익률이 기록되지 않았으면 클러스터링이 불가능하므로 **M 을 그대로**
        돌려줍니다 — 보수적인 쪽이 안전합니다.
        """
        mat, ids = self.return_matrix(family)
        m = len([t for t in self._trials
                 if not family or self._trials[t].get("family") == family])
        if mat is None or mat.shape[0] < 2:
            return {"n_hat": max(m, 1), "method": "count_only", "m": m,
                    "reason": "수익률 계열이 부족해 클러스터링하지 않았습니다."}

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(mat)
        corr = np.nan_to_num(corr, nan=0.0)
        k = corr.shape[0]

        # 단일연결 클러스터링 (union-find)
        parent = list(range(k))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(k):
            for j in range(i + 1, k):
                if abs(corr[i, j]) >= threshold:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        n_clusters = len({find(i) for i in range(k)})

        off = corr[~np.eye(k, dtype=bool)]
        rho_bar = float(np.mean(off)) if off.size else 0.0
        denom = 1.0 + (k - 1) * rho_bar
        analytic = (k * (1.0 - rho_bar) / denom) if denom > 1e-9 else float(k)
        analytic = float(min(max(analytic, 1.0), k))

        # 결과가 기록되지 않은 시행도 시행입니다. 클러스터 수에 그만큼 더합니다.
        unlogged = max(m - k, 0)
        return {
            "n_hat": int(max(n_clusters + unlogged, 1)),
            "method": "correlation_clusters",
            "m": m, "m_with_returns": k, "n_clusters": n_clusters,
            "unlogged_counted": unlogged,
            "mean_correlation": round(rho_bar, 4),
            "n_hat_analytic": round(analytic, 2),
            "threshold": threshold,
        }

    def deflated_sharpe(self, trial_id: str, family: str = None,
                        periods_per_year: int = 252) -> dict:
        """이 장부의 N̂ 으로 계산한 DSR.

        `engine.validation.deflated_sharpe_ratio` 를 그대로 쓰되, N 을 손으로
        넣지 않고 장부에서 가져옵니다. **N 을 손으로 넣는 순간 낙관 편향이
        들어옵니다** — 사람은 실패한 시행을 세지 않습니다.
        """
        from engine import validation

        res = self._results.get(trial_id)
        if res is None or res.sharpe is None:
            return {"dsr": None, "reason": "결과가 기록되지 않은 시행입니다."}
        eff = self.effective_n(family)
        trials = self.sharpes(family)
        rets = np.asarray(res.returns, dtype=float)
        n_obs = len(rets)
        skew = kurt = None
        if n_obs > 3:
            sd = float(rets.std(ddof=1))
            if sd > 1e-12:
                z = (rets - rets.mean()) / sd
                skew = float(np.mean(z ** 3))
                kurt = float(np.mean(z ** 4))
        out = validation.deflated_sharpe_ratio(
            observed_sr=res.sharpe, sr_trials=trials, n_obs=max(n_obs, 3),
            skew=skew or 0.0, kurtosis=kurt if kurt is not None else 3.0,
            n_trials=eff["n_hat"], annualized=True,
            periods_per_year=periods_per_year)
        out["effective_n"] = eff
        return out

    def summary(self) -> dict:
        families: dict = {}
        for tid, spec in self._trials.items():
            families[spec.get("family", "?")] = families.get(spec.get("family", "?"), 0) + 1
        return {"n_trials": self.n_trials, "n_with_results": len(self._results),
                "by_family": families, "path": self.path}


class RegistryError(RuntimeError):
    """미등록 시행을 실행하려 할 때."""


class RegistryGuard:
    """미등록 백테스트 실행을 **거부** 합니다.

    사용법::

        guard = RegistryGuard(registry)
        trial_id = guard.require(TrialSpec(family="xsec_mom", config=cfg))
        ...  # 백테스트 실행
        registry.log_result(trial_id, returns=daily_returns)

    왜 경고가 아니라 예외인가: 경고는 무시됩니다. 그리고 이 장부는 **소급
    적용이 불가능** 합니다 — 지난주에 돌린 40번을 지금 와서 셀 방법이 없습니다.
    빠뜨린 시행은 영구히 빠집니다. 그래서 실행 자체를 막는 쪽이 맞습니다.
    """

    def __init__(self, registry: TrialRegistry, enabled: bool = True):
        self.registry = registry
        self.enabled = bool(enabled)

    def require(self, spec: TrialSpec) -> str:
        if not self.enabled:
            return spec.hash()
        return self.registry.register(spec)

    def check(self, trial_id: str) -> None:
        if self.enabled and not self.registry.is_registered(trial_id):
            raise RegistryError(
                f"미등록 시행 {trial_id} — 장부에 등록되지 않은 백테스트는 "
                f"실행할 수 없습니다. TrialSpec 을 만들어 register() 하세요.")
