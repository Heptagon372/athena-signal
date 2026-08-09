"""
귀무분포 보정 (Null Calibration)
--------------------------------
추정량의 임계값을 **상수로 쓰지 않고 부트스트랩으로 만드는** 모듈입니다.

수록 항목
    stationary_bootstrap    Politis & Romano (1994) 정지 부트스트랩 — 변동성 군집 보존
    auto_block_length       평균 블록 길이 자동 선택 (|수익률| 자기상관 기반)
    NullSpec                한 (추정량, 창길이, 방법) 조합의 귀무분포 요약
    calibrate               임의의 추정량에 대해 NullSpec 생성 (캐시)
    calibrate_covariance    여러 추정량의 귀무 **공분산** — 마할라노비스 지수용
    NullBand                교정된 임계 밴드 + 히스테리시스 판정

왜 이것이 필요한가 — 0.45/0.55 는 민간전승입니다
    Hurst 지수를 0.45/0.55 와 비교해 "평균회귀/추세" 라고 부르는 관행이 널리
    퍼져 있습니다. 그런데 **그 폭은 추정 노이즈보다 좁습니다.**

        Kristoufek (2009): n=512 인 i.i.d. 계열에서 R/S 의 평균 Ĥ = 0.5763,
        즉 편향만으로 +0.076. n=512 에서 Ĥ=0.65 조차 90% 수준에서 랜덤워크를
        기각하지 못합니다.
        T=250 에서 σ(Ĥ) ≈ 0.057(R/S) / 0.074(DFA), T=128 에서 0.070 / 0.091.
        95% 대역이 0.5 ± 0.12 ~ ±0.18 입니다.

    0.45/0.55 는 그 대역 **안**에 통째로 들어갑니다. 그 임계값으로 만든 국면
    라벨은 난수입니다. 이 저장소의 `engine/econophysics.dfa_null_calibration`
    이 DFA 한 개에 대해 이미 같은 결론에 도달했습니다 — 랜덤워크에서 |t|>2 가
    52% 나왔습니다. 이 모듈은 그 처방을 **임의의 추정량으로 일반화**한 것입니다.

정지 부트스트랩을 쓰는 이유 (i.i.d. 부트스트랩이 아니라)
    **변동성 군집만으로도 가짜 장기기억이 생깁니다.** 수익률을 한 개씩 섞으면
    군집이 사라져서 귀무분포가 지나치게 좁아지고, 그 좁은 밴드와 비교하면
    실제로는 변동성 군집일 뿐인 것이 "유의한 장기기억" 으로 통과합니다.
    정지 부트스트랩은 기하분포 길이의 블록을 이어 붙여 군집과 두꺼운 꼬리를
    보존하면서 **예측 가능한 순서만** 파괴합니다.

    `engine/validation.block_bootstrap` 과 목적이 같고 형태가 다릅니다. 저쪽은
    고정 길이 블록으로 전략 성과의 귀무분포를 만들고, 이쪽은 가변 길이 블록으로
    **추정량**의 귀무분포를 만듭니다. 가변 길이여야 재표본이 정지성(stationary)을
    유지합니다 — 고정 길이는 블록 경계가 주기적으로 반복되어 미세한 인공물을
    남기는데, 스케일링 지수처럼 여러 시간 스케일을 동시에 보는 추정량은 바로
    그 인공물을 읽습니다.

비용과 캐시
    n_boot=2000 × 창 256 기준으로 추정량 한 개당 수 초입니다. 같은 (추정량,
    창길이, 방법, 시드) 조합은 캐시되므로 두 번째 호출부터 비용이 없습니다.
    캐시 키는 설정 해시이고, NullSpec.spec_hash 로 노출됩니다 — 백테스트
    기록(engine/registry.py)에 남겨서 "어느 임계값으로 판정했는가" 를 재현할
    수 있게 하기 위해서입니다.
"""

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

# 기본 부트스트랩 반복수. 설계 문헌은 10,000 을 권하지만 파이썬 추정량 호출이
# 지배적이라 2,000 으로 둡니다. 분위수 0.05/0.95 는 2,000회면 충분히 안정합니다
# (표준오차 ≈ sqrt(0.05*0.95/2000) ≈ 0.005).
DEFAULT_N_BOOT = 2000

_CACHE: dict[str, "NullSpec"] = {}


# ---------------------------------------------------------------------------
# 정지 부트스트랩
# ---------------------------------------------------------------------------

def auto_block_length(returns, lo: int = 2, hi: int = None,
                      window: int = None) -> float:
    """평균 블록 길이를 **|수익률| 의 자기상관**에서 고릅니다.

    통상적인 기본값 n^(1/3) 은 수익률 자체의 자기상관을 염두에 둔 것인데,
    우리가 보존하려는 것은 수익률이 아니라 **변동성** 의 기억입니다. 수익률의
    자기상관은 거의 0 이지만 |수익률| 의 자기상관은 수십 봉까지 살아 있습니다.
    그래서 |r| 의 자기상관이 백색잡음 대역 2/√n 아래로 처음 떨어지는 지연을
    평균 블록 길이로 씁니다.

    **`window` 상한이 이 함수에서 가장 중요한 부분입니다.**
        블록이 너무 짧으면 군집이 깨져 귀무분포가 좁아집니다(임계값이 관대해져
        가짜 알파를 통과시킵니다). 그런데 반대쪽 실패가 더 조용하고 더 위험합니다.
        평균 블록 길이가 재표본 길이에 가까워지면 재표본 하나가 **원본의 연속
        구간을 거의 그대로 복사**한 것이 됩니다. 그러면 만들어지는 분포는
        귀무분포가 아니라 그냥 *관측된 롤링 추정치의 분포* 입니다. 관측을
        관측과 비교하게 되므로 어떤 것도 유의하지 않게 나오고, 그 사실이
        화면에는 "엄격하게 검정했다" 로 보입니다.

        실측: 지속성 높은 GARCH(α+β=0.98) 수익률에서 |r| 의 자기상관은 140봉
        넘게 대역 위에 남아 있습니다. 창 256 에 그대로 쓰면 블록이 2개뿐입니다.
        그래서 window//8 로 자릅니다 — 재표본당 최소 8블록을 보장합니다.

    상한은 [lo, min(n//10, window//8)] 입니다.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float(lo)
    hi = int(hi or max(lo, n // 10))
    if window:
        hi = min(hi, max(int(lo), int(window) // 8))
    hi = max(hi, int(lo))

    dev = np.abs(x - x.mean())
    dev = dev - dev.mean()
    denom = float(np.sum(dev ** 2))
    if denom <= 0:
        return float(lo)

    band = 2.0 / math.sqrt(n)
    max_lag = min(hi * 3, n // 4)
    for lag in range(1, max(max_lag, 2)):
        acf = float(np.sum(dev[lag:] * dev[:-lag]) / denom)
        if acf < band:
            return float(min(max(lag, lo), hi))
    return float(hi)


def stationary_bootstrap(returns, length: int = None, mean_block: float = None,
                         rng=None) -> np.ndarray:
    """Politis & Romano (1994) 정지 부트스트랩 재표본 한 개.

    Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
    Journal of the American Statistical Association, 89(428), 1303-1313.

    절차
        1) 시작점을 균등하게 뽑는다
        2) 각 시점에서 확률 p = 1/mean_block 로 새 블록을 시작하고,
           그렇지 않으면 직전 인덱스의 **다음** 값을 이어 붙인다 (순환)
        3) 목표 길이까지 반복

    블록 길이가 기하분포를 따르므로 재표본이 정지성을 갖습니다. 고정 길이
    블록(engine.validation.block_bootstrap)은 이 성질이 없습니다 — 그쪽은
    전략 수익률용이라 문제가 되지 않지만, 스케일링 지수처럼 여러 시간 스케일을
    동시에 읽는 추정량에는 블록 경계의 주기성이 그대로 신호로 잡힙니다.

    `mean_block=1` 이면 각 블록이 길이 1 이므로 **i.i.d. 부트스트랩과 같습니다.**
    두 방법의 임계 폭 차이가 곧 "변동성 군집이 만드는 가짜 장기기억" 의 크기입니다.
    """
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    length = int(length or n)
    if length <= 0:
        return np.array([], dtype=float)

    rng = rng if rng is not None else np.random.default_rng()
    mb = float(mean_block) if mean_block else auto_block_length(x)
    mb = max(mb, 1.0)
    p = min(1.0 / mb, 1.0)

    if n == 1:
        return np.repeat(x, length)

    # 블록 길이를 넉넉히 뽑아 두고 필요한 만큼만 씁니다 (기대 블록수의 2배 + 여유)
    guess = int(length / mb) + 8
    lens = rng.geometric(p, size=guess)
    while int(lens.sum()) < length:
        lens = np.concatenate([lens, rng.geometric(p, size=guess)])

    starts = rng.integers(0, n, size=len(lens))
    # 블록 b 의 j 번째 원소 = x[(starts[b] + j) % n] — 완전 벡터화
    ends = np.cumsum(lens)
    offsets = np.arange(int(ends[-1])) - np.repeat(ends - lens, lens)
    idx = (np.repeat(starts, lens) + offsets) % n
    return x[idx[:length]]


# ---------------------------------------------------------------------------
# NullSpec — 한 조합의 귀무분포 요약
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NullSpec:
    """(추정량, 창길이, 부트스트랩 방법) 하나에 대한 귀무분포.

    `null_value` 는 이론적 귀무값입니다(Hurst·DFA α 는 0.5, VR 은 1.0).
    None 이면 판정을 분위수로만 합니다.
    """
    estimator: str
    n: int                       # 창 길이
    n_boot: int
    method: str                  # "stationary" | "iid"
    mean_block: float
    null_value: float | None
    mean: float
    sd: float
    q_lo: float
    q_hi: float
    quantiles: tuple = (0.05, 0.95)
    spec_hash: str = ""
    n_valid: int = 0             # 추정량이 None 을 돌려주지 않은 재표본 수
    _samples: tuple = field(default=(), repr=False, compare=False)

    # -- 판정 도구 ---------------------------------------------------------

    @property
    def bias(self) -> float | None:
        """유한표본 편향 = 귀무평균 − 이론값. 보고값에서 이걸 빼야 합니다."""
        if self.null_value is None:
            return None
        return self.mean - self.null_value

    def debias(self, value: float) -> float:
        """편향 보정된 추정치."""
        b = self.bias
        return float(value) if b is None else float(value) - b

    def zscore(self, value: float) -> float | None:
        """귀무분포 기준 z. **점추정을 임계값과 비교하지 말고 이걸 보세요.**"""
        if self.sd <= 1e-12:
            return None
        return float((float(value) - self.mean) / self.sd)

    def p_value(self, value: float, two_sided: bool = True) -> float | None:
        """경험적 p값. 재표본을 보관한 경우에만 계산합니다.

        (초과 횟수 + 1)/(시행 + 1) — engine.validation.falsification_audit 와
        같은 보수적 보정입니다. 시행이 적을 때 p=0 이 나오는 것을 막습니다.
        """
        if not self._samples:
            return None
        arr = np.asarray(self._samples, dtype=float)
        v = float(value)
        if two_sided:
            centre = self.mean
            hits = int(np.sum(np.abs(arr - centre) >= abs(v - centre)))
        else:
            hits = int(np.sum(arr >= v))
        return float((hits + 1) / (len(arr) + 1))

    def verdict(self, value: float) -> str:
        """교정된 밴드 기준 판정: "above" | "inside" | "below".

        의미는 추정량마다 다릅니다 — Hurst/DFA 면 above=지속성,
        VR 이면 above=추세. **inside 는 "랜덤워크다" 가 아니라 "이 표본
        크기로는 구분할 수 없다" 입니다.** 그 둘을 섞으면 안 됩니다.
        """
        v = float(value)
        if v > self.q_hi:
            return "above"
        if v < self.q_lo:
            return "below"
        return "inside"

    def to_dict(self) -> dict:
        return {
            "estimator": self.estimator, "n": self.n, "method": self.method,
            "mean_block": round(self.mean_block, 2), "n_boot": self.n_boot,
            "n_valid": self.n_valid,
            "null_value": self.null_value,
            "mean": round(self.mean, 4), "sd": round(self.sd, 4),
            "bias": round(self.bias, 4) if self.bias is not None else None,
            "q_lo": round(self.q_lo, 4), "q_hi": round(self.q_hi, 4),
            "band_width": round(self.q_hi - self.q_lo, 4),
            "spec_hash": self.spec_hash,
        }


def _hash_spec(estimator: str, n: int, n_boot: int, method: str,
               mean_block: float, quantiles, seed: int,
               source_fingerprint: str) -> str:
    """설정 해시 — 캐시 키이자 재현성 기록용.

    원본 수익률의 지문까지 포함합니다. 같은 추정량·같은 창이라도 **다른 시장,
    다른 기간의 수익률로 보정하면 다른 임계값** 이 나오기 때문입니다. 그것을
    같은 해시로 묶으면 캐시가 조용히 틀린 임계값을 돌려줍니다.
    """
    raw = "|".join([
        estimator, str(int(n)), str(int(n_boot)), method,
        f"{float(mean_block):.4f}",
        ",".join(f"{float(q):.4f}" for q in quantiles),
        str(int(seed)), source_fingerprint,
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _fingerprint(x: np.ndarray) -> str:
    """수익률 계열의 짧은 지문. 길이 + 요약통계로 충분합니다."""
    if len(x) == 0:
        return "empty"
    raw = f"{len(x)}:{x.mean():.8e}:{x.std():.8e}:{x[0]:.8e}:{x[-1]:.8e}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def calibrate(estimator, source_returns, window: int = None, *,
              name: str = None, null_value: float = None,
              n_boot: int = DEFAULT_N_BOOT, method: str = "stationary",
              mean_block: float = None, quantiles=(0.05, 0.95),
              seed: int = 20260808, keep_samples: bool = True,
              use_cache: bool = True) -> NullSpec | None:
    """임의의 추정량에 대한 귀무분포를 만듭니다.

    Parameters
    ----------
    estimator : callable
        `estimator(x: np.ndarray) -> float | None` 형태. 수익률 배열을 받아
        스칼라를 돌려줍니다. None 이나 NaN 을 돌려주면 그 재표본은 버립니다.
        **누적계열을 받는 추정량(GHE 등)은 호출부에서 감싸세요** — 이 함수는
        항상 수익률을 넘깁니다. (설계 문헌이 지목한 가장 흔한 버그가
        "로그가격에 DFA 를 돌려 α≈1.5 를 H=1.5 로 보고" 하는 것입니다.)
    source_returns : array-like
        **실제 시장 수익률**. 여기서 블록을 뽑습니다. 정규난수를 넣으면
        i.i.d. 귀무가 되어 이 모듈을 쓰는 의미가 없어집니다.
    window : int
        재표본 길이. 실제로 추정량에 넣는 창 길이와 같아야 합니다.
        귀무분포는 창 길이에 강하게 의존합니다 — 다른 창의 임계값을 돌려 쓰면
        보정이 무효입니다.
    method : "stationary" | "iid"
        "iid" 는 비교·진단용입니다. 두 밴드 폭의 차이가 곧 변동성 군집이
        만드는 가짜 장기기억의 크기입니다.

    Returns
    -------
    NullSpec | None — 유효 재표본이 30개 미만이면 None (판정하지 않습니다).
    """
    x = np.asarray(source_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 32:
        return None

    window = int(window or len(x))
    if window < 16:
        return None

    name = name or getattr(estimator, "__name__", "estimator")
    if method == "iid":
        mb = 1.0
    else:
        mb = (float(mean_block) if mean_block
              else auto_block_length(x, window=window))

    key = _hash_spec(name, window, n_boot, method, mb, quantiles, seed,
                     _fingerprint(x))
    if use_cache and key in _CACHE:
        return _CACHE[key]

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        sample = stationary_bootstrap(x, length=window, mean_block=mb, rng=rng)
        try:
            v = estimator(sample)
        except Exception:
            continue
        if v is None:
            continue
        v = float(v)
        if math.isfinite(v):
            values.append(v)

    if len(values) < 30:
        return None

    arr = np.asarray(values, dtype=float)
    q_lo, q_hi = (float(q) for q in np.quantile(arr, list(quantiles)))

    spec = NullSpec(
        estimator=name, n=window, n_boot=int(n_boot), method=method,
        mean_block=mb, null_value=null_value,
        mean=float(arr.mean()), sd=float(arr.std(ddof=1)),
        q_lo=q_lo, q_hi=q_hi, quantiles=tuple(quantiles), spec_hash=key,
        n_valid=len(arr),
        _samples=tuple(arr.tolist()) if keep_samples else (),
    )
    if use_cache:
        _CACHE[key] = spec
    return spec


def calibrate_covariance(estimators: dict, source_returns, window: int = None, *,
                         n_boot: int = DEFAULT_N_BOOT, method: str = "stationary",
                         mean_block: float = None,
                         seed: int = 20260808) -> dict | None:
    """여러 추정량의 귀무 **평균벡터와 공분산**.

    마할라노비스 효율성 지수(engine.econophysics.efficiency_index)가 이걸
    씁니다. 왜 유클리드 합이 아니라 공분산이 필요한가:

        같은 데이터로 계산한 Hurst 추정량 3개와 프랙탈 차원 4개는 **구조적으로
        강하게 상관**되어 있습니다. 그걸 제곱합하면 원소가 많은 계열이 지수를
        지배합니다 — 정보가 늘어난 게 아니라 같은 정보를 여러 번 센 것입니다.
        Σ⁻¹ 를 끼우면 상관이 나눠지고, 결과가 자유도 k 의 카이제곱 통계량으로
        **직접 해석**됩니다.

    **모든 추정량을 같은 재표본에서** 계산하는 것이 핵심입니다. 따로 돌리면
    상관 구조가 사라져 공분산이 대각이 되고, 그러면 유클리드 합과 같아집니다.
    """
    x = np.asarray(source_returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 32 or not estimators:
        return None
    window = int(window or len(x))

    names = list(estimators.keys())
    mb = 1.0 if method == "iid" else (
        float(mean_block) if mean_block
        else auto_block_length(x, window=window))
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(int(n_boot)):
        sample = stationary_bootstrap(x, length=window, mean_block=mb, rng=rng)
        row = []
        ok = True
        for nm in names:
            try:
                v = estimators[nm](sample)
            except Exception:
                ok = False
                break
            if v is None or not math.isfinite(float(v)):
                ok = False
                break
            row.append(float(v))
        if ok:
            rows.append(row)

    if len(rows) < max(30, len(names) * 5):
        return None

    mat = np.asarray(rows, dtype=float)
    mean = mat.mean(axis=0)
    cov = np.cov(mat, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)

    # 표본 공분산이 특이해질 수 있어 pinv 를 씁니다 (추정량들이 서로 거의
    # 완전상관이면 실제로 특이합니다 — 그 경우가 바로 D 와 H 를 함께 넣은 때입니다)
    inv = np.linalg.pinv(cov)

    return {
        "names": names, "mean": mean, "cov": cov, "inv": inv,
        "n_valid": len(rows), "method": method, "mean_block": mb,
        "window": window,
        # 조건수가 크면 추정량들이 사실상 같은 것을 재고 있다는 뜻입니다
        "condition_number": float(np.linalg.cond(cov)) if cov.size else None,
    }


def mahalanobis(values, calib: dict) -> float | None:
    """관측 벡터의 마할라노비스 거리. `calibrate_covariance` 결과를 씁니다.

    자유도 k 의 카이제곱 통계량으로 읽습니다 — 즉 √(카이제곱) 이므로
    k=3 이면 랜덤워크에서 대략 2.8 을 넘는 일이 5% 입니다.
    """
    if not calib:
        return None
    v = np.asarray(values, dtype=float)
    if v.shape[0] != len(calib["names"]) or not np.all(np.isfinite(v)):
        return None
    d = v - calib["mean"]
    q = float(d @ calib["inv"] @ d)
    return math.sqrt(max(q, 0.0))


# ---------------------------------------------------------------------------
# 히스테리시스 밴드
# ---------------------------------------------------------------------------

@dataclass
class NullBand:
    """교정된 밴드 + Schmitt 히스테리시스 상태기계.

    고정 임계 교차는 추정오차만으로 국면을 계속 뒤집습니다. 진입 임계와 이탈
    임계를 분리하면(Schmitt 트리거) 경계 근처에서 떨리지 않습니다. 여기에
    최소 체류 기간을 더해 이중으로 막습니다.

        진입: q_hi 를 넘어야 "above" 로 간다
        이탈: q_hi 아래로 내려가는 것으로는 부족하고, `exit_frac` 만큼
              중앙 쪽으로 더 들어와야 "inside" 로 돌아온다

    **이건 지연을 사는 대가로 안정을 얻는 거래입니다.** 국면이 실제로 바뀌었을 때
    반응이 늦어집니다. 국면 탐지의 가치는 대부분 *전환* 을 맞히는 데 있으므로
    (지속 구간을 맞히는 건 쉽습니다), exit_frac 을 키우는 것은 공짜가 아닙니다.
    """
    spec: NullSpec
    exit_frac: float = 0.5       # 진입 임계와 중앙 사이 어디까지 되돌아와야 이탈인가
    min_dwell: int = 3           # 최소 체류 봉 수
    state: str = "inside"
    dwell: int = 0

    def update(self, value: float) -> str:
        """새 관측 하나를 넣고 갱신된 국면 라벨을 돌려줍니다."""
        v = float(value)
        centre = self.spec.mean
        f = min(max(float(self.exit_frac), 0.0), 1.0)
        exit_hi = centre + (self.spec.q_hi - centre) * (1.0 - f)
        exit_lo = centre - (centre - self.spec.q_lo) * (1.0 - f)

        self.dwell += 1
        if self.state == "inside":
            if v > self.spec.q_hi:
                self.state, self.dwell = "above", 0
            elif v < self.spec.q_lo:
                self.state, self.dwell = "below", 0
        elif self.state == "above":
            if v < exit_hi and self.dwell >= self.min_dwell:
                self.state, self.dwell = "inside", 0
        elif self.state == "below":
            if v > exit_lo and self.dwell >= self.min_dwell:
                self.state, self.dwell = "inside", 0
        return self.state


def clear_cache() -> None:
    """캘리브레이션 캐시 비우기 (테스트용)."""
    _CACHE.clear()
