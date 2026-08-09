"""
인과성 · 누수 게이트 (Causality & Leakage Gates)
------------------------------------------------
"이 지표가 t 시점에 **정말로 알 수 있었던 것만** 쓰는가" 를 자동으로 검사합니다.

수록 항목
    assert_causal_series    롤링 지표: 미래를 지우고도 t 시점 값이 같은가
    assert_causal_at        (계열, t) → 스칼라 형태의 지표에 대한 같은 검사
    causality_report        예외 대신 보고서로 (CI 대시보드용)
    scan_callables          지표 여러 개를 한 번에 훑기
    leakage_shift_test      피처 시각을 +1 시프트하면 IC 가 무너지는가
    label_shuffle_test      라벨을 섞으면 성과가 무작위 수준이 되는가
    batch_permutation_test  배치 순서를 섞어도 예측이 불변인가
    centred_moving_average  **의도적으로 비인과인** 참조 구현 (게이트 검증용)

왜 이 한 개의 테스트가 가장 값싼가
    백테스트를 무효로 만드는 결함은 대부분 같은 모양입니다 — **t 시점 계산에
    t 이후 데이터가 들어간다.** 겉으로는 전혀 드러나지 않습니다. 오히려 성과가
    좋아지므로 "잘 되네" 로 읽힙니다.

    실제로 이 한 가지 검사가 잡아내는 것들:

      · 중앙이동평균 / centred DMA(θ=0.5)  — 창의 절반이 미래입니다
      · HMM smoothed state / Viterbi        — 전 표본을 씁니다. filtered 만 인과적
      · `bfill()`                            — 미래값을 워밍업 구간에 채웁니다
      · 전 구간에서 적합한 정규화 스케일러   — 평균·표준편차에 미래가 들어갑니다
      · 배치 축을 공간 축으로 취급한 컨볼루션 — 배치가 시간순 슬라이딩 윈도우면
                                              그대로 미래 참조가 됩니다

    마지막 항목은 실제 사례가 있습니다. L1 호가 예측 노트북 하나가 3-클래스
    정확도 83.6% 를 보고했는데, 입력이 (batch, T, 3) 3차원인데 커널이 2D 라
    프레임워크가 배치 축을 공간 축으로 계산했습니다. 결과적으로 시점 r 의 예측이
    r+6 까지의 호가를 썼습니다 — 39 upd/s 에서 **약 154ms 의 미래 참조** 입니다.
    `batch_permutation_test` 한 줄이면 잡혔을 일입니다.

두 가지 마스킹 모드 — 둘 다 필요합니다
    truncate : t 이후를 **잘라내고** 다시 계산합니다. "미래 데이터가 없어도
               같은 값이 나오는가" 를 봅니다. 인과성의 정의 그 자체입니다.
    mask     : t 이후를 NaN 으로 **덮고** 다시 계산합니다. 배열 길이는 그대로라서
               `len(x)` 나 전체 통계량(`x.mean()`)을 참조하는 코드를 잡습니다.
               truncate 만 하면 이런 것이 통과합니다 — 길이가 같이 줄어드니까요.
"""

import math

import numpy as np


class CausalityError(AssertionError):
    """t 시점 값이 미래 데이터에 의존할 때."""


def _probe_points(n: int, n_probes: int, warmup: int, rng) -> list:
    """검사 지점을 고릅니다. 워밍업 구간과 마지막 지점은 제외합니다.

    마지막 지점을 빼는 이유: t = n−1 에서는 자를 미래가 없어 모든 함수가
    자동으로 통과합니다. 그 지점만 검사하는 게이트는 아무것도 검사하지 않습니다.
    """
    lo = max(int(warmup) + 2, 2)
    hi = n - 2
    if hi <= lo:
        return []
    k = min(int(n_probes), hi - lo)
    return sorted(int(v) for v in rng.choice(np.arange(lo, hi), size=k, replace=False))


def _apply_mask(x: np.ndarray, t: int, mode: str) -> np.ndarray:
    """t 이후를 제거(truncate)하거나 NaN 으로 덮습니다(mask)."""
    if mode == "truncate":
        return x[: t + 1].copy()
    y = x.copy().astype(float)
    y[t + 1:] = np.nan
    return y


def assert_causal_series(fn, x, *, n_probes: int = 20, warmup: int = 0,
                         mode: str = "truncate", atol: float = 0.0,
                         seed: int = 20260808, name: str = None) -> dict:
    """롤링 지표 `fn(1-D array) -> 1-D array` 의 인과성을 검사합니다.

    t 시점 출력이 미래를 안 쓴다면, x 를 t 에서 자르고 다시 계산해도
    **정확히 같은 값** 이 나와야 합니다.

    Parameters
    ----------
    atol : 허용 오차. 기본 0 — 부동소수점 재결합으로 인한 미세 차이까지
        엄격히 봅니다. 누적합 순서가 바뀌어 1e-15 수준 차이가 나는 정상적인
        구현이라면 atol=1e-12 정도를 주세요. **1e-6 이상을 주면 안 됩니다** —
        진짜 누수를 그 아래로 숨길 수 있습니다.

    Raises
    ------
    CausalityError — 하나라도 어긋나면.
    """
    name = name or getattr(fn, "__name__", "fn")
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    probes = _probe_points(n, n_probes, warmup, rng)
    if not probes:
        return {"name": name, "passed": None, "n_probes": 0,
                "reason": "표본이 짧아 검사 지점을 잡지 못했습니다."}

    full = np.asarray(fn(arr), dtype=float)
    if len(full) != n:
        raise CausalityError(
            f"{name}: 출력 길이 {len(full)} ≠ 입력 길이 {n}. "
            f"assert_causal_series 는 입력과 같은 길이를 내는 롤링 지표용입니다.")

    failures = []
    for t in probes:
        masked = _apply_mask(arr, t, mode)
        try:
            out = np.asarray(fn(masked), dtype=float)
        except Exception as exc:
            failures.append((t, f"마스킹 후 예외: {exc!r}"))
            continue
        if len(out) <= t:
            failures.append((t, f"마스킹 후 출력이 짧습니다 ({len(out)} ≤ {t})"))
            continue
        a, b = full[t], out[t]
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) != np.isnan(b):
            failures.append((t, f"NaN 불일치: 전체={a}, 마스킹={b}"))
            continue
        if abs(a - b) > atol:
            failures.append((t, f"값 불일치: 전체={a!r}, 마스킹={b!r}, 차이={a - b:.3e}"))

    result = {"name": name, "mode": mode, "n_probes": len(probes),
              "n_failures": len(failures), "passed": not failures,
              "failures": failures[:5]}
    if failures:
        head = "; ".join(f"t={t}: {msg}" for t, msg in failures[:3])
        raise CausalityError(
            f"{name} 은(는) 인과적이지 않습니다 — {len(failures)}/{len(probes)} "
            f"지점에서 미래 데이터에 의존합니다. {head}")
    return result


def assert_causal_at(fn, x, *, n_probes: int = 20, warmup: int = 0,
                     mode: str = "truncate", atol: float = 0.0,
                     seed: int = 20260808, name: str = None) -> dict:
    """`fn(array, t) -> scalar` 형태의 지표에 대한 같은 검사.

    전략·국면 판정처럼 "전체 계열과 현재 시각을 받아 지금의 값을 내는" 함수용입니다.
    마스킹된 입력에서도 t 를 그대로 넘기므로, 함수가 `x[t+1:]` 를 건드리면 즉시 드러납니다.
    """
    name = name or getattr(fn, "__name__", "fn")
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    probes = _probe_points(len(arr), n_probes, warmup, rng)
    if not probes:
        return {"name": name, "passed": None, "n_probes": 0,
                "reason": "표본이 짧아 검사 지점을 잡지 못했습니다."}

    failures = []
    for t in probes:
        try:
            a = fn(arr, t)
            b = fn(_apply_mask(arr, t, mode), t)
        except Exception as exc:
            failures.append((t, f"예외: {exc!r}"))
            continue
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            failures.append((t, f"None 불일치: 전체={a}, 마스킹={b}"))
            continue
        a, b = float(a), float(b)
        if math.isnan(a) and math.isnan(b):
            continue
        if abs(a - b) > atol:
            failures.append((t, f"값 불일치: 전체={a!r}, 마스킹={b!r}"))

    if failures:
        head = "; ".join(f"t={t}: {msg}" for t, msg in failures[:3])
        raise CausalityError(
            f"{name} 은(는) 인과적이지 않습니다 — "
            f"{len(failures)}/{len(probes)} 지점 불일치. {head}")
    return {"name": name, "mode": mode, "n_probes": len(probes),
            "n_failures": 0, "passed": True}


def causality_report(fn, x, *, kind: str = "series", **kw) -> dict:
    """예외를 던지지 않고 보고서를 돌려줍니다 (대시보드·CI 요약용)."""
    checker = assert_causal_series if kind == "series" else assert_causal_at
    try:
        return checker(fn, x, **kw)
    except CausalityError as exc:
        return {"name": kw.get("name") or getattr(fn, "__name__", "fn"),
                "passed": False, "error": str(exc)}


def scan_callables(mapping: dict, x, *, kind: str = "series", **kw) -> dict:
    """지표 여러 개를 한 번에 검사합니다. `{이름: 함수}` 를 넘기세요."""
    out = {}
    for name, fn in (mapping or {}).items():
        out[name] = causality_report(fn, x, kind=kind, name=name, **kw)
    n_fail = sum(1 for r in out.values() if r.get("passed") is False)
    out["_summary"] = {"n_checked": len(mapping or {}), "n_failed": n_fail,
                       "all_passed": n_fail == 0}
    return out


# ---------------------------------------------------------------------------
# 누수 테스트
# ---------------------------------------------------------------------------

def _rank(a: np.ndarray) -> np.ndarray:
    """평균 순위 (동점은 평균). scipy 없이 씁니다."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # 동점 처리
    uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if len(uniq) < len(a):
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def information_coefficient(feature, forward_returns, method: str = "spearman") -> float | None:
    """IC — 피처와 미래 수익률의 상관. 기본은 순위상관(이상치에 강건)."""
    f = np.asarray(feature, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    n = min(len(f), len(r))
    if n < 8:
        return None
    f, r = f[:n], r[:n]
    ok = np.isfinite(f) & np.isfinite(r)
    if int(ok.sum()) < 8:
        return None
    f, r = f[ok], r[ok]
    if method == "spearman":
        f, r = _rank(f), _rank(r)
    fs, rs = f.std(), r.std()
    if fs <= 1e-12 or rs <= 1e-12:
        return None
    return float(np.mean((f - f.mean()) * (r - r.mean())) / (fs * rs))


def leakage_shift_test(feature, forward_returns, shift: int = 1,
                       collapse_ratio: float = 0.5) -> dict:
    """피처를 **미래로 한 칸 밀면** IC 가 무너져야 합니다.

    직관: 피처가 t 시점 정보라면, 그것을 t+1 의 수익률 예측에 쓰는 것이 원래
    용도입니다. 피처 자체를 한 칸 뒤로 밀면(= 원래보다 더 오래된 정보로 예측)
    예측력이 크게 줄어야 정상입니다. **줄지 않는다면** 둘 중 하나입니다:

      1) 피처에 이미 미래가 들어 있었다 (그래서 한 칸 미뤄도 여전히 미래를 봄)
      2) 그 "피처" 는 사실 아주 느리게 변하는 가격의 변장이다

    특히 국내 뉴스 피처에 반드시 적용하세요. 「특징주」 XX 급등 류 기사는
    **이미 일어난 가격 움직임의 서술** 이고 수 분~수 시간 뒤에 발행됩니다.
    동시점 스코어링하면 화려하고 완전히 가짜인 IC 가 나옵니다.
    """
    f = np.asarray(feature, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    n = min(len(f), len(r))
    if n < 16 or shift < 1:
        return {"passed": None, "reason": "표본이 부족합니다."}

    base = information_coefficient(f[:n], r[:n])
    shifted = information_coefficient(f[: n - shift], r[shift:n])
    if base is None or shifted is None:
        return {"passed": None, "reason": "IC 를 계산할 수 없습니다."}

    # 기저 IC 가 애초에 0 이면 "무너졌는가" 를 물을 수 없습니다.
    # 이 가드가 없으면 0/0 에 가까운 비율이 난수가 되어, 아무 정보 없는 피처가
    # 무작위로 "누수" 판정을 받습니다. 판정하지 않는 것이 옳습니다.
    se = 1.0 / math.sqrt(n)
    if abs(base) < 2.0 * se:
        return {"ic": round(base, 4), "ic_shifted": round(shifted, 4),
                "shift": shift, "passed": None,
                "reason": (f"기저 IC {base:+.4f} 가 표준오차({se:.4f}) 대비 "
                           f"유의하지 않습니다 — 붕괴 여부를 판정할 수 없습니다.")}

    ratio = abs(shifted) / abs(base)
    return {
        "ic": round(base, 4), "ic_shifted": round(shifted, 4),
        "retained_ratio": round(ratio, 3), "shift": shift,
        "passed": bool(ratio <= collapse_ratio),
        "reason": ("정상 — 시프트 후 IC 가 충분히 무너졌습니다."
                   if ratio <= collapse_ratio else
                   f"⚠ 시프트 후에도 IC 의 {ratio:.0%} 가 남아 있습니다. "
                   f"누수이거나, 이 피처가 가격의 변장입니다."),
    }


def label_shuffle_test(evaluate_fn, feature, forward_returns,
                       n_shuffle: int = 200, seed: int = 20260808) -> dict:
    """라벨을 섞으면 성과가 무작위 수준으로 내려가야 합니다.

    `evaluate_fn(feature, labels) -> float` 를 넘기세요. 기본 용도는
    "파이프라인이 라벨과 무관하게도 좋은 숫자를 만드는가" 입니다.
    """
    f = np.asarray(feature, dtype=float)
    y = np.asarray(forward_returns, dtype=float)
    n = min(len(f), len(y))
    if n < 32:
        return {"passed": None, "reason": "표본이 부족합니다."}
    f, y = f[:n], y[:n]

    observed = evaluate_fn(f, y)
    if observed is None:
        return {"passed": None, "reason": "관측값을 계산할 수 없습니다."}

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(int(n_shuffle)):
        v = evaluate_fn(f, rng.permutation(y))
        if v is not None and math.isfinite(float(v)):
            nulls.append(float(v))
    if len(nulls) < 20:
        return {"passed": None, "reason": "귀무 표본이 부족합니다."}

    from engine.validation import falsification_audit
    out = falsification_audit(float(observed), nulls, label="label_shuffle")
    out["passed"] = bool(out.get("p_value") is not None and out["p_value"] <= 0.05)
    return out


def batch_permutation_test(predict_fn, X, n_trials: int = 5,
                           atol: float = 1e-9, seed: int = 20260808) -> dict:
    """배치 순서를 섞어도 **예측이 정확히 불변** 이어야 합니다.

    `predict_fn(X) -> array` 를 넘기세요. X 의 첫 축이 배치입니다.

    이 한 줄짜리 테스트의 값어치: L1 호가 예측 노트북에서 (batch, T, 3) 입력에
    2D 커널을 쓰는 바람에 프레임워크가 배치 축을 공간 축으로 계산했고, 배치
    원소가 시간순 슬라이딩 윈도우였기 때문에 시점 r 의 예측이 r+6 까지를
    참조했습니다. 배치를 섞기만 해도 로짓이 0.267 움직였습니다.
    신경망을 쓰는 모든 경로에 이 테스트를 걸어 두세요.
    """
    arr = np.asarray(X)
    if arr.ndim < 1 or arr.shape[0] < 4:
        return {"passed": None, "reason": "배치가 너무 작습니다."}

    base = np.asarray(predict_fn(arr), dtype=float)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(int(n_trials)):
        perm = rng.permutation(arr.shape[0])
        out = np.asarray(predict_fn(arr[perm]), dtype=float)
        if out.shape[0] != base.shape[0]:
            return {"passed": False, "reason": "순열 후 출력 크기가 달라졌습니다."}
        # 역치환해서 원래 순서로 되돌린 뒤 비교
        restored = np.empty_like(out)
        restored[perm] = out
        diff = float(np.nanmax(np.abs(restored - base))) if base.size else 0.0
        worst = max(worst, diff)

    return {
        "passed": bool(worst <= atol), "max_abs_diff": worst,
        "n_trials": int(n_trials),
        "reason": ("정상 — 배치 순서와 무관합니다."
                   if worst <= atol else
                   f"⚠ 배치 순서만 바꿔도 예측이 {worst:.3e} 움직입니다. "
                   f"배치 축이 공간 축으로 취급되고 있습니다 — 배치가 시간순이면 "
                   f"이것은 그대로 미래 참조입니다."),
    }


# ---------------------------------------------------------------------------
# 게이트 자체를 검증하기 위한 참조 구현
# ---------------------------------------------------------------------------

def centred_moving_average(x, window: int = 11) -> np.ndarray:
    """**의도적으로 비인과인** 중앙이동평균.

    이건 쓰라고 있는 함수가 아닙니다. `assert_causal_series` 가 실제로 무언가를
    잡아내는지 확인하는 **양성 대조군** 입니다. 게이트가 이걸 통과시키면
    게이트가 고장난 것입니다.

    창 절반이 미래이므로, 창 11 이면 t 시점 값이 t+5 까지를 씁니다.
    """
    a = np.asarray(x, dtype=float)
    w = int(window) | 1                     # 홀수로
    half = w // 2
    out = np.full(len(a), np.nan, dtype=float)
    for t in range(len(a)):
        lo, hi = max(t - half, 0), min(t + half + 1, len(a))
        seg = a[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[t] = float(seg.mean())
    return out


def trailing_moving_average(x, window: int = 11) -> np.ndarray:
    """올바른(인과적) 후행 이동평균 — **음성 대조군**."""
    a = np.asarray(x, dtype=float)
    w = max(int(window), 1)
    out = np.full(len(a), np.nan, dtype=float)
    for t in range(len(a)):
        seg = a[max(t - w + 1, 0): t + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) == min(w, t + 1):
            out[t] = float(seg.mean())
    return out
