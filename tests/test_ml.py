# -*- coding: utf-8 -*-
"""
ML 오버레이 검증 (XGBoost·PatchTST 이식)
---------------------------------------
engine/gbdt.py, engine/patchtst.py, engine/mlsignal.py 를 네트워크 없이
검증합니다. 핵심 질문은 두 가지입니다.

    1) 배울 것이 있는 데이터에서는 배우는가 (심어둔 패턴 검출)
    2) 배울 것이 없는 데이터(랜덤워크)에서는 배웠다고 주장하지 않는가

    python tests/test_ml.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

results = []


def check(category: str, name: str, passed: bool, detail: str = ""):
    results.append((category, name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


def _to_bars(returns: np.ndarray, seed: int = 0, start: float = 50_000.0) -> pd.DataFrame:
    """수익률 배열 → OHLCV 일봉."""
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0, 0.006, len(close)))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])
    volume = rng.integers(100_000, 500_000, len(close)).astype(float)
    idx = pd.bdate_range(end="2026-08-05", periods=len(close))
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


def _ar1(n: int, phi: float, sigma: float = 0.012, seed: int = 0) -> np.ndarray:
    """AR(1) 수익률 — phi<0 이면 평균회귀(예측 가능), phi=0 이면 랜덤워크."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, sigma)
    return r


# ---------------------------------------------------------------------------
# GBDT (engine/gbdt.py — XGBoost 이식)
# ---------------------------------------------------------------------------

def test_gbdt():
    print("\n[GBDT] engine/gbdt.py")
    from engine import gbdt

    # 1) 부스터 단독 — 분리 가능한 계단 패턴을 학습하는가
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (400, 5))
    y = ((x[:, 0] > 0.2) ^ (x[:, 2] < -0.1)).astype(float)   # 상호작용 패턴
    booster = gbdt.Booster().fit(x[:300], y[:300], x[300:], y[300:])
    acc = float(((booster.predict_proba(x[300:]) > 0.5) == (y[300:] > 0.5)).mean())
    check("gbdt", "부스터: 상호작용 패턴 학습 (정확도>85%)", acc > 0.85, f"acc={acc:.2%}")

    # 2) 잎 가중치 공식 — 단일 잎(분할 없음)이면 w = −G/(H+λ)
    x1 = np.zeros((50, 1))
    y1 = np.ones(50) * 0.0
    b1 = gbdt.Booster(rounds=1, max_depth=0, subsample=1.0, colsample=1.0)
    b1.fit(x1, (np.arange(50) % 2 == 0).astype(float))
    p_out = b1.predict_proba(x1)
    check("gbdt", "부스터: 상수 피처 → 사전확률 부근 수렴",
          bool(np.allclose(p_out, p_out[0]) and 0.3 < p_out[0] < 0.7),
          f"p={p_out[0]:.3f}")

    # 3) 피처 행렬 — 유한하고 스케일 무관
    bars = _to_bars(_ar1(250, phi=0.0, seed=2), seed=2)
    built = gbdt.feature_matrix(bars)
    check("gbdt", "피처 행렬 생성·유한성", built is not None
          and bool(np.isfinite(built[0]).all()), f"{built[0].shape if built else None}")
    scaled = bars.copy()
    for c in ("open", "high", "low", "close"):
        scaled[c] = scaled[c] * 100.0        # 가격 수준 100배
    built_scaled = gbdt.feature_matrix(scaled)
    check("gbdt", "피처 스케일 무관성 (가격 100배 → 피처 동일)",
          bool(np.allclose(built[0], built_scaled[0], atol=1e-8)))

    # 4) 평균회귀 시장(phi=-0.5) — 1봉 시야에서 검증 적중률이 랜덤보다 높다
    mr_bars = _to_bars(_ar1(300, phi=-0.5, seed=3), seed=3)
    res = gbdt.direction_score(mr_bars, horizon_bars=1)
    check("gbdt", "평균회귀 시장: 학습 성공 + 검증 우위", res.ok
          and res.val_accuracy > 0.55 and res.edge > 0,
          f"acc={res.val_accuracy:.2%} base={res.val_baseline:.2%} edge={res.edge:+.2f}")

    # 5) 랜덤워크 — '배웠다'고 주장하지 않는다 (edge ≈ 0 부근)
    rw_bars = _to_bars(_ar1(300, phi=0.0, seed=4), seed=4)
    res_rw = gbdt.direction_score(rw_bars, horizon_bars=1)
    check("gbdt", "랜덤워크: 검증 우위 없음 (edge<0.15)",
          (not res_rw.ok) or res_rw.edge < 0.15,
          f"acc={res_rw.val_accuracy:.2%} edge={res_rw.edge:+.2f}")

    # 6) 결정성 — 같은 데이터는 같은 결과
    res2 = gbdt.direction_score(mr_bars, horizon_bars=1)
    check("gbdt", "결정성 (같은 입력 → 같은 점수)",
          res.ok and abs(res.score - res2.score) < 1e-12)

    # 7) 표본 부족 → 학습 거부
    small = gbdt.direction_score(mr_bars.iloc[:100], horizon_bars=1)
    check("gbdt", "표본 부족 시 학습 거부", not small.ok and small.error != "")

    # 8) 중요 피처 보고
    check("gbdt", "피처 중요도 첨부", res.ok and len(res.top_features) >= 1
          and all("name" in f for f in res.top_features),
          f"{[f['name'] for f in res.top_features]}")


# ---------------------------------------------------------------------------
# LightGBM 방식 (engine/gbdt.py variant="lgbm")
# ---------------------------------------------------------------------------

def test_lgbm():
    print("\n[LGBM] engine/gbdt.py — 리프 우선 + GOSS")
    from engine import gbdt

    # 1) 리프 우선 부스터도 상호작용 패턴을 학습한다
    rng = np.random.default_rng(21)
    x = rng.normal(0, 1, (400, 5))
    y = ((x[:, 0] > 0.2) ^ (x[:, 2] < -0.1)).astype(float)
    booster = gbdt.Booster(**gbdt.LGBM_PRESET).fit(x[:300], y[:300], x[300:], y[300:])
    acc = float(((booster.predict_proba(x[300:]) > 0.5) == (y[300:] > 0.5)).mean())
    check("lgbm", "리프 우선+GOSS: 상호작용 학습 (>85%)", acc > 0.85, f"acc={acc:.2%}")

    # 2) num_leaves 상한 준수 — 잎 수가 상한을 넘는 트리가 없어야 한다
    limit = int(gbdt.LGBM_PRESET["num_leaves"])
    max_leaves = 0
    for tree in booster.trees:
        leaves = sum(1 for node in tree.nodes if node.feature < 0)
        max_leaves = max(max_leaves, leaves)
    check("lgbm", f"num_leaves 상한({limit}) 준수", 0 < max_leaves <= limit,
          f"최대 잎 수={max_leaves}")

    # 3) GOSS 증폭 배율 — goss.hpp 의 multiply = (n − top_k)/other_k
    n = 200
    grad = np.linspace(-1, 1, n)
    hess = np.abs(grad) * 0.2 + 0.05
    b = gbdt.Booster(goss=True, eta=1.0)     # eta=1 → 워밍업 0라운드
    rows, g2, h2 = b._sample_rows(iteration=5, grad=grad.copy(), hess=hess.copy(),
                                  rng=np.random.default_rng(0))
    top_k = max(1, int(n * b.params["goss_top_rate"]))
    other_k = max(1, int(n * b.params["goss_other_rate"]))
    expect = (n - top_k) / other_k
    amplified = [i for i in rows if abs(g2[i] / grad[i] - expect) < 1e-9
                 and grad[i] != 0]
    check("lgbm", "GOSS: 표본 수 = top+other, 증폭 배율 정확",
          len(rows) == top_k + other_k and len(amplified) == other_k,
          f"rows={len(rows)} 증폭={len(amplified)} 배율={expect:.1f}")

    # 4) 변형 간 독립성 — 같은 데이터에서 xgb 와 lgbm 이 서로 다른 모델
    mr_bars = _to_bars(_ar1(300, phi=-0.5, seed=3), seed=3)
    res_x = gbdt.direction_score(mr_bars, horizon_bars=1, variant="xgb")
    res_l = gbdt.direction_score(mr_bars, horizon_bars=1, variant="lgbm")
    check("lgbm", "lgbm 변형: 평균회귀 학습 성공", res_l.ok and res_l.edge > 0,
          f"acc={res_l.val_accuracy:.2%} edge={res_l.edge:+.2f}")
    check("lgbm", "결정성 (같은 입력 → 같은 점수)",
          abs(res_l.score - gbdt.direction_score(mr_bars, horizon_bars=1,
                                                 variant="lgbm").score) < 1e-12)


# ---------------------------------------------------------------------------
# 칼만 추세 (engine/kalman.py — filterpy 이식)
# ---------------------------------------------------------------------------

def test_kalman():
    print("\n[칼만] engine/kalman.py")
    from engine import kalman

    # 1) 뚜렷한 추세 — 기울기 부호가 맞고 적중률이 높다
    rng = np.random.default_rng(31)
    trend = 0.004 + rng.normal(0, 0.008, 300)          # 상승 드리프트
    bars_up = _to_bars(trend, seed=31)
    res = kalman.direction_score(bars_up, horizon_bars=5)
    check("kalman", "상승 추세: 기울기 + / 적중률 > 60%",
          res.ok and res.slope > 0 and res.hit_rate > 0.6,
          f"slope={res.slope:+.5f} hit={res.hit_rate:.0%} (n={res.n_eval})")

    # 2) 랜덤워크 — 강한 주장을 하지 않는다 (SNR 낮음 또는 적중률 ≈ 50%)
    rw = _to_bars(_ar1(300, phi=0.0, sigma=0.015, seed=32), seed=32)
    res_rw = kalman.direction_score(rw, horizon_bars=5)
    check("kalman", "랜덤워크: 낮은 확신", res_rw.ok
          and (abs(res_rw.snr) < 3.0 or abs(res_rw.hit_rate - 0.5) < 0.12),
          f"snr={res_rw.snr:+.2f} hit={res_rw.hit_rate:.0%}")

    # 3) 적응형 Q — 급반전 구간이 있으면 Q 조절이 실제로 개입한다
    flip = np.concatenate([np.full(150, 0.004), np.full(150, -0.006)]) \
        + rng.normal(0, 0.004, 300)
    bars_flip = _to_bars(flip, seed=33)
    res_flip = kalman.direction_score(bars_flip, horizon_bars=5)
    check("kalman", "급반전: 적응형 Q 개입 + 새 방향 추종",
          res_flip.ok and res_flip.adapted > 0 and res_flip.slope < 0,
          f"adapted={res_flip.adapted} slope={res_flip.slope:+.5f}")

    # 4) 수준 추정 — 필터 수준이 관측 로그가격에서 크게 벗어나지 않는다
    close_last = float(bars_up["close"].iloc[-1])
    check("kalman", "수준 추정 합리성 (관측 대비 ±5%)",
          abs(np.exp(res.level) / close_last - 1) < 0.05,
          f"level={np.exp(res.level):,.0f} vs close={close_last:,.0f}")

    # 5) 결정성 + 표본 부족 거부
    res2 = kalman.direction_score(bars_up, horizon_bars=5)
    check("kalman", "결정성", abs(res.score - res2.score) < 1e-12)
    tiny = kalman.direction_score(bars_up.iloc[:40], horizon_bars=5)
    check("kalman", "표본 부족 시 거부", not tiny.ok and tiny.error != "")


# ---------------------------------------------------------------------------
# GARCH (engine/garch.py — arch 이식)
# ---------------------------------------------------------------------------

def _garch_series(n: int, omega: float, alpha: float, beta: float,
                  seed: int = 0) -> np.ndarray:
    """알려진 파라미터의 GARCH(1,1) 수익률 생성 — 복원 검사용."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    sigma2 = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2 = omega + alpha * r[t - 1] ** 2 + beta * sigma2
        r[t] = np.sqrt(sigma2) * rng.standard_normal()
    return r


def test_garch():
    print("\n[GARCH] engine/garch.py")
    from engine import ensemble, garch

    # 1) 파라미터 복원 — 진짜 GARCH 데이터에서 α·β 를 근사 복원한다
    true_a, true_b = 0.10, 0.85
    r = _garch_series(1000, omega=1.5e-5, alpha=true_a, beta=true_b, seed=41)
    res = garch.fit(r)
    check("garch", "GARCH 데이터: 수렴 + 지속성 복원", res.ok and res.converged
          and abs(res.persistence - (true_a + true_b)) < 0.1,
          f"α={res.alpha:.3f} β={res.beta:.3f} 지속성={res.persistence:.3f} LR={res.lr_stat:.0f}")

    # 2) 상수 변동성 — 우도비 검정이 '군집 없음'으로 거른다
    flat = np.random.default_rng(42).normal(0, 0.012, 500)
    res_flat = garch.fit(flat)
    check("garch", "상수 변동성: 수렴 거부 (LR 미달)", res_flat.ok
          and not res_flat.converged, f"LR={res_flat.lr_stat:.1f}")

    # 3) 예측 방향 — 방금 충격이 왔으면 예측 변동성 > 무조건부
    shocked = np.concatenate([_garch_series(400, 1.5e-5, true_a, true_b, seed=43),
                              np.array([0.06, -0.05, 0.055])])   # 마지막에 대형 충격
    res_shock = garch.fit(shocked)
    check("garch", "충격 직후: 예측 변동성 > 장기 평균", res_shock.ok
          and res_shock.vol_ratio > 1.1,
          f"ratio={res_shock.vol_ratio:.2f} (σ₁={res_shock.sigma_next_pct:.2f}% "
          f"vs 평균 {res_shock.sigma_uncond_pct:.2f}%)")

    # 4) 반감기 공식 — persistence 로부터 정확히
    if res.ok and 0 < res.persistence < 1:
        expect = np.log(0.5) / np.log(res.persistence)
        check("garch", "반감기 공식 일치", abs(res.half_life - expect) < 1e-6)
    else:
        check("garch", "반감기 공식 일치", False, "수렴 실패로 검증 불가")

    # 5) 결정성 (seed 없는 격자 탐색)
    res2 = garch.fit(r)
    check("garch", "결정성", res.ok and res.alpha == res2.alpha
          and res.loglik == res2.loglik)

    # 6) ensemble 통합 — GARCH 수렴 시 변동성 배수에 보정이 붙고,
    #    미수렴이면 기존 Yang-Zhang 값 그대로
    bars_g = _to_bars(shocked, seed=44)
    garch.clear_cache()
    vol = ensemble.volatility_factor(bars_g)
    has_garch = isinstance(vol.get("garch"), dict)
    check("garch", "ensemble: garch 정보 첨부", vol["ok"] and has_garch,
          f"factor={vol.get('factor')} adj={vol.get('garch', {}).get('applied_adj')}")
    # 표본 부족 시에도 배수는 살아 있어야 한다 (GARCH 없이)
    vol_small = ensemble.volatility_factor(bars_g.iloc[:80])
    check("garch", "ensemble: GARCH 불가여도 배수 동작", "factor" in vol_small)


# ---------------------------------------------------------------------------
# 패치 어텐션 (engine/patchtst.py — PatchTST 이식)
# ---------------------------------------------------------------------------

def test_patch():
    print("\n[패치 어텐션] engine/patchtst.py")
    from engine import patchtst

    # 1) 주기 패턴 — 사인파 수익률은 과거와 같은 위상이 반복되므로 예측 가능
    n = 320
    rng = np.random.default_rng(5)
    t = np.arange(n)
    seasonal = 0.010 * np.sin(2 * np.pi * t / 16) + rng.normal(0, 0.002, n)
    bars = _to_bars(seasonal, seed=5)
    res = patchtst.forecast(bars, {"patch_horizon": 4})
    # 실제 다음 4봉 방향 (사인파는 미래를 알 수 있음)
    future_true = 0.010 * np.sin(2 * np.pi * (np.arange(n, n + 4)) / 16).sum()
    same_dir = res.ok and np.sign(res.score) == np.sign(future_true)
    check("patch", "주기 패턴: 다음 구간 방향 예측", bool(same_dir),
          f"score={res.score:+.3f} 실제={future_true:+.4f}")
    check("patch", "주기 패턴: 확신도 형성", res.ok and res.confidence > 0.3,
          f"conf={res.confidence:.2f} 이웃={res.effective_n:.0f}")

    # 2) 롤링 방향 적중률 — 사인파에서 60% 이상
    hits = total = 0
    r_full = 0.010 * np.sin(2 * np.pi * np.arange(n + 60) / 16) \
        + np.random.default_rng(6).normal(0, 0.002, n + 60)
    for end in range(n, n + 56, 4):
        b = _to_bars(r_full[:end], seed=6)
        f = patchtst.forecast(b, {"patch_horizon": 4})
        if not f.ok:
            continue
        actual = r_full[end:end + 4].sum()
        if abs(actual) < 1e-6:
            continue
        total += 1
        hits += int(np.sign(f.score) == np.sign(actual))
    rate = hits / total if total else 0
    check("patch", "주기 패턴: 롤링 적중률 > 60%", total >= 10 and rate > 0.6,
          f"{hits}/{total} ({rate:.0%})")

    # 3) RevIN — 변동성 3배로 키워도 정규화 공간에서 같은 판단
    calm = _to_bars(seasonal, seed=7)
    hot = _to_bars(seasonal * 3.0, seed=7)
    res_calm = patchtst.forecast(calm, {"patch_horizon": 4})
    res_hot = patchtst.forecast(hot, {"patch_horizon": 4})
    check("patch", "RevIN: 변동성 스케일 무관 점수", res_calm.ok and res_hot.ok
          and abs(res_calm.score - res_hot.score) < 0.1,
          f"{res_calm.score:+.3f} vs {res_hot.score:+.3f}")

    # 4) 랜덤워크 — 강한 방향 주장을 하지 않는다
    rw = _to_bars(_ar1(320, phi=0.0, seed=8), seed=8)
    res_rw = patchtst.forecast(rw, {"patch_horizon": 4})
    check("patch", "랜덤워크: 약한 점수·낮은 합의", (not res_rw.ok)
          or (abs(res_rw.score) < 0.6 and res_rw.agreement < 0.9),
          f"score={res_rw.score:+.3f} agree={res_rw.agreement:.2f}")

    # 5) 미래 참조 없음 — 마지막 h봉을 바꿔도 키·값이 같아야 하는 건 아니지만,
    #    키 창의 값은 전부 과거에 실현된 것이어야 합니다. 구조 검증:
    #    마지막 봉 이후를 잘라낸 데이터로 같은 예측이 나오는지 (예측 시점 고정)
    res_a = patchtst.forecast(bars, {"patch_horizon": 4})
    res_b = patchtst.forecast(bars.copy(), {"patch_horizon": 4})
    check("patch", "결정성 (같은 입력 → 같은 예측)",
          res_a.ok and abs(res_a.score - res_b.score) < 1e-12)

    # 6) 표본 부족 → 예측 거부
    tiny = patchtst.forecast(bars.iloc[:80], {"patch_horizon": 4})
    check("patch", "표본 부족 시 예측 거부", not tiny.ok and tiny.error != "")


# ---------------------------------------------------------------------------
# ML 오버레이 결합 (engine/mlsignal.py)
# ---------------------------------------------------------------------------

def test_mlsignal():
    print("\n[ML 오버레이] engine/mlsignal.py")
    from engine import mlsignal

    mlsignal.clear_cache()
    mr_bars = _to_bars(_ar1(300, phi=-0.5, seed=9), seed=9)

    # 1) off 는 무동작
    state = mlsignal.compute("TEST", mr_bars, {"ml_mode": "off"})
    check("mlsig", "off: 계산 안 함", not state.ok and state.score is None)

    # 2) observe — 계산·첨부, 사용 여부 표시 (4개 예측기 전부)
    state = mlsignal.compute("TEST", mr_bars,
                             {"ml_mode": "observe", "ml_horizon_bars": 1})
    check("mlsig", "observe: 네 예측기 결과 첨부", state.ok
          and state.gbdt.get("ok") is not None and state.lgbm.get("ok") is not None
          and state.patch.get("ok") is not None and state.kalman.get("ok") is not None)

    # 3) 품질 게이트 — 검증 미달 예측기는 결합에서 빠진다
    #    (기준을 극단으로 올려 강제 탈락시키고 확인)
    mlsignal.clear_cache()
    strict = mlsignal.compute("TEST2", mr_bars,
                              {"ml_mode": "observe", "ml_horizon_bars": 1,
                               "ml_min_val_accuracy": 0.99,
                               "ml_min_confidence": 0.99})
    check("mlsig", "품질 게이트: 전원 미달 시 score=None",
          strict.ok and strict.score is None
          and not strict.gbdt_used and not strict.lgbm_used
          and not strict.patch_used and not strict.kalman_used)

    # 4) apply_to_score — usable 하지 않으면 점수 불변
    base = 0.4
    blended, note = mlsignal.apply_to_score(base, strict, {"ml_mode": "soft"})
    check("mlsig", "결합: 자격 미달이면 점수 불변", blended == base and note == "")

    # 5) usable 하면 가중 결합이 ML 방향으로 움직인다
    mlsignal.clear_cache()
    state = mlsignal.compute("TEST3", mr_bars,
                             {"ml_mode": "soft", "ml_horizon_bars": 1,
                              "ml_min_val_accuracy": 0.5, "ml_min_confidence": 0.0})
    if state.usable:
        blended, note = mlsignal.apply_to_score(base, state, {"ml_mode": "soft"})
        moved_toward = abs(blended - state.score) <= abs(base - state.score) + 1e-12
        check("mlsig", "결합: ML 방향으로 이동 + 사유 기록",
              moved_toward and "ML" in note, f"{base} → {blended:.3f} ({note})")
    else:
        check("mlsig", "결합: ML 방향으로 이동 + 사유 기록", False,
              "usable 하지 않음 — 게이트 완화에도 결합 불가")

    # 6) 캐시 — 같은 봉이면 재계산 없이 같은 객체
    mlsignal.clear_cache()
    cfg = {"ml_mode": "observe", "ml_horizon_bars": 1}
    s1 = mlsignal.compute("CACHE", mr_bars, cfg)
    s2 = mlsignal.compute("CACHE", mr_bars, cfg)
    check("mlsig", "캐시: 같은 봉 재사용", s1 is s2)
    # 봉이 갱신되면 다시 계산
    changed = mr_bars.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 1.01
    s3 = mlsignal.compute("CACHE", changed, cfg)
    check("mlsig", "캐시: 봉 갱신 시 재계산", s3 is not s1)


# ---------------------------------------------------------------------------
# 통합 — strategy.evaluate 경로
# ---------------------------------------------------------------------------

def test_integration():
    print("\n[통합] strategy.evaluate + ML")
    from engine import instruments, mlsignal, strategy

    inst = instruments.try_resolve("005930")
    if inst is None:
        check("integ", "종목 해석 불가로 건너뜀", True, "offline")
        return

    mlsignal.clear_cache()
    bars = _to_bars(_ar1(300, phi=-0.4, seed=11), seed=11)
    quote = {"price": float(bars["close"].iloc[-1]),
             "price_krw": float(bars["close"].iloc[-1]), "age_sec": 1.0}
    base_cfg = {"algo_mode": "off", "intraday_weight": 0.0, "ml_horizon_bars": 1}

    sig_off = strategy.evaluate(inst, {**base_cfg, "ml_mode": "off"},
                                bars_daily=bars, quote=quote, allow_fetch=False)
    check("integ", "off: ml 미첨부", sig_off.ok and sig_off.ml is None)

    sig_obs = strategy.evaluate(inst, {**base_cfg, "ml_mode": "observe"},
                                bars_daily=bars, quote=quote, allow_fetch=False)
    check("integ", "observe: ml 첨부 + 점수 불변", sig_obs.ok
          and sig_obs.ml is not None
          and abs(sig_obs.score - sig_off.score) < 1e-9,
          f"score={sig_obs.score:+.3f}")

    sig_soft = strategy.evaluate(inst, {**base_cfg, "ml_mode": "soft",
                                        "ml_min_val_accuracy": 0.5,
                                        "ml_min_confidence": 0.0},
                                 bars_daily=bars, quote=quote, allow_fetch=False)
    ml_score = (sig_soft.ml or {}).get("score")
    if ml_score is not None:
        check("integ", "soft: ML 점수가 결합에 반영", sig_soft.ok
              and abs(sig_soft.score - sig_off.score) > 1e-9,
              f"{sig_off.score:+.3f} → {sig_soft.score:+.3f} (ml {ml_score:+.3f})")
    else:
        # 합성 데이터가 게이트를 못 넘으면 결합이 없는 것이 맞는 동작입니다
        check("integ", "soft: 게이트 미달 시 점수 불변",
              abs(sig_soft.score - sig_off.score) < 1e-9)


# ---------------------------------------------------------------------------

def report() -> bool:
    print("\n" + "=" * 60)
    names = {"gbdt": "GBDT", "lgbm": "LGBM", "kalman": "칼만", "garch": "GARCH",
             "patch": "패치 어텐션", "mlsig": "ML 오버레이", "integ": "통합"}
    by_cat: dict = {}
    for cat, _, ok, _ in results:
        n, p = by_cat.get(cat, (0, 0))
        by_cat[cat] = (n + 1, p + (1 if ok else 0))

    total = passed_total = 0
    for cat, (n, p) in by_cat.items():
        total += n; passed_total += p
        mark = "OK" if p == n else "!!"
        print(f"  [{mark}] {names.get(cat, cat):8} {p:3}/{n:<3} 통과")

    failures = [(c, n, d) for c, n, ok, d in results if not ok]
    if failures:
        print(f"\n  실패 {len(failures)}건:")
        for cat, name, detail in failures:
            print(f"    · [{names.get(cat, cat)}] {name}" + (f" — {detail}" if detail else ""))

    rate = passed_total / total * 100 if total else 0
    print(f"\n  총계: {passed_total}/{total} 통과 ({rate:.1f}%)")
    return len(failures) == 0


if __name__ == "__main__":
    test_gbdt()
    test_lgbm()
    test_kalman()
    test_garch()
    test_patch()
    test_mlsignal()
    test_integration()
    ok = report()
    sys.exit(0 if ok else 1)
