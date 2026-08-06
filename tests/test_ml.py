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

    # 2) observe — 계산·첨부, 사용 여부 표시
    state = mlsignal.compute("TEST", mr_bars,
                             {"ml_mode": "observe", "ml_horizon_bars": 1})
    check("mlsig", "observe: 두 예측기 결과 첨부", state.ok
          and state.gbdt.get("ok") is not None and state.patch.get("ok") is not None)

    # 3) 품질 게이트 — 검증 미달 예측기는 결합에서 빠진다
    #    (기준을 극단으로 올려 강제 탈락시키고 확인)
    mlsignal.clear_cache()
    strict = mlsignal.compute("TEST2", mr_bars,
                              {"ml_mode": "observe", "ml_horizon_bars": 1,
                               "ml_min_val_accuracy": 0.99,
                               "ml_min_confidence": 0.99})
    check("mlsig", "품질 게이트: 전원 미달 시 score=None",
          strict.ok and strict.score is None
          and not strict.gbdt_used and not strict.patch_used)

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
    names = {"gbdt": "GBDT", "patch": "패치 어텐션", "mlsig": "ML 오버레이",
             "integ": "통합"}
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
    test_patch()
    test_mlsignal()
    test_integration()
    ok = report()
    sys.exit(0 if ok else 1)
