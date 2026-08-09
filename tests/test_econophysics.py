"""경제물리학 추정량 QA — 참값을 아는 인공 데이터에서 복원되는가.

engine/econophysics.py 의 각 추정량을 몬테카를로로 검증합니다.
네트워크가 필요 없습니다.

    python tests/test_econophysics.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from engine import econophysics as ep

rng = np.random.default_rng(42)
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


def fbm_returns(n, H, rng):
    """fractional Gaussian noise via Davies-Harte (spectral) — 참 H를 갖는 증분."""
    # 자기공분산 gamma(k) = 0.5*(|k-1|^2H - 2|k|^2H + |k+1|^2H)
    k = np.arange(n)
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    circ = np.concatenate([g, g[-2:0:-1]])
    lam = np.fft.fft(circ).real
    lam[lam < 0] = 0
    m = len(circ)
    w = rng.normal(size=m) + 1j * rng.normal(size=m)
    y = np.fft.fft(np.sqrt(lam / (2 * m)) * w)
    return y[:n].real


print("=" * 70)
print("  1. DFA — 참 H를 복원하는가 (fGn, n=2000, 20회 평균)")
print("=" * 70)
for true_h in (0.3, 0.5, 0.7):
    est = [ep.dfa(fbm_returns(2000, true_h, rng))["alpha"] for _ in range(20)]
    mean, sd = float(np.mean(est)), float(np.std(est))
    check(f"DFA H={true_h} -> {mean:.3f} (sd {sd:.3f})", abs(mean - true_h) < 0.06)

# 추세 강건성 — R/S 와 비교
print()
print("=" * 70)
print("  2. DFA 추세 강건성 — 드리프트를 넣어도 alpha가 안 부푸는가")
print("=" * 70)
from engine import quant

# (a) 상수 드리프트 — R/S 도 구간별 평균을 빼므로 둘 다 강건해야 정상입니다
d1, r1 = [], []
for _ in range(25):
    r = rng.normal(0, 0.02, 800) + 0.0015
    d1.append(ep.dfa(r)["alpha"])
    h = quant.hurst_exponent(100 * np.exp(np.cumsum(r)))
    if h:
        r1.append(h["hurst"])
print(f"      (a) 상수 드리프트(참값 0.5):   DFA={np.mean(d1):.3f}   고전 R/S={np.mean(r1):.3f}")
check("상수 드리프트: 둘 다 강건 (기대된 동작)",
      abs(np.mean(d1) - 0.5) < 0.06 and abs(np.mean(r1) - 0.5) < 0.06)

# (b) 시변 추세 — 구간 평균 제거로는 안 없어지는 비정상 추세.
#     DFA 는 구간별 다항 추세를 명시적으로 빼므로 여기서 차이가 납니다.
d2, r2, d2o2 = [], [], []
for _ in range(25):
    n = 800
    t = np.arange(n) / n
    trend = 0.010 * np.sin(2 * np.pi * t * 1.5) + 0.004 * t   # 진폭이 변하는 추세
    r = rng.normal(0, 0.02, n) + trend
    d2.append(ep.dfa(r, order=1)["alpha"])
    d2o2.append(ep.dfa(r, order=2)["alpha"])       # 모듈 기본값
    h = quant.hurst_exponent(100 * np.exp(np.cumsum(r)))
    if h:
        r2.append(h["hurst"])
d_m, d_m2, r_m = float(np.mean(d2)), float(np.mean(d2o2)), float(np.mean(r2))
print(f"      (b) 시변 추세(참값 0.5):       DFA1={d_m:.3f}  DFA2={d_m2:.3f}  고전 R/S={r_m:.3f}")
check(f"2차 DFA(기본값)가 참값 복원 ({d_m2:.3f})", abs(d_m2 - 0.5) < 0.06)
check(f"2차가 1차보다 추세에 강건 ({d_m2:.3f} vs {d_m:.3f})", abs(d_m2 - 0.5) < abs(d_m - 0.5))
# 통념 반증 기록: 1차 DFA 는 긴 스케일에서 R/S 보다 오히려 나쁩니다
check(f"[기록] 1차 DFA는 시변추세에서 R/S보다 부풀음 — 통념과 반대",
      abs(d_m - 0.5) > abs(r_m - 0.5))

print()
print("=" * 70)
print("  2-b. DFA t통계 귀무보정 — 랜덤워크에서 5% 만 기각하는가")
print("=" * 70)
cal = ep.dfa_null_calibration(252)
print(f"      귀무분포: 평균 {cal['mean']:+.3f}  표준편차 {cal['sd']:.3f}")
print(f"      보정 전 |t|>2 기각률 {cal['reject_rate_naive']:.1%} (정규분포라면 4.6%)")
check("OLS 표준오차가 t를 크게 부풀림 (보정 필요성)", cal["reject_rate_naive"] > 0.30)
_ts = np.array([t for t in
                (ep.dfa(rng.normal(0, 0.02, 252))["t_calibrated"] for _ in range(300))
                if t is not None])
rate = float(np.mean(np.abs(_ts) > 2))
print(f"      보정 후: 평균 {_ts.mean():+.3f}  표준편차 {_ts.std(ddof=1):.3f}  "
      f"|t|>2 기각률 {rate:.1%}")
check(f"보정 후 기각률이 5% 근처 ({rate:.1%})", 0.02 <= rate <= 0.09)
check("보정 후 평균이 0 근처", abs(float(_ts.mean())) < 0.2)

print()
print("=" * 70)
print("  3. Hill — Pareto 참 alpha를 복원하는가 (좌측 꼬리)")
print("=" * 70)
for true_a in (2.0, 3.0, 4.0):
    est = []
    for _ in range(20):
        # 좌측 꼬리만 Pareto(true_a), 우측은 얇게 — 비대칭 상황
        u = rng.random(1500)
        left = -(u[:750] ** (-1.0 / true_a)) * 0.01
        right = rng.normal(0, 0.005, 750)
        r = np.concatenate([left, right])
        rng.shuffle(r)
        h = ep.hill_tail(r, side="left")
        if h:
            est.append(h["alpha"])
    mean, sd = float(np.mean(est)), float(np.std(est))
    check(f"Hill left alpha={true_a} -> {mean:.3f} (sd {sd:.3f})", abs(mean - true_a) < 0.45)

# 좌우 비대칭을 실제로 구분하는가
print()
u = rng.random(4000)
left_heavy = np.where(rng.random(4000) < 0.5, -(u ** (-1 / 2.2)) * 0.01, rng.normal(0, 0.005, 4000))
hl = ep.hill_tail(left_heavy, side="left")
hr = ep.hill_tail(left_heavy, side="right")
print(f"      좌꼬리 두껍게 만든 표본:  left alpha={hl['alpha']:.2f}   right alpha={hr['alpha']:.2f}")
check("좌우 꼬리를 구분함 (left < right)", hl["alpha"] < hr["alpha"])

print()
print("=" * 70)
print("  4. Marchenko-Pastur — 순수 잡음의 최대 고유값이 경계 안에 드는가")
print("=" * 70)
inside = 0
TRIALS = 30
for _ in range(TRIALS):
    noise = rng.normal(size=(40, 500))       # 상관 전혀 없음
    res = ep.rmt_decompose(noise)
    if res and not res["market_mode_valid"]:
        inside += 1
check(f"순수 잡음에서 '시장 모드 없음' 판정 {inside}/{TRIALS}", inside >= TRIALS * 0.8)

# 진짜 공통 인자를 넣으면 잡아내는가
found = 0
for _ in range(30):
    factor = rng.normal(size=500)
    noise = rng.normal(size=(40, 500))
    data = 0.6 * factor[None, :] + 0.8 * noise      # 공통 시장 인자
    res = ep.rmt_decompose(data)
    if res and res["market_mode_valid"]:
        found += 1
check(f"공통 인자가 있으면 시장 모드 탐지 {found}/30", found >= 28)

res = ep.rmt_decompose(0.6 * rng.normal(size=500)[None, :] + 0.8 * rng.normal(size=(40, 500)))
print(f"      공통인자 표본:  lambda_max={res['lambda_max']:.2f}  "
      f"lambda_plus={res['lambda_plus']:.2f}  신호모드={res['n_signal']}  "
      f"시장비중={res['market_share']:.3f}")

check("자산 수 부족(N=5)이면 계산 거부", ep.rmt_decompose(rng.normal(size=(5, 500))) is None)

print()
print("=" * 70)
print("  5. Lo(1991) 수정 R/S — 단기 자기상관을 장기기억으로 오독하지 않는가")
print("=" * 70)
classic_reject = modified_reject = 0
TRIALS = 60
for _ in range(TRIALS):
    # AR(1) — 단기 의존성만 있고 장기기억은 없음
    e = rng.normal(0, 1, 600)
    ar = np.zeros(600)
    for t in range(1, 600):
        ar[t] = 0.45 * ar[t - 1] + e[t]
    res = ep.modified_rs(ar)
    if res:
        if res["classic_v"] > 1.862:
            classic_reject += 1
        if res["v_stat"] > 1.862:
            modified_reject += 1
print(f"      AR(1) phi=0.45 (장기기억 없음), {TRIALS}회:")
print(f"        고전 R/S 오탐  {classic_reject}/{TRIALS} ({classic_reject/TRIALS:.0%})")
print(f"        Lo 수정 R/S 오탐 {modified_reject}/{TRIALS} ({modified_reject/TRIALS:.0%})")
check("Lo 수정판이 고전보다 오탐이 적음", modified_reject < classic_reject)
check("Lo 수정판 오탐률 <= 20%", modified_reject / TRIALS <= 0.20)

print()
print("=" * 70)
print("  6. Chow-Denning — 다중검정 임계값이 단일검정보다 보수적인가")
print("=" * 70)
cd = ep.chow_denning(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
print(f"      임계값: 다중 {cd['critical']:.3f}  vs  단일 1.96  (m={cd['m_tests']})")
check("다중검정 임계값이 더 높음", cd["critical"] > 1.96)

false_single = false_multi = 0
TRIALS = 100
for _ in range(TRIALS):
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))   # 순수 랜덤워크
    cd = ep.chow_denning(p)
    if cd:
        if any(abs(s["z"]) > 1.96 for s in cd["by_q"]):
            false_single += 1
        if cd["reject_random_walk"]:
            false_multi += 1
print(f"      순수 랜덤워크 {TRIALS}회 위양성:")
print(f"        q별 개별검정 중 하나라도 유의  {false_single}/{TRIALS} ({false_single/TRIALS:.0%})")
print(f"        Chow-Denning 동시검정          {false_multi}/{TRIALS} ({false_multi/TRIALS:.0%})")
check("동시검정 위양성이 개별검정보다 낮음", false_multi <= false_single)
check("동시검정 위양성 <= 10%", false_multi / TRIALS <= 0.10)

print()
print("=" * 70)
print("  6-b. 시장 진단 — 동조화/독립을 구분하고 경고를 내는가 (표시 전용)")
print("=" * 70)
import pandas as pd
_dates = pd.date_range("2024-01-01", periods=250, freq="B")
_mkt = rng.normal(0, 0.01, 250)
_coupled = {f"S{i}": pd.Series(0.8 * _mkt + rng.normal(0, 0.004, 250), index=_dates)
            for i in range(18)}
_indep = {f"I{i}": pd.Series(rng.normal(0, 0.01, 250), index=_dates) for i in range(18)}
_dc = ep.market_diagnostics(_coupled)
_di = ep.market_diagnostics(_indep)
print(f"      동조 종목: 시장모드 {_dc['rmt']['market_share']:.3f} -> {_dc['stress_level']}")
print(f"      독립 종목: 시장모드 {_di['rmt']['market_share']:.3f} -> {_di['stress_level']}")
check("동조 시장을 high 로 판정", _dc["stress_level"] == "high")
check("독립 종목은 '시장 모드 없음'", _di["stress_level"] == "none")
check("동조 시 경고 메시지 생성", len(_dc["messages"]) >= 1)
check("종목 수 부족(10개)이면 RMT 계산 안 함",
      ep.market_diagnostics({k: v for k, v in list(_indep.items())[:10]})["rmt"] is None)

# 날짜가 어긋난 종목이 섞여도 정렬되는가
_halted = dict(_coupled)
_halted["S0"] = _coupled["S0"].drop(_coupled["S0"].index[40:70])
check("거래정지로 길이가 달라도 날짜로 정렬",
      ep.market_diagnostics(_halted)["rmt"] is not None)

# LPPLS 경보가 버블에서만 뜨는가
_rw = np.log(100) + np.cumsum(rng.normal(0, 0.012, 400))
_dg = ep.market_diagnostics(_indep, index_log_prices=_rw)
check("랜덤워크 지수에는 버블 경보 없음",
      not any("초지수적으로 상승" in m for m in _dg["messages"]))

print()
print("=" * 70)
print("  7. 제곱근 시장충격 — 참여율의 sqrt에 비례하는가")
print("=" * 70)
c = ep.sqrt_impact_cap(daily_volume=1_000_000, daily_volatility=0.03, max_impact_bps=10)
print(f"      일변동성 3%, 허용충격 10bp -> 참여율 상한 {c['max_participation']:.4%}")
print(f"      흔히 쓰는 '거래대금 0.5%' 규칙의 실제 충격: {c['impact_at_half_pct_bps']:.1f}bp")
# 충격 4배 -> 참여율 16배
c2 = ep.sqrt_impact_cap(1_000_000, 0.03, max_impact_bps=40)
ratio = c2["max_participation"] / c["max_participation"]
check(f"충격 4배 허용 -> 참여율 16배 (실측 {ratio:.1f}배)", abs(ratio - 16.0) < 0.1)

print()
print("=" * 70)
print(f"  결과: {len(PASS)} 통과 / {len(FAIL)} 실패")
if FAIL:
    for f in FAIL:
        print(f"    실패: {f}")
print("=" * 70)
sys.exit(1 if FAIL else 0)

