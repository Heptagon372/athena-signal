"""페어 트레이딩 QA — 공적분 검정이 참값을 복원하는가.

engine/pairs.py 의 ADF·Engle-Granger·스프레드 품질 필터를 인공 데이터로
검증합니다. 네트워크가 필요 없습니다.

    python tests/test_pairs.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from engine import pairs

rng = np.random.default_rng(31)
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


def ou(n, theta, sigma, rng):
    """Ornstein-Uhlenbeck — 참 반감기 = ln(2)/theta."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - theta * x[t - 1] + rng.normal(0, sigma)
    return x


print("=" * 72)
print("  1. ADF — 정상/비정상을 구분하는가")
print("=" * 72)
rw_reject = st_reject = 0
TRIALS = 100
for _ in range(TRIALS):
    if pairs.adf_test(np.cumsum(rng.normal(0, 1, 400)))["stationary_5pct"]:
        rw_reject += 1
    if pairs.adf_test(ou(400, 0.05, 1.0, rng))["stationary_5pct"]:
        st_reject += 1
print(f"      랜덤워크(비정상)를 '정상'이라 오판: {rw_reject}/{TRIALS} "
      f"({rw_reject/TRIALS:.0%}, 5% 근처여야 정상)")
print(f"      OU(정상)를 '정상'이라 판정:        {st_reject}/{TRIALS} "
      f"({st_reject/TRIALS:.0%}, 높을수록 좋음)")
check("랜덤워크 위양성 <= 12%", rw_reject / TRIALS <= 0.12)
check("OU 검출력 >= 70%", st_reject / TRIALS >= 0.70)

print()
print("=" * 72)
print("  2. Engle-Granger — 참 헤지비율을 복원하는가")
print("=" * 72)
for true_beta in (0.5, 1.0, 2.0):
    est = []
    for _ in range(20):
        x = np.cumsum(rng.normal(0, 0.01, 500)) + 5.0
        y = 1.2 + true_beta * x + ou(500, 0.08, 0.01, rng)   # 공적분 관계
        eg = pairs.engle_granger(y, x)
        if eg:
            est.append(eg["hedge_ratio"])
    m = float(np.mean(est))
    print(f"      참 beta={true_beta}  ->  추정 {m:.4f} (표준편차 {np.std(est):.4f})")
    check(f"헤지비율 복원 (beta={true_beta})", abs(m - true_beta) < 0.05)

# 공적분되지 않은 쌍은 걸러내는가
coint_false = 0
for _ in range(60):
    a = np.cumsum(rng.normal(0, 0.01, 500)) + 5
    b = np.cumsum(rng.normal(0, 0.01, 500)) + 5      # 서로 무관한 랜덤워크
    eg = pairs.engle_granger(a, b)
    if eg and eg["cointegrated_5pct"]:
        coint_false += 1
print(f"      무관한 랜덤워크 쌍을 '공적분'이라 오판: {coint_false}/60 "
      f"({coint_false/60:.0%})")
check("가짜 공적분 위양성 <= 12%", coint_false / 60 <= 0.12)

# 진짜 공적분 쌍은 잡아내는가
coint_true = 0
for _ in range(60):
    x = np.cumsum(rng.normal(0, 0.01, 500)) + 5
    y = 1.0 + 1.5 * x + ou(500, 0.08, 0.01, rng)
    eg = pairs.engle_granger(y, x)
    if eg and eg["cointegrated_5pct"]:
        coint_true += 1
print(f"      진짜 공적분 쌍 검출: {coint_true}/60 ({coint_true/60:.0%})")
check("공적분 검출력 >= 80%", coint_true / 60 >= 0.80)

print()
print("=" * 72)
print("  3. 스프레드 품질 — 참 반감기와 Hurst 를 맞추는가")
print("=" * 72)
for theta, lab in ((0.15, "빠름"), (0.05, "보통"), (0.02, "느림")):
    true_hl = math.log(2) / theta
    hls, hus = [], []
    for _ in range(15):
        q = pairs.spread_quality(ou(600, theta, 0.01, rng))
        if q.get("half_life"):
            hls.append(q["half_life"]); hus.append(q["hurst"])
    print(f"      theta={theta} ({lab})  참 반감기 {true_hl:.1f}일  "
          f"-> 추정 {np.mean(hls):.1f}일   Hurst {np.mean(hus):.3f}")
    check(f"반감기 복원 ({lab})", abs(np.mean(hls) - true_hl) < true_hl * 0.45)
    check(f"평균회귀를 반지속으로 판정 ({lab}, Hurst<0.5)", np.mean(hus) < 0.5)

# 랜덤워크 스프레드는 거래 불가로 판정해야 함
rw_tradable = sum(1 for _ in range(40)
                  if pairs.spread_quality(np.cumsum(rng.normal(0, 1, 600)))["tradable"])
print(f"      랜덤워크 스프레드를 '거래 가능'이라 판정: {rw_tradable}/40")
check("랜덤워크 스프레드는 거부", rw_tradable <= 4)

good = sum(1 for _ in range(40)
           if pairs.spread_quality(ou(600, 0.08, 0.01, rng))["tradable"])
print(f"      좋은 OU 스프레드를 '거래 가능'이라 판정: {good}/40")
check("좋은 스프레드는 통과 >= 60%", good >= 24)

print()
print("=" * 72)
print("  4. find_pairs — 심어놓은 페어만 찾아내는가")
print("=" * 72)
# 앞 섹션이 소비한 난수에 따라 데이터가 달라지지 않도록 전용 시드를 씁니다.
# (이 테스트는 특정 난수 추출에 의존하면 안 됩니다)
r4 = np.random.default_rng(2026)
n = 600
base = {}
for i in range(8):
    base[f"N{i}"] = np.exp(np.cumsum(r4.normal(0, 0.012, n)) + 4)     # 무관
x = np.cumsum(r4.normal(0, 0.012, n)) + 4
base["P_A"] = np.exp(x)
base["P_B"] = np.exp((x - 0.3) / 1.0 + ou(n, 0.10, 0.012, r4))        # P_A 와 공적분
found = pairs.find_pairs(base)
names = {tuple(sorted((p["y"], p["x"]))) for p in found}
print(f"      찾은 페어 {len(found)}개: "
      + ", ".join(f"{p['y']}~{p['x']}(H={p['hurst']:.2f},HL={p['half_life']:.0f})"
                  for p in found[:5]))
check("심어놓은 페어를 찾음", ("P_A", "P_B") in names)
check("무관한 쌍은 거의 안 잡힘 (<= 3개)", len(found) - (1 if ("P_A","P_B") in names else 0) <= 3)

print()
print("=" * 72)
print("  5. z-score 신호")
print("=" * 72)
sp = ou(400, 0.08, 1.0, rng)
sig = pairs.zscore_signal(sp, window=60)
z = sig["z"][~np.isnan(sig["z"])]
print(f"      z 평균 {z.mean():+.3f}  표준편차 {z.std():.3f}  "
      f"|z|>2 비율 {np.mean(np.abs(z)>2):.1%}")
check("z 가 표준화됨 (평균 0 근처)", abs(z.mean()) < 0.35)
# 이동창 표준편차는 자기상관 때문에 무조건분산을 과소추정하고, 그만큼 z 가
# 부풀립니다(실측 1.3 부근). 버그가 아니라 구조적 성질입니다 — 다만 "z>2 = 2시그마"
# 라고 읽으면 안 되고, 실제 발동 빈도는 명목보다 높습니다.
check("z 표준편차가 1~1.5 (이동창 자기상관 효과 포함)", 0.8 <= z.std() <= 1.5)
check("손절 임계값이 진입보다 큼", sig["stop"] > sig["entry"])

print()
print("=" * 72)
print(f"  결과: {len(PASS)} 통과 / {len(FAIL)} 실패")
for f in FAIL:
    print(f"    실패: {f}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
