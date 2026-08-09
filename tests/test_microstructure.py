"""LPPLS · Avellaneda-Stoikov 검증."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from engine import econophysics as ep
from engine import microstructure as ms

rng = np.random.default_rng(11)
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


def synth_lppls(n, tc, m, w, a=5.0, b=-0.02, c=0.0012, noise=0.0, rng=None):
    """감쇠조건 D = m|B|/(w|C|) >= 0.8 을 만족하는 '유효한' 버블을 만듭니다.
    (m=0.45, w=9, B=-0.02, C=0.0012 -> D = 0.83)"""
    t = np.arange(n, dtype=float)
    dt = np.abs(tc - t) + 1e-8
    y = a + np.power(dt, m) * (b + c * np.cos(w * np.log(dt)))
    if noise > 0:
        y = y + rng.normal(0, noise, n)
    return y


print("=" * 72)
print("  1. LPPLS — 인공 버블에서 임계시각 t_c 를 복원하는가")
print("=" * 72)
for true_tc_ahead in (20.0, 40.0):
    errs, ms_, ws_ = [], [], []
    for _ in range(8):
        n = 250
        tc = (n - 1) + true_tc_ahead
        y = synth_lppls(n, tc, m=0.45, w=9.0, noise=0.004, rng=rng)
        r = ep.lppls_fit(y, tc_max_frac=0.5)
        if r:
            errs.append(r["tc_ahead"] - true_tc_ahead)
            ms_.append(r["m"]); ws_.append(r["omega"])
    me = float(np.mean(np.abs(errs)))
    print(f"      참 t_c={true_tc_ahead:.0f}봉 앞:  평균절대오차 {me:.1f}봉  "
          f"m={np.mean(ms_):.2f}(참 0.45)  omega={np.mean(ws_):.1f}(참 9.0)")
    check(f"t_c 복원 오차 <= 12봉 (t_c={true_tc_ahead:.0f})", me <= 12.0)

# 노이즈 없는 완전한 LPPLS
y = synth_lppls(300, 299 + 30, m=0.5, w=8.0)
r = ep.lppls_fit(y, tc_max_frac=0.5)
print(f"      무노이즈:  t_c_ahead={r['tc_ahead']:.1f}(참 30)  R2={r['r_squared']:.4f}  "
      f"필터통과={r['qualified']}  부호={r['bubble_sign']}")
check("무노이즈 적합 R2 >= 0.99", r["r_squared"] >= 0.99)
check("양의 버블로 판정 (B<0)", r["bubble_sign"] == "positive")

print()
print("=" * 72)
print("  2. LPPLS 위양성 — 랜덤워크를 버블로 오판하는가 (가장 중요)")
print("=" * 72)
qual = 0
TRIALS = 40
confs = []
for _ in range(TRIALS):
    y = np.log(100) + np.cumsum(rng.normal(0, 0.015, 250))
    r = ep.lppls_fit(y)
    if r and r["qualified"]:
        qual += 1
print(f"      단일적합 필터통과 {qual}/{TRIALS} ({qual/TRIALS:.0%})")

for _ in range(20):
    y = np.log(100) + np.cumsum(rng.normal(0, 0.015, 400))
    c = ep.lppls_confidence(y)
    if c:
        confs.append(c["positive_conf"])
print(f"      신뢰지표(다중창) 양의버블 신뢰도: 평균 {np.mean(confs):.3f}  "
      f"최대 {np.max(confs):.3f}")
check("랜덤워크 신뢰지표 평균 <= 0.35", float(np.mean(confs)) <= 0.35)

# 진짜 버블에서는 신뢰도가 높아야 함
bub = []
for _ in range(12):
    y = synth_lppls(400, 399 + 35, m=0.45, w=9.0, noise=0.006, rng=rng)
    c = ep.lppls_confidence(y)
    if c:
        bub.append(c["positive_conf"])
print(f"      인공 버블 신뢰도: 평균 {np.mean(bub):.3f}")
check("인공 버블 신뢰도 > 랜덤워크 신뢰도",
      float(np.mean(bub)) > float(np.mean(confs)))

print()
print("=" * 72)
print("  3. Avellaneda-Stoikov — 재고가 호가를 미는가")
print("=" * 72)
base = ms.avellaneda_stoikov(100.0, inventory=0, volatility=0.5, time_left=1.0)
long_inv = ms.avellaneda_stoikov(100.0, inventory=5, volatility=0.5, time_left=1.0)
short_inv = ms.avellaneda_stoikov(100.0, inventory=-5, volatility=0.5, time_left=1.0)
print(f"      재고  0: 유보 {base['reservation_price']:.3f}  "
      f"[{base['bid']:.3f}, {base['ask']:.3f}]  스프레드 {base['spread']:.3f}")
print(f"      재고 +5: 유보 {long_inv['reservation_price']:.3f}  "
      f"[{long_inv['bid']:.3f}, {long_inv['ask']:.3f}]  밀림 {long_inv['inventory_skew']:+.3f}")
print(f"      재고 -5: 유보 {short_inv['reservation_price']:.3f}  "
      f"[{short_inv['bid']:.3f}, {short_inv['ask']:.3f}]  밀림 {short_inv['inventory_skew']:+.3f}")
check("재고 0이면 유보가격 = 중간가", abs(base["inventory_skew"]) < 1e-12)
check("롱 재고면 호가가 내려감 (팔기 쉬워짐)", long_inv["ask"] < base["ask"])
check("숏 재고면 호가가 올라감", short_inv["bid"] > base["bid"])
check("밀림이 재고에 대칭", abs(long_inv["inventory_skew"] + short_inv["inventory_skew"]) < 1e-12)

hi_vol = ms.avellaneda_stoikov(100.0, 0, volatility=1.0, time_left=1.0)
check(f"변동성 2배 -> 스프레드 확대 ({base['spread']:.3f} -> {hi_vol['spread']:.3f})",
      hi_vol["spread"] > base["spread"])
short_t = ms.avellaneda_stoikov(100.0, 0, volatility=0.5, time_left=0.1)
check(f"마감 임박 -> 스프레드 축소 ({base['spread']:.3f} -> {short_t['spread']:.3f})",
      short_t["spread"] < base["spread"])

print()
print("=" * 72)
print("  4. 체결강도 보정 — 참 kappa 를 복원하는가")
print("=" * 72)
true_A, true_k = 140.0, 1.5
d = np.linspace(0.05, 2.0, 40)
lam = true_A * np.exp(-true_k * d) * np.exp(rng.normal(0, 0.05, 40))
cal = ms.calibrate_intensity(d, lam)
print(f"      추정 A={cal['A']:.1f}(참 {true_A})  kappa={cal['kappa']:.3f}(참 {true_k})  "
      f"R2={cal['r_squared']:.4f}")
check("kappa 복원 오차 < 5%", abs(cal["kappa"] - true_k) / true_k < 0.05)
check("모형과 반대 데이터는 거부",
      ms.calibrate_intensity(d, true_A * np.exp(+true_k * d)) is None)

print()
print("=" * 72)
print("  5. A-S 시뮬레이션 — 재고 통제가 실제로 작동하는가")
print("=" * 72)


def simulate(gamma, n_sim=300, N=200):
    """참조 구현(fedecaccia)과 같은 설정, 스프레드만 원논문 유한시간 식."""
    sigma, T, k = 2.0, 1.0, 1.5
    dt = T / N
    A = 1.0 / dt / math.exp(k * (100 / 200) / 2)
    pnls, max_abs_q = [], []
    r2 = np.random.default_rng(99)
    for _ in range(n_sim):
        s = 100 + np.cumsum(r2.normal(0, sigma * math.sqrt(dt), N + 1))
        q = cash = 0.0
        peak = 0.0
        for n in range(N + 1):
            quote = ms.avellaneda_stoikov(s[n], q, sigma, T - dt * n,
                                          gamma=gamma, kappa=k)
            da, db = quote["ask"] - s[n], s[n] - quote["bid"]
            if r2.random() < 1 - math.exp(-A * math.exp(-k * da) * dt):
                cash += quote["ask"]; q -= 1
            if r2.random() < 1 - math.exp(-A * math.exp(-k * db) * dt):
                cash -= quote["bid"]; q += 1
            peak = max(peak, abs(q))
        pnls.append(cash + q * s[-1])
        max_abs_q.append(peak)
    return np.array(pnls), np.array(max_abs_q)


for g in (0.01, 0.1, 0.5):
    pnl, mq = simulate(g)
    print(f"      gamma={g:<5}  PnL 평균 {pnl.mean():+7.2f}  표준편차 {pnl.std():6.2f}  "
          f"최대재고 평균 {mq.mean():5.1f}")

p_lo, q_lo = simulate(0.01)
p_hi, q_hi = simulate(0.5)
check("위험회피가 크면 재고를 덜 쌓음", q_hi.mean() < q_lo.mean())
check("위험회피가 크면 PnL 변동성이 작음", p_hi.std() < p_lo.std())

print()
print("=" * 72)
print(f"  결과: {len(PASS)} 통과 / {len(FAIL)} 실패")
for f in FAIL:
    print(f"    실패: {f}")
print("=" * 72)
sys.exit(1 if FAIL else 0)

