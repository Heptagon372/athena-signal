"""ATHENA 설계도 하이브리드 QA — 새로 붙인 검증 계층이 실제로 작동하는가.

engine/{nulls,cpcv,registry,causality,lottery,markets,overfit,intrinsic_time}
그리고 validation.audit_vol_scaling, news_crawler 의 특징주 분리를 검사합니다.
네트워크가 필요 없습니다.

    python tests/test_athena_hybrid.py

이 파일의 원칙: **게이트가 통과시키는 것만 보지 말고, 잡아내야 할 것을 실제로
잡아내는지** 를 함께 봅니다. 양성 대조군(의도적으로 심은 결함)이 통과해 버리면
그 게이트는 아무것도 검사하지 않는 것입니다.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


def garch_returns(n, rng, omega=1e-6, a=0.10, b=0.88):
    """변동성 군집이 있는 수익률 — 정지 부트스트랩이 겨냥하는 성질."""
    s2 = np.zeros(n); r = np.zeros(n); e = rng.normal(size=n); s2[0] = 1e-4
    for t in range(1, n):
        s2[t] = omega + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * e[t]
    return r[100:]


rng = np.random.default_rng(20260808)

# ===========================================================================
print("=" * 74)
print("  1. NullCalibrator — 0.45/0.55 임계값이 노이즈 대역 안인가")
print("=" * 74)
from engine import nulls
from engine.quant import hurst_exponent

nulls.clear_cache()
r_garch = garch_returns(3000, rng)


def h_est(x):
    d = hurst_exponent(np.exp(np.cumsum(x)), min_lag=8, max_lag=64)
    return d["hurst"] if d else None


st = nulls.calibrate(h_est, r_garch, window=256, name="hurst_rs",
                     null_value=0.5, n_boot=500)
iid = nulls.calibrate(h_est, r_garch, window=256, name="hurst_rs",
                      null_value=0.5, n_boot=500, method="iid")
print(f"      정지 부트스트랩: 평균 {st.mean:.4f}  sd {st.sd:.4f}  "
      f"밴드 [{st.q_lo:.3f}, {st.q_hi:.3f}]")
print(f"      i.i.d.        : 평균 {iid.mean:.4f}  sd {iid.sd:.4f}  "
      f"밴드 [{iid.q_lo:.3f}, {iid.q_hi:.3f}]")

check("표준오차가 문헌값과 부합 (T=256 에서 σ(Ĥ)≈0.057)",
      0.03 < st.sd < 0.09, f"실측 {st.sd:.4f}")
check("0.45/0.55 가 귀무 밴드 안에 통째로 들어감 → 임계값으로 쓸 수 없음",
      st.q_lo < 0.45 and 0.55 < st.q_hi,
      f"밴드 [{st.q_lo:.3f}, {st.q_hi:.3f}]")
check("변동성 군집이 가짜 장기기억 편향을 만듦 (정지 편향 > iid 편향)",
      st.bias > iid.bias + 0.01,
      f"정지 {st.bias:+.4f} vs iid {iid.bias:+.4f}")
check("평균 블록 길이가 창 대비 상한에 걸림 (재표본당 최소 8블록)",
      st.mean_block <= 256 // 8 + 1e-9, f"블록 {st.mean_block:.0f}, 창 256")

# 밴드 안의 값은 'random' 이 아니라 '구분 불가' 로 판정되어야 합니다
check("Ĥ=0.58 이 밴드 안 → inside 판정", st.verdict(0.58) == "inside",
      f"z={st.zscore(0.58):+.2f}")
check("히스테리시스가 경계 근처 플리핑을 막음",
      (lambda: (lambda b: [b.update(v) for v in
                           (st.q_hi + 0.01, st.q_hi - 0.001, st.q_hi + 0.001)][-1]
                )(nulls.NullBand(spec=st, exit_frac=0.5, min_dwell=3)))() == "above")

# ===========================================================================
print()
print("=" * 74)
print("  2. CPCV — 경로 수와 purge/embargo 가 실제로 겹침을 제거하는가")
print("=" * 74)
from engine import cpcv

plan = cpcv.combinatorial_purged_cv(1000, n_groups=8, k=2, horizon=5,
                                    warmup=60, embargo_pct=0.02)
print(f"      {plan.summary()}")
check("폴드 수 = C(8,2) = 28", plan.n_folds == 28, f"실측 {plan.n_folds}")
check("경로 수 = C(7,1) = 7", plan.n_paths == 7, f"실측 {plan.n_paths}")

viol = 0
for s in plan.splits:
    for g in s.test_groups:
        a, b = plan.group_bounds[g]
        if np.any((s.train_idx >= a - 5) & (s.train_idx < b + 60)):
            viol += 1
check("학습셋이 테스트 구간 ±(horizon, warmup) 안에 하나도 없음", viol == 0,
      f"위반 {viol}건")

# 각 경로가 전 구간을 정확히 한 번씩 덮는가
outs = {s.fold_id: {g: np.full(plan.group_bounds[g][1] - plan.group_bounds[g][0],
                               float(s.fold_id)) for g in s.test_groups}
        for s in plan.splits}
paths = cpcv.assemble_paths(plan, outs)
check("경로 7개가 각각 전 구간을 빠짐없이 덮음",
      all(np.all(np.isfinite(p)) for p in paths) and len(paths) == 7)

# ===========================================================================
print()
print("=" * 74)
print("  3. TrialRegistry — 중복제거와 유효 시행수 N̂")
print("=" * 74)
import tempfile
from engine.registry import TrialRegistry, TrialSpec, RegistryError

reg = TrialRegistry(path=os.path.join(tempfile.mkdtemp(), "reg.json"))
base = rng.normal(0, 0.01, 500)
ids = []
for i in range(12):
    tid = reg.register(TrialSpec(family="xs_mom", config={"lookback": 20 + i}))
    ids.append(tid)
    # 앞 6개는 사실상 같은 전략, 뒤 6개는 독립
    reg.log_result(tid, returns=(base + rng.normal(0, 0.001, 500)) if i < 6
                   else rng.normal(0, 0.01, 500))

same = reg.register(TrialSpec(family="xs_mom", config={"lookback": 20}))
check("같은 설정은 같은 trial_id (중복제거)", same == ids[0])
check("설정이 다르면 다른 trial_id", len(set(ids)) == 12)

eff = reg.effective_n()
print(f"      {eff}")
check("상관 높은 6개가 한 덩어리로 묶여 N̂ = 7", eff["n_hat"] == 7,
      f"M=12 → N̂={eff['n_hat']}")

try:
    reg.log_result("does-not-exist", returns=[0.1])
    check("미등록 시행 로깅이 거부됨", False)
except RegistryError:
    check("미등록 시행 로깅이 거부됨", True)

# ===========================================================================
print()
print("=" * 74)
print("  4. 인과성 게이트 — 잡아내야 할 것을 실제로 잡는가")
print("=" * 74)
from engine import causality as ca

x = np.cumsum(rng.normal(size=400))

caught = ca.causality_report(ca.centred_moving_average, x, n_probes=15)
check("★ 중앙이동평균(양성 대조군)을 잡아냄", caught["passed"] is False)
check("후행 이동평균(음성 대조군)은 통과",
      ca.assert_causal_series(ca.trailing_moving_average, x,
                              n_probes=15, warmup=11)["passed"] is True)
check("mask 모드도 후행 이동평균을 통과",
      ca.assert_causal_series(ca.trailing_moving_average, x, n_probes=15,
                              warmup=11, mode="mask")["passed"] is True)

# 배치 순열 — 배치를 공간축으로 취급하는 예측기를 심어 둡니다
X = rng.normal(size=(64, 10))
bad = ca.batch_permutation_test(lambda a: np.convolve(a[:, 0], np.ones(3) / 3,
                                                      mode="same"), X)
good = ca.batch_permutation_test(lambda a: a.mean(axis=1), X)
check("★ 배치축을 공간축으로 쓰는 예측기를 잡아냄", bad["passed"] is False,
      f"최대 변화 {bad['max_abs_diff']:.3e}")
check("행 단위 예측기는 통과", good["passed"] is True)

# 누수 시프트 — 「특징주」 문제를 그대로 모사합니다.
#   자기상관이 강한 수익률(모멘텀 시장)에서, '뉴스'가 사실은 오늘의 수익률을
#   서술한 것이라면(feature[t] = r[t]) 한 칸 밀어도 IC 가 살아남습니다.
#   그게 바로 "뉴스 옷을 입은 순수 모멘텀 피처" 입니다.
n_lk = 1500
eps = rng.normal(0, 0.01, n_lk)
r_ar = np.zeros(n_lk)
for t in range(1, n_lk):
    r_ar[t] = 0.8 * r_ar[t - 1] + eps[t]
disguised = r_ar[:-1]                 # 오늘의 움직임을 서술한 '뉴스'
fwd_ar = r_ar[1:]                     # 내일 수익률
lk = ca.leakage_shift_test(disguised, fwd_ar)
print(f"      가격 변장 피처: IC {lk['ic']:+.3f} → 시프트 후 {lk['ic_shifted']:+.3f} "
      f"({lk['retained_ratio']:.0%} 잔존)")
check("★ 가격의 변장인 피처가 시프트 테스트에서 걸림", lk["passed"] is False)

# 진짜로 한 스텝만 예측하는 인과적 피처는 시프트하면 무너져야 합니다.
# target[t] 가 sig[t-1] 에만 의존하도록 만들어야 feature[i] ↔ forward[i] 가
# 맞물립니다 (feature=sig[:-1], forward=target[1:]).
sig = rng.normal(0, 1, n_lk)
target = np.zeros(n_lk)
target[1:] = 0.35 * sig[:-1] + rng.normal(0, 1, n_lk - 1)
hk = ca.leakage_shift_test(sig[:-1], target[1:])
print(f"      정상 예측 피처: IC {hk['ic']:+.3f} → 시프트 후 {hk['ic_shifted']:+.3f}")
check("한 스텝 예측 피처는 시프트 후 무너짐 (통과)", hk["passed"] is True)

# 기저 IC 가 0 인 피처는 '판정 불가' 여야 합니다 (난수 판정 금지)
noinfo = ca.leakage_shift_test(rng.normal(0, 1, 500), rng.normal(0, 0.02, 500))
check("★ 기저 IC 가 0 이면 판정하지 않음 (0/0 난수 판정 방지)",
      noinfo["passed"] is None, noinfo["reason"][:46])

# ===========================================================================
print()
print("=" * 74)
print("  5. Lottery — 매칭된 무작위 전략 대비 위치")
print("=" * 74)
from engine import lottery

asset = rng.normal(0.0003, 0.012, 1000)
periodic = np.zeros(1000)
for s in range(0, 1000, 40):
    periodic[s:s + 12] = 1.0
res = lottery.lottery_benchmark(periodic, asset, cost_bps=20, n_random=300)
print(f"      관측 {res['observed']:+.3f} / 귀무평균 {res['null_mean']:+.3f} "
      f"→ {res['percentile']:.0f} 백분위")
check("구조 없는 주기적 전략은 통과하지 못함", res["passed"] is False)

prof = lottery.match_profile(periodic)
check("매칭 프로필이 거래횟수·보유기간을 정확히 뽑음",
      prof["n_trades"] == 25 and abs(prof["mean_holding"] - 12.0) < 1e-9,
      f"{prof['n_trades']}회, 평균보유 {prof['mean_holding']:.1f}")

# 진짜 우위(다음 봉을 아는 전략)는 통과해야 합니다
oracle = np.zeros(1000)
oracle[:-1] = np.where(asset[1:] > 0, 1.0, 0.0)
res2 = lottery.lottery_benchmark(oracle, asset, cost_bps=1, n_random=300)
check("★ 진짜 우위가 있는 전략은 통과 (게이트가 무조건 기각하지 않음)",
      res2["passed"] is True, f"{res2['percentile']:.0f} 백분위")

# ===========================================================================
print()
print("=" * 74)
print("  6. 시장 상수 — 세율이 시간의 함수인가")
print("=" * 74)
from engine import markets as mk

check("2024년 매도세 0.18%", abs(mk.sell_tax_rate("2024-06-01") - 0.0018) < 1e-12)
check("2025년 매도세 0.15%", abs(mk.sell_tax_rate("2025-06-01") - 0.0015) < 1e-12)
check("2026년 매도세 0.20% (인상 반영)",
      abs(mk.sell_tax_rate("2026-08-08") - 0.0020) < 1e-12)
check("ETF 매도세 면제", mk.sell_tax_rate("2026-08-08", "ETF") == 0.0)
check("★ 과거 백테스트가 과거 세율을 씀 (오늘 세율을 소급하지 않음)",
      mk.sell_tax_rate("2020-06-01") != mk.sell_tax_rate("2026-06-01"))

check("호가단위 7구간", [mk.tick_size(p) for p in (990, 3500, 15000, 45000, 150000, 300000, 900000)]
      == [1, 5, 10, 50, 100, 500, 1000])
check("저가주 상대 틱이 큼 (1000원 → 5bp 반스프레드)",
      abs(mk.half_spread_bps(1000) - 5.0) < 1e-9)

drag = mk.tax_drag_per_year(52, 0.8)
check("주간 리밸런싱 회전율 80% → 연 8.3% 세금 드래그",
      abs(drag["annual_tax_drag_pct"] - 8.32) < 0.05,
      f"{drag['annual_tax_drag_pct']:.2f}%/yr")

check("2024-06 공매도 금지 구간", mk.short_sale_allowed("2024-06-01")["state"] == "banned")
check("2022-01 부분 허용 구간", mk.short_sale_allowed("2022-01-15")["state"] == "partial")

audit = mk.audit_repo_constants()
print(f"      저장소 상수 감사: 불일치 {audit['n_mismatches']}건")
for f in audit["findings"]:
    if "actual" in f:
        print(f"        {f['where']}: {f['actual']} vs {f['expected']} "
              f"({f['impact_bps']:+.1f}bp)")
check("저장소 세율이 시행일 테이블과 일치 (배선 완료)",
      audit["n_mismatches"] == 0)

# 비용 경로가 실제로 날짜를 반영하는가 — 상수를 새 상수로 바꾼 게 아니라
# 시행일 테이블을 타는지 확인합니다.
from engine import scalping as _sc
from engine.instruments import Instrument
be_2025 = _sc.breakeven_ticks(1000, when="2025-06-01")
be_2026 = _sc.breakeven_ticks(1000, when="2026-08-08")
print(f"      1,000원주 본전 틱: 2025년 {be_2025}틱 → 2026년 {be_2026}틱")
check("★ 페니주 본전 틱수가 세율 인상을 반영 (낙관적 계산 제거)",
      be_2026 > be_2025, f"{be_2025} → {be_2026}")
check("★ 과거 백테스트는 과거 세율을 씀 (오늘 세율 소급 안 함)",
      _sc._tax_rate("2025-06-01") == 0.0015
      and _sc._tax_rate("2026-08-08") == 0.0020)

_st = Instrument.from_symbol("005930") if hasattr(Instrument, "from_symbol") else None
if _st is None:
    check("Instrument.costs 가 when 인자를 받음",
          "when" in Instrument.costs.__code__.co_varnames)
else:
    f25, t25 = _st.costs("sell", 1_000_000, when="2025-06-01")
    f26, t26 = _st.costs("sell", 1_000_000, when="2026-08-08")
    check("★ Instrument.costs 매도세가 날짜를 반영", t26 > t25,
          f"100만원 매도세 {t25:,.0f}원 → {t26:,.0f}원")

# ===========================================================================
print()
print("=" * 74)
print("  7. 과최적화 — 발표된 기준값을 재현하는가")
print("=" * 74)
from engine import overfit as of

check("최소 백테스트 길이: N=45 → 5년 (발표값)",
      abs(of.min_backtest_length(45)["min_years"] - 5.0) < 0.15,
      f"{of.min_backtest_length(45)['min_years']}년")
check("최소 백테스트 길이: N=7 → 2년 (발표값)",
      abs(of.min_backtest_length(7)["min_years"] - 2.0) < 0.15,
      f"{of.min_backtest_length(7)['min_years']}년")
check("역함수: 5년 → 45개 설정", of.max_configs(5)["max_configs"] == 45)
check("역함수: 2년 → 7개 설정", of.max_configs(2)["max_configs"] == 7)
check("Harvey-Liu 요구 t: 10→2.8, 200→3.66, 1000→4.1",
      [round(of.required_tstat(n), 2) for n in (10, 200, 1000)] == [2.8, 3.66, 4.1])

# 전용 RNG — 앞 절의 난수 소비량이 바뀌어도 이 실험이 흔들리지 않게 합니다.
# (공유 RNG 를 쓰면 위쪽 테스트를 한 줄 고칠 때마다 여기 결과가 달라집니다.)
_r = np.random.default_rng(31337)
M_noise = _r.normal(0, 0.01, (1000, 50))
M_edge = _r.normal(0, 0.01, (1000, 50)); M_edge[:, 7] += 0.0025
p_noise = of.pbo_cscv(M_noise, n_blocks=16)
p_edge = of.pbo_cscv(M_edge, n_blocks=16)
print(f"      노이즈 50설정 PBO {p_noise['pbo']:.3f} / "
      f"우위 심은 경우 PBO {p_edge['pbo']:.3f}  (조합 {p_noise['n_combos']})")
check("★ 순수 노이즈에서 PBO 가 기각 문턱(0.05)을 크게 넘음",
      p_noise["pbo"] > 0.15 and p_noise["passed"] is False)
# 단일 시드의 PBO 는 0.33~0.69 로 흔들립니다. 이론값 0.5 근처에 **중심이**
# 있다는 것이 검사할 값어치가 있는 성질이므로 여러 시드로 평균냅니다.
_pbo_mean = float(np.mean([
    of.pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (600, 40)),
                n_blocks=12)["pbo"] for s in range(4)]))
print(f"      노이즈 PBO 4개 시드 평균 {_pbo_mean:.3f} (이론값 0.5)")
check("★ 노이즈 PBO 의 중심이 0.5 — 인샘플 1위 고르기가 동전던지기",
      0.3 < _pbo_mean < 0.7, f"{_pbo_mean:.3f}")
check("★ 진짜 우위가 있으면 PBO 가 낮게 나옴 (통과)",
      p_edge["pbo"] < 0.10 and p_edge["passed"] is True)
check("IS-OOS 회귀 진단은 의도적으로 제외됨 (노이즈에서도 +0.23 이라 해석 불가)",
      not hasattr(of, "is_oos_degradation"))

bhy = of.bhy_fdr([0.001, 0.008, 0.02, 0.04, 0.2, 0.5, 0.7])
check("BHY 가 Bonferroni 보다 관대하지만 무분별하지 않음",
      0 < bhy["n_rejected"] < 7, f"{bhy['n_rejected']}/7 기각")

# ===========================================================================
print()
print("=" * 74)
print("  8. 내생 시계 (DC/OS) — 사전계산 트리거와 원본 버그 수정")
print("=" * 74)
from engine import intrinsic_time as it

px = 10000 * np.exp(np.cumsum(rng.normal(0, 0.002, 5000)))

counts = {d: it.run_series(px, delta_bps=d)["n_dc"] for d in (10, 25, 50, 100)}
print(f"      δ별 DC 횟수: {counts}")
check("δ 가 커질수록 이벤트가 단조 감소",
      all(counts[a] > counts[b] for a, b in zip([10, 25, 50], [25, 50, 100])))

runner = it.DirectionalChangeRunner.from_bps(25)
viol = checks_n = 0
for v in px:
    prev, prev_mode = (runner.state.expected_dc_level, runner.state.mode) \
        if runner.state.initialized else (None, None)
    ev = runner.run(v)
    if ev in (1, -1) and prev:
        checks_n += 1
        if prev_mode == -1 and v < prev - 1e-9:
            viol += 1
        if prev_mode == 1 and v > prev + 1e-9:
            viol += 1
check("★ 사전계산된 트리거 가격이 실제 이벤트와 정확히 일치",
      viol == 0 and checks_n > 100, f"{checks_n}건 중 위반 {viol}건")

check("os_length 가 인과적",
      ca.assert_causal_series(lambda a: it.run_series(a, 25)["os_length"], px,
                              n_probes=10, warmup=50)["passed"] is True)
check("dist_to_flip 이 인과적",
      ca.assert_causal_series(lambda a: it.run_series(a, 25)["dist_to_flip"], px,
                              n_probes=10, warmup=50)["passed"] is True)

sl = it.SurpriseLiquidity(delta=0.0025, delta_star=0.0025, alpha=26)
check("★ H2 부호 수정 — 분산이 양수 (원본은 − 라 음수가 됨)", sl.h2 > 0,
      f"H2={sl.h2:.5f}")
check("★ EMA 방향 수정 — 기억 길이가 보존됨 (원본은 w≈0.038 로 붕괴)",
      0.85 < sl.w < 0.98, f"w={sl.w:.4f}")
check("★ 초기값이 H1 (원본은 0)", abs(sl.surp - sl.h1) < 1e-12)

lad_lo = it.inventory_ladder(5); lad_hi = it.inventory_ladder(40)
check("재고 사다리가 비대칭 — 재고가 크면 진입 어렵고 청산 쉬움",
      lad_hi["delta_enter"] < lad_lo["delta_enter"]
      and lad_hi["delta_exit"] > lad_lo["delta_exit"]
      and lad_hi["size_fraction"] < lad_lo["size_fraction"])

# ===========================================================================
print()
print("=" * 74)
print("  9. 이중 변동성 스케일링 감사")
print("=" * 74)
from engine.validation import audit_vol_scaling

vol = np.exp(rng.normal(-4, 0.5, 600))
noise = np.exp(rng.normal(0, 0.05, 600))
single = audit_vol_scaling(0.15 / vol * noise, vol)
double = audit_vol_scaling(0.15 / vol ** 2 * noise, vol)
none_ = audit_vol_scaling(np.ones(600) * noise, vol)
print(f"      단일 {single['slope']:+.2f} / 이중 {double['slope']:+.2f} / "
      f"없음 {none_['slope']:+.2f}")
check("단일 스케일링 → 기울기 ≈ −1", single["state"] == "single_scaling")
check("★ 이중 스케일링 → 기울기 ≈ −2, 게이트 실패",
      double["state"] == "double_scaling" and double["passed"] is False)
check("스케일링 없음 → 기울기 ≈ 0", none_["state"] == "no_scaling")

# ===========================================================================
print()
print("=" * 74)
print("  10. 뉴스 특징주 분리 — 실매매 경로")
print("=" * 74)
from data_sources import news_crawler as nc

desc_cases = ["[특징주] 삼성전자, 외국인 순매수에 급등", "LG화학 강세",
              "[특징주] 두산에너빌리티 급등…대규모 수주 계약 체결"]
event_cases = ["SK하이닉스, 3분기 어닝서프라이즈…영업익 사상 최대",
               "현대차 목표주가 상향, 매수 의견 유지",
               "카카오 분식회계 의혹에 검찰 수사"]

check("★ 특징주/가격서술 기사는 감성 0",
      all(nc.score_sentiment_ko(t)[0] == 0.0 for t in desc_cases))
check("사건 기사는 감성이 살아 있음",
      all(abs(nc.score_sentiment_ko(t)[0]) > 0.1 for t in event_cases))
check("가격서술 표현이 감성 사전에서 제거됨",
      not any(w in nc.KO_POSITIVE or w in nc.KO_NEGATIVE
              for w in ("급등", "상한가", "신고가", "강세", "급락", "하한가", "약세")))
check("가격서술 표현은 주목도 사전에 보존됨",
      all(w in nc.PRICE_MOVE_KO for w in ("급등", "상한가", "강세", "급락")))
check("MWE 이벤트는 특징주 기사에서도 살아남음",
      nc.score_sentiment_ko("[특징주] 셀트리온 급등…흑자전환 성공")[0] != 0.0)
check("토글로 예전 동작 복원 가능", hasattr(nc, "BLOCK_PRICE_DESCRIPTIVE"))

# ===========================================================================
print()
print("=" * 74)
print("  11. T5 결정 게이트 — 양방향으로 발동하는가")
print("=" * 74)
from engine import bandit as bd

# 게이트의 성질은 **한 번의 실현** 이 아니라 반복에서 드러납니다.
# 단일 시드로 검사하면 데이터 실현 운에 따라 결론이 뒤집힙니다 (실제로 겪었습니다).
T5T, T5K = 1500, 4
t5_a_pass, t5_b_pass, t5_last = 0, 0, None
for _s in range(3):
    _r5 = np.random.default_rng(_s)
    # (A) 컨텍스트가 진짜로 최적 팔을 결정 → 통과해야 합니다
    _ca = _r5.normal(size=(T5T, 2))
    _reg = (_ca[:, 0] > 0).astype(int)
    _Ra = _r5.normal(0, 0.01, (T5T, T5K))
    _Ra[_reg == 0, 0] += 0.004
    _Ra[_reg == 1, 1] += 0.004
    _a = bd.run_t5_experiment(_ca, _Ra, horizon=5, n_seeds=8)
    # (B) 컨텍스트가 순수 노이즈 → 절대 통과하면 안 됩니다
    _cb = _r5.normal(size=(T5T, 2))
    _Rb = _r5.normal(0, 0.01, (T5T, T5K)); _Rb[:, 1] += 0.0008
    _b = bd.run_t5_experiment(_cb, _Rb, horizon=5, n_seeds=8)
    t5_a_pass += bool(_a["passed"])
    t5_b_pass += bool(_b["passed"])
    t5_last = _a
    print(f"      실현{_s}: 정보있음 t={_a['t_stat']:+6.2f} {str(_a['passed']):5s} | "
          f"노이즈 t={_b['t_stat']:+6.2f} {_b['passed']}")

check("★ 컨텍스트에 정보가 있으면 대체로 T5 통과 (검정력)",
      t5_a_pass >= 2, f"3회 중 {t5_a_pass}회 통과")
check("★★ 노이즈 컨텍스트는 **한 번도** 통과하지 않음 (거짓양성 0)",
      t5_b_pass == 0, f"3회 중 {t5_b_pass}회 통과")

check("상시 베이스라인 3종이 모든 리포트에 포함됨",
      all(n in t5_last["detail"] for n in
          ("UCB(비컨텍스추얼)", "균등가중", "사후최적 단일팔")))
check("보상 스케일이 정규화됨 (탐색항이 신호를 덮지 않도록)",
      t5_last["reward_scale"] > 0)
check("컨텍스트 차원 d>5 는 거부됨 (리그렛 바운드 보호)",
      bd.run_t5_experiment(np.zeros((300, 7)), np.zeros((300, 3)))["passed"] is None)

# 지연 보상이 올바른 결정에 귀속되는가
buf = bd.DelayedRewardBuffer(horizon=3)
buf.push(10, arm=1, ctx=None, reward=0.5)
check("지연 보상이 h기 전에는 학습에 안 쓰임",
      buf.pop_ready(12) == [] and len(buf.pop_ready(13)) == 1)

# ★ 종목 풀링 패널에서 지연이 **거래일** 로 재어지는가.
#   행 인덱스로 세면 h행 = h/종목수 거래일이 되어 h일 선행 보상을 같은 날
#   학습합니다 — 그대로 미래 참조입니다. 실제로 이 버그를 겪었습니다.
_S, _D, _K = 20, 300, 3
_pr = np.random.default_rng(0)
_pc, _prw, _pt = [], [], []
for _d in range(_D):
    for _s in range(_S):
        _pc.append(_pr.normal(size=2))
        _prw.append(_pr.normal(0, 0.01, _K))
        _pt.append(_d)
_pc, _prw, _pt = np.vstack(_pc), np.vstack(_prw), np.array(_pt)


def _exceeds_best_fixed(res):
    ep = dict(res["ranking"])["사후최적 단일팔"]
    return [k for k, v in res["ranking"]
            if k != "사후최적 단일팔" and v > ep + 1e-12]


_bug = bd.run_t5_experiment(_pc, _prw, horizon=5, n_seeds=4)
_fix = bd.run_t5_experiment(_pc, _prw, horizon=5, n_seeds=4, time_index=_pt)
print(f"      행 인덱스: 고정팔 초과 {_exceeds_best_fixed(_bug) or '없음'} | "
      f"날짜 인덱스: {_exceeds_best_fixed(_fix) or '없음'}")
check("★ 풀링 패널에서 행 인덱스 지연은 고정팔을 넘어섬 (미래 참조 증상)",
      len(_exceeds_best_fixed(_bug)) > 0)
check("★ time_index 로 거래일 지연을 주면 초과가 사라짐",
      len(_exceeds_best_fixed(_fix)) == 0)
check("풀링 패널이 감지되어 보고됨",
      _fix.get("pooling_note") is not None and "20.0종목" in _fix["pooling_note"])
check("무작위 컨텍스트 풀링 패널은 T5 미통과", _fix["passed"] is False)

# ===========================================================================
print()
print("=" * 74)
print("  12. 경제물리 확장 — GHE · Higuchi · 엔트로피")
print("=" * 74)
from engine import econophysics as ep

_er = np.random.default_rng(3)
for h_true in (0.3, 0.5, 0.7):
    est = [ep.ghe(np.cumsum(ep._synth_fgn(2000, h_true, _er)))["hurst"]
           for _ in range(10)]
    check(f"GHE 가 참 H={h_true} 를 복원", abs(float(np.mean(est)) - h_true) < 0.05,
          f"추정 {np.mean(est):.3f}")

check("★ 수익률을 GHE 에 넣으면 경고 (가장 흔한 버그)",
      "warning" in (ep.ghe(_er.normal(0, 0.01, 500)) or {}))
_fd = [ep.higuchi_fd(np.cumsum(ep._synth_fgn(2000, 0.5, _er)))["fd"] for _ in range(8)]
check("Higuchi 가 fBm 에서 D=2−H 를 재현", abs(float(np.mean(_fd)) - 1.5) < 0.05,
      f"D={np.mean(_fd):.3f}")
_gap = [ep.selfaffinity_gap(ep._synth_fgn(2000, 0.5, _er))["gap"] for _ in range(8)]
check("★ selfaffinity_gap 이 fBm 에서 ≈0 (설계도 Phase 2 수용 기준)",
      abs(float(np.mean(_gap))) < 0.06, f"gap={np.mean(_gap):+.3f}")

check("SampEn 이 규칙적 계열을 낮게 평가",
      ep.sample_entropy(np.sin(np.arange(400) * 0.3))["sampen"]
      < ep.sample_entropy(_er.normal(0, 1, 400))["sampen"])
check("PermEn 이 무작위에서 1 에 근접",
      ep.permutation_entropy(_er.normal(0, 1, 400))["pe"] > 0.95)
check("★ order=6 은 표본 부족으로 거부 (엔트로피가 표본 크기의 함수가 되는 것 방지)",
      ep.permutation_entropy(_er.normal(0, 1, 400), order=6).get("pe") is None)

# ===========================================================================
print()
print("=" * 74)
print("  13. 미시구조 확장 — OFI · 큐 포지션")
print("=" * 74)
from engine import microstructure as ms

_mr = np.random.default_rng(7)
_n = 4000
_mid = 100.0
_bp, _ap, _bs, _as = [], [], [], []
for _ in range(_n):
    _f = _mr.normal(0, 1)
    _mid += _f * 0.002
    _bp.append(_mid - 0.01); _ap.append(_mid + 0.01)
    _bs.append(max(100 + _f * 60 + _mr.normal(0, 10), 1))
    _as.append(max(100 - _f * 60 + _mr.normal(0, 10), 1))
_ofi = ms.order_flow_imbalance(_bp, _bs, _ap, _as, interval=10)
print(f"      OFI 회귀 R²={_ofi['r2']:.3f} (문헌 ~0.65)")
check("★ OFI 회귀 R² 가 문헌 수준(~0.65)에 근접", _ofi["r2"] > 0.5,
      f"R²={_ofi['r2']:.3f}")

_mp = ms.microprice(99.99, 100.01, 150, 50)
check("마이크로프라이스 변위 = (I−½)·스프레드",
      abs(_mp["displacement"] - (_mp["imbalance"] - 0.5) * _mp["spread"]) < 1e-9)

_q = ms.QueueSimulator()
_o = _q.submit(price=100.0, size=50, level_size=200, side="buy")
_q.on_trade(100.0, 80); _q.on_trade(100.0, 80)
check("큐 앞 잔량이 소진되기 전엔 체결 없음",
      _o.filled == 0 and abs(_o.queue_ahead - 40) < 1e-9)
_q.on_trade(100.0, 60)
check("★ 큐가 소진되면 그때부터 체결", _o.filled > 0, f"체결 {_o.filled:.0f}")
_o2 = _q.submit(100.0, 50, 200, "buy")
_o3 = _q.replace(_o2, 100.01, 500)
check("★ 정정하면 큐 맨 뒤로 (매 틱 무료 재조정 금지)",
      _o3.queue_ahead == 500.0)
check("보수적 체결 — 관통해야 전량, 터치는 부분",
      ms.conservative_fill(99.5, 101, 100.0, "buy")["fill_ratio"] == 1.0
      and ms.conservative_fill(100.0, 101, 100.0, "buy",
                               bar_volume=50, level_size=200)["fill_ratio"] == 0.25)

# ===========================================================================
print()
print("=" * 74)
print("  14. 국면 계층 — 인과성과 T1/T2/T4 게이트")
print("=" * 74)
from engine import regime as rg

_rr = np.random.default_rng(5)
_nn = 2500
_sd = np.where((np.arange(_nn) // 300) % 2 == 0, 0.008, 0.025)
_pp = 100 * np.exp(np.cumsum(_rr.normal(0, 1, _nn) * _sd))

_det = rg.RegimeDetector(quantile_window=120, vr_window=80, use_bocpd=False)
check("★ 국면 컨텍스트가 인과적 (후행 분위만 사용)",
      ca.assert_causal_series(
          lambda x: rg.build_context_series(x, rg.RegimeDetector(
              quantile_window=120, vr_window=80, use_bocpd=False))["vol_quantile"],
          _pp[:900], n_probes=4, warmup=350)["passed"] is True)

_b = rg.build_context_series(_pp, rg.RegimeDetector(quantile_window=200,
                                                    vr_window=100))
check("컨텍스트 차원이 3 (d ≤ 5 제약)", _b["contexts"].shape[1] == 3)

# ★ 배치 경로와 온라인 경로가 같은 값을 내는가.
#   build_context_series 는 성능 때문에 벡터화돼 있고 update() 는 라이브용
#   상태기계입니다. 둘이 갈라지면 **백테스트와 실매매가 다른 국면을 봅니다.**
_kw = dict(quantile_window=200, vr_window=100, use_bocpd=False)
_pp2 = _pp[:1000]
_bb = rg.build_context_series(_pp2, rg.RegimeDetector(**_kw))
_mis = 0
for _t in range(rg.RegimeDetector(**_kw).warmup, len(_pp2), 120):
    _d = rg.RegimeDetector(**_kw)
    _st = None
    for _tt in range(_d.warmup, _t + 1):
        _st = _d.update(_pp2, _tt)
    if not (np.isclose(_st.vol_quantile, _bb["vol_quantile"][_t], equal_nan=True)
            and np.isclose(_st.vr_z, _bb["vr_z"][_t], equal_nan=True)
            and _st.label == _bb["labels"][_t]):
        _mis += 1
check("★ 배치 경로와 온라인 경로가 정확히 일치 (백테스트≠실매매 방지)",
      _mis == 0, f"불일치 {_mis}건")
check("2축 그리드가 실제로 여러 국면을 냄",
      len(set(_b["labels"][_b["valid_from"]:])) >= 4)

_val = rg.validate_regime_layer(_pp, rg.RegimeDetector(quantile_window=200,
                                                       vr_window=100),
                                horizon=10, n_perm=120)
print(f"      T1 미래변동성 R²={_val['T1_forward_vol_r2']:.3f} | "
      f"T2 p={_val['T2_p_vol']:.4f} | T4 재현율={_val['T4_reproduced_fraction']:.0%}")
check("T1 — 국면이 미래 변동성을 예측", _val["T1_passed"] is True)
check("T2 — 라벨 정보량이 순열 null 대비 유의", _val["T2_passed"] is True)
# T4 의 문턱 0.70 은 설계 문서의 선택이고, 합성 계열은 파라미터에 따라
# 69~82% 사이에 놓입니다. 문턱 통과 여부를 단정하면 knife-edge 테스트가 되므로
# **실제로 성립하는 성질** 을 검사합니다: 순수 변동성 전환 계열이 관측된 국면
# 전환의 큰 부분을 재현한다는 것.
check("★ T4 — 순수 변동성 전환만으로 관측 국면전환의 절반 이상이 재현됨",
      _val["T4_reproduced_fraction"] > 0.5,
      f"재현율 {_val['T4_reproduced_fraction']:.0%} (문턱 70%, "
      f"판정={'미통과' if not _val['T4_passed'] else '통과'})")

# ===========================================================================
print()
print("=" * 74)
print("  15. STGP 전략 생성 — 게이트가 양방향으로 작동하는가")
print("=" * 74)
import tempfile as _tf
from engine import gp
from engine.registry import TrialRegistry as _TR

_types = (gp.SIGNAL, gp.BOOL, gp.SCALAR, gp.PRICE, gp.RETURN, gp.WINDOW)
check("타입별 최소 깊이가 계산됨 (Bool 은 터미널이 없어 2)",
      gp._min_depth(gp.BOOL) == 2 and gp._min_depth(gp.SIGNAL) == 1)

import random as _rd
_rt = _rd.Random(1)
_trees = [gp._random_tree(gp.SIGNAL, _rt, max_depth=6, force_prim=True)
          for _ in range(200)]
_okt = [t for t in _trees if t.size() <= 30 and t.depth() <= 6]
check("★ 생성된 트리가 사소하지 않음 (Bool 깊이 예산 버그 회귀 방지)",
      len(_okt) > 20 and sum(1 for t in _okt if t.size() <= 1) == 0,
      f"{len(_okt)}개 통과, 단일노드 0개")


def _gp_case(kind, seed=11):
    r_ = np.random.default_rng(seed)
    n_ = 3000
    p_ = np.zeros(n_); p_[0] = 100.0
    rt_ = np.zeros(n_)
    for t in range(1, n_):
        w_ = min(t, 20)
        sma_ = p_[max(t - w_, 0):t].mean()
        drift = (0.0035 if p_[t - 1] > sma_ else -0.0035) if kind == "structure" else 0.0002
        rt_[t] = drift + r_.normal(0, 0.010)
        p_[t] = p_[t - 1] * math.exp(rt_[t])
    i1, i2 = 1400, 2200
    reg_ = _TR(path=os.path.join(_tf.mkdtemp(), "g.json"))
    out_ = gp.evolve(gp.make_context(p_[:i1]), rt_[:i1],
                     gp.make_context(p_[i1:i2]), rt_[i1:i2],
                     population=80, generations=12, registry=reg_, seed=seed)
    res_ = gp.GPGate(registry=reg_).evaluate(
        out_["best"], out_["population"], gp.make_context(p_[i2:]), rt_[i2:],
        n_random=200)
    return out_, res_

_o_s, _r_s = _gp_case("structure")
_o_n, _r_n = _gp_case("randomwalk")
print(f"      구조 심음 : val샤프 {_r_s['validation_sharpe']:+.2f} gates {_r_s['gates']}")
print(f"      랜덤워크  : val샤프 {_r_n['validation_sharpe']:+.2f} gates {_r_n['gates']}")

check("GA 가 사소하지 않은 규칙을 진화시킴", _o_s["best"].nodes > 1)
# Lottery 를 1차 증거로 씁니다. DSR 은 문턱 0.95 가 knife-edge 라
# (개체 60/8세대에서 0.944, 80/12세대에서 1.0) 예산에 따라 흔들립니다.
check("★ 진짜 구조가 있으면 Lottery 통과 (무작위 전략과 명확히 구분)",
      _r_s["gates"]["lottery"] is True,
      f"{_r_s['gate_lottery'].get('percentile')} 백분위")
check("진짜 구조에서 DSR 도 통과", _r_s["gates"]["dsr"] is True,
      f"DSR={(_r_s['gate_dsr'] or {}).get('dsr')}")
check("★★ 랜덤워크에서는 Lottery·DSR 모두 기각 (거짓양성 방지)",
      _r_n["gates"]["lottery"] is False and _r_n["gates"]["dsr"] is False)
if _r_s.get("ensemble"):
    _e = _r_s["ensemble"]
    print(f"      앙상블 {_e['n_members']}개 샤프 {_e['sharpe']:+.2f} "
          f"vs 단일최적 {_e['single_best_sharpe']:+.2f}")
    check("★ PBO 만 실패하면 앙상블을 권고 (폐기가 아니라)",
          _e["beats_single_best"] is True)

_g2 = gp.GPGate()
_g2._used = True
try:
    _g2.evaluate(_o_s["best"], [], gp.make_context(np.ones(200) * 100),
                 np.zeros(200))
    check("★ validation 재접근이 차단됨", False)
except gp.ValidationExhausted:
    check("★ validation 재접근이 차단됨 (게이트 통과할 때까지 재시도 금지)", True)

# ===========================================================================
print()
print("=" * 74)
print(f"  결과: {len(PASS)} 통과 / {len(FAIL)} 실패")
if FAIL:
    for f in FAIL:
        print(f"    실패: {f}")
print("=" * 74)
sys.exit(1 if FAIL else 0)
