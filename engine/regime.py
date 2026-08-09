"""
국면 계층 (Regime Layer)
------------------------
"지금이 어떤 장세인가" 를 **인과적으로** 판정하고, 밴딧이 먹을 컨텍스트 벡터를 냅니다.

수록 항목
    RegimeState         이산 라벨 + 연속 컨텍스트 x_t (d ≤ 5)
    RegimeDetector      2축 그리드 + Schmitt 히스테리시스 (온라인, O(1) 갱신)
    build_context_series 백테스트용 일괄 계산 → T5 실험 입력
    BOCPD               Adams-MacKay 온라인 변화점 — **노후화 경보 전용**
    validate_regime_layer  ★ T1/T2/T4 — 국면 계층 단독 검증

축 설계 — 무엇을 주축으로 삼는가
    **[MUST] 실현변동성이 주축입니다.** 신호대잡음비가 높고 지속성이 강하며
    거의 보편적으로 복제되는 유일한 국면 차원입니다. 지루하고, 작동합니다.

    **부호축은 `VR(q) − 1`(z2) 입니다. `Ĥ − 0.5` 가 아닙니다.**
    분산비는 닫힌 형태 표준오차가 있고, 스케일 집합 임의성이 없고, log-log
    회귀가 없습니다. Hurst 보다 명확히 우월한 부호 통계량입니다.

    Hurst·프랙탈·엔트로피는 **부트스트랩 null 로 보정된 밴드를 가진 보조
    확인 피처** 로 강등했습니다. 근거는 `engine/nulls.py` 에 실측이 있습니다 —
    창 256에서 교정 밴드가 [0.442, 0.619]라 0.45/0.55 임계값이 통째로 그 안에
    들어갑니다.

```
                 저변동성            고변동성
VR > q_hi     추세·평온            추세·스트레스
q_lo≤VR≤q_hi  랜덤                 랜덤·스트레스
VR < q_lo     평균회귀·평온         평균회귀·스트레스
```

[MUST] 컨텍스트 차원 d ≤ 5 — 밴딧 리그렛 산수 때문입니다
    일봉 5년 = 결정 1,250회. LinUCB 리그렛 Õ(d√T) 에서
        d=10 → 10·√1250 ≈ 354, 라운드당 평균 리그렛이 보상 범위의 ~28%
        d=3  → ~8.5%
    경제물리 측도 10개를 그대로 먹이면 바운드가 공허해집니다. 그래서 기본
    컨텍스트는 **3차원** 입니다: [변동성 분위, VR z, 유동성/추세 보조].

[MUST] 인과성 — 이 모듈의 모든 것은 t 시점 정보만 씁니다
    · HMM 은 filtered 만. Viterbi/smoothed 는 전 표본을 씁니다 → 아예 안 넣었습니다.
    · centred DMA(θ=0.5) 금지. 중앙이동평균 금지.
    · 분위수도 **후행 확장창** 으로만 계산합니다. 전 구간 분위를 쓰면 그 자체가
      미래 참조입니다 — 흔하고 잡기 어려운 누수입니다.
    `tests/test_athena_hybrid.py` 가 `causality.assert_causal_series` 로 검사합니다.

기존 `indicators.detect_regime()` 과의 관계
    저쪽은 **화면에 근거 문장을 뿌리는 용도** 이고 `trend_score` 에 반영되지
    않습니다(AUTOTRADE.md 16장). 이 모듈은 **밴딧 컨텍스트를 만드는 용도** 입니다.
    둘을 합치지 않았습니다 — 표시용과 의사결정용이 같은 코드를 쓰면 표시를
    고치다가 매매가 바뀝니다.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from engine.quant import variance_ratio

# 국면 라벨
VOL_LOW, VOL_HIGH = "calm", "stress"
SIGN_TREND, SIGN_RANDOM, SIGN_REVERT = "trend", "random", "revert"


@dataclass(frozen=True)
class RegimeState:
    """한 시점의 국면. `context` 가 밴딧이 먹는 것입니다."""
    label: str
    context: np.ndarray                # d ≤ 5, 표준화됨
    confidence: float = 0.0
    stale: bool = False                # BOCPD 노후화 경보
    vol_quantile: float = float("nan")
    vr_z: float = float("nan")
    as_of: int = -1

    def context_vector(self) -> np.ndarray:
        return self.context

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 3),
                "stale": self.stale,
                "vol_quantile": round(self.vol_quantile, 4),
                "vr_z": round(self.vr_z, 3),
                "context": [round(float(v), 4) for v in self.context]}


# ---------------------------------------------------------------------------
# BOCPD — 노후화 경보 전용
# ---------------------------------------------------------------------------

class BOCPD:
    """Adams & MacKay (2007) 베이지안 온라인 변화점 탐지.

    런길이 사후 `P(r_t | x_{1:t})` 를 해저드 함수와 함께 재귀 갱신합니다.
    완전 인과적이고, 불확실성이 보정되며, 가지치기하면 상수 시간입니다.

    **[SHOULD] 국면 라벨로 쓰지 말고 노후화 경보로만 쓰세요.**
    변화점 확률이 높으면 → 모델 신뢰도 하향, 신규 진입 중단, 재적합 트리거.
    이건 국면 라벨이 *예측력* 을 가질 것을 요구하지 않는 견고한 사용법입니다.
    변화점을 "지금부터 추세다" 로 읽는 순간 검증되지 않은 주장이 됩니다.

    정규-감마 켤레사전 + Student-t 예측분포. scipy 없이 lgamma 로 구현합니다.
    """

    def __init__(self, hazard: float = 250.0, mu0: float = 0.0,
                 kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0,
                 max_run: int = 400):
        self.h = 1.0 / float(hazard)
        self.mu0, self.k0, self.a0, self.b0 = mu0, kappa0, alpha0, beta0
        self.max_run = int(max_run)
        self.reset()

    def reset(self) -> None:
        self.mu = np.array([self.mu0], dtype=float)
        self.kappa = np.array([self.k0], dtype=float)
        self.alpha = np.array([self.a0], dtype=float)
        self.beta = np.array([self.b0], dtype=float)
        self.rl = np.array([1.0], dtype=float)     # 런길이 사후

    def _student_t_pdf(self, x: float) -> np.ndarray:
        """예측분포 — 각 런길이 가설에 대한 Student-t 밀도."""
        df = 2.0 * self.alpha
        scale2 = self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)
        scale2 = np.maximum(scale2, 1e-300)
        z = (float(x) - self.mu) ** 2 / scale2
        log_pdf = (np.vectorize(math.lgamma)((df + 1.0) / 2.0)
                   - np.vectorize(math.lgamma)(df / 2.0)
                   - 0.5 * np.log(math.pi * df * scale2)
                   - (df + 1.0) / 2.0 * np.log1p(z / df))
        return np.exp(np.clip(log_pdf, -700, 700))

    def update(self, x: float) -> dict:
        """관측 하나를 반영하고 변화점 확률을 돌려줍니다."""
        if not math.isfinite(float(x)):
            return {"changepoint_prob": 0.0, "expected_run_length": float("nan")}

        pred = self._student_t_pdf(x)
        growth = self.rl * pred * (1.0 - self.h)      # 런이 이어질 경우
        cp = float(np.sum(self.rl * pred * self.h))   # 변화점이 일어날 경우

        new_rl = np.concatenate([[cp], growth])
        total = float(new_rl.sum())
        new_rl = new_rl / total if total > 0 else np.ones_like(new_rl) / len(new_rl)

        # 켤레사전 갱신 (런길이 0 은 사전으로 리셋)
        xx = float(x)
        new_mu = np.concatenate([[self.mu0],
                                 (self.kappa * self.mu + xx) / (self.kappa + 1.0)])
        new_kappa = np.concatenate([[self.k0], self.kappa + 1.0])
        new_alpha = np.concatenate([[self.a0], self.alpha + 0.5])
        new_beta = np.concatenate([
            [self.b0],
            self.beta + self.kappa * (xx - self.mu) ** 2 / (2.0 * (self.kappa + 1.0))])

        # 가지치기 — 꼬리 확률이 무시할 수준이면 자릅니다 (상수 시간 유지)
        if len(new_rl) > self.max_run:
            new_rl = new_rl[: self.max_run]
            new_mu, new_kappa = new_mu[: self.max_run], new_kappa[: self.max_run]
            new_alpha, new_beta = new_alpha[: self.max_run], new_beta[: self.max_run]
            s = float(new_rl.sum())
            new_rl = new_rl / s if s > 0 else new_rl

        self.rl, self.mu, self.kappa = new_rl, new_mu, new_kappa
        self.alpha, self.beta = new_alpha, new_beta

        return {
            "changepoint_prob": float(self.rl[0]),
            "expected_run_length": float(np.sum(self.rl * np.arange(len(self.rl)))),
        }


# ---------------------------------------------------------------------------
# 국면 탐지기
# ---------------------------------------------------------------------------

class RegimeDetector:
    """실현변동성 주축 × VR 부호축 2D 그리드 + Schmitt 히스테리시스.

    Parameters
    ----------
    vol_window : 실현변동성 창 (EWMA 가 아니라 단순 창 — 해석이 쉽고 인과적)
    vr_q : 분산비의 q
    vr_window : VR 을 계산할 창
    quantile_window : 변동성 **분위** 를 재는 후행 확장창의 최소 길이.
        전 구간 분위를 쓰면 미래 참조입니다. 반드시 후행으로만.
    enter_q / exit_q : Schmitt 트리거. 진입은 높게, 이탈은 낮게.
    min_dwell : 최소 체류 봉 수 — 경계에서 떠는 것을 막습니다.
    """

    def __init__(self, vol_window: int = 20, vr_q: int = 5, vr_window: int = 120,
                 quantile_window: int = 250, enter_q: float = 0.80,
                 exit_q: float = 0.60, vr_z_enter: float = 1.5,
                 vr_z_exit: float = 0.7, min_dwell: int = 5,
                 hazard: float = 250.0, use_bocpd: bool = True):
        self.vol_window = int(vol_window)
        self.vr_q, self.vr_window = int(vr_q), int(vr_window)
        self.quantile_window = int(quantile_window)
        self.enter_q, self.exit_q = float(enter_q), float(exit_q)
        self.vr_z_enter, self.vr_z_exit = float(vr_z_enter), float(vr_z_exit)
        self.min_dwell = int(min_dwell)
        self.bocpd = BOCPD(hazard=hazard) if use_bocpd else None

        self.vol_state = VOL_LOW
        self.sign_state = SIGN_RANDOM
        self.dwell_vol = self.dwell_sign = 0

    @property
    def warmup(self) -> int:
        """유효한 국면이 나오기까지 필요한 봉 수."""
        return max(self.vr_window, self.quantile_window) + self.vol_window + 2

    def _realized_vol(self, rets: np.ndarray) -> float:
        seg = rets[-self.vol_window:]
        seg = seg[np.isfinite(seg)]
        if len(seg) < max(self.vol_window // 2, 3):
            return float("nan")
        return float(seg.std(ddof=1) * math.sqrt(252))

    def update(self, prices, t: int = None) -> RegimeState:
        """`prices[:t+1]` 만 보고 t 시점 국면을 냅니다. **미래를 안 봅니다.**"""
        p = np.asarray(prices, dtype=float)
        if t is None:
            t = len(p) - 1
        p = p[: int(t) + 1]
        p = p[np.isfinite(p) & (p > 0)]
        if len(p) < self.warmup:
            return RegimeState(label="warmup", context=np.zeros(3),
                               confidence=0.0, as_of=int(t))

        rets = np.diff(np.log(p))

        # --- 주축: 실현변동성의 **후행** 분위 ---------------------------
        cur_vol = self._realized_vol(rets)
        hist = np.array([
            float(rets[max(i - self.vol_window, 0):i].std(ddof=1) * math.sqrt(252))
            for i in range(max(len(rets) - self.quantile_window, self.vol_window),
                           len(rets))
            if i - self.vol_window >= 0], dtype=float)
        hist = hist[np.isfinite(hist)]
        if len(hist) < 20 or not math.isfinite(cur_vol):
            vol_q = float("nan")
        else:
            vol_q = float(np.mean(hist <= cur_vol))

        # --- 부호축: VR(q) 의 이분산 강건 z ------------------------------
        vr = variance_ratio(p[-self.vr_window:], q=self.vr_q)
        vr_z = float(vr["z"]) if vr and vr.get("z") is not None else float("nan")

        # --- Schmitt 히스테리시스 ---------------------------------------
        self.dwell_vol += 1
        if math.isfinite(vol_q):
            if self.vol_state == VOL_LOW and vol_q >= self.enter_q:
                self.vol_state, self.dwell_vol = VOL_HIGH, 0
            elif (self.vol_state == VOL_HIGH and vol_q < self.exit_q
                  and self.dwell_vol >= self.min_dwell):
                self.vol_state, self.dwell_vol = VOL_LOW, 0

        self.dwell_sign += 1
        if math.isfinite(vr_z):
            if self.sign_state == SIGN_RANDOM:
                if vr_z >= self.vr_z_enter:
                    self.sign_state, self.dwell_sign = SIGN_TREND, 0
                elif vr_z <= -self.vr_z_enter:
                    self.sign_state, self.dwell_sign = SIGN_REVERT, 0
            elif self.dwell_sign >= self.min_dwell:
                if self.sign_state == SIGN_TREND and vr_z < self.vr_z_exit:
                    self.sign_state, self.dwell_sign = SIGN_RANDOM, 0
                elif self.sign_state == SIGN_REVERT and vr_z > -self.vr_z_exit:
                    self.sign_state, self.dwell_sign = SIGN_RANDOM, 0

        # --- 노후화 경보 -------------------------------------------------
        stale, cp_prob = False, 0.0
        if self.bocpd is not None and len(rets):
            out = self.bocpd.update(float(rets[-1]))
            cp_prob = float(out["changepoint_prob"])
            stale = cp_prob > 0.5

        # --- 컨텍스트 (d=3) ----------------------------------------------
        # 밴딧이 먹기 좋게 대략 [-1, 1] 로 맞춥니다. 표준화는 T5 쪽에서
        # 다시 하므로 여기서는 스케일만 정돈합니다.
        ctx = np.array([
            (vol_q - 0.5) * 2.0 if math.isfinite(vol_q) else 0.0,
            math.tanh(vr_z / 2.0) if math.isfinite(vr_z) else 0.0,
            cp_prob * 2.0 - 1.0,
        ], dtype=float)

        label = f"{self.sign_state}·{self.vol_state}"
        conf = 0.0
        if math.isfinite(vr_z):
            # |z| 가 클수록, 그리고 변화점 경보가 낮을수록 확신
            conf = float(min(abs(vr_z) / 3.0, 1.0) * (1.0 - cp_prob))

        return RegimeState(label=label, context=ctx, confidence=conf,
                           stale=stale, vol_quantile=vol_q, vr_z=vr_z,
                           as_of=int(t))


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """후행 창 표준편차 (ddof=1). 누적합 두 개로 O(T) 에 끝냅니다.

    `out[i]` 는 `x[i-w:i]` 의 표준편차입니다 — **i 는 포함하지 않습니다.**
    수익률 인덱스 i 가 t 시점 봉에서 나온 것이므로, 그것까지 쓰면 그 봉의
    종가를 아는 셈이 됩니다. 경계 하나 차이가 곧 누수입니다.
    """
    n = len(x)
    out = np.full(n + 1, np.nan)
    if n < w or w < 2:
        return out[:n]
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
    idx = np.arange(w, n + 1)
    s1 = c1[idx] - c1[idx - w]
    s2 = c2[idx] - c2[idx - w]
    var = (s2 - s1 ** 2 / w) / (w - 1)
    out[idx] = np.sqrt(np.maximum(var, 0.0))
    return out[:n + 1]


def build_context_series(prices, detector: RegimeDetector = None,
                         step: int = 1) -> dict:
    """백테스트·T5 실험용 일괄 계산. **벡터화 경로입니다.**

    Returns
    -------
    dict — `contexts` (T, 3), `labels`, `vol_quantile`, `vr_z`, `valid_from`.
        **T5 에 넣을 때는 `valid_from` 이후만 쓰세요** — 워밍업 구간의
        컨텍스트는 0 이라 밴딧이 그것을 "정보" 로 학습합니다.

    `RegimeDetector.update()` 와 **같은 값을 냅니다.** 다만 저쪽은 호출마다
    롤링 변동성 이력을 처음부터 다시 만들어서 O(T · quantile_window · vol_window)
    입니다 — 4,000봉이면 종목당 2천만 연산이라 20종목 패널이 30분을 넘습니다.
    여기서는 롤링 변동성을 누적합으로 한 번만 구하고 분위·히스테리시스만
    한 번 훑습니다.

    `update()` 를 지우지 않은 이유: 라이브에서는 한 시점만 계산하므로 비용이
    무의미하고, 온라인 경로와 배치 경로가 **서로를 검증** 하기 때문입니다.
    `tests/test_athena_hybrid.py` 가 두 경로의 일치를 고정합니다.
    """
    det = detector or RegimeDetector()
    p = np.asarray(prices, dtype=float)
    p = np.where(np.isfinite(p) & (p > 0), p, np.nan)
    n = len(p)
    ctxs = np.zeros((n, 3), dtype=float)
    labels = ["warmup"] * n
    vol_q = np.full(n, np.nan)
    vr_z = np.full(n, np.nan)
    valid_from = det.warmup
    if n <= valid_from:
        return {"contexts": ctxs, "labels": labels, "vol_quantile": vol_q,
                "vr_z": vr_z, "valid_from": valid_from, "n": n}

    rets = np.diff(np.log(p))
    ann = math.sqrt(252)
    # rv[i] = rets[i-vol_window:i] 의 연율 변동성 (i 미포함)
    rv = _rolling_std(rets, det.vol_window) * ann

    bocpd = BOCPD(hazard=1.0 / det.bocpd.h) if det.bocpd is not None else None
    vol_state, sign_state = VOL_LOW, SIGN_RANDOM
    dwell_v = dwell_s = 0
    cur_vr = float("nan")

    for t in range(valid_from, n):
        # --- 주축: 실현변동성의 후행 분위 ---------------------------------
        # rets 인덱스 t-1 까지가 t 시점 봉까지의 수익률입니다
        m = t                     # rv 유효 길이
        cur_vol = rv[m] if m < len(rv) else np.nan
        lo = max(m - det.quantile_window, det.vol_window)
        hist = rv[lo:m]
        hist = hist[np.isfinite(hist)]
        q = (float(np.mean(hist <= cur_vol))
             if len(hist) >= 20 and math.isfinite(cur_vol) else float("nan"))
        vol_q[t] = q

        # --- 부호축: VR(q) z ------------------------------------------------
        if (t - valid_from) % max(int(step), 1) == 0 or not math.isfinite(cur_vr):
            seg = p[max(t - det.vr_window + 1, 0): t + 1]
            vr = variance_ratio(seg, q=det.vr_q)
            cur_vr = float(vr["z"]) if vr and vr.get("z") is not None else float("nan")
        vr_z[t] = cur_vr

        # --- Schmitt 히스테리시스 -------------------------------------------
        dwell_v += 1
        if math.isfinite(q):
            if vol_state == VOL_LOW and q >= det.enter_q:
                vol_state, dwell_v = VOL_HIGH, 0
            elif (vol_state == VOL_HIGH and q < det.exit_q
                  and dwell_v >= det.min_dwell):
                vol_state, dwell_v = VOL_LOW, 0

        dwell_s += 1
        if math.isfinite(cur_vr):
            if sign_state == SIGN_RANDOM:
                if cur_vr >= det.vr_z_enter:
                    sign_state, dwell_s = SIGN_TREND, 0
                elif cur_vr <= -det.vr_z_enter:
                    sign_state, dwell_s = SIGN_REVERT, 0
            elif dwell_s >= det.min_dwell:
                if sign_state == SIGN_TREND and cur_vr < det.vr_z_exit:
                    sign_state, dwell_s = SIGN_RANDOM, 0
                elif sign_state == SIGN_REVERT and cur_vr > -det.vr_z_exit:
                    sign_state, dwell_s = SIGN_RANDOM, 0

        # --- 노후화 경보 -----------------------------------------------------
        cp = 0.0
        if bocpd is not None and t - 1 < len(rets) and math.isfinite(rets[t - 1]):
            cp = float(bocpd.update(float(rets[t - 1]))["changepoint_prob"])

        labels[t] = f"{sign_state}·{vol_state}"
        ctxs[t] = (
            (q - 0.5) * 2.0 if math.isfinite(q) else 0.0,
            math.tanh(cur_vr / 2.0) if math.isfinite(cur_vr) else 0.0,
            cp * 2.0 - 1.0,
        )

    return {"contexts": ctxs, "labels": labels, "vol_quantile": vol_q,
            "vr_z": vr_z, "valid_from": valid_from, "n": n}


# ---------------------------------------------------------------------------
# ★ 국면 계층 단독 검증 — T1 / T2 / T4
# ---------------------------------------------------------------------------

def _mutual_information(labels, values, n_bins: int = 4) -> float:
    """이산 라벨과 연속값 사이의 상호정보량 (nats). 값은 분위로 이산화합니다."""
    lab = np.asarray(labels)
    val = np.asarray(values, dtype=float)
    ok = np.isfinite(val)
    lab, val = lab[ok], val[ok]
    if len(val) < 40:
        return float("nan")
    edges = np.quantile(val, np.linspace(0, 1, n_bins + 1)[1:-1])
    vb = np.digitize(val, edges)
    uniq_l = {v: i for i, v in enumerate(sorted(set(lab.tolist())))}
    lb = np.array([uniq_l[v] for v in lab])

    n = len(lb)
    mi = 0.0
    for i in range(len(uniq_l)):
        for j in range(n_bins):
            pij = float(np.mean((lb == i) & (vb == j)))
            if pij <= 0:
                continue
            pi, pj = float(np.mean(lb == i)), float(np.mean(vb == j))
            if pi > 0 and pj > 0:
                mi += pij * math.log(pij / (pi * pj))
    return float(mi)


def validate_regime_layer(prices, detector: RegimeDetector = None, *,
                          horizon: int = 10, n_perm: int = 300,
                          seed: int = 20260808) -> dict:
    """★ 국면 계층이 값어치가 있는지 단독으로 검증합니다.

    **[MUST] 이 게이트를 통과하지 못하면 GA·RL 코드를 작성하지 않습니다.**

    T1. 예측력 — 시점 t 의 국면 라벨이 [t, t+h] 의 **사전 등록된** 통계량
        (실현변동성)을 무조건부 베이스라인보다 잘 예측하는가?
        국면이 예측해야 하는 것은 수익률이 아니라 **변동성** 입니다.
        변동성은 지속성이 강해 실제로 예측 가능하고, 수익률은 아닙니다.
        여기서 실패하면 국면 정의 자체가 잘못된 것입니다.

    T2. 상호정보량 — MI(라벨, 미래수익률) 과 MI(라벨, 미래변동성) 이
        **순열 null 대비** 유의한가? 라벨 자체가 무작위여도 MI 는 0보다 큽니다
        (유한표본 편향). 그래서 순열 분포와 비교해야 합니다.

    T4. 구조변화 반증 — i.i.d. 수익률에 **관측된 변동성 국면 전환 시점을
        그대로 심은** 합성 계열을 만들고 같은 국면 탐지기를 돌립니다.
        관측된 라벨 변동의 몇 %가 재현되는가?
        **70% 이상 재현되면 경제물리 계층은 그냥 변동성 국면 변화를 측정하고
        있는 것입니다.** 그럴 경우 실현변동성을 직접 쓰세요. 더 싸고 노이즈가 적습니다.

    (T3 전환 타이밍과 T5 밴딧 A/B 는 각각 별도입니다 —
     T5 는 `engine.bandit.run_t5_experiment` 에 있습니다.)
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    det = detector or RegimeDetector()
    if len(p) < det.warmup + horizon + 200:
        return {"passed": None,
                "reason": f"표본 부족 — 최소 {det.warmup + horizon + 200}봉 필요, "
                          f"{len(p)}봉 있음."}

    built = build_context_series(p, det)
    v0 = built["valid_from"]
    rets = np.diff(np.log(p))
    n = len(p)

    # 미래 통계량 (t 이후 h봉) — 평가용이지 피처가 아닙니다
    fwd_vol = np.full(n, np.nan)
    fwd_ret = np.full(n, np.nan)
    for t in range(v0, n - horizon - 1):
        seg = rets[t: t + horizon]
        if len(seg) == horizon:
            fwd_vol[t] = float(seg.std(ddof=1) * math.sqrt(252))
            fwd_ret[t] = float(seg.sum())

    idx = np.arange(v0, n - horizon - 1)
    labels = np.array([built["labels"][t] for t in idx])
    fv, fr = fwd_vol[idx], fwd_ret[idx]
    ok = np.isfinite(fv) & np.isfinite(fr)
    labels, fv, fr = labels[ok], fv[ok], fr[ok]
    if len(fv) < 100:
        return {"passed": None, "reason": "유효 표본이 100개 미만입니다."}

    # --- T1: 라벨별 평균이 무조건부 평균보다 잘 맞는가 (R²) --------------
    uniq = sorted(set(labels.tolist()))
    pred = np.array([fv[labels == l].mean() for l in uniq])
    pred_map = {l: v for l, v in zip(uniq, pred)}
    yhat = np.array([pred_map[l] for l in labels])
    ss_res = float(np.sum((fv - yhat) ** 2))
    ss_tot = float(np.sum((fv - fv.mean()) ** 2))
    r2_vol = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- T2: MI vs 순열 null --------------------------------------------
    rng = np.random.default_rng(seed)
    mi_vol = _mutual_information(labels, fv)
    mi_ret = _mutual_information(labels, fr)
    null_vol, null_ret = [], []
    for _ in range(int(n_perm)):
        perm = rng.permutation(labels)
        null_vol.append(_mutual_information(perm, fv))
        null_ret.append(_mutual_information(perm, fr))
    nv = np.asarray([v for v in null_vol if math.isfinite(v)])
    nr = np.asarray([v for v in null_ret if math.isfinite(v)])
    p_vol = float((np.sum(nv >= mi_vol) + 1) / (len(nv) + 1)) if len(nv) else None
    p_ret = float((np.sum(nr >= mi_ret) + 1) / (len(nr) + 1)) if len(nr) else None

    # --- T4: 구조변화 반증 ------------------------------------------------
    # 관측된 변동성 경로를 그대로 쓰되 **자기상관과 장기기억을 제거한** i.i.d.
    # 표준정규를 곱합니다. 즉 "변동성 국면 변화만 있고 다른 구조는 없는" 세계.
    obs_vol_q = built["vol_quantile"][v0:]
    obs_switches = int(np.sum(np.abs(np.diff(
        (obs_vol_q[np.isfinite(obs_vol_q)] >= det.enter_q).astype(int))) > 0))

    roll_sd = np.array([rets[max(i - det.vol_window, 0):i].std(ddof=1)
                        for i in range(det.vol_window, len(rets))])
    synth_switches = []
    for _ in range(5):
        z = rng.normal(size=len(roll_sd))
        synth_rets = roll_sd * z
        synth_p = np.exp(np.concatenate([[math.log(p[0])],
                                         math.log(p[0]) + np.cumsum(synth_rets)]))
        sb = build_context_series(synth_p, RegimeDetector(
            vol_window=det.vol_window, vr_q=det.vr_q, vr_window=det.vr_window,
            quantile_window=det.quantile_window, enter_q=det.enter_q,
            exit_q=det.exit_q, min_dwell=det.min_dwell, use_bocpd=False))
        sq = sb["vol_quantile"][sb["valid_from"]:]
        sq = sq[np.isfinite(sq)]
        if len(sq) > 10:
            synth_switches.append(int(np.sum(np.abs(np.diff(
                (sq >= det.enter_q).astype(int))) > 0)))

    reproduced = (float(np.mean(synth_switches)) / obs_switches
                  if obs_switches > 0 and synth_switches else float("nan"))

    t1 = bool(r2_vol > 0.05)
    t2 = bool(p_vol is not None and p_vol <= 0.05)
    t4 = bool(math.isfinite(reproduced) and reproduced < 0.70)

    return {
        "T1_forward_vol_r2": round(r2_vol, 4),
        "T1_passed": t1,
        "T2_mi_vol": round(mi_vol, 5), "T2_p_vol": p_vol,
        "T2_mi_ret": round(mi_ret, 5), "T2_p_ret": p_ret,
        "T2_passed": t2,
        "T4_observed_switches": obs_switches,
        "T4_synthetic_switches_mean": round(float(np.mean(synth_switches)), 1)
        if synth_switches else None,
        "T4_reproduced_fraction": round(reproduced, 3)
        if math.isfinite(reproduced) else None,
        "T4_passed": t4,
        "n_eval": int(len(fv)), "n_labels": len(uniq), "labels": uniq,
        "passed": bool(t1 and t2 and t4),
        "verdict": _verdict(t1, t2, t4, r2_vol, p_ret, reproduced),
    }


def _verdict(t1, t2, t4, r2_vol, p_ret, reproduced) -> str:
    if t1 and t2 and t4:
        msg = ("통과 — 국면 라벨이 미래 변동성을 예측하고(T1), 순열 null 대비 "
               "유의하며(T2), 단순 변동성 브레이크로 환원되지 않습니다(T4). "
               "다음은 engine.bandit.run_t5_experiment 입니다.")
    else:
        fails = []
        if not t1:
            fails.append(f"T1 실패(미래변동성 R²={r2_vol:.3f}) — 국면 정의 자체를 의심하세요")
        if not t2:
            fails.append("T2 실패 — 라벨의 정보량이 순열 null과 구분되지 않습니다")
        if not t4:
            fails.append(
                f"T4 실패(재현율 {reproduced:.0%}) — 이 국면 탐지기는 그냥 "
                f"변동성 국면 변화를 재고 있습니다. 실현변동성을 직접 쓰세요")
        msg = "미통과 — " + " / ".join(fails)
    if p_ret is not None and p_ret > 0.05:
        msg += (" · 참고: 미래 *수익률* 에 대한 정보량은 유의하지 않습니다 — "
                "정상입니다. 국면은 방향이 아니라 변동성을 예측하는 것입니다.")
    return msg
