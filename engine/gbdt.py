"""
그래디언트 부스팅 방향 예측기 (XGBoost 이식)
--------------------------------------------
dmlc/xgboost 의 핵심 알고리즘을 numpy 로 옮긴 것입니다. 라이브러리를 통째로
들여오지 않은 이유: 이 프로젝트는 PyInstaller 단일 exe 로도 배포되는데,
xgboost 네이티브 바이너리는 그 크기를 몇 배로 키우고, 우리가 쓰는 것은
전체 기능의 아주 작은 조각(이진 분류, 작은 표본)뿐입니다.

원본에서 그대로 가져온 것 (xgboost/src/tree/param.h 의 공식)
    잎 가중치     w* = −G / (H + λ)
    잎 이득       G² / (H + λ)
    분할 이득     ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ
    2차 부스팅    로지스틱 손실의 grad = p − y, hess = p(1−p)  (Newton boosting)
    히스토그램 분할  피처를 분위수 구간으로 양자화해 구간 경계만 후보로 봄
    축소(eta) · 행/열 서브샘플링 · min_child_weight · 조기 종료

    Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting
    System." KDD '16.

무엇을 예측하는가
    "다음 h봉의 방향이 위인가" — 지표들이 각자 한 가지 관점만 보는 것과 달리,
    부스팅은 피처 조합(예: 'RSI 낮음 + 거래량 급증 + 단기 하락')의 상호작용을
    데이터에서 직접 찾습니다. 매 봉이 하나의 학습 표본이고, 학습은 항상
    **시간순 분할**입니다 — 검증 구간은 학습 구간보다 미래이므로 미래 정보가
    학습에 새어들 수 없습니다.

한계를 숨기지 않습니다
    일봉 180개면 표본이 ~120개입니다. 이 크기에서 부스팅은 쉽게 과적합하므로
    (1) 얕은 트리·강한 규제 기본값, (2) 검증 적중률이 기준 미달이면 점수를
    쓰지 않음(engine/mlsignal.py 의 품질 게이트), (3) 결과에 검증 성적을
    그대로 첨부 — 세 겹으로 방어합니다.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 소표본 재보정 기본값. xgboost 기본(eta 0.3, depth 6)은 수만 표본용이라
# 일봉 수백 개에서는 몇 라운드 만에 학습 데이터를 통째로 외웁니다.
DEFAULTS = {
    "rounds": 80,               # 최대 부스팅 라운드 (조기 종료가 먼저 끊는 게 보통)
    "eta": 0.10,                # 축소 (xgboost learning_rate)
    "max_depth": 3,
    "reg_lambda": 1.0,          # 잎 가중치 L2 (xgboost 기본과 동일)
    "gamma": 0.0,               # 분할 최소 이득
    "min_child_weight": 3.0,    # 잎의 최소 헤시안 합 — 소표본이라 기본 1보다 높임
    "subsample": 0.8,
    "colsample": 0.8,
    "bins": 32,                 # 히스토그램 구간 수
    "early_stop_rounds": 12,
    "seed": 7,                  # 결정성 — 같은 데이터는 항상 같은 모델
}

MIN_SAMPLES = 80                # 이보다 적으면 학습하지 않습니다
MIN_VAL = 15                    # 검증 표본 최소 개수
VAL_FRACTION = 0.2              # 뒤쪽 20% 를 검증으로 (시간순)


# ---------------------------------------------------------------------------
# 피처 — 전부 스케일 무관(수익률·비율·z점수)으로 만듭니다.
# 가격 수준을 그대로 넣으면 삼성전자(7만원)와 페니주(700원)가 다른 모델이 됩니다.
# ---------------------------------------------------------------------------

def feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Index] | None:
    """OHLCV → (X, 피처 이름, 유효 행 인덱스). 워밍업 구간(NaN)은 버립니다."""
    if df is None or len(df) < 90:
        return None
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else None

    log_c = np.log(close.clip(lower=1e-12))
    ret1 = log_c.diff()
    feats: dict[str, pd.Series] = {}

    # 수익률 (여러 시야) — 추세·모멘텀의 원재료
    for lag in (1, 2, 3, 5, 10, 20):
        feats[f"ret_{lag}"] = log_c.diff(lag)

    # 변동성 비율 — 최근 변동성이 평소 대비 커졌는가
    vol5 = ret1.rolling(5).std()
    vol20 = ret1.rolling(20).std()
    vol60 = ret1.rolling(60).std()
    feats["vol_5_20"] = vol5 / vol20.replace(0, np.nan)
    feats["vol_20_60"] = vol20 / vol60.replace(0, np.nan)

    # RSI(14) — Wilder 평활
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi"] = (100 - 100 / (1 + rs)) / 100.0        # 0~1 로 축소

    # MACD 히스토그램 — 변동성으로 나눠 스케일 제거
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    hist = macd_line - macd_line.ewm(span=9, adjust=False).mean()
    feats["macd_hist"] = hist / (close * vol20).replace(0, np.nan)

    # 가격 위치 — 구간 고저 대비 어디에 있는가 (0~1)
    for win in (20, 60):
        hi = high.rolling(win).max()
        lo = low.rolling(win).min()
        feats[f"pos_{win}"] = (close - lo) / (hi - lo).replace(0, np.nan)

    # 이동평균 이격 — ATR 단위
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    for win in (20, 60):
        sma = close.rolling(win).mean()
        feats[f"ma_gap_{win}"] = (close - sma) / atr.replace(0, np.nan)

    # 일중 진폭 비율
    feats["range_ratio"] = (high - low) / close.replace(0, np.nan) \
        / ((high - low) / close.replace(0, np.nan)).rolling(20).mean().replace(0, np.nan)

    # 거래량 z점수 (로그) — 거래량 없는 소스면 0 으로 둠
    if volume is not None and float(volume.fillna(0).abs().sum()) > 0:
        log_v = np.log1p(volume.clip(lower=0))
        feats["vol_z"] = ((log_v - log_v.rolling(20).mean())
                          / log_v.rolling(20).std().replace(0, np.nan))
    else:
        feats["vol_z"] = pd.Series(0.0, index=close.index)

    frame = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 30:
        return None
    return frame.to_numpy(dtype=float), list(frame.columns), frame.index


# ---------------------------------------------------------------------------
# 히스토그램 GBDT
# ---------------------------------------------------------------------------

def _quantile_bins(x_train: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """피처별 분위수 경계. **학습 구간만으로** 계산해 미래 분포를 훔쳐보지 않습니다."""
    edges = []
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    for j in range(x_train.shape[1]):
        e = np.unique(np.quantile(x_train[:, j], qs))
        edges.append(e)
    return edges


def _binned(x: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    out = np.empty(x.shape, dtype=np.int32)
    for j, e in enumerate(edges):
        out[:, j] = np.searchsorted(e, x[:, j], side="left")
    return out


@dataclass
class _Node:
    feature: int = -1           # -1 이면 잎
    threshold_bin: int = 0      # 이 구간 이하 → 왼쪽
    left: int = -1
    right: int = -1
    weight: float = 0.0         # 잎 가중치 (logit 기여)
    gain: float = 0.0


class _Tree:
    """깊이 제한 히스토그램 회귀 트리 (한 라운드의 약한 학습기)."""

    def __init__(self):
        self.nodes: list[_Node] = []

    def build(self, binned: np.ndarray, grad: np.ndarray, hess: np.ndarray,
              rows: np.ndarray, cols: np.ndarray, params: dict, n_bins: int):
        self._grow(binned, grad, hess, rows, cols, params, n_bins, depth=0)
        return self

    def _grow(self, binned, grad, hess, rows, cols, params, n_bins, depth) -> int:
        lam = float(params["reg_lambda"])
        g_sum = float(grad[rows].sum())
        h_sum = float(hess[rows].sum())
        node_id = len(self.nodes)
        node = _Node(weight=-g_sum / (h_sum + lam))
        self.nodes.append(node)

        if depth >= int(params["max_depth"]) or len(rows) < 2:
            return node_id

        parent_gain = g_sum * g_sum / (h_sum + lam)
        best = (0.0, -1, -1)          # (gain, feature, bin)
        mcw = float(params["min_child_weight"])

        for j in cols:
            b = binned[rows, j]
            hist_g = np.bincount(b, weights=grad[rows], minlength=n_bins)
            hist_h = np.bincount(b, weights=hess[rows], minlength=n_bins)
            gl = np.cumsum(hist_g)[:-1]
            hl = np.cumsum(hist_h)[:-1]
            gr = g_sum - gl
            hr = h_sum - hl
            ok = (hl >= mcw) & (hr >= mcw)
            if not ok.any():
                continue
            # xgboost param.h: ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ
            gain = 0.5 * (gl * gl / (hl + lam) + gr * gr / (hr + lam)
                          - parent_gain) - float(params["gamma"])
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), int(j), k)

        if best[1] < 0 or best[0] <= 0:
            return node_id            # 이득 없는 분할은 하지 않음 (잎으로 종료)

        node.feature, node.threshold_bin, node.gain = best[1], best[2], best[0]
        mask = binned[rows, best[1]] <= best[2]
        node.left = self._grow(binned, grad, hess, rows[mask], cols, params,
                               n_bins, depth + 1)
        node.right = self._grow(binned, grad, hess, rows[~mask], cols, params,
                                n_bins, depth + 1)
        return node_id

    def predict(self, binned: np.ndarray) -> np.ndarray:
        out = np.empty(len(binned))
        for i in range(len(binned)):
            k = 0
            while self.nodes[k].feature >= 0:
                node = self.nodes[k]
                k = node.left if binned[i, node.feature] <= node.threshold_bin \
                    else node.right
            out[i] = self.nodes[k].weight
        return out

    def gain_by_feature(self, acc: dict):
        for node in self.nodes:
            if node.feature >= 0:
                acc[node.feature] = acc.get(node.feature, 0.0) + node.gain


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


class Booster:
    """이진 분류 부스터. fit 후 predict_proba 로 확률을 냅니다."""

    def __init__(self, **overrides):
        self.params = {**DEFAULTS, **overrides}
        self.trees: list[_Tree] = []
        self.edges: list[np.ndarray] | None = None
        self.base_logit = 0.0
        self.best_iteration = 0
        self.val_logloss = None

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        p = self.params
        rng = np.random.default_rng(int(p["seed"]))
        self.edges = _quantile_bins(x_train, int(p["bins"]))
        bt = _binned(x_train, self.edges)
        bv = _binned(x_val, self.edges) if x_val is not None else None
        n_bins = int(p["bins"]) + 1

        # 초기값 = 학습 구간 사전확률의 logit (xgboost 의 base_score 학습과 같은 역할)
        rate = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
        self.base_logit = float(np.log(rate / (1 - rate)))
        logit_t = np.full(len(y_train), self.base_logit)
        logit_v = np.full(len(y_val), self.base_logit) if y_val is not None else None

        n_rows, n_feat = x_train.shape
        best_loss, best_iter, since_best = np.inf, 0, 0
        kept: list[_Tree] = []

        for _ in range(int(p["rounds"])):
            prob = _sigmoid(logit_t)
            grad = prob - y_train                 # 로지스틱 1차
            hess = prob * (1 - prob)              # 로지스틱 2차 (Newton)

            rows = np.flatnonzero(rng.random(n_rows) < float(p["subsample"]))
            if len(rows) < 10:
                rows = np.arange(n_rows)
            n_cols = max(1, int(round(n_feat * float(p["colsample"]))))
            cols = rng.choice(n_feat, size=n_cols, replace=False)

            tree = _Tree().build(bt, grad, hess, rows, cols, p, n_bins)
            kept.append(tree)
            logit_t += float(p["eta"]) * tree.predict(bt)

            if bv is not None:
                logit_v += float(p["eta"]) * tree.predict(bv)
                loss = _logloss(y_val, _sigmoid(logit_v))
                if loss < best_loss - 1e-6:
                    best_loss, best_iter, since_best = loss, len(kept), 0
                else:
                    since_best += 1
                    if since_best >= int(p["early_stop_rounds"]):
                        break

        self.trees = kept[:best_iter] if bv is not None and best_iter else kept
        self.best_iteration = len(self.trees)
        self.val_logloss = best_loss if bv is not None else None
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        b = _binned(x, self.edges)
        logit = np.full(len(x), self.base_logit)
        for tree in self.trees:
            logit += float(self.params["eta"]) * tree.predict(b)
        return _sigmoid(logit)

    def feature_importance(self) -> dict[int, float]:
        acc: dict[int, float] = {}
        for tree in self.trees:
            tree.gain_by_feature(acc)
        return acc


# ---------------------------------------------------------------------------
# 방향 점수
# ---------------------------------------------------------------------------

@dataclass
class GBDTResult:
    ok: bool = False
    score: float = 0.0            # 2p−1 ∈ [−1, +1]
    prob_up: float = 0.5
    val_accuracy: float = 0.5     # 시간순 검증 구간 적중률
    val_baseline: float = 0.5     # 다수 클래스만 찍었을 때의 적중률
    val_n: int = 0
    samples: int = 0
    rounds: int = 0
    horizon_bars: int = 0
    top_features: list = field(default_factory=list)
    error: str = ""

    @property
    def edge(self) -> float:
        """다수 클래스 대비 얼마나 더 맞혔는가 — 이게 0 이하면 모델이 무가치합니다."""
        return self.val_accuracy - max(self.val_baseline, 1 - self.val_baseline)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "score": round(self.score, 4),
                "prob_up": round(self.prob_up, 4),
                "val_accuracy": round(self.val_accuracy, 4),
                "val_baseline": round(self.val_baseline, 4),
                "edge": round(self.edge, 4), "val_n": self.val_n,
                "samples": self.samples, "rounds": self.rounds,
                "horizon_bars": self.horizon_bars,
                "top_features": self.top_features[:3], "error": self.error}


def direction_score(bars: pd.DataFrame, horizon_bars: int = 5,
                    overrides: dict = None) -> GBDTResult:
    """일봉 → 다음 horizon_bars 봉이 오를 확률.

    학습·검증·예측이 전부 시간순입니다.
        [   학습 80%   |  검증 20%  ] → 마지막 행으로 예측
    라벨은 t+1..t+h 의 수익률이라, 마지막 h개 행은 라벨이 없어 학습에서
    빠지고 예측에만 쓰입니다. 미래를 보는 경로가 구조적으로 없습니다.
    """
    result = GBDTResult(horizon_bars=int(horizon_bars))
    built = feature_matrix(bars)
    if built is None:
        result.error = "피처를 만들 봉이 부족합니다."
        return result
    x_all, names, index = built

    close = bars["close"].astype(float).reindex(index)
    log_c = np.log(close.to_numpy())
    h = max(1, int(horizon_bars))
    future = np.full(len(log_c), np.nan)
    future[:-h] = log_c[h:] - log_c[:-h]
    labeled = np.isfinite(future) & (np.abs(future) > 0)

    x = x_all[labeled]
    y = (future[labeled] > 0).astype(float)
    result.samples = len(y)
    if len(y) < MIN_SAMPLES:
        result.error = f"학습 표본 부족 ({len(y)}/{MIN_SAMPLES})"
        return result

    n_val = max(MIN_VAL, int(len(y) * VAL_FRACTION))
    if len(y) - n_val < MIN_SAMPLES // 2:
        result.error = "검증을 떼면 학습 표본이 남지 않습니다."
        return result
    x_tr, y_tr = x[:-n_val], y[:-n_val]
    x_va, y_va = x[-n_val:], y[-n_val:]
    if len(np.unique(y_tr)) < 2:
        result.error = "학습 구간 라벨이 한쪽뿐입니다 (일방향 시장)."
        return result

    booster = Booster(**(overrides or {})).fit(x_tr, y_tr, x_va, y_va)
    prob_va = booster.predict_proba(x_va)
    result.val_accuracy = float(((prob_va > 0.5) == (y_va > 0.5)).mean())
    result.val_baseline = float(y_va.mean())
    result.val_n = int(n_val)
    result.rounds = booster.best_iteration

    prob = float(booster.predict_proba(x_all[-1:])[0])
    result.prob_up = prob
    result.score = float(np.clip(2 * prob - 1, -1, 1))

    importance = booster.feature_importance()
    ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    total_gain = sum(importance.values()) or 1.0
    result.top_features = [
        {"name": names[j], "share": round(g / total_gain, 3)} for j, g in ranked[:3]]
    result.ok = True
    return result


__all__ = ["Booster", "GBDTResult", "direction_score", "feature_matrix",
           "DEFAULTS", "MIN_SAMPLES"]
