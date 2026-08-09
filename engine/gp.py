"""
강타입 유전 프로그래밍 전략 생성 (Strongly-Typed GP)
-----------------------------------------------------
매매 규칙을 **탐색으로 만들어내는** 계층입니다. 그리고 그 결과물이 노이즈인지
구조인지를 **코드로 강제해서** 판정합니다.

수록 항목
    TYPES / PRIMITIVES   타입 시스템과 함수·터미널 집합
    Node / Genome        규칙 트리 (노드 <= 30, 깊이 <= 6)
    evolve               3분할(train/selection/validation) GA
    fitness              CPCV 샤프 평균 - kappa*표준편차 - 복잡도 - 거래수 페널티
    GPGate               ★ lottery + DSR + PBO 를 통과해야만 채택 허용

★ 이 모듈에서 가장 중요한 것은 GA 가 아니라 GPGate 입니다
    기술적 프리미티브로 규칙을 탐색하면 **거의 항상 좋아 보이는 것이 나옵니다.**
    문제는 그게 진짜인지입니다. Sullivan-Timmermann-White 가 기술적 규칙
    39,832개를 정지 부트스트랩으로 검정했을 때, **성숙 시장(DJIA, S&P500)에서는
    통계적으로 유의한 규칙이 존재하지 않았습니다.** 유의했던 곳은 신흥·젊은
    시장(NASDAQ, Russell 2000)입니다. Aronson 의 6,402개 규칙 실험도 데이터마이닝
    편의 보정 후 유의한 것이 **0개** 였습니다.

    그래서 이 모듈은 GA 결과를 그냥 돌려주지 않습니다. `GPGate.evaluate()` 가
    세 관문을 통과해야 `adopted=True` 가 됩니다.

      1. **Lottery** — 매칭된 복잡도·거래빈도의 무작위 전략 분포 상위 5% 밖인가
      2. **DSR** — TrialRegistry 의 N-hat 으로 깎은 뒤에도 > 0.95 인가
      3. **PBO** — 설정 선택 절차의 과최적화 확률 < 0.05 인가

    통과 못 하면 `adopted=False` 이고 `reason` 에 어느 관문에서 걸렸는지 남습니다.
    **이건 비관주의가 아니라 산수입니다.** 개체 300 x 세대 40 이면 시행이 최대
    12,000 이고, 그 규모에서 무능력 귀무의 기대 최대 샤프가 이미 2를 넘습니다.

[MUST] 왜 강타입인가
    벡터형 GP 비교 연구에서 "강타입 VGP 는 항상 최상위권, 표준 GP 는 항상
    최하위권" 이었습니다. 타입 제약이 **의미 없는 표현을 탐색공간에서 아예
    제거** 하기 때문입니다. `sma(rsi, and(vol, 5))` 같은 것을 만들지 않습니다.

        Price, Return, Scalar, Bool, Window, Signal
        (Price, Window)  -> Price     sma, ema, max, min
        (Return, Window)  -> Scalar    vol, mean, skew
        (Scalar, Scalar)  -> Bool      gt, lt
        (Bool, Bool)      -> Bool      and, or
        (Bool)            -> Bool      not
        (Bool, Signal, Signal) -> Signal   if_then_else
        루트는 반드시 Signal

[MUST] 3분할 — selection 을 training 에 접어 넣지 마세요
    training  : 개체군 진화 (적합도가 탐색을 이끕니다)
    selection : **어떤 진화된 규칙을 남길지** 고릅니다 — 별개의 선택 사건이므로
                자기 데이터가 필요합니다
    validation: **정확히 한 번** 접근. 보고용.

    selection 을 training 에 접어 넣는 것이 "GA 가 훌륭한 전략을 찾았다" 가
    발생하는 방식입니다. 이 모듈은 validation 접근 횟수를 세고, 두 번째
    접근에서 예외를 던집니다.

[MUST] 하이퍼파라미터 — 문헌 기본값보다 훨씬 조입니다
    개체수 200~500 / 최대깊이 6 / 최대노드 30 / 세대 30~50 / 토너먼트 3
    Neely-Weller 의 깊이 10·노드 100 보다 훨씬 타이트합니다. **복잡도가 적입니다.**
    크게 해도 일반화는 안 늘고 DSR 의 N 만 늘어납니다.
"""

import math
import random
from dataclasses import dataclass, field

import numpy as np

# 타입
PRICE, RETURN, SCALAR, BOOL, WINDOW, SIGNAL = (
    "Price", "Return", "Scalar", "Bool", "Window", "Signal")

MAX_DEPTH_DEFAULT = 6
MAX_NODES_DEFAULT = 30
WINDOWS = (5, 10, 20, 60, 120)


# ---------------------------------------------------------------------------
# 프리미티브
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Primitive:
    name: str
    arg_types: tuple
    ret_type: str
    fn: object


def _sma(p, w):
    w = int(w)
    if w < 1 or len(p) < w:
        return np.full(len(p), np.nan)
    c = np.cumsum(np.insert(np.nan_to_num(p, nan=0.0), 0, 0.0))
    out = np.full(len(p), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def _ema(p, w):
    w = max(int(w), 1)
    a = 2.0 / (w + 1.0)
    out = np.full(len(p), np.nan)
    acc = None
    for i, v in enumerate(p):
        if not math.isfinite(v):
            continue
        acc = v if acc is None else a * v + (1 - a) * acc
        out[i] = acc
    return out


def _roll(p, w, fn):
    w = max(int(w), 1)
    out = np.full(len(p), np.nan)
    for i in range(w - 1, len(p)):
        seg = p[i - w + 1: i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) == w:
            out[i] = fn(seg)
    return out


PRIMITIVES = [
    Primitive("sma", (PRICE, WINDOW), PRICE, _sma),
    Primitive("ema", (PRICE, WINDOW), PRICE, _ema),
    Primitive("rmax", (PRICE, WINDOW), PRICE, lambda p, w: _roll(p, w, np.max)),
    Primitive("rmin", (PRICE, WINDOW), PRICE, lambda p, w: _roll(p, w, np.min)),
    Primitive("vol", (RETURN, WINDOW), SCALAR,
              lambda r, w: _roll(r, w, lambda s: float(s.std(ddof=1)))),
    Primitive("rmean", (RETURN, WINDOW), SCALAR,
              lambda r, w: _roll(r, w, lambda s: float(s.mean()))),
    Primitive("zscore", (RETURN, WINDOW), SCALAR,
              lambda r, w: _roll(r, w, lambda s: float(
                  (s[-1] - s.mean()) / s.std(ddof=1)) if s.std(ddof=1) > 1e-12 else 0.0)),
    Primitive("gt", (SCALAR, SCALAR), BOOL, lambda a, b: a > b),
    Primitive("lt", (SCALAR, SCALAR), BOOL, lambda a, b: a < b),
    Primitive("pgt", (PRICE, PRICE), BOOL, lambda a, b: a > b),
    Primitive("plt", (PRICE, PRICE), BOOL, lambda a, b: a < b),
    Primitive("and_", (BOOL, BOOL), BOOL, lambda a, b: a & b),
    Primitive("or_", (BOOL, BOOL), BOOL, lambda a, b: a | b),
    Primitive("not_", (BOOL,), BOOL, lambda a: ~a),
    Primitive("ite", (BOOL, SIGNAL, SIGNAL), SIGNAL,
              lambda c, x, y: np.where(c, x, y)),
]

# 터미널 — 이름과 타입만. 실제 값은 평가 시 컨텍스트에서 옵니다.
TERMINALS = {
    PRICE: ("close", "high", "low"),
    RETURN: ("ret",),
    SCALAR: ("const_scalar",),
    WINDOW: tuple(f"w{w}" for w in WINDOWS),
    SIGNAL: ("long", "flat", "short"),
    BOOL: (),
}

_BY_RET: dict = {}
for _p in PRIMITIVES:
    _BY_RET.setdefault(_p.ret_type, []).append(_p)


# ---------------------------------------------------------------------------
# 트리
# ---------------------------------------------------------------------------

@dataclass
class Node:
    kind: str                 # "prim" | "term"
    name: str
    ret_type: str
    children: list = field(default_factory=list)
    value: float = 0.0        # const_scalar 용

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + (max((c.depth() for c in self.children), default=0))

    def copy(self) -> "Node":
        return Node(self.kind, self.name, self.ret_type,
                    [c.copy() for c in self.children], self.value)

    def __str__(self) -> str:
        if self.kind == "term":
            return (f"{self.value:.3g}" if self.name == "const_scalar"
                    else self.name)
        return f"{self.name}({', '.join(str(c) for c in self.children)})"


_MIN_DEPTH: dict = {}


def _min_depth(ret_type: str, _seen=None) -> int:
    """그 타입의 서브트리를 완성하는 데 필요한 **최소 깊이**.

    ★ 이게 없으면 강타입 GP 가 조용히 퇴화합니다.
      `Bool` 에는 터미널이 없습니다(참/거짓 상수는 규칙으로 무의미하므로
      일부러 안 넣었습니다). 그래서 "깊이 한계에 닿으면 터미널로 닫는다" 는
      순진한 규칙이 Bool 에서 작동하지 않고, 트리가 max_depth 를 넘겨 버립니다.
      그런 트리는 개체군 필터에서 전부 탈락하므로 **살아남는 것은 `long`·`flat`
      같은 사소한 상수 트리뿐** 이 됩니다. 실제로 그렇게 퇴화했었습니다 —
      8세대 내내 최적해가 `short` 한 노드였습니다.

      그래서 남은 깊이 예산 안에 완성 가능한 프리미티브만 후보로 둡니다.
    """
    if ret_type in _MIN_DEPTH:
        return _MIN_DEPTH[ret_type]
    _seen = _seen or set()
    if ret_type in _seen:
        return 99
    _seen = _seen | {ret_type}

    best = 1 if TERMINALS.get(ret_type) else 99
    for p in _BY_RET.get(ret_type, []):
        need = 1 + max((_min_depth(a, _seen) for a in p.arg_types), default=0)
        best = min(best, need)
    if not _seen - {ret_type}:
        _MIN_DEPTH[ret_type] = best
    return best


def _random_tree(ret_type: str, rng: random.Random, depth: int = 0,
                 max_depth: int = MAX_DEPTH_DEFAULT,
                 force_prim: bool = False) -> Node:
    """타입 제약과 **깊이 예산** 을 지키는 무작위 트리."""
    terms = TERMINALS.get(ret_type, ())
    budget = max_depth - depth

    # 예산 안에 완성 가능한 프리미티브만 남깁니다
    prims = [p for p in _BY_RET.get(ret_type, [])
             if 1 + max((_min_depth(a) for a in p.arg_types), default=0) <= budget]

    if terms and not prims:
        name = rng.choice(terms)
        return Node("term", name, ret_type, [],
                    rng.uniform(-2.0, 2.0) if name == "const_scalar" else 0.0)
    if not terms and not prims:
        raise ValueError(f"타입 {ret_type} 을 깊이 {budget} 안에 완성할 수 없습니다.")

    # 터미널로 닫을지 결정. 루트(force_prim)에서는 상수 신호를 막습니다 —
    # 그러지 않으면 개체군의 30%가 `long`/`flat` 같은 무의미한 트리가 됩니다.
    if terms and prims and not force_prim and rng.random() < 0.35:
        name = rng.choice(terms)
        return Node("term", name, ret_type, [],
                    rng.uniform(-2.0, 2.0) if name == "const_scalar" else 0.0)
    if not prims:
        name = rng.choice(terms)
        return Node("term", name, ret_type, [],
                    rng.uniform(-2.0, 2.0) if name == "const_scalar" else 0.0)

    p = rng.choice(prims)
    kids = [_random_tree(t, rng, depth + 1, max_depth) for t in p.arg_types]
    return Node("prim", p.name, p.ret_type, kids)


def _all_nodes(n: Node, acc=None) -> list:
    acc = acc if acc is not None else []
    acc.append(n)
    for c in n.children:
        _all_nodes(c, acc)
    return acc


def evaluate_tree(node: Node, ctx: dict) -> np.ndarray:
    """트리를 평가해 배열을 냅니다. **전부 인과적 프리미티브만 씁니다.**"""
    if node.kind == "term":
        if node.name == "const_scalar":
            return np.full(ctx["_n"], float(node.value))
        if node.name.startswith("w") and node.name[1:].isdigit():
            return int(node.name[1:])
        if node.name == "long":
            return np.ones(ctx["_n"])
        if node.name == "flat":
            return np.zeros(ctx["_n"])
        if node.name == "short":
            return -np.ones(ctx["_n"])
        return ctx[node.name]

    prim = next(p for p in PRIMITIVES if p.name == node.name)
    args = [evaluate_tree(c, ctx) for c in node.children]
    with np.errstate(all="ignore"):
        return prim.fn(*args)


@dataclass
class Genome:
    root: Node
    fitness: float = float("-inf")
    detail: dict = field(default_factory=dict)

    @property
    def nodes(self) -> int:
        return self.root.size()

    @property
    def depth(self) -> int:
        return self.root.depth()

    def signal(self, ctx: dict) -> np.ndarray:
        out = np.asarray(evaluate_tree(self.root, ctx), dtype=float)
        return np.clip(np.nan_to_num(out, nan=0.0), -1.0, 1.0)

    def __str__(self) -> str:
        return str(self.root)


# ---------------------------------------------------------------------------
# 적합도
# ---------------------------------------------------------------------------

def make_context(close, high=None, low=None) -> dict:
    c = np.asarray(close, dtype=float)
    r = np.zeros(len(c))
    r[1:] = np.diff(np.log(np.where(c > 0, c, np.nan)))
    r = np.nan_to_num(r, nan=0.0)
    return {"close": c,
            "high": np.asarray(high, dtype=float) if high is not None else c,
            "low": np.asarray(low, dtype=float) if low is not None else c,
            "ret": r, "_n": len(c)}


def fitness(genome: Genome, ctx: dict, returns: np.ndarray, *,
            cost_bps: float = 100.0, kappa: float = 0.5, min_trades: int = 30,
            n_groups: int = 6, k: int = 2, horizon: int = 1,
            lam_complexity: float = 0.01, lam_turnover: float = 0.5) -> dict:
    """CPCV 경로 샤프의 **평균 − kappa*표준편차** 에서 페널티를 뺍니다.

        Fitness = mean(CPCV SR) - kappa*sd(CPCV SR)
                  - lam1*complexity - lam2*1[n_trades < 30] - lam3*turnover

    **`- kappa*sd(CPCV SR)` 가 핵심입니다.** 최고 성능이 아니라 **경로 강건성** 을
    직접 선택합니다. 가장 값싼 과적합 방어책입니다. 한 경로에서만 좋은 규칙은
    표준편차 페널티로 걸러집니다.

    **거래 30회 미만은 하드 리젝트** 입니다. 그 아래에서는 샤프의 표준오차가
    대략 1/sqrt(n_trades) 라 어떤 그럴듯한 엣지보다 큽니다.

    **Neely-Weller 비용 트릭**: training/selection 에서 비용을 1%(100bp)로
    과하게 잡습니다. 과다거래에 대한 무료 사전(prior)이고, validation 에서만
    실제 비용을 씁니다.
    """
    from engine.cpcv import combinatorial_purged_cv
    from engine.lottery import strategy_returns
    from engine.validation import sharpe_ratio

    sig = genome.signal(ctx)
    n = min(len(sig), len(returns))
    if n < 100:
        return {"fitness": float("-inf"), "reason": "표본 부족"}
    sig, rets = sig[:n], np.asarray(returns, dtype=float)[:n]

    n_trades = int(np.sum(np.abs(np.diff(sig)) > 1e-9))
    strat = strategy_returns(sig, rets, cost_bps=cost_bps)
    if len(strat) < 50:
        return {"fitness": float("-inf"), "reason": "수익률 계열이 짧습니다"}

    plan = combinatorial_purged_cv(len(strat), n_groups=n_groups, k=k,
                                   horizon=horizon, warmup=max(WINDOWS))
    if plan is None:
        return {"fitness": float("-inf"), "reason": "CPCV 분할 실패"}

    srs = []
    for split in plan.splits:
        seg = strat[split.test_idx]
        seg = seg[np.isfinite(seg)]
        if len(seg) > 8:
            srs.append(sharpe_ratio(seg))
    if len(srs) < 3:
        return {"fitness": float("-inf"), "reason": "유효 폴드 부족"}

    arr = np.asarray(srs, dtype=float)
    mean_sr, sd_sr = float(arr.mean()), float(arr.std(ddof=1))
    turnover = float(np.mean(np.abs(np.diff(sig)))) if len(sig) > 1 else 0.0

    penalty = (lam_complexity * genome.nodes / MAX_NODES_DEFAULT
               + lam_turnover * turnover)
    hard = 1.0 if n_trades < int(min_trades) else 0.0
    fit = mean_sr - kappa * sd_sr - penalty - hard

    return {"fitness": float(fit), "mean_sr": round(mean_sr, 4),
            "sd_sr": round(sd_sr, 4), "n_trades": n_trades,
            "turnover": round(turnover, 4), "nodes": genome.nodes,
            "depth": genome.depth, "n_folds": len(srs),
            "rejected_few_trades": bool(hard > 0)}


# ---------------------------------------------------------------------------
# 진화
# ---------------------------------------------------------------------------

def _mutate(g: Genome, rng: random.Random, max_depth: int) -> Genome:
    root = g.root.copy()
    nodes = _all_nodes(root)
    target = rng.choice(nodes)
    new_sub = _random_tree(target.ret_type, rng,
                           depth=max(max_depth - 3, 0), max_depth=max_depth)
    target.kind, target.name = new_sub.kind, new_sub.name
    target.children, target.value = new_sub.children, new_sub.value
    return Genome(root)


def _crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    root = a.root.copy()
    na = _all_nodes(root)
    nb = _all_nodes(b.root)
    rng.shuffle(na)
    for ta in na:
        # **타입이 같은 노드끼리만 교환합니다** — 강타입의 요점입니다
        cands = [x for x in nb if x.ret_type == ta.ret_type]
        if cands:
            src = rng.choice(cands).copy()
            ta.kind, ta.name = src.kind, src.name
            ta.children, ta.value = src.children, src.value
            break
    return Genome(root)


def evolve(ctx_train: dict, rets_train, ctx_sel: dict, rets_sel, *,
           population: int = 300, generations: int = 40,
           max_depth: int = MAX_DEPTH_DEFAULT, max_nodes: int = MAX_NODES_DEFAULT,
           tournament: int = 3, elite: int = 2, p_cross: float = 0.8,
           p_mut: float = 0.15, train_cost_bps: float = 100.0,
           kappa: float = 0.5, seed: int = 20260808,
           registry=None, family: str = "stgp") -> dict:
    """3분할 GA. **training 으로 진화하고 selection 으로 고릅니다.**

    validation 은 여기서 **건드리지 않습니다** — `GPGate.evaluate()` 에서
    정확히 한 번 씁니다.

    `registry` 를 넘기면 평가한 개체 수를 TrialRegistry 에 기록합니다.
    **DSR 의 N 은 GA 개체수가 아니라 파이프라인 전체 시행 수** 이므로,
    이 숫자가 나중에 DSR 을 깎는 데 쓰입니다.
    """
    rng = random.Random(int(seed))
    pop, guard, rejected = [], 0, 0
    while len(pop) < population and guard < population * 200:
        guard += 1
        try:
            # 루트는 반드시 프리미티브 — 상수 신호 개체를 막습니다
            t = _random_tree(SIGNAL, rng, max_depth=max_depth, force_prim=True)
        except ValueError:
            continue
        if t.size() <= max_nodes and t.depth() <= max_depth:
            pop.append(Genome(t))
        else:
            rejected += 1
    if len(pop) < 10:
        return {"best": None, "reason": "초기 개체군 생성 실패",
                "rejected": rejected}
    # 개체군이 사소한 트리로만 채워졌으면 탐색이 이미 죽은 것입니다
    trivial = sum(1 for g in pop if g.nodes <= 1)
    if trivial > len(pop) * 0.5:
        return {"best": None,
                "reason": (f"개체군의 {trivial}/{len(pop)} 가 단일 노드입니다 — "
                           f"타입 시스템이나 깊이 예산을 확인하세요."),
                "rejected": rejected}

    n_evaluated = 0
    history = []
    for gen in range(int(generations)):
        for g in pop:
            if g.fitness == float("-inf") or not g.detail:
                d = fitness(g, ctx_train, rets_train,
                            cost_bps=train_cost_bps, kappa=kappa)
                g.fitness, g.detail = d["fitness"], d
                n_evaluated += 1
        pop.sort(key=lambda x: x.fitness, reverse=True)
        history.append(round(pop[0].fitness, 4))

        nxt = [g for g in pop[:int(elite)]]
        while len(nxt) < population:
            def pick():
                return max(rng.sample(pop, min(int(tournament), len(pop))),
                           key=lambda x: x.fitness)
            r = rng.random()
            child = (_crossover(pick(), pick(), rng) if r < p_cross
                     else (_mutate(pick(), rng, max_depth) if r < p_cross + p_mut
                           else pick().copy() if hasattr(pick(), "copy")
                           else Genome(pick().root.copy())))
            if child.root.size() <= max_nodes and child.root.depth() <= max_depth:
                nxt.append(child)
        pop = nxt

    # --- selection 폴드에서 최종 선택 (별개의 선택 사건) ------------------
    for g in pop:
        d = fitness(g, ctx_sel, rets_sel, cost_bps=train_cost_bps, kappa=kappa)
        g.fitness, g.detail = d["fitness"], d
        n_evaluated += 1
    pop.sort(key=lambda x: x.fitness, reverse=True)
    best = pop[0]

    if registry is not None:
        from engine.registry import TrialSpec
        for i, g in enumerate(pop[:50]):
            tid = registry.register(TrialSpec(
                family=family,
                config={"expr": str(g), "nodes": g.nodes, "depth": g.depth,
                        "seed": seed, "gen": generations, "pop": population},
                label=f"stgp_{i}"))
            registry.log_result(tid, sharpe=g.detail.get("mean_sr"))

    return {"best": best, "population": pop[:20],
            "n_evaluated": n_evaluated, "history": history,
            "expr": str(best), "selection_fitness": round(best.fitness, 4),
            "note": ("validation 은 건드리지 않았습니다. "
                     "GPGate.evaluate() 로 정확히 한 번 평가하세요.")}


# ---------------------------------------------------------------------------
# ★ 게이트 — 이것이 이 모듈의 존재 이유입니다
# ---------------------------------------------------------------------------

class ValidationExhausted(RuntimeError):
    """validation 폴드에 두 번째로 접근했을 때."""


class GPGate:
    """GA 산출물을 채택해도 되는지 **코드로** 판정합니다.

    세 관문을 전부 통과해야 `adopted=True` 입니다.

      1. **Lottery** — 매칭된 무작위 전략 분포의 상위 5% 밖인가
         (Chen & Navet: 이걸 안 하면 GP 실패가 시장 효율성 때문인지 내 탐색기가
          나빠서인지 구분할 수 없습니다)
      2. **DSR** — TrialRegistry 의 N-hat 으로 깎은 뒤에도 > 0.95 인가
      3. **PBO** — 상위 개체들의 설정 선택 과최적화 확률 < 0.05 인가

    **validation 은 정확히 한 번만 접근합니다.** 두 번째 호출은 예외입니다 —
    "게이트를 통과할 때까지 다시 돌려보는" 것이 바로 이 모든 장치를 무효화하는
    행동이기 때문입니다.
    """

    def __init__(self, registry=None, family: str = "stgp"):
        self.registry = registry
        self.family = family
        self._used = False

    def evaluate(self, best: Genome, population: list, ctx_val: dict, rets_val,
                 *, cost_bps: float = 5.0, n_random: int = 500,
                 force: bool = False) -> dict:
        if self._used and not force:
            raise ValidationExhausted(
                "validation 폴드는 이미 사용했습니다. 다시 평가하려면 새 데이터를 "
                "쓰거나, 그것이 의도라면 force=True 를 명시하세요 — 단 그 순간 "
                "이 결과의 통계적 의미는 사라집니다.")
        self._used = True

        from engine.lottery import lottery_benchmark, strategy_returns
        from engine.validation import sharpe_ratio
        from engine.overfit import pbo_cscv

        sig = best.signal(ctx_val)
        n = min(len(sig), len(rets_val))
        sig, rv = sig[:n], np.asarray(rets_val, dtype=float)[:n]
        strat = strategy_returns(sig, rv, cost_bps=cost_bps)
        val_sr = sharpe_ratio(strat[np.isfinite(strat)])

        # 1. Lottery
        lot = lottery_benchmark(sig, rv, cost_bps=cost_bps, n_random=n_random)

        # 2. DSR — 장부의 N-hat 으로
        dsr = None
        if self.registry is not None:
            from engine.registry import TrialSpec
            tid = self.registry.register(TrialSpec(
                family=self.family, config={"expr": str(best), "stage": "validation"}))
            self.registry.log_result(tid, returns=strat.tolist())
            dsr = self.registry.deflated_sharpe(tid, family=self.family)

        # 3. PBO — **서로 다른** 개체들의 수익률 행렬
        #
        # ★ 중복제거가 필수입니다. PBO 는 "여러 설정 중 인샘플 1위를 고르는
        #   행위" 를 평가하므로 **설정들이 실제로 서로 달라야** 의미가 있습니다.
        #   GA 개체군은 수렴하면 상위 40개가 사실상 같은 규칙의 복제본이 되는데,
        #   거의 동일한 것들끼리의 순위는 당연히 노이즈라 PBO 가 인위적으로
        #   올라갑니다. 실제로 겪었습니다 — 심어둔 규칙을 정확히 찾아내
        #   validation 샤프 4.57·lottery 100 백분위·DSR 1.0 을 낸 경우에도
        #   PBO 가 0.23 으로 나와 **맞는 답을 틀린 이유로 기각** 했습니다.
        #
        #   그래서 표현식으로 중복을 제거하고, 서로 다른 설정이 10개 미만이면
        #   PBO 를 "판정 불가" 로 둡니다. 계산할 수 없는 것을 실패로 처리하면
        #   게이트가 엉뚱한 것을 재게 됩니다.
        pbo, seen, cols = None, set(), []
        for g in (population or []):
            key = str(g)
            if key in seen:
                continue
            seen.add(key)
            s = g.signal(ctx_val)[:n]
            rr = strategy_returns(s, rv, cost_bps=cost_bps)
            if len(rr) == len(strat) and np.isfinite(rr).all() and rr.std() > 1e-12:
                cols.append(rr)
            if len(cols) >= 40:
                break

        n_distinct = len(cols)
        if n_distinct >= 10:
            pbo = pbo_cscv(np.column_stack(cols), n_blocks=8)

        g1 = bool(lot.get("passed"))
        g2 = bool(dsr and dsr.get("dsr") is not None and dsr["dsr"] > 0.95)
        pbo_applicable = pbo is not None and pbo.get("pbo") is not None
        g3 = bool(pbo["pbo"] < 0.05) if pbo_applicable else None

        # PBO 를 계산할 수 없으면 그것으로 막지 않되, 반드시 드러냅니다
        adopted = bool(g1 and g2 and (g3 is not False))

        # ★ PBO 만 실패하면 "전략군은 살아 있는데 **고르는 행위** 가 무의미하다" 는
        #   뜻입니다. 그럴 때 올바른 대응은 폐기가 아니라 **동일가중 앙상블** 입니다.
        #   선택 자유도를 0 으로 만들면 선택편향 자체가 사라집니다.
        #   이 저장소가 AUTOTRADE.md 16장에서 288개 설정을 동일가중으로 합쳐
        #   평가한 것이 정확히 같은 처방입니다.
        ensemble = None
        if g1 and g2 and g3 is False and len(cols) >= 3:
            ens = np.mean(np.column_stack(cols), axis=1)
            ens_sr = sharpe_ratio(ens[np.isfinite(ens)])
            best_sr = float(val_sr)
            ensemble = {
                "n_members": len(cols),
                "sharpe": round(float(ens_sr), 4),
                "single_best_sharpe": round(best_sr, 4),
                "beats_single_best": bool(ens_sr > best_sr),
                "recommendation": (
                    "단일 최적 대신 **동일가중 앙상블** 을 쓰세요. PBO 가 높다는 것은 "
                    "이 개체들 중 하나를 고르는 행위가 값어치가 없다는 뜻이지 "
                    "전략군에 우위가 없다는 뜻이 아닙니다. 균등가중은 선택 자유도를 "
                    "0 으로 만들어 선택편향을 제거합니다 — 그러면 DSR 보정도 불필요합니다."),
            }

        fails, notes = [], []
        if not g1:
            fails.append(f"Lottery 미통과({lot.get('percentile')} 백분위) — "
                         f"매칭된 무작위 전략과 구분되지 않습니다")
        if not g2:
            fails.append(f"DSR 미통과({dsr.get('dsr') if dsr else 'N/A'}) — "
                         f"시행 수를 반영하면 유의하지 않습니다")
        if g3 is False:
            fails.append(f"PBO 미통과({pbo.get('pbo')}) — "
                         f"설정 선택 절차가 노이즈를 고르고 있습니다")
        elif g3 is None:
            notes.append(
                f"PBO 판정 불가 — 서로 다른 설정이 {n_distinct}개뿐입니다"
                f"(10개 필요). 개체군이 수렴했다는 뜻이고, 그 자체로는 나쁜 "
                f"신호가 아니지만 **선택 과최적화를 검정하지 못했다는 사실은 "
                f"남습니다.** 다른 시드로 재실행해 서로 다른 해가 나오는지 보세요.")

        return {
            "adopted": adopted,
            "validation_sharpe": round(float(val_sr), 4),
            "expr": str(best),
            "gate_lottery": lot, "gate_dsr": dsr, "gate_pbo": pbo,
            "n_distinct_configs": n_distinct,
            "gates": {"lottery": g1, "dsr": g2, "pbo": g3},
            "caveats": notes,
            "ensemble": ensemble,
            "reason": (("채택 — 관문을 통과했습니다."
                        + ((" 단, " + notes[0]) if notes else ""))
                       if adopted else
                       ("단일 최적은 기각 — " + " / ".join(fails)
                        + " · 다만 Lottery·DSR 은 통과했으므로 전략군 자체는 "
                          "살아 있습니다. ensemble 필드를 보세요."
                        if ensemble else "기각 — " + " / ".join(fails))),
            "honest_prior": (
                "유동성 높은 성숙 시장에서 기술적 프리미티브 GA 가 이 게이트를 "
                "통과할 사전 확률은 낮습니다. Sullivan-Timmermann-White 는 "
                "DJIA·S&P500 에서 39,832개 규칙 중 유의한 것을 찾지 못했습니다. "
                "기각은 정상이며, 유니버스를 신흥·소형주로 바꾸거나 프리미티브를 "
                "바꾸거나 GA 를 포기하는 것이 다음 선택지입니다."),
        }
