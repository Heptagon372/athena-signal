"""
게이트 실행기 — 실제 시장 데이터로 T1/T2/T4 와 T5 를 돌립니다.

    python run_gates.py                 # KR + US, 기본 유니버스
    python run_gates.py --market kr
    python run_gates.py --no-cache      # 캐시 무시하고 다시 받기

무엇을 하는가
    1) 실제 일봉을 받아 캐시합니다 (.cache/gates_*.pkl)
    2) 종목별로 국면 컨텍스트를 만듭니다 (engine/regime.py)
    3) 국면 계층 단독 검증 T1/T2/T4 를 돌립니다
    4) 전략 팔들의 보상을 만들고 **종목 풀링** 으로 T5 를 돌립니다
       (engine/bandit.py — 컨텍스추얼 밴딧 vs 비컨텍스추얼 UCB)

★ 이 스크립트가 답하는 질문
    "국면 컨텍스트에 정보가 있는가?"
    없으면 경제물리 계층을 버리고 실현변동성 단일 축으로 갑니다. GA·RL 은
    쓰지 않습니다. 두 설계 문서가 공통으로 "이걸 먼저 돌리라" 고 지목한
    실험이고, 비용은 하루이고 정보 가치는 프로젝트 전체입니다.

⚠ 이 실험의 알려진 한계 — 결과를 읽기 전에
    **생존편향이 있습니다.** 유니버스를 *오늘 기준으로* 유동성 있는 종목에서
    골랐습니다. 상장폐지된 종목이 빠져 있어 모든 전략의 수익이 위로 편향됩니다.
    T5 는 팔들 사이의 *상대* 비교라 이 편향이 상당 부분 상쇄되지만, 절대
    성과 숫자는 믿지 마세요. 제대로 하려면 PIT 유니버스가 필요합니다
    (AUTOTRADE.md 17.6 의 미구현 항목).
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

CACHE_DIR = ".cache"

# 오늘 기준 유동성으로 고른 유니버스 — 생존편향이 있습니다(모듈 docstring 참조)
KR_UNIVERSE = [
    "005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS",
    "000270.KS", "005490.KS", "051910.KS", "035420.KS", "035720.KS",
    "012330.KS", "068270.KS", "028260.KS", "105560.KS", "055550.KS",
    "096770.KS", "017670.KS", "015760.KS", "034730.KS", "003550.KS",
]
US_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "JNJ",
    "V", "PG", "XOM", "HD", "KO", "PEP", "MRK", "ABBV",
    "CVX", "WMT", "MCD", "CSCO",
]

ARM_NAMES = ("tsmom_60", "reversal_5", "breakout_20", "flat")


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def fetch(symbols, start="2010-01-01", end="2026-08-01", use_cache=True):
    """일봉 종가를 받아 캐시합니다. **시장을 섞어 받지 않습니다** —
    휴장일이 달라 NaN 이 대량 생깁니다."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = os.path.join(CACHE_DIR,
                       f"gates_{abs(hash(tuple(symbols)))}_{start}_{end}.pkl")
    if use_cache and os.path.exists(key):
        with open(key, "rb") as f:
            return pickle.load(f)

    import yfinance as yf
    raw = yf.download(symbols, start=start, end=end, progress=False,
                      auto_adjust=True)
    close = raw["Close"] if "Close" in raw else raw
    out = {}
    for s in symbols:
        if s not in close.columns:
            continue
        ser = close[s].dropna()
        if len(ser) > 1500:
            out[s] = ser
    with open(key, "wb") as f:
        pickle.dump(out, f)
    return out


# ---------------------------------------------------------------------------
# 전략 팔 — 서로 진짜로 달라야 합니다
# ---------------------------------------------------------------------------

def arm_signals(prices: np.ndarray) -> np.ndarray:
    """(T, K) 팔별 포지션. 전부 **t 시점까지의 정보만** 씁니다.

    팔이 서로 상관이 높으면 밴딧이 탐색을 낭비합니다. 그래서 방향이 반대이거나
    (모멘텀 vs 반전) 성격이 다른(돌파, 현금) 것으로 골랐습니다.
    """
    n = len(prices)
    p = np.asarray(prices, dtype=float)
    logp = np.log(p)
    sig = np.zeros((n, len(ARM_NAMES)))

    # 1) 시계열 모멘텀 60일
    L = 60
    for t in range(L, n):
        sig[t, 0] = 1.0 if logp[t] > logp[t - L] else -1.0

    # 2) 단기 반전 5일 (부호 반대)
    R = 5
    for t in range(R, n):
        sig[t, 1] = -1.0 if logp[t] > logp[t - R] else 1.0

    # 3) 돌파 (Donchian 20)
    D = 20
    for t in range(D, n):
        hi, lo = p[t - D:t].max(), p[t - D:t].min()
        sig[t, 2] = 1.0 if p[t] >= hi else (-1.0 if p[t] <= lo else 0.0)

    # 4) 현금
    sig[:, 3] = 0.0
    return sig


def arm_rewards(prices: np.ndarray, horizon: int = 5,
                cost_bps: float = 33.0) -> np.ndarray:
    """(T, K) 각 시점에 각 팔을 골랐을 때의 **비용 반영** 보상.

    보상은 [t, t+h] 구간 수익률입니다. 그래서 t 시점에는 알 수 없고,
    밴딧은 `DelayedRewardBuffer(horizon=h)` 로 h 기 뒤에 학습합니다.

    비용은 포지션 **변화량** 에 부과합니다. 기본 33bp 는 국내 1,000원대 주식의
    왕복비용(수수료 3 + 거래세 20 + 최소 스프레드 10)입니다 —
    `engine.markets.round_trip_cost_bps()` 산출값.
    """
    p = np.asarray(prices, dtype=float)
    n = len(p)
    sig = arm_signals(p)
    logp = np.log(p)

    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = logp[horizon:] - logp[: n - horizon]

    rew = np.full((n, sig.shape[1]), np.nan)
    for k in range(sig.shape[1]):
        turn = np.zeros(n)
        turn[1:] = np.abs(np.diff(sig[:, k]))
        rew[:, k] = sig[:, k] * fwd - turn * (cost_bps / 1e4)
    return rew


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def build_panel(data: dict, horizon: int, cost_bps: float, max_symbols=None):
    """종목별 (컨텍스트, 보상) 을 만들어 **날짜 우선 순서로 쌓습니다.**

    종목 풀링이 이 설계에서 가장 레버리지 높은 변경입니다 — 20종목이면
    유효 결정 수 T 가 20배가 됩니다. 밴딧 리그렛이 sqrt(T) 로 줄기 때문에
    이게 없으면 일봉 5년(1,250 결정)으로는 어떤 결론도 못 냅니다.
    """
    from engine.regime import RegimeDetector, build_context_series

    det_kw = dict(vol_window=20, vr_q=5, vr_window=120, quantile_window=250)
    rows = []
    per_symbol = {}
    syms = list(data.keys())[:max_symbols] if max_symbols else list(data.keys())

    for i, s in enumerate(syms, 1):
        px = data[s].values.astype(float)
        if len(px) < 900:
            continue
        det = RegimeDetector(**det_kw)
        built = build_context_series(px, det)
        rew = arm_rewards(px, horizon=horizon, cost_bps=cost_bps)
        v0 = built["valid_from"]
        dates = data[s].index
        for t in range(v0, len(px) - horizon - 1):
            if not np.all(np.isfinite(rew[t])):
                continue
            rows.append((dates[t], built["contexts"][t], rew[t]))
        per_symbol[s] = (px, built)
        print(f"    [{i}/{len(syms)}] {s:12s} {len(px):5d}봉  "
              f"유효 {len(px) - v0 - horizon - 1:5d}", flush=True)

    if not rows:
        return None, None, per_symbol, None
    rows.sort(key=lambda r: r[0])              # 날짜 우선 — 실제 운용 순서
    ctx = np.vstack([r[1] for r in rows])
    rew = np.vstack([r[2] for r in rows])
    # ★ 날짜 서수 — 밴딧의 보상 지연을 **거래일** 로 재게 합니다.
    #   이걸 안 넘기면 지연이 행 단위가 되어 h행 = h/종목수 거래일이 됩니다.
    uniq = sorted({r[0] for r in rows})
    ordinal = {d: i for i, d in enumerate(uniq)}
    tidx = np.array([ordinal[r[0]] for r in rows], dtype=np.int64)
    return ctx, rew, per_symbol, tidx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="both", choices=["kr", "us", "both"])
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=33.0)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-symbols", type=int, default=None)
    args = ap.parse_args()

    from engine.bandit import run_t5_experiment
    from engine.regime import RegimeDetector, validate_regime_layer

    markets = ([("KR", KR_UNIVERSE)] if args.market == "kr" else
               [("US", US_UNIVERSE)] if args.market == "us" else
               [("KR", KR_UNIVERSE), ("US", US_UNIVERSE)])

    for name, universe in markets:
        print("=" * 78)
        print(f"  {name} 시장 — 게이트 실행")
        print("=" * 78)
        print("  데이터 수집…", flush=True)
        data = fetch(universe, use_cache=not args.no_cache)
        if not data:
            print("  데이터를 받지 못했습니다.")
            continue
        print(f"  {len(data)}종목 확보\n")

        # ---- 국면 계층 단독 검증 (대표 종목) --------------------------
        print("-" * 78)
        print("  [1] 국면 계층 단독 검증 T1/T2/T4")
        print("-" * 78)
        results = []
        for s in list(data.keys())[:5]:
            px = data[s].values.astype(float)
            v = validate_regime_layer(px, RegimeDetector(), horizon=10,
                                      n_perm=200)
            if v.get("T1_forward_vol_r2") is None:
                continue
            results.append(v)
            print(f"    {s:12s} T1 R²={v['T1_forward_vol_r2']:+.3f}  "
                  f"T2 p={v['T2_p_vol']:.4f}  "
                  f"T4 재현율={v['T4_reproduced_fraction']:.0%}  "
                  f"→ {'통과' if v['passed'] else '미통과'}", flush=True)
        if results:
            n_pass = sum(1 for v in results if v["passed"])
            t4 = np.mean([v["T4_reproduced_fraction"] for v in results
                          if v["T4_reproduced_fraction"] is not None])
            print(f"\n    종합: {n_pass}/{len(results)} 통과 · "
                  f"T4 평균 재현율 {t4:.0%} (문턱 70%)")

        # ---- T5 ------------------------------------------------------
        print()
        print("-" * 78)
        print("  [2] 패널 구성 (종목 풀링)")
        print("-" * 78)
        ctx, rew, _, tidx = build_panel(data, args.horizon, args.cost_bps,
                                        args.max_symbols)
        if ctx is None:
            print("  패널을 만들지 못했습니다.")
            continue

        print(f"\n    유효 결정 수 T = {len(ctx):,}  "
              f"(컨텍스트 d={ctx.shape[1]}, 팔 K={rew.shape[1]})")
        print(f"    팔: {', '.join(ARM_NAMES)}")
        print(f"    보상 호라이즌 h={args.horizon}, 비용 {args.cost_bps}bp")

        print()
        print("-" * 78)
        print("  [3] ★ T5 — 컨텍스추얼 밴딧 vs 비컨텍스추얼 UCB")
        print("-" * 78)
        out = run_t5_experiment(ctx, rew, horizon=args.horizon,
                                n_seeds=args.seeds, time_index=tidx)
        if out.get("pooling_note"):
            print(f"    {out['pooling_note']}")
        if out.get("integrity_warning"):
            print()
            for line in _wrap(out["integrity_warning"], 74):
                print(f"    {line}")
            print()
        if out.get("passed") is None:
            print("   ", out.get("verdict") or out.get("reason"))
            continue

        print(f"\n    {'컨트롤러':<22s} {'결정당 평균보상':>16s}")
        for nm, v in out["ranking"]:
            star = " ←" if nm == "LinTS" else ""
            print(f"    {nm:<22s} {v:>16.8f}{star}")

        print(f"\n    구간 대응 t = {out['t_stat']:+.3f}  (블록 {out['n_blocks']}개)")
        print(f"    판정: {'통과' if out['passed'] else '미통과'}")
        print()
        for line in _wrap(out["verdict"], 74):
            print(f"    {line}")
        print()


def _wrap(text, width):
    words, cur, out = text.split(), "", []
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out


if __name__ == "__main__":
    main()
