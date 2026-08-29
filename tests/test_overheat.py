# -*- coding: utf-8 -*-
"""
진입 타이밍 검증 (과열 게이트 + 눌림목 재진입 + 공선성 보정)
----------------------------------------------------------
"급등 고점에서 매수 신호가 난다"는 실제 증상을 재현하고, 세 처방이 각각
제 몫만 하는지 확인합니다. 네트워크 없이 합성 봉으로 돕니다.

    engine/strategy.overheat_check    고점 추격 진입 차단
    engine/strategy.pullback_ready    급등 뒤 첫 눌림에서 문턱 완화
    engine/indicators._apply_family_cap   같은 질문을 하는 지표군의 비중 상한
    engine/indicators._regime_multipliers 국면 배수 연속화 + 되돌림 하한

    python tests/test_overheat.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from engine import indicators, strategy

results = []


def check(name: str, passed: bool, detail: str = ""):
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


def _bars(n: int = 200, drift: float = 0.012, runlen: int = 40,
          seed: int = 7) -> pd.DataFrame:
    """조용한 구간 뒤에 drift 만큼의 추세가 runlen 봉 이어지는 합성 일봉."""
    rng = np.random.default_rng(seed)
    base = 10_000 + np.cumsum(rng.normal(0, 60, n - runlen))
    run = base[-1] * np.cumprod(1 + rng.normal(drift, 0.008, runlen))
    close = np.concatenate([base, run])
    return pd.DataFrame(
        {"open": close * 1.001, "high": close * 1.012, "low": close * 0.988,
         "close": close, "volume": rng.integers(80_000, 200_000, n)},
        index=pd.date_range("2025-01-01", periods=n, freq="B"))


def _facts(df: pd.DataFrame, smooth: bool = None, extreme_at: float = None) -> dict:
    """게이트에 넣을 재료 — 실매매와 같은 경로(indicators.analyze)로 뽑습니다.

    smooth·extreme_at 을 주면 그 설정으로만 계산하고 원래 값을 되돌립니다
    (예전 동작과 지금 동작을 같은 봉으로 비교할 때 씁니다).
    """
    saved = dict(indicators.REGIME_MULTIPLIER)
    try:
        if smooth is not None:
            indicators.REGIME_MULTIPLIER["smooth"] = smooth
        if extreme_at is not None:
            indicators.REGIME_MULTIPLIER["extreme_at"] = extreme_at
        analysis = indicators.analyze(df)
    finally:
        indicators.REGIME_MULTIPLIER.clear()
        indicators.REGIME_MULTIPLIER.update(saved)
    out = {"score": analysis.score, "rsi": None, "bb": None}
    for item in analysis.indicators:
        if item.key == "rsi":
            out["rsi"] = (item.values or {}).get("rsi")
        elif item.key == "bollinger":
            out["bb"] = (item.values or {}).get("pct_b")
    price = float(df["close"].iloc[-1])
    out["price"] = price
    out["atr"] = strategy._atr(df, price)[0]
    return out


CFG = {}


def test_symptom():
    """증상 재현 — 보정을 끄면 급등 고점에서 강한 매수 점수가 그대로 납니다.

    보정을 끈 상태를 일부러 재현하는 이유: 이 점수가 고쳐졌다는 것을 숫자로
    남겨두지 않으면, 나중에 상한을 만졌을 때 무엇이 되돌아간 것인지 알 수 없습니다.
    """
    print("\n[증상 재현]")
    df = _bars(drift=0.012)
    now = _facts(df)                                       # 기본 설정 (감쇠해제 꺼짐)
    unclamped = _facts(df, smooth=True, extreme_at=0.45)   # 켰을 때

    check("급등 고점에서 RSI 가 70 을 넘는다", now["rsi"] > 70,
          f"RSI {now['rsi']:.1f}")
    check("기본 설정에서는 기술점수가 진입 임계값(0.35)을 크게 넘는다",
          now["score"] > 0.5, f"기본 {now['score']:+.3f} — 그래서 게이트가 필요합니다")
    check("감쇠해제를 켜면 고점 점수가 뚜렷하게 낮아진다",
          unclamped["score"] < now["score"] - 0.1,
          f"{now['score']:+.3f} → {unclamped['score']:+.3f}")
    check("켜도 여전히 양수다 — 점수만으로는 못 막는다는 것이 게이트의 존재 이유",
          unclamped["score"] > 0, f"{unclamped['score']:+.3f}")


def test_blocks_blowoff():
    """수직 급등은 막는다."""
    print("\n[고점 차단]")
    f = _facts(_bars(drift=0.012))
    h = strategy.overheat_check(strategy.LONG, f["price"], f["atr"],
                                f["rsi"], f["bb"], _bars(drift=0.012), CFG)
    check("급등 고점 매수 차단", h["blocked"], h["note"][:60])
    check("세 재료가 모두 걸렸다", h["hits"] == 3, f"{h['hits']}표")
    check("이격이 한도를 넘었다", h["detail"]["ext_atr"] > 3.0,
          f"{h['detail']['ext_atr']:.1f} ATR")


def test_passes_healthy_trend():
    """완만한 정상 상승은 통과시킨다 — 게이트가 정지 버튼이 되면 안 됩니다."""
    print("\n[정상 추세 통과]")
    df = _bars(drift=0.0025, runlen=60, seed=3)
    f = _facts(df)
    h = strategy.overheat_check(strategy.LONG, f["price"], f["atr"],
                                f["rsi"], f["bb"], df, CFG)
    check("완만한 상승은 통과", not h["blocked"], f"{h['hits']}표")
    check("통과해도 사유는 남는다", bool(h["note"]) or h["hits"] == 0)


def test_quiet_on_chop():
    """횡보에서는 아무 표도 서지 않는다."""
    print("\n[횡보]")
    df = _bars(drift=0.0, runlen=40, seed=11)
    f = _facts(df)
    h = strategy.overheat_check(strategy.LONG, f["price"], f["atr"],
                                f["rsi"], f["bb"], df, CFG)
    check("횡보는 과열 표 없음", h["hits"] == 0 and not h["blocked"])


def test_short_side_symmetry():
    """숏도 대칭으로 막는다 — 급락 바닥에서 추격매도 금지."""
    print("\n[숏 대칭]")
    df = _bars(drift=-0.012)
    f = _facts(df)
    short = strategy.overheat_check(strategy.SHORT, f["price"], f["atr"],
                                    f["rsi"], f["bb"], df, CFG)
    long_ = strategy.overheat_check(strategy.LONG, f["price"], f["atr"],
                                    f["rsi"], f["bb"], df, CFG)
    check("급락 바닥 숏 진입 차단", short["blocked"], short["note"][:60])
    check("같은 자리에서 롱은 막지 않는다", not long_["blocked"])


def test_switches():
    """설정으로 끄고 조일 수 있다."""
    print("\n[설정]")
    df = _bars(drift=0.012)
    f = _facts(df)
    off = strategy.overheat_check(strategy.LONG, f["price"], f["atr"],
                                  f["rsi"], f["bb"], df, {"overheat_gate": False})
    check("게이트를 끄면 막지 않는다", not off["blocked"])

    loose = strategy.overheat_check(strategy.LONG, f["price"], f["atr"],
                                    f["rsi"], f["bb"], df, {"overheat_min_hits": 3})
    check("min_hits 3 이어도 3표면 여전히 막힌다", loose["blocked"])

    df2 = _bars(drift=0.0025, runlen=60, seed=3)
    g = _facts(df2)
    tight = strategy.overheat_check(strategy.LONG, g["price"], g["atr"],
                                    g["rsi"], g["bb"], df2, {"overheat_min_hits": 1})
    check("min_hits 1 이면 완만한 상승도 막힌다", tight["blocked"],
          "1표 기준은 과하다는 근거")

    p = strategy.overheat_params({"overheat_rsi": 999, "overheat_min_hits": 99})
    check("범위 밖 설정은 안전 범위로 조인다",
          p["overheat_rsi"] == 99.0 and p["overheat_min_hits"] == 3)


def test_no_materials():
    """판정 재료가 없으면 막지 않는다 — 모르는 것을 위험으로 취급하지 않습니다."""
    print("\n[재료 없음]")
    h = strategy.overheat_check(strategy.LONG, 10_000.0, 0.0, None, None, None, CFG)
    check("RSI·%B·이격이 모두 없으면 통과", not h["blocked"] and h["hits"] == 0)
    flat = strategy.overheat_check(strategy.FLAT, 10_000.0, 100.0, 99.0, 1.0,
                                   _bars(), CFG)
    check("관망(FLAT)이면 판정 자체를 하지 않는다", not flat["blocked"])




# ---------------------------------------------------------------------------
# 눌림목 재진입
# ---------------------------------------------------------------------------

def _surge_then(after_pct: float, after_bars: int, surge_bars: int = 25,
                seed: int = 5) -> pd.DataFrame:
    """조용한 구간 → 급등 → after_pct 씩 after_bars 봉. 눌림 깊이를 만들 때 씁니다."""
    rng = np.random.default_rng(seed)
    base = list(10_000 + np.cumsum(rng.normal(0, 60, 120)))
    surge = list(base[-1] * np.cumprod(1 + rng.normal(0.015, 0.005, surge_bars)))
    tail = list(surge[-1] * np.cumprod(1 + np.full(after_bars, after_pct))) if after_bars else []
    close = np.array(base + surge + tail, float)
    return pd.DataFrame(
        {"open": close * 1.001, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": np.full(len(close), 100_000)},
        index=pd.date_range("2025-01-01", periods=len(close), freq="B"))


# 눌림목 재진입은 **기본이 꺼져 있습니다** (검증 하네스가 수익 개선을 기각).
# 동작 자체는 계속 검사해야 하므로 테스트에서만 명시적으로 켭니다.
PB_ON = {"pullback_entry": True}


def _ready(df: pd.DataFrame, cfg: dict = None) -> dict:
    price = float(df["close"].iloc[-1])
    return strategy.pullback_ready(df, price, strategy._atr(df, price)[0],
                                   {**PB_ON, **(cfg or {})})


def test_pullback():
    print("\n[눌림목 재진입]")
    top = _ready(_surge_then(0.0, 0))
    check("급등 직후 고점은 눌림목이 아니다", not top["ready"],
          f"현재 이격 {top['detail'].get('now_atr')} ATR")

    pull = _ready(_surge_then(-0.02, 5))
    check("급등 뒤 이동평균까지 되돌린 자리는 눌림목", pull["ready"],
          f"최고 {pull['detail'].get('peak_atr')} → 현재 {pull['detail'].get('now_atr')} ATR")

    broken = _ready(_surge_then(-0.022, 22))
    check("추세가 무너지면 눌림목이 아니다 (떨어지는 칼)", not broken["ready"],
          f"이동평균 상승 {broken['detail'].get('ma_rising')}")

    chop = _ready(_surge_then(0.0, 0, surge_bars=0))
    check("강세가 없었으면 눌림목도 없다", not chop["ready"],
          f"최고 이격 {chop['detail'].get('peak_atr')} ATR")

    off = _ready(_surge_then(-0.02, 5), {"pullback_entry": False})
    check("설정으로 끌 수 있다", not off["ready"])

    from storage import autotrade as store
    check("기본값은 꺼짐 (검정이 수익 개선을 기각해서)",
          not store.DEFAULT_CONFIG["pullback_entry"]
          and not strategy.PULLBACK_DEFAULTS["pullback_entry"])

    p = strategy.pullback_params({"pullback_entry_ratio": 0.01, "pullback_lookback": 999})
    check("범위 밖 설정은 안전 범위로 조인다",
          p["pullback_entry_ratio"] == 0.30 and p["pullback_lookback"] == 60,
          "문턱을 0.3 배 아래로 낮추는 것은 임계값을 없애는 것")

    short_df = _surge_then(-0.02, 5).iloc[-15:]
    check("봉이 부족하면 판정하지 않는다", not _ready(short_df)["ready"])


def test_pullback_lowers_threshold():
    """눌림목이면 실제로 진입 문턱이 낮아진다 — evaluate 를 통째로 통과시켜 확인."""
    print("\n[눌림목 → 문턱 완화]")
    from data_sources.symbol_registry import ResolvedSymbol
    from engine.instruments import STOCK, Instrument
    from storage import autotrade as store

    df = _surge_then(-0.02, 5)
    price = float(df["close"].iloc[-1])
    quote = {"price": price, "price_krw": price, "age_sec": 1.0}
    inst = Instrument(
        key="TEST01", name="테스트종목", asset_class=STOCK, market="KOSPI",
        currency="KRW", multiplier=1.0, margin_rate=1.0, shortable=False,
        symbol=ResolvedSymbol(key="TEST01", name="테스트종목", market="KOSPI",
                              yahoo_symbol="TEST01", currency="KRW"))
    cfg = dict(store.DEFAULT_CONFIG, intraday_weight=0.0, use_news=False,
               algo_mode="off", ml_mode="off", nnfx_mode="off", learn_mode="off",
               entry_score=0.35, pullback_entry=True)

    on = strategy.evaluate(inst, cfg, bars_daily=df, quote=quote, allow_fetch=False)
    off = strategy.evaluate(inst, dict(cfg, pullback_entry=False), bars_daily=df,
                            quote=quote, allow_fetch=False)
    check("눌림목 판정이 신호에 실린다", bool((on.pullback or {}).get("ready")))
    check("같은 점수인데 완화 여부만 다르다", abs(on.score - off.score) < 1e-9,
          f"점수 {on.score:+.3f}")
    stages = {s["key"]: s for s in on.stages}
    check("완화된 문턱이 계산 과정에 남는다", "pullback" in stages,
          stages.get("pullback", {}).get("detail", "")[:40])


# ---------------------------------------------------------------------------
# 공선성 보정 · 국면 배수
# ---------------------------------------------------------------------------

def _share(analysis, family: str) -> float:
    total = sum(i.weight for i in analysis.indicators) or 1.0
    return sum(i.weight for i in analysis.indicators if i.family == family) / total


def test_family_cap():
    """지표군 비중 상한 — 기본은 꺼짐. **효과가 없다는 측정**을 못으로 박습니다.

    추세 지표 9개가 전체 가중의 78% 를 먹는 것은 사실이고, 상한을 씌우면 그
    비중은 실제로 내려갑니다. 그런데 고점과 정상 추세의 **순서가 바뀌지 않습니다**
    — 둘 다 같이 내려갈 뿐입니다. 판별력이 아니라 척도 이동이라는 뜻이고,
    사용자가 저장해 둔 entry_score 는 옛 척도 기준이라 그대로 두면 계정이 굶습니다.
    나중에 누가 다시 켜고 싶어질 때 이 측정이 남아 있어야 합니다.
    """
    print("\n[지표군 비중 상한 — 기본 꺼짐]")
    top, trend = _bars(drift=0.012), _bars(drift=0.0025, runlen=60, seed=3)

    check("기본값은 상한 없음", not indicators.INDICATOR_FAMILY_CAP,
          "척도만 흔들고 판별력은 없어서 껐습니다")

    off_top, off_tr = indicators.analyze(top), indicators.analyze(trend)
    saved = dict(indicators.INDICATOR_FAMILY_CAP)
    try:
        indicators.INDICATOR_FAMILY_CAP.update({"trend": 0.50, "meanrev": 0.50})
        on_top, on_tr = indicators.analyze(top), indicators.analyze(trend)
    finally:
        indicators.INDICATOR_FAMILY_CAP.clear()
        indicators.INDICATOR_FAMILY_CAP.update(saved)

    check("켜면 추세군 비중이 실제로 상한까지 내려간다 (기능은 동작함)",
          _share(on_top, "trend") <= 0.5001 < _share(off_top, "trend"),
          f"{_share(off_top, 'trend'):.1%} → {_share(on_top, 'trend'):.1%}")
    check("그런데 고점과 정상 추세가 **같이** 내려간다",
          on_top.score < off_top.score and on_tr.score < off_tr.score,
          f"고점 {off_top.score:+.3f}→{on_top.score:+.3f} · "
          f"추세 {off_tr.score:+.3f}→{on_tr.score:+.3f}")
    check("순서가 바뀌지 않는다 = 판별력이 아니라 척도 이동",
          (off_top.score > off_tr.score) == (on_top.score > on_tr.score),
          "고점이 여전히 정상 추세보다 높은 점수를 받습니다")


def test_regime_multipliers():
    print("\n[국면 배수]")
    check("기본값은 예전 계단 방식 (검정이 수익 개선을 기각해서)",
          not indicators.REGIME_MULTIPLIER["smooth"]
          and indicators.REGIME_MULTIPLIER["extreme_at"] > 1.0,
          "점수 산식이 원본과 동일하게 유지됩니다")

    saved = dict(indicators.REGIME_MULTIPLIER)
    try:
        indicators.REGIME_MULTIPLIER["smooth"] = True
        prev, jumps = None, []
        for i in range(-100, 101):
            m = indicators._regime_multipliers(i / 100.0)[2]
            if prev is not None:
                jumps.append(abs(m["trend"] - prev))
            prev = m["trend"]
        strong = indicators._regime_multipliers(1.0)[2]
        old_style = indicators._regime_multipliers(0.9)[2]
    finally:
        indicators.REGIME_MULTIPLIER.clear()
        indicators.REGIME_MULTIPLIER.update(saved)

    check("켜면 배수가 계단이 아니라 연속이다", max(jumps) < 0.02,
          f"최대 급변 {max(jumps):.4f} (계단 방식은 경계에서 0.65)")

    check("뚜렷한 추세에서는 예전과 같은 배수에 도달한다 (점수 척도가 밀리지 않음)",
          abs(strong["trend"] - 1.45) < 1e-6 and abs(old_style["trend"] - 1.45) < 1e-6,
          f"추세점수 0.9 에서 ×{old_style['trend']:.2f}")
    check("강한 추세에서는 여전히 추세군이 더 크다",
          strong["trend"] > strong["meanrev"],
          f"추세 ×{strong['trend']:.2f} vs 되돌림 ×{strong['meanrev']:.2f}")


def test_selective_unclamp():
    """순환논리 차단 — 되돌림 지표가 실제로 경고할 때만 감쇠를 푼다."""
    print("\n[선택적 감쇠 해제]")
    top = _bars(drift=0.012)                       # RSI 99 — 되돌림군이 경고 중
    healthy = _bars(drift=0.0025, runlen=60, seed=3)  # 되돌림군이 거의 중립

    top_off = _facts(top)["score"]                             # 기본 (꺼짐)
    top_on = _facts(top, smooth=True, extreme_at=0.45)["score"]
    ok_off = _facts(healthy)["score"]
    ok_on = _facts(healthy, smooth=True, extreme_at=0.45)["score"]

    check("과열 자리에서는 점수가 내려간다", top_on < top_off - 0.1,
          f"{top_off:+.3f} → {top_on:+.3f}")
    check("되돌림군이 조용한 정상 추세는 거의 건드리지 않는다",
          abs(ok_on - ok_off) < 0.05, f"{ok_off:+.3f} → {ok_on:+.3f}")
    check("두 상황의 간격이 좁혀진다 (= 판별력)",
          (top_on - ok_on) < (top_off - ok_off),
          f"간격 {top_off - ok_off:+.3f} → {top_on - ok_on:+.3f}")
    saved = dict(indicators.REGIME_MULTIPLIER)
    try:
        indicators.REGIME_MULTIPLIER["extreme_at"] = 0.45
        note = indicators.analyze(top).note
    finally:
        indicators.REGIME_MULTIPLIER.clear()
        indicators.REGIME_MULTIPLIER.update(saved)
    check("근거 문장에 감쇠를 푼 사실이 남는다", "감쇠를 풀었습니다" in note)


def report() -> bool:
    print("\n" + "=" * 60)
    failed = [r for r in results if not r[1]]
    print(f"  총계: {len(results) - len(failed)}/{len(results)} 통과")
    for name, _, detail in failed:
        print(f"  FAIL: {name} {detail}")
    return not failed


if __name__ == "__main__":
    test_symptom()
    test_blocks_blowoff()
    test_passes_healthy_trend()
    test_quiet_on_chop()
    test_short_side_symmetry()
    test_switches()
    test_no_materials()
    test_pullback()
    test_pullback_lowers_threshold()
    test_family_cap()
    test_regime_multipliers()
    test_selective_unclamp()
    sys.exit(0 if report() else 1)
