# -*- coding: utf-8 -*-
"""
전처리·앙상블·보호장치 검증
--------------------------
freqtrade / FinRL / hummingbot 에서 이식한 세 모듈을 네트워크 없이 검증합니다.

    python tests/test_algo.py
"""

import sys
from datetime import datetime, timedelta
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


def _bars(n: int = 200, seed: int = 7, start_price: float = 50_000.0) -> pd.DataFrame:
    """합성 일봉 — 로그정규 랜덤워크 + 일중 진폭 + 거래량."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.015, n)
    close = start_price * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.008, n))
    high = close * (1 + spread)
    low = close * (1 - spread)
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    high = np.maximum.reduce([high, close, open_])
    low = np.minimum.reduce([low, close, open_])
    volume = rng.integers(100_000, 500_000, n).astype(float)
    idx = pd.bdate_range(end="2026-08-05", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


# ---------------------------------------------------------------------------
# 전처리 (engine/preprocess.py)
# ---------------------------------------------------------------------------

def test_preprocess():
    print("\n[전처리] engine/preprocess.py")
    from engine import preprocess

    # 1) 정상 데이터는 손대지 않는다
    df = _bars(150)
    out, q = preprocess.clean(df)
    check("prep", "정상 봉은 수정 없이 통과", q.ok and len(out) == 150
          and not q.repairs and q.score > 0.9, f"score={q.score}")
    check("prep", "정상 봉의 가격 보존", bool(np.allclose(out["close"], df["close"])))

    # 2) 중복 봉 병합
    dup = pd.concat([df, df.iloc[[50, 51]]]).sort_index()
    out, q = preprocess.clean(dup)
    check("prep", "같은 시각 중복 봉 병합", len(out) == 150
          and q.dropped.get("중복 봉") == 2)

    # 3) 종가 0·NaN 봉 제거
    bad = df.copy()
    bad.iloc[10, bad.columns.get_loc("close")] = 0.0
    bad.iloc[20, bad.columns.get_loc("close")] = np.nan
    out, q = preprocess.clean(bad)
    check("prep", "종가 0·결측 봉 제거", len(out) == 148
          and q.dropped.get("종가 결측·0 이하") == 2)

    # 4) OHLC 정합성 복원 — 고가 < 종가
    broken = df.copy()
    broken.iloc[30, broken.columns.get_loc("high")] = \
        broken.iloc[30]["close"] * 0.5
    out, q = preprocess.clean(broken)
    row = out.iloc[30]
    check("prep", "고가<종가 모순 복원", row["high"] >= row["close"] - 1e-9
          and row["high"] >= row["open"] - 1e-9)

    # 5) 액면분할 역조정 — 5:1 분할 (가격 1/5, 거래량 5배)
    split = df.copy()
    k = 100
    split.iloc[k:, [split.columns.get_loc(c) for c in ("open", "high", "low", "close")]] /= 5.0
    split.iloc[k:, split.columns.get_loc("volume")] *= 5.0
    out, q = preprocess.clean(split)
    jump = out["close"].iloc[k] / out["close"].iloc[k - 1]
    check("prep", "5:1 액면분할 감지·역조정", q.flags.get("splits_adjusted")
          and 0.5 < jump < 2.0, f"경계 수익률 {jump:.3f}")
    # 조정 후 수익률 시계열이 원본(분할 없는)과 같아야 합니다
    orig_ret = np.diff(np.log(df["close"].values))
    adj_ret = np.diff(np.log(out["close"].values))
    check("prep", "분할 조정 후 수익률 보존",
          bool(np.allclose(orig_ret, adj_ret, atol=1e-9)))

    # 6) 진짜 급락(-40%, 되돌림 없음, 비표준 비율)은 조정하지 않는다
    crash = df.copy()
    crash.iloc[k:, crash.columns.get_loc("close")] *= 0.63   # 1:1.59 — 실제 분할비 아님
    for c in ("open", "high", "low"):
        crash.iloc[k:, crash.columns.get_loc(c)] *= 0.63
    out, q = preprocess.clean(crash)
    check("prep", "비표준 비율 급락은 그대로 둠 (실제 폭락 보호)",
          not q.flags.get("splits_adjusted"))

    # 7) 단일 봉 오틱 복원 — +25% 튀고 다음 봉에서 복귀
    spike = df.copy()
    j = 80
    orig_close = float(spike.iloc[j]["close"])
    spike.iloc[j, spike.columns.get_loc("close")] = orig_close * 1.25
    spike.iloc[j, spike.columns.get_loc("high")] = orig_close * 1.26
    out, q = preprocess.clean(spike)
    repaired = float(out.iloc[j]["close"])
    check("prep", "되돌아온 오틱 복원", q.flags.get("spikes_repaired", 0) >= 1
          and abs(repaired / orig_close - 1) < 0.05,
          f"{orig_close * 1.25:,.0f} → {repaired:,.0f}")

    # 8) 상한가 연속(되돌림 없음)은 고치지 않는다
    limit_up = df.copy()
    for step, mult in ((90, 1.29), (91, 1.29 * 1.29)):
        for c in ("open", "high", "low", "close"):
            limit_up.iloc[step, limit_up.columns.get_loc(c)] = \
                float(df.iloc[89]["close"]) * mult
    out, q = preprocess.clean(limit_up)
    check("prep", "연속 상한가는 보존 (진짜 추세 보호)",
          not q.flags.get("spikes_repaired"),
          f"repairs={q.repairs}")

    # 9) 거래정지 표시 (수정하지 않고 표시만)
    halted = df.copy()
    for step in range(60, 66):
        price = float(halted.iloc[59]["close"])
        for c in ("open", "high", "low", "close"):
            halted.iloc[step, halted.columns.get_loc(c)] = price
        halted.iloc[step, halted.columns.get_loc("volume")] = 0.0
    out, q = preprocess.clean(halted)
    check("prep", "거래정지 봉은 표시만", q.flags.get("halted_bars", 0) >= 6
          and len(out) == 150)
    check("prep", "거래정지가 품질 점수를 깎음", q.score < 0.99, f"score={q.score}")

    # 10) 열 이름 별칭·시간 컬럼 표준화
    alien = df.reset_index().rename(columns={
        "index": "date", "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"})
    out, q = preprocess.clean(alien)
    check("prep", "컬럼 별칭·date 컬럼 표준화", q.ok and len(out) == 150
          and isinstance(out.index, pd.DatetimeIndex))

    # 11) 빈/깨진 입력에서 죽지 않는다
    out, q = preprocess.clean(pd.DataFrame())
    check("prep", "빈 DataFrame 안전 처리", not q.ok and q.error != "")
    out, q = preprocess.clean(pd.DataFrame({"foo": [1, 2, 3]}))
    check("prep", "OHLC 없는 입력 안전 처리", not q.ok)

    # 12) 워밍업 계산
    check("prep", "warmup_bars(20,26,14) = 26*3+5", preprocess.warmup_bars(20, 26, 14) == 83)
    check("prep", "trim_warmup 은 여유 없으면 안 자름",
          len(preprocess.trim_warmup(df.iloc[:50], 83)) == 50)


# ---------------------------------------------------------------------------
# 앙상블 (engine/ensemble.py)
# ---------------------------------------------------------------------------

def test_ensemble():
    print("\n[앙상블] engine/ensemble.py")
    from engine import ensemble

    # 1) 시평선 합치 — 같은 방향이면 가중평균 그대로
    score, info = ensemble.blend_timeframes(0.6, 0.4, 0.35)
    expected = 0.65 * 0.6 + 0.35 * 0.4
    check("ens", "동의 시 가중평균 유지", abs(score - expected) < 1e-9,
          f"{score:.4f} vs {expected:.4f}")

    # 2) 반대 방향이면 깎인다 (강한 반대일수록 많이)
    weak_conflict, _ = ensemble.blend_timeframes(0.9, -0.05, 0.35)
    strong_conflict, info = ensemble.blend_timeframes(0.9, -0.6, 0.35)
    plain = 0.65 * 0.9 + 0.35 * -0.6
    check("ens", "불일치 시 확신 감산", strong_conflict < plain and not info["agree"])
    check("ens", "반대가 강할수록 더 깎음",
          abs(strong_conflict) / abs(plain) < abs(weak_conflict) / abs(0.65 * 0.9 - 0.35 * 0.05))

    # 3) 분봉 없으면 일봉 그대로
    score, info = ensemble.blend_timeframes(0.5, None, 0.35)
    check("ens", "분봉 없으면 일봉 점수 유지", score == 0.5 and not info["used"])

    # 4) 난기류 — 평상시 마지막 날은 낮고, 충격일은 높다
    calm = _bars(300, seed=11)
    t_calm = ensemble.turbulence(calm)
    shocked = calm.copy()
    last = shocked.index[-1]
    shocked.loc[last, "close"] = shocked["close"].iloc[-2] * 0.88   # -12% 급락
    shocked.loc[last, "low"] = shocked.loc[last, "close"] * 0.97
    shocked.loc[last, "high"] = shocked["close"].iloc[-2] * 1.01
    shocked.loc[last, "volume"] = float(shocked["volume"].iloc[-30:].mean() * 8)
    t_shock = ensemble.turbulence(shocked)
    check("ens", "평상일 난기류 백분위 < 90",
          t_calm is not None and t_calm["percentile"] < 90,
          f"{t_calm['percentile'] if t_calm else None}")
    check("ens", "급락+거래폭증일 난기류 > 95 백분위",
          t_shock is not None and t_shock["percentile"] >= 95
          and t_shock["elevated"],
          f"{t_shock['percentile'] if t_shock else None}")

    # 5) 횡단면 난기류 (FinRL 원본형)
    closes = pd.DataFrame({f"T{i}": _bars(300, seed=20 + i)["close"].values
                           for i in range(5)},
                          index=_bars(300, seed=20).index)
    mt = ensemble.market_turbulence(closes)
    shocked_closes = closes.copy()
    shocked_closes.iloc[-1] = shocked_closes.iloc[-2] * 0.90   # 전 종목 동반 -10%
    mt_shock = ensemble.market_turbulence(shocked_closes)
    check("ens", "횡단면 난기류 계산", mt is not None and mt["tickers"] == 5)
    check("ens", "동반 급락일 횡단면 난기류 상승",
          mt_shock is not None and mt_shock["distance"] > mt["distance"]
          and mt_shock["elevated"])

    # 6) 변동성 배수 — 최근 변동성을 3배로 키우면 배수 > 1
    calm_vol = ensemble.volatility_factor(calm)
    hot = calm.copy()
    tail = hot.index[-10:]
    center = hot.loc[tail, "close"]
    hot.loc[tail, "high"] = center * 1.06
    hot.loc[tail, "low"] = center * 0.94
    hot_vol = ensemble.volatility_factor(hot)
    check("ens", "변동성 배수 기본 동작", calm_vol["ok"] and 0.5 < calm_vol["factor"] < 1.6,
          f"factor={calm_vol['factor']}")
    check("ens", "변동성 급증 시 배수 > 1.2", hot_vol["ok"] and hot_vol["factor"] > 1.2,
          f"factor={hot_vol['factor']}")

    # 7) 배리어 스케일 — 고정 % 만 건드리고 ATR·보유기간은 안 건드림
    cfg = {"stop_loss_pct": 4.0, "take_profit_pct": 8.0, "trailing_stop_pct": 2.0,
           "atr_stop_mult": 2.0, "max_hold_days": 15}
    scaled = ensemble.scale_barriers(cfg, 1.5)
    check("ens", "배리어 스케일 적용", scaled.get("stop_loss_pct") == 6.0
          and scaled.get("take_profit_pct") == 12.0)
    check("ens", "ATR 배수·보유기간은 스케일 제외",
          "atr_stop_mult" not in scaled and "max_hold_days" not in scaled)
    check("ens", "배수 1.0 은 무변경", ensemble.scale_barriers(cfg, 1.0) == {})

    # 8) 비용 게이트 — 한국 주식 0.2% 목표는 왕복비용을 못 이김
    from engine.instruments import try_resolve
    inst = try_resolve("005930")
    if inst is None:
        check("ens", "비용 게이트 (종목 해석 불가로 건너뜀)", True, "offline")
    else:
        tiny = ensemble.cost_edge(inst, 70000, 70000 * 1.002)     # +0.2% 목표
        fat = ensemble.cost_edge(inst, 70000, 70000 * 1.05)       # +5% 목표
        check("ens", "0.2% 목표는 비용 게이트 거부", not tiny["ok"],
              f"cost={tiny.get('cost_pct')}% target={tiny.get('target_pct')}%")
        check("ens", "5% 목표는 비용 게이트 통과", fat["ok"],
              f"ratio={fat.get('ratio')}")

    # 9) compute 통합 — observe 는 점수를 바꾸지 않는다
    state = ensemble.compute(calm, 0.9, -0.6, {"algo_mode": "observe",
                                               "intraday_weight": 0.35})
    check("ens", "observe 모드는 점수 불변", state.ok
          and state.score == state.base_score and state.vol_factor == 1.0)

    # 10) soft 는 불일치 감산이 반영된다
    state_soft = ensemble.compute(calm, 0.9, -0.6, {"algo_mode": "soft",
                                                    "intraday_weight": 0.35})
    check("ens", "soft 모드는 불일치 감산 반영", state_soft.ok
          and abs(state_soft.score) < abs(0.65 * 0.9 - 0.35 * 0.6))

    # 11) gate + 난기류 극단이면 차단
    state_gate = ensemble.compute(shocked, 0.9, 0.8,
                                  {"algo_mode": "gate", "intraday_weight": 0.35,
                                   "algo_turbulence_pct": 90.0})
    blocked_when_extreme = (state_gate.turbulence.get("level") != "extreme"
                            or state_gate.block)
    check("ens", "gate 모드 난기류 극단 차단 규칙", state_gate.ok and blocked_when_extreme,
          f"level={state_gate.turbulence.get('level')} block={state_gate.block}")

    # 12) off 는 아무것도 하지 않는다
    state_off = ensemble.compute(calm, 0.5, 0.5, {"algo_mode": "off"})
    check("ens", "off 모드는 무동작", not state_off.ok)


# ---------------------------------------------------------------------------
# 보호장치 (engine/protections.py)
# ---------------------------------------------------------------------------

def _trade(symbol: str, minutes_ago: float, pnl: float, reason: str = "",
           now: datetime = None) -> dict:
    now = now or datetime(2026, 8, 5, 14, 0, 0)
    return {"symbol": symbol, "realized_pnl": pnl, "reason": reason,
            "closed_at": (now - timedelta(minutes=minutes_ago)).isoformat()}


def test_protections():
    print("\n[보호장치] engine/protections.py")
    from engine import protections

    now = datetime(2026, 8, 5, 14, 0, 0)
    base_cfg = {"protect_enabled": True}

    # 1) 꺼져 있으면(기본값) 아무것도 잠그지 않는다
    trades = [_trade("005930", 5, -50_000, "손절 도달", now)] * 5
    locks = protections.evaluate(1, "paper", {}, equity=10_000_000, now=now,
                                 trades=trades)
    check("prot", "기본값(꺼짐)은 무동작", not locks.enabled and not locks.locks)

    # 2) 쿨다운 — 방금 청산한 종목은 잠기고, 오래된 종목은 풀린다
    trades = [_trade("005930", 10, 30_000, "목표가 도달", now),
              _trade("000660", 90, 10_000, "목표가 도달", now)]
    locks = protections.evaluate(1, "paper",
                                 {**base_cfg, "protect_cooldown_min": 30},
                                 equity=10_000_000, now=now, trades=trades)
    check("prot", "쿨다운: 10분 전 청산 종목 잠금",
          locks.for_symbol("005930", now) is not None)
    check("prot", "쿨다운: 90분 전 청산 종목은 통과",
          locks.for_symbol("000660", now) is None)
    check("prot", "쿨다운은 전역 잠금이 아님", not locks.global_reasons(now))

    # 3) 손절 감시 — 4시간 안 손절 3회면 전역 잠금
    trades = [_trade("005930", 30, -80_000, "손절 도달 (68,000 vs 68,500)", now),
              _trade("035720", 60, -60_000, "손실 한도 4% 초과", now),
              _trade("000660", 100, -90_000, "고점 대비 3.1% 되돌림 (트레일링 3%)", now)]
    cfg = {**base_cfg, "protect_cooldown_min": 0,
           "protect_stoploss_count": 3, "protect_stoploss_lookback_min": 240,
           "protect_stoploss_stop_min": 60}
    locks = protections.evaluate(1, "paper", cfg, equity=10_000_000, now=now,
                                 trades=trades)
    check("prot", "손절 3회 → 전역 잠금", len(locks.global_reasons(now)) == 1,
          "; ".join(locks.global_reasons(now)))

    # 4) 손절 2회면 잠기지 않는다
    locks = protections.evaluate(1, "paper", cfg, equity=10_000_000, now=now,
                                 trades=trades[:2])
    check("prot", "손절 2회는 통과", not locks.global_reasons(now))

    # 5) 익절은 손절로 세지 않는다
    profit_trades = [_trade("005930", 30, 80_000, "목표가 도달", now)] * 5
    locks = protections.evaluate(1, "paper", cfg, equity=10_000_000, now=now,
                                 trades=profit_trades)
    check("prot", "익절 5회는 손절 감시 미발동", not locks.global_reasons(now))

    # 6) 잠금 만료 — 마지막 손절 100분 뒤면 60분 잠금은 풀려 있다
    old = [_trade("005930", 150, -80_000, "손절 도달", now),
           _trade("035720", 130, -60_000, "손절 도달", now),
           _trade("000660", 110, -90_000, "손절 도달", now)]
    locks = protections.evaluate(1, "paper", cfg, equity=10_000_000, now=now,
                                 trades=old)
    check("prot", "잠금 시간 경과 후 자동 해제", not locks.global_reasons(now))

    # 7) 낙폭 감시 — 실현손익 낙폭 5% 초과 시 전역 잠금
    dd_cfg = {**base_cfg, "protect_cooldown_min": 0, "protect_stoploss_count": 0,
              "protect_drawdown_pct": 5.0, "protect_drawdown_lookback_min": 1440,
              "protect_drawdown_min_trades": 3, "protect_drawdown_stop_min": 240}
    dd_trades = [_trade("A", 500, 200_000, "목표가 도달", now),
                 _trade("B", 400, -300_000, "손절 도달", now),
                 _trade("C", 300, -200_000, "손절 도달", now),
                 _trade("D", 200, -150_000, "손절 도달", now)]
    # peak 200k → 저점 -450k : 낙폭 650k = 자산 10M 의 6.5%
    locks = protections.evaluate(1, "paper", dd_cfg, equity=10_000_000, now=now,
                                 trades=dd_trades)
    check("prot", "실현손익 낙폭 6.5% > 5% → 전역 잠금",
          len(locks.global_reasons(now)) == 1, "; ".join(locks.global_reasons(now)))
    locks = protections.evaluate(1, "paper", dd_cfg, equity=100_000_000, now=now,
                                 trades=dd_trades)
    check("prot", "자산이 크면(낙폭 0.65%) 통과", not locks.global_reasons(now))

    # 8) 부진 종목 — 그 종목만 잠기고 다른 종목은 통과
    lp_cfg = {**base_cfg, "protect_cooldown_min": 0, "protect_stoploss_count": 0,
              "protect_drawdown_pct": 0,
              "protect_lowprofit_pct": 0.0, "protect_lowprofit_lookback_min": 1440,
              "protect_lowprofit_min_trades": 3, "protect_lowprofit_stop_min": 120}
    lp_trades = [_trade("222810", 60, -40_000, "손절 도달", now),
                 _trade("222810", 120, -35_000, "손절 도달", now),
                 _trade("222810", 180, -30_000, "신호 반전", now),
                 _trade("005930", 60, 90_000, "목표가 도달", now),
                 _trade("005930", 120, 80_000, "목표가 도달", now),
                 _trade("005930", 180, 70_000, "목표가 도달", now)]
    locks = protections.evaluate(1, "paper", lp_cfg, equity=10_000_000, now=now,
                                 trades=lp_trades)
    check("prot", "3연패 종목만 잠금", locks.for_symbol("222810", now) is not None)
    check("prot", "수익 종목은 통과", locks.for_symbol("005930", now) is None)
    check("prot", "부진 종목 잠금은 전역이 아님", not locks.global_reasons(now))

    # 9) 표본 미달이면 판정하지 않는다
    locks = protections.evaluate(1, "paper", lp_cfg, equity=10_000_000, now=now,
                                 trades=lp_trades[:2])
    check("prot", "표본 2건은 부진 판정 안 함", not locks.locks)

    # 10) 이력이 없어도 죽지 않는다
    locks = protections.evaluate(1, "paper", base_cfg, equity=10_000_000, now=now,
                                 trades=[])
    check("prot", "빈 이력 안전 처리", locks.enabled and not locks.locks)


# ---------------------------------------------------------------------------
# 통합 — strategy.evaluate 경로 (주입 데이터, 네트워크 없음)
# ---------------------------------------------------------------------------

def test_strategy_integration():
    print("\n[통합] strategy.evaluate + 앙상블")
    from engine import instruments, strategy

    inst = instruments.try_resolve("005930")
    if inst is None:
        check("integ", "종목 해석 불가로 건너뜀", True, "offline")
        return

    bars = _bars(200, seed=3)
    quote = {"price": float(bars["close"].iloc[-1]),
             "price_krw": float(bars["close"].iloc[-1]), "age_sec": 1.0}

    # observe: 앙상블 진단이 붙고, 점수 로직은 기존과 같다
    sig_obs = strategy.evaluate(inst, {"algo_mode": "observe", "intraday_weight": 0.0},
                                bars_daily=bars, quote=quote, allow_fetch=False)
    check("integ", "observe: 신호 생성 + 앙상블 진단 첨부",
          sig_obs.ok and sig_obs.ensemble is not None
          and sig_obs.ensemble["mode"] == "observe")
    check("integ", "observe: 변동성 배수 미적용", sig_obs.vol_factor == 1.0)

    sig_off = strategy.evaluate(inst, {"algo_mode": "off", "intraday_weight": 0.0},
                                bars_daily=bars, quote=quote, allow_fetch=False)
    check("integ", "off: 앙상블 미첨부, 점수 동일",
          sig_off.ok and sig_off.ensemble is None
          and abs(sig_off.score - sig_obs.score) < 1e-9)

    sig_soft = strategy.evaluate(inst, {"algo_mode": "soft", "intraday_weight": 0.0},
                                 bars_daily=bars, quote=quote, allow_fetch=False)
    check("integ", "soft: 신호 생성 + 배수 부착", sig_soft.ok
          and sig_soft.vol_factor > 0)

    # plan_entry 의 배리어 스케일 — vol_factor 1.5 를 강제 주입해 확인
    sig_soft.vol_factor = 1.5
    sig_soft.direction = strategy.LONG
    sig_soft.atr = 0.0                      # ATR 폴백을 끄고 고정 % 손절 경로로
    cfg = {"algo_mode": "soft", "atr_stop_mult": 0, "stop_loss_pct": 4.0,
           "take_profit_pct": 8.0, "risk_per_trade_pct": 1.0,
           "position_pct": 20.0, "min_order_krw": 0}
    account = {"total_value": 10_000_000, "available_cash": 10_000_000}
    plan = strategy.plan_entry(inst, sig_soft, cfg, account)
    price = sig_soft.price
    expected_stop = price * (1 - 0.06)       # 4% × 1.5 = 6%
    check("integ", "plan_entry: 손절 폭 변동성 스케일 (4%→6%)",
          plan.ok and abs(plan.stop_price - expected_stop) / price < 1e-6,
          f"stop={plan.stop_price:,.0f} expected={expected_stop:,.0f}")

    # gate 의 비용 게이트 — 목표 0.2% 는 거부돼야 한다
    sig_gate = strategy.evaluate(inst, {"algo_mode": "off", "intraday_weight": 0.0},
                                 bars_daily=bars, quote=quote, allow_fetch=False)
    sig_gate.direction = strategy.LONG
    sig_gate.atr = 0.0
    sig_gate.vol_factor = 1.0
    gate_cfg = {"algo_mode": "gate", "atr_stop_mult": 0, "stop_loss_pct": 4.0,
                "take_profit_pct": 0.2, "risk_per_trade_pct": 1.0,
                "position_pct": 20.0, "min_order_krw": 0}
    plan = strategy.plan_entry(inst, sig_gate, gate_cfg, account)
    check("integ", "plan_entry: gate 모드 0.2% 목표 비용 거부",
          not plan.ok and "거래비용" in plan.reason, plan.reason[:60])


# ---------------------------------------------------------------------------

def report() -> bool:
    print("\n" + "=" * 60)
    names = {"prep": "전처리", "ens": "앙상블", "prot": "보호장치", "integ": "통합"}
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
    test_preprocess()
    test_ensemble()
    test_protections()
    test_strategy_integration()
    ok = report()
    sys.exit(0 if ok else 1)
