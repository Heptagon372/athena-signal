# -*- coding: utf-8 -*-
"""포트폴리오 계층 & 보유·제도 비용 QA.

    python tests/test_portfolio.py

engine/portfolio.py (슬리브 합성)와 engine/holding.py (공매도 금지 이력·
차입비용)를 검증합니다. 네트워크가 필요 없습니다.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from engine import registry
registry.DEFAULT_PATH = os.path.join(tempfile.mkdtemp(), "trials.json")  # 사용자 장부 격리

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from engine import fills, holding, portfolio
from engine.instruments import ETF, FUTURES, STOCK, Instrument
from models import MARKET_US, ResolvedSymbol

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


def section(title):
    print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72)


def stock(key="P01", asset=STOCK, market="KOSPI", shortable=False) -> Instrument:
    return Instrument(
        key=key, name=key, asset_class=asset, market=market, currency="KRW",
        multiplier=1.0, margin_rate=1.0, shortable=shortable,
        symbol=ResolvedSymbol(key=key, name=key, market=market,
                              yahoo_symbol=key, currency="KRW"))


def make_bars(seed, n=400, drift=0.0035, common=None, beta=0.0, start=None):
    rng = np.random.default_rng(seed)
    idio = rng.normal(drift, 0.016, n)
    shocks = (idio if common is None
              else beta * common + np.sqrt(max(1e-9, 1 - beta ** 2)) * idio)
    closes = 10_000 * np.exp(np.cumsum(shocks))
    rows = []
    for i, c in enumerate(closes):
        hi = c * (1 + abs(rng.normal(0, 0.010)))
        lo = c * (1 - abs(rng.normal(0, 0.010)))
        op = lo + (hi - lo) * rng.random()
        rows.append({"date": (start or datetime(2024, 1, 1)) + timedelta(days=i),
                     "open": op, "high": max(hi, c, op), "low": min(lo, c, op),
                     "close": c, "volume": 1_000_000})
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
section("1. 공매도 금지 이력 — 숏 진입을 제도적으로 막는가")
# ---------------------------------------------------------------------------
# 없으면: 2020-03~2021-05, 2023-11~2025-03 처럼 실제로는 숏을 낼 수 없었던
#         구간의 수익이 그대로 남습니다. 하필 하락장이라 백테스트가 가장
#         좋아 보이는 구간이 가장 불가능했던 구간입니다.

inst = stock(shortable=True)          # 연구용으로 숏을 열었다고 가정
INSTITUTIONAL = {"assume_retail": False}

banned = holding.short_entry_allowed(inst, "2024-06-01", INSTITUTIONAL)
check("2024-06 전면금지 구간 — 숏 차단", not banned["allowed"], banned["reason"][:60])
check("금지 상태가 사유에 남는다", banned["state"] == "banned")

partial = holding.short_entry_allowed(inst, "2022-01-15", INSTITUTIONAL)
check("2022-01 부분허용 — 지수편입 모르면 막는다 (보수적)",
      not partial["allowed"] and partial["state"] == "partial")
check("주요지수 편입이면 부분허용 구간에서 통과",
      holding.short_entry_allowed(
          inst, "2022-01-15", {**INSTITUTIONAL, "short_in_major_index": True})["allowed"])

allowed = holding.short_entry_allowed(inst, "2025-12-01", INSTITUTIONAL)
check("2025-12 허용 구간 — 기관 기준 통과", allowed["allowed"], allowed["state"])

retail = holding.short_entry_allowed(inst, "2025-12-01", {})
check("기본값(개인)은 허용 구간이어도 막는다",
      not retail["allowed"], retail["reason"][:50])

check("선물 매도 진입은 공매도 규제 대상이 아니다",
      holding.short_entry_allowed(stock(key="101H6000", asset=FUTURES,
                                        shortable=True), "2024-06-01")["allowed"])
check("미국 종목에는 국내 이력을 적용하지 않는다",
      holding.short_entry_allowed(stock(key="AAPL", market=MARKET_US,
                                        shortable=True), "2024-06-01")["allowed"])

# ---------------------------------------------------------------------------
section("2. 차입비용 — 숏을 들고 있는 동안 나가는 돈")
# ---------------------------------------------------------------------------
cost = holding.borrow_cost(inst, 10_000_000, 365)
check("연 명목가 × 이자율", abs(cost - 10_000_000 * holding.DEFAULT_BORROW_RATE_KR) < 1,
      f"{cost:,.0f}원 (연 {holding.DEFAULT_BORROW_RATE_KR:.2%})")
check("보유 기간에 비례", abs(holding.borrow_cost(inst, 10_000_000, 30) * 12
                        - cost) < cost * 0.03)
check("선물은 차입비용 없음 (증거금 상품)",
      holding.borrow_cost(stock(key="101H6000", asset=FUTURES), 10_000_000, 365) == 0.0)
check("설정으로 이자율을 덮어쓸 수 있다",
      holding.borrow_cost(inst, 10_000_000, 365,
                          {"short_borrow_rate_annual": 0.20}) > cost * 5)

# 달력일 — 금요일에서 월요일은 3일치
fri, mon = datetime(2025, 3, 7), datetime(2025, 3, 10)
weekend = holding.accrue(inst, 10_000_000, fri, mon)
oneday = holding.accrue(inst, 10_000_000, fri, datetime(2025, 3, 8))
check("주말 이자가 붙는다 (금→월 = 3일치)", abs(weekend - oneday * 3) < 1,
      f"{weekend:,.0f}원 vs 1일 {oneday:,.0f}원")
check("같은 날은 이자 0", holding.accrue(inst, 10_000_000, fri, fri) == 0.0)

check("거부 사유가 fills 라벨 표에 등록돼 있다",
      fills.REJECTION_LABELS.get(holding.SHORT_BANNED) is not None,
      fills.REJECTION_LABELS.get(holding.SHORT_BANNED, ""))

# ---------------------------------------------------------------------------
section("3. 백테스트 루프가 실제로 숏을 막는가")
# ---------------------------------------------------------------------------
from engine import autotrade
from storage import autotrade as store

down = make_bars(seed=3, drift=-0.004, start=datetime(2024, 1, 1))   # 금지 구간 하락장
cfg_short = {**store.DEFAULT_CONFIG, "entry_score": 0.2, "allow_short": True,
             "intraday_weight": 0.0, "use_news": False, "assume_retail": False}
run = autotrade._run_backtest(stock(key="SHORTTEST", shortable=True), down,
                              cfg_short, 10_000_000)
banned_hits = [r for r in run["rejections"].items if r.reason == holding.SHORT_BANNED]
check("금지 구간 하락장에서 숏 진입이 거부된다", len(banned_hits) > 0,
      f"{len(banned_hits)}건 거부 · 총 매매 {len(run['trades'])}건")
check("거부에 사유 문장이 남는다",
      all(r.detail for r in banned_hits),
      banned_hits[0].detail[:60] if banned_hits else "")
check("거부된 숏은 매매로 잡히지 않는다",
      not any(t["side"] == "short" for t in run["trades"]))

# ---------------------------------------------------------------------------
section("4. 슬리브 합성 — 자본 보존 · 캘린더 정렬")
# ---------------------------------------------------------------------------
CASH = 30_000_000


def fake_run(bars, seed_cash):
    """자산곡선만 있는 최소 run (합성 로직만 봅니다)."""
    closes = bars["close"].to_numpy(float)
    qty = seed_cash / closes[0]
    return {"curve": [{"date": str(d)[:10], "equity": round(c * qty), "exposure": 1.0}
                      for d, c in zip(bars.index, closes)],
            "trades": [],
            "benchmark": [{"date": str(d)[:10], "equity": round(c * qty)}
                          for d, c in zip(bars.index, closes)]}


b1, b2 = make_bars(11), make_bars(23)
runs = {"AAA": fake_run(b1, CASH / 2), "BBB": fake_run(b2, CASH / 2)}
out = portfolio.combine(runs, {}, CASH)
check("합성 성공", out.get("ok"))
check("동일가중이 기본", all(abs(w - 0.5) < 1e-9 for w in out["weights"].values()),
      str(out["weights"]))
alloc = sum(v["allocated"] for v in out["by_symbol"].values())
check("배분 합계 = 투입 자본", abs(alloc - CASH) <= 2, f"{alloc:,} vs {CASH:,}")
check("연율화 기준일이 명시된다 (자동추정 아님)", out["periods_per_year"] == 252,
      str(out["periods_per_year"]))
# curve 는 뒤 250개만 실려 오므로 first_equity 로 봐야 합니다.
# (curve[0] 으로 검사하면 잘린 구간의 중간값과 비교하게 되어 늘 통과합니다)
check("첫날 자산 = 투입 자본", abs(out["first_equity"] - CASH) <= 2,
      f"{out['first_equity']:,} vs {CASH:,}")

# 캘린더가 어긋난 두 슬리브 — 한쪽이 짧아도 폭락으로 잡히면 안 됩니다
short_bars = b2.iloc[:200]
mixed = {"AAA": fake_run(b1, CASH / 2), "BBB": fake_run(short_bars, CASH / 2)}
out2 = portfolio.combine(mixed, {}, CASH)
eq = np.array([p["equity"] for p in out2["curve"]], dtype=float)
worst = float(np.min(eq[1:] / eq[:-1] - 1)) if len(eq) > 1 else 0.0
check("짧은 슬리브가 끝나도 포트폴리오가 폭락하지 않는다 (직전값 유지)",
      worst > -0.5, f"최악 일간 변동 {worst:+.1%}")
check("합집합 캘린더 = 더 긴 쪽 길이", out2["n_points"] == len(b1),
      f"{out2['n_points']} vs 긴쪽 {len(b1)} / 짧은쪽 {len(short_bars)}")
check("짧은 슬리브 종료 후에도 그 자산이 유지된다 (0 이 아님)",
      out2["curve"][-1]["equity"] > CASH / 2,
      f"최종 {out2['curve'][-1]['equity']:,}")

# 비중 정규화
skew = portfolio.combine(runs, {"AAA": 3, "BBB": 1}, CASH)
check("비중 합이 1이 아니어도 정규화", abs(skew["weights"]["AAA"] - 0.75) < 1e-9,
      str(skew["weights"]))

# ---------------------------------------------------------------------------
section("5. 분산 효과 — 이 계층을 만든 이유")
# ---------------------------------------------------------------------------
common = np.random.default_rng(5).normal(0.0035, 0.016, 400)
indep = {"A": fake_run(make_bars(11), CASH / 3),
         "B": fake_run(make_bars(23), CASH / 3),
         "C": fake_run(make_bars(37), CASH / 3)}
corr = {"D": fake_run(make_bars(11, common=common, beta=0.95), CASH / 3),
        "E": fake_run(make_bars(23, common=common, beta=0.95), CASH / 3),
        "F": fake_run(make_bars(37, common=common, beta=0.95), CASH / 3)}

d_indep = portfolio.combine(indep, {}, CASH)["diversification"]
d_corr = portfolio.combine(corr, {}, CASH)["diversification"]
check("무상관 쪽 분산비율이 더 높다",
      d_indep["ratio"] > d_corr["ratio"],
      f"무상관 {d_indep['ratio']} vs 고상관 {d_corr['ratio']}")
check("고상관 평균상관이 실제로 높게 잡힌다",
      d_corr["avg_correlation"] > d_indep["avg_correlation"] + 0.3,
      f"{d_corr['avg_correlation']:+.2f} vs {d_indep['avg_correlation']:+.2f}")
check("고상관이면 '늘려도 소용없다' 고 말해준다",
      "제한적" in d_corr["note"] or "같이 움직" in d_corr["note"], d_corr["note"][:60])
check("무상관 기준선(√n)이 상한이 아니라 기준선으로 표기됨",
      "zero_corr_reference" in d_indep and "max_possible" not in d_indep)
check("상관행렬 대각은 1", all(abs(d_indep["correlation"][k][k] - 1.0) < 1e-6
                          for k in d_indep["correlation"]))
check("슬리브 1개면 분산 효과 없음",
      portfolio.combine({"A": indep["A"]}, {}, CASH)["diversification"]["ratio"] == 1.0)

# ---------------------------------------------------------------------------
section("6. 포트폴리오 백테스트 전 구간")
# ---------------------------------------------------------------------------
bars_map = {"AAA": make_bars(11), "BBB": make_bars(23), "CCC": make_bars(37)}
autotrade._resolve_universe_item = lambda q: stock(key=q)
autotrade.feed.bars = lambda i, tf="day", count=120: bars_map[i.key]

pf = autotrade.simulate_portfolio(list(bars_map), {"entry_score": 0.2},
                                  days=380, initial_cash=CASH)
check("포트폴리오 백테스트 실행", pf.get("ok"),
      f"수익률 {pf.get('total_return_pct')}% · 매매 {pf.get('trade_count')}회")
check("종목이 1개면 거부한다",
      not autotrade.simulate_portfolio(["AAA"], {}, days=380).get("ok"))
check("슬리브별 성과가 따로 나온다", len(pf.get("by_symbol", {})) == 3)
check("거부 장부가 합산된다", "rejections" in pf)
check("게이트가 포트폴리오 레벨에서 한 번 돈다",
      pf.get("gates", {}).get("ok") is not None)
check("시행 장부에 포트폴리오로 1건 등록",
      str(pf.get("gates", {}).get("registry", {}).get("trial_id", "")) != "",
      f"trial {pf.get('gates', {}).get('registry', {}).get('trial_id')}")
check("슬리브 노출이 포트폴리오 노출로 합산됨",
      any(abs(p.get("exposure", 0)) > 1e-9 for p in pf.get("curve", [])))
check("한계를 결과에 명시한다",
      "리밸런싱" in (pf.get("execution", {}).get("note") or ""))

# 자본 보존 — 존재하지 않는 종목이 섞여도 남은 종목에 전액이 배분돼야 합니다
autotrade._resolve_universe_item = lambda q: None if q == "NOPE" else stock(key=q)
bars_map["NOPE"] = make_bars(99)
pf2 = autotrade.simulate_portfolio(["AAA", "BBB", "NOPE"], {"entry_score": 0.2},
                                   days=380, initial_cash=CASH)
if pf2.get("ok"):
    alloc2 = sum(v["allocated"] for v in pf2["by_symbol"].values())
    check("못 찾은 종목이 있어도 자본이 남김없이 배분된다",
          abs(alloc2 - CASH) <= 2 and len(pf2["by_symbol"]) == 2,
          f"{alloc2:,} / {CASH:,} · 슬리브 {len(pf2['by_symbol'])}개")
    check("못 찾은 종목이 경고로 남는다", bool(pf2.get("warnings")),
          str(pf2.get("warnings"))[:60])
else:
    check("못 찾은 종목이 있어도 자본이 남김없이 배분된다", False, pf2.get("error", ""))

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(f"  결과: {len(PASS)} 통과 / {len(FAIL)} 실패")
for f_ in FAIL:
    print(f"    실패: {f_}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
