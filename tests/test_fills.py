# -*- coding: utf-8 -*-
"""체결 현실성 QA — engine/fills.py 의 판정 규칙이 실제로 그렇게 도는가.

각 검사는 **그 규칙이 없으면 백테스트가 어떻게 낙관적으로 틀리는지** 를
한 줄로 붙여 놓았습니다. 규칙을 지우고 싶어질 때 그 줄을 읽으세요.

    python tests/test_fills.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from engine import fills
from engine.instruments import ETF, STOCK, Instrument
from models import ResolvedSymbol

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


def section(title):
    print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72)


def stock(key="FILL01", asset=STOCK, market="KOSPI") -> Instrument:
    return Instrument(
        key=key, name="체결테스트", asset_class=asset, market=market,
        currency="KRW", multiplier=1.0, margin_rate=1.0, shortable=False,
        symbol=ResolvedSymbol(key=key, name="체결테스트", market=market,
                              yahoo_symbol=key, currency="KRW"),
    )


def bar(o, h, lo, c, v=1_000_000):
    return pd.Series({"open": float(o), "high": float(h), "low": float(lo),
                      "close": float(c), "volume": v})


INST = stock()

# ---------------------------------------------------------------------------
section("1. 시장가는 다음 봉 시가에 체결된다")
# ---------------------------------------------------------------------------
# 없으면: 종가를 보고 판단해서 그 종가에 체결 = 실전 불가능한 가정.
#         벡터라이즈 백테스터가 대개 이 형태이고, 성과가 통째로 부풀려집니다.

f, why = fills.market_fill(INST, bar(10_200, 10_400, 10_100, 10_300),
                           fills.BUY, slippage=0.0)
check("매수는 그 봉의 시가에 체결", f is not None and f.price == 10_200,
      f"체결가 {f.price if f else why}")
check("체결 사유가 기록됨", f is not None and f.reason == fills.OPEN)

f, _ = fills.market_fill(INST, bar(10_200, 10_400, 10_100, 10_300),
                         fills.BUY, slippage=0.001)
check("슬리피지는 매수에 불리하게(+) 붙는다", f.price > 10_200, f"{f.price:,.0f}")

f, _ = fills.market_fill(INST, bar(10_200, 10_400, 10_100, 10_300),
                         fills.SELL, slippage=0.001)
check("슬리피지는 매도에 불리하게(−) 붙는다", f.price < 10_200, f"{f.price:,.0f}")

# ---------------------------------------------------------------------------
section("2. 손절·익절은 봉 안(고가/저가)에서 판정된다")
# ---------------------------------------------------------------------------
# 없으면: 장중에 손절선을 뚫었다가 종가에 회복한 봉이 전부 '무손실'로 기록됩니다.
#         실제로는 그 자리에서 잘렸습니다.

b = bar(o=10_000, h=10_300, lo=8_900, c=10_100)     # 저가만 손절선 아래
f = fills.protective_fill(INST, b, fills.LONG, stop=9_000, target=None)
check("종가가 회복해도 저가가 손절선을 뚫었으면 손절", f is not None,
      f"체결가 {f.price if f else '체결 없음'}")
check("손절 체결가는 손절선 근처", f is not None and 8_900 <= f.price <= 9_000)

b = bar(o=10_000, h=11_200, lo=9_800, c=10_050)     # 고가만 목표가 위
f = fills.protective_fill(INST, b, fills.LONG, stop=9_000, target=11_000)
check("고가가 목표가에 닿으면 익절", f is not None and f.reason == fills.TARGET)

b = bar(o=10_000, h=10_300, lo=9_700, c=10_100)     # 둘 다 미도달
check("아무것도 안 닿으면 체결 없음",
      fills.protective_fill(INST, b, fills.LONG, 9_000, 11_000) is None)

# ---------------------------------------------------------------------------
section("3. 같은 봉에 손절·익절이 모두 닿으면 손절 우선")
# ---------------------------------------------------------------------------
# 없으면: 봉 내부 경로를 모르면서 유리한 쪽을 골라 없는 승률이 생깁니다.

b = bar(o=10_000, h=11_500, lo=8_500, c=10_000)     # 둘 다 관통한 큰 봉
f = fills.protective_fill(INST, b, fills.LONG, stop=9_000, target=11_000)
check("롱: 손절이 이긴다 (비관적 판정)",
      f is not None and f.reason == fills.STOP, f"사유 {f.reason if f else '—'}")

f = fills.protective_fill(INST, b, fills.SHORT, stop=11_000, target=9_000)
check("숏: 손절이 이긴다", f is not None and f.reason == fills.STOP)

# ---------------------------------------------------------------------------
section("4. 갭으로 손절선을 관통하면 체결가는 시가")
# ---------------------------------------------------------------------------
# 없으면: 손절 9,000원짜리가 7,000원에 갭 하락 시작해도 9,000원에 판 것으로
#         기록됩니다. 갭 위험이 백테스트에서 통째로 사라집니다.

b = bar(o=7_000, h=7_400, lo=6_800, c=7_100)
f = fills.protective_fill(INST, b, fills.LONG, stop=9_000, target=11_000)
check("갭 하락 시 손절값이 아니라 시가에 체결",
      f is not None and f.price <= 7_000, f"체결가 {f.price:,.0f} (손절선 9,000)")
check("갭 관통이 사유로 남는다",
      f is not None and f.reason == fills.GAP_THROUGH_STOP and f.gapped)

b = bar(o=13_000, h=13_400, lo=12_800, c=13_100)
f = fills.protective_fill(INST, b, fills.SHORT, stop=11_000, target=9_000)
check("숏도 동일 — 갭 상승 시 시가 체결",
      f is not None and f.reason == fills.GAP_THROUGH_STOP and f.price >= 13_000,
      f"체결가 {f.price:,.0f}")

b = bar(o=12_000, h=12_400, lo=11_800, c=12_100)
f = fills.protective_fill(INST, b, fills.LONG, stop=9_000, target=11_000)
check("유리한 갭(목표가 관통)도 시가 체결",
      f is not None and f.reason == fills.GAP_THROUGH_TARGET and f.price >= 11_000,
      f"체결가 {f.price:,.0f} (목표 11,000)")

# ---------------------------------------------------------------------------
section("5. 체결가는 호가 격자 위에, 반올림은 나에게 불리한 쪽으로")
# ---------------------------------------------------------------------------
# 없으면: 절반은 유리한 쪽으로 떨어져 수천 번 반복되며 성과가 부풀려집니다.

# 국내 주식 10,000원대는 호가 10원 단위
buy = fills.snap_adverse(INST, 10_003.0, fills.BUY)
sell = fills.snap_adverse(INST, 10_003.0, fills.SELL)
check("매수는 위쪽 호가로 올림", buy >= 10_003 and buy % 10 == 0, f"{buy:,.0f}")
check("매도는 아래쪽 호가로 버림", sell <= 10_003 and sell % 10 == 0, f"{sell:,.0f}")
check("매수가 매도보다 불리(비쌈)", buy > sell, f"{buy:,.0f} vs {sell:,.0f}")

exact = fills.snap_adverse(INST, 10_000.0, fills.BUY)
check("이미 호가에 맞으면 한 틱 밀리지 않는다", exact == 10_000, f"{exact:,.0f}")

# ---------------------------------------------------------------------------
section("6. 가격제한폭에 닿은 방향으로는 체결되지 않는다")
# ---------------------------------------------------------------------------
# 없으면: 백테스트가 상한가에서 자유롭게 사고 하한가에서 깔끔하게 손절합니다.
#         실전에서 가장 크게 깨지는 두 상황입니다.

prev = 10_000.0
check("상한가 인식", fills.limit_state(INST, prev, 13_000.0) == fills.LIMIT_UP)
check("하한가 인식", fills.limit_state(INST, prev, 7_000.0) == fills.LIMIT_DOWN)
check("평상시는 제한 없음", fills.limit_state(INST, prev, 10_500.0) == "")

f, why = fills.market_fill(INST, bar(13_000, 13_000, 13_000, 13_000),
                           fills.BUY, prev_close=prev)
check("상한가에서는 매수 체결 실패", f is None and why == fills.LIMIT_UP, f"사유 {why}")

f, why = fills.market_fill(INST, bar(13_000, 13_000, 13_000, 13_000),
                           fills.SELL, prev_close=prev)
check("상한가에서 매도는 가능", f is not None)

f = fills.protective_fill(INST, bar(7_000, 7_000, 7_000, 7_000), fills.LONG,
                          stop=9_500, target=None, prev_close=prev)
check("하한가에 갇히면 손절도 못 나간다 (다음 봉으로)", f is None)

# 규칙이 적용되는 범위 — 국내 주식/ETF 뿐입니다.
# 미국은 일일 제한폭 대신 변동성완화장치(LULD)라 같은 판정이 아니고,
# 파생은 제한폭이 단계적으로 확대되므로 ±30% 한 줄로 표현할 수 없습니다.
from models import MARKET_US

check("미국 주식에는 ±30% 제한폭을 적용하지 않는다",
      fills.limit_state(stock(key="AAPL", market=MARKET_US), prev, 13_000.0) == "")
check("ETF 에는 적용한다 (국내 상장)",
      fills.limit_state(stock(key="069500", asset=ETF), prev, 13_000.0)
      == fills.LIMIT_UP)

# ---------------------------------------------------------------------------
section("7. 거부 장부 — 사라진 신호가 숫자로 남는가")
# ---------------------------------------------------------------------------
log = fills.RejectionLog()
log.add(10, "2026-01-05", fills.INSUFFICIENT_CASH, "long", "필요 1,000 > 보유 500")
log.add(11, "2026-01-06", fills.INSUFFICIENT_CASH, "long")
log.add(12, "2026-01-07", fills.LIMIT_UP, "long")
s = log.summary()
check("총 건수 집계", s["total"] == 3, f"{s['total']}건")
check("사유별 집계", s["by_reason"][fills.INSUFFICIENT_CASH]["count"] == 2)
check("사유에 사람이 읽을 라벨이 붙는다",
      s["by_reason"][fills.LIMIT_UP]["label"] == "상한가 — 매수 불가")
check("최근 표본을 남긴다", len(s["recent"]) == 3)

# ---------------------------------------------------------------------------
section("8. 백테스트 루프 통합 — 규칙이 실제로 적용되는가")
# ---------------------------------------------------------------------------
from engine import autotrade, strategy as strat


def synth(n=200, seed=11):
    """상승 추세 합성 일봉 — 진입이 실제로 일어나게."""
    import numpy as np
    from datetime import datetime, timedelta
    rng = np.random.default_rng(seed)
    closes = 10_000 * np.exp(np.cumsum(rng.normal(0.005, 0.012, n)))
    rows = []
    for i, c in enumerate(closes):
        hi = c * (1 + abs(rng.normal(0, 0.006)))
        lo = c * (1 - abs(rng.normal(0, 0.006)))
        rows.append({"date": datetime(2025, 1, 1) + timedelta(days=i),
                     "open": (hi + lo) / 2, "high": max(hi, c),
                     "low": min(lo, c), "close": c, "volume": 1_000_000})
    return pd.DataFrame(rows).set_index("date")


bars = synth()
cfg = {**__import__("storage.autotrade", fromlist=["x"]).DEFAULT_CONFIG,
       "entry_score": 0.2, "intraday_weight": 0.0, "use_news": False}
run = autotrade._run_backtest(INST, bars, cfg, 10_000_000)

check("백테스트가 돈다", len(run["curve"]) > 50, f"{len(run['curve'])}포인트")
check("거부 장부가 결과에 실린다", "rejections" in run)
check("자산 곡선 길이 = 봉 수 − 워밍업",
      len(run["curve"]) == len(bars) - run["start_index"],
      f"{len(run['curve'])} vs {len(bars) - run['start_index']}")

if run["trades"]:
    check("모든 매매에 체결 사유가 붙는다",
          all(t.get("fill") for t in run["trades"]),
          str(sorted({t["fill"] for t in run["trades"]})))
    entries = {t["entry"] for t in run["trades"]}
    closes_set = set(bars["close"].round(4))
    check("진입가가 그 봉의 종가가 아니다 (= 종가 체결이 아님)",
          not entries.issubset(closes_set),
          f"매매 {len(run['trades'])}건")
else:
    check("매매가 최소 1건은 발생", False, "0건 — 신호 설정을 확인하세요")

# 구간 끝 미청산 포지션 정산 (finalize)
check("구간 끝에 미청산 포지션이 남지 않는다",
      any("구간 종료" in t["reason"] for t in run["trades"])
      or all(t["exit_at"] < str(bars.index[-1])[:10] for t in run["trades"]),
      "마지막 포지션이 매매 통계에 포함됨")

# ★ 가장 강한 불변식 — 이게 깨지면 두 숫자 중 하나는 반드시 거짓말입니다.
#   자산 곡선이 미실현 손익을 포함하는데 매매 통계는 미청산 포지션을 빼거나,
#   진입 수수료가 현금에서만 빠지고 매매 손익에는 안 잡히면 여기서 걸립니다.
pnl_sum = sum(t["pnl"] for t in run["trades"])
equity_delta = run["curve"][-1]["equity"] - 10_000_000
gap = abs(pnl_sum - equity_delta)
check("매매 손익 합계 = 최종자산 − 시작자산 (진입비용·미청산 포함)",
      gap <= max(len(run["trades"]) * 2, 10),
      f"매매합 {pnl_sum:,.0f} vs 자산증감 {equity_delta:,.0f} (차이 {gap:,.0f})")

check("매매 기록에 수수료·세금이 분리되어 남는다",
      all("fees" in t and "taxes" in t for t in run["trades"]),
      f"예: 수수료 {run['trades'][0]['fees']:,.0f} / 세금 "
      f"{run['trades'][0]['taxes']:,.0f}" if run["trades"] else "")

# 세율은 봉 날짜 기준 (2026-01-01 인상 전후가 달라야 함)
from engine import markets
old = markets.sell_tax_rate(pd.Timestamp("2024-06-01"), "KOSPI")
new = markets.sell_tax_rate(pd.Timestamp("2026-06-01"), "KOSPI")
check("증권거래세가 봉 날짜에 따라 다르다 (백테스트가 과거 세율을 쓴다)",
      old != new, f"2024 {old:.4%} vs 2026 {new:.4%}")

# ---------------------------------------------------------------------------
print()
print("=" * 72)
print(f"  결과: {len(PASS)} 통과 / {len(FAIL)} 실패")
for f_ in FAIL:
    print(f"    실패: {f_}")
print("=" * 72)
sys.exit(1 if FAIL else 0)
