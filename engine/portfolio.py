"""
포트폴리오 계층 (L3) — 종목별 백테스트를 하나의 자산곡선으로 합칩니다
---------------------------------------------------------------------

단일 종목 백테스트를 여러 개 돌려 놓고 수익률을 눈으로 비교하는 것은
포트폴리오가 아닙니다. **분산 효과는 합쳐 봐야만 보입니다.**

    종목 A 샤프 0.8 · 종목 B 샤프 0.8  →  합치면?
    상관 +1.0 이면 여전히 0.8, 상관 0.0 이면 1.13, 상관 −0.5 면 1.6.
    개별 숫자만 보면 이 셋을 구분할 방법이 없습니다.


슬리브(sleeve) 모델
-------------------
자본 C 를 종목별로 **미리** 쪼개고, 각 슬리브를 독립적으로 돌린 뒤 합칩니다.

    C = Σ wᵢ·C  →  각 종목을 wᵢ·C 로 백테스트  →  포트폴리오 = Σ Eᵢ(t)

정직하게 말해 두는 한계 셋 — 이걸 모르고 쓰면 결과를 과신하게 됩니다.

  1. **리밸런싱이 없습니다.** 한 슬리브가 두 배가 되어도 비중을 되돌리지
     않습니다. 실제 운용은 주기적으로 되돌리므로, 승자를 계속 태우는
     이 모델보다 보수적입니다.
  2. **슬리브 간 자본 공유가 없습니다.** A 가 현금이 남아도 B 의 진입에
     못 씁니다. 그래서 `insufficient_cash` 거부가 실제보다 많이 나옵니다.
     (거부 장부를 합산해 보여주는 이유입니다.)
  3. **증거금 상계가 없습니다.** 파생 롱·숏을 같이 들어도 증거금이 줄지
     않습니다. 실제보다 자본 효율이 나쁘게 나옵니다.

세 한계 모두 **결과를 나쁜 쪽으로 밀어냅니다.** 그래서 시작점으로 안전합니다.


★ 연율화 기준일을 인자로 받는 이유
----------------------------------
샤프는 `√(연간 관측수)` 로 스케일됩니다. 주식(252)과 24/7 자산(365)을 섞어
놓고 종목마다 다른 값으로 연율화하면 합산 지표가 의미를 잃습니다.
그래서 **자동 추정에 맡기지 않고 명시적으로 하나를 받아** 결과에 기록합니다.
지금 아테나는 KRX/US 주식뿐이라 252 가 맞습니다.
"""

import math

import numpy as np

from engine import validation

TRADING_DAYS_KR = 252


def combine(runs: dict, weights: dict, initial_cash: float,
            periods_per_year: int = TRADING_DAYS_KR) -> dict:
    """종목별 백테스트 결과 → 포트폴리오 자산곡선 + 지표.

    `runs`    {종목키: {"curve": [...], "trades": [...], "benchmark": [...]}}
    `weights` {종목키: 비중}. 합이 1이 아니면 정규화합니다.

    캘린더 합집합 + 직전값 채우기(ffill) 로 정렬합니다. 종목마다 거래일이
    다를 수 있어서입니다 — 한쪽이 휴장인 날 그 슬리브는 **전일 평가금액을
    유지**해야 합니다. 0 으로 두면 그날 포트폴리오가 폭락한 것으로 잡힙니다.
    """
    if not runs:
        return {"ok": False, "error": "합칠 결과가 없습니다."}

    weights = _normalize(weights, list(runs.keys()))
    dates = _union_dates(runs)
    if len(dates) < 2:
        return {"ok": False, "error": "공통 구간이 너무 짧습니다."}

    sleeves, bench_sleeves = {}, {}
    for key, run in runs.items():
        seed = initial_cash * weights[key]
        sleeves[key] = _align(run.get("curve") or [], dates, seed)
        bench_sleeves[key] = _align(run.get("benchmark") or [], dates, seed)

    equity = np.sum(np.vstack(list(sleeves.values())), axis=0)
    bench = np.sum(np.vstack(list(bench_sleeves.values())), axis=0)

    # 포트폴리오 노출 = Σ(슬리브 명목가) / 포트폴리오 자산.
    # 슬리브 노출은 자기 자본 대비이므로 명목가로 되돌린 뒤 더해야 합니다
    # (비율을 그냥 더하면 자본이 큰 슬리브가 과소반영됩니다).
    notional = np.sum(np.vstack([
        _align_field(runs[k].get("curve") or [], dates, "exposure") * sleeves[k]
        for k in runs]), axis=0)
    exposure = np.divide(notional, equity, out=np.zeros_like(notional),
                         where=np.abs(equity) > 1e-9)

    curve = [{"date": d, "equity": round(float(v)), "exposure": round(float(x), 4)}
             for d, v, x in zip(dates, equity, exposure)]
    bench_curve = [{"date": d, "equity": round(float(v))} for d, v in zip(dates, bench)]

    rets = validation.to_returns(curve)
    out = {
        "ok": True,
        "periods_per_year": periods_per_year,
        "symbols": list(runs.keys()),
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "initial_cash": initial_cash,
        "final_equity": curve[-1]["equity"],
        "total_return_pct": round((curve[-1]["equity"] - initial_cash)
                                  / initial_cash * 100, 2),
        "max_drawdown_pct": round(validation.max_drawdown(curve), 2),
        "sharpe": round(validation.sharpe_ratio(rets, periods_per_year), 4),
        "trade_count": sum(len(r.get("trades") or []) for r in runs.values()),
        # 곡선은 화면 전송량 때문에 뒤 250개만 보냅니다. 잘리기 전 길이와
        # 시작 자산을 따로 실어야 "첫날 자본이 맞나" 를 확인할 수 있습니다
        # (안 그러면 curve[0] 이 첫날이 아닌데 첫날인 줄 알고 검사하게 됩니다).
        "n_points": len(curve),
        "first_equity": curve[0]["equity"],
        "start_date": dates[0],
        "curve": curve[-250:],
        "benchmark_curve": bench_curve[-250:],
    }
    out.update(validation.active_metrics(curve, bench_curve, periods_per_year))
    out["diversification"] = diversification(sleeves, weights, periods_per_year)
    out["by_symbol"] = {
        key: {
            "weight": round(weights[key], 4),
            "allocated": round(initial_cash * weights[key]),
            "final_equity": round(float(sleeves[key][-1])),
            "return_pct": round((float(sleeves[key][-1]) - initial_cash * weights[key])
                                / (initial_cash * weights[key]) * 100, 2)
            if weights[key] > 0 else 0.0,
            "trade_count": len(runs[key].get("trades") or []),
            "sharpe": round(validation.sharpe_ratio(
                validation.to_returns(list(sleeves[key])), periods_per_year), 4),
        }
        for key in runs
    }
    return out


def diversification(sleeves: dict, weights: dict, periods_per_year: int) -> dict:
    """분산이 실제로 이득을 줬는가 — 이 계층을 만든 이유 그 자체.

    분산비율 = (비중가중 개별 변동성 합) / (포트폴리오 변동성)

        1.0  전혀 분산되지 않음 (슬리브가 다 같이 움직임)
        >1   분산이 변동성을 줄여 줌. 클수록 좋음

    `zero_corr_reference` 는 **상한이 아니라 기준선** 입니다. 동일 비중·동일
    변동성·무상관일 때의 값(√n)이고, 상관이 음수면 이 값을 넘어갑니다.
    상한으로 읽으면 정상적인 결과를 버그로 오해합니다.

    상관행렬도 같이 돌려줍니다. 분산비율이 1에 가까우면 **종목을 늘려도
    소용없다** 는 뜻이고, 그때는 유니버스를 바꿔야지 비중을 만져도 안 됩니다.
    """
    keys = [k for k in sleeves if weights.get(k, 0) > 0]
    if len(keys) < 2:
        return {"ratio": 1.0, "note": "슬리브가 1개라 분산 효과가 없습니다."}

    rets = {k: validation.to_returns(list(sleeves[k])) for k in keys}
    n = min(len(r) for r in rets.values())
    if n < 20:
        return {"ratio": None, "note": f"표본이 {n}일뿐이라 상관을 추정하지 않습니다."}

    matrix = np.vstack([rets[k][-n:] for k in keys])
    vols = matrix.std(axis=1, ddof=1)
    w = np.array([weights[k] for k in keys], dtype=float)
    w = w / w.sum()

    port = (matrix * w[:, None]).sum(axis=0)
    port_vol = float(port.std(ddof=1))
    weighted_vol = float((w * vols).sum())
    ratio = (weighted_vol / port_vol) if port_vol > 1e-12 else None

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix)
    corr = np.nan_to_num(corr, nan=0.0)

    off = [corr[a, b] for a in range(len(keys)) for b in range(a + 1, len(keys))]
    return {
        "ratio": round(ratio, 3) if ratio else None,
        "zero_corr_reference": round(math.sqrt(len(keys)), 3),
        "avg_correlation": round(float(np.mean(off)), 3) if off else None,
        "correlation": {keys[a]: {keys[b]: round(float(corr[a, b]), 3)
                                  for b in range(len(keys))}
                        for a in range(len(keys))},
        "annualized_vol_pct": round(port_vol * math.sqrt(periods_per_year) * 100, 2),
        "note": _diversification_note(ratio, float(np.mean(off)) if off else 0.0),
    }


def _diversification_note(ratio, avg_corr) -> str:
    if ratio is None:
        return ""
    if ratio < 1.05:
        return (f"평균 상관 {avg_corr:+.2f} — 슬리브들이 사실상 같이 움직입니다. "
                f"종목 수를 늘려도 위험은 줄지 않습니다. 유니버스를 바꾸세요.")
    if ratio < 1.3:
        return f"평균 상관 {avg_corr:+.2f} — 분산 효과가 제한적입니다."
    return (f"평균 상관 {avg_corr:+.2f} — 분산이 변동성을 "
            f"{(1 - 1 / ratio) * 100:.0f}% 줄였습니다.")


# ---------------------------------------------------------------------------
# 내부 도우미
# ---------------------------------------------------------------------------

def _normalize(weights: dict, keys: list) -> dict:
    """비중 정규화. 안 주면 동일가중.

    동일가중을 기본으로 두는 이유: 비중 최적화는 표본 내에서 거의 항상
    이기고 표본 밖에서 거의 항상 집니다(추정오차가 수익률 신호보다 큼).
    동일가중이 이기기 어려운 벤치마크라는 것이 알려진 결과입니다.
    """
    if not weights:
        return {k: 1.0 / len(keys) for k in keys}
    clean = {k: max(0.0, float(weights.get(k, 0.0))) for k in keys}
    total = sum(clean.values())
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / total for k, v in clean.items()}


def _union_dates(runs: dict) -> list:
    """모든 슬리브 날짜의 합집합 (정렬)."""
    dates = set()
    for run in runs.values():
        for point in (run.get("curve") or []):
            dates.add(point["date"])
    return sorted(dates)


def _align(curve: list, dates: list, seed: float) -> np.ndarray:
    """슬리브 곡선을 공통 캘린더에 맞춥니다 — 빈 날은 **직전값 유지**.

    첫 관측 이전 구간은 아직 투입되지 않은 자본이므로 배분액(seed)으로 둡니다.
    0 으로 두면 포트폴리오가 0에서 시작해 첫날 폭등한 것으로 잡힙니다.
    """
    lookup = {p["date"]: float(p.get("equity") or 0.0) for p in (curve or [])}
    out, last = [], float(seed)
    for d in dates:
        if d in lookup:
            last = lookup[d]
        out.append(last)
    return np.asarray(out, dtype=float)


def _align_field(curve: list, dates: list, field: str) -> np.ndarray:
    """`equity` 외 필드(노출 등)를 공통 캘린더에 맞춥니다. 빈 날은 0.

    노출은 자산과 달리 직전값을 유지하면 안 됩니다 — 휴장일에 포지션이
    유지되는 건 맞지만, 그 슬리브의 자산도 같이 고정되므로 곱하면 명목가가
    보존됩니다. 여기서 또 ffill 하면 이중 반영입니다.
    """
    lookup = {p["date"]: float(p.get(field) or 0.0) for p in (curve or [])}
    return np.asarray([lookup.get(d, 0.0) for d in dates], dtype=float)
