# -*- coding: utf-8 -*-
"""시장별 분리 운용 — 한국과 미국을 각자의 예산·상한으로 돌립니다.
=================================================================
예전에는 한 계좌 안에서 한국 종목과 미국 종목이 **같은 지갑**을 놓고
선착순으로 다퉜습니다. 그래서 이런 일이 벌어졌습니다.

    총자산 26만원 · 주문가능 4만원 상태에서 미국 종목 하나가 8주(3.7만원)를
    먼저 채가면, 그 뒤에 온 종목들은 전부 "주문가능금액" 한도에 걸려 1주씩만
    샀습니다. 한국장이 열릴 때쯤에는 쓸 현금이 남아 있지 않았습니다.

여기서 하는 일은 하나입니다 — **시장마다 자기 몫의 돈과 자기 상한을 준다.**

    한국 배분 40% · 미국 배분 40% 이면, 100만원 계좌에서 한국은 40만원,
    미국은 40만원 안에서만 삽니다. 나머지 20% 는 어느 쪽도 손대지 않는
    현금입니다. 꺼진 시장의 몫도 남는 쪽으로 넘기지 않습니다 — 넘기면
    "한국을 잠깐 껐더니 미국에 몰빵됐다"가 됩니다.

배분은 **총자산 기준**입니다. 주문가능현금 기준이 아닙니다
    통합증거금 계좌는 해외 매수대금이 결제일까지 예수금에 남아 있어서,
    주문가능현금이 실제 여력보다 훨씬 작게 보입니다(실측 예수금 132,226원 /
    주문가능 3,325원). 그걸 기준으로 나누면 배분이 시시각각 요동칩니다.
    총자산으로 몫을 정하고, 실제로 낼 수 있는 돈은 그 몫과 주문가능현금
    중 **작은 쪽**으로 다시 조입니다.

이 모듈은 판단만 합니다 — 주문·기록은 engine/autotrade.py 가 합니다
    engine/split.py, engine/rotation.py 와 같은 규약입니다. 부작용이 없어야
    테스트할 수 있고, 왜 그렇게 판단했는지를 그대로 남길 수 있습니다.
"""

from engine.instruments import MARKET_US

KR = "KR"
US = "US"
ALL = "ALL"
SCOPES = (KR, US)

LABELS = {KR: "한국", US: "미국", ALL: "전체"}
SUFFIX = {KR: "kr", US: "us"}

# 시장별로 덮어쓸 수 있는 전역 설정 키. 0(또는 빈 값)이면 전역값을 그대로 씁니다
# — "안 정했다"와 "0으로 정했다"를 가르기 위해서입니다. 종목당 비중 0% 를
# 원하는 사람은 그 시장을 끄면 됩니다.
#
# max_positions 는 여기에 없습니다. 리스크 엔진의 종목 수 상한은 **계좌 전체**
# 보유 수를 세는데, 거기에 한 시장의 상한(1)을 꽂으면 미국을 한 종목 들고 있는
# 동안 한국이 통째로 막힙니다. 시장별 상한은 gate() 가 그 시장의 보유 수만
# 세어서 따로 겁니다.
OVERRIDABLE = ("risk_per_trade_pct", "position_pct",
               "max_order_krw", "min_order_krw")

DEFAULTS = {
    # 나눠서 굴릴 것인가. 끄면 예전처럼 한 지갑을 함께 씁니다.
    "market_split": True,
    # 시장별 스위치 — 끄면 그 시장은 **신규 진입만** 멈춥니다.
    # 청산·손절은 계속 돕니다(가진 것을 못 파는 상태를 만들면 안 됩니다).
    "kr_enabled": True,
    "us_enabled": True,
    # 배분 (총자산 대비 %). 합이 100 을 넘으면 비례 축소합니다.
    "kr_alloc_pct": 50.0,
    "us_alloc_pct": 50.0,
    # 시장별 상한 — 기본 1종목. 소액 계좌에서 종목 수를 늘리면 종목당 금액이
    # 1주 값 밑으로 떨어져 결국 "전부 1주씩"이 됩니다.
    "kr_max_positions": 1,
    "us_max_positions": 1,
    # 0 이면 전역 설정을 그대로 씁니다 (OVERRIDABLE 참고)
    "kr_risk_per_trade_pct": 0.0,
    "us_risk_per_trade_pct": 0.0,
    "kr_position_pct": 0.0,
    "us_position_pct": 0.0,
}


# ---------------------------------------------------------------------------
# 이 종목은 어느 시장인가
# ---------------------------------------------------------------------------

def _attr(target, name: str, default=None):
    """객체든 dict 든 같은 이름으로 읽습니다.

    포지션은 두 모양으로 옵니다 — 엔진 회전은 Position 객체를, 콘솔 스냅샷은
    이미 to_dict() 된 값을 넘깁니다. 부르는 쪽마다 변환하게 하면 한쪽이
    빠졌을 때 화면과 엔진의 시장 구분이 조용히 갈립니다.
    """
    if isinstance(target, dict):
        return target.get(name, default)
    return getattr(target, name, default)


def scope_of(target) -> str:
    """Instrument · Position · 포지션 dict · 종목코드 문자열 → "KR" | "US".

    Instrument 는 market 을 들고 있으니 그대로 씁니다. 포지션·주문 기록처럼
    코드만 있는 경우에는 코드 모양으로 가릅니다 — 한국은 6자리 숫자이고,
    파생 단축코드(101S3000 …)도 숫자로 시작합니다. 미국 티커는 영문입니다.
    종목을 다시 해석(resolve)하지 않는 이유: 이 함수는 화면 집계에서 종목당
    여러 번 불리는데, 해석은 캐시가 비면 외부 조회로 이어집니다.
    """
    market = _attr(target, "market")
    if market:
        return US if market == MARKET_US else KR
    key = str((_attr(target, "key", target) if not isinstance(target, str)
               else target) or "").strip()
    if not key:
        return KR
    return KR if key[0].isdigit() else US


def label(scope: str) -> str:
    return LABELS.get(scope, scope)


# ---------------------------------------------------------------------------
# 설정 읽기
# ---------------------------------------------------------------------------

def enabled(cfg: dict) -> bool:
    """시장을 나눠서 굴리는 중인가."""
    return bool((cfg or {}).get("market_split", DEFAULTS["market_split"]))


def _f(cfg: dict, key: str, fallback: float) -> float:
    try:
        return float(cfg.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)


def alloc_pcts(cfg: dict) -> dict:
    """시장별 배분 비율 (합이 100 을 넘지 않도록 조인 값).

    합이 100 을 넘으면 **비례 축소**합니다. 앞에서부터 잘라내지 않는 이유는,
    60/60 을 넣었을 때 "한국 60 · 미국 40" 처럼 한쪽 편을 드는 대신 사람이
    의도한 비율(1:1)을 지켜주기 위해서입니다.
    """
    cfg = cfg or {}
    raw = {s: max(0.0, min(_f(cfg, f"{SUFFIX[s]}_alloc_pct",
                              DEFAULTS[f"{SUFFIX[s]}_alloc_pct"]), 100.0))
           for s in SCOPES}
    total = sum(raw.values())
    if total > 100.0 and total > 0:
        raw = {s: v * 100.0 / total for s, v in raw.items()}
    return raw


def market_on(cfg: dict, scope: str) -> bool:
    """그 시장의 신규 진입 스위치. 분리를 끄면 항상 켜진 것으로 봅니다."""
    if not enabled(cfg):
        return True
    key = f"{SUFFIX.get(scope, '')}_enabled"
    return bool((cfg or {}).get(key, DEFAULTS.get(key, True)))


def max_positions(cfg: dict, scope: str) -> int:
    key = f"{SUFFIX.get(scope, '')}_max_positions"
    try:
        value = int((cfg or {}).get(key, DEFAULTS.get(key, 1)))
    except (TypeError, ValueError):
        value = 1
    return max(0, value)


def global_cfg(cfg: dict) -> dict:
    """계좌 전체에 적용할 설정 사본.

    나눠서 굴리는 동안 계좌 전체 종목 상한은 **두 시장 상한의 합**입니다.
    이걸 맞춰두지 않으면 리스크 엔진의 전체 상한(기본 1종목)이 시장 상한보다
    먼저 걸려서, 미국을 한 종목 들고 있는 동안 한국이 통째로 막힙니다.
    엔진 회전과 화면이 같은 함수를 통과해야 두 곳의 숫자가 갈리지 않습니다.
    """
    cfg = cfg or {}
    if not enabled(cfg):
        return cfg
    return {**cfg, "max_positions": sum(max_positions(cfg, s) for s in SCOPES)}


def market_cfg(cfg: dict, scope: str) -> dict:
    """이 시장에 적용할 설정 **사본**.

    시장별로 값을 정해 둔 항목만 덮습니다. 저장된 설정을 건드리지 않고 사본을
    만드는 이유는 engine/ensemble.py 의 변동성 스케일과 같습니다 — 이번 판단에만
    쓰이고 백테스트나 다음 회전으로 흘러들면 안 됩니다.
    """
    cfg = cfg or {}
    if not enabled(cfg) or scope not in SCOPES:
        return cfg
    out = dict(cfg)
    prefix = SUFFIX[scope]
    for key in OVERRIDABLE:
        try:
            value = float(cfg.get(f"{prefix}_{key}"))
        except (TypeError, ValueError):
            continue
        if value > 0:
            out[key] = value
    return global_cfg(out)


# ---------------------------------------------------------------------------
# 예산
# ---------------------------------------------------------------------------

def _scope_positions(scope: str, positions: list) -> list:
    return [p for p in (positions or []) if scope_of(p) == scope]


def market_value(scope: str, positions: list) -> float:
    """그 시장이 지금 들고 있는 평가금액(원화)."""
    return float(sum(abs(_attr(p, "market_value", 0) or 0)
                     for p in _scope_positions(scope, positions)))


def budget(cfg: dict, scope: str, account: dict, positions: list) -> dict:
    """이 시장이 쓸 수 있는 돈.

        예산     = 총자산 × 배분%
        사용     = 그 시장 보유 평가금액
        여유     = 예산 − 사용
        주문가능 = min(계좌 주문가능현금, 여유)

    분리가 꺼져 있으면 계좌 전체를 그대로 돌려줍니다 — 부르는 쪽이 분기하지
    않아도 되도록 형태를 맞춥니다.
    """
    cfg = cfg or {}
    account = account or {}
    total = float(account.get("total_value") or 0)
    cash = float(account.get("available_cash") or 0)
    used = market_value(scope, positions)
    rows = _scope_positions(scope, positions)

    if not enabled(cfg) or scope not in SCOPES:
        return {"scope": scope, "split": False, "alloc_pct": 100.0,
                "budget_krw": total, "used_krw": used,
                "room_krw": max(total - used, 0.0), "cash_krw": cash,
                "total_krw": total, "positions": len(rows),
                "max_positions": int(cfg.get("max_positions", 1) or 0),
                "enabled": True}

    pct = alloc_pcts(cfg).get(scope, 0.0)
    size = total * pct / 100.0
    room = max(size - used, 0.0)
    return {
        "scope": scope, "split": True, "alloc_pct": pct,
        "budget_krw": size, "used_krw": used, "room_krw": room,
        "cash_krw": min(cash, room), "total_krw": total,
        "positions": len(rows), "max_positions": max_positions(cfg, scope),
        "enabled": market_on(cfg, scope),
    }


def scoped_account(cfg: dict, scope: str, account: dict, positions: list) -> dict:
    """수량 계산(strategy.plan_entry)에 넘길 **그 시장만의 계좌**.

    총자산을 시장 예산으로 바꿔 넘기는 것이 핵심입니다. plan_entry 의 세 한도
    (1회 위험 예산 · 종목당 비중 · 주문가능금액)가 전부 총자산에서 나오므로,
    총자산 하나만 바꾸면 세 한도가 한꺼번에 그 시장 몫으로 줄어듭니다.
    """
    if not enabled(cfg) or scope not in SCOPES:
        return account or {}
    book = budget(cfg, scope, account, positions)
    return {**(account or {}),
            "total_value": book["budget_krw"],
            "available_cash": book["cash_krw"],
            # 원래 계좌 크기 — 로그·화면이 "계좌 전체 대비 몇 %"를 말할 때 씁니다
            "account_total_value": float((account or {}).get("total_value") or 0)}


def gate(cfg: dict, scope: str, account: dict, positions: list,
         key: str = "", entry: bool = True) -> str:
    """신규 진입을 막을 이유. 없으면 빈 문자열.

    청산에는 쓰지 않습니다 — 배분을 넘겼다고 못 팔게 하면 그게 사고입니다.
    """
    if not entry or not enabled(cfg) or scope not in SCOPES:
        return ""
    name = label(scope)
    if not market_on(cfg, scope):
        return f"{name} 자동매매가 꺼져 있습니다."

    book = budget(cfg, scope, account, positions)
    if book["alloc_pct"] <= 0:
        return f"{name} 배분이 0% 입니다 — 배분을 올려야 살 수 있습니다."

    cap = book["max_positions"]
    holding = {_attr(p, "key") for p in _scope_positions(scope, positions)}
    if cap > 0 and key not in holding and len(holding) >= cap:
        return f"{name} 보유 종목 상한 {cap}개에 도달했습니다."

    if book["room_krw"] <= 0:
        return (f"{name} 배분 예산 {book['budget_krw']:,.0f}원"
                f"(총자산의 {book['alloc_pct']:g}%)을 이미 채웠습니다.")
    return ""


def overview(cfg: dict, account: dict, positions: list) -> list[dict]:
    """화면이 그대로 그릴 수 있는 시장별 요약."""
    out = []
    for scope in SCOPES:
        book = budget(cfg, scope, account, positions)
        rows = _scope_positions(scope, positions)
        out.append({
            **book,
            "label": label(scope),
            "unrealized_krw": float(sum(_attr(p, "unrealized_pnl", 0) or 0
                                        for p in rows)),
            "symbols": [_attr(p, "key") for p in rows],
            "blocked": gate(cfg, scope, account, positions),
        })
    return out


def describe(cfg: dict) -> list[dict]:
    """설정 요약 (한도 표에 함께 띄웁니다)."""
    if not enabled(cfg):
        return [{"key": "market_split", "label": "한국·미국 분리 운용",
                 "value": "끔 (한 지갑을 함께 씁니다)"}]
    pcts = alloc_pcts(cfg)
    rows = [{"key": "market_split", "label": "한국·미국 분리 운용", "value": "켬"}]
    for scope in SCOPES:
        state = "켬" if market_on(cfg, scope) else "끔"
        rows.append({
            "key": f"{SUFFIX[scope]}_alloc_pct",
            "label": f"{label(scope)} 배분",
            "value": (f"{pcts[scope]:g}%   최대 {max_positions(cfg, scope)}종목   "
                      f"신규 진입 {state}"),
        })
    left = 100.0 - sum(pcts.values())
    if left > 0.01:
        rows.append({"key": "cash_reserve", "label": "어느 쪽도 쓰지 않는 현금",
                     "value": f"{left:g}%"})
    return rows
