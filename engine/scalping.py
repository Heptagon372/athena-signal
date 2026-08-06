"""
페니주식 초단타 (Penny Scalping) — 틱 단위 진입/청산
====================================================

이 전략은 이 프로그램에서 가장 위험합니다.
저가주는 유동성이 얇고 호가 스프레드가 넓으며, 작전·펌프앤덤프의 주무대입니다.
초단타는 여기에 회전율을 곱합니다.

무엇이 바뀌었나 (이전 버전과의 차이)
    · 규모 한도를 **비율이 아니라 금액**으로 잡습니다.
      `budget_krw` 하나가 이 전략이 쓸 수 있는 전부입니다. 종목당 비중(%),
      동시 보유 개수는 여기서 파생되는 값이라 따로 두지 않습니다.
      (총자산 1,000원에 "비중 5%" 를 곱하면 50원이 되어 아무것도 못 삽니다.
       소액에서 비율 기반 예산은 그냥 고장납니다)
    · 손절·익절을 **% 가 아니라 틱**으로 잡습니다.
      100원짜리에서 "손절 1.5%" 는 1.5원인데 호가는 1원 단위입니다.
      입력한 값이 반올림돼서 그대로 먹지 않습니다. 틱은 호가창과 1:1 입니다.
    · 하루 매매 횟수 제한을 없앴습니다. 초단타에 "10회" 는 앞뒤가 맞지 않습니다.
      대신 그날을 끊는 것은 **일일 손실 금액** 하나입니다.
    · 추적 종목 수를 사용자가 정하지 않습니다. 타점 조건(거래량 급증·회전율·
      변동폭)을 통과한 종목을 전부 추적하고, 실시간 시세 구독 한도로만 잘립니다.

왜 여전히 위험한지 숫자로
    · 100원짜리 종목의 호가단위는 1원 = **1%**. 사는 순간 스프레드로 마이너스.
    · 저가주는 하루 ±30% 가 흔합니다. 손절이 늦으면 한 번에 다 사라집니다.
    · 유동성이 얇으면 팔고 싶을 때 사줄 사람이 없습니다(진짜 위험은 이것입니다).

그래서 남겨둔 하드 한도는 두 종류뿐입니다.
    1) **못 빠져나오는 상황**을 막는 것 (유동성, 상·하한가, 마감 직전 진입)
    2) **정한 금액 밖으로 나가는 것**을 막는 것 (budget_krw, daily_loss_krw)
    나머지(비중·동시보유·매매횟수)는 사용자의 선택으로 돌려놨습니다.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from engine import feed
from engine.instruments import ETF, STOCK, Instrument

# ===========================================================================
# 거래 비용 — 틱 손익분기의 근거
# ===========================================================================
# 이 두 값이 "몇 틱을 먹어야 본전인가" 를 결정합니다. 증권사·시장마다 다르므로
# 보수적인 쪽(비싼 쪽)으로 잡아둡니다. 실제보다 싸게 잡으면 손익분기를 낮게
# 계산해서, 사실은 지는 자리에 들어가게 됩니다.
FEE_RATE = 0.00015      # 위탁수수료 (편도, 온라인 기준)
TAX_RATE = 0.00150      # 증권거래세 + 농어촌특별세 (매도 시에만)

# KRX 호가단위 (2023.1.25 개편 이후 · 코스피/코스닥 동일)
# (이 가격 미만이면, 호가단위)
KRX_TICK_TABLE = [
    (2_000, 1),
    (5_000, 5),
    (20_000, 10),
    (50_000, 50),
    (200_000, 100),
    (500_000, 500),
]
KRX_TICK_TOP = 1_000        # 500,000원 이상

US_TICK = 0.01              # 미국 주식은 $1 이상이면 1센트 고정


# ===========================================================================
# 하드 한도 — 설정으로 완화할 수 없습니다
# ===========================================================================
# 여기 남은 것은 전부 "못 빠져나오는 상황" 또는 "정한 금액 밖으로 나가는 것" 을
# 막는 항목입니다. 취향 문제는 하나도 없습니다.
HARD_LIMITS = {
    # --- 종목 자격 (= 팔 수 있는가) ---
    "kr_min_price": 100,             # 이보다 싸면 호가단위 대비 노이즈가 가격을 지배
    "kr_max_price": 3_000,           # 이보다 비싸면 '페니'가 아닙니다
    "us_min_price": 1.0,             # $1 미만은 상장폐지 경고 구간(나스닥 규정)
    "us_max_price": 10.0,
    "min_trading_value_krw": 1_000_000_000,   # 일 거래대금 10억 미만 = 못 빠져나옴
    "max_spread_ticks": 3,           # 스프레드가 3틱을 넘으면 왕복비용이 수익을 삼킴
    "limit_zone_pct": 28.0,          # 상·하한가(±30%) 근처는 거래가 멈춰 못 팝니다
    "max_order_impact_pct": 0.5,     # 하루 거래대금의 0.5% 넘는 주문은 내가 호가를 밈

    # --- 금액 (= 정한 돈 밖으로 안 나감) ---
    "max_budget_krw": 10_000_000,    # 이 전략에 넣을 수 있는 총액의 상한

    # --- 청산 (= 물리지 않음) ---
    "max_stop_loss_ticks": 5,        # 손절을 이보다 넓게 잡으면 초단타가 아닙니다
    "max_hold_sec": 600,             # 10분을 넘기면 그건 초단타가 아니라 물린 것
    "min_reentry_cooldown_sec": 5,   # 같은 종목 연속 타격 방지(자기 호가끼리 충돌)

    # --- 시세 신선도 ---
    # 몇 틱 승부의 전제는 "지금 가격을 보고 있다" 입니다. 해외 실시간은 계정에
    # 따라 지연 시세가 오는데, 지연된 값으로 3틱을 재면 이미 지나간 자리에
    # 주문을 냅니다. 이건 설정으로 풀 수 있는 값이 아닙니다.
    "max_quote_delay_sec": 5,

    # --- 시간대 ---
    "no_entry_first_minutes": 5,     # 개장 직후 5분은 변동성이 비이성적
    "no_entry_last_minutes": 20,     # 마감 20분 전부터는 신규 진입 금지(못 빠져나옴)

    # --- 인프라 ---
    # 시세 구독 한도. 사용자가 정하는 값이 아니라 KIS 실시간체결 WebSocket 이
    # 계정당 41건까지만 허용하기 때문에 생기는 물리적 상한입니다.
    "max_tracked": 40,
    "min_universe_refresh_sec": 10,  # REST 순위 API 초당 한도 보호 (EGW00201)
}

# 기본 설정 (사용자가 조정 가능 — 단 위 한도 안에서만)
DEFAULT_SCALP = {
    "enabled": False,
    "markets": ["KR"],

    # --- 규모: 이 두 개가 전부입니다 ---
    "budget_krw": 100_000,           # 이 전략이 쓸 수 있는 총액
    "daily_loss_krw": 10_000,        # 오늘 이만큼 잃으면 종료

    # --- 대상 가격대 ---
    "kr_price_range": [100, 3_000],
    "us_price_range": [1.0, 10.0],

    # --- 진입/청산 (틱 단위) ---
    "take_profit_ticks": 3,          # 목표 도달 즉시 청산
    "stop_loss_ticks": 2,
    "max_hold_sec": 180,
    "reentry_cooldown_sec": 30,
    "min_net_ticks": 1,              # 비용 제하고 최소 이만큼 남아야 진입
    "entry_score": 0.55,
    "entry_order_type": "limit_bid",  # limit_bid(매수호가 지정가) | market(시장가)

    # --- 타점 (스크리너) ---
    # 초단타는 **자기 매매 대상을 따로 들고 있습니다**. 일반 자동매매의
    # universe 와 섞으면, 초단타 스크리너가 15초마다 갈아끼우는 동전주가
    # 자동매매 콘솔의 매매 대상을 통째로 밀어냅니다.
    "auto_universe": True,           # 조건에 맞는 종목을 알아서 찾아 교체
    "universe": [],                  # 초단타 전용 매매 대상 (스크리너가 채웁니다)
    "pinned": [],                    # 내가 직접 지정한 종목 — 갱신해도 안 빠집니다
    "rank_basis": "vol_increase",    # vol_increase | turnover | value | volume
    "min_vol_increase_pct": 200.0,   # 거래량증가율 (평소 대비 +200% = 3배)
    "min_turnover_pct": 1.0,         # 거래회전율 (상장주식수 대비 거래량)
    "min_range_pct": 3.0,            # 당일 고저 변동폭
    "universe_refresh_sec": 15,
}

LONG = "long"
FLAT = "flat"


# ===========================================================================
# 호가단위와 틱 계산
# ===========================================================================

def tick_size(price: float, market: str = "KR") -> float:
    """이 가격대의 호가단위.

    초단타의 모든 계산이 여기서 출발합니다. 호가단위를 모르면 "3틱 익절" 이
    얼마인지 알 수 없고, 손절을 호가 사이에 걸어 영원히 체결되지 않게 됩니다.
    """
    if market == "US":
        return US_TICK
    price = float(price or 0)
    for ceiling, unit in KRX_TICK_TABLE:
        if price < ceiling:
            return float(unit)
    return float(KRX_TICK_TOP)


def ticks_to_price(price: float, ticks: int, market: str = "KR") -> float:
    """현재가에서 n틱 떨어진 가격 (호가단위에 정렬).

    단순히 `price + ticks * tick_size` 로 계산하면 안 됩니다. 가격대 경계를
    넘어가면 호가단위가 바뀌기 때문입니다 (1,998원에서 +3틱은 2,001원이
    아니라 2,005원입니다 — 2,000원부터 5원 단위).
    """
    out = float(price)
    step = 1 if ticks >= 0 else -1
    for _ in range(abs(int(ticks))):
        unit = tick_size(out if step > 0 else out - 0.0001, market)
        out += step * unit
    return round(out, 4 if market == "US" else 0)


def breakeven_ticks(fill_price: float, market: str = "KR") -> int:
    """체결가 기준, 수수료·세금을 덮으려면 몇 틱 올라야 본전인가.

    매수는 수수료만, 매도는 수수료+거래세가 붙습니다. 그래서 손익분기는
    체결가보다 항상 위에 있고, 저가주일수록 틱 하나가 차지하는 비율이 커서
    **오히려 유리합니다**(100원에서 1틱은 1%). 진짜 비용은 세금이 아니라
    스프레드인데, 그건 진입 방식(지정가/시장가)으로 다룹니다.
    """
    price = float(fill_price or 0)
    if price <= 0:
        return 0
    unit = tick_size(price, market)
    needed = price * ((1 + FEE_RATE) / (1 - FEE_RATE - TAX_RATE) - 1)
    return max(1, math.ceil(needed / unit)) if needed > 0 else 0


def tick_economics(price: float, cfg: dict, market: str = "KR") -> dict:
    """이 설정으로 이 가격대에서 매매하면 산수가 맞는가.

    화면에 그대로 띄우기 위한 값들입니다. **필요 승률**이 핵심입니다 —
    "3틱 먹고 2틱 손절" 이 좋아 보여도, 비용을 넣으면 실제로는 2틱 먹고
    3틱 잃는 구조라 60% 를 이겨야 본전입니다. 그걸 모르고 하는 것과
    알고 하는 것은 다릅니다.
    """
    cfg = clamp_config(cfg)
    unit = tick_size(price, market)
    be = breakeven_ticks(price, market)
    tp = int(cfg["take_profit_ticks"])
    sl = int(cfg["stop_loss_ticks"])

    net_win = tp - be           # 익절했을 때 실제로 남는 틱
    net_loss = sl + be          # 손절했을 때 실제로 잃는 틱
    total = net_win + net_loss
    win_rate = (net_loss / total * 100) if total > 0 else 100.0

    # "그럼 몇 틱으로 잡아야 하나" 에 대한 답. 부적합 판정만 내리고 대안을
    # 안 주면 사용자는 숫자를 아무렇게나 바꿔가며 다시 걸리는 것을 반복합니다.
    min_viable_tp = be + max(int(cfg["min_net_ticks"]), 1)

    return {
        "price": price,
        "tick_size": unit,
        "tick_pct": round(unit / price * 100, 3) if price > 0 else 0.0,
        "breakeven_ticks": be,
        "take_profit_ticks": tp,
        "stop_loss_ticks": sl,
        "net_win_ticks": net_win,
        "net_loss_ticks": net_loss,
        "required_win_rate": round(win_rate, 1),
        "viable": net_win >= int(cfg["min_net_ticks"]),
        "min_viable_tp_ticks": min_viable_tp,
        "min_viable_tp_price": ticks_to_price(price, min_viable_tp, market),
        "target_price": ticks_to_price(price, tp, market),
        "stop_price": ticks_to_price(price, -sl, market),
    }


# ===========================================================================
# 설정 클램프
# ===========================================================================

def clamp_config(cfg: dict) -> dict:
    """사용자 설정을 하드 한도 안으로 강제로 끌어옵니다.

    반환값에는 무엇이 어떻게 잘렸는지(`_clamped`)가 함께 담깁니다 —
    화면에서 "당신이 넣은 값은 이렇게 제한되었습니다"를 보여주기 위해서입니다.
    """
    out = {**DEFAULT_SCALP, **(cfg or {})}
    clamped = []

    def cap(key: str, limit: float, label: str, unit: str = ""):
        try:
            value = float(out.get(key) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value > limit:
            clamped.append(f"{label} {value:g}{unit} → {limit:g}{unit}")
            out[key] = limit

    def floor(key: str, limit: float, label: str, unit: str = ""):
        try:
            value = float(out.get(key) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value < limit:
            clamped.append(f"{label} {value:g}{unit} → {limit:g}{unit}")
            out[key] = limit

    cap("budget_krw", HARD_LIMITS["max_budget_krw"], "투자금액", "원")
    cap("stop_loss_ticks", HARD_LIMITS["max_stop_loss_ticks"], "손절", "틱")
    cap("max_hold_sec", HARD_LIMITS["max_hold_sec"], "최대 보유", "초")
    floor("reentry_cooldown_sec", HARD_LIMITS["min_reentry_cooldown_sec"],
          "재진입 쿨다운", "초")
    floor("universe_refresh_sec", HARD_LIMITS["min_universe_refresh_sec"],
          "대상 갱신 주기", "초")

    # 틱은 정수여야 합니다 — 호가는 쪼개지지 않습니다
    for key in ("take_profit_ticks", "stop_loss_ticks", "min_net_ticks"):
        try:
            out[key] = max(int(float(out.get(key) or 0)), 0)
        except (TypeError, ValueError):
            out[key] = int(DEFAULT_SCALP[key])
    out["take_profit_ticks"] = max(out["take_profit_ticks"], 1)
    out["stop_loss_ticks"] = max(out["stop_loss_ticks"], 1)

    # 일일 손실 한도는 투자금액을 넘을 수 없습니다 (넘어봐야 의미가 없습니다)
    budget = float(out.get("budget_krw") or 0)
    if float(out.get("daily_loss_krw") or 0) > budget > 0:
        clamped.append(f"일일 손실 한도 → {budget:,.0f}원 (투자금액과 동일)")
        out["daily_loss_krw"] = budget
    if float(out.get("daily_loss_krw") or 0) <= 0:
        out["daily_loss_krw"] = max(budget * 0.1, 1)

    kr_lo, kr_hi = (list(out.get("kr_price_range") or [100, 3000]) + [3000])[:2]
    kr_lo = max(int(kr_lo), HARD_LIMITS["kr_min_price"])
    kr_hi = min(int(kr_hi), HARD_LIMITS["kr_max_price"])
    if [kr_lo, kr_hi] != list(out.get("kr_price_range") or []):
        clamped.append(f"한국 가격대 → {kr_lo:,}~{kr_hi:,}원")
    out["kr_price_range"] = [kr_lo, max(kr_hi, kr_lo + 1)]

    us_lo, us_hi = (list(out.get("us_price_range") or [1.0, 10.0]) + [10.0])[:2]
    us_lo = max(float(us_lo), HARD_LIMITS["us_min_price"])
    us_hi = min(float(us_hi), HARD_LIMITS["us_max_price"])
    if [us_lo, us_hi] != list(out.get("us_price_range") or []):
        clamped.append(f"미국 가격대 → ${us_lo:g}~${us_hi:g}")
    out["us_price_range"] = [us_lo, max(us_hi, us_lo + 0.01)]

    if out.get("entry_order_type") not in ("limit_bid", "market"):
        out["entry_order_type"] = "limit_bid"
    if out.get("rank_basis") not in ("vol_increase", "turnover", "value", "volume"):
        out["rank_basis"] = "vol_increase"

    # 수동 지정 종목 — 자동 갱신이 돌아도 빠지지 않습니다.
    # 구독 한도를 통째로 차지하면 자동 발굴이 아무것도 못 담으므로 절반까지만.
    pinned, seen = [], set()
    for code in (out.get("pinned") or []):
        code = str(code).strip()
        if code and code not in seen:
            seen.add(code)
            pinned.append(code)
    cap = max(HARD_LIMITS["max_tracked"] // 2, 1)
    if len(pinned) > cap:
        clamped.append(f"지정 종목 {len(pinned)}개 → {cap}개 (구독 한도)")
        pinned = pinned[:cap]
    out["pinned"] = pinned

    # 초단타 전용 매매 대상 — 지정 종목이 항상 앞에 옵니다(갱신에도 안 빠짐).
    # 여기서 구독 한도로 잘라두면, 이후 어떤 경로로 읽어도 한도 안입니다.
    tracked = []
    for code in [*pinned, *(out.get("universe") or [])]:
        code = str(code).strip()
        if code and code not in tracked:
            tracked.append(code)
    limit = HARD_LIMITS["max_tracked"]
    if len(tracked) > limit:
        clamped.append(f"초단타 대상 {len(tracked)}개 → {limit}개 (구독 한도)")
        tracked = tracked[:limit]
    out["universe"] = tracked

    out["_clamped"] = clamped
    return out


def max_tracked(cfg: dict = None) -> int:
    """동시에 추적할 수 있는 종목 수.

    사용자 설정이 아닙니다. KIS 실시간체결 WebSocket 이 계정당 41건까지만
    구독을 허용하기 때문에 생기는 물리적 상한입니다. 타점 조건을 통과한
    종목이 이보다 많으면 순위 상위부터 자릅니다.
    """
    return int(HARD_LIMITS["max_tracked"])


# ===========================================================================
# 종목 자격 심사
# ===========================================================================

@dataclass
class Eligibility:
    ok: bool = False
    reasons: list = field(default_factory=list)     # 탈락 사유
    notes: list = field(default_factory=list)       # 통과했지만 알아야 할 것


def _attr(candidate, name: str, default=None):
    """스크리너 소스마다 채우는 필드가 달라서(미국 경로엔 회전율이 없음)
    없는 값을 None 으로 받아 '판단 보류' 로 다룹니다. 0 으로 받으면
    '조건 미달' 이 되어 멀쩡한 종목이 전부 탈락합니다."""
    value = getattr(candidate, name, default)
    return default if value is None else value


def screen_candidate(candidate, cfg: dict) -> Eligibility:
    """이 종목을 초단타 대상으로 삼아도 되는가.

    타점 기준은 세 개입니다 — **거래량 급증 · 거래 회전율 · 가격 변동폭**.
    셋 다 "지금 사람이 몰려서 실제로 움직이는 종목인가" 를 다른 각도에서 봅니다.
    거래량만 보면 원래 큰 종목이 늘 1등이고, 회전율만 보면 품절주가 걸리고,
    변동폭만 보면 거래 없이 호가만 튄 종목이 걸립니다.

    **탈락 사유를 모두 모읍니다.** 첫 번째에서 멈추면 사용자는 하나 고치고
    다시 걸리는 것을 반복하게 됩니다.
    """
    verdict = Eligibility()
    cfg = clamp_config(cfg)
    is_us = candidate.market == "US"
    market = "US" if is_us else "KR"

    lo, hi = (cfg["us_price_range"] if is_us else cfg["kr_price_range"])
    if not (lo <= candidate.price <= hi):
        verdict.reasons.append(
            f"가격대 밖 ({candidate.price:,.2f} / 대상 {lo:,}~{hi:,})")

    # --- 팔 수 있는가 (이건 취향이 아니라 물리) ---
    value = candidate.trading_value_krw or candidate.trading_value
    if value < HARD_LIMITS["min_trading_value_krw"]:
        verdict.reasons.append(
            f"거래대금 부족 ({value / 1e8:.1f}억 / 최소 "
            f"{HARD_LIMITS['min_trading_value_krw'] / 1e8:.0f}억) — 팔고 싶을 때 못 팝니다")

    limit_zone = HARD_LIMITS["limit_zone_pct"]
    if candidate.change_rate >= limit_zone:
        verdict.reasons.append(
            f"{candidate.change_rate:+.1f}% — 상한가 근처는 거래가 멈춰 못 팝니다")
    if candidate.change_rate <= -limit_zone:
        verdict.reasons.append(
            f"{candidate.change_rate:+.1f}% — 하한가 근처는 받아줄 사람이 없습니다")

    if candidate.flags:
        verdict.reasons.append(f"위험 표식: {', '.join(candidate.flags)}")

    # --- 타점 세 가지 ---
    vol_increase = _attr(candidate, "vol_increase_pct")
    if vol_increase is not None and vol_increase < float(cfg["min_vol_increase_pct"]):
        verdict.reasons.append(
            f"거래량 증가율 {vol_increase:,.0f}% / 기준 {cfg['min_vol_increase_pct']:,.0f}%")

    turnover = _attr(candidate, "turnover_pct")
    if turnover is not None and turnover < float(cfg["min_turnover_pct"]):
        verdict.reasons.append(
            f"회전율 {turnover:.2f}% / 기준 {cfg['min_turnover_pct']:.2f}%")

    day_range = _attr(candidate, "day_range_pct")
    if day_range is not None and day_range < float(cfg["min_range_pct"]):
        verdict.reasons.append(
            f"변동폭 {day_range:.1f}% / 기준 {cfg['min_range_pct']:.1f}% — 먹을 폭이 없습니다")

    # --- 산수가 맞는가 ---
    econ = tick_economics(candidate.price, cfg, market)
    if not econ["viable"]:
        verdict.reasons.append(
            f"익절 {econ['take_profit_ticks']}틱 중 비용 {econ['breakeven_ticks']}틱 — "
            f"남는 게 {econ['net_win_ticks']}틱뿐입니다 "
            f"(이 종목은 익절 {econ['min_viable_tp_ticks']}틱 이상이어야 합니다)")

    # --- 통과했더라도 알아야 할 것 ---
    if value < HARD_LIMITS["min_trading_value_krw"] * 3:
        verdict.notes.append("거래대금이 넉넉하지 않습니다 — 수량을 작게 가세요")
    if econ["required_win_rate"] >= 60:
        verdict.notes.append(
            f"이 설정은 승률 {econ['required_win_rate']:.0f}% 를 넘겨야 본전입니다")
    if turnover is None and not is_us:
        verdict.notes.append("회전율을 못 받았습니다 — 이 항목은 심사에서 빠졌습니다")

    verdict.ok = not verdict.reasons
    return verdict


def spread_pct(quote: dict) -> float | None:
    """호가 스프레드(%) — 왕복 비용의 대부분을 차지합니다."""
    if not quote:
        return None
    bid, ask = quote.get("bid"), quote.get("ask")
    if bid and ask and bid > 0:
        return (ask - bid) / bid * 100
    return None


def spread_ticks(quote: dict, market: str = "KR") -> int | None:
    """스프레드가 몇 틱인가.

    페니 초단타에서 진짜 비용은 세금이 아니라 이것입니다. 1틱이면 정상,
    3틱을 넘으면 사는 순간 3틱을 잃고 시작하므로 어떤 설정으로도 못 이깁니다.
    """
    if not quote:
        return None
    bid, ask = quote.get("bid"), quote.get("ask")
    if not (bid and ask and bid > 0 and ask >= bid):
        return None
    unit = tick_size(bid, market)
    return int(round((ask - bid) / unit)) if unit > 0 else None


# ===========================================================================
# 초단타 신호
# ===========================================================================

@dataclass
class ScalpSignal:
    key: str
    ok: bool = False
    score: float = 0.0
    direction: str = FLAT
    price: float = 0.0
    vwap: float | None = None
    rvol: float | None = None
    rsi2: float | None = None
    orb_high: float | None = None
    momentum: float | None = None
    spread_ticks: int | None = None
    bars_used: int = 0
    economics: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "ok": self.ok, "score": round(self.score, 3),
                "direction": self.direction, "price": self.price,
                "vwap": round(self.vwap, 2) if self.vwap else None,
                "rvol": round(self.rvol, 2) if self.rvol else None,
                "rsi2": round(self.rsi2, 1) if self.rsi2 else None,
                "momentum": round(self.momentum, 3) if self.momentum else None,
                "spread_ticks": self.spread_ticks,
                "bars_used": self.bars_used, "economics": self.economics,
                "reasons": self.reasons[:5], "error": self.error}


def _vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["volume"].replace(0, 1)
    return (typical * volume).cumsum() / volume.cumsum()


def _rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))


def evaluate(inst: Instrument, cfg: dict, bars: pd.DataFrame = None,
             quote: dict = None, allow_fetch: bool = True) -> ScalpSignal:
    """진입 신호.

    방향은 분봉으로 잡고, 진입 여부는 틱(스프레드)으로 확정합니다.
    분봉은 "지금 이 종목이 움직이는 방향" 을 보는 데 쓰고, 실제로 들어갈지는
    지금 호가창이 왕복비용을 감당할 수 있는지로 결정합니다 — 아무리 좋은
    신호여도 스프레드가 3틱이면 들어가는 순간 집니다.
    """
    sig = ScalpSignal(key=inst.key)
    cfg = clamp_config(cfg)
    market = "US" if inst.market == "US" else "KR"

    if quote is None and allow_fetch:
        quote = feed.quote(inst)
    if not quote or not quote.get("price"):
        sig.error = "현재가를 조회할 수 없습니다."
        return sig
    sig.price = float(quote["price"])
    sig.economics = tick_economics(sig.price, cfg, market)

    # 스프레드 게이트 — 여기서 막히면 신호를 볼 필요도 없습니다
    sig.spread_ticks = spread_ticks(quote, market)
    if sig.spread_ticks is not None and sig.spread_ticks > HARD_LIMITS["max_spread_ticks"]:
        sig.ok = True
        sig.direction = FLAT
        sig.reasons.append(
            f"스프레드 {sig.spread_ticks}틱 — 사는 순간 {sig.spread_ticks}틱 손실")
        return sig

    if not sig.economics["viable"]:
        sig.ok = True
        sig.direction = FLAT
        sig.reasons.append(
            f"익절 {sig.economics['take_profit_ticks']}틱에서 비용 "
            f"{sig.economics['breakeven_ticks']}틱을 빼면 남는 게 없습니다")
        return sig

    if bars is None and allow_fetch:
        bars = feed.bars(inst, "minute", count=180)
    if bars is None or len(bars) < 30:
        sig.error = f"분봉이 부족합니다 ({0 if bars is None else len(bars)}개)"
        return sig

    sig.bars_used = len(bars)
    close = bars["close"]

    # 1) VWAP — 위에 있으면 매수 우위, 아래면 매도 우위
    vwap = _vwap(bars)
    sig.vwap = float(vwap.iloc[-1])
    vwap_gap = (sig.price - sig.vwap) / sig.vwap * 100 if sig.vwap else 0.0

    # 2) RVOL — 최근 5봉 거래량이 평소 대비 몇 배인가
    recent_vol = float(bars["volume"].iloc[-5:].mean())
    base_vol = float(bars["volume"].iloc[:-5].mean()) or 1.0
    sig.rvol = recent_vol / base_vol

    # 3) RSI(2) — 초단기 과매도/과매수
    sig.rsi2 = float(_rsi(close, 2).iloc[-1])

    # 4) ORB — 초반 30봉의 고가 돌파
    opening = bars.iloc[:30]
    sig.orb_high = float(opening["high"].max())
    orb_break = sig.price > sig.orb_high

    # 5) 모멘텀 — 최근 10봉 수익률
    sig.momentum = float((close.iloc[-1] / close.iloc[-10] - 1) * 100) if len(close) > 10 else 0.0

    # --- 점수 조합 (전부 같은 방향일 때만 임계값을 넘도록) ---
    score = 0.0
    if vwap_gap > 0:
        score += min(vwap_gap / 2.0, 1.0) * 0.30
        sig.reasons.append(f"VWAP 위 {vwap_gap:+.2f}%")
    else:
        score -= min(abs(vwap_gap) / 2.0, 1.0) * 0.30

    if sig.rvol >= 2.0:
        score += min((sig.rvol - 1) / 3.0, 1.0) * 0.30
        sig.reasons.append(f"거래량 급증 {sig.rvol:.1f}배")
    elif sig.rvol < 1.0:
        score -= 0.20          # 거래가 죽어가는 종목은 초단타 대상이 아닙니다

    if orb_break:
        score += 0.20
        sig.reasons.append(f"장초반 고가 {sig.orb_high:,.0f} 돌파")

    if sig.momentum > 0:
        score += min(sig.momentum / 3.0, 1.0) * 0.20
        sig.reasons.append(f"모멘텀 {sig.momentum:+.2f}%")
    else:
        score -= min(abs(sig.momentum) / 3.0, 1.0) * 0.20

    # RSI(2)는 과열 브레이크로만 씁니다 — 90 이상이면 이미 늦었습니다
    if sig.rsi2 >= 90:
        score -= 0.35
        sig.reasons.append(f"RSI(2) {sig.rsi2:.0f} 과열 — 감점")
    elif sig.rsi2 <= 10:
        score -= 0.15          # 급락 중 반등 베팅은 하지 않습니다

    sig.score = max(-1.0, min(1.0, score))
    entry = max(float(cfg.get("entry_score", 0.55)), 0.45)   # 임계값 하한도 강제
    sig.direction = LONG if sig.score >= entry else FLAT
    sig.ok = True
    return sig


# ===========================================================================
# 주문 계획 — 틱을 실제 가격과 수량으로
# ===========================================================================

@dataclass
class ScalpPlan:
    ok: bool = False
    quantity: int = 0
    entry_price: float = 0.0        # 지정가로 낼 가격 (시장가면 참고용)
    order_type: str = "limit"
    target_price: float = 0.0
    stop_price: float = 0.0
    order_krw: float = 0.0
    economics: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "quantity": self.quantity,
                "entry_price": self.entry_price, "order_type": self.order_type,
                "target_price": self.target_price, "stop_price": self.stop_price,
                "order_krw": round(self.order_krw), "economics": self.economics,
                "reasons": self.reasons}


def plan_order(inst: Instrument, quote: dict, cfg: dict,
               budget_left: float, trading_value_krw: float = 0) -> ScalpPlan:
    """진입 주문 하나를 만듭니다.

    **진입은 지정가(매수호가), 청산은 시장가** 가 기본입니다. 비대칭인 이유:
    진입은 못 잡아도 손해가 없지만, 청산은 못 빠져나오면 그게 손실입니다.
    100원짜리에서 시장가로 사면 매도호가에 체결돼 1틱(=1%)을 잃고 시작합니다.
    """
    plan = ScalpPlan()
    cfg = clamp_config(cfg)
    market = "US" if inst.market == "US" else "KR"

    last = float(quote.get("price") or 0)
    if last <= 0:
        plan.reasons.append("현재가 없음")
        return plan

    bid = float(quote.get("bid") or 0)
    use_limit = cfg["entry_order_type"] == "limit_bid" and bid > 0
    plan.entry_price = bid if use_limit else last
    plan.order_type = "limit" if use_limit else "market"

    # 체결가 기준으로 손익 구조를 계산합니다. 시장가면 매도호가에 체결되므로
    # 한 틱 불리한 쪽을 가정해야 나중에 "왜 계산과 다르냐" 가 안 생깁니다.
    assumed_fill = plan.entry_price if use_limit else ticks_to_price(last, 1, market)
    plan.economics = tick_economics(assumed_fill, cfg, market)
    if not plan.economics["viable"]:
        plan.reasons.append(
            f"비용 {plan.economics['breakeven_ticks']}틱을 빼면 "
            f"{plan.economics['net_win_ticks']}틱 남음 — 진입하지 않습니다")
        return plan

    plan.target_price = plan.economics["target_price"]
    plan.stop_price = plan.economics["stop_price"]

    # --- 수량: 남은 예산이 유일한 기준입니다 ---
    if budget_left < assumed_fill:
        plan.reasons.append(
            f"남은 예산 {budget_left:,.0f}원으로 {assumed_fill:,.0f}원짜리를 못 삽니다")
        return plan
    quantity = int(budget_left // assumed_fill)

    # 내 주문이 하루 거래대금 대비 너무 크면 내가 호가를 밀게 됩니다.
    # 그러면 계산한 진입가에 못 사고, 팔 때는 더 나쁩니다.
    if trading_value_krw > 0:
        impact_cap = trading_value_krw * HARD_LIMITS["max_order_impact_pct"] / 100.0
        if quantity * assumed_fill > impact_cap:
            capped = int(impact_cap // assumed_fill)
            if capped < 1:
                plan.reasons.append("거래대금 대비 최소 수량조차 부담이 큽니다")
                return plan
            plan.reasons.append(
                f"거래대금 대비 주문이 커서 {quantity:,}주 → {capped:,}주로 줄임")
            quantity = capped

    plan.quantity = quantity
    plan.order_krw = quantity * assumed_fill
    plan.ok = quantity >= 1
    if plan.ok:
        plan.reasons.append(
            f"{quantity:,}주 · 목표 {plan.target_price:,.0f} / 손절 {plan.stop_price:,.0f} "
            f"(실질 +{plan.economics['net_win_ticks']}틱 / -{plan.economics['net_loss_ticks']}틱)")
    return plan


# ===========================================================================
# 진입 가능 시간대
# ===========================================================================

def session_guard(inst: Instrument, now=None) -> tuple[bool, str]:
    """지금 이 종목에 초단타 진입을 해도 되는 시간인가."""
    return market_session_guard("US" if inst.market == "US" else "KR",
                                now=now, status=feed.market_status(inst))


def market_session_guard(market: str, now=None, status: dict = None) -> tuple[bool, str]:
    """지금 초단타 진입을 해도 되는 시간인가 — **종목 없이 시장만으로** 판정.

    개장 직후와 마감 직전은 초단타에 가장 나쁜 구간입니다.
    개장 직후는 가격이 아직 균형을 못 찾았고, 마감 직전은 **빠져나올 시간이
    없습니다** — 초단타에서 못 빠져나오는 것이 가장 큰 손실 원인입니다.

    종목을 받지 않는 판이 따로 있는 이유: "지금 초단타가 도는 중인가"를
    화면에 보여줄 때는 추적 종목이 하나도 없을 수도 있습니다. 그때도 시간
    판정은 나와야 합니다.
    """
    from data_sources import market_clock

    now = now or market_clock.now_kst()
    if status is None:
        status = market_clock.status_for("US" if market == "US" else "KOSPI")
    if not status.get("is_regular"):
        return False, f"정규장이 아닙니다 ({status.get('label', '')})"

    if market != "US":
        minutes = now.hour * 60 + now.minute
        open_min = 9 * 60             # 09:00 개장
        close_min = 15 * 60 + 20      # 15:20 정규장 마감
        if minutes < open_min + HARD_LIMITS["no_entry_first_minutes"]:
            return False, f"개장 직후 {HARD_LIMITS['no_entry_first_minutes']}분은 진입 금지"
        if minutes > close_min - HARD_LIMITS["no_entry_last_minutes"]:
            return False, f"마감 {HARD_LIMITS['no_entry_last_minutes']}분 전부터 신규 진입 금지"
    return True, "진입 가능 시간"


def risk_warnings(cfg: dict) -> list[str]:
    """화면에 띄울 경고 문구. 설정과 무관하게 항상 보여줍니다."""
    cfg = clamp_config(cfg)
    budget = float(cfg["budget_krw"])
    loss_cap = float(cfg["daily_loss_krw"])
    warnings = [
        "페니주식 초단타는 이 프로그램에서 가장 위험한 기능입니다.",
        f"이 전략은 {budget:,.0f}원까지만 씁니다. "
        f"오늘 {loss_cap:,.0f}원을 잃으면 그날은 종료합니다.",
        "저가주는 유동성이 얇아 팔고 싶을 때 사줄 사람이 없을 수 있습니다.",
        "작전·펌프앤덤프의 주 무대입니다. 급등의 상당수는 의도된 것입니다.",
        "손절이 호가를 건너뛰어 예상보다 훨씬 낮은 가격에 체결될 수 있습니다.",
        "하루 매매 횟수 제한이 없습니다 — 그날을 끊는 것은 손실 금액 하나뿐입니다.",
        "잃어도 생활에 지장이 없는 금액으로만 하세요.",
    ]
    if cfg.get("_clamped"):
        warnings.append("입력값 일부가 안전 한도로 제한되었습니다: "
                        + " / ".join(cfg["_clamped"]))
    return warnings


def describe_hard_limits() -> list[dict]:
    """설정으로 못 뚫는 한도를 화면에 그대로 보여줍니다."""
    return [
        {"label": "한국 대상 가격", "value": f"{HARD_LIMITS['kr_min_price']:,}~{HARD_LIMITS['kr_max_price']:,}원"},
        {"label": "미국 대상 가격", "value": f"${HARD_LIMITS['us_min_price']:g}~${HARD_LIMITS['us_max_price']:g}"},
        {"label": "최소 거래대금", "value": f"{HARD_LIMITS['min_trading_value_krw'] / 1e8:.0f}억원"},
        {"label": "투자금액 상한", "value": f"{HARD_LIMITS['max_budget_krw']:,}원"},
        {"label": "스프레드 상한", "value": f"{HARD_LIMITS['max_spread_ticks']}틱"},
        {"label": "상·하한가 근처", "value": f"±{HARD_LIMITS['limit_zone_pct']:g}% 진입 금지"},
        {"label": "손절 상한", "value": f"{HARD_LIMITS['max_stop_loss_ticks']}틱"},
        {"label": "최대 보유", "value": f"{HARD_LIMITS['max_hold_sec']}초"},
        {"label": "재진입 쿨다운", "value": f"최소 {HARD_LIMITS['min_reentry_cooldown_sec']}초"},
        {"label": "동시 추적", "value": f"{HARD_LIMITS['max_tracked']}종목 (시세 구독 한도)"},
        {"label": "대상 갱신", "value": f"최소 {HARD_LIMITS['min_universe_refresh_sec']}초"},
        {"label": "진입 금지 시간", "value": f"개장 {HARD_LIMITS['no_entry_first_minutes']}분 / "
                                           f"마감 {HARD_LIMITS['no_entry_last_minutes']}분"},
    ]


# ===========================================================================
# 자금 맞춤 추천
# ===========================================================================

def recommend(candidates: list, available_cash: float, total_value: float,
              cfg: dict) -> list[dict]:
    """가용 현금에 맞는 후보를 순위대로.

    "좋은 종목"이 아니라 **"이 돈으로 살 수 있고, 팔 수 있는 종목"** 을 고릅니다.
    예산은 `budget_krw` 이며, 계좌 현금이 그보다 적으면 현금이 상한입니다.
    (`total_value` 는 더 이상 예산 계산에 쓰지 않습니다 — 비율 기반을 걷어냈습니다.
     호출부 호환을 위해 인자만 남겨둡니다)
    """
    cfg = clamp_config(cfg)
    budget_cap = min(float(cfg["budget_krw"]), float(available_cash or 0))

    out = []
    for candidate in candidates:
        market = "US" if candidate.market == "US" else "KR"
        verdict = screen_candidate(candidate, cfg)
        unit = candidate.price_krw or candidate.price
        quantity = int(budget_cap // unit) if unit > 0 else 0
        affordable = quantity >= 1

        value = candidate.trading_value_krw or candidate.trading_value or 1
        order_krw = quantity * unit
        # 하루 거래대금의 0.5% 를 넘는 주문은 내가 호가를 밀게 됩니다
        impact_pct = order_krw / value * 100 if value else 100
        econ = tick_economics(unit, cfg, market)

        # 적합도: 살 수 있는가 + 빠져나올 수 있는가 + 타점이 살아 있는가
        fit = 0.0
        if affordable:
            fit += 0.30
        fit += min(value / (HARD_LIMITS["min_trading_value_krw"] * 5), 1.0) * 0.20
        fit += max(0.0, 1.0 - impact_pct / HARD_LIMITS["max_order_impact_pct"]) * 0.20
        vol_increase = _attr(candidate, "vol_increase_pct", 0.0)
        fit += min(vol_increase / 500.0, 1.0) * 0.15
        turnover = _attr(candidate, "turnover_pct", 0.0)
        fit += min(turnover / 5.0, 1.0) * 0.15

        out.append({
            **candidate.to_dict(),
            "eligible": verdict.ok,
            "reasons": verdict.reasons,
            "notes": verdict.notes,
            "affordable": affordable,
            "max_quantity": quantity,
            "order_krw": round(order_krw),
            "budget_cap": round(budget_cap),
            "impact_pct": round(impact_pct, 3),
            "economics": econ,
            "fit": round(fit, 3),
        })

    out.sort(key=lambda c: (c["eligible"], c["affordable"], c["fit"]), reverse=True)
    return out


# ===========================================================================
# 초봉 차트 — 지표를 시계열로 펼치기
# ===========================================================================
# 여기 주기는 전부 **봉 개수**입니다. 1초봉이면 EMA9 는 9초, 5초봉이면 45초입니다.
# 일봉용 파라미터(MA20/60)를 그대로 쓰면 1초봉에서 20초·60초가 되어버려서
# 의미가 달라집니다. 그래서 초단타 전용으로 짧게 잡았습니다.
CHART_EMA_FAST = 9
CHART_EMA_SLOW = 21
CHART_BB_PERIOD = 20
CHART_BB_STD = 2.0
CHART_RSI_PERIOD = 2


def _ema_series(values: list, period: int) -> list:
    """지수이동평균. 워밍업 구간은 None 으로 둡니다.

    첫 봉부터 값을 채우면 시드(첫 종가)가 그대로 찍혀서, 차트 왼쪽 끝에
    실제로는 계산되지 않은 선이 그려집니다.
    """
    out, prev, alpha = [], None, 2.0 / (period + 1)
    for i, value in enumerate(values):
        prev = value if prev is None else (value - prev) * alpha + prev
        out.append(round(prev, 4) if i >= period - 1 else None)
    return out


def _bollinger_series(values: list, period: int, num_std: float):
    mid, upper, lower = [], [], []
    for i in range(len(values)):
        if i < period - 1:
            mid.append(None), upper.append(None), lower.append(None)
            continue
        window = values[i - period + 1:i + 1]
        mean = sum(window) / period
        sd = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
        mid.append(round(mean, 4))
        upper.append(round(mean + num_std * sd, 4))
        lower.append(round(mean - num_std * sd, 4))
    return mid, upper, lower


def _rsi_series(values: list, period: int) -> list:
    """RSI — Wilder 방식. 초단타는 기간 2 를 씁니다(Connors)."""
    out = [None] * len(values)
    if len(values) <= period:
        return out
    gain = loss = 0.0
    alpha = 1.0 / period
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        up, down = max(delta, 0.0), max(-delta, 0.0)
        if i == 1:
            gain, loss = up, down
        else:
            gain += alpha * (up - gain)
            loss += alpha * (down - loss)
        if i >= period:
            out[i] = 100.0 if loss <= 0 else round(100 - 100 / (1 + gain / loss), 2)
    return out


def _vwap_series(bars: list) -> list:
    """누적 VWAP. 거래량이 0인 봉은 **체결 건수**로 대신 가중합니다.

    저가주 초봉에는 거래량 필드가 0으로 오는 봉이 섞입니다. 그대로 두면
    분모가 늘지 않아 VWAP 이 한 자리에 얼어붙습니다.
    """
    out, price_volume, volume = [], 0.0, 0.0
    for bar in bars:
        typical = (bar["h"] + bar["l"] + bar["c"]) / 3
        weight = bar["v"] or bar["n"] or 1
        price_volume += typical * weight
        volume += weight
        out.append(round(price_volume / volume, 4) if volume else None)
    return out


def chart_series(bars: list, cfg: dict = None, market: str = "KR") -> dict:
    """초봉 + 보조지표를 화면이 그대로 그릴 수 있는 형태로.

    `t` 는 자정으로부터의 초입니다. 봉이 없는 시각(체결이 없던 초)은 애초에
    만들지 않으므로, 화면은 **봉 인덱스**로 그리고 축 라벨만 시각으로 씁니다.
    체결이 없는 구간을 빈칸으로 벌려 그리면 초단타 차트가 대부분 공백이 됩니다.
    """
    cfg = clamp_config(cfg or {})
    bars = list(bars or [])
    closes = [b["c"] for b in bars]

    mid, upper, lower = _bollinger_series(closes, CHART_BB_PERIOD, CHART_BB_STD)
    series = {
        "t": [b["t"] for b in bars],
        "open": [b["o"] for b in bars],
        "high": [b["h"] for b in bars],
        "low": [b["l"] for b in bars],
        "close": closes,
        "volume": [b["v"] for b in bars],
        "trades": [b["n"] for b in bars],          # 봉당 체결 건수
        "vwap": _vwap_series(bars),
        "ema_fast": _ema_series(closes, CHART_EMA_FAST),
        "ema_slow": _ema_series(closes, CHART_EMA_SLOW),
        "bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
        "rsi": _rsi_series(closes, CHART_RSI_PERIOD),
    }

    last = closes[-1] if closes else 0.0
    econ = tick_economics(last, cfg, market) if last else {}
    return {
        "bars": len(bars),
        "series": series,
        "economics": econ,
        "levels": {
            # 지금 이 가격에서 들어간다면 목표·손절이 어디인지 (가로선용)
            "last": last,
            "target": econ.get("target_price"),
            "stop": econ.get("stop_price"),
            "tick_size": econ.get("tick_size"),
        },
        "indicator_periods": {
            "ema_fast": CHART_EMA_FAST, "ema_slow": CHART_EMA_SLOW,
            "bollinger": CHART_BB_PERIOD, "rsi": CHART_RSI_PERIOD,
        },
    }


def is_penny_target(inst: Instrument, price: float, cfg: dict) -> bool:
    """이 종목이 페니 초단타 대상 가격대인가 (자산군도 함께 확인)."""
    if inst.asset_class not in (STOCK, ETF):
        return False
    cfg = clamp_config(cfg)
    lo, hi = (cfg["us_price_range"] if inst.market == "US" else cfg["kr_price_range"])
    return lo <= price <= hi


def why_not_target(inst: Instrument, price: float, cfg: dict) -> str:
    """대상이 아니라면 왜 아닌지 한 줄로.

    이게 없으면 종목을 지정해놓고 "초단타가 안 도는데 이유를 모르는" 상태가
    됩니다. 예전 코드는 대상이 아니면 조용히 일반 전략으로 빠졌습니다.
    """
    cfg = clamp_config(cfg)
    if inst.asset_class not in (STOCK, ETF):
        return f"{inst.asset_class} 는 초단타 대상이 아닙니다 (주식·ETF만)"
    lo, hi = (cfg["us_price_range"] if inst.market == "US" else cfg["kr_price_range"])
    if price < lo:
        return f"{price:,.0f} < 대상 하한 {lo:,.0f}"
    if price > hi:
        return f"{price:,.0f} > 대상 상한 {hi:,.0f}"
    return ""
