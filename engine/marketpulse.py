"""
Market Pulse — 오닐 시장 방향(M) 상태기계 (prism-insight 이식)
------------------------------------------------------------
dragon1086/prism-insight 의 cores/market_pulse.py 를 이식한 것입니다. 원본은
William O'Neil (IBD, "How to Make Money in Stocks") 의 시장 판독 규칙을
결정적 유한상태기계로 옮긴 순수 모듈입니다 — 상수는 60년 시장사에서 나온
IBD 공식 방법론 값이지, 특정 계좌 이력에 맞춘 튜닝값이 아닙니다.

무엇이 새로운가
    아테나의 기존 국면 판별은 **종목 단위** 통계(Hurst·분산비율)이고, 난기류는
    분포 이탈 탐지입니다. Market Pulse 는 **시장 전체(지수)** 를 규칙 기반으로
    읽습니다 — 세 층이 서로 다른 질문에 답합니다.

상태
    UPTREND         분산일(DD) ≤ 3 — 정상
    UNDER_PRESSURE  분산일 4~5 — 기관이 조용히 파는 중
    CORRECTION      분산일 ≥ 6, 또는 롤링 고점 대비 −10% (폭포 하락은 천장
                    과정을 건너뜁니다 — Rev.2 엣지 트리거)

규칙 (원본 주석의 IBD 출처 표 그대로)
    분산일(DD)      종가 −0.2% 이상 하락 + 거래량 전일比 증가.
                    거래량 정보가 없으면 DD 로 **세지 않습니다** (가격 하락만으로는
                    기관 매도 확인이 안 됩니다 — IBD 정의).
    DD 만료         25거래일 경과, 또는 그 DD 종가 대비 +5% 회복
    랠리 시도       조정 저점 이후 첫 상승 종가 = 1일차. 시도 저점 이탈 시 리셋
    팔로우스루(FTD) 랠리 4일차 이후 +1.25% 급등 + 거래량 증가 → 조정 탈출
                    (거래량 미제공 지수는 가격 조건만으로 판정 — 문서화된 완화)
    가격 회복 탈출  조정 전 고점 위 종가 = 정의상 신고가 = 상승장 (Rev.1 —
                    FTD 는 바닥을 일찍 잡는 장치지 유일한 출구가 아닙니다)
    재발 방지       조정 탈출 시 기준 고점을 탈출 종가로 리셋 — 다시 −10% 가
                    **새로** 나와야 재진입 (Rev.2 anti-flap)

이 상태로 무엇을 하는가 — **줄이되 막지 않습니다**
    원본의 V2 매매 표본 감사가 "CORRECTION = 전면 매수 중단" 정책을 **기각**
    했습니다: 조정 구간 매수는 손절율 38% 로 무섭지만 순손익은 +25.3% 였습니다
    (폭락 후 반등 대어가 이 구간에 삽니다). 그래서 원본은 조정장에서 분석
    횟수만 줄였고, 아테나도 같은 이유로 점수 감쇠만 하고 진입 차단은 하지
    않습니다 (engine/ensemble.py 통합부).

지수 프록시
    KIS·네이버 무료 경로로 코스피 지수 자체의 일봉·거래량을 안정적으로 받기
    어려워, 기존 피드가 그대로 지원하는 **KODEX 200 (069500)** 을 한국 시장
    프록시로, **SPY** 를 미국 프록시로 씁니다. ETF 종가는 지수를 추종하고
    거래량은 시장 참여 강도를 반영합니다 — DD/FTD 의 거래량 확인 조건에
    지수 거래량 대신 쓸 수 있는 가장 가까운 공개 데이터입니다.
"""

import threading
import time
from dataclasses import dataclass

import pandas as pd

# IBD 상수 — 원본 market_pulse.py 의 값 그대로
DISTRIBUTION_WINDOW = 25          # DD 를 세는 구간 (거래일)
DISTRIBUTION_DROP_PCT = 0.2       # 종가 하락 임계 (%)
DISTRIBUTION_RECOVERY_PCT = 5.0   # 이만큼 회복하면 그 DD 는 만료
UNDER_PRESSURE_MIN_DD = 4
CORRECTION_MIN_DD = 6
DRAWDOWN_CORRECTION_PCT = 10.0    # 롤링 고점 대비 −10% → 조정 (Rev.2)
FTD_MIN_RALLY_DAY = 4
FTD_MIN_GAIN_PCT = 1.25           # 오닐 HTMMIS 정본 (Rev.1: 1.7 은 과보수로 완화)

UPTREND, UNDER_PRESSURE, CORRECTION = "UPTREND", "UNDER_PRESSURE", "CORRECTION"

LABELS = {UPTREND: "상승 추세", UNDER_PRESSURE: "압박 국면", CORRECTION: "조정 국면"}

# 시장 → 지수 프록시 종목
PROXIES = {"KR": "069500", "KOSPI": "069500", "KOSDAQ": "069500", "US": "SPY"}

_cache: dict = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 600.0                # 지수 상태는 분 단위로 바뀌지 않습니다


@dataclass(frozen=True)
class DailyBar:
    date: str
    close: float
    volume: float | None = None


def _count_distribution_days(closes, vols, window=DISTRIBUTION_WINDOW,
                             start_idx=0) -> int:
    """살아 있는 분산일 개수 (원본 _count_distribution_days 미러).

    DD 성립: 종가 −0.2% 이상 하락 + 거래량 전일比 증가 (거래량 결측 → 불성립)
    DD 만료: 창(25일) 밖으로 밀려나거나, 이후 종가가 그 DD 종가의 +5% 이상
    start_idx: FTD 이후 창 리셋 — 그 이전의 DD 는 더 이상 세지 않습니다
    """
    n = len(closes)
    if n < 2:
        return 0
    start = max(1, n - window, start_idx)
    running_max_after = float("-inf")
    flags = []
    # 역방향 패스 — running_max_after 가 'i 이후의 최고 종가'가 되게
    for i in range(n - 1, start - 1, -1):
        prev_c, cur_c = closes[i - 1], closes[i]
        if prev_c <= 0:
            flags.append((i, False, running_max_after))
            running_max_after = max(running_max_after, cur_c)
            continue
        pct = (cur_c - prev_c) / prev_c * 100.0
        vi, vp = vols[i], vols[i - 1]
        vol_up = vi is not None and vp is not None and vi > vp
        flags.append((i, pct <= -DISTRIBUTION_DROP_PCT and vol_up, running_max_after))
        running_max_after = max(running_max_after, cur_c)

    kept = 0
    for i, is_dist, max_after in flags:
        if is_dist and max_after < closes[i] * (1 + DISTRIBUTION_RECOVERY_PCT / 100.0):
            kept += 1
    return kept


class MarketPulse:
    """봉을 시간순으로 feed 하면 그 시점까지의 상태를 돌려주는 FSM.

    as-of 의미론 — 상태는 지금까지 넣은 봉에만 의존하므로, 같은 시퀀스를
    다시 돌리면 항상 같은 결과가 나옵니다 (재현 가능).
    """

    def __init__(self):
        self._closes: list[float] = []
        self._vols: list[float | None] = []
        self._state = UPTREND
        self._last_dd = 0
        self._dd_window_start = 0
        self._correction_low = None
        self._rally_active = False
        self._rally_day = 0
        self._rally_start_low = None
        self._pre_correction_peak = None
        self._reference_peak = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def distribution_days(self) -> int:
        return self._last_dd

    def feed(self, bar: DailyBar) -> str:
        self._closes.append(float(bar.close))
        self._vols.append(None if bar.volume is None else float(bar.volume))
        n = len(self._closes)
        cur = self._closes[-1]

        # 롤링 기준 고점 — 신고가는 고점만 올리고 자신을 트리거하지 않습니다
        if self._reference_peak is None or cur > self._reference_peak:
            self._reference_peak = cur

        self._last_dd = _count_distribution_days(
            self._closes, self._vols, DISTRIBUTION_WINDOW, self._dd_window_start)

        if self._state == CORRECTION:
            self._update_correction(n)
        else:
            # UPTREND/UNDER_PRESSURE 는 매일 DD 개수로 재판정.
            # CORRECTION 만 '끈적'합니다 (FTD·가격 회복으로만 탈출).
            drawdown = (self._reference_peak is not None and
                        cur < self._reference_peak
                        * (1.0 - DRAWDOWN_CORRECTION_PCT / 100.0))
            if drawdown or self._last_dd >= CORRECTION_MIN_DD:
                self._enter_correction()
            elif self._last_dd >= UNDER_PRESSURE_MIN_DD:
                self._state = UNDER_PRESSURE
            else:
                self._state = UPTREND
        return self._state

    def _enter_correction(self):
        self._state = CORRECTION
        self._correction_low = self._closes[-1]
        self._pre_correction_peak = self._reference_peak
        self._rally_active = False
        self._rally_day = 0
        self._rally_start_low = None

    def _update_correction(self, n: int):
        cur = self._closes[-1]
        prev = self._closes[-2] if n >= 2 else cur
        cur_vol, prev_vol = self._vols[-1], (self._vols[-2] if n >= 2 else None)

        if self._correction_low is None:
            self._correction_low = cur

        # 가격 회복 탈출 (Rev.1) — 조정 전 고점 위 종가 = 정의상 신고가
        if self._pre_correction_peak is not None and cur > self._pre_correction_peak:
            self._exit_to_uptrend(n)
            return

        if not self._rally_active:
            if n >= 2 and cur > prev:
                self._rally_active = True         # 랠리 시도 1일차
                self._rally_day = 1
                self._rally_start_low = self._correction_low
            elif cur < self._correction_low:
                self._correction_low = cur
            return

        if self._rally_start_low is not None and cur < self._rally_start_low:
            # 시도 시작 저점 이탈 → 랠리 실패, 리셋
            self._rally_active = False
            self._rally_day = 0
            self._correction_low = min(self._correction_low, cur)
            self._rally_start_low = None
            return

        self._rally_day += 1
        gain = (cur - prev) / prev * 100.0 if prev > 0 else 0.0
        has_vol = cur_vol is not None and prev_vol is not None
        vol_ok = (cur_vol > prev_vol) if has_vol else True   # 결측 시 가격 조건만
        if self._rally_day >= FTD_MIN_RALLY_DAY and gain >= FTD_MIN_GAIN_PCT and vol_ok:
            self._exit_to_uptrend(n)              # 팔로우스루 성립

    def _exit_to_uptrend(self, n: int):
        """조정 탈출 공통 처리 (FTD·가격 회복): UPTREND + DD 창 리셋 + anti-flap."""
        self._state = UPTREND
        self._dd_window_start = n
        self._last_dd = 0
        self._rally_active = False
        self._rally_day = 0
        self._rally_start_low = None
        self._correction_low = None
        self._pre_correction_peak = None
        # 기준 고점을 오늘 종가로 리셋 — 새 −10% 가 나와야 재트리거됩니다
        self._reference_peak = self._closes[-1]


# ---------------------------------------------------------------------------
# DataFrame 진입점
# ---------------------------------------------------------------------------

def pulse_of(bars: pd.DataFrame) -> dict | None:
    """지수(프록시) 일봉 → 현재 상태. 순수 함수 — 네트워크 없음."""
    if bars is None or len(bars) < 30 or "close" not in bars.columns:
        return None
    machine = MarketPulse()
    has_volume = "volume" in bars.columns
    for i in range(len(bars)):
        close = float(bars["close"].iloc[i])
        if close <= 0:
            continue
        volume = None
        if has_volume:
            v = bars["volume"].iloc[i]
            volume = float(v) if pd.notna(v) and v > 0 else None
        machine.feed(DailyBar(date=str(bars.index[i])[:10], close=close,
                              volume=volume))
    state = machine.state
    return {
        "state": state,
        "label": LABELS.get(state, state),
        "distribution_days": machine.distribution_days,
        "in_rally_attempt": machine._rally_active,
        "rally_day": machine._rally_day,
        "bars": len(bars),
    }


def market_state(market: str) -> dict | None:
    """시장 문자열("KR"/"KOSPI"/"US" 등) → 프록시 지수의 현재 상태.

    실패하면 **None** 을 돌려주고 절대 예외를 내지 않습니다 (fail-open —
    원본 regime_policy.get_market_pulse_state 와 같은 계약). 지수 상태를 몰라서
    매매가 멈추는 것은 이 장치의 목적이 아닙니다.
    """
    proxy = PROXIES.get(str(market or "").upper())
    if proxy is None:
        proxy = PROXIES["KR"] if str(market or "").upper() != "US" else "SPY"

    now = time.time()
    with _cache_lock:
        hit = _cache.get(proxy)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]

    result = None
    try:
        from engine import feed, instruments
        inst = instruments.try_resolve(proxy)
        if inst is not None:
            bars = feed.bars(inst, "day", count=300)
            result = pulse_of(bars)
            if result is not None:
                result["proxy"] = proxy
    except Exception:
        result = None

    with _cache_lock:
        _cache[proxy] = (now, result)
    return result


def clear_cache():
    with _cache_lock:
        _cache.clear()


__all__ = ["MarketPulse", "DailyBar", "pulse_of", "market_state", "clear_cache",
           "UPTREND", "UNDER_PRESSURE", "CORRECTION", "LABELS"]
