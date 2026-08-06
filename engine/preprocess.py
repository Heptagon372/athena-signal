"""
시세 전처리 (OHLCV Preprocessing)
--------------------------------
지표 엔진에 들어가기 **전에** 봉 데이터를 한 번 걸러내는 곳입니다.

왜 필요한가
    지금까지 engine/feed.py 가 받아온 봉은 제공처(토스·KIS·네이버·야후)가 준
    그대로 indicators.analyze 로 들어갔습니다. 그런데 무료 시세에는 다음이 섞여 옵니다.

      · 같은 시각 봉이 두 번        제공처를 갈아타는 구간에서 발생
      · high < close 같은 정합성 위반  반올림·병합 오류
      · 거래정지일의 0거래량 무변동 봉  변동성을 실제보다 낮게 만듭니다
      · 액면분할 당일의 −98% 수익률   ATR·분산비율·Hurst 를 통째로 오염시킵니다
      · 단일 봉 오틱(bad tick)      다음 봉에서 그대로 되돌아오는 튐

    이 중 하나만 남아도 손절 폭(ATR)과 국면 판정이 조용히 틀어집니다. 값이
    틀린 줄 모른 채 주문이 나가는 것이 자동매매에서 가장 위험합니다.

설계 원칙 — **가격을 함부로 고치지 않는다**
    전처리가 진짜 급등락을 지우면, 지우기 전보다 더 나쁩니다(상한가를 '오류'로
    판정해 삭제하면 추세를 통째로 잃습니다). 그래서 이 모듈은 세 가지만 고칩니다.

      1) 구조적 오류   중복·역순·NaN·음수가격·OHLC 정합성 위반
      2) 자본 변동     액면분할/병합 — 가격 자체가 재정의된 사건이라 반드시 조정
      3) 되돌아온 튐   한 봉만 극단으로 갔다가 다음 봉에서 그대로 복귀한 경우

    나머지(거래정지·유동성 부족·큰 갭)는 **고치지 않고 표시만** 합니다.
    표시는 Quality 로 나가고, 상위 판단이 그 값을 보고 스스로 결정합니다.

참고 구현
    freqtrade 의 clean_ohlcv_dataframe / ohlcv_fill_up_missing_data 가 하는
    "중복 제거 → 미완성 봉 처리 → 결측 채움" 순서를 따랐습니다. 다만 결측 채움은
    주식에 그대로 쓸 수 없어 바꿨습니다 — 24시간 돌아가는 암호화폐와 달리 주식은
    주말·공휴일·거래정지가 정상이고, 달력 기준으로 채우면 없는 거래일을 만들어
    ATR 과 이동평균의 기간 의미가 무너집니다.
    FinRL 의 FeatureEngineer.clean_data 가 하는 "종목별 결측 정리"도 같은 이유로
    횡단면이 아니라 종목 단위로 재구성했습니다.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")

# 액면분할/병합에서 실제로 쓰이는 비율. 임의의 실수를 허용하면 진짜 급락이
# "1:1.37 분할"로 둔갑합니다. 실제 공시에 나오는 비율만 인정합니다.
KNOWN_SPLIT_RATIOS = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 25.0, 50.0, 100.0)

SPLIT_RETURN_THRESHOLD = 0.35   # |로그수익률| 이 이보다 커야 분할 후보로 봅니다
SPLIT_RATIO_TOL = 0.04          # 알려진 비율에서 ±4% 안이어야 인정
SPLIT_VOLUME_HINT = 1.5         # 거래량이 이 배수 이상 늘면 분할 정황 보강

SPIKE_Z = 8.0                   # 로버스트 z (MAD 기준) — 이보다 크면 튐 후보
SPIKE_REVERT_TOL = 0.35         # 다음 봉이 이만큼 안쪽으로 되돌리면 '오틱'으로 판정

# 데이터 신뢰도가 이 밑이면 상위에서 신호를 만들지 않는 것을 권장합니다
MIN_TRUSTWORTHY_QUALITY = 0.6


@dataclass
class Quality:
    """전처리 결과 보고서. **무엇을 고쳤는지 숨기지 않는 것**이 목적입니다."""
    ok: bool = False
    rows_in: int = 0
    rows_out: int = 0
    score: float = 0.0                              # 0~1 데이터 신뢰도
    dropped: dict = field(default_factory=dict)     # 사유별 삭제 봉 수
    repairs: list = field(default_factory=list)     # 사람이 읽는 수정 내역
    flags: dict = field(default_factory=dict)       # 고치지 않고 표시만 한 것들
    error: str = ""

    @property
    def trustworthy(self) -> bool:
        return self.ok and self.score >= MIN_TRUSTWORTHY_QUALITY

    def to_dict(self) -> dict:
        return {"ok": self.ok, "rows_in": self.rows_in, "rows_out": self.rows_out,
                "score": round(self.score, 3), "trustworthy": self.trustworthy,
                "dropped": self.dropped, "repairs": self.repairs[:8],
                "flags": self.flags, "error": self.error}


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame, timeframe: str = "day", *,
          drop_incomplete: bool = False, adjust_splits: bool = True,
          repair_spikes: bool = True) -> tuple[pd.DataFrame, Quality]:
    """봉 데이터를 정제하고 (정제된 DataFrame, 품질보고서) 를 돌려줍니다.

    drop_incomplete
        마지막 봉을 버립니다. 오늘 장중의 일봉은 아직 만들어지는 중이라
        고가·저가·종가가 확정값이 아닙니다. 백테스트처럼 '확정된 봉만' 써야 하는
        곳에서 True 로 켭니다. 실매매는 오늘 흐름을 봐야 하므로 기본 False 이고,
        대신 flags["forming_last_bar"] 로 알려줍니다.

    실패해도 예외를 던지지 않습니다. 전처리가 죽어서 매매 루프가 멈추는 것보다,
    원본을 그대로 돌려주고 error 를 남기는 편이 안전합니다.
    """
    q = Quality()
    if df is None or len(df) == 0:
        q.error = "봉 데이터가 비어 있습니다."
        return (df if df is not None else pd.DataFrame()), q

    q.rows_in = len(df)
    try:
        work = _normalize(df)
    except Exception as exc:
        q.error = f"표준화 실패: {type(exc).__name__}: {exc}"
        return df, q

    if work is None or work.empty:
        q.error = "OHLC 컬럼을 찾을 수 없습니다."
        return df, q

    try:
        work = _dedupe(work, q)
        work = _drop_invalid(work, q)
        if work.empty:
            q.error = "유효한 봉이 남지 않았습니다."
            return work, q

        work = _repair_ohlc(work, q)
        if adjust_splits:
            work = _adjust_splits(work, q)
        if repair_spikes:
            work = _repair_spikes(work, q)
        _flag_only(work, timeframe, q)

        if drop_incomplete and len(work) > 1:
            work = work.iloc[:-1]
            q.flags["dropped_last_bar"] = True
    except Exception as exc:
        # 여기까지 왔으면 부분적으로 정제된 상태입니다. 그대로 쓰되 사실을 남깁니다.
        q.error = f"정제 중 오류: {type(exc).__name__}: {exc}"

    q.rows_out = len(work)
    q.ok = q.rows_out > 0
    q.score = _quality_score(q)
    return work, q


def clean_bars(df: pd.DataFrame, timeframe: str = "day", **kwargs) -> pd.DataFrame:
    """clean() 의 DataFrame 만 필요한 곳을 위한 편의 함수.

    품질보고서는 `df.attrs["quality"]` 에 붙여둡니다. pandas 의 attrs 는
    슬라이싱(백테스트의 `bars.iloc[:i]`)을 넘어 유지되므로, 나중에라도
    "이 봉으로 계산한 신호가 어떤 데이터에서 나왔는지" 되짚을 수 있습니다.
    """
    out, q = clean(df, timeframe, **kwargs)
    try:
        out.attrs["quality"] = q.to_dict()
    except Exception:
        pass
    return out


def quality_of(df: pd.DataFrame) -> dict:
    """clean_bars 가 붙여둔 품질보고서를 꺼냅니다 (없으면 빈 dict)."""
    try:
        value = df.attrs.get("quality")
    except AttributeError:
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# 1) 표준화 — 컬럼 이름·자료형·시간축
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Volume": "volume", "시가": "open", "고가": "high", "저가": "low",
    "종가": "close", "거래량": "volume",
}
_DATE_COLUMNS = ("date", "datetime", "time", "timestamp", "일자")


def _normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    out = df.copy()
    out = out.rename(columns={c: _COLUMN_ALIASES.get(c, c) for c in out.columns})

    # 시간축이 컬럼으로 들어온 제공처가 있습니다 (야후 reset_index 경로)
    if not isinstance(out.index, pd.DatetimeIndex):
        for name in _DATE_COLUMNS:
            if name in out.columns:
                out[name] = pd.to_datetime(out[name], errors="coerce")
                out = out.dropna(subset=[name]).set_index(name)
                break

    if "close" not in out.columns:
        return None

    # open/high/low 가 없는 제공처(종가만 주는 경로)는 종가로 채웁니다.
    # 0 으로 채우면 ATR 이 종가만큼 커져서 손절 폭이 터무니없어집니다.
    for col in OHLC:
        if col not in out.columns:
            out[col] = out["close"]
    if "volume" not in out.columns:
        out["volume"] = 0.0

    for col in (*OHLC, "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    keep = [*OHLC, "volume"] + [c for c in out.columns if c not in (*OHLC, "volume")]
    return out[keep]


# ---------------------------------------------------------------------------
# 2) 중복·정렬
# ---------------------------------------------------------------------------

def _dedupe(df: pd.DataFrame, q: Quality) -> pd.DataFrame:
    """같은 시각 봉을 하나로 합치고 시간순으로 정렬합니다.

    합치는 방식은 freqtrade 와 같습니다 — 시가는 첫 값, 고가는 최댓값, 저가는
    최솟값, 종가는 마지막 값. 거래량은 **합이 아니라 최댓값**입니다. 중복은
    "두 번 거래된 것"이 아니라 "같은 봉이 두 번 전달된 것"이기 때문입니다.
    """
    before = len(df)
    if df.index.has_duplicates:
        agg = {"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "max"}
        others = {c: "last" for c in df.columns if c not in agg}
        df = df.groupby(level=0, sort=True).agg({**agg, **others})
        removed = before - len(df)
        if removed:
            q.dropped["중복 봉"] = removed
            q.repairs.append(f"같은 시각 봉 {removed}개를 하나로 합쳤습니다.")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        q.repairs.append("시간 역순으로 들어온 봉을 정렬했습니다.")
    return df


# ---------------------------------------------------------------------------
# 3) 불량 봉 제거
# ---------------------------------------------------------------------------

def _drop_invalid(df: pd.DataFrame, q: Quality) -> pd.DataFrame:
    """종가가 없거나 0 이하인 봉을 버립니다.

    0 원짜리 봉 하나가 로그수익률을 −inf 로 만들고, 그 뒤 모든 통계량이
    NaN 이 됩니다. 0 을 '가격'으로 취급하는 경로는 만들지 않습니다.
    """
    bad_close = df["close"].isna() | (df["close"] <= 0)
    if bad_close.any():
        n = int(bad_close.sum())
        q.dropped["종가 결측·0 이하"] = n
        q.repairs.append(f"종가가 없거나 0 이하인 봉 {n}개를 버렸습니다.")
        df = df[~bad_close]

    if df.empty:
        return df

    # 시가/고가/저가가 비었으면 종가로 메웁니다 (freqtrade 와 같은 처리).
    for col in ("open", "high", "low"):
        missing = df[col].isna() | (df[col] <= 0)
        if missing.any():
            n = int(missing.sum())
            df.loc[missing, col] = df.loc[missing, "close"]
            q.repairs.append(f"{col} 결측 {n}개를 종가로 채웠습니다.")

    negative_volume = df["volume"] < 0
    if negative_volume.any():
        df.loc[negative_volume, "volume"] = 0.0
        q.repairs.append(f"음수 거래량 {int(negative_volume.sum())}개를 0 으로 바꿨습니다.")
    df["volume"] = df["volume"].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# 4) OHLC 정합성
# ---------------------------------------------------------------------------

def _repair_ohlc(df: pd.DataFrame, q: Quality) -> pd.DataFrame:
    """고가 < 종가 같은 모순을 바로잡습니다.

    고가는 정의상 그 봉에서 나온 모든 가격 중 최댓값입니다. 제공처가 고가를
    반올림하거나 서로 다른 소스를 합치면서 이 관계가 깨지면, ATR(진폭)이
    음수 방향으로 왜곡되고 스토캐스틱·윌리엄스가 0으로 나눕니다.
    실제 값을 모르므로 **가장 보수적인 복원**을 씁니다 — 고가는 넷 중 최댓값,
    저가는 넷 중 최솟값. 진폭을 넓히는 방향이라 손절이 헐거워지지 않습니다.
    """
    hi = df[list(OHLC)].max(axis=1)
    lo = df[list(OHLC)].min(axis=1)
    broken = (df["high"] < hi - 1e-12) | (df["low"] > lo + 1e-12)
    if broken.any():
        n = int(broken.sum())
        df.loc[broken, "high"] = hi[broken]
        df.loc[broken, "low"] = lo[broken]
        q.repairs.append(f"OHLC 정합성 위반 {n}개를 고가=최댓값/저가=최솟값으로 복원했습니다.")
        q.flags["ohlc_repaired"] = n
    return df


# ---------------------------------------------------------------------------
# 5) 액면분할 / 병합 역조정
# ---------------------------------------------------------------------------

def _adjust_splits(df: pd.DataFrame, q: Quality) -> pd.DataFrame:
    """액면분할·병합으로 생긴 가격 점프를 과거 구간에 되돌려 반영합니다.

    왜 이것만은 반드시 고쳐야 하는가
        5:1 액면분할은 하루에 −80% 로 보입니다. 이건 손실이 아니라 **가격의
        재정의**입니다. 그대로 두면 ATR 이 다섯 배로 뛰고, 분산비율과 Hurst 가
        "극단적 평균회귀"로 오판하고, 60일 이동평균은 몇 달간 현재가의 다섯 배
        위에 머뭅니다. 즉 지표 전체가 몇 달 동안 조용히 고장납니다.

    오탐을 막는 3중 조건
        1) 로그수익률의 절댓값이 35% 초과 (하루 등락으로는 드문 크기)
        2) 가격 배율이 실제 공시에 쓰이는 분할비의 ±4% 안
        3) 다음 봉에서 되돌아오지 않음 (되돌아오면 분할이 아니라 오틱)
        거래량이 함께 뛰었으면 정황이 보강되지만, 필수 조건으로 두지는
        않습니다 — 거래량을 안 주는 제공처가 있습니다.

    대부분의 제공처는 이미 수정주가를 줍니다. 그래서 이 함수는 보통 아무것도
    하지 않고, 수정이 안 된 소스로 폴백했을 때만 작동합니다.
    """
    if len(df) < 5:
        return df

    close = df["close"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    log_ret = np.diff(np.log(close))

    adjusted = []
    for i in np.flatnonzero(np.abs(log_ret) > SPLIT_RETURN_THRESHOLD):
        bar = i + 1                       # log_ret[i] 는 bar 시점의 수익률
        prev_price, price = close[i], close[bar]
        if price <= 0 or prev_price <= 0:
            continue

        ratio = prev_price / price        # 분할이면 >1, 병합이면 <1
        matched = _match_split_ratio(ratio)
        if matched is None:
            continue

        # 다음 봉에서 그대로 되돌아오면 분할이 아니라 오틱입니다
        if bar + 1 < len(close):
            back = close[bar + 1] / price
            if abs(np.log(back) + log_ret[i]) < abs(log_ret[i]) * SPIKE_REVERT_TOL:
                continue

        hint = ""
        if volume[bar] > 0 and volume[i] > 0:
            vol_ratio = volume[bar] / volume[i]
            expected = matched if matched > 1 else 1 / matched
            if vol_ratio >= min(SPLIT_VOLUME_HINT, expected * 0.5):
                hint = f", 거래량 {vol_ratio:.1f}배"

        # 과거 구간을 현재 기준으로 끌어내립니다(분할) / 끌어올립니다(병합).
        df.iloc[:bar, [df.columns.get_loc(c) for c in OHLC]] /= matched
        df.iloc[:bar, df.columns.get_loc("volume")] *= matched
        adjusted.append((df.index[bar], matched))
        q.repairs.append(
            f"{_stamp(df.index[bar])} 액면{'분할' if matched > 1 else '병합'} "
            f"1:{matched:g} 감지 — 이전 {bar}개 봉을 역조정했습니다{hint}.")

    if adjusted:
        q.flags["splits_adjusted"] = [{"at": _stamp(t), "ratio": r} for t, r in adjusted]
    return df


def _match_split_ratio(ratio: float) -> float | None:
    """관측된 가격 배율이 알려진 분할비 중 하나인가."""
    for known in KNOWN_SPLIT_RATIOS:
        for candidate in (known, 1.0 / known):
            if abs(ratio / candidate - 1.0) <= SPLIT_RATIO_TOL:
                return candidate
    return None


# ---------------------------------------------------------------------------
# 6) 단일 봉 오틱 복원
# ---------------------------------------------------------------------------

def _repair_spikes(df: pd.DataFrame, q: Quality) -> pd.DataFrame:
    """한 봉만 극단으로 갔다가 다음 봉에서 그대로 돌아온 값을 되돌립니다.

    "튀었다"만으로는 고치지 않습니다 — 상한가와 실적 쇼크가 그렇게 생겼습니다.
    **튄 뒤 되돌아왔다**는 조건이 붙어야 데이터 오류입니다. 그래서 판정은
    두 봉을 함께 봅니다.

        r_t 가 로버스트 z 로 8σ 를 넘고
        r_{t+1} 이 r_t 를 65% 이상 되돌리면 -> 오틱

    로버스트 z 는 중앙값·MAD 로 계산합니다. 평균·표준편차를 쓰면 튐 자체가
    표준편차를 키워서 자기 자신을 정상으로 만듭니다.
    """
    if len(df) < 12:
        return df

    close = df["close"].to_numpy(dtype=float).copy()
    log_ret = np.diff(np.log(close))
    z = _robust_z(log_ret)
    if z is None:
        return df

    fixed = []
    for i in np.flatnonzero(np.abs(z) > SPIKE_Z):
        if i + 1 >= len(log_ret):
            continue                       # 마지막 봉은 되돌림을 확인할 수 없습니다
        if np.sign(log_ret[i + 1]) == np.sign(log_ret[i]):
            continue
        if abs(log_ret[i] + log_ret[i + 1]) > abs(log_ret[i]) * SPIKE_REVERT_TOL:
            continue

        bar = i + 1
        # 앞뒤 종가의 기하평균으로 되돌립니다 (로그 공간에서의 선형 보간)
        repaired = float(np.sqrt(close[bar - 1] * close[bar + 1]))
        if not np.isfinite(repaired) or repaired <= 0:
            continue
        old = close[bar]
        df.iloc[bar, df.columns.get_loc("close")] = repaired
        # 고가·저가가 그 튐 때문에 벌어져 있으면 함께 좁힙니다
        row = df.iloc[bar]
        df.iloc[bar, df.columns.get_loc("high")] = max(row["open"], repaired, row["low"])
        df.iloc[bar, df.columns.get_loc("low")] = min(row["open"], repaired, row["high"])
        fixed.append((df.index[bar], old, repaired))

    if fixed:
        q.repairs.append(
            f"되돌아온 단일 봉 튐 {len(fixed)}개를 보정했습니다 "
            f"(예: {_stamp(fixed[0][0])} {fixed[0][1]:,.4g} → {fixed[0][2]:,.4g}).")
        q.flags["spikes_repaired"] = len(fixed)
    return df


def _robust_z(values: np.ndarray) -> np.ndarray | None:
    """중앙값·MAD 기반 로버스트 z. 표본이 모자라거나 MAD 가 0 이면 None."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return None
    median = float(np.median(v))
    mad = float(np.median(np.abs(v - median)))
    if mad <= 0:
        return None
    # 0.6745 = 정규분포에서 MAD 를 표준편차로 환산하는 상수
    return 0.6745 * (np.asarray(values, dtype=float) - median) / mad


# ---------------------------------------------------------------------------
# 7) 고치지 않고 표시만 하는 것들
# ---------------------------------------------------------------------------

def _flag_only(df: pd.DataFrame, timeframe: str, q: Quality):
    """거래정지·유동성·시간 간격처럼 '사실이지만 오류는 아닌' 상태를 기록합니다."""
    n = len(df)
    if n == 0:
        return

    # 거래정지 / 거래 없음 — 0 거래량이면서 고가=저가인 봉.
    # 이 봉들은 변동성을 실제보다 낮게 만들어 손절 폭을 좁힙니다.
    halted = ((df["volume"] <= 0) & (df["high"] <= df["low"] + 1e-12))
    if halted.any():
        q.flags["halted_bars"] = int(halted.sum())
        q.flags["halted_ratio"] = round(float(halted.mean()), 4)

    zero_volume = int((df["volume"] <= 0).sum())
    if zero_volume:
        q.flags["zero_volume_bars"] = zero_volume

    # 시간 간격 — 일봉은 주말·공휴일이 정상이라 '이상'이 아닙니다. 다만 한 달
    # 넘게 비어 있으면 거래정지·상장폐지 직전일 수 있어 알려줍니다.
    if isinstance(df.index, pd.DatetimeIndex) and n >= 2:
        gaps = df.index.to_series().diff().dropna()
        if not gaps.empty:
            biggest = gaps.max()
            q.flags["max_gap_days" if timeframe == "day" else "max_gap_minutes"] = (
                round(biggest.total_seconds() / (86400 if timeframe == "day" else 60), 2))
            if timeframe == "day" and biggest.days >= 30:
                q.flags["long_halt_suspected"] = True

    # 마지막 봉이 아직 만들어지는 중인지 — 확정 여부는 여기서 알 수 없으므로
    # 사실만 남깁니다(장중이면 거의 항상 미완성입니다).
    q.flags["last_bar_at"] = _stamp(df.index[-1])
    q.flags["forming_last_bar"] = True


# ---------------------------------------------------------------------------
# 품질 점수
# ---------------------------------------------------------------------------

def _quality_score(q: Quality) -> float:
    """0~1. 상위 판단이 "이 데이터를 믿고 주문을 낼 것인가"를 정하는 근거입니다.

    감점 폭은 그 결함이 지표를 얼마나 망가뜨리는지에 비례합니다.
      · 삭제 봉    표본이 줄어 통계량이 불안정해집니다
      · 거래정지    변동성을 과소추정해 손절이 좁아집니다 (가장 위험)
      · 수복 흔적   원본이 부정확했다는 신호입니다
    """
    if q.rows_out <= 0:
        return 0.0
    score = 1.0
    dropped = sum(q.dropped.values())
    if q.rows_in > 0:
        score -= 0.5 * min(1.0, dropped / q.rows_in)
    score -= 0.4 * float(q.flags.get("halted_ratio", 0.0))
    score -= 0.05 * min(4, len(q.repairs))
    if q.flags.get("long_halt_suspected"):
        score -= 0.15
    if q.error:
        score -= 0.2
    return float(max(0.0, min(1.0, score)))


def _stamp(value) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M").replace(" 00:00", "")
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# 워밍업 (freqtrade 의 startup_candle_count)
# ---------------------------------------------------------------------------

def warmup_bars(*periods: int, safety: int = 5) -> int:
    """지표가 안정된 값을 내기까지 필요한 최소 봉 개수.

    EMA·ADX 처럼 지수가중을 쓰는 지표는 첫 봉부터 값이 나오지만 **그 값이
    맞지는 않습니다**. 초기값이 남아 있어서, 같은 종목이라도 몇 개부터
    계산했느냐에 따라 다른 신호가 나옵니다. freqtrade 가 startup_candle_count
    만큼 앞의 봉을 잘라 쓰는 이유가 이것입니다.

    지수가중은 이론상 무한 기간을 보므로 관례대로 기간의 3배를 잡습니다
    (3τ 이후 초기값의 영향이 5% 아래로 떨어집니다).
    """
    longest = max([int(p) for p in periods if p] or [0])
    return longest * 3 + safety


def has_enough(df: pd.DataFrame, need: int) -> bool:
    return df is not None and len(df) >= need


def trim_warmup(df: pd.DataFrame, warmup: int) -> pd.DataFrame:
    """앞쪽 warmup 개를 잘라냅니다 — 남는 봉이 없으면 자르지 않습니다.

    자를지 말지를 호출부가 매번 판단하면 조건이 흩어집니다. "자를 여유가
    있을 때만 자른다"는 규칙을 여기 한 곳에 둡니다.
    """
    if df is None or warmup <= 0 or len(df) <= warmup + 10:
        return df
    return df.iloc[warmup:]


__all__ = ["clean", "clean_bars", "quality_of", "Quality", "warmup_bars",
           "has_enough", "trim_warmup", "MIN_TRUSTWORTHY_QUALITY"]
