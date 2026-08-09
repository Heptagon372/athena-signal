"""
내생 시계 (Intrinsic Time · Directional Change)
-----------------------------------------------
가격을 **벽시계가 아니라 가격 기하학으로** 샘플링합니다.

수록 항목
    DirectionalChangeRunner  단일 δ 러너 — O(1) 상태 갱신, 사전계산 트리거 가격
    MultiScaleDC             여러 δ 를 동시에 돌리는 뱅크
    run_series               봉/틱 계열 일괄 처리 (백테스트용)
    SurpriseLiquidity        엔트로피 기반 유동성 지표 (원본 버그 3개 수정)
    inventory_ladder         재고 의존 임계값·사이즈 사다리

출처와 라이선스
    Golub, Glattfelder & Olsen (2017), *The Alpha Engine*,
    in *High Performance Computing in Finance* (SSRN 2951348).
    **논문에서 독립 재구현했습니다.** 공개된 참조 구현(AntonVonGolub/Code)은
    GPL-3.0 이므로 코드를 옮기지 않았습니다. 알고리즘과 수식은 저작권 대상이
    아닙니다. 어차피 그 구현은 컴파일되지 않고(javac 21 기준 4개 에러) 손익목표
    로직이 통째로 빠져 있으며 아래 수학 버그들이 있어서, 깨끗한 재구현이
    코드 품질 면에서도 낫습니다.

왜 이것이 이 저장소에 필요한가 — 다중 주기를 하나로 흡수합니다
    이 저장소는 일봉(추천·백테스트), 분봉, 그리고 페니주 초단타의 **틱** 을
    동시에 다룹니다. 지금은 각각 다른 코드 경로입니다.

    DC 이벤트 시계는 그 셋을 **같은 피처 공간** 으로 사상합니다. 이벤트를
    정의하는 것이 시간이 아니라 가격 움직임의 크기(δ)이기 때문입니다.
    δ=25bp 로 두면 일봉에서는 며칠에 한 번, 틱에서는 초에 몇 번 이벤트가
    나지만 **피처의 의미는 같습니다.** 자산군별 분기 없이 δ 만 조정합니다.

★ 이 구현의 핵심 — 사전계산된 트리거 가격
    보통의 DC 탐지기는 매 틱마다 "지금 로그비가 δ 를 넘었나?" 를 계산합니다.
    이 구현은 반대로 **"다음 이벤트가 정확히 어느 가격에서 일어나는가"** 를
    미리 계산해 캐시합니다(`expected_dc_level`). 결과적으로

      1) 매 틱 비교가 `price >= trigger` 한 번의 분기로 끝납니다 (O(1), 무할당)
      2) 백테스트에서 봉 전체를 건너뛸 수 있습니다 — `if bar.high < trigger: continue`
      3) **지정가 주문의 큐 포지션을 접수 시점부터 누적** 할 수 있습니다.
         다음 이벤트 가격을 미리 아니까 그 자리에 주문을 미리 걸어 둡니다.

    3번이 실전에서 가장 값어치 있습니다. 참조 구현의 치명적 결함이 정확히
    거기였습니다 — 지정가가 가격에 닿으면 **무조건 전량 즉시 체결** 로 처리하고
    (큐 없음·부분체결 없음·거부 없음), 매 틱 무료로 지정가를 재조정했습니다.
    실제로는 매 재조정이 취소/정정이고 **큐 맨 뒤로** 갑니다. 이 전략은 주문을
    정확히 DC 레벨, 즉 시장이 곧 뚫고 지나갈 자리에 놓기 때문에 이 가정의
    왜곡이 특히 큽니다. **논문 수익성의 상당 부분이 여기서 나왔을 수 있습니다.**
    그래서 이 모듈은 이벤트 시계만 제공하고 체결은 다루지 않습니다.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from engine.quant import _norm_cdf

# 이벤트 코드
NONE, DC_UP, DC_DOWN, OS_UP, OS_DOWN = 0, 1, -1, 2, -2


@dataclass
class DCState:
    """러너의 전체 상태. 전부 스칼라라 복사가 쌉니다."""
    mode: int = -1                # +1 상승 국면(extreme 은 고점) / −1 하락 국면
    extreme: float = 0.0          # 현재 국면의 극값
    reference: float = 0.0        # 마지막 OS 갱신 기준가
    last_dc_price: float = 0.0    # 마지막 DC 가 일어난 가격
    expected_dc_level: float = 0.0
    expected_os_level: float = 0.0
    os_length: float = 0.0        # log(extreme / last_dc_price) / δ
    n_dc: int = 0
    n_os: int = 0
    ticks_since_dc: int = 0
    initialized: bool = False


class DirectionalChangeRunner:
    """단일 스케일 δ 의 DC/OS 러너.

    Parameters
    ----------
    delta_up, delta_down : 로그수익률 단위 임계 (bp 가 아니라 log 비율).
        `from_bps()` 생성자를 쓰면 bp 로 줄 수 있습니다.
    delta_star_up, delta_star_down : 오버슛 임계. 기본은 δ 와 같습니다
        (논문의 δ* = δ 설정). 재고 사다리에서 비대칭으로 바뀝니다.

    `run(price)` 는 이벤트 코드를 돌려줍니다:
        0 없음 / +1 상승 DC / −1 하락 DC / +2 상승 OS / −2 하락 OS
    """

    def __init__(self, delta_up: float = 0.0025, delta_down: float = None,
                 delta_star_up: float = None, delta_star_down: float = None):
        self.delta_up = float(delta_up)
        self.delta_down = float(delta_down if delta_down is not None else delta_up)
        self.d_star_up = float(delta_star_up if delta_star_up is not None else self.delta_up)
        self.d_star_down = float(delta_star_down if delta_star_down is not None
                                 else self.delta_down)
        self.state = DCState()

    @classmethod
    def from_bps(cls, delta_bps: float, **kw):
        """δ 를 bp 로 주는 편의 생성자. 25bp → 0.0025."""
        return cls(delta_up=float(delta_bps) / 1e4, **kw)

    # -- 트리거 가격 사전계산 (이 클래스의 핵심) --------------------------

    def _refresh_dc_level(self) -> None:
        s = self.state
        if s.mode == -1:
            # 하락 국면 — 저점에서 delta_up 만큼 오르면 상승 DC
            s.expected_dc_level = math.exp(math.log(s.extreme) + self.delta_up)
        else:
            s.expected_dc_level = math.exp(math.log(s.extreme) - self.delta_down)

    def _refresh_os_level(self) -> None:
        s = self.state
        if s.mode == -1:
            s.expected_os_level = math.exp(math.log(s.reference) - self.d_star_down)
        else:
            s.expected_os_level = math.exp(math.log(s.reference) + self.d_star_up)

    def _refresh_os_length(self) -> None:
        """오버슛 길이 = log(extreme / 직전 DC 가격) / δ.

        참조 구현의 리라이트에서 **빠져 있던** 값입니다(원본 code.java 에는
        있습니다). 오버슛이 δ 의 몇 배까지 갔는지를 재는 무차원 량이라
        스케일 간 비교가 됩니다. 이게 없으면 "지금 추세가 얼마나 늘어났나" 를
        말할 수 없습니다.
        """
        s = self.state
        if s.last_dc_price > 0 and s.extreme > 0:
            delta = self.delta_up if s.mode == 1 else self.delta_down
            if delta > 0:
                s.os_length = abs(math.log(s.extreme / s.last_dc_price)) / delta

    # -- 상태 갱신 --------------------------------------------------------

    def run(self, price: float) -> int:
        """가격 하나를 넣고 이벤트 코드를 돌려줍니다. O(1), 할당 없음."""
        p = float(price)
        if not (p > 0) or not math.isfinite(p):
            return NONE
        s = self.state

        if not s.initialized:
            s.extreme = s.reference = s.last_dc_price = p
            s.mode = -1
            s.initialized = True
            self._refresh_dc_level()
            self._refresh_os_level()
            return NONE

        s.ticks_since_dc += 1

        if s.mode == -1:
            if p >= s.expected_dc_level:
                # 상승 방향 전환
                s.mode = 1
                s.extreme = s.reference = s.last_dc_price = p
                s.n_dc += 1
                s.ticks_since_dc = 0
                s.os_length = 0.0
                self._refresh_dc_level()
                self._refresh_os_level()
                return DC_UP
            if p < s.extreme:
                s.extreme = p
                self._refresh_dc_level()
                self._refresh_os_length()
                if p <= s.expected_os_level:
                    s.reference = p
                    s.n_os += 1
                    self._refresh_os_level()
                    return OS_DOWN
        else:
            if p <= s.expected_dc_level:
                s.mode = -1
                s.extreme = s.reference = s.last_dc_price = p
                s.n_dc += 1
                s.ticks_since_dc = 0
                s.os_length = 0.0
                self._refresh_dc_level()
                self._refresh_os_level()
                return DC_DOWN
            if p > s.extreme:
                s.extreme = p
                self._refresh_dc_level()
                self._refresh_os_length()
                if p >= s.expected_os_level:
                    s.reference = p
                    s.n_os += 1
                    self._refresh_os_level()
                    return OS_UP
        return NONE

    # -- 피처 -------------------------------------------------------------

    def features(self, price: float) -> dict:
        """현재 상태에서 뽑은 피처. **전부 t 시점 정보만 씁니다.**"""
        s = self.state
        p = float(price)
        if not s.initialized or p <= 0:
            return {}
        return {
            "dc_mode": int(s.mode),
            "os_length": round(s.os_length, 4),
            # 다음 국면 전환까지의 정규화 거리 — 연속 피처라 밴딧 컨텍스트에 적합
            "dist_to_flip": round((s.expected_dc_level - p) / p, 6),
            "ticks_since_dc": int(s.ticks_since_dc),
            "n_dc": int(s.n_dc),
            "n_os": int(s.n_os),
            "expected_dc_level": s.expected_dc_level,
            "expected_os_level": s.expected_os_level,
        }


def run_series(prices, delta_bps: float = 25.0) -> dict:
    """계열 전체를 한 번에 돌립니다 (백테스트·연구용).

    Returns
    -------
    dict — `events` 는 각 시점의 이벤트 코드, `os_length`/`dist_to_flip` 은
        시점별 피처 배열입니다. 길이는 입력과 같고 **인과적** 입니다
        (t 시점 값은 t 까지의 가격만 씁니다).
    """
    p = np.asarray(prices, dtype=float)
    n = len(p)
    runner = DirectionalChangeRunner.from_bps(delta_bps)
    events = np.zeros(n, dtype=np.int8)
    os_len = np.full(n, np.nan, dtype=float)
    dist = np.full(n, np.nan, dtype=float)
    mode = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if not (p[i] > 0) or not math.isfinite(p[i]):
            continue
        events[i] = runner.run(p[i])
        f = runner.features(p[i])
        if f:
            os_len[i] = f["os_length"]
            dist[i] = f["dist_to_flip"]
            mode[i] = f["dc_mode"]

    return {
        "events": events, "os_length": os_len, "dist_to_flip": dist,
        "mode": mode,
        "n_dc": int(runner.state.n_dc), "n_os": int(runner.state.n_os),
        "delta_bps": float(delta_bps),
        # 이벤트가 너무 적으면 그 δ 는 이 계열에 안 맞습니다
        "events_per_bar": round((runner.state.n_dc + runner.state.n_os) / max(n, 1), 5),
    }


class MultiScaleDC:
    """여러 δ 를 동시에 돌리는 뱅크.

    단일 δ 는 그 크기의 움직임만 봅니다. 뱅크로 돌리면 "작은 스케일에서는
    상승인데 큰 스케일에서는 하락" 같은 **스케일 간 구조** 가 보입니다.
    그것이 국면 컨텍스트로 쓸 만한 몇 안 되는 연속 피처입니다.

    기본 δ = 25 / 50 / 100 bp. 일봉이면 더 크게(100~300), 틱이면 더 작게
    (5~25) 잡으세요. 기준은 **이벤트가 봉 수 대비 너무 드물지도 잦지도 않은**
    지점입니다 — `run_series()` 의 `events_per_bar` 로 확인하세요.
    """

    def __init__(self, deltas_bps=(25.0, 50.0, 100.0)):
        self.deltas = tuple(float(d) for d in deltas_bps)
        self.runners = [DirectionalChangeRunner.from_bps(d) for d in self.deltas]

    def update(self, price: float) -> dict:
        events = [r.run(price) for r in self.runners]
        feats = {}
        modes, os_lengths = [], []
        for d, r, ev in zip(self.deltas, self.runners, events):
            f = r.features(price)
            key = f"d{int(d)}"
            if f:
                feats[f"{key}_mode"] = f["dc_mode"]
                feats[f"{key}_os_length"] = f["os_length"]
                feats[f"{key}_dist_to_flip"] = f["dist_to_flip"]
                modes.append(f["dc_mode"])
                os_lengths.append(f["os_length"])
            feats[f"{key}_event"] = ev

        if modes:
            # 스케일 합의도 — +1 이면 모든 스케일이 상승, 0 이면 갈림
            feats["scale_agreement"] = round(float(np.mean(modes)), 4)
            # 오버슛 비대칭 — 큰 스케일이 더 늘어나 있으면 추세 성숙
            if len(os_lengths) >= 2:
                feats["os_asymmetry"] = round(float(os_lengths[-1] - os_lengths[0]), 4)
        return feats


# ---------------------------------------------------------------------------
# 서프라이즈 유동성 — 원본의 수학 버그 3개를 고쳐 재구현
# ---------------------------------------------------------------------------

@dataclass
class SurpriseLiquidity:
    """엔트로피 기반 "서프라이즈" 유동성 지표.

    발상: 오버슛이 무기억이라면 각 내생 이벤트가 OS 일 확률은
    `p = exp(−δ*/δ)` 입니다. 실제 관측된 이벤트 열의 서프라이즈
    `−ln p_obs` 를 지수평활해서 기대 엔트로피 H1 과 비교합니다.
    시장이 "평소보다 예측 가능한" 패턴을 보이면 유동성이 낮다고 읽습니다.

    ★ 참조 구현의 버그 3개를 고쳤습니다 (LocalLiquidity.java)

      1) **EMA 방향이 뒤집혀 있었습니다.**
         원본 `surp = alphaWeight*관측 + (1−alphaWeight)*surp`, alphaWeight=0.9615.
         새 관측에 96%, 누적 평균에 4% 를 줍니다 — 기억 길이가 ~26 에서 ~1 로
         붕괴합니다. 같은 저장소의 code.java 에는 올바른 방향으로 쓰여 있습니다.
      2) **H2 의 부호가 틀렸습니다.** 원본 `p(ln p)² − (1−p)(ln(1−p))² − H1²`.
         분산 정의상 두 항 모두 `+` 여야 합니다.
      3) **초기값이 0 이었습니다.** H1 로 초기화해야 워밍업 구간에서 지표가
         가짜 극단값을 내지 않습니다.

      이 두 버그(1·2)의 결과로 원본의 유동성 지표는 사실상 **"직전 이벤트가
      DC 냐 OS 냐" 의 이진 함수** 로 붕괴하고, 논문의 유동성 기반 포지션
      사이징이 "DC 후 풀사이즈 / OS 후 10%" 로 퇴화합니다.

      추가로 원본의 하드코딩 상수(0.08338161, 2.525729)를 쓰지 않고
      **자기 δ*/δ 에서 재유도** 합니다.
    """
    delta: float = 0.0025
    delta_star: float = 0.0025
    alpha: float = 26.0            # EMA 기억 길이 (이벤트 수)
    surp: float = field(default=None)
    n_events: int = 0

    def __post_init__(self):
        p = math.exp(-self.delta_star / self.delta) if self.delta > 0 else 0.5
        self.p = min(max(p, 1e-12), 1 - 1e-12)
        q = 1.0 - self.p
        self.h1 = -(self.p * math.log(self.p) + q * math.log(q))
        # ★ 두 항 모두 + (원본은 두 번째가 −)
        self.h2 = (self.p * math.log(self.p) ** 2
                   + q * math.log(q) ** 2 - self.h1 ** 2)
        self.w = math.exp(-2.0 / (self.alpha + 1.0))
        if self.surp is None:
            self.surp = self.h1          # ★ 원본은 0

    def update(self, event: int) -> dict | None:
        """내생 이벤트 하나를 반영합니다. event 는 run() 의 반환값."""
        if event == NONE:
            return None
        is_os = abs(event) == 2
        p_obs = self.p if is_os else (1.0 - self.p)
        surprise = -math.log(max(p_obs, 1e-12))
        # ★ 누적 평균에 w, 새 관측에 (1−w) — 원본은 반대
        self.surp = self.w * self.surp + (1.0 - self.w) * surprise
        self.n_events += 1

        if self.h2 <= 0:
            return {"surprise": self.surp, "liquidity": None}
        z = math.sqrt(self.alpha) * (self.surp - self.h1) / math.sqrt(self.h2)
        return {
            "surprise": round(self.surp, 6),
            "h1": round(self.h1, 6),
            "liquidity": round(1.0 - _norm_cdf(z), 6),
            "z": round(z, 4),
            "n_events": self.n_events,
            "warming_up": self.n_events < self.alpha,
        }


# ---------------------------------------------------------------------------
# 재고 사다리
# ---------------------------------------------------------------------------

def inventory_ladder(inventory: float, base_delta: float = 0.0025) -> dict:
    """재고 크기에 따라 진입 임계와 사이즈를 조절합니다.

    논문의 템플릿을 그대로 씁니다. 재고가 쌓일수록 **추가 진입은 어렵게,
    청산은 쉽게** 만듭니다 — 비대칭이 핵심입니다.

        |inv| < 15   → (δ_진입, δ_청산) = (1.00δ, 1.00δ),  단위크기 1
        15 ≤ |inv| < 30 → (0.75δ, 1.50δ),                 단위크기 1/2
        |inv| ≥ 30   → (0.50δ, 2.00δ),                    단위크기 1/4

    간결하고 검증된 재고 편향 템플릿입니다. `engine/microstructure.py` 의
    Avellaneda-Stoikov 유보가격이 같은 일을 연속적으로 하므로, 둘을 **동시에
    쓰지 마세요** — 재고 페널티를 두 번 매기게 됩니다.
    """
    inv = abs(float(inventory))
    if inv < 15:
        enter, exit_, size = 1.00, 1.00, 1.0
    elif inv < 30:
        enter, exit_, size = 0.75, 1.50, 0.5
    else:
        enter, exit_, size = 0.50, 2.00, 0.25
    return {
        "delta_enter": base_delta * enter,
        "delta_exit": base_delta * exit_,
        "size_fraction": size,
        "inventory": float(inventory),
        "tier": "low" if inv < 15 else ("mid" if inv < 30 else "high"),
    }
