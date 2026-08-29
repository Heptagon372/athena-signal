"""
탐색 범위 (Universe) — "어디에서 찾을 것인가"
--------------------------------------------
지금까지 스크리너는 **시장 전체**의 거래량순위에서 출발했습니다. 그러면 늘
동전주·테마주가 상위를 채웁니다. 거래량이 폭발하는 종목은 대체로 그런 종목이기
때문입니다. "코스피200 안에서만", "반도체 안에서만", "나스닥100 안에서만"
찾고 싶다는 요구는 여기서 나옵니다.

이 모듈은 그 **범위**만 책임집니다. 범위 안에서 무엇을 살지는 여전히
`engine/recommender.py`(일반)와 `engine/scalping.py`(초단타)가 정합니다.

범위를 얻는 경로 — 시장마다 다릅니다
    한국
      1) 지수 : KIS 거래량순위의 `FID_INPUT_ISCD` 파라미터
               (0001 코스피 / 1001 코스닥 / 2001 코스피200 / 4001 KRX100)
               → 지수 **전체**를 서버가 순위 매겨 줍니다. 가장 정확하고 쌉니다.
      2) ETF : KIS ETF 구성종목시세 (FHKST121600C0)
               → ETF 한 종목만 물어보면 **구성종목 + 시세**가 한 번에 옵니다.
               코스닥150(229200)·반도체(091160)처럼 지수 파라미터가 없는 범위는
               그 지수를 추종하는 ETF 를 대신 물어봅니다.
      섹터도 같은 방식입니다 — 국내에는 섹터 지수 API 가 없지만 섹터 ETF 는
      있으므로, **섹터 = 섹터 ETF 의 구성종목**으로 정의했습니다.
    미국
      1) 나스닥 스크리너 (api.nasdaq.com) — 한 번의 호출로 미국 상장 전 종목의
         시세·거래량·시총·**섹터**가 옵니다. 섹터 필터는 여기서 끝납니다.
      2) 지수 편입 종목 — 나스닥100은 나스닥 공식 API, S&P500은 slickcharts.
         편입 목록은 분기마다 바뀌므로 디스크에 캐시하고 하루 지나면 갱신합니다.

없으면 없다고 말합니다
    구성종목을 못 받아왔는데 조용히 시장 전체로 되돌아가면, 사용자는 "코스피200
    안에서 찾았다"고 믿는데 실제로는 아무 종목이나 나옵니다. 그래서 실패는
    `MemberSet.error` 에 담아 호출부가 화면에 띄우도록 합니다.
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field

from config import CACHE_DIR
from data_sources import http_client

# ---------------------------------------------------------------------------
# 세부시장 (Segment) — "코스피 / 코스닥 / 나스닥"
# ---------------------------------------------------------------------------
# 예전에는 KR / US 두 개였습니다. 매수 대상을 고를 때 코스피와 코스닥은 성격이
# 전혀 다르고(변동성·유동성·세금 체계), 미국도 나스닥과 뉴욕이 다릅니다.
# `region` 은 기존 코드(환율 변환·장시간·세금)가 쓰던 구분을 그대로 유지합니다.

SEGMENTS = {
    "KOSPI":  {"label": "코스피",  "region": "KR", "kis_iscd": "0001"},
    "KOSDAQ": {"label": "코스닥",  "region": "KR", "kis_iscd": "1001"},
    "NASDAQ": {"label": "나스닥",  "region": "US", "nasdaq_exchange": "NASDAQ"},
    # 화면에는 노출하지 않지만, 예전 설정("US")을 손실 없이 풀어내려면 필요합니다.
    "NYSE":   {"label": "뉴욕(NYSE)", "region": "US", "nasdaq_exchange": "NYSE"},
}

# 화면에 체크박스로 내보내는 순서 (사용자가 고르는 3개)
PRIMARY_SEGMENTS = ["KOSPI", "KOSDAQ", "NASDAQ"]

# 예전 설정값 호환. 저장된 설정이 ["KR"] 이면 코스피+코스닥으로 풀어 줍니다 —
# 사용자가 새 화면에서 저장하기 전까지 탐색 범위가 좁아지면 안 됩니다.
LEGACY_SEGMENTS = {
    "KR": ["KOSPI", "KOSDAQ"],
    "US": ["NASDAQ", "NYSE"],
    "KRX": ["KOSPI", "KOSDAQ"],
}


def normalize_segments(markets) -> list[str]:
    """설정에 저장된 시장 목록을 세부시장 코드로 펼칩니다 (중복 제거, 순서 유지)."""
    out: list[str] = []
    for raw in (markets or []):
        token = str(raw or "").strip().upper()
        for seg in LEGACY_SEGMENTS.get(token, [token]):
            if seg in SEGMENTS and seg not in out:
                out.append(seg)
    return out or ["KOSPI", "KOSDAQ"]


def regions_of(segments: list[str]) -> list[str]:
    """세부시장 → 지역(KR/US). 환율·장시간 판단은 아직 지역 단위입니다."""
    out = []
    for seg in segments:
        region = SEGMENTS.get(seg, {}).get("region")
        if region and region not in out:
            out.append(region)
    return out


def segment_label(segment: str) -> str:
    return SEGMENTS.get(segment, {}).get("label", segment)


# ---------------------------------------------------------------------------
# 유니버스 카탈로그
# ---------------------------------------------------------------------------

@dataclass
class Universe:
    """탐색 범위 하나.

    `kis_iscd`  : KIS 거래량순위에 그대로 넘길 수 있는 지수 코드. 이게 있으면
                  지수 **전체**를 순위 매겨 받을 수 있어 가장 좋습니다.
    `etf`       : 구성종목을 물어볼 국내 ETF 종목코드. 상위 30종목(비중순)까지
                  옵니다 — API 응답 한도이며, 비중 상위가 곧 유동성 상위입니다.
    `us_index`  : 미국 지수 편입 종목 목록 키 (NASDAQ100 / SP500).
    `us_sector` : 나스닥 스크리너의 sector 값 (Technology, Finance …).
    """
    key: str
    label: str
    group: str                    # index | sector
    region: str                   # KR | US
    segments: list                # 이 범위가 실제로 걸치는 세부시장
    note: str = ""
    kis_iscd: str = ""
    etf: str = ""
    us_index: str = ""
    us_sector: str = ""

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "group": self.group,
                "region": self.region, "segments": list(self.segments),
                "note": self.note}


# 국내 섹터 ETF 는 2026-08 기준 거래대금·구성종목 수를 보고 골랐습니다.
# TOP10 류(구성 10종목)보다 구성종목이 넓은 쪽을 우선했습니다 — 탐색 범위가
# 10종목이면 "찾는다"고 할 게 없기 때문입니다.
CATALOG: list[Universe] = [
    # --- 국내 지수 ---
    Universe("KOSPI200", "코스피200", "index", "KR", ["KOSPI"],
             "시가총액·유동성 상위 200종목. 거래량 상위 30은 매 갱신, "
             "나머지는 순환 계산으로 전 종목을 평가합니다.",
             kis_iscd="2001", etf="069500"),
    Universe("KOSDAQ150", "코스닥150", "index", "KR", ["KOSDAQ"],
             "코스닥 대표 150종목. 비중 상위 30은 매 갱신, "
             "나머지는 순환 계산으로 전 종목을 평가합니다.",
             etf="229200"),
    Universe("KRX100", "KRX100", "index", "KR", ["KOSPI", "KOSDAQ"],
             "코스피·코스닥 통합 우량 100종목.",
             kis_iscd="4001"),

    # --- 국내 섹터 (섹터 ETF 구성종목) ---
    Universe("KR_SEMI", "반도체", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "KODEX 반도체 구성종목", etf="091160"),
    Universe("KR_BATTERY", "2차전지", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "KODEX 2차전지산업 구성종목", etf="305720"),
    Universe("KR_AUTO", "자동차", "sector", "KR", ["KOSPI"],
             "KODEX 자동차 구성종목", etf="091180"),
    Universe("KR_BANK", "은행", "sector", "KR", ["KOSPI"],
             "KODEX 은행 구성종목", etf="091170"),
    Universe("KR_BROKER", "증권", "sector", "KR", ["KOSPI"],
             "KODEX 증권 구성종목", etf="102970"),
    Universe("KR_HEALTH", "바이오·헬스케어", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "TIGER 헬스케어 구성종목", etf="227540"),
    Universe("KR_SOFTWARE", "인터넷·소프트웨어", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "TIGER 소프트웨어 구성종목", etf="157490"),
    Universe("KR_GAME", "게임", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "KODEX 게임산업 구성종목", etf="300950"),
    Universe("KR_MEDIA", "미디어·엔터", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "TIGER 미디어컨텐츠 구성종목", etf="228810"),
    Universe("KR_SHIP", "조선", "sector", "KR", ["KOSPI"],
             "TIGER 조선TOP10 구성종목", etf="494670"),
    Universe("KR_DEFENSE", "방산·우주항공", "sector", "KR", ["KOSPI"],
             "PLUS K방산 구성종목", etf="449450"),
    Universe("KR_NUCLEAR", "원자력", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "HANARO 원자력iSelect 구성종목", etf="434730"),
    Universe("KR_CONSTRUCT", "건설", "sector", "KR", ["KOSPI"],
             "KODEX 건설 구성종목", etf="117700"),
    Universe("KR_STEEL", "철강·금속", "sector", "KR", ["KOSPI"],
             "KODEX 철강 구성종목", etf="117680"),
    Universe("KR_CHEM", "화학·에너지", "sector", "KR", ["KOSPI"],
             "KODEX 에너지화학 구성종목", etf="117460"),
    Universe("KR_REIT", "리츠·부동산", "sector", "KR", ["KOSPI"],
             "TIGER 리츠부동산인프라 구성종목", etf="329200"),
    Universe("KR_DIVIDEND", "고배당", "sector", "KR", ["KOSPI", "KOSDAQ"],
             "PLUS 고배당주 구성종목", etf="161510"),

    # --- 미국 지수 ---
    Universe("NASDAQ100", "나스닥100", "index", "US", ["NASDAQ"],
             "나스닥 대표 100종목 (나스닥 공식 목록).", us_index="NASDAQ100"),
    Universe("SP500", "S&P 500", "index", "US", ["NASDAQ", "NYSE"],
             "미국 대표 500종목. 뉴욕·나스닥에 걸쳐 있습니다.", us_index="SP500"),
]

# 미국 섹터는 나스닥 스크리너가 돌려주는 값을 그대로 씁니다 — 우리가 임의로
# 분류하지 않습니다. 표시용 한글 이름만 여기서 붙입니다.
US_SECTOR_LABELS = {
    "Technology": "기술",
    "Finance": "금융",
    "Health Care": "헬스케어",
    "Consumer Discretionary": "경기소비재",
    "Consumer Staples": "필수소비재",
    "Industrials": "산업재",
    "Energy": "에너지",
    "Utilities": "유틸리티",
    "Real Estate": "부동산",
    "Basic Materials": "소재",
    "Telecommunications": "통신",
    "Miscellaneous": "기타",
}

for _sector, _label in US_SECTOR_LABELS.items():
    CATALOG.append(Universe(
        key="US_" + re.sub(r"[^A-Z]", "", _sector.upper()),
        label=_label, group="sector", region="US", segments=["NASDAQ", "NYSE"],
        note=f"나스닥 스크리너 섹터 분류 · {_sector}", us_sector=_sector))

_BY_KEY = {u.key: u for u in CATALOG}


def get(key: str) -> Universe | None:
    return _BY_KEY.get(str(key or "").strip().upper())


def normalize_pools(value) -> list[str]:
    """설정에 저장된 탐색 범위를 **목록**으로 펼칩니다 (중복 제거, 순서 유지).

    받는 모양이 셋입니다 — 예전 설정이 문자열 하나였기 때문입니다.
        ""                       범위 지정 없음 (시장 전체)
        "KOSPI200"               예전 단일 선택
        "KOSPI200,KOSDAQ150"     콤마로 이어붙인 값 (URL·수동 편집)
        ["KOSPI200", "SP500"]    지금 화면이 저장하는 모양

    `normalize_segments` 와 같은 규약입니다: 어떤 경로로 들어온 값이든 여기
    한 곳을 지나면 같은 모양이 됩니다. 카탈로그에 없는 키도 **버리지 않고**
    그대로 둡니다 — 조용히 지우면 오타를 낸 사람은 왜 결과가 시장 전체인지
    알 수 없습니다. 알 수 없는 키는 screener.scan 이 경고로 남깁니다.
    """
    if isinstance(value, str):
        raw = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    elif value is None:
        raw = []
    else:
        raw = [value]
    out: list[str] = []
    for item in raw:
        key = str(item or "").strip().upper()
        if key and key not in out:
            out.append(key)
    return out


def describe_many(keys) -> list[dict]:
    """고른 범위들의 설명 — 화면이 "지금 어디를 보고 있는지"를 적을 때 씁니다."""
    return [u.to_dict() for u in (get(k) for k in normalize_pools(keys)) if u]


def list_universes(segments: list[str] = None) -> list[dict]:
    """화면 드롭다운용 목록. 고른 세부시장과 겹치는 범위만 돌려줍니다.

    코스피만 켜 놓고 '나스닥100'을 고를 수 있으면 결과가 0건으로 나오고,
    사용자는 이유를 알 수 없습니다.
    """
    if not segments:
        return [u.to_dict() for u in CATALOG]
    chosen = set(normalize_segments(segments))
    return [u.to_dict() for u in CATALOG if chosen & set(u.segments)]


# ---------------------------------------------------------------------------
# 캐시 (구성종목은 분기마다 바뀝니다 — 매번 받을 이유가 없습니다)
# ---------------------------------------------------------------------------

_mem: dict[str, tuple[float, object]] = {}
_mem_lock = threading.Lock()

MEMBER_TTL = 6 * 3600.0        # 지수 편입 목록 — 하루에 몇 번이면 충분
TABLE_TTL = 300.0              # 미국 스크리너 시세 표 — 5분


def _cached(key: str, ttl: float, build):
    now = time.time()
    with _mem_lock:
        hit = _mem.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = build()
    if value is not None:
        with _mem_lock:
            _mem[key] = (now, value)
    return value


def _disk_path(name: str):
    return CACHE_DIR / f"universe_{name}.json"


def _disk_load(name: str, max_age_sec: float):
    path = _disk_path(name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    saved_at = float(payload.get("saved_at") or 0)
    fresh = (time.time() - saved_at) < max_age_sec
    return payload.get("value"), fresh


def _disk_save(name: str, value):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _disk_path(name).write_text(
            json.dumps({"saved_at": time.time(), "value": value}, ensure_ascii=False),
            encoding="utf-8")
    except OSError:
        pass


def clear_cache():
    with _mem_lock:
        _mem.clear()


# ---------------------------------------------------------------------------
# 결과 자료형
# ---------------------------------------------------------------------------

@dataclass
class Member:
    """유니버스 구성종목 한 줄. 시세는 소스가 줄 때만 채워집니다."""
    key: str
    name: str = ""
    market: str = ""              # KOSPI | KOSDAQ | US
    segment: str = ""             # KOSPI | KOSDAQ | NASDAQ | NYSE
    weight: float | None = None   # 유니버스 내 비중(%)
    sector: str = ""
    price: float = 0.0
    change_rate: float = 0.0
    volume: float = 0.0
    trading_value: float = 0.0
    market_cap: float = 0.0
    prev_volume: float | None = None
    listed_shares: float | None = None


@dataclass
class MemberSet:
    """구성종목 + **어디서 왔는지 / 왜 실패했는지**."""
    universe: str = ""
    members: list = field(default_factory=list)
    source: str = ""
    error: str = ""
    partial: bool = False          # 전체가 아니라 상위 일부만 받은 경우

    @property
    def codes(self) -> set:
        return {m.key for m in self.members}

    def __len__(self):
        return len(self.members)


# ---------------------------------------------------------------------------
# 한국 — ETF 구성종목시세
# ---------------------------------------------------------------------------

KIS_ETF_COMPONENT_PATH = "/uapi/etfetn/v1/quotations/inquire-component-stock-price"
KIS_ETF_COMPONENT_TR = "FHKST121600C0"


def _num(value, default=0.0) -> float:
    try:
        return float(str(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return default


def kr_entry(code: str) -> dict:
    """상장 마스터(KRX 상장법인목록, 로컬 캐시)에서 이 종목의 등록 정보.

    ETF 구성종목 응답에도, KIS 거래량순위 응답에도 시장 구분과 업종이 없습니다.
    둘 다 이 마스터에 이미 들어 있고 디스크 캐시라 호출 비용이 없습니다.
    """
    try:
        from data_sources import symbol_registry
        return symbol_registry.get_krx_master()["by_code"].get(str(code)) or {}
    except Exception:
        return {}


def kr_market_of(code: str, default: str = "KOSPI") -> str:
    """종목코드 → KOSPI / KOSDAQ.

    코스닥 종목을 코스피로 표시하면 화면의 시장 배지가 전부 틀리고, 코스피/코스닥
    중 하나만 켜고 찾을 때 엉뚱한 종목이 걸러집니다.
    """
    market = kr_entry(code).get("market") or ""
    return market if market in ("KOSPI", "KOSDAQ") else default


# 순환 평가의 모집단 — 코스피·코스닥 **전 종목**.
# 마스터가 통째로 깨진 채(빈 목록·몇 건) 돌아오면, 화면은 "시장 전체를 훑는다"고
# 말하는데 실제로는 몇 종목만 보게 됩니다. 최소 건수를 두고 그보다 적으면 빈
# 목록을 돌려주어, 호출부가 예전 경로(순위 API 상위 30)로만 돌게 합니다.
KR_MASTER_MIN = 1_000


def kr_listed(segments: list[str] = None, skip_spac: bool = True) -> list[dict]:
    """코스피·코스닥 상장 **전 종목** 명단 (KRX 상장법인목록, 로컬 캐시 3일).

    거래량순위 API 는 상위 30행까지만 줍니다. "지금 거래가 터지는 종목"을 찾는
    데는 그걸로 충분하지만, **시장 전체를 순환하며 팩터를 계산하려면** 순위와
    무관한 전체 명단이 필요합니다. 그 명단이 여기 있습니다 — 이미 시장 판정
    (`kr_market_of`)에 쓰고 있던 마스터라 추가 네트워크 호출이 없습니다.

    시세는 들어 있지 않습니다(명단일 뿐입니다). 팩터 계산이 어차피 일봉을
    받으므로, 여기서 종목마다 시세를 붙이면 수천 번의 중복 호출이 됩니다.

    제외되는 것
        ETF·ETN  상장법인목록은 '회사'만 수록해 애초에 없습니다.
        코넥스    세부시장(SEGMENTS)에 없어 선택 자체가 불가능합니다.
        스팩      합병 발표 전까지 공모가 근처에서 거의 움직이지 않습니다.
                  모멘텀·추세·상대강도가 전부 무의미한 종목에 계산 창을
                  내주면, 그만큼 실제로 움직이는 종목이 늦게 평가됩니다.
    """
    chosen = set(normalize_segments(segments) if segments else ["KOSPI", "KOSDAQ"])
    chosen &= {"KOSPI", "KOSDAQ"}
    if not chosen:
        return []

    try:
        from data_sources import symbol_registry
        entries = symbol_registry.get_krx_master().get("entries") or []
    except Exception:
        return []
    if len(entries) < KR_MASTER_MIN:
        return []

    # KIND 응답에는 같은 행이 두 번 오는 경우가 있습니다 (2026-08 실측 42건).
    # 마스터의 by_code 는 dict 라 이 사실이 가려져 있지만, 명단을 그대로 돌리면
    # 그 종목이 계산 창을 두 칸 먹고 추천 목록에도 두 번 오릅니다.
    out, seen = [], set()
    for entry in entries:
        code = entry.get("code") or ""
        market = entry.get("market")
        if market not in chosen or code in seen:
            continue
        name = entry.get("name") or ""
        if skip_spac and ("스팩" in name or "기업인수목적" in name):
            continue
        seen.add(code)
        out.append({"key": code, "name": name, "market": market,
                    "segment": market, "sector": entry.get("sector") or ""})
    return out


def etf_components(etf_code: str) -> MemberSet:
    """국내 ETF 한 종목의 구성종목 + 시세 (KIS 공식 API, 1회 호출).

    응답은 비중 상위 30종목까지입니다. 구성종목이 30개를 넘는 ETF(코스피200 등)
    에서는 `partial=True` 로 표시해 화면이 "전체가 아니다"를 말할 수 있게 합니다.
    """
    from data_sources import kis_client

    result = MemberSet(universe=etf_code, source="KIS ETF 구성종목시세")
    if not kis_client.is_configured():
        result.error = "KIS 키가 없어 ETF 구성종목을 조회할 수 없습니다."
        return result

    headers = kis_client.auth_headers(KIS_ETF_COMPONENT_TR)
    if not headers:
        result.error = "KIS 접근토큰 발급 실패."
        return result

    kis_client.throttle()
    status, data, raw = http_client.get_full(
        kis_client.base_url() + KIS_ETF_COMPONENT_PATH, headers=headers,
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": str(etf_code),
                "FID_COND_SCR_DIV_CODE": "11216"}, timeout=12)

    if status != 200 or not isinstance(data, dict) or str(data.get("rt_cd")) != "0":
        reason = (data or {}).get("msg1") if isinstance(data, dict) else raw[:80]
        result.error = f"ETF({etf_code}) 구성종목 조회 실패 — {reason or f'HTTP {status}'}"
        return result

    total = int(_num((data.get("output1") or {}).get("etf_cnfg_issu_cnt"), 0))
    for row in (data.get("output2") or []):
        code = str(row.get("stck_shrn_iscd") or "").strip()
        price = _num(row.get("stck_prpr"))
        if not code or price <= 0:
            continue
        volume = _num(row.get("acml_vol"))
        # 전일거래량 = 누적거래량 - 전일대비거래량. 거래량 급증 판정의 분모입니다.
        prev_vol = volume - _num(row.get("prdy_vrss_vol"))
        # 시가총액은 '억원' 단위로 옵니다 → 상장주식수를 역산해 회전율을 씁니다.
        cap = _num(row.get("hts_avls")) * 1e8
        entry = kr_entry(code)
        market = entry.get("market") if entry.get("market") in ("KOSPI", "KOSDAQ") else "KOSPI"
        result.members.append(Member(
            key=code, name=row.get("hts_kor_isnm") or code,
            market=market, segment=market, sector=entry.get("sector") or "",
            weight=_num(row.get("etf_cnfg_issu_rlim")) or None,
            price=price, change_rate=_num(row.get("prdy_ctrt")),
            volume=volume, trading_value=_num(row.get("acml_tr_pbmn")),
            market_cap=cap,
            prev_volume=prev_vol if prev_vol > 0 else None,
            listed_shares=(cap / price) if cap > 0 and price > 0 else None,
        ))

    if not result.members:
        result.error = f"ETF({etf_code}) 구성종목이 비어 있습니다 (장 시작 전이면 정상)."
    result.partial = bool(total and len(result.members) < total)
    if result.partial:
        result.source += f" (구성 {total}종목 중 비중 상위 {len(result.members)}종목)"
    return result


# ---------------------------------------------------------------------------
# 미국 — 나스닥 스크리너 / 지수 편입 목록
# ---------------------------------------------------------------------------

NASDAQ_SCREENER = "https://api.nasdaq.com/api/screener/stocks"
NASDAQ_INDEX_LIST = "https://api.nasdaq.com/api/quote/list-type/{name}"
SLICKCHARTS = "https://www.slickcharts.com/{name}"

NASDAQ_HEADERS = {"Accept": "application/json, text/plain, */*"}


def _money(text) -> float:
    """'$313.33' / '1,525,210' / '41,226,590,866.00' → float"""
    return _num(str(text or "").replace("$", ""))


def us_screener_rows(exchange: str = "") -> list[dict]:
    """미국 상장 종목 표 — 시세·거래량·시총·**섹터**가 한 번에 옵니다.

    종목마다 시세를 따로 물어보면 수천 번 호출해야 합니다. 이 한 번으로 끝나기
    때문에 미국 쪽 섹터·지수 탐색이 현실적으로 가능해졌습니다.
    """
    def build():
        data = http_client.get_json(
            NASDAQ_SCREENER, headers=NASDAQ_HEADERS,
            params={"tableonly": "false", "limit": "0", "download": "true",
                    **({"exchange": exchange} if exchange else {})}, timeout=40)
        rows = ((data or {}).get("data") or {}).get("rows")
        return rows if isinstance(rows, list) and rows else None

    return _cached(f"us_screener:{exchange or 'ALL'}", TABLE_TTL, build) or []


def _nasdaq100_symbols() -> list[str]:
    data = http_client.get_json(NASDAQ_INDEX_LIST.format(name="nasdaq100"),
                                headers=NASDAQ_HEADERS, timeout=20)
    rows = (((data or {}).get("data") or {}).get("data") or {}).get("rows") or []
    return [str(r.get("symbol") or "").strip().upper() for r in rows if r.get("symbol")]


def _slickcharts_symbols(name: str) -> list[str]:
    """slickcharts 편입 목록 (S&P500 은 공식 무료 API 가 없습니다).

    HTML 을 긁는 경로라 언제든 깨질 수 있습니다. 그래서 실패는 예외 대신 빈
    목록으로 두고, 호출부가 디스크 캐시(만료본 포함)로 버티게 했습니다.
    """
    html = http_client.get_text(SLICKCHARTS.format(name=name),
                                headers={"Accept": "text/html"}, timeout=25)
    if not html:
        return []
    seen, out = set(), []
    for symbol in re.findall(r'href="/symbol/([A-Z][A-Z.\-]{0,6})"', html):
        if symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


US_INDEX_SOURCES = {
    "NASDAQ100": [("나스닥 공식 목록", _nasdaq100_symbols),
                  ("slickcharts", lambda: _slickcharts_symbols("nasdaq100"))],
    "SP500": [("slickcharts", lambda: _slickcharts_symbols("sp500"))],
}

# 편입 종목이 이보다 적게 오면 파싱이 깨진 것으로 봅니다 (지수 규모의 절반).
US_INDEX_MIN = {"NASDAQ100": 60, "SP500": 300}


def us_index_symbols(index_key: str) -> tuple[list[str], str, str]:
    """(심볼 목록, 소스, 오류). 실패하면 만료된 디스크 캐시라도 씁니다.

    편입 종목은 분기 리밸런싱 때만 바뀝니다. 어제 목록으로 오늘 찾는 것은
    거의 문제가 없지만, 목록을 통째로 못 받아서 시장 전체로 되돌아가는 것은
    사용자가 고른 범위를 배신하는 일입니다.
    """
    index_key = str(index_key or "").upper()
    cache_name = f"us_index_{index_key.lower()}"

    def build():
        for label, fetch in US_INDEX_SOURCES.get(index_key, []):
            try:
                symbols = fetch()
            except Exception:
                symbols = []
            if len(symbols) >= US_INDEX_MIN.get(index_key, 20):
                _disk_save(cache_name, {"symbols": symbols, "source": label})
                return {"symbols": symbols, "source": label}
        return None

    payload = _cached(f"us_index:{index_key}", MEMBER_TTL, build)
    if payload:
        return payload["symbols"], payload["source"], ""

    stale, fresh = _disk_load(cache_name, MEMBER_TTL)
    if stale and stale.get("symbols"):
        age = "" if fresh else " (최신 목록을 못 받아 저장본 사용)"
        return stale["symbols"], stale.get("source", "저장본") + age, ""
    return [], "", f"{index_key} 편입 종목 목록을 받지 못했습니다."


def us_members(universe: Universe = None, exchange: str = "") -> MemberSet:
    """미국 구성종목 + 시세. `universe` 를 비우면 그 거래소 상장 전 종목.

    지수든 섹터든 시세는 결국 같은 스크리너 표에서 옵니다. 지수는 편입 목록으로
    걸러내고, 섹터는 표의 sector 열로 걸러냅니다 — 범위를 안 주면 아무것도
    걸러내지 않을 뿐, 만드는 방법은 같습니다.
    """
    result = MemberSet(universe=universe.key if universe else "")
    rows = us_screener_rows(exchange)
    if not rows:
        result.error = "나스닥 스크리너 응답이 비었습니다 (차단되었거나 네트워크 문제)."
        return result

    allow: set | None = None
    if universe is not None and universe.us_index:
        symbols, source, error = us_index_symbols(universe.us_index)
        if error:
            result.error = error
            return result
        allow = set(symbols)
        result.source = f"나스닥 스크리너 + {universe.label} 편입목록({source})"
    elif universe is not None and universe.us_sector:
        result.source = "나스닥 스크리너 (섹터 분류)"
    else:
        result.source = "나스닥 스크리너"

    sector = universe.us_sector if universe else ""
    segment = exchange if exchange in SEGMENTS else "NASDAQ"
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or (allow is not None and symbol not in allow):
            continue
        if sector and (row.get("sector") or "") != sector:
            continue
        price = _money(row.get("lastsale"))
        if price <= 0:
            continue
        volume = _money(row.get("volume"))
        cap = _money(row.get("marketCap"))
        result.members.append(Member(
            key=symbol, name=row.get("name") or symbol,
            market="US", segment=segment,
            sector=row.get("sector") or "",
            price=price, change_rate=_num(str(row.get("pctchange") or "").rstrip("%")),
            volume=volume, trading_value=price * volume,
            market_cap=cap,
            # 상장주식수 = 시총 ÷ 현재가 (국내 ETF 경로와 같은 역산).
            # 이게 있어야 회전율(거래량 ÷ 상장주식수)이 전 종목에 공짜로 생기고,
            # 후보 순위를 거래대금(= 늘 같은 대형주) 말고 회전율로 매길 수 있습니다.
            listed_shares=(cap / price) if cap > 0 else None,
        ))

    if not result.members:
        where = f"{universe.label} 범위에서" if universe else "선택한 거래소에서"
        result.error = (f"{where} 종목을 찾지 못했습니다 "
                        f"(선택한 시장에 해당 종목이 없을 수 있습니다).")
    return result


# ---------------------------------------------------------------------------
# 공용 진입점
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 지수 구성종목 전체 명단 (네이버)
# ---------------------------------------------------------------------------
# KIS 는 거래량순위 30행 · ETF 구성종목 30종목이 한계입니다. "코스피200 을
# 골랐으면 200종목 전부"를 순환 평가에 태우려면 명단이 따로 필요하고, 네이버
# 모바일 API 가 지수 편입종목을 페이지로 줍니다. 명단(종목코드)만 쓰고 시세는
# 버립니다 — 순환 평가의 팩터 계산이 어차피 종목마다 일봉을 받습니다.
NAVER_INDEX_CODES = {"KOSPI200": "KPI200", "KOSDAQ150": "KQI150"}
NAVER_INDEX_URL = "https://m.stock.naver.com/api/index/{code}/enrollStocks"
NAVER_INDEX_HEADERS = {"Referer": "https://m.stock.naver.com/",
                       "Accept": "application/json"}
NAVER_INDEX_PAGE = 60           # pageSize 는 100 부터 400 응답 (2026-08 확인)


def index_member_codes(universe_key: str, segments: list[str] = None) -> list[str]:
    """지수 **전체** 구성종목의 종목코드 명단. 못 받으면 빈 목록.

    지원하는 범위(코스피200·코스닥150)가 아니면 빈 목록이고, 호출 쪽은 기존
    경로(순위 API·ETF 구성종목)로만 동작합니다. 네트워크가 죽어 있으면 디스크
    캐시(3일)로 버팁니다 — 편입 종목은 분기에 한 번 바뀌는 정보입니다.
    """
    naver = NAVER_INDEX_CODES.get(str(universe_key or "").strip().upper())
    if not naver:
        return []

    def build():
        codes = []
        for page in range(1, 11):      # 200종목 = 60행 × 4쪽 — 10쪽이면 넉넉
            rows = http_client.get_json(
                NAVER_INDEX_URL.format(code=naver),
                params={"page": page, "pageSize": NAVER_INDEX_PAGE},
                headers=NAVER_INDEX_HEADERS)
            if not isinstance(rows, list) or not rows:
                break
            codes += [str(r.get("itemCode") or "").strip() for r in rows
                      if str(r.get("itemCode") or "").strip()]
            if len(rows) < NAVER_INDEX_PAGE:
                break
        if codes:
            _disk_save(f"idx_{naver}", codes)
            return codes
        return None                    # 실패를 6시간 동안 캐시하지 않습니다

    codes = _cached(f"idx_members:{naver}", MEMBER_TTL, build)
    if not codes:
        codes, _ = _disk_load(f"idx_{naver}", 3 * 86400)
    codes = list(dict.fromkeys(codes or []))
    if segments:
        chosen = set(normalize_segments(segments))
        # 마스터가 모르는 코드는 뺍니다 — 신형 알파벳 코드('0126Z0' 류)는 상장
        # 마스터에 없어 어차피 팩터 해석이 안 되고, 남기면 "코스피만" 골랐는데
        # 시장 미상 종목이 섞입니다.
        codes = [c for c in codes if kr_entry(c).get("market") in chosen]
    return codes


def members(universe_key: str, segments: list[str] = None) -> MemberSet:
    """유니버스 키 → 구성종목. 시장별 경로 선택을 여기서 감춥니다."""
    universe = get(universe_key)
    if universe is None:
        return MemberSet(universe=str(universe_key),
                         error=f"알 수 없는 탐색 범위: {universe_key}")

    if universe.region == "KR":
        if not universe.etf:
            # 지수 파라미터(kis_iscd)만 있는 범위는 구성종목 목록 자체가 없습니다.
            # 이 경우 스크리너가 순위 API 에 지수코드를 넘겨 직접 훑습니다.
            return MemberSet(universe=universe.key, source="KIS 지수 순위",
                             error="이 범위는 구성종목 목록 대신 지수 순위로 조회합니다.")
        found = etf_components(universe.etf)
        found.universe = universe.key
        if segments and found.members:
            chosen = set(normalize_segments(segments))
            kept = [m for m in found.members if m.segment in chosen]
            if not kept:
                # 조용히 0건으로 두면 "종목이 없다"고 오해합니다. 실제로는
                # 범위와 시장 선택이 어긋난 것이고, 시장을 켜면 바로 나옵니다.
                labels = "·".join(segment_label(s) for s in sorted(chosen))
                found.error = (f"{universe.label} 구성종목 중 {labels}에 상장된 종목이 "
                               f"없습니다 — 시장 선택을 확인하세요.")
            found.members = kept
        return found

    chosen = normalize_segments(segments) if segments else ["NASDAQ", "NYSE"]
    wanted = [s for s in chosen if SEGMENTS.get(s, {}).get("region") == "US"]
    if not wanted:
        return MemberSet(universe=universe.key,
                         error="미국 시장이 선택되어 있지 않습니다.")

    merged = MemberSet(universe=universe.key)
    for segment in wanted:
        part = us_members(universe, exchange=SEGMENTS[segment]["nasdaq_exchange"])
        merged.members += part.members
        merged.source = merged.source or part.source
        if part.error and not merged.members:
            merged.error = part.error
    if merged.members:
        merged.error = ""
    return merged


def describe(universe_key: str) -> dict:
    """화면에 "지금 어디서 찾고 있는지"를 한 줄로 보여주기 위한 설명."""
    universe = get(universe_key)
    if universe is None:
        return {}
    return universe.to_dict()
