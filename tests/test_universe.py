# -*- coding: utf-8 -*-
"""
탐색 범위(유니버스) QA
----------------------
"코스피200 안에서 찾는다"고 화면에 써 놓고 실제로는 아무 종목이나 나오는 것이
이 기능에서 가장 나쁜 실패입니다. 그래서 **범위를 벗어난 종목이 섞이지 않는지**
와 **못 찾았을 때 이유를 말하는지** 를 집중적으로 봅니다.

    python tests/test_universe.py            # 전체 (네트워크 사용)
    python tests/test_universe.py --offline  # 네트워크 없이 순수 로직만

offline 검사는 실제 API 를 부르지 않습니다. 시장이 닫혀 있거나 KIS 키가 없어도
깜빡이면 안 되는 부분(코드 매핑·필터·설정 호환)만 봅니다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from data_sources import screener, universe

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = ""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 오프라인 — 네트워크 없이 확인할 수 있는 것
# ---------------------------------------------------------------------------

def test_segment_compat():
    """예전 설정("KR"/"US")이 세부시장으로 손실 없이 펼쳐지는가.

    여기가 깨지면 이미 설정을 저장해 둔 사용자의 탐색 시장이 조용히 좁아집니다.
    """
    print("\n[세부시장 호환]")
    check("KR → 코스피+코스닥", universe.normalize_segments(["KR"]) == ["KOSPI", "KOSDAQ"])
    check("US → 나스닥+뉴욕", universe.normalize_segments(["US"]) == ["NASDAQ", "NYSE"])
    check("새 코드는 그대로", universe.normalize_segments(["KOSDAQ"]) == ["KOSDAQ"])
    check("중복 제거", universe.normalize_segments(["KR", "KOSPI"]) == ["KOSPI", "KOSDAQ"])
    check("빈 값이면 국내 기본", universe.normalize_segments([]) == ["KOSPI", "KOSDAQ"])
    check("모르는 값은 무시", universe.normalize_segments(["ZZZ"]) == ["KOSPI", "KOSDAQ"])
    check("지역 판정", universe.regions_of(["KOSPI", "NASDAQ"]) == ["KR", "US"])


def test_catalog():
    """카탈로그가 화면이 기대하는 모양인가."""
    print("\n[카탈로그]")
    keys = {u.key for u in universe.CATALOG}
    check("주요 지수가 있다",
          {"KOSPI200", "KOSDAQ150", "NASDAQ100", "SP500"} <= keys)
    check("국내 섹터가 10개 이상",
          len([u for u in universe.CATALOG if u.group == "sector" and u.region == "KR"]) >= 10)
    check("미국 섹터가 10개 이상",
          len([u for u in universe.CATALOG if u.group == "sector" and u.region == "US"]) >= 10)

    bad = [u.key for u in universe.CATALOG
           if not (u.kis_iscd or u.etf or u.us_index or u.us_sector)]
    check("모든 범위에 수집 경로가 있다", not bad, f"경로 없음: {bad}")

    bad = [u.key for u in universe.CATALOG
           if any(s not in universe.SEGMENTS for s in u.segments)]
    check("segments 값이 전부 유효", not bad, f"이상: {bad}")

    # 화면은 고른 시장과 겹치는 범위만 보여줍니다 — 겹치지 않는 것을 고를 수 있으면
    # 결과가 0건으로 나오고 사용자는 이유를 알 수 없습니다.
    kospi_only = {u["key"] for u in universe.list_universes(["KOSPI"])}
    check("코스피만 켜면 나스닥100이 안 보인다", "NASDAQ100" not in kospi_only)
    check("코스피만 켜도 코스피200은 보인다", "KOSPI200" in kospi_only)
    nasdaq_only = {u["key"] for u in universe.list_universes(["NASDAQ"])}
    check("나스닥만 켜면 코스닥150이 안 보인다", "KOSDAQ150" not in nasdaq_only)

    check("모르는 키는 None", universe.get("NOPE") is None)
    check("키는 대소문자 무관", universe.get("kospi200") is not None)


def test_kr_full_market():
    """국내 전 종목 명단이 순환 평가의 모집단으로 쓸 만한 모양인가.

    이게 비거나 코넥스·스팩이 섞이면, 화면은 "코스피·코스닥 전 종목을 돈다"고
    말하는데 실제로는 엉뚱한 종목에 계산 창을 내주게 됩니다.
    """
    print("\n[국내 전 종목 명단]")
    rows = universe.kr_listed()
    check("명단이 충분히 크다", len(rows) >= universe.KR_MASTER_MIN, f"{len(rows)}종목")
    markets = {r["market"] for r in rows}
    check("코스피·코스닥만 있다", markets == {"KOSPI", "KOSDAQ"}, str(markets))
    check("코넥스가 없다", "KONEX" not in markets)
    check("스팩이 없다", not [r for r in rows if "스팩" in r["name"]])
    check("종목코드 6자리", all(len(r["key"]) == 6 and r["key"].isdigit() for r in rows))
    check("코드 중복 없음", len({r["key"] for r in rows}) == len(rows))

    kospi = universe.kr_listed(["KOSPI"])
    kosdaq = universe.kr_listed(["KOSDAQ"])
    check("코스피만 고르면 코스피만", {r["market"] for r in kospi} == {"KOSPI"})
    check("코스닥만 고르면 코스닥만", {r["market"] for r in kosdaq} == {"KOSDAQ"})
    check("둘을 합치면 전체", len(kospi) + len(kosdaq) == len(rows))
    check("예전 설정 KR 도 풀린다", len(universe.kr_listed(["KR"])) == len(rows))
    check("미국만 고르면 빈 목록", universe.kr_listed(["NASDAQ"]) == [])


def test_rotation_window():
    """순환이 실제로 **앞으로 나아가는가** (같은 자리에서 맴돌지 않는가).

    계산 실패 종목을 창에서 빼지 않으면 매 회전 같은 종목이 앞자리를 차지해
    시장 전체 커버가 영원히 끝나지 않습니다.
    """
    print("\n[순환 평가 창]")
    from engine.autotrade import rotate_window

    pool = [f"{i:06d}" for i in range(10)]
    ident = lambda k: k

    window, extras, skipped = rotate_window(pool, ident, 3, set(), set())
    check("첫 회전은 앞에서 3개", window == pool[:3])
    check("캐시가 없으면 순위에 올릴 것도 없다", extras == [])

    cached = set(window)
    window2, extras2, _ = rotate_window(pool, ident, 3, cached, set())
    check("다음 회전은 그 다음 3개", window2 == pool[3:6])
    check("이미 계산한 것은 캐시로 순위에", extras2 == pool[:3])

    # 앞 두 종목이 영구 실패 — 여기서 멈추면 안 됩니다
    window3, _, skipped3 = rotate_window(pool, ident, 3, set(), {pool[0], pool[1]})
    check("실패 종목은 창을 차지하지 않는다", window3 == pool[2:5], str(window3))
    check("건너뛴 수를 보고한다", skipped3 == 2)

    full, _, _ = rotate_window(pool, ident, 3, set(pool), set())
    check("전부 계산됐으면 새로 계산할 것이 없다", full == [])


def test_scope_mismatch():
    """범위와 시장이 어긋나면 **이유를 말하고** 0건이어야 합니다.

    조용히 0건이면 사용자는 "종목이 없다"고 읽지만, 실제로는 시장 하나만
    켜면 바로 나옵니다. 더 나쁜 것은 조용히 시장 전체로 되돌아가는 것입니다.
    """
    print("\n[범위·시장 불일치]")
    result = screener.scan(markets=["NASDAQ"], kr_price=(1000, 2_000_000),
                           us_price=(3.0, 2000.0), limit=5, use_cache=False,
                           universe="KOSPI200")
    check("후보 0건", len(result.candidates) == 0)
    check("이유를 설명한다", any("코스피200" in e for e in result.errors),
          " / ".join(result.errors) or "설명 없음")

    result = screener.scan(markets=["KOSPI"], kr_price=(1000, 2_000_000),
                           us_price=(3.0, 2000.0), limit=5, use_cache=False,
                           universe="ZZZ_NOPE")
    check("모르는 범위는 알리고 전체로", any("알 수 없는" in e for e in result.errors))


def test_scalp_config():
    """초단타 설정이 새 값·옛 값 모두에서 안전하게 정리되는가."""
    print("\n[초단타 설정 정규화]")
    from engine import scalping
    cfg = scalping.clamp_config({"markets": ["KR"], "universe_pool": "kospi200"})
    check("옛 시장값이 펼쳐진다", cfg["markets"] == ["KOSPI", "KOSDAQ"], str(cfg["markets"]))
    check("범위 키가 대문자로", cfg["universe_pool"] == "KOSPI200")

    cfg = scalping.clamp_config({"universe_pool": "존재하지않는범위"})
    check("모르는 범위는 비운다", cfg["universe_pool"] == "")
    check("비운 사실을 알린다", any("탐색 범위" in c for c in cfg.get("_clamped", [])))

    check("기본값은 시장 전체", scalping.DEFAULT_SCALP["universe_pool"] == "")


# ---------------------------------------------------------------------------
# 온라인 — 실제 API 로 범위가 지켜지는지
# ---------------------------------------------------------------------------

def test_kr_universes():
    print("\n[국내 범위 — 실제 조회]")
    from data_sources import kis_client
    if not kis_client.is_configured():
        print("  SKIP KIS 키가 없어 국내 범위 조회를 건너뜁니다.")
        return

    found = universe.members("KOSDAQ150", ["KOSDAQ"])
    check("코스닥150 구성종목을 받는다", len(found) > 0, found.error)
    if len(found):
        check("전부 코스닥", all(m.market == "KOSDAQ" for m in found.members),
              str({m.market for m in found.members}))
        check("시세가 채워져 있다", all(m.price > 0 for m in found.members))
        check("비중이 있다", any(m.weight for m in found.members))
        check("상장주식수를 역산했다", all(m.listed_shares for m in found.members))
        check("일부만 받았음을 표시", found.partial)

    found = universe.members("KR_SEMI")
    check("반도체 섹터를 받는다", len(found) > 0, found.error)
    if len(found):
        codes = found.codes
        check("SK하이닉스가 반도체에 있다", "000660" in codes)

    # 시장 선택과 범위가 어긋나면 조용히 0건이 아니라 이유가 있어야 합니다
    found = universe.members("KOSDAQ150", ["KOSPI"])
    check("코스피만 켜면 코스닥150은 비고 이유가 있다",
          len(found) == 0 and bool(found.error), found.error)


def test_us_universes():
    print("\n[미국 범위 — 실제 조회]")
    found = universe.members("NASDAQ100", ["NASDAQ"])
    check("나스닥100 편입 종목을 받는다", len(found) >= 90, f"{len(found)}종목 {found.error}")
    if len(found):
        check("전부 나스닥", all(m.segment == "NASDAQ" for m in found.members))
        check("섹터가 채워져 있다",
              sum(1 for m in found.members if m.sector) > len(found) * 0.8)
        check("시세가 채워져 있다", all(m.price > 0 for m in found.members))

    found = universe.members("SP500", ["NASDAQ", "NYSE"])
    check("S&P500 편입 종목을 받는다", len(found) >= 400, f"{len(found)}종목 {found.error}")

    found = universe.members("US_TECHNOLOGY", ["NASDAQ"])
    check("기술 섹터를 받는다", len(found) > 50, f"{len(found)}종목 {found.error}")
    if len(found):
        check("전부 Technology 섹터",
              all(m.sector == "Technology" for m in found.members))


def test_scan_stays_in_scope():
    """스캔 결과가 고른 시장·범위를 벗어나지 않는가 (이 기능의 핵심 약속)."""
    print("\n[스캔이 범위를 지키는가]")
    from data_sources import kis_client

    if kis_client.is_configured():
        result = screener.scan(markets=["KOSDAQ"], kr_price=(1000, 2_000_000),
                               us_price=(3.0, 2000.0), limit=10, use_cache=False)
        check("코스닥만 골랐을 때 후보가 있다", len(result.candidates) > 0,
              " / ".join(result.errors))
        if result.candidates:
            check("코스피가 섞이지 않는다",
                  all(c.market == "KOSDAQ" for c in result.candidates),
                  str({c.market for c in result.candidates}))

        result = screener.scan(markets=["KOSDAQ"], kr_price=(1000, 2_000_000),
                               us_price=(3.0, 2000.0), limit=10, use_cache=False,
                               universe="KOSDAQ150")
        check("코스닥150 범위에서 후보가 있다", len(result.candidates) > 0,
              " / ".join(result.errors))
        if result.candidates:
            allow = universe.members("KOSDAQ150", ["KOSDAQ"]).codes
            outside = [c.key for c in result.candidates if c.key not in allow]
            check("범위 밖 종목이 없다", not outside, f"밖: {outside}")
    else:
        print("  SKIP KIS 키가 없어 국내 스캔을 건너뜁니다.")

    result = screener.scan(markets=["NASDAQ"], kr_price=(1000, 2_000_000),
                           us_price=(3.0, 2000.0), limit=10, use_cache=False,
                           universe="NASDAQ100")
    check("나스닥100 범위에서 후보가 있다", len(result.candidates) > 0,
          " / ".join(result.errors))
    if result.candidates:
        allow = set(universe.us_index_symbols("NASDAQ100")[0])
        outside = [c.key for c in result.candidates if c.key not in allow]
        check("범위 밖 종목이 없다", not outside, f"밖: {outside}")
        check("원화 환산이 되어 있다",
              all(c.price_krw > c.price for c in result.candidates))


def main():
    offline = "--offline" in sys.argv
    print("=" * 70)
    print("탐색 범위(유니버스) QA" + (" — offline" if offline else ""))
    print("=" * 70)

    test_segment_compat()
    test_catalog()
    test_scalp_config()
    test_kr_full_market()      # 로컬 캐시(상장법인목록)만 봅니다
    test_rotation_window()
    if not offline:
        test_scope_mismatch()
        test_kr_universes()
        test_us_universes()
        test_scan_stays_in_scope()

    print("\n" + "=" * 70)
    print(f"통과 {len(PASS)} · 실패 {len(FAIL)}")
    if FAIL:
        for name in FAIL:
            print(f"  ✗ {name}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
