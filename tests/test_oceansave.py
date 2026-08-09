# -*- coding: utf-8 -*-
"""세이브(SAVE) 크롤러 QA
------------------------
    python tests/test_oceansave.py            # 전체 (실서버 스모크 포함)
    python tests/test_oceansave.py --offline  # 네트워크 없이 파싱/필터 로직만

오프라인 검사는 API 응답을 모킹해 파싱·티커 재검증·실패 흡수를 봅니다.
실서버 스모크는 saveticker.com 공개 API가 여전히 열려 있는지 확인합니다
(스키마가 바뀌면 여기서 먼저 깨져야 운영 중에 조용히 빈 뉴스가 되지 않습니다).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

OFFLINE = "--offline" in sys.argv
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


from data_sources import http_client, oceansave_crawler as oc

# ---------------------------------------------------------------------------
# 1. 오프라인 — 응답 파싱과 티커 재검증
# ---------------------------------------------------------------------------
print("=" * 70)
print("  1. 파싱/필터 (모킹)")
print("=" * 70)

SAMPLE = {
    "news_list": [
        {   # NVDA 태그가 실제로 달린 정상 항목
            "id": "188888", "title": "엔비디아, 사상 최대 분기 실적 발표",
            "source": "reuters", "created_at": "2026-08-08T00:22:17.485947Z",
            "is_deleted": False,
            "tickers": [{"symbol": "NVDA", "name": "NVIDIA Corporation"}],
        },
        {   # 태그 없는 항목 — 서버가 모르는 티커에 전체 피드를 돌려줄 때의 모습
            "id": "188889", "title": "키이우서 강력한 폭발음",
            "source": "financial-juice", "created_at": "2026-08-08T00:10:00Z",
            "is_deleted": False, "tickers": [],
        },
        {   # 삭제 표시 항목 — 걸러져야 함
            "id": "188890", "title": "엔비디아 삭제된 기사",
            "source": "reuters", "created_at": "2026-08-08T00:05:00Z",
            "is_deleted": True,
            "tickers": [{"symbol": "NVDA", "name": "NVIDIA Corporation"}],
        },
    ],
    "total_count": 3, "page": 1, "page_size": 20,
}

_real_get_json = http_client.get_json
http_client.get_json = lambda *a, **k: SAMPLE
try:
    items = oc.get_news("NVDA", limit=10)
    check("태깅된 항목만 통과 (3건 중 1건)", len(items) == 1,
          f"-> {len(items)}건")
    if items:
        it = items[0]
        check("제목 파싱", it.title == "엔비디아, 사상 최대 분기 실적 발표")
        check("source 에 SAVE 표기 (reasoning.py 속보 태그 조건)",
              "SAVE" in it.source and "reuters" in it.source, f"-> {it.source}")
        check("URL 생성", it.url == "https://saveticker.com/news/188888")
        check("한국어 감성사전 적용 (사상 최대=+)", it.sentiment_score > 0,
              f"-> {it.sentiment_score:+.2f} {it.matched_keywords}")
        # UTC 00:22 -> 로컬 naive. 기대값을 timestamp 산술로 독립 계산해 비교
        utc = datetime(2026, 8, 8, 0, 22, 17, 485947, tzinfo=timezone.utc)
        expected = datetime.fromtimestamp(utc.timestamp())
        check("UTC -> 로컬 naive 변환", it.published_at == expected,
              f"-> {it.published_at}")

    # 한국 종목코드: 전체 피드가 내려와도 태그 재검증으로 전부 걸러져야 함
    check("한국 종목코드는 빈 리스트", oc.get_news("005930") == [])

    feed = oc.get_market_feed(limit=10)
    check("전체 피드는 삭제 항목만 제외 (3건 중 2건)", len(feed) == 2,
          f"-> {len(feed)}건")

    # 카테고리 피드 payload (대시보드 패널용)
    feed = oc.feed_payload("breaking", limit=10)
    check("카테고리 피드 payload 생성", len(feed) == 2, f"-> {len(feed)}건")
    if feed:
        p = feed[0]
        check("payload 에 배지/라벨/티커 포함",
              p["badge"] == "블룸버그 정보" and "labels" in p and "tickers" in p,
              f"-> labels={p['labels']} tickers={p['tickers']}")
    check("모르는 카테고리는 빈 리스트", oc.feed_payload("없는카테고리") == [])

    # 리포트: 쿠키 없으면 호출 자체를 안 하고 빈 리스트
    import os as _os
    _saved_cookie = _os.environ.pop("SAVETICKER_COOKIE", None)
    called = []
    http_client.get_json = lambda *a, **k: called.append(1) or SAMPLE
    check("리포트는 쿠키 없으면 미호출", oc.feed_payload("report") == [] and not called)
    check("PDF 도 쿠키 없으면 미호출", oc.fetch_report_pdf("abc") is None)

    # 쿠키가 있을 때의 리포트 파싱 (스키마: reports[])
    _os.environ["SAVETICKER_COOKIE"] = "dummy=1"
    REPORT_SAMPLE = {"reports": [
        {"id": "abc123", "title": "2026년 08월 07일 (금)", "content": "",
         "created_at": "2026-08-07T22:01:01.017000+09:00",
         "author_name": "오선", "has_pdf": True, "view_count": 3057,
         "tag_names": []},
        {"id": "", "title": "id 없는 항목", "has_pdf": True},   # 걸러져야 함
    ]}
    http_client.get_json = lambda *a, **k: REPORT_SAMPLE
    reports = oc.feed_payload("report", limit=5)
    check("리포트 파싱 (2건 중 1건)", len(reports) == 1, f"-> {len(reports)}건")
    if reports:
        rp = reports[0]
        check("리포트 제목에 날짜 포함",
              rp["title"] == "SAVE 마감 리포트 — 2026년 08월 07일 (금)", f"-> {rp['title']}")
        check("리포트 링크는 우리 서버 PDF 중계 경로",
              rp["url"] == "/api/save/report/abc123/pdf", f"-> {rp['url']}")
        check("리포트 감성점수는 0 (제목이 날짜뿐)", rp["sentiment"] == 0.0)
        check("리포트 배지", rp["badge"] == "블룸버그 정보")
    if _saved_cookie is not None:
        _os.environ["SAVETICKER_COOKIE"] = _saved_cookie
    else:
        _os.environ.pop("SAVETICKER_COOKIE", None)

    http_client.get_json = lambda *a, **k: None
    check("API 실패는 빈 리스트로 흡수", oc.get_news("NVDA") == [])
finally:
    http_client.get_json = _real_get_json

# ---------------------------------------------------------------------------
# 2. 실서버 스모크 — 공개 API가 여전히 열려 있는가
# ---------------------------------------------------------------------------
if not OFFLINE:
    print()
    print("=" * 70)
    print("  2. 실서버 스모크 (saveticker.com)")
    print("=" * 70)
    raw = oc.fetch_raw_items(page_size=5)
    check("전체 피드 응답", len(raw) > 0, f"-> {len(raw)}건")
    if raw:
        first = raw[0]
        check("필수 필드 존재",
              all(k in first for k in ("id", "title", "source", "created_at")),
              f"-> {sorted(first.keys())[:6]}...")
        print(f"       최신: {first.get('created_at', '')[:19]} | "
              f"{first.get('source')} | {str(first.get('title'))[:40]}")
    nvda = oc.get_news("NVDA", limit=5)
    check("NVDA 태그 뉴스", len(nvda) > 0, f"-> {len(nvda)}건")
    for n in nvda[:3]:
        print(f"       {n.published_at:%m-%d %H:%M} | {n.title[:44]} | {n.sentiment_score:+.2f}")

    # 카테고리 피드 4종 + 리포트(쿠키 있을 때만)
    for cat, label in (("top", "주요뉴스"), ("breaking", "속보"),
                       ("reuters", "로이터"), ("news", "뉴스")):
        rows = oc.feed_payload(cat, limit=5)
        check(f"{label}({cat}) 피드", len(rows) > 0, f"-> {len(rows)}건")
        if rows:
            r = rows[0]
            print(f"       {r['published_at'][5:16]} | {r['source'][:14]:14} | {r['title'][:38]}")

    import os
    if os.environ.get("SAVETICKER_COOKIE"):
        rows = oc.feed_payload("report", limit=3)
        check("리포트(report) 피드 — 쿠키 인증", len(rows) > 0, f"-> {len(rows)}건")
        for r in rows[:3]:
            print(f"       {r['published_at'][:16]} | {r['source']} | {r['title'][:44]}")
        if rows:
            rid = rows[0]["url"].split("/")[-2]
            pdf = oc.fetch_report_pdf(rid)
            check("리포트 PDF 원본 수신",
                  pdf is not None and pdf[0].startswith(b"%PDF-"),
                  f"-> {len(pdf[0]):,} bytes" if pdf else "실패")
    else:
        print("  [SKIP] 리포트 — SAVETICKER_COOKIE 환경변수 없음")

print()
print(f"결과: PASS {len(PASS)} / FAIL {len(FAIL)}")
if FAIL:
    print("실패:", *FAIL, sep="\n  - ")
sys.exit(1 if FAIL else 0)
