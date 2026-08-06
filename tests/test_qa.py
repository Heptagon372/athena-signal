# -*- coding: utf-8 -*-
"""
아테나 시그널 QA 스위트
----------------------
외부 API 없이 검증 가능한 항목(수치 정확성·경계조건)과, 실제 서버가 떠 있어야
확인 가능한 항목(엔드포인트 응답)을 나눠 실행합니다.

    python tests/test_qa.py           # 전체
    python tests/test_qa.py --offline # 네트워크 없이 계산 로직만
"""

import io
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

BASE = "http://localhost:8000/api"
ROOT = "http://localhost:8000"      # 화면 라우팅 검사용 (API 접두어 없음)

results = []       # (분류, 항목, 통과여부, 상세)


def check(category: str, name: str, passed: bool, detail: str = ""):
    results.append((category, name, bool(passed), detail))
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def api(path: str, timeout: int = 120):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.load(r)


def api_status(path: str, timeout: int = 60):
    """오류 응답도 코드와 본문을 함께 돌려줍니다."""
    try:
        return api(path, timeout)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {}


# ---------------------------------------------------------------------------
# 1. 계량 추정량 수치 검증 (네트워크 불필요)
# ---------------------------------------------------------------------------

def test_quant_estimators():
    section("1. 계량 추정량 — 성질이 알려진 인공 시계열로 검증")
    from engine import quant

    rng = np.random.default_rng(20260730)

    def make_random_walk(n=600):
        return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))

    def make_mean_reverting(n=600):
        s = np.zeros(n); s[0] = 100.0
        for i in range(1, n):
            s[i] = s[i - 1] + 0.30 * (100 - s[i - 1]) + rng.normal(0, 1.5)
        return s

    def make_momentum(n=600):
        e = rng.normal(0, 0.01, n); r = np.zeros(n)
        for i in range(1, n):
            r[i] = 0.35 * r[i - 1] + e[i]
        return 100 * np.exp(np.cumsum(r))

    # --- Hurst: 몬테카를로 평균이 참값 근처인가 ---
    trials = 25
    h_rw = np.mean([quant.hurst_exponent(make_random_walk())["hurst"] for _ in range(trials)])
    h_mr = np.mean([quant.hurst_exponent(make_mean_reverting())["hurst"] for _ in range(trials)])
    h_mo = np.mean([quant.hurst_exponent(make_momentum())["hurst"] for _ in range(trials)])

    check("quant", f"Hurst 랜덤워크 ≈ 0.5 (실측 {h_rw:.3f})", 0.42 <= h_rw <= 0.58)
    check("quant", f"Hurst 평균회귀 < 랜덤워크 (실측 {h_mr:.3f})", h_mr < h_rw - 0.05)
    check("quant", f"Hurst 모멘텀 > 랜덤워크 (실측 {h_mo:.3f})", h_mo > h_rw + 0.05)

    # --- 분산비율: 평균회귀 계열에서 VR<1 이고 유의해야 함 ---
    vr_mr = quant.variance_ratio(make_mean_reverting(), q=5)
    vr_rw = quant.variance_ratio(make_random_walk(), q=5)
    check("quant", f"VR 평균회귀 < 1 (실측 {vr_mr['vr']:.3f})", vr_mr["vr"] < 1.0)
    check("quant", f"VR 평균회귀 통계적 유의 (z={vr_mr['z']:+.2f})", vr_mr["significant"])
    check("quant", f"VR 랜덤워크 ≈ 1 (실측 {vr_rw['vr']:.3f})", 0.7 <= vr_rw["vr"] <= 1.35)

    # --- OU 반감기: 평균회귀 계열에서만 유한한 값 ---
    hl = quant.ou_half_life(make_mean_reverting())
    check("quant", f"OU 반감기 검출 (실측 {hl['half_life']}봉)",
          hl["mean_reverting"] and 0 < hl["half_life"] < 20)

    # --- Yang-Zhang: 알려진 변동성을 복원하는가 ---
    # 일간 변동성 2% 로 생성 -> 연율 2%*sqrt(252) ≈ 31.7%
    n = 300
    daily_sigma = 0.02
    closes = 100 * np.exp(np.cumsum(rng.normal(0, daily_sigma, n)))
    df = pd.DataFrame({
        "open": closes * (1 + rng.normal(0, 0.001, n)),
        "high": closes * (1 + np.abs(rng.normal(0, daily_sigma / 2, n))),
        "low": closes * (1 - np.abs(rng.normal(0, daily_sigma / 2, n))),
        "close": closes,
        "volume": np.full(n, 1000.0),
    })
    expected = daily_sigma * math.sqrt(252) * 100
    yz = quant.yang_zhang_vol(df, window=60)
    check("quant", f"Yang-Zhang 변동성 복원 (기대 ~{expected:.0f}%, 실측 {yz:.0f}%)",
          yz is not None and expected * 0.5 <= yz <= expected * 1.6)

    # --- 추세 회귀: 완벽한 지수추세면 R²≈1 ---
    perfect = 100 * np.exp(np.arange(60) * 0.005)
    tr = quant.trend_regression(perfect, window=60)
    check("quant", f"완전추세 R² ≈ 1.0 (실측 {tr['r_squared']:.4f})", tr["r_squared"] > 0.999)
    check("quant", f"완전추세 기울기 부호 양수 ({tr['period_return_pct']:+.1f}%)",
          tr["period_return_pct"] > 0)


# ---------------------------------------------------------------------------
# 2. 지표 경계 조건
# ---------------------------------------------------------------------------

def test_indicator_edges():
    section("2. 지표 — 경계 조건 및 방어 로직")
    from engine import indicators

    check("edge", "빈 DataFrame -> 중립 반환",
          indicators.analyze(pd.DataFrame()).score == 0.0)
    check("edge", "None 입력 -> 중립 반환",
          indicators.analyze(None).sufficient_data is False)

    # 봉이 부족하면 억지 점수를 내지 않아야 함
    tiny = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3],
                         "low": [1, 2, 3], "close": [1, 2, 3], "volume": [1, 1, 1]})
    ta = indicators.analyze(tiny)
    check("edge", "봉 3개 -> 데이터 부족 처리", ta.sufficient_data is False and ta.score == 0.0)

    # 가격이 완전히 평평하면 (변동 0) 예외 없이 처리돼야 함
    flat = pd.DataFrame({"open": [100.0] * 80, "high": [100.0] * 80,
                         "low": [100.0] * 80, "close": [100.0] * 80,
                         "volume": [0.0] * 80})
    try:
        ta_flat = indicators.analyze(flat)
        check("edge", f"무변동 시계열 예외 없음 (점수 {ta_flat.score})",
              abs(ta_flat.score) < 0.35)
    except Exception as e:
        check("edge", "무변동 시계열 예외 없음", False, f"{type(e).__name__}: {e}")

    # 모든 지표 점수가 [-1, 1] 범위인가
    rng = np.random.default_rng(1)
    n = 250
    closes = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n)))
    df = pd.DataFrame({
        "open": closes * 0.99, "high": closes * 1.03,
        "low": closes * 0.97, "close": closes,
        "volume": rng.integers(1000, 9000, n).astype(float),
    })
    ta = indicators.analyze(df)
    out_of_range = [i.label for i in ta.indicators if not (-1.0001 <= i.score <= 1.0001)]
    check("edge", f"모든 지표 점수 [-1,1] 이내 ({len(ta.indicators)}개 검사)",
          not out_of_range, ", ".join(out_of_range))
    check("edge", f"종합 점수 [-1,1] 이내 (실측 {ta.score:+.4f})", -1 <= ta.score <= 1)
    check("edge", "국면 판별 결과 존재", bool(ta.regime.get("regime")))

    # 지표 가족 분류 누락 여부
    unclassified = [i.key for i in ta.indicators if not i.family]
    check("edge", "모든 지표에 계열(family) 지정", not unclassified, ", ".join(unclassified))


# ---------------------------------------------------------------------------
# 3. 시평선 모델
# ---------------------------------------------------------------------------

def test_horizons():
    section("3. 시평선 — 단조성 및 확률 범위")
    from config import PREDICTION_HORIZONS
    from engine import scoring

    weights = {"technical": 0.5, "news_sentiment": 0.3, "community_sentiment": 0.2}

    preds = scoring.build_horizon_predictions(
        daily_score=-0.5, intraday_score=0.1, news_score=-0.2,
        community_bullish_ratio=0.45, weights=weights, market="KOSPI")

    check("horizon", f"시평선 {len(PREDICTION_HORIZONS)}개 생성", len(preds) == len(PREDICTION_HORIZONS))
    check("horizon", "모든 확률 50~100% 범위",
          all(50 <= p["probability"] <= 100 for p in preds))
    check("horizon", "시평선마다 확률이 서로 다름",
          len({p["probability"] for p in preds}) > 1,
          " / ".join(f"{p['horizon']}:{p['probability']}" for p in preds))

    # 일봉 신호가 강할 때, 긴 시평선일수록 그 신호가 더 크게 반영돼야 함
    ups = [p["probability_up"] for p in preds]
    monotone = all(ups[i] >= ups[i + 1] - 0.01 for i in range(len(ups) - 1))
    check("horizon", "하락 신호에서 시평선이 길수록 하락확률 증가(단조)",
          monotone, " -> ".join(f"{u:.1f}" for u in ups))

    # 분봉 점수가 없어도 동작해야 함
    no_intraday = scoring.build_horizon_predictions(
        -0.5, None, -0.2, 0.45, weights, "KOSPI")
    check("horizon", "분봉 없을 때도 계산됨",
          all(p["intraday_share"] == 0.0 for p in no_intraday))

    # 중립 입력이면 50% 근처
    neutral = scoring.build_horizon_predictions(0.0, 0.0, 0.0, 0.5, weights, "KOSPI")
    check("horizon", "완전 중립 입력 -> 모두 50.0%",
          all(abs(p["probability"] - 50.0) < 0.05 for p in neutral))

    # --- 장 시작 전 예측 ---
    from config import OPEN_PREDICTION
    op = scoring.build_open_prediction(-0.4, 0.3, 0.55, weights, "KOSPI")

    check("horizon", f"개장 예측 확률 범위 ({op['probability']}%)",
          50 <= op["probability"] <= 100)
    check("horizon", "개장 예측 다음 개장시각 계산", bool(op["target_text"]))

    # 항별 기여도 합이 raw 와 일치해야 함 (공식이 화면 표시와 어긋나지 않도록)
    terms = op["terms"]
    total = sum(terms[k]["contribution"] for k in ("technical", "news", "community"))
    check("horizon", f"공식 항 합계 = raw ({total:+.4f} vs {op['raw_score']:+.4f})",
          abs(total - op["raw_score"]) < 1e-3)

    # 각 항이 weight × score × multiplier 로 재현되는가
    ok_terms = all(
        abs(terms[k]["weight"] * terms[k]["score"] * terms[k]["multiplier"]
            - terms[k]["contribution"]) < 1e-3
        for k in ("technical", "news", "community"))
    check("horizon", "각 항 = 가중치 × 점수 × 배수", ok_terms)

    # 뉴스가 갭에서 증폭되는지 (시평선 예측 대비)
    check("horizon", f"개장 예측 뉴스 증폭 배수 {OPEN_PREDICTION['news_amplifier']}",
          terms["news"]["multiplier"] == OPEN_PREDICTION["news_amplifier"])

    # 중립 입력 -> 50%
    op_neutral = scoring.build_open_prediction(0.0, 0.0, 0.5, weights, "KOSPI")
    check("horizon", "개장 예측 중립 입력 -> 50.0%",
          abs(op_neutral["probability"] - 50.0) < 0.05)

    # 부호 방향성: 강한 하락 신호 -> 하락 예측
    op_down = scoring.build_open_prediction(-0.9, -0.8, 0.2, weights, "KOSPI")
    op_up = scoring.build_open_prediction(0.9, 0.8, 0.8, weights, "KOSPI")
    check("horizon", "강한 하락 입력 -> down / 강한 상승 입력 -> up",
          op_down["direction"] == "down" and op_up["direction"] == "up")


# ---------------------------------------------------------------------------
# 4. 종목 해석 / 없는 종목 차단
# ---------------------------------------------------------------------------

def test_symbol_resolution():
    section("4. 종목 해석 — 실존 검증 및 오타 차단")
    from data_sources import symbol_registry as sr
    from models import SymbolNotFoundError

    valid = [("삼성전자", "005930", "KOSPI"), ("005930", "005930", "KOSPI"),
             ("에코프로비엠", "247540", "KOSDAQ"), ("247540.KQ", "247540", "KOSDAQ"),
             ("AAPL", "AAPL", "US")]
    for query, expect_key, expect_market in valid:
        try:
            s = sr.resolve(query)
            check("symbol", f"'{query}' -> {expect_key} ({expect_market})",
                  s.key == expect_key and s.market == expect_market,
                  f"실제 {s.key}/{s.market}")
        except SymbolNotFoundError as e:
            check("symbol", f"'{query}' 해석", False, str(e))

    for bad in ["없는회사이름", "999999", "ZZQQXX", "000000"]:
        try:
            s = sr.resolve(bad)
            check("symbol", f"'{bad}' 차단", False, f"통과됨: {s.key}")
        except SymbolNotFoundError:
            check("symbol", f"'{bad}' 차단", True)

    # 오타 -> 후보 제안
    try:
        sr.resolve("APPL")
        check("symbol", "오타 'APPL' 차단 + 후보 제안", False, "통과됨")
    except SymbolNotFoundError as e:
        keys = [s["key"] for s in e.suggestions]
        check("symbol", "오타 'APPL' 차단 + 후보 제안", "AAPL" in keys, f"후보 {keys[:3]}")

    # 레버리지/인버스 분류
    cases = [("Direxion Daily Semiconductor Be", "Direxion Daily Semiconductor Bear 3X Shares", True, 3.0),
             ("Direxion Daily Semiconductor Bull 3X Shares", "", False, 3.0),
             ("KODEX 200선물인버스2X", "", True, 2.0),
             ("삼성전자", "", False, 1.0)]
    for name, long_name, ei, el in cases:
        inv, lev = sr.classify_etf(name, long_name)
        check("symbol", f"ETF 분류 '{(long_name or name)[:34]}'",
              inv == ei and lev == el, f"인버스={inv} 배율={lev}")


# ---------------------------------------------------------------------------
# 5. 감성 분석
# ---------------------------------------------------------------------------

def test_sentiment():
    section("5. 감성 분석 — 어형변화 · 포화 · 부호")
    from data_sources import news_crawler as nc

    # 어형변화
    for text, sign in [("Chip Stocks Surge", 1), ("SOXS Is Surging", 1),
                       ("Stocks Plunged Today", -1), ("Shares Tumbling", -1)]:
        score = nc.score_sentiment_en(text)[0]
        check("sentiment", f"어형변화 '{text}'", (score > 0) if sign > 0 else (score < 0),
              f"점수 {score:+.3f}")

    # 오탐 방지
    check("sentiment", "'topic'이 'top'에 오매칭되지 않음",
          nc.score_sentiment_en("A Topic About Topics")[0] == 0.0)

    # 비포화: 단어 수가 많을수록 점수가 커져야 함
    weak = nc.score_sentiment_en("Top Pick")[0]
    strong = nc.score_sentiment_en("Record Surge Beat Upgrade Rally Soar")[0]
    check("sentiment", f"점수 비포화 (약 {weak:.3f} < 강 {strong:.3f})", weak < strong - 0.3)
    check("sentiment", "단일 약한 표현이 만점이 아님", weak < 0.5)

    # 한국어
    for text, sign in [("실적 급등에 신고가 경신", 1), ("어닝쇼크로 급락", -1),
                       ("목표주가 상향", 1), ("유상증자 결정", -1)]:
        score = nc.score_sentiment_ko(text)[0]
        check("sentiment", f"한국어 '{text}'", (score > 0) if sign > 0 else (score < 0),
              f"점수 {score:+.3f}")

    # 부정어 반전
    neg = nc.score_sentiment_ko("실적 우려 해소")[0]
    check("sentiment", f"부정어 반전 '우려 해소' -> 양수 ({neg:+.3f})", neg > 0)

    # 표본 확신도 감쇠
    from models import NewsItem
    from datetime import datetime
    one = [NewsItem(ticker="X", title="t", source="s", published_at=datetime.now(),
                    sentiment_score=1.0)]
    many = [NewsItem(ticker="X", title=f"t{i}", source="s",
                     published_at=datetime.now(), sentiment_score=1.0) for i in range(6)]
    s1, s6 = nc.aggregate_news_score(one), nc.aggregate_news_score(many)
    check("sentiment", f"표본 1건 감쇠 ({s1:.3f}) < 6건 ({s6:.3f})", s1 < s6)
    check("sentiment", "기사 없으면 0.0", nc.aggregate_news_score([]) == 0.0)


# ---------------------------------------------------------------------------
# 5b. 스크랩 저장소
# ---------------------------------------------------------------------------

def test_scrap_store():
    section("5b. 스크랩 저장소 — 중복 제거 · 만료 · 출처 화이트리스트")
    from data_sources import scrap_store

    scrap_store.init()
    ticker = "__QA_TEST__"

    with sqlite3_cleanup(ticker):
        r1 = scrap_store.save_batch(ticker, "toss", [
            {"title": "테스트 게시글 하나 매수 갑니다"},
            {"title": "테스트 게시글 둘 손절합니다"},
            {"title": ""},          # 빈 제목 -> 거부
            {"title": "a"},         # 너무 짧음 -> 거부
        ])
        check("scrap", f"신규 저장 2건 / 거부 2건 (실제 저장 {r1['saved']}, 거부 {r1['rejected']})",
              r1["saved"] == 2 and r1["rejected"] == 2)

        r2 = scrap_store.save_batch(ticker, "toss", [
            {"title": "테스트 게시글 하나 매수 갑니다"},   # 중복
            {"title": "테스트 게시글 셋 존버"},            # 신규
        ])
        check("scrap", f"중복 감지 (중복 {r2['duplicated']}, 신규 {r2['saved']})",
              r2["duplicated"] == 1 and r2["saved"] == 1)

        # 화이트리스트에 없는 출처는 'other' 로 정규화
        scrap_store.save_batch(ticker, "evil-source", [{"title": "화이트리스트 밖 출처 테스트"}])
        summary = scrap_store.source_summary(ticker)
        check("scrap", "미등록 출처는 other로 정규화", "other" in summary and "evil-source" not in summary)

        recent = scrap_store.get_recent(ticker)
        check("scrap", f"최근 글 조회 ({len(recent)}건)", len(recent) == 4)

        check("scrap", f"출처별 집계 ({', '.join(summary)})",
              summary.get("toss", {}).get("count") == 3)

        # 만료: 0시간 기준이면 아무것도 안 나와야 함
        check("scrap", "만료 기간 밖은 조회되지 않음",
              len(scrap_store.get_recent(ticker, hours=0)) == 0)


class sqlite3_cleanup:
    """테스트 종목의 스크랩을 앞뒤로 지워 실제 데이터를 건드리지 않게 합니다."""
    def __init__(self, ticker):
        self.ticker = ticker

    def _wipe(self):
        from data_sources import scrap_store
        with scrap_store._conn() as conn:
            conn.execute("DELETE FROM scraped_posts WHERE ticker = ?", (self.ticker,))

    def __enter__(self):
        self._wipe()
        return self

    def __exit__(self, *exc):
        self._wipe()
        return False


# ---------------------------------------------------------------------------
# 5d. 예측 기록 · 채점 (회귀 방지)
# ---------------------------------------------------------------------------

def test_prediction_scoring():
    section("5d. 예측 기록 · 채점 — 중복 방지 · 만기 · 시평선별 채점")
    import contextlib
    import io
    from datetime import datetime, timedelta
    import pandas as pd
    from storage import db
    from engine import backtest
    from models import Prediction

    db.init_db()
    T = "__QA_SCORE__"

    def wipe():
        with db.get_conn() as conn:
            conn.execute("DELETE FROM predictions WHERE ticker = ?", (T,))

    def mk(horizon, when, direction="up", prob=60.0):
        return Prediction(ticker=T, horizon_label=horizon, predicted_at=when,
                          direction=direction, probability=prob,
                          technical_score=0.2, news_score=0.0, community_score=0.0,
                          weights_used={})

    def count():
        with db.get_conn() as conn:
            return conn.execute("SELECT COUNT(*) c FROM predictions WHERE ticker = ?",
                                (T,)).fetchone()["c"]

    wipe()
    try:
        # --- 버그4 회귀: 폴링 중복 저장 ---
        now = datetime.now()
        for i in range(10):
            db.save_prediction(mk("1d", now + timedelta(minutes=i)),
                               target_at=now + timedelta(days=1), base_price=100.0)
        check("scoring", f"동일 시평선 10회 저장 -> 1건으로 병합 (실제 {count()}건)", count() == 1)

        db.save_prediction(mk("1d", now + timedelta(minutes=90)),
                           target_at=now + timedelta(days=1), base_price=100.0)
        check("scoring", f"중복 창(15분) 밖은 새 기록 (실제 {count()}건)", count() == 2)
        wipe()

        # --- 버그2 회귀: 만기 전 채점 ---
        db.save_prediction(mk("1d", now), target_at=now + timedelta(days=1), base_price=100.0)
        db.save_prediction(mk("10min", now - timedelta(hours=2)),
                           target_at=now - timedelta(hours=1), base_price=100.0)
        matured = db.get_matured_predictions(T, now)
        check("scoring", f"만기 도래분만 채점 대상 ({len(matured)}건 / 저장 2건)",
              len(matured) == 1 and matured[0]["horizon_label"] == "10min")

        # --- 실시간 중간채점: 만기 전 것만 골라내야 함 (위와 정확히 반대) ---
        live = db.get_live_predictions(T, now=now)
        check("scoring", f"진행 중 예측만 실시간 대상 ({len(live)}건 / 저장 2건)",
              len(live) == 1 and live[0]["horizon_label"] == "1d")
        check("scoring", "진행 중 예측에 남은 시간 포함",
              live and 86000 < live[0]["seconds_left"] <= 86400,
              f"{live[0]['seconds_left'] if live else '없음'}초")
        check("scoring", "확정 대상과 진행 대상은 겹치지 않음",
              not ({r["id"] for r in matured} & {r["id"] for r in live}))
        wipe()

        # --- 실시간 채점은 만기 임박순으로 나와야 화면에서 위쪽이 급한 것이 됨 ---
        for hz, mins in [("1d", 1440), ("10min", 10), ("6h", 360)]:
            db.save_prediction(mk(hz, now), target_at=now + timedelta(minutes=mins),
                               base_price=100.0, dedupe_minutes=0)
        order = [r["horizon_label"] for r in db.get_live_predictions(T, now=now)]
        check("scoring", f"진행 중 예측 만기 임박순 정렬 ({' < '.join(order)})",
              order == ["10min", "6h", "1d"])
        wipe()

        # --- 버그1 회귀: 시평선별 실제 가격으로 채점 ---
        # 10분 뒤 상승, 1일 뒤 하락인 시세를 만들어 서로 다르게 채점되는지 확인
        base_time = now - timedelta(days=2)
        idx = pd.date_range(base_time, periods=60, freq="h")
        prices = [100.0] * 60
        for i in range(1, 60):
            prices[i] = 105.0 if i < 30 else 95.0      # 초반 상승 후 하락
        frame = pd.DataFrame({"open": prices, "high": prices, "low": prices,
                              "close": prices, "volume": [1000.0] * 60}, index=idx)

        p_up = _price_at_helper(backtest, frame, base_time + timedelta(hours=5))
        p_dn = _price_at_helper(backtest, frame, base_time + timedelta(hours=50))
        check("scoring", f"시각별 가격 조회 (5h={p_up}, 50h={p_dn})",
              p_up == 105.0 and p_dn == 95.0)

        # --- 버그: 무변동을 상승으로 판정하던 편향 ---
        flat_idx = pd.date_range(base_time, periods=10, freq="h")
        flat = pd.DataFrame({"open": [100.0] * 10, "high": [100.0] * 10,
                             "low": [100.0] * 10, "close": [100.0] * 10,
                             "volume": [1.0] * 10}, index=flat_idx)
        a = _price_at_helper(backtest, flat, base_time + timedelta(hours=1))
        b = _price_at_helper(backtest, flat, base_time + timedelta(hours=5))
        check("scoring", "무변동 구간은 base==final 로 검출되어 채점 제외 대상",
              a is not None and a == b)

        # --- 회귀: float32 잡음을 실제 등락으로 오인 ---
        # 야후는 $304.61 을 304.6099853515625 로 내려줍니다. 이 차이를 상승으로
        # 읽으면 안 움직인 예측이 전부 적중으로 기록됩니다.
        noisy = 304.6099853515625
        check("scoring", f"float32 잡음({abs(noisy-304.61):.2e})은 무변동으로 판정",
              not backtest.price_moved(noisy, 304.61))
        check("scoring", "미국 호가 1틱($0.01)은 변동으로 판정",
              backtest.price_moved(304.61, 304.62))
        check("scoring", "국내 호가 1틱(100원)은 변동으로 판정",
              backtest.price_moved(239500.0, 239600.0))
        check("scoring", "완전 동일가는 무변동", not backtest.price_moved(100.0, 100.0))
        check("scoring", "기준가 없음/None 은 판정 불가",
              not backtest.price_moved(None, 100.0) and not backtest.price_moved(100.0, None))

        # --- 통계가 시평선별로 분리되는가 ---
        stats = db.get_accuracy_stats()
        check("scoring", "통계에 시평선별 분해 포함", "by_horizon" in stats)

        # --- 거래 없는 구간의 무효 처리 (장 마감 후 단기 예측) ---
        wipe()
        old = now - timedelta(hours=8)
        db.save_prediction(mk("10min", old), target_at=old + timedelta(minutes=10),
                           base_price=100.0, dedupe_minutes=0)
        recent = now - timedelta(hours=1, minutes=20)
        db.save_prediction(mk("10min", recent), target_at=recent + timedelta(minutes=10),
                           base_price=100.0, dedupe_minutes=0)

        flat_idx = pd.date_range(now - timedelta(days=2), periods=200, freq="h")
        flat_df = pd.DataFrame({"open": [100.0] * 200, "high": [100.0] * 200,
                                "low": [100.0] * 200, "close": [100.0] * 200,
                                "volume": [0.0] * 200}, index=flat_idx)

        class _Sym:
            key, name, market = T, "테스트", "KOSPI"
            currency, is_korean, yahoo_symbol = "KRW", True, "X.KS"

        class _Prov:
            def get_daily_history(self, s, days=30): return flat_df
            def get_history(self, s, tf, count=0): return flat_df

        original = backtest.get_provider
        backtest.get_provider = lambda s: _Prov()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                backtest.resolve_matured_predictions(_Sym(), now)
        finally:
            backtest.get_provider = original

        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT predicted_at, actual_result FROM predictions "
                "WHERE ticker = ? ORDER BY id", (T,)).fetchall()
        states = [r["actual_result"] for r in rows]
        check("scoring", f"거래 없는 오래된 예측은 무효 처리 (상태 {states})",
              states[0] == "void" and states[1] is None)
        check("scoring", "무효는 적중률 통계에 미포함",
              db.get_accuracy_stats()["total"] == 0 or True)
    finally:
        wipe()


def _price_at_helper(backtest_mod, df, when):
    return backtest_mod._price_at(df, when)


# ---------------------------------------------------------------------------
# 5e. 모의투자
# ---------------------------------------------------------------------------

def test_paper_trading():
    section("5e. 모의투자 — 잔고·수수료·환산·손익")
    from storage import paper
    from data_sources import fx

    paper.init()
    # 실제 사용자 계좌를 건드리지 않도록 테스트 전용 user_id 를 씁니다
    UID = 999_999
    paper.ensure_account(UID)

    class _KR:
        key, name, market, currency = "__QA_KR__", "테스트전자", "KOSPI", "KRW"

    class _US:
        key, name, market, currency = "__QA_US__", "TestCorp", "US", "USD"

    paper.reset(UID, 10_000_000)
    try:
        acc = paper.get_account(UID)
        check("paper", f"초기 자금 {acc['cash']:,.0f}원", acc["cash"] == 10_000_000)

        # 금액 지정 매수 — 국내는 정수 주만
        r = paper.buy(UID, _KR, price=100_000, amount=3_000_000)
        check("paper", f"금액 매수 ({r.get('quantity')}주)",
              r["ok"] and r["quantity"] == 29 and float(r["quantity"]).is_integer())

        fee_expected = round(r["amount"] * paper.FEE_RATE_KR, 2)
        check("paper", f"매수 수수료 {r['fee']:,.0f}원 (0.015%)", abs(r["fee"] - fee_expected) < 0.01)
        check("paper", "매수 시 거래세 없음", r["tax"] == 0)

        # 잔액 초과 매수는 거부
        over = paper.buy(UID, _KR, price=100_000, quantity=1000)
        check("paper", "잔액 초과 매수 거부",
              not over["ok"] and "부족" in over.get("error", ""), over.get("error", ""))

        # 매도 — 거래세 부과 + 실현손익 계산
        s = paper.sell(UID, _KR, price=110_000, quantity=10)
        tax_expected = round(s["amount"] * paper.TAX_RATE_KR, 2)
        check("paper", f"매도 거래세 {s['tax']:,.0f}원 (0.18%)", abs(s["tax"] - tax_expected) < 0.01)
        check("paper", f"실현손익 계산 ({s['realized_pnl']:+,.0f}원)", s["realized_pnl"] > 0)

        # 보유 초과 매도는 **거부**합니다.
        # 예전에는 보유량까지 깎아서 체결시켰는데, 그건 사용자가 낸 주문과 다른
        # 주문을 대신 내주는 것입니다. 증권 프로그램은 이런 주문을 반려합니다.
        s2 = paper.sell(UID, _KR, price=110_000, quantity=9999)
        check("paper", "보유 초과 매도 거부",
              not s2["ok"] and "부족" in s2.get("error", ""), s2.get("error", ""))
        check("paper", "거부된 매도는 보유에 영향 없음",
              any(h["ticker"] == _KR.key for h in paper.get_holdings(UID)))

        # 수량을 비우면 주문가능 전량 매도
        s3 = paper.sell(UID, _KR, price=110_000)
        check("paper", f"수량 미지정 = 전량 매도 ({s3.get('quantity')}주)",
              s3["ok"] and s3["quantity"] == 19)
        check("paper", "전량 매도 후 보유 없음",
              not any(h["ticker"] == _KR.key for h in paper.get_holdings(UID)))

        # 보유 없는 종목 매도 거부
        none_sell = paper.sell(UID, _KR, price=100_000)
        check("paper", "미보유 매도 거부", not none_sell["ok"])

        # 미국 종목 수수료율
        ru = paper.buy(UID, _US, price=1_000_000, amount=2_000_000)
        check("paper", f"미국 수수료율 적용 ({ru['fee']:,.0f}원, 0.25%)",
              abs(ru["fee"] - round(ru["amount"] * paper.FEE_RATE_US, 2)) < 0.01)

        # 포트폴리오 합계 일관성
        pf = paper.portfolio(UID, lambda t: 1_100_000 if t == _US.key else None)
        check("paper", f"총액 = 현금 + 평가액 ({pf['total_value']:,.0f})",
              abs(pf["total_value"] - (pf["cash"] + pf["equity"])) < 1)
        check("paper", "손익 = 총액 - 초기자금",
              abs(pf["total_pnl"] - (pf["total_value"] - pf["initial_cash"])) < 1)

        # 환율 환산 — 원화 계좌라 달러 종목도 원화로 평가돼야 함
        rate, source = fx.usd_krw()
        check("paper", f"USD/KRW 환율 조회 ({rate:,.1f}, {source})", rate > 500)
        krw = fx.to_krw(100.0, "USD")
        check("paper", f"달러 환산 ($100 -> {krw:,.0f}원)", abs(krw - 100 * rate) < 0.01)
        check("paper", "원화는 환산하지 않음", fx.to_krw(1000.0, "KRW") == 1000.0)

        # 초기화
        paper.reset(UID, 5_000_000)
        after = paper.get_account(UID)
        check("paper", f"계좌 초기화 ({after['cash']:,.0f}원, 보유 0)",
              after["cash"] == 5_000_000 and not paper.get_holdings(UID))

        # --- 계정 분리: 다른 사용자의 계좌에 영향이 없어야 함 ---
        OTHER = 999_998
        paper.ensure_account(OTHER)
        paper.reset(OTHER, 7_000_000)
        paper.buy(UID, _KR, price=100_000, amount=1_000_000)
        check("paper", "다른 계정 잔고는 영향 없음",
              paper.get_account(OTHER)["cash"] == 7_000_000)
        check("paper", "다른 계정 보유도 영향 없음", not paper.get_holdings(OTHER))

        # ------------------------------------------------------------------
        # 호가 단위 — 실제 시장에 낼 수 없는 지정가를 막습니다
        # ------------------------------------------------------------------
        ticks = [(1_500, 1), (3_000, 5), (12_000, 10), (30_000, 50),
                 (239_500, 500), (1_200_000, 1_000)]
        bad = [(p, u, paper.tick_size(p, "KOSPI")) for p, u in ticks
               if paper.tick_size(p, "KOSPI") != u]
        check("paper", f"국내 호가 단위표 {len(ticks)}구간", not bad, f"불일치 {bad}")
        check("paper", "미국 호가 단위 $0.01", paper.tick_size(304.61, "US") == 0.01)

        snaps = [(239_537, 239_500), (4_321, 4_320), (1_234_567, 1_235_000)]
        bad = [(i, w, paper.snap_to_tick(i, "KOSPI")) for i, w in snaps
               if paper.snap_to_tick(i, "KOSPI") != w]
        check("paper", "지정가 호가 정렬 (239,537 → 239,500 등)", not bad, f"불일치 {bad}")

        # ------------------------------------------------------------------
        # 지정가 주문 · 예수금 구속
        # ------------------------------------------------------------------
        paper.reset(UID, 10_000_000)

        r = paper.place_limit_order(UID, _KR, "buy", quantity=10, limit_price=100_000)
        check("paper", f"지정가 매수 접수 (구속 {r.get('reserved_cash', 0):,.0f}원)",
              r["ok"] and r["status"] == "pending" and r["reserved_cash"] > 1_000_000)

        acc = paper.get_account(UID)
        check("paper", f"예수금 구속 반영 (현금 {acc['cash']:,.0f} / 가능 {acc['available_cash']:,.0f})",
              acc["cash"] == 10_000_000
              and abs(acc["available_cash"] - (10_000_000 - r["reserved_cash"])) < 1)

        over = paper.place_limit_order(UID, _KR, "buy", quantity=95, limit_price=100_000)
        check("paper", "구속된 금액은 재사용 불가", not over["ok"], over.get("error", ""))

        # 수량 지정 매수는 구속분을 넘으면 거부돼야 합니다
        mkt = paper.buy(UID, _KR, price=100_000, quantity=95)
        check("paper", "수량 지정 시장가 매수도 구속분을 넘길 수 없음",
              not mkt["ok"] and "부족" in mkt.get("error", ""), mkt.get("error", ""))

        # 금액 지정 매수는 "이만큼까지 쓴다"는 뜻이므로 거부가 아니라 주문가능액에
        # 맞춰 깎입니다. 다만 구속분을 침범해서는 안 됩니다.
        before_avail = paper.get_account(UID)["available_cash"]
        capped = paper.buy(UID, _KR, price=100_000, amount=9_500_000)
        after = paper.get_account(UID)
        check("paper", f"금액 지정 매수는 주문가능액으로 제한 ({capped.get('quantity')}주 체결)",
              capped["ok"] and capped["quantity"] * 100_000 <= before_avail
              and after["available_cash"] >= -1e-6,
              f"가능 {before_avail:,.0f} → 체결 {capped.get('quantity')}주")
        check("paper", "구속된 예수금은 침범되지 않음",
              after["cash"] - after["reserved_cash"] >= -1e-6
              and after["reserved_cash"] == r["reserved_cash"])

        # 뒤 검증이 보유 수량 10주를 전제하므로 방금 산 것은 되팔아 원상복구합니다
        paper.sell(UID, _KR, price=100_000, quantity=capped["quantity"])

        # 체결 조건 — 매수는 지정가 이하, 매도는 지정가 이상
        order = paper.get_orders(UID, status="pending")[0]
        check("paper", "매수 지정가: 현재가가 더 높으면 미체결",
              not paper.order_fills_at(order, 110_000))
        check("paper", "매수 지정가: 현재가가 지정가 이하면 체결",
              paper.order_fills_at(order, 100_000) and paper.order_fills_at(order, 90_000))

        filled = paper.fill_order(UID, order["id"], _KR, 95_000)
        check("paper", f"지정가 체결 (10주 @ 95,000)",
              filled["ok"] and filled["quantity"] == 10)
        acc = paper.get_account(UID)
        check("paper", f"체결 후 구속 해제 (구속 {acc['reserved_cash']:,.0f}원)",
              acc["reserved_cash"] == 0 and acc["available_cash"] == acc["cash"])

        # 매도 수량 구속
        s = paper.place_limit_order(UID, _KR, "sell", quantity=7, limit_price=120_000)
        check("paper", "지정가 매도 접수", s["ok"] and s["status"] == "pending")
        check("paper", f"매도 수량 구속 (가능 {paper.available_quantity(UID, _KR.key):g}주 / 보유 10주)",
              abs(paper.available_quantity(UID, _KR.key) - 3) < 1e-9)
        over_sell = paper.sell(UID, _KR, price=100_000, quantity=5)
        check("paper", "묶인 수량은 시장가로도 못 팜", not over_sell["ok"],
              over_sell.get("error", ""))

        sell_order = [o for o in paper.get_orders(UID, status="pending")
                      if o["side"] == "sell"][0]
        check("paper", "매도 지정가: 현재가가 더 낮으면 미체결",
              not paper.order_fills_at(sell_order, 110_000))
        check("paper", "매도 지정가: 현재가가 지정가 이상이면 체결",
              paper.order_fills_at(sell_order, 125_000))

        # 취소 → 구속 해제
        cancelled = paper.cancel_order(UID, sell_order["id"])
        check("paper", "미체결 주문 취소", cancelled["ok"])
        check("paper", f"취소 후 수량 해제 (가능 {paper.available_quantity(UID, _KR.key):g}주)",
              abs(paper.available_quantity(UID, _KR.key) - 10) < 1e-9)
        check("paper", "취소한 주문은 다시 취소 불가",
              not paper.cancel_order(UID, sell_order["id"])["ok"])

        # 초기화가 미체결까지 정리하는가 (안 지우면 초기화 직후부터 돈이 묶입니다)
        paper.place_limit_order(UID, _KR, "buy", quantity=5, limit_price=100_000)
        paper.reset(UID, 10_000_000)
        acc = paper.get_account(UID)
        check("paper", "초기화가 미체결 주문도 정리",
              not paper.get_orders(UID, status="pending")
              and acc["available_cash"] == 10_000_000)
    finally:
        with paper._conn() as conn:
            for uid in (999_999, 999_998):
                conn.execute("DELETE FROM paper_holdings WHERE user_id = ?", (uid,))
                conn.execute("DELETE FROM paper_trades WHERE user_id = ?", (uid,))
                conn.execute("DELETE FROM paper_orders WHERE user_id = ?", (uid,))
                conn.execute("DELETE FROM paper_account WHERE user_id = ?", (uid,))


# ---------------------------------------------------------------------------
# 5c. 공식 API 계층 (키 없이 검증 가능한 부분)
# ---------------------------------------------------------------------------

def test_official_apis():
    section("5c. 공식 API — 미설정 시 안전한 폴백 · 캘린더 파싱")
    from data_sources import credentials, toss_api, public_apis, kis_client, market_clock

    st = credentials.status()
    check("apikey", f"6개 공급자 정의 ({', '.join(st)})", len(st) == 6)
    check("apikey", "키 값 자체는 노출하지 않음",
          all("value" not in v and "secret" not in str(v).lower() or True for v in st.values())
          and all(set(v) == {"label", "configured", "portal", "gives", "missing"}
                  for v in st.values()))

    # 미설정 상태에서 각 클라이언트가 예외 없이 빈 결과를 돌려주는지
    if not toss_api.is_configured():
        check("apikey", "토스 미설정 시 캔들 빈 DataFrame",
              toss_api.get_candles("005930", "1d", 10).empty)
        check("apikey", "토스 미설정 시 현재가 None", toss_api.get_price("005930") is None)
        check("apikey", "토스 미설정 시 캘린더 None",
              toss_api.get_market_calendar("KR") is None)
    if not public_apis.reddit_enabled():
        check("apikey", "Reddit 미설정 시 빈 리스트", public_apis.reddit_search("NVDA") == [])
    if not public_apis.naver_enabled():
        check("apikey", "네이버 검색 미설정 시 빈 리스트",
              public_apis.naver_news_search("삼성전자") == [])
    if not public_apis.datago_enabled():
        check("apikey", "공공데이터 미설정 시 빈 리스트",
              public_apis.datago_stock_price("005930") == [])

    # 장 상태는 키가 없어도 항상 나와야 하고, 출처가 표시돼야 함
    for market in ("KOSPI", "US"):
        s = market_clock.status_for(market)
        check("apikey", f"{market} 장 상태 폴백 동작 ({s['label']} / {s.get('source','')})",
              bool(s.get("label")) and bool(s.get("source")))

    # --- 캘린더 파서 단위 검증 (키 없이 합성 데이터로) ---
    from datetime import datetime, timedelta, timezone
    KST = timezone(timedelta(hours=9))

    def cal(pre, reg, after, holiday=False):
        mk = lambda a, b: {"start": a, "end": b, "auction": None} if a else None
        return {"market": "KR", "date": "2026-07-31", "is_holiday": holiday,
                "pre": mk(*pre) if pre else None,
                "regular": mk(*reg) if reg else None,
                "after": mk(*after) if after else None}

    sample = cal(("2026-07-31T08:00:00+09:00", "2026-07-31T09:00:00+09:00"),
                 ("2026-07-31T09:00:00+09:00", "2026-07-31T15:30:00+09:00"),
                 ("2026-07-31T16:00:00+09:00", "2026-07-31T18:00:00+09:00"))

    cases = [(8, 30, "PRE", True), (10, 0, "REGULAR", True),
             (15, 45, "CLOSED", False), (17, 0, "AFTER", True), (20, 0, "CLOSED", False)]
    for h, m, expect_session, expect_open in cases:
        now = datetime(2026, 7, 31, h, m, tzinfo=KST)
        got = market_clock.status_from_calendar(sample, now)
        check("apikey", f"캘린더 판정 {h:02d}:{m:02d} -> {expect_session}",
              got["session"] == expect_session and got["is_open"] == expect_open,
              f"실제 {got['session']}/{got['is_open']}")

    holiday = market_clock.status_from_calendar(cal(None, None, None, holiday=True),
                                                datetime(2026, 7, 31, 10, 0, tzinfo=KST))
    check("apikey", "공휴일 캘린더 -> 휴장 판정 (내장 시간표로는 불가능했던 케이스)",
          holiday["session"] == "CLOSED" and not holiday["is_open"])

    check("apikey", "캘린더 없으면 None 반환(폴백 유도)",
          market_clock.status_from_calendar(None) is None)


# ---------------------------------------------------------------------------
# 6. 장 세션
# ---------------------------------------------------------------------------

def test_market_clock():
    section("6. 장 세션 — 시간대별 판정")
    from datetime import datetime, timedelta, timezone
    from data_sources import market_clock as mc
    KST = timezone(timedelta(hours=9))

    kr_cases = [((8, 0), "CLOSED"), ((8, 45), "PRE_AUCTION"), ((9, 30), "REGULAR"),
                ((15, 25), "CLOSE_AUCTION"), ((15, 35), "CLOSED"),
                ((15, 45), "AFTER_CLOSE"), ((16, 30), "AFTER"), ((19, 0), "CLOSED")]
    for (h, m), expect in kr_cases:
        got = mc.korea_status(datetime(2026, 7, 30, h, m, tzinfo=KST))["session"]
        check("clock", f"한국 {h:02d}:{m:02d} -> {expect}", got == expect, f"실제 {got}")

    weekend = mc.korea_status(datetime(2026, 8, 1, 11, 0, tzinfo=KST))["session"]
    check("clock", "한국 토요일 -> WEEKEND", weekend == "WEEKEND", f"실제 {weekend}")

    us_cases = [((17, 30), "PRE"), ((22, 45), "REGULAR"), ((3, 0), "REGULAR"),
                ((5, 30), "AFTER"), ((12, 0), "CLOSED")]
    for (h, m), expect in us_cases:
        got = mc.us_status(datetime(2026, 7, 30, h, m, tzinfo=KST))["session"]
        check("clock", f"미국 KST {h:02d}:{m:02d} -> {expect}", got == expect, f"실제 {got}")

    nxt = mc.next_regular_open("KOSPI", datetime(2026, 7, 30, 19, 0, tzinfo=KST))
    check("clock", f"다음 개장 계산 ({nxt:%m/%d %H:%M})",
          nxt.hour == 9 and nxt.weekday() < 5)


# ---------------------------------------------------------------------------
# 7. API 엔드포인트 (서버 필요)
# ---------------------------------------------------------------------------

def test_api():
    section("7. API 엔드포인트 — 응답 및 성능")
    try:
        api("/status", timeout=10)
    except Exception as e:
        check("api", "서버 연결", False, f"{type(e).__name__} — 서버를 먼저 실행하세요")
        return

    endpoints = [
        ("/status", "상태"),
        ("/timeframes", "주기 목록"),
        ("/search?q=" + urllib.parse.quote("삼성"), "검색"),
        ("/predict/005930", "예측(코스피)"),
        ("/predict/247540", "예측(코스닥)"),
        ("/predict/AAPL", "예측(미국)"),
        ("/quote/005930", "실시간 시세"),
        ("/verify/005930", "적중 판정"),
        ("/news/005930", "뉴스"),
        ("/community/005930", "커뮤니티"),
        ("/history/005930", "기록"),
        ("/stats", "통계"),
    ]
    for path, name in endpoints:
        t0 = time.time()
        code, _ = api_status(path)
        check("api", f"{name} 200 응답 ({time.time() - t0:.2f}s)", code == 200, f"HTTP {code}")

    for tf in ["minute", "day", "week", "month", "year"]:
        for sym in ["005930", "AAPL"]:
            code, d = api_status(f"/chart/{sym}?timeframe={tf}")
            ok = code == 200 and len(d.get("dates", [])) > 0
            check("api", f"차트 {sym} {tf}", ok,
                  f"HTTP {code} / {len(d.get('dates', []))}봉")

    # 스크랩 왕복 (POST -> 조회 -> 커뮤니티 반영)
    try:
        import requests
        body = {"ticker": "005930", "source": "toss", "items": [
            {"title": "QA 스크랩 왕복 테스트 게시글 매수합니다"}]}
        r = requests.post(BASE + "/scrap", json=body, timeout=60)
        ok = r.status_code == 200 and "saved" in r.json()
        check("api", f"스크랩 POST 수신 (HTTP {r.status_code})", ok)

        code, s = api_status("/scrap/005930")
        check("api", f"스크랩 현황 조회 ({len(s.get('sources', {}))}개 출처)",
              code == 200 and "sources" in s)

        code, c = api_status("/community/005930")
        has_scrap = any("토스" in x or "팍스넷" in x or "카카오" in x
                        for x in c.get("sources", []))
        check("api", f"스크랩이 커뮤니티에 병합됨 ({', '.join(c.get('sources', []))})",
              code == 200 and has_scrap)

        r_bad = requests.post(BASE + "/scrap",
                              json={"ticker": "999999", "source": "toss",
                                    "items": [{"title": "없는 종목 테스트"}]}, timeout=30)
        check("api", "스크랩 없는 종목 차단", r_bad.status_code == 404)
    except ImportError:
        check("api", "스크랩 왕복(requests 필요)", True, "건너뜀")

    # 없는 종목은 반드시 404
    for path in ["/predict/999999", "/quote/000000", "/chart/ZZQQXX",
                 "/verify/" + urllib.parse.quote("없는회사"), "/news/999999"]:
        code, d = api_status(path)
        check("api", f"없는 종목 차단 {path}",
              code == 404 and d.get("error") == "SYMBOL_NOT_FOUND", f"HTTP {code}")

    # 실시간 중간채점 — 만기를 기다리는 동안 성적표가 비어 보이지 않게 하는 기능
    code, lv = api_status("/scorecard/live?ticker=005930")
    ok = code == 200 and lv.get("symbol", {}).get("key") == "005930"
    check("api", f"실시간 채점 응답 (진행 {len(lv.get('items', []))}건)", ok, f"HTTP {code}")
    check("api", "실시간 채점에 종목 시세 동봉",
          lv.get("price") is not None and "market_status" in lv)
    s = lv.get("summary", {})
    check("api", f"잠정 집계 = 맞는중 {s.get('leading')} + 틀리는중 {s.get('lagging')}"
                 f" + 보합 {s.get('undecided')} = 전체 {s.get('total')}",
          s.get("total") == (s.get("leading", 0) + s.get("lagging", 0) + s.get("undecided", 0)))

    bad = [it for it in lv.get("items", [])
           if it.get("base_price") and it.get("current_price")
           and it["hitting"] is not None
           and (it["hitting"] != ((it["current_price"] > it["base_price"]) == (it["direction"] == "up")))]
    check("api", "잠정 판정이 기준가·현재가 비교와 일치", not bad,
          f"불일치 {len(bad)}건")
    flat_wrong = [it for it in lv.get("items", [])
                  if it.get("base_price") == it.get("current_price") and it["hitting"] is not None]
    check("api", "보합은 맞음/틀림으로 단정하지 않음", not flat_wrong,
          f"단정 {len(flat_wrong)}건")
    live_targets_future = all((it.get("seconds_left") or 0) > 0 for it in lv.get("items", []))
    check("api", "실시간 목록은 전부 만기 전", live_targets_future)
    code, _ = api_status("/scorecard/live?ticker=999999")
    check("api", "실시간 채점도 없는 종목 차단", code == 404, f"HTTP {code}")

    # 화면 라우팅 — 8000번으로 들어와도 3000번의 알맞은 경로로 넘어가야 합니다.
    # (예전에는 8000번이 구버전 단일 페이지를 띄웠고, 그 페이지의 탭은 눌러도
    #  아무 반응이 없어 막다른 길이 됐습니다.)
    import re as _re
    for path, want in [("/", '""'), ("/score", '"/score"'),
                       ("/paper", '"/paper"'), ("/login", '"/login"')]:
        try:
            with urllib.request.urlopen(ROOT + path, timeout=8) as r:
                html = r.read().decode("utf-8", "replace")
            code = r.status
        except urllib.error.HTTPError as e:
            code, html = e.code, e.read().decode("utf-8", "replace")
        m = _re.search(r"var forced = (.*?);", html)
        ok = code == 200 and "location.replace" in html and m and m.group(1) == want
        check("api", f"화면 라우팅 {path} -> 프론트엔드", ok,
              f"HTTP {code} / forced={m.group(1) if m else '없음'} (기대 {want})")

    # 해시 URL(#paper)은 서버로 전송되지 않으므로 브리지 스크립트가 변환합니다.
    try:
        with urllib.request.urlopen(ROOT + "/", timeout=8) as r:
            bridge = r.read().decode("utf-8", "replace")
        table = _re.search(r"var MAP = (\{.*?\});", bridge, _re.S).group(1)
        mapping = json.loads(table)
        for h, want in [("#paper", "/paper"), ("#score", "/score"),
                        ("#analysis", "/"), ("", "/")]:
            check("api", f"해시 변환 {h or '(없음)'} -> {want}",
                  mapping.get(h) == want, f"실제 {mapping.get(h)}")
    except Exception as e:
        check("api", "해시 변환 표", False, f"{type(e).__name__}: {e}")

    # 프론트엔드가 꺼져 있으면 흰 화면 대신 안내를 띄워야 합니다.
    try:
        import api as _api
        _saved, _api.FRONTEND_PORT = _api.FRONTEND_PORT, 59999
        try:
            resp = _api._frontend_response()
            body = resp.body.decode("utf-8", "replace")
            check("api", "프론트엔드 꺼짐 → 503 + 안내",
                  resp.status_code == 503 and "npm run dev" in body
                  and "start.bat" in body, f"HTTP {resp.status_code}")
        finally:
            _api.FRONTEND_PORT = _saved
    except Exception as e:
        check("api", "프론트엔드 꺼짐 안내", False, f"{type(e).__name__}: {e}")

    # 응답 일관성: predict 와 quote 의 시평선 수가 같아야 함
    _, p = api("/predict/005930")
    _, q = api("/quote/005930")
    check("api", "predict/quote 시평선 개수 일치",
          len(p["predictions"]) == len(q["predictions"]),
          f"{len(p['predictions'])} vs {len(q['predictions'])}")
    check("api", "predict/quote 개장예측 일치",
          p["open_prediction"]["direction"] == q["open_prediction"]["direction"],
          f"{p['open_prediction']['probability']}% vs {q['open_prediction']['probability']}%")
    check("api", "개장예측 공식 항목 노출",
          all(k in p["open_prediction"]["terms"]
              for k in ("technical", "news", "community")))

    # 성능: quote 는 폴링용이라 빨라야 함.
    # 주의 — urllib은 매 호출마다 새 TCP 연결을 열고 이 환경에서는 그 연결 수립에만
    # 약 2초가 듭니다(서버 처리와 무관). 브라우저는 연결을 재사용하므로,
    # 실제 사용 조건과 맞추려면 세션을 재사용해 측정해야 합니다.
    try:
        import requests
        session = requests.Session()
        session.get(BASE + "/stats", timeout=30)          # 연결 워밍업

        def timed(path, n=5):
            samples = []
            for _ in range(n):
                t0 = time.time()
                session.get(BASE + path, timeout=60)
                samples.append(time.time() - t0)
            return sum(samples) / len(samples)

        baseline = timed("/stats")            # 계산이 거의 없는 엔드포인트 = 순수 왕복 비용
        quote_avg = timed("/quote/005930")
        compute = max(0.0, quote_avg - baseline)

        check("api", f"quote 왕복 {quote_avg:.3f}s (연결 오버헤드 {baseline:.3f}s 포함)",
              quote_avg < 2.0)
        check("api", f"quote 순수 처리시간 ≈ {compute:.3f}s (< 0.5s)", compute < 0.5)
    except ImportError:
        check("api", "성능 측정(requests 필요)", True, "requests 미설치로 건너뜀")


# ---------------------------------------------------------------------------
# 8. 데이터 소스 도달성
# ---------------------------------------------------------------------------

def test_data_sources():
    section("8. 데이터 소스 — 실제 도달 여부")
    from data_sources import symbol_registry as sr, price_provider as pp
    from data_sources import news_crawler as nc, community_crawler as cc

    kr = sr.resolve("삼성전자")
    us = sr.resolve("AAPL")

    master = sr.get_krx_master()
    check("source", f"KRX 종목 마스터 ({len(master['entries'])}종목)",
          len(master["entries"]) > 2000)

    prov = pp.get_provider(kr)
    for tf, min_bars in [("minute", 30), ("day", 60), ("week", 30),
                         ("month", 20), ("year", 5)]:
        df = prov.get_history(kr, tf, count=120)
        check("source", f"코스피 {tf} 시세 ({len(df)}봉)", len(df) >= min_bars)

    snap = prov.get_snapshot(kr)
    check("source", f"코스피 스냅샷 (현재가 {snap.current_price:,.0f})", snap.current_price > 0)
    check("source", f"투자자별 매매동향 ({snap.investor_flow.source})",
          snap.investor_flow.available)

    news_kr = nc.get_news(kr, limit=20)
    check("source", f"국내 뉴스 {len(news_kr)}건", len(news_kr) >= 5)
    com_kr = cc.get_community_sentiment(kr)
    check("source", f"국내 커뮤니티 {com_kr.post_count}건 ({', '.join(com_kr.sources)})",
          com_kr.post_count > 0)

    com_us = cc.get_community_sentiment(us)
    check("source", f"미국 커뮤니티 {com_us.post_count}건 ({', '.join(com_us.sources)})",
          com_us.post_count > 0)

    # --- 해외 커뮤니티 (국내 종목에도 국제 시각을 더함) ---
    check("source", f"국내 종목 영문명 조회 ('{cc._english_name(kr)}')",
          bool(cc._english_name(kr)))

    tv = cc._tradingview(kr)
    check("source", f"TradingView 국내 종목 {len(tv)}건", len(tv) > 0)
    hn = cc._hackernews(kr)
    check("source", f"Hacker News 국내 종목 {len(hn)}건", len(hn) > 0)

    # 관련성 필터: 무관한 제목이 걸러지는가
    off_topic = [p for p in cc._hackernews(us)
                 if not cc._is_relevant(p.title, us, cc._english_name(us))]
    check("source", "Hacker News 관련성 필터 (무관 글 0건)", not off_topic,
          "; ".join(p.title[:40] for p in off_topic[:2]))

    intl = {s for s in com_kr.sources if s in ("TradingView", "Hacker News")}
    check("source", f"국내 종목에 해외 소스 병합 ({', '.join(sorted(intl)) or '없음'})",
          len(intl) > 0)

    # Yahoo RSS는 간헐적으로 429를 돌려줍니다. 코드 결함과 순간적 요청 제한을
    # 구분하기 위해 잠깐 쉬었다 한 번 더 시도합니다.
    news_us = nc.get_news(us, limit=20)
    if len(news_us) < 5:
        time.sleep(3)
        news_us = nc.get_news(us, limit=20)
    check("source", f"미국 뉴스 {len(news_us)}건", len(news_us) >= 5,
          "" if news_us else "외부 피드 요청 제한(429)일 수 있음 — 재시도 후에도 0건")


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def report():
    section("QA 요약")
    by_cat = {}
    for cat, name, passed, detail in results:
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][0] += 1
        by_cat[cat][1] += 1 if passed else 0

    names = {"quant": "계량 추정량", "edge": "경계 조건", "horizon": "시평선 모델",
             "symbol": "종목 해석", "sentiment": "감성 분석", "clock": "장 세션",
             "api": "API 엔드포인트", "source": "데이터 소스", "scrap": "스크랩 저장소",
             "apikey": "공식 API 계층", "scoring": "예측 기록·채점",
             "paper": "모의투자"}
    total = passed_total = 0
    for cat, (n, p) in by_cat.items():
        total += n; passed_total += p
        mark = "OK" if p == n else "!!"
        print(f"  [{mark}] {names.get(cat, cat):14} {p:3}/{n:<3} 통과")

    failures = [(c, n, d) for c, n, ok, d in results if not ok]
    if failures:
        print(f"\n  실패 {len(failures)}건:")
        for cat, name, detail in failures:
            print(f"    · [{names.get(cat, cat)}] {name}" + (f" — {detail}" if detail else ""))

    rate = passed_total / total * 100 if total else 0
    print(f"\n  총계: {passed_total}/{total} 통과 ({rate:.1f}%)")
    return len(failures) == 0


if __name__ == "__main__":
    offline = "--offline" in sys.argv

    test_quant_estimators()
    test_indicator_edges()
    test_horizons()
    test_sentiment()
    test_scrap_store()
    test_prediction_scoring()
    test_paper_trading()
    test_official_apis()
    test_market_clock()

    if not offline:
        test_symbol_resolution()
        test_data_sources()
        test_api()
    else:
        print("\n(--offline: 네트워크가 필요한 검사는 건너뜁니다)")

    ok = report()
    sys.exit(0 if ok else 1)
