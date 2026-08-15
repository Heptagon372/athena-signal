# -*- coding: utf-8 -*-
"""
손실 학습 QA (engine/lessons.py)
--------------------------------
학습은 점수를 깎는 기능이라, "배운다"보다 "엉뚱한 것을 배우지 않는다"를
확인해야 합니다. 표본 부족이면 침묵하는지, 상한을 지키는지, 부호를 뒤집지
않는지가 정상 경로만큼 중요합니다.

    python tests/test_lessons.py     # 네트워크·서버 없이 전부 오프라인
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from engine import lessons, strategy
from storage import autotrade as store

TEST_USER = 999_901          # 실제 계정과 겹치지 않는 테스트 전용 id
MODE = "paper"
results = []


def check(name, passed, detail=""):
    results.append((name, bool(passed), detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return passed


def section(title):
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def fresh():
    """테스트 흔적을 지우고 캐시를 비웁니다 — 이전 실행이 남긴 것을 배우면 안 됩니다."""
    store.lessons_purge(TEST_USER)
    lessons.invalidate(TEST_USER, MODE)


def snap(**kw) -> dict:
    base = {"score": 0.5, "direction": "long", "regime": "상승추세",
            "daily_score": 0.4, "intraday_score": 0.6, "news_score": None,
            "rsi": 55.0, "bb_pct": 0.5, "atr_pct": 2.0, "vol_factor": 1.0}
    base.update(kw)
    return base


def fake_signal(**kw) -> strategy.Signal:
    sig = strategy.Signal(key="TEST01", ok=True)
    sig.score = kw.get("score", 0.5)
    sig.regime = kw.get("regime", "상승추세")
    sig.daily_score = kw.get("daily_score", 0.4)
    sig.intraday_score = kw.get("intraday_score", 0.6)
    sig.news_score = kw.get("news_score")
    sig.rsi = kw.get("rsi", 55.0)
    sig.bb_pct = kw.get("bb_pct", 0.5)
    sig.atr_pct = kw.get("atr_pct", 2.0)
    return sig


# ---------------------------------------------------------------------------
section("1. 컨텍스트 태그 — 진입과 신호 계산이 같은 언어를 쓰는가")

tags = lessons.tags_of(snap(), symbol="TEST01")
check("국면 태그", "국면:상승추세" in tags, str(tags))
check("주도 성분 태그 (분봉이 최대)", "주도:분봉" in tags)
check("종목 태그", "종목:TEST01" in tags)

tags = lessons.tags_of(snap(rsi=75.0))
check("RSI 70 이상 롱 → 과열진입", "과열진입" in tags)
tags = lessons.tags_of(snap(rsi=75.0, score=-0.5))
check("숏 방향이면 RSI 75 는 과열이 아님", "과열진입" not in tags)

tags = lessons.tags_of(snap(daily_score=-0.2, intraday_score=0.5))
check("일봉·분봉 부호 반대 + 간격 → 불일치", "일봉분봉불일치" in tags)
tags = lessons.tags_of(snap(daily_score=0.1, intraday_score=0.2))
check("같은 방향이면 불일치 아님", "일봉분봉불일치" not in tags)

tags = lessons.tags_of(snap(news_score=0.5, daily_score=0.05))
check("뉴스가 세고 일봉이 약함 → 뉴스의존", "뉴스의존" in tags)
tags = lessons.tags_of(snap(news_score=0.5, daily_score=0.4))
check("일봉이 받쳐주면 뉴스의존 아님", "뉴스의존" not in tags)

tags = lessons.tags_of(snap(atr_pct=5.5))
check("ATR 4% 이상 → 고변동성", "고변동성" in tags)
check("값이 없어도 죽지 않음", isinstance(lessons.tags_of({}), list))

# ---------------------------------------------------------------------------
section("2. 청산 사유 분류와 원인 추론")

check("손절 분류", lessons.exit_class("손절 도달 (900 vs 950, -5.00%)") == "손절")
check("트레일링 분류", lessons.exit_class("고점 대비 3.20% 되돌림 (트레일링 3%)") == "트레일링")
check("시간 청산 분류", lessons.exit_class("보유 4일 (최대 3일) — 시간 청산") == "시간청산")
check("신호 반전 분류", lessons.exit_class("신호 반전 (점수 -0.21)") == "신호반전")

cause = lessons.diagnose(snap(daily_score=-0.2, intraday_score=0.5),
                         ["일봉분봉불일치"], "손절 도달", -4.2)
check("불일치 손실 → 단기 추격 진단", "반대" in cause and "-4.20%" in cause, cause)
cause = lessons.diagnose(snap(rsi=76.0), ["과열진입"], "손절 도달", -3.0)
check("과열 손실 → 추격 매수 진단", "RSI 76" in cause, cause)
cause = lessons.diagnose(snap(atr_pct=5.0), ["고변동성"], "손절 도달", -2.0)
check("고변동성 손절 → 노이즈 손절 진단", "변동성" in cause, cause)
cause = lessons.diagnose(snap(), [], "보유 3일 (최대 3일) — 시간 청산", -1.0)
check("시간 청산 → 근거 약함 진단", "방향이 나오지 않았습니다" in cause, cause)
cause = lessons.diagnose(snap(), ["주도:일봉"], "외부 청산", None)
check("손익률을 몰라도 문장이 성립", "%" not in cause.split("(")[-1], cause)

# ---------------------------------------------------------------------------
section("3. 저장소 원장 — 열고, 중복 없이, 닫는다")

fresh()
lid = store.lesson_open(TEST_USER, MODE, "TEST01", name="테스트종목", side="long",
                        entry_price=10_000, quantity=10,
                        context=snap(), tags=["국면:상승추세", "주도:분봉"])
check("교훈 생성", isinstance(lid, int) and lid > 0)
lid2 = store.lesson_open(TEST_USER, MODE, "TEST01", entry_price=10_100, quantity=5)
check("같은 종목 중복 생성 안 함 (추가 매수는 같은 매매)", lid2 == lid)

row = store.lesson_find_open(TEST_USER, MODE, "TEST01")
check("열린 교훈 조회", row and row["entry_price"] == 10_000 and
      row["tags"] == ["국면:상승추세", "주도:분봉"])
check("없는 종목은 None", store.lesson_find_open(TEST_USER, MODE, "NOPE") is None)

store.lesson_close(lid, exit_price=9_500, exit_reason="손절 도달",
                   realized_krw=-5_000, pnl_pct=-5.0, cause="테스트 원인")
check("닫힌 뒤에는 열린 교훈이 없음",
      store.lesson_find_open(TEST_USER, MODE, "TEST01") is None)
closed = store.lessons_closed(TEST_USER, MODE)
check("닫힌 교훈 집계 조회", len(closed) == 1 and closed[0]["pnl_pct"] == -5.0
      and closed[0]["cause"] == "테스트 원인")

# ---------------------------------------------------------------------------
section("4. 엔진 훅 — close_lesson 이 손익을 계산하고 원인을 남기는가")

fresh()
lessons.open_lesson(TEST_USER, MODE, "TEST02", "테스트2", "long",
                    snap(rsi=76.0, score=0.6), 20_000, 5)
lessons.close_lesson(TEST_USER, MODE, "TEST02", name="테스트2",
                     exit_reason="손절 도달 (19000 vs 19400)", exit_price=19_000,
                     realized_krw=-5_200)
closed = store.lessons_closed(TEST_USER, MODE)
check("손익률 자동 계산 (-5%)", len(closed) == 1
      and abs(closed[0]["pnl_pct"] - (-5.0)) < 0.01,
      f"pnl={closed[0]['pnl_pct'] if closed else None}")
check("손실이면 원인 추론이 붙음", closed and "RSI 76" in closed[0]["cause"],
      closed[0]["cause"] if closed else "")

lessons.open_lesson(TEST_USER, MODE, "TEST03", "테스트3", "long",
                    snap(), 10_000, 5)
lessons.close_lesson(TEST_USER, MODE, "TEST03", exit_reason="목표가 도달",
                     exit_price=11_000, realized_krw=+5_000)
closed = store.lessons_closed(TEST_USER, MODE)
win = next(r for r in closed if r["symbol"] == "TEST03")
check("이긴 매매도 기록 (집계용)", abs(win["pnl_pct"] - 10.0) < 0.01)
check("이긴 매매에는 원인 추론 없음", win["cause"] == "")

lessons.open_lesson(TEST_USER, MODE, "TEST04", "테스트4", "long", snap(), 10_000, 5)
lessons.close_lesson(TEST_USER, MODE, "TEST04", exit_reason="외부 청산 (계좌 대조로 정리)")
closed = store.lessons_closed(TEST_USER, MODE)
gone = next(r for r in closed if r["symbol"] == "TEST04")
check("외부 청산은 손익 없이 닫힘 (집계 제외)", gone["pnl_pct"] is None)

# ---------------------------------------------------------------------------
section("5. 감쇠표 — 표본이 모자라면 침묵하고, 모이면 배운다")

fresh()
cfg = dict(store.DEFAULT_CONFIG)

# 같은 태그(과열진입)로 2번 잃음 — learn_min_trades=3 미달
for i in range(2):
    lid = store.lesson_open(TEST_USER, MODE, f"LOSS{i}", entry_price=10_000,
                            quantity=1, context=snap(rsi=75.0),
                            tags=["과열진입", "국면:상승추세"])
    store.lesson_close(lid, exit_price=9_600, exit_reason="손절 도달",
                       realized_krw=-400, pnl_pct=-4.0)
lessons.invalidate(TEST_USER, MODE)
pen = lessons.build_penalties(TEST_USER, MODE, cfg)
under = pen is None or "과열진입" not in (pen or {}).get("tags", {})
check("유효 표본 2회 → 아직 판단 보류", under, str(pen))

# 3번째 손실 — 이제 배울 수 있음
lid = store.lesson_open(TEST_USER, MODE, "LOSS2", entry_price=10_000,
                        quantity=1, context=snap(rsi=75.0),
                        tags=["과열진입", "국면:상승추세"])
store.lesson_close(lid, exit_price=9_600, exit_reason="손절 도달",
                   realized_krw=-400, pnl_pct=-4.0)
lessons.invalidate(TEST_USER, MODE)
pen = lessons.build_penalties(TEST_USER, MODE, cfg)
check("표본 3회부터 감쇠표에 등장", pen and "과열진입" in pen["tags"], str(pen))
info = pen["tags"]["과열진입"]
check("평균 -4% > scale -3% → 태그 감쇠가 상한(0.10)",
      abs(info["pen"] - cfg["learn_max_penalty"]) < 1e-9, str(info))

# 이긴 매매가 섞이면 기대수익이 개선되어 감쇠가 줄어야 함 (선택 편향 방지)
for i in range(3):
    lid = store.lesson_open(TEST_USER, MODE, f"WIN{i}", entry_price=10_000,
                            quantity=1, context=snap(rsi=75.0),
                            tags=["과열진입"])
    store.lesson_close(lid, exit_price=10_500, exit_reason="목표가 도달",
                       realized_krw=+500, pnl_pct=+5.0)
lessons.invalidate(TEST_USER, MODE)
pen2 = lessons.build_penalties(TEST_USER, MODE, cfg)
softer = (pen2 is None or "과열진입" not in pen2["tags"]
          or pen2["tags"]["과열진입"]["pen"] < info["pen"])
check("승리 표본이 섞이면 감쇠 축소 (편향 없이 기대수익으로)", softer, str(pen2))

check("learn_mode=off 면 감쇠표 없음",
      lessons.build_penalties(TEST_USER, MODE, {**cfg, "learn_mode": "off"}) is None)
check("inject 는 off 면 cfg 그대로",
      "_lesson_penalties" not in lessons.inject(TEST_USER, MODE,
                                                {**cfg, "learn_mode": "off"}))

# ---------------------------------------------------------------------------
section("6. 점수 적용 — observe 는 안 바꾸고, soft 는 깎되 상한과 부호를 지킨다")

fresh()
# 과열진입 태그에 강한 손실 이력을 만들어 둡니다
for i in range(4):
    lid = store.lesson_open(TEST_USER, MODE, f"HOT{i}", entry_price=10_000,
                            quantity=1, context=snap(rsi=75.0),
                            tags=["과열진입", "종목:TEST01"])
    store.lesson_close(lid, exit_price=9_500, exit_reason="손절 도달",
                       realized_krw=-500, pnl_pct=-5.0)
lessons.invalidate(TEST_USER, MODE)

observe_cfg = {**cfg, "learn_mode": "observe"}
observe_cfg = lessons.inject(TEST_USER, MODE, observe_cfg)
check("observe 도 감쇠표는 실림", "_lesson_penalties" in observe_cfg)
sig = fake_signal(score=0.6, rsi=76.0)
new, note = lessons.apply_to_score(0.6, sig, observe_cfg)
check("observe — 점수 유지", new == 0.6)
check("observe — 예정 감쇠를 설명", "soft 면" in note, note)

soft_cfg = lessons.inject(TEST_USER, MODE, {**cfg, "learn_mode": "soft"})
new, note = lessons.apply_to_score(0.6, sig, soft_cfg)
check("soft — 점수 감쇠", new < 0.6, f"{0.6} → {new} ({note})")
expected = 0.6 - min(cfg["learn_total_cap"],
                     soft_cfg["_lesson_penalties"]["tags"]["과열진입"]["pen"]
                     + soft_cfg["_lesson_penalties"]["tags"]["종목:TEST01"]["pen"])
check("soft — 감쇠량이 계산식과 일치", abs(new - expected) < 1e-9,
      f"기대 {expected:.4f}, 실제 {new:.4f}")

cool = fake_signal(score=0.6, rsi=50.0)
cool.key = "OTHER"
new, note = lessons.apply_to_score(0.6, cool, soft_cfg)
check("해당 태그가 없으면 그대로", new == 0.6 and note == "")

hot_short = fake_signal(score=-0.4, rsi=76.0)     # 숏 신호 — 과열 태그 안 붙음
new, _ = lessons.apply_to_score(-0.4, hot_short, soft_cfg)
check("부호를 뒤집지 않음 (음수 신호는 음수 유지)", new <= 0.0, str(new))

tiny = fake_signal(score=0.05, rsi=76.0)
new, _ = lessons.apply_to_score(0.05, tiny, soft_cfg)
check("감쇠가 커도 0 아래로 뚫지 않음", new >= 0.0, str(new))

# 총합 상한 — 태그를 잔뜩 맞춰도 learn_total_cap 을 넘지 못함
many = {"mode": "soft",
        "tags": {t: {"e": -5.0, "n": 10.0, "pen": 0.10}
                 for t in ("국면:상승추세", "주도:분봉", "과열진입", "종목:TEST01")}}
sig_all = fake_signal(score=0.9, rsi=76.0, intraday_score=0.8)
new, _ = lessons.apply_to_score(0.9, sig_all, {**cfg, "_lesson_penalties": many})
check("감쇠 총합이 learn_total_cap(0.25) 에서 멈춤",
      abs((0.9 - new) - cfg["learn_total_cap"]) < 1e-9, f"감쇠 {0.9 - new:.3f}")

# ---------------------------------------------------------------------------
section("7. 반감기 — 오래된 실수는 잊혀진다")

fresh()
old_day = (datetime.now() - timedelta(days=140)).isoformat()   # 반감기 14일 × 10
for i in range(3):
    lid = store.lesson_open(TEST_USER, MODE, f"OLD{i}", entry_price=10_000,
                            quantity=1, context=snap(), tags=["고변동성"])
    store.lesson_close(lid, exit_price=9_000, exit_reason="손절 도달",
                       realized_krw=-1_000, pnl_pct=-10.0)
with store._conn() as conn:
    conn.execute("UPDATE at_lessons SET closed_at = ? WHERE user_id = ?",
                 (old_day, TEST_USER))
lessons.invalidate(TEST_USER, MODE)
pen = lessons.build_penalties(TEST_USER, MODE, cfg)
check("140일 전 손실 3건 → 유효 표본이 줄어 침묵",
      pen is None or "고변동성" not in (pen or {}).get("tags", {}), str(pen))

# ---------------------------------------------------------------------------
fresh()
failed = [r for r in results if not r[1]]
print(f"\n{'=' * 68}")
print(f"  결과: {len(results) - len(failed)}/{len(results)} 통과"
      + (f" — 실패 {len(failed)}건" if failed else ""))
for name, _, detail in failed:
    print(f"    FAIL {name}" + (f" — {detail}" if detail else ""))
print("=" * 68)
sys.exit(1 if failed else 0)
