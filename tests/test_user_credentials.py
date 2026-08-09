# -*- coding: utf-8 -*-
"""사용자별 API 키 QA
--------------------
    python tests/test_user_credentials.py

네트워크도 MongoDB도 필요 없습니다 (가짜 Mongo). 이 파일이 지키는 것:

    · 암호화     Atlas 에 평문이 들어가지 않는가. 키가 바뀌면 조용히 실패하는가
    · 격리       A 의 키가 B 의 컨텍스트에서 보이지 않는가. 컨텍스트가 끝나면
                 오버레이가 사라지는가 (오래 사는 스레드에서의 누수가 가장 무섭습니다)
    · 화이트리스트 사용자가 MONGODB_URI 같은 서버 설정을 갈아끼울 수 없는가
    · 실거래 게이트 구글 계정이 자기 키 없이 KIS 주문 모드로 못 들어가는가,
                 서버의 실거래 스위치를 물려받지 않는가
    · 마스킹     화면 자료에 비밀값 전체가 실려 나가지 않는가

설계 근거는 storage/user_credentials.py 모듈 docstring 과 ACCOUNTS.md 에 있습니다.
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}  {detail}")


from data_sources import credentials
from storage import accounts, user_credentials as uc

# ---------------------------------------------------------------------------
# 준비: credentials 를 가로채고 (서버 키를 고정), 가짜 Mongo 를 답니다
# ---------------------------------------------------------------------------

_real_cred_get = credentials.get
_server = {
    "KIS_APP_KEY": "server-app-key",
    "KIS_APP_SECRET": "server-app-secret",
    "KIS_ACCOUNT": "99999999-01",
    "KIS_LIVE_TRADING": "1",            # 서버는 실전 — 이게 새면 안 되는 값입니다
    "NAVER_CLIENT_ID": "server-naver",
    "MONGODB_URI": "mongodb://server-only",
    "ATHENA_CRED_KEY": "",              # 아래에서 채웁니다
}


def _patched_get(name, default=""):
    over = credentials._from_overlay(name)
    if over is not None:
        return over
    if name in _server:
        return _server[name]
    return _real_cred_get(name, default)


credentials.get = _patched_get

# 암호화 키는 파일(api_keys.json)에 쓰지 않도록 미리 만들어 끼웁니다
from cryptography.fernet import Fernet

_server["ATHENA_CRED_KEY"] = Fernet.generate_key().decode("ascii")


class FakeCollection:
    def __init__(self):
        self.docs = []

    def find_one(self, spec, projection=None):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in spec.items()):
                return doc
        return None

    def _apply(self, doc, update):
        for path, value in update.get("$set", {}).items():
            parts = path.split(".")
            target = doc
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        for path in update.get("$unset", {}):
            parts = path.split(".")
            target = doc
            for part in parts[:-1]:
                target = target.get(part) or {}
            target.pop(parts[-1], None)

    def update_one(self, spec, update, upsert=False):
        doc = self.find_one(spec)
        if doc is None:
            if not upsert:
                return
            doc = dict(spec)
            self.docs.append(doc)
        self._apply(doc, update)


class FakeDB:
    def __init__(self):
        self.user_credentials = FakeCollection()

    def __getitem__(self, name):
        return getattr(self, name)


fake = FakeDB()
_real_db = accounts._db
accounts._db = lambda: fake

ALICE = accounts.USER_ID_OFFSET + 800_000_001      # 구글 계정 (검사 대역)
BOB = accounts.USER_ID_OFFSET + 800_000_002        # 구글 계정
LOCAL = 3                                          # 로컬 계정

try:
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  1. 저장 — 화이트리스트와 암호화")
    print("=" * 70)

    result = uc.save_keys(ALICE, {
        "KIS_APP_KEY": "alice-app-key",
        "KIS_APP_SECRET": "alice-app-secret",
        "KIS_ACCOUNT": "11111111-01",
        "NAVER_CLIENT_ID": "alice-naver",
    })
    check("정상 키 저장", result["ok"] and len(result["saved"]) == 4, str(result))

    stored = fake.user_credentials.find_one({"user_id": ALICE})
    raw = str(stored)
    check("Mongo 문서에 평문이 없다 (암호화 확인)",
          "alice-app-key" not in raw and "alice-app-secret" not in raw,
          "Atlas 유출 = 계좌 유출이 되면 안 됩니다")
    check("복호화하면 원래 값", uc.load_keys(ALICE)["KIS_APP_KEY"] == "alice-app-key")

    bad = uc.save_keys(ALICE, {"MONGODB_URI": "mongodb://evil", "KIS_APP_KEY": "x"})
    check("서버 설정 키는 거부된다", "MONGODB_URI" in bad["rejected"], str(bad))
    check("거부되면 ok=False", bad["ok"] is False)
    check("거부되면 아무것도 저장되지 않는다 (원자적)",
          uc.load_keys(ALICE)["KIS_APP_KEY"] == "alice-app-key",
          "반쯤 저장되면 화면과 실제가 어긋납니다")
    for name in ("FIREBASE_API_KEY", "GOOGLE_CLIENT_SECRET", "ATHENA_CRED_KEY",
                 "ATHENA_PUBLIC_ORIGIN"):
        r = uc.save_keys(ALICE, {name: "x"})
        check(f"거부: {name}", name in r["rejected"])

    uc.save_keys(ALICE, {"NAVER_CLIENT_ID": ""})
    check("빈 값 저장 = 그 키 삭제", "NAVER_CLIENT_ID" not in uc.load_keys(ALICE))

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  2. 오버레이 — 우선순위와 격리")
    print("=" * 70)

    uc.save_keys(BOB, {"KIS_APP_KEY": "bob-app-key"})

    with credentials.use_user(uc.overlay_for(ALICE)):
        check("내 컨텍스트에서는 내 키", credentials.get("KIS_APP_KEY") == "alice-app-key")
        check("없는 키는 서버로 폴백 (네이버)",
              credentials.get("NAVER_CLIENT_ID") == "server-naver")
        with credentials.use_user(uc.overlay_for(BOB)):
            check("중첩: 안쪽 컨텍스트는 B 의 키",
                  credentials.get("KIS_APP_KEY") == "bob-app-key")
        check("중첩이 끝나면 다시 내 키", credentials.get("KIS_APP_KEY") == "alice-app-key")

    check("컨텍스트 밖에서는 서버 키", credentials.get("KIS_APP_KEY") == "server-app-key")

    # 다른 스레드로 새지 않는가 — 자동매매 루프가 사용자를 번갈아 도는 상황
    seen = {}

    def _worker():
        seen["value"] = credentials.get("KIS_APP_KEY")

    with credentials.use_user(uc.overlay_for(ALICE)):
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
    check("먼저 뜬 다른 스레드로는 오버레이가 새지 않는다",
          seen["value"] == "server-app-key", f"보인 값: {seen['value']}")

    # 오버레이에 심어도 화이트리스트 밖이면 읽는 쪽이 거른다 (2중 방어)
    with credentials.use_user({"MONGODB_URI": "mongodb://evil"}):
        check("읽는 쪽 화이트리스트: MONGODB_URI 오버레이 무시",
              credentials.get("MONGODB_URI") == "mongodb://server-only")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  3. 실거래 안전선 — 서버 스위치를 물려받지 않는다")
    print("=" * 70)

    # 서버는 KIS_LIVE_TRADING=1 (실전) 인 상태입니다
    with credentials.use_user(uc.overlay_for(ALICE)):
        check("구글 계정은 서버의 실거래 스위치를 물려받지 않는다",
              credentials.get_bool("KIS_LIVE_TRADING") is False,
              "자기가 켠 적 없는 실거래가 켜지면 안 됩니다")
        check("구글 계정은 서버 계좌번호를 물려받지 않는다",
              credentials.get("KIS_ACCOUNT") == "11111111-01",
              "자기 계좌를 저장했으니 자기 것이어야 합니다")
        check("모의 서버가 기본값", credentials.get("KIS_MOCK") == "1")

    with credentials.use_user(uc.overlay_for(BOB)):
        # BOB 은 계좌를 저장하지 않았습니다
        check("계좌 미저장 구글 계정에게 서버 계좌가 보이지 않는다",
              credentials.get("KIS_ACCOUNT") == "",
              "서버 주인의 계좌로 주문이 나가는 경로를 원천 차단")

    with credentials.use_user(uc.overlay_for(LOCAL)):
        check("로컬 계정은 서버 값 그대로 (기존 동작 유지)",
              credentials.get("KIS_APP_KEY") == "server-app-key"
              and credentials.get_bool("KIS_LIVE_TRADING") is True)

    from engine import autotrade

    check("구글 계정 + 자기 키 없음 → KIS 모의 주문 차단",
          autotrade._own_keys_gate(BOB, "mock") != "")
    check("구글 계정 + 자기 키 없음 → KIS 실전 주문 차단",
          autotrade._own_keys_gate(BOB, "live") != "")
    check("가상 모의투자(paper)는 키 없이 허용",
          autotrade._own_keys_gate(BOB, "paper") == "")
    check("구글 계정 + 자기 키 있음 → 허용",
          autotrade._own_keys_gate(ALICE, "mock") == "")
    check("로컬 계정은 서버 키로 허용 (기존 동작)",
          autotrade._own_keys_gate(LOCAL, "live") == "")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  4. 화면 자료 — 비밀값이 실려 나가지 않는다")
    print("=" * 70)

    view = uc.overview(ALICE)
    text = str(view)
    check("전체 값이 어디에도 없다",
          "alice-app-key" not in text and "alice-app-secret" not in text)
    kis = view["providers"]["kis"]
    field = next(f for f in kis["fields"] if f["name"] == "KIS_APP_KEY")
    check("마스킹은 끝 4자리만", field["masked"] == "····-key", field["masked"])
    check("서버 폴백 여부를 알려준다 (네이버)",
          next(f for f in view["providers"]["naver"]["fields"]
               if f["name"] == "NAVER_CLIENT_ID")["server_fallback"] is True)
    check("실거래 준비 상태", view["kis_trade_ready"] is True)
    check("BOB 은 실거래 준비 안 됨", uc.overview(BOB)["kis_trade_ready"] is False)

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  5. 삭제와 캐시")
    print("=" * 70)

    uc.save_keys(ALICE, {"KIS_APP_KEY": "alice-key-2"})
    check("저장 직후 캐시가 비워져 새 값이 보인다",
          uc.load_keys(ALICE)["KIS_APP_KEY"] == "alice-key-2",
          "잘못된 키를 고쳤는데 1분간 옛 키로 돌면 안 됩니다")

    gone = uc.delete_provider(ALICE, "kis")
    check("서비스 단위 삭제", gone["ok"] and "KIS_APP_KEY" not in uc.load_keys(ALICE))
    check("삭제하면 실거래 준비도 풀린다", uc.has_own_kis_keys(ALICE) is False)
    check("모르는 서비스는 거부", uc.delete_provider(ALICE, "mongo")["ok"] is False)

    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  6. 암호화 키 사고 — 조용한 오동작이 없어야 합니다")
    print("=" * 70)

    uc.save_keys(BOB, {"KIS_APP_SECRET": "bob-secret"})
    _server["ATHENA_CRED_KEY"] = Fernet.generate_key().decode("ascii")   # 키 분실 상황
    uc._invalidate(BOB)
    after = uc.load_keys(BOB)
    check("복호화 키가 바뀌면 그 값은 조용히 제외 (예외로 죽지 않음)",
          "KIS_APP_SECRET" not in after, str(after))
    check("읽을 수 없는 키는 '없음'으로 — 실거래 게이트가 자동으로 닫힌다",
          uc.has_own_kis_keys(BOB) is False)

finally:
    accounts._db = _real_db
    credentials.get = _real_cred_get

# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"  통과 {len(PASS)} · 실패 {len(FAIL)}")
print("=" * 70)
if FAIL:
    for name in FAIL:
        print(f"  FAIL  {name}")
sys.exit(1 if FAIL else 0)
