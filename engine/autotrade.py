"""
자동매매 엔진 (Auto Trading Engine)
----------------------------------
한 번의 회전(tick)에서 하는 일은 정해져 있습니다.

    1. 설정을 읽고 브로커를 준비한다
    2. 계좌·포지션 스냅샷을 뜬다
    3. 오늘의 기준 평가금액을 갱신한다 (손실 한도 판정용)
    4. 안전장치를 확인한다 — 걸리면 신규 진입 금지, 청산은 계속
    5. **청산 먼저** 검사한다 (손절이 늦으면 안 됩니다)
    6. 남은 여력으로 신규 진입을 검사한다
    7. 모든 판단을 로그로 남긴다

왜 청산이 먼저인가
    진입을 먼저 처리하면 현금을 다 쓴 뒤에 손절해야 할 종목을 발견합니다.
    실제 사고는 대부분 "팔았어야 할 때 사고 있었다"에서 나옵니다.

중복 주문 방지 3중 장치
    1) client_order_id = 사용자:종목:동작:분 → 같은 분에 같은 주문은 한 번만
    2) at_orders.client_order_id UNIQUE 제약 (DB 레벨)
    3) 사용자별 실행 락 — 루프와 수동 실행이 겹쳐도 동시에 돌지 않습니다

이 엔진은 **실제로 주문을 냅니다**. mode=paper 면 내부 가상계좌, mock 이면
KIS 모의투자 서버, live 면 실제 자금입니다. dry_run=True 면 판단만 하고
주문은 내지 않습니다(설정 검증용).
"""

import math
import threading
import time
import traceback
from datetime import datetime

import numpy as np

from data_sources import credentials
from engine import broker as broker_module
from engine import (feed, fills, holding, instruments, portfolio, protections,
                    risk, rotation, split, strategy, validation)
from engine.broker import make_client_order_id
from engine.instruments import Instrument
from storage import autotrade as store

# 포지션의 주인 — 초단타와 일반 자동매매의 경계입니다. 왜 나누는지는 아래
# "초단타와 일반 자동매매의 경계" 단락에 적어 두었습니다.
# 이 두 값이 여기(맨 위)에 있어야 하는 이유: split_applies() 가 이 값을 기본
# 인자로 씁니다. 기본 인자는 def 를 만나는 순간, 즉 임포트 중에 계산되므로
# 함수보다 뒤에 정의하면 모듈이 통째로 임포트되지 않습니다(NameError).
OWNER_AUTO, OWNER_SCALP = "auto", "scalp"

# 사용자별 실행 락 — 같은 계좌에 두 회전이 동시에 들어가면 잔고 계산이 깨집니다
_user_locks: dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def _user_lock(user_id: int) -> threading.Lock:
    with _locks_guard:
        if user_id not in _user_locks:
            _user_locks[user_id] = threading.Lock()
        return _user_locks[user_id]


# 같은 사유의 신호 실패를 매 회전 기록하면(60초마다) 로그가 그 얘기로 가득 찹니다.
# 그렇다고 조용히 넘기면 "왜 이 종목만 아무 일도 안 하지"를 알 수 없습니다.
# 사유가 바뀌거나 30분이 지났을 때만 남깁니다.
_SIGNAL_ERROR_REPEAT_SEC = 1800
_last_signal_error: dict[tuple, tuple] = {}

# "이 종목은 관리 안 합니다" 안내를 매 회전 찍지 않기 위한 기록
_skipped_notice: dict[int, set] = {}

# 브로커가 주문을 거부한 종목의 재시도 대기 — 증권사가 거부하는 사유(외화 부족,
# 미신청 서비스 등)는 다음 회전에 저절로 풀리지 않습니다. 그대로 두면 60초마다
# 같은 주문을 계속 내서 실계좌 원장이 거부 기록으로 가득 찹니다.
_BROKER_REJECT_COOLDOWN_SEC = 1800
_broker_rejects: dict[tuple, tuple] = {}


def _reject_cooldown_left(user_id: int, key: str) -> tuple[float, str]:
    """(남은 대기 초, 사유). 대기 중이 아니면 (0, "")."""
    hit = _broker_rejects.get((user_id, key))
    if not hit:
        return 0.0, ""
    left = _BROKER_REJECT_COOLDOWN_SEC - (time.time() - hit[0])
    return (left, hit[1]) if left > 0 else (0.0, "")


def note_broker_reject(user_id: int, key: str, reason: str):
    _broker_rejects[(user_id, key)] = (time.time(), reason)


def clear_broker_reject(user_id: int, key: str = ""):
    """설정·잔고를 고쳤을 때 바로 다시 시도할 수 있게 대기를 풉니다."""
    if key:
        _broker_rejects.pop((user_id, key), None)
    else:
        for slot in [s for s in _broker_rejects if s[0] == user_id]:
            _broker_rejects.pop(slot, None)


def _log_signal_error(user_id: int, key: str, name: str, message: str):
    slot = (user_id, key)
    previous = _last_signal_error.get(slot)
    now = time.time()
    if previous and previous[1] == message and now - previous[0] < _SIGNAL_ERROR_REPEAT_SEC:
        return
    _last_signal_error[slot] = (now, message)
    store.log_event(user_id, "reject", f"{name} 신호를 만들 수 없습니다 — {message}",
                    level="warn", symbol=key, name=name)


def _log_plan_reject(user_id: int, key: str, name: str, reason: str, detail: dict = None):
    """진입 불가 사유 기록 — 같은 사유는 30분에 한 번만.

    소액 계좌에서는 '리스크 예산 미달' 같은 사유가 매 회전(60초) 반복되는데,
    그걸 다 기록하면 하룻밤에 수백 건이 쌓여 정작 중요한 로그가 묻힙니다.
    """
    slot = (user_id, f"plan:{key}")
    previous = _last_signal_error.get(slot)
    now = time.time()
    if previous and previous[1] == reason and now - previous[0] < _SIGNAL_ERROR_REPEAT_SEC:
        return
    _last_signal_error[slot] = (now, reason)
    store.log_event(user_id, "reject", f"{name} 진입 불가 — {reason}",
                    symbol=key, name=name, detail=detail)


# ---------------------------------------------------------------------------
# 한 번의 회전
# ---------------------------------------------------------------------------

def _user_credentials_scope(user_id: int):
    """이 사용자의 저장 키를 겹친 credentials 컨텍스트.

    자동매매 루프는 오래 사는 스레드 하나에서 모든 사용자를 돌기 때문에,
    회전마다 그 사용자의 키를 장착하고 **끝나면 반드시 되돌려야** 합니다 —
    안 되돌리면 다음 사용자의 회전이 앞 사람의 KIS 키로 주문을 냅니다.
    use_user() 가 컨텍스트 매니저라 예외가 나도 되돌아갑니다.

    키 조회가 실패하면(Mongo 다운 등) 빈 오버레이 대신 **구글 계정은 안전
    기본값만 있는** 오버레이가 옵니다 (user_credentials.overlay_for 참고).
    """
    from storage import user_credentials
    try:
        return credentials.use_user(user_credentials.overlay_for(user_id))
    except Exception:                                              # noqa: BLE001
        return credentials.use_user(None)


def _own_keys_gate(user_id: int, mode: str) -> str:
    """구글 계정이 KIS 주문 모드를 쓰려면 자기 키가 있어야 합니다.

    반환: 막아야 하면 사유 문자열, 통과면 "".

    서버 키로 폴백시키면 아무 구글 사용자가 자동매매를 켰을 때 **서버 주인의
    실계좌**로 주문이 나갑니다. 로컬 계정(이 PC 에서 만든 계정)은 지금까지처럼
    서버 키를 씁니다 — 기존 동작을 깨지 않습니다.
    """
    if mode == broker_module.PAPER:
        return ""                       # 모의투자는 키가 필요 없습니다
    from storage import accounts, user_credentials
    if user_id < accounts.USER_ID_OFFSET:
        return ""                       # 로컬 계정 — 서버 키 사용 (기존 동작)
    if user_credentials.has_own_kis_keys(user_id):
        return ""
    return ("KIS 모의/실전 주문에는 본인 KIS 키가 필요합니다. "
            "웹 → 설정 → API 키에서 KIS 앱 키·시크릿·계좌번호를 저장해 주세요.")


def run_once(user_id: int, force: bool = False) -> dict:
    """자동매매 1회전. force=True 면 enabled 가 꺼져 있어도 판단만 수행합니다.

    반환값은 이번 회전에서 무엇을 봤고 무엇을 했는지의 요약입니다.
    """
    lock = _user_lock(user_id)
    if not lock.acquire(blocking=False):
        return {"ok": False, "skipped": "이전 회전이 아직 실행 중입니다."}

    started = time.time()
    try:
        with _user_credentials_scope(user_id):
            return _tick(user_id, force=force, started=started)
    except Exception as exc:
        detail = traceback.format_exc(limit=4)
        store.log_event(user_id, "error", f"회전 실패: {type(exc).__name__}: {exc}",
                        level="error", detail={"traceback": detail})
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        lock.release()


# 보호 회전이 쓰는 현재가 신선도 — 손절선을 20초 전 가격으로 재면 짧게
# 도는 의미가 없습니다. 보호 주기와 비슷하게 잡아 같은 값을 두 번 받지 않습니다.
GUARD_QUOTE_MAX_AGE = 5.0

# 전체 회전이 읽어둔 보유 목록 — 보호 회전이 재사용합니다.
# {user_id: (기록시각, mode, [Position, ...])}
#
# 잔고 조회(KIS stock/deriv/overseas 3종)는 5초 캐시가 있어도 보호 주기마다
# 만료돼 매번 다시 붙습니다. 실측으로 그것만 3~7초가 걸렸습니다.
# 보유 **종목과 수량**은 우리가 주문을 내지 않는 한 바뀌지 않으므로, 전체
# 회전이 읽어둔 목록을 그대로 씁니다. 정말 실시간이어야 하는 값(현재가)은
# 보호 회전이 직접 새로 받습니다.
_positions_cache: dict[int, tuple[float, str, list]] = {}
GUARD_POSITIONS_TTL = 300.0        # 이보다 오래된 목록은 믿지 않고 다시 읽습니다


def _remember_positions(user_id: int, mode: str, positions: list):
    _positions_cache[user_id] = (time.time(), mode, list(positions or []))


def invalidate_positions(user_id: int):
    """주문을 냈으면 보유 목록이 곧 바뀝니다 — 캐시를 버립니다."""
    _positions_cache.pop(user_id, None)


def run_guard(user_id: int) -> dict:
    """고속 보호 회전 — 보유 포지션의 **손절·트레일링·시간 청산만** 확인합니다.

    왜 따로 도는가
        전체 회전은 종목마다 일봉·분봉·지표 19종·ML 위원회를 계산해서 몇 초가
        걸립니다. 그래서 주기를 60초보다 짧게 잡기 어렵고, 그 사이 손절선이
        뚫려도 최대 60초를 기다렸습니다. 실계좌에서 그 60초는 손실입니다.

        이 회전은 **보유 종목의 현재가만** 받아서 이미 정해져 있는 손절가·
        트레일링·보유시간과 비교합니다. 종목당 시세 조회 한 번이라 몇 초 주기로
        돌 수 있습니다.

    하지 않는 것
        · 신규 진입 (전체 회전의 몫)
        · 익절·신호 반전 (지표가 없으므로 판단 근거가 없습니다)
        · 계좌 기준선 갱신 (입출금 판별은 전체 회전에서 한 번만)

    청산 주문 경로·리스크 게이트·기록은 전체 회전과 **완전히 같은 코드**를
    씁니다. 빠른 길에만 있는 별도 청산 규칙은 만들지 않습니다.
    """
    cfg = store.get_config(user_id)
    if not cfg.get("enabled"):
        return {"ok": False, "skipped": "자동매매가 꺼져 있습니다."}
    if int(cfg.get("guard_interval_sec", 5) or 0) <= 0:
        return {"ok": False, "skipped": "보호 회전이 꺼져 있습니다."}

    mode = cfg.get("mode", "paper")
    # 지켜야 할 포지션이 없으면 브로커를 부르지도 않습니다 (DB 조회는 공짜).
    states = store.get_position_states(user_id, mode)
    if not states:
        return {"ok": True, "skipped": "보유 포지션 없음", "checked": 0}

    # 보호 회전도 주문(청산)을 내므로 같은 게이트를 지납니다
    blocked = _own_keys_gate(user_id, mode)
    if blocked:
        return {"ok": False, "skipped": blocked}

    lock = _user_lock(user_id)
    if not lock.acquire(blocking=False):
        return {"ok": False, "skipped": "다른 회전이 실행 중입니다."}

    started = time.time()
    result = {"ok": True, "guard": True, "at": datetime.now().isoformat(),
              "mode": mode, "exits": [], "rejects": [], "errors": [],
              "signals": [], "checked": 0, "asleep": 0}
    try:
        with _user_credentials_scope(user_id):
            brk = broker_module.get_broker(user_id, mode, cfg)
            if not brk.health().get("ready"):
                return {"ok": False, "skipped": "브로커 준비 안 됨"}

            cached = _positions_cache.get(user_id)
            if (cached and cached[1] == mode
                    and time.time() - cached[0] < GUARD_POSITIONS_TTL):
                positions = cached[2]
            else:
                positions = brk.positions()
                _remember_positions(user_id, mode, positions)

            # 보호 회전은 청산만 합니다. RiskEngine.check_exit 는 계좌·일자 값을
            # 보지 않으므로(현재가와 장 상태만 봄) 잔고를 다시 읽지 않습니다.
            # 계좌가 차단 상태여도 청산은 계속해야 합니다 — 위험을 줄이는 방향입니다.
            engine_risk = risk.RiskEngine(cfg, {}, positions, {})

            for position in positions:
                if not is_managed(cfg, position.key, states):
                    continue
                # 장이 닫혀 있으면 시세를 받아봐야 주문을 낼 수 없습니다.
                # 여기서 걸러야 밤새 무의미한 시세 조회가 쌓이지 않습니다.
                inst = instruments.try_resolve(position.key)
                if inst is not None and not feed.market_status(inst).get("is_open"):
                    result["asleep"] += 1
                    continue
                result["checked"] += 1
                try:
                    _handle_exit(user_id, cfg, brk, engine_risk, position,
                                 states.get(position.key, {}), result, fast=True)
                except Exception as exc:
                    result["errors"].append(f"{position.key} 보호 검사 실패: {exc}")
    except Exception as exc:
        detail = traceback.format_exc(limit=4)
        store.log_event(user_id, "error",
                        f"보호 회전 실패: {type(exc).__name__}: {exc}",
                        level="error", detail={"traceback": detail})
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        lock.release()

    result["elapsed_sec"] = round(time.time() - started, 2)
    # 아무 일도 없으면 로그를 남기지 않습니다 — 5초마다 찍히면 이벤트 로그가
    # 보호 회전 기록으로만 가득 차서 정작 봐야 할 판단이 묻힙니다.
    if result["exits"] or result["errors"]:
        store.log_event(
            user_id, "guard",
            f"보호 회전 — 확인 {result['checked']}건 / 청산 {len(result['exits'])}건"
            + (f" / 오류 {len(result['errors'])}건" if result["errors"] else ""),
            level="warn" if result["errors"] else "trade",
            detail={"elapsed_sec": result["elapsed_sec"],
                    "exits": result["exits"], "errors": result["errors"][:3]})
    return result


def _tick(user_id: int, force: bool, started: float) -> dict:
    cfg = split_legacy_universe(user_id, store.get_config(user_id))
    if not cfg.get("enabled") and not force:
        return {"ok": False, "skipped": "자동매매가 꺼져 있습니다."}

    mode = cfg.get("mode", "paper")
    result = {
        "ok": True, "at": datetime.now().isoformat(),
        "mode": cfg["mode"], "dry_run": bool(cfg.get("dry_run")),
        "halted": False, "halt_reasons": [],
        "signals": [], "entries": [], "exits": [], "rejects": [], "errors": [],
        "settled": [], "reconciled": [], "waiting": [],
    }

    # -- 브로커 준비 ------------------------------------------------------
    blocked = _own_keys_gate(user_id, mode)
    if blocked:
        store.set_state(user_id, store.HALTED, blocked)
        store.log_event(user_id, "halt", blocked, level="error")
        result.update(ok=False, halted=True, halt_reasons=[blocked])
        return result

    brk = broker_module.get_broker(user_id, cfg["mode"], cfg)
    health = brk.health()
    if not health.get("ready"):
        store.set_state(user_id, store.HALTED, health.get("detail", "브로커 준비 안 됨"))
        store.log_event(user_id, "halt", f"브로커를 쓸 수 없습니다: {health.get('detail')}",
                        level="error", detail=health)
        result.update(ok=False, halted=True, halt_reasons=[health.get("detail", "")])
        return result

    # 계좌를 바꿨으면 성적을 0원으로 되돌립니다. 정산·집계보다 먼저 해야
    # 옛 계좌의 체결이 새 계좌의 오늘 손익에 섞이지 않습니다
    # (storage/autotrade.sync_account).
    if store.sync_account(user_id, mode, broker_module.account_fingerprint(mode)):
        result["account_changed"] = True

    # -- 0단계: 지난 회전에 낸 주문부터 정리 -------------------------------
    # 이걸 먼저 하지 않으면 "체결됐는지도 모르는 주문"이 남은 채로 새 주문을 냅니다.
    _settle_open_orders(user_id, cfg, brk, result)

    account = brk.account()
    positions = brk.positions()
    # 보호 회전이 잔고를 다시 읽지 않도록 이 목록을 남겨둡니다
    _remember_positions(user_id, mode, positions)
    # 평가손익을 넘겨야 오늘 손익이 입출금과 무관해집니다
    # (storage/autotrade.touch_daily 참고).
    day = store.touch_daily(user_id, mode, account.get("total_value") or 0,
                            unrealized=sum(p.unrealized_pnl or 0 for p in positions),
                            cash=account.get("cash"),
                            settled=len(result.get("settled") or []))

    # -- 0.5단계: 실계좌와 내부 상태 대조 ---------------------------------
    # 잔고 조회가 일부라도 실패했으면 이 목록은 보유분 전체가 아닙니다.
    # 그 상태로 대조하면 "안 보이는 = 청산됨"으로 오판해 손절선을 지웁니다.
    _reconcile(user_id, cfg, positions, result,
               partial=bool(getattr(brk, "position_errors", [])))
    result["account"] = account
    result["day"] = day

    engine_risk = risk.RiskEngine(cfg, account, positions, day)
    halts = engine_risk.halt_reasons()
    result["halt_reasons"] = halts
    result["halted"] = bool(halts)

    if halts:
        store.set_state(user_id, store.HALTED, " / ".join(halts))
        store.log_event(user_id, "halt", "신규 진입 중단: " + " / ".join(halts),
                        level="warn", detail={"account": account, "day": day})
    elif cfg.get("enabled"):
        store.set_state(user_id, store.RUNNING, "")

    # -- 0.7단계: 초단타 대상 자동 추적 -----------------------------------
    # 페니 초단타는 오늘 움직이는 종목을 잠깐 타는 방식이라 대상이 계속 바뀝니다
    if auto_tracking_enabled(cfg) and not halts:
        cfg = _maybe_refresh_universe(user_id, cfg, account, result)

    states = store.get_position_states(user_id, mode)

    # -- 1단계: 청산 검사 --------------------------------------------------
    for position in positions:
        # 관리 대상이 아닌 보유분(사용자가 직접 산 주식)은 건드리지 않습니다
        if not is_managed(cfg, position.key, states):
            continue
        try:
            # 안전장치가 걸린 회전에서는 분할 추가 매수를 하지 않습니다
            # (신규 진입과 같은 성격입니다). 차수 매도는 그대로 봅니다.
            _handle_exit(user_id, cfg, brk, engine_risk, position,
                         states.get(position.key, {}), result,
                         allow_add=not halts)
        except Exception as exc:
            result["errors"].append(f"{position.key} 청산 검사 실패: {exc}")
            store.log_event(user_id, "error", f"{position.key} 청산 검사 실패: {exc}",
                            level="error", symbol=position.key)

    # -- 2단계: 신규 진입 --------------------------------------------------
    if not halts:
        # 청산이 일어났으면 계좌가 바뀌었으므로 다시 읽습니다
        account = brk.account()
        positions = brk.positions()
        engine_risk = risk.RiskEngine(cfg, account, positions, day)
        held = {p.key for p in positions}

        # 보호장치 (engine/protections.py) — 최근 매매 이력 기반 잠금.
        # 회전 시작에 한 번만 계산해 스냅샷으로 씁니다 (RiskEngine 과 같은 이유).
        # 청산(1단계)은 이미 끝났으므로 이 잠금은 신규 진입에만 닿습니다.
        locks = protections.evaluate(
            user_id, mode, cfg, equity=float((account or {}).get("total_value") or 0))
        if locks.locks or locks.error:
            result["protections"] = locks.to_dict()
        global_locks = locks.global_reasons()
        if global_locks:
            result["rejects"].append({"symbol": "*", "action": "entry",
                                      "reasons": global_locks})
            store.log_event(user_id, "reject",
                            "보호장치 잠금 — " + " / ".join(global_locks),
                            level="warn", detail=locks.to_dict())

        # 두 전략을 **각자의 목록으로** 돌립니다. 예전에는 초단타가 켜져 있으면
        # 자동매매 대상까지 전부 초단타 판단을 태워서, 콘솔에 넣어둔 종목이
        # "초단타 대상이 아닙니다"로 거부되고 아무 매매도 되지 않았습니다.
        scalp_keys = set(scalp_universe(cfg))
        if not global_locks:
            for query in (cfg.get("universe") or []):
                if str(query).strip() in scalp_keys:
                    continue          # 초단타가 보는 종목은 초단타가 관리합니다
                try:
                    _handle_entry(user_id, cfg, brk, engine_risk, query, held,
                                  result, locks=locks)
                except Exception as exc:
                    result["errors"].append(f"{query} 진입 검사 실패: {exc}")
                    store.log_event(user_id, "error", f"{query} 진입 검사 실패: {exc}",
                                    level="error", symbol=str(query))

        if _scalp_enabled(cfg) and not global_locks:
            # 자동 갱신을 끄고 지정 종목만 쓰는 경우에도 실시간 체결은 붙어
            # 있어야 합니다 — 안 그러면 20초 지연 시세로 틱 매매를 하려다
            # 전부 '시세 지연'으로 거부됩니다. (같은 목록이면 무동작입니다)
            _sync_tick_stream(user_id, scalp_universe(cfg), result)
            for query in scalp_universe(cfg):
                try:
                    _handle_entry(user_id, cfg, brk, engine_risk, query, held,
                                  result, scalp=True, locks=locks)
                except Exception as exc:
                    result["errors"].append(f"{query} 초단타 진입 검사 실패: {exc}")
                    store.log_event(user_id, "error",
                                    f"{query} 초단타 진입 검사 실패: {exc}",
                                    level="error", symbol=str(query))

    result["elapsed_sec"] = round(time.time() - started, 2)
    result["account"] = brk.account()
    extra = ""
    if result["settled"]:
        extra += f" / 체결확정 {len(result['settled'])}건"
    if result["reconciled"]:
        extra += f" / 재동기화 {len(result['reconciled'])}건"
    if result.get("waiting"):
        extra += f" / 개장대기 {len(result['waiting'])}건"
    errors = f" / 오류 {len(result['errors'])}건" if result["errors"] else ""
    store.log_event(
        user_id, "tick",
        f"회전 완료 — 신호 {len(result['signals'])}건 / 진입 {len(result['entries'])}건 / "
        f"청산 {len(result['exits'])}건 / 거부 {len(result['rejects'])}건{extra}{errors}",
        level="warn" if result["errors"] else "info",
        detail={"elapsed_sec": result["elapsed_sec"], "mode": cfg["mode"],
                "halted": result["halted"], "errors": result["errors"][:5]})
    return result


# ---------------------------------------------------------------------------
# 0단계: 미결 주문 정산 (접수 → 체결 확인 → 포지션 확정)
# ---------------------------------------------------------------------------

def _settle_open_orders(user_id: int, cfg: dict, brk, result: dict):
    """접수만 된 주문을 브로커에 조회해 결말을 짓습니다.

    실계좌에서 '주문을 냈다'와 '샀다'는 다른 사건입니다. 접수 응답만 보고
    포지션을 잡았다고 처리하면, 미체결·부분체결·거부가 전부 내부 상태에
    거짓으로 반영됩니다. 그래서 매 회전 여기서 먼저 확인합니다.

    처리 규칙
        전량 체결   원장 확정 + 실체결가로 손절·목표 재계산
        부분 체결   체결된 만큼만 상태에 반영하고 계속 대기
        미체결      order_timeout_sec 초과 시 취소 요청
        조회 실패   **아무것도 하지 않습니다** (모르면 건드리지 않는 것이 안전)
    """
    timeout = float(cfg.get("order_timeout_sec", 180) or 0)

    mode = cfg.get("mode", "paper")
    for record in store.open_orders(user_id, mode):
        try:
            status = brk.order_status(record)
        except Exception as exc:
            result["errors"].append(f"주문 {record['id']} 조회 실패: {exc}")
            continue

        age = _age_seconds(record.get("created_at"))
        store.bump_poll(record["id"])

        if not status.known:
            # 조회가 안 되는 주문이 오래 남아 있으면 사람이 봐야 합니다
            if timeout and age > timeout * 3:
                store.log_event(
                    user_id, "error",
                    f"{record['name'] or record['symbol']} 주문 상태를 {age / 60:.0f}분째 "
                    f"확인하지 못했습니다 — 증권사 화면에서 직접 확인하세요.",
                    level="error", symbol=record["symbol"],
                    detail={"order": record["client_order_id"], "why": status.detail})
            continue

        if status.status in ("filled", "partial") and status.filled_quantity > 0:
            _apply_fill(user_id, cfg, record, status, result)

        if status.status == "filled":
            store.update_order(record["id"], status="filled",
                               settled_at=datetime.now().isoformat())
            continue

        if status.status == "cancelled":
            store.update_order(record["id"], status="cancelled",
                               settled_at=datetime.now().isoformat(),
                               reason="브로커에서 취소됨")
            store.log_event(user_id, "reject",
                            f"{record['name'] or record['symbol']} 주문이 취소되었습니다.",
                            level="warn", symbol=record["symbol"])
            continue

        # 아직 미체결 — 시간이 지났으면 거둬들입니다.
        # 안 걷으면 장 마감까지 남아 원하지 않는 가격에 체결될 수 있습니다.
        if timeout and age > timeout and record["status"] != "cancel_requested":
            cancel = brk.cancel(record)
            if cancel.ok:
                store.update_order(record["id"], status="cancel_requested",
                                   reason=f"{age:.0f}초 미체결 — 자동 취소 요청")
                store.log_event(
                    user_id, "reject",
                    f"{record['name'] or record['symbol']} {age:.0f}초 미체결 — 주문을 취소했습니다.",
                    level="warn", symbol=record["symbol"],
                    detail={"order": record["client_order_id"]})
            elif cancel.status == "filled":
                # 취소하려 했는데 이미 체결된 경우 — 다음 회전에 체결로 잡힙니다
                store.update_order(record["id"], reason="취소 시도했으나 이미 처리됨")
            else:
                result["errors"].append(
                    f"{record['symbol']} 주문 취소 실패: {cancel.error}")


def _exit_realized_krw(inst: Instrument, side: str, entry_price, exit_price,
                       quantity: float) -> float | None:
    """청산 왕복의 실현손익(**원화**). 계산할 수 없으면 None.

    왜 엔진이 직접 계산하는가
        증권사 체결내역은 체결가와 수량만 줍니다. 손익은 주지 않습니다. 그래서
        `KISBroker` 는 `realized_pnl` 을 채울 수 없는데, 비워 두면 그 값을 재료로
        쓰는 안전장치가 전부 판단을 못 합니다 — 일일 손실 한도, 최대 낙폭,
        초단타 `daily_loss_krw`, 보호장치 4종(`closed_trades`), 승률 집계.
        실제로 실계좌에서는 **손절할 때마다 오늘 손익이 0으로 리셋**됐습니다.
        평가손실은 포지션과 함께 사라지는데 실현손실이 0으로 들어왔기 때문입니다.

    규약은 `storage/paper.py` 의 매도와 같습니다 — **왕복 비용을 전부 뺀 순손익**.
    진입 수수료까지 빼는 이유는 그것이 이 매매가 계좌에 남긴 실제 금액이고,
    한도 판정에서 손실을 크게 보는(=안전한) 방향이기 때문입니다.
    (`storage/derivatives.py` 는 진입 수수료를 진입 시점에 이미 차감해서 여기서
     빼지 않습니다. 그만큼 실계좌 파생이 모의계좌보다 조금 보수적으로 잡힙니다.)

    None 을 돌려주는 경우
        진입가를 모르는 포지션입니다(재동기화로 복원된 것 등). **0 으로 채우면
        손실이 없었던 것처럼 보이므로** 비워 두고, 부르는 쪽이 경고를 남깁니다.
    """
    try:
        entry = float(entry_price or 0)
        exit_ = float(exit_price or 0)
        quantity = float(quantity or 0)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or exit_ <= 0 or quantity <= 0:
        return None

    direction = -1 if str(side or "").strip() == "short" else 1
    multiplier = float(inst.multiplier or 1.0)
    entry_side = "buy" if direction > 0 else "sell"
    exit_side = "sell" if direction > 0 else "buy"

    entry_fee, entry_tax = inst.costs(entry_side, entry * quantity * multiplier)
    exit_fee, exit_tax = inst.costs(exit_side, exit_ * quantity * multiplier)
    gross = (exit_ - entry) * quantity * multiplier * direction
    realized = gross - entry_fee - entry_tax - exit_fee - exit_tax

    # 미국 종목의 체결가는 달러입니다. 환산하지 않으면 일일 손실 한도(원)와
    # 초단타 daily_loss_krw(원)가 달러 숫자를 원으로 읽어, 손실을 1/1400 로
    # 축소해서 봅니다 — 한도가 사실상 없는 것과 같아집니다.
    from data_sources import fx
    return float(fx.to_krw(realized, inst.currency))


def _apply_fill(user_id: int, cfg: dict, record: dict, status, result: dict):
    """체결된 수량을 원장과 포지션 상태에 반영합니다."""
    mode = cfg.get("mode", "paper")
    already = float(record.get("filled_quantity") or 0)
    newly = status.filled_quantity - already
    if newly <= 1e-9:
        return          # 새로 체결된 것이 없으면 할 일이 없습니다

    fill_price = status.avg_fill_price or record.get("price")
    intended = record.get("intended_price") or record.get("price")
    slippage = None
    if intended and fill_price:
        raw = (float(fill_price) - float(intended)) / float(intended) * 10_000
        selling = record.get("action") in ("sell", "close", "open_short")
        slippage = round(-raw if selling else raw, 2)

    is_exit = record.get("action") in ("close", "sell")
    inst = instruments.try_resolve(record["symbol"])
    # 청산이 전량 체결되면 상태를 지우므로, 진입가는 **지우기 전에** 읽어 둡니다
    state = store.get_position_states(user_id, mode).get(record["symbol"], {})

    # 청산 손익 — 증권사가 주지 않아 엔진이 계산합니다 (_exit_realized_krw 참고).
    # 부분 체결이면 이미 적어 둔 값 위에 **늘어난 만큼만** 더합니다.
    realized = record.get("realized_pnl")
    booked = float(realized or 0)
    if is_exit:
        total = (_exit_realized_krw(inst, record.get("side") or state.get("side"),
                                    state.get("entry_price"), fill_price,
                                    status.filled_quantity)
                 if inst is not None else None)
        if total is None:
            # 조용히 0 으로 채우면 손실이 없었던 것처럼 보입니다. 빠졌다는 사실을
            # 남겨야 나중에 "왜 한도가 안 걸렸지"를 추적할 수 있습니다.
            store.log_event(
                user_id, "error",
                f"{record['name'] or record['symbol']} 청산 손익을 계산하지 못했습니다 "
                f"— 진입가를 모르는 포지션입니다. 오늘 손익과 손실 한도 집계에서 "
                f"이 매매가 빠집니다.",
                level="warn", symbol=record["symbol"], name=record["name"],
                detail={"order": record["client_order_id"],
                        "entry_price": state.get("entry_price"),
                        "exit_price": fill_price})
        else:
            realized = round(total, 2)

    store.update_order(record["id"],
                       status="filled" if status.status == "filled" else "partial",
                       filled_quantity=status.filled_quantity,
                       avg_fill_price=fill_price, slippage_bps=slippage,
                       realized_pnl=realized)

    # 이 주문이 **차수 주문**인가 — 접수만 됐던 분할 주문이 뒤늦게 체결되는
    # 경로입니다. 즉시 체결이면 _handle_split 이 이미 원장을 고쳤고, 그런
    # 주문은 미결 목록에 없으므로 여기 오지 않습니다(이중 반영 없음).
    detail = record.get("detail") or {}
    if not isinstance(detail, dict):
        detail = {}
    split_info = detail.get("split") if isinstance(detail.get("split"), dict) else None

    if is_exit and split_info:
        # 차수 하나만 판 것이므로 **포지션 상태를 지우면 안 됩니다.**
        # 지우면 남은 차수가 손절선도 원장도 없이 계좌에만 남습니다.
        if realized is not None:
            store.record_daily_trade(user_id, mode, float(realized) - booked,
                                     count=1 if status.status == "filled" else 0)
        if status.status == "filled":
            ledger = split.load(state.get("splits"))
            split.remove_tranche(ledger, int(split_info.get("tranche") or 0),
                                 status.filled_quantity)
            if ledger.tranches:
                _save_ledger(user_id, mode, record["symbol"], ledger)
            else:
                # 원장이 비었으면 실제로 전량이 나간 것입니다 (원장과 계좌가
                # 어긋났던 경우). 그때는 예전 경로와 같이 기억을 지웁니다.
                store.clear_position_state(user_id, mode, record["symbol"])
    elif is_exit:
        if realized is not None:
            # 손익은 체결된 만큼 즉시 반영하고(안 하면 한도가 늦게 걸립니다),
            # 매매 건수는 그 매매가 끝났을 때 한 번만 셉니다.
            store.record_daily_trade(user_id, mode, float(realized) - booked,
                                     count=1 if status.status == "filled" else 0)
        # 청산이 전량 체결되면 손절·목표 기억을 지웁니다
        if status.status == "filled":
            store.clear_position_state(user_id, mode, record["symbol"])
    elif split_info:
        # 추가 차수 매수 체결 — 원장에 차수를 더하고 평단·수량을 다시 씁니다.
        # 손절선·목표는 건드리지 않습니다: 분할의 손절선은 1차 때 마지막
        # 차수까지 계산해 박아둔 값이고, 차수를 담았다고 움직이면 안 됩니다.
        ledger = split.load(state.get("splits"))
        if ledger.tranches and status.status == "filled":
            split.add_tranche(ledger, int(split_info.get("tranche") or ledger.next_n()),
                              status.filled_quantity, float(fill_price or 0))
            _save_ledger(user_id, mode, record["symbol"], ledger)
    else:
        # 진입 체결 — **실제 체결가 기준으로** 손절·목표를 다시 잡습니다.
        # 주문 낼 때 본 가격으로 손절을 걸어두면, 슬리피지만큼 손절폭이 어긋납니다.
        plan = detail.get("plan")
        entry = float(fill_price or 0)
        stop = target = None
        if plan and entry:
            planned_entry = float(plan.get("price") or entry)
            if planned_entry:
                stop_gap = planned_entry - float(plan.get("stop_price") or planned_entry)
                target_gap = float(plan.get("target_price") or planned_entry) - planned_entry
                stop = entry - stop_gap
                target = entry + target_gap

        # 분할 1차가 뒤늦게 체결된 경우 — 원장을 여기서 만듭니다. 안 만들면
        # 계좌에는 1차가 들어와 있는데 원장이 없어서, 다음 회전이 그 포지션을
        # '분할 대상 아님'으로 보고 2차를 영원히 담지 않습니다.
        splits_json = None
        sizing_split = ((plan or {}).get("sizing") or {}).get("split") \
            if isinstance(plan, dict) else None
        if sizing_split and entry and not (state.get("splits") or "").strip():
            sp = split.params(cfg)
            ledger = split.bootstrap(sp, entry, status.filled_quantity,
                                     sum(float(q) for q in (sizing_split.get("qty") or [])
                                         ) or status.filled_quantity,
                                     inst.round_quantity if inst else None)
            splits_json = ledger.to_json()
            stop, target = ledger.plan["stop"], 0.0

        # splits 는 **만들었을 때만** 넘깁니다. None 을 그대로 넘기면 이미 있는
        # 원장을 NULL 로 덮어써서, 담아둔 차수 기록이 통째로 사라집니다.
        extra = {"splits": splits_json} if splits_json else {}
        store.upsert_position_state(
            user_id, mode, record["symbol"],
            side=record.get("side") or state.get("side") or "long",
            entry_price=entry or state.get("entry_price"),
            stop_price=stop if stop is not None else state.get("stop_price"),
            target_price=target if target is not None else state.get("target_price"),
            peak_price=entry or state.get("peak_price"),
            quantity=status.filled_quantity,
            opened_at=state.get("opened_at") or datetime.now().isoformat(),
            # 주문을 낸 전략이 이 포지션의 주인입니다 (청산 규칙이 여기서 갈립니다)
            strategy=detail.get("strategy") or state.get("strategy"),
            **extra)

    result["settled"].append({
        "symbol": record["symbol"], "name": record["name"],
        "action": record["action"], "filled": status.filled_quantity,
        "ordered": record["quantity"], "avg_price": fill_price,
        "slippage_bps": slippage, "status": status.status,
        "realized_pnl": realized,
    })
    slip_text = f" (슬리피지 {slippage:+.1f}bp)" if slippage is not None else ""
    pnl_text = (f" · 실현 {float(realized):+,.0f}원"
                if is_exit and realized is not None else "")
    store.log_event(
        user_id, "fill",
        f"{record['name'] or record['symbol']} {status.filled_quantity:g}/"
        f"{record['quantity']:g} 체결 @ {float(fill_price or 0):,.2f}{slip_text}{pnl_text}",
        level="trade", symbol=record["symbol"], name=record["name"],
        detail={"order": record["client_order_id"], "status": status.to_dict(),
                "realized_pnl": realized})


def _age_seconds(created_at: str) -> float:
    try:
        return (datetime.now() - datetime.fromisoformat(str(created_at))).total_seconds()
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# 0.5단계: 실계좌 ↔ 내부 상태 재동기화
# ---------------------------------------------------------------------------

# 사용자별 마지막 유니버스 갱신 시각
_last_universe_refresh: dict[int, float] = {}
# 지금 갱신 스레드가 돌고 있는 사용자 (중복 실행 방지)
_refreshing_users: set = set()
_refresh_guard = threading.Lock()

# (사용자, 종목) 별 마지막 초단타 청산 시각 — 재진입 쿨다운의 기준입니다.
# 프로세스 메모리에만 둡니다. 재시작하면 쿨다운이 풀리지만, 재시작 자체가
# 이미 몇 초 이상 걸리므로 실질적인 차이가 없습니다.
_last_scalp_exit: dict[tuple, float] = {}


def reset_universe_timer(user_id: int):
    """다음 sweep 에서 즉시 갱신되도록 타이머를 지웁니다 (추적 시작 직후용)."""
    _last_universe_refresh.pop(user_id, None)


def tracking_status(user_id: int, cfg: dict) -> dict:
    """AI 추적이 지금 뭘 하고 있는지 — 화면 표시용."""
    enabled = bool(cfg.get("auto_universe"))
    last = _last_universe_refresh.get(user_id)
    period = max(float(cfg.get("auto_universe_refresh_min", 30)), 5) * 60
    remaining = None
    if enabled:
        remaining = 0 if not last else max(0, int(period - (time.time() - last)))
    return {
        "enabled": enabled,
        "refreshing": user_id in _refreshing_users,
        "last_refresh_at": (datetime.fromtimestamp(last).isoformat()
                            if last else None),
        "next_in_sec": remaining,
        "period_min": period / 60,
    }


def _tracking_refresh(user_id: int):
    """백그라운드 추적 1회 — 시장을 훑어 추천을 다시 계산하고 매매 대상을 갱신.

    매매 회전과 별도 스레드로 돕니다. 추천 계산은 30~60초가 걸리는데,
    그동안 손절 검사(회전)가 멈추면 안 되기 때문입니다.
    """
    try:
        cfg = store.get_config(user_id)
        if not cfg.get("auto_universe"):
            return
        # 별도 스레드라 요청·회전의 오버레이가 없습니다 — 여기서 직접 장착합니다
        with _user_credentials_scope(user_id):
            brk = broker_module.get_broker(user_id, cfg.get("mode", "paper"), cfg)
            picked = refresh_recommended_universe(user_id, cfg, brk.account())
        # 반환만 하고 저장을 안 하면 추천 목록엔 ★편입이 찍히는데
        # 정작 매매 대상은 그대로인 어긋난 상태가 됩니다
        if picked != list(cfg.get("universe") or []):
            store.save_config(user_id, {"universe": picked})
    except Exception as exc:
        store.log_event(user_id, "error", f"AI 추적 갱신 실패: {exc}", level="error")
    finally:
        with _refresh_guard:
            _refreshing_users.discard(user_id)


def auto_tracking_enabled(cfg: dict) -> bool:
    """AI 자동 추적이 켜져 있는가 (일반 추천기 또는 초단타 스크리너)."""
    from engine import scalping
    if cfg.get("auto_universe"):
        return True
    return bool(scalping.clamp_config(cfg.get("scalp") or {}).get("enabled")
                and scalping.clamp_config(cfg.get("scalp") or {}).get("auto_universe"))


def _maybe_refresh_universe(user_id: int, cfg: dict, account: dict, result: dict) -> dict:
    """설정한 주기가 되면 매매 대상을 다시 뽑습니다.

    매 회전마다 시장 전체를 훑으면 데이터 제공처에 차단당하므로 주기를 둡니다.
    초단타가 켜져 있으면 스크리너 경로를, 아니면 다중 팩터 추천기를 씁니다.
    """
    from engine import scalping

    scalp = scalping.clamp_config(cfg.get("scalp") or {})
    scalp_mode = bool(scalp.get("enabled") and scalp.get("auto_universe"))
    if not scalp_mode:
        # 일반 AI 추적은 EngineLoop 의 백그라운드 스레드가 담당합니다.
        # 회전 안에서 30~60초짜리 추천 계산을 돌리면 그동안 손절 검사가 멈춥니다.
        return cfg

    # 초 단위입니다. 초단타 대상은 장중에 몇 분 만에도 바뀝니다.
    # (아래 한도는 REST 순위 API 초당 제한을 보호하기 위한 것이고,
    #  개별 종목의 틱 추적은 이 주기와 무관하게 WebSocket 이 담당합니다)
    period = max(float(scalp.get("universe_refresh_sec", 15)),
                 scalping.HARD_LIMITS["min_universe_refresh_sec"])
    last = _last_universe_refresh.get(user_id, 0)
    if time.time() - last < period:
        return cfg

    _last_universe_refresh[user_id] = time.time()
    try:
        picked = refresh_scalp_universe(user_id, cfg, account)
    except Exception as exc:
        result["errors"].append(f"매매 대상 갱신 실패: {exc}")
        store.log_event(user_id, "error", f"AI 자동 추적 실패: {exc}", level="error")
        return cfg

    # **초단타 목록에만** 씁니다. 예전에는 여기서 cfg["universe"] 를 덮어써서,
    # 초단타를 켜는 순간 자동매매 콘솔의 매매 대상이 동전주로 갈아치워졌습니다.
    if picked != list(scalp.get("universe") or []):
        store.save_config(user_id, {"scalp": {"universe": picked}})
        result["scalp_universe_refreshed"] = picked
        return {**cfg, "scalp": {**(cfg.get("scalp") or {}), "universe": picked}}
    return cfg


def rotate_window(items: list, key_of, budget: int,
                  cached: set, missed: set) -> tuple[list, list, int]:
    """순환 평가의 한 회전 — (새로 계산할 것, 캐시로만 올릴 것, 건너뛴 수).

    `items` 는 모집단(시장 전체)이고 `budget` 이 이번 회전에서 새로 계산할
    수입니다. 이미 팩터가 있는 종목(`cached`)은 계산 없이 순위에만 올리므로,
    회전을 거듭할수록 순위 대상이 시장 전체로 넓어집니다.

    `missed` 를 창에서 빼는 것이 핵심입니다. 계산이 실패하는 종목(상장 직후라
    일봉이 모자라거나 제공처가 그 종목만 안 주는 경우)은 **영원히 캐시가 생기지
    않아서**, 빼지 않으면 매 회전 창의 같은 자리를 차지합니다. 순환이 그
    지점에서 멈추고 시장 전체 커버는 영원히 끝나지 않습니다.
    """
    window, extras, skipped = [], [], 0
    for item in items:
        key = key_of(item)
        if key in cached:
            extras.append(item)
        elif key in missed:
            skipped += 1
        elif len(window) < budget:
            window.append(item)
    return window, extras, skipped


def recommend_universe(user_id: int, cfg: dict, account: dict,
                       top_n: int = None) -> list:
    """다중 팩터 추천기로 후보를 뽑고 순위를 매깁니다 (편입은 하지 않음).

    1) 시장 **전체**를 모집단으로 잡고 (국내는 KRX 상장법인목록,
       미국은 나스닥 스크리너 표 — 둘 다 한 번의 조회로 명단이 옵니다)
    2) 그중 이번 갱신 몫만 팩터를 새로 계산하고 (순환 평가),
       하루 안에 계산해 둔 나머지는 캐시 그대로 순위에 올린 뒤
    3) 같은 시장 후보들끼리 표준화하고
    4) 지금 국면에 맞는 비중으로 합산합니다
    """
    from data_sources import screener, universe as universe_mod
    from engine import recommender

    markets = universe_mod.normalize_segments(cfg.get("auto_universe_markets"))
    pool = str(cfg.get("auto_universe_pool") or "")
    size = int(top_n or cfg.get("auto_universe_size", 5))

    # 팩터 계산은 종목마다 일봉을 받아야 해서 비쌉니다. 그래서 **순환 평가**:
    # 갱신마다 scan_limit 개만 새로 계산하고, 하루(FACTOR_STALE_TTL) 안에 계산해
    # 둔 나머지는 캐시 그대로 순위에 함께 올립니다. 몇 회전이면 시장 전체가
    # 순위 대상이 됩니다 (3,300종목 ÷ 300개 ≈ 11회전 = 30분 주기로 반나절).
    #   · 미국 — 표(전 종목)를 회전율 순으로 받아, 팩터가 아직 없는 종목부터
    #     계산 창(window)에 담습니다. 회전율 상위 = 오늘 움직이는 종목 우선.
    #   · 한국 — KIS 순위 API 는 상위 30행까지만 줍니다. 그래서 **명단은 순위가
    #     아니라 KRX 상장법인목록**(코스피 845 · 코스닥 1,787, 로컬 캐시)에서
    #     받고, 순위 30종목은 오늘 시세가 붙은 채로 매번 새로 평가합니다.
    # scan_limit 은 **시장별** 창 크기입니다 — 국내는 네이버·KIS, 미국은
    # 나스닥·야후라 서로 다른 제공처를 쓰므로 한도를 나눠 가질 이유가 없습니다.
    # 갱신 1회는 30개≈30초 → 300개≈2~4분 (백그라운드 스레드라 손절 검사는
    # 안 멈춥니다 — _tracking_refresh 참고). 국내는 종목당 일봉 + 수급 2회라
    # 같은 창 크기라도 미국보다 조금 더 걸립니다.
    scan_limit = int(cfg.get("auto_universe_scan_limit")
                     or store.DEFAULT_CONFIG["auto_universe_scan_limit"])
    scan_limit = min(max(scan_limit, 10), 300)   # 한 번에 300 초과는 일봉 제공처가 차단
    scan = screener.scan(
        markets=markets,
        kr_price=(1_000, 2_000_000),          # 일반 추천은 가격대를 넓게
        us_price=(3.0, 2_000.0),
        limit=30,                             # 한국 몫 — KIS 순위 API 의 최대치
        us_limit=100_000,                     # 미국은 전 종목 (창은 아래서 자릅니다)
        universe=pool,
        # 미국만 회전율 순 — 거래대금 순은 매번 같은 대형주가 상위를 채워서,
        # 계산 창이 늘 그 안에서만 돌았습니다. 회전율(시총 대비 손바뀜)은
        # 오늘 실제로 움직이는 종목을 앞세웁니다. 한국은 KIS 평균거래량 순 유지.
        us_rank_basis="turnover")

    us = [c for c in scan.candidates if c.market == "US"]
    kr = [c for c in scan.candidates if c.market != "US"]

    # 국내 전 종목 순환 — 탐색 범위를 지정하면 하지 않습니다. "코스피200 안에서
    # 찾겠다"고 해 놓고 뒤에서 시장 전체를 순위에 올리면 범위가 무의미해집니다.
    kr_segments = [s for s in markets
                   if universe_mod.SEGMENTS.get(s, {}).get("region") == "KR"]
    kr_pool: list[str] = []
    if kr_segments and not pool and cfg.get("auto_universe_full_market", True):
        seen = {c.key for c in kr}
        kr_pool = [row["key"] for row in universe_mod.kr_listed(kr_segments)
                   if row["key"] not in seen]

    if not scan.candidates and not kr_pool:
        store.log_event(
            user_id, "screen",
            "추천할 후보를 받지 못했습니다 — " + (" / ".join(scan.errors) or "원인 불명"),
            level="warn", detail={"errors": scan.errors})
        return []
    if not scan.candidates:
        # 명단만으로도 순위는 매길 수 있지만(캐시 + 순환), 오늘 시세가 붙은
        # 상위 종목이 통째로 빠진 상태입니다. 조용히 넘어가면 안 됩니다.
        store.log_event(
            user_id, "screen",
            "거래량순위를 받지 못해 상장 명단만으로 순위를 냅니다 — "
            + (" / ".join(scan.errors) or "원인 불명"),
            level="warn", detail={"errors": scan.errors})

    cached = recommender.cached_factor_keys()
    missed = recommender.uncomputable_keys()

    def _rotate(items: list, key_of, budget: int) -> tuple[list, list, int]:
        return rotate_window(items, key_of, budget, cached, missed)

    # 범위를 지정했으면 "시장 전체를 다 봤다"고 말하면 안 됩니다 — 실제로 다 본
    # 것은 그 범위뿐이고, 사용자는 시장 전체를 훑었다고 오해합니다.
    scope = "범위" if pool else "시장"

    def _log_rotation(label: str, total: int, fresh: int, have: int, skipped: int):
        if total <= 0:
            return
        remain = max(total - have - fresh - skipped, 0)
        store.log_event(
            user_id, "screen",
            f"순환 평가 — {label} {total:,}종목 중 팩터 보유 {have:,}종목, "
            f"이번 갱신에서 {fresh}종목 새로 계산"
            + (f" (전체까지 약 {(remain + scan_limit - 1) // scan_limit}회전 남음)"
               if remain else f" — {scope} 전체 커버 완료")
            + (f" · 계산 불가 {skipped:,}종목은 건너뜀" if skipped else ""),
            detail={"market": label, "total": total, "cached": have,
                    "fresh": fresh, "skipped": skipped, "remaining": remain})

    # 미국 — 스크리너 표(회전율 순) 그대로 순환
    us_window, us_extras, us_skipped = _rotate(us, lambda c: c.key, scan_limit)
    _log_rotation("미국", len(us), len(us_window), len(us_extras), us_skipped)

    # 한국 — 순위 30종목(오늘 시세 포함)은 늘 새로, 나머지 전 종목은 순환.
    # 명단은 종목코드 문자열입니다. rank() 가 문자열 후보를 받아 상장 마스터로
    # 해석하고, 가격은 팩터 계산이 받아온 일봉의 마지막 종가를 씁니다 —
    # 명단에 시세를 붙이려면 종목마다 조회가 한 번씩 더 나갑니다.
    kr_window, kr_extras, kr_skipped = _rotate(kr_pool, lambda k: k, scan_limit)
    _log_rotation("한국", len(kr) + len(kr_pool),
                  len(kr) + len(kr_window), len(kr_extras), kr_skipped)

    return recommender.rank(kr + kr_window + kr_extras + us_window + us_extras,
                            cfg, account, top_n=size * 3,
                            stale_keys={c.key for c in us_extras} | set(kr_extras))


def refresh_recommended_universe(user_id: int, cfg: dict, account: dict) -> list[str]:
    """AI 자동 추적 — 추천 상위 종목으로 매매 대상을 갱신합니다."""
    ranked = recommend_universe(user_id, cfg, account)
    if not ranked:
        return list(cfg.get("universe") or [])

    threshold = float(cfg.get("auto_universe_min_score", 0.55))
    size = int(cfg.get("auto_universe_size", 5))
    picked = [r.key for r in ranked if r.tradable and r.score >= threshold][:size]

    # 손으로 넣은 종목은 지키고 싶을 수 있습니다 (기본값: 유지)
    if cfg.get("auto_universe_keep_manual", True):
        manual = [s for s in (cfg.get("manual_universe") or [])
                  if s in (cfg.get("universe") or [])]
        if not manual:
            # manual_universe 가 없던 시절에 만든 계정 — 예전 방식으로 추정합니다.
            # (추정이라 AI 가 한 번 뽑은 종목은 못 지킵니다. 화면에서 한 번
            #  추가·삭제하면 그때부터 정확히 기록됩니다)
            previous_auto = {r["symbol"] for r in store.get_recommendations(user_id)
                             if r.get("picked")}
            manual = [s for s in (cfg.get("universe") or []) if s not in previous_auto]
        picked = list(dict.fromkeys(manual + picked))

    store.save_recommendations(user_id, ranked, picked)

    if not picked:
        store.log_event(
            user_id, "screen",
            f"추천 점수가 기준({threshold:.2f})을 넘는 종목이 없어 매매 대상을 비웠습니다. "
            f"1위 {ranked[0].name} {ranked[0].score:.2f}점.",
            level="warn")
        return []

    previous = set(cfg.get("universe") or [])
    if set(picked) != previous:
        top = ranked[0]
        store.log_event(
            user_id, "screen",
            f"AI 추천으로 매매 대상을 갱신했습니다 — {', '.join(picked)} "
            f"(1위 {top.name} {top.score:.2f}점 · {top.regime})",
            detail={"picked": picked,
                    "dropped": sorted(previous - set(picked)),
                    "top": [r.to_dict() for r in ranked[:5]]})
    return picked


def refresh_scalp_universe(user_id: int, cfg: dict, account: dict) -> list[str]:
    """AI 자동 추적 — 시장을 훑어 초단타 대상을 갈아끼웁니다.

    페니 초단타는 "좋은 종목을 오래 들고 가는" 것이 아니라 **오늘 움직이는
    것을 잠깐 타는** 방식이라, 대상이 매일(때로는 시간마다) 바뀝니다.
    그래서 유니버스를 사람이 고정하지 않고 주기적으로 다시 뽑습니다.
    """
    from data_sources import screener
    from engine import scalping

    scalp = scalping.clamp_config(cfg.get("scalp") or {})
    if not scalp.get("enabled"):
        return list(scalp.get("universe") or [])

    # 내가 지정한 종목은 갱신과 무관하게 항상 남습니다
    pinned = [c for c in (scalp.get("pinned") or [])]
    if not scalp.get("auto_universe"):
        return pinned

    # 자동 갱신 루프에서는 고저가 보강(enrich)을 끕니다 — 종목당 REST 1회라
    # 15초 주기에 맞지 않습니다. 추적 중인 종목의 변동폭은 실시간체결
    # 스트림이 고가/저가를 함께 주므로 거기서 채워집니다.
    scan = screener.scan(
        markets=scalp.get("markets"),
        kr_price=tuple(scalp["kr_price_range"]),
        us_price=tuple(scalp["us_price_range"]),
        limit=40, rank_basis=scalp.get("rank_basis", "vol_increase"),
        enrich=False, universe=str(scalp.get("universe_pool") or ""))
    ranked = scalping.recommend(scan.candidates, account.get("available_cash") or 0,
                                account.get("total_value") or 0, scalp)

    # 개수를 사용자가 정하지 않습니다. 조건을 통과한 것을 전부 담고,
    # 실시간 시세 구독 한도로만 자릅니다. 한도는 계정마다 다르고 HTS·MTS 와
    # 나눠 쓰므로, 문서값이 아니라 **서버가 실제로 받아준 수**를 씁니다.
    # 여기서 크게 잡으면 남는 종목은 틱 없이 REST(20초 지연)로 돌아갑니다.
    from data_sources import kis_realtime
    try:
        cap = min(scalping.max_tracked(), kis_realtime.subscription_room())
    except Exception:
        cap = scalping.max_tracked()
    room = max(cap - len(pinned), 0)
    found = [c["key"] for c in ranked
             if c["eligible"] and c["affordable"] and c["key"] not in pinned][:room]
    picked = pinned + found

    if not picked:
        why = " / ".join(scan.errors) if scan.errors else "타점·유동성 필터 통과 실패"
        store.log_event(user_id, "screen",
                        f"초단타 조건을 통과한 종목이 없어 매매 대상을 비웠습니다 — {why}",
                        level="warn", detail={"errors": scan.errors,
                                              "scanned": len(scan.candidates)})
        return []

    _sync_tick_stream(user_id, picked, result=None)

    previous = set(scalp.get("universe") or [])
    if set(picked) != previous:
        store.log_event(
            user_id, "screen",
            f"초단타 대상을 갱신했습니다 — {', '.join(picked)}"
            + (f" (지정 {len(pinned)}종목 고정)" if pinned else ""),
            detail={"picked": picked, "pinned": pinned,
                    "scanned": len(scan.candidates),
                    "dropped": sorted(previous - set(picked))})
    return picked


# 미국 티커의 거래소 조회 결과 캐시 (구독 키를 만들려면 NAS/NYS/AMS 가 필요)
_us_exchange_cache: dict[str, str] = {}


def _us_exchange_hint(code: str) -> str:
    """미국 티커의 거래소. 못 찾으면 빈 문자열(→ 나스닥으로 가정).

    거래소가 틀리면 구독은 받아들여지는데 데이터가 한 건도 안 옵니다.
    그래서 추측하지 않고 한 번은 조회해서 캐시해둡니다.
    """
    code = str(code).strip().upper()
    if code in _us_exchange_cache:
        return _us_exchange_cache[code]
    exchange = ""
    try:
        from data_sources import symbol_registry
        rows = symbol_registry._yahoo_lookup(code)
        hit = next((r for r in rows if r["symbol"].upper() == code), None)
        exchange = (hit or {}).get("exchange", "")
    except Exception:
        exchange = ""
    _us_exchange_cache[code] = exchange
    return exchange


def _sync_tick_stream(user_id: int, symbols: list[str], result: dict = None):
    """추적 종목을 실시간체결 스트림에 반영합니다. 국내·미국 모두.

    실패해도 매매는 계속되어야 합니다 — 스트림이 없으면 REST 폴링으로
    돌아가고, 그 사실은 상태로 보고합니다. 시세 경로가 하나 막혔다고
    보유 포지션의 손절 검사까지 멈추면 그게 더 큰 사고입니다.
    """
    from data_sources import kis_realtime

    codes = [str(s).strip().upper() for s in (symbols or []) if str(s).strip()]
    hints = {c: _us_exchange_hint(c) for c in codes
             if not kis_realtime.is_korean_code(c)}
    try:
        kis_realtime.start(codes, exchanges=hints)
    except Exception as exc:
        message = f"실시간 시세 연결 실패 — REST 폴링으로 진행합니다 ({exc})"
        if result is not None:
            result.setdefault("errors", []).append(message)
        store.log_event(user_id, "error", message, level="warn")


def universe_keys(cfg: dict) -> set:
    """유니버스가 가리키는 실제 종목코드 집합 (별칭도 해석해서 넣습니다)."""
    keys = set()
    for query in (cfg.get("universe") or []):
        inst = _resolve_universe_item(query)
        if inst:
            keys.add(inst.key)
        keys.add(str(query).strip())
    return keys


def is_managed(cfg: dict, key: str, states: dict) -> bool:
    """이 종목을 자동매매가 건드려도 되는가.

    기본값(manage_only_universe=True)에서는 **유니버스에 넣은 종목과
    엔진이 직접 산 종목만** 관리합니다. 계좌에 원래 있던 다른 주식은
    자동매매가 팔지 않습니다 — 실계좌에서 이건 사고입니다.
    """
    if not cfg.get("manage_only_universe", True):
        return True
    if key in states:
        return True          # 엔진이 직접 잡은 포지션
    # 두 목록 모두 '자동매매가 건드려도 되는 종목'입니다 (관리 주체만 다릅니다)
    return key in universe_keys(cfg) or key in set(scalp_universe(cfg))


def _reconcile(user_id: int, cfg: dict, positions: list, result: dict,
               partial: bool = False):
    """브로커의 실제 포지션과 엔진이 기억하는 상태를 맞춥니다.

    어긋나는 경우는 실제로 자주 생깁니다.
        · 서버를 껐다 켜는 사이 체결됨
        · 사용자가 증권사 앱에서 직접 사고팔았음
        · 부분 체결로 수량이 달라졌음

    맞추지 않으면 손절선 없는 포지션이 방치되거나, 이미 없는 포지션을
    청산하려 들면서 매 회전 오류가 납니다.
    """
    mode = cfg.get("mode", "paper")
    states = store.get_position_states(user_id, mode)
    live_keys = {p.key for p in positions}

    # (1) 포지션은 있는데 기억이 없다 → 손절·목표를 복원합니다
    for position in positions:
        if position.key in states:
            state = states[position.key]
            recorded = float(state.get("quantity") or 0)
            if recorded and abs(recorded - position.quantity) > 1e-6:
                store.upsert_position_state(user_id, mode, position.key,
                                            quantity=position.quantity)
                result["reconciled"].append({
                    "symbol": position.key, "action": "quantity_fixed",
                    "from": recorded, "to": position.quantity})
                store.log_event(
                    user_id, "reconcile",
                    f"{position.name} 수량을 실계좌 기준으로 맞췄습니다 "
                    f"({recorded:g} → {position.quantity:g})",
                    level="warn", symbol=position.key, name=position.name)
            continue

        # 유니버스 밖 종목 = 사용자가 직접 산 주식. 기본 설정에서는 손대지 않습니다.
        if not is_managed(cfg, position.key, states):
            if position.key not in _skipped_notice.get(user_id, set()):
                _skipped_notice.setdefault(user_id, set()).add(position.key)
                store.log_event(
                    user_id, "reconcile",
                    f"{position.name} 은(는) 매매 대상이 아니어서 자동매매가 관여하지 않습니다 "
                    f"(계좌에만 보유). 관리하려면 매매 대상에 추가하세요.",
                    symbol=position.key, name=position.name)
            continue

        inst = instruments.try_resolve(position.key)
        entry = float(position.avg_price or position.current_price or 0)
        if not entry:
            continue
        # 손절 폭은 ATR 기준으로 다시 잡습니다 (진입 근거를 모르므로 현재 변동성 사용)
        distance = None
        if inst:
            sig = strategy.evaluate(inst, cfg)
            if sig.ok:
                distance = strategy.stop_distance(inst, sig, cfg)
        if not distance:
            # 지표를 못 구해도 손절 없는 포지션을 방치하면 안 됩니다.
            # 고정 퍼센트로라도 반드시 선을 걸어 둡니다.
            distance = entry * float(cfg.get("stop_loss_pct", 4.0)) / 100.0
        direction = 1 if position.side == "long" else -1
        stop = entry - direction * distance
        target = entry + direction * distance * float(cfg.get("reward_risk", 2.0))

        # 진입 근거가 없으니 주인도 모릅니다. 지금 초단타 대상이면 초단타로
        # 보되(그 종목은 초단타가 산 것일 가능성이 높습니다), 아니면 일반으로
        # 둡니다 — 모르는 포지션을 초단타에 넘겨 3틱에 던지게 하면 안 됩니다.
        store.upsert_position_state(
            user_id, mode, position.key, side=position.side, entry_price=entry,
            stop_price=stop, target_price=target,
            peak_price=position.current_price or entry,
            quantity=position.quantity,
            opened_at=position.opened_at or datetime.now().isoformat(),
            note="재동기화로 복원 (진입 근거 없음)",
            strategy=position_owner(cfg, position.key, {}))
        result["reconciled"].append({"symbol": position.key, "action": "restored",
                                     "entry": entry, "stop": stop})
        store.log_event(
            user_id, "reconcile",
            f"{position.name} 포지션을 계좌에서 발견해 관리 대상으로 복원했습니다 "
            f"(평단 {entry:,.2f}, 손절 {stop:,.2f})",
            level="warn", symbol=position.key, name=position.name,
            detail={"restored": True})

    # (2) 기억은 있는데 포지션이 없다 → 밖에서 정리된 것이므로 기억을 지웁니다
    if partial:
        # 조회 실패로 안 보이는 것을 '청산됨'으로 처리하면 손절선을 잃습니다.
        # 다음 회전에 정상 조회되면 그때 대조합니다 (지연은 안전하지만 오판은 아닙니다).
        return
    for symbol, state in states.items():
        if symbol in live_keys:
            continue
        if store.has_open_order(user_id, mode, symbol):
            continue          # 아직 체결 대기 중인 주문이면 남겨둡니다
        store.clear_position_state(user_id, mode, symbol)
        result["reconciled"].append({"symbol": symbol, "action": "cleared"})
        store.log_event(
            user_id, "reconcile",
            f"{symbol} 포지션이 계좌에 없어 관리 목록에서 제외했습니다 "
            f"(증권사에서 직접 청산했거나 이미 정리된 종목).",
            level="warn", symbol=symbol, detail={"state": state})


# ---------------------------------------------------------------------------
# 청산
# ---------------------------------------------------------------------------

def _handle_exit(user_id: int, cfg: dict, brk, engine_risk, position,
                 state: dict, result: dict, fast: bool = False,
                 allow_add: bool = False):
    """청산 검사 한 종목.

    fast=True 는 고속 보호 회전(run_guard)이 쓰는 경로입니다. 지표·ML 을
    계산하지 않고 **신선한 현재가만** 받아 손절·트레일링·시간 청산을 봅니다.
    익절과 신호 반전은 보지 않습니다 (strategy.check_exit 의 protective_only).

    `allow_add` 는 분할 매매의 **추가 매수**를 허용할지입니다. 안전장치가
    걸린 회전에서는 False 로 들어옵니다 — 그때도 차수 매도는 그대로 봅니다.
    """
    mode = cfg.get("mode", "paper")
    inst = instruments.try_resolve(position.key)
    if inst is None:
        result["errors"].append(f"{position.key} 종목 정보를 해석할 수 없습니다.")
        return

    # 이 포지션을 누가 잡았는가 — 청산 규칙 전체가 여기서 갈립니다
    owner = position_owner(cfg, position.key, state)

    # 이미 낸 주문이 처리 중이면 또 내지 않습니다 (이중 청산 방지)
    if store.has_open_order(user_id, mode, position.key):
        if owner == OWNER_SCALP:
            mark_scalp_phase(user_id, position.key, HOLD, name=position.name,
                             note="낸 주문의 체결을 기다리는 중입니다")
        return

    # 보호 회전은 **캐시된 가격을 믿지 않습니다** — 손절선을 재는 데 20초 전
    # 가격을 쓰면 5초마다 확인하는 의미가 없습니다.
    quote = feed.quote(inst, max_age=GUARD_QUOTE_MAX_AGE) if fast else feed.quote(inst)
    # 초단타 포지션의 청산 판단은 **틱**으로 해야 합니다. 20초 전 가격으로
    # 3틱 목표를 재면 이미 지나간 자리에서 뒤늦게 나가게 됩니다.
    if quote and owner == OWNER_SCALP:
        quote = _scalp_quote(inst, quote)

    if fast:
        # 지표·ML 을 계산하지 않습니다. 점수 0 짜리 신호를 만들어 가격 배리어만
        # 재게 하고, 판정은 protective_only 로 손절 계열만 봅니다.
        if not quote or not quote.get("price"):
            return
        sig = strategy.Signal(key=inst.key, ok=True, score=0.0,
                              price=float(quote["price"]),
                              price_krw=float(quote.get("price_krw")
                                              or quote["price"]),
                              quote_age=float(quote.get("age_sec") or 0))
    else:
        sig = strategy.evaluate(inst, cfg, quote=quote)
        if sig.ok:
            result["signals"].append({**sig.to_dict(), "held": True})

    # 트레일링 스톱용 고점 갱신 (신호가 실패해도 가격만 있으면 갱신합니다)
    price = sig.price if sig.ok else (quote or {}).get("price")
    if price:
        state = strategy.update_peak(dict(state), price, position.side)
        store.upsert_position_state(user_id, mode, position.key,
                                    peak_price=state.get("peak_price"))

    # 초단타로 잡은 포지션은 **초 단위** 시간 청산이 걸려 있어야 합니다.
    # 목표·손절 가격 자체는 진입 때 틱으로 계산해 포지션 상태에 박아뒀고,
    # 아래 %는 그 값이 유실됐을 때를 위한 backstop 입니다.
    # 판정 기준은 **가격대가 아니라 주인**입니다. 값이 대상 가격대를 벗어났다고
    # 초단타 포지션을 일반 규칙으로 넘기면, 3틱짜리로 산 종목이 며칠짜리
    # 손절선을 달고 남습니다.
    exit_cfg = cfg
    if owner == OWNER_SCALP and sig.ok:
        from engine import scalping
        scalp = scalping.clamp_config(cfg.get("scalp") or {})
        market = "US" if inst.market == "US" else "KR"
        econ = scalping.tick_economics(sig.price, scalp, market)
        unit = econ["tick_size"]
        exit_cfg = {
            **cfg,
            "max_hold_sec": scalp["max_hold_sec"],
            "max_hold_minutes": 0,
            "max_hold_days": 0,
            "atr_stop_mult": 0,
            "stop_loss_pct": unit * scalp["stop_loss_ticks"] / sig.price * 100,
            "take_profit_pct": unit * scalp["take_profit_ticks"] / sig.price * 100,
        }
        # 들고 있는 동안은 '파는 조건을 지켜보는 중'입니다
        mark_scalp_phase(
            user_id, position.key, HOLD, name=position.name,
            note=f"보유 {position.quantity:g}{_unit(inst)} · "
                 f"목표 {state.get('target_price') or econ['target_price']:,.0f} / "
                 f"손절 {state.get('stop_price') or econ['stop_price']:,.0f} / "
                 f"최대 {scalp['max_hold_sec']:g}초",
            price=sig.price, quantity=position.quantity,
            entry_price=state.get("entry_price") or position.avg_price,
            target_price=state.get("target_price") or econ["target_price"],
            stop_price=state.get("stop_price") or econ["stop_price"])

    # 분할로 잡은 포지션은 **전량 익절을 하지 않습니다** — 차수별 매도가
    # 그 자리를 대신합니다. 지키는 조건(손절·트레일링·만기·시간·신호 반전)은
    # 그대로 전량으로 나갑니다.
    ledger = _split_ledger(state)
    split_on = bool(ledger.tranches) and split_applies(cfg, inst, owner)
    if split_on:
        # 분할 포지션의 손절은 **절대가(stop_price) 하나로만** 봅니다.
        #
        # 평단 대비 고정 %(stop_loss_pct)를 함께 두면 그쪽이 먼저 걸립니다 —
        # 기본값 4% 는 차수 간격 5% 보다 좁아서, 2차를 담을 자리에 닿기 전에
        # 전량 손절이 나갑니다. 실제로 그렇게 동작했고(실측 -6.00% 에서 청산),
        # 그러면 분할 매수는 영원히 1차에서 끝납니다.
        #
        # 절대가는 1차 진입 때 "마지막 차수까지 담고도 더 빠지면"으로 계산해
        # 박아둔 값입니다. 그 값이 유실됐으면 원장의 계획에서 되살리고,
        # 그것마저 없으면 **고정 %를 그대로 둡니다** — 손절을 없애는 것보다
        # 일찍 걸리는 편이 낫습니다.
        stop = state.get("stop_price") or (ledger.plan or {}).get("stop")
        if stop:
            state = {**state, "stop_price": stop}
            exit_cfg = {**exit_cfg, "stop_loss_pct": 0.0}

    decision = strategy.check_exit(inst, position, sig, exit_cfg, state,
                                   protective_only=fast,
                                   defer_take_profit=split_on)
    if not decision.should_exit:
        if split_on and not fast:
            _handle_split(user_id, cfg, brk, engine_risk, inst, position, state,
                          sig, result, allow_add=allow_add, quote=quote)
        return

    gate = engine_risk.check_exit(inst, quote, urgent=(decision.urgency == "urgent"))
    if not gate.approved:
        result["rejects"].append({"symbol": position.key, "action": "exit",
                                  "reasons": gate.rejects})
        store.log_event(user_id, "reject",
                        f"{inst.name} 청산 보류 — {' / '.join(gate.rejects)}",
                        level="warn", symbol=inst.key, name=inst.name,
                        detail={"decision": decision.reason})
        return

    if cfg.get("dry_run"):
        store.log_event(user_id, "exit", f"[모의판단] {inst.name} 청산 대상 — {decision.reason}",
                        level="info", symbol=inst.key, name=inst.name,
                        detail={"dry_run": True, "reason": decision.reason})
        result["exits"].append({"symbol": inst.key, "name": inst.name,
                                "dry_run": True, "reason": decision.reason})
        return

    _submit_close(user_id, cfg, brk, inst, position, state,
                  reason=decision.reason, result=result,
                  detail={"signal": sig.to_dict()})


def _submit_close(user_id: int, cfg: dict, brk, inst: Instrument, position,
                  state: dict, reason: str, result: dict, detail: dict = None):
    """청산 주문 제출 + 체결 기록. 낸 주문(또는 중복이면 None)을 돌려줍니다.

    일반 청산(_handle_exit)과 회전 매도(_maybe_rotate)가 같이 씁니다 —
    실현손익 계산·상태 정리·로그 규약이 두 경로에서 갈라지면 안 됩니다.
    """
    mode = cfg.get("mode", "paper")
    coid = make_client_order_id(user_id, inst.key, "close", mode=mode)
    if store.order_exists(coid):
        return None

    order = brk.close(inst, note=f"자동청산: {reason}", client_order_id=coid)
    _record(user_id, cfg, inst, order, reason=reason,
            side=position.side, detail=detail)

    if order.ok:
        # 청산 시각을 남깁니다 — 같은 종목 재진입 쿨다운의 기준입니다.
        # 접수 시점부터 재는 이유: 체결을 기다리는 동안 또 사면 안 됩니다.
        _last_scalp_exit[(user_id, inst.key)] = time.time()

        # 즉시 체결 브로커(모의계좌)는 손익을 함께 돌려줍니다. 비어 있으면
        # **0 으로 적지 않고** 직접 계산합니다 — 0 은 '손실이 없었다'로 읽혀서
        # 일일 손실 한도가 그만큼 늦게 걸립니다 (_exit_realized_krw 참고).
        realized = order.realized_pnl
        if realized is None and order.status == "filled":
            realized = _exit_realized_krw(
                inst, position.side,
                state.get("entry_price") or position.avg_price,
                order.avg_fill_price or order.price,
                order.filled_quantity or order.quantity)

        # 접수만 된 주문은 아직 포지션이 남아 있습니다.
        # 여기서 상태를 지우면 손절선을 잃은 채 포지션만 남습니다.
        if order.status == "filled":
            store.clear_position_state(user_id, mode, inst.key)
            # 끝까지 못 구하면 아예 적지 않습니다. 0 으로 채우면 승률과
            # 오늘 손익이 조용히 오염됩니다 (정산 패스와 같은 정책입니다).
            if realized is not None:
                store.record_daily_trade(user_id, mode, float(realized))
        result["exits"].append({
            "symbol": inst.key, "name": inst.name, "quantity": order.quantity,
            "price": order.price, "realized_pnl": realized,
            "reason": reason, "status": order.status,
        })
        pnl = f" (실현 {realized:+,.0f}원)" if realized is not None else ""
        store.log_event(user_id, "exit",
                        f"{inst.name} {order.quantity:g}{_unit(inst)} 청산 — "
                        f"{reason}{pnl}",
                        level="trade", symbol=inst.key, name=inst.name,
                        detail=order.to_dict())
    else:
        result["errors"].append(f"{inst.name} 청산 실패: {order.error}")
        store.log_event(user_id, "error", f"{inst.name} 청산 실패: {order.error}",
                        level="error", symbol=inst.key, name=inst.name,
                        detail=order.to_dict())
    return order


# ---------------------------------------------------------------------------
# 분할 매수·매도 — engine/split.py 가 판단하고, 여기서 실행합니다
# ---------------------------------------------------------------------------

def split_applies(cfg: dict, inst: Instrument, owner: str = OWNER_AUTO) -> bool:
    """이 종목에 분할 매매 규칙을 적용하는가.

    초단타는 제외합니다 — 3틱 안에 끝내는 매매에 "5% 내리면 2차"는 의미가
    없고, 예산 체계도 따로 있습니다.
    파생상품도 제외합니다 — 승수·증거금 때문에 차수당 금액이 주식과 전혀 다른
    의미가 되고, 만기가 있어 '내릴 때까지 담는다'는 전제 자체가 성립하지 않습니다.
    """
    if owner == OWNER_SCALP or inst.is_derivative:
        return False
    return bool(split.params(cfg)["split_enabled"])


def _split_ledger(state: dict) -> split.Ledger:
    return split.load((state or {}).get("splits"))


def _save_ledger(user_id: int, mode: str, key: str, ledger: split.Ledger):
    """원장 + 그 원장에서 유도되는 값(평단·수량)을 함께 저장합니다.

    평단을 같이 쓰는 이유: 손절 판정(check_exit)은 state['entry_price'] 를
    봅니다. 차수를 담거나 판 뒤에 평단을 갱신하지 않으면, 1차 가격 기준으로
    손익률을 재게 되어 손절이 엉뚱한 자리에서 걸립니다.

    원장이 비면 **평단·수량은 건드리지 않습니다.** 원장과 실제 보유가 어긋난
    상태(밖에서 직접 더 산 경우 등)에서 차수를 다 팔면 여기로 오는데, 그때
    평단을 NULL 로 덮으면 남은 실제 보유분이 진입가를 잃고 손익 계산이
    통째로 무너집니다. 분할 관리에서만 빠지고 포지션은 그대로 둡니다.
    """
    if not ledger.tranches:
        store.upsert_position_state(user_id, mode, key, splits="")
        return
    store.upsert_position_state(
        user_id, mode, key, splits=ledger.to_json(),
        entry_price=ledger.avg_price, quantity=ledger.quantity)


def _handle_split(user_id: int, cfg: dict, brk, engine_risk, inst: Instrument,
                  position, state: dict, sig, result: dict, allow_add: bool,
                  quote: dict = None):
    """차수별 매도 → 없으면 차수 추가 매수. 한 회전에 최대 하나만 실행합니다.

    매도를 먼저 보는 이유는 엔진 전체의 원칙과 같습니다 — 파는 판단이 사는
    판단보다 먼저입니다. 같은 회전에서 팔고 또 사면 수수료만 두 번 냅니다.

    `allow_add` 는 안전장치(일일 손실 한도 등)가 걸렸을 때 False 입니다.
    추가 매수는 신규 진입과 같은 성격이라 그때는 멈춰야 하지만, **차수 매도는
    막지 않습니다** — 잠금은 사는 것을 막는 장치이지 파는 것을 막는 장치가
    아닙니다.
    """
    mode = cfg.get("mode", "paper")
    ledger = _split_ledger(state)
    if not ledger.tranches:
        return                      # 분할로 잡은 포지션이 아닙니다

    price = sig.price if getattr(sig, "ok", False) else (position.current_price or 0)
    if not price:
        return
    rsi, bb = getattr(sig, "rsi", None), getattr(sig, "bb_pct", None)

    # -- 1) 차수 매도 -----------------------------------------------------
    sell = split.plan_sell(cfg, inst, ledger, price, rsi=rsi, bb_pct=bb,
                           held_quantity=position.quantity)
    if sell.ok and sell.quantity < position.quantity:
        _submit_split_sell(user_id, cfg, brk, inst, position, state, ledger,
                           sell, result)
        return
    if sell.ok:
        # 마지막 차수가 목표에 닿았습니다. 이건 부분 매도가 아니라 청산이므로
        # 전량 청산 경로로 보냅니다 — 실현손익 기록·상태 정리 규약이 그쪽에만
        # 있습니다. 이 분기가 없으면 마지막 차수는 익절이 영원히 없습니다
        # (분할 포지션은 전량 목표가를 비워 두기 때문입니다).
        if cfg.get("dry_run"):
            store.log_event(user_id, "exit", f"[모의판단] {inst.name} {sell.reason}",
                            symbol=inst.key, name=inst.name,
                            detail={"dry_run": True, "split": sell.to_dict()})
            result["exits"].append({"symbol": inst.key, "name": inst.name,
                                    "dry_run": True, "quantity": sell.quantity,
                                    "split_tranche": sell.tranche,
                                    "reason": sell.reason})
            return
        gate = engine_risk.check_exit(inst, quote, urgent=False)
        if not gate.approved:
            result["rejects"].append({"symbol": inst.key, "action": "exit",
                                      "reasons": gate.rejects})
            return
        _submit_close(user_id, cfg, brk, inst, position, state,
                      reason=sell.reason + " (마지막 차수 — 전량 청산)",
                      result=result, detail={"split": sell.to_dict()})
        return

    if not allow_add:
        return

    # -- 2) 차수 추가 매수 ------------------------------------------------
    buy = split.plan_buy(cfg, inst, ledger, price, rsi=rsi, bb_pct=bb)
    if not buy.ok:
        return

    # 리스크 게이트는 신규 진입과 **똑같이** 통과해야 합니다. 추가 매수는
    # 노출을 늘리는 행위이고, 분할이라는 이유로 한도를 비켜가면 안 됩니다.
    plan = strategy.EntryPlan(ok=True, quantity=buy.quantity, price=price,
                              stop_price=float((ledger.plan or {}).get("stop") or 0),
                              target_price=0.0,
                              notional_krw=inst.notional(sig.price_krw or price,
                                                         buy.quantity),
                              reason=buy.reason)
    verdict = engine_risk.check_entry(inst, "buy", buy.quantity, sig, plan, None,
                                      adding=True)
    if not verdict.approved:
        result["rejects"].append({"symbol": inst.key, "action": "split_buy",
                                  "reasons": verdict.rejects})
        store.log_event(user_id, "reject",
                        f"{inst.name} {buy.tranche}차 분할 매수 거부 — "
                        f"{' / '.join(verdict.rejects)}",
                        level="warn", symbol=inst.key, name=inst.name,
                        detail={"split": buy.to_dict(), "verdict": verdict.to_dict()})
        return

    if cfg.get("dry_run"):
        store.log_event(user_id, "entry",
                        f"[모의판단] {inst.name} {buy.reason}",
                        symbol=inst.key, name=inst.name,
                        detail={"dry_run": True, "split": buy.to_dict()})
        result["entries"].append({"symbol": inst.key, "name": inst.name,
                                  "dry_run": True, "quantity": buy.quantity,
                                  "split_tranche": buy.tranche})
        return

    coid = make_client_order_id(user_id, inst.key, f"split{buy.tranche}buy", mode=mode)
    if store.order_exists(coid):
        return
    order = brk.submit(inst, "buy", verdict.quantity, order_type="market",
                       note=f"분할매수: {buy.reason}", client_order_id=coid)
    _record(user_id, cfg, inst, order, reason=buy.reason, side=position.side,
            detail={"split": buy.to_dict(), "signal": sig.to_dict(),
                    "strategy": OWNER_AUTO})

    if not order.ok:
        note_broker_reject(user_id, inst.key, order.error or "사유 미상")
        result["errors"].append(f"{inst.name} {buy.tranche}차 분할 매수 실패: {order.error}")
        store.log_event(user_id, "error",
                        f"{inst.name} {buy.tranche}차 분할 매수 실패: {order.error}",
                        level="error", symbol=inst.key, name=inst.name,
                        detail=order.to_dict())
        return

    if order.status == "filled":
        fill_price = order.avg_fill_price or order.price or price
        filled = order.filled_quantity or order.quantity
        split.add_tranche(ledger, buy.tranche, filled, fill_price)
        _save_ledger(user_id, mode, inst.key, ledger)
    result["entries"].append({
        "symbol": inst.key, "name": inst.name, "quantity": order.quantity,
        "price": order.price, "split_tranche": buy.tranche,
        "status": order.status})
    store.log_event(
        user_id, "entry",
        f"{inst.name} {buy.tranche}차 분할 매수 {order.quantity:g}{_unit(inst)} "
        f"@ {order.price:,.2f} "
        f"{'주문 접수 (체결 대기)' if order.status != 'filled' else '체결'} — "
        f"{buy.reason} · 평단 {ledger.avg_price:,.2f} "
        f"({ledger.count}/{len((ledger.plan or {}).get('qty') or [])}차)",
        level="trade", symbol=inst.key, name=inst.name,
        detail={"order": order.to_dict(), "split": buy.to_dict()})


def _submit_split_sell(user_id: int, cfg: dict, brk, inst: Instrument, position,
                       state: dict, ledger: split.Ledger, decision, result: dict):
    """차수 하나만 파는 부분 매도."""
    mode = cfg.get("mode", "paper")

    if cfg.get("dry_run"):
        store.log_event(user_id, "exit", f"[모의판단] {inst.name} {decision.reason}",
                        symbol=inst.key, name=inst.name,
                        detail={"dry_run": True, "split": decision.to_dict()})
        result["exits"].append({"symbol": inst.key, "name": inst.name,
                                "dry_run": True, "quantity": decision.quantity,
                                "split_tranche": decision.tranche,
                                "reason": decision.reason})
        return

    coid = make_client_order_id(user_id, inst.key, f"split{decision.tranche}sell",
                               mode=mode)
    if store.order_exists(coid):
        return

    order = brk.close(inst, quantity=decision.quantity,
                      note=f"분할매도: {decision.reason}", client_order_id=coid)
    _record(user_id, cfg, inst, order, reason=decision.reason,
            side=position.side, detail={"split": decision.to_dict()})

    if not order.ok:
        result["errors"].append(f"{inst.name} {decision.tranche}차 분할 매도 실패: "
                                f"{order.error}")
        store.log_event(user_id, "error",
                        f"{inst.name} {decision.tranche}차 분할 매도 실패: {order.error}",
                        level="error", symbol=inst.key, name=inst.name,
                        detail=order.to_dict())
        return

    realized = None
    if order.status == "filled":
        filled = order.filled_quantity or order.quantity
        entry = float(decision.detail.get("entry") or 0)
        realized = order.realized_pnl
        if realized is None:
            realized = _exit_realized_krw(inst, position.side, entry,
                                          order.avg_fill_price or order.price, filled)
        # 원장에서 그 차수를 빼고 평단·수량을 다시 씁니다. 전량이 아니므로
        # 포지션 상태는 남습니다 (손절선·고점은 그대로 유지됩니다).
        split.remove_tranche(ledger, decision.tranche, filled)
        _save_ledger(user_id, mode, inst.key, ledger)
        if realized is not None:
            store.record_daily_trade(user_id, mode, float(realized))

    result["exits"].append({
        "symbol": inst.key, "name": inst.name, "quantity": order.quantity,
        "price": order.price, "realized_pnl": realized,
        "split_tranche": decision.tranche, "reason": decision.reason,
        "status": order.status})
    pnl = f" (실현 {realized:+,.0f}원)" if realized is not None else ""
    store.log_event(
        user_id, "exit",
        f"{inst.name} {decision.tranche}차 분할 매도 {order.quantity:g}{_unit(inst)} "
        f"@ {order.price:,.2f} — {decision.reason}{pnl} · "
        f"남은 차수 {ledger.count}개",
        level="trade", symbol=inst.key, name=inst.name,
        detail={"order": order.to_dict(), "split": decision.to_dict()})


# ---------------------------------------------------------------------------
# 회전(갈아타기) — engine/rotation.py 가 판단하고, 여기서 실행합니다
# ---------------------------------------------------------------------------

# 회전으로 판 종목의 재진입 금지 (메모리 — 재시작하면 풀립니다. 그 사이의
# 재매수는 protections 쿨다운이 켜져 있으면 그쪽이 한 번 더 막습니다)
_rotation_exits: dict[tuple, float] = {}


def _rotation_block_left(user_id: int, key: str, cfg: dict) -> float:
    """회전 매도 후 재진입 금지가 몇 초 남았는가. 0 이면 제한 없음.

    이게 없으면 A 를 팔아 B 를 산 다음 회전에 A 의 신호가 여전히 좋아서
    A 를 다시 사는 왕복(A→B→A)이 생깁니다 — 수수료만 두 번 냅니다.
    """
    at = _rotation_exits.get((user_id, str(key)), 0)
    if not at:
        return 0.0
    wait = rotation.params(cfg)["rotation_reentry_min"] * 60
    return max(0.0, at + wait - time.time())


def _rotations_today(user_id: int) -> int:
    """오늘 실제로 실행한 회전 횟수 (모의판단 dry_run 은 세지 않습니다)."""
    today = datetime.now().isoformat()[:10]
    rows = store.get_events(user_id, limit=50, kinds=["rotate"])
    return sum(1 for r in rows
               if str(r.get("created_at") or "").startswith(today)
               and not (r.get("detail") or {}).get("dry_run"))


def _maybe_rotate(user_id: int, cfg: dict, brk, engine_risk, inst: Instrument,
                  sig, plan, result: dict, reject_reasons: list):
    """재원 부족으로 거부된 새 후보를 위해 보유 종목 하나를 팔지 검토합니다.

    판다면 이번 회전에서는 **매도까지만** 합니다. 새 후보의 매수는 다음
    회전에서 평소 경로로 다시 시도합니다 — 매도 체결·결제가 확정되기 전에
    사면 이중지출이고, 다음 회전에 신호가 죽어 있으면 사지 않는 것이
    맞습니다 (재검증을 공짜로 얻습니다).
    """
    if not cfg.get("rotation_enabled"):
        return
    if result.get("rotations"):
        return          # 한 회전(tick)에 회전은 한 번 — 연쇄 매도를 막습니다

    # 재원 부족이 원인일 때만. 다른 거부(장 상태·점수 미달·비중 상한 등)는
    # 보유 종목을 판다고 해결되지 않습니다. "주문 수량이 0" 만으로는 부족합니다
    # — 종목 비중 상한이 자른 것이면 팔아도 총자산이 그대로라 여전히 못 삽니다.
    # 그래서 **어느 한도가 잘랐는지**까지 봅니다 (주문가능금액·총 노출·슬롯만 해당).
    text = " / ".join(str(r) for r in reject_reasons)
    capital_bound = any(s in text for s in (
        "보유 종목 수 상한", "주문가능금액", "총 노출"))
    if not capital_bound:
        return

    mode = cfg.get("mode", "paper")
    states = store.get_position_states(user_id, mode)
    held_sigs = {s.get("key"): s for s in result.get("signals") or []
                 if s.get("held")}

    holdings = []
    for pos in engine_risk.positions:
        if pos.key == inst.key:
            continue
        state = states.get(pos.key) or {}
        # 자동매매가 잡은 포지션만 — 사용자가 직접 산 주식과 초단타 포지션은
        # 회전의 재료가 아닙니다 (초단타는 초 단위 규칙이 따로 돕니다)
        if not is_managed(cfg, pos.key, states):
            continue
        if position_owner(cfg, pos.key, state) != OWNER_AUTO:
            continue
        if store.has_open_order(user_id, mode, pos.key):
            continue
        inst_v = instruments.try_resolve(pos.key)
        if inst_v is None:
            continue
        held_sig = held_sigs.get(pos.key) or {}
        holdings.append({
            "inst": inst_v, "key": pos.key, "name": pos.name or inst_v.name,
            "side": pos.side, "quantity": pos.quantity,
            "entry_price": state.get("entry_price") or pos.avg_price,
            # 이번 회전의 청산 검사가 계산한 신선한 값이 우선, 없으면 잔고 스냅샷
            "price": held_sig.get("price") or pos.current_price,
            "target_price": state.get("target_price"),
            "peak_price": state.get("peak_price"),
            "score": held_sig.get("score", state.get("signal_score")),
            "ma50": held_sig.get("ma50"),
            "opened_at": state.get("opened_at"),
            "position": pos, "state": state,
        })
    if not holdings:
        return

    decision = rotation.evaluate(
        cfg,
        {"inst": inst, "key": inst.key, "name": inst.name, "score": sig.score,
         "direction": 1 if sig.direction == strategy.LONG else -1,
         "price": sig.price, "target_price": plan.target_price},
        holdings, _exit_realized_krw, _rotations_today(user_id))

    if not decision.ok:
        # 매 회전 반복되는 사유라 30분에 한 번만 남깁니다 (_log_plan_reject 와 동일)
        if decision.rejects:
            _log_plan_reject(user_id, f"rotate:{inst.key}", inst.name,
                             "회전 보류 — " + decision.rejects[-1],
                             detail=decision.to_dict())
        return

    victim = decision.victim
    if cfg.get("dry_run"):
        store.log_event(user_id, "rotate", f"[모의판단] {decision.reason}",
                        level="info", symbol=inst.key, name=inst.name,
                        detail={**decision.detail, "dry_run": True})
        result.setdefault("rotations", []).append(
            {"victim": victim["key"], "for": inst.key, "dry_run": True,
             "reason": decision.reason})
        return

    # 청산이 나갈 수 있는 상태인지 (장 상태·시세) — 회전은 긴급이 아닙니다
    quote = feed.quote(victim["inst"])
    gate = engine_risk.check_exit(victim["inst"], quote, urgent=False)
    if not gate.approved:
        _log_plan_reject(user_id, f"rotate:{inst.key}", inst.name,
                         "회전 보류 — " + " / ".join(gate.rejects),
                         detail=decision.to_dict())
        return

    order = _submit_close(user_id, cfg, brk, victim["inst"], victim["position"],
                          victim["state"], reason=decision.reason, result=result,
                          detail={"rotation": decision.detail})
    if order is None or not order.ok:
        return

    _rotation_exits[(user_id, victim["key"])] = time.time()
    result.setdefault("rotations", []).append(
        {"victim": victim["key"], "for": inst.key, "reason": decision.reason,
         "status": order.status})
    store.log_event(
        user_id, "rotate",
        decision.reason + " — 매도부터 하고, 신규 진입은 다음 회전에서 다시 검증합니다",
        level="trade", symbol=inst.key, name=inst.name, detail=decision.detail)


# ---------------------------------------------------------------------------
# 진입
# ---------------------------------------------------------------------------

def _handle_entry(user_id: int, cfg: dict, brk, engine_risk, query: str,
                  held: set, result: dict, scalp: bool = False, locks=None):
    """진입 검사 한 종목.

    `scalp` 는 **부르는 쪽이 정합니다**. 예전에는 초단타 스위치가 켜져 있으면
    모든 종목이 초단타 경로를 탔습니다 — 콘솔에 넣어둔 삼성전자가 "초단타
    대상이 아닙니다"로 거부되고, 일반 매매가 통째로 멈췄습니다.
    """
    mode = cfg.get("mode", "paper")
    owner = OWNER_SCALP if scalp else OWNER_AUTO
    inst = _resolve_universe_item(query)
    if inst is None:
        result["errors"].append(f"'{query}' 종목을 찾을 수 없습니다.")
        _log_signal_error(user_id, str(query), str(query),
                          "종목을 찾을 수 없습니다 (코드·별칭을 확인하세요)")
        return

    if inst.key in held and not cfg.get("allow_pyramiding"):
        return

    # 장이 닫혀 있으면 **여기서 끝냅니다** — 시세도 지표도 계산하지 않습니다.
    # 예전에는 신호를 다 만든 뒤 주문 직전에야 리스크 게이트가 거부했습니다.
    # 그래서 미국장 시간(새벽)마다 한국 종목의 지표·ML 을 회전마다 계산해서
    # 버리고, 콘솔에는 "정규장이 아닙니다" 거부 로그만 쌓였습니다.
    # 이건 거부가 아니라 '개장 대기'이므로 로그에 남기지 않고, 회전 요약에만
    # 몇 종목이 쉬는지 셉니다.
    open_now, why, _session = feed.entry_allowed_now(inst, cfg)
    if not open_now:
        result.setdefault("waiting", []).append(
            {"symbol": inst.key, "name": inst.name, "reason": why})
        if scalp:
            mark_scalp_phase(user_id, inst.key, WATCH, name=inst.name, note=why)
        return

    # 종목별 보호 잠금 (쿨다운·부진 종목 등) — 신호 계산 전에 봅니다.
    # 잠긴 종목의 신호를 계산해봐야 어차피 못 들어가고, 시세 조회만 낭비합니다.
    if locks is not None:
        lock = locks.for_symbol(inst.key)
        if lock is not None:
            result["rejects"].append({"symbol": inst.key, "action": "entry",
                                      "reasons": [lock.reason]})
            if scalp:
                mark_scalp_phase(user_id, inst.key, WATCH, name=inst.name,
                                 note=lock.reason)
            return

    # 아직 결과가 안 나온 주문이 있으면 새로 내지 않습니다.
    # (미체결 주문을 무시하고 또 사면 같은 종목을 두 번 사게 됩니다)
    if store.has_open_order(user_id, mode, inst.key):
        if scalp:
            mark_scalp_phase(user_id, inst.key, READY, name=inst.name,
                             note="매수 주문을 냈고 체결을 기다리는 중입니다")
        return

    # 증권사가 거부한 직후에는 잠시 쉽니다 — 외화 부족·미신청 서비스 같은 사유는
    # 다음 회전에 저절로 풀리지 않아서, 재시도해봐야 거부 기록만 쌓입니다.
    left, why = _reject_cooldown_left(user_id, inst.key)
    if left > 0:
        result["rejects"].append({"symbol": inst.key, "action": "entry",
                                  "reasons": [f"증권사 거부 후 대기 ({why})"]})
        if scalp:
            mark_scalp_phase(user_id, inst.key, WATCH, name=inst.name,
                             note=f"증권사 거부 후 대기 ({why})")
        return

    # 회전으로 판 종목은 당분간 다시 사지 않습니다 (A→B→A 왕복 방지)
    left = _rotation_block_left(user_id, inst.key, cfg)
    if left > 0:
        result["rejects"].append({"symbol": inst.key, "action": "entry",
                                  "reasons": [f"회전 매도 후 재진입 대기 "
                                              f"({left / 60:.0f}분 남음)"]})
        return

    quote = feed.quote(inst)

    # 초단타 목록에서 온 종목만 틱 기반 경로를 탑니다
    entry_cfg, sig, scalp_plan = cfg, None, None
    if scalp:
        if not quote:
            _log_signal_error(user_id, inst.key, inst.name, "현재가를 조회할 수 없습니다.")
            mark_scalp_phase(user_id, inst.key, WATCH, name=inst.name,
                             note="현재가를 조회할 수 없습니다")
            return
        entry_cfg, sig, scalp_plan = _scalp_entry(
            user_id, cfg, inst, quote, engine_risk, result)
        if sig is None:
            return

    if sig is None:
        sig = strategy.evaluate(inst, cfg, quote=quote)
    if not sig.ok:
        result["errors"].append(f"{inst.name}: {sig.error}")
        _log_signal_error(user_id, inst.key, inst.name, sig.error)
        return

    result["signals"].append({**sig.to_dict(), "held": inst.key in held})

    if sig.direction == strategy.FLAT:
        return

    cfg = entry_cfg
    account = engine_risk.account
    if scalp_plan is not None:
        # 초단타는 수량을 예산으로 정합니다 (비중·리스크 예산 기반이 아님).
        # plan_entry 를 태우면 position_pct 로 다시 잘려서 소액에서 0주가 됩니다.
        plan = strategy.EntryPlan(
            ok=True, quantity=float(scalp_plan.quantity),
            price=scalp_plan.entry_price,
            stop_price=scalp_plan.stop_price,
            target_price=scalp_plan.target_price,
            notional_krw=scalp_plan.order_krw,
            risk_krw=abs(scalp_plan.entry_price - scalp_plan.stop_price) * scalp_plan.quantity,
            reason=" / ".join(scalp_plan.reasons))
    else:
        plan = strategy.plan_entry(inst, sig, cfg, account)
    if not plan.ok:
        result["rejects"].append({"symbol": inst.key, "action": "entry",
                                  "reasons": [plan.reason]})
        _log_plan_reject(user_id, inst.key, inst.name, plan.reason,
                         detail={"signal": sig.to_dict(), "plan": plan.to_dict()})
        # 현금이 없어 못 산 것이라면 — 보유 종목을 팔아 재원을 만들지 검토
        # (초단타는 예산이 따로 있어 회전 대상이 아닙니다)
        if not scalp:
            _maybe_rotate(user_id, cfg, brk, engine_risk, inst, sig, plan,
                          result, [plan.reason])
        return

    # -- 분할 매수 1차 -----------------------------------------------------
    # 전체 수량을 차수로 쪼개고, 손절선을 **마지막 차수까지 담을 수 있는 선**
    # 으로 다시 잡습니다. 이걸 안 하면 기본 손절(-4%)이 차수 간격(-5%)보다
    # 좁아서 2차를 살 기회가 영원히 오지 않습니다.
    # 숏은 대상이 아닙니다 — '내리면 더 담는다'가 성립하지 않습니다.
    split_plan = None
    if (not scalp and sig.direction == strategy.LONG
            and split_applies(cfg, inst, owner)):
        # 이미 차수 원장이 있는 종목이면 1차를 다시 잡지 않습니다. 추가 매수는
        # 계획대로 분할 엔진(_handle_split)이 합니다 — 여기서 또 1차를 만들면
        # 원장이 통째로 새 계획으로 덮여, 담아둔 차수 기록이 사라집니다.
        # (추가 진입 허용(allow_pyramiding)을 함께 켠 계정에서만 닿는 길입니다)
        if inst.key in held and _split_ledger(
                store.get_position_states(user_id, mode).get(inst.key, {})).tranches:
            return
        sp = split.params(cfg)
        split_plan = split.plan_for(sp, sig.price, plan.quantity,
                                    inst.round_quantity)
        first_qty = float(split_plan["qty"][0])
        if first_qty < plan.quantity:
            # 1차 수량이 전체보다 클 수는 없습니다(차수 최소 1주 보정 때문에
            # 같아질 수는 있습니다). 큰 경우엔 계획을 믿지 않고 원래대로 둡니다.
            plan.quantity = first_qty
            plan.notional_krw = inst.notional(sig.price_krw, first_qty)
            plan.margin_krw = inst.margin_required(sig.price_krw, first_qty, "buy")
            plan.risk_krw = abs(sig.price - split_plan["stop"]) * first_qty
        plan.stop_price = split_plan["stop"]
        plan.target_price = 0.0          # 익절은 차수별 분할 매도가 합니다
        plan.sizing["split"] = {
            "tranches": sp["split_tranches"], "qty": split_plan["qty"],
            "drop_pct": split_plan["drop"], "gain_pct": split_plan["gain"],
            "last_price": round(split_plan["last_price"], 4),
            "stop": round(split_plan["stop"], 4),
            "drawdown_pct": round(split.total_drawdown_pct(sp, sig.price), 2),
        }
        plan.reason = (f"1/{sp['split_tranches']}차 분할 매수 — " + plan.reason
                       + f" · 마지막 차수까지 담으면 1차가 대비 "
                         f"-{split.total_drawdown_pct(sp, sig.price):.1f}% 에서 전량 손절")

    side = "buy" if sig.direction == strategy.LONG else "sell"
    verdict = engine_risk.check_entry(inst, side, plan.quantity, sig, plan, quote)

    # '최소 1주 허용' 때문에 위험 예산을 넘긴 주문은 조용히 넘기지 않습니다.
    # 소액 계좌에서는 1주만 사도 계좌의 4분의 1이 될 수 있는데, 그걸 모르면
    # 리스크 설정이 지켜지고 있다고 착각하게 됩니다.
    #
    # **리스크 게이트를 통과한 뒤에** 남깁니다. 통과 전에 남기면 한도에 걸려
    # 어차피 못 나갈 주문까지 회전마다 경고를 찍어 로그가 그것만으로 찹니다.
    if plan.over_budget and verdict.approved:
        s = plan.sizing
        store.log_event(
            user_id, "sizing",
            f"{inst.name} 최소 1주 허용으로 {verdict.quantity:g}{_unit(inst)} 주문 — "
            f"손절 시 위험 {s.get('risk_krw', 0):,.0f}원({s.get('risk_pct', 0):.1f}%)이 "
            f"1회 위험 예산 {s.get('budget_pct', 0):g}%"
            f"({s.get('risk_budget_krw', 0):,.0f}원)를 넘습니다. "
            f"1주 {s.get('unit_cost_krw', 0):,.0f}원 · 총자산 "
            f"{s.get('total_value_krw', 0):,.0f}원",
            level="warn", symbol=inst.key, name=inst.name, detail=s)

    if not verdict.approved:
        result["rejects"].append({"symbol": inst.key, "action": "entry",
                                  "reasons": verdict.rejects})
        store.log_event(user_id, "reject",
                        f"{inst.name} 리스크 거부 — {' / '.join(verdict.rejects)}",
                        symbol=inst.key, name=inst.name,
                        detail={"signal": sig.to_dict(), "plan": plan.to_dict(),
                                "verdict": verdict.to_dict()})
        # 슬롯·현금·노출 한도가 원인이라면 — 회전으로 자리를 만들지 검토
        if not scalp:
            _maybe_rotate(user_id, cfg, brk, engine_risk, inst, sig, plan,
                          result, list(verdict.rejects))
        return

    if cfg.get("dry_run"):
        store.log_event(
            user_id, "entry",
            f"[모의판단] {inst.name} {sig.direction} {verdict.quantity:g}"
            f"{_unit(inst)} 진입 대상 (점수 {sig.score:+.2f})",
            symbol=inst.key, name=inst.name,
            detail={"dry_run": True, "signal": sig.to_dict(), "plan": plan.to_dict()})
        result["entries"].append({"symbol": inst.key, "name": inst.name,
                                  "dry_run": True, "quantity": verdict.quantity,
                                  "direction": sig.direction, "score": sig.score})
        return

    action = "buy" if side == "buy" else "sell"
    coid = make_client_order_id(user_id, inst.key, action, mode=mode)
    if store.order_exists(coid):
        return

    reason = f"점수 {sig.score:+.2f} · " + " · ".join(sig.reasons[:2])

    # 초단타 진입은 **지정가(매수호가)** 가 기본입니다. 시장가로 사면 매도호가에
    # 체결돼 1틱(저가주에선 1%)을 잃고 시작합니다. 못 잡으면 다음 기회가 있지만,
    # 1틱 손해로 시작한 자리는 되돌릴 방법이 없습니다.
    # (청산은 반대로 시장가입니다 — 못 빠져나오는 것이 더 큰 손실입니다)
    order_type, limit_price = "market", None
    if scalp_plan is not None and scalp_plan.order_type == "limit":
        order_type, limit_price = "limit", scalp_plan.entry_price

    order = brk.submit(inst, side, verdict.quantity, order_type=order_type,
                       price=limit_price,
                       note=f"자동진입: {reason}", client_order_id=coid)
    # 주문 원장에도 주인을 남깁니다 — 접수만 된 주문은 나중에 정산 패스가
    # 포지션을 만드는데, 그때 이 값이 없으면 주인을 잃습니다.
    _record(user_id, cfg, inst, order, reason=reason, side=sig.direction,
            detail={"signal": sig.to_dict(), "plan": plan.to_dict(),
                    "verdict": verdict.to_dict(), "strategy": owner})

    if order.ok:
        # 즉시 체결된 경우에만 손절·목표를 확정합니다. 접수만 된 주문은
        # 체결가를 모르므로, 정산 패스가 실제 체결가로 잡습니다.
        if order.status == "filled":
            entry_price = order.avg_fill_price or order.price or sig.price
            filled_qty = order.filled_quantity or order.quantity
            stop_price, ledger_json = plan.stop_price, None
            if split_plan is not None:
                # 계획을 **실제 체결가**로 다시 세웁니다. 슬리피지가 있으면
                # 차수 트리거와 손절선이 계획서와 어긋나는데, 원장에 남는 값이
                # 실제와 다르면 나중에 왜 그 자리에서 샀는지 설명할 수 없습니다.
                sp = split.params(cfg)
                ledger = split.bootstrap(sp, entry_price, filled_qty,
                                         sum(split_plan["qty"]),
                                         inst.round_quantity)
                stop_price = ledger.plan["stop"]
                ledger_json = ledger.to_json()
            store.upsert_position_state(
                user_id, mode, inst.key, side=sig.direction,
                entry_price=entry_price,
                stop_price=stop_price, target_price=plan.target_price,
                peak_price=entry_price, quantity=filled_qty,
                signal_score=sig.score, opened_at=datetime.now().isoformat(),
                note=reason, strategy=owner, splits=ledger_json)
            held.add(inst.key)

        result["entries"].append({
            "symbol": inst.key, "name": inst.name, "direction": sig.direction,
            "quantity": order.quantity, "price": order.price, "score": sig.score,
            "stop_price": plan.stop_price, "target_price": plan.target_price,
            "status": order.status,
        })
        pending = order.status != "filled"
        store.log_event(
            user_id, "entry",
            f"{inst.name} {'매수' if side == 'buy' else '매도(숏)'} "
            f"{order.quantity:g}{_unit(inst)} @ {order.price:,.2f} "
            f"{'주문 접수 (체결 대기)' if pending else '체결'} — {reason}",
            level="trade", symbol=inst.key, name=inst.name,
            detail={"order": order.to_dict(), "plan": plan.to_dict(),
                    "signal": sig.to_dict()})
    else:
        note_broker_reject(user_id, inst.key, order.error or "사유 미상")
        result["errors"].append(f"{inst.name} 진입 실패: {order.error}")
        hint = _reject_hint(inst, order.error)
        store.log_event(
            user_id, "error",
            f"{inst.name} 진입 실패: {order.error}{hint} "
            f"— {_BROKER_REJECT_COOLDOWN_SEC // 60}분간 이 종목 재시도를 멈춥니다.",
            level="error", symbol=inst.key, name=inst.name, detail=order.to_dict())


def _reject_hint(inst: Instrument, error: str) -> str:
    """증권사 거부 문구에 '그래서 뭘 해야 하는가'를 붙입니다.

    KIS 문구만 보면 원인을 알 수 없는 경우가 많습니다. 특히 미국 주식은
    원화가 있어도 **달러가 없으면** 주문가능금액 초과로 거부됩니다.
    """
    text = str(error or "")
    if inst.market != "US":
        return ""
    if "확인하세요" in text or "신청하세요" in text:
        return ""          # 사전 조회가 이미 구체적으로 안내한 경우
    if "주문가능금액" in text or "가능금액" in text:
        return (" — 미국 주식은 달러로 결제됩니다. 원화가 있어도 외화 예수금이 없으면 "
                "거부됩니다. 증권사 앱에서 환전하거나 통합증거금(원화로 해외주식 주문) "
                "서비스를 신청하세요")
    if "해당종목정보" in text:
        return " — 거래소 코드가 맞지 않습니다 (종목의 상장 거래소를 확인하세요)"
    if "장" in text and "시간" in text:
        return " — 프리마켓 주문은 증권사에 주간거래·프리마켓 서비스 신청이 필요합니다"
    return ""


def _scalp_enabled(cfg: dict) -> bool:
    from engine import scalping
    return bool(scalping.clamp_config(cfg.get("scalp") or {}).get("enabled"))


# ---------------------------------------------------------------------------
# 초단타와 일반 자동매매의 경계
# ---------------------------------------------------------------------------
# 두 전략은 **같은 계좌를 쓰지만 매매 대상도 청산 규칙도 완전히 다릅니다.**
#   · 일반 자동매매 : cfg["universe"]        — %·일 단위, AI 추천으로 갱신
#   · 페니 초단타   : cfg["scalp"]["universe"] — 틱·초 단위, 스크리너가 15초마다 교체
# 섞이면 사고가 납니다. 초단타 스크리너가 콘솔의 매매 대상을 밀어내고, 일반
# 종목이 3틱에 던져지고, 초단타가 산 동전주를 일반 전략이 며칠 들고 있습니다.
# 그래서 대상 목록도, 포지션의 주인도 따로 둡니다.
# (주인 값 OWNER_AUTO / OWNER_SCALP 는 임포트 순서 때문에 모듈 맨 위에 있습니다.)


def scalp_universe(cfg: dict) -> list[str]:
    """초단타 전용 매매 대상 (지정 종목이 항상 앞).

    초단타가 꺼져 있어도 목록 자체는 돌려줍니다 — 화면은 꺼진 상태에서도
    "무엇을 대상으로 잡아뒀는지"를 보여줘야 하기 때문입니다. 실제로 매매할지는
    부르는 쪽에서 `enabled` 로 판단합니다.
    """
    from engine import scalping

    return list(scalping.clamp_config(cfg.get("scalp") or {}).get("universe") or [])


def position_owner(cfg: dict, key: str, state: dict) -> str:
    """이 포지션은 누가 잡았는가 — 청산 규칙을 정하는 값입니다.

    진입 때 박아둔 값이 우선입니다. 초단타 대상은 15초마다 바뀌므로, 지금
    목록에 있는지로 판단하면 **들고 있는 동안 주인이 바뀝니다** (사놓고 목록에서
    빠지는 순간 일반 전략이 그 포지션의 손절을 가져갑니다).
    """
    tag = str((state or {}).get("strategy") or "").strip()
    if tag in (OWNER_AUTO, OWNER_SCALP):
        return tag
    # 주인을 기록하기 전에 잡힌 포지션 — 지금 초단타 대상이면 초단타로 봅니다
    return OWNER_SCALP if str(key) in set(scalp_universe(cfg)) else OWNER_AUTO


def split_legacy_universe(user_id: int, cfg: dict) -> dict:
    """옛 설정 한 번 갈라주기 — 초단타가 콘솔 대상을 밀어낸 흔적을 되돌립니다.

    분리 전에는 초단타 스크리너가 `universe` 를 통째로 덮어썼습니다. 그래서
    초단타를 켜둔 계정의 매매 대상은 지금 동전주로 채워져 있고, 그대로 두면
    이번 변경 뒤에 **일반 전략이 그 동전주를 며칠짜리 규칙으로 매매합니다.**
    사람이 직접 넣은 종목(manual_universe)만 콘솔에 남기고 나머지는 비웁니다.
    """
    scalp = dict(cfg.get("scalp") or {})
    if "universe" in scalp:
        return cfg                      # 이미 분리된 설정

    pinned = [str(s).strip() for s in (scalp.get("pinned") or []) if str(s).strip()]
    patch = {"scalp": {**scalp, "universe": pinned}}

    if scalp.get("enabled") and scalp.get("auto_universe"):
        manual = [s for s in (cfg.get("manual_universe") or [])
                  if s in (cfg.get("universe") or [])]
        dropped = [s for s in (cfg.get("universe") or []) if s not in manual]
        if dropped:
            patch["universe"] = manual
            store.log_event(
                user_id, "screen",
                "초단타와 자동매매 대상을 분리했습니다 — 초단타 스크리너가 넣었던 "
                f"{len(dropped)}종목을 자동매매 대상에서 뺐습니다 "
                f"({', '.join(dropped[:5])}{'…' if len(dropped) > 5 else ''}). "
                "초단타는 이제 자기 목록으로만 돕니다.",
                level="warn", detail={"dropped": dropped, "kept": manual})

    return store.save_config(user_id, patch)


def _scalp_reject(user_id: int, result: dict, inst: Instrument, reasons: list):
    result["rejects"].append({"symbol": inst.key, "action": "scalp_entry",
                              "reasons": reasons})
    # 진입 거부 사유는 곧 '지금 관망하는 이유'입니다 — 화면에 그대로 씁니다
    mark_scalp_phase(user_id, inst.key, WATCH, name=inst.name,
                     note=(reasons[0] if reasons else ""))


# ---------------------------------------------------------------------------
# 초단타가 지금 무엇을 하고 있는가 (화면 표시용)
# ---------------------------------------------------------------------------
# 회전이 **이미 내린 판단**을 그대로 담아둡니다. 화면이 물을 때마다 시세를 다시
# 조회해 새로 계산하면, 몇 틱 승부에 써야 할 REST 한도를 표시용으로 태우고,
# 그렇게 만든 숫자는 엔진이 실제로 본 것과 다릅니다. 늦은 화면이 다른 화면보다
# 낫습니다 — 대신 얼마나 늦었는지(age_sec)를 항상 같이 보여줍니다.

WATCH, READY, HOLD = "watch", "ready", "hold"
PHASE_LABELS = {WATCH: "관망중", READY: "매수 준비중", HOLD: "매도 준비중"}
_scalp_phase: dict[tuple[int, str], dict] = {}
_PHASE_TTL_SEC = 300.0          # 이보다 오래된 판단은 '지금 상태'가 아닙니다


def mark_scalp_phase(user_id: int, key: str, phase: str, note: str = "", **extra):
    """이번 회전에서 이 종목을 어떻게 판단했는지 기록합니다."""
    _scalp_phase[(user_id, key)] = {"phase": phase, "note": note,
                                    "at": time.time(), **extra}


def scalp_phases(user_id: int, cfg: dict) -> list[dict]:
    """추적 종목별 상태. 기록이 없거나 오래됐으면 **그 사실을 그대로** 돌려줍니다."""
    keys = scalp_universe(cfg)
    now, out = time.time(), []
    for key in keys:
        rec = _scalp_phase.get((user_id, key))
        age = round(now - rec["at"], 1) if rec else None
        fresh = rec is not None and age <= _PHASE_TTL_SEC
        phase = rec["phase"] if fresh else None
        row = {"key": key,
               "name": (rec or {}).get("name") or key,
               "phase": phase,
               "label": PHASE_LABELS.get(phase, "판단 전"),
               "note": rec["note"] if fresh else "아직 이번 회전에서 보지 않았습니다",
               "age_sec": age if fresh else None}
        if fresh:
            row.update({k: rec[k] for k in
                        ("price", "score", "entry_price", "target_price",
                         "stop_price", "quantity", "order_krw") if k in rec})
        out.append(row)
    return out


STATE_LABELS_SCALP = {"running": "진행중", "off": "중지됨",
                      "blocked": "차단됨", "closed": "장 시간 아님"}


def scalp_status(user_id: int, cfg: dict) -> dict:
    """페니 초단타가 지금 돌고 있는가, 돌고 있다면 무엇을 하는 중인가.

    '돌고 있다'는 초단타 스위치 하나로 정해지지 않습니다. 자동매매가 켜져
    있고, 엔진 루프가 살아 있고, 킬 스위치가 안 걸렸고, 장이 열려 있어야
    비로소 판단이 돕니다. 하나라도 빠지면 그 사실을 그대로 적습니다 —
    켜두기만 하고 실제로는 안 도는 상태를 '진행중'으로 보여주면, 사용자는
    자기 돈이 감시받고 있다고 믿게 됩니다.
    """
    from engine import scalping

    scalp = scalping.clamp_config(cfg.get("scalp") or {})
    mode = cfg.get("mode", "paper")
    phases = scalp_phases(user_id, cfg)

    # 손실 한도는 **초단타가 만든 손익만** 봅니다. 일반 자동매매의 손익이
    # 섞이면, 콘솔에서 난 손실 때문에 초단타가 멈추거나 그 반대가 됩니다.
    today = store.realized_today(user_id, mode, symbols=scalp_universe(cfg))
    loss_cap = float(scalp["daily_loss_krw"])

    state, reason = "running", ""
    if not scalp.get("enabled"):
        state, reason = "off", "초단타가 꺼져 있습니다 — [초단타 켜기] 를 켜고 저장하세요."
    elif not cfg.get("enabled"):
        state, reason = "off", "자동매매가 중지 상태입니다 — 위 [시작] 을 눌러야 판단이 돕니다."
    elif cfg.get("kill_switch"):
        state, reason = "blocked", "킬 스위치가 걸려 있습니다 — 신규 진입을 막고 있습니다."
    elif not loop.running:
        state, reason = "blocked", "엔진 루프가 멈춰 있습니다 — 서버를 다시 켜세요."
    elif cfg.get("state") == store.HALTED:
        state, reason = "blocked", cfg.get("state_reason") or "리스크 한도로 중단된 상태입니다."
    elif today["realized_pnl"] <= -loss_cap:
        state = "blocked"
        reason = (f"오늘 확정 손실 {abs(today['realized_pnl']):,.0f}원 — "
                  f"한도 {loss_cap:,.0f}원에 닿아 오늘은 종료했습니다.")
    else:
        checks = [(m, *scalping.market_session_guard(m))
                  for m in (scalp.get("markets") or ["KR"])]
        if not any(ok for _, ok, _ in checks):
            state = "closed"
            reason = " / ".join(f"{m} — {why}" for m, _, why in checks)
        elif not phases:
            reason = "추적 중인 종목이 없습니다 — [지금 종목 찾기] 로 대상을 정하세요."

    running = state == "running"
    phase = None
    if running:
        # 가장 앞선 단계를 대표로 보여줍니다 — 하나라도 들고 있으면 '매도 준비중'
        for candidate in (HOLD, READY, WATCH):
            if any(p["phase"] == candidate for p in phases):
                phase = candidate
                break

    return {
        "running": running,
        "state": state,
        "state_label": STATE_LABELS_SCALP[state],
        "reason": reason,
        "phase": phase,
        "phase_label": PHASE_LABELS.get(phase, ""),
        "symbols": phases,
        "interval_sec": max(int(cfg.get("interval_sec", 60)), 15),
        "realized_today": today["realized_pnl"],
        "daily_loss_krw": loss_cap,
        "checked_at": datetime.now().isoformat(),
    }


def scalp_budget_left(cfg: dict, positions: list) -> float:
    """이 전략이 아직 쓸 수 있는 금액.

    `budget_krw` 에서 **이미 초단타 대상에 들어가 있는 평가금액**을 뺍니다.
    비중(%)이 아니라 이 한 줄이 규모를 통제합니다. 계좌가 얼마든, 이 전략은
    여기 적힌 금액 밖으로 나가지 않습니다.
    """
    from engine import scalping

    scalp = scalping.clamp_config(cfg.get("scalp") or {})
    # 초단타 대상에 들어간 돈만 셉니다 — 콘솔이 산 종목까지 세면, 일반 매매가
    # 예산을 다 먹은 것처럼 보여 초단타가 한 주도 못 삽니다 (반대도 마찬가지).
    tracked = set(scalp.get("universe") or [])
    used = sum(float(p.market_value or 0) for p in (positions or [])
               if p.key in tracked)
    return max(float(scalp["budget_krw"]) - used, 0.0)


def _scalp_quote(inst: Instrument, quote: dict) -> dict:
    """실시간 체결이 있으면 그걸로 갈아끼웁니다. 국내·미국 모두.

    REST 시세는 캐시 TTL 이 20초입니다(feed.QUOTE_TTL). 몇 틱 승부에서 20초 전
    가격은 그냥 다른 종목입니다. 그리고 **호가(bid/ask)는 REST 현재가 응답에
    없어서**, 스프레드 판단은 실시간 스트림이 있어야만 가능합니다.

    해외 시세는 계정에 따라 **지연**되어 옵니다. 지연된 틱은 받아들이지 않고
    `quote_delay_sec` 로 표시해서, 상위(_scalp_entry)가 진입을 막게 합니다.
    지연된 값으로 초봉을 그리면 과거를 현재로 착각하게 됩니다.
    """
    from data_sources import kis_realtime
    from engine import scalping

    try:
        tick = kis_realtime.tick(inst.key)
    except Exception:
        return quote
    if not tick or tick.get("age_sec", 999) > 10:
        return quote          # 오래된 틱은 없느니만 못합니다

    delay = float(tick.get("delay_sec") or 0)
    if delay > scalping.HARD_LIMITS["max_quote_delay_sec"]:
        return {**quote, "quote_delay_sec": delay,
                "source": tick.get("source", "") + " (지연)"}
    # 미국 종목의 틱은 달러입니다. 원화로 환산하지 않고 price_krw 에 넣으면
    # 3달러짜리를 3원으로 보고 예산·수량을 계산합니다.
    from data_sources import fx
    price_krw = (fx.to_krw(tick["price"], inst.currency)
                 if inst.currency != "KRW" else tick["price"])

    return {
        **quote,
        "quote_delay_sec": delay,
        "price": tick["price"],
        "price_krw": price_krw,
        "bid": tick.get("bid"),
        "ask": tick.get("ask"),
        "high": tick.get("high"),
        "low": tick.get("low"),
        "day_range_pct": tick.get("day_range_pct"),
        "change_rate": tick.get("change_rate", quote.get("change_rate")),
        "volume": tick.get("volume", quote.get("volume")),
        "age_sec": tick.get("age_sec", 0),
        "source": tick.get("source", "KIS 실시간체결"),
    }


def _scalp_entry(user_id: int, cfg: dict, inst: Instrument, quote: dict,
                 engine_risk, result: dict):
    """페니 초단타 진입 판단. (설정, 신호, 주문계획) 을 돌려줍니다.

    부적격이면 (cfg, None, None). **부적격 사유는 항상 남깁니다** — 예전에는
    대상 가격대가 아니면 조용히 일반 전략으로 빠져서, 종목을 지정해놓고
    "초단타가 안 도는데 이유를 모르는" 상태가 됐습니다.

    통제 항목 (하루 매매 횟수는 없습니다 — 초단타에 맞지 않아 걷어냈습니다):
        · 대상 가격대 · 진입 가능 시간대 · 호가 스프레드
        · 남은 예산(budget_krw) · 오늘 손실 금액(daily_loss_krw)
        · 같은 종목 재진입 쿨다운
    """
    from engine import scalping

    scalp = scalping.clamp_config(cfg.get("scalp") or {})
    quote = _scalp_quote(inst, quote)
    price = float(quote.get("price") or 0)

    if not scalping.is_penny_target(inst, price, scalp):
        _scalp_reject(user_id, result, inst, [
            "초단타 대상이 아닙니다 — " + scalping.why_not_target(inst, price, scalp)])
        return cfg, None, None

    tradable, why = scalping.session_guard(inst)
    if not tradable:
        _scalp_reject(user_id, result, inst, [why])
        return cfg, None, None

    # 지연 시세로는 몇 틱 승부가 성립하지 않습니다. 해외 실시간은 계정에 따라
    # 지연되어 오는데, 그걸 모르고 3틱을 재면 이미 지나간 자리에 주문이 나갑니다.
    delay = float(quote.get("quote_delay_sec") or 0)
    delay_cap = scalping.HARD_LIMITS["max_quote_delay_sec"]
    if delay > delay_cap:
        _scalp_reject(user_id, result, inst, [
            f"시세가 {delay:.0f}초 지연됐습니다 (허용 {delay_cap}초) — "
            f"지연 시세로는 틱 매매를 하지 않습니다"])
        return cfg, None, None

    # 오늘 손실 금액 — 그날을 끊는 것은 이제 이것 하나입니다.
    # **초단타 대상에서 난 손익만** 셉니다 (일반 자동매매와 지갑이 다릅니다).
    mode = cfg.get("mode", "paper")
    today = store.realized_today(user_id, mode, symbols=scalp_universe(cfg))
    loss_cap = float(scalp["daily_loss_krw"])
    if today["realized_pnl"] <= -loss_cap:
        _scalp_reject(user_id, result, inst, [
            f"오늘 확정 손실 {abs(today['realized_pnl']):,.0f}원 — "
            f"한도 {loss_cap:,.0f}원 도달, 오늘은 종료합니다"])
        return cfg, None, None

    # 같은 종목 연속 타격 방지 — 자기 주문끼리 호가가 부딪히는 것을 막습니다
    cooldown = float(scalp["reentry_cooldown_sec"])
    last_exit = _last_scalp_exit.get((user_id, inst.key), 0)
    waited = time.time() - last_exit
    if last_exit and waited < cooldown:
        _scalp_reject(user_id, result, inst, [f"재진입 쿨다운 {waited:.0f}/{cooldown:.0f}초"])
        return cfg, None, None

    budget_left = scalp_budget_left(cfg, engine_risk.positions)
    if budget_left < price:
        _scalp_reject(user_id, result, inst, [
            f"남은 예산 {budget_left:,.0f}원 (투자금액 {scalp['budget_krw']:,.0f}원) — "
            f"{price:,.0f}원짜리를 살 수 없습니다"])
        return cfg, None, None

    signal = scalping.evaluate(inst, scalp, quote=quote)
    if not signal.ok:
        _log_signal_error(user_id, inst.key, inst.name, signal.error)
        mark_scalp_phase(user_id, inst.key, WATCH, name=inst.name,
                         note=signal.error or "신호를 계산하지 못했습니다")
        return cfg, None, None

    sig = strategy.Signal(
        key=inst.key, ok=True, score=signal.score,
        direction=(strategy.LONG if signal.direction == scalping.LONG else strategy.FLAT),
        confidence=min(abs(signal.score) / max(float(scalp["entry_score"]), 0.01), 1.0),
        price=signal.price, price_krw=float(quote.get("price_krw") or signal.price),
        bars_used=signal.bars_used, quote_age=float(quote.get("age_sec") or 0),
        reasons=signal.reasons + [
            f"초단타 · 스프레드 {signal.spread_ticks if signal.spread_ticks is not None else '?'}틱"],
    )

    # 수량·목표가·손절가를 틱으로 직접 계산합니다.
    # strategy.plan_entry 는 비중(%)·리스크 예산 기반이라 이 전략과 맞지 않습니다.
    plan = None
    if sig.direction == strategy.LONG:
        plan = scalping.plan_order(
            inst, quote, scalp, budget_left,
            trading_value_krw=float(quote.get("trading_value") or 0))
        if not plan.ok:
            _scalp_reject(user_id, result, inst, plan.reasons or ["주문 계획을 세울 수 없습니다"])
            return cfg, None, None
        mark_scalp_phase(
            user_id, inst.key, READY, name=inst.name,
            note=f"진입 점수 {signal.score:+.2f} ≥ 기준 {float(scalp['entry_score']):.2f} · "
                 f"{plan.quantity:g}주 {plan.entry_price:,.0f}원",
            price=signal.price, score=round(signal.score, 3),
            entry_price=plan.entry_price, target_price=plan.target_price,
            stop_price=plan.stop_price, quantity=plan.quantity,
            order_krw=round(plan.order_krw))
    else:
        # 스프레드·왕복비용 게이트에서 끊기면 점수를 아예 안 냅니다(bars_used=0).
        # 그때 '점수 미달'이라고 쓰면 신호가 약한 것처럼 보이지만, 실제로는
        # 신호를 볼 필요도 없이 산수가 안 맞는 자리입니다 — 사유를 그대로 씁니다.
        gated = not signal.bars_used and signal.reasons
        mark_scalp_phase(
            user_id, inst.key, WATCH, name=inst.name,
            note=(signal.reasons[0] if gated else
                  f"진입 점수 {signal.score:+.2f} · 기준 {float(scalp['entry_score']):.2f} 미달"),
            price=signal.price, score=round(signal.score, 3))

    # 청산 조건은 진입 때 확정합니다 (호가단위가 가격대마다 달라서,
    # 나중에 %로 되돌리면 실제 호가와 어긋납니다)
    econ = plan.economics if plan else scalping.tick_economics(
        price, scalp, "US" if inst.market == "US" else "KR")
    entry_cfg = {
        **cfg,
        "atr_stop_mult": 0,
        "stop_loss_pct": abs(econ["stop_price"] - econ["price"]) / econ["price"] * 100,
        "take_profit_pct": abs(econ["target_price"] - econ["price"]) / econ["price"] * 100,
        "max_hold_sec": scalp["max_hold_sec"],
        "max_hold_minutes": 0,
        "max_hold_days": 0,
        "min_one_unit": True,
    }
    return entry_cfg, sig, plan


def _resolve_universe_item(query) -> Instrument | None:
    """유니버스 항목을 Instrument 로. 'FUT_FRONT' 같은 별칭을 지원합니다.

    선물 코드는 만기가 지나면 죽으므로, 유니버스에 코드를 고정하지 않고
    별칭으로 적어두면 매번 최근월물로 자동 치환됩니다.
    """
    text = str(query or "").strip()
    if not text:
        return None
    upper = text.upper()
    if upper in ("FUT_FRONT", "K200_FUT", "코스피200선물"):
        return feed.front_month_futures("01")
    if upper in ("MINI_FUT", "미니선물"):
        return feed.front_month_futures("05")
    if upper in ("KQ150_FUT", "코스닥150선물"):
        return feed.front_month_futures("06")
    return instruments.try_resolve(text)


def _unit(inst: Instrument) -> str:
    return "계약" if inst.is_derivative else "주"


def _record(user_id: int, cfg: dict, inst: Instrument, order, reason: str,
            side: str = "", detail: dict = None):
    # 주문을 냈으면 보유 목록이 곧 바뀝니다. 보호 회전이 옛 목록으로 판단하면
    # 방금 판 종목을 또 팔려 들거나, 없는 포지션의 손절선을 재게 됩니다.
    invalidate_positions(user_id)
    store.record_order(user_id, {
        "client_order_id": order.client_order_id,
        "broker_mode": cfg.get("mode", "paper"),
        "broker_order_id": order.broker_order_id,
        "symbol": inst.key, "name": inst.name, "asset_class": inst.asset_class,
        "action": order.action, "side": side,
        "quantity": order.quantity, "price": order.price,
        "price_krw": order.price_krw, "fee": order.fee,
        "realized_pnl": order.realized_pnl,
        "status": order.status if order.ok else "rejected",
        "reason": reason if order.ok else (order.error or reason),
        "detail": detail,
        "intended_price": order.intended_price or order.price,
        "filled_quantity": order.filled_quantity,
        "avg_fill_price": order.avg_fill_price,
        "slippage_bps": order.slippage_bps,
    })


# ---------------------------------------------------------------------------
# 백그라운드 루프
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 장 상태 알림 — 세션이 바뀔 때 로그로 알려줍니다
# ---------------------------------------------------------------------------

SESSION_MESSAGES = {
    ("KR", "PRE_AUCTION"): "한국 장 시작 동시호가가 시작되었습니다 "
                           "(08:30~09:00 시초가 결정 · 08:30~08:40 장전 시간외 종가 병행).",
    ("KR", "REGULAR"): "한국 시장이 열렸습니다 — 정규장 시작 (09:00~15:30).",
    ("KR", "CLOSE_AUCTION"): "한국 장 마감 동시호가입니다 (15:20~15:30 종가 결정).",
    ("KR", "AFTER_CLOSE"): "한국 장후 시간외 종가 거래가 시작되었습니다 "
                           "(15:40~16:00, 당일 종가로만 체결).",
    ("KR", "AFTER"): "한국 시간외 단일가가 시작되었습니다 (16:00~18:00, 10분 단위 ±10%).",
    ("KR", "CLOSED"): "한국 시장이 마감되었습니다.",
    ("KR", "WEEKEND"): "한국 시장 주말 휴장입니다.",
    ("US", "PRE"): "미국 프리마켓이 열렸습니다.",
    ("US", "REGULAR"): "미국장이 열렸습니다 — 정규장 시작.",
    ("US", "AFTER"): "미국 정규장이 끝나고 애프터마켓입니다.",
    ("US", "CLOSED"): "미국장이 마감되었습니다.",
    ("US", "WEEKEND"): "미국 시장 주말 휴장입니다.",
}

_last_sessions: dict[str, str] = {}


def session_message(market: str, session: str) -> str:
    return SESSION_MESSAGES.get((market, session),
                                f"{'한국' if market == 'KR' else '미국'} 시장 세션: {session}")


def watch_market_sessions():
    """세션 전환을 감지해 자동매매 설정이 있는 모든 계정의 로그에 남깁니다.

    처음 관찰한 상태는 기록만 하고 알리지 않습니다 — 서버를 켤 때마다
    "장이 마감되었습니다"가 찍히면 진짜 전환과 구분이 안 됩니다.
    """
    from data_sources import market_clock

    changes = []
    for market, status_key in (("KR", "KOSPI"), ("US", "US")):
        try:
            session = market_clock.status_for(status_key).get("session", "")
        except Exception:
            continue
        if not session:
            continue
        previous = _last_sessions.get(market)
        _last_sessions[market] = session
        if previous is not None and previous != session:
            changes.append((market, session))

    if not changes:
        return
    users = store.configured_users()
    for market, session in changes:
        message = session_message(market, session)
        for user_id in users:
            store.log_event(user_id, "market", message,
                            detail={"market": market, "session": session})


class EngineLoop:
    """켜져 있는 모든 사용자의 자동매매를 주기적으로 돌리는 데몬 스레드."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_run: dict[int, float] = {}
        self._last_guard: dict[int, float] = {}
        self.last_tick_at: str = ""
        self.tick_count = 0
        self.guard_count = 0
        self.last_error = ""

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="athena-autotrade")
        self._thread.start()

    def stop(self):
        self._stop.set()

    # 다음 회전 시각을 계산하지 못했을 때의 안전 대기 (초)
    _FALLBACK_WAIT = 5.0
    _MAX_WAIT = 15.0

    def _run(self):
        # 서버 기동 직후에는 시세 소스가 아직 준비되지 않았을 수 있습니다
        self._stop.wait(10)
        while not self._stop.is_set():
            try:
                self._sweep()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            # **다음 회전이 실제로 필요한 시각까지만** 잡니다.
            #
            # 예전에는 무조건 15초씩 잤습니다. 회전 자체가 10초쯤 걸리므로
            # 실제 주기는 (작업시간 + 15초) 격자가 되고, 그 격자에서 설정값을
            # 처음 넘기는 칸에서야 돌았습니다. 60초로 맞춰둬도 71초마다 도는
            # 이유가 이것이었습니다(실측 02:56:55 → 02:58:06 → 02:59:17).
            # 짧은 주기를 설정할수록 오차가 커져서, 30초 설정이 41초가 됩니다.
            self._stop.wait(self._sleep_seconds())

    def _sleep_seconds(self) -> float:
        """가장 먼저 돌아올 회전(전체 또는 보호)의 예정 시각까지 남은 시간."""
        now = time.time()
        due = []
        try:
            for user_id in store.enabled_users():
                cfg = store.get_config(user_id)
                interval = max(int(cfg.get("interval_sec", 60)), 5)
                due.append(self._last_run.get(user_id, 0) + interval - now)
                guard = int(cfg.get("guard_interval_sec", 5) or 0)
                if guard > 0:
                    due.append(self._last_guard.get(user_id, 0)
                               + max(guard, 2) - now)
        except Exception:
            return self._FALLBACK_WAIT
        if not due:
            return self._MAX_WAIT      # 켜진 사용자가 없으면 느긋하게
        # 0.5초 하한 — 이미 지난 예정은 즉시 다시 돌되 CPU 를 태우지는 않습니다
        return max(0.5, min(self._MAX_WAIT, min(due)))

    def _sweep(self):
        # 장 상태 전환 알림 — 자동매매가 꺼져 있어도 알려줍니다
        try:
            watch_market_sessions()
        except Exception:
            pass
        # AI 자동 추적 — **매매를 켜지 않아도** 주기적으로 새 종목을 찾습니다.
        # (추적과 매매는 별개입니다: 찾기만 하고 매매는 사람이 결정할 수 있어야 합니다)
        try:
            self._sweep_tracking()
        except Exception as exc:
            self.last_error = f"추적 sweep 실패: {exc}"
        for user_id in store.enabled_users():
            cfg = store.get_config(user_id)
            interval = max(int(cfg.get("interval_sec", 60)), 5)
            last = self._last_run.get(user_id, 0)
            now = time.time()
            if now - last >= interval:
                # 다음 예정은 **이번 예정 시각 기준**으로 잡습니다. now 로 잡으면
                # 회전에 걸린 시간이 매번 더해져 주기가 조금씩 밀립니다(드리프트).
                # 오래 멈춰 있었다면(서버 재시작 등) 지금부터 다시 셉니다.
                self._last_run[user_id] = (last + interval
                                           if now - last < interval * 2 else now)
                # 전체 회전이 청산까지 봤으므로 보호 회전 타이머도 리셋합니다
                self._last_guard[user_id] = now
                run_once(user_id)
                self.tick_count += 1
                self.last_tick_at = datetime.now().isoformat()
                continue

            # 전체 회전 사이사이에 **손절만** 지켜보는 짧은 회전을 끼웁니다.
            guard = int(cfg.get("guard_interval_sec", 5) or 0)
            if guard <= 0:
                continue
            guard = max(guard, 2)
            last_guard = self._last_guard.get(user_id, 0)
            if now - last_guard >= guard:
                self._last_guard[user_id] = now
                try:
                    out = run_guard(user_id)
                    if out.get("ok") and out.get("checked"):
                        self.guard_count += 1
                except Exception as exc:
                    self.last_error = f"보호 회전 실패: {exc}"

    def _sweep_tracking(self):
        for user_id in store.configured_users():
            cfg = store.get_config(user_id)
            if not cfg.get("auto_universe"):
                continue
            period = max(float(cfg.get("auto_universe_refresh_min", 30)), 5) * 60
            if time.time() - _last_universe_refresh.get(user_id, 0) < period:
                continue
            with _refresh_guard:
                if user_id in _refreshing_users:
                    continue          # 이전 갱신이 아직 도는 중
                _refreshing_users.add(user_id)
            _last_universe_refresh[user_id] = time.time()
            threading.Thread(target=_tracking_refresh, args=(user_id,),
                             daemon=True, name=f"athena-track-{user_id}").start()

    def status(self) -> dict:
        return {
            "running": self.running,
            "tick_count": self.tick_count,
            "guard_count": self.guard_count,     # 손절만 보는 고속 회전 횟수
            "last_tick_at": self.last_tick_at,
            "last_error": self.last_error,
            "enabled_users": len(store.enabled_users()),
        }


loop = EngineLoop()


# ---------------------------------------------------------------------------
# 백테스트 (같은 신호·같은 청산 규칙으로 과거를 돌려봅니다)
# ---------------------------------------------------------------------------

def simulate(query: str, cfg: dict = None, days: int = 250,
             initial_cash: float = 10_000_000, falsify: int = 0) -> dict:
    """설정을 실제로 켜기 전에 과거 데이터로 검증합니다.

    라이브와 **같은 evaluate() / check_exit()** 를 씁니다. 백테스트 전용 로직을
    따로 만들면 "백테스트는 좋은데 실매매는 다르게 동작하는" 고전적 함정에 빠집니다.

    돌려주는 숫자를 읽는 순서 (총수익률부터 보면 안 됩니다)
        1) `excess_return_pct` — 그냥 사서 들고 있는 것보다 나았는가.
           상승장에서는 아무 전략이나 총수익률이 좋습니다. 매매가 실제로 무엇을
           보탰는지는 이 값에만 나옵니다.
        2) `active_sharpe` — 그 초과분이 흔들림 대비 얼마나 안정적인가
        3) `falsification` — 예측 가능한 순서를 없앤 가짜 데이터에서도 같은 성적이
           나오는가 (falsify>0 일 때만 계산합니다)

    한계 (정직하게)
        · 일봉 종가 기준 체결이라 장중 손절 도달은 다음 날 종가로 처리됩니다
        · 분봉·뉴스는 과거 데이터를 못 구해 반영하지 않습니다 (일봉 신호만)
        · 슬리피지는 고정 스프레드로 근사합니다
      따라서 결과는 "이 설정이 말이 되는가"의 점검이지 수익 예측이 아닙니다.
    """
    cfg = {**store.DEFAULT_CONFIG, **(cfg or {})}
    # 과거 분봉·뉴스를 재현할 수 없으므로 일봉 신호만 사용합니다
    cfg = {**cfg, "intraday_weight": 0.0, "use_news": False}

    inst = _resolve_universe_item(query)
    if inst is None:
        return {"ok": False, "error": f"'{query}' 종목을 찾을 수 없습니다."}

    bars = feed.bars(inst, "day", count=max(days + 60, 120))
    if bars is None or len(bars) < strategy.MIN_DAILY_BARS + 20:
        return {"ok": False, "error": "백테스트에 필요한 일봉이 부족합니다."}

    run = _run_backtest(inst, bars, cfg, initial_cash)
    curve, trades = run["curve"], run["trades"]

    # 벤치마크 = 같은 기간 첫날 사서 끝까지 들고 있기.
    # 이것을 못 이기면 매매는 수수료를 내며 시장을 따라간 것입니다.
    benchmark = _buy_and_hold_curve(inst, bars, run["start_index"], initial_cash)

    metrics = _metrics(curve, trades, initial_cash)
    metrics.update(validation.active_metrics(curve, benchmark))

    out = {"ok": True, "instrument": inst.to_dict(), **metrics,
           "trades": trades[-50:], "curve": curve[-250:],
           "benchmark_curve": benchmark[-250:],
           # 체결 가정을 결과에 같이 실어 보냅니다. 나중에 이 백테스트를 다시
           # 볼 때 "어떤 체결 모형에서 나온 숫자인가" 를 못 되짚으면, 모형을
           # 바꾼 전후 결과를 비교할 수 없습니다.
           "execution": {
               "fill": "next_bar_open",
               "intrabar_stops": True,
               "stop_priority": "stop_first",
               "gap_through_stop": "open",
               "finalize_open_position": True,
               "slippage_bps": float(cfg.get("slippage_bps", 5)),
               "note": "판단은 종가, 체결은 다음 봉 시가. 손절·익절은 봉 안 "
                       "고가/저가로 판정하며 같은 봉에 둘 다 닿으면 손절 우선.",
           },
           "rejections": run["rejections"].summary(),
           "bars_used": len(bars), "config": {
               k: cfg[k] for k in ("entry_score", "exit_score", "atr_stop_mult",
                                   "stop_loss_pct", "take_profit_pct",
                                   "trailing_stop_pct", "max_hold_days",
                                   "risk_per_trade_pct", "position_pct")
               if k in cfg}}
    if falsify:
        out["falsification"] = _falsification(inst, bars, cfg, initial_cash,
                                              metrics, trials=falsify)
    out["gates"] = _run_gates(inst, cfg, curve, benchmark, trades, days,
                              out["rejections"])
    return out


def simulate_portfolio(queries: list, cfg: dict = None, days: int = 250,
                       initial_cash: float = 10_000_000,
                       weights: dict = None) -> dict:
    """여러 종목을 **하나의 포트폴리오로** 백테스트합니다 (슬리브 모델).

    종목별 백테스트를 따로 돌려 수익률을 나열하는 것과 다릅니다. 분산 효과는
    합쳐 봐야만 보이고, 그게 이 함수의 존재 이유입니다 (engine/portfolio.py).

    읽는 순서
        1) `diversification.ratio`  — 종목을 늘린 게 실제로 위험을 줄였는가.
           1에 가까우면 유니버스가 사실상 한 종목입니다.
        2) `excess_return_pct`      — 같은 비중으로 그냥 사서 들고 있는 것보다 나은가.
        3) `rejections`             — 슬리브별 자본이 쪼개져 있어 개별 백테스트보다
           예수금 부족 거부가 많이 납니다. 많으면 종목 수를 줄이거나 자본을 키우세요.

    시행 장부에는 **포트폴리오 전체가 1건** 으로 올라갑니다. 종목별로 N건을
    올리면 DSR 의 N 이 부풀어 문턱이 엉뚱하게 높아집니다.
    """
    cfg = {**store.DEFAULT_CONFIG, **(cfg or {})}
    cfg = {**cfg, "intraday_weight": 0.0, "use_news": False}

    queries = [q for q in (queries or []) if str(q).strip()]
    if len(queries) < 2:
        return {"ok": False, "error": "포트폴리오는 종목이 2개 이상이어야 합니다."}

    # -- 1단계: 데이터가 실제로 있는 종목만 추립니다 -----------------------
    #
    # 자본 배분 전에 걸러야 합니다. 배분 후에 거르면 실패한 종목 몫이 허공에
    # 남아서, "1/3씩 넣었다" 고 보고하면서 실제로는 2/3만 투입된 결과가 됩니다.
    resolved, bars_by_key, failures = {}, {}, []
    raw_weights = dict(weights or {})
    for query in queries:
        inst = _resolve_universe_item(query)
        if inst is None:
            failures.append(f"'{query}' 종목을 찾을 수 없습니다.")
            continue
        bars = feed.bars(inst, "day", count=max(days + 60, 120))
        if bars is None or len(bars) < strategy.MIN_DAILY_BARS + 20:
            failures.append(f"{inst.name}: 일봉이 부족합니다.")
            continue
        resolved[inst.key] = inst
        bars_by_key[inst.key] = bars
        # 사용자가 준 비중은 입력 문자열 기준이므로 종목키로 옮겨 둡니다
        if query in raw_weights:
            raw_weights[inst.key] = raw_weights.pop(query)

    if len(resolved) < 2:
        return {"ok": False,
                "error": "합칠 수 있는 종목이 2개 미만입니다. "
                         + (" / ".join(failures) if failures else "")}

    # -- 2단계: 살아남은 종목에만 자본을 배분하고 돌립니다 ------------------
    weights = portfolio._normalize(raw_weights, list(resolved.keys()))
    runs = {}
    rejections = fills.RejectionLog()
    for key, inst in resolved.items():
        sleeve_cash = initial_cash * weights[key]
        if sleeve_cash <= 0:
            continue
        run = _run_backtest(inst, bars_by_key[key], cfg, sleeve_cash)
        runs[key] = {
            "curve": run["curve"], "trades": run["trades"],
            "benchmark": _buy_and_hold_curve(inst, bars_by_key[key],
                                             run["start_index"], sleeve_cash),
        }
        rejections.items.extend(run["rejections"].items)

    if len(runs) < 2:
        return {"ok": False,
                "error": "합칠 수 있는 종목이 2개 미만입니다. "
                         + (" / ".join(failures) if failures else "")}

    out = portfolio.combine(runs, weights, initial_cash,
                            periods_per_year=validation.TRADING_DAYS)
    if not out.get("ok"):
        return out

    out["instruments"] = {k: v.to_dict() for k, v in resolved.items()}
    out["rejections"] = rejections.summary()
    out["warnings"] = failures
    out["execution"] = {
        "model": "sleeve",
        "note": "종목별로 자본을 미리 쪼개 독립 백테스트한 뒤 합칩니다. "
                "리밸런싱·슬리브 간 자본 공유·증거금 상계는 없습니다 "
                "(셋 다 결과를 보수적인 쪽으로 밉니다).",
    }
    all_trades = [t for r in runs.values() for t in (r["trades"] or [])]
    out["gates"] = _run_gates(
        next(iter(resolved.values())), cfg, out["curve"], out["benchmark_curve"],
        all_trades, days, out["rejections"],
        trial_label=f"portfolio:{'+'.join(sorted(runs.keys()))}")
    return out


def _run_gates(inst: Instrument, cfg: dict, curve: list, benchmark: list,
               trades: list, days: int, rejections: dict = None,
               trial_label: str = "") -> dict:
    """★ 백테스트 결과에 **검증 하네스** 를 붙입니다 (설계도 §12).

    총수익률·샤프만 보고 판단하지 않게 만드는 것이 목적입니다. 다섯 가지를 답니다.

      registry  이 설정이 **몇 번째 시행인가**. 장부에 자동 등록됩니다.
                DSR 의 N 은 사람이 세면 반드시 과소평가됩니다 — 실패한 시행을
                안 세기 때문입니다. 그래서 실행 자체가 등록이 되게 했습니다.
      cpcv      단일 경로가 아니라 **경로 분포**. 중앙값보다 최악 경로가 중요합니다.
      lottery   같은 횟수로 아무렇게나 매매한 것보다 나은가.
      vol_audit 변동성 스케일링이 두 번 적용되고 있지 않은가 (기울기 ≈ −1 이어야 정상).
      overfit   시행 수를 반영한 요구 t통계와 최소 백테스트 길이.
      execution 신호가 **체결되지 못하고 사라진 비율**. 아래 설명 참고.

    **실패해도 백테스트를 막지 않습니다.** 숫자를 함께 보여줄 뿐입니다 —
    막으면 사람이 게이트를 끄는 쪽으로 움직입니다. 대신 `verdict` 한 줄로
    "이 결과를 믿어도 되는가" 를 명시합니다.
    """
    from engine import cpcv as _cpcv, lottery as _lottery, overfit as _of
    from engine.registry import TrialRegistry, TrialSpec

    rets = validation.to_returns(curve)
    bench = validation.to_returns(benchmark)
    n = len(rets)
    if n < 60:
        return {"ok": False, "reason": f"게이트를 돌리기에 표본이 부족합니다 ({n}일)."}

    out: dict = {"ok": True, "n_obs": n}

    # -- 1) 시행 장부 -----------------------------------------------------
    try:
        reg = TrialRegistry()
        spec = TrialSpec(
            family=(f"portfolio:{inst.asset_class}" if trial_label
                    else f"autotrade:{inst.asset_class}"),
            config={k: cfg.get(k) for k in sorted((
                "entry_score", "exit_score", "atr_stop_mult", "stop_loss_pct",
                "take_profit_pct", "trailing_stop_pct", "max_hold_days",
                "risk_per_trade_pct", "position_pct", "algo_mode", "ml_mode",
                "protect_enabled", "slippage_bps")) if k in cfg},
            data_fingerprint=f"{trial_label or inst.key}:{days}",
            label=trial_label or inst.key)
        tid = reg.register(spec)
        reg.log_result(tid, returns=rets.tolist())
        eff = reg.effective_n(spec.family)
        out["registry"] = {"trial_id": tid, "m_trials": reg.n_trials,
                           "effective_n": eff["n_hat"], "method": eff["method"]}
        dsr = reg.deflated_sharpe(tid, family=spec.family)
        out["dsr"] = {k: dsr.get(k) for k in
                      ("dsr", "observed_sr", "expected_max_sr_under_null",
                       "n_trials", "significant_at_95")}
        if out["dsr"].get("dsr") is None:
            # 시행이 2개 미만이면 "여러 번 시도해서 최고를 골랐다" 는 보정 자체가
            # 성립하지 않습니다. 값이 없는 게 정상이고, **아직 안전하다는 뜻이
            # 아닙니다** — 설정을 바꿔 다시 돌릴수록 이 장부의 N 이 늘고
            # 문턱이 올라갑니다.
            out["dsr"]["reason"] = (
                f"이 전략군의 시행이 {reg.n_trials}건뿐이라 DSR 을 계산하지 "
                f"않았습니다. 설정을 바꿔 재실행할 때마다 장부에 쌓이고, "
                f"2건부터 보정이 시작됩니다.")
    except Exception as exc:
        out["registry"] = {"error": repr(exc)[:120]}

    # -- 2) CPCV 경로 분포 -------------------------------------------------
    try:
        plan = _cpcv.combinatorial_purged_cv(
            n, n_groups=6, k=2, horizon=int(cfg.get("max_hold_days", 5) or 5),
            warmup=strategy.MIN_DAILY_BARS, embargo_pct=0.02)
        if plan is not None:
            srs = []
            for sp in plan.splits:
                seg = rets[sp.test_idx]
                seg = seg[np.isfinite(seg)]
                if len(seg) > 8:
                    srs.append(validation.sharpe_ratio(seg))
            if len(srs) >= 3:
                arr = np.asarray(srs, dtype=float)
                out["cpcv"] = {
                    "n_folds": plan.n_folds, "n_paths": plan.n_paths,
                    "sharpe_median": round(float(np.median(arr)), 4),
                    "sharpe_iqr": [round(float(np.quantile(arr, .25)), 4),
                                   round(float(np.quantile(arr, .75)), 4)],
                    "sharpe_worst": round(float(arr.min()), 4),
                    "share_positive": round(float(np.mean(arr > 0)), 3),
                }
    except Exception as exc:
        out["cpcv"] = {"error": repr(exc)[:120]}

    # -- 3) Lottery + 4) 이중 vol 스케일링 감사 ----------------------------
    exposure = np.array([float(p.get("exposure") or 0.0) for p in curve],
                        dtype=float)
    try:
        if np.any(np.abs(exposure) > 1e-9) and len(bench) >= n - 1:
            asset = bench[:len(exposure)]
            lot = _lottery.lottery_benchmark(
                exposure[:len(asset)], asset,
                cost_bps=float(cfg.get("slippage_bps", 5)) * 2 + 20,
                n_random=400)
            out["lottery"] = {k: lot.get(k) for k in
                              ("percentile", "passed", "observed", "null_mean",
                               "p_value")}
    except Exception as exc:
        out["lottery"] = {"error": repr(exc)[:120]}

    try:
        rv = np.array([float(np.std(rets[max(i - 20, 0):i], ddof=1))
                       if i >= 21 else np.nan for i in range(len(rets))])
        out["vol_audit"] = validation.audit_vol_scaling(exposure[1:len(rv) + 1], rv)
    except Exception as exc:
        out["vol_audit"] = {"error": repr(exc)[:120]}

    # -- 5) 다중검정 -------------------------------------------------------
    try:
        n_tr = int(out.get("registry", {}).get("effective_n") or 1)
        sr = float(metrics_sr(curve))
        hc = _of.haircut_sharpe(sr, n_trials=max(n_tr, 1), n_obs=n)
        out["overfit"] = {
            "required_tstat": hc.get("required_tstat"),
            "t_stat": hc.get("t_stat"),
            "passes_required_t": hc.get("passes_required_t"),
            "haircut_pct": hc.get("haircut_pct"),
            "min_backtest_years": _of.min_backtest_length(max(n_tr, 2))["min_years"],
            "have_years": round(n / 252.0, 2),
        }
    except Exception as exc:
        out["overfit"] = {"error": repr(exc)[:120]}

    # -- 6) 체결 실패율 ----------------------------------------------------
    #
    # 거부가 많다는 것은 "이 백테스트가 보여준 매매를 실제로는 낼 수 없었다" 는
    # 뜻입니다. 성과 숫자보다 먼저 봐야 하는 값인데, 거부를 기록하지 않으면
    # 매매 횟수가 조용히 줄어들 뿐이라 좋은 결과처럼 보입니다.
    rej = rejections or {}
    if rej.get("total"):
        attempted = len(trades) + rej["total"]
        out["execution"] = {
            "rejected": rej["total"],
            "attempted": attempted,
            "reject_rate": round(rej["total"] / attempted, 4) if attempted else 0.0,
            "by_reason": rej.get("by_reason", {}),
        }

    out["verdict"] = _gate_verdict(out, len(trades))
    return out


def metrics_sr(curve: list) -> float:
    return validation.sharpe_ratio(validation.to_returns(curve))


def _gate_verdict(g: dict, n_trades: int) -> str:
    """게이트 결과를 한 줄로. **가장 나쁜 신호를 앞에 씁니다.**"""
    bad = []
    if n_trades < 30:
        bad.append(f"거래 {n_trades}회 — 30회 미만이면 샤프 표준오차가 "
                   f"어떤 엣지보다 큽니다. 통계적으로 판단 불가")
    c = g.get("cpcv") or {}
    if c.get("share_positive") is not None and c["share_positive"] < 0.6:
        bad.append(f"CPCV 경로의 {c['share_positive']:.0%} 만 양수 "
                   f"(최악 경로 샤프 {c.get('sharpe_worst')})")
    lot = g.get("lottery") or {}
    if lot.get("passed") is False:
        bad.append(f"무작위 매매 대비 {lot.get('percentile')} 백분위 — "
                   f"같은 횟수로 아무렇게나 매매한 것과 구분되지 않습니다")
    va = g.get("vol_audit") or {}
    if va.get("passed") is False:
        bad.append(f"이중 변동성 스케일링 의심 (기울기 {va.get('slope')})")
    of = g.get("overfit") or {}
    if of.get("passes_required_t") is False:
        bad.append(f"t={of.get('t_stat')} < 요구 {of.get('required_tstat')} "
                   f"(시행 {g.get('registry', {}).get('effective_n')}회 반영)")
    if of.get("min_backtest_years") and of.get("have_years") and \
            of["have_years"] < of["min_backtest_years"]:
        bad.append(f"데이터 {of['have_years']}년 < 이 시행 수를 지탱하는 "
                   f"최소 {of['min_backtest_years']}년")
    ex = g.get("execution") or {}
    if ex.get("reject_rate", 0) >= 0.10:
        top = next(iter(ex.get("by_reason", {}).values()), {}).get("label", "")
        bad.append(f"신호의 {ex['reject_rate']:.0%}({ex['rejected']}건)가 "
                   f"체결되지 못했습니다{f' — 주로 {top}' if top else ''}. "
                   f"이 성과는 낼 수 없었던 주문을 뺀 나머지입니다")

    if not bad:
        return "게이트 통과 — 이 결과는 통상적인 과최적화 검사를 견딥니다."
    return "⚠ " + " · ".join(bad)


def _buy_and_hold_curve(inst: Instrument, bars, start_index: int,
                        initial_cash: float) -> list:
    """벤치마크 자산곡선 — 첫날 살 수 있는 만큼 사서 그대로 들고 있기.

    수수료를 한 번 물립니다. 매매 전략은 수수료를 계속 내므로, 벤치마크만
    공짜로 두면 비교가 전략에 불리하게 기울어집니다.
    """
    closes = bars["close"].to_numpy(dtype=float)[start_index:]
    dates = [str(d)[:10] for d in bars.index[start_index:]]
    if len(closes) == 0 or closes[0] <= 0:
        return []
    unit = closes[0] * inst.multiplier
    quantity = math.floor(initial_cash / unit) if unit > 0 else 0
    if quantity <= 0:
        return []
    fee, _ = inst.costs("buy", closes[0] * quantity * inst.multiplier)
    cash_left = initial_cash - closes[0] * quantity * inst.multiplier - fee
    return [{"date": d, "equity": round(cash_left + c * quantity * inst.multiplier)}
            for d, c in zip(dates, closes)]


def _synthetic_bars(bars, rng, block: int = 5):
    """예측 가능한 **순서만** 파괴한 가짜 일봉.

    일간 수익률을 블록 부트스트랩으로 다시 이어 붙이고, 각 봉의 고가·저가 모양은
    그대로 유지합니다(같은 배율로 옮김). 변동성 군집과 봉 형태는 남고 "이 다음에
    무엇이 오는가"만 사라지므로, 전략이 먹고 사는 구조가 정확히 제거됩니다.
    """
    close = bars["close"].to_numpy(dtype=float)
    if len(close) < 3 or close[0] <= 0:
        return None
    rets = np.diff(close) / close[:-1]
    boot = validation.block_bootstrap(rets, block=block, rng=rng)
    new_close = np.empty(len(close), dtype=float)
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.cumprod(1 + boot)

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(close > 0, new_close / close, 1.0)
    out = bars.copy()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col].to_numpy(dtype=float) * scale
    return out


def _falsification(inst: Instrument, bars, cfg: dict, initial_cash: float,
                   observed: dict, trials: int = 10) -> dict:
    """같은 전략을 **우위가 없는 데이터**에 통과시켜 봅니다.

    여기서도 비슷한 성적이 나오면, 좋은 숫자를 만든 것은 전략이 아니라 절차입니다.
    이 검사를 통과하지 못한 백테스트로 실매매를 켜면 안 됩니다.
    """
    trials = max(min(int(trials), 50), 1)
    rng = np.random.default_rng(20260806)      # 고정 시드 — 같은 입력이면 같은 판정
    null_returns, null_sharpes = [], []
    for _ in range(trials):
        fake = _synthetic_bars(bars, rng)
        if fake is None:
            continue
        try:
            run = _run_backtest(inst, fake, cfg, initial_cash)
        except Exception:
            continue          # 가짜 가격에서 지표가 깨질 수 있습니다 — 그 시행만 버립니다
        if not run["curve"]:
            continue
        null_returns.append((run["curve"][-1]["equity"] - initial_cash) / initial_cash * 100)
        null_sharpes.append(validation.sharpe_ratio(validation.to_returns(run["curve"])))

    audit = validation.falsification_audit(
        float(observed.get("total_return_pct") or 0.0), null_returns,
        label="총수익률")
    audit["sharpe"] = validation.falsification_audit(
        float(observed.get("sharpe") or 0.0), null_sharpes, label="샤프")
    audit["verdict"] = (
        "가짜 데이터와 구분되지 않습니다 — 이 성과는 전략의 우위로 보기 어렵습니다."
        if audit.get("passed") is False else
        "가짜 데이터에서는 재현되지 않았습니다 — 전략이 실제 구조를 쓰고 있습니다."
        if audit.get("passed") else
        "판정할 표본이 부족합니다.")
    return audit


def _run_backtest(inst: Instrument, bars, cfg: dict, initial_cash: float) -> dict:
    """백테스트 루프 본체 (실데이터·합성데이터가 같은 경로를 쓰게 분리했습니다).

    ★ 봉 하나 안에서의 순서가 이 함수의 전부입니다 (engine/fills.py 참고).

        ① 지난 봉에서 낸 주문을 **이 봉 시가** 에 체결        ← 한 봉 지연
        ② 보유 중이면 이 봉의 고가/저가로 **손절·익절 판정**   ← 봉 안 체결
        ③ 종가로 신호를 계산하고 **다음 봉에 낼 주문** 을 예약
        ④ 종가로 평가금액 기록

    ①과 ③을 갈라놓은 것이 핵심입니다. 예전 구현은 종가로 판단해서 **그 종가에**
    체결했습니다. 실전에서 불가능한 가정이고, 벡터라이즈 백테스터가 대개 이
    형태라 "백테스트만 좋은" 결과의 단골 원인입니다.

    ②가 없으면 장중에 손절선을 뚫었다가 회복한 봉이 전부 무손실로 기록됩니다.

    바뀐 뒤 성과는 **거의 항상 나빠집니다.** 그게 맞습니다 — 없어진 만큼이
    원래부터 받을 수 없던 몫입니다.
    """
    slippage = float(cfg.get("slippage_bps", 5)) / 10_000.0
    cash = float(initial_cash)
    position = None          # {side, qty, entry, stop, target, peak, opened_at}
    pending = None           # 다음 봉 시가에 체결할 주문
    trades, curve = [], []
    rejections = fills.RejectionLog()
    start_index = max(strategy.MIN_DAILY_BARS, 30)

    def close_position(pos: dict, fill, when, reason: str) -> None:
        """청산 회계 한 곳. 세 경로(보호주문·신호·구간끝)가 모두 여기를 지납니다.

        손익에 **진입 비용까지** 포함시킵니다. 진입 수수료는 진입 시점에 이미
        현금에서 빠지므로 자산 곡선에는 반영돼 있는데, 매매 기록에서 빼먹으면
        `sum(매매 손익) ≠ 최종자산` 이 됩니다. 그 차이는 승률·손익비를 전부
        조금씩 좋은 쪽으로 밀어냅니다 — 매매가 잦을수록 커집니다.
        """
        nonlocal cash, position
        direction = 1 if pos["side"] == strategy.LONG else -1
        notional = fill.price * pos["qty"] * inst.multiplier
        gross = (fill.price - pos["entry"]) * pos["qty"] * inst.multiplier * direction
        # when= 을 넘기는 것이 중요합니다. 빼면 2020년 매매에 오늘 세율이 붙어
        # 과거 구간이 실제보다 비관적으로 평가됩니다 (instruments.costs 주석).
        fee, tax = inst.costs("sell" if direction > 0 else "buy", notional, when=when)
        # 차입비용은 보유 중 매 봉 현금에서 이미 빠졌습니다. 손익에도 반영해야
        # `sum(매매 손익) = 자산 증감` 이 유지됩니다.
        pnl = gross - fee - tax - pos["entry_cost"] - pos["borrow_cost"]
        cash += pos["margin"] + gross - fee - tax
        trades.append({
            "entry_at": pos["opened_at"], "exit_at": str(when)[:10],
            "side": pos["side"], "quantity": pos["qty"],
            "entry": round(pos["entry"], 4), "exit": round(fill.price, 4),
            "pnl": round(pnl),
            "pnl_pct": round(pnl / pos["margin"] * 100, 2) if pos["margin"] else 0,
            "reason": reason, "fill": fill.reason,
            "fees": round(fee + pos["entry_fee"]), "taxes": round(tax + pos["entry_tax"]),
            "borrow": round(pos["borrow_cost"]),
        })
        position = None

    for i in range(start_index, len(bars)):
        row = bars.iloc[i]
        bar_date = bars.index[i]
        day = str(bar_date)[:10]
        price = float(row["close"])
        prev_close = float(bars.iloc[i - 1]["close"]) if i else 0.0

        # -- ① 지난 봉의 주문을 이 봉 시가에 체결 -------------------------
        if pending:
            order, pending = pending, None
            if order["action"] == "exit" and position:
                fill, why = fills.market_fill(
                    inst, row, fills.SELL if position["side"] == strategy.LONG
                    else fills.BUY, slippage, prev_close, i, day)
                if fill:
                    close_position(position, fill, bar_date, order["reason"])
                else:
                    rejections.add(i, day, why, "exit", order["reason"])

            elif order["action"] == "enter" and not position:
                side = fills.BUY if order["side"] == strategy.LONG else fills.SELL
                # 공매도 금지 구간이었는지 **체결 시점 날짜로** 확인합니다.
                # 이걸 안 보면 2020-03~2021-05, 2023-11~2025-03 처럼 실제로는
                # 숏을 낼 수 없었던 구간의 수익이 그대로 남습니다 —
                # 하필 하락장이라 백테스트가 가장 좋아 보이는 구간입니다.
                short_ok = ({"allowed": True} if order["side"] == strategy.LONG
                            else holding.short_entry_allowed(inst, bar_date, cfg))
                fill, why = (None, holding.SHORT_BANNED) if not short_ok["allowed"] \
                    else fills.market_fill(inst, row, side, slippage,
                                           prev_close, i, day)
                if why == holding.SHORT_BANNED:
                    rejections.add(i, day, holding.SHORT_BANNED, order["side"],
                                   short_ok["reason"], order["qty"])
                elif not fill:
                    rejections.add(i, day, why, order["side"],
                                   "가격제한폭", order["qty"])
                else:
                    margin = inst.margin_required(fill.price, order["qty"], side)
                    # 진입도 side 를 그대로 넘깁니다. 숏 진입은 '매도' 라서
                    # 증권거래세가 붙습니다 — "매수" 로 고정하면 그 세금이 빠집니다.
                    fee, tax = inst.costs(
                        side, fill.price * order["qty"] * inst.multiplier,
                        when=bar_date)
                    if margin + fee + tax > cash:
                        # 예전에는 여기서 조용히 건너뛰었습니다. 신호가 증발해도
                        # 매매 횟수가 줄 뿐이라 아무도 몰랐습니다.
                        rejections.add(
                            i, day, fills.INSUFFICIENT_CASH, order["side"],
                            f"필요 {margin + fee + tax:,.0f} > 보유 {cash:,.0f}",
                            order["qty"])
                    else:
                        cash -= margin + fee + tax
                        position = {
                            "side": order["side"], "qty": order["qty"],
                            "entry": fill.price, "stop": order["stop"],
                            "target": order["target"], "peak": fill.price,
                            "margin": margin, "opened_at": day,
                            "entry_fee": fee, "entry_tax": tax,
                            "entry_cost": fee + tax,
                            "borrow_cost": 0.0, "last_accrued": bar_date,
                        }

        # -- ② 봉 안에서의 손절·익절 (같은 봉에 진입했어도 판정합니다) -----
        if position:
            protective = fills.protective_fill(
                inst, row, position["side"], position["stop"],
                position["target"], slippage, prev_close, i, day)
            if protective:
                label = {fills.STOP: "손절 도달",
                         fills.TARGET: "목표가 도달",
                         fills.GAP_THROUGH_STOP: "갭 하락으로 손절선 관통 — 시가 체결",
                         fills.GAP_THROUGH_TARGET: "갭 상승으로 목표가 관통 — 시가 체결",
                         }.get(protective.reason, protective.reason)
                close_position(position, protective, bar_date,
                               f"{label} ({protective.price:g})")

        # -- ③ 종가로 판단 → 다음 봉 주문 예약 ------------------------------
        window = bars.iloc[: i + 1]
        quote = {"price": price, "price_krw": price, "age_sec": 0.0,
                 "market_open": True}
        sig = strategy.evaluate(inst, cfg, bars_daily=window, quote=quote,
                                allow_fetch=False)

        if sig.ok and position:
            position["peak"] = (max(position["peak"], price)
                                if position["side"] == strategy.LONG
                                else min(position["peak"], price))
            fake = type("P", (), {
                "side": position["side"], "avg_price": position["entry"],
                "current_price": price, "quantity": position["qty"],
            })()
            state = {"entry_price": position["entry"], "stop_price": position["stop"],
                     "target_price": position["target"], "peak_price": position["peak"],
                     "opened_at": position["opened_at"]}
            # 손절·익절은 ②가 이미 봉 안에서 처리했습니다. 여기서 보는 것은
            # 신호 반전·보유기간 초과처럼 **종가로만 알 수 있는** 사유입니다.
            decision = strategy.check_exit(inst, fake, sig, cfg, state,
                                           now=_as_datetime(bar_date))
            if decision.should_exit:
                pending = {"action": "exit", "reason": decision.reason}

        elif sig.ok and not position and sig.direction != strategy.FLAT:
            account = {"total_value": cash, "available_cash": cash}
            plan = strategy.plan_entry(inst, sig, cfg, account)
            if plan.ok and plan.quantity > 0:
                # 손절·목표선은 **판단 시점(종가)** 기준으로 잡힌 값을 그대로
                # 들고 갑니다. 체결가(다음 봉 시가) 기준으로 다시 잡지 않는 것은
                # 라이브가 그렇게 동작하기 때문입니다 — 신호를 보고 손절선을
                # 정한 뒤 주문을 냅니다. 여기서만 재계산하면 백테스트와 실매매가
                # 갈라집니다.
                pending = {"action": "enter", "side": sig.direction,
                           "qty": plan.quantity, "stop": plan.stop_price,
                           "target": plan.target_price}
            elif plan.ok:
                rejections.add(i, day, fills.ZERO_QUANTITY, sig.direction,
                               "계산된 주문 수량이 0주")

        # -- ④ 숏 차입비용 누적 → 종가로 평가금액 기록 -----------------------
        #
        # 달력일 기준이라 주말 이자도 붙습니다. 영업일로 세면 연 2.5% 짜리
        # 이자의 28%가 조용히 사라집니다 (금→월은 3일치).
        if position and position["side"] == strategy.SHORT:
            cost = holding.accrue(
                inst, price * position["qty"] * inst.multiplier,
                position["last_accrued"], bar_date, cfg)
            if cost:
                cash -= cost
                position["borrow_cost"] += cost
            position["last_accrued"] = bar_date

        equity = cash
        exposure = 0.0
        if position:
            direction = 1 if position["side"] == strategy.LONG else -1
            equity += position["margin"] + (price - position["entry"]) * \
                position["qty"] * inst.multiplier * direction
            # 노출 = 포지션 명목가 / 자산. 게이트(이중 vol 스케일링 감사·
            # lottery 매칭)가 이 계열을 씁니다. 없으면 "얼마나 들고 있었나" 를
            # 사후에 복원할 수 없어 두 검정 모두 불가능해집니다.
            if equity > 0:
                notional = price * position["qty"] * inst.multiplier
                exposure = direction * notional / equity
        curve.append({"date": str(bar_date)[:10], "equity": round(equity),
                      "exposure": round(exposure, 4)})

    # -- 구간 끝에 남은 포지션을 청산해서 매매 통계에 넣습니다 --------------
    #
    # 안 하면 자산 곡선에는 미실현 손익이 들어가는데 매매 통계에는 안 들어가서,
    # 승률·매매횟수·손익비가 전부 어긋납니다. 오래 들고 가는 전략일수록 심하고,
    # "수익률은 +30%인데 매매 0회" 같은 결과가 나옵니다.
    if position and len(bars) > start_index:
        last = bars.iloc[-1]
        exit_side = fills.SELL if position["side"] == strategy.LONG else fills.BUY
        raw = float(last["close"]) * ((1 - slippage) if exit_side == fills.SELL
                                      else (1 + slippage))
        close_position(position,
                       fills.Fill(fills.snap_adverse(inst, raw, exit_side),
                                  "close", len(bars) - 1, str(bars.index[-1])[:10]),
                       bars.index[-1], "구간 종료 — 미청산 포지션 강제 정산")
        # 마지막 평가금액도 정산 비용을 반영해야 곡선과 매매 통계가 어긋나지
        # 않습니다 (곡선은 세금 없이 평가, 매매는 세금 차감 — 이 둘이 다르면
        # `final_equity` 와 매매 손익 합계가 맞지 않습니다).
        if curve:
            curve[-1]["equity"] = round(cash)

    return {"curve": curve, "trades": trades, "start_index": start_index,
            "rejections": rejections}


def _as_datetime(value) -> datetime:
    try:
        return value.to_pydatetime()
    except AttributeError:
        return datetime.now()


def _metrics(curve: list, trades: list, initial_cash: float) -> dict:
    """수익률·최대낙폭·승률. 표본이 적으면 그대로 적게 나오는 게 정직합니다.

    주의: 여기 숫자만 보면 안 됩니다. 총수익률은 상승장이면 전략과 무관하게
    올라갑니다. 시장 대비 초과분은 validation.active_metrics 가 붙여줍니다.
    """
    if not curve:
        return {"total_return_pct": 0.0, "max_drawdown_pct": 0.0, "win_rate": 0.0,
                "trade_count": 0, "final_equity": initial_cash, "profit_factor": 0.0}

    final = curve[-1]["equity"]
    peak, max_dd = curve[0]["equity"], 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        if peak > 0:
            max_dd = min(max_dd, (point["equity"] - peak) / peak * 100)

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))

    return {
        "total_return_pct": round((final - initial_cash) / initial_cash * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "trade_count": len(trades),
        "final_equity": final,
        "avg_win": round(gross_win / len(wins)) if wins else 0,
        "avg_loss": round(-gross_loss / len(losses)) if losses else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else 0.0,
    }
