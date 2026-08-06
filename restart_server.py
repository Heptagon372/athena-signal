"""
서버 재시작 (Safe Restart)
--------------------------
코드를 고친 뒤 백엔드를 껐다 켜는 일을 사람이 손으로 하지 않게 합니다.
**실계좌 자동매매가 도는 서버**라, 그냥 죽였다 켜는 것이 아니라 순서를 지킵니다.

    1. 코드가 실제로 임포트되는지 먼저 확인한다  ← 여기서 걸리면 서버를 건드리지 않음
    2. 지금 체결 대기 중인 주문이 있는지 본다
    3. 돌던 서버를 정중히 세운다 (안 되면 강제)
    4. 새 서버를 띄우고 **응답할 때까지 기다린다**
    5. 안 뜨면 크게 알린다 (조용히 죽어 있는 것이 가장 위험합니다)

왜 1번이 먼저인가
    문법 오류 하나로 서버가 안 뜨면, 고치는 몇 분 동안 손절 검사가 통째로
    멈춥니다. 멀쩡히 돌던 서버는 새 코드가 최소한 임포트는 된다는 것을
    확인한 뒤에 내립니다.

왜 --reload 를 쓰지 않는가
    uvicorn 의 자동 리로드는 파일을 저장할 때마다 프로세스를 갈아칩니다.
    주문을 낸 직후에 갈리면 접수는 됐는데 원장에 기록이 없는 주문이 생깁니다.
    실계좌에서는 "언제 재시작할지"를 사람이 정하는 편이 안전합니다.

사용법
    python restart_server.py              # 확인 후 재시작
    python restart_server.py --check      # 재시작 없이 점검만
    python restart_server.py --force      # 미체결 주문이 있어도 진행
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
HEALTH_PATH = "/api/status"
START_TIMEOUT = 60           # 새 서버가 응답하기까지 기다리는 최대 초
STOP_TIMEOUT = 12            # 정중한 종료를 기다리는 최대 초

sys.stdout.reconfigure(encoding="utf-8")


def say(mark: str, message: str):
    print(f"  {mark} {message}", flush=True)


# ---------------------------------------------------------------------------
# 1단계: 코드 점검 — 서버를 내리기 전에
# ---------------------------------------------------------------------------

def code_imports() -> tuple[bool, str]:
    """새 코드가 임포트되는지 **별도 프로세스에서** 확인합니다.

    지금 프로세스에서 import 하면 이미 로드된 모듈이 섞여 오류를 놓칩니다.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import api; import engine.autotrade"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace")
    if probe.returncode == 0:
        return True, ""
    detail = (probe.stderr or probe.stdout or "").strip().splitlines()
    return False, "\n      ".join(detail[-6:]) if detail else "원인 불명"


# ---------------------------------------------------------------------------
# 2단계: 지금 재시작해도 되는 상황인가
# ---------------------------------------------------------------------------

def pending_orders() -> list[dict]:
    """접수됐는데 결말이 안 난 주문. 있으면 재시작 타이밍을 다시 생각해야 합니다.

    (재시작 자체가 주문을 취소하지는 않습니다. 다만 다음 회전의 정산 패스가
     돌기 전까지는 아무도 그 주문을 지켜보지 않는 공백이 생깁니다.)
    """
    try:
        sys.path.insert(0, str(ROOT))
        from storage import autotrade as store
        out = []
        for user_id in store.configured_users():
            cfg = store.get_config(user_id)
            for row in store.open_orders(user_id, cfg.get("mode", "paper")):
                out.append({"user": user_id, "symbol": row.get("symbol"),
                            "name": row.get("name"), "status": row.get("status")})
        return out
    except Exception as exc:
        say("!", f"미체결 주문을 확인하지 못했습니다 ({type(exc).__name__}) — 그냥 진행합니다.")
        return []


# ---------------------------------------------------------------------------
# 3단계: 세우기
# ---------------------------------------------------------------------------

def _pids(script: str) -> list[int]:
    res = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    return [int(x) for x in (res.stdout or "").split() if x.strip().isdigit()]


def running_servers() -> list[int]:
    """지금 떠 있는 uvicorn 프로세스 PID 목록."""
    return _pids(
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'uvicorn' } | "
        "Select-Object -ExpandProperty ProcessId")


def orphan_windows() -> list[int]:
    """안의 서버는 죽었는데 남아 있는 콘솔 창.

    창은 `cmd /k` 로 띄웁니다 — 서버가 스스로 죽었을 때 오류 메시지를 읽을 수
    있어야 하기 때문입니다. 대신 우리가 의도적으로 재시작할 때는 빈 창이
    쌓이지 않게 여기서 걷어냅니다. (프론트엔드 창과 start.bat 런처는
    명령줄에 uvicorn 이 없으므로 걸리지 않습니다)
    """
    alive = set(running_servers())
    windows = _pids(
        "Get-CimInstance Win32_Process -Filter \"Name = 'cmd.exe'\" | "
        "Where-Object { $_.CommandLine -match 'uvicorn' } | "
        "Select-Object -ExpandProperty ProcessId")
    if not windows:
        return []
    # 살아 있는 서버의 부모 창은 남겨 둡니다
    parents = set(_pids(
        "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'uvicorn' } | "
        "Select-Object -ExpandProperty ParentProcessId")) if alive else set()
    return [pid for pid in windows if pid not in parents]


def stop_servers() -> int:
    """돌던 서버를 세웁니다. 정중히 → 안 되면 강제."""
    pids = running_servers()
    if not pids:
        say("·", "돌고 있는 서버가 없습니다.")
        return 0

    for pid in pids:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Id {pid} -ErrorAction SilentlyContinue"],
                       capture_output=True)

    deadline = time.time() + STOP_TIMEOUT
    while time.time() < deadline:
        if not running_servers():
            say("✓", f"서버 {len(pids)}개를 세웠습니다 (PID {', '.join(map(str, pids))})")
            return len(pids)
        time.sleep(1)

    stubborn = running_servers()
    for pid in stubborn:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                       capture_output=True)
    time.sleep(2)
    say("✓", f"서버를 강제로 세웠습니다 (PID {', '.join(map(str, stubborn))})")
    return len(pids)


def sweep_orphan_windows():
    """서버가 빠져나간 빈 콘솔 창을 닫습니다 (재시작마다 쌓이는 것 방지)."""
    orphans = orphan_windows()
    for pid in orphans:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                       capture_output=True)
    if orphans:
        say("✓", f"빈 서버 창 {len(orphans)}개를 닫았습니다 "
                 f"(PID {', '.join(map(str, orphans))})")


# ---------------------------------------------------------------------------
# 4단계: 띄우고, 응답할 때까지 확인
# ---------------------------------------------------------------------------

def start_server():
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Start-Process cmd -ArgumentList '/k','chcp 65001 > nul && "
         f"python -m uvicorn api:app --port {PORT} --no-use-colors' "
         f"-WorkingDirectory '{ROOT}' -WindowStyle Minimized"],
        capture_output=True)


def wait_healthy(timeout: int = START_TIMEOUT) -> tuple[bool, str]:
    """서버가 실제로 응답할 때까지 기다립니다.

    '프로세스가 떴다'와 '서비스가 산다'는 다릅니다. 포트만 보고 성공이라고
    하면, 기동 중 예외로 죽는 경우를 못 잡습니다.
    """
    deadline = time.time() + timeout
    last = "응답 없음"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE + HEALTH_PATH, timeout=4) as res:
                if res.status == 200:
                    try:
                        body = json.loads(res.read().decode() or "{}")
                    except json.JSONDecodeError:
                        body = {}
                    return True, (body.get("version") or body.get("status") or "정상")
                last = f"HTTP {res.status}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = str(getattr(exc, "reason", exc))[:60]
        time.sleep(1.5)
    return False, last


# ---------------------------------------------------------------------------

def main() -> int:
    force = "--force" in sys.argv
    check_only = "--check" in sys.argv

    print("=" * 62)
    print("  아테나 시그널 — 서버 재시작")
    print("=" * 62)

    # 1) 코드부터
    ok, detail = code_imports()
    if not ok:
        say("✗", "새 코드가 임포트되지 않습니다. **서버를 건드리지 않았습니다.**")
        print(f"      {detail}")
        say("·", "고친 뒤 다시 실행하세요. 지금 돌던 서버는 그대로 살아 있습니다.")
        return 1
    say("✓", "코드 임포트 확인")

    # 2) 지금 재시작해도 되는가
    waiting = pending_orders()
    if waiting:
        names = ", ".join(f"{o['name'] or o['symbol']}({o['status']})" for o in waiting[:5])
        say("!", f"체결 대기 중인 주문 {len(waiting)}건: {names}")
        if not force:
            say("·", "재시작하면 다음 회전까지 이 주문들을 아무도 지켜보지 않습니다.")
            say("·", "그래도 진행하려면: python restart_server.py --force")
            return 2
        say("!", "--force 로 진행합니다.")
    else:
        say("✓", "체결 대기 중인 주문 없음")

    alive, orphans = running_servers(), orphan_windows()
    say("·", f"서버 {len(alive)}개 실행 중"
             + (f" (PID {', '.join(map(str, alive))})" if alive else "")
             + (f" · 빈 창 {len(orphans)}개" if orphans else ""))
    if len(alive) > 1:
        say("!", "서버가 중복 실행 중입니다 — 재시작하면 하나로 정리됩니다.")

    if check_only:
        if orphans:
            say("·", "--clean 을 붙이면 빈 창만 닫습니다 (서버는 그대로).")
        say("·", "--check 이므로 재시작하지 않고 끝냅니다.")
        return 0

    if "--clean" in sys.argv:
        sweep_orphan_windows()
        say("·", "빈 창만 정리했습니다 (서버는 건드리지 않았습니다).")
        return 0

    # 3) 세우고 — 빈 창까지 정리해서 재시작마다 창이 쌓이지 않게 합니다
    stop_servers()
    sweep_orphan_windows()

    # 4) 띄우고 확인
    start_server()
    say("·", f"기동 중… 최대 {START_TIMEOUT}초 기다립니다.")
    healthy, detail = wait_healthy()
    if not healthy:
        say("✗", f"서버가 응답하지 않습니다 ({detail})")
        say("✗", "**지금 자동매매가 멈춰 있습니다.** 서버 창의 오류를 확인하세요.")
        return 1

    pids = running_servers()
    say("✓", f"서버 정상 ({detail}) · PID {', '.join(map(str, pids)) or '?'}")
    if len(pids) > 1:
        say("!", f"서버가 {len(pids)}개 떠 있습니다 — 중복 실행이면 하나만 남기세요.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
