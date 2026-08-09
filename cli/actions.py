"""메뉴 항목이 실제로 하는 일.

예전 .bat 파일들이 하던 일을 그대로 옮겨 왔습니다. 물어보는 순서와
멈추는 조건은 일부러 똑같이 뒀습니다 — 특히 실거래 스위치는 그렇습니다.

각 함수는 자기 화면을 직접 그리고, 끝나면 그냥 돌아옵니다.
"돌아가시겠습니까" 를 묻는 것은 메뉴 쪽 일입니다.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from . import ui

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = sys.executable or "python"
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
REPO_URL = "https://github.com/Heptagon372/athena-signal"


# 공통 -----------------------------------------------------------------------

def run(command: list[str], cwd: Path | None = None) -> int:
    """자식 프로세스를 이 창에 그대로 흘려보내고 종료 코드를 돌려줍니다."""
    try:
        return subprocess.call(command, cwd=str(cwd or ROOT))
    except FileNotFoundError:
        ui.err(f"{command[0]} 을(를) 찾을 수 없습니다.")
        return 127
    except KeyboardInterrupt:
        ui.out()
        ui.warn("중단했습니다.")
        return 130


def new_window(title: str, command: str) -> None:
    """새 콘솔 창에서 명령을 띄웁니다 (그 창을 닫으면 서비스가 꺼집니다).

    python 을 sys.executable 대신 이름으로 부릅니다. 경로에 공백이 있으면
    cmd 의 중첩 따옴표가 깨지는데, start.bat 시절부터 이 방식으로 돌아왔습니다.
    """
    subprocess.Popen(
        f'start "{title}" cmd /k "chcp 65001 >nul && {command}"',
        shell=True,
        cwd=str(ROOT),
    )


def port_open(port: int) -> bool:
    """이 PC 의 포트가 열려 있는지 (= 뭔가 듣고 있는지)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.15)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_port(port: int, seconds: int) -> bool:
    """포트가 열릴 때까지 한 줄짜리 진행 표시를 그리며 기다립니다."""
    deadline = time.time() + seconds
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    tick = 0
    while time.time() < deadline:
        if port_open(port):
            sys.stdout.write("\r" + " " * 60 + "\r")
            return True
        sys.stdout.write(f"\r  {ui.GOLD}{frames[tick % len(frames)]}{ui.RESET}"
                         f"  {ui.FAINT}준비되기를 기다립니다…{ui.RESET}")
        sys.stdout.flush()
        tick += 1
        time.sleep(0.4)
    sys.stdout.write("\r" + " " * 60 + "\r")
    return False


def _git(*args: str) -> str:
    try:
        done = subprocess.run(
            ["git", *args], cwd=str(ROOT), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _credentials():
    """설정 저장소. 못 불러오면 화면에 알리고 None 을 돌려줍니다."""
    try:
        from data_sources import credentials
        return credentials
    except Exception as exc:                          # noqa: BLE001 - 원인이 무엇이든 메뉴는 살아야 합니다
        ui.err(f"설정 모듈을 불러오지 못했습니다: {exc}")
        return None


def _user_env_names() -> set[str]:
    """setx 로 이 PC 에 심어 둔 사용자 환경변수 이름들."""
    names: set[str] = set()
    if os.name != "nt":
        return names
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            for i in range(winreg.QueryInfoKey(key)[1]):
                names.add(winreg.EnumValue(key, i)[0])
    except (OSError, ImportError):
        pass
    return names


def remember(values: dict, credentials) -> None:
    """설정을 api_keys.json 에 적습니다.

    예전 setup_*.bat 은 같은 값을 setx 로 심었습니다. 그 환경변수가 남아
    있으면 파일보다 **먼저** 읽혀서 "분명히 고쳤는데 그대로"가 됩니다.
    그래서 이미 심어져 있는 이름만 골라 같이 갱신합니다.
    """
    credentials.save(values)
    planted = _user_env_names()
    for name, value in values.items():
        os.environ[name] = value              # 이 창에서 띄우는 서버부터 바로 적용
        if name in planted:
            subprocess.run(["setx", name, value], capture_output=True)


def mask(value: str) -> str:
    """키를 알아볼 만큼만 보여줍니다 (어깨너머로 새지 않게)."""
    if not value:
        return ""
    if len(value) <= 6:
        return value[0] + "•" * (len(value) - 1)
    return f"{value[:3]}•••{value[-2:]}"


# 지금 상태 -------------------------------------------------------------------

@dataclass
class Snapshot:
    backend: bool = False
    frontend: bool = False
    branch: str = ""
    dirty: int = 0
    ahead: int = 0
    behind: int = 0
    api_done: int = 0
    api_total: int = 0
    google: bool = False
    trading: str = "미설정"


_git_cache: tuple[float, tuple[str, int, int, int]] = (0.0, ("", 0, 0, 0))


def forget_git() -> None:
    """다음 새로고침 때 git 상태를 다시 읽게 합니다 (올리기·받아오기 직후)."""
    global _git_cache
    _git_cache = (0.0, ("", 0, 0, 0))


def _git_state() -> tuple[str, int, int, int]:
    """브랜치 · 안 올린 변경 수 · 앞선 커밋 · 뒤처진 커밋 (10초 캐시).

    git 호출은 100ms 쯤 걸립니다. 화살표를 누를 때마다 부르면 메뉴가
    끈적해지므로 잠깐 재활용합니다.
    """
    global _git_cache
    now = time.time()
    if now - _git_cache[0] < 10:
        return _git_cache[1]

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = len([l for l in _git("status", "--porcelain").splitlines() if l.strip()])
    behind = ahead = 0
    counts = _git("rev-list", "--left-right", "--count", "@{u}...HEAD")
    if counts:
        parts = counts.replace("\t", " ").split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            behind, ahead = int(parts[0]), int(parts[1])

    _git_cache = (now, (branch, dirty, ahead, behind))
    return _git_cache[1]


def snapshot() -> Snapshot:
    """메뉴 윗줄에 띄울 '지금 이 PC 의 상태'."""
    state = Snapshot()
    state.backend = port_open(BACKEND_PORT)
    state.frontend = port_open(FRONTEND_PORT)
    state.branch, state.dirty, state.ahead, state.behind = _git_state()

    try:
        from data_sources import credentials

        report = credentials.status()
        state.api_total = len(report)
        state.api_done = sum(1 for info in report.values() if info["configured"])
        # Firebase 든 예전 OAuth 든, 하나라도 되면 구글 로그인은 켜집니다
        state.google = (credentials.is_configured("firebase")
                        or credentials.is_configured("google")) \
            and credentials.is_configured("mongo")
        if not credentials.get("KIS_ACCOUNT"):
            state.trading = "미설정"
        elif credentials.get_bool("KIS_LIVE_TRADING") and not credentials.get_bool("KIS_MOCK"):
            state.trading = "실전"
        else:
            state.trading = "모의"
    except Exception:                                 # noqa: BLE001 - 상태 표시 하나 때문에 메뉴가 죽으면 안 됩니다
        pass
    return state


# 실행 -----------------------------------------------------------------------

def start_all() -> None:
    ui.head("시작", "백엔드와 웹 화면을 켜고 브라우저를 엽니다")

    if shutil.which("node") is None:
        ui.err("Node.js 를 찾을 수 없습니다. https://nodejs.org 에서 설치해 주세요.")
        return

    if not (ROOT / "frontend" / "node_modules").exists():
        npm = shutil.which("npm")
        if npm is None:
            ui.err("npm 을 찾을 수 없습니다. Node.js 를 다시 설치해 주세요.")
            return
        ui.info("웹 화면 패키지를 설치합니다. 처음 한 번은 몇 분 걸립니다…")
        ui.out()
        if run([npm, "install", "--omit=optional", "--no-fund", "--no-audit"],
               cwd=ROOT / "frontend"):
            ui.out()
            ui.err("설치에 실패했습니다. 위 메시지를 확인하세요.")
            return
        ui.out()

    # 이미 떠 있는 것은 다시 띄우지 않습니다. 그냥 실행하면 "포트가 이미
    # 사용 중"이라며 창만 하나 더 쌓이고, 어느 쪽이 진짜인지 알 수 없게 됩니다.
    if port_open(BACKEND_PORT):
        ui.note("백엔드", "이미 켜져 있어 건너뜁니다 — 코드를 고쳤다면 [3] 업데이트")
    else:
        ui.note("백엔드", f"띄웁니다 · localhost:{BACKEND_PORT}")
        new_window("Athena Signal - 백엔드",
                   f"python -m uvicorn api:app --port {BACKEND_PORT} --no-use-colors")

    if port_open(FRONTEND_PORT):
        ui.note("웹", "이미 켜져 있어 건너뜁니다")
    else:
        ui.note("웹", f"띄웁니다 · localhost:{FRONTEND_PORT}")
        time.sleep(2)
        new_window("Athena Signal - 웹", "cd frontend && npm run dev")

    ui.out()
    if wait_port(FRONTEND_PORT, 60):
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
        ui.ok(f"브라우저를 열었습니다 — http://localhost:{FRONTEND_PORT}")
        ui.out()
        ui.info("끄려면 새로 열린 두 창을 닫으면 됩니다. 이 창은 그대로 둬도 됩니다.")
    else:
        ui.warn("화면이 아직 뜨지 않았습니다. 새로 열린 창의 메시지를 확인하세요.")


def status_check() -> None:
    ui.head("상태 확인", "서버를 건드리지 않습니다")
    run([PY, "restart_server.py", "--check"])
    ui.out()
    if ui.confirm("빈 창만 정리할까요? (돌고 있는 서버는 그대로 둡니다)"):
        ui.out()
        run([PY, "restart_server.py", "--clean"])


def update(force: bool = False) -> None:
    ui.head("업데이트 (강제)" if force else "업데이트",
            "체결 대기 주문이 있어도 진행합니다" if force else "고친 코드로 서버만 다시 올립니다")

    code = run([PY, "restart_server.py"] + (["--force"] if force else []))
    ui.out()
    if code == 0:
        ui.ok("완료되었습니다.")
    elif code == 2:
        ui.warn("체결 대기 중인 주문이 있어 멈췄습니다. 서버는 그대로 돌고 있습니다.")
        ui.info("그래도 올리려면 메뉴에서 [4] 강제 업데이트 를 고르세요.")
    else:
        ui.err("실패했습니다. 위 메시지를 확인하세요. 서버는 그대로 살아 있습니다.")


# 깃허브 ---------------------------------------------------------------------

def _powershell(script: str) -> int:
    """upload.ps1 / download.ps1 을 이 창에서 그대로 돌립니다.

    git 이 얽힌 부분(스태시·되돌리기·충돌 안내)은 이미 다듬어져 있어
    다시 옮겨 적지 않았습니다. -NoPause 는 스크립트가 스스로 멈추지 않고
    메뉴로 돌아오게 합니다.
    """
    powershell = shutil.which("powershell") or "powershell"
    return run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(ROOT / script), "-NoPause"])


def upload() -> None:
    ui.head("올리기", f"이 PC 의 작업을 GitHub 에 저장합니다 · {REPO_URL}")
    _powershell("upload.ps1")


def download() -> None:
    ui.head("받아오기", f"다른 PC 에서 올린 최신 코드를 내려받습니다 · {REPO_URL}")
    _powershell("download.ps1")


# 설정 -----------------------------------------------------------------------

_API_ORDER = ["toss", "kis", "naver", "datago", "krx", "reddit"]


def setup_api() -> None:
    ui.head("API 키", "넣으면 정확해지고, 비워 둬도 지금 그대로 돕니다")
    credentials = _credentials()
    if credentials is None:
        return

    ui.block([
        "하나도 넣지 않아도 기존 공개 경로로 정상 동작합니다.",
        "필요한 것만 넣고 나머지는 Enter 로 건너뛰세요.",
    ])
    ui.out()

    values: dict[str, str] = {}
    for name in _API_ORDER:
        spec = credentials.PROVIDERS[name]
        done = credentials.is_configured(name)
        badge = f"{ui.GREEN}설정됨{ui.RESET}" if done else f"{ui.FAINT}미설정{ui.RESET}"
        ui.out(f"  {ui.GOLD}{spec['label']}{ui.RESET}  {badge}")
        ui.out(f"  {ui.FAINT}{spec['gives']}{ui.RESET}")
        ui.out(f"  {ui.FAINT}발급  {spec['portal']}{ui.RESET}")

        for field in spec["fields"]:
            current = credentials.get(field)
            typed = ui.ask(ui.pad(field, 22),
                           f"지금 {mask(current)} · Enter 면 그대로" if current else "")
            if typed:
                values[field] = typed
        ui.out()

    if not values:
        ui.info("바뀐 것이 없습니다.")
        return

    remember(values, credentials)
    ui.ok(f"{len(values)}개 항목을 저장했습니다 — api_keys.json")
    ui.out()
    ui.info("서버가 켜져 있다면 [3] 업데이트 를 한 번 돌려야 새 키가 반영됩니다.")


def setup_google() -> None:
    ui.head("구글 로그인", "아이디/비밀번호 없이 구글 계정으로 바로 들어옵니다")
    credentials = _credentials()
    if credentials is None:
        return

    ui.block([
        "건너뛰어도 기존 아이디/비밀번호 로그인은 그대로 됩니다.",
        "",
        f"{ui.GREY}1단계 · Firebase 프로젝트{ui.RESET}   약 3분, 무료",
        "  https://console.firebase.google.com  에서 프로젝트 생성",
        "  Authentication → Sign-in method → Google 사용 설정",
        "  프로젝트 설정 → 내 앱 → </> (웹 앱 추가) → firebaseConfig 확인",
        f"  {ui.FAINT}거기 나오는 projectId 와 apiKey 를 아래에 넣으면 됩니다.{ui.RESET}",
        f"  {ui.FAINT}apiKey 는 비밀이 아닙니다 — 브라우저에 그대로 실리는 공개값입니다.{ui.RESET}",
        "",
        f"  {ui.GREY}승인된 도메인{ui.RESET}  Authentication → 설정 → 승인된 도메인",
        f"  {ui.FAINT}localhost 는 처음부터 들어 있습니다. 외부 주소로 열어 둘 때만 추가하세요.{ui.RESET}",
        f"  {ui.FAINT}리디렉션 URI 등록은 필요 없습니다 (예전 OAuth 방식과 다른 점).{ui.RESET}",
        "",
        f"{ui.GREY}2단계 · MongoDB 준비{ui.RESET}",
        "  Atlas(무료 M0)  https://cloud.mongodb.com  → 접속 문자열 복사",
        "  혼자 쓴다면      mongodb://localhost:27017",
    ])
    ui.out()

    values: dict[str, str] = {}
    for field, label in (("FIREBASE_PROJECT_ID", "Firebase 프로젝트 ID"),
                         ("FIREBASE_API_KEY", "Firebase 웹 API 키"),
                         ("MONGODB_URI", "MongoDB 접속 문자열")):
        current = credentials.get(field)
        typed = ui.ask(ui.pad(label, 22),
                       f"지금 {mask(current)} · Enter 면 그대로" if current else "")
        if typed:
            values[field] = typed

    if not values:
        ui.out()
        ui.info("바뀐 것이 없습니다. 기존 로그인 방식이 그대로 쓰입니다.")
        return

    remember(values, credentials)
    ui.out()
    ui.ok("저장했습니다 — api_keys.json")

    if not credentials.get("MONGODB_URI"):
        ui.warn("MongoDB 주소가 없으면 구글 로그인은 켜지지 않습니다.")

    ui.out()
    if ui.confirm("MongoDB 드라이버(pymongo)를 지금 설치할까요?", default=True):
        ui.out()
        run([PY, "-m", "pip", "install", "pymongo[srv]"])

    # cryptography 가 있으면 토큰 서명을 이 PC 에서 직접 확인합니다. 없으면 구글에
    # 물어보는 경로로 도는데, 웹 API 키에 리퍼러 제한을 걸어둔 경우 그 경로가
    # 막힙니다. 설치해 두는 편이 실패할 구석이 하나 적습니다.
    ui.out()
    if ui.confirm("로그인 토큰을 이 PC 에서 직접 검증할까요? (cryptography 설치)",
                  default=True):
        ui.out()
        run([PY, "-m", "pip", "install", "cryptography"])

    ui.out()
    ui.info("확인:  python -m storage.accounts")
    ui.info("서버가 켜져 있다면 [3] 업데이트 를 한 번 돌리세요.")


def setup_trading() -> None:
    ui.head("자동매매", "주문을 낼 계좌와 모의/실전을 정합니다")
    credentials = _credentials()
    if credentials is None:
        return

    ui.block([
        "APP KEY / SECRET 은 [7] API 키 의 한국투자증권 칸에서 넣습니다.",
        "",
        f"{ui.GREY}모의{ui.RESET}  KIS 모의투자 서버 — 가짜 돈, 주문 흐름만 확인",
        f"{ui.GREY}실전{ui.RESET}  진짜 돈이 오갑니다",
        "",
        "설정하지 않아도 화면 안 모의(paper) 자동매매는 그대로 돕니다.",
    ])
    ui.out()

    current = credentials.get("KIS_ACCOUNT")
    account = ui.ask(ui.pad("계좌번호", 22),
                     f"지금 {mask(current)} · Enter 면 그대로" if current
                     else "예: 12345678-01 · 건너뛰려면 Enter")

    if not account and not current:
        ui.out()
        ui.info("건너뛰었습니다. 화면 안 모의 자동매매는 그대로 쓸 수 있습니다.")
        ui.info("콘솔:  http://localhost:8000/autotrade")
        return

    values: dict[str, str] = {}
    if account:
        values["KIS_ACCOUNT"] = account

    deriv = ui.ask(ui.pad("선물옵션 계좌", 22), "없으면 Enter")
    if deriv:
        values["KIS_DERIV_ACCOUNT"] = deriv

    ui.out()
    if ui.confirm("모의투자 서버로 시작할까요?", default=True):
        values["KIS_MOCK"] = "1"
        values["KIS_LIVE_TRADING"] = "0"
        remember(values, credentials)
        ui.out()
        ui.ok("모의투자 서버로 맞췄습니다. 가짜 돈으로 주문 경로를 확인합니다.")
    else:
        values["KIS_MOCK"] = "0"
        ui.out()
        ui.out(f"  {ui.RED}{'─' * (ui.term_width() - 2)}{ui.RESET}")
        ui.err("실전 서버에서는 진짜 돈으로 주문이 체결됩니다.")
        ui.out(f"  {ui.RED}{'─' * (ui.term_width() - 2)}{ui.RESET}")
        ui.out()
        ui.info("실주문까지 허용하려면 아래에 대문자로 LIVE 라고만 적으세요.")
        ui.info("그냥 Enter 를 치면 실전 서버에 붙되 주문은 막힌 상태가 됩니다.")
        ui.out()
        typed = ui.ask(ui.pad("입력", 22))
        values["KIS_LIVE_TRADING"] = "1" if typed == "LIVE" else "0"
        remember(values, credentials)
        ui.out()
        if typed == "LIVE":
            ui.warn("실주문이 허용되었습니다. 자동매매 콘솔에서 모드를 live 로 바꾸고")
            ui.warn("거기서 한 번 더 LIVE 를 입력해야 실제로 나갑니다.")
        else:
            ui.ok("실주문은 막은 채로 뒀습니다. 조회만 됩니다.")

    ui.out()
    ui.info("확인:  python -m data_sources.kis_trading")
    ui.info("콘솔:  http://localhost:8000/autotrade   ·   설명: AUTOTRADE.md")
    ui.info("서버가 켜져 있다면 [3] 업데이트 를 한 번 돌리세요.")


# 도구 -----------------------------------------------------------------------

def build_exe() -> None:
    ui.head("EXE 만들기", "혼자 도는 실행 파일 하나로 굽습니다 · 1~2분")

    if not ui.confirm("지금 만들까요?", default=False):
        ui.out()
        ui.info("그만뒀습니다.")
        return

    ui.out()
    for requirement in ("requirements.txt", "requirements-build.txt"):
        if run([PY, "-m", "pip", "install", "-r", requirement]):
            ui.out()
            ui.err(f"{requirement} 설치에 실패했습니다.")
            return

    hidden = [
        "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
        "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan", "uvicorn.lifespan.on",
        "bs4", "feedparser", "yfinance", "pandas", "numpy", "requests",
        "data_sources.symbol_registry", "data_sources.kr_market",
        "data_sources.kis_client", "data_sources.market_clock",
        "data_sources.http_client", "data_sources.price_provider",
        "data_sources.news_crawler", "data_sources.community_crawler",
        "engine.indicators", "engine.scoring", "engine.backtest", "storage.db",
    ]
    command = [PY, "-m", "PyInstaller", "--noconfirm", "--onefile", "--console",
               "--name", "AthenaSignal", "--add-data", "web;web"]
    for module in hidden:
        command += ["--hidden-import", module]
    command.append("run_server.py")

    ui.out()
    run(command)

    ui.out()
    if (ROOT / "dist" / "AthenaSignal.exe").exists():
        ui.ok("dist\\AthenaSignal.exe 를 만들었습니다.")
        ui.info("이제 그 파일만 두 번 눌러도 서버가 켜지고 브라우저가 열립니다.")
    else:
        ui.err("만들지 못했습니다. 위 오류 메시지를 그대로 복사해 주세요.")
