"""화면과 키 입력 — 콘솔에서 보이는 모든 것.

표준 라이브러리만 씁니다. GitHub 에서 막 받아온 PC 에서 pip install 없이
바로 떠야 하기 때문입니다.

한글은 터미널에서 두 칸을 차지합니다. len() 으로 자리를 맞추면 줄이
어긋나므로, 이 파일의 폭 계산은 전부 dw() 를 거칩니다.
"""

import os
import re
import shutil
import sys
import unicodedata

WIN = os.name == "nt"

# 색 -------------------------------------------------------------------------
# 256색만 씁니다 (트루컬러는 옛 콘솔에서 그대로 글자로 새어 나옵니다).
RESET   = "\x1b[0m"
BOLD    = "\x1b[1m"
GOLD    = "\x1b[38;5;179m"      # 아테나 = 금빛. 제목과 고른 줄에 씁니다
GOLD_HI = "\x1b[38;5;222m"
GREY    = "\x1b[38;5;245m"
FAINT   = "\x1b[38;5;240m"      # 설명글 — 읽히되 눈에 걸리지 않을 만큼만
GREEN   = "\x1b[38;5;108m"
RED     = "\x1b[38;5;174m"
BLUE    = "\x1b[38;5;110m"
SEL     = "\x1b[48;5;238m"      # 고른 줄의 배경 띠

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_ARROWS_WIN   = {"H": "up", "P": "down", "K": "left", "M": "right"}
_ARROWS_POSIX = {"A": "up", "B": "down", "D": "left", "C": "right"}


# 준비 -----------------------------------------------------------------------

def setup(title: str = "아테나 시그널") -> None:
    """콘솔을 UTF-8 + 색이 나오는 상태로 맞춥니다.

    윈도우 콘솔은 기본이 cp949 라 한글을 찍다가 죽고, 색 코드를 글자
    그대로 뱉습니다. 둘 다 여기서 한 번만 켜 둡니다.
    """
    if WIN:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
            for handle_id in (-11, -12):            # stdout, stderr
                handle = kernel32.GetStdHandle(handle_id)
                mode = ctypes.c_uint32()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (OSError, AttributeError, ImportError):
            pass                                     # 색이 없어도 쓸 수는 있습니다

    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    sys.stdout.write(f"\x1b]0;{title}\x07")
    sys.stdout.flush()


# 폭 계산 ---------------------------------------------------------------------

def dw(text: str) -> int:
    """터미널에서 이 문자열이 차지하는 칸 수 (한글·이모지는 두 칸)."""
    plain = _ANSI.sub("", text)
    width = 0
    for ch in plain:
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def pad(text: str, columns: int) -> str:
    """오른쪽을 공백으로 채워 정확히 columns 칸을 차지하게 만듭니다."""
    return text + " " * max(0, columns - dw(text))


def cut(text: str, columns: int) -> str:
    """넘치면 잘라내고 … 을 붙입니다."""
    if dw(text) <= columns:
        return text
    out_chars, width = [], 0
    for ch in _ANSI.sub("", text):
        step = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if width + step > columns - 1:
            break
        out_chars.append(ch)
        width += step
    return "".join(out_chars) + "…"


def term_width() -> int:
    """그림을 그릴 폭. 창이 아무리 넓어도 110칸을 넘기지 않습니다."""
    return max(64, min(shutil.get_terminal_size((92, 30)).columns - 1, 110))


# 출력 -----------------------------------------------------------------------

def out(line: str = "") -> None:
    print(line)


def clear() -> None:
    sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
    sys.stdout.flush()


def cursor(visible: bool) -> None:
    sys.stdout.write("\x1b[?25h" if visible else "\x1b[?25l")
    sys.stdout.flush()


def rule(label: str = "", width: int | None = None) -> str:
    """'실행 ──────' 처럼 제목이 붙은 가로줄.

    부르는 쪽에서 왼쪽에 두 칸을 들여 쓰므로, 여기서는 width - 2 칸만
    채웁니다. 그래야 메뉴 줄과 오른쪽 끝이 딱 맞습니다.
    """
    width = width or term_width()
    if not label:
        return f"{FAINT}{'─' * (width - 2)}{RESET}"
    dashes = max(3, width - 3 - dw(label))
    return f"{GREY}{label} {FAINT}{'─' * dashes}{RESET}"


def head(title: str, subtitle: str = "") -> None:
    """메뉴에서 항목을 고른 직후 그리는 머리말."""
    clear()
    cursor(True)
    out()
    line = f"  {GOLD}{BOLD}{title}{RESET}"
    if subtitle:
        line += f"   {FAINT}{subtitle}{RESET}"
    out(line)
    out(f"  {rule()}")
    out()


def ok(message: str) -> None:
    out(f"  {GREEN}✓{RESET}  {message}")


def err(message: str) -> None:
    out(f"  {RED}✗{RESET}  {message}")


def warn(message: str) -> None:
    out(f"  {GOLD_HI}!{RESET}  {message}")


def info(message: str) -> None:
    out(f"  {FAINT}·{RESET}  {message}")


def note(label: str, message: str) -> None:
    """'백엔드   띄웁니다' 처럼 이름과 상태를 나란히 보여줍니다."""
    out(f"  {BLUE}{pad(label, 10)}{RESET}{FAINT}{message}{RESET}")


def block(lines: list[str]) -> None:
    """안내문 덩어리 — 왼쪽에 세로줄을 그어 본문과 구분합니다."""
    for line in lines:
        out(f"  {FAINT}│{RESET}  {line}")


# 입력 -----------------------------------------------------------------------

def ask(label: str, hint: str = "") -> str:
    """한 줄 입력. 그냥 Enter 를 치면 빈 문자열이 돌아옵니다.

    안내는 물음표 앞에 붙여 한 줄로 둡니다 — 항목마다 두 줄씩 쓰면
    키를 열 개쯤 넣는 화면이 금세 읽기 싫어집니다.
    """
    prompt = f"  {GOLD}{label}{RESET}"
    if hint:
        prompt += f" {FAINT}{hint}{RESET}"
    cursor(True)
    try:
        return input(f"{prompt} {GOLD}›{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        out()
        return ""


def confirm(question: str, default: bool = False) -> bool:
    """예/아니오. 기본값이 아닌 쪽을 고르려면 또렷하게 눌러야 합니다."""
    tail = "(Y/n)" if default else "(y/N)"
    answer = ask(f"{question} {tail}").lower()
    if not answer:
        return default
    return answer in ("y", "yes", "네", "ㅇ")


def pause(message: str = "아무 키나 누르면 메뉴로 돌아갑니다") -> None:
    out()
    out(f"  {FAINT}{message}{RESET}")
    cursor(False)
    key()


def key() -> str:
    """키 하나를 기다립니다. 화살표는 'up' / 'down' 같은 이름으로 옵니다."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        if not line:
            return "q"
        return line.strip()[:1] or "enter"

    if WIN:
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return _ARROWS_WIN.get(msvcrt.getwch(), "")
        if ch == "\r":
            return "enter"
        if ch == "\x1b":
            return "esc"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            if select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.read(1)                     # '['
                return _ARROWS_POSIX.get(sys.stdin.read(1), "")
            return "esc"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    if ch == "\r":
        return "enter"
    if ch == "\x03":
        raise KeyboardInterrupt
    return ch
