"""아테나 시그널 — 실행 콘솔.

    python athena.py        (또는 아테나.bat 을 두 번 누르세요)

예전에는 할 일마다 .bat 파일이 하나씩 있었습니다. 이제 이 창 하나에서
화살표로 고르고 Enter 로 실행합니다. 하는 일은 그대로입니다.
"""

import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 자식으로 띄우는 파이썬들도 한글을 그대로 찍게 해 둡니다.
# (윈도우 기본 cp949 로는 restart_server.py 의 ✓ 하나에 죽습니다)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from cli import actions, ui                                  # noqa: E402
from cli.actions import Snapshot                             # noqa: E402

LABEL_COLUMN = 20
BADGE_COLUMN = 6
GUTTER = 25 + BADGE_COLUMN + 2      # 표시·글쇠·이름 + 배지 자리


# 메뉴 -----------------------------------------------------------------------

@dataclass
class Item:
    key: str
    label: str
    desc: str
    run: Callable[[], None] | None                 # None 이면 끝내기
    badge: Callable[[Snapshot], tuple[str, str]] | None = None


def _running(state: Snapshot) -> tuple[str, str]:
    if state.backend and state.frontend:
        return "켜짐", ui.GREEN
    if state.backend or state.frontend:
        return "반쪽", ui.GOLD_HI
    return "", ""


def _api(state: Snapshot) -> tuple[str, str]:
    if not state.api_total:
        return "", ""
    done = f"{state.api_done}/{state.api_total}"
    return done, ui.GREEN if state.api_done else ui.FAINT


def _google(state: Snapshot) -> tuple[str, str]:
    return ("연결됨", ui.GREEN) if state.google else ("", "")


def _trading(state: Snapshot) -> tuple[str, str]:
    return {
        "실전": ("실전", ui.RED),
        "모의": ("모의", ui.BLUE),
    }.get(state.trading, ("", ""))


def _changes(state: Snapshot) -> tuple[str, str]:
    if not state.dirty and not state.ahead:
        return "", ""
    return f"{state.dirty + state.ahead}건", ui.GOLD_HI


def _incoming(state: Snapshot) -> tuple[str, str]:
    return (f"{state.behind}건", ui.BLUE) if state.behind else ("", "")


SECTIONS: list[tuple[str, list[Item]]] = [
    ("실행", [
        Item("1", "시작", "서버를 켜고 브라우저를 엽니다", actions.start_all, _running),
        Item("2", "상태 확인", "무엇이 켜져 있는지만 봅니다", actions.status_check),
        Item("3", "업데이트", "고친 코드로 서버만 다시 올립니다", actions.update),
        Item("4", "강제 업데이트", "체결 대기 주문이 있어도 진행합니다",
             lambda: actions.update(force=True)),
    ]),
    ("깃허브", [
        Item("5", "올리기", "이 PC 의 작업을 GitHub 에 저장합니다", actions.upload, _changes),
        Item("6", "받아오기", "다른 PC 에서 올린 최신 코드를 받습니다", actions.download, _incoming),
    ]),
    ("설정", [
        Item("7", "API 키", "서버 공용 키 (계정별 키는 웹 → 설정)", actions.setup_api, _api),
        Item("8", "구글 로그인", "구글 계정 로그인 · MongoDB 저장소", actions.setup_google, _google),
        Item("9", "자동매매", "주문 계좌 · 모의/실전 전환", actions.setup_trading, _trading),
    ]),
    ("도구", [
        Item("B", "EXE 만들기", "혼자 도는 실행 파일 하나로 굽습니다", actions.build_exe),
        Item("Q", "끝내기", "", None),
    ]),
]

ITEMS = [item for _, group in SECTIONS for item in group]


# 그리기 ---------------------------------------------------------------------

def status_line(state: Snapshot, width: int) -> str:
    """지금 이 PC 가 어떤 상태인지 한 줄로."""
    def lamp(alive: bool, name: str, port: int) -> str:
        dot = f"{ui.GREEN}●{ui.RESET}" if alive else f"{ui.FAINT}○{ui.RESET}"
        return f"{dot} {ui.GREY if alive else ui.FAINT}{name} {port}{ui.RESET}"

    parts = [lamp(state.backend, "백엔드", actions.BACKEND_PORT),
             lamp(state.frontend, "웹", actions.FRONTEND_PORT)]

    if state.branch:
        git = f"{ui.FAINT}git {state.branch}"
        if state.dirty:
            git += f" · 안 올린 변경 {state.dirty}"
        if state.ahead:
            git += f" · 올릴 커밋 {state.ahead}"
        if state.behind:
            git += f" · 받을 커밋 {state.behind}"
        parts.append(git + ui.RESET)

    return ui.cut("   ".join(parts), width - 4)


def row(item: Item, selected: bool, state: Snapshot, width: int) -> str:
    text, color = item.badge(state) if item.badge else ("", "")
    badge = " " * max(0, BADGE_COLUMN - ui.dw(text)) + text

    label = ui.pad(item.label, LABEL_COLUMN)
    desc = ui.pad(ui.cut(item.desc, width - 2 - GUTTER), width - 2 - GUTTER)
    mark = "▸" if selected else " "

    if selected:
        # 고른 줄은 배경 띠 하나로 통째로 칠합니다. 중간에 색을 되돌리면
        # 그 자리에서 띠가 끊겨 보입니다.
        return (f"  {ui.SEL}{ui.GOLD_HI}{ui.BOLD}"
                f"{mark} {item.key}  {label}{desc}  {badge}{ui.RESET}")

    return (f"  {mark} {ui.GOLD}{item.key}{ui.RESET}  "
            f"{label}{ui.FAINT}{desc}{ui.RESET}  "
            f"{color}{badge}{ui.RESET}")


def draw(index: int, state: Snapshot) -> None:
    width = ui.term_width()
    ui.clear()
    ui.cursor(False)

    ui.out()
    ui.out(f"  {ui.GOLD}{ui.BOLD}ATHENA SIGNAL{ui.RESET}"
           f"   {ui.FAINT}아테나 시그널 실행 콘솔{ui.RESET}")
    ui.out(f"  {ui.rule(width=width)}")
    ui.out(f"  {status_line(state, width)}")

    position = 0
    for title, group in SECTIONS:
        ui.out()
        ui.out(f"  {ui.rule(title, width)}")
        for item in group:
            ui.out(row(item, position == index, state, width))
            position += 1

    hint = ("↑↓ 고르기   Enter 실행   글쇠를 바로 눌러도 됩니다   R 새로고침   Q 끝내기"
            if width >= 80 else "↑↓ 고르기   Enter 실행   Q 끝내기")
    ui.out()
    ui.out(f"  {ui.FAINT}{hint}{ui.RESET}")
    ui.out()


# 실행 -----------------------------------------------------------------------

def launch(item: Item) -> None:
    try:
        item.run()
    except KeyboardInterrupt:
        ui.out()
        ui.warn("중단했습니다.")
    except Exception:                                # noqa: BLE001 - 무엇이 터져도 메뉴로는 돌아와야 합니다
        ui.out()
        ui.err("예상 못한 오류입니다. 아래 내용을 그대로 알려 주세요.")
        ui.out(f"{ui.FAINT}{traceback.format_exc()}{ui.RESET}")
    ui.pause()


def main() -> int:
    ui.setup()
    index = 0

    try:
        while True:
            state = actions.snapshot()
            draw(index, state)

            try:
                pressed = ui.key()
            except KeyboardInterrupt:
                return 0

            if pressed in ("q", "Q", "esc"):
                return 0
            if pressed in ("up", "k"):
                index = (index - 1) % len(ITEMS)
                continue
            if pressed in ("down", "j"):
                index = (index + 1) % len(ITEMS)
                continue
            if pressed in ("r", "R"):
                actions.forget_git()
                continue

            chosen: Item | None = None
            if pressed == "enter":
                chosen = ITEMS[index]
            else:
                for position, item in enumerate(ITEMS):
                    if item.key.lower() == pressed.lower():
                        index, chosen = position, item
                        break

            if chosen is None:
                continue
            if chosen.run is None:
                return 0

            launch(chosen)
            actions.forget_git()      # 방금 한 일이 상태에 바로 보이게
    finally:
        ui.cursor(True)
        ui.out()


if __name__ == "__main__":
    sys.exit(main())
