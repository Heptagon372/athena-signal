"""
아테나 시그널 - 더블클릭 실행용 런처
------------------------------------
이 파일을 PyInstaller로 감싸면 AthenaSignal.exe 가 됩니다.
실행하면: 백엔드 서버(uvicorn)를 켜고 -> 잠깐 기다렸다가 -> 기본 브라우저로
대시보드를 자동으로 엽니다. 창을 닫으면(Ctrl+C 또는 콘솔 종료) 서버도 같이 꺼집니다.
"""

import threading
import time
import webbrowser
import sys
import os

# 윈도우 기본 콘솔(cp949)에서 한글 안내문이 깨지지 않도록 출력 인코딩을 UTF-8로 고정
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# PyInstaller로 묶었을 때도 같은 폴더의 모듈들을 찾을 수 있도록 경로 보정
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def open_browser_after_delay():
    time.sleep(1.8)  # 서버가 뜰 시간을 잠깐 기다림
    webbrowser.open(f"http://{HOST}:{PORT}/")


def main():
    print("=" * 50)
    print("  아테나 시그널 (Athena Signal) 서버를 시작합니다")
    print(f"  대시보드: http://{HOST}:{PORT}/")
    print("  이 창을 닫으면 서버가 종료됩니다.")
    print("=" * 50)

    threading.Thread(target=open_browser_after_delay, daemon=True).start()

    from api import app  # noqa: E402 (경로 보정 이후 import)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
