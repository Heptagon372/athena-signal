@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ============================================================
echo   상태 확인 (서버를 건드리지 않습니다)
echo ============================================================
echo.
python restart_server.py --check
echo.
echo   빈 창만 정리하려면: python restart_server.py --clean
echo.
pause
