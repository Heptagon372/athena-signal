@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================================
echo   서버 업데이트 (코드를 고친 뒤 실행하세요)
echo ============================================================
echo.

python restart_server.py %*
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" (
    echo   완료되었습니다. 이 창은 닫으셔도 됩니다.
) else if "%RC%"=="2" (
    echo   체결 대기 중인 주문이 있어 멈췄습니다.
    echo   그래도 진행하려면 "업데이트-강제.bat" 를 실행하세요.
) else (
    echo   실패했습니다. 위 메시지를 확인하세요.
    echo   서버는 그대로 살아 있습니다.
)
echo.
pause
