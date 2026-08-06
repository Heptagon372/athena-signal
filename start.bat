@echo off
cd /d "%~dp0"

echo ============================================================
echo   아테나 시그널 (Athena Signal)
echo ============================================================
echo.
echo   백엔드(분석 엔진)와 프론트엔드(웹)를 함께 실행합니다.
echo.
echo     웹      http://localhost:3000   (여기로 접속하세요)
echo     백엔드  http://localhost:8000   (분석 · API - 들어오면 3000으로 넘어갑니다)
echo.
echo   두 개의 창이 열립니다. 종료하려면 두 창을 모두 닫으세요.
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [오류] 파이썬을 찾을 수 없습니다. https://python.org 에서 설치해 주세요.
    pause
    exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
    echo [오류] Node.js 를 찾을 수 없습니다. https://nodejs.org 에서 설치해 주세요.
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [준비] 프론트엔드 패키지를 설치합니다. 처음 한 번만 걸립니다...
    pushd frontend
    call npm install --omit=optional --no-fund --no-audit
    popd
    echo.
)

REM 
REM 이미 떠 있는 서버는 다시 띄우지 않습니다.
REM 그냥 실행하면 "address already in use" 로 죽으면서 빈 창만 쌓이고,
REM 어느 것이 진짜 서버인지 알 수 없게 됩니다.
set BACKEND_UP=
set FRONTEND_UP=
netstat -ano | findstr /r /c:":8000 .*LISTENING" >nul 2>&1 && set BACKEND_UP=1
netstat -ano | findstr /r /c:":3000 .*LISTENING" >nul 2>&1 && set FRONTEND_UP=1

if defined BACKEND_UP (
    echo [1/2] 백엔드가 이미 실행 중입니다 - 건너뜁니다.
    echo       코드를 고쳐 다시 올리려면:  python restart_server.py
) else (
echo [1/2] 백엔드 시작...
    start "Athena Signal - 백엔드" cmd /k "chcp 65001 > nul && python -m uvicorn api:app --port 8000 --no-use-colors"
)

if defined FRONTEND_UP (
    echo [2/2] 프론트엔드가 이미 실행 중입니다 - 건너뜁니다.
) else (
echo [2/2] 프론트엔드 시작...
    timeout /t 3 /nobreak > nul
    start "Athena Signal - 웹" cmd /k "chcp 65001 > nul && cd frontend && npm run dev"
)

echo.
echo 잠시 후 브라우저가 자동으로 열립니다...
timeout /t 8 /nobreak > nul
start http://localhost:3000

echo.
echo 실행 중입니다. 이 창은 닫아도 됩니다.
echo (백엔드/웹 창을 닫으면 서비스가 종료됩니다)
echo.
pause
