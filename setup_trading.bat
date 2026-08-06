@echo off
setlocal

echo ============================================================
echo   아테나 시그널 - 자동매매(주문) 설정
echo ============================================================
echo.
echo   이 설정은 "실제 주문"에 관한 것입니다.
echo   APP KEY / SECRET 은 setup_kis.bat 에서 먼저 등록하세요.
echo.
echo   모드
echo     모의투자  KIS 모의투자 서버 - 가상 자금, 실제 주문 경로 검증
echo     실전      실제 자금이 움직입니다
echo.
echo   설정하지 않아도 내부 가상계좌(paper) 자동매매는 그대로 동작합니다.
echo ------------------------------------------------------------
echo.

set /p ACCOUNT="계좌번호 (예: 12345678-01, 건너뛰려면 Enter): "
if "%ACCOUNT%"=="" goto :skipped

setx KIS_ACCOUNT "%ACCOUNT%" > nul
echo   계좌번호를 저장했습니다.

echo.
set /p DERIV="선물옵션 계좌가 따로 있으면 입력 (없으면 Enter): "
if not "%DERIV%"=="" (
    setx KIS_DERIV_ACCOUNT "%DERIV%" > nul
    echo   선물옵션 계좌를 저장했습니다.
)

echo.
set /p MOCKMODE="모의투자 서버를 쓰시겠습니까? (Y/n): "
if /i "%MOCKMODE%"=="n" goto :realserver

setx KIS_MOCK "1" > nul
setx KIS_LIVE_TRADING "0" > nul
echo.
echo   모의투자 서버로 설정했습니다. 가상 자금으로 실제 주문 경로를 검증합니다.
goto :done

:realserver
echo.
echo ------------------------------------------------------------
echo   경고: 실전 서버는 실제 자금으로 주문이 체결됩니다.
echo ------------------------------------------------------------
echo.
echo   실주문을 허용하려면 아래에 정확히  LIVE  라고 입력하세요.
echo   (그냥 Enter 를 누르면 실전 서버로 두되 주문은 잠긴 상태가 됩니다)
echo.
set /p CONFIRM="입력: "

setx KIS_MOCK "0" > nul
if /i "%CONFIRM%"=="LIVE" (
    setx KIS_LIVE_TRADING "1" > nul
    echo.
    echo   실주문이 허용되었습니다. 자동매매 콘솔에서 모드를 live 로 바꾸고
    echo   시작할 때 한 번 더 LIVE 를 입력해야 실제로 동작합니다.
) else (
    setx KIS_LIVE_TRADING "0" > nul
    echo.
    echo   실주문은 잠긴 상태로 두었습니다. 조회만 가능합니다.
)

:done
echo.
echo ============================================================
echo   설정 완료
echo ============================================================
echo.
echo 연결 확인 (새 명령 창에서):
echo.
echo     python -m data_sources.kis_trading
echo.
echo 자동매매 콘솔:  http://localhost:8000/autotrade
echo 자세한 설명  :  AUTOTRADE.md
echo.
echo * 이미 열려 있는 콘솔 창/서버에는 적용되지 않습니다.
echo   반드시 새 창에서 서버를 다시 시작하세요.
echo.
goto :end

:skipped
echo.
echo 설정을 건너뛰었습니다. 내부 가상계좌(paper) 모드는 그대로 사용할 수 있습니다.
echo 콘솔: http://localhost:8000/autotrade
echo.

:end
pause
endlocal
