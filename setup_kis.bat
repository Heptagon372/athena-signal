@echo off
setlocal

echo ============================================================
echo   아테나 시그널 - 한국투자증권 KIS 공식 API 연결 설정
echo ============================================================
echo.
echo 이 설정을 하면 실시간 체결가와 "장중" 투자자별 매매동향
echo (개인/외국인/기관)을 증권사 공식 API로 받아옵니다.
echo.
echo 설정하지 않아도 토스증권 + 네이버금융 경로로 정상 동작합니다.
echo.
echo ------------------------------------------------------------
echo  [키 발급 방법]  약 5분, 무료
echo ------------------------------------------------------------
echo   1. https://apiportal.koreainvestment.com  접속 후 로그인
echo   2. 상단 [API 신청] - 실전투자 또는 모의투자 선택
echo   3. APP KEY / APP SECRET 발급받아 복사
echo.
echo   ※ 한국투자증권 계좌가 필요합니다 (비대면 개설 가능)
echo   ※ 이 창에 입력한 키는 이 PC의 사용자 환경변수에만 저장됩니다
echo.
echo ------------------------------------------------------------
echo.

set /p APPKEY="APP KEY 를 붙여넣고 Enter (건너뛰려면 그냥 Enter): "
if "%APPKEY%"=="" goto :skipped

set /p APPSECRET="APP SECRET 을 붙여넣고 Enter: "
if "%APPSECRET%"=="" goto :skipped

echo.
set /p MOCKMODE="모의투자 서버를 사용하시겠습니까? (y/N): "

setx KIS_APP_KEY "%APPKEY%" > nul
setx KIS_APP_SECRET "%APPSECRET%" > nul

if /i "%MOCKMODE%"=="y" (
    setx KIS_MOCK "1" > nul
    echo   모의투자 서버로 설정했습니다.
) else (
    setx KIS_MOCK "0" > nul
    echo   실전투자 서버로 설정했습니다.
)

echo.
echo ============================================================
echo   저장 완료
echo ============================================================
echo.
echo 연결이 되는지 확인하려면 ^(새 명령 창을 열고^):
echo.
echo     python -m data_sources.kis_client
echo.
echo 그 다음 서버를 재시작하면 대시보드 상단 배너가
echo "한국투자증권 KIS 공식 API 연결됨" 으로 바뀝니다.
echo.
echo ※ 이미 열려 있는 명령 창/서버에는 적용되지 않습니다.
echo    반드시 새 창에서 서버를 다시 실행하세요.
echo.
goto :end

:skipped
echo.
echo 설정을 건너뛰었습니다. 토스증권 + 네이버금융 경로로 계속 동작합니다.
echo.

:end
pause
endlocal
