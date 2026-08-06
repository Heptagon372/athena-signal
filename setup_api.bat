@echo off
cd /d "%~dp0"

echo ============================================================
echo   아테나 시그널 - 공식 API 키 설정
echo ============================================================
echo.
echo   키를 넣으면 비공식 크롤링 대신 공식 API를 씁니다.
echo   하나도 넣지 않아도 기존 공개 경로로 정상 동작합니다.
echo.
echo   필요한 것만 넣고 나머지는 그냥 Enter 로 건너뛰세요.
echo.

python -c "import sys" 2>nul
if errorlevel 1 (
    echo [오류] 파이썬을 찾을 수 없습니다.
    pause
    exit /b 1
)

echo ------------------------------------------------------------
echo  1. 토스증권 Open API        https://developers.tossinvest.com
echo     제공: 현재가, 캔들, 종목마스터, 장 운영 캘린더(공휴일 반영)
echo ------------------------------------------------------------
set /p TOSS_ID="   TOSS_CLIENT_ID     : "
if not "%TOSS_ID%"=="" set /p TOSS_SECRET="   TOSS_CLIENT_SECRET : "

echo.
echo ------------------------------------------------------------
echo  2. 한국투자증권 KIS         https://apiportal.koreainvestment.com
echo     제공: 종목별 투자자 매매동향(장중 갱신)
echo ------------------------------------------------------------
set /p KIS_ID="   KIS_APP_KEY        : "
if not "%KIS_ID%"=="" set /p KIS_SECRET="   KIS_APP_SECRET     : "

echo.
echo ------------------------------------------------------------
echo  3. Reddit API               https://www.reddit.com/prefs/apps
echo     제공: 해외 커뮤니티 (RSS 429 제한 해소) / script 타입으로 생성
echo ------------------------------------------------------------
set /p RD_ID="   REDDIT_CLIENT_ID   : "
if not "%RD_ID%"=="" set /p RD_SECRET="   REDDIT_CLIENT_SECRET: "

echo.
echo ------------------------------------------------------------
echo  4. 네이버 검색 API          https://developers.naver.com/apps/#/register
echo     제공: 뉴스 검색 (기사 수집량 증가)
echo ------------------------------------------------------------
set /p NV_ID="   NAVER_CLIENT_ID    : "
if not "%NV_ID%"=="" set /p NV_SECRET="   NAVER_CLIENT_SECRET: "

echo.
echo ------------------------------------------------------------
echo  5. 공공데이터포털           https://www.data.go.kr
echo     제공: 금융위 주식시세 (검색: 금융위원회_주식시세정보)
echo ------------------------------------------------------------
set /p DG_KEY="   DATAGO_SERVICE_KEY : "

echo.
echo ------------------------------------------------------------
echo  6. KRX Data Marketplace     http://openapi.krx.co.kr
echo ------------------------------------------------------------
set /p KRX_KEY="   KRX_AUTH_KEY       : "

echo.
python -X utf8 -c "from data_sources import credentials; import os; credentials.save({'TOSS_CLIENT_ID':os.environ.get('_T1',''),'TOSS_CLIENT_SECRET':os.environ.get('_T2',''),'KIS_APP_KEY':os.environ.get('_K1',''),'KIS_APP_SECRET':os.environ.get('_K2',''),'REDDIT_CLIENT_ID':os.environ.get('_R1',''),'REDDIT_CLIENT_SECRET':os.environ.get('_R2',''),'NAVER_CLIENT_ID':os.environ.get('_N1',''),'NAVER_CLIENT_SECRET':os.environ.get('_N2',''),'DATAGO_SERVICE_KEY':os.environ.get('_D1',''),'KRX_AUTH_KEY':os.environ.get('_X1','')})" 2>nul
set "_T1=%TOSS_ID%"
set "_T2=%TOSS_SECRET%"
set "_K1=%KIS_ID%"
set "_K2=%KIS_SECRET%"
set "_R1=%RD_ID%"
set "_R2=%RD_SECRET%"
set "_N1=%NV_ID%"
set "_N2=%NV_SECRET%"
set "_D1=%DG_KEY%"
set "_X1=%KRX_KEY%"

python -X utf8 -c "from data_sources import credentials; import os; credentials.save({k: os.environ.get(v,'') for k,v in {'TOSS_CLIENT_ID':'_T1','TOSS_CLIENT_SECRET':'_T2','KIS_APP_KEY':'_K1','KIS_APP_SECRET':'_K2','REDDIT_CLIENT_ID':'_R1','REDDIT_CLIENT_SECRET':'_R2','NAVER_CLIENT_ID':'_N1','NAVER_CLIENT_SECRET':'_N2','DATAGO_SERVICE_KEY':'_D1','KRX_AUTH_KEY':'_X1'}.items()})"

echo.
echo ============================================================
echo   저장 완료 - api_keys.json
echo ============================================================
echo.
echo   설정 현황 확인:
python -X utf8 -m data_sources.credentials

echo.
echo   각 API 연결 점검:
echo     python -m data_sources.toss_api
echo     python -m data_sources.kis_client
echo     python -m data_sources.public_apis
echo.
echo   서버를 재시작하면 공식 API가 우선 사용됩니다.
echo.
pause
