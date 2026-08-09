@echo off
setlocal

echo ============================================================
echo   아테나 시그널 - 구글 로그인 + MongoDB 계정 저장소 설정
echo ============================================================
echo.
echo 이 설정을 하면 아이디/비밀번호를 만들지 않고 구글 계정으로
echo 바로 로그인할 수 있습니다. 계정 정보는 MongoDB 에 저장되므로
echo 다른 PC 에서 접속해도 같은 기록을 그대로 씁니다.
echo.
echo 설정하지 않아도 기존 아이디/비밀번호 로그인은 그대로 됩니다.
echo (구글 버튼만 화면에 안 보입니다)
echo.
echo ------------------------------------------------------------
echo  [1단계] 구글 클라이언트 발급   약 5분, 무료
echo ------------------------------------------------------------
echo   1. https://console.cloud.google.com  접속 - 프로젝트 생성
echo   2. [API 및 서비스] - [OAuth 동의 화면]
echo        User Type: 외부(External) / 앱 이름: 아테나 시그널
echo        범위: email, profile, openid  (그 이상은 필요 없습니다)
echo   3. [사용자 인증 정보] - [사용자 인증 정보 만들기]
echo        - OAuth 클라이언트 ID - 웹 애플리케이션
echo   4. 승인된 리디렉션 URI 에 아래 값을 **그대로** 추가:
echo.
echo        http://localhost:8000/api/auth/google/callback
echo.
echo   5. 클라이언트 ID / 클라이언트 보안 비밀 복사
echo.
echo   ※ 글자 하나만 달라도 redirect_uri_mismatch 오류가 납니다.
echo   ※ localhost 는 https 인증서 없이도 허용됩니다.
echo.
echo ------------------------------------------------------------
echo  [2단계] MongoDB 준비
echo ------------------------------------------------------------
echo   [A] Atlas (무료, 추천 - 다른 PC 에서도 같은 계정 사용)
echo        1. https://cloud.mongodb.com - M0 무료 클러스터 생성
echo        2. Database Access - DB 사용자 생성
echo        3. Network Access - 접속할 IP 추가
echo        4. Connect - Drivers - Python - 접속 문자열 복사
echo           mongodb+srv://사용자:비번@클러스터.mongodb.net/...
echo.
echo   [B] 로컬 (혼자 쓰는 경우)
echo        MongoDB Community Server 설치 후
echo           mongodb://localhost:27017
echo.
echo ------------------------------------------------------------
echo.

set /p CLIENTID="클라이언트 ID 를 붙여넣고 Enter (건너뛰려면 그냥 Enter): "
if "%CLIENTID%"=="" goto :skipped

set /p CLIENTSECRET="클라이언트 보안 비밀 을 붙여넣고 Enter: "
if "%CLIENTSECRET%"=="" goto :skipped

echo.
set /p MONGOURI="MongoDB 접속 문자열 (비우면 mongodb://localhost:27017): "
if "%MONGOURI%"=="" set MONGOURI=mongodb://localhost:27017

setx GOOGLE_CLIENT_ID "%CLIENTID%" > nul
setx GOOGLE_CLIENT_SECRET "%CLIENTSECRET%" > nul
setx MONGODB_URI "%MONGOURI%" > nul

echo.
echo ------------------------------------------------------------
echo  [3단계] 드라이버 설치
echo ------------------------------------------------------------
echo.
python -m pip install "pymongo[srv]"

echo.
echo ============================================================
echo   설정 완료
echo ============================================================
echo.
echo 연결이 되는지 확인하려면 (새 명령 창에서):
echo.
echo     python -m storage.accounts
echo.
echo 서버를 재시작하면 로그인 화면에 "구글로 계속하기" 버튼이 생깁니다.
echo.
echo ※ 이미 열려 있는 명령 창/서버에는 반영되지 않습니다.
echo    반드시 새 창에서 서버를 다시 시작하세요. (start.bat)
echo.
goto :end

:skipped
echo.
echo 설정을 건너뛰었습니다. 기존 아이디/비밀번호 로그인은 그대로 동작합니다.
echo.

:end
pause
endlocal
