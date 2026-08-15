# 공개 배포 — 오라클 클라우드

> 목표: **아테나 시그널을 인터넷에 열어 다른 사람도 계정을 만들어 쓰게 하되,
> 서버와 사용자들의 API 키·계좌를 외부에서 건드릴 수 없게.**
>
> 계정·설정창·키 암호화는 이미 앱 안에 있습니다 (ACCOUNTS.md). 이 문서는
> 그것을 오라클 서버에 올리고 바깥 문단속을 하는 절차입니다.

## 0. 구조 — 무엇이 어디에 열리는가

```
인터넷 ──443(https)──► nginx ──► /api·/autotrade·/static ─► uvicorn :8000 (127.0.0.1)
            80 은 443 으로 넘김 └► 나머지 화면 ───────────► Next.js  :3000 (127.0.0.1)

외부에 열리는 포트: 80, 443, (22는 내 IP 만)   ← 그 외 전부 차단
8000·3000 은 127.0.0.1 에만 묶임               ← nginx 를 거치지 않고는 닿을 수 없음
```

| 층 | 지키는 것 | 어디서 |
|---|---|---|
| 오라클 보안 목록 + 서버 방화벽 | 80/443 외 전부 차단 | 3장 |
| nginx + Let's Encrypt | 전 구간 암호화, X-Forwarded-For 위조 차단 | 6장 |
| 레이트 리밋 | 로그인·가입 무차별 대입 | `security.py` (자동) |
| 공개 서버 모드 | **모든** 계정이 KIS 주문에 자기 키·자기 계좌 필수 | `ATHENA_PUBLIC_ORIGIN` 설정 시 자동 |
| Fernet 암호화 | Atlas 유출 ≠ 계좌 유출 | 이미 내장 (ACCOUNTS.md 0-B-3) |

## 1. 준비물

- 오라클 클라우드 계정 (Always Free 의 A1.Flex 면 충분합니다)
- 도메인 하나 (예: `athena.example.com`) — Let's Encrypt 인증서와 Firebase
  승인 도메인에 필요합니다. IP 만으로는 https 를 제대로 못 겁니다.
- MongoDB Atlas (무료 M0) — 계정·세션·사용자별 키 저장소
- Firebase 프로젝트 — 구글 로그인 (ACCOUNTS.md 0-5)

## 2. VM 만들기

1. 오라클 콘솔 → Compute → Instance 생성
   - 이미지: **Ubuntu 24.04**, Shape: A1.Flex (2 OCPU / 12GB 권장, 1/6 도 동작)
   - SSH 공개키 등록, 공용 IP 할당
2. 도메인의 DNS A 레코드를 이 공용 IP 로

## 3. 네트워크 문단속 (오라클은 두 겹입니다)

**① 오라클 보안 목록(Security List)** — VCN → 서브넷 → Security List:

| 방향 | 포트 | 소스 | 비고 |
|---|---|---|---|
| Ingress | 22 | **내 집/사무실 IP/32** | 전세계에 열지 마세요 |
| Ingress | 80, 443 | 0.0.0.0/0 | 웹 |

8000·3000 은 **추가하지 않습니다.**

**② 서버 안 iptables** — 오라클 우분투 이미지는 자체 iptables 규칙이 22 외
전부를 REJECT 합니다 (보안 목록에서 열었는데 접속이 안 되는 단골 원인):

```bash
sudo iptables -I INPUT 5 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

## 4. 소프트웨어 + 코드

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip nginx git
# Node 20 (Next.js 16 요구사항)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install -y nodejs

# 서비스 전용 사용자 — root 로 돌리지 않습니다
sudo useradd -r -m -d /opt/athena-signal -s /usr/sbin/nologin athena

# 코드 올리기 (GitHub 사용 시; scp 로 통째로 복사해도 됩니다)
sudo -u athena git clone <저장소주소> /opt/athena-signal
cd /opt/athena-signal

sudo -u athena python3 -m venv .venv
sudo -u athena .venv/bin/pip install -r requirements.txt

cd frontend && sudo -u athena npm ci && sudo -u athena npm run build && cd ..
```

## 5. 서버 설정 — `api_keys.json`

서버에는 **서버 인프라 키만** 둡니다. `/opt/athena-signal/api_keys.json`:

```json
{
  "MONGODB_URI": "mongodb+srv://...(Atlas 접속 문자열)",
  "FIREBASE_PROJECT_ID": "...",
  "FIREBASE_API_KEY": "AIzaSy...",
  "ATHENA_PUBLIC_ORIGIN": "https://athena.example.com"
}
```

```bash
sudo chown athena:athena api_keys.json && sudo chmod 600 api_keys.json
```

- **`ATHENA_PUBLIC_ORIGIN` 이 공개 모드 스위치입니다.** 설정하면
  ① CORS·OAuth 오리진 허용 목록에 들어가고 ② **로컬(아이디/비번) 계정도
  KIS 주문에 자기 키·자기 계좌가 필수**가 됩니다. 공개 서버에서는 가입이
  누구에게나 열려 있으므로, 이 값 없이 열면 방금 가입한 외부인이 서버
  공용 키로 주문을 낼 수 있습니다. **반드시 설정하세요.**
- 같은 이유로 공개 서버의 `api_keys.json` 에는 **KIS 키·계좌를 넣지 마세요.**
  시세는 토스·공공데이터 등 공용 키로 충분하고, 주문 키는 각자 웹 설정
  (`/settings`)에서 자기 것을 넣는 구조입니다.
- `ATHENA_CRED_KEY` 는 첫 키 저장 때 자동 생성됩니다 — 사용자 키 암호화의
  마스터 키이므로 **이 파일을 백업**해 두세요. 잃으면 모든 사용자가 키를
  다시 입력해야 합니다 (계좌가 털리는 건 아니고, 다시 입력하면 됩니다).

바깥 서비스 쪽 등록 두 가지:

1. **Atlas** → Network Access 에 서버 공용 IP 추가
2. **Firebase** → Authentication → 설정 → 승인된 도메인에 `athena.example.com` 추가

## 6. 서비스 등록 + nginx + 인증서

```bash
sudo cp deploy/athena-api.service deploy/athena-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now athena-api athena-web

# nginx — 파일 안의 athena.example.com 을 실제 도메인으로 바꾼 뒤
sudo cp deploy/nginx-athena.conf /etc/nginx/sites-available/athena
sudo ln -s /etc/nginx/sites-available/athena /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# TLS (자동 갱신 포함)
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d athena.example.com
```

## 7. 확인 체크리스트

```bash
# 화면·API 가 https 로 뜨는가
curl -sI https://athena.example.com | head -3
curl -s https://athena.example.com/api/status | head -1

# 8000/3000 이 밖에서 닫혀 있는가 (둘 다 실패해야 정상)
curl -m 3 http://<공용IP>:8000/ ; curl -m 3 http://<공용IP>:3000/

# 레이트 리밋 — 로그인을 11번 두드리면 429
for i in $(seq 11); do curl -s -o /dev/null -w "%{http_code} " -X POST \
  https://athena.example.com/api/auth/login \
  -H 'Content-Type: application/json' -d '{"username":"x","password":"y"}'; done
```

브라우저에서: 회원가입 → `/settings` 에서 KIS 키·계좌 저장 → 마스킹(····1234)만
보이는지 → 자동매매는 자기 키 없이는 "본인 KIS 키가 필요합니다"로 막히는지.

## 8. 업데이트

```bash
cd /opt/athena-signal
sudo -u athena git pull
sudo -u athena .venv/bin/pip install -r requirements.txt
cd frontend && sudo -u athena npm ci && sudo -u athena npm run build && cd ..
sudo systemctl restart athena-api athena-web
```

## 9. 하지 말아야 할 것

| 하지 말 것 | 이유 |
|---|---|
| 보안 목록·iptables 에 8000/3000 열기 | nginx(레이트 리밋 헤더·TLS)를 우회한 직통 경로가 생깁니다 |
| 공개 서버 `api_keys.json` 에 KIS 키·계좌 넣기 | 로컬 계정 폴백이 없어도, 서버가 뚫리면 파일째 계좌 권한입니다 |
| `ATHENA_PUBLIC_ORIGIN` 없이 공개 | 외부인이 서버 키로 주문 가능 (5장) |
| nginx 없이 `ATHENA_BEHIND_PROXY=1` | X-Forwarded-For 위조로 레이트 리밋이 우회됩니다 |
| root 로 서비스 실행 | 앱이 뚫리면 서버 전체가 넘어갑니다 |
| `uvicorn --host 0.0.0.0` | 8000 직통 개방과 같습니다 — 유닛 파일 그대로 두세요 |
