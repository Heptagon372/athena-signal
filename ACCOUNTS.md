# 계정 · 구글 로그인 · MongoDB

> 목표: **아무나 자기 구글 계정으로 바로 들어와서, 자기 데이터를 자기 것으로 쓰는 것.**
>
> 지금은 아이디/비밀번호를 직접 만들어야 하고, 그 계정은 이 PC의 `athena.db` 안에만
> 존재합니다. 다른 사람이 접속해도 계정을 새로 파야 하고, 서버를 옮기면 계정이
> 사라집니다. 구글 로그인 + MongoDB 계정 저장소로 이 두 가지를 끊습니다.

---

## 1. 지금 상태 (as-is)

정확히 알고 시작해야 뭘 건드리면 안 되는지가 정해집니다.

| 구성요소 | 위치 | 현재 동작 |
|---|---|---|
| 계정 | `storage/users.py` → SQLite `users` | 아이디/비번, PBKDF2-SHA256 20만 회 |
| 세션 | SQLite `sessions` | `secrets.token_urlsafe(32)`, 30일, 만료분은 로그인 때 청소 |
| 토큰 전달 | `frontend/app/lib/api.js` | localStorage + `athena_token` 쿠키 양쪽 |
| 세션 복원 | `frontend/app/providers.jsx` | 부팅 시 `GET /api/auth/me` |
| 사용자 식별자 | **정수 `user_id`** | `predictions` · `paper_account` · `watchlist` · 자동매매 테이블 전반의 FK |
| 화면 | Next.js `:3000` | FastAPI `:8000` 은 API 전담, `/api/*` 는 Next 라우트 핸들러가 프록시 |

핵심 제약이 하나 있습니다. **`user_id` 는 정수입니다.** 앱 데이터 전체가 이 정수를
붙들고 있습니다. Mongo 의 `ObjectId` 로 갈아타려면 스키마와 쿼리를 전부 다시 써야
하고, 이미 쌓인 예측·매매 기록의 마이그레이션이 필요합니다. 그래서 **정수 `user_id`
는 그대로 둡니다.** 이 결정이 아래 설계 전부를 지배합니다.

---

## 2. 설계 결정

### 2-1. MongoDB 는 `MONGODB_URI` 하나만 본다

코드는 접속 문자열만 압니다. 로컬이든 Atlas 든 구분하지 않습니다.

```
mongodb://localhost:27017                             # 로컬 mongod
mongodb+srv://user:pw@cluster0.xxxx.mongodb.net/...   # Atlas 무료 M0
```

로컬로 시작했다가 Atlas 로 옮길 때 코드 수정이 0 입니다. Atlas 의 `mongodb+srv://`
는 DNS SRV 조회가 필요하므로 의존성은 `pymongo[srv]` (= pymongo + dnspython)로
잡습니다. 지금 이 PC에는 pymongo·dnspython·mongod 모두 없습니다 — 설치가 4장의
설정 절차에 들어갑니다.

### 2-2. Mongo 는 "신원", SQLite 는 "앱 데이터"

| | MongoDB | SQLite (`athena.db`) |
|---|---|---|
| 담는 것 | 구글 계정 프로필, 세션, OAuth 임시 상태 | 예측 기록, 모의투자 계좌, 관심종목, 자동매매 |
| 키 | `google_sub` (구글이 주는 불변 식별자) | 정수 `user_id` |
| 잇는 값 | 문서 안의 `user_id` 정수 | 같은 정수 |

Mongo 문서 하나가 "이 구글 계정은 `user_id=100003` 이다" 를 말해주고, 그 뒤부터는
기존 코드가 하던 대로 정수로 굽니다. 새로 짜는 코드는 신원 확인 구간뿐입니다.

### 2-3. `user_id` 발급은 Mongo 의 원자적 카운터로

이게 미묘하지만 중요합니다. SQLite `AUTOINCREMENT` 로 발급하면, **여러 사람이 각자
자기 PC에서 앱을 띄우고 같은 Atlas 클러스터를 바라볼 때 같은 구글 계정이 PC마다 다른
`user_id` 를 받습니다.** 계정이 안 따라옵니다.

그래서 `user_id` 는 Mongo `counters` 문서에서 `find_one_and_update($inc)` 로 발급합니다.
클러스터를 공유하는 모든 인스턴스가 같은 번호를 봅니다.

기존 로컬 계정의 SQLite `AUTOINCREMENT` id 와 부딪히면 안 되므로 **카운터는 100000
에서 시작**합니다. 로컬 계정이 10만 개가 되는 일은 없으니 충돌하지 않습니다.
발급한 뒤에는 SQLite `users` 에 `INSERT OR IGNORE` 로 같은 id 의 행을 하나 만듭니다 —
FK 대상이 실제로 있어야 하고, `paper.ensure_account(user_id)` 도 그 행을 전제합니다.

### 2-4. 기존 아이디/비번 로그인은 그대로 둔다

없애지 않습니다. 로그인 화면에 "구글로 계속하기" 가 **추가**됩니다.

- 로컬 계정 세션 → SQLite `sessions` (지금 그대로)
- 구글 계정 세션 → Mongo `sessions` (TTL 인덱스로 자동 만료)
- `current_user()` 가 **Mongo 를 먼저, 없으면 SQLite** 를 봅니다

세션을 Mongo 에 두는 이유는 계정과 같습니다 — 클러스터를 공유하면 세션도 같이
따라옵니다. TTL 인덱스를 걸면 만료 청소를 우리가 안 해도 됩니다.

**계정 연결(link):** 이미 `username` 을 자기 구글 이메일로 만들어 둔 로컬 계정이
있으면, 그 계정의 `user_id` 를 재사용합니다. 즉 이메일이 같으면 기존 기록에
그대로 들어갑니다. 별도 UI 없이 이메일 일치만으로 판단합니다.

### 2-5. Mongo 가 없으면 구글 로그인만 조용히 꺼진다

지금 이 앱은 **외부 설정 없이도 돌아갑니다.** 그 성질을 깨면 안 됩니다.

`MONGODB_URI` 나 구글 클라이언트 ID 가 없으면:
- `GET /api/auth/google/config` 가 `{"configured": false, "reason": ...}` 를 돌려주고
- 로그인 화면은 구글 버튼을 **아예 그리지 않고**
- 아이디/비번 로그인은 전혀 영향 없이 계속 동작합니다

Mongo 연결이 도중에 끊겨도 구글 로그인만 실패하고, 로컬 로그인·분석·자동매매는
멀쩡해야 합니다. 그래서 `pymongo` 는 **모듈 최상단에서 import 하지 않고** 필요할 때
지연 import 합니다. 라이브러리가 없는 PC에서 `api.py` 가 아예 안 뜨는 사고를 막습니다.

---

## 3. 인증 흐름

### 3-1. 왜 리다이렉트 흐름인가 — 파이어폭스

파폭에서 쓰려면 이 선택이 강제됩니다.

구글이 미는 **Google Identity Services (One Tap / `<div id="g_id_onload">`)** 는
서드파티 쿠키와 iframe 에 의존합니다. 파이어폭스는 **Total Cookie Protection** 이
기본값이라 서드파티 쿠키를 오리진별로 격리합니다. One Tap 은 파폭에서 안 뜨거나
조용히 실패합니다. FedCM 도 파폭 지원이 확정적이지 않습니다.

**표준 OAuth 2.0 Authorization Code 흐름(전체 페이지 이동)은 서드파티 쿠키를 쓰지
않습니다.** 브라우저가 `accounts.google.com` 으로 직접 이동하니 거기서는 구글이
1st-party 입니다. 파폭·크롬·사파리 전부 동일하게 동작합니다. 그래서 이걸 씁니다.

### 3-2. `redirect_uri` 를 8000번으로 두는 이유

Next 프록시(`frontend/app/api/[...path]/route.js`)는 응답에서 `content-type` 과
`cache-control` **두 개만** 되돌려줍니다. `Location` 과 `Set-Cookie` 는 버립니다.
`redirect_uri` 를 `:3000/api/...` 로 잡으면 구글의 콜백 리다이렉트가 프록시에서
사라집니다.

그래서 `redirect_uri` 는 **`http://localhost:8000/api/auth/google/callback`** —
브라우저가 FastAPI 로 직접 갑니다. 프록시를 타지 않으니 302 가 온전합니다.

반대로 **start 는 JSON 으로** 돌려줍니다. 302 를 보내면 프록시가 먹으니까,
`{"auth_url": ...}` 를 주고 프론트가 `location.href` 로 이동합니다. 프록시는
JSON 은 잘 통과시킵니다. 흐름의 두 방향이 서로 다른 이유가 이것입니다.

### 3-3. 세션 토큰을 URL 에 싣지 않는다 — 핸드오프 코드

콜백 시점에 세션 토큰이 서버에 있고, 그걸 브라우저(localStorage)에 넘겨야 합니다.
`?token=...` 으로 붙이면 브라우저 히스토리·`Referer`·서버 로그에 30일짜리 세션
토큰이 남습니다. 프래그먼트(`#token=`)는 서버엔 안 가지만 히스토리엔 남습니다.

그래서 **일회용 핸드오프 코드**를 씁니다.

1. 콜백이 세션 토큰을 만들고, 별도의 랜덤 `handoff` 코드를 **60초 TTL·1회용**으로
   Mongo 에 저장 (`handoff → token` 매핑)
2. `:3000/auth/callback?handoff=<code>` 로 302
3. 프론트가 `POST /api/auth/google/exchange {handoff}` → 진짜 토큰을 응답 본문으로 받음
4. 서버는 그 즉시 핸드오프 문서를 삭제 (`find_one_and_delete` — 원자적 1회용)

긴 수명의 토큰은 URL 에 한 번도 등장하지 않습니다. 유출돼도 60초·1회용입니다.

### 3-4. 전체 순서

```
[브라우저]                    [FastAPI :8000]              [구글]        [MongoDB]

로그인 화면
  "구글로 계속하기" 클릭
      │
      ├─ GET /api/auth/google/start ──────►
      │   (Next :3000 프록시 경유, JSON)
      │                          state 생성 + PKCE verifier
      │                          호출한 프론트 오리진 기록 ──────────────► oauth_state
      │                                                                    (10분 TTL)
      ◄── {auth_url} ────────────┤
      │
      ├─ location.href = auth_url ──────────────────────► 구글 동의 화면
      │                                                        │
      │        ◄───────── 302 ?code=&state= ───────────────────┘
      │            (redirect_uri = :8000 — 프록시 안 탐)
      ├─ GET /api/auth/google/callback?code=&state= ──►
      │                          state 검증 (1회용 소비) ◄──────────────► oauth_state
      │                          code + verifier ──────► /token
      │                          ◄──── id_token, access_token ────┘
      │                          id_token 검증 (iss/aud/exp/email_verified)
      │                          userinfo 교차 확인 ──► /userinfo
      │
      │                          구글 계정 조회 or 신규 ◄────────────────► accounts
      │                            신규면: user_id 발급 (counters $inc)
      │                                    SQLite users 행 INSERT OR IGNORE
      │                                    paper.ensure_account(user_id)
      │                            기존이면: 프로필·last_login_at 갱신
      │                          세션 토큰 발급 ────────────────────────► sessions
      │                          핸드오프 코드 발급 ──────────────────► handoffs
      ◄── 302 :3000/auth/callback?handoff=… ─┤                           (60초 TTL)
      │
/auth/callback 페이지
      ├─ POST /api/auth/google/exchange {handoff} ──►
      │                          핸드오프 소비 (find_one_and_delete)
      ◄── {token, user} ─────────┤
      │
      setToken() → 세션 복원 → next 경로로 이동
```

### 3-5. 오리진 보존

사용자가 `127.0.0.1:3000` 으로 들어왔는데 콜백이 `localhost:3000` 으로 되돌리면,
**localStorage 는 오리진별이라 토큰이 딴 곳에 저장됩니다.** 로그인한 것처럼 보였다가
새로고침하면 풀립니다.

그래서 start 가 요청의 `Origin`/`Referer` 에서 프론트 오리진을 읽어 `oauth_state`
문서에 함께 저장하고, 콜백은 **그 오리진으로** 되돌립니다. 허용 목록
(`localhost`/`127.0.0.1` + 설정된 오리진)으로 제한해 오픈 리다이렉트를 막습니다.

---

## 4. 데이터 모델 (MongoDB)

DB 이름은 `MONGODB_DB` (기본 `athena`).

### `accounts` — 계정 본체
```js
{
  _id: ObjectId,
  provider: "google",
  google_sub: "1174…",          // 구글의 불변 식별자. 이메일이 바뀌어도 안 바뀜
  user_id: 100003,              // ★ SQLite 앱 데이터와 잇는 정수
  email: "someone@gmail.com",
  email_verified: true,
  display_name: "홍길동",
  picture: "https://lh3.googleusercontent.com/…",
  locale: "ko",
  created_at: ISODate,
  last_login_at: ISODate,
  login_count: 7,
  linked_local_user_id: 4 | null // 이메일이 같은 로컬 계정을 물려받은 경우
}
```
- 유니크 인덱스: `google_sub`, `user_id`
- 이메일이 아니라 `google_sub` 로 찾습니다. 이메일은 바뀔 수 있고, 바뀌었다고
  계정이 새로 생기면 그 사람 기록이 통째로 사라져 보입니다.

### `sessions` — 세션
```js
{
  token: "…",  user_id: 100003,  account_id: ObjectId,
  created_at: ISODate,  expires_at: ISODate,     // 30일
  user_agent: "Mozilla/5.0 …"                    // 어디서 로그인했는지 확인용
}
```
- 유니크 인덱스 `token`, **TTL 인덱스 `expires_at`** (`expireAfterSeconds: 0`)

### `oauth_state` — CSRF/PKCE 임시 상태
```js
{
  state: "…", code_verifier: "…", nonce: "…",
  redirect_origin: "http://localhost:3000",
  next: "/paper", created_at: ISODate, expires_at: ISODate   // 10분
}
```
- TTL 인덱스 `expires_at`. 콜백에서 `find_one_and_delete` 로 **1회용** 소비
- `nonce` 는 `id_token` 안에 되돌아오는 값과 대조합니다 (id_token 재사용 차단)

### `handoffs` — 토큰 인계용 일회용 코드
```js
{ code: "…", token: "…", expires_at: ISODate }   // 60초, TTL 인덱스
```

### `counters` — `user_id` 발급
```js
{ _id: "user_id", seq: 100003 }
```

---

## 5. API 명세

| 메서드 | 경로 | 인증 | 응답 |
|---|---|---|---|
| GET | `/api/auth/google/config` | — | `{configured, reason?}` — 버튼 표시 여부 판단 |
| GET | `/api/auth/google/start?next=/paper` | — | `{auth_url}` **JSON** (프록시 통과용) |
| GET | `/api/auth/google/callback?code=&state=` | — | **302** → `<origin>/auth/callback?handoff=…` |
| POST | `/api/auth/google/exchange` | — | `{token, user}` — 핸드오프 1회 소비 |

실패는 302 로 `/auth/callback?error=<사유>` 로 보냅니다. 콜백은 브라우저가 직접
여는 화면이라 JSON 에러를 띄우면 사용자가 날 JSON 을 보게 됩니다.

기존 엔드포인트는 시그니처가 안 바뀝니다. `current_user()` 만 Mongo 세션을 먼저
보도록 확장하므로, `/api/auth/me` · `/api/watchlist` · 모의투자 · 자동매매가 구글
계정에서도 그대로 동작합니다.

---

## 6. 파일별 작업

| 파일 | 상태 | 내용 |
|---|---|---|
| `storage/accounts.py` | **신규** | Mongo 연결(지연 import), 인덱스 보장, `user_id` 카운터, 계정 upsert/조회, 세션·핸드오프 |
| `data_sources/google_oauth.py` | **신규** | authorize URL(state+PKCE S256), 코드 교환, `id_token` 검증, userinfo 교차 확인 |
| `api.py` | 수정 | 엔드포인트 4개 추가, `current_user()`·`auth_logout()` 확장 |
| `storage/users.py` | 수정 | `find_by_username()` · `ensure_external_user()`, 비밀번호 없는 계정의 로그인 차단 |
| `data_sources/http_client.py` | 수정 | `post_form_full()` — 토큰 엔드포인트의 400 본문(`invalid_grant` 등)을 읽기 위해 |
| `data_sources/credentials.py` | 수정 | `PROVIDERS` 에 `google` · `mongo` 등록 → 설정 현황에 노출 |
| `frontend/app/providers.jsx` | 수정 | `loginWithGoogle()` · `completeGoogleLogin()` |
| `frontend/app/login/page.jsx` | 수정 | "구글로 계속하기" (미설정이면 미표시) |
| `frontend/app/auth/callback/page.jsx` | **신규** | 핸드오프 교환 → 토큰 저장 → `next` 이동 |
| `frontend/app/pages.css` | 수정 | 구글 버튼·구분선 스타일 (라이트/다크 양쪽) |
| `api.py` (라우팅) | 수정 | `/auth/callback` 을 프론트로 넘기는 브리지 |
| `requirements.txt` | 수정 | `pymongo[srv]` |
| `setup_google.bat` | **신규** | 클라이언트 ID/시크릿·Mongo URI 입력 → `api_keys.json` |
| `tests/test_google_auth.py` | **신규** | 오프라인 검사 (아래 8장) |

---

## 7. 설정 (다른 사람이 붙이는 절차)

### 7-1. 구글 클라우드 콘솔

1. <https://console.cloud.google.com> → 프로젝트 생성
2. **API 및 서비스 → OAuth 동의 화면**: 외부(External), 앱 이름 `아테나 시그널`,
   범위는 `email` · `profile` · `openid` 만. 테스트 모드면 테스터에 쓸 계정을 등록
3. **사용자 인증 정보 → OAuth 클라이언트 ID → 웹 애플리케이션**
   - 승인된 리디렉션 URI: **`http://localhost:8000/api/auth/google/callback`**
   - `127.0.0.1` 로도 접속한다면 `http://127.0.0.1:8000/api/auth/google/callback` 도 추가
   - (`localhost` 는 http 로도 허용됩니다 — 인증서 없이 됩니다)
4. 클라이언트 ID / 시크릿 확보

### 7-2. MongoDB

**Atlas (다른 PC에서도 같은 계정을 쓰려면 이쪽)**
1. <https://cloud.mongodb.com> → 무료 M0 클러스터
2. Database Access 에서 DB 사용자 생성
3. Network Access 에 접속할 IP 추가
4. Connect → Drivers → Python 접속 문자열 복사

**로컬 (혼자 쓰면 이쪽)**
1. MongoDB Community Server 설치 → 서비스로 자동 실행
2. URI 는 `mongodb://localhost:27017`

### 7-3. 키 넣기

```bash
python -m pip install "pymongo[srv]"
```

```bash
setup_google.bat
```

또는 직접 `api_keys.json` / 환경변수:

| 키 | 예 |
|---|---|
| `GOOGLE_CLIENT_ID` | `1234-abc.apps.googleusercontent.com` |
| `GOOGLE_CLIENT_SECRET` | `GOCSPX-…` |
| `GOOGLE_REDIRECT_URI` | (선택) 기본 `http://localhost:8000/api/auth/google/callback` |
| `MONGODB_URI` | `mongodb://localhost:27017` 또는 `mongodb+srv://…` |
| `MONGODB_DB` | (선택) 기본 `athena` |

`api_keys.json` 과 `.env` 는 이미 `.gitignore` 에 있습니다. **시크릿은 커밋되지
않습니다.** 4번 항목의 지연 import 덕분에, 키를 안 넣으면 구글 버튼만 안 보이고
나머지는 지금과 똑같이 돕니다.

---

## 8. 검증 계획

```bash
python tests/test_google_auth.py
```

네트워크도 MongoDB도 없이 돕니다 (**79개 검사, 전부 통과**). 실제 서버를 띄우지
않고 엔드포인트 함수를 직접 부릅니다 — 이 PC 는 실계좌에 붙어 있어서, 검사용으로
두 번째 인스턴스를 띄우면 `startup` 이 자동매매 루프를 하나 더 시작합니다.

1. **authorize URL** — `response_type=code`, `code_challenge_method=S256`,
   `scope` 는 `openid email profile` 뿐, `redirect_uri` 일치,
   `client_secret` 이 URL 에 없음(히스토리에 남으므로)
2. **PKCE** — `S256` 변환이 RFC 7636 부록 B 예시와 일치, 패딩 없음
3. **`id_token` 검증** — `aud`/`iss` 불일치, `exp` 만료, `nonce` 불일치,
   `email_verified: false`, `sub`/`email` 누락이 각각 거부되는지
4. **오픈 리다이렉트** — `http://evil.com`·`javascript:` 는 기본 오리진으로 대체,
   `127.0.0.1` 은 **그대로 유지**(토큰이 딴 오리진에 저장되면 로그인이 풀림),
   `next` 의 `//evil.com`·절대 URL 거부
5. **1회용 규칙** — 같은 `state`·같은 핸드오프의 두 번째 사용이 실패,
   만료된 핸드오프는 문서가 남아 있어도 거부(TTL 인덱스 지연 구간 방어)
6. **전체 흐름** (가짜 Mongo + 구글 응답 스텁) — 콜백 302, URL 에 세션 토큰이 아니라
   `handoff` 만 실림, 계정이 Mongo 에 저장, `user_id` 가 오프셋 위에서 발급,
   SQLite 에 같은 id 의 행 생성, 핸드오프 교환, **재로그인 시 계정이 하나 유지되고
   같은 `user_id`** (= "계정을 불러온다"), 취소·잘못된 `state`·재생 공격 거부
7. **미설정 폴백** — 키가 비면 `configured == false`, Mongo 없이 `user_from_token`
   이 예외 대신 `None`, 그리고 **아이디/비번 가입·로그인·세션 복원이 정상 동작**
   (회귀 방지의 핵심). 비밀번호 없는 구글 전용 계정은 어떤 비밀번호로도 못 들어옴

검사는 이 PC 의 실제 키 설정과 무관하게 같은 결과를 냅니다 (`credentials.get` 을
가로채므로, `setup_google.bat` 을 돌린 뒤에도 깨지지 않습니다).

수동 확인 (실제 구글 계정 필요 — 여기까지는 자동 검사로 대신할 수 없습니다):
- 파이어폭스에서 로그인 → 계정 생성 → 모의투자 계좌가 자동 개설되는지
- 로그아웃 후 재로그인 → **같은 `user_id`, 기록 유지** (= "계정을 불러온다")
- Atlas 라면 다른 브라우저/PC 에서 같은 구글 계정으로 → 같은 데이터
- 이메일이 같은 로컬 계정이 있던 경우 → 기존 기록으로 들어가는지

---

## 9. 범위 밖 (지금 안 함)

- **예측·매매 기록의 Mongo 이관** — `user_id` 정수를 쓰는 SQLite 그대로 둡니다 (2장)
- **`id_token` 서명 검증** — 코드 교환을 우리가 연 TLS 채널로 구글 토큰 엔드포인트에
  직접 하므로 채널 자체가 출처를 보장합니다(OpenID Connect Core §3.1.3.7 주석).
  `iss`/`aud`/`exp`/`email_verified` 는 검증하고 userinfo 로 교차 확인합니다.
  서명까지 검증하려면 `cryptography` + JWKS 캐시가 필요한데, **id_token 을 제3자에게서
  받는 구조로 바뀌면 그때는 반드시 추가해야 합니다.**
- **리프레시 토큰 보관** — 구글 API 를 대신 호출할 일이 없어 `access_token` 을 저장하지
  않습니다. 로그인 확인용으로 한 번 쓰고 버립니다. 저장 안 하는 게 유출면이 적습니다.
- **HTTPS·공개 배포** — 현재 전제는 localhost 입니다. 외부에 열려면 도메인·TLS·
  `Secure`/`HttpOnly` 쿠키·리디렉션 URI 재등록이 별도 작업으로 필요합니다.
- 다른 소셜 로그인, 계정 삭제/이메일 변경 UI, 역할·권한 분리.

---

## 10. 위험과 대응

| 위험 | 대응 |
|---|---|
| `pymongo` 없는 PC 에서 서버가 안 뜬다 | 지연 import + `configured=false` 폴백 (2-5) |
| 파폭에서 One Tap 이 안 뜬다 | 애초에 안 씀 — 전체 페이지 리다이렉트 (3-1) |
| Next 프록시가 `Location` 을 먹는다 | start 는 JSON, `redirect_uri` 는 8000번 직접 (3-2) |
| 세션 토큰이 URL·히스토리에 남는다 | 60초·1회용 핸드오프 코드 (3-3) |
| `127.0.0.1` ↔ `localhost` 로 토큰이 흩어진다 | 시작 오리진을 state 에 보존 (3-5) |
| PC 마다 `user_id` 가 달라진다 | Mongo 카운터로 발급 (2-3) |
| 기존 로컬 계정 id 와 충돌 | 카운터 100000 오프셋 (2-3) |
| CSRF / 인가 코드 가로채기 | `state` 1회용 + PKCE S256 + `nonce` 대조 |
| 오픈 리다이렉트 | `redirect_origin` 허용 목록 (3-5) |
| 구글에서 이메일을 바꿨다 | `google_sub` 로 조회 (4장) |
