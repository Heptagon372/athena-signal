/**
 * 백엔드 통신 + 인증 토큰 관리
 *
 * 토큰은 localStorage 와 쿠키 양쪽에 둡니다.
 *   - localStorage: 클라이언트 컴포넌트가 Authorization 헤더에 실어 보냄
 *   - 쿠키: 서버 컴포넌트/미들웨어가 로그인 여부를 판단할 때 사용
 */

const TOKEN_KEY = "athena_token";

export function getToken() {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

// https 로 서비스될 때는 Secure — 세션 토큰이 평문 http 로 실려 나가면 안 됩니다
function cookieFlags() {
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  return `path=/; SameSite=Lax${secure}`;
}

export function setToken(token) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
  } catch {}
  // 미들웨어가 읽을 수 있도록 쿠키에도 (30일)
  document.cookie = `${TOKEN_KEY}=${token}; max-age=${60 * 60 * 24 * 30}; ${cookieFlags()}`;
}

export function clearToken() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {}
  document.cookie = `${TOKEN_KEY}=; max-age=0; ${cookieFlags()}`;
}

/**
 * href 에 넣어도 되는 링크만 통과시킵니다. 아니면 "" (= 링크 없이 글자만).
 *
 * 여기 들어오는 url 은 뉴스 크롤러와 커뮤니티 스크랩이 실어 보낸 값입니다 —
 * 우리가 쓴 문자열이 아닙니다. React 는 본문 텍스트는 이스케이프해 주지만
 * `javascript:` 로 시작하는 href 는 경고만 하고 넘어가는 버전이 있어서,
 * 클릭 한 번이 스크립트 실행이 되는 길을 직접 막습니다. 그 스크립트가
 * 가져갈 수 있는 것은 localStorage 의 세션 토큰 — 곧 KIS 주문 권한입니다.
 *
 * 스킴 판정 전에 제어문자를 걷어내는 이유: `java\tscript:` 처럼 사이에 탭·개행을
 * 끼우면 브라우저는 무시하고 실행하지만 순진한 검사는 통과시킵니다.
 *
 * 서버(security.safe_external_url)도 저장 단계에서 같은 검사를 합니다. 이미
 * 저장된 옛 데이터가 있어 그리는 쪽에서도 한 번 더 봅니다.
 */
export function safeUrl(url) {
  const raw = String(url ?? "").trim();
  const probe = Array.from(raw)
    .filter((c) => c.charCodeAt(0) > 32)
    .join("")
    .toLowerCase();
  // 우리 서버가 중계하는 리포트는 "/api/..." 상대경로로 옵니다.
  // "//evil.com" 은 프로토콜 상대 URL 이라 같은 출처가 아닙니다 — 제외합니다.
  if (probe.startsWith("/") && !probe.startsWith("//")) return raw;
  if (probe.startsWith("http://") || probe.startsWith("https://")) return raw;
  return "";
}

/** 종목이 실존하지 않을 때 서버가 돌려주는 404 를 구분하기 위한 에러 */
export class SymbolNotFound extends Error {
  constructor(payload) {
    super(payload?.message || "종목을 찾을 수 없습니다.");
    this.payload = payload;
    this.notFound = true;
  }
}

export class Unauthorized extends Error {
  constructor(message = "로그인이 필요합니다.") {
    super(message);
    this.unauthorized = true;
  }
}

async function request(path, { method = "GET", body, timeout = 30000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  let res;
  try {
    res = await fetch(`/api${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError") throw new Error("서버 응답이 너무 늦습니다.");
    throw new Error("분석 서버에 연결할 수 없습니다. 백엔드가 켜져 있는지 확인해 주세요.");
  }
  clearTimeout(timer);

  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }

  if (res.status === 401) throw new Unauthorized(data?.detail || data?.error);
  if (res.status === 404 && data?.error === "SYMBOL_NOT_FOUND") throw new SymbolNotFound(data);
  if (!res.ok) {
    throw new Error(data?.error || data?.detail || `요청 실패 (${res.status})`);
  }
  return data;
}

export const api = {
  get: (path, opts) => request(path, { ...opts, method: "GET" }),
  post: (path, body, opts) => request(path, { ...opts, method: "POST", body: body ?? {} }),
  put: (path, body, opts) => request(path, { ...opts, method: "PUT", body: body ?? {} }),
  del: (path, opts) => request(path, { ...opts, method: "DELETE" }),
};

/* ---------------- 표시용 포맷 ---------------- */

export function fmtNum(n, digits = 0) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
  return Number(n).toLocaleString("ko-KR", { maximumFractionDigits: digits });
}

export function fmtMoney(n, currency = "KRW") {
  if (n === null || n === undefined) return "-";
  return currency === "USD" ? `$${fmtNum(n, 2)}` : `${fmtNum(n)}원`;
}

export function fmtBigWon(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "-";
  const abs = Math.abs(n);
  if (abs >= 1e12) return `${(n / 1e12).toFixed(2)}조원`;
  if (abs >= 1e8) return `${(n / 1e8).toFixed(0)}억원`;
  if (abs >= 1e4) return `${(n / 1e4).toFixed(0)}만원`;
  return `${fmtNum(n)}원`;
}

export function signColor(v) {
  if (v > 0) return "var(--up)";
  if (v < 0) return "var(--down)";
  return "var(--muted)";
}

export const HORIZON_LABEL = {
  "10min": "10분 뒤",
  "1h": "1시간 뒤",
  "6h": "6시간 뒤",
  "12h": "12시간 뒤",
  "18h": "18시간 뒤",
  "1d": "1일 뒤",
  open: "개장",
  "5min": "+5분",
  "30min": "+30분",
  close: "마감",
};
