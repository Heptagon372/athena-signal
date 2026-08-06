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

export function setToken(token) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
  } catch {}
  // 미들웨어가 읽을 수 있도록 쿠키에도 (30일)
  document.cookie = `${TOKEN_KEY}=${token}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
}

export function clearToken() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {}
  document.cookie = `${TOKEN_KEY}=; path=/; max-age=0; SameSite=Lax`;
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
  del: (path, opts) => request(path, { ...opts, method: "DELETE" }),
};

/* ---------------- 표시용 포맷 ---------------- */

export function fmtNum(n, digits = 0) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString("ko-KR", { maximumFractionDigits: digits });
}

export function fmtMoney(n, currency = "KRW") {
  if (n === null || n === undefined) return "—";
  return currency === "USD" ? `$${fmtNum(n, 2)}` : `${fmtNum(n)}원`;
}

export function fmtBigWon(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
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
