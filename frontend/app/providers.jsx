"use client";

/**
 * 전역 상태 — 로그인 사용자와 테마
 *
 * 클라이언트에서만 의미가 있어 Context 로 묶었습니다.
 * 페이지가 각각 독립된 라우트라, 새로고침해도 토큰으로 세션을 복원합니다.
 */

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { api, clearToken, getToken, setToken, Unauthorized } from "./lib/api";

const AuthContext = createContext(null);
const ThemeContext = createContext(null);
const SymbolContext = createContext(null);

/** 선택한 종목이 없을 때만 쓰는 최후의 기본값 */
export const DEFAULT_SYMBOL = { key: "삼성전자", name: "삼성전자", market: "KOSPI" };
const SYMBOL_KEY = "athena-symbol";

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth 는 Providers 안에서만 쓸 수 있습니다.");
  return ctx;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme 은 Providers 안에서만 쓸 수 있습니다.");
  return ctx;
}

/** 지금 보고 있는 종목 — 페이지를 옮겨다녀도 유지됩니다. */
export function useSymbol() {
  const ctx = useContext(SymbolContext);
  if (!ctx) throw new Error("useSymbol 은 Providers 안에서만 쓸 수 있습니다.");
  return ctx;
}

function readStoredSymbol() {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(SYMBOL_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && parsed.key ? parsed : null;
  } catch {
    return null;
  }
}

export function Providers({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  const [theme, setTheme] = useState("dark");

  // 선택 종목. 서버 렌더와 첫 클라이언트 렌더가 어긋나면 hydration 경고가 나므로,
  // 초기값은 항상 기본값으로 두고 마운트 후에 저장값으로 교체합니다.
  const [symbol, setSymbolState] = useState(DEFAULT_SYMBOL);
  const [symbolReady, setSymbolReady] = useState(false);

  useEffect(() => {
    // URL 의 ?q= 가 저장값보다 우선입니다 (링크로 받은 종목을 존중).
    let fromUrl = null;
    try {
      const q = new URLSearchParams(window.location.search).get("q");
      if (q) fromUrl = { key: q, name: q, market: "" };
    } catch {}
    const picked = fromUrl || readStoredSymbol();
    if (picked) setSymbolState(picked);
    setSymbolReady(true);
  }, []);

  const setSymbol = useCallback((next) => {
    if (!next?.key) return;
    setSymbolState(next);
    try { window.localStorage.setItem(SYMBOL_KEY, JSON.stringify(next)); } catch {}
  }, []);

  // 저장된 토큰으로 세션 복원
  useEffect(() => {
    let alive = true;
    (async () => {
      if (!getToken()) {
        if (alive) setReady(true);
        return;
      }
      try {
        const data = await api.get("/auth/me", { timeout: 10000 });
        if (alive) setUser(data.user);
      } catch (err) {
        if (err instanceof Unauthorized) clearToken();
      } finally {
        if (alive) setReady(true);
      }
    })();
    return () => { alive = false; };
  }, []);

  // 테마: 저장값 → OS 설정 순
  useEffect(() => {
    let saved = null;
    try { saved = localStorage.getItem("athena-theme"); } catch {}
    const initial = saved
      || (window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark");
    setTheme(initial);
    document.documentElement.setAttribute("data-theme", initial);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next = prev === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("athena-theme", next); } catch {}
      return next;
    });
  }, []);

  const login = useCallback(async (username, password) => {
    const data = await api.post("/auth/login", { username, password });
    setToken(data.token);
    setUser(data.user);
    return data.user;
  }, []);

  const register = useCallback(async (username, password, displayName) => {
    const data = await api.post("/auth/register", {
      username, password, display_name: displayName || username,
    });
    setToken(data.token);
    setUser(data.user);
    return data.user;
  }, []);

  /**
   * 구글 로그인 시작 — 동의 화면으로 **페이지 전체를 이동**시킵니다.
   *
   * 팝업이나 iframe 을 쓰지 않는 이유: 파이어폭스는 Total Cookie Protection 이
   * 기본값이라 서드파티 쿠키를 격리합니다. 구글 One Tap 류는 파폭에서 조용히
   * 실패합니다. 전체 페이지 이동은 그 문제를 아예 만들지 않습니다.
   *
   * origin 을 직접 실어 보내는 이유: 이 요청은 브라우저가 아니라 Next 서버가
   * 프록시로 대신 보내므로 Origin·Referer 헤더가 백엔드까지 가지 않습니다.
   * 127.0.0.1 로 들어온 사람을 localhost 로 되돌리면 localStorage 가 오리진별이라
   * 토큰이 딴 곳에 저장돼 로그인이 풀립니다.
   */
  const loginWithGoogle = useCallback(async (next = "/") => {
    const query = new URLSearchParams({ next, origin: window.location.origin });
    const data = await api.get(`/auth/google/start?${query}`, { timeout: 15000 });
    if (!data?.auth_url) throw new Error("구글 로그인 주소를 받지 못했습니다.");
    window.location.href = data.auth_url;
  }, []);

  /** 콜백 페이지에서 호출 — 1회용 핸드오프 코드를 세션 토큰으로 교환합니다. */
  const completeGoogleLogin = useCallback(async (handoff) => {
    const data = await api.post("/auth/google/exchange", { handoff });
    setToken(data.token);
    setUser(data.user);
    return data.user;
  }, []);

  const logout = useCallback(async () => {
    try { await api.post("/auth/logout"); } catch {}
    clearToken();
    setUser(null);
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      <SymbolContext.Provider value={{ symbol, setSymbol, symbolReady }}>
        <AuthContext.Provider
          value={{ user, ready, login, register, logout,
                   loginWithGoogle, completeGoogleLogin }}>
          {children}
        </AuthContext.Provider>
      </SymbolContext.Provider>
    </ThemeContext.Provider>
  );
}
