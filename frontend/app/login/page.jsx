"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useAuth } from "../providers";

/** 구글 로고 — 외부 이미지를 불러오지 않도록 인라인 SVG 로 둡니다. */
function GoogleMark() {
  return (
    <svg width="17" height="17" viewBox="0 0 18 18" aria-hidden="true">
      <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.91c1.7-1.57 2.69-3.88 2.69-6.62z" />
      <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.91-2.26c-.81.54-1.84.86-3.05.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.34A8.99 8.99 0 0 0 9 18z" />
      <path fill="#FBBC05" d="M3.97 10.72a5.41 5.41 0 0 1 0-3.44V4.94H.96a9 9 0 0 0 0 8.12l3.01-2.34z" />
      <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.59C13.46.89 11.43 0 9 0A8.99 8.99 0 0 0 .96 4.94l3.01 2.34C4.68 5.16 6.66 3.58 9 3.58z" />
    </svg>
  );
}

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { user, ready, login, register, loginWithGoogle } = useAuth();

  const [mode, setMode] = useState("login");     // login | register
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  // null = 아직 확인 중. 확인 전에는 버튼을 그리지 않아 깜빡임을 막습니다.
  const [googleReady, setGoogleReady] = useState(null);

  const next = params.get("next") || "/";

  // 이미 로그인돼 있으면 돌려보냅니다
  useEffect(() => {
    if (ready && user) router.replace(next);
  }, [ready, user, next, router]);

  // 구글 로그인이 설정돼 있는지 — 안 됐으면 버튼을 아예 그리지 않습니다.
  // 눌러도 안 되는 버튼을 보여주는 것보다 없는 게 낫습니다.
  useEffect(() => {
    let alive = true;
    api.get("/auth/google/config", { timeout: 8000 })
      .then((data) => { if (alive) setGoogleReady(!!data?.configured); })
      .catch(() => { if (alive) setGoogleReady(false); });
    return () => { alive = false; };
  }, []);

  const startGoogle = async () => {
    setError("");
    setBusy(true);
    try {
      await loginWithGoogle(next);
      // 성공하면 페이지가 구글로 이동하므로 여기 아래는 실행되지 않습니다
    } catch (err) {
      setError(err.message || "구글 로그인을 시작할 수 없습니다.");
      setBusy(false);
    }
  };

  const submit = async (e) => {
    e.preventDefault();
    setError("");

    if (mode === "register" && password !== passwordConfirm) {
      setError("비밀번호가 서로 다릅니다.");
      return;
    }

    setBusy(true);
    try {
      if (mode === "login") await login(username.trim(), password);
      else await register(username.trim(), password, displayName.trim());
      router.replace(next);
    } catch (err) {
      setError(err.message || "요청에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth-wrap">
      <div className="auth-card">
        <div className="eyebrow">{mode === "login" ? "로그인" : "회원가입"}</div>
        <h1 className="auth-title">
          {mode === "login" ? "다시 오셨군요" : "계정을 만듭니다"}
        </h1>
        <p className="auth-lede">
          계정마다 <strong>예측 기록·성적표·모의투자 계좌</strong>가 따로 저장됩니다.
          같은 종목을 봐도 성적과 잔고는 계정별로 독립적입니다.
        </p>

        {googleReady && (
          <>
            <button type="button" className="btn-google" onClick={startGoogle} disabled={busy}>
              <GoogleMark />
              <span>구글로 계속하기</span>
            </button>
            <div className="auth-or"><span>또는 아이디로</span></div>
          </>
        )}

        <form onSubmit={submit} className="auth-form">
          <label className="field">
            <span>아이디</span>
            <input className="input" value={username} autoComplete="username"
                   onChange={(e) => setUsername(e.target.value)}
                   placeholder="3자 이상 · 한글/영문/숫자" required />
          </label>

          {mode === "register" && (
            <label className="field">
              <span>표시 이름 <em>(선택)</em></span>
              <input className="input" value={displayName}
                     onChange={(e) => setDisplayName(e.target.value)}
                     placeholder="비우면 아이디를 씁니다" />
            </label>
          )}

          <label className="field">
            <span>비밀번호</span>
            <input className="input" type="password" value={password}
                   autoComplete={mode === "login" ? "current-password" : "new-password"}
                   onChange={(e) => setPassword(e.target.value)}
                   placeholder="6자 이상" required />
          </label>

          {mode === "register" && (
            <label className="field">
              <span>비밀번호 확인</span>
              <input className="input" type="password" value={passwordConfirm}
                     autoComplete="new-password"
                     onChange={(e) => setPasswordConfirm(e.target.value)} required />
            </label>
          )}

          {error && <div className="auth-error">{error}</div>}

          <button className="btn btn-gold auth-submit" type="submit" disabled={busy}>
            {busy ? "처리 중…" : mode === "login" ? "로그인" : "가입하고 시작하기"}
          </button>
        </form>

        <div className="auth-switch">
          {mode === "login" ? (
            <>계정이 없으신가요?{" "}
              <button onClick={() => { setMode("register"); setError(""); }}>회원가입</button>
            </>
          ) : (
            <>이미 계정이 있으신가요?{" "}
              <button onClick={() => { setMode("login"); setError(""); }}>로그인</button>
            </>
          )}
        </div>

        <div className="auth-note">
          비밀번호는 PBKDF2-SHA256(20만 회)로 해싱해 저장하며 평문은 남기지 않습니다.
          {googleReady && " 구글로 로그인하면 계정 정보(이메일·이름)만 MongoDB에 저장되고, 비밀번호는 받지 않습니다."}
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="loading">불러오는 중…</div>}>
      <LoginForm />
    </Suspense>
  );
}
