"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { useAuth } from "../providers";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const { user, ready, login, register } = useAuth();

  const [mode, setMode] = useState("login");     // login | register
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const next = params.get("next") || "/";

  // 이미 로그인돼 있으면 돌려보냅니다
  useEffect(() => {
    if (ready && user) router.replace(next);
  }, [ready, user, next, router]);

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
          이 앱은 사용자 PC에서만 도는 로컬 서비스입니다.
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
