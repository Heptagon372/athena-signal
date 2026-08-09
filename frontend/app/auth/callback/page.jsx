"use client";

/**
 * 구글 로그인 마무리
 *
 * 백엔드(:8000)가 구글 콜백을 처리한 뒤 브라우저를 여기로 보냅니다. URL 에는
 * 세션 토큰이 아니라 **60초·1회용 핸드오프 코드**만 들어 있습니다. 30일짜리
 * 세션 토큰이 브라우저 히스토리·Referer·서버 로그에 남지 않게 하려고 한 단계를
 * 더 둔 것입니다 (ACCOUNTS.md 3-3).
 *
 * 이 페이지가 하는 일은 세 줄입니다.
 *   1) handoff 를 토큰으로 교환
 *   2) 히스토리에서 코드를 지움 (뒤로 가기로 되돌아와 재시도하지 않게)
 *   3) 원래 가려던 경로로 이동
 */

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { useAuth } from "../../providers";

// 백엔드가 실어 보내는 실패 사유 → 사람이 읽을 문장
const ERROR_TEXT = {
  access_denied: "구글 로그인을 취소했습니다.",
  missing_code: "구글이 인증 정보를 보내지 않았습니다. 다시 시도해 주세요.",
  bad_state: "로그인 요청이 만료되었거나 이미 처리됐습니다. 다시 시도해 주세요.",
  mongo_unavailable: "계정 데이터베이스(MongoDB)에 연결할 수 없습니다.",
  redirect_uri_mismatch: "구글 콘솔의 리디렉션 URI 설정이 일치하지 않습니다.",
  invalid_client: "구글 클라이언트 ID/시크릿이 올바르지 않습니다.",
  invalid_grant: "인증 코드가 만료됐습니다. 다시 로그인해 주세요.",
  email_unverified: "이메일이 확인되지 않은 구글 계정입니다.",
  not_configured: "구글 로그인이 설정되지 않았습니다.",
};

function Callback() {
  const router = useRouter();
  const params = useSearchParams();
  const { completeGoogleLogin } = useAuth();
  const [error, setError] = useState("");

  // React 18+ 개발 모드는 effect 를 두 번 실행합니다. 핸드오프 코드는 1회용이라
  // 두 번째 호출이 반드시 실패합니다 — 성공했는데 실패 화면이 뜨는 원인입니다.
  const startedRef = useRef(false);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;

    const handoff = params.get("handoff");
    const failure = params.get("error");
    const detail = params.get("message");
    const next = params.get("next") || "/";

    if (failure) {
      setError(detail || ERROR_TEXT[failure] || `로그인에 실패했습니다. (${failure})`);
      return;
    }
    if (!handoff) {
      setError("로그인 정보가 없습니다. 로그인 화면에서 다시 시도해 주세요.");
      return;
    }

    (async () => {
      try {
        await completeGoogleLogin(handoff);
        // 코드를 URL 에서 지워 뒤로 가기로 재시도되지 않게 합니다
        window.history.replaceState({}, "", "/auth/callback");
        router.replace(next);
      } catch (err) {
        setError(err.message || "로그인을 마무리하지 못했습니다.");
      }
    })();
  }, [params, completeGoogleLogin, router]);

  if (error) {
    return (
      <div className="auth-wrap">
        <div className="auth-card">
          <div className="eyebrow">로그인 실패</div>
          <h1 className="auth-title">들어오지 못했습니다</h1>
          <div className="auth-error" style={{ marginBottom: 20 }}>{error}</div>
          <button className="btn btn-gold auth-submit"
                  onClick={() => router.replace("/login")}>
            로그인 화면으로
          </button>
        </div>
      </div>
    );
  }

  return <div className="loading">구글 계정을 확인하고 있습니다…</div>;
}

export default function AuthCallbackPage() {
  return (
    <Suspense fallback={<div className="loading">불러오는 중…</div>}>
      <Callback />
    </Suspense>
  );
}
