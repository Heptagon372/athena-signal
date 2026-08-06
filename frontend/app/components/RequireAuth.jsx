"use client";

/**
 * 로그인이 필요한 페이지를 감싸는 가드.
 * 세션 확인이 끝나기 전에는 깜빡임을 막기 위해 로딩만 보여줍니다.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "../providers";

export default function RequireAuth({ children, what = "이 기능" }) {
  const { user, ready } = useAuth();
  const pathname = usePathname();

  if (!ready) return <div className="loading">세션 확인 중…</div>;

  if (!user) {
    return (
      <section className="section">
        <div className="card auth-gate">
          <div className="eyebrow">로그인 필요</div>
          <h2>{what}은 로그인 후 이용할 수 있습니다</h2>
          <p>
            계정마다 예측 기록·성적표·모의투자 계좌가 따로 저장되기 때문에,
            어느 계정의 데이터인지 알아야 합니다.
          </p>
          <Link href={`/login?next=${encodeURIComponent(pathname)}`}
                className="btn btn-gold" style={{ textDecoration: "none", display: "inline-block" }}>
            로그인 / 회원가입
          </Link>
        </div>
      </section>
    );
  }

  return children;
}
