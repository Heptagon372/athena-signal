/**
 * Next.js 설정
 *
 * /api/* 는 app/api/[...path]/route.js 프록시 핸들러가 백엔드(8000)로 넘깁니다.
 * 예전에는 여기 rewrite 로 넘겼는데, rewrite 는 오류 처리가 안 돼서 백엔드
 * 재시작(업데이트.bat) 때마다 ECONNRESET 스택 트레이스가 터미널에 쏟아졌습니다.
 * 핸들러 쪽은 재시도 + 깔끔한 503 으로 그 공백을 흡수합니다.
 *
 * 주의: rewrite(afterFiles)는 동적 라우트보다 먼저 평가되므로, /api rewrite 를
 * 여기 되살리면 프록시 핸들러가 조용히 무시됩니다.
 *
 * 브라우저 입장에서는 같은 출처라 CORS 문제가 없고, 토큰을 쿠키로 다뤄도
 * 안전하게 붙습니다.
 */
const BACKEND = process.env.ATHENA_API ?? "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      // 자동매매 콘솔은 백엔드가 직접 그리는 단독 화면입니다 (메인 + 하위 페이지).
      // 여기로 넘겨두면 3000번에서도 같은 주소로 열려 로그인 세션이 이어집니다.
      { source: "/autotrade", destination: `${BACKEND}/autotrade` },
      { source: "/autotrade/:path*", destination: `${BACKEND}/autotrade/:path*` },
      { source: "/static/:path*", destination: `${BACKEND}/static/:path*` },
    ];
  },
};

export default nextConfig;
