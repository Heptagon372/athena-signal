/**
 * Next.js 설정
 *
 * /api/* 는 app/api/[...path]/route.js 프록시 핸들러가 백엔드(8000)로 넘깁니다.
 * 예전에는 여기 rewrite 로 넘겼는데, rewrite 는 오류 처리가 안 돼서 백엔드
 * 재시작([3] 업데이트) 때마다 ECONNRESET 스택 트레이스가 터미널에 쏟아졌습니다.
 * 핸들러 쪽은 재시도 + 깔끔한 503 으로 그 공백을 흡수합니다.
 *
 * 주의: rewrite(afterFiles)는 동적 라우트보다 먼저 평가되므로, /api rewrite 를
 * 여기 되살리면 프록시 핸들러가 조용히 무시됩니다.
 *
 * 브라우저 입장에서는 같은 출처라 CORS 문제가 없고, 토큰을 쿠키로 다뤄도
 * 안전하게 붙습니다.
 */
const BACKEND = process.env.ATHENA_API ?? "http://127.0.0.1:8000";
const DEV = process.env.NODE_ENV !== "production";

/*
 * 보안 헤더
 * ---------
 * 백엔드(api.py)는 자기가 그리는 화면에만 헤더를 붙일 수 있습니다. 이 앱의
 * 화면은 Next(3000)가 그리므로 여기서 따로 붙여야 합니다.
 *
 * script-src 에 'unsafe-inline' 이 있는 이유
 *     Next 는 하이드레이션 부트스트랩을 인라인 <script> 로 심습니다. 이걸
 *     막으려면 요청마다 nonce 를 발급하는 미들웨어가 필요한데, 지금 구조를
 *     크게 바꾸는 일이라 다음으로 미룹니다. 대신 **새어 나가는 쪽**을 잠급니다 —
 *     connect-src 가 우리 서버와 구글 인증 엔드포인트뿐이면, 스크립트가
 *     주입되더라도 localStorage 의 세션 토큰을 공격자 서버로 보낼 수 없습니다.
 *     그 토큰은 KIS 주문 권한이라 유출 경로를 막는 것이 실행을 막는 것 다음으로
 *     중요합니다.
 *
 * 개발 모드에서 'unsafe-eval' 과 ws: 를 여는 이유
 *     HMR(핫 리로드)이 eval 과 웹소켓을 씁니다. 프로덕션 빌드에는 넣지 않습니다.
 *
 * 외부 호스트가 필요한 곳
 *     cdn.jsdelivr.net / fonts.* : Pretendard·Noto Sans KR 웹폰트 (layout.jsx)
 *     *.googleapis.com           : Firebase 로그인 (토큰 발급·검증)
 *     *.firebaseapp.com, accounts.google.com : 구글 계정 선택 팝업
 */
const CSP = [
  "default-src 'self'",
  `script-src 'self' 'unsafe-inline'${DEV ? " 'unsafe-eval'" : ""} https://apis.google.com`,
  "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
  "font-src 'self' data: https://cdn.jsdelivr.net https://fonts.gstatic.com",
  "img-src 'self' data: blob:",
  "media-src 'self'",
  `connect-src 'self' https://*.googleapis.com https://*.firebaseapp.com${DEV ? " ws: http://127.0.0.1:8000" : ""}`,
  "frame-src https://*.firebaseapp.com https://accounts.google.com",
  "form-action 'self'",
  "base-uri 'none'",
  "object-src 'none'",
  "frame-ancestors 'none'",
].join("; ");

const SECURITY_HEADERS = [
  { key: "Content-Security-Policy", value: CSP },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "same-origin" },
  {
    key: "Permissions-Policy",
    value: "accelerometer=(), camera=(), geolocation=(), gyroscope=(), " +
           "magnetometer=(), microphone=(), payment=(), usb=()",
  },
];

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 서버 종류·버전을 굳이 알려주지 않습니다 (nginx 의 server_tokens off 와 짝)
  poweredByHeader: false,
  async headers() {
    return [{ source: "/:path*", headers: SECURITY_HEADERS }];
  },
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
