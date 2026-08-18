import "./globals.css";
import "./pages.css";
import { Providers } from "./providers";
import Chrome from "./components/Chrome";

export const metadata = {
  title: "ATHENA SIGNAL 주식 전략 신호 콘솔",
  description:
    "기술적 지표 19종, 국내외 뉴스, 커뮤니티 여론을 종합해 상승/하락 확률을 계산하는 실험적 분석 콘솔",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko" data-theme="light" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://cdn.jsdelivr.net" crossOrigin="" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        {/* 본문 한글 폰트: Pretendard (없으면 Noto Sans KR → 시스템 한글 폰트) */}
        <link
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.css"
          rel="stylesheet"
          crossOrigin=""
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Providers>
          <Chrome>{children}</Chrome>
        </Providers>
      </body>
    </html>
  );
}
