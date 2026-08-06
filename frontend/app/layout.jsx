import "./globals.css";
import "./pages.css";
import { Providers } from "./providers";
import Chrome from "./components/Chrome";

export const metadata = {
  title: "ATHENA SIGNAL — 코스피·코스닥 전략 신호 콘솔",
  description:
    "기술적 지표 19종, 국내외 뉴스, 커뮤니티 여론을 종합해 상승/하락 확률을 계산하는 실험적 분석 콘솔",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }) {
  return (
    <html lang="ko" data-theme="dark" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
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
