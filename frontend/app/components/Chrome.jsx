"use client";

/**
 * 공통 헤더 + 내비게이션
 *
 * 해시 탭이 아니라 **실제 라우트**입니다. 각 페이지는 독립된 URL 을 갖고,
 * 새로고침·뒤로가기·즐겨찾기가 모두 정상 동작합니다.
 */

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useAuth, useTheme } from "../providers";

const NAV = [
  { href: "/", label: "분석", icon: "M3 17l5-6 4 4 6-8 M3 21h18" },
  { href: "/score", label: "성적표", icon: "M9 11l3 3 6-6" },
  { href: "/paper", label: "모의투자", icon: "M12 1v22 M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" },
  // 자동매매 콘솔은 백엔드(8000번)가 그리는 단독 화면이라 라우터를 태우지 않고
  // 일반 링크로 엽니다. next.config 의 rewrite 가 같은 주소로 연결해 줍니다.
  { href: "/autotrade", label: "자동매매", icon: "M13 2L4 14h6l-1 8 9-12h-6l1-8z", external: true },
  { href: "/settings", label: "설정", icon: "M12 8a4 4 0 100 8 4 4 0 000-8z M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 11-4 0v-.09a1.65 1.65 0 00-1-1.51 1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 110-4h.09a1.65 1.65 0 001.51-1 1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33h0a1.65 1.65 0 001-1.51V3a2 2 0 114 0v.09a1.65 1.65 0 001 1.51h0a1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82v0a1.65 1.65 0 001.51 1H21a2 2 0 110 4h-.09a1.65 1.65 0 00-1.51 1z" },
];

function Icon({ d }) {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none"
         stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      {d.split(" M").map((seg, i) => (
        <path key={i} d={i === 0 ? seg : `M${seg}`} />
      ))}
    </svg>
  );
}

/** 다음 장 세션까지 남은 시간 */
function Countdown() {
  const [status, setStatus] = useState(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await api.get("/status", { timeout: 8000 });
        if (alive) setStatus(s.korea_market);
      } catch {}
    };
    load();
    const poll = setInterval(load, 60000);
    const timer = setInterval(() => setTick((t) => t + 1), 1000);
    return () => { alive = false; clearInterval(poll); clearInterval(timer); };
  }, []);

  if (!status) return null;

  // 한국장 기준 다음 전환 시각
  const MARKS = [
    [8, 30, "동시호가까지"], [9, 0, "정규장 개장까지"],
    [15, 20, "마감 동시호가까지"], [15, 30, "동시호가 종료까지"],
    [15, 40, "장후 시간외 종가까지"], [16, 0, "시간외 단일가까지"],
    [18, 0, "시간외 종료까지"],
  ];
  const now = new Date();
  const mins = now.getHours() * 60 + now.getMinutes() + now.getSeconds() / 60;
  let next = MARKS.find(([h, m]) => h * 60 + m > mins);
  let remaining;
  if (next) {
    remaining = next[0] * 60 + next[1] - mins;
  } else {
    next = MARKS[0];
    remaining = 24 * 60 - mins + next[0] * 60 + next[1];
  }

  const total = Math.max(0, Math.round(remaining * 60));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const text = h > 0 ? `${h}시간 ${String(m).padStart(2, "0")}분`
                     : `${m}:${String(s).padStart(2, "0")}`;

  const open = status.is_open;
  return (
    <div className={`countdown${open ? " open" : h < 1 ? " soon" : ""}`}
         title="다음 장 세션까지 남은 시간">
      <span className="cd-dot" />
      <span className="cd-label">{open ? `${status.label} · ${next[2]}` : next[2]}</span>
      <span className="cd-time mono">{text}</span>
    </div>
  );
}

export default function Chrome({ children }) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, ready, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();

  const onLogout = async () => {
    await logout();
    router.push("/login");
  };

  return (
    <>
      <header className="site-header">
        <div className="header-inner">
          <Link href="/" className="brand">
            <svg width="30" height="30" viewBox="0 0 40 40" fill="none">
              <circle cx="20" cy="20" r="18.5" stroke="var(--gold)" strokeWidth="1.2" />
              <circle cx="20" cy="20" r="10" stroke="var(--gold)" strokeWidth="1" />
              <circle cx="20" cy="20" r="3.4" fill="var(--gold)" />
              <path d="M20 1.5V6M20 34V38.5M1.5 20H6M34 20H38.5"
                    stroke="var(--olive)" strokeWidth="1" />
            </svg>
            <div>
              <div className="brand-text">ATHENA SIGNAL</div>
              <div className="brand-sub">코스피 · 코스닥 · 미국</div>
            </div>
          </Link>

          <div className="header-right">
            <Countdown />
            {ready && (user ? (
              <div className="user-chip">
                <span className="mono">{user.display_name}</span>
                <button className="btn-mini" onClick={onLogout}>로그아웃</button>
              </div>
            ) : (
              <Link href="/login" className="btn btn-gold" style={{ textDecoration: "none" }}>
                로그인
              </Link>
            ))}
            <button className="theme-toggle" onClick={toggleTheme}
                    title="다크/라이트 전환" aria-label="테마 전환">
              {theme === "light" ? (
                <svg viewBox="0 0 24 24" width="17" height="17" fill="none"
                     stroke="currentColor" strokeWidth="1.8">
                  <path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" width="17" height="17" fill="none"
                     stroke="currentColor" strokeWidth="1.8">
                  <circle cx="12" cy="12" r="4.2" />
                  <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </header>

      <nav className="navbar">
        <div className="navbar-inner">
          {NAV.map((item) => (
            item.external ? (
              <a key={item.href} href={item.href} className="nav-link">
                <Icon d={item.icon} />
                {item.label}
              </a>
            ) : (
              <Link key={item.href} href={item.href}
                    className={`nav-link${pathname === item.href ? " active" : ""}`}>
                <Icon d={item.icon} />
                {item.label}
              </Link>
            )
          ))}
        </div>
      </nav>

      <main className="shell">{children}</main>

      <footer className="shell">
        <div className="disclaimer">
          ATHENA SIGNAL이 표시하는 확률은 검증된 투자 자문이 아니라, 기술적 지표·뉴스·커뮤니티
          여론을 결합한 실험적 휴리스틱 점수입니다. 기술적 지표는 과거 가격의 통계적 요약일 뿐
          미래 수익을 보장하지 않으며, 종목토론방 여론은 노이즈와 의도적 선동이 섞인 채널입니다.
          모의투자는 가상 자금이며 실제 주문은 발생하지 않습니다. 실제 매매 판단의 유일한 근거로
          사용하지 마세요.
        </div>
      </footer>
    </>
  );
}
