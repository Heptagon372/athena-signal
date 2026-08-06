/**
 * 커뮤니티 스크래퍼 (Browser-Assisted Scraping)
 * ---------------------------------------------
 * 서버가 직접 접근할 수 없는 커뮤니티를 사용자 브라우저에서 읽어
 * 로컬 분석 서버(localhost:8000)로 보냅니다.
 *
 * 왜 확장프로그램이 해야 하는가
 *   토스증권 커뮤니티는 비로그인 요청이 401/405, 팍스넷은 404,
 *   카카오페이증권은 연결 자체가 거부됩니다. 반면 이 스크립트는
 *   **사용자가 이미 보고 있는 페이지 안에서** 돌기 때문에,
 *   화면에 떠 있는 공개 게시글을 그대로 읽을 수 있습니다.
 *
 * 지키는 원칙
 *   · 사용자가 직접 연 종목 페이지에서만 동작합니다 (자동 순회 없음)
 *   · 화면에 이미 렌더링된 공개 게시글 제목만 읽습니다
 *   · 로그인 정보·쿠키·개인정보는 읽지도 보내지도 않습니다
 *   · 전송 대상은 사용자 PC의 localhost 뿐입니다 (외부 서버 없음)
 *   · 같은 페이지를 반복 전송하지 않도록 간격 제한을 둡니다
 *
 * 사이트 DOM 구조는 예고 없이 바뀝니다. 셀렉터가 안 맞으면 조용히 아무것도
 * 보내지 않고 넘어갑니다 (사용자 브라우징을 방해하지 않기 위함).
 */

const API_BASE = "http://localhost:8000/api";
const MIN_INTERVAL_MS = 60000;   // 같은 종목 재전송 최소 간격
const MAX_ITEMS = 60;

let lastSent = {};               // { "source:ticker": timestamp }

/* ---------------- 사이트별 어댑터 ---------------- */

const ADAPTERS = [
  {
    source: "toss",
    match: (h) => h.includes("tossinvest.com"),
    ticker: () => {
      const m = location.pathname.match(/\/stocks\/A?(\d{6}|[A-Z.]{1,8})/i);
      return m ? m[1].toUpperCase() : null;
    },
    // 토스 커뮤니티 글은 클래스명이 자주 바뀌어, 구조 대신 '댓글처럼 생긴 것'을 찾습니다
    collect: () => {
      const nodes = document.querySelectorAll(
        '[class*="comment"] [class*="body"], [class*="Comment"] p, article p, [data-testid*="comment"]');
      return textsFrom(nodes);
    },
  },
  {
    source: "paxnet",
    match: (h) => h.includes("paxnet.co.kr"),
    ticker: () => {
      const p = new URLSearchParams(location.search);
      const code = p.get("code") || p.get("stockCode");
      return code && /^\d{6}$/.test(code) ? code : null;
    },
    collect: () => textsFrom(document.querySelectorAll(
      "td.subject a, .list_area .subject a, table.type2 td.title a, .tbbs-list a")),
  },
  {
    source: "kakaopay",
    match: (h) => h.includes("kakaopay") && h.includes("securities"),
    ticker: () => {
      const m = location.pathname.match(/(\d{6})/);
      return m ? m[1] : null;
    },
    collect: () => textsFrom(document.querySelectorAll(
      '[class*="comment"] [class*="text"], [class*="post"] [class*="content"], article p')),
  },
  {
    source: "naver",
    match: (h) => h.includes("finance.naver.com") || h.includes("stock.naver.com"),
    ticker: () => {
      const p = new URLSearchParams(location.search);
      if (p.get("code") && /^\d{6}$/.test(p.get("code"))) return p.get("code");
      const m = location.pathname.match(/\/stock\/(\d{6})/);
      return m ? m[1] : null;
    },
    collect: () => textsFrom(document.querySelectorAll("td.title a, .tb_cmt td a")),
  },
];

/** DOM 노드 목록 -> 정제된 텍스트 배열 */
function textsFrom(nodes) {
  const out = [];
  const seen = new Set();
  for (const n of nodes) {
    const raw = (n.getAttribute && n.getAttribute("title")) || n.textContent || "";
    const text = raw.replace(/\s+/g, " ").trim();
    // 너무 짧은 것(버튼/라벨)과 너무 긴 것(본문 전체)은 제외
    if (text.length < 4 || text.length > 280) continue;
    if (seen.has(text)) continue;
    seen.add(text);
    out.push({ title: text, url: n.href || location.href });
    if (out.length >= MAX_ITEMS) break;
  }
  return out;
}

/* ---------------- 전송 ---------------- */

async function sendScrap(source, ticker, items) {
  if (!ticker || !items.length) return;

  const key = `${source}:${ticker}`;
  const now = Date.now();
  if (lastSent[key] && now - lastSent[key] < MIN_INTERVAL_MS) return;
  lastSent[key] = now;

  try {
    const res = await fetch(`${API_BASE}/scrap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker, source, kind: "community", items }),
    });
    if (!res.ok) return;
    const data = await res.json();
    if (data.saved > 0) {
      console.log(`[Athena] ${source} → ${ticker}: 새 글 ${data.saved}건 전송 ` +
                  `(중복 ${data.duplicated})`);
      chrome.runtime.sendMessage({
        type: "SCRAP_SENT", source, ticker, saved: data.saved,
      }).catch(() => {});
    }
  } catch (e) {
    // 서버가 꺼져 있으면 조용히 넘어갑니다 (브라우징 방해 금지)
  }
}

function run() {
  const host = location.hostname;
  const adapter = ADAPTERS.find((a) => a.match(host));
  if (!adapter) return;

  let ticker = null;
  try { ticker = adapter.ticker(); } catch (e) { return; }
  if (!ticker) return;

  let items = [];
  try { items = adapter.collect(); } catch (e) { return; }

  sendScrap(adapter.source, ticker, items);
}

// 최초 + SPA 라우팅/무한스크롤 대응 (DOM이 안정된 뒤 수집)
let debounce = null;
function schedule() {
  clearTimeout(debounce);
  debounce = setTimeout(run, 1500);
}

schedule();
new MutationObserver(schedule).observe(document.body, { childList: true, subtree: true });
