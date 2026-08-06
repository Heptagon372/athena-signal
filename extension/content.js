/**
 * 종목 자동 감지 (토스증권 / 네이버 금융)
 * ---------------------------------------
 * 토스증권 URL 패턴
 *     https://tossinvest.com/stocks/A005930/...      -> 한국 종목 (A + 6자리 코드)
 *     https://tossinvest.com/stocks/US19801221001/... -> 미국 종목 (내부 ID)
 *     https://tossinvest.com/stocks/AAPL/...
 * 네이버 금융 URL 패턴
 *     https://finance.naver.com/item/main.naver?code=005930
 *     https://m.stock.naver.com/domestic/stock/005930/total
 *
 * 백엔드는 종목명·종목코드·티커를 모두 받아주므로, 코드를 못 잡으면
 * 페이지 제목에서 종목명을 추출해 넘겨도 정상 동작합니다.
 * (실존하지 않는 값이면 서버가 404로 걸러냅니다.)
 */

function fromTossUrl() {
  const match = window.location.pathname.match(/\/stocks\/([A-Za-z0-9-]+)/);
  if (!match) return null;
  const raw = match[1];

  // A005930 형태 -> 한국 6자리 종목코드
  const korean = raw.match(/^A(\d{6})$/i);
  if (korean) return korean[1];

  // 6자리 숫자만 있는 경우
  if (/^\d{6}$/.test(raw)) return raw;

  // 영문 티커 (미국)
  const ticker = raw.replace(/-US$|-KR$/i, "");
  if (/^[A-Za-z.\-]{1,12}$/.test(ticker)) return ticker.toUpperCase();

  return null;
}

function fromNaverUrl() {
  const params = new URLSearchParams(window.location.search);
  const code = params.get("code");
  if (code && /^\d{6}$/.test(code)) return code;

  const path = window.location.pathname.match(/\/stock\/(\d{6})/);
  if (path) return path[1];

  return null;
}

function fromDom() {
  // 문서 제목에서 6자리 코드 -> 영문 티커 -> 한글 종목명 순으로 시도
  const title = document.title || "";

  const code = title.match(/\b(\d{6})\b/);
  if (code) return code[1];

  const ticker = title.match(/\(([A-Z]{1,6})\)/);
  if (ticker) return ticker[1];

  // "삼성전자 - 토스증권" / "삼성전자 : 네이버페이 증권" 같은 형태에서 종목명만 분리
  const korean = title.split(/[-|:·]/)[0].trim();
  if (/[가-힣]/.test(korean) && korean.length <= 20) return korean;

  return null;
}

function detectAndBroadcast() {
  const host = window.location.hostname;
  let detected = null;

  if (host.includes("tossinvest.com")) detected = fromTossUrl();
  else if (host.includes("naver.com")) detected = fromNaverUrl();

  detected = detected || fromDom();

  if (detected) {
    chrome.runtime.sendMessage({
      type: "TICKER_DETECTED",
      ticker: detected,
      url: window.location.href,
    });
  }
}

// 최초 로드 + SPA 라우팅 대응 (URL이 새로고침 없이 바뀌는 경우 감지)
detectAndBroadcast();
let lastUrl = window.location.href;
new MutationObserver(() => {
  if (window.location.href !== lastUrl) {
    lastUrl = window.location.href;
    setTimeout(detectAndBroadcast, 500); // 라우팅 후 DOM 반영 대기
  }
}).observe(document.body, { childList: true, subtree: true });
