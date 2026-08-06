/**
 * 백그라운드 서비스 워커
 * 콘텐츠 스크립트가 보낸 이벤트를 세션 저장소에 기록해 팝업이 읽게 합니다.
 */

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "TICKER_DETECTED" && message.ticker) {
    chrome.storage.session.set({
      lastDetectedTicker: message.ticker,
      lastDetectedUrl: message.url,
      detectedAt: Date.now(),
    });
  }

  if (message.type === "SCRAP_SENT") {
    // 팝업에서 "방금 어디서 몇 건 수집했는지" 보여주기 위한 누적 기록
    chrome.storage.session.get(["scrapLog"], (data) => {
      const log = data.scrapLog || {};
      const key = message.source;
      log[key] = {
        ticker: message.ticker,
        saved: (log[key]?.saved || 0) + message.saved,
        at: Date.now(),
      };
      chrome.storage.session.set({ scrapLog: log });
    });
  }
});
