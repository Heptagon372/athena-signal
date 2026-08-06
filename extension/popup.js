const API_BASE = "http://localhost:8000/api";
const HORIZON_LABELS = { open: "개장", "5min": "+5분", "30min": "+30분", close: "마감" };
const MARKET_LABELS = { KOSPI: "코스피", KOSDAQ: "코스닥", KONEX: "코넥스", US: "미국" };

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function dirColor(dir) { return dir === "up" ? "var(--up)" : "var(--down)"; }

function setBadge(text, cls) {
  $("mode-badge").textContent = text;
  $("mode-badge").className = "badge " + cls;
}

/**
 * 종목이 실존하지 않으면 서버가 404 + SYMBOL_NOT_FOUND 를 돌려줍니다.
 * 그 경우 예시 데이터로 덮지 않고 "없는 종목" 화면을 그대로 보여줍니다.
 */
async function fetchPrediction(query) {
  const res = await fetch(`${API_BASE}/predict/${encodeURIComponent(query)}`,
    { signal: AbortSignal.timeout(25000) });

  if (res.status === 404) {
    const body = await res.json().catch(() => null);
    if (body && body.error === "SYMBOL_NOT_FOUND") {
      const err = new Error(body.message);
      err.notFound = body;
      throw err;
    }
  }
  if (!res.ok) throw new Error("api_error");
  return res.json();
}

function renderNotFound(payload) {
  setBadge("종목 없음", "demo");
  const chips = (payload.suggestions || []).map((s) =>
    `<button class="suggest-chip" data-key="${esc(s.key)}">${esc(s.name)}
       <span>${esc(s.key)} · ${esc(MARKET_LABELS[s.market] || s.market)}</span></button>`).join("");

  $("content").innerHTML = `
    <div class="notfound">
      <div class="nf-head">'${esc(payload.query)}' — 존재하지 않는 종목</div>
      <div class="nf-msg">${esc(payload.message)}</div>
      ${chips ? `<div class="nf-chips">${chips}</div>` : ""}
      <div class="nf-hint">코스피 · 코스닥 · 코넥스 · 미국 시장에서 조회했습니다.<br>예: 삼성전자 / 005930 / 에코프로비엠 / AAPL</div>
    </div>`;

  document.querySelectorAll(".suggest-chip").forEach((btn) =>
    btn.addEventListener("click", () => loadTicker(btn.dataset.key)));
}

function renderOffline(query) {
  setBadge("서버 미연결", "demo");
  $("content").innerHTML = `
    <div class="notfound">
      <div class="nf-head">분석 서버에 연결할 수 없습니다</div>
      <div class="nf-msg">'${esc(query)}' 을(를) 분석하려면 백엔드가 켜져 있어야 합니다.
        <code>AthenaSignal.exe</code> 를 실행하거나 <code>uvicorn api:app --port 8000</code> 을 실행해 주세요.</div>
      <div class="nf-hint">실시간 시세·뉴스·커뮤니티 데이터가 필요하므로 오프라인 추정치는 표시하지 않습니다.</div>
    </div>`;
}

function render(data) {
  setBadge("실시간 연결됨", "live");
  const sym = data.symbol;
  const snap = data.market_snapshot || {};
  const changeColor = snap.change_rate >= 0 ? "var(--up)" : "var(--down)";
  const unit = sym.currency === "USD" ? "$" : "원";
  const price = sym.currency === "USD"
    ? "$" + Number(snap.current_price || 0).toLocaleString("ko-KR", { maximumFractionDigits: 2 })
    : Number(snap.current_price || 0).toLocaleString("ko-KR") + unit;

  const rows = data.predictions.map((p) => {
    const color = dirColor(p.direction);
    return `
      <div class="gauge-item">
        <span class="h-label">${HORIZON_LABELS[p.horizon] || p.horizon}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${p.probability}%;background:${color}"></div></div>
        <span class="pct" style="color:${color}">${p.direction === "up" ? "▲" : "▼"} ${p.probability}%</span>
      </div>`;
  }).join("");

  // 최종 확률에 가장 크게 기여한 기술적 지표 3개를 근거로 노출
  const topIndicators = ((data.technical && data.technical.indicators) || [])
    .filter((i) => i.weight > 0 && Math.abs(i.score) > 0.05)
    .slice(0, 3)
    .map((i) => {
      const color = i.score > 0 ? "var(--up)" : "var(--down)";
      return `<div class="ind-line">
        <span class="ind-label">${esc(i.label)}</span>
        <span class="ind-value" style="color:${color}">${esc(i.value_text)}</span>
        <span class="ind-contrib" style="color:${color}">${i.contribution >= 0 ? "+" : ""}${i.contribution}</span>
      </div>`;
    }).join("");

  const w = data.weights || {};
  const news = data.news_summary || {};
  const com = data.community || {};

  $("content").innerHTML = `
    <div class="ticker-title">${esc(sym.name)}
      <span class="market-tag ${esc(sym.market)}">${esc(sym.market_label)}</span></div>
    <div class="ticker-sub">${esc(sym.key)} · ${new Date(data.generated_at).toLocaleString("ko-KR")}</div>
    <div class="price-line">
      <span class="price">${price}</span>
      <span class="change" style="color:${changeColor}">${snap.change_rate >= 0 ? "+" : ""}${snap.change_rate}%</span>
    </div>
    <div class="gauge-list">${rows}</div>
    ${topIndicators ? `<div class="section-label">기술적 근거 (기여도 순)</div><div class="ind-list">${topIndicators}</div>` : ""}
    <div class="section-label">데이터</div>
    <div class="data-line">뉴스 ${news.total || 0}건 (긍정 ${news.positive || 0} / 부정 ${news.negative || 0})</div>
    <div class="data-line">커뮤니티 ${com.post_count || 0}건 (매수 ${com.bullish_count || 0} / 매도 ${com.bearish_count || 0})</div>
    <div class="weights-line">
      <span>기술 ${Math.round((w.technical || 0) * 100)}%</span>
      <span>뉴스 ${Math.round((w.news_sentiment || 0) * 100)}%</span>
      <span>커뮤니티 ${Math.round((w.community_sentiment || 0) * 100)}%</span>
    </div>`;
}

async function loadTicker(query) {
  query = (query || "").trim();
  if (!query) return;
  $("content").innerHTML = `<div class="loading">${esc(query)} 분석 중…</div>`;
  try {
    render(await fetchPrediction(query));
    chrome.storage.session.set({ lastViewedTicker: query });
  } catch (e) {
    if (e.notFound) renderNotFound(e.notFound);
    else renderOffline(query);
  }
}

$("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const val = $("ticker-input").value;
  $("ticker-input").value = "";
  if (val) loadTicker(val);
});

$("open-web-link").addEventListener("click", (e) => {
  e.preventDefault();
  chrome.tabs.create({ url: "http://localhost:8000/" });
});

(async function init() {
  let query = "삼성전자";
  try {
    const stored = await chrome.storage.session.get(["lastDetectedTicker", "lastViewedTicker"]);
    if (stored.lastDetectedTicker) {
      query = stored.lastDetectedTicker;
      $("detected-hint").style.display = "block";
      $("detected-hint").innerHTML = `현재 페이지에서 <b>${esc(query)}</b> 감지됨 — 이 결과를 보여드립니다.`;
    } else if (stored.lastViewedTicker) {
      query = stored.lastViewedTicker;
    }
  } catch (e) {
    // chrome.storage 접근 불가 환경(테스트 등) 대비
  }
  $("ticker-input").value = "";
  await loadTicker(query);
})();
