/* ==========================================================================
   자동매매 콘솔 공통 스크립트
   --------------------------------------------------------------------------
   모든 페이지가 이 파일 하나로 다음을 공유합니다.
     · 로그인(토큰) 처리 + 로그인 오버레이
     · 서버 호출 call() + 404/버전 불일치 감지 (서버 재시작 배너)
     · 공통 헤더(부엉이·내비게이션·상태 배지·시작/중지·킬 스위치) 주입
     · 계좌 정체 표시줄(가상/실제 자금) + 실전 모드 붉은 테두리
     · 7초 주기 상태 새로고침 → 각 페이지의 PAGE.onSnapshot(snap) 호출

   페이지 쪽 계약:
     window.PAGE = {
       nav: "main" | "ai" | "penny" | "deriv" | "backtest",
       title, desc,                  // 페이지 머리글
       init(),                       // 첫 로딩 후 1회
       onSnapshot(snap),             // 상태 새로고침마다
     }
   ========================================================================== */

(function () {
  const API = "/api";
  // api.py 의 CONSOLE_API_VERSION 과 짝 — 서버가 낮으면 재시작 배너를 띄웁니다
  const REQUIRED_API = 9;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const won = (v) => (v == null || isNaN(v)) ? "—" : Math.round(v).toLocaleString("ko-KR") + "원";
  const num = (v, d = 2) => (v == null || isNaN(v)) ? "—"
    : Number(v).toLocaleString("ko-KR", {maximumFractionDigits: d});
  const pct = (v) => (v == null || isNaN(v)) ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(2) + "%";
  const sign = (v) => v > 0 ? "up" : v < 0 ? "down" : "muted";

  let TOKEN = "";
  let SNAP = null;

  // ---- 토큰: 3000번(Next)과 8000번이 쿠키를 공유합니다 ----
  function readToken() {
    const m = document.cookie.match(/(?:^|;\s*)athena_token=([^;]+)/);
    if (m) return decodeURIComponent(m[1]);
    try { return localStorage.getItem("athena_token") || ""; } catch { return ""; }
  }
  function saveToken(t) {
    TOKEN = t;
    try { localStorage.setItem("athena_token", t); } catch {}
    document.cookie = `athena_token=${t}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
  }

  // ---- 서버 호출 ----
  async function call(path, opts = {}) {
    const res = await fetch(API + path, {
      method: opts.method || "GET",
      headers: {"Content-Type": "application/json",
                ...(TOKEN ? {Authorization: `Bearer ${TOKEN}`} : {})},
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch {}
    if (res.status === 401) { showLogin(); throw new Error("로그인이 필요합니다."); }
    // 404 라고 다 구버전 서버가 아닙니다. 서버가 직접 만든 404 는
    // {error, message} 를 담아 옵니다 (예: 종목 없음) — 그건 아래의 일반
    // 오류 처리로 보내고, 라우트 자체가 없는 FastAPI 기본 404
    // ({"detail":"Not Found"}) 만 재시작 배너를 띄웁니다.
    if (res.status === 404 && path.startsWith("/autotrade") && !(data && data.error)) {
      showStaleBanner();
      throw new Error("서버가 이전 버전 코드로 실행 중입니다 — 서버를 재시작하세요.");
    }
    if (!res.ok) throw new Error(
      (data && (data.message || data.error || data.detail)) || `HTTP ${res.status}`);
    return data;
  }

  function toast(msg, bad) {
    const el = document.createElement("div");
    el.className = "toast" + (bad ? " bad" : "");
    el.textContent = msg;
    $("toast").appendChild(el);
    setTimeout(() => el.remove(), 5200);
  }

  function showStaleBanner() {
    if ($("staleBanner")) return;
    const el = document.createElement("div");
    el.id = "staleBanner";
    el.className = "warnbox";
    el.style.cssText = "position:sticky;top:96px;z-index:30;border-color:var(--up);color:#ffb4a6";
    el.innerHTML = "<b>서버 재시작이 필요합니다.</b> 이 화면은 최신이지만 서버는 이전 버전 " +
      "코드로 실행 중입니다. 서버 창을 닫고 <code>python -m uvicorn api:app --port 8000</code> " +
      "으로 다시 켜세요.";
    document.querySelector("main").prepend(el);
  }

  // ---- 부엉이 (아테나의 상징 — 눈을 깜빡이며 지켜봅니다) ----
  function owlSVG(size, hero) {
    const cls = hero ? "owl-hero" : "owl";
    return `<svg class="${cls}" width="${size}" height="${size}" viewBox="0 0 64 64" aria-hidden="true">
      <ellipse cx="32" cy="38" rx="22" ry="21" fill="#262019" stroke="#3a3226" stroke-width="1.5"/>
      <path d="M14 24 Q10 12 18 14 Q22 8 27 15 Z" fill="#262019" stroke="#3a3226" stroke-width="1.2"/>
      <path d="M50 24 Q54 12 46 14 Q42 8 37 15 Z" fill="#262019" stroke="#3a3226" stroke-width="1.2"/>
      <ellipse cx="32" cy="44" rx="11" ry="12" fill="#1c1812"/>
      <path d="M22 46 Q32 41 42 46 M23 51 Q32 46 41 51" stroke="#3a3226" stroke-width="1.2" fill="none"/>
      <circle cx="23.5" cy="30" r="8.6" fill="#14110d" stroke="#c9a227" stroke-width="1.6"/>
      <circle cx="40.5" cy="30" r="8.6" fill="#14110d" stroke="#c9a227" stroke-width="1.6"/>
      <circle class="pupil" cx="24.5" cy="30.5" r="3.4" fill="#c9a227"/>
      <circle class="pupil" cx="41.5" cy="30.5" r="3.4" fill="#c9a227"/>
      <circle cx="25.6" cy="29.2" r="1.1" fill="#ece3d0"/>
      <circle cx="42.6" cy="29.2" r="1.1" fill="#ece3d0"/>
      <ellipse class="lid" cx="23.5" cy="30" rx="8.6" ry="8.6" fill="#262019"/>
      <ellipse class="lid r" cx="40.5" cy="30" rx="8.6" ry="8.6" fill="#262019"/>
      <path d="M32 34 L29.5 38.5 L34.5 38.5 Z" fill="#c9762a"/>
      <path d="M26 58 l-2 4 M32 58 l0 4 M38 58 l2 4" stroke="#c9762a" stroke-width="1.6"/>
    </svg>`;
  }

  // ---- 공통 헤더 + 내비게이션 주입 ----
  const NAV = [
    ["main", "/autotrade", "메인"],
    ["ai", "/autotrade/ai", "AI 추천·추적"],
    ["penny", "/autotrade/penny", "페니 초단타"],
    ["deriv", "/autotrade/deriv", "선물·옵션"],
    ["backtest", "/autotrade/backtest", "백테스트·기타"],
  ];

  function injectChrome(page) {
    const nav = NAV.map(([key, href, label]) =>
      `<a href="${href}" class="${key === page.nav ? "active" : ""}${key === "penny" ? " pennytab" : ""}">${label}</a>`
    ).join("");
    document.body.insertAdjacentHTML("afterbegin", `
      <header>
        <div class="bar">
          <div class="brand">${owlSVG(34)}<span>ATHENA<small>자동매매 콘솔</small></span></div>
          <span id="stateBadge" class="pill off"><span class="dot"></span><span>불러오는 중…</span></span>
          <span id="modeBadge" class="pill"></span>
          <div style="flex:1"></div>
          <button id="btnRun" class="sm">지금 1회전</button>
          <button id="btnKill" class="danger sm">킬 스위치</button>
          <button id="btnToggle" class="primary">시작</button>
        </div>
        <nav class="navrow">${nav}</nav>
      </header>`);
    document.body.insertAdjacentHTML("beforeend", `<div id="toast"></div>
      <div id="loginOverlay" class="hide"><section>
        <h2>로그인</h2>
        <div class="field"><label>아이디</label><input id="lgUser" autocomplete="username"></div>
        <div class="field"><label>비밀번호</label><input id="lgPass" type="password" autocomplete="current-password"></div>
        <button id="btnLogin" class="primary" style="width:100%">로그인</button>
        <div class="note" id="lgMsg" style="margin-top:10px"></div>
      </section></div>`);

    const main = document.querySelector("main");
    main.insertAdjacentHTML("afterbegin", `
      <div class="pagehead">${owlSVG(52, true)}
        <div><h1>${esc(page.title || "")}</h1><p>${esc(page.desc || "")}</p></div></div>
      <div id="moneyBar" class="moneybar"></div>
      <div id="liveWarn" class="warnbox hide"><b>실거래 모드입니다.</b>
        이 화면의 모든 숫자와 주문은 실제 계좌·실제 자금입니다.</div>`);

    $("btnRun").onclick = runOnce;
    $("btnKill").onclick = killSwitch;
    $("btnToggle").onclick = toggleRun;
    $("btnLogin").onclick = doLogin;
    $("lgPass").onkeydown = (e) => { if (e.key === "Enter") doLogin(); };
  }

  // ---- 공통 상태 렌더 ----
  function renderCommon(snap) {
    SNAP = snap;
    if ((snap.api_version || 0) < REQUIRED_API) showStaleBanner();
    else $("staleBanner")?.remove();

    const cfg = snap.config || {};
    const badge = $("stateBadge");
    const st = snap.state;
    badge.className = "pill " + (st === "running" ? "on" : st === "halted" ? "halt" : "off");
    badge.lastElementChild.textContent =
      (snap.state_label || "") + (snap.state_reason ? ` — ${snap.state_reason}` : "");
    $("btnToggle").textContent = snap.enabled ? "중지" : "시작";
    $("btnToggle").className = snap.enabled ? "danger" : "primary";
    $("btnKill").textContent = cfg.kill_switch ? "킬 해제" : "킬 스위치";

    const id = snap.identity || {};
    const real = !!id.real_money;
    document.body.classList.toggle("live", real);
    $("modeBadge").className = "pill " + (real ? "live" : "");
    $("modeBadge").textContent = (id.label || cfg.mode || "") + (cfg.dry_run ? " · 모의판단" : "");
    $("liveWarn").classList.toggle("hide", !real);
    $("moneyBar").className = "moneybar" + (real ? " real" : "");
    $("moneyBar").innerHTML =
      `<span class="badge">${real ? "실제 자금" : "가상 자금"}</span>
       <span>지금 보는 숫자는 <b>${esc(id.label || "")}</b> 계좌의 것입니다 ·
       계좌 <b>${esc(id.account || "")}</b></span>
       <span class="muted" style="margin-left:auto">${esc(id.server || "")}</span>`;
  }

  // ---- 공통 동작 ----
  // lite 페이지(포지션을 안 그리는 페이지)는 시세·잔고 조회를 건너뛰는
  // 가벼운 스냅샷을 씁니다 — 로딩이 즉시 끝납니다. 메인만 전체를 받습니다.
  function isLite() { return window.PAGE?.lite !== false; }

  async function refresh() {
    try {
      const snap = await call("/autotrade" + (isLite() ? "?lite=1" : ""));
      renderCommon(snap);
      window.PAGE?.onSnapshot?.(snap);
    } catch (e) { if (!/로그인/.test(e.message)) console.warn(e.message); }
  }

  async function toggleRun() {
    const enable = !SNAP?.enabled;
    let confirmText = "";
    if (enable && SNAP?.config?.mode === "live") {
      confirmText = prompt("실제 자금이 움직입니다. 계속하려면 LIVE 를 입력하세요.") || "";
      if (confirmText.trim().toUpperCase() !== "LIVE") return toast("취소했습니다.", true);
    }
    try {
      renderCommon(await call("/autotrade/enable",
        {method: "POST", body: {enabled: enable, confirm: confirmText}}));
      toast(enable ? "자동매매를 시작했습니다." : "자동매매를 중지했습니다.");
      window.PAGE?.onSnapshot?.(SNAP);
    } catch (e) { toast(e.message, true); }
  }

  async function runOnce() {
    $("btnRun").disabled = true; $("btnRun").textContent = "실행 중…";
    try {
      const data = await call("/autotrade/run", {method: "POST"});
      renderCommon(data);
      const r = data.result || {};
      toast(r.skipped ? r.skipped
        : `회전 완료 — 신호 ${(r.signals || []).length} · 진입 ${(r.entries || []).length} · `
          + `청산 ${(r.exits || []).length} · 거부 ${(r.rejects || []).length}`);
      window.PAGE?.onSnapshot?.(SNAP);
    } catch (e) { toast(e.message, true); }
    finally { $("btnRun").disabled = false; $("btnRun").textContent = "지금 1회전"; }
  }

  async function killSwitch() {
    const on = SNAP?.config?.kill_switch;
    try {
      renderCommon(await call(on ? "/autotrade/resume" : "/autotrade/kill", {method: "POST"}));
      toast(on ? "킬 스위치를 해제했습니다." : "킬 스위치 — 신규 진입을 차단했습니다.");
    } catch (e) { toast(e.message, true); }
  }

  // ---- 로그인 ----
  function showLogin() { $("loginOverlay").classList.remove("hide"); }
  async function doLogin() {
    try {
      const res = await fetch(API + "/auth/login", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({username: $("lgUser").value, password: $("lgPass").value}),
      });
      const data = await res.json();
      if (!res.ok || !data.token) throw new Error(data.error || "로그인 실패");
      saveToken(data.token);
      $("loginOverlay").classList.add("hide");
      boot();
    } catch (e) { $("lgMsg").textContent = e.message; }
  }

  // ---- 부팅 ----
  // 페이지 데이터(로그·추천·한도)는 스냅샷을 기다리지 않고 병렬로 불러옵니다.
  // 직렬로 하면 첫 화면이 가장 느린 호출만큼 늦게 뜹니다.
  let booted = false;
  async function boot() {
    if (!booted) {
      booted = true;
      try { window.PAGE?.init?.(); } catch (e) { console.warn(e); }
      // 설정 변경은 드물어서 lite 페이지는 폴링을 느리게 (서버 부담 ↓)
      setInterval(refresh, isLite() ? 15000 : 7000);
    }
    await refresh();
  }

  // ------------------------------------------------------------------
  // 탐색 범위 선택기 — "어느 시장에서 · 어느 범위 안에서 찾을 것인가"
  // ------------------------------------------------------------------
  // AI 추적 화면과 페니 화면이 같은 선택지를 써야 합니다. 한쪽에만 코스닥이
  // 있으면 사용자는 두 화면이 다른 시장을 보고 있다고 오해합니다.
  let SCOPE = null;                     // 카탈로그 (한 번만 받아 캐시)

  async function scopeCatalog() {
    if (!SCOPE) SCOPE = await call("/autotrade/universes");
    return SCOPE;
  }

  /** 시장 체크박스 3개를 그립니다. containerId 안에 <label> 들을 채웁니다. */
  async function renderSegments(containerId, selected, onChange) {
    const box = $(containerId);
    if (!box) return;
    const cat = await scopeCatalog();
    const chosen = new Set(selected && selected.length ? selected : ["KOSPI", "KOSDAQ"]);
    box.innerHTML = cat.segments.map((s) =>
      `<label class="chk" style="margin:0"><input type="checkbox" data-seg="${s.key}"
        ${chosen.has(s.key) ? "checked" : ""}> ${esc(s.label)}</label>`).join("");
    box.querySelectorAll("[data-seg]").forEach((el) => {
      el.onchange = () => onChange?.(readSegments(containerId));
    });
  }

  function readSegments(containerId) {
    const box = $(containerId);
    if (!box) return [];
    return [...box.querySelectorAll("[data-seg]")]
      .filter((el) => el.checked).map((el) => el.dataset.seg);
  }

  /** 탐색 범위 드롭다운. 고른 시장과 겹치는 범위만 남깁니다. */
  async function renderUniverse(selectId, segments, selected) {
    const sel = $(selectId);
    if (!sel) return;
    const cat = await scopeCatalog();
    const chosen = new Set(segments && segments.length ? segments
                                                       : ["KOSPI", "KOSDAQ", "NASDAQ"]);
    const fits = cat.universes.filter((u) => u.segments.some((s) => chosen.has(s)));
    const group = (key, label) => {
      const rows = fits.filter((u) => u.group === key);
      if (!rows.length) return "";
      return `<optgroup label="${esc(label)}">` + rows.map((u) =>
        `<option value="${u.key}" title="${esc(u.note)}">${esc(u.label)}</option>`
      ).join("") + "</optgroup>";
    };
    sel.innerHTML = `<option value="">시장 전체</option>`
      + group("index", "지수 · ETF") + group("sector", "섹터");
    // 시장을 바꿔서 선택지에서 사라진 범위는 '전체'로 되돌립니다 —
    // 남겨두면 화면에는 보이는데 결과가 0건인 상태가 됩니다.
    sel.value = fits.some((u) => u.key === selected) ? selected : "";
    const hit = fits.find((u) => u.key === sel.value);
    const note = $(selectId + "Note");
    if (note) note.textContent = hit ? hit.note : "";
    return sel.value;
  }

  window.AT = {call, toast, esc, num, won, pct, sign, owlSVG,
               get SNAP() { return SNAP; }, get TOKEN() { return TOKEN; },
               refresh, $, scopeCatalog, renderSegments, readSegments, renderUniverse};

  document.addEventListener("DOMContentLoaded", () => {
    injectChrome(window.PAGE || {nav: "", title: "", desc: ""});
    TOKEN = readToken();
    if (!TOKEN) showLogin();
    else boot().catch(showLogin);
  });
})();
