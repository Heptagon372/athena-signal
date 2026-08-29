/* ==========================================================================
   자동매매 콘솔 공통 스크립트
   --------------------------------------------------------------------------
   모든 페이지가 이 파일 하나로 다음을 공유합니다.
       로그인(토큰) 처리 + 로그인 오버레이
       서버 호출 call() + 404/버전 불일치 감지 (서버 재시작 배너)
       공통 헤더(부엉이, 내비게이션, 상태 배지, 시작/중지, 킬 스위치) 주입
       계좌 정체 표시줄(가상/실제 자금) + 실전 모드 붉은 테두리
       7초 주기 상태 새로고침 → 각 페이지의 PAGE.onSnapshot(snap) 호출

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
  const REQUIRED_API = 13;      // 13: AI 에이전트 분석 화면 (/api/agents/...)

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
    (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

  const won = (v) => (v == null || isNaN(v)) ? "-" : Math.round(v).toLocaleString("ko-KR") + "원";
  const num = (v, d = 2) => (v == null || isNaN(v)) ? "-"
    : Number(v).toLocaleString("ko-KR", {maximumFractionDigits: d});
  const pct = (v) => (v == null || isNaN(v)) ? "-" : (v >= 0 ? "+" : "") + Number(v).toFixed(2) + "%";
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
    // https 로 서비스될 때는 Secure — 토큰이 평문 http 로 새지 않게
    const secure = location.protocol === "https:" ? "; Secure" : "";
    document.cookie = `athena_token=${t}; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax${secure}`;
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
      throw new Error("서버가 이전 버전 코드로 실행 중입니다. 서버를 재시작하세요.");
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

  // ---- 걸어다니는 부엉이 (수익률 옆에서 계좌를 지켜봅니다) ----
  // 수익이면 신이 나서 빨리 걷고, 손실이면 느리게 걷습니다. 장식이지만
  // "지금 벌고 있나"를 숫자를 읽기 전에 알려주는 역할도 합니다.
  function owlWalkSVG(size = 44) {
    return `<svg class="owl-walk" width="${size}" height="${size}" viewBox="0 0 64 64"
                 aria-hidden="true">
      <g class="legs">
        <path class="leg l" d="M26 52 L26 58 M22 58 h8" stroke="#c9762a"
              stroke-width="2.2" fill="none" stroke-linecap="round"/>
        <path class="leg r" d="M38 52 L38 58 M34 58 h8" stroke="#c9762a"
              stroke-width="2.2" fill="none" stroke-linecap="round"/>
      </g>
      <g class="body">
        <path d="M15 25 Q11 13 19 15 Q23 9 28 16 Z" fill="#3a2f22"/>
        <path d="M49 25 Q53 13 45 15 Q41 9 36 16 Z" fill="#3a2f22"/>
        <ellipse cx="32" cy="37" rx="20" ry="19" fill="#3a2f22"/>
        <ellipse cx="32" cy="43" rx="10" ry="11" fill="#4a3c2b"/>
        <path d="M23 45 Q32 40 41 45 M24 50 Q32 45 40 50" stroke="#5c4a34"
              stroke-width="1.2" fill="none"/>
        <g class="wing l"><path d="M14 30 Q8 40 16 50 Q20 42 19 31 Z" fill="#4a3c2b"/></g>
        <g class="wing r"><path d="M50 30 Q56 40 48 50 Q44 42 45 31 Z" fill="#4a3c2b"/></g>
        <circle cx="24" cy="30" r="8.2" fill="#14110d" stroke="#c9a227" stroke-width="1.6"/>
        <circle cx="40" cy="30" r="8.2" fill="#14110d" stroke="#c9a227" stroke-width="1.6"/>
        <circle class="pupil" cx="25" cy="30.5" r="3.3" fill="#c9a227"/>
        <circle class="pupil" cx="41" cy="30.5" r="3.3" fill="#c9a227"/>
        <circle cx="26" cy="29.2" r="1" fill="#ece3d0"/>
        <circle cx="42" cy="29.2" r="1" fill="#ece3d0"/>
        <ellipse class="lid" cx="24" cy="30" rx="8.2" ry="8.2" fill="#3a2f22"/>
        <ellipse class="lid r" cx="40" cy="30" rx="8.2" ry="8.2" fill="#3a2f22"/>
        <path d="M32 34 L29.5 38.5 L34.5 38.5 Z" fill="#c9762a"/>
      </g>
    </svg>`;
  }

  // ---- 공통 헤더 + 내비게이션 주입 ----
  const NAV = [
    ["main", "/autotrade", "메인"],
    ["unified", "/autotrade/unified", "통합"],
    ["kr", "/autotrade/kr", "한국 주식"],
    ["us", "/autotrade/us", "미국 주식"],
    ["ai", "/autotrade/ai", "AI 추천"],
    // 위의 AI 추천과 다른 화면입니다. 저쪽은 팩터 점수로 종목을 고르고,
    // 이쪽은 LLM 에이전트들이 토론해 매수 매도 보유를 정합니다.
    ["agents", "/autotrade/agents", "AI 에이전트"],
    ["penny", "/autotrade/penny", "페니 초단타"],
    ["deriv", "/autotrade/deriv", "선물 옵션"],
    ["backtest", "/autotrade/backtest", "백테스트"],
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
    // 페이지 머리글과 계좌 정체 표시줄은 넓은 화면에서 한 줄에 나란히 섭니다
    main.insertAdjacentHTML("afterbegin", `
      <div class="topline">
        <div class="pagehead">${owlSVG(46, true)}
          <div><h1>${esc(page.title || "")}</h1><p>${esc(page.desc || "")}</p></div></div>
        <div id="moneyBar" class="moneybar"></div>
      </div>
      <div id="liveWarn" class="warnbox hide"><b>실거래 모드입니다.</b>
        이 화면의 모든 숫자와 주문은 실제 계좌, 실제 자금입니다.</div>`);

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
      (snap.state_label || "") + (snap.state_reason ? `, ${snap.state_reason}` : "");
    $("btnToggle").textContent = snap.enabled ? "중지" : "시작";
    $("btnToggle").className = snap.enabled ? "danger" : "primary";
    $("btnKill").textContent = cfg.kill_switch ? "킬 해제" : "킬 스위치";

    const id = snap.identity || {};
    const real = !!id.real_money;
    document.body.classList.toggle("live", real);
    $("modeBadge").className = "pill " + (real ? "live" : "");
    $("modeBadge").textContent = (id.label || cfg.mode || "") + (cfg.dry_run ? "   모의판단" : "");
    $("liveWarn").classList.toggle("hide", !real);
    $("moneyBar").className = "moneybar" + (real ? " real" : "");
    $("moneyBar").innerHTML =
      `<span class="badge">${real ? "실제 자금" : "가상 자금"}</span>
       <span>지금 보는 숫자는 <b>${esc(id.label || "")}</b> 계좌의 것입니다 , 
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
        : `회전 완료, 신호 ${(r.signals || []).length}   진입 ${(r.entries || []).length}   `
          + `청산 ${(r.exits || []).length}   거부 ${(r.rejects || []).length}`);
      window.PAGE?.onSnapshot?.(SNAP);
    } catch (e) { toast(e.message, true); }
    finally { $("btnRun").disabled = false; $("btnRun").textContent = "지금 1회전"; }
  }

  async function killSwitch() {
    const on = SNAP?.config?.kill_switch;
    try {
      renderCommon(await call(on ? "/autotrade/resume" : "/autotrade/kill", {method: "POST"}));
      toast(on ? "킬 스위치를 해제했습니다." : "킬 스위치, 신규 진입을 차단했습니다.");
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

  /**
   * 탐색 범위 **다중 선택** — 고른 범위들의 합집합에서 찾습니다.
   * 아무것도 고르지 않으면 시장 전체입니다.
   *
   * 드롭다운이 아니라 칩(누르면 켜지는 단추)인 이유: 범위가 34개라 다중 선택
   * <select> 로 만들면 Ctrl 을 누른 채 골라야 하고, 지금 무엇이 켜져 있는지
   * 스크롤해야 보입니다. 칩은 켜진 것이 한눈에 보입니다.
   *
   * 돌려주는 값은 **실제로 남은 선택**입니다 — 시장을 바꿔서 겹치지 않게 된
   * 범위는 빠집니다(코스피만 켜 놓고 '나스닥100'이 남아 있으면 결과가 0건이
   * 되는데 사용자는 이유를 알 수 없습니다).
   */
  async function renderPools(containerId, segments, selected, onChange) {
    const box = $(containerId);
    if (!box) return [];
    const cat = await scopeCatalog();
    const segs = new Set(segments && segments.length ? segments
                                                     : ["KOSPI", "KOSDAQ", "NASDAQ"]);
    const fits = cat.universes.filter((u) => u.segments.some((s) => segs.has(s)));
    const fitKeys = new Set(fits.map((u) => u.key));
    const chosen = new Set((selected || [])
      .map((k) => String(k || "").trim().toUpperCase())
      .filter((k) => fitKeys.has(k)));

    const group = (key, label) => {
      const rows = fits.filter((u) => u.group === key);
      if (!rows.length) return "";
      return `<div class="poolgroup"><span class="lb">${esc(label)}</span>`
        + rows.map((u) => `<button type="button" data-pool="${u.key}"
             class="poolchip${chosen.has(u.key) ? " on" : ""}"
             title="${esc(u.note || "")}">${esc(u.label)}</button>`).join("")
        + `</div>`;
    };
    box.innerHTML = `<div class="poolbox">${group("index", "지수 ETF")}`
      + `${group("sector", "섹터")}</div>`;
    box.querySelectorAll("[data-pool]").forEach((el) => {
      el.onclick = () => {
        el.classList.toggle("on");
        onChange?.(readPools(containerId));
      };
    });
    return [...chosen];
  }

  function readPools(containerId) {
    const box = $(containerId);
    if (!box) return [];
    return [...box.querySelectorAll("[data-pool].on")].map((el) => el.dataset.pool);
  }

  /** 탐색 범위 드롭다운 (단일 선택). 페니 초단타 화면이 씁니다. */
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
      + group("index", "지수   ETF") + group("sector", "섹터");
    // 시장을 바꿔서 선택지에서 사라진 범위는 '전체'로 되돌립니다 —
    // 남겨두면 화면에는 보이는데 결과가 0건인 상태가 됩니다.
    sel.value = fits.some((u) => u.key === selected) ? selected : "";
    const hit = fits.find((u) => u.key === sel.value);
    const note = $(selectId + "Note");
    if (note) note.textContent = hit ? hit.note : "";
    return sel.value;
  }

  // ------------------------------------------------------------------
  // 순수익률 곡선 — 오늘 기점으로 그려 나가는 그래프
  // ------------------------------------------------------------------
  // 세 화면(메인·시장별·통합)이 같은 그림을 씁니다. 화면마다 따로 그리면
  // 같은 계좌를 두고 서로 다른 축·다른 정의의 그래프가 생깁니다.
  //
  // 라이브러리를 쓰지 않고 SVG 를 직접 그립니다 — 이 콘솔은 인터넷이 끊긴
  // 환경에서도 열려야 하고(자동매매를 멈추려는데 화면이 안 뜨면 그게 사고),
  // 선 하나에 CDN 을 물릴 이유가 없습니다.

  const CURVE_COLORS = {ALL: "#2e251a", KR: "#8f6d17", US: "#5b6942"};
  const CURVE_NAMES = {ALL: "전체", KR: "한국", US: "미국"};

  /** 종목코드 → "KR" | "US". engine/allocation.py 의 scope_of 와 같은 규칙입니다
   *  (한국은 6자리 숫자, 미국은 영문 티커). 규칙이 갈리면 화면이 미국으로
   *  분류한 종목을 엔진은 한국 예산에서 사게 됩니다. */
  const scopeOf = (key) => /^\d/.test(String(key || "")) ? "KR" : "US";

  const hhmm = (sec) => {
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  };

  /** 점 목록 → SVG path (선을 그릴 수 없으면 빈 문자열). */
  function curvePath(points, x, y) {
    if (!points.length) return "";
    return points.map((p, i) => `${i ? "L" : "M"}${x(p).toFixed(1)},${y(p).toFixed(1)}`)
      .join(" ");
  }

  /**
   * 곡선을 그립니다.
   *   host   : svg 를 넣을 요소
   *   data   : /autotrade/curve 응답
   *   scopes : 그릴 시장 (["ALL"] · ["KR","US"] · ["ALL","KR","US"])
   *   field  : "pct"(그날 기준) | "cum_pct"(지난 날들까지 복리로 누적)
   */
  function drawCurve(host, data, {scopes = ["ALL"], field = "pct", height = 220} = {}) {
    const series = (data && data.series) || {};
    const shown = scopes.filter((s) => (series[s] || []).length);
    if (!shown.length) {
      host.innerHTML = `<div class="curve-empty">아직 표본이 없습니다.
        자동매매를 켜거나 이 화면을 열어두면 30초마다 한 점씩 그려집니다.</div>`;
      return;
    }

    // viewBox 를 **실제 픽셀 폭에 맞춥니다.** 고정 폭(1000)으로 그리고 CSS 로
    // 100% 늘리면 좁은 칸에서는 통째로 축소돼 글씨가 9px 까지 줄어듭니다
    // (실측: 626px 칸에서 15px 눈금이 9.4px 로 렌더링). 1 사용자단위 = 1 CSS px
    // 로 맞추면 눈금 글씨가 적어둔 크기 그대로 나옵니다.
    const inner = Math.round((host.clientWidth || 0) - 8);
    const W = Math.max(360, inner > 0 ? inner : 900);
    const H = height, PAD = {l: 62, r: 62, t: 16, b: 28};
    const multiDay = (data.days || 1) > 1;
    // x 축 — 하루치는 시각(초), 여러 날은 표본 순서를 그대로 씁니다
    // (장이 안 열린 밤을 실제 시간으로 비워두면 그래프의 90% 가 빈칸입니다).
    const index = new Map();
    if (multiDay) {
      const stamps = [...new Set(shown.flatMap((s) => series[s].map((p) => p.ts)))].sort();
      stamps.forEach((ts, i) => index.set(ts, i));
    }
    const xs = shown.flatMap((s) => series[s].map(
      (p) => multiDay ? index.get(p.ts) : p.sec));
    const ys = shown.flatMap((s) => series[s].map((p) => p[field] ?? 0));
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(0, ...ys), y1 = Math.max(0, ...ys);
    const span = (y1 - y0) || 1;
    y0 -= span * 0.12; y1 += span * 0.12;

    const px = (p) => PAD.l + ((multiDay ? index.get(p.ts) : p.sec) - x0)
      / ((x1 - x0) || 1) * (W - PAD.l - PAD.r);
    const py = (p) => PAD.t + (y1 - (p[field] ?? 0)) / ((y1 - y0) || 1) * (H - PAD.t - PAD.b);
    const yOf = (v) => PAD.t + (y1 - v) / ((y1 - y0) || 1) * (H - PAD.t - PAD.b);

    // y 눈금 — 0% 는 반드시 넣습니다 (본전선이 없으면 벌었는지 알 수 없습니다)
    const ticks = [];
    const step = Math.max(0.05, Number(((y1 - y0) / 4).toPrecision(1)));
    for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) ticks.push(v);
    if (!ticks.some((v) => Math.abs(v) < 1e-9)) ticks.push(0);

    const first = series[shown[0]];
    const primary = shown[0];
    const area = `${curvePath(series[primary], px, py)} L${px(first[first.length - 1]).toFixed(1)},`
      + `${yOf(0).toFixed(1)} L${px(first[0]).toFixed(1)},${yOf(0).toFixed(1)} Z`;
    const lastVal = series[primary][series[primary].length - 1][field] ?? 0;

    host.innerHTML = `
      <svg class="curve-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"
           role="img" aria-label="순수익률 곡선">
        <defs>
          <linearGradient id="curveFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="${lastVal >= 0 ? "#cf2f27" : "#2560bd"}"
                  stop-opacity=".16"/>
            <stop offset="1" stop-color="${lastVal >= 0 ? "#cf2f27" : "#2560bd"}"
                  stop-opacity="0"/>
          </linearGradient>
        </defs>
        ${ticks.map((v) => `
          <line class="grid ${Math.abs(v) < 1e-9 ? "zero" : ""}" x1="${PAD.l}"
                x2="${W - PAD.r}" y1="${yOf(v).toFixed(1)}" y2="${yOf(v).toFixed(1)}"/>
          <text class="ytick" x="${PAD.l - 8}" y="${(yOf(v) + 4).toFixed(1)}"
                text-anchor="end">${Math.abs(v) < 1e-9 ? "" : v > 0 ? "+" : ""}${v.toFixed(2)}%</text>`).join("")}
        <path class="curve-area" d="${area}" fill="url(#curveFill)"/>
        ${shown.map((s) => `<path class="curve-line" d="${curvePath(series[s], px, py)}"
            stroke="${CURVE_COLORS[s] || "#2e251a"}"
            stroke-width="${s === primary ? 2.6 : 1.8}"
            stroke-dasharray="${s === primary ? "" : "5 4"}"/>`).join("")}
        ${shown.map((s) => {
          const p = series[s][series[s].length - 1];
          return `<circle class="curve-dot" cx="${px(p).toFixed(1)}" cy="${py(p).toFixed(1)}"
                          r="4" fill="${CURVE_COLORS[s] || "#2e251a"}"/>`;
        }).join("")}
        <text class="xtick" x="${PAD.l}" y="${H - 6}">${
          multiDay ? esc(series[primary][0].date) : hhmm(series[primary][0].sec)}</text>
        <text class="xtick" x="${W - PAD.r}" y="${H - 6}" text-anchor="end">${
          multiDay ? esc(series[primary][series[primary].length - 1].date)
                   : hhmm(series[primary][series[primary].length - 1].sec)}</text>
      </svg>`;
  }

  /**
   * 수익률 카드 한 장 (큰 숫자 + 걸어다니는 부엉이 + 곡선).
   * 돌려주는 객체의 reload() 를 부르면 새 표본을 받아 다시 그립니다.
   */
  function curveCard(hostId, {scopes = ["ALL"], title = "오늘 순수익률",
                             days = 1, height = 220} = {}) {
    const host = $(hostId);
    if (!host) return {reload: async () => {}};
    let state = {days, data: null};

    host.innerHTML = `
      <div class="curve-card">
        <div class="curve-head">
          <div class="owlbox">${owlWalkSVG(52)}</div>
          <div class="curve-now">
            <div class="k">${esc(title)}</div>
            <div class="v" data-v>0.00%</div>
            <div class="s" data-s></div>
          </div>
          <div class="curve-range" data-range>
            ${[[1, "오늘"], [7, "7일"], [30, "30일"]].map(([d, l]) =>
              `<button class="sm ${d === days ? "on" : ""}" data-days="${d}">${l}</button>`
            ).join("")}
          </div>
        </div>
        <div class="curve-legend" data-legend></div>
        <div class="curve-plot" data-plot style="min-height:${height}px"></div>
        <div class="note" data-note></div>
      </div>`;

    host.querySelectorAll("[data-range] button").forEach((b) => {
      b.onclick = () => {
        state.days = Number(b.dataset.days);
        host.querySelectorAll("[data-range] button")
            .forEach((x) => x.classList.toggle("on", x === b));
        api.reload();
      };
    });

    const api = {
      state,
      async reload() {
        try {
          const data = await call(`/autotrade/curve?days=${state.days}`);
          state.data = data;
          api.draw();
        } catch (e) {
          host.querySelector("[data-plot]").innerHTML =
            `<div class="curve-empty">${esc(e.message)}</div>`;
        }
      },
      draw() {
        const data = state.data;
        if (!data) return;
        const field = state.days > 1 ? "cum_pct" : "pct";
        drawCurve(host.querySelector("[data-plot]"), data, {scopes, field, height});

        const head = (data.series || {})[scopes[0]] || [];
        const last = head[head.length - 1];
        const value = last ? (last[field] ?? 0) : 0;
        const box = host.querySelector("[data-v]");
        box.textContent = pct(value);
        box.className = "v " + sign(value);
        host.querySelector("[data-s]").innerHTML = last
          ? `${won(last.pnl)}   기준 자본 ${won(last.base)}`
          : "표본을 모으는 중입니다";
        const owl = host.querySelector(".owl-walk");
        owl.classList.toggle("up", value > 0);
        owl.classList.toggle("down", value < 0);

        host.querySelector("[data-legend]").innerHTML = scopes.map((s) => {
          const rows = (data.series || {})[s] || [];
          const v = rows.length ? (rows[rows.length - 1][field] ?? 0) : null;
          return `<span class="lg-item"><i style="background:${CURVE_COLORS[s]}"></i>
            ${esc(CURVE_NAMES[s] || s)}
            <b class="${v == null ? "muted" : sign(v)}">${v == null ? "-" : pct(v)}</b></span>`;
        }).join("");
        host.querySelector("[data-note]").textContent =
          state.days > 1
            ? "지난 날들의 마감 수익률을 복리로 이어 붙인 값입니다. 수수료, 거래세, "
              + "그리고 지금 다 팔았을 때 더 낼 비용까지 뺀 순수익률입니다."
            : "오늘 첫 표본이 0% 입니다. 수수료, 거래세, 그리고 지금 다 팔았을 때 "
              + "더 낼 비용까지 뺀 순수익률입니다.";
      },
    };
    // 창 폭이 바뀌면 다시 그립니다. viewBox 를 픽셀 폭에 맞춰 그리기 때문에,
    // 다시 그리지 않으면 늘어난 칸에서 예전 폭 그대로 확대돼 글씨가 커집니다.
    // 자료를 다시 받지는 않습니다 (이미 받아둔 것으로 다시 그리기만).
    if (window.ResizeObserver) {
      let last = 0, timer = null;
      new ResizeObserver((entries) => {
        const width = Math.round(entries[0].contentRect.width);
        if (Math.abs(width - last) < 12) return;
        last = width;
        clearTimeout(timer);
        timer = setTimeout(() => api.draw(), 120);
      }).observe(host.querySelector("[data-plot]"));
    }

    api.reload();
    return api;
  }

  // ------------------------------------------------------------------
  // 시장 카드 — 한국·미국이 지금 얼마를 쓰고 있는가
  // ------------------------------------------------------------------
  function marketCards(hostId, snap, {only = null, link = true} = {}) {
    const host = $(hostId);
    if (!host) return;
    const rows = (snap && snap.markets) || [];
    if (!rows.length) {
      host.innerHTML = `<div class="note">계좌를 읽는 중입니다.</div>`;
      return;
    }
    const href = {KR: "/autotrade/kr", US: "/autotrade/us"};
    host.innerHTML = rows.filter((m) => !only || m.scope === only).map((m) => {
      const usedPct = m.budget_krw > 0 ? m.used_krw / m.budget_krw * 100 : 0;
      return `<div class="mcard ${m.enabled ? "" : "off"}">
        <div class="mtop">
          <b>${esc(m.label)} 주식</b>
          <span class="pill ${m.enabled ? "on" : "off"}"><span class="dot"></span>
            <span>${m.enabled ? "매수 켬" : "매수 끔"}</span></span>
          ${link ? `<a class="go" href="${href[m.scope]}">${esc(m.label)} 페이지 가기</a>` : ""}
        </div>
        <div class="mbar"><i style="width:${Math.min(usedPct, 100).toFixed(1)}%"></i></div>
        <div class="mgrid">
          <div><span>배분</span><b>${num(m.alloc_pct, 1)}%</b></div>
          <div><span>예산</span><b>${won(m.budget_krw)}</b></div>
          <div><span>쓴 돈</span><b>${won(m.used_krw)}</b></div>
          <div><span>더 살 수 있는 돈</span><b>${won(m.cash_krw)}</b></div>
          <div><span>보유</span><b>${m.positions} / ${m.max_positions}종목</b></div>
          <div><span>평가손익</span><b class="${sign(m.unrealized_krw)}">${won(m.unrealized_krw)}</b></div>
        </div>
        ${m.blocked ? `<div class="mblock">${esc(m.blocked)}</div>` : ""}
      </div>`;
    }).join("");
  }

  window.AT = {call, toast, esc, num, won, pct, sign, owlSVG, owlWalkSVG,
               get SNAP() { return SNAP; }, get TOKEN() { return TOKEN; },
               refresh, $, scopeCatalog, renderSegments, readSegments, renderUniverse,
               renderPools, readPools,
               curveCard, drawCurve, marketCards, scopeOf,
               CURVE_COLORS, CURVE_NAMES};

  document.addEventListener("DOMContentLoaded", () => {
    injectChrome(window.PAGE || {nav: "", title: "", desc: ""});
    TOKEN = readToken();
    if (!TOKEN) showLogin();
    else boot().catch(showLogin);
  });
})();
