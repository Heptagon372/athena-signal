"use client";

/** 분석 페이지 (/) — 종목 검색, 개장 예측, 시평선, 차트, 지표, 뉴스·커뮤니티 */

import { useCallback, useEffect, useRef, useState } from "react";
import CandleChart from "./components/CandleChart";
import Owl from "./components/Owl";
import { api, fmtBigWon, fmtMoney, fmtNum, HORIZON_LABEL, signColor, SymbolNotFound } from "./lib/api";
import { useSymbol } from "./providers";

const TIMEFRAMES = [
  { key: "minute", label: "분봉" }, { key: "day", label: "일봉" },
  { key: "week", label: "주봉" }, { key: "month", label: "월봉" },
  { key: "year", label: "년봉" },
];
const FAMILY_LABEL = { trend: "추세추종", meanrev: "평균회귀", confirm: "수급확인", info: "참고" };

export default function AnalysisPage() {
  const { symbol: saved, setSymbol, symbolReady } = useSymbol();
  const [query, setQuery] = useState(saved.key);
  const [input, setInput] = useState("");
  const [data, setData] = useState(null);
  const [chart, setChart] = useState(null);
  const [quote, setQuote] = useState(null);
  const [timeframe, setTimeframe] = useState("day");
  const [notFound, setNotFound] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [suggests, setSuggests] = useState([]);
  const pollRef = useRef(null);

  /* ---- 종목 분석 ---- */
  const load = useCallback(async (q) => {
    setLoading(true); setError(""); setNotFound(null);
    try {
      const d = await api.get(`/predict/${encodeURIComponent(q)}`, { timeout: 60000 });
      setData(d); setQuery(q);
      // 조회에 성공한 종목만 기억합니다. 오타로 조회에 실패한 문자열이 저장돼
      // 다음 방문 때마다 "종목 없음"이 뜨는 일을 막기 위함입니다.
      if (d?.symbol?.key) {
        setSymbol({ key: d.symbol.key, name: d.symbol.name, market: d.symbol.market });
        try {
          const url = new URL(window.location.href);
          url.searchParams.set("q", d.symbol.key);
          window.history.replaceState(null, "", url);
        } catch {}
      }
      return d;
    } catch (err) {
      if (err instanceof SymbolNotFound) setNotFound(err.payload);
      else setError(err.message);
      setData(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, [setSymbol]);

  // 저장된 종목을 읽기 전에 기본값으로 한 번 조회해 버리면 화면이 삼성전자로
  // 깜빡였다가 바뀝니다. symbolReady 를 기다렸다가 한 번만 부릅니다.
  const bootRef = useRef(false);
  useEffect(() => {
    if (!symbolReady || bootRef.current) return;
    bootRef.current = true;
    load(saved.key);
  }, [symbolReady, saved.key, load]);

  /* ---- 차트 ---- */
  useEffect(() => {
    if (!data?.symbol) return;
    let alive = true;
    setChart(null);
    api.get(`/chart/${encodeURIComponent(data.symbol.key)}?timeframe=${timeframe}&days=120`,
            { timeout: 45000 })
      .then((c) => { if (alive) setChart(c); })
      .catch(() => { if (alive) setChart({ empty: true, note: "차트를 불러오지 못했습니다." }); });
    return () => { alive = false; };
  }, [data?.symbol?.key, timeframe]);

  /* ---- 실시간 시세 폴링 ---- */
  useEffect(() => {
    if (!data?.symbol) return;
    const key = data.symbol.key;
    let alive = true;

    const tick = async () => {
      if (!alive) return;
      try {
        const q = await api.get(`/quote/${encodeURIComponent(key)}`, { timeout: 20000 });
        if (alive && q?.price) setQuote(q);
      } catch {}
      if (alive) {
        const open = quoteRef.current?.market_status?.is_open;
        pollRef.current = setTimeout(tick, open ? 4000 : 60000);
      }
    };
    tick();
    return () => { alive = false; clearTimeout(pollRef.current); };
  }, [data?.symbol?.key]);

  const quoteRef = useRef(null);
  quoteRef.current = quote;

  /* ---- 자동완성 ---- */
  useEffect(() => {
    if (input.trim().length < 1) { setSuggests([]); return; }
    const t = setTimeout(async () => {
      try {
        const r = await api.get(`/search?q=${encodeURIComponent(input.trim())}`, { timeout: 8000 });
        setSuggests(r.results || []);
      } catch { setSuggests([]); }
    }, 200);
    return () => clearTimeout(t);
  }, [input]);

  const submit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    load(input.trim());
    setInput(""); setSuggests([]);
  };

  const openPred = quote?.open_prediction || data?.open_prediction;
  const preds = quote?.predictions || data?.predictions || [];
  const marketOpen = (quote?.market_status || data?.market_status)?.is_open ?? false;
  const snap = data?.market_snapshot;
  const sym = data?.symbol;
  const price = quote?.price ?? snap?.current_price;
  const changeRate = quote?.change_rate ?? snap?.change_rate;

  return (
    <>
      {/* 검색 */}
      <section className="section search-section">
        <form onSubmit={submit} className="search-form">
          <div className="search-wrap">
            <input className="input" value={input} onChange={(e) => setInput(e.target.value)}
                   placeholder="종목명 또는 코드 (예: 삼성전자, 005930, AAPL)" autoComplete="off" />
            {suggests.length > 0 && (
              <div className="suggest">
                {suggests.map((s) => (
                  <button type="button" key={s.key} className="suggest-item"
                          onMouseDown={(e) => { e.preventDefault(); load(s.key); setInput(""); setSuggests([]); }}>
                    <span>{s.name}</span>
                    <span className="mono suggest-meta">{s.key} · {s.market}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <button className="btn btn-gold" type="submit">분석</button>
        </form>
      </section>

      {loading && <div className="loading">분석 중…</div>}
      {error && <section className="section"><div className="empty error-text">{error}</div></section>}

      {notFound && (
        <section className="section">
          <div className="card notfound-card">
            <div className="eyebrow">종목 조회 실패</div>
            <h2>&lsquo;{notFound.query}&rsquo; — 존재하지 않는 종목입니다</h2>
            <p>{notFound.message}</p>
            {notFound.suggestions?.length > 0 && (
              <div className="nf-chips">
                {notFound.suggestions.map((s) => (
                  <button key={s.key} className="nf-chip" onClick={() => load(s.key)}>
                    {s.name}<span className="mono">{s.key} · {s.market}</span>
                  </button>
                ))}
              </div>
            )}
            <div className="nf-hint">
              확인한 시장 · 코스피 · 코스닥 · 코넥스 · 미국(NYSE/NASDAQ)<br />
              입력 예시 · 삼성전자 / 005930 / 에코프로비엠 / AAPL
            </div>
          </div>
        </section>
      )}

      {data && sym && (
        <>
          {/* 히어로 */}
          <section className="section hero">
            <div>
              <div className="eyebrow">오늘의 전략 판단</div>
              <h1 className="hero-title">
                올빼미의 눈은<br />
                <em>{sym.name}</em>
                <span className={`market-badge ${sym.market}`}>{sym.market_label}</span><br />
                을 향해 열려 있다.
              </h1>

              {sym.leverage_warning && (
                <div className="leverage-warning">
                  ⚠ <b>{sym.long_name || sym.name}</b> — {sym.leverage_warning}
                </div>
              )}

              <div className="live-price">
                <div className="lp-row">
                  <span className="lp-value" style={{ color: signColor(changeRate) }}>
                    {fmtMoney(price, sym.currency)}
                  </span>
                  <span className="lp-change" style={{ color: signColor(changeRate) }}>
                    {changeRate >= 0 ? "▲" : "▼"} {changeRate >= 0 ? "+" : ""}{changeRate}%
                  </span>
                </div>
                <div className="lp-meta">
                  <span className={`market-state${marketOpen ? " open" : ""}`}>
                    <i />{(quote?.market_status || data.market_status)?.label}
                    {marketOpen ? " · 실시간" : ` · ${(quote?.market_status || data.market_status)?.next_event || ""}`}
                  </span>
                  <span>{quote?.source || snap?.source}</span>
                </div>
              </div>

              <div className="verified">
                종목 실존 확인 <b>{sym.verified_by}</b> · 코드 {sym.key}
              </div>
            </div>

            {/* 장 시작 전 예측 */}
            {openPred && (
              <div className="open-panel">
                <div className="open-eyebrow">장 시작 전 예측</div>
                <OpenGauge pred={openPred} />
                <div className="open-target">
                  {(quote?.market_status || data.market_status)?.label} · 다음 개장{" "}
                  <b>{openPred.target_text}</b><br />
                  전일 종가 대비 <b>{openPred.direction === "up" ? "상승" : "하락"} 출발</b> 확률
                </div>
                <div className="open-formula">
                  <div style={{ marginBottom: 7 }}>
                    raw = <b>w</b>기술×기술점수×0.85 + <b>w</b>뉴스×뉴스점수×1.35 + <b>w</b>커뮤×커뮤점수
                  </div>
                  {[["기술", "technical", "갭 설명력 낮춤"],
                    ["뉴스", "news", "밤사이 뉴스 = 갭 주동인"],
                    ["커뮤", "community", ""]].map(([label, key, note]) => {
                    const t = openPred.terms?.[key];
                    if (!t) return null;
                    return (
                      <div className="f-term" key={key}>
                        <span>{label} = {t.weight} × {t.score >= 0 ? "+" : ""}{t.score} × {t.multiplier}
                          {note && <i> {note}</i>}</span>
                        <span style={{ color: signColor(t.contribution) }}>
                          {t.contribution >= 0 ? "+" : ""}{t.contribution.toFixed(4)}
                        </span>
                      </div>
                    );
                  })}
                  <div className="f-term f-total">
                    <span>raw 합계</span>
                    <span style={{ color: signColor(openPred.raw_score) }}>
                      {openPred.raw_score >= 0 ? "+" : ""}{openPred.raw_score}
                    </span>
                  </div>
                  <div className="f-term">
                    <span>확률 = 100 / (1 + e^(−{openPred.sigmoid_k} × raw))</span>
                    <span style={{ color: signColor(openPred.raw_score) }}>
                      {openPred.probability_up}% 상승
                    </span>
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* 시평선 */}
          <section className="section">
            <div className="section-head">
              <div>
                <span className="eyebrow">시평선별 예측</span>
                <p>짧을수록 노이즈가 커 50%에 수렴합니다. 분봉/일봉 배합 비율과 신뢰도가 시평선마다 다릅니다.</p>
              </div>
            </div>
            <div className="hz-strip">
              {preds.map((p) => (
                <div className="hz-cell" key={p.horizon}>
                  <div className="hz-label">{p.label || HORIZON_LABEL[p.horizon]}</div>
                  <div className="hz-prob" style={{ color: p.direction === "up" ? "var(--up)" : "var(--down)" }}>
                    {p.direction === "up" ? "▲" : "▼"} {p.probability}%
                  </div>
                  <div className="hz-bar">
                    <i style={{ width: `${Math.max(0, Math.min(100, (p.probability - 50) * 2))}%`,
                                background: p.direction === "up" ? "var(--up)" : "var(--down)" }} />
                  </div>
                  <div className="hz-meta">분봉 {Math.round(p.intraday_share * 100)}% · 신뢰 {p.confidence}</div>
                </div>
              ))}
            </div>
          </section>

          {/* 올빼미 + 시세 */}
          <section className="section owl-section">
            <Owl direction={openPred?.direction || "up"}
                 probability={openPred?.probability || 50}
                 marketOpen={marketOpen} name={sym.name} />
            <div>
              <div className="section-head" style={{ marginBottom: 14 }}>
                <div>
                  <div className="eyebrow">시세 현황</div>
                  <h2 style={{ fontSize: 21 }}>{sym.name} ({sym.key})</h2>
                </div>
              </div>
              {snap && <Snapshot snap={snap} currency={sym.currency} />}
            </div>
          </section>

          {/* 차트 */}
          <section className="section">
            <div className="section-head">
              <div>
                <div className="eyebrow">차트</div>
                <h2>가격이 실제로 그린 그림</h2>
                <p>캔들·이동평균·볼린저밴드·거래량·RSI·MACD 모두 실제 수신한 데이터로 그립니다.
                   차트 오른쪽 위 <b>[지표]</b>에서 CCI·스토캐스틱·MFI·ATR·OBV 를 켜고 끌 수 있습니다.</p>
              </div>
              <div className="tf-picker">
                {TIMEFRAMES.map((tf) => (
                  <button key={tf.key} className={timeframe === tf.key ? "active" : ""}
                          onClick={() => setTimeframe(tf.key)}>{tf.label}</button>
                ))}
              </div>
            </div>
            {chart ? <CandleChart data={chart} /> : <div className="loading">차트 불러오는 중…</div>}
          </section>

          {/* 국면 + 지표 */}
          <section className="section">
            <div className="section-head">
              <div>
                <div className="eyebrow">기술적 지표 근거</div>
                <h2>숫자로 확인하는 판단</h2>
                <p>지표마다 실제 계산값과 판단 근거, 최종 점수 기여도를 함께 표시합니다.</p>
              </div>
            </div>
            {data.technical?.regime && <Regime regime={data.technical.regime} />}
            <div className="ind-rows">
              <div className="row head ind-row"><span>지표</span><span>계산값</span>
                <span>판단 근거</span><span>기여도</span></div>
              {data.technical?.indicators?.map((i) => (
                <div className="row ind-row" key={i.key}>
                  <div>
                    <span className={`ind-dot ${i.verdict}`} />
                    {i.label}
                    {i.family && <span className={`fam-tag ${i.family}`}>{FAMILY_LABEL[i.family]}</span>}
                  </div>
                  <div className="m" style={{ color: signColor(i.score) }}>{i.value_text}</div>
                  <div className="ind-reason">
                    {i.reason}
                    <div className="row-sub">점수 {i.score >= 0 ? "+" : ""}{i.score} × 비중 {i.weight}
                      {i.formula ? ` | ${i.formula}` : ""}</div>
                  </div>
                  <div className="m" style={{ color: signColor(i.contribution), textAlign: "right" }}>
                    {i.contribution >= 0 ? "+" : ""}{i.contribution}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 뉴스 · 커뮤니티 */}
          <section className="section">
            <div className="section-head">
              <div>
                <div className="eyebrow">근거 & 여론</div>
                <h2>왜 이 확률이 나왔을까</h2>
              </div>
            </div>
            <div className="rc-grid">
              <div className="rationale-list">
                {data.rationale?.slice(0, 10).map((it, idx) => (
                  <div className="rationale-item" key={idx}>
                    <span className={`rationale-tag ${it.source_type}`}>
                      {{ technical: "기술", news: "뉴스", community: "커뮤" }[it.source_type] || it.source_type}
                    </span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="rationale-text">{it.text}</div>
                      {it.source_name && <div className="row-sub">{it.source_name}</div>}
                    </div>
                    <span className="m" style={{ color: signColor(it.influence) }}>
                      {it.influence >= 0 ? "+" : ""}{it.influence}
                    </span>
                  </div>
                ))}
              </div>
              <div>
                <div className="feed-head"><span className="eyebrow">뉴스</span>
                  <span className="mono row-sub">
                    {data.news_summary?.total}건 · 긍정 {data.news_summary?.positive} / 부정 {data.news_summary?.negative}
                  </span>
                </div>
                <div className="feed-list">
                  {data.news_sample?.slice(0, 6).map((nItem, i) => (
                    <a className="feed-item" key={i} href={nItem.url} target="_blank" rel="noopener noreferrer">
                      <div className={`feed-stance ${nItem.sentiment > 0.1 ? "bullish" : nItem.sentiment < -0.1 ? "bearish" : "neutral"}`} />
                      <div>
                        <div className="feed-title">{nItem.title}</div>
                        <div className="row-sub">{nItem.source} · {nItem.sentiment >= 0 ? "+" : ""}{nItem.sentiment.toFixed(2)}</div>
                      </div>
                    </a>
                  ))}
                </div>

                <div className="feed-head" style={{ marginTop: 18 }}>
                  <span className="eyebrow">커뮤니티</span>
                  <span className="mono row-sub">
                    {data.community?.post_count}건 · 매수 {data.community?.bullish_count} / 매도 {data.community?.bearish_count}
                  </span>
                </div>
                <div className="feed-sources">
                  {Object.entries(data.community?.source_counts || {}).map(([src, cnt]) => (
                    <span className="src-chip" key={src}>{src} <b>{cnt}</b></span>
                  ))}
                </div>
                <div className="feed-list">
                  {data.community?.recent_posts?.slice(0, 6).map((p, i) => (
                    <div className="feed-item" key={i}>
                      <div className={`feed-stance ${p.stance}`} />
                      <div>
                        <div className="feed-title">{p.title}</div>
                        <div className="row-sub">{p.source}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </>
      )}
    </>
  );
}

function OpenGauge({ pred }) {
  const size = 168, r = 74;
  const circ = 2 * Math.PI * r;
  const off = circ * (1 - pred.probability / 100);
  const col = pred.direction === "up" ? "var(--up)" : "var(--down)";
  return (
    <svg width={size} height={size} viewBox="0 0 168 168">
      <circle cx="84" cy="84" r={r} fill="none" stroke="var(--hairline)" strokeWidth="8" />
      <circle cx="84" cy="84" r={r} fill="none" stroke={col} strokeWidth="8" strokeLinecap="round"
              strokeDasharray={circ} strokeDashoffset={off} transform="rotate(-90 84 84)" />
      <circle cx="84" cy="84" r={r * 0.42} fill={col} opacity="0.12" />
      <text x="84" y="82" textAnchor="middle" fill="var(--text)" fontSize="34" fontWeight="600">
        {pred.probability}%
      </text>
      <text x="84" y="104" textAnchor="middle" fill={col} fontFamily="monospace" fontSize="13">
        {pred.direction === "up" ? "▲ 상승 출발" : "▼ 하락 출발"}
      </text>
    </svg>
  );
}

function Snapshot({ snap, currency }) {
  const cells = [
    ["현재가", fmtMoney(snap.current_price, currency), snap.change_rate],
    ["등락률", `${snap.change_rate >= 0 ? "+" : ""}${snap.change_rate}%`, snap.change_rate],
    ["시가", fmtMoney(snap.open, currency)], ["전일종가", fmtMoney(snap.prev_close, currency)],
    ["고가", fmtMoney(snap.high, currency)], ["저가", fmtMoney(snap.low, currency)],
    ["거래량", `${fmtNum(snap.volume)}주`],
    ["거래대금", currency === "KRW" ? fmtBigWon(snap.trading_value) : `$${fmtNum(snap.trading_value)}`],
  ];
  if (snap.high_52w) cells.push(["52주 최고", fmtMoney(snap.high_52w, currency)]);
  if (snap.low_52w) cells.push(["52주 최저", fmtMoney(snap.low_52w, currency)]);
  if (snap.market_cap_text) cells.push(["시가총액", snap.market_cap_text]);
  if (snap.per) cells.push(["PER / PBR", `${snap.per} / ${snap.pbr || "—"}`]);

  const flow = snap.investor_flow || {};
  return (
    <>
      <div className="stat-grid">
        {cells.map(([label, value, color]) => (
          <div key={label}>
            <div className="stat-label">{label}</div>
            <div className="stat-value" style={color !== undefined ? { color: signColor(color) } : undefined}>
              {value}
            </div>
          </div>
        ))}
      </div>
      <div className="investor-row">
        {[["개인", flow.individual_net, flow.individual_net_value],
          ["외국인", flow.foreign_net, flow.foreign_net_value],
          ["기관", flow.institution_net, flow.institution_net_value]].map(([label, qty, val]) => (
          <div key={label}>
            <div className="stat-label">{label}</div>
            <div className="stat-value" style={{ fontSize: 14, color: signColor(qty) }}>
              {qty == null ? "—" : `${qty > 0 ? "+" : ""}${fmtNum(qty)}주`}
            </div>
            <div className="stat-sub">{val == null ? "" : fmtBigWon(val)}</div>
          </div>
        ))}
      </div>
      <div className="source-note">
        시세 출처 · {snap.source}<br />
        투자자별 순매매 · {flow.source || "데이터 없음"}
        {flow.trade_date ? ` (${flow.trade_date} 기준)` : ""}
      </div>
    </>
  );
}

function Regime({ regime }) {
  const cls = { TREND: "trend", MEAN_REVERT: "meanrev", RANDOM: "random" }[regime.regime] || "random";
  const m = regime.multipliers || {};
  return (
    <div className={`regime-card ${cls}`}>
      <div className="regime-head">
        <span className={`regime-badge ${cls}`}>{regime.label}</span>
        <span className="mono row-sub">추세점수 {regime.trend_score >= 0 ? "+" : ""}{regime.trend_score}</span>
      </div>
      <div className="regime-strategy">{regime.strategy}</div>
      <div className="regime-evidence">{(regime.evidence || []).join(" · ")}</div>
      <div className="regime-evidence" style={{ marginTop: 8 }}>
        적용 배수 — 추세추종 <b>×{m.trend}</b> &nbsp; 평균회귀 <b>×{m.meanrev}</b> &nbsp; 수급확인 <b>×{m.confirm}</b>
      </div>
    </div>
  );
}
