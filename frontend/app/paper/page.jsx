"use client";

/**
 * 모의투자 페이지 (/paper) — 가상 자금 트레이딩 데스크
 *
 * 종목 검색 → 실시간 시세 → 차트 → 주문표 → 미체결 → 보유 → 거래내역 순으로,
 * 증권사 프로그램의 흐름을 그대로 따릅니다.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import CandleChart from "../components/CandleChart";
import RequireAuth from "../components/RequireAuth";
import { api, fmtNum, signColor, SymbolNotFound } from "../lib/api";
import { useAuth, useSymbol } from "../providers";
import OrderTicket from "./OrderTicket";

const TIMEFRAMES = [
  { key: "minute", label: "분봉" }, { key: "day", label: "일봉" },
  { key: "week", label: "주봉" }, { key: "month", label: "월봉" },
];

function PaperDesk() {
  const { symbol, setSymbol } = useSymbol();
  const [data, setData] = useState(null);
  const [quote, setQuote] = useState(null);
  const [chart, setChart] = useState(null);
  const [timeframe, setTimeframe] = useState("day");
  const [input, setInput] = useState("");
  const [suggests, setSuggests] = useState([]);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [flash, setFlash] = useState(false);
  const pollRef = useRef(null);
  const prevPrice = useRef(null);

  /* ---- 계좌 ---- */
  const load = useCallback(async () => {
    try {
      const d = await api.get("/paper", { timeout: 45000 });
      setData(d);
      setError("");
      if (d.just_filled?.length) {
        setResult({
          ok: true,
          message: d.just_filled.map((f) =>
            `지정가 체결 — ${f.name} ${f.side === "buy" ? "매수" : "매도"} ` +
            `${fmtNum(f.quantity, 4)}주 @ ${fmtNum(f.price)}원`).join(" / "),
        });
      }
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  /* ---- 주문표 시세 폴링 (장중 3초 / 마감 20초) ---- */
  useEffect(() => {
    if (!symbol?.key) return;
    let alive = true;

    const tick = async () => {
      if (!alive) return;
      try {
        const q = await api.get(`/paper/quote/${encodeURIComponent(symbol.key)}`,
                                { timeout: 20000 });
        if (!alive) return;
        if (prevPrice.current != null && q.price !== prevPrice.current) {
          setFlash(true);
          setTimeout(() => setFlash(false), 500);
        }
        prevPrice.current = q.price;
        setQuote(q);
      } catch (err) {
        if (alive && err instanceof SymbolNotFound) setQuote(null);
      }
      if (alive) {
        const open = quote?.market_status?.is_open;
        pollRef.current = setTimeout(tick, open ? 3000 : 20000);
      }
    };
    tick();
    return () => { alive = false; clearTimeout(pollRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol?.key]);

  /* ---- 차트 ---- */
  useEffect(() => {
    if (!symbol?.key) return;
    let alive = true;
    setChart(null);
    api.get(`/chart/${encodeURIComponent(symbol.key)}?timeframe=${timeframe}&days=120`,
            { timeout: 45000 })
      .then((c) => { if (alive) setChart(c); })
      .catch(() => { if (alive) setChart({ empty: true, note: "차트를 불러오지 못했습니다." }); });
    return () => { alive = false; };
  }, [symbol?.key, timeframe]);

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

  const pick = (s) => {
    setSymbol({ key: s.key, name: s.name, market: s.market });
    setInput(""); setSuggests([]); setResult(null);
    prevPrice.current = null;
  };

  const search = async (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q) return;
    try {
      const r = await api.get(`/resolve/${encodeURIComponent(q)}`, { timeout: 20000 });
      pick(r.symbol || r);
    } catch (err) {
      setResult({ ok: false, message: err.message });
    }
  };

  /* ---- 주문 ---- */
  const placeOrder = async (ticket) => {
    setBusy(true); setResult(null);
    try {
      const r = await api.post("/paper/order",
        { ticker: symbol.key, ...ticket }, { timeout: 60000 });

      if (r.order_type === "limit" && r.status === "pending" && !r.filled_now) {
        setResult({
          ok: true,
          message: `지정가 ${ticket.side === "buy" ? "매수" : "매도"} 접수 — ` +
            `${fmtNum(r.quantity, 4)}주 @ ${fmtNum(r.limit_price)} · 미체결 대기` +
            (r.reserved_cash ? ` (예수금 ${fmtNum(r.reserved_cash)}원 구속)` : ""),
        });
      } else {
        const f = r.filled_now || r;
        setResult({
          ok: true,
          message: `${ticket.side === "buy" ? "매수" : "매도"} 체결 — ` +
            `${fmtNum(f.quantity, 4)}주 @ ${fmtNum(f.price)}원` +
            (r.fee != null ? ` (수수료 ${fmtNum(r.fee)}원${r.tax ? `, 거래세 ${fmtNum(r.tax)}원` : ""})` : "") +
            (r.realized_pnl != null
              ? ` · 실현손익 ${r.realized_pnl >= 0 ? "+" : ""}${fmtNum(r.realized_pnl)}원` : ""),
        });
      }
      await load();
      const q = await api.get(`/paper/quote/${encodeURIComponent(symbol.key)}`, { timeout: 20000 });
      setQuote(q);
    } catch (err) {
      setResult({ ok: false, message: err.message });
    } finally {
      setBusy(false);
    }
  };

  const cancelOrder = async (id) => {
    setBusy(true);
    try {
      await api.del(`/paper/orders/${id}`, { timeout: 30000 });
      setResult({ ok: true, message: `주문 #${id} 을 취소했습니다. 묶여 있던 예수금이 풀렸습니다.` });
      await load();
    } catch (err) {
      setResult({ ok: false, message: err.message });
    } finally {
      setBusy(false);
    }
  };

  const quickSell = async (ticker, name, market) => {
    setSymbol({ key: ticker, name, market });
    setBusy(true); setResult(null);
    try {
      // 수량을 보내지 않으면 서버가 '주문가능 전량'으로 처리합니다.
      // 화면에 그려둔 수량을 그대로 보내면, 그 사이 보유가 줄었을 때 반려됩니다.
      const r = await api.post("/paper/order",
        { ticker, side: "sell", order_type: "market" }, { timeout: 60000 });
      setResult({
        ok: true,
        message: `시장가 매도 체결 — ${fmtNum(r.quantity, 4)}주 @ ${fmtNum(r.price)}원` +
          (r.realized_pnl != null
            ? ` · 실현손익 ${r.realized_pnl >= 0 ? "+" : ""}${fmtNum(r.realized_pnl)}원` : ""),
      });
      await load();
    } catch (err) {
      setResult({ ok: false, message: err.message });
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    if (!confirm("모의투자 계좌를 초기화합니다.\n보유 종목·미체결 주문·거래 내역이 모두 삭제됩니다.")) return;
    setBusy(true);
    try {
      await api.post("/paper/reset", {}, { timeout: 30000 });
      setResult({ ok: true, message: "계좌를 초기화했습니다." });
      await load();
    } catch (err) {
      setResult({ ok: false, message: err.message });
    } finally {
      setBusy(false);
    }
  };

  if (error) return <section className="section"><div className="empty error-text">{error}</div></section>;
  if (!data) return <div className="loading">계좌 불러오는 중…</div>;

  const pf = data.portfolio || {};
  const isUS = quote?.symbol?.market === "US";
  const cur = (v, d = 0) => (isUS ? `$${fmtNum(v, 2)}` : `${fmtNum(v, d)}원`);

  return (
    <section className="section">
      <div className="section-head">
        <div>
          <div className="eyebrow">가상머니</div>
          <h2>모의투자</h2>
          <p>
            시장가·지정가 주문을 실제 증권 프로그램처럼 넣어봅니다. 수수료·거래세까지
            반영한 <strong>실제 손익</strong>을 보여줍니다.
            <strong> 실제 주문은 절대 나가지 않습니다.</strong>
          </p>
        </div>
        <button className="btn btn-ghost" onClick={reset} disabled={busy}>계좌 초기화</button>
      </div>

      {/* 계좌 요약 */}
      <div className="stat-grid">
        <div><div className="stat-label">총 평가액</div>
          <div className="stat-value">{fmtNum(pf.total_value)}원</div>
          <div className="stat-sub">초기 {fmtNum(pf.initial_cash)}원</div></div>
        <div><div className="stat-label">총 손익</div>
          <div className="stat-value" style={{ color: signColor(pf.total_pnl) }}>
            {pf.total_pnl >= 0 ? "+" : ""}{fmtNum(pf.total_pnl)}원</div>
          <div className="stat-sub" style={{ color: signColor(pf.total_pnl) }}>
            {pf.total_pnl_pct >= 0 ? "+" : ""}{(pf.total_pnl_pct ?? 0).toFixed(2)}%</div></div>
        <div><div className="stat-label">주문가능 현금</div>
          <div className="stat-value">{fmtNum(pf.available_cash)}원</div>
          <div className="stat-sub">
            {pf.reserved_cash > 0
              ? `미체결 구속 ${fmtNum(pf.reserved_cash)}원`
              : `예수금 ${fmtNum(pf.cash)}원`}</div></div>
        <div><div className="stat-label">평가손익 (미실현)</div>
          <div className="stat-value" style={{ color: signColor(pf.unrealized_pnl) }}>
            {(pf.unrealized_pnl ?? 0) >= 0 ? "+" : ""}{fmtNum(pf.unrealized_pnl)}원</div>
          <div className="stat-sub">보유 {pf.position_count ?? 0}종목</div></div>
        <div><div className="stat-label">실현손익</div>
          <div className="stat-value" style={{ color: signColor(pf.realized_pnl) }}>
            {(pf.realized_pnl ?? 0) >= 0 ? "+" : ""}{fmtNum(pf.realized_pnl)}원</div>
          <div className="stat-sub">매도 확정분</div></div>
      </div>

      {/* 종목 검색 */}
      <form onSubmit={search} className="search-form desk-search">
        <div className="search-wrap">
          <input className="input" value={input} onChange={(e) => setInput(e.target.value)}
                 placeholder="종목명 또는 코드 (예: 삼성전자, 005930, AAPL)" autoComplete="off" />
          {suggests.length > 0 && (
            <div className="suggest">
              {suggests.map((s) => (
                <button type="button" className="suggest-item" key={s.key}
                        onClick={() => pick(s)}>
                  <span>{s.name}</span>
                  <span className="suggest-meta">{s.key} · {s.market}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="btn btn-gold" type="submit">조회</button>
      </form>

      {/* 트레이딩 데스크 */}
      <div className="desk">
        <div className="desk-main">
          {quote ? (
            <>
              <div className="desk-quote">
                <div>
                  <div className="dq-name">
                    {quote.symbol.name}
                    <span className="market-badge">{quote.symbol.market_label}</span>
                  </div>
                  <div className="dq-code">{quote.symbol.key}</div>
                </div>
                <div className="dq-right">
                  <div className={`dq-price ${flash ? "flash" : ""}`}
                       style={{ color: signColor(quote.change_rate) }}>
                    {cur(quote.price)}
                  </div>
                  <div className="dq-change" style={{ color: signColor(quote.change_rate) }}>
                    {quote.change_rate >= 0 ? "▲" : "▼"} {Math.abs(quote.change_rate ?? 0).toFixed(2)}%
                    {quote.change_amount != null && ` (${cur(Math.abs(quote.change_amount))})`}
                  </div>
                  <div className="dq-meta">
                    {quote.market_status?.label}
                    {isUS && ` · 원화 ${fmtNum(quote.price_krw)}원`}
                  </div>
                </div>
              </div>

              <div className="tf-picker">
                {TIMEFRAMES.map((t) => (
                  <button key={t.key} className={timeframe === t.key ? "on" : ""}
                          onClick={() => setTimeframe(t.key)}>{t.label}</button>
                ))}
              </div>
              <div className="chart-box">
                {chart ? <CandleChart data={chart} />
                       : <div className="loading">차트 불러오는 중…</div>}
              </div>
            </>
          ) : (
            <div className="empty">위에서 종목을 검색해 주세요.</div>
          )}
        </div>

        <div className="desk-side">
          <OrderTicket quote={quote} busy={busy} onSubmit={placeOrder} />
        </div>
      </div>

      {result && (
        <div className="order-result" style={{ color: result.ok ? "var(--up)" : "var(--down)" }}>
          {result.message}
        </div>
      )}

      {/* 미체결 주문 */}
      <h3 className="sub-head">
        미체결 주문
        {data.orders?.length > 0 && <span className="count-chip">{data.orders.length}</span>}
      </h3>
      <div className="rows">
        <div className="row head order-row">
          <span>접수시각</span><span>구분</span><span>종목</span>
          <span className="ta-r">수량</span><span className="ta-r">지정가</span><span />
        </div>
        {data.orders?.length ? data.orders.map((o) => (
          <div className="row order-row" key={o.id}>
            <span className="m">{new Date(o.created_at).toLocaleString("ko-KR",
              { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}</span>
            <span><span className={`pill ${o.side === "buy" ? "ok" : "no"}`}>
              {o.side === "buy" ? "매수" : "매도"}</span></span>
            <div>
              <div className="row-name">{o.name || o.ticker}</div>
              <div className="row-sub">{o.ticker} · 지정가</div>
            </div>
            <span className="m ta-r">{fmtNum(o.quantity, 4)}</span>
            <span className="m ta-r">{fmtNum(o.limit_price, o.currency === "USD" ? 2 : 0)}</span>
            <span>
              <button className="btn-mini" disabled={busy}
                      onClick={() => cancelOrder(o.id)}>취소</button>
            </span>
          </div>
        )) : (
          <div className="row">
            <span style={{ color: "var(--muted)" }}>
              미체결 주문이 없습니다. 지정가로 주문하면 조건에 닿을 때까지 여기서 대기합니다.
            </span>
          </div>
        )}
      </div>
      {data.orders?.length > 0 && (
        <div className="order-hint">
          지정가는 <b>매수 = 현재가가 지정가 이하</b>, <b>매도 = 현재가가 지정가 이상</b>일 때 체결됩니다.
          이 화면을 열어두면 시세를 확인할 때마다 자동으로 체결 여부를 검사합니다.
        </div>
      )}

      {/* 보유 종목 */}
      <h3 className="sub-head">보유 종목</h3>
      <div className="rows">
        <div className="row head hold-row">
          <span>종목</span><span className="ta-r">수량</span><span className="ta-r">평균단가</span>
          <span className="ta-r">현재가</span><span className="ta-r">평가손익</span><span />
        </div>
        {pf.positions?.length ? pf.positions.map((p) => (
          <div className="row hold-row" key={p.ticker}>
            <div>
              <div className="row-name">{p.name || p.ticker}</div>
              <div className="row-sub">{p.ticker} · {p.market}</div>
            </div>
            <div className="m ta-r">
              {fmtNum(p.quantity, 4)}
              {p.reserved_quantity > 0 && (
                <div className="row-sub">{fmtNum(p.reserved_quantity, 4)}주 주문중</div>
              )}
            </div>
            <div className="m ta-r">{fmtNum(p.avg_price)}</div>
            <div className="m ta-r">{p.price_available ? fmtNum(p.current_price) : "—"}</div>
            <div className="m ta-r" style={{ color: signColor(p.pnl) }}>
              {p.pnl >= 0 ? "+" : ""}{fmtNum(p.pnl)}
              <div className="row-sub" style={{ color: signColor(p.pnl) }}>
                {p.pnl_pct >= 0 ? "+" : ""}{p.pnl_pct.toFixed(2)}%
              </div>
            </div>
            <div className="hold-actions">
              <button className="btn-mini" disabled={busy}
                      onClick={() => setSymbol({ key: p.ticker, name: p.name, market: p.market })}>
                주문
              </button>
              <button className="btn-mini danger" disabled={busy || p.available_quantity <= 0}
                      onClick={() => quickSell(p.ticker, p.name, p.market)}>
                전량매도
              </button>
            </div>
          </div>
        )) : (
          <div className="row">
            <span style={{ color: "var(--muted)" }}>보유 종목이 없습니다. 위에서 매수해 보세요.</span>
          </div>
        )}
      </div>

      {/* 거래 내역 */}
      <h3 className="sub-head">거래 내역</h3>
      <div className="rows">
        <div className="row head trade-row">
          <span>시각</span><span>구분</span><span>종목</span>
          <span>수량</span><span>단가</span><span>실현손익</span>
        </div>
        {data.trades?.length ? data.trades.map((t) => (
          <div className="row trade-row" key={t.id}>
            <span className="m">{new Date(t.traded_at).toLocaleString("ko-KR",
              { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}</span>
            <span><span className={`pill ${t.side === "buy" ? "ok" : "no"}`}>
              {t.side === "buy" ? "매수" : "매도"}</span></span>
            <div>
              <div className="row-name">{t.name || t.ticker}</div>
              {t.note && <div className="row-sub">{t.note}</div>}
            </div>
            <span className="m">{fmtNum(t.quantity, 4)}</span>
            <span className="m">{fmtNum(t.price)}</span>
            <span className="m" style={{ color: t.realized_pnl == null ? "var(--muted)" : signColor(t.realized_pnl) }}>
              {t.realized_pnl == null ? "—" : `${t.realized_pnl >= 0 ? "+" : ""}${fmtNum(t.realized_pnl)}`}
            </span>
          </div>
        )) : (
          <div className="row"><span style={{ color: "var(--muted)" }}>거래 내역이 없습니다.</span></div>
        )}
      </div>

      <div className="order-hint">
        수수료 국내 {data.fees?.kr_fee_pct}% · 매도 거래세 {data.fees?.kr_tax_pct}% · 미국 {data.fees?.us_fee_pct}%
        {data.fx && ` · 환율 ${fmtNum(data.fx.usd_krw)}원/USD (${data.fx.source})`}
        {" "}— 미국 종목도 원화로 환산해 거래합니다.
      </div>
    </section>
  );
}

export default function PaperPage() {
  const { user } = useAuth();
  return (
    <RequireAuth what="모의투자">
      <PaperDesk key={user?.id} />
    </RequireAuth>
  );
}
