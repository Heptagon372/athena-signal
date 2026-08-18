"use client";

/**
 * 실시간 중간채점 패널
 * -------------------
 * 확정 채점은 만기(10분~1일)가 지나야 나옵니다. 그때까지 성적표가 텅 비어 있으면
 * "채점이 되고 있긴 한가?" 를 알 수 없습니다. 이 패널은 진행 중인 예측을 현재가와
 * 대조해 **지금 시점의 잠정 결과**를 보여줍니다.
 *
 * 잠정치임을 화면에서 분명히 합니다. 만기까지 가격이 뒤집히면 결과도 뒤집히고,
 * 이 수치는 적중률 통계에 들어가지 않습니다.
 */

import { useEffect, useRef, useState } from "react";
import { api, fmtNum, signColor, SymbolNotFound } from "../lib/api";
import { useSymbol } from "../providers";

function countdown(seconds) {
  if (seconds == null) return "-";
  if (seconds <= 0) return "채점 대기";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (d > 0) return `${d}일 ${h}시간`;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export default function LiveScoring({ onResolved }) {
  const { symbol } = useSymbol();
  const [live, setLive] = useState(null);
  const [error, setError] = useState("");
  const [, setTick] = useState(0);              // 1초마다 다시 그려 카운트다운을 흐르게 함
  const [flash, setFlash] = useState(false);
  const timerRef = useRef(null);
  const prevPrice = useRef(null);
  const prevTotal = useRef(null);
  // 폴링 클로저는 생성 시점의 live 를 붙잡고 있어 항상 null 입니다.
  // 장중 여부로 주기를 바꾸려면 최신값을 ref 로 따로 들고 있어야 합니다.
  const liveRef = useRef(null);
  liveRef.current = live;
  // 서버시각 - 브라우저시각. 응답을 받은 그 순간에만 갱신합니다.
  const offsetRef = useRef(0);

  /* 서버 폴링 — 장중 5초, 장 마감 30초 */
  useEffect(() => {
    if (!symbol?.key) return;
    let alive = true;

    const poll = async () => {
      if (!alive) return;
      try {
        const d = await api.get(
          `/scorecard/live?ticker=${encodeURIComponent(symbol.key)}`, { timeout: 20000 });
        if (!alive) return;
        if (d.server_time) {
          const parsed = Date.parse(d.server_time);
          if (!Number.isNaN(parsed)) offsetRef.current = parsed - Date.now();
        }
        if (prevPrice.current != null && d.price != null && d.price !== prevPrice.current) {
          setFlash(true);
          setTimeout(() => setFlash(false), 600);
        }
        prevPrice.current = d.price;
        // 진행 중 건수가 줄었다는 것은 그만큼 확정 채점으로 넘어갔다는 뜻입니다.
        if (prevTotal.current != null && d.items.length < prevTotal.current) onResolved?.();
        prevTotal.current = d.items.length;
        setLive(d); setError("");
      } catch (err) {
        if (alive) setError(err instanceof SymbolNotFound ? err.message : err.message);
      }
      if (alive) {
        const open = liveRef.current?.market_status?.is_open;
        timerRef.current = setTimeout(poll, open ? 5000 : 30000);
      }
    };

    poll();
    return () => { alive = false; clearTimeout(timerRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol?.key]);

  /* 남은 시간은 서버를 다시 부르지 않고 로컬에서 흐르게 합니다 */
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, []);

  if (error) {
    return (
      <aside className="live-panel">
        <div className="lv-head"><span className="eyebrow">실시간 채점</span></div>
        <div className="empty error-text">{error}</div>
      </aside>
    );
  }
  if (!live) {
    return (
      <aside className="live-panel">
        <div className="lv-head"><span className="eyebrow">실시간 채점</span></div>
        <div className="loading">불러오는 중…</div>
      </aside>
    );
  }

  const s = live.summary;
  const sym = live.symbol || symbol;
  const decided = s.leading + s.lagging;
  const barLead = decided ? (s.leading / decided) * 100 : 0;

  // 달러 종목을 원화처럼 정수로 반올림하면 $304.80 과 $304.62 가 똑같이 305 로 보여
  // "왜 틀렸다는 거지?" 가 됩니다. 통화에 맞춰 소수 자리를 정합니다.
  const digits = sym?.currency === "USD" ? 2 : 0;
  const px = (v) => fmtNum(v, digits);

  // 남은 시간은 target_at 에서 직접 계산합니다. 서버가 준 seconds_left 를 그대로
  // 쓰면 폴링 간격(최대 30초) 동안 숫자가 얼어붙습니다.
  //
  // 브라우저 시계가 서버와 어긋나 있을 수 있어 보정하는데, 그 보정값(offset)은
  // **응답을 받은 순간에 한 번** 재야 합니다. 매 렌더마다 Date.now() 로 다시 재면
  // 흐른 시간이 그대로 상쇄되어 카운트다운이 멈춰버립니다.
  const secondsLeft = (targetAt) => {
    if (!targetAt) return null;
    const t = Date.parse(targetAt);
    if (Number.isNaN(t)) return null;
    const serverNow = Date.now() + offsetRef.current;
    return Math.max(Math.round((t - serverNow) / 1000), 0);
  };

  return (
    <aside className="live-panel">
      <div className="lv-head">
        <span className="eyebrow">실시간 채점</span>
        <span className={`lv-dot ${live.market_status?.is_open ? "on" : ""}`} />
        <span className="lv-market">{live.market_status?.label || "-"}</span>
      </div>

      {/* 종목 — 지금 보고 있는 주식 */}
      <div className="lv-symbol">
        <div className="lv-name">
          {sym?.name || sym?.key}
          {sym?.market_label && <span className="market-badge">{sym.market_label}</span>}
        </div>
        <div className="lv-code">{sym?.key}</div>
        <div className={`lv-price ${flash ? "flash" : ""}`}>
          {/* 달러는 앞에, 원은 뒤에 붙습니다 ("304.52$" 가 되지 않도록) */}
          {digits === 2 && <span className="lv-cur pre">$</span>}
          {live.price != null ? px(live.price) : "-"}
          {digits !== 2 && <span className="lv-cur">원</span>}
        </div>
        {live.change_rate != null && (
          <div className="lv-change" style={{ color: signColor(live.change_rate) }}>
            {live.change_rate >= 0 ? "▲" : "▼"} {Math.abs(live.change_rate).toFixed(2)}%
          </div>
        )}
      </div>

      {/* 잠정 집계 */}
      <div className="lv-summary">
        <div><b style={{ color: "var(--up)" }}>{s.leading}</b><span>맞는 중</span></div>
        <div><b style={{ color: "var(--down)" }}>{s.lagging}</b><span>틀리는 중</span></div>
        <div><b style={{ color: "var(--muted)" }}>{s.undecided}</b><span>보합</span></div>
      </div>

      {decided > 0 && (
        <>
          <div className="lv-bar"><div className="lv-bar-fill" style={{ width: `${barLead}%` }} /></div>
          <div className="lv-bar-label">
            잠정 적중률 <b style={{ color: s.provisional_accuracy >= 50 ? "var(--up)" : "var(--down)" }}>
              {s.provisional_accuracy}%
            </b>
            <span>   판정 가능 {decided}건</span>
          </div>
        </>
      )}

      <div className="lv-caveat">
        {decided === 0 && s.undecided > 0 && !live.market_status?.is_open ? (
          <>
            장이 닫혀 있어 가격이 <b>{px(live.price)}</b> 에 멈춰 있습니다.
            기준가와 같으면 방향을 정할 수 없어 전부 보합입니다.
            다음 장이 열리면 하나씩 갈립니다.
          </>
        ) : (
          <>
            만기 전 잠정치입니다. 가격이 뒤집히면 결과도 뒤집히며,
            위 성적표의 적중률에는 <b>들어가지 않습니다</b>.
          </>
        )}
      </div>

      {/* 진행 중 예측 */}
      <div className="lv-list">
        {live.items.length === 0 && (
          <div className="empty">
            진행 중인 예측이 없습니다. 분석 페이지에서 이 종목을 조회하면
            6개 시평선의 예측이 새로 기록됩니다.
          </div>
        )}
        {live.items.map((it) => {
          const left = secondsLeft(it.target_at) ?? it.seconds_left;
          const state = it.hitting === true ? "ok" : it.hitting === false ? "no" : "flat";
          return (
            <div className={`lv-item ${state}`} key={it.id}>
              <div className="lv-item-top">
                <span className="lv-hz">{it.label}</span>
                <span className="lv-dir" style={{ color: it.direction === "up" ? "var(--up)" : "var(--down)" }}>
                  {it.direction === "up" ? "▲상승" : "▼하락"} {it.probability}%
                </span>
                <span className={`pill ${state}`}>
                  {it.hitting === true ? "맞는 중" : it.hitting === false ? "틀리는 중" : "보합"}
                </span>
              </div>
              <div className="lv-item-bot">
                <span className="lv-move">
                  {px(it.base_price)} → {px(it.current_price)}
                  {it.change_pct != null && (
                    <b style={{ color: signColor(it.change_pct) }}>
                      {" "}{it.change_pct >= 0 ? "+" : ""}{it.change_pct}%
                    </b>
                  )}
                </span>
                <span className="lv-left">남은 {countdown(left)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}
