"use client";

/** 성적표 페이지 (/score) — AI가 지금까지 얼마나 맞췄는지 */

import { useCallback, useEffect, useState } from "react";
import LiveScoring from "../components/LiveScoring";
import RequireAuth from "../components/RequireAuth";
import { api, fmtNum, HORIZON_LABEL, signColor } from "../lib/api";
import { useAuth } from "../providers";

function Scorecard() {
  const [sc, setSc] = useState(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setSc(await api.get("/scorecard", { timeout: 30000 }));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const resolveNow = async () => {
    setBusy(true); setNote("채점 중…");
    try {
      const r = await api.post("/scorecard/resolve", {}, { timeout: 180000 });
      setNote(`방금 ${r.scored}건 채점했습니다.`);
      await load();
    } catch (err) {
      setNote(`채점 실패: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  if (error) return <section className="section"><div className="empty error-text">{error}</div></section>;
  if (!sc) return <div className="loading">성적표 불러오는 중…</div>;

  const o = sc.overall || {};
  const wrong = (o.total || 0) - (o.correct || 0);
  const acc = o.accuracy;

  return (
    <div className="score-layout">
    <section className="section">
      <div className="section-head">
        <div>
          <div className="eyebrow">자가검증</div>
          <h2>AI 성적표</h2>
          <p>
            만기가 지난 예측을 실제 가격과 대조해 채점한 기록입니다.
            서버가 10분마다 자동으로 채점합니다.
            {sc.scoped_to_user && " 내 계정이 만든 예측만 집계합니다."}
          </p>
        </div>
        <button className="btn btn-gold" onClick={resolveNow} disabled={busy}>
          {busy ? "채점 중…" : "지금 채점"}
        </button>
      </div>

      <div className="stat-grid">
        <div><div className="stat-label">채점 완료</div>
          <div className="stat-value big">{o.total ?? 0}</div><div className="stat-sub">건</div></div>
        <div><div className="stat-label">맞춘 예측</div>
          <div className="stat-value big" style={{ color: "var(--up)" }}>{o.correct ?? 0}</div>
          <div className="stat-sub">건</div></div>
        <div><div className="stat-label">틀린 예측</div>
          <div className="stat-value big" style={{ color: "var(--down)" }}>{wrong}</div>
          <div className="stat-sub">건</div></div>
        <div><div className="stat-label">적중률</div>
          <div className="stat-value big"
               style={{ color: acc == null ? "var(--muted)" : acc >= 50 ? "var(--up)" : "var(--down)" }}>
            {acc ?? "—"}{acc != null ? "%" : ""}
          </div>
          <div className="stat-sub">
            {acc == null ? "표본 없음" : acc >= 50 ? "동전던지기보다 나음" : "동전던지기보다 못함"}
          </div></div>
        <div><div className="stat-label">채점 대기</div>
          <div className="stat-value big" style={{ color: "var(--muted)" }}>{sc.pending ?? 0}</div>
          <div className="stat-sub">만기 전이거나 시세 대기</div></div>
      </div>

      <div className="resolve-note">
        {note && <div style={{ marginBottom: 6 }}>{note}</div>}
        {sc.last_auto_resolve
          ? `마지막 자동 채점 ${new Date(sc.last_auto_resolve.checked_at).toLocaleString("ko-KR")} · ${sc.last_auto_resolve.scored}건 처리`
          : "서버가 10분마다 만기가 지난 예측을 자동 채점합니다."}
        {sc.voided > 0 && ` · 무효 ${sc.voided}건(만기 구간에 거래 없음)`}
      </div>

      <h3 className="sub-head">시평선별 적중률</h3>
      <div className="hz-strip">
        {sc.by_horizon?.length ? sc.by_horizon.map((h) => (
          <div className="hz-cell" key={h.horizon}>
            <div className="hz-label">{h.label}{h.legacy ? " (구)" : ""}</div>
            <div className="hz-prob" style={{ color: (h.accuracy ?? 0) >= 50 ? "var(--up)" : "var(--down)" }}>
              {h.accuracy ?? "—"}%
            </div>
            <div className="hz-meta">{h.correct}/{h.total}건</div>
          </div>
        )) : <div className="empty">아직 채점된 예측이 없습니다.</div>}
      </div>

      <h3 className="sub-head">종목별 적중률</h3>
      <div className="hz-strip">
        {sc.by_ticker?.length ? sc.by_ticker.map((t) => (
          <div className="hz-cell" key={t.ticker}>
            <div className="hz-label">{t.name}</div>
            <div className="hz-prob" style={{ color: (t.accuracy ?? 0) >= 50 ? "var(--up)" : "var(--down)" }}>
              {t.accuracy ?? "—"}%
            </div>
            <div className="hz-meta">{t.correct}/{t.total}건</div>
          </div>
        )) : <div className="empty">아직 채점된 종목이 없습니다.</div>}
      </div>

      <h3 className="sub-head">채점 상세 — 예측이 실제로 어떻게 됐는지</h3>
      <div className="rows">
        <div className="row head scored-row">
          <span>종목</span><span>시평선</span><span>예측 → 실제</span>
          <span>가격 변화</span><span>결과</span>
        </div>
        {sc.recent?.length ? sc.recent.map((r) => {
          const dirTxt = (d) => (d === "up" ? "▲상승" : "▼하락");
          const dirCol = (d) => (d === "up" ? "var(--up)" : "var(--down)");
          return (
            <div className="row scored-row" key={r.id}>
              <div>
                <div className="row-name">{r.name || r.ticker}</div>
                <div className="row-sub">{r.ticker}</div>
              </div>
              <div className="m">{HORIZON_LABEL[r.horizon_label] || r.horizon_label}</div>
              <div className="m">
                <span style={{ color: dirCol(r.direction) }}>{dirTxt(r.direction)} {r.probability}%</span>
                <span style={{ color: "var(--muted)" }}> → </span>
                <span style={{ color: dirCol(r.actual_result) }}>{dirTxt(r.actual_result)}</span>
              </div>
              <div className="m">
                {r.base_price != null ? `${fmtNum(r.base_price)} → ${fmtNum(r.resolved_price)}` : "—"}
                {r.change_pct != null && (
                  <div className="row-sub" style={{ color: signColor(r.change_pct) }}>
                    {r.change_pct >= 0 ? "+" : ""}{r.change_pct}%
                  </div>
                )}
              </div>
              <div><span className={`pill ${r.correct ? "ok" : "no"}`}>{r.correct ? "적중" : "오답"}</span></div>
            </div>
          );
        }) : (
          <div className="row">
            <span style={{ color: "var(--muted)" }}>
              채점된 예측이 아직 없습니다. 분석 페이지에서 종목을 조회하면 예측이 기록되고,
              만기(빠르면 10분)가 지나면 자동으로 채점됩니다.
            </span>
          </div>
        )}
      </div>
    </section>

    <LiveScoring onResolved={load} />
    </div>
  );
}

export default function ScorePage() {
  const { user } = useAuth();
  return (
    <RequireAuth what="성적표">
      <Scorecard key={user?.id} />
    </RequireAuth>
  );
}
