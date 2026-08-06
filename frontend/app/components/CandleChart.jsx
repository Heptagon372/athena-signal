"use client";

/**
 * 캔들차트 — 외부 라이브러리 없이 인라인 SVG로 직접 그립니다.
 *
 * 가격 패널(캔들 + 이동평균 + 볼린저) 아래에 보조지표 패널을 쌓습니다.
 * 어떤 지표를 그릴지는 우측 상단 [지표] 버튼에서 켜고 끕니다.
 * 선택은 localStorage 에 남아 다음 방문에도 유지됩니다.
 *
 * 패널 높이는 켜진 지표 수에 따라 계산되므로, 지표를 끄면 차트가
 * 그만큼 짧아집니다(빈 칸이 남지 않습니다).
 *
 * 색은 CSS 변수로 지정해 테마 전환 시 자동으로 따라갑니다.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { fmtNum } from "../lib/api";

const W = 1100, PADL = 60, PADR = 16;
const PRICE = { top: 12, h: 286 };
const GAP = 30;                     // 패널 사이 여백 (라벨 자리)
const PLOT = W - PADL - PADR;
const STORE_KEY = "athena_chart_indicators";

/** 가격 패널 위에 겹쳐 그리는 것들 */
const OVERLAYS = [
  { key: "ma5", label: "5선", color: "#e6b422" },
  { key: "ma20", label: "20선", color: "#7a9cc6" },
  { key: "ma60", label: "60선", color: "var(--muted)" },
  { key: "bb", label: "볼린저(20,2σ)", color: "var(--olive)" },
];

/**
 * 아래에 따로 쌓는 보조지표 패널
 *   kind  bars(막대) | range(0~100 고정) | osc(0 기준 대칭) | line(자동 범위)
 *   band  음영으로 표시할 구간 [lo, hi]
 */
const PANELS = [
  { key: "volume", label: "거래량", h: 58, kind: "bars" },
  { key: "rsi", label: "RSI(14)", h: 60, kind: "range", band: [30, 70],
    series: [["rsi", "var(--gold)"]] },
  { key: "macd", label: "MACD(12,26,9)", h: 64, kind: "osc", hist: "macd_hist",
    series: [["macd", "var(--gold)"], ["macd_signal", "#7a9cc6"]] },
  { key: "cci", label: "CCI(20)", h: 60, kind: "osc", band: [-100, 100],
    series: [["cci", "var(--gold)"]] },
  { key: "stoch", label: "스토캐스틱(14,3,3)", h: 60, kind: "range", band: [20, 80],
    series: [["stoch_k", "var(--gold)"], ["stoch_d", "#7a9cc6"]] },
  { key: "mfi", label: "MFI(14)", h: 60, kind: "range", band: [20, 80],
    series: [["mfi", "var(--olive)"]] },
  { key: "atr", label: "ATR(14)", h: 52, kind: "line",
    series: [["atr", "var(--olive)"]] },
  { key: "obv", label: "OBV", h: 52, kind: "line",
    series: [["obv", "#7a9cc6"]] },
];

// 처음 열었을 때의 기본 구성 (기존 화면과 동일)
const DEFAULTS = {
  ma5: true, ma20: true, ma60: true, bb: true,
  volume: true, rsi: true, macd: true,
  cci: false, stoch: false, mfi: false, atr: false, obv: false,
};

function loadSettings() {
  if (typeof window === "undefined") return DEFAULTS;
  try {
    const saved = JSON.parse(window.localStorage.getItem(STORE_KEY) || "{}");
    return { ...DEFAULTS, ...saved };
  } catch {
    return DEFAULTS;
  }
}

function niceTicks(lo, hi, count) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
  return out;
}

const nn = (a) => (a || []).filter((v) => v !== null && v !== undefined);

/** 축약 표기 (OBV 처럼 자릿수가 큰 값용) */
function shortNum(v) {
  const abs = Math.abs(v);
  if (abs >= 1e12) return (v / 1e12).toFixed(1) + "조";
  if (abs >= 1e8) return (v / 1e8).toFixed(1) + "억";
  if (abs >= 1e4) return (v / 1e4).toFixed(1) + "만";
  if (abs >= 100) return v.toFixed(0);
  return v.toFixed(2);
}

export default function CandleChart({ data }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);
  const [settings, setSettings] = useState(DEFAULTS);
  const [open, setOpen] = useState(false);

  // localStorage 는 서버 렌더링 때 없으므로 마운트 후에 읽습니다
  useEffect(() => { setSettings(loadSettings()); }, []);

  const toggle = (key) => {
    setSettings((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      try { window.localStorage.setItem(STORE_KEY, JSON.stringify(next)); } catch {}
      return next;
    });
  };

  const reset = () => {
    setSettings(DEFAULTS);
    try { window.localStorage.removeItem(STORE_KEY); } catch {}
  };

  // 켜져 있고, 실제로 값이 들어온 패널만 그립니다.
  // (예전 서버가 안 내려주는 지표를 켜면 빈 칸만 생기므로)
  const panels = useMemo(() => {
    if (!data) return [];
    let top = PRICE.top + PRICE.h + GAP;
    const out = [];
    for (const panel of PANELS) {
      if (!settings[panel.key]) continue;
      const keys = panel.kind === "bars" ? ["volume"] : panel.series.map((s) => s[0]);
      if (!keys.some((k) => nn(data[k]).length)) continue;
      out.push({ ...panel, top });
      top += panel.h + GAP;
    }
    return out;
  }, [data, settings]);

  const height = (panels.length ? panels[panels.length - 1].top + panels[panels.length - 1].h
                                : PRICE.top + PRICE.h) + 26;

  const geom = useMemo(() => {
    if (!data || data.empty || !data.dates?.length) return null;
    const n = data.dates.length;
    const step = PLOT / n;
    const cw = Math.max(1.2, Math.min(step * 0.68, 13));
    const x = (i) => PADL + (i + 0.5) * step;

    // 볼린저를 끄면 밴드가 가격 범위를 넓히지 않도록 제외합니다
    const extra = settings.bb ? [...nn(data.bb_lower)] : [];
    const extraHi = settings.bb ? [...nn(data.bb_upper)] : [];
    const lo = Math.min(...nn(data.low), ...extra);
    const hi = Math.max(...nn(data.high), ...extraHi);
    const pad = (hi - lo) * 0.06 || 1;
    const pLo = lo - pad, pHi = hi + pad;
    const py = (v) => PRICE.top + PRICE.h - ((v - pLo) / (pHi - pLo)) * PRICE.h;

    const line = (arr, yf) => {
      let d = "", pen = false;
      for (let i = 0; i < (arr || []).length; i++) {
        const v = arr[i];
        if (v === null || v === undefined) { pen = false; continue; }
        d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${yf(v).toFixed(1)} `;
        pen = true;
      }
      return d;
    };

    // 패널마다 y 변환 + 눈금을 만들어 둡니다
    const scales = {};
    for (const panel of panels) {
      if (panel.kind === "bars") {
        const max = Math.max(...nn(data.volume), 1);
        scales[panel.key] = {
          y: (v) => panel.top + panel.h - (v / max) * panel.h,
          ticks: [],
        };
      } else if (panel.kind === "range") {
        const y = (v) => panel.top + panel.h - (v / 100) * panel.h;
        scales[panel.key] = { y, ticks: panel.band };
      } else if (panel.kind === "osc") {
        const values = [
          ...panel.series.flatMap((s) => nn(data[s[0]])),
          ...(panel.hist ? nn(data[panel.hist]) : []),
        ].map(Math.abs);
        const max = Math.max(...values, panel.band ? Math.abs(panel.band[1]) : 0, 1);
        const y = (v) => panel.top + panel.h / 2 - (v / max) * (panel.h / 2 - 3);
        scales[panel.key] = { y, ticks: panel.band || [] };
      } else {
        const values = panel.series.flatMap((s) => nn(data[s[0]]));
        const min = Math.min(...values), max = Math.max(...values);
        const span = max - min || 1;
        const y = (v) => panel.top + panel.h - ((v - min) / span) * panel.h;
        scales[panel.key] = { y, ticks: [min, max], format: shortNum };
      }
    }

    return { n, step, cw, x, py, line, pLo, pHi, scales };
  }, [data, panels, settings.bb]);

  if (!data || data.empty || !data.dates?.length) {
    return <div className="chart-box"><div className="loading">
      {data?.note || "차트를 그릴 데이터가 없습니다."}</div></div>;
  }

  const { n, step, cw, x, py, line, pLo, pHi, scales } = geom;

  const onMove = (e) => {
    const svg = svgRef.current;
    if (!svg) return;
    const box = svg.getBoundingClientRect();
    const scale = W / box.width;
    const sx = (e.clientX - box.left) * scale;
    const i = Math.max(0, Math.min(n - 1, Math.floor((sx - PADL) / step)));
    if (data.close[i] === null) return;
    setHover({ i, left: x(i) / scale, top: py(data.close[i]) / scale });
  };

  // 볼린저 음영
  let bbPath = "";
  if (settings.bb) {
    const fwd = [], bwd = [];
    for (let i = 0; i < n; i++) {
      if (data.bb_upper[i] == null || data.bb_lower[i] == null) continue;
      fwd.push(`${x(i).toFixed(1)} ${py(data.bb_upper[i]).toFixed(1)}`);
      bwd.push(`${x(i).toFixed(1)} ${py(data.bb_lower[i]).toFixed(1)}`);
    }
    if (fwd.length) {
      bbPath = `M${fwd[0]} ${fwd.slice(1).map((p) => `L${p}`).join(" ")} ` +
               `${bwd.reverse().map((p) => `L${p}`).join(" ")} Z`;
    }
  }

  const tickEvery = Math.max(1, Math.round(n / 7));
  const h = hover ? hover.i : null;
  const activeCount = [...OVERLAYS, ...PANELS].filter((o) => settings[o.key]).length;

  // 툴팁에 띄울 보조지표 값 (켜둔 것만)
  const tipRows = hover ? panels.flatMap((panel) =>
    (panel.kind === "bars" ? [] : panel.series)
      .filter(([key]) => data[key]?.[hover.i] != null)
      .map(([key]) => [key.toUpperCase().replace("_", " "),
                       key === "obv" ? shortNum(data[key][hover.i])
                                     : data[key][hover.i].toFixed(2)])
  ) : [];

  return (
    <div className="chart-box">
      <div className="chart-legend">
        <span><i className="sw-box" style={{ background: "var(--up)" }} />양봉</span>
        <span><i className="sw-box" style={{ background: "var(--down)" }} />음봉</span>
        {OVERLAYS.filter((o) => settings[o.key]).map((o) => (
          <span key={o.key}><i style={{ background: o.color }} />{o.label}</span>
        ))}
        <span style={{ color: "var(--muted)" }}>
          · {data.timeframe_label || ""} {n}봉 ({data.dates[0]} ~ {data.dates[n - 1]})
        </span>

        <div className="ind-picker">
          <button className={open ? "active" : ""} onClick={() => setOpen((v) => !v)}>
            지표 {activeCount}
          </button>
          {open && (
            <>
              <div className="ind-scrim" onClick={() => setOpen(false)} />
              <div className="ind-menu">
                <div className="ind-menu-head">
                  <span>가격 위에 겹치기</span>
                  <button onClick={reset}>기본값</button>
                </div>
                {OVERLAYS.map((o) => (
                  <label key={o.key}>
                    <input type="checkbox" checked={!!settings[o.key]}
                           onChange={() => toggle(o.key)} />
                    <i style={{ background: o.color }} />{o.label}
                  </label>
                ))}
                <div className="ind-menu-head"><span>보조지표 패널</span></div>
                {PANELS.map((p) => {
                  const keys = p.kind === "bars" ? ["volume"] : p.series.map((s) => s[0]);
                  const ready = keys.some((k) => nn(data[k]).length);
                  return (
                    <label key={p.key} className={ready ? "" : "off"}
                           title={ready ? "" : "이 주기에서는 계산할 데이터가 부족합니다"}>
                      <input type="checkbox" checked={!!settings[p.key]} disabled={!ready}
                             onChange={() => toggle(p.key)} />
                      {p.label}
                    </label>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>

      <svg ref={svgRef} viewBox={`0 0 ${W} ${height}`} onMouseMove={onMove}
           onMouseLeave={() => setHover(null)} style={{ cursor: "crosshair" }}>
        {niceTicks(pLo, pHi, 5).map((v) => (
          <g key={v}>
            <line x1={PADL} y1={py(v)} x2={W - PADR} y2={py(v)}
                  stroke="var(--hairline)" strokeWidth="0.6" opacity="0.55" />
            <text x={PADL - 7} y={py(v) + 3.5} textAnchor="end" fill="var(--muted)"
                  fontFamily="monospace" fontSize="10">{fmtNum(v)}</text>
          </g>
        ))}

        {bbPath && <>
          <path d={bbPath} fill="var(--olive)" opacity="0.10" />
          <path d={line(data.bb_upper, py)} fill="none" stroke="var(--olive)"
                strokeWidth="0.8" opacity="0.55" strokeDasharray="4 3" />
          <path d={line(data.bb_lower, py)} fill="none" stroke="var(--olive)"
                strokeWidth="0.8" opacity="0.55" strokeDasharray="4 3" />
        </>}

        {data.close.map((cl, i) => {
          const o = data.open[i], hi2 = data.high[i], lo2 = data.low[i];
          if ([o, hi2, lo2, cl].some((v) => v == null)) return null;
          const rise = cl >= o;
          const col = rise ? "var(--up)" : "var(--down)";
          const yt = py(Math.max(o, cl)), yb = py(Math.min(o, cl));
          return (
            <g key={i}>
              <line x1={x(i)} y1={py(hi2)} x2={x(i)} y2={py(lo2)} stroke={col} strokeWidth="1" />
              <rect x={x(i) - cw / 2} y={yt} width={cw} height={Math.max(1, yb - yt)}
                    fill={rise ? col : "none"} stroke={col} strokeWidth="1" />
            </g>
          );
        })}

        {OVERLAYS.filter((o) => o.key !== "bb" && settings[o.key]).map((o) => (
          <path key={o.key} d={line(data[o.key], py)} fill="none"
                stroke={o.color} strokeWidth="1.3" />
        ))}

        {panels.map((panel) => {
          const { y, ticks, format } = scales[panel.key];
          return (
            <g key={panel.key}>
              <text x={PADL} y={panel.top - 6} fill="var(--muted)"
                    fontFamily="monospace" fontSize="9.5">{panel.label}</text>

              {/* 과열/침체 구간 음영 */}
              {panel.band && (
                <rect x={PADL} y={Math.min(y(panel.band[1]), y(panel.band[0]))}
                      width={PLOT} height={Math.abs(y(panel.band[0]) - y(panel.band[1]))}
                      fill="var(--gold)" opacity="0.05" />
              )}

              {(ticks || []).map((v, ti) => (
                <g key={ti}>
                  <line x1={PADL} y1={y(v)} x2={W - PADR} y2={y(v)}
                        stroke="var(--hairline)" strokeWidth="0.7" strokeDasharray="3 3" />
                  <text x={PADL - 7} y={y(v) + 3.5} textAnchor="end" fill="var(--muted)"
                        fontFamily="monospace" fontSize="9.5">
                    {format ? format(v) : v}
                  </text>
                </g>
              ))}

              {panel.kind === "osc" && (
                <line x1={PADL} y1={y(0)} x2={W - PADR} y2={y(0)}
                      stroke="var(--hairline)" strokeWidth="0.7" />
              )}

              {/* 거래량 막대 */}
              {panel.kind === "bars" && data.volume.map((v, i) => {
                if (v == null) return null;
                const rise = data.close[i] >= (data.open[i] ?? data.close[i]);
                return <rect key={i} x={x(i) - cw / 2} y={y(v)} width={cw}
                             height={Math.max(0.5, panel.top + panel.h - y(v))}
                             fill={rise ? "var(--up)" : "var(--down)"} opacity="0.5" />;
              })}

              {/* MACD 히스토그램 */}
              {panel.hist && (data[panel.hist] || []).map((v, i) => {
                if (v == null) return null;
                const y0 = y(0), y1 = y(v);
                return <rect key={i} x={x(i) - cw / 2} y={Math.min(y0, y1)} width={cw}
                             height={Math.max(0.6, Math.abs(y1 - y0))}
                             fill={v >= 0 ? "var(--up)" : "var(--down)"} opacity="0.55" />;
              })}

              {(panel.series || []).map(([key, color]) => (
                <path key={key} d={line(data[key], y)} fill="none"
                      stroke={color} strokeWidth="1.3" />
              ))}

              {panel.kind === "bars" && (
                <line x1={PADL} y1={panel.top + panel.h} x2={W - PADR} y2={panel.top + panel.h}
                      stroke="var(--hairline)" strokeWidth="0.7" />
              )}
            </g>
          );
        })}

        {data.dates.map((d, i) => i % tickEvery === 0 && (
          <text key={i} x={x(i)} y={height - 8} textAnchor="middle" fill="var(--muted)"
                fontFamily="monospace" fontSize="9.5">{d}</text>
        ))}

        {h !== null && (
          <g>
            <line x1={x(h)} y1={PRICE.top} x2={x(h)} y2={height - 26}
                  stroke="var(--gold)" strokeWidth="0.8" strokeDasharray="3 3" opacity="0.8" />
            <circle cx={x(h)} cy={py(data.close[h])} r="3" fill="var(--gold)" />
          </g>
        )}
      </svg>

      {hover && (
        <div className="chart-tip" style={{ left: hover.left + 14, top: Math.max(8, hover.top - 30) }}>
          <div className="tip-date">{data.dates[hover.i]}</div>
          {[["시가", data.open[hover.i]], ["고가", data.high[hover.i]],
            ["저가", data.low[hover.i]], ["종가", data.close[hover.i]],
            ["거래량", data.volume[hover.i]]].map(([k, v]) => (
            <div className="tip-row" key={k}><span>{k}</span><span>{fmtNum(v)}</span></div>
          ))}
          {tipRows.map(([k, v]) => (
            <div className="tip-row" key={k}><span>{k}</span><span>{v}</span></div>
          ))}
        </div>
      )}
    </div>
  );
}
