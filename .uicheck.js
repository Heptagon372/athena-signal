
const AT = { esc: (x) => String(x == null ? '' : x), pct: (x) => (x==null?'-':Number(x).toFixed(2)+'%') };
function gatesBox(g) {
    if (!g) return "";
    if (!g.ok) return `<div class="note" style="margin-top:10px">${AT.esc(g.reason || "")}</div>`;
    const pass = String(g.verdict || "").startsWith("게이트 통과");
    const num = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
    const c = g.cpcv || {}, o = g.overfit || {}, l = g.lottery || {}, dsr = g.dsr || {};
    const rows = [
      ["CPCV 최악 경로 샤프", c.sharpe_worst == null ? "—"
        : `${num(c.sharpe_worst)} (양수 경로 ${c.share_positive == null ? "—"
          : (c.share_positive * 100).toFixed(0) + "%"})`],
      ["무작위 매매 대비", l.percentile == null ? "—"
        : `${num(l.percentile, 1)} 백분위`],
      ["요구 t통계", o.required_tstat == null ? "—"
        : `${num(o.t_stat)} / ${num(o.required_tstat)} 필요`],
      ["시행 수 (DSR 보정)", g.registry?.effective_n == null ? "—"
        : `${g.registry.effective_n}회${dsr.dsr == null ? "" : ` · DSR ${num(dsr.dsr)}`}`],
      ["데이터 길이", o.have_years == null ? "—"
        : `${num(o.have_years)}년 / 최소 ${num(o.min_backtest_years)}년`],
    ];
    return `<div class="warnbox" style="margin-top:10px;${pass
        ? "border-color:var(--olive);color:#b6c496" : "border-color:var(--up);color:#f0917f"}">
      <b>${pass ? "과최적화 게이트 통과" : "과최적화 게이트 경고"}</b>
      <div style="margin-top:4px">${AT.esc(g.verdict || "")}</div>
      <div class="note" style="margin-top:6px">
        ${rows.map(([k, v]) => `${AT.esc(k)} <b>${AT.esc(String(v))}</b>`).join(" · ")}
      </div>
    </div>`;
}
function rejectBox(rej) {
    if (!rej || !rej.total) return "";
    const rows = Object.entries(rej.by_reason || {})
      .map(([, v]) => `${AT.esc(v.label)} ${v.count}건`).join(" · ");
    return `<div class="warnbox" style="margin-bottom:10px;border-color:var(--up);color:#f0917f">
      <b>신호 ${rej.total}건이 체결되지 못했습니다.</b>
      아래 성과는 이 주문들을 뺀 나머지로 만든 것입니다.
      <div class="note" style="margin-top:4px">${rows}</div>
    </div>`;
}
function executionNote(ex) {
    if (!ex) return "";
    return `<div class="note" style="margin-top:8px">
      체결 모형 — ${AT.esc(ex.note || "")}
      갭으로 손절선을 관통하면 손절값이 아니라 시가에 체결하고, 구간 끝에 남은
      포지션도 정산해 매매 통계에 포함합니다. 슬리피지 ${ex.slippage_bps}bp.
    </div>`;
}
const gates = {"ok": true, "n_obs": 369, "registry": {"trial_id": "9211936e", "effective_n": 2}, "dsr": {"dsr": 0.41, "observed_sr": 2.99}, "cpcv": {"sharpe_median": 1.2, "sharpe_worst": -0.4, "share_positive": 0.53}, "lottery": {"percentile": 37.5, "passed": false}, "vol_audit": {"passed": true, "slope": -0.94}, "overfit": {"required_tstat": 2.61, "t_stat": 1.9, "passes_required_t": false, "min_backtest_years": 3.2, "have_years": 1.46}, "verdict": "\u26a0 \ubb34\uc791\uc704 \ub9e4\ub9e4 \ub300\ube44 37.5 \ubc31\ubd84\uc704"};
const rej = {"total":7,"by_reason":{"short_banned":{"count":5,"label":"SHORT-BAN"},"insufficient_cash":{"count":2,"label":"NO-CASH"}}};
const ex = {"slippage_bps":5.0,"note":"NOTE"};
const cases = [['gatesBox(full)',()=>gatesBox(gates)],['gatesBox(null)',()=>gatesBox(null)],
 ['gatesBox(not-ok)',()=>gatesBox({ok:false,reason:'x'})],['gatesBox(sparse)',()=>gatesBox({ok:true,verdict:'게이트 통과'})],
 ['gatesBox(empty-sub)',()=>gatesBox({ok:true,verdict:'x',cpcv:{},overfit:{},lottery:{},dsr:{}})],
 ['rejectBox(full)',()=>rejectBox(rej)],['rejectBox(empty)',()=>rejectBox({total:0,by_reason:{}})],
 ['rejectBox(null)',()=>rejectBox(null)],['executionNote',()=>executionNote(ex)],['executionNote(null)',()=>executionNote(null)]];
const lines = [];
for (const [n,f] of cases) { try { lines.push(`  OK   ${n} -> ${f().length} chars`); }
  catch(e){ lines.push(`  FAIL ${n}: ${e.message}`); } }
lines.push('--- rendered text ---');
lines.push(gatesBox(gates).replace(/<[^>]+>/g,' ').replace(/\s+/g,' ').trim());
require('fs').writeFileSync(process.argv[2], lines.join('
'), 'utf8');
