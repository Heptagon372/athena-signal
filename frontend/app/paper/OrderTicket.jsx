"use client";

/**
 * 주문표 (Order Ticket)
 * --------------------
 * 증권사 HTS 의 주문 창에 해당합니다.
 *
 *   매수 / 매도 · 시장가 / 지정가 · 수량 · 지정가
 *
 * 값을 입력하는 즉시 **총 주문금액을 실시간으로 계산**해 보여줍니다. 수수료와
 * 거래세까지 더한 실제 출금액이라, 주문을 넣기 전에 얼마가 나가는지 알 수 있습니다.
 *
 * 지정가는 호가 단위에 맞춰 정렬합니다. 삼성전자에 239,537원 같은 주문은
 * 실제 시장에 존재할 수 없기 때문입니다.
 */

import { useEffect, useMemo, useState } from "react";
import { fmtNum } from "../lib/api";

const FEE_KR = 0.00015, TAX_KR = 0.0018, FEE_US = 0.0025;

/** 국내 호가 단위 (2023-01 개편) — 백엔드 paper.KR_TICK_TABLE 과 같은 표 */
function tickSize(price, market) {
  if (market === "US") return 0.01;
  if (price < 2000) return 1;
  if (price < 5000) return 5;
  if (price < 20000) return 10;
  if (price < 50000) return 50;
  if (price < 200000) return 100;
  if (price < 500000) return 500;
  return 1000;
}

function snap(price, market) {
  if (!price || price <= 0) return price;
  const unit = tickSize(price, market);
  const v = Math.round(price / unit) * unit;
  return market === "US" ? Math.round(v * 100) / 100 : Math.round(v);
}

export default function OrderTicket({ quote, busy, onSubmit }) {
  const [side, setSide] = useState("buy");
  const [orderType, setOrderType] = useState("market");
  const [qty, setQty] = useState("");
  const [limitPrice, setLimitPrice] = useState("");
  const [touchedLimit, setTouchedLimit] = useState(false);

  const sym = quote?.symbol;
  const market = sym?.market || "KOSPI";
  const isUS = market === "US";
  const acc = quote?.account || {};
  const rate = quote?.fx?.usd_krw || 1;

  // 종목이 바뀌면 지정가를 현재가로 다시 채웁니다 (직접 고친 값은 건드리지 않음)
  useEffect(() => {
    setTouchedLimit(false);
    setQty("");
  }, [sym?.key]);

  useEffect(() => {
    if (!touchedLimit && quote?.price) setLimitPrice(String(snap(quote.price, market)));
  }, [quote?.price, market, touchedLimit]);

  const unit = tickSize(quote?.price || 0, market);

  /* ---- 실시간 금액 계산 ---- */
  const calc = useMemo(() => {
    const n = Number(qty) || 0;
    // 시장가는 현재가, 지정가는 입력한 가격으로 계산합니다.
    const native = orderType === "limit" ? Number(limitPrice) || 0 : quote?.price || 0;
    const krw = isUS ? native * rate : native;
    const gross = krw * n;
    const feeRate = isUS ? FEE_US : FEE_KR;
    const fee = Math.round(gross * feeRate * 100) / 100;
    const tax = side === "sell" && !isUS ? Math.round(gross * TAX_KR * 100) / 100 : 0;
    return {
      n, native, krw, gross, fee, tax,
      total: side === "buy" ? gross + fee : gross - fee - tax,
    };
  }, [qty, limitPrice, orderType, quote?.price, side, isUS, rate]);

  const maxQty = side === "buy"
    ? (orderType === "limit" && Number(limitPrice) > 0
        // 지정가 매수는 현재가가 아니라 **지정가**로 최대 수량을 계산해야 맞습니다.
        ? Math.floor((acc.available_cash || 0)
            / ((isUS ? Number(limitPrice) * rate : Number(limitPrice)) * (1 + (isUS ? FEE_US : FEE_KR))))
        : acc.max_buy_quantity || 0)
    : acc.sellable_quantity || 0;

  const overMax = calc.n > maxQty + 1e-9;
  const canSubmit = calc.n > 0 && !overMax && !busy
    && (orderType === "market" || Number(limitPrice) > 0);

  const submit = () => {
    if (!canSubmit) return;
    onSubmit({
      side, order_type: orderType, quantity: calc.n,
      limit_price: orderType === "limit" ? snap(Number(limitPrice), market) : undefined,
    });
  };

  const setPct = (p) => setQty(String(isUS
    ? Math.round(maxQty * p * 10000) / 10000
    : Math.floor(maxQty * p)));

  const nudge = (dir) => {
    const base = Number(limitPrice) || quote?.price || 0;
    const u = tickSize(base, market);
    setTouchedLimit(true);
    setLimitPrice(String(snap(Math.max(base + dir * u, u), market)));
  };

  if (!quote) {
    return <div className="ticket"><div className="empty">종목을 먼저 선택해 주세요.</div></div>;
  }

  return (
    <div className="ticket">
      {/* 매수 / 매도 */}
      <div className="tk-sides">
        <button className={`tk-side buy ${side === "buy" ? "on" : ""}`}
                onClick={() => setSide("buy")}>매수</button>
        <button className={`tk-side sell ${side === "sell" ? "on" : ""}`}
                onClick={() => setSide("sell")}>매도</button>
      </div>

      {/* 시장가 / 지정가 */}
      <div className="tk-types">
        {[["market", "시장가"], ["limit", "지정가"]].map(([k, label]) => (
          <button key={k} className={`tk-type ${orderType === k ? "on" : ""}`}
                  onClick={() => setOrderType(k)}>{label}</button>
        ))}
      </div>

      {/* 지정가 입력 */}
      {orderType === "limit" && (
        <div className="tk-field">
          <label>주문가격</label>
          <div className="tk-stepper">
            <button onClick={() => nudge(-1)} aria-label="한 호가 내림">−</button>
            <input className="input" type="number" value={limitPrice} step={unit}
                   onChange={(e) => { setTouchedLimit(true); setLimitPrice(e.target.value); }}
                   onBlur={(e) => setLimitPrice(String(snap(Number(e.target.value), market)))} />
            <button onClick={() => nudge(1)} aria-label="한 호가 올림">＋</button>
          </div>
          <div className="tk-note">
            호가 단위 {isUS ? "$0.01" : `${fmtNum(unit)}원`}
            {quote.price != null && (
              <>   현재가 대비{" "}
                <b style={{ color: Number(limitPrice) >= quote.price ? "var(--up)" : "var(--down)" }}>
                  {(((Number(limitPrice) || 0) / quote.price - 1) * 100).toFixed(2)}%
                </b>
              </>
            )}
          </div>
        </div>
      )}

      {orderType === "market" && (
        <div className="tk-market-note">
          현재가 <b>{isUS ? `$${fmtNum(quote.price, 2)}` : `${fmtNum(quote.price)}원`}</b> 로 즉시 체결됩니다.
        </div>
      )}

      {/* 수량 */}
      <div className="tk-field">
        <label>주문수량</label>
        <input className="input" type="number" value={qty} min="0"
               step={isUS ? "0.0001" : "1"}
               onChange={(e) => setQty(e.target.value)} placeholder="0" />
        <div className="tk-pcts">
          {[0.1, 0.25, 0.5, 1].map((p) => (
            <button key={p} onClick={() => setPct(p)}>
              {p === 1 ? "최대" : `${p * 100}%`}
            </button>
          ))}
        </div>
        <div className="tk-note">
          {side === "buy"
            ? <>주문가능 {fmtNum(acc.available_cash)}원   최대 <b>{fmtNum(maxQty, isUS ? 4 : 0)}</b>주</>
            : <>매도가능 <b>{fmtNum(acc.sellable_quantity, isUS ? 4 : 0)}</b>주
                {acc.owned_quantity > acc.sellable_quantity &&
                  ` (보유 ${fmtNum(acc.owned_quantity, isUS ? 4 : 0)}주 중 ` +
                  `${fmtNum(acc.owned_quantity - acc.sellable_quantity, isUS ? 4 : 0)}주 미체결 주문에 묶임)`}</>}
        </div>
      </div>

      {/* 실시간 금액 */}
      <div className="tk-calc">
        <div><span>주문단가</span>
          <b>{isUS ? `$${fmtNum(calc.native, 2)}` : `${fmtNum(calc.native)}원`}</b></div>
        <div><span>주문금액</span><b>{fmtNum(calc.gross)}원</b></div>
        <div><span>수수료</span><b>{fmtNum(calc.fee)}원</b></div>
        {side === "sell" && !isUS && (
          <div><span>거래세</span><b>{fmtNum(calc.tax)}원</b></div>
        )}
        <div className="tk-total">
          <span>{side === "buy" ? "총 매수금액" : "정산 예상금액"}</span>
          <b style={{ color: side === "buy" ? "var(--up)" : "var(--down)" }}>
            {fmtNum(calc.total)}원
          </b>
        </div>
        {isUS && calc.n > 0 && (
          <div className="tk-fxnote">
            환율 {fmtNum(rate)}원/USD 적용   계좌는 원화 단일 통화입니다
          </div>
        )}
      </div>

      {overMax && (
        <div className="tk-warn">
          {side === "buy" ? "주문가능 금액" : "매도가능 수량"}을 초과했습니다
          (최대 {fmtNum(maxQty, isUS ? 4 : 0)}주)
        </div>
      )}

      <button className={`tk-submit ${side}`} onClick={submit} disabled={!canSubmit}>
        {busy ? "처리 중…"
          : `${orderType === "limit" ? "지정가 " : "시장가 "}${side === "buy" ? "매수" : "매도"}`}
      </button>

      <div className="tk-disclaimer">
        모의 주문입니다. 실제 주문은 어떤 경우에도 나가지 않습니다.
      </div>
    </div>
  );
}
