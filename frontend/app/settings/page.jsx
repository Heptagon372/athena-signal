"use client";

/**
 * 설정 → API 키 (/settings)
 *
 * 계정마다 자기 API 키를 저장하는 화면입니다. 저장된 키는 MongoDB 에
 * **암호화**되어 들어가고, 이 계정으로 로그인한 요청과 자동매매에서만
 * 서버 키 대신 쓰입니다 (storage/user_credentials.py).
 *
 * 화면 규칙 두 가지:
 *   · 저장된 값 전체는 서버가 절대 돌려주지 않습니다 — 마스킹(····abcd)만
 *     옵니다. 입력창이 비어 있으면 "그대로 둠", 채우면 "교체"입니다.
 *   · "서버 공용 키 사용 중" 표시 — 키를 안 넣었는데 시세가 나오는 게
 *     버그가 아니라 서버 키 덕분임을 화면이 설명해야 합니다.
 */

import { useCallback, useEffect, useState } from "react";
import RequireAuth from "../components/RequireAuth";
import { api } from "../lib/api";

/** 필드 이름 → 사람이 읽는 라벨 */
const FIELD_LABELS = {
  TOSS_CLIENT_ID: "클라이언트 ID",
  TOSS_CLIENT_SECRET: "클라이언트 시크릿",
  KIS_APP_KEY: "앱 키",
  KIS_APP_SECRET: "앱 시크릿",
  REDDIT_CLIENT_ID: "클라이언트 ID",
  REDDIT_CLIENT_SECRET: "클라이언트 시크릿",
  NAVER_CLIENT_ID: "클라이언트 ID",
  NAVER_CLIENT_SECRET: "클라이언트 시크릿",
  KRX_AUTH_KEY: "인증 키",
  DATAGO_SERVICE_KEY: "서비스 키",
  KIS_ACCOUNT: "계좌번호 (예: 12345678-01)",
  KIS_DERIV_ACCOUNT: "파생 계좌번호 (선택)",
  KIS_MOCK: "모의투자 서버 (1=모의, 0=실전)",
  KIS_LIVE_TRADING: "실거래 스위치 (1=켜짐)",
};

/** 값 숨김이 필요한 필드 (비밀번호 입력창으로) */
const SECRET_FIELD = /SECRET|_KEY$/;

function Field({ field, value, onChange }) {
  const label = FIELD_LABELS[field.name] || field.name;
  const secret = SECRET_FIELD.test(field.name);
  return (
    <label className="field">
      <span>
        {label} <em style={{ opacity: 0.6 }}>({field.name})</em>
        {field.set && <em className="key-state key-mine">저장됨 {field.masked}</em>}
        {!field.set && field.server_fallback && (
          <em className="key-state key-server">서버 공용 키 사용 중</em>
        )}
      </span>
      <input
        className="input"
        type={secret ? "password" : "text"}
        value={value ?? ""}
        autoComplete="off"
        placeholder={field.set ? "비워 두면 그대로 둡니다 · 지우려면 서비스 삭제" : "값 입력"}
        onChange={(e) => onChange(field.name, e.target.value)}
      />
    </label>
  );
}

function ProviderCard({ id, provider, onSaved }) {
  const [draft, setDraft] = useState({});
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);   // {ok, text}

  const change = (name, value) => setDraft((d) => ({ ...d, [name]: value }));
  const dirty = Object.values(draft).some((v) => (v ?? "").trim() !== "");

  const save = async () => {
    const values = Object.fromEntries(
      Object.entries(draft).filter(([, v]) => (v ?? "").trim() !== ""));
    if (!Object.keys(values).length) return;
    setBusy(true);
    setMsg(null);
    try {
      const data = await api.put("/keys", { values });
      setDraft({});
      setMsg({ ok: true, text: `저장했습니다 (${data.saved.join(", ")})` });
      onSaved(data);
    } catch (err) {
      setMsg({ ok: false, text: err.message || "저장에 실패했습니다." });
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (!window.confirm(`${provider.label} 키를 계정에서 삭제할까요?\n(서버 공용 키가 있으면 그쪽으로 되돌아갑니다)`)) return;
    setBusy(true);
    setMsg(null);
    try {
      const data = await api.del(`/keys/${id}`);
      setDraft({});
      setMsg({ ok: true, text: "삭제했습니다." });
      onSaved(data);
    } catch (err) {
      setMsg({ ok: false, text: err.message || "삭제에 실패했습니다." });
    } finally {
      setBusy(false);
    }
  };

  const anyMine = provider.fields.some((f) => f.set)
    || provider.extra.some((f) => f.set);

  return (
    <div className="card key-card">
      <div className="key-card-head">
        <div>
          <h3>{provider.label}</h3>
          <p className="key-gives">{provider.gives}</p>
        </div>
        {provider.mine
          ? <span className="key-badge key-badge-mine">내 키</span>
          : provider.fields.some((f) => f.server_fallback)
            ? <span className="key-badge key-badge-server">서버 키</span>
            : <span className="key-badge">미설정</span>}
      </div>

      {provider.fields.map((f) => (
        <Field key={f.name} field={f} value={draft[f.name]} onChange={change} />
      ))}

      {provider.extra.length > 0 && (
        <details className="key-extra" open={provider.extra.some((f) => f.set)}>
          <summary>거래 설정 (계좌·모의/실전)</summary>
          {provider.extra.map((f) => (
            <Field key={f.name} field={f} value={draft[f.name]} onChange={change} />
          ))}
          <p className="key-note">
            ⚠️ 실거래 스위치는 <strong>본인 키·본인 계좌</strong>에만 적용됩니다.
            구글 계정은 앱 키·시크릿·계좌번호를 모두 저장해야 KIS 주문(모의/실전)을
            낼 수 있습니다 — 저장 전에는 가상 자금 모의투자만 가능합니다.
          </p>
        </details>
      )}

      {msg && (
        <div className={msg.ok ? "key-msg-ok" : "auth-error"}>{msg.text}</div>
      )}

      <div className="key-actions">
        <a href={`https://${provider.portal.replace(/^https?:\/\//, "").split(" ")[0]}`}
           target="_blank" rel="noreferrer" className="key-portal">
          키 발급처 ↗
        </a>
        <div style={{ flex: 1 }} />
        {anyMine && (
          <button className="btn" onClick={remove} disabled={busy}>삭제</button>
        )}
        <button className="btn btn-gold" onClick={save} disabled={busy || !dirty}>
          {busy ? "저장 중…" : "저장"}
        </button>
      </div>
    </div>
  );
}

function SettingsBody() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setData(await api.get("/keys", { timeout: 15000 }));
      setError("");
    } catch (err) {
      setError(err.message || "키 정보를 불러오지 못했습니다.");
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  if (error) return <div className="auth-error" style={{ margin: 24 }}>{error}</div>;
  if (!data) return <div className="loading">불러오는 중…</div>;

  return (
    <section className="section">
      <div className="eyebrow">설정</div>
      <h1 className="settings-title">내 API 키</h1>
      <p className="settings-lede">
        여기 저장한 키는 <strong>내 계정에만</strong> 적용됩니다 — 시세·뉴스 조회와
        자동매매가 서버 공용 키 대신 내 키로 동작합니다. 키는 암호화되어
        저장되며, 저장된 값은 다시 표시되지 않습니다 (끝 4자리만 확인용으로 보여줍니다).
      </p>
      <div className="key-grid">
        {Object.entries(data.providers).map(([id, provider]) => (
          <ProviderCard key={id} id={id} provider={provider}
                        onSaved={(next) => setData(next)} />
        ))}
      </div>
    </section>
  );
}

export default function SettingsPage() {
  return (
    <RequireAuth what="API 키 설정">
      <SettingsBody />
    </RequireAuth>
  );
}
