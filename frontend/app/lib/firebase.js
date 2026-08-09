"use client";

/**
 * Firebase 로 구글 계정 고르기
 *
 * 이 파일이 하는 일은 하나입니다 — 계정 선택 창을 띄우고, 고른 계정의
 * **ID 토큰 문자열**을 돌려주는 것. 그 토큰으로 세션을 만드는 일은 서버
 * (POST /api/auth/firebase/session)가 하고, 검증 근거는
 * data_sources/firebase_auth.py 에 있습니다.
 *
 * 왜 signInWithPopup 인가 (signInWithRedirect 가 아니라)
 *     파이어폭스·사파리는 서드파티 저장소를 오리진별로 격리합니다. Firebase 의
 *     리디렉션 방식은 돌아온 결과를 authDomain(=프로젝트.firebaseapp.com) 쪽
 *     저장소에서 읽어오는 구조라, 그 브라우저들에서 **조용히 로그인이 안 된 채로**
 *     원래 페이지에 돌아옵니다 (자체 도메인에 auth 핸들러를 직접 호스팅해야 풀림).
 *     팝업은 결과를 postMessage 로 창끼리 직접 주고받아 그 문제가 없습니다.
 *     팝업이 차단된 경우는 조용히 실패하지 않고 사유를 그대로 띄웁니다.
 *
 * 왜 SDK 를 동적 import 하는가
 *     로그인 화면에서만 필요한 200KB 남짓을 첫 화면 번들에서 빼기 위해서입니다.
 *     동시에 서버 렌더 단계에서 firebase/auth 가 평가되는 일도 없어집니다.
 *
 * 왜 지속성을 메모리로 두는가
 *     세션의 주인은 우리 토큰(localStorage 의 athena_token)입니다. Firebase 까지
 *     따로 로그인 상태를 들고 있으면 "로그아웃했는데 아직 로그인돼 있는" 두 개의
 *     진실이 생깁니다. Firebase 는 자격증명 중개인으로만 쓰고 바로 지웁니다.
 */

import { api } from "./api";

let cached = null;      // { auth, mod } — 탭당 한 번만 초기화

/** 사람이 읽을 수 있는 문장으로. 여기 없는 코드는 원문을 그대로 보여줍니다. */
const ERROR_TEXT = {
  "auth/popup-blocked":
    "브라우저가 로그인 창을 막았습니다. 주소창의 팝업 차단을 허용한 뒤 다시 눌러 주세요.",
  "auth/popup-closed-by-user": "로그인 창을 닫았습니다.",
  "auth/cancelled-popup-request": "로그인 창을 닫았습니다.",
  "auth/user-cancelled": "구글 계정 연결을 취소했습니다.",
  "auth/unauthorized-domain":
    "이 주소는 Firebase 에 등록되지 않았습니다. Firebase 콘솔 → Authentication → " +
    "설정 → 승인된 도메인에 지금 주소의 호스트를 추가해 주세요.",
  "auth/operation-not-allowed":
    "Firebase 콘솔 → Authentication → Sign-in method 에서 Google 을 사용 설정해 주세요.",
  "auth/network-request-failed": "네트워크에 연결할 수 없습니다.",
  "auth/invalid-api-key": "FIREBASE_API_KEY 가 올바르지 않습니다.",
};

function readable(err) {
  const code = err?.code || "";
  return ERROR_TEXT[code] || err?.message || "구글 로그인에 실패했습니다.";
}

/** 서버가 내려준 공개 설정으로 Firebase 앱을 준비합니다. */
async function ensureAuth(config) {
  if (cached) return cached;

  const [appMod, authMod] = await Promise.all([
    import("firebase/app"),
    import("firebase/auth"),
  ]);

  // Next 개발 모드의 빠른 새로고침으로 이 모듈이 다시 평가되면 initializeApp 이
  // 두 번 불릴 수 있습니다. 이미 만든 앱이 있으면 그걸 씁니다.
  const app = appMod.getApps().length
    ? appMod.getApp()
    : appMod.initializeApp({
        apiKey: config.apiKey,
        authDomain: config.authDomain,
        projectId: config.projectId,
      });

  const auth = authMod.getAuth(app);
  auth.useDeviceLanguage();                         // 동의 화면을 브라우저 언어로
  await authMod.setPersistence(auth, authMod.inMemoryPersistence);

  cached = { auth, mod: authMod };
  return cached;
}

/**
 * 계정 선택 창을 띄우고 Firebase ID 토큰을 돌려줍니다.
 *
 * @param {{apiKey: string, authDomain: string, projectId: string}} config
 * @returns {Promise<string>} ID 토큰
 */
export async function pickGoogleAccount(config) {
  if (!config?.apiKey || !config?.projectId) {
    throw new Error("Firebase 설정을 받지 못했습니다.");
  }

  let auth;
  let mod;
  try {
    ({ auth, mod } = await ensureAuth(config));
  } catch (err) {
    throw new Error(`Firebase 를 초기화하지 못했습니다: ${err?.message || err}`);
  }

  const provider = new mod.GoogleAuthProvider();
  // 이게 없으면 브라우저에 로그인된 계정으로 조용히 넘어가 버립니다.
  // 계정이 여러 개인 사람이 원하는 계정을 고를 수 있어야 합니다.
  provider.setCustomParameters({ prompt: "select_account" });

  let credential;
  try {
    credential = await mod.signInWithPopup(auth, provider);
  } catch (err) {
    throw new Error(readable(err));
  }

  try {
    return await credential.user.getIdToken();
  } finally {
    // 서버에 넘길 토큰을 손에 넣은 순간부터 Firebase 세션은 쓸모가 없습니다.
    // 남겨두면 우리 세션과 별개의 로그인 상태가 하나 더 생깁니다.
    mod.signOut(auth).catch(() => {});
  }
}

/** 로그인 화면이 어떤 버튼을 그릴지 — 서버 설정을 그대로 전달합니다. */
export async function fetchAuthProviders() {
  try {
    const data = await api.get("/auth/providers", { timeout: 8000 });
    return {
      firebase: data?.firebase || { configured: false },
      google: data?.google || { configured: false },
    };
  } catch {
    // 백엔드가 잠깐 내려가 있어도 아이디/비번 로그인은 계속 보여야 합니다
    return { firebase: { configured: false }, google: { configured: false } };
  }
}
