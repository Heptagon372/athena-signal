/**
 * /api/* → 파이썬 백엔드(8000) 프록시
 *
 * 원래는 next.config.mjs 의 rewrite 로 넘겼습니다. rewrite 는 오류 처리를
 * 끼워 넣을 수 없어서, 백엔드를 재시작([3] 업데이트)하는 동안
 *
 *   Failed to proxy http://127.0.0.1:8000/... Error: read ECONNRESET
 *
 * 스택 트레이스가 터미널에 쏟아졌습니다. 두 가지 경우입니다.
 *   1) 서버가 내려가는 순간 진행 중이던 요청이 끊김
 *   2) 죽은 서버와 맺어둔 keep-alive 소켓을 재사용하다 리셋됨
 *
 * 이 핸들러는 rewrite 를 대체합니다 (next.config.mjs 에서 /api rewrite 제거).
 *   · GET 은 짧게 기다렸다 최대 2번 재시도 — 재시작 공백(수 초)과
 *     죽은 keep-alive 소켓을 그대로 흡수합니다
 *   · POST 등 상태를 바꾸는 요청은 재시도하지 않습니다 — 첫 요청이 서버에
 *     닿은 뒤 응답만 끊겼을 수 있어, 눈감고 다시 보내면 이중 실행 위험이
 *     있습니다 (주문은 client_order_id 로 막혀 있지만 원칙은 지킵니다)
 *   · 끝내 실패하면 스택 트레이스 대신 깔끔한 JSON 503 을 돌려줍니다.
 *     클라이언트(app/lib/api.js)는 data.error 를 그대로 화면에 띄우므로
 *     "백엔드 재시작 중" 안내가 사용자에게 보입니다.
 */

const BACKEND = process.env.ATHENA_API ?? "http://127.0.0.1:8000";

const UPSTREAM_TIMEOUT_MS = 60_000;   // 느린 조회(스크리너 등)도 있어 넉넉히
const RETRY_DELAY_MS = 700;           // 재시작 공백을 건너뛰기 위한 간격
const GET_ATTEMPTS = 3;               // GET 한정 (1 + 재시도 2)

// 백엔드로 넘길 요청 헤더 — host/connection 류는 넘기면 안 됩니다
const FORWARD_HEADERS = ["authorization", "content-type", "cookie", "accept"];

async function forward(req, ctx) {
  const { path = [] } = await ctx.params;
  const search = new URL(req.url).search;
  const target = `${BACKEND}/api/${path.map(encodeURIComponent).join("/")}${search}`;

  const headers = new Headers();
  for (const name of FORWARD_HEADERS) {
    const value = req.headers.get(name);
    if (value) headers.set(name, value);
  }

  const method = req.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD";
  const body = hasBody ? await req.arrayBuffer() : undefined;
  const attempts = hasBody ? 1 : GET_ATTEMPTS;

  let lastError;
  for (let i = 0; i < attempts; i++) {
    if (i > 0) await new Promise((r) => setTimeout(r, RETRY_DELAY_MS * i));
    try {
      const res = await fetch(target, {
        method,
        headers,
        body,
        cache: "no-store",
        redirect: "manual",
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
      // 상태·본문은 그대로 통과 — 401/404 판별은 클라이언트 몫입니다
      const out = new Headers();
      for (const name of ["content-type", "cache-control"]) {
        const value = res.headers.get(name);
        if (value) out.set(name, value);
      }
      // 이 헤더가 보이면 rewrite 가 아니라 이 핸들러를 타고 있는 것입니다
      out.set("x-athena-proxy", "route-handler");
      return new Response(res.body, { status: res.status, headers: out });
    } catch (err) {
      lastError = err;
      // 타임아웃은 서버가 살아 있는데 느린 것 — 재시도해봐야 부하만 늡니다
      if (err?.name === "TimeoutError" || err?.name === "AbortError") break;
    }
  }

  const isTimeout = lastError?.name === "TimeoutError" || lastError?.name === "AbortError";
  return Response.json(
    {
      error: isTimeout
        ? "백엔드 응답이 너무 늦습니다."
        : "백엔드 서버에 연결할 수 없습니다. 재시작(업데이트) 중이면 잠시 뒤 저절로 복구됩니다.",
      backend: BACKEND,
      restarting_hint: !isTimeout,
    },
    { status: isTimeout ? 504 : 503 },
  );
}

export {
  forward as GET,
  forward as POST,
  forward as PUT,
  forward as PATCH,
  forward as DELETE,
  forward as HEAD,
};
