"""
한국투자증권 KIS Open API 어댑터 (증권사 공식 API)
--------------------------------------------------
네이버/토스는 서비스 내부용 비공식 엔드포인트라 언제든 막힐 수 있습니다.
한국투자증권은 개인 개발자에게 **공식 REST API**를 무료로 열어주므로, 키를 발급받으면
이쪽을 우선 사용하도록 했습니다.

키 발급 방법 (5분, 무료, 계좌 필요)
    1. https://apiportal.koreainvestment.com 접속 -> 로그인
    2. [API 신청] -> 실전투자 또는 모의투자 선택
    3. APP KEY / APP SECRET 발급
    4. 아래 환경변수 설정 후 서버 재시작

        set KIS_APP_KEY=발급받은_APP_KEY
        set KIS_APP_SECRET=발급받은_APP_SECRET
        set KIS_ACCOUNT=12345678-01          (선택)
        set KIS_MOCK=1                        (모의투자 서버를 쓸 때만)

    PowerShell 이라면 set 대신:
        $env:KIS_APP_KEY="..."

설정하지 않으면 이 모듈은 조용히 비활성 상태로 남고, 기존 네이버/토스 경로가 그대로
동작합니다 (is_configured() 로 확인).

KIS가 네이버 대비 좋은 점
    · 공식 API라 약관상 안전하고 구조가 잘 바뀌지 않음
    · **투자자별 매매동향(개인/외국인/기관)을 장중에도** 조회 가능
      (네이버 trend는 전일 확정치 위주)
    · 호가/체결 등 더 세밀한 데이터 접근 가능

주의: 이 어댑터는 KIS 공식 문서의 스펙(tr_id, 필드명)에 맞춰 작성했지만, 제가 실제
      계정 키로 응답을 검증하지는 못했습니다. 키를 넣고 처음 실행할 때
      `python -m data_sources.kis_client` 로 자체 점검을 돌려보세요.
"""

import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from data_sources import credentials, http_client

REAL_BASE = "https://openapi.koreainvestment.com:9443"
MOCK_BASE = "https://openapivts.koreainvestment.com:29443"

_token_lock = threading.Lock()
_token: str | None = None
_token_expires_at: float = 0.0


def app_key() -> str:
    return credentials.get("KIS_APP_KEY")


def app_secret() -> str:
    return credentials.get("KIS_APP_SECRET")


def is_configured() -> bool:
    return bool(app_key() and app_secret())


def is_mock() -> bool:
    """모의투자 서버를 쓰는지 여부. 주문 어댑터가 tr_id 를 고를 때도 씁니다."""
    return credentials.get("KIS_MOCK") == "1"


def use_for_quotes() -> bool:
    """시세 조회에도 KIS 를 쓸지 (기본 예).

    KIS 는 계정당 초당 호출 한도가 있는데, 시세 폴링이 그 한도를 대부분
    잡아먹습니다. `KIS_QUOTES=0` 으로 끄면 시세는 네이버/토스 무료 경로로 돌리고
    **KIS 한도는 주문·잔고·체결에만** 씁니다. 자동매매 안정성에는 이쪽이 낫습니다.
    (선물·옵션 시세는 다른 경로가 없어 이 설정과 무관하게 KIS 를 씁니다)
    """
    return credentials.get("KIS_QUOTES", "1") != "0"


def _base_url() -> str:
    return MOCK_BASE if is_mock() else REAL_BASE


# ---------------------------------------------------------------------------
# 호출 속도 제한 (KIS는 계정당 초당 호출 수를 제한합니다)
# ---------------------------------------------------------------------------
# 실전은 초당 약 20건, 모의투자는 초당 2건까지만 받아줍니다. 넘기면
# "초당 거래건수를 초과하였습니다"로 거부됩니다. 자동매매는 종목마다 시세·잔고를
# 연속으로 조회하므로 제한에 아주 쉽게 걸립니다.
# 시세(kis_client)와 주문(kis_trading)이 같은 계정을 쓰므로 **한 곳에서** 조절합니다.
_rate_lock = threading.Lock()
_last_call_at = 0.0

# 문서상 한도는 실전 초당 20건이지만, 실제로는 그보다 낮게 걸리는 계정이 있습니다.
# 게다가 이 프로그램은 한 키를 여러 곳이 나눠 씁니다 —
# 웹 화면 시세 폴링 · 성적표 실시간 채점 · 모의투자 평가 · 자동매매 루프 · 진단.
# 그래서 문서값에 맞추지 않고 보수적으로 잡습니다. 조회가 조금 느려지는 것이
# "초당 초과"로 통째로 실패하는 것보다 낫습니다.
REAL_MIN_INTERVAL = 0.35      # 초당 약 3회
MOCK_MIN_INTERVAL = 0.60      # 모의는 초당 2회 제한


def min_interval() -> float:
    """호출 간격(초). KIS_MIN_INTERVAL 로 덮어쓸 수 있습니다.

    계정마다 실제 한도가 달라서, 코드 수정 없이 조절할 수 있어야 합니다.
    """
    override = credentials.get("KIS_MIN_INTERVAL")
    if override:
        try:
            return max(float(override), 0.0)
        except ValueError:
            pass
    return MOCK_MIN_INTERVAL if is_mock() else REAL_MIN_INTERVAL


def throttle():
    """직전 호출과 최소 간격을 확보합니다 (호출 직전에 부릅니다).

    이 프로그램 안의 **모든** KIS 호출이 이 한 줄을 통과합니다.
    시세와 주문이 같은 계정 한도를 나눠 쓰기 때문입니다.
    """
    global _last_call_at
    interval = min_interval()
    with _rate_lock:
        wait = interval - (time.time() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.time()


def _get_token() -> str | None:
    """접근토큰 발급 (KIS는 24시간 유효, 재발급 호출에 제한이 있어 캐시 필수)."""
    global _token, _token_expires_at

    if not is_configured():
        return None

    with _token_lock:
        if _token and time.time() < _token_expires_at - 300:
            return _token

        res = http_client.post_json(
            _base_url() + "/oauth2/tokenP",
            json_body={
                "grant_type": "client_credentials",
                "appkey": app_key(),
                "appsecret": app_secret(),
            },
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if not res or not res.get("access_token"):
            return None

        _token = res["access_token"]
        # expires_in 이 없으면 보수적으로 12시간만 신뢰
        _token_expires_at = time.time() + float(res.get("expires_in") or 43200)
        return _token


def _headers(tr_id: str) -> dict | None:
    token = _get_token()
    if not token:
        return None
    return {
        "authorization": f"Bearer {token}",
        "appkey": app_key(),
        "appsecret": app_secret(),
        "tr_id": tr_id,
        "custtype": "P",          # 개인
        "Content-Type": "application/json",
    }


# 주문 어댑터(kis_trading)가 토큰·주소·헤더를 재사용할 수 있도록 공개한 접근자입니다.
# 토큰 발급은 호출 횟수 제한이 있어 반드시 이 캐시를 공유해야 합니다.
def base_url() -> str:
    return _base_url()


def access_token() -> str | None:
    return _get_token()


def auth_headers(tr_id: str, extra: dict | None = None) -> dict | None:
    headers = _headers(tr_id)
    if headers and extra:
        headers.update(extra)
    return headers


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 시세
# ---------------------------------------------------------------------------

def get_quote(code: str) -> dict | None:
    """주식현재가 시세 (tr_id: FHKST01010100).

    장중에는 실시간 체결가, 장 마감 후에는 종가를 돌려줍니다.
    """
    headers = _headers("FHKST01010100")
    if not headers:
        return None

    throttle()
    data = http_client.get_json(
        _base_url() + "/uapi/domestic-stock/v1/quotations/inquire-price",
        headers=headers,
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        timeout=10,
    )
    out = (data or {}).get("output")
    if not out:
        return None

    return {
        "close": _num(out.get("stck_prpr")),          # 현재가
        "prev_close": _num(out.get("stck_sdpr")),     # 전일 종가(기준가)
        "change": _num(out.get("prdy_vrss")),         # 전일 대비
        "change_rate": _num(out.get("prdy_ctrt")),    # 등락률(%)
        "open": _num(out.get("stck_oprc")),
        "high": _num(out.get("stck_hgpr")),
        "low": _num(out.get("stck_lwpr")),
        "volume": _num(out.get("acml_vol")),          # 누적 거래량
        "trading_value": _num(out.get("acml_tr_pbmn")),  # 누적 거래대금
        "high_52w": _num(out.get("w52_hgpr")),
        "low_52w": _num(out.get("w52_lwpr")),
        "per": out.get("per"),
        "pbr": out.get("pbr"),
        "market_cap": _num(out.get("hts_avls")),      # 시가총액(억원)
        "upper_limit": _num(out.get("stck_mxpr")),    # 상한가
        "lower_limit": _num(out.get("stck_llam")),    # 하한가
        "source": "한국투자증권 KIS",
    }


def get_investor_flow(code: str) -> dict | None:
    """투자자별 매매동향 (tr_id: FHKST01010900).

    네이버 trend 와 달리 **장중에도** 갱신되는 것이 큰 장점입니다.
    응답은 최신일이 배열 첫 번째로 옵니다.
    """
    headers = _headers("FHKST01010900")
    if not headers:
        return None

    throttle()
    data = http_client.get_json(
        _base_url() + "/uapi/domestic-stock/v1/quotations/inquire-investor",
        headers=headers,
        params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        timeout=10,
    )
    rows = (data or {}).get("output")
    if not rows:
        return None

    latest = rows[0] if isinstance(rows, list) else rows
    raw_date = str(latest.get("stck_bsop_date") or "")
    return {
        "date": (f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                 if len(raw_date) == 8 else raw_date),
        "individual": _num(latest.get("prsn_ntby_qty")),    # 개인 순매수 수량
        "foreign": _num(latest.get("frgn_ntby_qty")),       # 외국인 순매수 수량
        "institution": _num(latest.get("orgn_ntby_qty")),   # 기관계 순매수 수량
        "individual_value": _num(latest.get("prsn_ntby_tr_pbmn")),
        "foreign_value": _num(latest.get("frgn_ntby_tr_pbmn")),
        "institution_value": _num(latest.get("orgn_ntby_tr_pbmn")),
        "source": "한국투자증권 KIS 투자자별 매매동향",
    }


def get_daily_chart(code: str, days: int = 120) -> pd.DataFrame:
    """기간별 시세 (tr_id: FHKST03010100).

    이 TR 은 **호출당 최대 약 100건**만 돌려줍니다. 예전에는 한 번만 불러서,
    260봉을 요청해도 조용히 100봉이 나왔습니다 — ML 학습(표본 145봉 필요)이
    "표본 부족"으로 꺼지는 원인이었습니다. 그래서 받은 것 중 가장 오래된 날짜
    앞으로 창을 옮겨 가며 필요한 만큼 이어 붙입니다 (역방향 페이지네이션).
    """
    headers = _headers("FHKST03010100")
    if not headers:
        return pd.DataFrame()

    parsed: list[dict] = []
    end = datetime.now()
    # 페이지당 ~100 거래일 ≈ 달력 150일. 여유를 두고 필요한 페이지 수를 잡되,
    # API 이상으로 무한히 과거로 가지 않게 상한을 둡니다.
    max_pages = min(6, days // 90 + 2)

    for _ in range(max_pages):
        start = end - timedelta(days=150)
        throttle()
        data = http_client.get_json(
            _base_url() + "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            headers=headers,
            params={
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": end.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",   # 일봉
                "FID_ORG_ADJ_PRC": "0",       # 수정주가 반영
            },
            timeout=15,
        )
        rows = (data or {}).get("output2") or []
        page = []
        for r in rows:
            date = str(r.get("stck_bsop_date") or "")
            close = _num(r.get("stck_clpr"))
            if len(date) != 8 or close is None:
                continue
            page.append({
                "date": datetime.strptime(date, "%Y%m%d"),
                "open": _num(r.get("stck_oprc")) or close,
                "high": _num(r.get("stck_hgpr")) or close,
                "low": _num(r.get("stck_lwpr")) or close,
                "close": close,
                "volume": _num(r.get("acml_vol")) or 0,
            })
        if not page:
            break                     # 더 과거가 없거나(상장 초) 조회 실패
        parsed.extend(page)
        if len(parsed) >= days:
            break
        end = min(p["date"] for p in page) - timedelta(days=1)

    if not parsed:
        return pd.DataFrame()

    df = pd.DataFrame(parsed).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.tail(days) if len(df) > days else df


def self_check():
    """키를 넣은 뒤 연결이 되는지 확인하는 자체 점검 (python -m data_sources.kis_client)."""
    print("=" * 56)
    print("  한국투자증권 KIS API 자체 점검")
    print("=" * 56)

    if not is_configured():
        print("\n❌ KIS_APP_KEY / KIS_APP_SECRET 환경변수가 설정되지 않았습니다.")
        print("\n   설정 방법 (명령 프롬프트):")
        print("     set KIS_APP_KEY=발급받은_APP_KEY")
        print("     set KIS_APP_SECRET=발급받은_APP_SECRET")
        print("\n   PowerShell:")
        print('     $env:KIS_APP_KEY="발급받은_APP_KEY"')
        print('     $env:KIS_APP_SECRET="발급받은_APP_SECRET"')
        print("\n   키 발급: https://apiportal.koreainvestment.com")
        print("\n   ※ 설정하지 않아도 네이버/토스 경로로 정상 동작합니다.")
        return

    mode = "모의투자" if is_mock() else "실전투자"
    print(f"\n서버: {mode} ({_base_url()})")

    print("\n[1/4] 접근토큰 발급...", end=" ")
    token = _get_token()
    print("성공" if token else "실패 — APP KEY/SECRET을 확인하세요")
    if not token:
        return

    print("[2/4] 현재가 조회 (삼성전자)...", end=" ")
    q = get_quote("005930")
    print(f"{q['close']:,.0f}원 ({q['change_rate']:+.2f}%)" if q else "실패")

    print("[3/4] 투자자별 매매동향...", end=" ")
    f = get_investor_flow("005930")
    if f:
        print(f"개인 {f['individual']:+,.0f} / 외국인 {f['foreign']:+,.0f} / 기관 {f['institution']:+,.0f}")
    else:
        print("실패")

    print("[4/4] 일봉 조회...", end=" ")
    df = get_daily_chart("005930", days=60)
    print(f"{len(df)}봉" if not df.empty else "실패")

    print("\n✅ 점검 완료 — 서버를 재시작하면 KIS가 우선 사용됩니다.")


if __name__ == "__main__":
    self_check()
