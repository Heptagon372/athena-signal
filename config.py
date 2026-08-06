"""
아테나 시그널 - 설정 파일
공식(가중치)은 여기서 관리하고, backtest 결과에 따라 자동으로 갱신됩니다.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

# 데이터가 실제로 저장될 위치.
# exe(PyInstaller onefile)로 실행하면 __file__ 은 실행할 때마다 지워지는 임시
# 압축해제 폴더를 가리키므로, DB/캐시를 거기 두면 종료 시 예측 기록이 사라집니다.
# 그래서 exe 로 돌 때는 실행 파일이 놓인 폴더를 데이터 폴더로 씁니다.
if getattr(sys, "frozen", False):
    DATA_DIR = Path(sys.executable).parent
else:
    DATA_DIR = BASE_DIR

DB_PATH = str(DATA_DIR / "athena.db")

# 캐시 디렉터리 (KRX 상장종목 마스터 등)
CACHE_DIR = DATA_DIR / ".cache"
SYMBOL_CACHE_TTL_DAYS = 3  # 상장/폐지 반영을 위해 3일마다 종목 마스터 갱신

# 외부 요청 공통 설정
HTTP_TIMEOUT = 10
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# 예측 시평선 (Prediction Horizons)
# ---------------------------------------------------------------------------
# key      : 내부 식별자 (DB의 horizon_label 로 저장)
# minutes  : 지금으로부터 몇 분 뒤를 예측하는가
# label    : 화면 표기
# intraday : 분봉 기반 단기 점수의 반영 비율 (나머지는 일봉 기반 점수)
# conf     : 그 시평선에서 신호를 얼마나 신뢰할지 (확률을 50%로 끌어당기는 정도)
#
# intraday / conf 값의 근거
#   · 10분~1시간 : 일봉 지표는 이 구간을 설명하지 못하므로 분봉 점수를 주로 씁니다.
#                  다만 초단기는 호가 노이즈가 커서 신뢰도(conf)를 낮게 잡았습니다.
#   · 6~12시간   : 하루의 절반 이상 — 분봉과 일봉을 섞습니다.
#   · 18시간~1일 : 일봉 지표의 본래 예측 대상 구간이라 conf가 가장 높습니다.
#
#   이 수치들은 이론적으로 유도된 상수가 아니라 설계 판단값입니다.
#   backtest 로 시평선별 적중률이 쌓이면 그 결과로 조정하는 것이 맞습니다.
PREDICTION_HORIZONS = [
    {"key": "10min", "minutes": 10,   "label": "10분 뒤", "intraday": 0.85, "conf": 0.55},
    {"key": "1h",    "minutes": 60,   "label": "1시간 뒤", "intraday": 0.70, "conf": 0.70},
    {"key": "6h",    "minutes": 360,  "label": "6시간 뒤", "intraday": 0.45, "conf": 0.85},
    {"key": "12h",   "minutes": 720,  "label": "12시간 뒤", "intraday": 0.30, "conf": 0.95},
    {"key": "18h",   "minutes": 1080, "label": "18시간 뒤", "intraday": 0.20, "conf": 1.00},
    {"key": "1d",    "minutes": 1440, "label": "1일 뒤",   "intraday": 0.12, "conf": 1.00},
]

# 단기(분봉) 기술점수 계산에 쓸 분봉 개수
INTRADAY_BARS = 240

# ---------------------------------------------------------------------------
# 장 시작 전 예측 (Opening Gap Forecast)
# ---------------------------------------------------------------------------
# "다음 정규장이 열릴 때 전일 종가 대비 위로 뜰 것인가"를 따로 계산합니다.
#
# 시가 갭은 장중 흐름과 동인이 다릅니다.
#   · 밤사이 쌓인 뉴스와 해외 증시가 갭의 주 동인입니다  -> 뉴스 비중을 키움
#   · 분봉 기반 단기 신호는 장이 닫힌 동안 의미가 없습니다 -> 아예 쓰지 않음
#   · 일봉 기술 신호는 갭보다 '세션 중 흐름'을 설명하므로 약간 낮춤
#
# 이 배수들도 이론값이 아니라 설계 판단값입니다. backtest 로 시가 방향 적중률이
# 쌓이면 그 결과로 조정하는 것이 옳습니다.
OPEN_PREDICTION = {
    "news_amplifier": 1.35,        # 밤사이 뉴스 = 갭의 주 동인
    "technical_damping": 0.85,     # 일봉 기술신호는 갭 설명력이 상대적으로 낮음
    "community_damping": 1.0,
}

# 초기 가중치 (기술적 지표 / 뉴스 감성 / 커뮤니티 여론)
# 합이 1.0이 되도록 유지. backtest.py 가 이 값을 자동 조정합니다.
INITIAL_WEIGHTS = {
    "technical": 0.5,
    "news_sentiment": 0.3,
    "community_sentiment": 0.2,
}

# 가중치 학습률 (한 번의 backtest 사이클마다 최대 얼마나 이동할지)
WEIGHT_LEARNING_RATE = 0.05

# 확률을 0~100% 로 스케일링할 때 쓰는 시그모이드 민감도
SIGMOID_SENSITIVITY = 4.0

# ---------------------------------------------------------------------------
# 기술적 지표 가중치
# ---------------------------------------------------------------------------
# engine/indicators.py 가 계산하는 개별 지표들의 상대 비중입니다.
# 기술점수 = Σ(가중치 × 개별지표점수) / Σ(계산에 성공한 지표의 가중치)
#
# 비중 근거:
#   - 추세추종 계열(MACD/이동평균/ROC)에 가장 큰 비중 → 단기 방향성 예측의 주력
#   - 오실레이터 계열(RSI/스토캐스틱/볼린저)은 과열·침체 되돌림 신호 → 보조
#   - 수급 계열(거래량/OBV)은 추세의 신뢰도 확인 → 보조
#   - ATR은 방향성이 없는 변동성 지표라 가중치 0 (참고 정보로만 표시)
#
# 이 값은 "기본 비중"이고, engine/indicators.detect_regime() 이 판별한 국면에 따라
# 지표군(추세/평균회귀)별 배수가 곱해져 최종 비중이 정해집니다.
INDICATOR_WEIGHTS = {
    # 추세추종형
    "macd": 1.2,
    "ma_cross": 1.2,
    "adx": 1.1,           # Wilder 추세 강도
    "trend_reg": 1.0,     # 회귀 기울기 × R²
    "roc": 0.9,
    "ma_trend": 0.8,
    "aroon": 0.7,
    "obv": 0.6,
    "price_position": 0.5,
    # 평균회귀형
    "rsi": 1.0,
    "bollinger": 0.8,
    "cci": 0.8,
    "stochastic": 0.7,
    "williams": 0.6,
    # 확인형(수급)
    "mfi": 0.7,
    "cmf": 0.6,
    "volume": 0.6,
    # 참고용 (방향성 없음 -> 확률에 미반영)
    "atr": 0.0,
    "yz_vol": 0.0,
}

# 기술적 분석에 필요한 최소 일봉 개수 (60일선/볼린저 계산에 필요)
MIN_BARS_FOR_TECHNICAL = 20
PREFERRED_BARS = 120
