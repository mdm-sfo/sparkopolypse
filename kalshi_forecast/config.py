from pathlib import Path

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

WEATHER_SERIES = [
    "KXHIGHNY",   # NYC high temp
    "KXHIGHCHI",  # Chicago high temp
    "KXHIGHLAX",  # LA high temp
]

FINANCIAL_SERIES = [
    "KXINX",   # S&P 500
    "KXBTC",   # Bitcoin
    "KXETH",   # Ethereum
]

ALL_SERIES = WEATHER_SERIES + FINANCIAL_SERIES

DATA_DIR = Path(__file__).parent.parent / "data"

# Candlestick intervals
INTERVAL_1MIN = 1
INTERVAL_1HR = 60
INTERVAL_1DAY = 1440

# Rate limit: 20 req/sec on basic tier, stay under with some margin
RATE_LIMIT_DELAY = 0.1  # seconds between requests

# TimesFM settings
TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MAX_CONTEXT = 512
TIMESFM_MAX_HORIZON = 128
TIMESFM_BATCH_SIZE = 32
