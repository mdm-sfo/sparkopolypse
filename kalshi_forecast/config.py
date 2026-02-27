from pathlib import Path

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

WEATHER_SERIES = [
    # Active (backtested profitable)
    "KXHIGHNY",   # NYC high temp — 20c threshold
    "KXHIGHAUS",  # Austin high temp — 10c threshold
    "KXHIGHTDC",  # Washington DC high temp — 10c threshold
    "KXHIGHTLV",  # Las Vegas high temp — 10c threshold
    "KXHIGHLAX",  # LA high temp — 20c threshold (watch list)
    # Disabled (data collection only, not traded)
    "KXHIGHCHI",  # Chicago — never profitable
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
