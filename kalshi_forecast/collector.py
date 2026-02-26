"""Collect historical candlestick data from Kalshi weather markets."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .api import KalshiAPI
from . import config


def _parse_timestamp(ts_str: str) -> int:
    """Parse ISO timestamp string to unix timestamp."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return int(dt.timestamp())


def _candle_to_row(candle: dict, ticker: str) -> dict:
    """Flatten a candlestick response into a flat dict."""
    price = candle.get("price", {})
    return {
        "timestamp": candle["end_period_ts"],
        "ticker": ticker,
        "price_open": price.get("open", 0),
        "price_high": price.get("high", 0),
        "price_low": price.get("low", 0),
        "price_close": price.get("close", 0),
        "price_mean": price.get("mean", 0),
        "volume": candle.get("volume", 0),
        "open_interest": candle.get("open_interest", 0),
    }


def collect_series(
    api: KalshiAPI,
    series_ticker: str,
    output_dir: Path,
    period_interval: int = config.INTERVAL_1HR,
) -> int:
    """Collect all settled market candlesticks for a series.

    Groups markets by event and saves one parquet per event.
    Returns total number of candlesticks collected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching settled markets for {series_ticker}...")
    markets = api.list_markets(series_ticker, status="settled")
    print(f"  found {len(markets)} settled markets")

    # Group markets by event_ticker
    events: dict[str, list[dict]] = {}
    for m in markets:
        event = m["event_ticker"]
        events.setdefault(event, []).append(m)

    print(f"  across {len(events)} events")

    total_candles = 0
    for event_ticker, event_markets in sorted(events.items()):
        outfile = output_dir / f"{event_ticker}.parquet"
        if outfile.exists():
            continue

        rows = []
        for m in event_markets:
            ticker = m["ticker"]
            start_ts = _parse_timestamp(m["open_time"])
            close_ts = _parse_timestamp(m["close_time"])

            candles = api.get_candlesticks(
                series_ticker, ticker, start_ts, close_ts, period_interval
            )
            for c in candles:
                rows.append(_candle_to_row(c, ticker))

        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(outfile, index=False)
            total_candles += len(rows)
            print(f"  {event_ticker}: {len(rows)} candles -> {outfile.name}")

    return total_candles


def collect_open_markets(
    api: KalshiAPI,
    series_ticker: str,
    period_interval: int = config.INTERVAL_1HR,
) -> pd.DataFrame:
    """Collect current candlestick data for all open markets in a series."""
    import time as _time

    markets = api.get_open_markets(series_ticker)
    now = int(_time.time())

    rows = []
    for m in markets:
        ticker = m["ticker"]
        start_ts = _parse_timestamp(m["open_time"])
        candles = api.get_candlesticks(
            series_ticker, ticker, start_ts, now, period_interval
        )
        for c in candles:
            rows.append(_candle_to_row(c, ticker))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def collect_all(period_interval: int = config.INTERVAL_1HR) -> None:
    """Collect historical data for all configured weather series."""
    with KalshiAPI() as api:
        for series in config.WEATHER_SERIES:
            output_dir = config.DATA_DIR / series
            total = collect_series(api, series, output_dir, period_interval)
            print(f"  {series} total new candles: {total}\n")
