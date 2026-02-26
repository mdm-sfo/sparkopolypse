"""Kalshi public market data API client (no auth required)."""

import time
import httpx
from . import config


class KalshiAPI:
    def __init__(self):
        self.client = httpx.Client(base_url=config.API_BASE, timeout=30)
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        delay = 0.15  # ~6-7 req/sec, well under 20/sec limit
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(5):
            self._rate_limit()
            resp = self.client.get(path, params=params)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    def list_markets(
        self,
        series_ticker: str,
        status: str = "settled",
        max_results: int | None = None,
    ) -> list[dict]:
        """Fetch markets for a series, paginating through results.

        Args:
            max_results: If set, stop after fetching this many markets.
        """
        all_markets = []
        cursor = None
        page_size = 1000  # max allowed by API
        while True:
            params = {
                "series_ticker": series_ticker,
                "status": status,
                "limit": page_size,
            }
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params)
            markets = data.get("markets", [])
            if not markets:
                break
            all_markets.extend(markets)
            if max_results and len(all_markets) >= max_results:
                all_markets = all_markets[:max_results]
                break
            cursor = data.get("cursor")
            if not cursor:
                break
            print(f"  fetched {len(all_markets)} markets so far...")
        return all_markets

    def get_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = config.INTERVAL_1HR,
    ) -> list[dict]:
        """Get candlestick data for a single market."""
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        data = self._get(path, {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        })
        return data.get("candlesticks", [])

    def get_open_markets(self, series_ticker: str) -> list[dict]:
        """Get currently open markets for a series."""
        return self.list_markets(series_ticker, status="open")

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
