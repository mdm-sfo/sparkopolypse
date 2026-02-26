#!/usr/bin/env python3
"""
Fast parallel Kalshi data collector.

Splits work between Spark and EC2 by event date (even/odd).
Skips low-volume markets during listing phase.
Runs faster by reducing delay between requests.

Usage:
    python3 fast_collector.py --node spark --series KXBTC KXETH
    python3 fast_collector.py --node ec2 --series KXBTC KXETH
"""

import argparse
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
INTERVAL_1HR = 60

# Minimum total volume across all markets in an event to bother collecting
MIN_EVENT_VOLUME = 0  # we filter at market level instead


class FastCollector:
    def __init__(self, output_dir: str, node_id: int, total_nodes: int,
                 min_markets_per_event: int = 3):
        self.output_dir = Path(output_dir)
        self.node_id = node_id  # 0, 1, 2, ...
        self.total_nodes = total_nodes
        self.min_markets = min_markets_per_event
        self.client = httpx.Client(base_url=API_BASE, timeout=30)
        self._last_req = 0.0
        self._req_count = 0

    def _rate_limit(self):
        """Target ~15 req/sec to stay safely under 20/sec limit."""
        elapsed = time.time() - self._last_req
        delay = 0.07  # ~14 req/sec
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_req = time.time()
        self._req_count += 1

    def _get(self, path: str, params: dict = None) -> dict:
        for attempt in range(5):
            self._rate_limit()
            try:
                resp = self.client.get(path, params=params)
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                print(f"  timeout, retry {attempt+1}...")
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        return {}

    def _is_my_event(self, event_ticker: str) -> bool:
        """Deterministic split: hash the event ticker, assign to node."""
        h = int(hashlib.md5(event_ticker.encode()).hexdigest(), 16)
        return (h % self.total_nodes) == self.node_id

    def list_markets(self, series_ticker: str) -> list[dict]:
        """Fetch all settled markets for a series."""
        all_markets = []
        cursor = None
        while True:
            params = {
                "series_ticker": series_ticker,
                "status": "settled",
                "limit": 1000,
            }
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params)
            markets = data.get("markets", [])
            if not markets:
                break
            all_markets.extend(markets)
            cursor = data.get("cursor")
            if not cursor:
                break
            if len(all_markets) % 10000 < 1000:
                print(f"  fetched {len(all_markets)} markets...", flush=True)
        return all_markets

    def collect_series(self, series_ticker: str):
        """Collect candle data for a series, splitting work by node."""
        series_dir = self.output_dir / series_ticker
        series_dir.mkdir(parents=True, exist_ok=True)

        print(f"Collecting {series_ticker} (node {self.node_id}/{self.total_nodes})...", flush=True)
        print(f"Fetching settled markets...", flush=True)
        markets = self.list_markets(series_ticker)
        print(f"  found {len(markets)} settled markets", flush=True)

        # Group by event
        events: dict[str, list[dict]] = {}
        for m in markets:
            events.setdefault(m["event_ticker"], []).append(m)

        total_events = len(events)
        my_events = {k: v for k, v in events.items() if self._is_my_event(k)}
        print(f"  {total_events} total events, {len(my_events)} assigned to node {self.node_id}", flush=True)

        # Skip events with very few markets (likely illiquid)
        if self.min_markets > 1:
            before = len(my_events)
            my_events = {k: v for k, v in my_events.items() if len(v) >= self.min_markets}
            skipped = before - len(my_events)
            if skipped:
                print(f"  skipped {skipped} events with < {self.min_markets} markets", flush=True)

        collected = 0
        skipped_existing = 0
        for event_ticker in sorted(my_events.keys()):
            outfile = series_dir / f"{event_ticker}.parquet"
            if outfile.exists():
                skipped_existing += 1
                continue

            event_markets = my_events[event_ticker]
            rows = []
            for m in event_markets:
                ticker = m["ticker"]
                start_ts = int(datetime.fromisoformat(
                    m["open_time"].replace("Z", "+00:00")
                ).timestamp())
                close_ts = int(datetime.fromisoformat(
                    m["close_time"].replace("Z", "+00:00")
                ).timestamp())

                path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
                data = self._get(path, {
                    "start_ts": start_ts,
                    "end_ts": close_ts,
                    "period_interval": INTERVAL_1HR,
                })
                for c in data.get("candlesticks", []):
                    price = c.get("price", {})
                    rows.append({
                        "timestamp": c["end_period_ts"],
                        "ticker": ticker,
                        "price_open": price.get("open", 0),
                        "price_high": price.get("high", 0),
                        "price_low": price.get("low", 0),
                        "price_close": price.get("close", 0),
                        "price_mean": price.get("mean", 0),
                        "volume": c.get("volume", 0),
                        "open_interest": c.get("open_interest", 0),
                    })

            if rows:
                df = pd.DataFrame(rows)
                df.to_parquet(outfile, index=False)
                collected += 1
                if collected % 10 == 0 or collected < 5:
                    print(f"  [{collected}] {event_ticker}: {len(rows)} candles", flush=True)

        print(f"  Done: {collected} new, {skipped_existing} existing", flush=True)
        print(f"  Total API requests this run: {self._req_count}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fast parallel Kalshi collector")
    parser.add_argument("--node-id", required=True, type=int,
                        help="This node's ID (0-indexed)")
    parser.add_argument("--total-nodes", required=True, type=int,
                        help="Total number of collector nodes")
    parser.add_argument("--series", nargs="+", default=["KXBTC", "KXETH"],
                        help="Series to collect")
    parser.add_argument("--output", default=os.path.expanduser("~/kalshi-forecast/data"),
                        help="Output directory")
    parser.add_argument("--min-markets", type=int, default=3,
                        help="Skip events with fewer than this many markets")
    parser.add_argument("--loop", action="store_true",
                        help="Loop continuously with 1hr sleep between runs")
    args = parser.parse_args()

    collector = FastCollector(args.output, args.node_id, args.total_nodes, args.min_markets)

    while True:
        for series in args.series:
            try:
                collector.collect_series(series)
            except Exception as e:
                print(f"ERROR collecting {series}: {e}", flush=True)

        if not args.loop:
            break
        print(f"Sleeping 3600s before next run...", flush=True)
        time.sleep(3600)


if __name__ == "__main__":
    main()
