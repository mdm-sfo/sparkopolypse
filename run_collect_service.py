#!/usr/bin/env python3
"""Kalshi data collector service — runs continuously, collects all series."""

import time
from kalshi_forecast.collector import collect_series
from kalshi_forecast.api import KalshiAPI
from kalshi_forecast import config

SERIES_TO_COLLECT = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX", "KXINX", "KXBTC", "KXETH"]
SLEEP_BETWEEN_RUNS = 3600  # re-scan every hour for new settled markets


def main():
    while True:
        with KalshiAPI() as api:
            for series in SERIES_TO_COLLECT:
                output_dir = config.DATA_DIR / series
                print(f"Collecting {series}...", flush=True)
                try:
                    total = collect_series(api, series, output_dir)
                    print(f"  {series}: {total} new candles", flush=True)
                except Exception as e:
                    print(f"  {series}: ERROR - {e}", flush=True)

        print(f"All series done. Sleeping {SLEEP_BETWEEN_RUNS}s before next run...", flush=True)
        time.sleep(SLEEP_BETWEEN_RUNS)


if __name__ == "__main__":
    main()
