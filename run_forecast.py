#!/usr/bin/env python3
"""Run TimesFM forecast on open Kalshi weather markets."""

import sys
from kalshi_forecast.pipeline import run_forecast

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "KXHIGHNY"
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    signals = run_forecast(series_ticker=series, horizon=horizon)
