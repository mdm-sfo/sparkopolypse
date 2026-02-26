#!/usr/bin/env python3
"""Collect historical Kalshi weather market data."""

from kalshi_forecast.collector import collect_all

if __name__ == "__main__":
    collect_all()
