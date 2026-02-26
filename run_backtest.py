#!/usr/bin/env python3
"""Run backtest of TimesFM forecasts against historical Kalshi data."""

import sys
from kalshi_forecast.backtest import run_backtest, print_report

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "KXHIGHNY"
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    results = run_backtest(series_ticker=series, horizon=horizon)
    print_report(results)

    # Save results to CSV for further analysis
    outfile = f"backtest_{series}_{horizon}h.csv"
    results.to_csv(outfile, index=False)
    print(f"Detailed results saved to {outfile}")
