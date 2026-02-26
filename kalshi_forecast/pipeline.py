"""End-to-end pipeline: collect live data -> forecast -> generate signals."""

import pandas as pd

from .api import KalshiAPI
from .collector import collect_open_markets
from .forecaster import Forecaster
from . import config


def run_forecast(
    series_ticker: str = "KXHIGHNY",
    horizon: int = 12,
    signal_threshold: float = 3.0,
) -> list[dict]:
    """Run forecast on all open markets for a series.

    Args:
        series_ticker: Kalshi series to forecast.
        horizon: Number of hourly candles ahead to forecast.
        signal_threshold: Minimum price movement (in cents) to flag as signal.

    Returns:
        List of signal dicts for markets with meaningful predicted movement.
    """
    print(f"=== Kalshi Forecast: {series_ticker} ===\n")

    # 1. Collect current market data
    print("Fetching open markets and current candles...")
    with KalshiAPI() as api:
        candles_df = collect_open_markets(api, series_ticker)

    if candles_df.empty:
        print("No open markets found.")
        return []

    tickers = candles_df["ticker"].unique()
    print(f"Found {len(tickers)} open markets with candle data.\n")

    # 2. Load model and forecast
    forecaster = Forecaster()
    forecaster.load_model()

    signals = []
    print(f"{'Ticker':<35} {'Now':>5} {'Fcst':>5} {'Dir':>5} {'Mag':>5}  Signal")
    print("-" * 85)

    for ticker in sorted(tickers):
        result = forecaster.forecast_market(candles_df, ticker, horizon=horizon)
        if "error" in result:
            continue

        is_signal = result["magnitude"] >= signal_threshold
        marker = " <-- SIGNAL" if is_signal else ""

        print(
            f"{result['ticker']:<35} "
            f"{result['current_price']:>5.0f} "
            f"{result['forecast_mean']:>5.1f} "
            f"{result['direction']:>5} "
            f"{result['magnitude']:>5.1f}"
            f"{marker}"
        )

        if is_signal:
            signals.append(result)

    print(f"\n{len(signals)} trade signals found (threshold: {signal_threshold} cents)")
    return signals
