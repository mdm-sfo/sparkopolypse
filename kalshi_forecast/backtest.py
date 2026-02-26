"""Backtest TimesFM forecasts against historical Kalshi weather market data."""

from pathlib import Path

import numpy as np
import pandas as pd

from .forecaster import Forecaster
from . import config


KALSHI_FEE_RATE = 0.07  # 7% fee on profit


def load_historical_data(series_ticker: str) -> pd.DataFrame:
    """Load all parquet files for a series into one DataFrame."""
    data_dir = config.DATA_DIR / series_ticker
    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory for {series_ticker}")

    frames = []
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["event"] = f.stem
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    return pd.concat(frames, ignore_index=True)


def run_backtest(
    series_ticker: str = "KXHIGHNY",
    horizon: int = 6,
    min_history: int = 8,
    entry_frac: float = 0.5,
    signal_threshold: float = 2.0,
) -> pd.DataFrame:
    """Run backtest across all historical markets for a series.

    For each market, we simulate entering at `entry_frac` of the way through
    its price history, forecast `horizon` steps ahead, and compare against
    what actually happened.

    Args:
        series_ticker: Which series to backtest.
        horizon: How many candles ahead to forecast.
        min_history: Minimum candles before entry point (need enough context).
        entry_frac: Fraction of the price series to use as "known" history.
            0.5 = enter halfway through the market's life.
        signal_threshold: Minimum predicted magnitude (cents) to count as a trade.

    Returns:
        DataFrame with one row per trade: ticker, direction, predicted/actual
        move, profit, etc.
    """
    print(f"=== Backtest: {series_ticker} ===")
    print(f"  horizon={horizon}, entry_frac={entry_frac}, threshold={signal_threshold}\n")

    data = load_historical_data(series_ticker)
    tickers = data["ticker"].unique()
    print(f"Loaded {len(tickers)} markets across {data['event'].nunique()} events\n")

    forecaster = Forecaster()
    forecaster.load_model()

    trades = []
    skipped = 0

    for i, ticker in enumerate(sorted(tickers)):
        market = data[data["ticker"] == ticker].sort_values("timestamp")
        prices = market["price_close"].values.astype(float)

        # Need enough data to split into history + future
        if len(prices) < min_history + horizon:
            skipped += 1
            continue

        # Split: use first entry_frac as history, measure next `horizon` candles
        entry_idx = max(min_history, int(len(prices) * entry_frac))
        if entry_idx + horizon > len(prices):
            entry_idx = len(prices) - horizon
        if entry_idx < min_history:
            skipped += 1
            continue

        history = prices[:entry_idx]
        future = prices[entry_idx:entry_idx + horizon]

        # Skip markets with no price movement (illiquid / all zeros)
        if np.all(history == 0) or np.all(np.isnan(history)) or np.std(history) == 0:
            skipped += 1
            continue

        entry_price = history[-1]
        actual_mean = float(np.mean(future))
        actual_end = float(future[-1])
        actual_direction = "UP" if actual_mean > entry_price else "DOWN"
        actual_move = actual_mean - entry_price

        # Forecast
        try:
            results = forecaster.forecast([history], horizon=horizon)
        except Exception:
            skipped += 1
            continue
        point = results[0]["point"]
        forecast_mean = float(np.mean(point))
        forecast_direction = "UP" if forecast_mean > entry_price else "DOWN"
        forecast_move = forecast_mean - entry_price

        magnitude = abs(forecast_move)
        if magnitude < signal_threshold:
            skipped += 1
            continue

        # Did we get the direction right?
        correct = forecast_direction == actual_direction

        # P&L calculation (in cents)
        # If correct direction: profit = |actual_move|, minus fees
        # If wrong direction: loss = |actual_move|
        if correct:
            gross_profit = abs(actual_move)
            fee = gross_profit * KALSHI_FEE_RATE
            pnl = gross_profit - fee
        else:
            pnl = -abs(actual_move)

        trades.append({
            "ticker": ticker,
            "event": market["event"].iloc[0],
            "entry_price": entry_price,
            "forecast_mean": forecast_mean,
            "forecast_direction": forecast_direction,
            "forecast_magnitude": magnitude,
            "actual_mean": actual_mean,
            "actual_end": actual_end,
            "actual_direction": actual_direction,
            "actual_move": actual_move,
            "correct": correct,
            "pnl": pnl,
            "history_len": len(history),
            "future_len": len(future),
        })

        if (i + 1) % 200 == 0:
            n = len(trades)
            if n > 0:
                acc = sum(t["correct"] for t in trades) / n * 100
                print(f"  processed {i+1}/{len(tickers)} markets, {n} trades, accuracy so far: {acc:.1f}%")

    results_df = pd.DataFrame(trades)
    print(f"\nSkipped {skipped} markets (insufficient data or below threshold)")

    return results_df


def print_report(results: pd.DataFrame) -> None:
    """Print a summary report of backtest results."""
    if results.empty:
        print("No trades to report.")
        return

    n = len(results)
    correct = results["correct"].sum()
    accuracy = correct / n * 100
    total_pnl = results["pnl"].sum()
    avg_pnl = results["pnl"].mean()
    win_rate = (results["pnl"] > 0).sum() / n * 100
    avg_winner = results[results["pnl"] > 0]["pnl"].mean() if (results["pnl"] > 0).any() else 0
    avg_loser = results[results["pnl"] <= 0]["pnl"].mean() if (results["pnl"] <= 0).any() else 0

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total trades:         {n}")
    print(f"  Directional accuracy: {accuracy:.1f}% ({correct}/{n})")
    print(f"  Win rate (after fees): {win_rate:.1f}%")
    print(f"  Total P&L:            {total_pnl:+.1f} cents")
    print(f"  Avg P&L per trade:    {avg_pnl:+.2f} cents")
    print(f"  Avg winner:           {avg_winner:+.2f} cents")
    print(f"  Avg loser:            {avg_loser:+.2f} cents")
    print(f"{'='*60}")

    # Breakdown by signal strength
    print(f"\n  By signal magnitude:")
    bins = [(2, 5), (5, 10), (10, 20), (20, 100)]
    for lo, hi in bins:
        subset = results[(results["forecast_magnitude"] >= lo) & (results["forecast_magnitude"] < hi)]
        if subset.empty:
            continue
        sub_n = len(subset)
        sub_acc = subset["correct"].sum() / sub_n * 100
        sub_pnl = subset["pnl"].mean()
        print(f"    {lo}-{hi} cents: {sub_n} trades, {sub_acc:.1f}% accuracy, avg P&L {sub_pnl:+.2f}")

    # Breakdown by direction
    print(f"\n  By predicted direction:")
    for direction in ["UP", "DOWN"]:
        subset = results[results["forecast_direction"] == direction]
        if subset.empty:
            continue
        sub_n = len(subset)
        sub_acc = subset["correct"].sum() / sub_n * 100
        sub_pnl = subset["pnl"].mean()
        print(f"    {direction}: {sub_n} trades, {sub_acc:.1f}% accuracy, avg P&L {sub_pnl:+.2f}")

    print()
