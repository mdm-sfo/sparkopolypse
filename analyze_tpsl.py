#!/usr/bin/env python3
"""Analyze optimal Take Profit / Stop Loss levels using full price paths.

Unlike the basic backtest which only stores actual_move (mean), this script
walks through the future price candles one-by-one to simulate when TP or SL
would trigger.
"""

import sys
import numpy as np
import pandas as pd

from kalshi_forecast.backtest import load_historical_data
from kalshi_forecast.forecaster import Forecaster


KALSHI_FEE_RATE = 0.07


def simulate_tpsl(
    entry_price: float,
    direction: str,
    future_prices: np.ndarray,
    tp_cents: float,
    sl_cents: float,
) -> dict:
    """Simulate a trade with TP/SL on a candle-by-candle price path.

    Args:
        entry_price: Price when we enter the trade.
        direction: "UP" (we bought Yes) or "DOWN" (we bought No).
        future_prices: Array of future close prices, one per candle.
        tp_cents: Take profit distance in cents from entry.
        sl_cents: Stop loss distance in cents from entry.

    Returns:
        Dict with exit_reason ("tp", "sl", "expire"), exit_price, pnl, candles_held.
    """
    for i, price in enumerate(future_prices):
        if np.isnan(price):
            continue  # skip NaN candles

        if direction == "UP":
            move = price - entry_price
        else:
            # If we bet DOWN, we profit when price drops
            move = entry_price - price

        # Check TP first (optimistic), then SL
        if move >= tp_cents:
            gross = tp_cents
            fee = gross * KALSHI_FEE_RATE
            return {
                "exit_reason": "tp",
                "exit_price": price,
                "pnl": gross - fee,
                "candles_held": i + 1,
            }
        if move <= -sl_cents:
            return {
                "exit_reason": "sl",
                "exit_price": price,
                "pnl": -sl_cents,
                "candles_held": i + 1,
            }

    # Expired — held to end. Use last non-NaN price.
    valid = future_prices[~np.isnan(future_prices)]
    if len(valid) == 0:
        return {"exit_reason": "expire", "exit_price": entry_price, "pnl": 0.0, "candles_held": len(future_prices)}

    last_price = valid[-1]
    if direction == "UP":
        move = last_price - entry_price
    else:
        move = entry_price - last_price

    if move > 0:
        fee = move * KALSHI_FEE_RATE
        pnl = move - fee
    else:
        pnl = move  # loss (negative)

    return {
        "exit_reason": "expire",
        "exit_price": float(future_prices[-1]),
        "pnl": pnl,
        "candles_held": len(future_prices),
    }


def run_tpsl_analysis(
    series_ticker: str = "KXHIGHNY",
    horizon: int = 6,
    min_history: int = 8,
    entry_frac: float = 0.5,
    signal_threshold: float = 2.0,
):
    """Run the full TP/SL analysis: first collect trades with full price paths,
    then grid-search TP/SL combinations."""

    print(f"=== TP/SL Analysis: {series_ticker} ===")
    print(f"  horizon={horizon}, entry_frac={entry_frac}, threshold={signal_threshold}\n")

    data = load_historical_data(series_ticker)
    tickers = data["ticker"].unique()
    print(f"Loaded {len(tickers)} markets\n")

    forecaster = Forecaster()
    forecaster.load_model()

    # Collect trades WITH full future price paths
    trades = []
    skipped = 0

    for i, ticker in enumerate(sorted(tickers)):
        market = data[data["ticker"] == ticker].sort_values("timestamp")
        prices = market["price_close"].values.astype(float)

        if len(prices) < min_history + horizon:
            skipped += 1
            continue

        entry_idx = max(min_history, int(len(prices) * entry_frac))
        if entry_idx + horizon > len(prices):
            entry_idx = len(prices) - horizon
        if entry_idx < min_history:
            skipped += 1
            continue

        history = prices[:entry_idx]
        future = prices[entry_idx:entry_idx + horizon]

        if np.all(history == 0) or np.all(np.isnan(history)) or np.std(history) == 0:
            skipped += 1
            continue

        # Skip if future has NaN or all zeros (can't simulate TP/SL)
        if np.any(np.isnan(future)) or np.all(future == 0):
            skipped += 1
            continue

        entry_price = history[-1]
        if np.isnan(entry_price) or entry_price == 0:
            skipped += 1
            continue

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

        trades.append({
            "ticker": ticker,
            "entry_price": entry_price,
            "forecast_direction": forecast_direction,
            "forecast_magnitude": magnitude,
            "future_prices": future,  # the key addition
        })

        if (i + 1) % 200 == 0 and trades:
            print(f"  processed {i+1}/{len(tickers)}, {len(trades)} trades so far")

    print(f"\nCollected {len(trades)} trades (skipped {skipped})")

    if not trades:
        print("No trades to analyze.")
        return

    # --- Part 1: Actual price move statistics ---
    print(f"\n{'='*60}")
    print("  ACTUAL PRICE MOVES (what happens after entry)")
    print(f"{'='*60}")

    all_moves = []
    for t in trades:
        fp = t["future_prices"]
        ep = t["entry_price"]
        d = t["forecast_direction"]
        # Directional move: positive = favorable
        if d == "UP":
            moves = fp - ep
        else:
            moves = ep - fp
        max_favorable = float(np.max(moves))
        max_adverse = float(np.min(moves))
        end_move = float(moves[-1])
        all_moves.append({
            "max_favorable": max_favorable,
            "max_adverse": max_adverse,
            "end_move": end_move,
            "magnitude": t["forecast_magnitude"],
        })

    moves_df = pd.DataFrame(all_moves)
    print(f"\n  Max favorable excursion (best unrealized gain):")
    print(f"    Mean:   {moves_df['max_favorable'].mean():+.1f}c")
    print(f"    Median: {moves_df['max_favorable'].median():+.1f}c")
    print(f"    P25:    {moves_df['max_favorable'].quantile(0.25):+.1f}c")
    print(f"    P75:    {moves_df['max_favorable'].quantile(0.75):+.1f}c")

    print(f"\n  Max adverse excursion (worst unrealized loss):")
    print(f"    Mean:   {moves_df['max_adverse'].mean():+.1f}c")
    print(f"    Median: {moves_df['max_adverse'].median():+.1f}c")
    print(f"    P25:    {moves_df['max_adverse'].quantile(0.25):+.1f}c")
    print(f"    P75:    {moves_df['max_adverse'].quantile(0.75):+.1f}c")

    print(f"\n  End-of-horizon move:")
    print(f"    Mean:   {moves_df['end_move'].mean():+.1f}c")
    print(f"    Median: {moves_df['end_move'].median():+.1f}c")
    winners = moves_df[moves_df["end_move"] > 0]
    losers = moves_df[moves_df["end_move"] <= 0]
    print(f"    Winners: {len(winners)} ({len(winners)/len(moves_df)*100:.1f}%)")
    print(f"    Losers:  {len(losers)} ({len(losers)/len(moves_df)*100:.1f}%)")

    # --- Part 2: Grid search TP/SL ---
    print(f"\n{'='*60}")
    print("  TP/SL GRID SEARCH")
    print(f"{'='*60}")

    tp_values = [3, 5, 8, 10, 15, 20, 30, 50]
    sl_values = [2, 3, 5, 8, 10, 15, 20]

    results_grid = []
    for tp in tp_values:
        for sl in sl_values:
            pnls = []
            tp_hits = 0
            sl_hits = 0
            expires = 0
            for t in trades:
                sim = simulate_tpsl(
                    entry_price=t["entry_price"],
                    direction=t["forecast_direction"],
                    future_prices=t["future_prices"],
                    tp_cents=tp,
                    sl_cents=sl,
                )
                pnls.append(sim["pnl"])
                if sim["exit_reason"] == "tp":
                    tp_hits += 1
                elif sim["exit_reason"] == "sl":
                    sl_hits += 1
                else:
                    expires += 1

            n = len(pnls)
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / n
            win_rate = sum(1 for p in pnls if p > 0) / n * 100

            results_grid.append({
                "tp": tp,
                "sl": sl,
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "tp_rate": tp_hits / n * 100,
                "sl_rate": sl_hits / n * 100,
                "expire_rate": expires / n * 100,
                "n_trades": n,
            })

    grid_df = pd.DataFrame(results_grid)
    nan_count = grid_df["avg_pnl"].isna().sum()
    if nan_count > 0:
        print(f"\n  WARNING: {nan_count}/{len(grid_df)} combos have NaN P&L")

    # Show top 15 by avg P&L
    print(f"\n  Top 15 TP/SL combos by avg P&L per trade:")
    print(f"  {'TP':>4} {'SL':>4} {'AvgPnL':>8} {'TotalPnL':>10} {'WinRate':>8} {'TP%':>6} {'SL%':>6} {'Exp%':>6}")
    print(f"  {'-'*4} {'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    top = grid_df.nlargest(15, "avg_pnl")
    for _, row in top.iterrows():
        print(f"  {row['tp']:4.0f} {row['sl']:4.0f} {row['avg_pnl']:+8.2f} {row['total_pnl']:+10.1f} {row['win_rate']:7.1f}% {row['tp_rate']:5.1f}% {row['sl_rate']:5.1f}% {row['expire_rate']:5.1f}%")

    # Show top by total P&L
    print(f"\n  Top 10 by total P&L:")
    top_total = grid_df.nlargest(10, "total_pnl")
    for _, row in top_total.iterrows():
        print(f"  TP={row['tp']:2.0f}c SL={row['sl']:2.0f}c → total {row['total_pnl']:+.1f}c over {row['n_trades']:.0f} trades, win rate {row['win_rate']:.1f}%")

    # Recommended settings
    valid_grid = grid_df.dropna(subset=["avg_pnl"])
    if valid_grid.empty:
        print("\n  ERROR: All P&L values are NaN - check data quality")
        return
    best = valid_grid.loc[valid_grid["avg_pnl"].idxmax()]
    print(f"\n  RECOMMENDED: TP={best['tp']:.0f}c, SL={best['sl']:.0f}c")
    print(f"    Avg P&L: {best['avg_pnl']:+.2f}c/trade, Win rate: {best['win_rate']:.1f}%")
    print(f"    TP hit: {best['tp_rate']:.1f}%, SL hit: {best['sl_rate']:.1f}%, Expired: {best['expire_rate']:.1f}%")

    # --- Part 3: Breakdown by signal magnitude ---
    print(f"\n{'='*60}")
    print("  BEST TP/SL BY SIGNAL MAGNITUDE")
    print(f"{'='*60}")

    mag_bins = [(2, 5), (5, 10), (10, 20), (20, 100)]
    for lo, hi in mag_bins:
        subset = [t for t in trades if lo <= t["forecast_magnitude"] < hi]
        if len(subset) < 5:
            continue

        best_combo = None
        best_avg = -999
        for tp in tp_values:
            for sl in sl_values:
                pnls = []
                for t in subset:
                    sim = simulate_tpsl(
                        entry_price=t["entry_price"],
                        direction=t["forecast_direction"],
                        future_prices=t["future_prices"],
                        tp_cents=tp,
                        sl_cents=sl,
                    )
                    pnls.append(sim["pnl"])
                avg = sum(pnls) / len(pnls)
                if avg > best_avg:
                    best_avg = avg
                    best_combo = (tp, sl, avg, sum(1 for p in pnls if p > 0) / len(pnls) * 100)

        if best_combo:
            tp, sl, avg, wr = best_combo
            print(f"\n  Signal {lo}-{hi}c ({len(subset)} trades):")
            print(f"    Best: TP={tp}c, SL={sl}c → avg {avg:+.2f}c/trade, win rate {wr:.1f}%")

    # Save grid results
    outfile = f"tpsl_grid_{series_ticker}.csv"
    grid_df.to_csv(outfile, index=False)
    print(f"\nGrid results saved to {outfile}")


if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "KXHIGHNY"
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    run_tpsl_analysis(series_ticker=series, horizon=horizon)
