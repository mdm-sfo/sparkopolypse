#!/usr/bin/env python3
"""Realistic backtest: simulates actual TP/SL/trailing-stop execution.

Reuses forecast signals from existing backtest CSVs (no GPU needed), then
simulates candle-by-candle position monitoring with ALL current trading rules:

  - NO-side only (skip UP signals)
  - 5c signal threshold
  - 20-95c price range filter
  - Per-series TP/SL levels (from timesfm_server.py)
  - Trailing stops (weather: +15c activate / 8c trail, S&P: +20c / 10c)
  - Entry slippage model (up to 3c, 20% of magnitude)
  - SL gap slippage (exit at candle price, not SL level)
  - 7% Kalshi fee on profit
  - Full remaining price history (not just 6 candles)

Usage:
    python realistic_backtest.py                    # all series
    python realistic_backtest.py KXHIGHNY           # single series
    python realistic_backtest.py --output report.md # save report
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
#  Current Trading Rules (mirrors live config)
# ═══════════════════════════════════════════════════════════════════

TPSL_DEFAULTS = {
    "KXHIGH": {
        (2, 5): {"tp": 20, "sl": 3},
        (5, 10): {"tp": 20, "sl": 3},
        (10, 20): {"tp": 20, "sl": 3},
        (20, 100): {"tp": 30, "sl": 2},
    },
    "KXINX": {
        (2, 5): {"tp": 30, "sl": 5},
        (5, 10): {"tp": 50, "sl": 2},
        (10, 20): {"tp": 50, "sl": 8},
        (20, 100): {"tp": 50, "sl": 10},
    },
    "KXBTC": {
        (2, 5): {"tp": 40, "sl": 8},
        (5, 10): {"tp": 50, "sl": 8},
        (10, 20): {"tp": 50, "sl": 10},
        (20, 40): {"tp": 50, "sl": 12},
        (40, 70): {"tp": 60, "sl": 15},
        (70, 100): {"tp": 70, "sl": 15},
    },
    "KXETH": {
        (2, 5): {"tp": 40, "sl": 10},
        (5, 10): {"tp": 50, "sl": 10},
        (10, 20): {"tp": 50, "sl": 12},
        (20, 40): {"tp": 50, "sl": 15},
        (40, 70): {"tp": 60, "sl": 15},
        (70, 100): {"tp": 70, "sl": 18},
    },
}
TPSL_FALLBACK = {"tp": 20, "sl": 5}

TRAILING_STOP_CONFIG = {
    "KXHIGH": {"activation": 15, "trail": 8},
    "KXINX": {"activation": 20, "trail": 10},
}

KALSHI_FEE_RATE = 0.07
SIGNAL_THRESHOLD = 5.0
PRICE_MIN = 20
PRICE_MAX = 95
MIN_HISTORY = 8
ENTRY_FRAC = 0.5


def get_tpsl(series_ticker: str, magnitude: float) -> dict:
    """Look up TP/SL for a series and signal magnitude."""
    for prefix, buckets in TPSL_DEFAULTS.items():
        if series_ticker.startswith(prefix):
            for (lo, hi), levels in buckets.items():
                if lo <= magnitude < hi:
                    return levels
            return TPSL_FALLBACK
    return TPSL_FALLBACK


def get_trailing(series_ticker: str):
    """Look up trailing stop config for a series (None if not configured)."""
    for prefix, cfg in TRAILING_STOP_CONFIG.items():
        if series_ticker.startswith(prefix):
            return cfg
    return None


# ═══════════════════════════════════════════════════════════════════
#  Trade Simulator
# ═══════════════════════════════════════════════════════════════════

def simulate_trade(entry_price, future_prices, tp_cents, sl_cents,
                   trail_config, entry_slip):
    """Simulate candle-by-candle NO-side position monitoring.

    All prices are YES prices. For NO holder:
      - Profit when YES drops (NO rises)
      - Loss when YES rises (NO drops)
      - TP: YES drops by tp_cents from entry
      - SL: YES rises by sl_cents from entry

    Returns dict with exit details and P&L.
    """
    best_yes_low = entry_price  # lowest YES = max NO profit
    trailing_activated = False

    for i, price in enumerate(future_prices):
        # Update best price seen (lowest YES = best for NO)
        if price < best_yes_low:
            best_yes_low = price

        # ── TP check: YES dropped enough ──
        drop = entry_price - price
        if drop >= tp_cents:
            # Fill at TP level (limit order)
            exit_price = entry_price - tp_cents
            pnl = tp_cents - entry_slip
            fee = pnl * KALSHI_FEE_RATE if pnl > 0 else 0
            return {
                "exit_price": exit_price,
                "exit_type": "TP",
                "pnl_gross": pnl,
                "pnl_net": pnl - fee,
                "candles_held": i + 1,
                "best_profit_seen": entry_price - best_yes_low,
            }

        # ── Trailing stop check ──
        if trail_config:
            profit_from_best = entry_price - best_yes_low
            if profit_from_best >= trail_config["activation"]:
                trailing_activated = True

            if trailing_activated:
                retracement = price - best_yes_low
                if retracement >= trail_config["trail"]:
                    exit_price = price
                    pnl = entry_price - exit_price - entry_slip
                    fee = pnl * KALSHI_FEE_RATE if pnl > 0 else 0
                    return {
                        "exit_price": exit_price,
                        "exit_type": "TRAIL",
                        "pnl_gross": pnl,
                        "pnl_net": pnl - fee,
                        "candles_held": i + 1,
                        "best_profit_seen": entry_price - best_yes_low,
                    }

        # ── SL check: YES rose enough ──
        rise = price - entry_price
        if rise >= sl_cents:
            # Exit at candle price (may gap past SL = realistic slippage)
            exit_price = price
            pnl = entry_price - exit_price - entry_slip
            return {
                "exit_price": exit_price,
                "exit_type": "SL",
                "pnl_gross": pnl,
                "pnl_net": pnl,  # no fee on losses
                "candles_held": i + 1,
                "best_profit_seen": entry_price - best_yes_low,
            }

    # ── Market expired — close at last candle ──
    exit_price = future_prices[-1]
    pnl = entry_price - exit_price - entry_slip
    fee = pnl * KALSHI_FEE_RATE if pnl > 0 else 0
    return {
        "exit_price": exit_price,
        "exit_type": "EXPIRE",
        "pnl_gross": pnl,
        "pnl_net": pnl - fee,
        "candles_held": len(future_prices),
        "best_profit_seen": entry_price - best_yes_low,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main Backtest
# ═══════════════════════════════════════════════════════════════════

def load_parquet_data(series_ticker: str) -> pd.DataFrame:
    """Load all parquet files for a series."""
    data_dir = Path(__file__).parent / "data" / series_ticker
    if not data_dir.exists():
        raise FileNotFoundError(f"No data dir: {data_dir}")
    frames = []
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["event"] = f.stem
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def run_realistic_backtest(series_ticker: str, backtest_csv: str = None):
    """Run realistic backtest for a single series.

    Uses forecast signals from existing backtest CSV, then simulates
    real execution on the full price path.
    """
    # Load forecast signals from simplified backtest
    if backtest_csv is None:
        backtest_csv = Path(__file__).parent / f"backtest_{series_ticker}_6h.csv"
    signals_df = pd.read_csv(backtest_csv)

    # Pre-filter: drop NaN magnitudes, keep only DOWN + above threshold
    signals_df = signals_df.dropna(subset=["forecast_magnitude", "entry_price"])
    eligible = signals_df[
        (signals_df["forecast_direction"] == "DOWN") &
        (signals_df["forecast_magnitude"] >= SIGNAL_THRESHOLD) &
        (signals_df["entry_price"] >= PRICE_MIN) &
        (signals_df["entry_price"] <= PRICE_MAX)
    ]

    print(f"\n{'='*70}")
    print(f"  REALISTIC BACKTEST: {series_ticker}")
    print(f"{'='*70}")
    print(f"  Signals from CSV: {len(signals_df)} total, "
          f"{len(eligible)} eligible (DOWN, ≥5c, 20-95c)")

    if eligible.empty:
        print(f"  No eligible trades. Skipping.")
        return pd.DataFrame()

    # Only load parquet data for tickers we actually need
    needed_tickers = set(eligible["ticker"].unique())
    print(f"  Loading price data for {len(needed_tickers)} tickers...", flush=True)

    data_dir = Path(__file__).parent / "data" / series_ticker
    ticker_prices = {}
    files_loaded = 0
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        # Only process tickers we need
        relevant = df[df["ticker"].isin(needed_tickers)]
        if relevant.empty:
            continue
        files_loaded += 1
        for ticker in relevant["ticker"].unique():
            market = relevant[relevant["ticker"] == ticker].sort_values("timestamp")
            ticker_prices[ticker] = market["price_close"].values.astype(float)
        # Stop early if we found all needed tickers
        if len(ticker_prices) >= len(needed_tickers):
            break

    print(f"  Found {len(ticker_prices)}/{len(needed_tickers)} tickers "
          f"across {files_loaded} parquet files", flush=True)

    trades = []
    skip_reasons = {"no_data": 0, "up_signal": 0, "below_threshold": 0,
                    "price_range": 0, "no_future": 0}

    for _, row in signals_df.iterrows():
        ticker = row["ticker"]
        forecast_dir = row["forecast_direction"]
        magnitude = row["forecast_magnitude"]
        csv_entry_price = row["entry_price"]
        history_len = int(row["history_len"])

        # ── Filter: NO-side only ──
        if forecast_dir != "DOWN":
            skip_reasons["up_signal"] += 1
            continue

        # ── Filter: 5c threshold (also skip NaN) ──
        if np.isnan(magnitude) or magnitude < SIGNAL_THRESHOLD:
            skip_reasons["below_threshold"] += 1
            continue

        # ── Get raw price data ──
        if ticker not in ticker_prices:
            skip_reasons["no_data"] += 1
            continue
        prices = ticker_prices[ticker]

        # ── Reconstruct entry index ──
        # Match the original backtest's entry point
        entry_idx = history_len
        if entry_idx >= len(prices):
            skip_reasons["no_future"] += 1
            continue

        entry_price = prices[entry_idx - 1]  # last known price before entry

        # Verify entry price matches CSV (sanity check)
        if abs(entry_price - csv_entry_price) > 1.0:
            # Entry index might be off by one, try to find it
            diffs = np.abs(prices - csv_entry_price)
            best_idx = np.argmin(diffs)
            if diffs[best_idx] < 1.0 and best_idx + 1 < len(prices):
                entry_idx = best_idx + 1
                entry_price = prices[best_idx]
            else:
                skip_reasons["no_data"] += 1
                continue

        # ── Filter: price range ──
        if entry_price < PRICE_MIN or entry_price > PRICE_MAX:
            skip_reasons["price_range"] += 1
            continue

        # All remaining candles after entry (not just 6!)
        future_prices = prices[entry_idx:]
        if len(future_prices) < 1:
            skip_reasons["no_future"] += 1
            continue

        # ── Get trade parameters ──
        tpsl = get_tpsl(series_ticker, magnitude)
        tp_cents = tpsl["tp"]
        sl_cents = tpsl["sl"]
        trail_config = get_trailing(series_ticker)

        # Entry slippage: up to 20% of magnitude, capped at 3c
        entry_slip = min(int(magnitude * 0.20), 3)

        # ── Simulate execution ──
        result = simulate_trade(
            entry_price=entry_price,
            future_prices=future_prices,
            tp_cents=tp_cents,
            sl_cents=sl_cents,
            trail_config=trail_config,
            entry_slip=entry_slip,
        )

        trades.append({
            "ticker": ticker,
            "event": row.get("event", ""),
            "entry_price": entry_price,
            "magnitude": magnitude,
            "tp_cents": tp_cents,
            "sl_cents": sl_cents,
            "has_trailing": trail_config is not None,
            "entry_slip": entry_slip,
            "exit_price": result["exit_price"],
            "exit_type": result["exit_type"],
            "pnl_gross": result["pnl_gross"],
            "pnl_net": result["pnl_net"],
            "candles_held": result["candles_held"],
            "best_profit_seen": result["best_profit_seen"],
            "future_candles": len(future_prices),
        })

    print(f"\n  Skipped: {sum(skip_reasons.values())} total")
    for reason, count in skip_reasons.items():
        if count > 0:
            print(f"    {reason}: {count}")

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════

def print_report(series: str, df: pd.DataFrame, file=None):
    """Print detailed report for one series."""
    def p(msg=""):
        print(msg, file=file)

    if df.empty:
        p(f"\n  {series}: No trades after filtering.")
        return

    n = len(df)
    wins = (df["pnl_net"] > 0).sum()
    losses = (df["pnl_net"] <= 0).sum()
    win_rate = wins / n * 100
    total_pnl = df["pnl_net"].sum()
    avg_pnl = df["pnl_net"].mean()
    avg_winner = df[df["pnl_net"] > 0]["pnl_net"].mean() if wins else 0
    avg_loser = df[df["pnl_net"] <= 0]["pnl_net"].mean() if losses else 0
    best = df["pnl_net"].max()
    worst = df["pnl_net"].min()
    avg_hold = df["candles_held"].mean()

    p(f"\n  Trades: {n}  (W:{wins} / L:{losses})")
    p(f"  Win rate:       {win_rate:.1f}%")
    p(f"  Total P&L:      {total_pnl:+.1f}c  (${total_pnl/100:+.2f} per contract)")
    p(f"  Avg P&L/trade:  {avg_pnl:+.2f}c")
    p(f"  Avg winner:     {avg_winner:+.2f}c")
    p(f"  Avg loser:      {avg_loser:+.2f}c")
    p(f"  Best trade:     {best:+.2f}c")
    p(f"  Worst trade:    {worst:+.2f}c")
    p(f"  Avg hold:       {avg_hold:.1f} candles")

    # Profit factor
    gross_wins = df[df["pnl_net"] > 0]["pnl_net"].sum() if wins else 0
    gross_losses = abs(df[df["pnl_net"] <= 0]["pnl_net"].sum()) if losses else 1
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")
    p(f"  Profit factor:  {pf:.2f}")

    # ── Exit type breakdown ──
    p(f"\n  Exit type breakdown:")
    for et in ["TP", "TRAIL", "SL", "EXPIRE"]:
        sub = df[df["exit_type"] == et]
        if sub.empty:
            continue
        sub_n = len(sub)
        sub_pnl = sub["pnl_net"].mean()
        sub_total = sub["pnl_net"].sum()
        p(f"    {et:>6}: {sub_n:>4} ({sub_n/n*100:>5.1f}%)  "
          f"avg {sub_pnl:+.2f}c  total {sub_total:+.1f}c")

    # ── SL slippage analysis ──
    sl_trades = df[df["exit_type"] == "SL"]
    if not sl_trades.empty:
        # Slippage = how much past SL level we exited
        sl_expected_loss = sl_trades["sl_cents"] + sl_trades["entry_slip"]
        sl_actual_loss = sl_trades["pnl_net"].abs()
        sl_slippage = sl_actual_loss - sl_expected_loss
        avg_slip = sl_slippage.mean()
        max_slip = sl_slippage.max()
        p(f"\n  SL slippage (gap past level):")
        p(f"    Avg: {avg_slip:+.1f}c   Max: {max_slip:+.1f}c")

    # ── TP capture analysis (trailing stop effectiveness) ──
    if df["has_trailing"].any():
        trail_trades = df[df["exit_type"] == "TRAIL"]
        if not trail_trades.empty:
            avg_best = trail_trades["best_profit_seen"].mean()
            avg_captured = trail_trades["pnl_gross"].mean()
            capture_pct = (avg_captured / avg_best * 100) if avg_best > 0 else 0
            p(f"\n  Trailing stop effectiveness:")
            p(f"    Avg best profit seen: {avg_best:.1f}c")
            p(f"    Avg captured:         {avg_captured:.1f}c")
            p(f"    Capture rate:         {capture_pct:.0f}%")

    # ── By magnitude bucket ──
    p(f"\n  By signal magnitude:")
    p(f"    {'Mag':>8}  {'Trades':>6}  {'WR%':>5}  {'AvgPnL':>8}  {'TotPnL':>9}  {'PF':>5}")
    p(f"    {'-'*48}")
    for lo, hi in [(5, 10), (10, 20), (20, 40), (40, 100)]:
        sub = df[(df["magnitude"] >= lo) & (df["magnitude"] < hi)]
        if sub.empty:
            continue
        sub_n = len(sub)
        sub_wr = (sub["pnl_net"] > 0).sum() / sub_n * 100
        sub_avg = sub["pnl_net"].mean()
        sub_tot = sub["pnl_net"].sum()
        gw = sub[sub["pnl_net"] > 0]["pnl_net"].sum() if (sub["pnl_net"] > 0).any() else 0
        gl = abs(sub[sub["pnl_net"] <= 0]["pnl_net"].sum()) if (sub["pnl_net"] <= 0).any() else 1
        sub_pf = gw / gl if gl > 0 else float("inf")
        p(f"    {lo:>2}-{hi:<4}c  {sub_n:>6}  {sub_wr:>4.0f}%  {sub_avg:>+7.2f}c  {sub_tot:>+8.1f}c  {sub_pf:>5.2f}")

    # ── Dollar P&L at different position sizes ──
    p(f"\n  Estimated $ P&L (total, assuming fixed contract size):")
    for contracts in [10, 30, 50]:
        dollar_pnl = total_pnl * contracts / 100
        p(f"    {contracts:>3} contracts/trade: ${dollar_pnl:+.2f}")
    p()


def comparison_table(all_results: dict, file=None):
    """Print cross-series comparison."""
    def p(msg=""):
        print(msg, file=file)

    p(f"\n{'='*70}")
    p(f"  CROSS-SERIES COMPARISON (Realistic Execution)")
    p(f"{'='*70}")
    p(f"  {'Series':<12} {'N':>5} {'WR%':>5} {'AvgPnL':>8} {'TotPnL':>9} "
      f"{'PF':>5} {'TP%':>5} {'SL%':>5} {'TRL%':>5} {'EXP%':>5}")
    p(f"  {'-'*72}")

    grand_total = 0
    grand_n = 0

    for series in sorted(all_results.keys()):
        df = all_results[series]
        if df.empty:
            p(f"  {series:<12}  (no trades)")
            continue

        n = len(df)
        wr = (df["pnl_net"] > 0).sum() / n * 100
        avg = df["pnl_net"].mean()
        tot = df["pnl_net"].sum()
        gw = df[df["pnl_net"] > 0]["pnl_net"].sum() if (df["pnl_net"] > 0).any() else 0
        gl = abs(df[df["pnl_net"] <= 0]["pnl_net"].sum()) if (df["pnl_net"] <= 0).any() else 1
        pf = gw / gl if gl > 0 else float("inf")

        tp_pct = len(df[df["exit_type"] == "TP"]) / n * 100
        sl_pct = len(df[df["exit_type"] == "SL"]) / n * 100
        trl_pct = len(df[df["exit_type"] == "TRAIL"]) / n * 100
        exp_pct = len(df[df["exit_type"] == "EXPIRE"]) / n * 100

        p(f"  {series:<12} {n:>5} {wr:>4.0f}% {avg:>+7.2f}c {tot:>+8.1f}c "
          f"{pf:>5.2f} {tp_pct:>4.0f}% {sl_pct:>4.0f}% {trl_pct:>4.0f}% {exp_pct:>4.0f}%")

        grand_total += tot
        grand_n += n

    p(f"  {'-'*72}")
    if grand_n > 0:
        grand_wr = sum((all_results[s]["pnl_net"] > 0).sum()
                       for s in all_results if not all_results[s].empty) / grand_n * 100
        grand_avg = grand_total / grand_n
        p(f"  {'TOTAL':<12} {grand_n:>5} {grand_wr:>4.0f}% {grand_avg:>+7.2f}c "
          f"{grand_total:>+8.1f}c")

    p(f"\n  Assumptions:")
    p(f"    - NO-side only, 5c signal threshold")
    p(f"    - Entry slippage: min(magnitude*0.20, 3)c")
    p(f"    - SL slippage: exit at candle price (may gap past SL)")
    p(f"    - Trailing stops: weather (+15c/8c), S&P (+20c/10c)")
    p(f"    - Full remaining price path (not capped at 6 candles)")
    p(f"    - 7% Kalshi fee on profit")
    p()


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    series_list = ["KXHIGHNY", "KXHIGHCHI", "KXINX", "KXBTC", "KXETH"]

    # Parse args
    output_path = None
    requested = []
    for arg in sys.argv[1:]:
        if arg.startswith("--output"):
            continue
        elif sys.argv[sys.argv.index(arg) - 1] == "--output" if arg != sys.argv[0] else False:
            output_path = arg
        elif arg in series_list:
            requested.append(arg)

    # Handle --output=path or --output path
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--output="):
            output_path = arg.split("=", 1)[1]
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]

    if requested:
        series_list = requested

    # Check which CSVs exist
    available = []
    for s in series_list:
        csv_path = Path(__file__).parent / f"backtest_{s}_6h.csv"
        if csv_path.exists():
            available.append(s)
        else:
            print(f"  Skipping {s}: no backtest CSV found at {csv_path}")

    if not available:
        print("No backtest CSVs found. Run the simplified backtest first.")
        sys.exit(1)

    # Run backtests
    all_results = {}
    for series in available:
        try:
            df = run_realistic_backtest(series)
            all_results[series] = df
            print_report(series, df)
        except Exception as e:
            print(f"\n  ERROR backtesting {series}: {e}")
            import traceback
            traceback.print_exc()
            all_results[series] = pd.DataFrame()

    # Cross-series comparison
    comparison_table(all_results)

    # Save to file if requested
    if output_path:
        with open(output_path, "w") as f:
            f.write(f"# Realistic Backtest Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("```\n")
            for series in available:
                if series in all_results:
                    print_report(series, all_results[series], file=f)
            comparison_table(all_results, file=f)
            f.write("```\n")

            # Also save raw trade data
            f.write("\n## Raw Trade Data\n\n")
            for series in available:
                df = all_results.get(series)
                if df is not None and not df.empty:
                    csv_out = output_path.replace(".md", f"_{series}.csv")
                    df.to_csv(csv_out, index=False)
                    f.write(f"- {series}: {csv_out} ({len(df)} trades)\n")

        print(f"\n  Report saved to: {output_path}")

    # Save combined CSV
    all_trades = []
    for series, df in all_results.items():
        if not df.empty:
            df = df.copy()
            df["series"] = series
            all_trades.append(df)
    if all_trades:
        combined = pd.concat(all_trades, ignore_index=True)
        csv_path = Path(__file__).parent / "realistic_backtest_all.csv"
        combined.to_csv(csv_path, index=False)
        print(f"  Combined CSV: {csv_path} ({len(combined)} trades)")


if __name__ == "__main__":
    main()
