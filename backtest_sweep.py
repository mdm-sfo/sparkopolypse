#!/usr/bin/env python3
"""Parameter sweep: find optimal SL/TP/trailing config per series.

Tests combinations of:
  - Trailing stop: ON vs OFF
  - SL overrides: current server values, 5c, 8c, 10c, 15c
  - Focused on weather (most data) but runs all series
"""

import sys
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════
#  Config (same as realistic_backtest.py)
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


def get_tpsl(series_ticker, magnitude, sl_override=None):
    """Get TP/SL, optionally overriding the SL value."""
    for prefix, buckets in TPSL_DEFAULTS.items():
        if series_ticker.startswith(prefix):
            for (lo, hi), levels in buckets.items():
                if lo <= magnitude < hi:
                    tp = levels["tp"]
                    sl = sl_override if sl_override is not None else levels["sl"]
                    return {"tp": tp, "sl": sl}
            sl = sl_override if sl_override is not None else TPSL_FALLBACK["sl"]
            return {"tp": TPSL_FALLBACK["tp"], "sl": sl}
    sl = sl_override if sl_override is not None else TPSL_FALLBACK["sl"]
    return {"tp": TPSL_FALLBACK["tp"], "sl": sl}


def simulate_trade(entry_price, future_prices, tp_cents, sl_cents,
                   trail_config, entry_slip):
    """Simulate candle-by-candle NO-side trade. Same as realistic_backtest.py."""
    best_yes_low = entry_price
    trailing_activated = False

    for i, price in enumerate(future_prices):
        if price < best_yes_low:
            best_yes_low = price

        # TP
        if entry_price - price >= tp_cents:
            exit_price = entry_price - tp_cents
            pnl = tp_cents - entry_slip
            fee = pnl * KALSHI_FEE_RATE if pnl > 0 else 0
            return {"exit_type": "TP", "pnl_net": pnl - fee,
                    "best_profit": entry_price - best_yes_low, "candles": i + 1}

        # Trailing stop
        if trail_config:
            if entry_price - best_yes_low >= trail_config["activation"]:
                trailing_activated = True
            if trailing_activated:
                if price - best_yes_low >= trail_config["trail"]:
                    pnl = entry_price - price - entry_slip
                    fee = pnl * KALSHI_FEE_RATE if pnl > 0 else 0
                    return {"exit_type": "TRAIL", "pnl_net": pnl - fee,
                            "best_profit": entry_price - best_yes_low, "candles": i + 1}

        # SL
        if price - entry_price >= sl_cents:
            pnl = entry_price - price - entry_slip
            return {"exit_type": "SL", "pnl_net": pnl,
                    "best_profit": entry_price - best_yes_low, "candles": i + 1}

    # Expire
    pnl = entry_price - future_prices[-1] - entry_slip
    fee = pnl * KALSHI_FEE_RATE if pnl > 0 else 0
    return {"exit_type": "EXPIRE", "pnl_net": pnl - fee,
            "best_profit": entry_price - best_yes_low, "candles": len(future_prices)}


def load_series_data(series_ticker, backtest_csv):
    """Load and pre-filter signals + price data for a series.
    Returns (eligible_signals_df, ticker_prices_dict).
    """
    csv_path = Path(__file__).parent / backtest_csv
    signals_df = pd.read_csv(csv_path)
    signals_df = signals_df.dropna(subset=["forecast_magnitude", "entry_price"])

    eligible = signals_df[
        (signals_df["forecast_direction"] == "DOWN") &
        (signals_df["forecast_magnitude"] >= SIGNAL_THRESHOLD) &
        (signals_df["entry_price"] >= PRICE_MIN) &
        (signals_df["entry_price"] <= PRICE_MAX)
    ].copy()

    if eligible.empty:
        return eligible, {}

    needed = set(eligible["ticker"].unique())
    data_dir = Path(__file__).parent / "data" / series_ticker
    ticker_prices = {}
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        relevant = df[df["ticker"].isin(needed)]
        if relevant.empty:
            continue
        for ticker in relevant["ticker"].unique():
            market = relevant[relevant["ticker"] == ticker].sort_values("timestamp")
            ticker_prices[ticker] = market["price_close"].values.astype(float)
        if len(ticker_prices) >= len(needed):
            break

    return eligible, ticker_prices


def run_scenario(eligible, ticker_prices, series_ticker,
                 sl_override=None, trailing_on=True):
    """Run one backtest scenario. Returns summary dict."""
    trail_config_base = TRAILING_STOP_CONFIG.get(
        next((p for p in TRAILING_STOP_CONFIG if series_ticker.startswith(p)), ""),
        None
    )
    # Look up trailing config properly
    trail_config = None
    if trailing_on:
        for prefix, cfg in TRAILING_STOP_CONFIG.items():
            if series_ticker.startswith(prefix):
                trail_config = cfg
                break

    trades = []
    for _, row in eligible.iterrows():
        ticker = row["ticker"]
        magnitude = row["forecast_magnitude"]
        csv_entry = row["entry_price"]
        history_len = int(row["history_len"])

        if ticker not in ticker_prices:
            continue

        prices = ticker_prices[ticker]
        entry_idx = history_len
        if entry_idx >= len(prices):
            continue

        entry_price = prices[entry_idx - 1]
        if abs(entry_price - csv_entry) > 1.0:
            diffs = np.abs(prices - csv_entry)
            best_idx = np.argmin(diffs)
            if diffs[best_idx] < 1.0 and best_idx + 1 < len(prices):
                entry_idx = best_idx + 1
                entry_price = prices[best_idx]
            else:
                continue

        future_prices = prices[entry_idx:]
        if len(future_prices) < 1:
            continue

        tpsl = get_tpsl(series_ticker, magnitude, sl_override)
        entry_slip = min(int(magnitude * 0.20), 3)

        result = simulate_trade(
            entry_price, future_prices,
            tpsl["tp"], tpsl["sl"],
            trail_config, entry_slip,
        )
        trades.append(result)

    if not trades:
        return {"n": 0, "wins": 0, "wr": 0, "total_pnl": 0, "avg_pnl": 0,
                "pf": 0, "tp_pct": 0, "sl_pct": 0, "trail_pct": 0,
                "exp_pct": 0, "avg_sl_loss": 0, "worst": 0}

    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["pnl_net"] > 0).sum()
    losses = (df["pnl_net"] <= 0).sum()
    total_pnl = df["pnl_net"].sum()
    gw = df[df["pnl_net"] > 0]["pnl_net"].sum() if wins else 0
    gl = abs(df[df["pnl_net"] <= 0]["pnl_net"].sum()) if losses else 1
    pf = gw / gl if gl > 0 else float("inf")

    sl_trades = df[df["exit_type"] == "SL"]
    avg_sl_loss = sl_trades["pnl_net"].mean() if not sl_trades.empty else 0

    return {
        "n": n,
        "wins": wins,
        "wr": wins / n * 100,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / n,
        "pf": pf,
        "tp_pct": len(df[df["exit_type"] == "TP"]) / n * 100,
        "sl_pct": len(df[df["exit_type"] == "SL"]) / n * 100,
        "trail_pct": len(df[df["exit_type"] == "TRAIL"]) / n * 100,
        "exp_pct": len(df[df["exit_type"] == "EXPIRE"]) / n * 100,
        "avg_sl_loss": avg_sl_loss,
        "worst": df["pnl_net"].min(),
        "avg_winner": df[df["pnl_net"] > 0]["pnl_net"].mean() if wins else 0,
    }


def main():
    output_path = Path(__file__).parent.parent / "wormhole" / "spark_deepdives" / "STUDY_parameter_sweep.md"

    # Series to sweep
    series_configs = [
        ("KXHIGHNY", "backtest_KXHIGHNY_6h.csv"),
        ("KXHIGHCHI", "backtest_KXHIGHCHI_6h.csv"),
        ("KXINX", "backtest_KXINX_6h.csv"),
        ("KXBTC", "backtest_KXBTC_6h.csv"),
        ("KXETH", "backtest_KXETH_6h.csv"),
    ]

    # Parameters to sweep
    sl_options = {
        "KXHIGH": [None, 5, 8, 10, 15, 20],  # None = use server defaults (2-3c)
        "KXINX": [None, 5, 8, 10, 15],
        "KXBTC": [None, 10, 15, 20],
        "KXETH": [None, 12, 15, 20],
    }
    trailing_options = [True, False]

    lines = []
    def p(msg=""):
        print(msg)
        lines.append(msg)

    p(f"# Parameter Sweep: Optimal SL / Trailing Stop Config")
    p(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    for series, csv_file in series_configs:
        p(f"\n{'='*75}")
        p(f"  {series}")
        p(f"{'='*75}")

        csv_path = Path(__file__).parent / csv_file
        if not csv_path.exists():
            p(f"  Skipped (no CSV)")
            continue

        # Load data once
        eligible, ticker_prices = load_series_data(series, csv_file)
        if eligible.empty:
            p(f"  No eligible trades")
            continue

        p(f"  {len(eligible)} eligible signals, {len(ticker_prices)} markets with data\n")

        # Determine which SL values to test
        prefix = next((k for k in sl_options if series.startswith(k)), None)
        sls = sl_options.get(prefix, [None, 5, 10, 15])

        # Header
        p(f"  {'Trail':>5} {'SL':>4}  {'N':>4} {'WR%':>5} {'AvgPnL':>8} {'TotPnL':>9} "
          f"{'PF':>5} {'TP%':>5} {'SL%':>5} {'TRL%':>5} {'EXP%':>5} "
          f"{'AvgSLloss':>10} {'Worst':>7} {'AvgWin':>7}")
        p(f"  {'-'*105}")

        best_pnl = -float("inf")
        best_config = ""

        for trailing_on in trailing_options:
            for sl in sls:
                result = run_scenario(eligible, ticker_prices, series,
                                      sl_override=sl, trailing_on=trailing_on)
                if result["n"] == 0:
                    continue

                trail_label = "ON" if trailing_on else "OFF"
                sl_label = f"{sl}c" if sl is not None else "svr"

                marker = ""
                if result["total_pnl"] > best_pnl:
                    best_pnl = result["total_pnl"]
                    best_config = f"trail={trail_label}, SL={sl_label}"

                p(f"  {trail_label:>5} {sl_label:>4}  {result['n']:>4} "
                  f"{result['wr']:>4.0f}% {result['avg_pnl']:>+7.2f}c "
                  f"{result['total_pnl']:>+8.1f}c {result['pf']:>5.2f} "
                  f"{result['tp_pct']:>4.0f}% {result['sl_pct']:>4.0f}% "
                  f"{result['trail_pct']:>4.0f}% {result['exp_pct']:>4.0f}% "
                  f"{result['avg_sl_loss']:>+9.1f}c {result['worst']:>+6.1f}c "
                  f"{result['avg_winner']:>+6.1f}c")

        p(f"\n  >>> BEST: {best_config}  (total P&L: {best_pnl:+.1f}c)")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    p(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
