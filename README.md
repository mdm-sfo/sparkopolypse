# Sparkopolypse

Automated [Kalshi](https://kalshi.com) prediction market trading system powered by Google's [TimesFM 2.5](https://github.com/google-research/timesfm) time-series foundation model running on an NVIDIA DGX Spark GPU.

## How It Works

1. **Collect** historical candlestick data from Kalshi's public API across 6 market series (weather, S&P 500, BTC, ETH)
2. **Forecast** price movements using TimesFM 2.5 200M on GPU
3. **Trade** NO-side contracts when predicted downward magnitude exceeds 5 cents
4. **Monitor** positions with auto TP/SL exit via REST polling

## Architecture

```
Spark (DGX GPU)              EC2 (Trading)              Mac Mini (M1)
┌──────────────┐  HTTP/Tailscale  ┌──────────────┐     ┌──────────────┐
│ TimesFM 2.5  │◄───────────────►│ Signal Trader │     │              │
│ Forecast API │  POST /signal   │ Kalshi Orders │     │              │
│ Port 8787    │                 │ TP/SL Monitor │     │              │
│              │                 │ Pushover Alerts│    │              │
│ Collector    │  rsync (cron)   │ Collector     │     │ Collector    │
│ (node 0)    │◄────────────────│ (node 1)      │     │ (node 2)     │
│ Backtesting  │◄────────────────┼───────────────┘     └──────┘
└──────────────┘                  rsync (cron)
```

Data collection is distributed across 3 nodes using MD5 hash partitioning to parallelize against per-IP rate limits.

## Strategy

- **NO-only** — DOWN predictions are 99%+ accurate; UP predictions are garbage
- **Medium+ confidence, 5c+ magnitude** — filters out noise
- **20-95c price range** — no edge on extreme prices
- **Magnitude-based slippage** — `min(magnitude * 0.20, 3c)` for better fill rates
- **Optimized TP/SL** from grid search on historical data (e.g. BTC: TP=50c, SL=10c)

## Backtest Results

| Series | Events | Trades | Accuracy | Total P&L | Avg/Trade |
|--------|--------|--------|----------|-----------|-----------|
| BTC | 2,042 | 1,160 | 94.5% | +158.2c | +3.04c |
| ETH | 947 | 765 | 95.8% | +23.6c | +2.36c |
| S&P 500 | 973 | 3,551 | 96.7% | +2.3c | +0.06c |
| NYC Weather | 1,658 | 4,450 | 85.4% | +5.8c | +0.13c |

Headline accuracy is inflated (most contracts drift to 0). Real edge is in the 5-10c signal magnitude bucket.

## Quick Start

### Forecast Server (GPU machine)
```bash
source ~/timesfm/.venv/bin/activate
TIMESFM_REMOTE_TOKEN=your_token HF_TOKEN=your_hf_token python timesfm_server.py
```

### Backtest
```bash
python run_backtest.py KXBTC 6    # series, horizon
python analyze_tpsl.py KXBTC 6    # TP/SL grid search
```

### Data Collection
```bash
python fast_collector.py --node-id 0 --total-nodes 3 --series KXBTC KXETH --loop
```

## Project Structure

```
kalshi_forecast/        # Core Python package
  api.py                #   Kalshi public API client
  collector.py          #   Historical data collection
  forecaster.py         #   TimesFM model wrapper
  backtest.py           #   Backtesting engine
  pipeline.py           #   Live signal generation
  config.py             #   Series config, API URLs
timesfm_server.py       # HTTP forecast server (systemd)
fast_collector.py       # Distributed data collector
sync_collectors.sh      # Rsync from remote nodes (cron)
analyze_tpsl.py         # TP/SL grid search optimization
run_backtest.py         # CLI: run backtests
run_forecast.py         # CLI: live forecasting
run_collect.py          # CLI: data collection
```

The trading bot (`timesfm_signal_trader.py`) lives separately on EC2 with authenticated Kalshi API access.

## Requirements

- Python 3.12+
- CUDA-capable GPU (TimesFM inference)
- `timesfm`, `torch`, `pandas`, `httpx`, `numpy`
- Kalshi API credentials (for live trading)
