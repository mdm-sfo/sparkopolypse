# Kalshi TimesFM Forecast Trading System

**Last updated:** 2026-02-26
**Status:** LIVE trading ($3/trade), distributed data collection across 3 nodes

---

## What This Is

An automated Kalshi prediction market trading system powered by Google's TimesFM 2.5 (time-series foundation model) running on the DGX Spark GPU. The system:

1. Collects historical candlestick data from Kalshi's public API
2. Forecasts price movements using TimesFM on GPU
3. Generates trade signals with direction, magnitude, and confidence
4. Places NO-side limit orders on Kalshi with magnitude-based slippage
5. Monitors positions and auto-closes at TP/SL targets via REST polling

---

## Architecture

```
┌─────────────────────────┐       Tailscale HTTP        ┌─────────────────────────┐
│     SPARK (DGX GPU)     │◄───────────────────────────►│     EC2 (Trading)       │
│  100.98.36.51           │                              │  ec2.buri-kelvin.ts.net │
│                         │  POST /signal                │                         │
│  TimesFM 2.5 200M      │  {markets, horizon}          │  timesfm_signal_trader  │
│  systemd: timesfm-server│                              │  systemd: timesfm-trader│
│  Port 8787              │  → {signals with TP/SL}      │                         │
│                         │                              │  Kalshi API (auth)      │
│  Data collection (node 0)│    Syncthing wormhole       │  Data collection (node 1)│
│  Backtesting            │◄───────────────────────────►│  Pushover notifications │
│  TP/SL optimization     │     ~/wormhole/ sync         │  RSA-PSS signing        │
└─────────────────────────┘                              └─────────────────────────┘

┌─────────────────────────┐
│     MAC MINI (M1)       │
│  Data collection (node 2)│
│  rsync → Spark every 10m│
└─────────────────────────┘
```

---

## Servers

### Spark (DGX)
- **Hardware:** NVIDIA DGX Spark, aarch64 ARM, GB10 GPU (121.7 GB VRAM), 3.7TB disk
- **Tailscale IP:** 100.98.36.51
- **SSH:** `ssh spark`
- **User:** matthewmurray
- **Passwordless sudo:** yes

### EC2
- **Instance:** t3.xlarge (4 vCPU, 16 GB RAM)
- **Tailscale:** ec2.buri-kelvin.ts.net
- **SSH:** `ssh ec2`
- **User:** ec2-user

### Mac Mini
- **Hardware:** M1
- **SSH:** `ssh mini`
- **Role:** Data collection node only

---

## Services

### On Spark (auto-start on boot):

| Service | Description | Status |
|---------|-------------|--------|
| `timesfm-server` | GPU forecast server on port 8787 | **enabled** |
| `syncthing@matthewmurray` | Wormhole sync with EC2 | **enabled** |

### On EC2 (auto-start on boot):

| Service | Description | Status |
|---------|-------------|--------|
| `timesfm-trader` | **LIVE** signal trader ($3/trade, 5c threshold) | **enabled** |

### Distributed Data Collection (background processes, not services):
- Spark: `fast_collector.py --node-id 0 --total-nodes 3`
- EC2: `fast_collector.py --node-id 1 --total-nodes 3`
- Mac Mini: `fast_collector.py --node-id 2 --total-nodes 3`
- Rsync cron on Spark: syncs EC2 and Mini data every 10 minutes

---

## Trading Strategy

### Rules (all enforced in `timesfm_signal_trader.py`):
- **NO-side only** — never buy YES contracts (DOWN predictions are 99%+ accurate)
- **Medium+ confidence** — skip low confidence signals
- **5c+ magnitude** — minimum predicted price move
- **20-95c price range** — skip extreme prices (no edge, SL triggers too easily)
- **10c max spread** — skip illiquid markets
- **3 trades per series per scan** — forces diversification
- **100 contract max position** — caps total exposure (~$60-80)
- **$3 per trade wager**

### Slippage (added 2026-02-26):
- `slippage = min(int(magnitude * 0.20), 3)` — up to 20% of signal magnitude, capped at 3c
- Limit orders placed at `ask + slippage` to improve fill rates on liquid markets
- Rationale: A +10c edge trade with -2c slippage (+8c net) beats an unfilled trade (0c)

### Resting Order Handling:
- If an order doesn't fill within 15 seconds, it's automatically cancelled
- Only filled orders get TP/SL monitors and trade logging

### TP/SL Monitoring:
- AutoCloseMonitor polls bid prices every 5 seconds via REST API
- Places sell orders when TP or SL price is hit
- 10-second grace period after entry before SL can trigger
- 2 consecutive polls below SL required before triggering (avoids noise)
- Pushover notifications on every trade, TP hit, and SL hit

---

## Optimized TP/SL Levels (from grid search on historical data)

| Asset | Take Profit | Stop Loss | Avg P&L | Win Rate |
|-------|------------|-----------|---------|----------|
| Weather | 20c | 3c | +2.53c | 34% |
| S&P 500 | 30c | 5c | +4.55c | 35% |
| Bitcoin | 50c | 10c | +6.25c | 60% |
| Ethereum | 50c | 15c | TBD | TBD |

These are baked into the TimesFM server (`timesfm_server.py` → `TPSL_DEFAULTS`).

---

## Backtest Results Summary

### Bitcoin (2,042 events, 1,160 trades) — LATEST
- Directional accuracy: 94.5%
- Total P&L: +158.2 cents
- Avg per trade: +3.04 cents
- DOWN: 99.2% accuracy (1,085 trades)
- UP: 26.7% accuracy (75 trades) — confirms NO-only strategy
- **5-10c signals: +10.31c avg** (sweet spot)

### Ethereum (947 events, 765 trades) — LATEST
- Directional accuracy: 95.8%
- Total P&L: +23.6 cents
- Avg per trade: +2.36 cents
- DOWN: 99.5% accuracy (733 trades)
- UP: 12.5% accuracy (32 trades)
- Thinner edge than BTC (~2.4c vs ~3.0c per trade)

### NYC Weather (4,450 trades)
- Directional accuracy: 85.4%
- Total P&L: +5.76 cents

### Chicago Weather (4,712 trades)
- Directional accuracy: 85.1%
- Total P&L: +1.46 cents

### LA Weather (1,312 trades)
- Directional accuracy: 81.7%
- Total P&L: +310.0 cents

### S&P 500 (3,551 trades)
- Directional accuracy: 96.7%
- Total P&L: +2.26 cents

### Key Insight
Headline accuracy (85-99%) is inflated — most contracts drift to zero so "predict DOWN" is almost always right. The real edge is in the **signal magnitude**: trades where TimesFM predicts 5+ cent moves are genuinely profitable.

---

## Data Collection Status

Distributed across 3 nodes using `fast_collector.py` with MD5-based event splitting. Each node handles events where `hash(event_ticker) % 3 == node_id`.

| Series | Events on Spark | Total Estimated | Status |
|--------|----------------|-----------------|--------|
| KXHIGHNY | 1,658 | ~1,658 | ✅ Complete |
| KXHIGHCHI | 1,646 | ~1,646 | ✅ Complete |
| KXHIGHLAX | 415 | ~415 | ✅ Complete |
| KXINX | 973 | ~973 | ✅ Complete |
| KXBTC | 2,042 | ~7,340 | 🔄 Collecting (3 nodes) |
| KXETH | 947 | ~2,500+ | 🔄 Collecting (3 nodes) |

### Sync:
```bash
# Cron on Spark (every 10 min):
*/10 * * * * /home/matthewmurray/kalshi-forecast/sync_collectors.sh
# Syncs from EC2 and Mini → Spark data/
```

---

## Key File Locations

### Spark: ~/kalshi-forecast/

```
kalshi-forecast/
├── kalshi_forecast/           # Python package
│   ├── api.py                 # Kalshi public API client (unauthenticated)
│   ├── collector.py           # Historical data collection → parquet
│   ├── forecaster.py          # TimesFM model wrapper
│   ├── pipeline.py            # Live signal generation
│   ├── config.py              # Series, API URLs, model settings
│   └── backtest.py            # Backtesting engine
│
├── timesfm_server.py          # HTTP forecast server (runs as systemd service)
├── timesfm-server.service     # systemd unit file
├── fast_collector.py          # Distributed collector (deployed to all 3 nodes)
├── sync_collectors.sh         # Rsync from EC2/Mini (cron every 10 min)
├── analyze_tpsl.py            # TP/SL grid search optimization
├── run_collect.py             # CLI: data collection
├── run_forecast.py            # CLI: live forecasting
├── run_backtest.py            # CLI: backtesting
│
├── data/                      # Parquet data (synced from all nodes)
│   ├── KXHIGHNY/              # NYC weather
│   ├── KXHIGHCHI/             # Chicago weather
│   ├── KXHIGHLAX/             # LA weather
│   ├── KXINX/                 # S&P 500
│   ├── KXBTC/                 # Bitcoin
│   └── KXETH/                 # Ethereum
│
├── backtest_*_6h.csv          # Backtest results per series
└── tpsl_grid_*.csv            # TP/SL optimization results
```

### EC2: ~/projects/Kalshi/

```
Kalshi/
├── kalshi_trading_bot.py      # Core: KalshiAuth, KalshiRestClient, KalshiWebSocket
├── timesfm_signal_trader.py   # LIVE signal trader (systemd service)
├── paper_report.py            # Paper trading P&L report
├── fast_collector.py          # Distributed collector (node 1)
├── bloomberg.py               # Bloomberg-style terminal (MPT)
├── trigger.py                 # Phone-optimized rapid execution
├── secrets/
│   └── kalshi_private_key.pem # RSA private key for API auth
└── data/
    └── timesfm_trades/        # Trade logs (JSON per day)
```

### EC2: ~/.config/kalshi.env
Contains: KALSHI_API_KEY, KALSHI_PRIVATE_KEY_PATH, PUSHOVER_USER, PUSHOVER_TOKEN, TIMESFM_REMOTE_URL, TIMESFM_REMOTE_TOKEN

---

## Kalshi Account
- **Balance:** ~$182 (as of 2026-02-26)
- **Mode:** LIVE trading, $3/trade
- **API Key:** in ~/.config/kalshi.env on EC2

---

## How to Run Things

### Check live trader:
```bash
ssh ec2 "sudo systemctl status timesfm-trader"
ssh ec2 "sudo journalctl -u timesfm-trader -f"    # live logs
ssh ec2 "python3 ~/projects/Kalshi/paper_report.py"  # P&L report
```

### Run a backtest (on Spark):
```bash
source ~/timesfm/.venv/bin/activate
cd ~/kalshi-forecast
python3 -u run_backtest.py KXBTC 6
python3 -u run_backtest.py KXETH 6
```

### Run TP/SL optimization:
```bash
python3 -u analyze_tpsl.py KXBTC 6
```

### Check services:
```bash
# Spark
curl http://localhost:8787/health
sudo systemctl status timesfm-server

# EC2
ssh ec2 "sudo systemctl status timesfm-trader"
```

---

## After Reboot Checklist

### Spark:
1. Services auto-start — verify:
   ```bash
   sudo systemctl status timesfm-server
   sudo systemctl status syncthing@matthewmurray
   curl http://localhost:8787/health
   ```
2. Restart data collector (not a service):
   ```bash
   source ~/timesfm/.venv/bin/activate
   nohup python3 ~/kalshi-forecast/fast_collector.py --node-id 0 --total-nodes 3 --series KXBTC KXETH --loop > /tmp/collector.log 2>&1 &
   ```

### EC2:
1. Trader auto-starts — verify:
   ```bash
   sudo systemctl status timesfm-trader
   ```
2. Restart data collector (not a service):
   ```bash
   nohup python3 ~/fast_collector.py --node-id 1 --total-nodes 3 --series KXBTC KXETH --output ~/kalshi-data --loop > /tmp/collector.log 2>&1 &
   ```

---

## Pending / Future Work

1. **Monitor live trading performance** — accumulate data, compare to backtest expectations
2. **Revisit TP levels** — TP=99c on cheap NO contracts may be too aggressive (trying for 4x when 2x would be great). Revisit with more live data.
3. **ETH TP/SL optimization** — wait for more data, then re-run grid search
4. **Finish distributed data collection** — BTC has ~7,340 events, only ~2,042 collected so far
5. **Scale up wagers** — if live results confirm edge, increase from $3 to $5-10
6. **Mac Mini migration** — move TimesFM server from Spark to Mini (future, no rush)
7. **Bloomberg terminal integration** — add TimesFM signal overlay to bloomberg.py
