#!/usr/bin/env python3
"""TimesFM forecast server for Kalshi signal generation.

Serves TimesFM forecasts over HTTP. Designed to run on Spark (GPU) and be
called by the signal trader on EC2.

Endpoints:
    GET  /health              → {"ok": true}
    POST /forecast            → raw forecasts (backward compatible)
    POST /signal              → enriched signals with direction, magnitude, TP/SL

Usage:
    python timesfm_server.py
    # Or with env vars:
    TIMESFM_REMOTE_PORT=8787 TIMESFM_REMOTE_TOKEN=your_token python timesfm_server.py
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import timesfm


MODEL = None
MODEL_LOCK = threading.Lock()

# TP/SL recommendations from backtest optimization
# Format: {series_prefix: {(mag_lo, mag_hi): {"tp": cents, "sl": cents}}}
TPSL_DEFAULTS = {
    "KXHIGH": {  # Weather markets
        (2, 5): {"tp": 20, "sl": 3},
        (5, 10): {"tp": 20, "sl": 3},
        (10, 20): {"tp": 20, "sl": 3},
        (20, 100): {"tp": 30, "sl": 2},
    },
    "KXINX": {  # S&P 500
        (2, 5): {"tp": 30, "sl": 5},
        (5, 10): {"tp": 50, "sl": 2},
        (10, 20): {"tp": 50, "sl": 8},
        (20, 100): {"tp": 50, "sl": 10},
    },
    "KXBTC": {  # Bitcoin — wider stops, finer magnitude buckets
        (2, 5): {"tp": 40, "sl": 8},
        (5, 10): {"tp": 50, "sl": 8},
        (10, 20): {"tp": 50, "sl": 10},
        (20, 40): {"tp": 50, "sl": 12},
        (40, 70): {"tp": 60, "sl": 15},
        (70, 100): {"tp": 70, "sl": 15},
    },
    "KXETH": {  # Ethereum — similar to BTC, slightly wider SL (higher vol)
        (2, 5): {"tp": 40, "sl": 10},
        (5, 10): {"tp": 50, "sl": 10},
        (10, 20): {"tp": 50, "sl": 12},
        (20, 40): {"tp": 50, "sl": 15},
        (40, 70): {"tp": 60, "sl": 15},
        (70, 100): {"tp": 70, "sl": 18},
    },
}

# Fallback TP/SL for unknown series
TPSL_FALLBACK = {"tp": 20, "sl": 5}

# Confidence thresholds: {series_prefix: (high, med-high, med)}
# Crypto needs higher magnitude for same confidence (more volatile)
CONFIDENCE_THRESHOLDS = {
    "KXBTC": (40, 20, 8),
    "KXETH": (40, 20, 8),
}
CONFIDENCE_DEFAULT = (20, 10, 5)  # weather, equities, etc.

# Per-series minimum magnitude thresholds (overrides request-level threshold)
# Based on realistic backtest parameter sweep 2026-02-26
SERIES_MIN_MAGNITUDE = {
    # Profitable cities (realistic backtest 2026-02-26, all 18 cities tested)
    "KXHIGHNY": 20,     # NYC: PF 2.82 at 20c, 31 trades, 61% WR
    "KXHIGHAUS": 10,    # Austin: PF 1.39 at 10c, 34 trades, 47% WR
    "KXHIGHTDC": 10,    # Washington DC: PF 10.35 at 10c, 6 trades, 83% WR
    "KXHIGHTLV": 10,    # Las Vegas: PF 4.03 at 10c, 5 trades, 60% WR
    "KXHIGHLAX": 20,    # LA: PF 2.39 at 20c, 4 trades (watch list)
    # Disabled cities (losing at every threshold)
    "KXHIGHCHI": 999,   # Chicago: never profitable (lake effect)
    "KXHIGHDEN": 999,   # Denver: never profitable (mountain)
    "KXHIGHTATL": 999,  # Atlanta: losing
    "KXHIGHTSEA": 999,  # Seattle: losing (coastal)
    "KXHIGHTSFO": 999,  # San Francisco: worst performer -102c (coastal)
    "KXHIGHTHOU": 999,  # Houston: losing
    "KXHIGHTMIN": 999,  # Minneapolis: near breakeven but losing
    "KXHIGHTNOLA": 999, # New Orleans: near breakeven but losing
    "KXHIGHTPHX": 999,  # Phoenix: losing
    # Crypto + S&P use request-level threshold (default 5c) — already profitable
}


def _load_model():
    global MODEL
    if MODEL is not None:
        return MODEL

    print("Loading TimesFM 2.5 200M...", flush=True)
    torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        device_map="cuda",
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=512,
            max_horizon=128,
            per_core_batch_size=32,
        )
    )
    MODEL = model
    print("Model loaded and compiled on GPU.", flush=True)
    return MODEL


def _get_tpsl(series_ticker: str, magnitude: float) -> dict:
    """Look up optimal TP/SL for a series and signal magnitude."""
    # Match series prefix
    for prefix, buckets in TPSL_DEFAULTS.items():
        if series_ticker.startswith(prefix):
            for (lo, hi), levels in buckets.items():
                if lo <= magnitude < hi:
                    return levels
            return TPSL_FALLBACK
    return TPSL_FALLBACK


class Handler(BaseHTTPRequestHandler):
    def _write_json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        token = os.environ.get("TIMESFM_REMOTE_TOKEN", "").strip()
        if not token:
            return True
        supplied = self.headers.get("X-TimesFM-Token", "")
        return supplied == token

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", "ignore")
        return json.loads(raw)

    def do_GET(self):
        if self.path == "/health":
            self._write_json(200, {"ok": True, "model_loaded": MODEL is not None})
            return
        self._write_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if not self._authorized():
            self._write_json(401, {"ok": False, "error": "unauthorized"})
            return

        if self.path == "/forecast":
            self._handle_forecast()
        elif self.path == "/signal":
            self._handle_signal()
        else:
            self._write_json(404, {"ok": False, "error": "not found"})

    def _handle_forecast(self):
        """Backward-compatible raw forecast endpoint."""
        try:
            payload = self._read_body()
            contexts = payload.get("contexts", [])
            horizon = int(payload.get("horizon", 10))

            if not contexts:
                self._write_json(200, {"ok": True, "pred_last": []})
                return

            with MODEL_LOCK:
                model = _load_model()
                arr_inputs = [np.asarray(c, dtype=np.float32) for c in contexts]
                out = model.forecast(horizon=horizon, inputs=arr_inputs)

            point = out[0] if isinstance(out, tuple) else out
            point = np.asarray(point)
            pred_last = point[:, horizon - 1].tolist()

            self._write_json(200, {"ok": True, "pred_last": pred_last})

        except Exception as e:
            self._write_json(500, {"ok": False, "error": str(e)[:200]})

    def _handle_signal(self):
        """Enriched signal endpoint with direction, magnitude, TP/SL.

        Request body:
        {
            "markets": [
                {
                    "ticker": "KXHIGHNY-26FEB24-B32.5",
                    "series_ticker": "KXHIGHNY",
                    "prices": [52.0, 53.0, ...],  // price_close history
                    "current_price": 54.0
                },
                ...
            ],
            "horizon": 6,
            "signal_threshold": 2.0
        }

        Response:
        {
            "ok": true,
            "signals": [
                {
                    "ticker": "KXHIGHNY-26FEB24-B32.5",
                    "current_price": 54.0,
                    "forecast_mean": 56.3,
                    "forecast_end": 57.0,
                    "direction": "UP",
                    "side": "yes",
                    "magnitude": 2.3,
                    "point_forecast": [55.1, 55.8, ...],
                    "tp_cents": 20,
                    "sl_cents": 3,
                    "confidence": "medium"
                },
                ...
            ],
            "skipped": 3,
            "elapsed_ms": 142
        }
        """
        try:
            payload = self._read_body()
            markets = payload.get("markets", [])
            horizon = int(payload.get("horizon", 6))
            threshold = float(payload.get("signal_threshold", 2.0))

            if not markets:
                self._write_json(200, {"ok": True, "signals": [], "skipped": 0, "elapsed_ms": 0})
                return

            start = time.time()

            # Filter markets with enough price data
            valid_markets = []
            valid_prices = []
            skipped = 0

            for m in markets:
                prices = m.get("prices", [])
                if len(prices) < 3:
                    skipped += 1
                    continue
                arr = np.asarray(prices, dtype=np.float32)
                if np.all(arr == 0) or np.all(np.isnan(arr)) or np.std(arr) == 0:
                    skipped += 1
                    continue
                valid_markets.append(m)
                valid_prices.append(arr)

            if not valid_prices:
                self._write_json(200, {"ok": True, "signals": [], "skipped": skipped, "elapsed_ms": 0})
                return

            # Batch forecast
            with MODEL_LOCK:
                model = _load_model()
                out = model.forecast(horizon=horizon, inputs=valid_prices)

            point_forecast = out[0] if isinstance(out, tuple) else out
            point_forecast = np.asarray(point_forecast)

            # Generate signals
            signals = []
            for i, m in enumerate(valid_markets):
                current_price = float(m.get("current_price", valid_prices[i][-1]))
                point = point_forecast[i].tolist()
                forecast_mean = float(np.mean(point))
                forecast_end = float(point[-1])

                direction = "UP" if forecast_mean > current_price else "DOWN"
                magnitude = abs(forecast_mean - current_price)

                if magnitude < threshold:
                    skipped += 1
                    continue

                # Per-series minimum magnitude (from backtest optimization)
                series = m.get("series_ticker", "")
                series_min = None
                for prefix, min_mag in SERIES_MIN_MAGNITUDE.items():
                    if series.startswith(prefix):
                        series_min = min_mag
                        break
                if series_min is not None and magnitude < series_min:
                    skipped += 1
                    continue

                # Map direction to Kalshi side
                side = "yes" if direction == "UP" else "no"

                # Get TP/SL recommendation (series already set above)
                tpsl = _get_tpsl(series, magnitude)

                # Confidence level based on magnitude (series-specific)
                hi, mh, md = CONFIDENCE_DEFAULT
                for prefix, thresholds in CONFIDENCE_THRESHOLDS.items():
                    if series.startswith(prefix):
                        hi, mh, md = thresholds
                        break
                if magnitude >= hi:
                    confidence = "high"
                elif magnitude >= mh:
                    confidence = "medium-high"
                elif magnitude >= md:
                    confidence = "medium"
                else:
                    confidence = "low"

                signals.append({
                    "ticker": m.get("ticker", ""),
                    "series_ticker": series,
                    "current_price": current_price,
                    "forecast_mean": round(forecast_mean, 2),
                    "forecast_end": round(forecast_end, 2),
                    "direction": direction,
                    "side": side,
                    "magnitude": round(magnitude, 2),
                    "point_forecast": [round(p, 2) for p in point],
                    "tp_cents": tpsl["tp"],
                    "sl_cents": tpsl["sl"],
                    "confidence": confidence,
                })

            elapsed_ms = int((time.time() - start) * 1000)

            self._write_json(200, {
                "ok": True,
                "signals": signals,
                "skipped": skipped,
                "elapsed_ms": elapsed_ms,
            })

        except Exception as e:
            self._write_json(500, {"ok": False, "error": str(e)[:200]})

    def log_message(self, fmt, *args):
        # Suppress default request logging
        return


def main():
    host = os.environ.get("TIMESFM_REMOTE_HOST", "0.0.0.0")
    port = int(os.environ.get("TIMESFM_REMOTE_PORT", "8787"))

    # Pre-load model on startup
    print("Pre-loading model...", flush=True)
    _load_model()

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"ok": True, "listening": f"{host}:{port}"}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
