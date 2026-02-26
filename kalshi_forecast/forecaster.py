"""TimesFM forecasting on Kalshi market price series."""

import numpy as np
import pandas as pd

from . import config


class Forecaster:
    def __init__(self):
        self.model = None

    def load_model(self):
        """Load TimesFM 2.5 200M onto GPU."""
        import timesfm

        print("Loading TimesFM 2.5 200M...")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            config.TIMESFM_REPO, device_map="cuda"
        )
        fc = timesfm.ForecastConfig(
            max_context=config.TIMESFM_MAX_CONTEXT,
            max_horizon=config.TIMESFM_MAX_HORIZON,
            per_core_batch_size=config.TIMESFM_BATCH_SIZE,
        )
        self.model.compile(fc)
        print("Model loaded and compiled on GPU.")

    def forecast(
        self,
        price_series: list[np.ndarray],
        horizon: int = 12,
    ) -> list[dict]:
        """Forecast future prices for one or more price series.

        Args:
            price_series: List of 1D arrays, each a price_close time series.
            horizon: Number of future steps to forecast (in candlestick intervals).

        Returns:
            List of dicts with 'point' and 'quantiles' arrays per input series.
        """
        if self.model is None:
            self.load_model()

        point_forecast, quantile_forecast = self.model.forecast(horizon, price_series)

        results = []
        for i in range(point_forecast.shape[0]):
            q = None
            if quantile_forecast is not None and i < quantile_forecast.shape[0]:
                q = quantile_forecast[i]
            results.append({"point": point_forecast[i], "quantiles": q})
        return results

    def forecast_market(
        self,
        candles_df: pd.DataFrame,
        ticker: str,
        horizon: int = 12,
    ) -> dict:
        """Forecast a single market's price trajectory.

        Args:
            candles_df: DataFrame with columns [timestamp, ticker, price_close, ...]
            ticker: The specific market ticker to forecast.
            horizon: Steps ahead to forecast.

        Returns:
            Dict with current_price, point_forecast, direction, confidence info.
        """
        market_data = candles_df[candles_df["ticker"] == ticker].sort_values("timestamp")
        if market_data.empty:
            return {"ticker": ticker, "error": "no data"}

        prices = market_data["price_close"].values.astype(float)
        if len(prices) < 3:
            return {"ticker": ticker, "error": "insufficient data"}
        current_price = prices[-1]

        results = self.forecast([prices], horizon=horizon)
        point = results[0]["point"]
        quantiles = results[0]["quantiles"]

        forecast_mean = float(np.mean(point))
        direction = "UP" if forecast_mean > current_price else "DOWN"
        magnitude = abs(forecast_mean - current_price)

        output = {
            "ticker": ticker,
            "current_price": float(current_price),
            "forecast_mean": forecast_mean,
            "forecast_next": float(point[0]),
            "forecast_end": float(point[-1]),
            "direction": direction,
            "magnitude": magnitude,
            "point_forecast": point,
        }
        if quantiles is not None:
            output["quantiles"] = quantiles

        return output
