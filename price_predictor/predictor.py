from dataclasses import dataclass, field
from typing import List

import pandas as pd

from .data_fetcher import fetch_ohlcv
from .indicators import compute_indicators
from .grid_suggester import compute_grid_levels

_ADX_TRENDING_THRESHOLD = 25.0
_ATR_PCT_VOLATILE_THRESHOLD = 3.0
_ATR_FACTOR_TRENDING = 2.0
_ATR_FACTOR_VOLATILE = 1.5


@dataclass
class PricePredictor:
    exchange_id: str
    symbol: str
    timeframe: str = "1h"
    limit: int = 500
    grid_count: int = 10

    def _fetch_ohlcv(self) -> pd.DataFrame:
        return fetch_ohlcv(self.exchange_id, self.symbol, self.timeframe, self.limit)

    def predict(self) -> dict:
        df = self._fetch_ohlcv()
        ind = compute_indicators(df)
        row = ind.dropna().iloc[-1]

        regime, low, high = self._determine_regime_and_range(row)
        confidence = self._compute_confidence(row, regime)
        grid_levels = compute_grid_levels(low, high, self.grid_count)

        return {
            "predicted_low": round(float(low), 8),
            "predicted_high": round(float(high), 8),
            "confidence": round(float(confidence), 4),
            "grid_levels": grid_levels,
            "regime": regime,
        }

    def _determine_regime_and_range(self, row: pd.Series) -> tuple[str, float, float]:
        close = float(row["close"])
        atr = float(row["atr"])
        adx = float(row["adx"])
        atr_pct = float(row["atr_pct"])

        if adx > _ADX_TRENDING_THRESHOLD:
            half = atr * _ATR_FACTOR_TRENDING
            return "trending", max(close - half, 0.0), close + half

        if atr_pct > _ATR_PCT_VOLATILE_THRESHOLD:
            half = atr * _ATR_FACTOR_VOLATILE
            return "volatile", max(close - half, 0.0), close + half

        bb_lower = float(row["bb_lower"])
        bb_upper = float(row["bb_upper"])
        if bb_upper > bb_lower:
            return "ranging", bb_lower, bb_upper

        # ATR fallback
        half = atr * _ATR_FACTOR_VOLATILE
        return "ranging", max(close - half, 0.0), close + half

    def _compute_confidence(self, row: pd.Series, regime: str) -> float:
        rsi = float(row["rsi"])
        adx = float(row["adx"])

        # RSI near 50 = more balanced market = higher confidence
        rsi_score = 1.0 - abs(rsi - 50.0) / 50.0

        # Higher ADX = clearer trend = higher confidence for trending, lower for ranging
        adx_score = min(adx / 100.0, 1.0)
        if regime == "ranging":
            adx_score = 1.0 - adx_score

        return max(0.0, min(1.0, (rsi_score + adx_score) / 2.0))
