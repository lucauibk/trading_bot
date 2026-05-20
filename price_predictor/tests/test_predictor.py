import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from price_predictor.predictor import PricePredictor


def _make_ohlcv(n: int = 200, base: float = 50000.0, volatility: float = 200.0) -> pd.DataFrame:
    np.random.seed(7)
    close = base + np.cumsum(np.random.randn(n) * volatility)
    high = close + np.abs(np.random.randn(n) * volatility * 0.5)
    low = close - np.abs(np.random.randn(n) * volatility * 0.5)
    open_ = close + np.random.randn(n) * volatility * 0.3
    volume = np.random.randint(100, 5000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TestPricePredictorInit:
    def test_default_params(self):
        p = PricePredictor(exchange_id="binance", symbol="BTC/USDT")
        assert p.exchange_id == "binance"
        assert p.symbol == "BTC/USDT"

    def test_custom_timeframe(self):
        p = PricePredictor(exchange_id="kraken", symbol="ETH/USDT", timeframe="4h")
        assert p.timeframe == "4h"


class TestPricePredictorPredict:
    @pytest.fixture
    def predictor(self):
        return PricePredictor(exchange_id="binance", symbol="BTC/USDT", timeframe="1h")

    @pytest.fixture
    def sample_df(self):
        return _make_ohlcv()

    def test_predict_returns_dict(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        for key in ("predicted_low", "predicted_high", "confidence", "grid_levels", "regime"):
            assert key in result, f"Missing key: {key}"

    def test_predicted_low_less_than_high(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        assert result["predicted_low"] < result["predicted_high"]

    def test_confidence_bounded(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        assert 0.0 <= result["confidence"] <= 1.0

    def test_grid_levels_count(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        assert len(result["grid_levels"]) == 10

    def test_grid_levels_sorted(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        levels = result["grid_levels"]
        assert levels == sorted(levels)

    def test_grid_levels_within_range(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        for lvl in result["grid_levels"]:
            assert result["predicted_low"] <= lvl <= result["predicted_high"]

    def test_regime_valid_value(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        assert result["regime"] in ("ranging", "trending", "volatile")

    def test_regime_trending_high_adx(self, predictor):
        df = _make_ohlcv(n=200)
        # force trending regime via compute_indicators mock
        indicators_override = {
            "adx": pd.Series([30.0] * 200),
            "atr_pct": pd.Series([1.0] * 200),
        }
        with patch.object(predictor, "_fetch_ohlcv", return_value=df):
            with patch("price_predictor.predictor.compute_indicators") as mock_ind:
                from price_predictor.indicators import compute_indicators as real_ci
                real_result = real_ci(df)
                real_result["adx"] = 30.0
                real_result["atr_pct"] = 1.0
                mock_ind.return_value = real_result
                result = predictor.predict()
        assert result["regime"] == "trending"

    def test_regime_volatile_high_atr_pct(self, predictor):
        df = _make_ohlcv(n=200)
        with patch.object(predictor, "_fetch_ohlcv", return_value=df):
            with patch("price_predictor.predictor.compute_indicators") as mock_ind:
                from price_predictor.indicators import compute_indicators as real_ci
                real_result = real_ci(df)
                real_result["adx"] = 15.0
                real_result["atr_pct"] = 5.0
                mock_ind.return_value = real_result
                result = predictor.predict()
        assert result["regime"] == "volatile"

    def test_predicted_values_are_positive(self, predictor, sample_df):
        with patch.object(predictor, "_fetch_ohlcv", return_value=sample_df):
            result = predictor.predict()
        assert result["predicted_low"] > 0
        assert result["predicted_high"] > 0


class TestPricePredictorGridLevels:
    def test_custom_grid_count(self):
        p = PricePredictor(exchange_id="binance", symbol="BTC/USDT", grid_count=5)
        df = _make_ohlcv()
        with patch.object(p, "_fetch_ohlcv", return_value=df):
            result = p.predict()
        assert len(result["grid_levels"]) == 5
