import numpy as np
import pandas as pd
import pytest

from price_predictor.indicators import compute_indicators


def _make_ohlcv(n: int = 100, base: float = 100.0, volatility: float = 1.0) -> pd.DataFrame:
    np.random.seed(42)
    close = base + np.cumsum(np.random.randn(n) * volatility)
    high = close + np.abs(np.random.randn(n) * volatility * 0.5)
    low = close - np.abs(np.random.randn(n) * volatility * 0.5)
    open_ = close + np.random.randn(n) * volatility * 0.3
    volume = np.random.randint(1000, 10000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


class TestComputeIndicators:
    def test_returns_expected_columns(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        for col in ["atr", "bb_upper", "bb_lower", "bb_mid", "rsi", "vwap", "adx"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_atr_is_positive(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        atr = result["atr"].dropna()
        assert len(atr) > 0
        assert (atr > 0).all()

    def test_rsi_bounded(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        rsi = result["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_bollinger_ordering(self):
        df = _make_ohlcv()
        result = compute_indicators(df).dropna()
        assert (result["bb_upper"] >= result["bb_mid"]).all()
        assert (result["bb_mid"] >= result["bb_lower"]).all()

    def test_adx_bounded(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        adx = result["adx"].dropna()
        assert (adx >= 0).all() and (adx <= 100).all()

    def test_atr_percent_column(self):
        df = _make_ohlcv()
        result = compute_indicators(df)
        assert "atr_pct" in result.columns
        atr_pct = result["atr_pct"].dropna()
        assert (atr_pct >= 0).all()

    def test_handles_minimal_rows(self):
        df = _make_ohlcv(n=30)
        result = compute_indicators(df)
        assert len(result) == 30

    def test_does_not_mutate_input(self):
        df = _make_ohlcv()
        cols_before = set(df.columns)
        compute_indicators(df)
        assert set(df.columns) == cols_before
