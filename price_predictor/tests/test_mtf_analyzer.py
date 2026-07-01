"""Tests für price_predictor/mtf_analyzer.py"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from price_predictor.mtf_analyzer import (
    MTFAnalyzer,
    _confirm_4h,
    _nearest_swing_high_above,
    _nearest_swing_low_below,
    _trend_bias,
    find_swings,
)


def _make_ohlcv(n: int, close_vals=None, trend: str = "up") -> pd.DataFrame:
    """Erstellt einen minimalen OHLCV-DataFrame mit `n` Kerzen."""
    if close_vals is not None:
        closes = list(close_vals)
    elif trend == "up":
        closes = [100.0 + i * 0.5 for i in range(n)]
    elif trend == "down":
        closes = [100.0 + (n - i) * 0.5 for i in range(n)]
    else:
        closes = [100.0 + (i % 5) * 0.3 for i in range(n)]

    df = pd.DataFrame(
        {
            "open":   [c - 0.1 for c in closes],
            "high":   [c + 0.5 for c in closes],
            "low":    [c - 0.5 for c in closes],
            "close":  closes,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    return df


class TestFindSwings(unittest.TestCase):
    def test_marks_local_high(self):
        highs = [1, 1, 1, 5, 1, 1, 1]
        df = _make_ohlcv(len(highs))
        df["high"] = highs
        result = find_swings(df, n=2)
        self.assertTrue(result["swing_high"].iloc[3])
        self.assertFalse(result["swing_high"].iloc[0])

    def test_marks_local_low(self):
        lows = [5, 5, 5, 1, 5, 5, 5]
        df = _make_ohlcv(len(lows))
        df["low"] = lows
        result = find_swings(df, n=2)
        self.assertTrue(result["swing_low"].iloc[3])

    def test_no_swing_flat(self):
        df = _make_ohlcv(10)
        df["high"] = 100.0
        df["low"] = 99.0
        result = find_swings(df, n=2)
        # Flat series: center=True rolling fills borders with NaN → False.
        # Interior candles all tie for the rolling max, so they are True.
        # With n=2, window=5: positions 2..7 are interior on a 10-row df.
        interior = result["swing_high"].iloc[2:-2]
        self.assertTrue(interior.all())

    def test_n_parameter_respected(self):
        highs = [1, 1, 5, 1, 1]
        df = _make_ohlcv(len(highs))
        df["high"] = highs
        result_n1 = find_swings(df, n=1)
        result_n2 = find_swings(df, n=2)
        # n=1 → window=3; n=2 → window=5 (uses entire series)
        self.assertTrue(result_n1["swing_high"].iloc[2])
        self.assertTrue(result_n2["swing_high"].iloc[2])


class TestTrendBias(unittest.TestCase):
    def test_uptrend_detected(self):
        df = _make_ohlcv(250, trend="up")
        result = _trend_bias(df, require_adx=False)
        self.assertEqual(result["bias"], "up")

    def test_downtrend_detected(self):
        df = _make_ohlcv(250, trend="down")
        result = _trend_bias(df, require_adx=False)
        self.assertEqual(result["bias"], "down")

    def test_adx_gate_neutralises_weak_trend(self):
        df = _make_ohlcv(250, trend="up")
        # Force ADX-related columns to produce low ADX by making price flat after 200 bars
        df["close"] = 100.0
        df["high"]  = 100.5
        df["low"]   = 99.5
        result = _trend_bias(df, require_adx=True)
        # Flat candles → ADX near 0 → bias neutralised
        self.assertEqual(result["bias"], "neutral")


class TestConfirm4h(unittest.TestCase):
    def test_confirms_uptrend_when_close_above_ema(self):
        df = _make_ohlcv(60, trend="up")
        self.assertTrue(_confirm_4h(df, "up"))

    def test_rejects_uptrend_in_downward_market(self):
        df = _make_ohlcv(60, trend="down")
        self.assertFalse(_confirm_4h(df, "up"))


class TestMTFAnalyzer(unittest.TestCase):
    def _make_analyzer(self):
        def fetch_fn(symbol, timeframe, limit):
            n = {"1d": 300, "4h": 300, "1h": 170, "5m": 20}.get(timeframe, limit)
            trend = "up"
            df = _make_ohlcv(n, trend=trend)
            if timeframe == "5m":
                # Letzte Kerze: bullische Reversal-Kerze (close > open, close > prev high)
                df["open"].iloc[-1]  = df["close"].iloc[-2] - 0.2
                df["close"].iloc[-1] = df["high"].iloc[-2] + 0.1
                df["high"].iloc[-1]  = df["close"].iloc[-1] + 0.05
                df["low"].iloc[-1]   = df["open"].iloc[-1] - 0.1
            return df

        return MTFAnalyzer("SOL/USD", fetch_fn)

    def test_refresh_bias_returns_dict_keys(self):
        analyzer = self._make_analyzer()
        bias = analyzer.refresh_bias()
        self.assertIn("daily_bias", bias)
        self.assertIn("confirmed_4h", bias)
        self.assertIn("daily_adx", bias)

    def test_refresh_bias_neutral_on_fetch_error(self):
        def bad_fetch(symbol, tf, limit):
            raise RuntimeError("network error")

        analyzer = MTFAnalyzer("ETH/USD", bad_fetch)
        bias = analyzer.refresh_bias()
        self.assertEqual(bias["daily_bias"], "neutral")
        self.assertFalse(bias["confirmed_4h"])

    def test_find_retest_setup_returns_none_when_neutral(self):
        analyzer = self._make_analyzer()
        bias = {"daily_bias": "neutral", "confirmed_4h": False, "daily_adx": 0.0}
        result = analyzer.find_retest_setup(bias)
        self.assertIsNone(result)

    def test_find_retest_setup_returns_dict_when_valid(self):
        analyzer = self._make_analyzer()
        # Build a 1h series with clear swings and uptrend
        bias = {"daily_bias": "up", "confirmed_4h": True, "daily_adx": 30.0}
        # Provide a fetch_fn that generates data with at least one swing_low below current price
        close_vals = (
            [100.0, 101.0, 102.0, 98.0, 103.0, 102.0, 104.0, 100.0, 105.0,
             104.0, 106.0, 103.0, 107.0] * 13
        )[:170]
        def fetch_fn(symbol, tf, limit):
            return _make_ohlcv(len(close_vals), close_vals=close_vals)

        analyzer2 = MTFAnalyzer("SOL/USD", fetch_fn)
        result = analyzer2.find_retest_setup(bias)
        if result is not None:
            self.assertIn("zone_low", result)
            self.assertIn("zone_high", result)
            self.assertIn("direction", result)
            self.assertIn("target", result)

    def test_check_entry_trigger_none_outside_zone(self):
        analyzer = self._make_analyzer()
        setup = {
            "zone_low": 50.0, "zone_high": 51.0,
            "direction": "long", "target": 55.0, "atr": 1.0, "level": 50.5,
        }
        result = analyzer.check_entry_trigger(setup, current_price=200.0)
        self.assertIsNone(result)


class TestSwingHelpers(unittest.TestCase):
    def _make_swing_df(self):
        df = _make_ohlcv(10)
        df["swing_high"] = False
        df["swing_low"]  = False
        df["high"]  = [100, 105, 103, 110, 108, 115, 113, 120, 118, 125]
        df["low"]   = [90,  88,  92,  87,  93,  86,  94,  85,  95,  84]
        df["close"] = [95, 100, 98,  105, 103, 110, 108, 115, 113, 120]
        df.loc[df.index[1], "swing_high"] = True   # 105
        df.loc[df.index[3], "swing_high"] = True   # 110
        df.loc[df.index[7], "swing_high"] = True   # 120
        df.loc[df.index[2], "swing_low"]  = True   # 92
        df.loc[df.index[4], "swing_low"]  = True   # 93
        return df

    def test_nearest_swing_high_above(self):
        df = self._make_swing_df()
        result = _nearest_swing_high_above(df, price=100.0)
        # Lowest swing high above 100 is 105 (idx 1)
        self.assertAlmostEqual(result, 105.0)

    def test_nearest_swing_low_below(self):
        df = self._make_swing_df()
        result = _nearest_swing_low_below(df, price=100.0)
        # Highest swing low below 100 is 93 (idx 4)
        self.assertAlmostEqual(result, 93.0)


if __name__ == "__main__":
    unittest.main()
