"""
Technische Indikatoren auf OHLCV-DataFrame berechnen.
"""

import pandas as pd
import ta as ta_lib


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high, low = df["high"], df["low"]

    # EMAs
    df["ema_9"]  = ta_lib.trend.ema_indicator(close, window=9)
    df["ema_21"] = ta_lib.trend.ema_indicator(close, window=21)
    df["ema_50"] = ta_lib.trend.ema_indicator(close, window=50)
    df["ema_200"] = ta_lib.trend.ema_indicator(close, window=200)

    # RSI
    rsi_period = params.get("rsi_period", 14)
    df["rsi"] = ta_lib.momentum.rsi(close, window=rsi_period)

    # MACD
    macd = ta_lib.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    # Bollinger Bands
    bb_period = params.get("bb_period", 20)
    bb_std    = params.get("bb_std", 2.0)
    bb = ta_lib.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    # ATR
    atr_period = params.get("atr_period", 14)
    df["atr"] = ta_lib.volatility.average_true_range(high, low, close, window=atr_period)

    # Volume SMA
    df["volume_sma"] = df["volume"].rolling(window=20).mean()

    return df
