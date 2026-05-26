"""
Higher-Timeframe features: 4h and 1d trend context.
Trading against a higher-timeframe trend dramatically reduces win rate.
"""

import numpy as np
import pandas as pd
import ta as ta_lib

FEATURE_NAMES = [
    "trend_4h",           # slope of 4h EMA50: -1..+1
    "trend_1d",           # price vs 1d EMA200: negative = below
    "dist_4h_ema200_atr", # distance from 4h price to 4h EMA200, in ATR units
    "htf_rsi_4h",         # 4h RSI (momentum context)
]


def extract(df_1h: pd.DataFrame) -> np.ndarray:
    """
    Extract 4 higher-timeframe features from a 1h OHLCV DataFrame.
    Requires at least 500 rows (enough for 4h EMA200 and 1d EMA200).
    """
    if df_1h is None or len(df_1h) < 100:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    try:
        close_1h = df_1h["close"]

        # 4h: resample and compute EMA50
        close_4h = close_1h.resample("4h").last().dropna()
        if len(close_4h) < 50:
            trend_4h = 0.0
            dist_4h = 0.0
            rsi_4h = 0.5
        else:
            ema50_4h = ta_lib.trend.ema_indicator(close_4h, window=50)
            # Slope: (ema[-1] - ema[-5]) / ema[-5], clipped
            slope = float((ema50_4h.iloc[-1] - ema50_4h.iloc[-5]) / max(ema50_4h.iloc[-5], 1e-9))
            trend_4h = float(np.clip(slope * 100, -1.0, 1.0))

            # Distance to 4h EMA200
            if len(close_4h) >= 200:
                ema200_4h = ta_lib.trend.ema_indicator(close_4h, window=200)
                atr_4h = float(ta_lib.volatility.average_true_range(
                    df_1h["high"].resample("4h").max().dropna(),
                    df_1h["low"].resample("4h").min().dropna(),
                    close_4h, window=14,
                ).iloc[-1])
                dist_4h = float(np.clip(
                    (close_4h.iloc[-1] - ema200_4h.iloc[-1]) / max(atr_4h, 1e-9),
                    -5.0, 5.0
                ))
            else:
                dist_4h = 0.0

            rsi_4h_series = ta_lib.momentum.rsi(close_4h, window=14)
            rsi_4h = float(rsi_4h_series.iloc[-1]) / 100.0

        # 1d: resample and compute EMA200
        close_1d = close_1h.resample("1D").last().dropna()
        if len(close_1d) >= 200:
            ema200_1d = ta_lib.trend.ema_indicator(close_1d, window=200)
            dist_1d = float(np.clip(
                (close_1d.iloc[-1] - ema200_1d.iloc[-1]) / max(ema200_1d.iloc[-1], 1e-9),
                -0.5, 0.5
            ))
        else:
            dist_1d = 0.0

        feats = np.array([
            trend_4h,
            dist_1d,
            dist_4h,
            rsi_4h,
        ], dtype=np.float32)

        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
