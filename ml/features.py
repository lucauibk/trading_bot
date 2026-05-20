import numpy as np
import pandas as pd
import ta as ta_lib

FEATURE_NAMES = [
    "ema9_ratio", "ema21_ratio", "ema_cross",
    "rsi",
    "mom_1h", "mom_4h", "mom_12h",
    "macd_hist",
    "bb_pct", "bb_width",
    "vol_ratio",
    "atr_pct",
    "body_pct", "upper_shadow_pct", "lower_shadow_pct", "is_green",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract 16-dimensional feature vector from OHLCV DataFrame (min 60 rows)."""
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    volume = df["volume"]
    price  = float(close.iloc[-1])

    ema9  = float(ta_lib.trend.ema_indicator(close, window=9).iloc[-1])
    ema21 = float(ta_lib.trend.ema_indicator(close, window=21).iloc[-1])

    rsi   = float(ta_lib.momentum.rsi(close, window=14).iloc[-1])
    mom_1 = float(close.pct_change(1).iloc[-1])
    mom_4 = float(close.pct_change(4).iloc[-1])
    mom_12 = float(close.pct_change(12).iloc[-1])

    macd_line   = float(ta_lib.trend.macd(close).iloc[-1])
    macd_signal = float(ta_lib.trend.macd_signal(close).iloc[-1])
    macd_hist   = (macd_line - macd_signal) / price if price != 0 else 0.0

    bb_high = float(ta_lib.volatility.bollinger_hband(close).iloc[-1])
    bb_low  = float(ta_lib.volatility.bollinger_lband(close).iloc[-1])
    bb_pct  = (price - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5
    bb_width = (bb_high - bb_low) / price if price != 0 else 0.0

    vol_sma   = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = min(float(volume.iloc[-1]) / vol_sma, 5.0) if vol_sma > 0 else 1.0

    atr     = float(ta_lib.volatility.average_true_range(high, low, close, window=14).iloc[-1])
    atr_pct = atr / price if price != 0 else 0.0

    o = float(open_.iloc[-1]); h = float(high.iloc[-1])
    l = float(low.iloc[-1]);   c = float(close.iloc[-1])
    total_range  = h - l if h != l else 1e-9
    body         = abs(c - o)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    feats = np.array([
        np.clip((ema9  - price) / price, -0.10, 0.10),
        np.clip((ema21 - price) / price, -0.10, 0.10),
        np.clip((ema9 - ema21)  / price, -0.05, 0.05),
        rsi / 100.0,
        np.clip(mom_1,  -0.15, 0.15),
        np.clip(mom_4,  -0.25, 0.25),
        np.clip(mom_12, -0.40, 0.40),
        np.clip(macd_hist, -0.03, 0.03),
        np.clip(bb_pct,   0.0, 1.0),
        np.clip(bb_width, 0.0, 0.3),
        np.clip(vol_ratio, 0.0, 5.0),
        np.clip(atr_pct,  0.0, 0.1),
        body         / total_range,
        upper_shadow / total_range,
        lower_shadow / total_range,
        1.0 if c >= o else 0.0,
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
