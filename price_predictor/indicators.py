import pandas as pd
import ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    atr = ta.volatility.AverageTrueRange(
        high=out["high"], low=out["low"], close=out["close"], window=14
    )
    out["atr"] = atr.average_true_range().replace(0.0, float("nan"))
    out["atr_pct"] = out["atr"] / out["close"] * 100

    bb = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_mid"] = bb.bollinger_mavg()

    rsi = ta.momentum.RSIIndicator(close=out["close"], window=14)
    out["rsi"] = rsi.rsi()

    vwap = ta.volume.VolumeWeightedAveragePrice(
        high=out["high"], low=out["low"], close=out["close"], volume=out["volume"], window=14
    )
    out["vwap"] = vwap.volume_weighted_average_price()

    adx = ta.trend.ADXIndicator(
        high=out["high"], low=out["low"], close=out["close"], window=14
    )
    out["adx"] = adx.adx()

    return out
