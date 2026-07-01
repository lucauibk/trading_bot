"""
Multi-Timeframe (MTF) Setup-Analyse: Daily-Trend → 4h-Bestätigung → 1h-Retest-Zone → 5m-Entry.

Top-Down-Methodik: Tagestrend bestimmen, auf 4h bestätigen, auf 1h auf den Retest
einer gebrochenen Struktur (Swing-High/Low) warten, auf 5m den Einstieg triggern.
"""

import logging
from typing import Callable, Optional

import pandas as pd
import ta

from .indicators import compute_indicators

logger = logging.getLogger(__name__)

EMA_FAST_PERIOD = 50
EMA_SLOW_PERIOD = 200
DAILY_ADX_TREND_MIN = 20.0
FRACTAL_N = 2                     # 5-Kerzen-Fraktal (Bill-Williams-Stil)
RETEST_ZONE_ATR_MULT = 0.5
TARGET_ATR_FALLBACK_MULT = 2.5    # spiegelt DIRECTIONAL_TP_ATR
ENTRY_RSI_LOWER_BAND = 45.0
ENTRY_RSI_UPPER_BAND = 65.0
SWING_LOOKBACK_BARS_1H = 120

_DAILY_LIMIT = 300
_H4_LIMIT = 300
_H1_LIMIT = SWING_LOOKBACK_BARS_1H + 50
_M5_LIMIT = 20


def _add_emas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ta.trend.EMAIndicator(close=out["close"], window=EMA_FAST_PERIOD).ema_indicator()
    out["ema_slow"] = ta.trend.EMAIndicator(close=out["close"], window=EMA_SLOW_PERIOD).ema_indicator()
    return out


def find_swings(df: pd.DataFrame, n: int = FRACTAL_N) -> pd.DataFrame:
    """Fügt boolesche Spalten 'swing_high'/'swing_low' via N-Bar-Fraktal hinzu."""
    out = df.copy()
    window = 2 * n + 1
    is_max = out["high"].rolling(window, center=True).max() == out["high"]
    is_min = out["low"].rolling(window, center=True).min() == out["low"]
    out["swing_high"] = is_max.fillna(False)
    out["swing_low"] = is_min.fillna(False)
    return out


def _trend_bias(df: pd.DataFrame, require_adx: bool) -> dict:
    ind = _add_emas(compute_indicators(df))
    ind = ind.dropna(subset=["ema_slow", "adx"])
    if ind.empty:
        return {"bias": "neutral", "adx": 0.0}
    row = ind.iloc[-1]
    close, ema_fast, ema_slow, adx = (
        float(row["close"]), float(row["ema_fast"]), float(row["ema_slow"]), float(row["adx"])
    )

    if close > ema_fast > ema_slow:
        bias = "up"
    elif close < ema_fast < ema_slow:
        bias = "down"
    else:
        bias = "neutral"

    if require_adx and adx < DAILY_ADX_TREND_MIN:
        bias = "neutral"

    return {"bias": bias, "adx": adx}


def _confirm_4h(df: pd.DataFrame, daily_bias: str) -> bool:
    ind = _add_emas(compute_indicators(df)).dropna(subset=["ema_fast"])
    if ind.empty:
        return False
    row = ind.iloc[-1]
    close, ema_fast = float(row["close"]), float(row["ema_fast"])
    if daily_bias == "up":
        return close > ema_fast
    if daily_bias == "down":
        return close < ema_fast
    return False


def _latest_swing_low_below(swings: pd.DataFrame, price: float) -> Optional[float]:
    """Der jüngste (zuletzt gebrochene) Swing-Low unterhalb des aktuellen Preises."""
    lows = swings.loc[swings["swing_low"] & (swings["low"] < price), "low"]
    return float(lows.iloc[-1]) if not lows.empty else None


def _latest_swing_high_above(swings: pd.DataFrame, price: float) -> Optional[float]:
    highs = swings.loc[swings["swing_high"] & (swings["high"] > price), "high"]
    return float(highs.iloc[-1]) if not highs.empty else None


def _nearest_swing_high_above(swings: pd.DataFrame, price: float) -> Optional[float]:
    """Das nächstgelegene Ziel-Level (Widerstand) oberhalb des aktuellen Preises."""
    highs = swings.loc[swings["swing_high"] & (swings["high"] > price), "high"]
    return float(highs.min()) if not highs.empty else None


def _nearest_swing_low_below(swings: pd.DataFrame, price: float) -> Optional[float]:
    lows = swings.loc[swings["swing_low"] & (swings["low"] < price), "low"]
    return float(lows.max()) if not lows.empty else None


class MTFAnalyzer:
    """Multi-Timeframe Setup-Analyzer. fetch_ohlcv_fn hat Signatur (symbol, timeframe, limit) → DataFrame."""

    def __init__(self, symbol: str, fetch_ohlcv_fn: Callable):
        self.symbol = symbol
        self._fetch_ohlcv_fn = fetch_ohlcv_fn

    def _fetch_ohlcv(self, timeframe: str, limit: int) -> pd.DataFrame:
        return self._fetch_ohlcv_fn(self.symbol, timeframe, limit)

    def refresh_bias(self) -> dict:
        """Langsamer Takt (alle 30min): 1d-Trend + 4h-Bestätigung."""
        try:
            daily_df = self._fetch_ohlcv("1d", _DAILY_LIMIT)
            daily = _trend_bias(daily_df, require_adx=True)
        except Exception as e:
            logger.warning("MTF refresh_bias 1d-Fehler %s: %s", self.symbol, e)
            return {"daily_bias": "neutral", "confirmed_4h": False, "daily_adx": 0.0}

        confirmed_4h = False
        if daily["bias"] in ("up", "down"):
            try:
                h4_df = self._fetch_ohlcv("4h", _H4_LIMIT)
                confirmed_4h = _confirm_4h(h4_df, daily["bias"])
            except Exception as e:
                logger.warning("MTF refresh_bias 4h-Fehler %s: %s", self.symbol, e)

        logger.info("MTF %s: daily_bias=%s confirmed_4h=%s adx=%.1f",
                    self.symbol, daily["bias"], confirmed_4h, daily["adx"])
        return {"daily_bias": daily["bias"], "confirmed_4h": confirmed_4h, "daily_adx": daily["adx"]}

    def find_retest_setup(self, bias: dict) -> Optional[dict]:
        """Sucht auf 1h die zuletzt gebrochene Struktur (Retest-Zone) + nächstes Ziel-Level."""
        direction = bias.get("daily_bias")
        if direction not in ("up", "down") or not bias.get("confirmed_4h"):
            return None

        try:
            df = self._fetch_ohlcv("1h", _H1_LIMIT)
        except Exception as e:
            logger.warning("MTF find_retest_setup 1h-Fehler %s: %s", self.symbol, e)
            return None

        ind = compute_indicators(df)
        swings = find_swings(ind, n=FRACTAL_N).dropna(subset=["atr"]).tail(SWING_LOOKBACK_BARS_1H)
        if swings.empty:
            return None

        current_price = float(swings["close"].iloc[-1])
        atr = float(swings["atr"].iloc[-1])

        if direction == "up":
            level = _latest_swing_low_below(swings, current_price)
            target = _nearest_swing_high_above(swings, current_price)
            trade_direction = "long"
        else:
            level = _latest_swing_high_above(swings, current_price)
            target = _nearest_swing_low_below(swings, current_price)
            trade_direction = "short"

        if level is None or atr <= 0:
            return None

        zone_half = RETEST_ZONE_ATR_MULT * atr
        zone_low, zone_high = level - zone_half, level + zone_half

        target_is_fallback = target is None
        if target_is_fallback:
            target = (current_price + TARGET_ATR_FALLBACK_MULT * atr if trade_direction == "long"
                      else current_price - TARGET_ATR_FALLBACK_MULT * atr)

        logger.info("MTF %s: Retest-Zone %.4f–%.4f gefunden → target=%.4f",
                    self.symbol, zone_low, zone_high, target)
        return {
            "level": level, "zone_low": zone_low, "zone_high": zone_high,
            "direction": trade_direction, "target": float(target),
            "target_is_fallback": target_is_fallback, "atr": atr,
        }

    def check_entry_trigger(self, setup: dict, current_price: float) -> Optional[dict]:
        """5m-Einstiegs-Trigger: Reversal-Kerze + RSI-Erholungsband, nur innerhalb der Zone."""
        if not (setup["zone_low"] <= current_price <= setup["zone_high"]):
            return None

        try:
            df = self._fetch_ohlcv("5m", _M5_LIMIT)
        except Exception as e:
            logger.warning("MTF check_entry_trigger 5m-Fehler %s: %s", self.symbol, e)
            return None

        ind = compute_indicators(df).dropna(subset=["rsi"])
        if len(ind) < 2:
            return None

        last, prev = ind.iloc[-1], ind.iloc[-2]
        rsi = float(last["rsi"])
        if not (ENTRY_RSI_LOWER_BAND <= rsi <= ENTRY_RSI_UPPER_BAND):
            return None

        if setup["direction"] == "long":
            confirmed = bool(last["close"] > last["open"] and last["close"] > prev["high"])
        else:
            confirmed = bool(last["close"] < last["open"] and last["close"] < prev["low"])

        if not confirmed:
            return None

        logger.info("MTF %s: Entry-Trigger @ %.4f (RSI=%.1f)", self.symbol, float(last["close"]), rsi)
        return {"trigger_price": float(last["close"]), "rsi": rsi}
