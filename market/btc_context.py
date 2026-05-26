"""
BTC market context: trend regime, returns, realized vol, dominance.
Acts as a hard filter for new entries (BTC crash → no new alts).
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from core.context import BTCContext
from data_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

_CACHE_TTL = 3600  # refresh hourly
_DOMINANCE_TTL = 3600

_btc_cache: Optional[BTCContext] = None
_btc_cache_ts: float = 0.0

_dominance_cache: float = 0.5
_dominance_cache_ts: float = 0.0


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _realized_vol_annualized(close: pd.Series, window: int = 7 * 24) -> float:
    """Annualized realized volatility from hourly close prices (window = hours)."""
    log_returns = np.log(close / close.shift(1)).dropna()
    hourly_std = float(log_returns.tail(window).std())
    return hourly_std * np.sqrt(24 * 365)


def _fetch_btc_dominance() -> float:
    """Fetch BTC dominance from CoinGecko public API (no auth needed)."""
    global _dominance_cache, _dominance_cache_ts
    if time.time() - _dominance_cache_ts < _DOMINANCE_TTL:
        return _dominance_cache
    try:
        import urllib.request, json
        url = "https://api.coingecko.com/api/v3/global"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        dom = data["data"]["market_cap_percentage"].get("btc", 50.0) / 100
        _dominance_cache = dom
        _dominance_cache_ts = time.time()
        logger.debug("BTC dominance: %.1f%%", dom * 100)
        return dom
    except Exception as e:
        logger.debug("BTC dominance fetch failed: %s", e)
        return _dominance_cache


def get_btc_context(force_refresh: bool = False) -> Optional[BTCContext]:
    """Return cached BTCContext, refreshing if older than _CACHE_TTL."""
    global _btc_cache, _btc_cache_ts

    if not force_refresh and _btc_cache and time.time() - _btc_cache_ts < _CACHE_TTL:
        return _btc_cache

    try:
        df = fetch_ohlcv("BTC/USD", "1h", 500)
        close = df["close"]

        ema200_4h = _ema(close.resample("4h").last().dropna(), 200).iloc[-1]
        current_4h = close.resample("4h").last().dropna().iloc[-1]
        if current_4h > ema200_4h * 1.01:
            trend = "up"
        elif current_4h < ema200_4h * 0.99:
            trend = "down"
        else:
            trend = "range"

        r1h = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2])
        r4h = float((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5])
        r24h = float((close.iloc[-1] - close.iloc[-25]) / close.iloc[-25])
        vol7d = _realized_vol_annualized(close, window=7 * 24)
        dom = _fetch_btc_dominance()

        _btc_cache = BTCContext(
            trend=trend,
            return_1h=r1h,
            return_4h=r4h,
            return_24h=r24h,
            realized_vol_7d=vol7d,
            dominance=dom,
        )
        _btc_cache_ts = time.time()
        logger.info("BTC: trend=%s r1h=%+.2f%% r4h=%+.2f%% vol7d=%.0f%% dom=%.1f%%",
                    trend, r1h * 100, r4h * 100, vol7d * 100, dom * 100)
        return _btc_cache

    except Exception as e:
        logger.warning("BTCContext fetch failed: %s", e)
        return _btc_cache  # return stale cache if available
