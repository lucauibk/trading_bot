"""
Funding Rate + Open Interest from Kraken Futures (with Binance fallback).

Funding rate z-score is a strong directional signal:
  positive funding = longs pay shorts = bullish crowding
  negative funding = shorts pay longs = bearish crowding

Both are cached 1h in SQLite to avoid hitting API limits.
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np

from core.context import FundingInfo

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/perp_cache.db")
_CACHE_TTL = 3600  # 1h

# Map spot symbols to perp symbols on each exchange
_KRAKEN_MAP = {
    "SOL/USD": "SOL/USD:USD",
    "ETH/USD": "ETH/USD:USD",
    "AVAX/USD": "AVAX/USD:USD",
    "LINK/USD": "LINK/USD:USD",
    "XRP/USD": "XRP/USD:USD",
    "BTC/USD": "BTC/USD:USD",
}
_BINANCE_MAP = {
    "SOL/USD": "SOL/USDT",
    "ETH/USD": "ETH/USDT",
    "AVAX/USD": "AVAX/USDT",
    "LINK/USD": "LINK/USDT",
    "XRP/USD": "XRP/USDT",
    "BTC/USD": "BTC/USDT",
}

_kraken_futures: Optional[ccxt.Exchange] = None
_binance_futures: Optional[ccxt.Exchange] = None


def _get_kraken_futures() -> ccxt.Exchange:
    global _kraken_futures
    if _kraken_futures is None:
        _kraken_futures = ccxt.krakenfutures({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
    return _kraken_futures


def _get_binance_futures() -> ccxt.Exchange:
    global _binance_futures
    if _binance_futures is None:
        _binance_futures = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
    return _binance_futures


def _init_db():
    _DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS funding_cache (
            symbol      TEXT PRIMARY KEY,
            rate        REAL,
            oi_usd      REAL,
            history_json TEXT,
            updated_ts  REAL
        )
    """)
    con.commit()
    con.close()


def _load_cache(symbol: str) -> Optional[dict]:
    try:
        con = sqlite3.connect(_DB_PATH)
        row = con.execute(
            "SELECT * FROM funding_cache WHERE symbol=?", (symbol,)
        ).fetchone()
        con.close()
        if row and time.time() - row[4] < _CACHE_TTL:
            return {
                "rate": row[1],
                "oi_usd": row[2],
                "history": json.loads(row[3] or "[]"),
                "ts": row[4],
            }
    except Exception:
        pass
    return None


def _save_cache(symbol: str, rate: float, oi_usd: float, history: list):
    try:
        con = sqlite3.connect(_DB_PATH)
        con.execute(
            "INSERT OR REPLACE INTO funding_cache VALUES (?,?,?,?,?)",
            (symbol, rate, oi_usd, json.dumps(history), time.time()),
        )
        con.commit()
        con.close()
    except Exception:
        pass


def _fetch_from_binance(spot_symbol: str) -> Optional[dict]:
    perp = _BINANCE_MAP.get(spot_symbol)
    if not perp:
        return None
    try:
        ex = _get_binance_futures()
        fr = ex.fetch_funding_rate(perp)
        rate = float(fr.get("fundingRate", 0) or 0)

        # OI
        try:
            oi_data = ex.fetch_open_interest_history(perp, "1h", limit=24)
            oi_series = [float(x.get("openInterestValue", 0) or 0) for x in oi_data]
            oi_usd = oi_series[-1] if oi_series else 0.0
        except Exception:
            oi_usd = 0.0
            oi_series = []

        # Funding history for z-score
        try:
            hist = ex.fetch_funding_rate_history(perp, limit=7 * 3)  # ~7 days, 3/day
            history = [float(h.get("fundingRate", 0) or 0) for h in hist]
        except Exception:
            history = [rate]

        return {"rate": rate, "oi_usd": oi_usd, "oi_series": oi_series, "history": history}
    except Exception as e:
        logger.debug("Binance funding fetch failed for %s: %s", spot_symbol, e)
        return None


def get_funding(spot_symbol: str) -> Optional[FundingInfo]:
    """Return FundingInfo for a spot symbol, using cache when fresh."""
    _init_db()

    cached = _load_cache(spot_symbol)
    if cached:
        history = cached["history"]
        rate = cached["rate"]
        oi_usd = cached["oi_usd"]
    else:
        data = _fetch_from_binance(spot_symbol)
        if data is None:
            return None
        rate = data["rate"]
        oi_usd = data["oi_usd"]
        history = data.get("history", [rate])
        oi_series = data.get("oi_series", [])

        # Compute OI change
        _save_cache(spot_symbol, rate, oi_usd, history)

        # Compute z-score
        if len(history) >= 3:
            arr = np.array(history)
            z = float((rate - arr.mean()) / (arr.std() + 1e-9))
        else:
            z = 0.0

        oi_change_1h = 0.0
        oi_change_24h = 0.0
        if oi_series and len(oi_series) >= 2:
            oi_change_1h = (oi_series[-1] - oi_series[-2]) / max(abs(oi_series[-2]), 1)
        if oi_series and len(oi_series) >= 24:
            oi_change_24h = (oi_series[-1] - oi_series[-24]) / max(abs(oi_series[-24]), 1)

        info = FundingInfo(
            symbol=spot_symbol,
            rate=rate,
            rate_z7d=z,
            oi_change_1h=oi_change_1h,
            oi_change_24h=oi_change_24h,
        )
        logger.info("Funding %-12s rate=%+.4f%% z=%.2f OI_chg1h=%+.1f%%",
                    spot_symbol, rate * 100, z, oi_change_1h * 100)
        return info

    # From cache – recompute derived metrics
    arr = np.array(history) if history else np.array([rate])
    z = float((rate - arr.mean()) / (arr.std() + 1e-9))
    return FundingInfo(
        symbol=spot_symbol,
        rate=rate,
        rate_z7d=z,
        oi_change_1h=0.0,
        oi_change_24h=0.0,
    )
