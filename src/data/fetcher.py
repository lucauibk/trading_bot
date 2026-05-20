"""
Marktdaten-Fetcher mit SQLite-Cache, Retry-Logik und Rate Limiting.
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parents[2] / "data" / "ohlcv_cache.db"
_exchange_cache: dict = {}


def _get_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT, timeframe TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    """)
    con.commit()
    return con


def _build_exchange(exchange_id: str, api_key: str = "", api_secret: str = "",
                    public_only: bool = False) -> ccxt.Exchange:
    cls = getattr(ccxt, exchange_id)
    params: dict = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    if not public_only and api_key:
        params["apiKey"] = api_key
        params["secret"] = api_secret
    return cls(params)


def get_exchange(exchange_id: str, api_key: str = "", api_secret: str = "",
                 public_only: bool = False) -> ccxt.Exchange:
    key = (exchange_id, public_only)
    if key not in _exchange_cache:
        _exchange_cache[key] = _build_exchange(exchange_id, api_key, api_secret, public_only)
    return _exchange_cache[key]


def _fetch_with_retry(fn, retries: int = 3, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return fn()
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            wait = backoff ** attempt
            logger.warning("API-Fehler (Versuch %d/%d): %s – warte %.1fs", attempt + 1, retries, e, wait)
            time.sleep(wait)
        except ccxt.RateLimitExceeded:
            time.sleep(5)
    raise RuntimeError("Alle Retry-Versuche fehlgeschlagen")


def get_ohlcv(symbol: str, timeframe: str, limit: int = 500,
              exchange_id: str = "kraken") -> pd.DataFrame:
    ex = get_exchange(exchange_id, public_only=True)

    raw = _fetch_with_retry(lambda: ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
    df = _to_dataframe(raw)

    # In Cache schreiben
    try:
        con = _get_db()
        rows = [(symbol, timeframe, int(ts.timestamp() * 1000),
                 r.open, r.high, r.low, r.close, r.volume)
                for ts, r in df.iterrows()]
        con.executemany(
            "INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
    except Exception as e:
        logger.warning("Cache-Schreibfehler: %s", e)

    return df


def get_ohlcv_since(symbol: str, timeframe: str, since_iso: str,
                    limit: int = 1000) -> pd.DataFrame:
    binance = get_exchange("binance", public_only=True)
    since_ms = binance.parse8601(since_iso)
    now_ms = binance.milliseconds()
    all_candles: list = []

    while True:
        batch = _fetch_with_retry(
            lambda: binance.fetch_ohlcv(symbol, timeframe=timeframe,
                                        since=since_ms, limit=limit))
        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        since_ms = last_ts + 1
        if now_ms - last_ts < 2 * 60 * 60 * 1000:
            break
        time.sleep(binance.rateLimit / 1000)

    df = _to_dataframe(all_candles)
    return df[~df.index.duplicated(keep="last")]


def get_ticker(symbol: str, exchange_id: str = "kraken") -> dict:
    ex = get_exchange(exchange_id, public_only=True)
    return _fetch_with_retry(lambda: ex.fetch_ticker(symbol))


def get_balance(currency: str, exchange_id: str, api_key: str,
                api_secret: str) -> float:
    try:
        ex = get_exchange(exchange_id, api_key, api_secret)
        balance = ex.fetch_balance()
        return float(balance["free"].get(currency, 0.0))
    except Exception as e:
        logger.error("Balance-Fehler: %s", e)
        return 0.0


def _to_dataframe(raw) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)
