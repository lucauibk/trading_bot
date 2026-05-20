import ccxt
import pandas as pd
import time
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


def _build_exchange(public_only: bool = False) -> ccxt.Exchange:
    exchange_class = getattr(ccxt, config.EXCHANGE_ID)
    params = {"enableRateLimit": True, "timeout": 10000, "options": {"defaultType": "spot"}}
    if not public_only:
        params["apiKey"] = config.API_KEY
        params["secret"] = config.API_SECRET
    return exchange_class(params)


_exchange: Optional[ccxt.Exchange] = None
_public_exchange: Optional[ccxt.Exchange] = None


def get_exchange() -> ccxt.Exchange:
    global _exchange
    if _exchange is None:
        _exchange = _build_exchange(public_only=False)
    return _exchange


def get_public_exchange() -> ccxt.Exchange:
    global _public_exchange
    if _public_exchange is None:
        _public_exchange = _build_exchange(public_only=True)
    return _public_exchange


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    raw = get_public_exchange().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


def fetch_ticker(symbol: str) -> dict:
    last_exc = None
    for attempt in range(3):
        try:
            return get_public_exchange().fetch_ticker(symbol)
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise last_exc


def get_balance(currency: str = "USDT") -> float:
    try:
        balance = get_exchange().fetch_balance()
        return float(balance["free"].get(currency, 0.0))
    except Exception as e:
        logger.error("Balance-Fehler: %s", e)
        return 0.0


def fetch_ohlcv_since(symbol: str, timeframe: str, since_iso: str, limit: int = 1000) -> pd.DataFrame:
    # Binance hat die beste öffentliche OHLCV-API (bis 1000 Candles/Request, keine Auth nötig)
    binance = ccxt.binance({"enableRateLimit": True})
    since_ms = binance.parse8601(since_iso)
    now_ms = binance.milliseconds()
    all_candles: list = []

    while True:
        batch = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        since_ms = last_ts + 1

        if now_ms - last_ts < 2 * 60 * 60 * 1000:
            break

        time.sleep(binance.rateLimit / 1000)
        print(f"  ...{len(all_candles)} Candles geladen", end="\r")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df.astype(float)
