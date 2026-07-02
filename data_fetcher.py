import ccxt
import pandas as pd
import time
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5  # seconds — matches execution/kraken.py


def _with_retry(fn, *args, **kwargs):
    """Retry on transient network/timeout errors with exponential backoff.

    Mirrors the implementation in execution/kraken.py so both code paths
    handle Kraken rate-limit spikes and short outages consistently.
    ccxt.ExchangeError (bad symbol, auth failure, etc.) is re-raised immediately
    without retry since those are logic errors, not transient issues.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exc = e
            sleep = _BACKOFF_BASE ** attempt
            logger.warning(
                "data_fetcher network error (attempt %d/%d): %s – retry in %.1fs",
                attempt + 1, _MAX_RETRIES, e, sleep,
            )
            time.sleep(sleep)
        except ccxt.ExchangeError:
            raise  # don't retry logic errors (bad symbol, auth, etc.)
    raise last_exc


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
    raw = _with_retry(get_public_exchange().fetch_ohlcv, symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


def fetch_ticker(symbol: str) -> dict:
    return _with_retry(get_public_exchange().fetch_ticker, symbol)


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


def fetch_funding_history(perp_symbol: str, since_iso: str, limit: int = 1000) -> pd.DataFrame:
    """Historische Funding-Raten via Binance-Perp (8h-Intervall, bis 2 Jahre).

    perp_symbol: ccxt-Perp-Notation, z.B. "SOL/USDT:USDT".
    Rückgabe: DataFrame mit UTC-Index und Spalte "rate" (Funding-Rate, z.B. 0.0001).
    OI wird bewusst NICHT geladen — Binance hält nur ~30 Tage OI-Historie vor.
    """
    binance = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    since_ms = binance.parse8601(since_iso)
    now_ms = binance.milliseconds()
    rows: list = []

    while True:
        batch = _with_retry(
            binance.fetch_funding_rate_history, perp_symbol, since=since_ms, limit=limit
        )
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        since_ms = last_ts + 1

        # Funding wird alle 8h gepostet → wenn wir bis <8h an jetzt heran sind, fertig
        if now_ms - last_ts < 8 * 60 * 60 * 1000:
            break
        time.sleep(binance.rateLimit / 1000)
        print(f"  ...{len(rows)} Funding-Punkte geladen", end="\r")

    if not rows:
        return pd.DataFrame(columns=["rate"])

    df = pd.DataFrame(
        [{"timestamp": r["timestamp"], "rate": float(r["fundingRate"])} for r in rows]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def perp_symbol_for(spot_symbol: str) -> str:
    """Mappt Spot-Symbol ("SOL/USD") auf Binance-Perp ("SOL/USDT:USDT")."""
    base = spot_symbol.split("/")[0]
    return f"{base}/USDT:USDT"
