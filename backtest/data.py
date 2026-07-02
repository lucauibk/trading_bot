"""
OHLCV loader with SQLite cache (data/ohlcv_cache.db).

First call per symbol fetches missing history from Binance (public API,
/USD mapped to /USDT) and persists it; subsequent calls — e.g. hundreds of
sweep runs — are pure local reads.
"""

import datetime
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger("backtest.data")

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ohlcv_cache.db"

_TIMEFRAME_MS = {"5m": 300_000, "15m": 900_000, "1h": 3600_000,
                 "4h": 4 * 3600_000, "1d": 24 * 3600_000}


def _ensure_table(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT, timeframe TEXT, timestamp INTEGER,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timeframe, timestamp)
        )
    """)


def _binance_symbol(symbol: str) -> str:
    return symbol.replace("/USDT", "/USD").replace("/USD", "/USDT")


def load_ohlcv(symbol: str, timeframe: str = "1h", days: int = 180) -> pd.DataFrame:
    """Return OHLCV DataFrame (UTC datetime index) covering the last `days` days."""
    tf_ms = _TIMEFRAME_MS[timeframe]
    now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    since_ms = now_ms - days * 24 * 3600_000

    con = sqlite3.connect(DB_PATH)
    try:
        _ensure_table(con)
        row = con.execute(
            "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM ohlcv "
            "WHERE symbol=? AND timeframe=? AND timestamp>=?",
            (symbol, timeframe, since_ms),
        ).fetchone()
        cache_min, cache_max, cache_n = row
        expected = (now_ms - since_ms) // tf_ms

        # Refetch when coverage has holes at either end or rows are missing
        # (sparse middle gaps from exchange downtime are tolerated).
        complete = (
            cache_n >= expected * 0.97
            and cache_min is not None
            and cache_min <= since_ms + 2 * tf_ms
            and cache_max >= now_ms - 3 * tf_ms
        )
        if not complete:
            from data_fetcher import fetch_ohlcv_since
            since_iso = datetime.datetime.fromtimestamp(
                since_ms / 1000, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info("Cache miss %s %s (%s rows) → fetching %dd from Binance",
                        symbol, timeframe, cache_n, days)
            fresh = fetch_ohlcv_since(_binance_symbol(symbol), timeframe, since_iso)
            rows = [
                (symbol, timeframe, int(ts.value // 1_000_000),
                 float(r["open"]), float(r["high"]), float(r["low"]),
                 float(r["close"]), float(r["volume"]))
                for ts, r in fresh.iterrows()
            ]
            con.executemany(
                "INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows
            )
            con.commit()

        df = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol=? AND timeframe=? AND timestamp>=? ORDER BY timestamp",
            con, params=(symbol, timeframe, since_ms),
        )
    finally:
        con.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)
