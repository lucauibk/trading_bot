"""
Fill Reconciler – persists open order state and detects missed fills.

Replaces the old fetch_closed_orders(limit=20) approach.
Maintains a SQLite table of open orders (by client_id) and periodically
reconciles against the exchange via fetch_my_trades(since=last_ts).
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Callable, List, Optional

from core.strategy import Fill

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/trades.db")


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE IF NOT EXISTS open_orders (
            client_id          TEXT PRIMARY KEY,
            exchange_order_id  TEXT,
            symbol             TEXT,
            side               TEXT,
            price              REAL,
            qty                REAL,
            placed_ts          REAL,
            meta               TEXT DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS reconciler_state (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            last_fill_ts REAL DEFAULT 0
        );
        INSERT OR IGNORE INTO reconciler_state (id) VALUES (1);
        -- Dead-Letter für Fills, deren client_id nicht auflösbar war (#38):
        -- statt sie beim Vorschieben von last_fill_ts still zu verlieren,
        -- werden sie hier persistiert und bei jedem reconcile() erneut
        -- gegen open_orders aufgelöst.
        CREATE TABLE IF NOT EXISTS unresolved_fills (
            exchange_order_id TEXT PRIMARY KEY,
            symbol            TEXT,
            side              TEXT,
            price             REAL,
            qty               REAL,
            fee               REAL,
            ts                REAL,
            recorded_ts       REAL
        );
    """)
    con.commit()
    return con


class Reconciler:
    """Tracks open orders and reconciles fills from the exchange."""

    def __init__(self, fetch_fills_fn: Callable[[float], List[Fill]]):
        """
        fetch_fills_fn: callable that takes since_ts (unix) and returns list of Fill.
        Typically broker.reconcile_fills.
        """
        self._fetch = fetch_fills_fn
        self._last_fill_ts = self._load_last_ts()

    def _load_last_ts(self) -> float:
        try:
            con = _get_conn()
            row = con.execute("SELECT last_fill_ts FROM reconciler_state WHERE id=1").fetchone()
            con.close()
            return float(row["last_fill_ts"]) if row else time.time() - 3600
        except Exception as e:
            logger.warning("Reconciler: last_fill_ts laden fehlgeschlagen (%s) – Fallback -1h", e)
            return time.time() - 3600

    def _save_last_ts(self, ts: float):
        try:
            con = _get_conn()
            con.execute("UPDATE reconciler_state SET last_fill_ts=? WHERE id=1", (ts,))
            con.commit()
            con.close()
        except Exception as e:
            # Nicht schlucken: schlägt der Write still fehl, lädt ein Restart
            # einen veralteten Timestamp → bereits verarbeitete Fills würden
            # doppelt durch on_fill laufen (#38).
            logger.warning("Reconciler: last_fill_ts speichern fehlgeschlagen: %s", e)

    def track_order(self, client_id: str, exchange_id: str, symbol: str,
                    side: str, price: float, qty: float):
        try:
            con = _get_conn()
            con.execute(
                "INSERT OR REPLACE INTO open_orders VALUES (?,?,?,?,?,?,?,?)",
                (client_id, exchange_id, symbol, side, price, qty, time.time(), "{}"),
            )
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("reconciler track_order failed: %s", e)

    def remove_order(self, client_id: str):
        try:
            con = _get_conn()
            con.execute("DELETE FROM open_orders WHERE client_id=?", (client_id,))
            con.commit()
            con.close()
        except Exception:
            pass

    def reconcile(self) -> List[Fill]:
        """Fetch fills since last reconciliation, resolving client_id from open_orders table.

        Fills mit nicht auflösbarer client_id werden NICHT still verworfen (#38):
        sie landen in der unresolved_fills-Dead-Letter-Tabelle und werden bei
        jedem reconcile() erneut gegen open_orders aufgelöst. last_fill_ts darf
        deshalb weiter vorschieben, ohne Fills zu verlieren.
        """
        since = self._last_fill_ts
        fills = self._fetch(since)
        resolved: List[Fill] = []
        if fills:
            # Resolve any fills that still have empty client_id from DB-persisted orders
            try:
                con = _get_conn()
                for fill in fills:
                    if not fill.client_id and fill.exchange_order_id:
                        row = con.execute(
                            "SELECT client_id FROM open_orders WHERE exchange_order_id=?",
                            (fill.exchange_order_id,)
                        ).fetchone()
                        if row:
                            fill.client_id = row["client_id"]
                con.close()
            except Exception as e:
                logger.warning("Reconciler: client_id-Auflösung fehlgeschlagen: %s", e)

            unresolved = 0
            for fill in fills:
                if fill.client_id:
                    resolved.append(fill)
                else:
                    self._dead_letter(fill)
                    unresolved += 1

            max_ts = max(f.ts for f in fills)
            self._last_fill_ts = max_ts + 1
            self._save_last_ts(self._last_fill_ts)
            logger.info("Reconciler: %d new fills since %.0f (%d unaufgelöst → Dead-Letter)",
                        len(fills), since, unresolved)

        resolved.extend(self._retry_dead_letters())
        return resolved

    def _dead_letter(self, fill: Fill):
        """Persistiert einen nicht zuordenbaren Fill statt ihn zu verlieren."""
        try:
            con = _get_conn()
            con.execute(
                "INSERT OR IGNORE INTO unresolved_fills VALUES (?,?,?,?,?,?,?,?)",
                (fill.exchange_order_id or f"unknown_{fill.ts}", fill.symbol, fill.side,
                 fill.price, fill.qty, fill.fee, fill.ts, time.time()),
            )
            con.commit()
            con.close()
            logger.warning(
                "Reconciler: Fill ohne client_id → Dead-Letter (%s %s %.6f @ %.4f, oid=%s)",
                fill.symbol, fill.side, fill.qty, fill.price, fill.exchange_order_id,
            )
        except Exception as e:
            logger.error("Reconciler: Dead-Letter-Persist fehlgeschlagen: %s", e)

    def _retry_dead_letters(self) -> List[Fill]:
        """Versucht geparkte Fills erneut aufzulösen; Erfolge werden emittiert."""
        out: List[Fill] = []
        try:
            con = _get_conn()
            rows = con.execute("SELECT * FROM unresolved_fills").fetchall()
            for r in rows:
                row = con.execute(
                    "SELECT client_id FROM open_orders WHERE exchange_order_id=?",
                    (r["exchange_order_id"],)
                ).fetchone()
                if not row:
                    continue
                out.append(Fill(
                    client_id=row["client_id"], symbol=r["symbol"], side=r["side"],
                    price=r["price"], qty=r["qty"], fee=r["fee"], ts=r["ts"],
                    exchange_order_id=r["exchange_order_id"],
                ))
                con.execute("DELETE FROM unresolved_fills WHERE exchange_order_id=?",
                            (r["exchange_order_id"],))
                logger.info("Reconciler: Dead-Letter-Fill aufgelöst (%s, oid=%s)",
                            r["symbol"], r["exchange_order_id"])
            con.commit()
            con.close()
        except Exception as e:
            logger.warning("Reconciler: Dead-Letter-Retry fehlgeschlagen: %s", e)
        return out

    def get_tracked_orders(self, symbol: Optional[str] = None) -> list:
        try:
            con = _get_conn()
            if symbol:
                rows = con.execute(
                    "SELECT * FROM open_orders WHERE symbol=?", (symbol,)
                ).fetchall()
            else:
                rows = con.execute("SELECT * FROM open_orders").fetchall()
            con.close()
            return [dict(r) for r in rows]
        except Exception:
            return []
