import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

DB_PATH = Path("data/ml_training.db")


class MLDataStore:
    def __init__(self):
        DB_PATH.parent.mkdir(exist_ok=True)
        with sqlite3.connect(DB_PATH) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT    NOT NULL,
                    timestamp   INTEGER NOT NULL,
                    features    TEXT    NOT NULL,
                    entry_price REAL    NOT NULL,
                    label       INTEGER,
                    predicted   INTEGER,
                    UNIQUE(symbol, timestamp)
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_sym_ts ON samples(symbol, timestamp)")
            c.commit()

    def store(self, symbol: str, ts: int, features: np.ndarray, entry_price: float, predicted: int):
        with sqlite3.connect(DB_PATH) as c:
            c.execute(
                "INSERT OR IGNORE INTO samples "
                "(symbol, timestamp, features, entry_price, predicted) VALUES (?,?,?,?,?)",
                (symbol, ts, json.dumps(features.tolist()), entry_price, predicted),
            )
            c.commit()

    def set_label(self, symbol: str, ts: int, label: int):
        with sqlite3.connect(DB_PATH) as c:
            c.execute(
                "UPDATE samples SET label=? WHERE symbol=? AND timestamp=?",
                (label, symbol, ts),
            )
            c.commit()

    def get_unlabeled_before(self, before_ts: int) -> List[Tuple[str, int, float]]:
        """Returns [(symbol, timestamp, entry_price), ...] for samples without a label."""
        with sqlite3.connect(DB_PATH) as c:
            return c.execute(
                "SELECT symbol, timestamp, entry_price FROM samples "
                "WHERE label IS NULL AND timestamp <= ?",
                (before_ts,),
            ).fetchall()

    def get_labeled(self, symbol: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        q = "SELECT features, label FROM samples WHERE label IS NOT NULL"
        p: list = []
        if symbol:
            q += " AND symbol=?"
            p.append(symbol)
        q += " ORDER BY timestamp"
        with sqlite3.connect(DB_PATH) as c:
            rows = c.execute(q, p).fetchall()
        if not rows:
            return np.empty((0, 16), np.float32), np.empty(0, np.int32)
        X = np.array([json.loads(r[0]) for r in rows], np.float32)
        y = np.array([r[1] for r in rows], np.int32)
        return X, y

    def count_labeled(self, symbol: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) FROM samples WHERE label IS NOT NULL"
        p: list = []
        if symbol:
            q += " AND symbol=?"
            p.append(symbol)
        with sqlite3.connect(DB_PATH) as c:
            return c.execute(q, p).fetchone()[0]

    def count_new_labeled_since(self, since_ts: int) -> int:
        with sqlite3.connect(DB_PATH) as c:
            return c.execute(
                "SELECT COUNT(*) FROM samples WHERE label IS NOT NULL AND timestamp > ?",
                (since_ts,),
            ).fetchone()[0]
