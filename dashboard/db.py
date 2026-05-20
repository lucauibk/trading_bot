"""
SQLite-Datenbank für Trade-History und Bot-Status.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parents[1] / "data" / "trades.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    _init(con)
    return con


def _init(con: sqlite3.Connection):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT,
            symbol     TEXT,
            direction  TEXT,
            entry      REAL,
            exit       REAL,
            pnl        REAL,
            reason     TEXT,
            strategy   TEXT,
            mode       TEXT
        );

        CREATE TABLE IF NOT EXISTS equity (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            capital   REAL
        );

        CREATE TABLE IF NOT EXISTS bot_status (
            id            INTEGER PRIMARY KEY CHECK (id = 1),
            running       INTEGER DEFAULT 0,
            mode          TEXT    DEFAULT 'paper',
            strategy      TEXT    DEFAULT 'grid',
            capital       REAL    DEFAULT 0,
            updated_at    TEXT,
            session_start TEXT
        );

        INSERT OR IGNORE INTO bot_status (id) VALUES (1);

        CREATE TABLE IF NOT EXISTS grid_state (
            symbol         TEXT PRIMARY KEY,
            current_price  REAL,
            levels         TEXT,
            range_pct      REAL,
            investment     REAL,
            total_profit   REAL,
            trade_count    INTEGER,
            prediction     TEXT,
            updated_at     TEXT,
            predicted_low  REAL DEFAULT 0,
            predicted_high REAL DEFAULT 0,
            confidence     REAL DEFAULT 0,
            regime         TEXT DEFAULT ''
        );
    """)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS coin_settings (
            symbol         TEXT PRIMARY KEY,
            max_investment REAL DEFAULT 300.0,
            enabled        INTEGER DEFAULT 1,
            updated_at     TEXT
        );
    """)
    for col, coldef in [
        ("predicted_low",  "REAL DEFAULT 0"),
        ("predicted_high", "REAL DEFAULT 0"),
        ("confidence",     "REAL DEFAULT 0"),
        ("regime",         "TEXT DEFAULT ''"),
    ]:
        try:
            con.execute(f"ALTER TABLE grid_state ADD COLUMN {col} {coldef}")
        except Exception:
            pass
    con.commit()


def log_trade(symbol: str, direction: str, entry: float, exit_: float,
              pnl: float, reason: str, strategy: str = "grid", mode: str = "paper"):
    from datetime import datetime
    con = get_conn()
    con.execute(
        "INSERT INTO trades (timestamp,symbol,direction,entry,exit,pnl,reason,strategy,mode) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (datetime.utcnow().isoformat(), symbol, direction, entry, exit_, pnl, reason, strategy, mode)
    )
    con.commit()
    con.close()


def log_equity(capital: float):
    from datetime import datetime
    con = get_conn()
    con.execute("INSERT INTO equity (timestamp, capital) VALUES (?,?)",
                (datetime.utcnow().isoformat(), capital))
    con.commit()
    con.close()


def update_grid_state(symbol: str, current_price: float, orders: dict,
                      range_pct: float, investment: float,
                      total_profit: float, trade_count: int, prediction: str = "",
                      predicted_low: float = 0.0, predicted_high: float = 0.0,
                      confidence: float = 0.0, regime: str = ""):
    import json
    from datetime import datetime
    levels = [
        {
            "price":     price,
            "side":      o["side"],
            "filled":    o.get("filled", False),
            "bought_at": o.get("bought_at"),
        }
        for price, o in sorted(orders.items(), reverse=True)
    ]
    con = get_conn()
    con.execute(
        """INSERT INTO grid_state
               (symbol, current_price, levels, range_pct, investment,
                total_profit, trade_count, prediction, updated_at,
                predicted_low, predicted_high, confidence, regime)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(symbol) DO UPDATE SET
               current_price=excluded.current_price,
               levels=excluded.levels,
               range_pct=excluded.range_pct,
               investment=excluded.investment,
               total_profit=excluded.total_profit,
               trade_count=excluded.trade_count,
               prediction=excluded.prediction,
               updated_at=excluded.updated_at,
               predicted_low=excluded.predicted_low,
               predicted_high=excluded.predicted_high,
               confidence=excluded.confidence,
               regime=excluded.regime""",
        (symbol, current_price, json.dumps(levels), range_pct, investment,
         total_profit, trade_count, prediction, datetime.utcnow().isoformat(),
         predicted_low, predicted_high, confidence, regime)
    )
    con.commit()
    con.close()


def init_coin_settings(symbols_with_defaults: list):
    """Legt Standardwerte an für neue Symbole (überschreibt nichts)."""
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    con = get_conn()
    for symbol, default_investment in symbols_with_defaults:
        con.execute(
            "INSERT OR IGNORE INTO coin_settings (symbol, max_investment, enabled, updated_at) VALUES (?,?,1,?)",
            (symbol, default_investment, now)
        )
    con.commit()
    con.close()


def get_all_coin_settings() -> list:
    con = get_conn()
    rows = con.execute("SELECT * FROM coin_settings ORDER BY symbol").fetchall()
    con.close()
    return [dict(r) for r in rows]


def set_coin_setting(symbol: str, max_investment: float, enabled: int = 1):
    from datetime import datetime
    con = get_conn()
    con.execute(
        """INSERT INTO coin_settings (symbol, max_investment, enabled, updated_at)
           VALUES (?,?,?,?)
           ON CONFLICT(symbol) DO UPDATE SET
               max_investment=excluded.max_investment,
               enabled=excluded.enabled,
               updated_at=excluded.updated_at""",
        (symbol, max_investment, enabled, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()


def update_capital(capital: float):
    con = get_conn()
    con.execute("UPDATE bot_status SET capital=? WHERE id=1", (capital,))
    con.commit()
    con.close()


def set_status(running: bool, mode: str = "paper", strategy: str = "grid", capital: float = 0):
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    con = get_conn()
    if running:
        con.execute(
            "UPDATE bot_status SET running=?, mode=?, strategy=?, capital=?, updated_at=?, session_start=? WHERE id=1",
            (1, mode, strategy, capital, now, now)
        )
    else:
        con.execute(
            "UPDATE bot_status SET running=0, updated_at=? WHERE id=1",
            (now,)
        )
    con.commit()
    con.close()
