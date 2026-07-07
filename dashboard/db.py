"""
SQLite-Datenbank für Trade-History und Bot-Status.
"""

import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).parents[1] / "data" / "trades.db"

# Schema-Init nur einmal pro Prozess – nicht bei jeder Connection das komplette
# executescript (CREATE TABLE + ALTER-Migrationen) laufen lassen (#41).
_INIT_LOCK = threading.Lock()
_INITIALIZED_PATH: str = ""


def get_conn() -> sqlite3.Connection:
    global _INITIALIZED_PATH
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    # WAL + busy_timeout: Bot- und Dashboard-Prozess schreiben parallel in
    # dieselbe DB. Ohne WAL/Timeout drohen "database is locked"-Fehler und
    # stille Write-Verluste bei Schreib-Kollisionen (#41).
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    if _INITIALIZED_PATH != str(DB_PATH):
        with _INIT_LOCK:
            if _INITIALIZED_PATH != str(DB_PATH):
                _init(con)
                _INITIALIZED_PATH = str(DB_PATH)
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

        CREATE TABLE IF NOT EXISTS trade_context (
            id              INTEGER PRIMARY KEY,
            trade_id        INTEGER REFERENCES trades(id),
            symbol          TEXT,
            ts              TEXT,
            atr_pct         REAL,
            rsi             REAL,
            ema9            REAL,
            ema21           REAL,
            macd_hist       REAL,
            bb_position     REAL,
            volume_ratio    REAL,
            regime          TEXT,
            ml_prediction   TEXT,
            ml_confidence   REAL,
            predicted_low   REAL,
            predicted_high  REAL,
            grid_level_idx  INTEGER,
            holding_seconds REAL,
            fees_usdt       REAL
        );

        CREATE TABLE IF NOT EXISTS grid_sessions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol             TEXT,
            started_at         TEXT,
            ended_at           TEXT,
            total_profit       REAL,
            n_trades           INTEGER,
            n_wins             INTEGER,
            max_drawdown       REAL,
            range_pct_avg      REAL,
            levels             INTEGER,
            initial_investment REAL,
            final_investment   REAL,
            exit_reason        TEXT
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol           TEXT,
            ts               TEXT,
            prediction       TEXT,
            confidence       REAL,
            predicted_low    REAL,
            predicted_high   REAL,
            realized_high_6h REAL,
            realized_low_6h  REAL,
            hit              INTEGER
        );

        CREATE TABLE IF NOT EXISTS optimizer_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT,
            regime      TEXT,
            params_json TEXT,
            score       REAL,
            daily_pct   REAL,
            max_dd      REAL,
            sample_size INTEGER,
            created_at  TEXT
        );
    """)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS mtf_state (
            symbol       TEXT PRIMARY KEY,
            daily_bias   TEXT DEFAULT 'neutral',
            confirmed_4h INTEGER DEFAULT 0,
            daily_adx    REAL DEFAULT 0,
            zone_low     REAL DEFAULT 0,
            zone_high    REAL DEFAULT 0,
            direction    TEXT DEFAULT '',
            target       REAL DEFAULT 0,
            updated_at   TEXT
        );
    """)
    for col, coldef in [
        ("predicted_low",  "REAL DEFAULT 0"),
        ("predicted_high", "REAL DEFAULT 0"),
        ("confidence",     "REAL DEFAULT 0"),
        ("regime",         "TEXT DEFAULT ''"),
        ("directional",    "TEXT DEFAULT '{}'"),
        ("floor_sl",       "REAL DEFAULT 0"),
    ]:
        try:
            con.execute(f"ALTER TABLE grid_state ADD COLUMN {col} {coldef}")
        except Exception:
            pass
    for col, coldef in [
        ("mtf_auto_execute", "INTEGER DEFAULT 0"),
    ]:
        try:
            con.execute(f"ALTER TABLE coin_settings ADD COLUMN {col} {coldef}")
        except Exception:
            pass
    for col, coldef in [
        ("leverage", "REAL DEFAULT 1.0"),
        ("initial_capital", "REAL DEFAULT 1000.0"),
    ]:
        try:
            con.execute(f"ALTER TABLE bot_status ADD COLUMN {col} {coldef}")
        except Exception:
            pass
        try:
            con.execute(f"ALTER TABLE trades ADD COLUMN {col} {coldef}")
        except Exception:
            pass
    for col, coldef in [
        ("stop_mode", "TEXT DEFAULT NULL"),
        ("stats_reset_at", "TEXT DEFAULT NULL"),
        ("frozen", "INTEGER DEFAULT 0"),
        ("frozen_reason", "TEXT DEFAULT NULL"),
        ("paper_balances", "TEXT DEFAULT NULL"),
    ]:
        try:
            con.execute(f"ALTER TABLE bot_status ADD COLUMN {col} {coldef}")
        except Exception:
            pass
    # predictions: add ML-score + entry_price so hit-eval is based on the ML
    # direction rather than PricePredictor bounds (which are a different system).
    for col, coldef in [
        ("ml_score",    "REAL DEFAULT NULL"),
        ("entry_price", "REAL DEFAULT NULL"),
    ]:
        try:
            con.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coldef}")
        except Exception:
            pass
    con.commit()


def set_frozen(frozen: bool, reason: str = None):
    con = get_conn()
    con.execute(
        "UPDATE bot_status SET frozen=?, frozen_reason=? WHERE id=1",
        (1 if frozen else 0, reason if frozen else None)
    )
    con.commit()
    con.close()


def log_trade(symbol: str, direction: str, entry: float, exit_: float,
              pnl: float, reason: str, strategy: str = "grid", mode: str = "paper",
              context: dict = None, leverage: float = 1.0) -> int:
    """Schreibt einen Trade und optional seinen Kontext. Gibt die Trade-ID zurück."""
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    con = get_conn()
    cur = con.execute(
        "INSERT INTO trades (timestamp,symbol,direction,entry,exit,pnl,reason,strategy,mode,leverage) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (now, symbol, direction, entry, exit_, pnl, reason, strategy, mode, leverage)
    )
    trade_id = cur.lastrowid
    if context:
        con.execute(
            """INSERT INTO trade_context
               (trade_id, symbol, ts, atr_pct, rsi, ema9, ema21, macd_hist,
                bb_position, volume_ratio, regime, ml_prediction, ml_confidence,
                predicted_low, predicted_high, grid_level_idx, holding_seconds, fees_usdt)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trade_id, symbol, now,
                context.get("atr_pct"), context.get("rsi"),
                context.get("ema9"), context.get("ema21"),
                context.get("macd_hist"), context.get("bb_position"),
                context.get("volume_ratio"), context.get("regime"),
                context.get("ml_prediction"), context.get("ml_confidence"),
                context.get("predicted_low"), context.get("predicted_high"),
                context.get("grid_level_idx"), context.get("holding_seconds"),
                context.get("fees_usdt"),
            )
        )
    con.commit()
    con.close()
    return trade_id


def log_prediction(symbol: str, prediction: str, confidence: float,
                   predicted_low: float, predicted_high: float,
                   ml_score: float = 0.0, entry_price: float = 0.0):
    """Speichert eine ML-Vorhersage für spätere Kalibrierungsauswertung.

    ml_score   – LGBM+LLM blend-score (-1..+1), dient der Hit-Auswertung.
    entry_price – aktueller Preis bei der Vorhersage (für realized-return-Hit).
    predicted_low/high – PricePredictor-Range (separates System, nur zur Info).
    """
    from datetime import datetime
    con = get_conn()
    con.execute(
        """INSERT INTO predictions
               (symbol, ts, prediction, confidence, predicted_low, predicted_high,
                ml_score, entry_price)
           VALUES (?,?,?,?,?,?,?,?)""",
        (symbol, datetime.utcnow().isoformat(), prediction, confidence,
         predicted_low, predicted_high, ml_score, entry_price)
    )
    con.commit()
    con.close()


def update_prediction_outcomes(fetch_ohlcv_fn):
    """
    Füllt realized_high_6h, realized_low_6h und hit für Predictions nach, die
    älter als 6h sind und noch kein Ergebnis haben. Muss periodisch aufgerufen werden.
    fetch_ohlcv_fn(symbol, timeframe, limit) → DataFrame mit high/low-Spalten.

    Hit-Definition (ML-direction-based):
      up      → Kurs stieg ≥0.3 % über entry_price im 6h-Fenster
      down    → Kurs fiel  ≥0.3 % unter entry_price im 6h-Fenster
      neutral → Kurs blieb innerhalb ±0.3 % um entry_price

    0.3 % ≈ Kraken-Round-Trip-Fee (2 × 0.16 %) → ein Hit bedeutet die Prediction
    war zumindest kostendeckend. Legacy-Zeilen ohne entry_price fallen auf die alte
    PricePredictor-Bound-Logik zurück.
    """
    HIT_THRESHOLD = 0.003  # 0.3 % round-trip-fee-Niveau
    from datetime import datetime, timedelta, timezone
    con = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    rows = con.execute(
        "SELECT id, symbol, ts, prediction, predicted_low, predicted_high, "
        "COALESCE(entry_price, 0.0) as entry_price "
        "FROM predictions WHERE hit IS NULL AND ts <= ?",
        (cutoff,)
    ).fetchall()

    for row in rows:
        pid, symbol, ts_str, pred, p_low, p_high, entry_price = row
        try:
            df = fetch_ohlcv_fn(symbol, "1h", 8)
            ts_pred = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts_pred.tzinfo is None:
                ts_pred = ts_pred.replace(tzinfo=timezone.utc)
            after = df[df.index >= ts_pred] if hasattr(df.index, "tz") else df.tail(6)
            if after.empty:
                after = df.tail(6)
            r_high = float(after["high"].max())
            r_low  = float(after["low"].min())

            if entry_price and entry_price > 0:
                # ML-direction hit: did price move ≥threshold in the predicted direction?
                up_barrier   = entry_price * (1 + HIT_THRESHOLD)
                down_barrier = entry_price * (1 - HIT_THRESHOLD)
                if pred == "up":
                    hit = 1 if r_high >= up_barrier else 0
                elif pred == "down":
                    hit = 1 if r_low <= down_barrier else 0
                else:  # neutral: price stayed in band
                    hit = 1 if (r_high < up_barrier and r_low > down_barrier) else 0
            else:
                # Legacy rows (no entry_price): fall back to PricePredictor-bound logic
                if pred == "up":
                    hit = 1 if (p_high and r_high >= p_high) else 0
                elif pred == "down":
                    hit = 1 if (p_low and r_low <= p_low) else 0
                else:
                    hit = 1 if (p_high and p_low and r_high <= p_high and r_low >= p_low) else 0

            con.execute(
                "UPDATE predictions SET realized_high_6h=?, realized_low_6h=?, hit=? WHERE id=?",
                (r_high, r_low, hit, pid)
            )
        except Exception:
            pass

    con.commit()
    con.close()


def log_optimizer_run(symbol: str, regime: str, params: dict, score: float,
                      daily_pct: float, max_dd: float, sample_size: int):
    import json
    from datetime import datetime
    con = get_conn()
    con.execute(
        """INSERT INTO optimizer_runs (symbol, regime, params_json, score, daily_pct, max_dd, sample_size, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (symbol, regime, json.dumps(params), score, daily_pct, max_dd, sample_size,
         datetime.utcnow().isoformat())
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
                      confidence: float = 0.0, regime: str = "",
                      directional: dict = None, floor_sl: float = 0.0):
    import json
    from datetime import datetime
    # Enrich per-level data: qty + sl_price + pre_seeded enable correct PnL
    # calculation and pre-seed filtering in the dashboard's open-positions table.
    levels = [
        {
            "price":      o["price"],
            "side":       o["side"],
            "filled":     o.get("filled", False),
            "bought_at":  o.get("bought_at"),
            "qty":        o.get("qty", 0.0),
            "sl_price":   o.get("sl_price"),
            "pre_seeded": o.get("pre_seeded", False),
        }
        for o in sorted(orders.values(), key=lambda x: x.get("price", 0), reverse=True)
        if isinstance(o.get("price"), (int, float))
    ]
    con = get_conn()
    con.execute(
        """INSERT INTO grid_state
               (symbol, current_price, levels, range_pct, investment,
                total_profit, trade_count, prediction, updated_at,
                predicted_low, predicted_high, confidence, regime, directional, floor_sl)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
               regime=excluded.regime,
               directional=excluded.directional,
               floor_sl=excluded.floor_sl""",
        (symbol, current_price, json.dumps(levels), range_pct, investment,
         total_profit, trade_count, prediction, datetime.utcnow().isoformat(),
         predicted_low, predicted_high, confidence, regime,
         json.dumps(directional or {}), floor_sl)
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


def get_leverage() -> float:
    con = get_conn()
    row = con.execute("SELECT leverage FROM bot_status WHERE id=1").fetchone()
    con.close()
    lev = float(row["leverage"]) if row and row["leverage"] else 1.0
    # Negative/abwegige Werte (z. B. direkter DB-Write) nie durchreichen –
    # ein negativer Hebel würde Buys die Balance GUTSCHREIBEN (#37).
    return lev if 1.0 <= lev <= 10.0 else 1.0


def set_leverage(value: float):
    value = max(1.0, min(3.0, float(value)))
    con = get_conn()
    con.execute("UPDATE bot_status SET leverage=? WHERE id=1", (value,))
    con.commit()
    con.close()


def get_initial_capital() -> float:
    """Startkapital für den nächsten Bot-Start (Paper-Modus). Fallback: 1000."""
    con = get_conn()
    row = con.execute("SELECT initial_capital FROM bot_status WHERE id=1").fetchone()
    con.close()
    return float(row["initial_capital"]) if row and row["initial_capital"] else 1000.0


def set_initial_capital(value: float):
    value = max(10.0, float(value))
    con = get_conn()
    con.execute("UPDATE bot_status SET initial_capital=? WHERE id=1", (value,))
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
            "UPDATE bot_status SET running=0, frozen=0, frozen_reason=NULL, updated_at=? WHERE id=1",
            (now,)
        )
    con.commit()
    con.close()


def set_stop_mode(mode):
    """Setzt den gewünschten Stopp-Modus ('sell_all', 'wait_fills', oder None zum Löschen)."""
    con = get_conn()
    con.execute("UPDATE bot_status SET stop_mode=? WHERE id=1", (mode,))
    con.commit()
    con.close()


def get_stop_mode():
    """Liest den aktuellen Stopp-Modus aus der DB."""
    con = get_conn()
    row = con.execute("SELECT stop_mode FROM bot_status WHERE id=1").fetchone()
    con.close()
    return row["stop_mode"] if row else None


def reset_stats():
    """Setzt den Reset-Zeitpunkt für die Dashboard-Statistik (Trades/PnL/Heute) auf jetzt.

    Die Trades bleiben in der DB erhalten — nur die Anzeige zählt ab diesem
    Zeitpunkt neu (siehe dashboard/app.py: /api/trades, /api/equity, /api/summary).
    """
    from datetime import datetime
    now = datetime.utcnow().isoformat()
    con = get_conn()
    con.execute("UPDATE bot_status SET stats_reset_at=? WHERE id=1", (now,))
    con.commit()
    con.close()
    return now


def get_stats_reset_at():
    con = get_conn()
    row = con.execute("SELECT stats_reset_at FROM bot_status WHERE id=1").fetchone()
    con.close()
    return row["stats_reset_at"] if row else None


def save_paper_balances(balances: dict) -> None:
    import json
    con = get_conn()
    con.execute("UPDATE bot_status SET paper_balances=? WHERE id=1", (json.dumps(balances),))
    con.commit()
    con.close()


def load_paper_balances():
    import json
    con = get_conn()
    row = con.execute("SELECT paper_balances FROM bot_status WHERE id=1").fetchone()
    con.close()
    if row and row["paper_balances"]:
        return json.loads(row["paper_balances"])
    return None


def update_mtf_state(symbol: str, bias: dict, setup: dict = None):
    """UPSERT den aktuellen MTF-Status (Bias + optionale Retest-Zone) für ein Symbol."""
    from datetime import datetime
    con = get_conn()
    zone_low  = setup["zone_low"]  if setup else 0.0
    zone_high = setup["zone_high"] if setup else 0.0
    direction = setup["direction"] if setup else ""
    target    = setup["target"]    if setup else 0.0
    con.execute(
        """INSERT INTO mtf_state
               (symbol, daily_bias, confirmed_4h, daily_adx,
                zone_low, zone_high, direction, target, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(symbol) DO UPDATE SET
               daily_bias=excluded.daily_bias,
               confirmed_4h=excluded.confirmed_4h,
               daily_adx=excluded.daily_adx,
               zone_low=excluded.zone_low,
               zone_high=excluded.zone_high,
               direction=excluded.direction,
               target=excluded.target,
               updated_at=excluded.updated_at""",
        (symbol, bias.get("daily_bias", "neutral"),
         1 if bias.get("confirmed_4h") else 0,
         bias.get("daily_adx", 0.0),
         zone_low, zone_high, direction, target,
         datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()


def get_mtf_auto_execute(symbol: str) -> bool:
    """Gibt zurück ob MTF-Auto-Execute für dieses Symbol aktiviert ist."""
    con = get_conn()
    row = con.execute(
        "SELECT mtf_auto_execute FROM coin_settings WHERE symbol=?", (symbol,)
    ).fetchone()
    con.close()
    return bool(row["mtf_auto_execute"]) if row and row["mtf_auto_execute"] is not None else False


def set_mtf_auto_execute(symbol: str, enabled: bool) -> None:
    from datetime import datetime
    con = get_conn()
    con.execute(
        """INSERT INTO coin_settings (symbol, max_investment, enabled, mtf_auto_execute, updated_at)
           VALUES (?, 300.0, 1, ?, ?)
           ON CONFLICT(symbol) DO UPDATE SET
               mtf_auto_execute=excluded.mtf_auto_execute,
               updated_at=excluded.updated_at""",
        (symbol, 1 if enabled else 0, datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()
