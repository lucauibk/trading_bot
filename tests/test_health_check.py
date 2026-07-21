"""
Regression test for the UTC-vs-local timezone bug found 2026-07-21:
dashboard/db.py.log_equity() stores datetime.utcnow(), but
_last_equity_age() compared it against datetime.now() (local) — during
BST (UTC+1) this inflated every reading by ~3600s, exceeding MAX_AGE=600s
and making the watchdog kill+restart a perfectly healthy bot roughly
hourly. Skipped entirely outside UTC-offset timezones, so this test forces
the DB timestamp to "now" and asserts the computed age stays small
regardless of the machine's local timezone.
"""

import importlib
import sqlite3
from datetime import datetime


def test_last_equity_age_uses_utc_not_local(tmp_path, monkeypatch):
    db_path = tmp_path / "trades.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE equity (id INTEGER PRIMARY KEY, timestamp TEXT, capital REAL)")
    con.execute("INSERT INTO equity (timestamp, capital) VALUES (?, ?)",
                (datetime.utcnow().isoformat(), 500.0))
    con.commit()
    con.close()

    import scripts.health_check as health_check
    importlib.reload(health_check)
    monkeypatch.setattr(health_check, "DB_PATH", db_path)

    age = health_check._last_equity_age()
    assert age is not None
    # A UTC/local mix-up shows up as ~3600s (or more) off; a correct
    # comparison against a just-inserted row should be a few seconds at most.
    assert age < 60, f"equity age should be near-zero, got {age}s — UTC/local mismatch?"
