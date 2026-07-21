#!/usr/bin/env python3
"""
Health-Check: Erkennt einen HÄNGENDEN Bot (Prozess lebt, Loop steht) und
liefert den täglichen Status-Report via Telegram.

Liveness-Beweis: Der Bot schreibt alle ~15-20s eine Equity-Zeile in
data/trades.db. Versiegen die Einträge, obwohl der Prozess lebt und das
Internet erreichbar ist, hängt die Loop (Deadlock, blockierter API-Call).

Modi:
  --check         Für den Watchdog (alle 5 min). Still. Exit-Codes:
                    0 = gesund (oder offline/Bootstrap → kein Restart sinnvoll)
                    1 = hängend → Watchdog killt + restartet
                    2 = läuft nicht (PID tot/fehlt) → Watchdog-Standardpfad
  --daily-report  Telegram-Statusbericht (Cron, 1×/Tag 08:00).

Env:
  HEALTH_MAX_AGE    max. Alter der letzten Equity-Zeile in s (Default 600)
  HEALTH_BOOT_GRACE Bootstrap-Schonfrist nach Prozessstart in s (Default 900)
"""

from __future__ import annotations

import os
import socket
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DB_PATH = ROOT / "data" / "trades.db"
PID_FILE = ROOT / ".bot.pid"
WATCHDOG_LOG = ROOT / "logs" / "watchdog.log"

MAX_AGE = int(os.getenv("HEALTH_MAX_AGE", "600"))
BOOT_GRACE = int(os.getenv("HEALTH_BOOT_GRACE", "900"))


def _bot_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def _last_equity_age() -> float | None:
    """Sekunden seit der letzten Equity-Zeile; None wenn Tabelle leer/DB fehlt."""
    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
        row = con.execute("SELECT MAX(timestamp) FROM equity").fetchone()
        con.close()
        if not row or not row[0]:
            return None
        last = datetime.fromisoformat(row[0])
        # dashboard/db.py.log_equity() stores datetime.utcnow() — comparing
        # against local datetime.now() silently inflated every reading by
        # the local UTC offset (found 2026-07-21: +1h during BST, which
        # alone exceeds MAX_AGE=600s and made every check past boot-grace
        # report HÄNGEND regardless of actual bot health — see the ~hourly
        # watchdog restarts in logs/watchdog.log around that date).
        return (datetime.utcnow() - last).total_seconds()
    except Exception:
        return None


def _internet_ok(host: str = "api.kraken.com", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _proc_age() -> float:
    """Sekunden seit Bot-Start (Näherung: mtime des PID-Files, das start.sh
    beim Start schreibt)."""
    try:
        return time.time() - PID_FILE.stat().st_mtime
    except OSError:
        return 0.0


def check() -> int:
    if _bot_pid() is None:
        return 2

    age = _last_equity_age()
    if age is not None and age <= MAX_AGE:
        return 0

    if _proc_age() < BOOT_GRACE:
        # Bootstrap (BTC-History-Download etc.) schreibt noch keine Equity.
        print(f"health: equity stale ({age}), aber Prozess erst "
              f"{_proc_age():.0f}s alt (Bootstrap-Grace) — kein Restart")
        return 0

    if not _internet_ok():
        # _log_equity überspringt Schreibungen bei stale prices — Equity-Stille
        # bei Netz-Ausfall heißt NICHT hängend; ein Restart brächte nichts.
        print("health: equity stale, aber Internet offline — kein Restart")
        return 0

    print(f"health: HÄNGEND — letzte Equity vor {age:.0f}s (> {MAX_AGE}s), "
          f"Prozess lebt, Internet ok")
    return 1


def daily_report() -> None:
    pid = _bot_pid()
    age = _last_equity_age()
    healthy = pid is not None and age is not None and age <= MAX_AGE

    lines = []
    if healthy:
        uptime_h = _proc_age() / 3600
        lines.append(f"✅ Daily-Check: Bot läuft (seit {uptime_h:.1f}h)")
    elif pid is None:
        lines.append("⚠️ Daily-Check: Bot-Prozess läuft NICHT")
    else:
        lines.append(f"⚠️ Daily-Check: Bot hängt womöglich "
                     f"(letzte Equity vor {age:.0f}s)" if age is not None
                     else "⚠️ Daily-Check: keine Equity-Daten")

    try:
        con = sqlite3.connect(DB_PATH, timeout=10)
        cur = con.execute("SELECT capital FROM equity ORDER BY id DESC LIMIT 1").fetchone()
        equity_now = float(cur[0]) if cur else 0.0
        # Same UTC-vs-local fix as _last_equity_age(): equity.timestamp is UTC.
        yday = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        cur = con.execute(
            "SELECT capital FROM equity WHERE timestamp >= ? ORDER BY id LIMIT 1",
            (yday,)).fetchone()
        equity_24h = float(cur[0]) if cur else equity_now
        cur = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM trades "
            "WHERE timestamp >= ? AND pnl IS NOT NULL", (yday,)).fetchone()
        n_trades, pnl_24h = int(cur[0]), float(cur[1])
        con.close()
        delta = equity_now - equity_24h
        lines.append(f"💰 Equity: {equity_now:.2f} USDT ({delta:+.2f} / 24h)")
        lines.append(f"📊 Trades/24h: {n_trades} (PnL {pnl_24h:+.2f})")
        if age is not None:
            lines.append(f"⏱ Letzte Equity-Schreibung: vor {age:.0f}s")
    except Exception as e:
        lines.append(f"(DB-Auswertung fehlgeschlagen: {e})")

    # Watchdog-Restarts der letzten 24h zählen
    try:
        cutoff = datetime.now() - timedelta(hours=24)
        restarts = 0
        for line in WATCHDOG_LOG.read_text().splitlines():
            if "Restart ausgeloest" in line or "haengt" in line:
                try:
                    ts = datetime.fromisoformat(line[:19].replace(" ", "T"))
                    if ts >= cutoff:
                        restarts += 1
                except ValueError:
                    pass
        if restarts:
            lines.append(f"🐶 Watchdog-Restarts/24h: {restarts}")
    except OSError:
        pass

    msg = "\n".join(lines)
    print(msg)
    try:
        from notifier import _send
        _send(msg)
    except Exception as e:
        print(f"Telegram-Notify fehlgeschlagen: {e}")


if __name__ == "__main__":
    if "--daily-report" in sys.argv:
        daily_report()
        sys.exit(0)
    sys.exit(check())
