#!/usr/bin/env python3
"""
Export trades.db to flat CSV/JSON — no dashboard detour needed.

Motivation: building nq-video-bot/ (a deliberately minimal SQLite trade log,
directly readable with the sqlite3 CLI) made it obvious how much friction
there is in getting a quick, portable view of this bot's trades — right now
that requires either the dashboard or a hand-rolled sqlite3 query. This is a
read-only, purely additive export helper alongside scripts/optimize.py.

Verwendung:
  python scripts/export_trades.py --format csv [--days 30] [--symbol SOL/USD]
  python scripts/export_trades.py --format json --out results/trades_export.json
"""

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
DB_TRADES = ROOT / "data" / "trades.db"


def _conn(db_path: Path = DB_TRADES) -> sqlite3.Connection:
    if not db_path.exists():
        print(f"[FEHLER] DB nicht gefunden: {db_path}")
        sys.exit(1)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def fetch_trades(con: sqlite3.Connection, days: int = None, symbol: str = None) -> list:
    query = "SELECT * FROM trades WHERE 1=1"
    params = []
    if days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        query += " AND timestamp >= ?"
        params.append(cutoff)
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY timestamp"
    return [dict(row) for row in con.execute(query, params).fetchall()]


def write_csv(rows: list, out_path: Path) -> None:
    if not rows:
        print("Keine Trades im gewählten Zeitraum/Symbol.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} Trades exportiert nach {out_path}")


def write_json(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"{len(rows)} Trades exportiert nach {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--days", type=int, default=None, help="Nur Trades der letzten N Tage")
    parser.add_argument("--symbol", default=None, help="Nur dieses Symbol (z.B. SOL/USD)")
    parser.add_argument("--out", default=None, help="Zieldatei (Default: results/trades_export.<ext>)")
    args = parser.parse_args()

    con = _conn()
    try:
        rows = fetch_trades(con, days=args.days, symbol=args.symbol)
    finally:
        con.close()

    ext = args.format
    out_path = Path(args.out) if args.out else ROOT / "results" / f"trades_export.{ext}"

    if args.format == "csv":
        write_csv(rows, out_path)
    else:
        write_json(rows, out_path)


if __name__ == "__main__":
    main()
