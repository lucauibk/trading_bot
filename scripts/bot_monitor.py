#!/usr/bin/env python3
"""
Bot-Monitor: läuft alle 6h via launchd, analysiert Logs + trades.db und
öffnet GitHub Issues für jede gefundene Auffälligkeit.

Kategorien:
  bug        – technische Fehler (API-Ausfälle, Exceptions, insufficient balance)
  risk       – Freeze, Stop-Loss, Drawdown
  performance – Win-Rate, Trade-Frequenz, Equity-Stagnation
  config     – Trend-Filter dauerhaft aktiv, Grid nie gefüllt

Keine Code-Änderungen, keine PRs — rein beobachtend.
Kein Duplicate-Spam: jeder Check-Typ darf max. 1 offenes Issue haben.
"""

import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [monitor] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / "bot_monitor.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("monitor")

DB_PATH   = ROOT / "data" / "trades.db"
LOG_PATH  = ROOT / "logs" / "trading_bot.log"
REPO      = "lucauibk/trading_bot"
WINDOW_H  = 6        # Analysefenster in Stunden (entspricht Run-Intervall)
MIN_TRADES_FOR_WINRATE = 5  # Mindest-Trades bevor Win-Rate bewertet wird


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _since() -> str:
    """UTC-Timestamp vor WINDOW_H Stunden als ISO-String."""
    return (datetime.now(timezone.utc) - timedelta(hours=WINDOW_H)).isoformat()


def _open_issues() -> dict[str, list]:
    """Gibt {title_keyword: [issue_number, ...]} für alle offenen Issues zurück."""
    result = subprocess.run(
        ["gh", "issue", "list", "--repo", REPO, "--state", "open",
         "--json", "number,title", "--limit", "100"],
        capture_output=True, text=True
    )
    mapping = defaultdict(list)
    if result.returncode == 0:
        for issue in json.loads(result.stdout or "[]"):
            mapping[issue["title"]].append(issue["number"])
    return mapping


def _create_issue(title: str, body: str, labels: list[str], open_titles: dict) -> None:
    if title in open_titles:
        log.info("Issue bereits offen, überspringe: %s", title)
        return
    label_args = []
    for l in labels:
        label_args += ["--label", l]
    result = subprocess.run(
        ["gh", "issue", "create", "--repo", REPO,
         "--title", title, "--body", body] + label_args,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log.info("Issue erstellt: %s → %s", title, result.stdout.strip())
    else:
        log.warning("Issue-Erstellung fehlgeschlagen: %s\n%s", title, result.stderr)


def _recent_log_lines() -> list[str]:
    """Liest Log-Zeilen der letzten WINDOW_H Stunden."""
    if not LOG_PATH.exists():
        return []
    cutoff = datetime.now() - timedelta(hours=WINDOW_H)
    lines = []
    with open(LOG_PATH, "r", errors="replace") as f:
        for line in f:
            # Format: "2026-06-17 20:01:20,849 [INFO] ..."
            m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    if ts >= cutoff:
                        lines.append(line.rstrip())
                except ValueError:
                    pass
    return lines


# ── Checks ───────────────────────────────────────────────────────────────────

def check_insufficient_balance(lines: list[str], open_issues: dict) -> None:
    hits = [l for l in lines if "insufficient balance" in l]
    if len(hits) < 3:
        return
    coins = set()
    for l in hits:
        m = re.search(r"for (\S+) buy", l)
        if m:
            coins.add(m.group(1))
    title = "[bug] PaperBroker: insufficient balance bei Grid-Buys"
    body = (
        f"In den letzten {WINDOW_H}h wurden **{len(hits)}× `insufficient balance`**-Warnungen geloggt "
        f"für Coins: `{', '.join(sorted(coins)) or 'unbekannt'}`.\n\n"
        "Das bedeutet, die Strategy versucht größere Orders zu platzieren als der per-Symbol "
        "Cash-Bucket erlaubt. Entweder stimmt `investment` vs. Broker-Bucket nicht, oder der "
        "Inventory-Cap blockiert nicht früh genug.\n\n"
        f"**Betroffene Log-Zeilen (erste 5):**\n```\n" +
        "\n".join(hits[:5]) + "\n```"
    )
    _create_issue(title, body, ["bug", "broker"], open_issues)


def check_api_errors(lines: list[str], open_issues: dict) -> None:
    api_errors = [l for l in lines if "on_candle failed" in l or "Ticker failed" in l
                  or "BTCContext fetch failed" in l]
    if len(api_errors) < 5:
        return
    # Häufigste betroffene Symbole
    coins = defaultdict(int)
    for l in api_errors:
        for sym in ["SOL", "ETH", "AVAX", "LINK", "XRP", "BTC"]:
            if sym in l:
                coins[sym] += 1
    top = sorted(coins.items(), key=lambda x: -x[1])[:3]
    title = "[bug] Kraken API: wiederkehrende Fetch-Fehler"
    body = (
        f"In den letzten {WINDOW_H}h: **{len(api_errors)} API-Fehler** beim Laden von OHLCV/Ticker-Daten.\n\n"
        f"Häufigste betroffene Coins: {', '.join(f'`{c}` ({n}×)' for c, n in top)}\n\n"
        "Mögliche Ursachen: Rate-Limit, Netzwerk-Ausfall, Kraken API-Downtime.\n\n"
        "**Impact:** Der Bot arbeitet mit veralteten Preisen/Vorhersagen wenn OHLCV-Daten fehlen.\n\n"
        f"**Erste 5 Einträge:**\n```\n" + "\n".join(api_errors[:5]) + "\n```"
    )
    _create_issue(title, body, ["bug", "api"], open_issues)


def check_freeze(lines: list[str], open_issues: dict) -> None:
    freeze_lines = [l for l in lines if "FREEZE: daily drawdown" in l]
    if not freeze_lines:
        return
    title = "[risk] Daily-Drawdown-Freeze ausgelöst"
    body = (
        f"Der Bot wurde in den letzten {WINDOW_H}h **{len(freeze_lines)}× eingefroren** "
        "(kein neues Kaufen).\n\n"
        "Das deutet auf einen >10% Equity-Rückgang seit Session-Start hin.\n\n"
        f"**Log:**\n```\n" + "\n".join(freeze_lines) + "\n```\n\n"
        "**Zu prüfen:** War es ein echtes Drawdown-Ereignis oder ein Konfigurationsproblem "
        "(z. B. zu enge Schwelle, fehlendes `initial_capital`)?"
    )
    _create_issue(title, body, ["risk"], open_issues)


def check_stop_losses(lines: list[str], open_issues: dict) -> None:
    sl_lines = [l for l in lines if "[SL]" in l]
    if len(sl_lines) < 2:
        return
    # Verluste summieren
    losses = []
    for l in sl_lines:
        m = re.search(r"net=([-\d.]+)", l)
        if m:
            losses.append(float(m.group(1)))
    total_loss = sum(losses)
    title = "[risk] Mehrere Stop-Loss-Fires in kurzer Zeit"
    body = (
        f"In den letzten {WINDOW_H}h haben **{len(sl_lines)} Stop-Loss-Trades** ausgelöst "
        f"(Gesamtverlust: `{total_loss:.4f} USDT`).\n\n"
        "Häufige SL-Fires deuten auf ein Grid hin, das zu nah am aktuellen Preis liegt, "
        "oder auf einen starken Downtrend.\n\n"
        f"**Log:**\n```\n" + "\n".join(sl_lines) + "\n```"
    )
    _create_issue(title, body, ["risk"], open_issues)


def check_trend_filter_stuck(lines: list[str], open_issues: dict) -> None:
    """Warnt wenn ein Coin seit >3h dauerhaft im Downtrend-Block hängt."""
    paused = defaultdict(int)
    resumed = set()
    for l in lines:
        m = re.search(r"\[TREND\] (\S+) hard downtrend → grid buys paused", l)
        if m:
            paused[m.group(1)] += 1
        m2 = re.search(r"\[TREND\] (\S+) downtrend cleared", l)
        if m2:
            resumed.add(m2.group(1))
    stuck = {c: n for c, n in paused.items() if c not in resumed and n >= 6}
    if not stuck:
        return
    title = "[performance] Trend-Filter blockiert Käufe dauerhaft"
    body = (
        f"Folgende Coins sind seit >3h im Trend-Filter-Block und kaufen nicht:\n\n"
        + "\n".join(f"- `{c}`: {n} Wiederholungen" for c, n in stuck.items()) + "\n\n"
        "Der Trend-Filter (`EMA9 < EMA21 < EMA50` oder ADX bearish) verhindert neue Buys. "
        "Bei sehr langen Blockaden entgehen dem Bot potenzielle Trades.\n\n"
        "**Zu prüfen:** Ist der Downtrend real (dann korrekt), oder schlägt der Filter zu "
        "sensibel an? Prüfe `trend_adx_min` in `GridParams`."
    )
    _create_issue(title, body, ["performance", "config"], open_issues)


def check_winrate(open_issues: dict) -> None:
    if not DB_PATH.exists():
        return
    con = _db()
    since = _since()
    try:
        row = con.execute(
            "SELECT COUNT(*) as total, "
            "COUNT(CASE WHEN pnl > 0 THEN 1 END) as wins, "
            "SUM(pnl) as total_pnl "
            "FROM trades WHERE timestamp >= ? AND reason != 'stop_loss'",
            (since,)
        ).fetchone()
    finally:
        con.close()
    if not row or row["total"] < MIN_TRADES_FOR_WINRATE:
        return
    win_rate = row["wins"] / row["total"]
    if win_rate >= 0.50:
        return
    title = "[performance] Win-Rate unter 50%"
    body = (
        f"In den letzten {WINDOW_H}h: **{row['total']} Trades**, Win-Rate **{win_rate*100:.1f}%**, "
        f"Total PnL: `{row['total_pnl']:.4f} USDT`.\n\n"
        "Eine Win-Rate < 50% bei Grid-Trading deutet auf ein schlecht kalibriertes Grid hin "
        "(Grid-Range zu eng, Fills ohne Profit-Margin, oder häufige Rebuilds die Positionen "
        "abreißen bevor sie ins Plus laufen).\n\n"
        "**Zu prüfen:** Sind Fees > Grid-Step? Wird das Grid zu häufig neu gebaut?"
    )
    _create_issue(title, body, ["performance"], open_issues)


def check_equity_stagnation(open_issues: dict) -> None:
    """Warnt wenn die Equity seit WINDOW_H Stunden nicht gestiegen ist (Bot idle)."""
    if not DB_PATH.exists():
        return
    con = _db()
    since = _since()
    try:
        rows = con.execute(
            "SELECT capital FROM equity WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since,)
        ).fetchall()
        bot_running = con.execute(
            "SELECT running FROM bot_status WHERE id=1"
        ).fetchone()
    finally:
        con.close()
    if not rows or len(rows) < 10:
        return
    if not bot_running or not bot_running["running"]:
        return
    first_eq = rows[0]["capital"]
    last_eq  = rows[-1]["capital"]
    max_eq   = max(r["capital"] for r in rows)
    # Equity hat sich um weniger als 0.5% geändert und kein neues High
    if abs(last_eq - first_eq) / max(first_eq, 1) < 0.005 and max_eq == first_eq:
        title = "[performance] Equity stagniert – keine Trades in letzten 6h"
        body = (
            f"Die Equity ist seit {WINDOW_H}h nicht gestiegen: "
            f"`{first_eq:.2f}` → `{last_eq:.2f}` USDT.\n\n"
            "Mögliche Ursachen:\n"
            "- Trend-Filter blockiert alle Coins\n"
            "- Bot eingefroren (Freeze)\n"
            "- Grid-Range zu weit (Buys füllen nie)\n"
            "- API-Fehler beim Preis-Fetch\n\n"
            "**Zu prüfen:** Logs auf FREEZE, TREND-Blocks und API-Fehler checken."
        )
        _create_issue(title, body, ["performance"], open_issues)


def check_inactive_coins(open_issues: dict) -> None:
    """Warnt wenn bestimmte Coins seit >12h keinen einzigen Trade hatten."""
    if not DB_PATH.exists():
        return
    con = _db()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    try:
        active_coins = {
            r["symbol"] for r in con.execute(
                "SELECT DISTINCT symbol FROM trades WHERE timestamp >= ?", (cutoff,)
            ).fetchall()
        }
        all_coins = {
            r["symbol"] for r in con.execute(
                "SELECT symbol FROM grid_state"
            ).fetchall()
        }
        running = con.execute("SELECT running FROM bot_status WHERE id=1").fetchone()
    finally:
        con.close()
    if not running or not running["running"]:
        return
    silent = all_coins - active_coins
    if not silent:
        return
    title = "[performance] Coins ohne Trade seit >12h"
    body = (
        f"Folgende Coins haben in den letzten 12h **keinen einzigen Trade** erzeugt:\n\n"
        + "\n".join(f"- `{c}`" for c in sorted(silent)) + "\n\n"
        "Mögliche Ursachen: Trend-Filter dauerhaft aktiv, Grid-Range zu weit, "
        "Coin disabled in coin_settings, oder Preis außerhalb des Grid-Bereichs.\n\n"
        "**Zu prüfen:** `grid_state` im Dashboard für diese Coins, Logs auf TREND-Blocks."
    )
    _create_issue(title, body, ["performance"], open_issues)


def check_large_single_loss(lines: list[str], open_issues: dict) -> None:
    """Einzelner Trade mit Verlust > 2 USDT."""
    for l in lines:
        if "stop_loss" not in l and "[SL]" not in l:
            continue
        m = re.search(r"net=([-\d.]+)", l)
        if m and float(m.group(1)) < -2.0:
            title = "[risk] Einzelner Stop-Loss-Verlust > 2 USDT"
            body = (
                f"Ein einzelner Stop-Loss-Trade hat mehr als 2 USDT verloren:\n\n"
                f"```\n{l}\n```\n\n"
                "Bei 60 USDT/Coin und 3× Leverage ist das ein Verlust von >3% auf den Coin-Bucket. "
                "Prüfe ob der Floor-SL (`floor_sl_atr_mult`) zu weit unter dem Grid-Boden liegt."
            )
            _create_issue(title, body, ["risk"], open_issues)
            break


def check_grid_rebuild_storm(lines: list[str], open_issues: dict) -> None:
    """Warnt wenn Grid öfter als alle 5 Minuten rebuildet wird (out-of-range storm)."""
    rebuild_times = []
    for l in lines:
        if "[GRID] Built" not in l:
            continue
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", l)
        if m:
            try:
                rebuild_times.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
    if len(rebuild_times) < 20:
        return
    # Prüfe ob im Schnitt < 5 Minuten zwischen Rebuilds
    total_minutes = (rebuild_times[-1] - rebuild_times[0]).total_seconds() / 60
    avg_min = total_minutes / len(rebuild_times)
    if avg_min > 5:
        return
    title = "[performance] Grid-Rebuild-Storm: Grid wird zu häufig neu gebaut"
    body = (
        f"In den letzten {WINDOW_H}h: **{len(rebuild_times)} Grid-Rebuilds** "
        f"(⌀ alle {avg_min:.1f} Minuten).\n\n"
        "Zu häufige Rebuilds bedeuten:\n"
        "- Offene Positionen werden durch `state.orders = dict(open_positions)` ständig neu sortiert\n"
        "- Unnötige Kraken-API-Calls (OHLCV-Fetch bei jedem Rebuild)\n"
        "- Preis springt ständig aus dem Grid-Bereich (`out_of_range`)\n\n"
        "**Zu prüfen:** Grid-Range ±9% zu eng für das aktuelle Volatilitäts-Regime?\n"
        "Erwäge `GRID_REBUILD_CYCLES` zu erhöhen oder Range zu verbreitern."
    )
    _create_issue(title, body, ["performance", "config"], open_issues)


def check_ml_fallback_rate(lines: list[str], open_issues: dict) -> None:
    """Warnt wenn >30% der ML-Predictions auf den Rule-Based-Fallback zurückfallen.

    Ein 34→16-Feature-Downgrade lässt das 34-Feature-Modell (hold, 0.0) liefern →
    alle Predictions landen im Fallback, ohne sichtbares Signal auf INFO-Level.
    Seit dem Fix wird 'falling back to 16-feature' als WARNING geloggt.
    """
    real_preds   = [l for l in lines if "→ UP" in l or "→ DOWN" in l or "→ NEUTRAL" in l]
    fallback_lines = [l for l in lines if "Fallback" in l and ("→ UP" in l or "→ DOWN" in l or "→ NEUTRAL" in l)]
    total = len(real_preds)
    n_fallback = len(fallback_lines)
    if total < 10:
        return  # Zu wenig Daten für eine sinnvolle Quote
    fallback_pct = n_fallback / total
    if fallback_pct < 0.30:
        return
    # Prüfe auf 34→16-Downgrade-Warnungen
    downgrade_warnings = [l for l in lines if "falling back to 16-feature" in l or "34-feature extraction failed" in l]
    title = "[bug] ML-Fallback-Quote > 30% – Modell läuft möglicherweise nicht"
    body = (
        f"In den letzten {WINDOW_H}h: **{n_fallback}/{total} Predictions ({fallback_pct*100:.0f}%) "
        f"nutzen den Rule-Based-Fallback** statt das LightGBM-Modell.\n\n"
        + (f"**{len(downgrade_warnings)} 34→16-Feature-Downgrade-Warnungen** geloggt.\n\n"
           if downgrade_warnings else "Keine expliziten Downgrade-Warnungen im Log.\n\n") +
        "Mögliche Ursachen:\n"
        "- `extract_all_features` wirft (34→16-Downgrade) → Modell gibt immer `(hold, 0.0)` zurück\n"
        "- Modell nicht bereit (`is_ready()` False) – zu wenig Trainings-Samples?\n"
        "- `blended_conf < 0.45` dauerhaft (Model zu unsicher)\n\n"
        "**Zu prüfen:** `grep 'falling back to 16' logs/trading_bot.log`, "
        "`ls -la data/models/`, `sqlite3 data/ml_training.db 'SELECT symbol, COUNT(*) FROM samples GROUP BY symbol'`"
    )
    _create_issue(title, body, ["bug", "performance"], open_issues)


def check_exceptions(lines: list[str], open_issues: dict) -> None:
    """Unerwartete Python-Exceptions im Log."""
    exc_lines = [l for l in lines if "Traceback" in l or "Exception" in l or "Error:" in l
                 and "[WARNING]" in l or "[ERROR]" in l]
    errors = [l for l in lines if "[ERROR]" in l]
    if len(errors) < 2:
        return
    title = "[bug] Unerwartete Fehler im Bot-Log"
    body = (
        f"In den letzten {WINDOW_H}h: **{len(errors)} [ERROR]-Einträge** im Log.\n\n"
        f"**Erste 10:**\n```\n" + "\n".join(errors[:10]) + "\n```"
    )
    _create_issue(title, body, ["bug"], open_issues)


# ── Haupt-Labels sicherstellen ────────────────────────────────────────────────

def _ensure_labels() -> None:
    existing = subprocess.run(
        ["gh", "label", "list", "--repo", REPO, "--json", "name"],
        capture_output=True, text=True
    )
    if existing.returncode != 0:
        return
    names = {l["name"] for l in json.loads(existing.stdout or "[]")}
    needed = {
        "bug":         ("d73a4a", "Ein Fehler im Code"),
        "risk":        ("e11d48", "Risiko-Management-Auffälligkeit"),
        "performance": ("0075ca", "Performance-Problem"),
        "config":      ("cfd3d7", "Konfiguration / Parameter"),
        "api":         ("e4e669", "API / Daten-Problem"),
        "broker":      ("f9d0c4", "Broker / Order-Ausführung"),
        "monitor":     ("bfd4f2", "Automatisch erstellt von bot_monitor.py"),
    }
    for name, (color, desc) in needed.items():
        if name not in names:
            subprocess.run(
                ["gh", "label", "create", name,
                 "--repo", REPO, "--color", color, "--description", desc],
                capture_output=True
            )


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== Bot-Monitor Start (Fenster: letzte %dh) ===", WINDOW_H)

    _ensure_labels()
    open_issues = _open_issues()
    log.info("%d offene Issues gefunden", sum(len(v) for v in open_issues.values()))

    lines = _recent_log_lines()
    log.info("%d Log-Zeilen im Analysefenster", len(lines))

    # Technische Fehler
    check_insufficient_balance(lines, open_issues)
    check_api_errors(lines, open_issues)
    check_exceptions(lines, open_issues)

    # Risiko
    check_freeze(lines, open_issues)
    check_stop_losses(lines, open_issues)
    check_large_single_loss(lines, open_issues)

    # Performance
    check_trend_filter_stuck(lines, open_issues)
    check_equity_stagnation(open_issues)
    check_inactive_coins(open_issues)
    check_winrate(open_issues)
    check_grid_rebuild_storm(lines, open_issues)
    check_ml_fallback_rate(lines, open_issues)

    log.info("=== Bot-Monitor fertig ===")


if __name__ == "__main__":
    main()
