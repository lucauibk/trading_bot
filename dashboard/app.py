"""
Monitoring Dashboard für den Trading Bot.
Aufruf: python3 dashboard/app.py
Öffne: http://127.0.0.1:5001
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from flask import Flask, Response, render_template, jsonify, request
from dashboard.db import get_conn, set_status

app = Flask(__name__)
_bot_process: subprocess.Popen = None
_ROOT = Path(__file__).parents[1]


@app.after_request
def no_cache(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ── Bot starten / stoppen ─────────────────────────────────────────────────────

@app.route("/api/bot/start", methods=["POST"])
def api_start():
    global _bot_process

    # PID-File-bewusst prüfen (#162): ein via ./start.sh --bot extern gestarteter
    # Bot ist im Dashboard-Prozess als _bot_process=None sichtbar. Ein reiner
    # _bot_process-Guard griffe dann nicht → zweiter main.py würde gespawnt, .bot.pid
    # mit der PID des am Singleton-Lock scheiternden Kindprozesses überschrieben,
    # der echte Bot damit unkillbar. _is_running() liest zusätzlich .bot.pid.
    if _is_running():
        return jsonify({"ok": False, "msg": "Bot läuft bereits"})

    data = request.get_json() or {}
    mode = data.get("mode", "paper")

    if mode == "live":
        if not data.get("confirmed"):
            return jsonify({"ok": False, "confirm": True,
                            "msg": "Live Trading bestätigen?"})

    cmd = [sys.executable, "main.py", "--mode", mode, "--no-confirm"]
    _bot_process = subprocess.Popen(cmd, cwd=str(_ROOT))
    (_ROOT / ".bot.pid").write_text(str(_bot_process.pid))

    set_status(running=True, mode=mode, strategy="grid")
    return jsonify({"ok": True,
                    "msg": f"Bot gestartet ({mode.upper()})"})


def _stop_bot_process():
    """Beendet den Bot-Prozess – via Subprocess-Handle ODER PID-File (extern gestartet)."""
    global _bot_process

    # 1) Via Subprocess-Handle (Dashboard hat den Bot gestartet)
    if _bot_process and _bot_process.poll() is None:
        _bot_process.terminate()
        try:
            _bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _bot_process.kill()
        _bot_process = None

    # 2) Fallback: Bot wurde extern gestartet (z.B. ./start.sh --bot) → PID-File lesen
    pid_file = _ROOT / ".bot.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if pid != os.getpid():           # nie sich selbst killen
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(pid, 0)          # noch am Leben?
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass                     # bereits beendet
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        pid_file.unlink(missing_ok=True)


@app.route("/api/bot/stop", methods=["POST"])
def api_stop():
    _stop_bot_process()
    set_status(running=False)
    return jsonify({"ok": True, "msg": "Bot gestoppt"})


@app.route("/api/bot/stop-graceful", methods=["POST"])
def api_stop_graceful():
    """Sendet einen Graceful-Stop-Wunsch an den laufenden Bot via DB-Flag."""
    from dashboard.db import set_stop_mode
    mode = (request.get_json() or {}).get("mode", "sell_all")
    if mode not in {"sell_all", "wait_fills"}:
        return jsonify({"ok": False, "msg": "Ungültiger Modus"}), 400
    set_stop_mode(mode)
    msg = "Alle Positionen werden verkauft…" if mode == "sell_all" else "Sell-Only-Modus aktiv – Bot wartet auf Fills…"
    return jsonify({"ok": True, "msg": msg})


@app.route("/api/stats/reset", methods=["POST"])
def api_stats_reset():
    """Setzt Trade-Anzahl/Gesamt-PnL/Heute im Dashboard auf 0 zurück.

    Bewusst getrennt von Bot-Start/Restart (anders als die alte session_start-
    Logik) — die DB-Historie bleibt unangetastet, nur die Anzeige zählt neu.
    """
    from dashboard.db import reset_stats
    reset_stats()
    return jsonify({"ok": True, "msg": "Statistik zurückgesetzt"})


@app.route("/api/bot/restart", methods=["POST"])
def api_restart():
    data = request.get_json() or {}
    mode = data.get("mode")
    _stop_bot_process()
    set_status(running=False)
    time.sleep(1)
    if not mode:
        con = get_conn()
        row = con.execute("SELECT mode FROM bot_status LIMIT 1").fetchone()
        con.close()
        mode = row["mode"] if row and row["mode"] else "paper"
    cmd = [sys.executable, "main.py", "--mode", mode, "--no-confirm"]
    global _bot_process
    _bot_process = subprocess.Popen(cmd, cwd=str(_ROOT))
    (_ROOT / ".bot.pid").write_text(str(_bot_process.pid))
    set_status(running=True, mode=mode, strategy="grid")
    return jsonify({"ok": True, "msg": f"Bot neu gestartet ({mode.upper()})"})


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """Stoppt Bot + Dashboard komplett (nichts läuft danach im Hintergrund)."""
    _stop_bot_process()
    set_status(running=False)

    def _kill():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({"ok": True, "msg": "Dashboard wird beendet…"})


# ── Status prüfen (Prozess wirklich aktiv?) ───────────────────────────────────

def _is_running() -> bool:
    if _bot_process is not None and _bot_process.poll() is None:
        return True
    # Fallback: PID-File prüfen (Bot via start.sh gestartet)
    pid_file = _ROOT / ".bot.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Signal 0 = nur prüfen ob Prozess existiert
            return True
        except (ProcessLookupError, ValueError):
            pid_file.unlink(missing_ok=True)
    return False


# ── Daten-Endpunkte ───────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    con = get_conn()
    row = con.execute("SELECT * FROM bot_status WHERE id=1").fetchone()
    con.close()
    data = dict(row) if row else {}
    data["running"] = int(_is_running())
    return jsonify(data)


def _stats_reset_at():
    from dashboard.db import get_stats_reset_at
    return get_stats_reset_at()


@app.route("/api/trades", methods=["GET"])
def api_trades():
    limit    = request.args.get("limit", 50, type=int)
    reset_at = _stats_reset_at()
    con      = get_conn()
    if reset_at:
        rows = con.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
            (reset_at, limit)
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/equity", methods=["GET"])
def api_equity():
    reset_at = _stats_reset_at()
    con      = get_conn()
    if reset_at:
        rows = con.execute(
            "SELECT timestamp, capital FROM equity WHERE timestamp >= ? ORDER BY id DESC LIMIT 200",
            (reset_at,)
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT timestamp, capital FROM equity ORDER BY id DESC LIMIT 200"
        ).fetchall()
    con.close()
    return jsonify(list(reversed([dict(r) for r in rows])))


@app.route("/api/grids", methods=["GET"])
def api_grids():
    import json
    con  = get_conn()
    rows = con.execute("SELECT * FROM grid_state ORDER BY symbol").fetchall()
    con.close()
    result = []
    for r in rows:
        d = dict(r)
        d["levels"] = json.loads(d["levels"]) if d["levels"] else []
        d["directional"] = json.loads(d["directional"]) if d.get("directional") else {}
        result.append(d)
    return jsonify(result)


@app.route("/api/summary", methods=["GET"])
def api_summary():
    from datetime import datetime, timezone
    today_str = datetime.now(timezone.utc).date().isoformat()  # 'YYYY-MM-DD'
    reset_at  = _stats_reset_at()
    con = get_conn()
    if reset_at:
        total = con.execute("SELECT COALESCE(SUM(pnl),0) FROM trades WHERE timestamp >= ?", (reset_at,)).fetchone()[0]
        count = con.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (reset_at,)).fetchone()[0]
        wins  = con.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ? AND pnl > 0", (reset_at,)).fetchone()[0]
    else:
        total = con.execute("SELECT COALESCE(SUM(pnl),0) FROM trades").fetchone()[0]
        count = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        wins  = con.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0").fetchone()[0]
    # today_pnl: since midnight UTC, or since reset if that happened later today
    # (ISO timestamp comparison is lexicographic, so max() works as a string op)
    today_cutoff = max(today_str, reset_at) if reset_at else today_str
    today_pnl = con.execute(
        "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE timestamp >= ?", (today_cutoff,)
    ).fetchone()[0]
    con.close()
    return jsonify({"total_pnl": total, "trades": count,
                    "win_rate": f"{wins/count*100:.1f}%" if count else "–",
                    "today_pnl": today_pnl})


@app.route("/api/leverage", methods=["GET"])
def api_leverage_get():
    from dashboard.db import get_leverage
    return jsonify({"leverage": get_leverage()})


@app.route("/api/leverage", methods=["POST"])
def api_leverage_set():
    from dashboard.db import set_leverage, get_leverage
    data = request.get_json() or {}
    try:
        val = float(data.get("leverage", 1.0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "ungültiger leverage"}), 400
    set_leverage(val)
    # set_leverage() clamps to [1.0, 3.0]; echo the persisted value back so the
    # dashboard never shows a leverage the bot isn't actually running with (#150).
    actual = get_leverage()
    return jsonify({"ok": True, "leverage": actual, "msg": f"Hebel auf {actual:.1f}× gesetzt"})


@app.route("/api/capital", methods=["GET"])
def api_capital_get():
    from dashboard.db import get_initial_capital
    return jsonify({"initial_capital": get_initial_capital()})


@app.route("/api/capital", methods=["POST"])
def api_capital_set():
    from dashboard.db import set_initial_capital
    data = request.get_json() or {}
    try:
        val = float(data.get("initial_capital", 0))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "Ungültiger Wert"}), 400
    if val < 10:
        return jsonify({"ok": False, "msg": "Minimum 10 USDT"})
    set_initial_capital(val)
    return jsonify({"ok": True, "initial_capital": val,
                    "msg": f"Startkapital auf {val:.0f} USDT gesetzt (wirkt beim nächsten Bot-Start)"})


@app.route("/api/coin-settings", methods=["GET"])
def api_coin_settings_get():
    from dashboard.db import get_all_coin_settings
    rows = get_all_coin_settings()
    # Fallback: falls noch keine Einträge, grid_state-Symbole zurückgeben
    if not rows:
        con = get_conn()
        gs = con.execute("SELECT symbol FROM grid_state ORDER BY symbol").fetchall()
        con.close()
        rows = [{"symbol": r["symbol"], "max_investment": 300.0, "enabled": 1} for r in gs]
    return jsonify(rows)


@app.route("/api/coin-settings", methods=["POST"])
def api_coin_settings_post():
    from dashboard.db import set_coin_setting
    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"ok": False, "msg": "Symbol fehlt"})
    max_inv = float(data.get("max_investment", 300))
    enabled = int(data.get("enabled", 1))
    if max_inv < 10:
        return jsonify({"ok": False, "msg": "Minimum 10 USDT"})
    set_coin_setting(symbol, max_inv, enabled)
    return jsonify({"ok": True, "msg": f"{symbol} gespeichert: {max_inv:.0f} USDT"})


@app.route("/api/coin-settings/mtf", methods=["POST"])
def api_coin_mtf_post():
    from dashboard.db import set_mtf_auto_execute
    data = request.get_json() or {}
    symbol = data.get("symbol", "").strip().upper()
    if not symbol:
        return jsonify({"ok": False, "msg": "Symbol fehlt"})
    enabled = bool(data.get("mtf_auto_execute", False))
    set_mtf_auto_execute(symbol, enabled)
    state = "aktiviert" if enabled else "deaktiviert"
    return jsonify({"ok": True, "msg": f"MTF Auto-Execute {state} für {symbol}"})


@app.route("/stream", methods=["GET"])
def stream():
    def generate():
        while True:
            con = get_conn()
            try:
                row  = con.execute("SELECT * FROM bot_status WHERE id=1").fetchone()
                last = con.execute(
                    "SELECT symbol, pnl, timestamp FROM trades ORDER BY id DESC LIMIT 1"
                ).fetchone()
            finally:
                con.close()
            data = dict(row) if row else {}
            data["running"] = int(_is_running())
            yield f"data: {json.dumps({'status': data, 'last_trade': dict(last) if last else None})}\n\n"
            time.sleep(5)
    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
