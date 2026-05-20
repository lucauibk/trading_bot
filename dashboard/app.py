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

    if _bot_process and _bot_process.poll() is None:
        return jsonify({"ok": False, "msg": "Bot läuft bereits"})

    data     = request.get_json() or {}
    strategy = data.get("strategy", "grid")
    mode     = data.get("mode", "paper")

    if mode == "live":
        # Live-Modus: Bestätigung über Dashboard statt Terminal
        if not data.get("confirmed"):
            return jsonify({"ok": False, "confirm": True,
                            "msg": "Live Trading bestätigen?"})

    cmd = [sys.executable, "main.py", "--mode", mode,
           "--strategy", strategy, "--no-confirm"]
    _bot_process = subprocess.Popen(cmd, cwd=str(_ROOT))
    (_ROOT / ".bot.pid").write_text(str(_bot_process.pid))

    set_status(running=True, mode=mode, strategy=strategy)
    return jsonify({"ok": True,
                    "msg": f"Bot gestartet ({strategy.upper()}, {mode.upper()})"})


@app.route("/api/bot/stop", methods=["POST"])
def api_stop():
    global _bot_process

    if _bot_process and _bot_process.poll() is None:
        _bot_process.terminate()
        try:
            _bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _bot_process.kill()

    (_ROOT / ".bot.pid").unlink(missing_ok=True)
    set_status(running=False)
    return jsonify({"ok": True, "msg": "Bot gestoppt"})


@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    """Stoppt Bot + Dashboard komplett (nichts läuft danach im Hintergrund)."""
    global _bot_process

    if _bot_process and _bot_process.poll() is None:
        _bot_process.terminate()
        try:
            _bot_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _bot_process.kill()

    (_ROOT / ".bot.pid").unlink(missing_ok=True)
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


def _session_start():
    con = get_conn()
    row = con.execute("SELECT session_start FROM bot_status WHERE id=1").fetchone()
    con.close()
    return row["session_start"] if row else None


@app.route("/api/trades", methods=["GET"])
def api_trades():
    limit   = request.args.get("limit", 50, type=int)
    session = _session_start()
    con     = get_conn()
    if session:
        rows = con.execute(
            "SELECT * FROM trades WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
            (session, limit)
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/equity", methods=["GET"])
def api_equity():
    session = _session_start()
    con     = get_conn()
    if session:
        rows = con.execute(
            "SELECT timestamp, capital FROM equity WHERE timestamp >= ? ORDER BY id DESC LIMIT 200",
            (session,)
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
        result.append(d)
    return jsonify(result)


@app.route("/api/summary", methods=["GET"])
def api_summary():
    session = _session_start()
    con     = get_conn()
    if session:
        total = con.execute("SELECT COALESCE(SUM(pnl),0) FROM trades WHERE timestamp >= ?", (session,)).fetchone()[0]
        count = con.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ?", (session,)).fetchone()[0]
        wins  = con.execute("SELECT COUNT(*) FROM trades WHERE timestamp >= ? AND pnl > 0", (session,)).fetchone()[0]
    else:
        total = con.execute("SELECT COALESCE(SUM(pnl),0) FROM trades").fetchone()[0]
        count = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        wins  = con.execute("SELECT COUNT(*) FROM trades WHERE pnl > 0").fetchone()[0]
    con.close()
    return jsonify({"total_pnl": total, "trades": count,
                    "win_rate": f"{wins/count*100:.1f}%" if count else "–",
                    "today_pnl": total})


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


@app.route("/stream", methods=["GET"])
def stream():
    def generate():
        while True:
            con  = get_conn()
            row  = con.execute("SELECT * FROM bot_status WHERE id=1").fetchone()
            last = con.execute(
                "SELECT symbol, pnl, timestamp FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
            con.close()
            data = dict(row) if row else {}
            data["running"] = int(_is_running())
            yield f"data: {json.dumps({'status': data, 'last_trade': dict(last) if last else None})}\n\n"
            time.sleep(5)
    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
