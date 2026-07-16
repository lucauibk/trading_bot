#!/usr/bin/env bash
# watchdog.sh — hält den Bot (Paper-Modus) + Dashboard durchgehend am Laufen,
# damit kontinuierlich ML-Trainingsdaten gesammelt werden.
#
# Läuft als launchd-Job (com.tradingbot.watchdog, alle 5 Minuten) — nur wenn
# der Mac wach ist. Für 24/7-Betrieb ggf. `pmset repeat wakeorpoweron` setzen.
#
# Bewusster Stopp via ./stop.sh schreibt den Marker .bot.stopped —
# solange der existiert, startet der Watchdog nichts neu.
# ./start.sh entfernt den Marker wieder.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOT_PIDFILE="$ROOT/.bot.pid"
STOP_MARKER="$ROOT/.bot.stopped"
LOG="$ROOT/logs/watchdog.log"

mkdir -p "$ROOT/logs"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# Bewusst gestoppt → nichts tun
[ -f "$STOP_MARKER" ] && exit 0

bot_running() {
  if [ -f "$BOT_PIDFILE" ]; then
    local pid
    pid=$(cat "$BOT_PIDFILE")
    kill -0 "$pid" 2>/dev/null && return 0
  fi
  return 1
}

RESTART_MARKER="$ROOT/.watchdog.lastrestart"

if bot_running; then
  # Hang-Erkennung: Prozess lebt, aber die Loop schreibt keine Equity mehr
  # (health_check.py prueft Equity-Alter, Bootstrap-Grace und Internet).
  "${PYTHON:-python3}" "$ROOT/scripts/health_check.py" --check >> "$LOG" 2>&1
  HC=$?
  if [ "$HC" -ne 1 ]; then
    exit 0   # gesund (0) oder Sonderfall — nichts tun
  fi
  # Kill-Loop-Schutz: max. 1 Hang-Restart pro Stunde, sonst nur Alarm.
  if [ -f "$RESTART_MARKER" ] && [ $(( $(date +%s) - $(stat -c %Y "$RESTART_MARKER" 2>/dev/null || stat -f %m "$RESTART_MARKER") )) -lt 3600 ]; then
    log "Bot haengt erneut binnen 1h — KEIN weiterer Kill (systematisches Problem?)"
    cd "$ROOT" && "${PYTHON:-python3}" - <<'PY' >> "$LOG" 2>&1
try:
    from notifier import _send
    _send("🚨 Watchdog: Bot hängt wiederholt (2× binnen 1h) — kein Auto-Restart mehr, bitte anschauen!")
except Exception as e:
    print(f"Telegram-Notify fehlgeschlagen: {e}")
PY
    exit 0
  fi
  log "Bot haengt (Equity stale, PID $(cat "$BOT_PIDFILE")) — kill + restart"
  kill -9 "$(cat "$BOT_PIDFILE")" 2>/dev/null
  rm -f "$BOT_PIDFILE"
  touch "$RESTART_MARKER"
  # faellt durch in den Restart-Pfad unten
fi

log "Bot laeuft nicht — starte Dashboard + Bot (paper) neu via start.sh --bot"
if "$ROOT/start.sh" --bot --no-browser >> "$LOG" 2>&1; then
  log "Restart ausgeloest (Bot-PID: $(cat "$BOT_PIDFILE" 2>/dev/null || echo '?'))"
  # Telegram-Notify über bestehenden Notifier (best effort).
  # ${PYTHON:-python3}: auf dem Pi liegt der Stack im venv — System-python3
  # hat kein dotenv/requests.
  cd "$ROOT" && "${PYTHON:-python3}" - <<'PY' >> "$LOG" 2>&1
try:
    from notifier import _send
    _send("🐶 Watchdog: Bot war down und wurde im Paper-Modus neu gestartet.")
except Exception as e:
    print(f"Telegram-Notify fehlgeschlagen: {e}")
PY
else
  log "FEHLER: start.sh --bot fehlgeschlagen — siehe oben"
fi
