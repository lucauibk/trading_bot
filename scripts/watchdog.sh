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

if bot_running; then
  exit 0
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
