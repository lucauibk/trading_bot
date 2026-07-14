#!/usr/bin/env bash
# stop.sh — beendet Bot und Dashboard sauber.

ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$ROOT/.dashboard.pid"
BOT_PIDFILE="$ROOT/.bot.pid"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[stop]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

stop_pid() {
  local name="$1" pidfile="$2"
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      # Auf ECHTES Prozess-Ende warten: der Graceful-Shutdown beendet erst
      # seine laufende Loop-/Bootstrap-Iteration (30-60s) und haelt solange
      # den Singleton-Lock. Ohne Warten startet ein direkt folgendes
      # start.sh in die Lock-Kollision und der neue Bot stirbt leise
      # (Befund 2026-07-14).
      local waited=0
      while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 60 ]; do
        sleep 1; waited=$((waited + 1))
      done
      if kill -0 "$pid" 2>/dev/null; then
        warn "$name reagiert nach ${waited}s nicht auf SIGTERM — SIGKILL."
        kill -9 "$pid" 2>/dev/null
        sleep 1
      fi
      info "$name gestoppt (PID $pid, nach ${waited}s beendet)."
    else
      warn "$name war nicht mehr aktiv."
    fi
    rm -f "$pidfile"
  else
    # Fallback: via Port-Suche (Dashboard)
    if [ "$name" = "Dashboard" ]; then
      local port_pid
      port_pid=$(lsof -ti :5001 2>/dev/null || true)
      if [ -n "$port_pid" ]; then
        kill "$port_pid" && info "Dashboard per Port gestoppt."
      else
        warn "Dashboard läuft nicht."
      fi
    else
      warn "$name läuft nicht (kein PID-File)."
    fi
  fi
}

# Marker für den Watchdog: bewusst gestoppt → nicht automatisch neu starten
touch "$ROOT/.bot.stopped"

stop_pid "Bot"       "$BOT_PIDFILE"
stop_pid "Dashboard" "$PIDFILE"

echo -e "${GREEN}Alle Prozesse gestoppt.${NC}"
