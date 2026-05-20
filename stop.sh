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
      kill "$pid" && info "$name gestoppt (PID $pid)."
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

stop_pid "Bot"       "$BOT_PIDFILE"
stop_pid "Dashboard" "$PIDFILE"

echo -e "${GREEN}Alle Prozesse gestoppt.${NC}"
