#!/usr/bin/env bash
# start.sh -- startet Dashboard (+ optional Bot) falls noch nicht laufend.
#
# Verwendung:
#   ./start.sh           -> nur Dashboard starten (Bot dann per Browser steuern)
#   ./start.sh --bot     -> Dashboard + Bot im Paper-Modus
#   ./start.sh --live    -> Dashboard + Bot im Live-Modus

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
DASHBOARD_PORT=5001
PIDFILE="$ROOT/.dashboard.pid"
BOT_PIDFILE="$ROOT/.bot.pid"
LOG_DIR="$ROOT/logs"

cd "$ROOT"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${GREEN}[start]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()  { echo -e "${RED}[error]${NC} $*"; exit 1; }

# -- Argumente --
START_BOT=false
BOT_MODE="paper"
OPEN_BROWSER=true
for arg in "$@"; do
  case $arg in
    --bot)  START_BOT=true ;;
    --live) START_BOT=true; BOT_MODE="live" ;;
    --no-browser) OPEN_BROWSER=false ;;
  esac
done

# -- Voraussetzungen --
command -v "$PYTHON" >/dev/null 2>&1 || err "python3 nicht gefunden."
[ -f ".env" ] || err ".env fehlt. Kopiere .env.example und trage deine Keys ein."

if [ "$BOT_MODE" = "live" ]; then
  source .env 2>/dev/null || true
  [ -n "${KRAKEN_API_KEY:-}" ] || err "KRAKEN_API_KEY fehlt in .env"
  [ -n "${KRAKEN_API_SECRET:-}" ] || err "KRAKEN_API_SECRET fehlt in .env"
fi

mkdir -p "$LOG_DIR"

# Watchdog-Marker entfernen: Bot soll (wieder) laufen
rm -f "$ROOT/.bot.stopped"

# -- Dashboard starten falls nicht schon laufend --
dashboard_running() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid=$(cat "$PIDFILE")
    kill -0 "$pid" 2>/dev/null && return 0
    rm -f "$PIDFILE"
  fi
  lsof -ti ":$DASHBOARD_PORT" >/dev/null 2>&1
}

if dashboard_running; then
  warn "Dashboard laeuft bereits auf http://localhost:$DASHBOARD_PORT"
else
  info "Starte Dashboard auf Port $DASHBOARD_PORT ..."
  nohup "$PYTHON" dashboard/app.py \
    > "$LOG_DIR/dashboard.log" 2>&1 &
  echo $! > "$PIDFILE"
  info "Dashboard-PID: $(cat "$PIDFILE") | Log: logs/dashboard.log"

  for i in 1 2 3 4 5 6 7 8; do
    sleep 1
    if lsof -ti ":$DASHBOARD_PORT" >/dev/null 2>&1; then
      info "Dashboard bereit."
      break
    fi
    if [ "$i" = "8" ]; then
      warn "Dashboard antwortet noch nicht -- pruefen: tail -f logs/dashboard.log"
    fi
  done
fi

# -- Bot starten falls gewuenscht --
bot_running() {
  if [ -f "$BOT_PIDFILE" ]; then
    local pid
    pid=$(cat "$BOT_PIDFILE")
    kill -0 "$pid" 2>/dev/null && return 0
    rm -f "$BOT_PIDFILE"
  fi
  return 1
}

if $START_BOT; then
  if bot_running; then
    warn "Bot laeuft bereits (PID $(cat "$BOT_PIDFILE"))."
  else
    if [ "$BOT_MODE" = "live" ]; then
      echo -e "${RED}LIVE TRADING -- echtes Geld! Fortfahren? (ja/nein):${NC} "
      read -r confirm
      [ "$confirm" = "ja" ] || { info "Abgebrochen."; exit 0; }
    fi
    info "Starte Bot im ${BOT_MODE} Modus..."
    nohup "$PYTHON" main.py --mode "$BOT_MODE" --no-confirm \
      > "$LOG_DIR/trading_bot.log" 2>&1 &
    echo $! > "$BOT_PIDFILE"
    # macOS: Idle-Sleep verhindern solange der Bot lebt (-w endet mit dem Bot).
    # Auf Linux/Raspberry Pi gibt es caffeinate nicht -> kein Wrapper noetig.
    if command -v caffeinate >/dev/null 2>&1; then
      nohup caffeinate -i -w "$(cat "$BOT_PIDFILE")" >/dev/null 2>&1 &
    fi
    info "Bot-PID: $(cat "$BOT_PIDFILE") | Log: logs/trading_bot.log"
  fi
fi

# -- Browser oeffnen --
URL="http://localhost:$DASHBOARD_PORT"
if $OPEN_BROWSER; then
  if command -v open >/dev/null 2>&1; then
    open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL"
  fi
fi

# -- Status-Ausgabe --
echo ""
echo "============================================"
echo "  Grid Trading Bot"
echo "============================================"
echo "  Dashboard : http://localhost:$DASHBOARD_PORT"
if $START_BOT && bot_running; then
  echo "  Bot       : laufend ($BOT_MODE) | PID $(cat "$BOT_PIDFILE")"
else
  echo "  Bot       : nicht gestartet -- im Dashboard starten"
fi
echo ""
echo "  Logs:"
echo "    tail -f logs/trading_bot.log"
echo "    tail -f logs/dashboard.log"
echo ""
echo "  Stoppen:  ./stop.sh"
echo "============================================"
