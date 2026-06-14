#!/usr/bin/env bash
# logs.sh -- zeigt Live-Logs des Bots und Dashboards.
#
# Verwendung:
#   ./logs.sh          # beide Logs parallel (Strg+C zum Beenden)
#   ./logs.sh bot      # nur Bot-Log
#   ./logs.sh dash     # nur Dashboard-Log

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT/logs"

case "${1:-both}" in
  bot)
    tail -f "$LOG_DIR/trading_bot.log" ;;
  dash)
    tail -f "$LOG_DIR/dashboard.log" ;;
  *)
    # beide gleichzeitig mit Präfix
    tail -f "$LOG_DIR/trading_bot.log" "$LOG_DIR/dashboard.log" ;;
esac
