#!/usr/bin/env bash
# backtest.sh -- führt einen Backtest für ein Symbol aus.
#
# Verwendung:
#   ./backtest.sh                   # SOL/USD, 90 Tage
#   ./backtest.sh ETH/USD 60        # ETH, 60 Tage
#   ./backtest.sh SOL/USD 180       # SOL, 180 Tage

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SYMBOL="${1:-SOL/USD}"
DAYS="${2:-90}"

echo "[backtest] $SYMBOL | $DAYS Tage"
"$PYTHON" -m backtest.engine --symbol "$SYMBOL" --days "$DAYS"
