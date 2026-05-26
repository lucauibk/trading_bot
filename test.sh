#!/usr/bin/env bash
# test.sh -- führt die Test-Suite aus.
#
# Verwendung:
#   ./test.sh              # alle Tests
#   ./test.sh -v           # verbose
#   ./test.sh -k broker    # nur Tests mit "broker" im Namen

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

echo -e "${GREEN}[test]${NC} Starte Test-Suite..."
"$PYTHON" -m pytest tests/ "$@"
EXIT=$?

if [ $EXIT -eq 0 ]; then
  echo -e "${GREEN}[test]${NC} Alle Tests bestanden."
else
  echo -e "${RED}[test]${NC} Tests fehlgeschlagen (Exit $EXIT)."
fi
exit $EXIT
