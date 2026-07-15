#!/usr/bin/env bash
# kiosk.sh — Touchscreen-Dashboard auf dem Raspberry Pi.
# Wartet aufs Dashboard, dann Chromium-Kiosk — sonst zeigt der Browser
# nach dem Boot dauerhaft "Seite nicht erreichbar" (Race gegen den
# Watchdog, der Bot+Dashboard erst ~30-60s nach dem Desktop startet).
# Autostart: ~/.config/autostart/dashboard-kiosk.desktop → Exec=<dieses Script>
for i in $(seq 1 120); do
  curl -s --max-time 2 http://localhost:5001/api/status >/dev/null && break
  sleep 5
done
exec chromium --kiosk --noerrdialogs --disable-session-crashed-bubble http://localhost:5001
