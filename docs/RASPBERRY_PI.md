# Raspberry-Pi-Migration — Checkliste

Ziel: Der Bot zieht vom Mac (schläft → Uptime-Lücken) auf einen Raspberry Pi
mit Touchscreen als 24/7-Host. Der Code ist bereits Pi-portabel: alle
macOS-Spezifika (`caffeinate`) stehen hinter `command -v`/`shutil.which`-Guards
und sind auf Linux No-Ops.

## 1. Grundsetup auf dem Pi

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git sqlite3 chromium-browser
git clone https://github.com/lucauibk/trading_bot.git ~/trading-bot
cd ~/trading-bot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**LightGBM auf ARM64:** Falls das pip-Wheel fehlschlägt:
`sudo apt install -y cmake libomp-dev` und erneut versuchen, oder
`pip install lightgbm --no-binary lightgbm` (kompiliert lokal, dauert).

## 2. Daten & Secrets vom Mac übernehmen

Diese Dateien sind gitignored und müssen manuell kopiert werden
(z.B. per `scp` vom Mac):

```bash
scp ~/.../trading-bot/.env            pi@<pi-ip>:~/trading-bot/.env
scp ~/.../trading-bot/data/*.db       pi@<pi-ip>:~/trading-bot/data/
scp -r ~/.../trading-bot/data/models  pi@<pi-ip>:~/trading-bot/data/models
```

Vorher auf dem Mac den Bot stoppen (`./stop.sh`), damit die SQLite-DBs
konsistent sind (WAL-Dateien `*.db-wal`/`*.db-shm` mitkopieren oder vorher
`sqlite3 data/trades.db "PRAGMA wal_checkpoint(TRUNCATE)"`).

## 3. Test-Start

```bash
cd ~/trading-bot && ./start.sh --bot --no-browser
tail -f logs/trading_bot.log     # Loop muss alle 15s ticken
curl -s localhost:5001/api/status
```

## 4. launchd → systemd/cron

Auf dem Mac laufen drei launchd-Jobs (`~/Library/LaunchAgents/com.tradingbot.*`).
Auf dem Pi ersetzen durch cron (einfachste Variante):

```cron
# crontab -e
*/5 * * * *  /home/pi/trading-bot/scripts/watchdog.sh
0 5 * * *    cd /home/pi/trading-bot && ./venv/bin/python3 scripts/nightly_tune.py >> logs/nightly_tune.log 2>&1
0 5,11,17,23 * * * cd /home/pi/trading-bot && ./venv/bin/python3 scripts/bot_monitor.py >> logs/monitor.log 2>&1
```

Der Watchdog (`scripts/watchdog.sh`) ist portabel und wird auf dem Pi zur
reinen Crash-Absicherung — das Sleep-Problem existiert dort nicht.
`./stop.sh` setzt weiterhin `.bot.stopped`, damit bewusste Stopps nicht
neu gestartet werden.

## 5. Touchscreen: Dashboard im Kiosk-Modus

Chromium beim Boot im Vollbild auf das Dashboard zeigen lassen —
`~/.config/autostart/dashboard-kiosk.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Trading Dashboard
Exec=chromium-browser --kiosk --noerrdialogs --disable-session-crashed-bubble http://localhost:5001
```

Optional Bildschirmschoner aus: `sudo raspi-config` → Display → Screen Blanking off.

## 6. Mac stilllegen (nach erfolgreichem Umzug)

```bash
# Auf dem Mac:
./stop.sh                                   # setzt .bot.stopped → Watchdog startet nichts neu
launchctl bootout gui/$(id -u)/com.tradingbot.watchdog
launchctl bootout gui/$(id -u)/com.tradingbot.nightlytune
launchctl bootout gui/$(id -u)/com.tradingbot.monitor
```

Wichtig: **Nie beide Hosts gleichzeitig** mit denselben Kraken-Keys laufen
lassen — der Singleton-Lock (`core/lifecycle.py`, `fcntl.flock`) schützt nur
pro Maschine, nicht über Hosts hinweg.

## Offene Punkte beim Umzug

- [ ] Telegram-Notify testen (`.env`-Werte übernommen?)
- [ ] `gh` CLI auf dem Pi einrichten, falls nightly_tune Issues/PRs erstellen soll (`sudo apt install gh && gh auth login`)
- [ ] Zeitzone prüfen (`timedatectl`) — Cron-Zeiten sind Lokalzeit
- [ ] Nach 24h: Equity-Ticks/Tag prüfen (`sqlite3 data/trades.db "SELECT date(timestamp), COUNT(*) FROM equity GROUP BY 1 ORDER BY 1 DESC LIMIT 3"`) — Ziel: ~5700/Tag (15s-Takt)
