# Grid Trading Bot

Automatischer Multi-Coin Grid Trading Bot für **Kraken** mit KI-gestützter Richtungsvorhersage,
adaptiver Positionsgröße, Per-Position-Stop-Loss und einem eingebauten Trading-Optimizer.

> **Warnung:** Trading birgt erhebliche Verlustrisiken. Starte IMMER im Paper-Trading-Modus.
> Vergangene Performance garantiert keine zukünftigen Gewinne.

---

## Architektur

```
ccxt (Kraken)
    │
    ▼
data_fetcher.py ──► ohlcv_cache.db
    │
    ├──► price_predictor/   → Regime-Detection + Grid-Range (ATR/Bollinger)
    │         └──► PricePredictor.predict() → low / high / regime / confidence
    │
    ├──► ml/               → LightGBM-Richtungsvorhersage (up/neutral/down)
    │         └──► MLPredictor.predict() → direction + confidence
    │
    ▼
grid_bot.py
    ├── PaperGridBot  (Simulation)
    └── LiveGridBot   (echte Orders auf Kraken)
          │
          ├── setup_grid()          → Limit-Orders aufbauen
          ├── check_fills()         → Order-Fills + Per-Position-SL
          ├── _maybe_compound()     → Gewinn reinvestieren (Cap: 3×)
          ├── check_stop_loss()     → 8% Max-Loss-Bremse
          └── emergency_sell()      → Sofort-Verkauf bei DOWN-Signal
          │
          ▼
    src/risk/RiskManager      → Cross-Coin Daily-Drawdown (-3% Freeze-Mode)
          │
          ▼
    dashboard/db.py           → trades.db (Trades + Kontext + Grid-State)
          │
          ▼
    dashboard/app.py          → Flask-Dashboard (Port 5001)
```

---

## Setup

### 1. Abhängigkeiten installieren

```bash
pip3 install -r requirements.txt
```

### 2. `.env` konfigurieren

```bash
cp .env.example .env
```

Pflichtfelder:
```env
KRAKEN_API_KEY=dein_api_key
KRAKEN_API_SECRET=dein_api_secret
PAPER_TRADING=true          # Auf false für Live-Trading
TELEGRAM_BOT_TOKEN=...      # Optional: Telegram-Benachrichtigungen
TELEGRAM_CHAT_ID=...
```

### 3. Coins und Budget konfigurieren

In `grid_bot.py` (Zeile 30–36) oder nach dem Start über das Dashboard:

```python
GRIDS = [
    {"symbol": "SOL/USD",  "investment": 300.0, "levels": 8},
    {"symbol": "ETH/USD",  "investment": 300.0, "levels": 8},
    # ...
]
```

---

## Bot starten

```bash
# Paper-Trading (empfohlen)
python3 main.py --strategy grid --mode paper

# Live-Trading
python3 main.py --strategy grid --mode live

# Klassische Strategien
python3 main.py --strategy ema --mode paper
python3 main.py --strategy rsi --mode paper
```

---

## Dashboard

```
http://localhost:5001
```

Features:
- Echtzeit-Grid-Visualisierung pro Coin (Levels, Fills, ML-Badge, Regime)
- KPI-Karten: Gesamt-PnL, Heute, Win-Rate, Kapital
- Equity-Kurve (Chart.js, alle 10s Refresh)
- Trade-History (letzte 30)
- Coin-Budget-Editor (Toggle aktiv/deaktiv, max_investment pro Coin)
- Bot-Steuerung: Start/Stop, Strategie, Paper/Live-Mode

---

## Backtest & Optimizer

### Grid-Backtest
```bash
python3 grid_backtester.py
```

### Parameter-Sweep (Levels × Range)
```bash
python3 grid_optimizer.py
```

### Trading-Optimizer CLI (nach 7+ Tagen Paper-Trading empfohlen)
```bash
# Vollständige Trade-Analyse pro Regime und ML-Konfidenz-Bucket
python3 scripts/optimize.py --analyze-trades --days 30

# ML-Kalibrierungsbericht (Hit-Rate vs. Confidence)
python3 scripts/optimize.py --calibration-report

# Beste Grid-Parameter aus Backtest-Sweeps
python3 scripts/optimize.py --suggest-params

# Toxische Trade-Setups identifizieren
python3 scripts/optimize.py --pattern-mine --days 60

# Neuen Backtest-Sweep starten + in DB persistieren
python3 scripts/optimize.py --run-sweep --symbol SOL/USD
```

Oder direkt in Claude Code: **`/trading-optimizer`**

---

## Konfiguration

### `config/config.yaml`
Allgemeine Einstellungen: Exchange, Symbol-Liste, Timeframe.

### `config/strategy_params.yaml`
ATR-Perioden, Risk-Limits (`max_daily_drawdown: 0.03`), EMA-Fensterlängen.

### Dashboard-Override
Coin-Budget im Dashboard unter **Coin-Settings** bearbeitbar. Werte werden in
`data/trades.db → coin_settings` gespeichert und beim nächsten Start automatisch geladen.

---

## Risk-Management

| Mechanismus | Wert | Beschreibung |
|-------------|------|-------------|
| Per-Position Stop-Loss | 4% | Jede Buy-Fill: SL 4% unter Kaufpreis |
| Coin-Notbremse | 8% Investment | Stoppt Coin bei 8% Gesamtverlust |
| Cross-Coin Freeze-Mode | -3% Tages-PnL | Keine neuen Buys bis nächsten Tag |
| Compounding-Cap | 3× Initial | Investment max. 3× des Start-Budgets |
| ML Emergency-Sell | DOWN-Signal | Alle offenen Positionen sofort schließen |

### Risiko-Profil anpassen

Alle Parameter in `grid_bot.py` (Zeilen 44–47 und `run()` → `max_daily_drawdown`):

```python
# Konservativ
MAX_LOSS_PCT        = 0.05   # 5% Coin-Notbremse
PER_POS_SL_PCT      = 0.05   # 5% Per-Position-SL
MAX_INVESTMENT_MULT = 2.0    # max. 2× Startkapital
max_daily_drawdown  = 0.02   # Freeze ab -2% Tages-PnL

# Aktuell (moderat aggressiv)
MAX_LOSS_PCT        = 0.08   # 8%
PER_POS_SL_PCT      = 0.04   # 4%
MAX_INVESTMENT_MULT = 3.0    # max. 3× Startkapital
max_daily_drawdown  = 0.03   # Freeze ab -3%

# Aggressiv
MAX_LOSS_PCT        = 0.15   # 15%
PER_POS_SL_PCT      = 0.025  # 2.5%
MAX_INVESTMENT_MULT = 5.0    # max. 5× Startkapital
max_daily_drawdown  = 0.05   # Freeze ab -5%
```

Nach Änderungen Bot neu starten:
```bash
./stop.sh && ./start.sh --bot
```

---

## ML-Pipeline

### Modell
- **LightGBM** + `CalibratedClassifierCV(method="isotonic")` → kalibrierte Wahrscheinlichkeiten
- **Walk-Forward-Validierung** (TimeSeriesSplit, 5 Folds) — Modell wird nur gespeichert wenn OOS-F1 ≥ 0.30
- **Triple-Barrier-Labels** (López de Prado): ATR-skalierte Gewinn-/Verlust-Barrieren

### Features (16 Dimensionen)
EMA9/21-Ratios, EMA-Cross, RSI/100, Momentum 1h/4h/12h, MACD-Histogram, BB-%, BB-Breite, Volumen-Ratio, ATR%, Candle-Body/Shadows, is_green

### Lifecycle
1. **Bootstrap** beim Start: 1000 1h-Candles → Triple-Barrier-Labels → Training
2. **Predict**: Features extrahieren → LightGBM → kalibrierte Konfidenz
3. **Retrain**: Asynchron im Hintergrund (alle 50 neue gelabelte Samples)

---

## Datenbankschema

| Tabelle | Inhalt |
|---------|--------|
| `trades` | Trade-History: timestamp, symbol, entry, exit, pnl, reason |
| `trade_context` | Kontext zum Trade: atr_pct, rsi, ema9/21, regime, ml_confidence, … |
| `predictions` | ML-Vorhersagen mit realisiertem Outcome (für Kalibrierung) |
| `optimizer_runs` | Backtest-Sweep-Ergebnisse (params, score, daily_pct, max_dd) |
| `grid_state` | Live Grid-Zustand pro Coin |
| `equity` | Kapital-Kurve |
| `coin_settings` | Dashboard-Budget-Override |

---

## Tests

```bash
# Alle Tests
python3 -m pytest tests/ -v

# Grid-Bot Tests (Risk-Mgmt, Compounding, Stop-Loss, SL)
python3 -m pytest tests/test_grid_bot.py -v

# PricePredictor Tests
python3 -m pytest price_predictor/tests/ -v

# Strategie-Tests (EMA, RSI, RiskManager)
python3 -m pytest tests/test_strategy.py -v
```

---

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| Dashboard nicht erreichbar | Port ist `5001`, nicht 5000 |
| ML-Modell lädt nicht | `data/models/` löschen → Bootstrap beim nächsten Start |
| Hohe ATR-Fehler-Rate | `ATR_CANDLES` temporär auf 5 setzen bis mehr History vorhanden |
| Live-Order schlägt fehl | `PAPER_TRADING=true` in `.env` prüfen |
| Freeze-Mode aktiv | 3% Daily-Drawdown erreicht → automatisch um Mitternacht aufgehoben |

### Wichtige Log-Tags

```
[PAPER GRID] BUY/SELL   → Grid-Fill
[STOP-LOSS]             → Per-Position-Stop-Loss ausgelöst
NOTBREMSE               → 8%-Coin-Limit erreicht
FREEZE-MODE             → Cross-Coin Daily-Drawdown
♻️  COMPOUND            → Gewinn reinvestiert
Walk-Forward OOS F1     → ML-Modell Qualitätsmetrik nach Retrain
```
