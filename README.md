# Trading Bot

Automatisierter Krypto-Trading-Bot mit Grid-, EMA- und RSI-Strategie, Risikomanagement, Backtesting und Web-Dashboard.

> ⚠️ **Warnung:** Trading birgt erhebliche Verlustrisiken. Starte IMMER im Paper-Trading-Modus. Vergangene Performance garantiert keine zukünftigen Gewinne.

---

## Installation

```bash
git clone <repo>
cd trading-bot
pip3 install -r requirements.txt
```

Kopiere `.env.example` zu `.env` und trage deine Werte ein:

```bash
cp .env.example .env
```

`.env`:
```
KRAKEN_API_KEY=dein_api_key
KRAKEN_API_SECRET=dein_api_secret
PAPER_TRADING=true
INITIAL_CAPITAL=1000
TELEGRAM_TOKEN=dein_telegram_bot_token
TELEGRAM_CHAT_ID=deine_chat_id
```

---

## Konfiguration

Alle Parameter in `config/config.yaml` und `config/strategy_params.yaml`.

Wichtigste Einstellungen in `config/config.yaml`:
```yaml
exchange: kraken
paper_trading: true       # false = echtes Geld!
initial_capital: 1000

risk:
  max_risk_per_trade: 0.01    # 1% pro Trade
  max_daily_drawdown: 0.03    # Bot stoppt bei 3% Tagesverlust
```

Grid-Parameter in `config/strategy_params.yaml`:
```yaml
grid:
  investment_per_coin: 300
  levels: 6
  range_min: 0.08
  range_max: 0.25
```

---

## Bot starten

```bash
# Grid Bot (empfohlen)
python3 main.py --strategy grid

# EMA Crossover
python3 main.py --strategy ema

# RSI Mean Reversion
python3 main.py --strategy rsi

# Live Trading (fragt Bestätigung!)
python3 main.py --mode live --strategy grid
```

---

## Backtest starten

```bash
python3 -m backtest.backtester --strategy rsi --symbol SOL/USDT --since 2024-01-01T00:00:00Z
```

Ergebnis wird als `backtest_report.html` gespeichert (im Browser öffnen).

---

## Dashboard starten

```bash
python3 dashboard/app.py
```

Öffne [http://localhost:5000](http://localhost:5000) im Browser.

Das Dashboard zeigt:
- Bot-Status (Running/Stopped, Paper/Live)
- Gesamt-PnL, Tages-PnL, Win-Rate
- Equity Curve (live aktualisierend)
- Trade-History

---

## Tests ausführen

```bash
python3 -m pytest tests/ -v
```

---

## Projektstruktur

```
trading-bot/
├── config/                  # YAML Konfiguration
│   ├── config.yaml
│   └── strategy_params.yaml
├── src/
│   ├── data/
│   │   ├── fetcher.py       # OHLCV mit Cache + Retry
│   │   └── processor.py     # Technische Indikatoren
│   ├── strategy/
│   │   ├── base_strategy.py
│   │   ├── ema_crossover.py
│   │   └── rsi_mean_rev.py
│   ├── risk/
│   │   └── risk_manager.py  # Drawdown, Position Sizing
│   └── execution/
│       └── broker.py        # PaperBroker / LiveBroker
├── backtest/
│   └── backtester.py        # Backtest + HTML-Report
├── dashboard/
│   ├── app.py               # Flask Dashboard
│   ├── db.py                # SQLite Trade-History
│   └── templates/index.html
├── tests/
│   └── test_strategy.py
├── grid_bot.py              # Grid Trading Bot
├── notifier.py              # Telegram
└── main.py                  # Einstiegspunkt
```
