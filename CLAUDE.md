# Grid Trading Bot – Vollständige Projektdokumentation

## Projektübersicht

Multi-Coin Grid Trading Bot für Kraken (Paper + Live). Kombiniert dynamisches Grid-Trading
mit ML-Vorhersagen (LightGBM 34 Features + Claude Haiku), adaptiver Positionsgröße und
Directional Trades.

**Ziel:** Maximaler Tagesgewinn bei minimalem Loss durch:
- ATR/Bollinger-basierte Grid-Ranges (PricePredictor)
- LightGBM (34 Features) + Claude Haiku Blended-Richtungsvorhersage
- Directional Trades bei starkem UP-Signal (20% des Investments, Leverage aus DB)
- Floor-Stop-Loss (SL unter Gridboden) → verhindert Kaskaden-Dumps
- Cross-Coin Daily-Drawdown-Bremse (-10%, config.yaml max_daily_drawdown)
- Compounding mit Investment-Cap (3× Initial)

---

## Start & Betrieb

```bash
./start.sh           # nur Dashboard starten (Bot dann per Browser)
./start.sh --bot     # Dashboard + Bot im Paper-Modus
./start.sh --live    # Dashboard + Bot im Live-Modus
./stop.sh            # alles stoppen

# Logs live verfolgen
tail -f logs/trading_bot.log
tail -f logs/dashboard.log
```

**Voraussetzung:** `.env` muss existieren. Falls weg: `cp .env.example .env`
→ Die BINANCE_API_KEY/SECRET im .env sind tatsächlich die Kraken-Keys (falsch benannt).
→ `config.py` liest automatisch beide Namen als Fallback.

---

## Architektur: Wie alles zusammenläuft

```
start.sh
  ├── startet dashboard/app.py  (Flask, Port 5001) als Hintergrundprozess
  └── startet main.py --mode paper|live  als Hintergrundprozess

Browser → http://localhost:5001
  │
  ├── API-Calls (start/stop/settings)
  │       ↓
  │   dashboard/app.py  schreibt in data/trades.db
  │       ↓
  │   core/engine.py  liest DB jede Loop-Iteration (CHECK_INTERVAL = 15s)
  │
  └── SSE-Stream /stream  (Live-Updates alle 15s)
```

**Kommunikation Dashboard ↔ Bot ausschließlich über SQLite (`data/trades.db`):**
- Dashboard schreibt: `bot_status.leverage`, `bot_status.stop_mode`, `coin_settings`
- Bot liest: diese Felder jede Loop-Iteration
- Bot schreibt: `trades`, `grid_state`, `equity`, `predictions`
- Dashboard liest: alle diese Tabellen für Anzeige

**Prozess-Verwaltung:**
- Dashboard startet Bot via `subprocess.Popen` → PID in `.bot.pid`, Lock in `.bot.lock`
- Singleton-Guard via `fcntl.flock` (race-free OS-Lock) in `core/lifecycle.py`
- Stop via SIGTERM an PID aus `.bot.pid`
- Graceful Stop: Dashboard setzt `bot_status.stop_mode` → Bot reagiert selbst

---

## Graceful Shutdown

"⏹ Stoppen"-Button im Dashboard öffnet Modal mit drei Optionen:

| Option | Was passiert |
|--------|-------------|
| 💰 Sofort alle Positionen verkaufen | `stop_mode='sell_all'` → Engine ruft `_emergency_sell_all()` auf, dann Exit |
| ⏳ Auf Grid-Fills warten | `stop_mode='wait_fills'` → `with_position=False` für alle Coins, läuft weiter bis alle Sells gefüllt, dann Auto-Exit |
| Abbrechen | Bot läuft weiter |

**Endpoint:** `POST /api/bot/stop-graceful` mit `{"mode": "sell_all"|"wait_fills"}`
**Bot-Check:** Jede Loop-Iteration via `get_stop_mode()` aus `dashboard/db.py`

---

## Projektstruktur

```
trading-bot/
├── main.py               # Einstiegspunkt: --mode paper|live
├── config.py             # ENV-Variablen (BINANCE_* = Kraken-Fallback)
├── data_fetcher.py       # OHLCV + Ticker via ccxt (Kraken)
├── notifier.py           # Telegram-Benachrichtigungen
├── start.sh / stop.sh    # Prozess-Management-Scripts
│
├── core/                 # Engine-Kern
│   ├── engine.py         # Event-Loop: ticker → on_tick → desired_orders → broker
│   ├── context.py        # MarketContext: BTC-Regime, Funding, Equity, Positionen
│   ├── lifecycle.py      # Singleton-Lock (fcntl.flock), ShutdownFlag
│   └── strategy.py       # Strategy ABC: Order, Fill, Hooks
│
├── strategies/           # Trading-Strategien
│   ├── grid.py           # GridStrategy: Grid-Logik, SL/TP, Directional Trades
│   └── grid_params.py    # GridParams Dataclass (alle Tuning-Parameter)
│
├── execution/            # Broker-Abstraktion
│   ├── broker.py         # Broker ABC
│   ├── paper.py          # PaperBroker: per-Symbol-Budget, Margin-Accounting
│   ├── kraken.py         # KrakenBroker: ccxt, PostOnly, Retry
│   └── reconciler.py     # Fill-Reconciliation für Live-Modus
│
├── risk/                 # Risiko-Management
│   ├── manager.py        # RiskManager: Daily-Drawdown, Position Caps
│   ├── sizing.py         # Position-Sizing
│   └── correlation.py    # Korrelations-Tracker (BTC-Bucket)
│
├── market/               # Markt-Kontext
│   ├── btc_context.py    # BTC-Trend, Returns (gecacht)
│   └── perp.py           # Funding-Rate Cache
│
├── ml/                   # KI-Richtungsvorhersage
│   ├── model.py          # TradingModel: LightGBM + CalibratedClassifier + Walk-Forward
│   ├── trainer.py        # Triple-Barrier-Labels, Bootstrap, Retrain (mit echtem Rollback)
│   ├── features/         # 34-Feature-Vektoren
│   │   ├── technical.py  # 16 technische Features (EMA, RSI, Momentum, BB, ATR, …)
│   │   ├── htf.py        # 4 Higher-Timeframe-Features (4h/1d Trend, RSI)
│   │   ├── market.py     # 5 BTC-Kontext-Features (Returns, Korrelation, Dominanz)
│   │   ├── perp.py       # 4 Perp-Features (Funding-Rate, OI)
│   │   ├── seasonality.py # 5 Saisonalitäts-Features (Stunde, Wochentag)
│   │   └── combined.py   # extract_all() → 34-Feature-Vektor
│   ├── predictor.py      # MLPredictor: predict() + async Retrain + LLM-Blending
│   ├── llm_analyst.py    # Claude Haiku: Marktanalyse + Score-Blending
│   └── data_store.py     # SQLite-Persistenz für ML-Samples
│
├── price_predictor/      # Regelbasierter Range-Predictor (ATR/Bollinger)
│   ├── predictor.py      # PricePredictor: predict() → low/high/regime/confidence
│   ├── indicators.py     # ATR(14), Bollinger(20,2σ), RSI(14), VWAP, ADX(14)
│   └── grid_suggester.py # Grid-Levels aus Range berechnen
│
├── backtest/             # OOS-Backtester (für Sweep)
│   ├── engine.py         # Backtest-Engine
│   ├── data.py           # Historische Daten
│   └── metrics.py        # Calmar, Drawdown, Win-Rate
│
├── dashboard/            # Flask-Dashboard Port 5001
│   ├── app.py            # API-Endpoints + SSE-Stream + Bot-Prozess-Management
│   ├── db.py             # SQLite-Schema + Helpers (log_trade, set_stop_mode, etc.)
│   └── templates/        # index.html: Grid-Visualisierung, Charts, Stop-Modal
│
├── scripts/
│   ├── optimize.py       # Trading-Optimizer CLI (--analyze-trades, --suggest-params, …)
│   ├── sweep.py          # OOS Parameter-Sweep (Calmar-optimiert)
│   └── nightly_tune.py   # Nightly Auto-Tune: Branch + Issue + PR (täglich 05:00)
│
├── data/                 # SQLite-DBs (nicht in Git)
│   ├── trades.db         # Trades + grid_state + bot_status + predictions + equity
│   ├── ml_training.db    # ML-Samples (features + labels)
│   └── ohlcv_cache.db    # OHLCV-Cache
│
├── config/
│   ├── config.yaml       # Symbol-Liste, Risk-Parameter, Initial-Capital
│   └── grid_params.json  # Aktuelle Sweep-Winner-Parameter (auto-generiert)
│
├── results/              # Sweep-Ergebnisse (auto-generiert, nicht in Git)
├── logs/                 # Logfiles (trading_bot.log, dashboard.log)
├── .env                  # API-Keys (niemals einchecken) – BINANCE_* = Kraken-Fallback
└── requirements.txt
```

---

## Schlüssel-Konstanten (echte Werte, Stand simplify-bot)

### core/engine.py

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `CHECK_INTERVAL` | 15 | Sekunden zwischen Loop-Iterationen |
| `PREDICTION_RECHECK` | 5 | Alle 5 Ticks (~75s) Prediction neu prüfen |
| `GRID_REBUILD_CYCLES` | 60 | Alle 60 Ticks (~15 Min) Grid neu aufbauen |
| `EMERGENCY_STOP_PCT` | 0.12 | 12% Realized-Loss pro Coin → Symbol pausieren |

### strategies/grid.py + grid_params.py (GridParams Defaults)

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `sl_mode` | `"floor"` | SL unter Gridboden (kaskadensicher) |
| `floor_sl_atr_mult` | 1.0 | Floor = grid_lower − 1.0 × ATR |
| `per_pos_sl_max_pct` | 0.04 | Hard-Cap: kein Per-Position-SL > 4% |
| `levels_by_regime` | ranging:14, trending:6, volatile:20 | Levels pro Regime |
| `trend_filter_enabled` | True | EMA/ADX-Trend-Filter (Buys pausieren bei Downtrend) |
| `COMPOUND_EVERY_TRADES` | 3 | Gewinn-Reinvestition alle 3 Trades |
| `MAX_INVESTMENT_MULT` | 3.0 | Compounding-Cap: max. 3× Initial-Investment |
| `KRAKEN_FEE` | 0.0016 | Maker-Fee 0.16% — immer diese verwenden |

### Directional-Trade-Parameter (GridParams)

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| `directional_score_min` | 0.12 | Score-Schwelle für neuen Einstieg |
| `directional_pct` | 0.20 | 20% des Investments pro Directional |
| `directional_tp_atr` | 3.0 | Take-Profit: Einstieg + 3.0 × ATR |
| `directional_sl_atr` | 1.5 | Stop-Loss: Einstieg − 1.5 × ATR |
| `DIRECTIONAL_DOWN_TRAIL_PCT` | 0.005 | Trail-Lock: erst nach 0.5% Rückgang verkaufen |
| `DIRECTIONAL_RECHECK_SCORE_MIN` | 0.25 | Nach SL: Score > 0.25 für Re-Entry |

---

## Stop-Loss-Design

**Standard: `sl_mode = "floor"` (kaskadensicher)**
- Ein einziger SL-Level unter dem Gridboden (`grid_lower - 1×ATR`)
- Kein einzelnes Grid-Level kann alleine stoppen → kein Cascade-Dump
- Bei Grid-Rebuild: Floor-SL wird nie gesenkt (Ratchet in `grid.py:748-752`)

**Fallback `sl_mode = "per_position"` (für Backtests)**
- SL pro Position = `step_pct × 1.5`, mind. 0.8%, **max. 4%** (Hard-Cap)
- Hard-Cap verhindert -7%/-9%-Exits bei weiten Grids

---

## Multi-Coin & Balance-Accounting

**PaperBroker (`execution/paper.py`) mit per-Symbol-Budgets:**
- `initial_balance / n_symbols` pro Coin-Bucket → SOL kann nicht das ETH-Budget leeren
- Margin-Accounting: Buy deducted = `notional / leverage + fee`
- Sell credited = `margin_return + leveraged_PnL - fee`
- Pre-seeded Sells (oben im Grid ohne echten Buy): nur Profit-Anteil gutgeschrieben

**Für Live-Modus (KrakenBroker):** `meta` aus Order.meta wird weitergeleitet (für Logging),
Margin-Accounting übernimmt die echte Exchange.

---

## ML-Vorhersage-Pipeline (34 Features)

```
predict(symbol)
  │
  ├── extract_all(df_1h, funding, btc, btc_corr, dt)
  │     → 34-Feature-Vektor:
  │       technical(16) + perp(4) + market(5) + htf(4) + seasonality(5)
  │
  ├── LightGBM (CalibratedClassifier, Walk-Forward 5-Folds)
  │     └── lgbm_score + lgbm_conf
  │
  ├── Claude Haiku (llm_analyst.py, gecacht ~1h)
  │     └── llm_score + llm_conf + reason
  │
  ├── blend_scores: 0.55 × lgbm + 0.45 × llm
  │
  ├── if blended_conf >= MIN_CONFIDENCE (0.45):
  │     score > +0.15 → "up", score < -0.15 → "down", sonst "neutral"
  │
  └── else: Fallback _rule_based() → score ±0.5
```

**Retrain:** Async via ThreadPoolExecutor, echter Rollback wenn F1 − 0.05 schlechter
**OOS-Gate:** Modell nur gespeichert wenn Walk-Forward-F1 ≥ `MIN_OOS_F1 = 0.30`
**Feature-Mismatch:** `predict()` gibt `(hold, 0.0)` zurück wenn Feature-Anzahl falsch → Bootstrap
**Perp-Features:** Im Training als 0 (kein historisches Funding), Live via `market/perp.py`

---

## Regime-Logik (PricePredictor + GridParams)

| Regime | Bedingung | Grid-Range | Levels |
|--------|-----------|-----------|--------|
| trending | ADX > 25 | ATR × 2.0 | 6 |
| volatile | ATR% > 3% | ATR × 1.5 | 20 |
| ranging | sonst | Bollinger Bands | 14 |

---

## Datenbanken

### `data/trades.db`

| Tabelle | Inhalt |
|---------|--------|
| `trades` | Trade-History: timestamp, symbol, direction, entry, exit, pnl, reason, leverage |
| `trade_context` | Kontext: atr_pct, rsi, ema9/21, regime, ml_confidence, … |
| `grid_state` | Aktueller Grid-Status pro Coin (Dashboard-Anzeige) |
| `grid_sessions` | Pro Session: profit, trades, max_dd, range_pct, levels |
| `predictions` | ML/PricePredictor-Vorhersagen mit realized outcome |
| `optimizer_runs` | Backtest-Sweep-Ergebnisse |
| `equity` | Kapital-Kurve (alle 15s, Mark-to-Market) |
| `coin_settings` | Dashboard-Override: max_investment, enabled pro Symbol |
| `bot_status` | Einzeilig: running, mode, leverage, **stop_mode** (Graceful-Shutdown-Flag) |

**Wichtig:** `bot_status.stop_mode` ist das Kommunikations-Flag für Graceful Shutdown.
Wird vom Bot nach Lesen automatisch auf NULL zurückgesetzt.

---

## Nightly Auto-Tune (05:00 täglich)

`scripts/nightly_tune.py` läuft als lokaler launchd-Job (`~/Library/LaunchAgents/com.tradingbot.nightlytune.plist`,
geladen via `launchctl bootstrap gui/$(id -u) ...`) — **kein Cloud-Routine**, weil Trade-Analyse (`trades.db`) und
Telegram-Notify lokale, gitignored Dateien (`data/trades.db`, `.env`) brauchen, auf die eine Cloud-Sandbox keinen
Zugriff hat. Läuft nur, wenn der Mac um 05:00 wach ist. Log: `logs/nightly_tune.log`.
1. Branch `auto-tune/YYYY-MM-DD` von main
2. Trade-Analyse (7 Tage) + Pattern Mining
3. OOS-Sweep je Symbol (180 Tage, 120 Train)
4. Wenn Sweep-Winner besser: `config/grid_params.json` committen
5. GitHub-Issue mit Findings
6. Draft-PR nach main (User reviewed + merged manuell am nächsten Tag)
7. Telegram-Benachrichtigung

**Sanity:** Niemals main direkt, niemals Live-Bot neu starten.

---

## Dashboard-API (wichtigste Endpoints)

| Endpoint | Methode | Funktion |
|----------|---------|----------|
| `/api/bot/start` | POST | Bot als Subprocess starten |
| `/api/bot/stop` | POST | Bot sofort via SIGTERM beenden |
| `/api/bot/stop-graceful` | POST `{"mode":"sell_all"\|"wait_fills"}` | Graceful Stop via DB-Flag |
| `/api/status` | GET | Bot-Status, Leverage, Mode |
| `/api/grids` | GET | Alle Grid-States (Coins, Orders, PnL) |
| `/api/leverage` | POST `{"leverage": 3.0}` | Hebel ändern (wirkt sofort) |
| `/api/coin-settings` | POST | max_investment, enabled pro Symbol |
| `/stream` | GET (SSE) | Live-Updates alle 15s |

---

## Konventionen (PFLICHT)

- **Fees:** Immer `KRAKEN_FEE = 0.0016` nutzen, nie hardcoded.
- **RiskManager:** Jede neue Strategie muss `risk/manager.py` einbinden.
- **ATR:** Immer Fallback auf ATR wenn PricePredictor fehlschlägt.
- **Kein Hardcoding:** Exchange, Symbol und Fees immer als Konstanten.
- **ML-Retrain:** Nie im predict()-Hot-Path blockierend — ThreadPoolExecutor nutzen.
- **Walk-Forward:** Modelle nur speichern wenn OOS-F1 ≥ `MIN_OOS_F1 = 0.30`.
- **db.log_trade():** `context`-Dict mitliefern für Pattern-Mining.
- **stop_mode:** Nach Lesen immer auf NULL zurücksetzen — sonst blockiert Bot-Neustart.
- **Singleton:** `core/lifecycle.acquire_singleton()` am Start — verhindert Doppelstarts.
- **Feature-Count:** Modell-Predict prüft Feature-Anzahl → bei Mismatch Bootstrap-Retrain.

---

## Trading-Optimizer Skill

Aufruf in Claude Code: `/trading-optimizer`

Analysiert `trades.db` und gibt Empfehlungen für Grid-Parameter, ML-Confidence-Threshold und Risiko-Filter.

```bash
python3 scripts/optimize.py --analyze-trades --days 30
python3 scripts/optimize.py --calibration-report
python3 scripts/optimize.py --suggest-params
python3 scripts/optimize.py --pattern-mine
python3 scripts/optimize.py --run-sweep --symbol SOL/USD

# Nightly Tune manuell testen
python3 scripts/nightly_tune.py
```
