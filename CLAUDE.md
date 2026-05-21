# Grid Trading Bot – Vollständige Projektdokumentation

## Projektübersicht

Multi-Coin Grid Trading Bot für Kraken (Paper + Live). Kombiniert dynamisches Grid-Trading
mit ML-Vorhersagen (LightGBM + Claude Haiku), adaptiver Positionsgröße und Directional Trades.

**Ziel:** Maximaler Tagesgewinn bei minimalem Loss durch:
- ATR/Bollinger-basierte Grid-Ranges (PricePredictor)
- LightGBM + Claude Haiku Blended-Richtungsvorhersage
- Directional Trades bei starkem UP-Signal (15% des Investments, 3× Leverage)
- Per-Position-Stop-Loss (4% unter Buy-Preis)
- Cross-Coin Daily-Drawdown-Bremse (-3%)
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
  │   grid_bot.py  liest DB jede Loop-Iteration (CHECK_INTERVAL = 15s)
  │
  └── SSE-Stream /stream  (Live-Updates alle 15s)
```

**Kommunikation Dashboard ↔ Bot ausschließlich über SQLite (`data/trades.db`):**
- Dashboard schreibt: `bot_status.leverage`, `bot_status.stop_mode`, `coin_settings`
- Bot liest: diese Felder jede Loop-Iteration
- Bot schreibt: `trades`, `grid_state`, `equity`, `predictions`
- Dashboard liest: alle diese Tabellen für Anzeige

**Prozess-Verwaltung:**
- Dashboard startet Bot via `subprocess.Popen` → PID in `.bot.pid`
- Stop via SIGTERM an PID aus `.bot.pid`
- Graceful Stop: Dashboard setzt `bot_status.stop_mode` → Bot reagiert selbst

---

## Graceful Shutdown (NEU)

"⏹ Stoppen"-Button im Dashboard öffnet Modal mit drei Optionen:

| Option | Was passiert |
|--------|-------------|
| 💰 Sofort alle Positionen verkaufen | `stop_mode='sell_all'` → Bot ruft `_sell_all_on_exit()` + `emergency_sell()` auf, dann Exit |
| ⏳ Auf Grid-Fills warten | `stop_mode='wait_fills'` → `with_position=False` für alle Bots, läuft weiter bis alle Sells gefüllt, dann Auto-Exit |
| Abbrechen | Bot läuft weiter |

**Endpoint:** `POST /api/bot/stop-graceful` mit `{"mode": "sell_all"|"wait_fills"}`
**Bot-Check:** Jede Loop-Iteration via `get_stop_mode()` aus `dashboard/db.py`

---

## Projektstruktur

```
trading-bot/
├── grid_bot.py           # Hauptlogik: PaperGridBot + LiveGridBot + run()
├── main.py               # Einstiegspunkt: --strategy grid|ema|rsi --mode paper|live
├── config.py             # ENV-Variablen (BINANCE_* = Kraken-Fallback), PAPER_TRADING-Flag
├── data_fetcher.py       # OHLCV + Ticker via ccxt (Kraken)
├── grid_optimizer.py     # Backtest-Sweep über Levels × Range-Kombinationen
├── grid_backtester.py    # Einzelner Grid-Backtest
├── notifier.py           # Telegram-Benachrichtigungen
├── start.sh / stop.sh    # Prozess-Management-Scripts
│
├── ml/                   # KI-Richtungsvorhersage
│   ├── model.py          # TradingModel: LightGBM + CalibratedClassifier + Walk-Forward
│   ├── trainer.py        # Triple-Barrier-Labels, Bootstrap, Retrain
│   ├── features.py       # 16 Features (EMA, RSI, Momentum, BB, ATR, Candles)
│   ├── predictor.py      # MLPredictor: predict() + async Retrain + LLM-Blending
│   ├── llm_analyst.py    # Claude Haiku: Marktanalyse + Score-Blending
│   └── data_store.py     # SQLite-Persistenz für ML-Samples
│
├── price_predictor/      # Regelbasierter Range-Predictor (ATR/Bollinger)
│   ├── predictor.py      # PricePredictor: predict() → low/high/regime/confidence
│   ├── indicators.py     # ATR(14), Bollinger(20,2σ), RSI(14), VWAP, ADX(14)
│   ├── grid_suggester.py # Grid-Levels aus Range berechnen
│   └── tests/            # pytest Tests für Indikatoren + Predictor
│
├── backtest/             # Strategie-Backtester (EMA/RSI)
├── dashboard/            # Flask-Dashboard Port 5001
│   ├── app.py            # API-Endpoints + SSE-Stream + Bot-Prozess-Management
│   ├── db.py             # SQLite-Schema + Helpers (log_trade, set_stop_mode, etc.)
│   └── templates/        # index.html: Grid-Visualisierung, Charts, Stop-Modal
│
├── src/                  # EMA/RSI-Strategie-Bots (klassisch)
│   ├── strategy/         # base_strategy, ema_crossover, rsi_mean_rev
│   ├── risk/             # RiskManager: Daily-Drawdown, Position Sizing
│   └── execution/        # PaperBroker, LiveBroker (ccxt)
│
├── scripts/
│   └── optimize.py       # Trading-Optimizer CLI
│
├── data/                 # SQLite-DBs (nicht in Git)
│   ├── trades.db         # Trades + grid_state + bot_status + predictions + equity
│   ├── ml_training.db    # ML-Samples (features + labels)
│   └── ohlcv_cache.db    # OHLCV-Cache
│
├── config/
│   ├── config.yaml       # Symbol-Liste (SOL, LINK, INJ, AVAX, NEAR), Risk-Parameter
│   └── strategy_params.yaml  # ATR-Perioden, Risk-Parameter
│
├── logs/                 # Logfiles (trading_bot.log, dashboard.log)
├── .env                  # API-Keys (niemals einchecken) – BINANCE_* = Kraken-Fallback
└── requirements.txt
```

---

## Schlüssel-Konstanten in grid_bot.py

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `CHECK_INTERVAL` | 15 | Sekunden zwischen Loop-Iterationen |
| `KRAKEN_FEE` | 0.0016 | Maker-Fee 0.16% — immer diese verwenden |
| `ATR_CANDLES` | 24 | ATR über 24h berechnen |
| `MAX_LOSS_PCT` | 0.05 | 5% vom Investment = Notbremse pro Coin |
| `PER_POS_SL_PCT` | 0.04 | 4% unter Buy-Preis = Per-Position-SL |
| `MAX_INVESTMENT_MULT` | 3.0 | Compounding-Cap: max. 3× Initial-Investment |
| `COMPOUND_EVERY_TRADES` | 5 | Gewinn-Reinvestition alle 5 Trades |
| `DEFAULT_LEVERAGE` | 1.0 | Startwert; änderbar per Dashboard |
| `PREDICTION_RECHECK` | 20 | Alle 20 Zyklen (5 Min) Vorhersage neu prüfen |
| `GRID_REBUILD_CYCLES` | 240 | Stündlicher Zwangs-Rebuild des Grids |
| `REGIME_CONFIGS` | dict | Level-Anzahl pro Regime (auto-tuned nach Win-Rate) |

### Directional-Trade-Konstanten

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `DIRECTIONAL_SCORE_MIN` | 0.08 | Score-Schwelle für neuen Einstieg |
| `DIRECTIONAL_PCT` | 0.15 | 15% des Investments pro Trade |
| `DIRECTIONAL_TP_ATR` | 2.5 | Take-Profit: Einstieg + 2.5 × ATR |
| `DIRECTIONAL_SL_ATR` | 1.5 | Stop-Loss: Einstieg − 1.5 × ATR |
| `DIRECTIONAL_DOWN_TRAIL_PCT` | 0.005 | Bei ML:DOWN + Position im Plus: erst nach 0.5% Rückgang verkaufen |
| `DIRECTIONAL_RECHECK_SCORE_MIN` | 0.25 | Nach SL: frische Live-Prediction braucht Score > 0.25 |

---

## Directional-Trade-Logik (wichtige Sonderfälle)

**ML:DOWN-Verhalten bei offener Directional-Position:**
- Position im **Plus**: Trail-Lock bei Signal-Zeitpunkt-Preis → Verkauf erst wenn Preis 0.5% fällt
- Position im **Minus**: ML:DOWN ignorieren → SL-Level übernimmt den Exit (Kurs kann noch wenden)
- Signal dreht zurück auf UP: Trail-Lock wird aufgehoben, Position bleibt offen

**Nach Directional-SL (wichtig gegen Instant-Re-Entry):**
- `_directional_needs_recheck = True` wird gesetzt
- Nächster Einstiegsversuch: frische Live-Prediction (kein Cache) + strikter Score > 0.25
- Verhindert das "INJ-Szenario": SL getroffen → sofort neuer Kauf in fallenden Coin

**Score-Konsistenz:**
- `predict_direction()` erzwingt: Score-Vorzeichen muss zur Direction passen
- Fallback Rule-Based aktualisiert `_last_scores` (war Bug: altes positives Score blieb bei DOWN)

---

## ML-Vorhersage-Pipeline

```
predict_direction(symbol)
  │
  ├── LightGBM (16 Features, Walk-Forward, CalibratedClassifier)
  │     └── lgbm_score + lgbm_conf
  │
  ├── Claude Haiku (llm_analyst.py)
  │     └── llm_score + llm_conf + reason
  │
  ├── blend_scores(lgbm, llm) → blended_score, blended_conf
  │
  ├── if blended_conf >= MIN_CONFIDENCE (0.45):
  │     score > +0.15 → "up", score < -0.15 → "down", sonst "neutral"
  │
  └── else: Fallback _rule_based() → aktualisiert _last_scores mit ±0.5
```

**Retrain:** Async via Threading, nur speichern wenn OOS-F1 ≥ 0.30
**ML-Refresh:** Täglich (24h), 720 OHLCV-Candles (1h)
**LLM-Cache:** `data/llm_cache.db` — verhindert doppelte API-Calls

---

## Regime-Logik (PricePredictor)

| Regime | Bedingung | Grid-Range | Levels (default) |
|--------|-----------|-----------|--------|
| trending | ADX > 25 | ATR × 2.0 | 6 |
| volatile | ATR% > 3% | ATR × 1.5 | 10 |
| ranging | sonst | Bollinger Bands | 8 |

REGIME_CONFIGS wird automatisch nach echten Win-Rates angepasst (Auto-Tune).

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
| `equity` | Kapital-Kurve (alle 15s) |
| `coin_settings` | Dashboard-Override: max_investment, enabled pro Symbol |
| `bot_status` | Einzeilig: running, mode, leverage, **stop_mode** (Graceful-Shutdown-Flag) |

**Wichtig:** `bot_status.stop_mode` ist das Kommunikations-Flag für Graceful Shutdown.
Nie manuell auf einen Wert setzen lassen – wird vom Bot nach Lesen automatisch auf NULL zurückgesetzt.

### `data/ml_training.db`
| `samples` | ML-Training-Samples: features (JSON), entry_price, label, predicted |

### `data/ohlcv_cache.db`
| `ohlcv` | OHLCV-Cache: symbol, timeframe, timestamp, OHLCV-Werte |

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
- **RiskManager:** Jede neue Strategie muss `src/risk/risk_manager.py` einbinden.
- **ATR:** Immer Fallback auf ATR wenn PricePredictor fehlschlägt.
- **Kein Hardcoding:** Exchange, Symbol und Fees immer als Konstanten.
- **ML-Retrain:** Nie im predict()-Hot-Path blockierend — Threading nutzen.
- **Walk-Forward:** Modelle nur speichern wenn OOS-F1 ≥ `MIN_OOS_F1 = 0.30`.
- **db.log_trade():** `context`-Dict mitliefern für Pattern-Mining.
- **Score-Vorzeichen:** Direction und Score müssen übereinstimmen (enforce in `predict_direction`).
- **stop_mode:** Nach Lesen immer auf NULL zurücksetzen — sonst blockiert Bot-Neustart.

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
```
