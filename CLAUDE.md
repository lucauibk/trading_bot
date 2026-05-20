# Grid Trading Bot – Vollständige Projektdokumentation

## Projektübersicht

Multi-Coin Grid Trading Bot für Kraken (Paper + Live). Kombiniert dynamisches Grid-Trading
mit ML-Vorhersagen (LightGBM + Walk-Forward), adaptiver Positionsgröße und einem
automatischen Trading-Optimizer-Skill.

**Ziel:** Maximaler Tagesgewinn bei minimalem Loss durch:
- ATR/Bollinger-basierte Grid-Ranges (PricePredictor)
- LightGBM-Richtungsvorhersage mit kalibrierter Konfidenz
- Per-Position-Stop-Loss (4% unter Buy-Preis)
- Cross-Coin Daily-Drawdown-Bremse (-3%)
- Compounding mit Investment-Cap (3× Initial)

---

## Projektstruktur

```
trading-bot/
├── grid_bot.py           # Hauptlogik: PaperGridBot + LiveGridBot + run()
├── main.py               # Einstiegspunkt: --strategy grid|ema|rsi --mode paper|live
├── config.py             # ENV-Variablen, API-Keys, PAPER_TRADING-Flag
├── data_fetcher.py       # OHLCV + Ticker via ccxt (Kraken)
├── grid_optimizer.py     # Backtest-Sweep über Levels × Range-Kombinationen
├── grid_backtester.py    # Einzelner Grid-Backtest
├── notifier.py           # Telegram-Benachrichtigungen
│
├── ml/                   # KI-Richtungsvorhersage (LightGBM)
│   ├── model.py          # TradingModel: LightGBM + CalibratedClassifier + Walk-Forward
│   ├── trainer.py        # Triple-Barrier-Labels, Bootstrap, Retrain
│   ├── features.py       # 16 Features (EMA, RSI, Momentum, BB, ATR, Candles)
│   ├── predictor.py      # MLPredictor: predict() + async Retrain
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
│   ├── app.py            # API-Endpoints + SSE-Stream
│   ├── db.py             # SQLite-Schema + Helpers: log_trade, log_prediction, etc.
│   └── templates/        # index.html: Grid-Visualisierung, Charts, Coin-Budget
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
│   ├── trades.db         # Trades + trade_context + grid_state + predictions + optimizer_runs
│   ├── ml_training.db    # ML-Samples (features + labels)
│   └── ohlcv_cache.db    # OHLCV-Cache
│
├── config/
│   ├── config.yaml       # Symbol-Liste, allgemeine Einstellungen
│   └── strategy_params.yaml  # ATR-Perioden, Risk-Parameter
│
├── logs/                 # Logfiles (grid_bot.log, trading_bot.log)
├── .env                  # API-Keys (niemals einchecken)
└── requirements.txt      # ccxt, pandas, ta, lightgbm, flask, scikit-learn, joblib
```

---

## Technischer Stack

- **Python 3.11+** (getestet mit 3.9+)
- **ccxt** — Exchange-Daten (Kraken)
- **pandas + ta** — Indikatoren
- **LightGBM + scikit-learn** — ML-Modell mit Walk-Forward + Kalibrierung
- **Flask** — Dashboard (Port 5001)
- **SQLite** — Persistenz (3 DBs)
- **pytest** — Tests

---

## Schlüssel-Konstanten in grid_bot.py

| Konstante | Wert | Bedeutung |
|-----------|------|-----------|
| `ATR_CANDLES` | 24 | ATR über 24h berechnen |
| `KRAKEN_FEE` | 0.0016 | Maker-Fee 0.16% — immer diese verwenden |
| `MAX_LOSS_PCT` | 0.08 | 8% vom aktuellen Investment = Notbremse |
| `PER_POS_SL_PCT` | 0.04 | 4% unter Buy-Preis = Per-Position-SL |
| `MAX_INVESTMENT_MULT` | 3.0 | Compounding-Cap: max. 3× Initial-Investment |
| `COMPOUND_EVERY_TRADES` | 5 | Gewinn-Reinvestition alle 5 Trades |
| `REGIME_CONFIGS` | dict | Level-Anzahl pro Regime (ranging/trending/volatile) |

---

## Konventionen (PFLICHT)

- **Fees:** Immer `KRAKEN_FEE = 0.0016` nutzen, nie hardcoded `0.001` oder `0.002`.
- **RiskManager:** Jede neue Strategie muss `src/risk/risk_manager.py` einbinden.
- **TDD:** Tests zuerst schreiben, dann Implementierung.
- **ATR:** Immer Fallback auf ATR wenn PricePredictor fehlschlägt.
- **Kein Hardcoding:** Exchange, Symbol und Fees nie hardcoded — immer Konstanten.
- **ML-Retrain:** Nie im predict()-Hot-Path blockierend — Threading nutzen.
- **Walk-Forward:** Modelle nur speichern wenn OOS-F1 ≥ `MIN_OOS_F1 = 0.30`.
- **db.log_trade():** `context`-Dict mitliefern für späteres Pattern-Mining.

---

## Datenbanken

### `data/trades.db`

| Tabelle | Inhalt |
|---------|--------|
| `trades` | Trade-History: timestamp, symbol, direction, entry, exit, pnl, reason, strategy |
| `trade_context` | Kontext zum Trade-Zeitpunkt: atr_pct, rsi, ema9/21, regime, ml_confidence, … |
| `grid_state` | Aktueller Grid-Status pro Coin (Live-Dashboard) |
| `grid_sessions` | Pro Session: profit, trades, max_dd, range_pct, levels |
| `predictions` | ML/PricePredictor-Vorhersagen mit realized outcome für Kalibrierung |
| `optimizer_runs` | Backtest-Sweep-Ergebnisse (params, score, daily_pct, max_dd) |
| `equity` | Kapital-Kurve (alle 15s) |
| `coin_settings` | Dashboard-Override: max_investment, enabled pro Symbol |

### `data/ml_training.db`

| Tabelle | Inhalt |
|---------|--------|
| `samples` | ML-Training-Samples: features (JSON), entry_price, label, predicted |

### `data/ohlcv_cache.db`

| Tabelle | Inhalt |
|---------|--------|
| `ohlcv` | OHLCV-Cache: symbol, timeframe, timestamp, OHLCV-Werte |

---

## Regime-Logik (PricePredictor)

| Regime | Bedingung | Grid-Range | Levels |
|--------|-----------|-----------|--------|
| trending | ADX > 25 | ATR × 2.0 | 6 |
| volatile | ATR% > 3% | ATR × 1.5 | 10 |
| ranging | sonst | Bollinger Bands | 8 |

---

## Trading-Optimizer Skill

Aufruf in Claude Code: `/trading-optimizer`

Analysiert `trades.db` und gibt Empfehlungen für:
- Grid-Parameter (REGIME_CONFIGS, range_pct)
- ML-Confidence-Threshold (MIN_CONFIDENCE in `ml/predictor.py`)
- Risiko-Filter (toxische Setups aus Pattern-Mining)

Manuell ausführen:
```bash
python3 scripts/optimize.py --analyze-trades --days 30
python3 scripts/optimize.py --calibration-report
python3 scripts/optimize.py --suggest-params
python3 scripts/optimize.py --pattern-mine
python3 scripts/optimize.py --run-sweep --symbol SOL/USD
```
