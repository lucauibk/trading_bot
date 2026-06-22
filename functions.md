# Grid Trading Bot — Funktions- und Algorithmus-Referenz

**Zweck:** Verifikations-Baseline. Jede Funktion beschreibt ihr **Soll-Verhalten** und die
**Invarianten**, gegen die man bei Änderungen oder Bugs prüft. Wenn Ist ≠ Soll, ist das ein Bug.

**Stand:** 2026-06-18 — aus vollständiger Lektüre aller Module erstellt.

---

## ⚠️ Bekannte Quirks — stehende Checks

Diese Punkte sind dokumentierte Abweichungen oder Risiken, die bei jeder Review geprüft werden
sollten:

| # | Quirk | Wo | Risiko |
|---|-------|----|--------|
| Q1 | **Silenter 34→16-Feature-Fallback** | `predictor.py:82-88`, `model.py:124-129` | Bei ANY Exception in `extract_all()` fällt die Pipeline auf 16 technische Features zurück. Das 34-Feature-Modell erkennt Dim-Mismatch → gibt `(hold, 0.0)` zurück. Damit fällt **jede Prediction** auf `_rule_based()` zurück — nur 1 WARNING im Log. Sieht aus wie normaler Betrieb. |
| Q2 | **btc_corr Train=0.0 / Live=0.7** | `trainer.py:35`, `predictor.py:74-77` | Training immer `btc_corr=0.0` (LGBM splittet nie auf konstantem Feature). Live: 0.7. Angeblich No-op — aber: wenn Modell je mit variantem btc_corr trainiert wird, verhält es sich anders. Stehender Check nach jeder Trainer-Änderung. |
| Q3 | **htf-Slot `trend_1d` enthält `dist_1d`** | `htf.py:63-72` | `FEATURE_NAMES[1]="trend_1d"`, aber `extract()` legt dort die EMA200-Distanz (1d) ab — keine Slope. Name ist misleading, Reihenfolge intern konsistent. Kein Bug, aber Verwirrungspotenzial beim Feature-Debugging. |
| Q4 | **perp Cache-Pfad: `oi_change_1h/24h = 0.0`** | `market/perp.py` | Wenn ein Cache-Hit vorliegt, wird `oi_series` nicht gecacht → `oi_change_1h/24h` ist 0.0. Nur frische API-Calls liefern echte OI-Änderungen. |
| Q5 | **Emergency-Sell PaperBroker** | `engine.py:366` | Synthetisiert Fills nur für sell-orders mit `bought_at` (echte Positionen). Pre-seeded Sells werden übersprungen — korrekt, weil kein Kapital dafür deponiert. |

---

## Schlüssel-Konstanten

| Konstante | Wert | Datei | Bedeutung |
|-----------|------|-------|-----------|
| `KRAKEN_FEE` | 0.0016 | `grid.py:27`, `paper.py:31`, `engine.py:482` | Maker-Fee 0.16% — immer diese verwenden |
| `CHECK_INTERVAL` | 15 | `engine.py:24` | Sekunden zwischen Loop-Iterationen |
| `PREDICTION_RECHECK` | 5 | `engine.py:25` | Alle 5 Ticks (~75s) Prediction neu |
| `GRID_REBUILD_CYCLES` | 60 | `engine.py:26` | Alle 60 Ticks (~15min) Grid neu aufbauen |
| `BTC_REFRESH_CYCLES` | 4 | `engine.py:27` | Alle 4 Ticks (~1min) BTC-Kontext neu |
| `FUNDING_REFRESH_CYCLES` | 240 | `engine.py:28` | Alle 240 Ticks (~1h) Funding neu |
| `EMERGENCY_STOP_PCT` | 0.12 | `engine.py:32` | 12% Realized-Loss pro Coin → Symbol pausieren |
| `COMPOUND_EVERY_TRADES` | 3 | `grid.py:29` | Compounding alle 3 Trades |
| `MAX_INVESTMENT_MULT` | 3.0 | `grid.py:30` | Compounding-Cap: max. 3× Initial |
| `ADAPTIVE_SIZING` | True | `grid.py:32` | Bullish → mehr Budget auf unteren Levels |
| `SIZE_BIAS_FACTOR` | 0.30 | `grid.py:33` | Bias-Stärke für adaptive Allokation |
| `DIRECTIONAL_RECHECK_SCORE_MIN` | 0.25 | `grid.py:36` | Score-Schwelle für Re-Entry nach SL |
| `DIRECTIONAL_DOWN_TRAIL_PCT` | 0.005 | `grid.py:37` | 0.5% Rückgang nach Signal-Flip → Verkauf |
| `DIRECTIONAL_COOLOFF_SECONDS` | 14400 | `grid.py:38` | 4h Cooloff nach Directional-SL |
| `NO_DIRECTIONAL_HOURS` | {5,6,7,8} | `grid.py:39` | UTC-Stunden mit negativem EV |
| `MIN_STEP_FEE_MULTIPLE` | 4.0 | `grid.py:28` | Step muss ≥4× Round-Trip-Fee sein |
| `MIN_SAMPLES` | 100 | `model.py:23` | Mindest-Samples für Training |
| `MIN_OOS_F1` | 0.30 | `model.py:19` | Modell nur gespeichert wenn OOS-F1 ≥ 0.30 |
| `MIN_CONFIDENCE` | 0.45 | `predictor.py:17` | Min Blend-Confidence für direktionales Signal |
| `LLM_WEIGHT` | 0.45 | `llm_analyst.py:18` | Blend: 0.55×LGBM + 0.45×LLM |
| `LLM_CONFIDENCE_MIN` | 0.60 | `llm_analyst.py:17` | LLM wird nur geblended wenn conf ≥ 0.60 |
| `CACHE_SECONDS` (LLM) | 3600 | `llm_analyst.py:19` | LLM-Ergebnis 1h gecacht |
| `LOOKFORWARD_H` | 2 | `trainer.py:15` | Triple-Barrier Timeout = 2 Stunden |
| `RETRAIN_EVERY_N` | 50 | `trainer.py:16` | Retrain nach 50 neuen gelabelten Samples |
| `TB_UPPER_ATR` | 0.5 | `trainer.py:19` | Gewinn-Barriere = 0.5× ATR |
| `TB_LOWER_ATR` | 0.5 | `trainer.py:20` | Verlust-Barriere = 0.5× ATR |
| `N_FEATURES` | 34 | `combined.py:23` | technical(16) + perp(4) + market(5) + htf(4) + seasonality(5) |
| `SLIPPAGE_BPS` | 3 | `paper.py:32` | 3 Basispunkte Slippage im Paper-Modus |
| `_ADX_TRENDING_THRESHOLD` | 25.0 | `price_predictor/predictor.py:14` | ADX>25 → trending |
| `_ATR_PCT_VOLATILE_THRESHOLD` | 3.0 | `price_predictor/predictor.py:15` | ATR%>3% → volatile |

### GridParams-Defaults (frozen dataclass, `grid_params.py`)

| Parameter | Default | Bedeutung |
|-----------|---------|-----------|
| `sl_mode` | `"floor"` | Floor-SL unter Gridboden (kaskadensicher) |
| `floor_sl_atr_mult` | 1.0 | Floor = grid_lower − 1.0×ATR |
| `per_pos_sl_max_pct` | 0.04 | Hard-Cap: Per-Position-SL max. 4% |
| `momentum_hold_score` | 0.35 | SL verzögern wenn score > 0.35 |
| `momentum_hold_max` | 2 | Max 2 Ticks SL-Verzögerung |
| `levels_by_regime` | ranging:14, trending:6, volatile:20 | Grid-Level pro Regime |
| `range_atr_mult_trending` | 2.0 | Half-Range = 2×ATR im trending-Regime |
| `range_atr_mult_volatile` | 1.5 | Half-Range = 1.5×ATR im volatile-Regime |
| `min_step_pct` | 0.006 | Mindest-Step 0.6% (≈1.9× Round-Trip) |
| `trend_filter_enabled` | True | EMA/ADX-Trend-Filter aktiv |
| `trend_adx_min` | 25.0 | ADX-Schwelle für Hard-Downtrend |
| `leverage` | 0.0 | 0 = live aus Dashboard-DB lesen |
| `directional_enabled` | True | Directional-Trades aktiv |
| `directional_score_min` | 0.12 | Min-Score für neuen Directional-Entry |
| `directional_pct` | 0.20 | 20% des Investments pro Directional |
| `directional_tp_atr` | 3.0 | TP = Entry + 3.0×ATR |
| `directional_sl_atr` | 1.5 | SL = Entry − 1.5×ATR |
| `max_inventory_notional_mult` | 2.0 | Stop neue Buys wenn deployed ≥ 2×Investment |
| `min_confidence_to_buy` | 0.0 | PricePredictor-Conf-Gate (0=off) |
| `floor_sl_per_cohort` | False | Jeder Rebuild-Cohort behält eigenen Floor |

---

## End-to-End-Flows

### A — Normaler 15s Tick-Loop

```
Engine.run()  →  while ShutdownFlag.is_running():
  Engine._tick()
    1. _check_dashboard_stop()       # DB: stop_mode → sell_all | wait_fills
    2. _refresh_btc() (alle 4 Ticks)
    3. _refresh_funding() (alle 240 Ticks)
    4. _check_daily_drawdown()       # freeze wenn DD > 10%
    5. fetch_ticker() alle Coins     # prices dict
    6. Pro Coin:
       a. Emergency-Stop-Check (total_profit < -12% investment) → skip
       b. on_candle() alle 5 Ticks   # ATR, trend-filter, prediction refresh
       c. out_of_range prüfen        # price außerhalb [lo*0.99, hi*1.01]
       d. on_tick_safety()           # SL/TP (auch bei freeze)
       e. on_tick() (wenn !frozen)   # _check_position_stops, _check_directional,
                                     # _maybe_open_directional, _update_trailing_stops
       f. setup_grid() (wenn out_of_range ODER alle 60 Ticks)
       g. _sync_orders() (wenn !frozen) # fills → on_fill → cancel/place
       h. _update_dashboard()
    7. _log_equity()
    8. _update_prediction_outcomes() (alle 60 Ticks)
  sleep(CHECK_INTERVAL=15s)
```

### B — Prediction-Pipeline (MLPredictor.predict)

```
predict(symbol)
  1. fetch 120×1h OHLCV
  2. extract_all(df, funding, btc, btc_corr=0.7, dt) → 34-Feature-Vektor
     └─ Bei Exception: WARNING + 16-Feature-Fallback extract_features(df)   [⚠️ Q1]
  3. model.predict(feats)
     └─ Bei Dim-Mismatch: return (hold=1, 0.0)                              [⚠️ Q1]
  4. lgbm_score = label_to_sign × lgbm_conf  (sell=-1, hold=0, buy=+1)
  5. record sample + submit async label_and_maybe_retrain
  6. llm_analyst.analyse(symbol, indicators) → {direction, confidence, score}
  7. blend_scores(lgbm, llm) → (blended_score, blended_conf)
     └─ Blend nur wenn llm_conf ≥ 0.60; else pure LGBM
  8. if blended_conf ≥ 0.45:
       score > +0.15  → "up"
       score < −0.15  → "down"
       else           → "neutral"
  9. else: _rule_based() → ±0.5
  Store score → state._direction_score
  state.with_position = (direction != "down")
```

### C — Fill → Sell → Compound

```
PaperBroker.update_price(symbol, price)
  └─ Buy-Order getriggert:
       cost = fill_price × qty / leverage + fee
       _deduct(symbol, cost)                    # Margin deponieren
       → strategy.on_fill(fill)
           → _handle_buy_fill()
               Sell-Order erstellen (einen Level höher)
               sl_price = floor_sl (oder per-position mit Hard-Cap 4%)
               ctx.add_position()

  └─ Sell-Order getriggert:
       credit = margin_return + leveraged_PnL − fee  (normal)
       credit = (fill_price − bought_at) × qty − fee  (pre_seeded: kein margin)
       _credit(symbol, credit)
       → strategy.on_fill(fill)
           → _handle_sell_fill()
               net = (sell_price − buy_price) × qty − fees
               state.total_profit += net
               state.trade_count += 1
               Smart-Replenish: neuer Buy (1 Level höher wenn bullish)
               ctx.remove_position()
               _maybe_compound()
                 wenn (trade_count − last_compound_at) ≥ 3:
                   state.investment = min(investment + delta, 3× initial)
```

### D — Graceful Shutdown

```
Dashboard → POST /api/bot/stop-graceful {"mode": "sell_all"|"wait_fills"}
  └─ db.set_stop_mode(mode)

Engine._check_dashboard_stop() (jeden Tick):
  mode = db.get_stop_mode()
  set_stop_mode(None)              # ← sofort zurücksetzen, MUSS passieren
  
  "sell_all":
    _emergency_sell_all()          # cancel_all + market sell / synth fills
    _shutdown.stop()
    
  "wait_fills":
    state.with_position = False    # alle Coins: keine neuen Buys
    _waiting_for_fills = True      # auto-terminate wenn alle Sells gefüllt
```

---

## core/engine.py

### `Engine.__init__`
Instanziiert Strategy, Broker, Symbols, MarketContext, Reconciler, initial_capital.
`_active_orders: Dict[symbol, Dict[client_id, BrokerOrder]]` — eigene Orderbuch-Kopie.
`_last_prices: Dict[symbol, float]` — für MTM-Equity und ml_score-Logging.
`_shutdown: ShutdownFlag` — SIGTERM/SIGINT Handler.

### `Engine.run()`
Führt beim Start aus: stale `stop_mode` auf NULL setzen, `strategy.init()`, initiale
BTC/Funding-Refresh, erstes `on_candle + setup_grid` pro Coin. Dann Event-Loop (→ Flow A).
**Invariante:** stop_mode wird VOR erstem Tick gecleart (verhindert Phantom-Stop von vorheriger Session).

### `Engine._tick()`
Kern-Iteration (→ Flow A).
**Invariante:** `on_tick_safety()` wird **immer** aufgerufen (auch bei freeze) — Stop-Losses
laufen auch während DD-Freeze. `setup_grid()` läuft auch bei freeze, aber `_sync_orders()` nicht
(neue Orders werden nicht platziert). Emergency-Stop pro Coin: `total_profit ≤ −12% × investment`
→ Symbol-Loop `continue` (kein Order-Management mehr für diesen Coin in diesem Tick).

### `Engine._sync_orders(symbol, price)`
1. `process_paper_fills()` — PaperBroker: `update_price()` → Fills → `strategy.on_fill()`.
2. `desired_orders()` vom Strategy → Set der gewünschten Client-IDs.
3. Aktive Orders die nicht mehr im desired-Set → `broker.cancel()`.
4. Gewünschte Orders die noch nicht aktiv → `broker.place_limit()`.
**Invariante:** Cancel-before-place — verhindert Doppelorders. Bei live: Reconciler trackt
Exchange-IDs.

### `Engine._check_daily_drawdown()`
Baseline = `_initial_capital` (aus Config/Dashboard), nicht Mid-Session-Equity.
`rm.set_daily_start(baseline)` → `daily_drawdown_ok(total_equity)`.
Wenn `dd > max_daily_drawdown (10%)`: `ctx.set_freeze(True)` → keine neuen Buys/Sells via
`_sync_orders()`, aber SL-Checks laufen weiter via `on_tick_safety()`.
**Invariante:** Baseline = initial_capital, NICHT current equity (verhindert doppeltes Zählen).

### `Engine._log_equity()`
`total = cash_balance + MTM open positions`.
MTM per Position = `qty × bought_at / leverage + qty × (price − bought_at)`
= Margin-Rückgabe + unrealisierter PnL.
**Invariante:** Pre-seeded sells werden NICHT zu MTM gezählt (kein Kapital deponiert).
Würde man `qty × price` nehmen, wäre Equity bei Leverage>1 überhöht.

### `Engine._emergency_sell_all()`
Paper: Synth-Fills für alle sell-with-bought_at (echte Positionen) zum aktuellen Preis.
Live: `broker.place_market()` für alle Context-Positions.
**Invariante:** Pre-seeded sells übersprungen (kein echtes Kapital gebunden).

### `Engine._check_dashboard_stop()`
Liest `stop_mode` aus DB, setzt es **sofort auf NULL** zurück — bevor Aktion ausgeführt wird.
**Invariante:** stop_mode MUSS nach Lesen NULL sein, sonst blockiert ein Bot-Neustart.

---

## strategies/grid.py + grid_params.py

### `GridStrategy.init(symbols, ctx)`
Erstellt `_GridState` pro Symbol. Initialisiert `MLPredictor` (bootstrap wenn kein Modell).
Initialisiert `PricePredictor` pro Symbol. Beides Exception-safe (Bot läuft auch ohne ML).

### `on_candle(symbol, df, ctx)`
Berechnet ATR(14), speichert `_last_df`. Führt `_update_trend_filter()` und
`_refresh_prediction()` aus. Wird alle 5 Ticks (~75s) plus bei out-of-range aufgerufen.

### `_update_trend_filter(state, df)`
**Hard-Downtrend-Erkennung** (ML-unabhängig):
`hard_down = (EMA9 < EMA21 < EMA50)  OR  (ADX > trend_adx_min=25.0 AND DI− > DI+)`
Wenn `hard_down=True`: `state._hard_trend_down = True`, `_trend_up_count = 0`.
**Exit-Hysterese:** Erst nach **2 aufeinanderfolgenden** non-hard-down Candles cleared.
**Invariante:** False-Positives (kurzer Bounce) lösen keinen vorzeitigen Resume aus.

### `_buys_allowed(state) → bool`
Gate für ALLE Buy-Emissionen (setup_grid, desired_orders, smart-replenish).
Gibt False wenn:
1. `not state.with_position` (ML-Prediction = "down")
2. `trend_filter_enabled AND _hard_trend_down`
3. `_deployed_notional ≥ max_inventory_notional_mult × investment` (Inventory-Cap)
4. `min_confidence_to_buy > 0 AND _last_confidence < min_confidence_to_buy`
**Invariante:** Alle Buys gehen durch dieses Gate — es gibt keinen anderen Buy-Pfad.

### `_refresh_prediction(symbol, df, ctx)`
Ruft `ml_predictor.predict()` auf. Normalisiert Score-Vorzeichen (up→positiv, down→negativ).
**Setzt `state.with_position = (direction != "down")`** — das ist der Mechanismus für die
ML-basierte Buy-Sperre. Fallback wenn kein ML: EMA9 vs EMA21 + RSI.

### `_build_grid_params(symbol, price, state) → (lower, upper, levels, range_pct, regime, conf)`
Versucht zuerst `PricePredictor.predict()`. Fallback: Inline ATR/BB-Berechnung aus `_last_df`.
Regime-Lookup für levels: `ranging→14, trending→6, volatile→20` (aus GridParams).
Minimum-Range: `KRAKEN_FEE × levels × MIN_STEP_FEE_MULTIPLE` — stellt sicher dass jeder Step
die Fees übertrifft. Level-Cap: wenn `step_pct < min_step_pct (0.006)`, reduziere levels
(range bleibt erhalten).

### `setup_grid(symbol, price, ctx)`
1. Berechnet Grid-Parameter via `_build_grid_params()`.
2. `grid_lines = [lower + (i+0.5)×step for i in range(levels)]` — Level-Mittelpunkte.
3. Floor-SL setzen: `floor_sl = grid_lower − 1.0×ATR`.
4. **Floor-Ratchet (Soll-Invariante):** Beim Rebuild werden bestehende Positionen NICHT unter
   ihren aktuellen SL gesenkt: `o["sl_price"] = max(o["sl_price"], state.floor_sl)`.
   Ausnahme: `floor_sl_per_cohort=True` — dann behält jede Cohort ihren eigenen Floor.
5. Bestehende echte Positionen (sell + bought_at + nicht pre_seeded) werden **übernommen**.
   Pre-seeded Sells werden immer neu erstellt (sonst akkumulieren sie über Rebuilds).
6. Unter aktuellem Preis (Buys): nur wenn `_buys_allowed()` → buy-Order.
7. Über aktuellem Preis (Sells): immer → pre_seeded sell mit `bought_at=current_price`.
8. Adaptive Allokation via `_calc_level_allocations()`: bullish (score>0.05) → mehr Budget
   auf unteren Levels.

### `desired_orders(symbol, price, ctx) → List[Order]`
Liefert die vom Engine gewünschte Ordermenge. Filtert:
- gefüllte Orders
- Buys wenn `not _buys_allowed()`
- neue Buys (ohne bought_at) wenn BTC-Crash: `btc.trend=="down" AND btc.return_1h < −2%`

### `_handle_buy_fill(fill, state, ctx)`
Bei Buy-Fill: erstellt sofort Sell-Order einen Level höher.
SL-Zuweisung:
- `sl_mode="floor"`: `sl_price = order["cohort_floor"] if floor_sl_per_cohort else state.floor_sl`
- `sl_mode="per_position"`: `step_pct × 1.5`, mind. 0.8%, **max. 4% (Hard-Cap)**
Registriert Position in `ctx`.

### `_handle_sell_fill(fill, state, ctx)`
PnL = `(sell − buy) × qty − (sell + buy) × qty × KRAKEN_FEE`.
`state.total_profit += net; state.trade_count += 1`.
Logged Trade in dashboard DB mit Context-Dict (regime, ml-Felder).
**Smart-Replenish:** Wenn `_buys_allowed()`, neuer Buy-Order — bei bullish (score>0.1) einen
Level höher, sonst auf demselben Level.
`_maybe_compound()` nach jedem Sell-Fill.

### `_maybe_compound(price, state)`
Alle `COMPOUND_EVERY_TRADES=3` Trades:
`state.investment = min(investment + new_profit_delta, initial × 3.0)`
`state.usdt_per_grid = investment / levels`
**Invariante:** Investment überschreitet niemals `3 × _initial_investment`.

### `_check_position_stops(symbol, price, state, ctx)`
Iteriert alle offenen Sell-Orders mit `sl_price`.
Wenn `price ≤ sl_price`:
- Momentum-Hold: wenn `_direction_score > 0.35` und Hold-Count < 2 → Tick überspringen.
- Sonst: SL-Execution — PnL berechnen, order["filled"]=True, log_trade("stop_loss").
**Invariante:** SL-Checks laufen auch während freeze (via `on_tick_safety()`).

### `_update_trailing_stops(symbol, price, state)`
Trailing-Stop-Aktivierung: wenn `profit_in_atr ≥ 1.0` → SL auf Break-Even heben.
Nach Aktivierung: `trailing_sl = price − 1.5×ATR`, nur wenn neuer SL > alter SL (Ratchet).

### `_maybe_open_directional(symbol, price, state, ctx)`
**Gates** (alle müssen True sein):
1. `directional_enabled AND _buys_allowed()`
2. Kein offener Directional
3. UTC-Stunde NOT in `{5,6,7,8}` (NO_DIRECTIONAL_HOURS)
4. Cooloff seit letztem SL < 4h
5. `_directional_needs_recheck = False` ODER (score ≥ 0.25 → Flag resetten)
6. `_last_prediction == "up" AND _direction_score ≥ 0.12`
7. `btc.trend != "down" AND btc.return_1h ≥ −3%` (BTC-Makro-Guard)
8. `RiskManager.can_open()` = True
Entry: `qty = investment × 0.20 × leverage / price`, `TP = price + 3×ATR`, `SL = price − 1.5×ATR`.

### `_check_directional(symbol, price, state, ctx)`
Exit-Trigger:
- `price ≥ tp` → TP
- `price ≤ sl` → SL (setzt `_directional_needs_recheck=True`, `_directional_sl_ts=now`)
- **Signal-Flip:** score < 0 (signal_down) UND `pnl_pct ≥ 0` (in profit) UND Preis fällt
  ≥0.5% unter den Preis beim ersten Signal-Down → "Signal-Flip"-Sell.
  Wenn Signal wieder positiv wird, `signal_down_price` löschen (kein Flip mehr).

---

## execution/paper.py — PaperBroker

### Per-Symbol-Budget-Isolation
`__init__`: `_balances = {symbol: initial_balance / n_symbols}` — jeder Coin hat eigenes Bucket.
`_sym_balance(symbol)` → Bucket des Coins (oder globaler Fallback wenn Symbol nicht registriert).
**Invariante:** Ein Coin kann nicht das Budget eines anderen leeren.

### `update_price(symbol, price) → List[Fill]`
Prüft alle offenen Orders dieses Symbols auf Fill:
- Buy: `price ≤ order.price`
- Sell: `price ≥ order.price`
- Same-Tick-Guard: Order die in diesem Tick platziert wurde, kann nicht sofort fillen.
Slippage: 3 Basispunkte (Buy: aufschlagen, Sell: abzíehen).

**Margin-Accounting bei Buy:**
`cost = fill_price × qty / leverage + fee`
→ `_deduct(symbol, cost)` — nur Margin, nicht voller Notional.

**Margin-Accounting bei Sell (normal):**
`credit = (bought_at × qty / leverage) + (fill_price − bought_at) × qty − fee`
= Margin-Rückgabe + leveraged PnL − Fee.

**Pre-seeded Sell:**
`credit = (fill_price − bought_at) × qty − fee` — kein Margin-Return (nie deponiert).
**Invariante:** Pre-seeded Sell bucht nicht mehr als Profit-Delta — sonst würde Kapital aus dem
Nichts entstehen.

### `get_balance("USD") → float`
Summe aller Symbol-Buckets (oder globaler Fallback).

---

## risk/manager.py — RiskManager

### `can_open(symbol, usdt_size, ctx) → (bool, reason)`
5 Checks in Reihenfolge:
1. **Daily Drawdown:** `daily_drawdown_ok(equity)` = `(equity − start) / start > −10%`
2. **BTC Crash:** `btc.return_1h < −3%` ODER `btc.realized_vol_7d > 150%` (annualisiert) → block
3. **Max Open Positions:** `ctx.open_position_count() ≥ 3` (aus config.yaml) → block
4. **Position Size Cap:** `usdt_size > equity × 10%` → block
5. **Korrelations-Bucket:** Wenn Symbol hochkorreliert (>0.85) und schon ≥2 hochkorrelierte
   Positionen offen → block
Gibt `(True, "")` wenn alle Checks pass.

### `daily_drawdown_ok(equity) → bool`
`dd = (equity − daily_start) / daily_start`
True wenn `dd > −max_daily_drawdown (−10%)`.
**Invariante:** Wenn `daily_start ≤ 0`, gibt True zurück (kein false-negative).

### `set_daily_start(equity)`
Setzt Baseline nur bei Tageswechsel oder wenn noch 0.0.
**Invariante:** Baseline wird nicht mid-day geändert (keine Drift des DD-Ankerpunkts).

### `position_size(symbol, equity, ...) → float`
Empfehlung via `compute_position_usdt()` (Kelly + Vol-Target, gecappt bei 10% equity).
Wird von Engine/Strategy nicht für Grid-Sizing verwendet (die nutzen investment / levels direkt).
Wird für Directional über `can_open()` gate-check genutzt.

---

## risk/sizing.py

### `kelly_fraction(win_rate, win_loss_ratio, kelly_factor=0.25) → float`
Full Kelly = `(p × b − q) / b`. Quarter-Kelly = `full × 0.25`. Gecappt bei `kelly_factor`.
Fallback 0.01 wenn Parameter invalid.

### `vol_target_size(equity, target_risk_pct, realized_vol, price, leverage) → float`
Qty sodass 1-Tag-PnL-Std ≈ `target_risk_pct × equity`.
`qty = equity × target_risk_pct / (price × realized_vol × leverage)`

### `compute_position_usdt(...) → float`
`min(kelly_usdt, vol_target_usdt)`, gecappt bei `max_position_pct × equity (10%)`.
Floor: `0.5% × equity`.

---

## ml/model.py — TradingModel

### `train(X, y)`
Walk-Forward mit `TimeSeriesSplit(n_splits=5)`. Letzter Fold = OOS.
Base: `LightGBM(n_estimators=400, max_depth=6, lr=0.05, class_weight="balanced")`.
**OOS-Gate:** Wenn Walk-Forward Macro-F1 < `MIN_OOS_F1=0.30` → Modell wird NICHT gespeichert.
Final: `CalibratedClassifierCV(method="isotonic")`.
Label-Map: 0=sell, 1=hold, 2=buy.

### `predict(x) → (label_int, confidence)`
Gibt `(1, 0.0)` = hold wenn Modell nicht ready.
**Dim-Mismatch-Guard:** Wenn `len(x) != stored_feature_count` → `(1, 0.0)` statt Exception.
→ Das ist die zweite Verteidigungslinie gegen den 34→16-Fallback (⚠️ Q1).

### `_load()`
Lädt joblib+meta. Wenn gespeicherte Feature-Count ≠ 34 → `_clf=None` (Modell verworfen).
→ Bootstrap-Retrain wird beim nächsten `initialize()` getriggert.

---

## ml/trainer.py

### `_compute_label_triple_barrier(df, idx, atr_pct) → 0|1|2`
López-de-Prado Triple-Barrier:
- `upper = p0 × (1 + 0.5 × atr_pct)`, `lower = p0 × (1 − 0.5 × atr_pct)`
- Scan `idx+1..idx+LOOKFORWARD_H`: erstes `high ≥ upper → 2 (buy)`, `low ≤ lower → 0 (sell)`
- Timeout → 1 (hold)
- Wenn `idx + LOOKFORWARD_H ≥ len(df)` → 1 (kein Future-Blick möglich)

### `bootstrap_from_history(symbol, df, store, model)`
Generiert Features+Labels aus historischem OHLCV, speichert in DB, trainiert Modell.
Benötigt ≥100 Samples.

### `refresh_from_recent_history(symbol, df, store, model)`
Täglicher LightGBM-Retrain (ohne LLM). Rollback wenn neues OOS-F1 < altes F1 − 0.05.

### `ModelTrainer.label_and_maybe_retrain(symbol, current_df)`
Labelt gereifte Samples (älter als 2h). Retrain wenn `≥50` neue gelabelte Samples seit letztem
Retrain vorhanden. Async (ThreadPoolExecutor, max_workers=1).

---

## ml/predictor.py — MLPredictor

### `predict(symbol) → "up"|"down"|"neutral"`
(→ Flow B)
**Score-Vorzeichen-Konvention:** sell=−1, hold=0, buy=+1, jeweils ×confidence.
Direction-Bänder: score > +0.15 → "up", score < −0.15 → "down", sonst "neutral".
Fallback `_rule_based()` wenn blended_conf < 0.45.

### `_rule_based(df) → str`
Rein regelbasierter Fallback (≠ PricePredictor):
Integer-Score aus EMA-Cross, 4h-Momentum, RSI, MACD, BB%, Volumen-Surge, Kerzenformationen.
`≥3 → "up"`, `≤−3 → "down"`, sonst "neutral".

---

## ml/llm_analyst.py — Claude Haiku Analyst

### `analyse(symbol, indicators) → dict|None`
Ruft Claude Haiku auf (`claude-haiku-4-5-20251001`), cached 1h in RAM + SQLite.
Gibt None wenn kein API-Key oder API-Fehler.
LLM-Score = `{up:+1, neutral:0, down:−1}[direction] × confidence`.

### `blend_scores(lgbm_score, lgbm_conf, llm_result) → (score, conf)`
Wenn `llm_result is None` ODER `llm_conf < 0.60` → LGBM unverändert.
Sonst: `0.55 × lgbm + 0.45 × llm` (Scores + Confidences).

---

## ml/features/ — 34-Feature-Vektor

**Reihenfolge (fest, model-kritisch):**
`technical(16) + perp(4) + market(5) + htf(4) + seasonality(5)`

| Gruppe | Features | Modul |
|--------|---------|-------|
| technical | ema9_ratio, ema21_ratio, ema_cross, rsi/100, mom_1h/4h/12h, macd_hist, bb_pct, bb_width, vol_ratio, atr_pct, body_pct, upper/lower_shadow_pct, is_green | `technical.py` |
| perp | funding_rate×1000, funding_z7d, oi_change_1h, oi_change_24h | `perp.py` |
| market | btc_return_1h/4h/24h, btc_corr_30d, btc_dominance | `market.py` |
| htf | trend_4h (EMA50-Slope), **dist_1d** (Name: trend_1d ⚠️Q3), dist_4h_ema200, htf_rsi_4h | `htf.py` |
| seasonality | hour_sin, hour_cos, dow_sin, dow_cos, is_weekend | `seasonality.py` |

Alle Extraktoren geben 0-Vektoren wenn Input fehlt (graceful degradation). `nan_to_num` am Ende.

---

## price_predictor/ — Regelbasierter Range-Predictor

**Vollständig unabhängig von ML** — eigene Fetch, eigene Indicators, nur ATR/Bollinger-Logik.

### `PricePredictor.predict() → dict`
Gibt `{predicted_low, predicted_high, confidence, grid_levels, regime}`.

### `_determine_regime_and_range(row) → (regime, low, high)`
| Bedingung | Regime | Methode |
|-----------|--------|---------|
| ADX > 25 | trending | `price ± 2.0×ATR` |
| ATR% > 3% | volatile | `price ± 1.5×ATR` |
| BB valide | ranging | BB-Bänder |
| Fallback | ranging | `price ± 1.5×ATR` |

### `compute_grid_levels(low, high, count=10) → List[float]`
`np.linspace(low, high, count)`, gerundet auf 8dp. Raise ValueError wenn count<2 oder low≥high.

---

## market/btc_context.py

### `get_btc_context(force_refresh=False) → BTCContext|None`
Gecacht 1h. Fetcht 500×1h BTC/USD. Trend via 4h EMA200 mit ±1% Band:
- `4h_close > EMA200×1.01` → "up"
- `4h_close < EMA200×0.99` → "down"
- sonst → "range"
Returns stale cache bei Fehler (nie None wenn je erfolgreich geladen).
**Verwendung:** BTC ist kein getradeter Coin — er ist Makro-Barometer für Alt-Risiko.

---

## execution/kraken.py — KrakenBroker

### `place_limit(...)`
PostOnly-Order via ccxt. Retry mit Exponential Backoff (`_with_retry`).
`client_id` als `clientOrderId` übergeben.
Leveraged Orders: `params["leverage"] = meta["leverage"]`.

### `reconcile_fills(since_ts) → List[Fill]`
Fetcht `fetchMyTrades` ab `since_ts`. Mappt Exchange-Fills auf interne `Fill`-Objekte.

---

## execution/reconciler.py — Reconciler (Live-Modus)

### `track_order(client_id, exchange_id, symbol, side, price, qty)`
Speichert Order in SQLite (`data/reconciler.db`).

### `reconcile() → List[Fill]`
Fetcht neue Fills vom Exchange ab `_last_ts`. Matched Exchange-Order-IDs mit tracked orders.
Gibt gematchte Fills zurück, updatet `_last_ts`.

---

## dashboard/db.py — Datenbankschema & Helpers

### Tabellen

| Tabelle | Inhalt |
|---------|--------|
| `trades` | Trade-History: timestamp, symbol, direction, entry, exit, pnl, reason, leverage |
| `trade_context` | Kontext pro Trade: regime, atr_pct, rsi, ema9/21, ml_confidence, ... |
| `grid_state` | Aktueller Grid pro Coin: price, orders (JSON), range_pct, investment, profit, prediction |
| `grid_sessions` | Pro Session: profit, trades, max_dd, range_pct, levels |
| `predictions` | ML-Vorhersagen: prediction, confidence, ml_score, entry_price, realized_high/low/hit |
| `optimizer_runs` | Backtest-Sweep-Ergebnisse |
| `equity` | Kapital-Kurve (alle 15s, Mark-to-Market) |
| `coin_settings` | Dashboard-Override: max_investment, enabled pro Symbol |
| `bot_status` | Einzeilig: running, mode, leverage, **stop_mode** (Graceful-Flag) |

### Wichtige Helpers

| Funktion | Soll |
|----------|------|
| `log_trade(symbol, ...)` | INSERT trade + trade_context. Context-Dict für Pattern-Mining. |
| `log_prediction(symbol, ..., ml_score, entry_price)` | INSERT prediction. ml_score+entry_price für ML-Kalibrierung getrennt von PricePredictor-Bounds. |
| `update_prediction_outcomes(fetch_ohlcv_fn)` | Füllt realized_high/low/hit für gereifte Predictions (~6h alt). ML-Logik: up-hit wenn r_high ≥ entry×1.003, down-hit wenn r_low ≤ entry×0.997. |
| `update_grid_state(symbol, ...)` | UPSERT grid_state. Orders als JSON gespeichert. |
| `get_stop_mode()` | Liest stop_mode aus bot_status. Bot setzt es danach auf NULL. |
| `set_stop_mode(mode)` | Schreibt stop_mode. Dashboard nutzt dies für Graceful-Stop. |
| `set_leverage(value)` | Schreibt Leverage. Bot liest via `_get_leverage()` jede Iteration. |
| `reset_stats()` | Setzt trades, equity, grid_state, predictions zurück. |

---

## dashboard/app.py — Flask-Routes

| Route | Methode | Funktion |
|-------|---------|----------|
| `/` | GET | index.html Dashboard |
| `/api/bot/start` | POST | Bot als Subprocess starten (`main.py --mode paper\|live`) |
| `/api/bot/stop` | POST | Bot sofort via SIGTERM beenden |
| `/api/bot/stop-graceful` | POST `{"mode":"sell_all"\|"wait_fills"}` | Graceful Stop via DB-Flag |
| `/api/bot/restart` | POST | Stop + Start |
| `/api/shutdown` | POST | Dashboard-Prozess selbst beenden |
| `/api/status` | GET | Bot-Status, Leverage, Mode, running |
| `/api/trades` | GET | Trade-History (letzte N) |
| `/api/equity` | GET | Equity-Kurve |
| `/api/grids` | GET | Alle Grid-States |
| `/api/summary` | GET | Zusammenfassung: PnL, Trade-Count, Win-Rate |
| `/api/leverage` GET/POST | GET/POST | Leverage lesen/setzen |
| `/api/capital` GET/POST | GET/POST | Initial-Capital lesen/setzen |
| `/api/coin-settings` GET/POST | GET/POST | max_investment, enabled pro Symbol |
| `/api/stats/reset` | POST | Statistiken zurücksetzen |
| `/stream` | GET (SSE) | Live-Updates alle 15s |

---

## Weitere Module (Übersicht)

### core/lifecycle.py

| Funktion | Soll |
|----------|------|
| `acquire_singleton()` | `fcntl.flock(LOCK_EX\|LOCK_NB)` — verhindert doppelten Bot-Start. Fehler → sys.exit(1). |
| `release_singleton()` | Lock freigeben beim Shutdown. |
| `ShutdownFlag` | SIGTERM/SIGINT → `is_running()=False`. Thread-safe via `threading.Event`. |

### core/context.py — MarketContext

Thread-safe Container für geteilten Zustand zwischen Engine und Strategy.

| Methode | Soll |
|---------|------|
| `set_btc / get_btc` | BTC-Kontext (für Guards) |
| `set_funding / get_funding` | Funding-Info pro Symbol |
| `add_position / remove_position` | Positionsliste (für RiskManager) |
| `symbol_position_usdt(symbol)` | Summe der USDT-Werte offener Positionen |
| `set_freeze / is_frozen` | DD-Freeze-Flag |
| `set_stop_mode / get_stop_mode` | In-memory Kopie von stop_mode (parallel zu DB) |

### data_fetcher.py

| Funktion | Soll |
|----------|------|
| `fetch_ohlcv(symbol, tf, limit)` | ccxt Kraken OHLCV → DataFrame mit open/high/low/close/volume |
| `fetch_ticker(symbol)` | Aktueller Preis via ccxt |
| `get_balance(currency)` | Live-Balance vom Exchange |
| `fetch_ohlcv_since(symbol, tf, since_iso, limit)` | OHLCV ab Datum (für Backtest-Data) |

### notifier.py

| Funktion | Soll |
|----------|------|
| `notify_trade_open/close` | Telegram-Nachricht (async Thread). Stille wenn kein Token. |
| `notify_error` | Fehler-Alert |
| `notify_startup` | Bot-Start-Meldung |

### backtest/

| Funktion | Soll |
|----------|------|
| `run_backtest(params, symbol, df)` | Simuliert Grid-Trading auf historischen Daten. Gibt dict mit pnls, equity_curve. |
| `load_ohlcv(symbol, tf, days)` | OHLCV aus Cache-DB oder Binance-Download |
| `sharpe / sortino / max_drawdown / calmar / hit_rate / profit_factor` | Standard-Performance-Metriken |
| `summary(pnls, equity_curve, days)` | Zusammenfassung aller Metriken als dict |

### scripts/optimize.py, sweep.py, nightly_tune.py, bot_monitor.py

| Script | Soll |
|--------|------|
| `optimize.py --analyze-trades` | Trade-Pattern-Analyse aus trades.db |
| `optimize.py --suggest-params` | Parameter-Empfehlungen basierend auf Trade-Historie |
| `sweep.py` | OOS-Parameter-Sweep (Calmar-optimiert), schreibt results/ |
| `nightly_tune.py` | **Read-only** (seit 2026-06-17): Analyse + GitHub Issue. KEIN Branch/Commit/PR. Garantie: "this script NEVER modifies any file or branch." |
| `bot_monitor.py` | Health-Checks: Grid-Rebuild-Storm, ML-Fallback-Rate (>30% rule-based → Issue), Stuck-Bot, DD-Breach |

---

## Konfiguration

### config/config.yaml

```yaml
symbols: [SOL/USD, ETH/USD, AVAX/USD, LINK/USD, XRP/USD]
initial_capital: 1000
risk:
  max_daily_drawdown: 0.10    # 10% → Bot-Freeze
  max_position_size: 0.10     # 10% Equity max pro Position
  max_open_positions: 3
```

### config/grid_params.json
Auto-generiert vom Sweep-Winner. Wird von `GridParams.from_dict()` geladen wenn vorhanden.

### .env / config.py
`BINANCE_API_KEY/SECRET` = Kraken-Keys (falsch benannt, Fallback in config.py).
`ANTHROPIC_API_KEY` = für Claude Haiku LLM-Analyst.
