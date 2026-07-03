# PROGRESS — Bot-Korrektheits-Loop

## LOOP COMPLETE (2026-07-03)

**Schluss-Urteil: Verdient der Bot real Geld? — NEIN.**

- 37 Tage Paper (2026-05-26→07-02): ausgewiesen +847.57, davon +1054.55 Phantom-PnL
  aus pre-seeded Sells → **reale Ökonomie −206.97 (−5.59/Tag)**.
- Seit Kapital-Reset 06-17 (300 USDT): +21.89 gebucht, **Equity real −30.79**
  (298.73→267.93; Differenz = Fill-Drop-Cash-Leck, gefixt in 4817034).
- Ehrlicher OOS-Vergleich nach Phantom-Fix (5m, 02.06.–02.07. inkl. Crash, 8 Configs
  × 5 Symbole à 200 USDT, Lev 3): **alle 8 Configs negativ** — per_position −34.5 (m2)
  bis −56.1 (m6), floor −81.0 bis −132.3 (mit 2–4 Emergency-Halts). Nur 51–69 echte
  Trades/Monat; deckt sich mit Live (~64/Monat) → Backtest ist kalibriert.
- Directional: 5 Strategie-Familien unabhängig negativ getestet (ML 12% WR in-sample,
  Funding IC≈0, MTF N≈2/Jahr, Breakout-Trailing PF 0.46–0.74, Mean-Reversion PF
  0.27–0.82 IS+OOS). Bleibt AUS.
- Konsequenz: Buchhaltung ist jetzt ehrlich; die Strategie selbst hat keine positive
  Expectancy. **Live-Modus ist hart gesperrt** (366297d) — zu Recht. Nächster sinnvoller
  Schritt: Bot mit gefixtem Code neu starten (vorher Graceful-Stop `wait_fills`!) und
  1–2 Wochen ehrliche Paper-Daten sammeln, bevor irgendeine Optimierung diskutiert wird.

Suite: **82/82 grün** (Baseline 79 + 3 Regressionstests). Working Tree clean auf `dev`.

---

## Checkliste

- [x] **Working Tree committen; `git checkout .` reaktiviert NICHT Directional** — `3deaa13`.
  Beleg: `git show HEAD:config/grid_params.json` → `directional_enabled: false`; Tree clean.
- [x] **P0 Phantom-PnL** — `e5170a4`. Beleg: Repro PaperBroker +0.81 Cash ohne Kauf (jetzt 0.00);
  trades.db-Gap-Verteilung: nur 359/1896 grid_fills echte 1-Level-Zyklen (Step immer 1.28%),
  Phantom +1054.55 von +1313.68. Fix: Seed-Fill = Non-Event (kein PnL/Cash/Trade-Log);
  ab e5170a4 sind alle geloggten grid_fills real → Trennung real/phantom via Commit-Zeitpunkt.
  Regressionstest `test_pre_seeded_fill_books_no_pnl_and_no_cash`.
- [x] **P0 Cash-Leck/Fill-Drop** — `4817034`. Beleg: Repro (Rebuild → Fill für entfernten cid →
  −16.75 Balance, 0 Positionen); DB: +21.89 gebucht vs. Equity −30.79 seit 06-17. Fix:
  Orphan-BUY wird als Position adoptiert (Sell+SL), Orphan-SELL aus meta rekonstruiert —
  korrekt für Paper UND Live. Regressionstest `test_orphan_buy_fill_is_adopted_not_dropped`.
  Hinweis: Equity-Rekonziliation < Toleranz ist erst nach Bot-Neustart mit neuem Code messbar
  (laufender Prozess seit 01.07. fährt alten Code).
- [x] **P1 Emergency-Stop** — `a0cf1f6`. Beleg: `continue` lag vor `on_tick_safety` (Code).
  Fix: SL-Überwachung läuft im Halt weiter; Halt persistiert via `coin_settings.enabled=0`
  (neuer Helper `set_coin_enabled`); main.py schließt disabled Symbole jetzt wirklich aus
  (Toggle war wirkungslos); SL cancelt Broker-Order sofort (Doppel-Kredit im Freeze).
- [x] **P1 RiskManager remove_position** — `83cc418`. Beleg: Repro 3 Buys + 1 Sell → 0 statt 2.
  Fix: entfernt genau eine Position (Match entry_price/qty). Regressionstest
  `test_remove_position_removes_only_one`.
- [x] **P1 Paper==Live-Parität → Live hart gegated** — `366297d`. Beleg: Spot-only
  (kraken.py defaultType=spot), in-memory Order-Map, keine echten SL-Market-Orders,
  pre-seeded Sells auf Spot unmöglich. Fix: `LIVE_PARITY_OK=False` + Blocker-Liste in
  execution/kraken.py; main.py verweigert `--mode live` fail-fast (validiert: Exit 2);
  Dashboard gibt klare Fehlermeldung. Freischalten nur per bewusster Code-Änderung.
- [x] **P2 Grid-Geometrie-Sweep auf 5m nach Phantom-Fix** — Infrastruktur `4d5d5b3` + `f2e9666`
  (PaperBroker-O(n²)-Pruning; nötig, weil Rebuild alle 3 Candles das Orderbuch sprengte).
  Durchführung: sequentieller Vergleich (multiprocessing-Sweep auf macOS/Py3.9 fragil —
  Orphan-Spawn-Worker, Refetch-Stürme; Skript `seq_compare.py` im Session-Scratchpad,
  Ergebnis-Tabelle oben im Urteil). floor-vs-per_position hat die OOS-Phase erreicht:
  per_position dominiert floor auf dem Crash-Fenster, aber ALLE Configs negativ →
  bewusst KEINE Config-Änderung (wäre Optimierung von "weniger Verlust").
- [x] **Ehrliches Schluss-Urteil** — dieses Dokument, Abschnitt oben.

## Bekannte offene Punkte (außerhalb des Loop-Scopes, dokumentiert)

1. **Positions-Persistenz über Restart fehlt**: `state.orders` wird nicht persistiert; Hard-Restart
   verwaist die Margin offener Positionen. Workaround: Graceful-Stop `wait_fills` vor Neustart.
2. **Laufender Bot fährt alten Code** (Start 01.07. 15:45, alte floor-Config) und produziert
   weiter Phantom-PnL in der DB, bis er neu gestartet wird.
3. **backtest/data.py Frische-Check** (`cache_max >= now − 3 Candles`) löst bei 5m-Daten
   Full-Refetch-Stürme aus (jeder Worker holt 90d neu, sobald der Cache >15 min alt ist).
4. **Live-Parität** (Margin-Trading, persistente Order-Map, echte SL-Orders) ist ein eigenes
   Projekt; bis dahin greift das Gate.
5. **Dashboard-Gesamt-PnL** summiert historische Phantom-Trades (vor e5170a4) weiter mit;
   für ehrliche Anzeige nach Neustart `/api/stats/reset` nutzen.
