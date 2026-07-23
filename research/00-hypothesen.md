# Phase 0 — Pre-Registration: Hypothesen, Kill-Kriterien, Dev/Vault-Split

Stand: 2026-07-22. Dieses Dokument wird **vor** jeder Hypothesenprüfung fixiert und
danach nicht mehr rückwirkend verändert (nur ergänzt, mit Datum). Ziel: verhindern,
dass wir uns durch wiederholtes Nachjustieren selbst betrügen (Data-Snooping).
Referenz: `/Users/lucasturz/.claude-personal/plans/du-bist-eine-person-floofy-snowflake.md`.

## 0. Ist-Stand bei Pre-Registration (Fakten, keine Interpretation)

- **Aktiver Prozess:** Der Bot läuft **live im Paper-Modus auf dem Raspberry Pi**
  (`pi@raspberrypi.local`, PID seit 2026-07-22 10:38), nicht auf dem Mac (dort seit
  15.07. gestoppt). Equity dort: ~478,74 von 500 Start. Leverage 3.0, `sl_mode:
  per_position`. Live-Trading (`--mode live`) bleibt hart gesperrt (`LIVE_PARITY_OK=False`).
- **Neuester ehrlicher Sweep** (`results/sweep_20260721_1903/report.md`, Header
  „ehrliche Metriken — ohne Phantom-PnL, ab Commit e5170a4", stage_a-Grid, 180d @1h,
  Train 120d/Test 60d, 5 Symbole, Leverage 3.0): **Top-10-Configs allesamt negativ** —
  OOS Calmar −4,92 bis −5,23, OOS-Return −4,4 % bis −4,8 %. **Train-Calmar ebenfalls
  negativ** (−2,77 bis −2,80) — deutet auf strukturellen Kostendrag statt reines
  Overfitting hin (bestätigt unabhängig das Urteil aus `PROGRESS.md`, 03.07.).
- **⚠️ Roter Faden „Aggressiv"-Modus (Commit `99de896`, 13.07., 10 Tage nach dem
  Negativ-Urteil):** Martingale-artiges DCA-Sizing (`dca_size_mult`, Level i × mult^depth)
  + Runner-Modus + Leverage-Cap 3× → 5×, explizit begründet mit „mehr Risiko für mehr
  Gewinn". **Aktuell per Default aus** (`dca_size_mult=1.0`, `runner_enabled=false` in
  Code-Defaults und im aktiven `config/grid_params.json`). Aber: Martingale-Sizing nach
  einem Negativ-Befund ist ein bekanntes Warnsignal (verwandelt leicht-negative
  Expectancy in selten-aber-katastrophales Ruin-Risiko). **Bleibt in diesem Programm
  aus, bis das Grundsystem ohne Martingale eine belegte positive Expectancy hat** —
  Martingale auf einen Non-Edge draufzusetzen wäre das Gegenteil von „profitabel machen".
- **`sweep.py` pinnt `LEVERAGE=3.0`** als Modul-Konstante (Zeile 32); kein `--leverage`-
  CLI-Flag. Für unsere Lev=1×-Pflicht (siehe Plan) muss das vor Phase 1 ergänzt werden
  (kleiner, additiver Patch — kein Eingriff in Risk-Logik).
- **Instrument-Verifikation (Plan Phase 0.1):** `scripts/sweep.py` mit `--grid stage_a`
  produziert bereits explizit als „ehrliche Metriken" gelabelte Reports und lief zuletzt
  (21.07.) durch, ohne die in `PROGRESS.md` beschriebenen Orphan-Worker-/Refetch-Probleme
  erkennbar zu reproduzieren (Laufzeiten sequentiell ~1–1,5 h für 5 Symbole, plausibel).
  → **Vorläufiges Urteil: `scripts/sweep.py --grid stage_a` ist das zu nutzende
  Instrument**, mit `--min-trades` als eingebautem Trade-Zahl-Filter. Offen: Bestätigen,
  dass diese Zahlen sich mit einem unabhängigen Kontrolllauf decken, bevor darauf eine
  Kill-Entscheidung gebaut wird (siehe §3).

## 1. Dev/Vault-Split (einmalig, vor jeder Exploration)

**⚠️ Korrektur 2026-07-22 (gleicher Tag wie Pre-Registration, vor jeder Kill-Entscheidung
entdeckt):** Die ursprüngliche Vault-Definition unten (07-15 bis Phase-4-Gate) wurde durch
die ersten Ausführungsschritte selbst kontaminiert — `scripts/sweep.py` und
`scripts/optimize.py --analyze-trades` kappen Daten nie nach oben, sondern reichen immer
bis „jetzt". Der erste Leverage-1×-Sweep (SOL/USD, `--days 180`) hatte ein OOS-Testfenster
2026-05-23→07-22, die Fee-Drag-Analyse (`--days 30`) lief über 06-22→07-22 — beides deckt
den vordefinierten Vault-Zeitraum bereits ab. Root Cause: reines Tooling-Problem (kein
`--as-of`-Cap), kein Interpretationsfehler. Fix: `--as-of`-Flag zu `scripts/sweep.py`
ergänzt (kappt `df.index <= as_of` vor dem Split), 64/64 Tests weiterhin grün.
**Konsequenz:** Vault wird unten neu definiert als **rein zukünftiges Fenster ab dem
Pre-Registration-Stichtag**, statt als historisches Fenster — dadurch ist nichts
tatsächlich verbrannt (es existierte schlicht noch keine Zukunfts-Vault-Zeit, die man
hätte ansehen können), aber die ursprüngliche Definition war in sich widersprüchlich
(„zukünftig unberührt" + „von Natur aus im OOS-Fenster jedes Laufs enthalten").

Verfügbare Historie laut Sweep-Setup: 180 Tage @ 1h, Train 120 / Test 60 (rollend).
Um Data-Snooping über Phase 1–3 zu verhindern:

- **Dev-Set:** Alle Kursdaten und Trade-Historie **bis einschließlich 2026-07-22**
  (heutiger Pre-Registration-/Korrektur-Stichtag). Hier finden alle Hypothesentests aus
  Phase 1–3 statt — inklusive der bereits gelaufenen Befunde in `research/01-kosten.md`.
- **Vault-Fenster:** Alle Kursdaten und echten Trades **ab 2026-07-23** (dem Tag nach
  diesem Stichtag) **bis zum Tag des Phase-4-Gates**. Wird von jetzt an durch die bereits
  laufende Paper-Session auf dem Pi genuin unberührt gesammelt — deckt sich mit dem in
  `PROGRESS.md` vorgeschlagenen nächsten Schritt „1–2 Wochen ehrliche Paper-Daten
  sammeln". Jeder künftige Dev-Set-Sweep (Phase 1–3-Fortsetzung) **muss** `--as-of
  2026-07-22` (oder das jeweils gültige Dev-Grenzdatum) setzen, sonst kontaminiert er
  erneut. Vor Phase 4 nicht anschauen, nicht backtesten, nicht zur Parameterwahl nutzen.
- Zusätzlich: der historische **06/2026-Crash** (bereits Teil der `PROGRESS.md`-Analyse,
  liegt vor dem heutigen Dev/Vault-Schnitt) zählt als unabhängiges Stress-Fenster in
  Phase 4, nicht zur Hypothesenwahl in Phase 1–3.

## 2. Hypothesenliste (vorab fixiert, nicht erweiterbar ohne neues Pre-Reg-Datum)

Bewusst kurz gehalten — mehr Hypothesen erhöhen die Falsch-Positiv-Rate:

| # | Hypothese | Hebel | Phase | Status (22.07.) |
|---|-----------|-------|-------|-------------------|
| H1a | Strikter Maker/Post-Only-Zwang senkt Fee-Drag messbar | Kosten | 1 | ✅ bereits erfüllt (PostOnly Default), kein zusätzlicher Effekt möglich |
| H1b | Weiteres Grid-Spacing (weniger Flips) verbessert Netto-Expectancy | Kosten | 1 | ❌ getestet (21.07.-Sweep), kein messbarer Effekt |
| H1c | Mindest-Profit pro Flip > 2×Fee+Slippage filtert Level, die strukturell nicht tragen | Kosten | 1 | ❌ getestet, Effekt +0,29 Calmar — Richtung bestätigt, Größenordnung zu klein |
| H2 | Grid nur in robust erkanntem Regime statt Dauerbetrieb | Regime | 2 | ❌ Beide Varianten getestet (Trend-Filter-Schwelle: Rauschen; Regime-Hard-Gate: Trade-Weniger-Artefakt, Profit-Factor bleibt <1,0 mit und ohne Gate, kein echter Qualitätsgewinn) |
| H3 | Nur Mean-Reversion-Paare (Hurst < 0,5) gridden, Trender raus | Instrumente | 3 | ungetestet |

**Nicht Teil dieses Programms** (bewusst ausgeschlossen, um die Hypothesenzahl klein zu
halten): Geometrie-Feintuning (bereits gesweept, alle negativ), ML/Directional
(F1 schwach, 12 % WR in-sample, bleibt aus), Aggressiv-Modus/Martingale (s. §0).

## 3. Kill-Kriterien (Latte, aus dem Plan übernommen — hier verbindlich verankert)

Für die Endkonfiguration in Phase 4 gilt, **alle** vier Bedingungen gleichzeitig:

1. **Leverage = 1× (Spot-Ökonomie)** — `--leverage 1.0`-Flag in `scripts/sweep.py`
   ergänzt und getestet (2026-07-22, 64/64 Tests grün).
2. **Bootstrap-Konfidenzintervall der Expectancy schließt 0 aus** (nicht nur positiver
   Punktschätzer).
3. **≥ 30 echte Trades pro Fenster, gepoolt über den 5-Symbol-Basket** (nicht pro
   Einzelsymbol). **Korrektur 2026-07-22**, empirisch erzwungen: Der erste Lev=1×-Sweep
   (SOL/USD allein, 120d Train) erzeugte nur 10–31 Trades (Median 16) über alle 39
   Configs — ein Einzelsymbol erreicht bei echter Spot-Ökonomie in einem 60-Tage-
   OOS-Fenster strukturell nie 30 Trades. Konsequenz: Expectancy/Bootstrap-CI wird auf
   dem gepoolten Trade-Set aller 5 Symbole je Fenster berechnet, nicht separat pro Coin.
4. **Positive Expectancy in der Mehrheit der qualifizierenden Fenster** (Vault +
   Bull/Bär/Crash/Seitwärts), nicht nur in einem herausgepickten.

**Nicht erfüllt → ehrlicher Stopp, kein Live-Antrag, Vault-Fenster gilt als verbraucht.**
Calmar > 1,0 bleibt Richtwert, kein hartes Gate (zu verrauscht auf kurzen Fenstern).

## 4. Nächste konkrete Schritte (Phase 0 → 1 Übergang)

1. ✅ `scripts/sweep.py`: additiver `--leverage`-Flag (Default 3.0 unverändert für
   Nightly-Automatik) — erledigt 2026-07-22.
2. ✅ `scripts/sweep.py`: additiver `--as-of`-Flag gegen die in §1 dokumentierte
   Vault-Kontamination — erledigt 2026-07-22, 64/64 Tests grün. **Ab jetzt Pflicht bei
   jedem weiteren Dev-Set-Sweep:** `--as-of 2026-07-22` (oder das dann gültige
   Dev-Grenzdatum) setzen.
3. ✅ Fee-Drag auf dem Dev-Set quantifiziert (`research/01-kosten.md`): Ranging-Regime
   verliert trotz 72 % Win-Rate (Fee-Drag-Signatur), ML-Brier-Score 0,39 (schlechter als
   Zufall) → H2 geschärft, ML/Directional-Ausschluss bestätigt.
4. Läuft: Leverage-1×-Sweep SOL/USD (`--grid stage_a`, ohne `--as-of`, also nach der
   Vault-Neudefinition in §1 unproblematisch als Dev-Set-Lauf). Ergebnis wird in
   `research/01-kosten.md` ergänzt.
5. Offen: gleicher Sweep für die übrigen 4 Symbole (ETH/AVAX/LINK/XRP) bei Leverage 1×,
   dann H1a–H1c einzeln gegen diese Baseline testen (H1a strukturell bereits erfüllt —
   PostOnly ist in `execution/kraken.py`/`execution/paper.py` bereits Default).
6. Ergebnisse pro Hypothese weiter in `research/01-kosten.md` (Phase 1),
   `research/02-regime.md` (Phase 2), `research/03-instrumente.md` (Phase 3)
   dokumentieren — Format: Hypothese, Delta zur Baseline, Trade-Zahl, Entscheidung
   (behalten/verwerfen).
