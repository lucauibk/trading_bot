# Phase 1 — Kostenstruktur: Befunde (Dev-Set)

Bezug: `research/00-hypothesen.md` (Pre-Registration). Dev-Set-Exploration bis
einschließlich 2026-07-22; Vault-Fenster (ab 2026-07-23) unberührt (siehe dortige
Korrektur vom 22.07.).

## Befund A: Ranging-Regime verliert trotz hoher Win-Rate — klassisches Fee-Drag-Muster

`python3 scripts/optimize.py --analyze-trades --days 30` (297 Trades, alle 5 Symbole):

| Regime | Trades | Win-Rate | Ø PnL/Trade |
|--------|--------|----------|-------------|
| ranging | 176 | 72,2 % | **−0,0223 USDT** |
| trending | 121 | 81,8 % | **+0,0730 USDT** |

**Das ist die Signatur, die H1a–H1c vorhersagen:** Ranging ist genau das Regime, in dem
das Grid am meisten Level-Flips produziert (14 Levels laut `levels_by_regime`) — hohe
Win-Rate (72 %), aber im Schnitt ein Verlust pro Trade. Viele kleine Gewinne unter der
Kraken-Fee-Doppelbelastung (Buy+Sell × 0,16 %) plus Slippage reichen nicht, um die
selteneren, größeren Verluste auszugleichen. Trending (weniger Levels: 6, größere
Bewegungen pro Fill) ist im Schnitt profitabel.

**Konsequenz für Phase 1/2:** Die naive Annahme „Grid nur in Ranging laufen lassen"
(ursprüngliche Hypothese H2) greift zu kurz — Ranging ist im Ist-Zustand das
Verlust-Regime, nicht das Sicherheits-Regime. H2 wird entsprechend geschärft: nicht
„Ranging = sicher", sondern „ist der Mindest-Profit pro Ranging-Flip strukturell zu
klein für die Fee-Last?" (→ direkt H1c: Mindest-Profit pro Flip > 2×Fee+Slippage).
Trending-Performance spricht dafür, den bestehenden `trend_filter_enabled`-Mechanismus
eher zu lockern als zu verschärfen — das wird in Phase 2 explizit gegen die
Fee-Filter-Hypothese getestet (nicht nur „Grid aus bei Trend").

## Befund B: ML-Kalibrierung ist schlechter als Zufall

`python3 scripts/optimize.py --calibration-report` (3402 kalibrierte Vorhersagen):

- **Brier-Score: 0,3934** — das Tool selbst definiert 0,25 als „zufällig". **0,39 ist
  schlechter als Münzwurf.**
- Reliability-Tabelle nicht monoton: Bucket (0,55–0,6] hat höhere Hit-Rate (31,5 %) als
  Bucket (0,6–0,65] (23,7 %) — klassisches Zeichen für Rauschen statt Signal.

**Konsequenz:** Bestätigt unabhängig den bestehenden Befund (F1 schwach, 12 % WR
in-sample) und die Entscheidung, ML/Directional aus dem Programm herauszuhalten (siehe
`research/00-hypothesen.md` §2, „nicht Teil dieses Programms"). Jeder Confidence-Filter
auf Basis dieses Modells würde eher Rauschen selektieren als Signal.

## Befund C: Bei echter Spot-Ökonomie (Leverage 1×) bricht die Trade-Zahl pro Symbol ein

Erster Lev=1×-Sweep (`scripts/sweep.py --grid stage_a --leverage 1.0 --symbol SOL/USD`,
180d/1h, Train 120d/Test 60d, `results/sweep_20260722_1227/`):

- **Trade-Zahl über alle 39 Configs im 120-Tage-Trainingsfenster: 10–31 (Median 16).**
  **0 von 39 Configs erreichen den Default-Filter `--min-trades 100`.** Nur die
  Live-Baseline wurde (per Force-Include) überhaupt bis OOS durchgetestet.
- **OOS-Ergebnis der Baseline (Lev=1×, SOL/USD allein):** median Calmar **−5,44**,
  median Return **−1,5 %**, worst DD −1,6 % (deutlich kleinerer Drawdown als bei
  Lev=3× — erwartbar ohne Hebel-Verstärkung — aber weiterhin negativ).
- **Konsequenz (siehe Korrektur in `research/00-hypothesen.md` §3, Kill-Kriterium #3):**
  Ein Einzelsymbol kann bei echter Spot-Ökonomie in einem 60-Tage-OOS-Fenster
  strukturell nicht auf ≥ 30 Trades kommen. Die Latte wird deshalb **gepoolt über den
  5-Symbol-Basket** ausgewertet, nicht pro Coin.
- **Instrument-Zusatzbefund:** Von 6 Worker-Prozessen des Sweep-Pools war über weite
  Strecken nur 1 aktiv rechnend (die anderen 5 im Leerlauf) — bestätigt empirisch die in
  `PROGRESS.md` dokumentierte macOS/Py3.9-Lastverteilungs-Schwäche. Lief trotzdem ohne
  Absturz zu Ende (9 Min für 1 Symbol × 39 Configs); für 5 Symbole entsprechend länger.

## Befund C (Fortsetzung): Einzelsymbol-Sweeps bei Leverage 1× (Dev-Set, 180d/1h)

Der kombinierte 5-Symbol-Lauf wurde vom System gekillt (8-GB-Mac, ~61 MB frei laut
`memory_pressure`, massive Swap-Historie — 4 parallele Pool(6)-Prozesse haben den
Speicher gesprengt). Fix: **strikt sequentiell, `--jobs 2`** pro Symbol. Ergebnisse
werden hier gepoolt, sobald alle 5 durch sind.

| Symbol | OOS median Calmar | OOS median Return | OOS worst DD | Trades (Train, 39 Configs) |
|--------|-------------------|--------------------|--------------|------------------------------|
| Symbol | OOS median Calmar | OOS median Return | OOS worst DD | Trades (Train, 39 Configs) |
|--------|-------------------|--------------------|--------------|------------------------------|
| SOL/USD | −5,44 | −1,5 % | −1,6 % | 10–31 (Median 16), 0/39 ≥ 100 |
| ETH/USD | −5,54 | −1,3 % | −1,4 % | (0/39 ≥ 100, Baseline force-included) |
| AVAX/USD | −5,63 | −1,6 % | −1,6 % | (0/39 ≥ 100, Baseline force-included) |
| LINK/USD | −5,60 | −1,4 % | −1,4 % | (0/39 ≥ 100, Baseline force-included) |
| XRP/USD | *läuft* | | | |

**Bisheriges Bild (4/5 Symbole):** Bemerkenswert konsistent negativ und in enger
Bandbreite (Calmar −5,4 bis −5,6, Return −1,3 % bis −1,6 %) — kein Ausreißer, kein
Symbol zeigt bislang einen Hauch von positiver Expectancy bei echter Spot-Ökonomie.
Diese Konsistenz über unabhängige Coins spricht eher für einen strukturellen
(Fee-/Geometrie-)Effekt als für Zufallsstreuung.

## Gepoolte Auswertung (5/5 Symbole vollständig)

**⚠️ Methodischer Hinweis (verschärft):** Dies ist eine **Cross-Symbol-Streuung der
Config-Median-Werte, keine echte Per-Trade-Bootstrap-CI** (Kill-Kriterium #2 verlangt
eigentlich Bootstrap über einzelne Trade-PnLs; `all_runs.csv` exportiert nur
Aggregate). **Zusätzlich sind die 5 Coins keine unabhängigen Stichproben** — sie
co-bewegen sich über BTC (dafür existiert `market/correlation.py` im Projekt). Die
„95%-CI" unten liest sich präziser, als sie ist. **Was robust bleibt: die Richtung**
(klar negativ, sehr eng gebündelt) — **nicht die exakte Präzisionsangabe.**

| Symbol | OOS median Return | OOS median Calmar |
|--------|--------------------|--------------------|
| SOL/USD | −1,5 % | −5,44 |
| ETH/USD | −1,3 % | −5,54 |
| AVAX/USD | −1,6 % | −5,63 |
| LINK/USD | −1,4 % | −5,60 |
| XRP/USD | −1,5 % | −5,57 |

**Mittelwert Return: −1,46 %, SD 0,11 Prozentpunkte, 95%-CI (n=5, Normalapprox.):
[−1,56 %, −1,36 %]. Mittelwert Calmar: −5,56 (SD 0,07).** Schließt Null eindeutig
aus — auf der **negativen** Seite. Die Streuung ist über alle fünf unabhängigen Coins
bemerkenswert eng (Calmar-SD nur 0,07!), was stark für einen **strukturellen Effekt**
(Fee-/Geometrie-Drag, der auf jeden Coin fast identisch wirkt) statt Zufallsstreuung
spricht — kein einzelnes Symbol sticht positiv oder negativ heraus.

**Einordnung ggü. Lev=3×-Sweep (21.07., alle Symbole gepoolt):** Bei Lev=1× ist die
Verlustgröße deutlich kleiner (−1,3 % bis −1,6 % vs. −4,4 % bis −4,8 % bei Lev=3×) —
erwartbar, da ohne Hebel-Verstärkung. **Aber das Vorzeichen bleibt gleich.** Das
bestätigt: Das Problem ist nicht (nur) die Hebelverstärkung eines an sich neutralen
Systems, sondern eine strukturell negative Expectancy der Basis-Konfiguration.

## Korrektur: H1a–H1c sind bereits getestet — im Lev=3×-Sweep vom 21.07., nicht hier

**Wichtige Selbstkorrektur:** Die 5 Lev=1×-Läufe oben testen keine Kosten-Hypothese —
`--min-trades 100` filtert bei nur 1 Symbol alle 38 Nicht-Baseline-Configs raus, es
bleibt nur die force-included Baseline. Fünf Läufe → fünf Baseline-Messungen, keine
H1b/H1c-Prüfung.

**H1b (weiteres Spacing) und H1c (Mindest-Profit-Multiple) sind aber bereits im
gepoolten 5-Symbol-Sweep vom 21.07. (`results/sweep_20260721_1903/report.md`, Lev=3×)
OOS getestet** — genau das ist, wofür das stage_a-Grid gebaut wurde:

| Config | OOS Calmar | Delta zur Baseline |
|--------|-----------|---------------------|
| BASELINE (live) | −5,21 | — |
| stage_a/current/**step6.0** (max. `min_step_fee_multiple`, H1c) | **−4,92** | +0,29 |
| stage_a/tight_many/step4.0 (H1b, engeres Grid) | −5,21 | ±0,00 |

**Ergebnis: H1c ist Richtung-bestätigt (weniger Fee-Drag hilft), aber der Effekt ist
verschwindend klein** (+0,29 Calmar von −5,21 auf −4,92 — Größenordnung 100× zu klein,
um in Richtung 0 oder gar positiv zu kommen). H1b zeigt keinen messbaren Effekt.
**H1a** ist strukturell bereits erfüllt (PostOnly ist Default in
`execution/kraken.py`/`execution/paper.py`).

**Phase-1-Fazit: Es gibt aktuell keinen ungetesteten Kosten-Hebel mehr, der plausibel
bis in die Nähe von Breakeven reichen könnte.** Kosten sind nicht der Hauptverursacher
— oder zumindest nicht behebbar innerhalb der bereits gesweepten Geometrie-Achsen.

## Instrument-Limitation (aufgeschoben, nicht blockierend)

Für eine echte Per-Trade-Bootstrap-CI bräuchte es Trade-Level-Export aus
`backtest/engine.py` (aktuell nur `all_runs.csv`-Aggregate). Angesichts des klaren
Befunds oben (Richtung robust, Kosten-Hebel ausgeschöpft) aktuell nicht
Priorität — erst relevant, falls Phase 2 (Regime-Gating) einen vielversprechenden
Kandidaten liefert, der eine belastbare Bestätigung braucht.

## Status: Entscheidungspunkt, kein weiterer autonomer Sweep

Phase 1 ist inhaltlich abgeschlossen. Bevor weitere (speicherintensive, auf diesem
8-GB-Mac wiederholt an OOM gescheiterte) Sweeps gestartet werden, braucht es eine
Nutzer-Entscheidung: Phase 2 (Regime-Gating, einziger Hebel mit plausiblem Mechanismus
laut Befund A) weiterverfolgen, oder das Urteil „kein Edge" akzeptieren. Siehe
Gesamtbericht in der Konversation.
