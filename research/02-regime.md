# Phase 2 — Regime-Gating: Befunde (Dev-Set)

Bezug: `research/00-hypothesen.md` (H2), `research/01-kosten.md` Befund A (Ranging
verliert trotz hoher Win-Rate, Trending gewinnt).

## Wichtige Klarstellung vor den Zahlen: zwei verschiedene Mechanismen

Befund A basiert auf der **Regime-Klassifikation des PricePredictor** (ranging/
trending/volatile — bestimmt Grid-Range UND Level-Zahl: 14/6/20). Der in diesem
Abschnitt getestete `trend_filter_enabled`/`trend_adx_min`-Mechanismus ist etwas
**anderes**: ein separates EMA/ADX/DI-Signal, das NUR neue Buys während eines hart
erkannten Abwärtstrends pausiert (`_hard_trend_down` in `strategies/grid.py`). Beide
Mechanismen sind im Code unabhängig. **Das Testen von `trend_adx_min` prüft also nicht
direkt „Grid aus im Ranging-Regime"** — das wäre ein invasiverer Code-Eingriff (siehe
unten).

## Befund D: Trend-Filter-Schwelle hat auf SOL/USD praktisch keinen Effekt

`scripts/sweep.py --grid stage_b --leverage 1.0 --symbol SOL/USD --min-trades 5`
(`results/sweep_20260722_1504/report.md`):

| Config | OOS Calmar | OOS Return | OOS worst DD |
|--------|-----------|------------|--------------|
| BASELINE (trend_filter an, ADX 25) | −5,44 | −1,5 % | −1,6 % |
| stage_b/adx15_strict (strenger) | −5,44 | −1,5 % | −1,5 % |
| stage_b/adx25_baseline (Referenz) | −5,44 | −1,5 % | −1,6 % |
| stage_b/adx35_loose (lockerer) | −5,44 | −1,5 % | −1,6 % |
| stage_b/filter_off (Filter ganz aus) | −5,64 | −1,4 % | −1,4 % |

**Strenger, Standard und lockerer ADX-Schwellwert sind für SOL/USD praktisch
identisch** — der Filter-Threshold selbst bewegt fast nichts. Nur das komplette
Abschalten des Filters weicht minimal ab (leicht schlechterer Calmar, leicht besserer
Rohertrag — vermutlich weil kleinere Drawdowns den Calmar-Nenner verändern, nicht weil
absolut mehr verdient wird). **Erste Einschätzung: Dieser konkrete Hebel (ADX-
Schwellwert-Tuning) ist so gut wie wirkungslos** — ähnlich schwach wie H1b/H1c beim
Kosten-Hebel.

**ETH/USD bestätigt (`results/sweep_20260722_1508/report.md`):** BASELINE/adx25/adx35
praktisch identisch (−5,54, −1,3 %). adx15_strict leicht schlechter (−5,57, −1,6 %).
filter_off diesmal **schlechter, nicht besser** (−5,60, −1,7 %, Train-Return −2,7 %
statt −1,6 % — deutlich schlechter in-sample). **Kein konsistentes Signal in
irgendeine Richtung über die zwei bisherigen Symbole — sieht nach Rauschen aus, nicht
nach Effekt.**

**AVAX/USD bestätigt (`results/sweep_20260722_1511/report.md`):** Alle 5 Configs in
enger Bandbreite (−5,63 bis −5,66), diesmal ist adx15_strict beim Return leicht
vorn (−1,3 %), aber beim Calmar am schlechtesten — wieder kein konsistentes
Vorzeichen. **3/5 Symbole zeigen dasselbe Bild: Trend-Filter-Schwellenwert bewegt
nichts Systematisches.**

**LINK/USD bestätigt (`results/sweep_20260722_1514/report.md`):** Alle 5 Configs in
enger Bandbreite (−5,59 bis −5,65), kein konsistentes Signal. **4/5 Symbole
bestätigt.**

**XRP/USD bestätigt (`results/sweep_20260722_1517/report.md`):** Alle 5 Configs in
enger Bandbreite (−5,56 bis −5,57), diesmal ist sogar rechnerisch adx35_loose
„Gewinner" nach Ranking — aber nur um 0,01 Calmar vor der Baseline, praktisch
identisch. Kein konsistentes Signal.

## Phase-2-Fazit (Trend-Filter-Achse): toter Hebel, alle 5 Symbole bestätigt

| Symbol | Bandbreite OOS Calmar über 5 Configs | Bestes Config-Vorzeichen ggü. Baseline |
|--------|----------------------------------------|-------------------------------------------|
| SOL/USD | −5,44 bis −5,64 | uneinheitlich (Filter aus: leicht besser Return, schlechter Calmar) |
| ETH/USD | −5,54 bis −5,60 | Filter aus schlechter (nicht besser) |
| AVAX/USD | −5,63 bis −5,66 | uneinheitlich |
| LINK/USD | −5,59 bis −5,65 | uneinheitlich |
| XRP/USD | −5,56 bis −5,57 | uneinheitlich, Unterschiede <0,02 |

**Der `trend_filter_enabled`/`trend_adx_min`-Mechanismus hat über alle 5 Symbole
hinweg keinen systematischen, richtungsstabilen Effekt** — die Schwankungen liegen im
Rauschbereich (0,01–0,3 Calmar-Punkte), nicht in der Größenordnung, die nötig wäre, um
auch nur in die Nähe von Breakeven (Calmar 0) zu kommen (Abstand: ~5,5 Punkte).

**Einordnung:** Diese konkrete Operationalisierung von H2 ist damit **so gut wie
sicher tot** — genau wie der Kosten-Hebel in Phase 1.

## Stage C: Harter Regime-Gate (H2, harte Variante) — Code-Änderung umgesetzt

**Nutzer-Entscheidung 2026-07-22:** Härteren Mechanismus bauen statt Urteil zu
akzeptieren. Umgesetzt: `ranging_gate_enabled: bool = False` (neues Feld,
`strategies/grid_params.py`), additive Gate-Bedingung in
`strategies/grid.py::_buys_allowed()` (pausiert neue Buys komplett, wenn
`state._last_regime == "ranging"` — Positionsverwaltung/Sells laufen unverändert
weiter, exakt wie beim bestehenden `hard_trend_down`-Gate). Default aus, ändert nichts
am Live-Bot. 3 neue Regressionstests (`TestRangingGate`), 67/67 Tests grün. Neue
Sweep-Option `--grid stage_c` (2 Configs: Gate an/aus ggü. Baseline).

**⚠️ Methodischer Hinweis:** `calmar()` in `backtest/metrics.py` annualisiert den
Return (`(1+r)^(1/years)-1`, hier years≈0,164 bei 60 Tagen OOS — Exponent ≈6,1×), was
kleine Return-Unterschiede stark verstärkt. **Primär wird deshalb der rohe OOS-Return
verglichen, Calmar nur als Richtwert** (siehe ETH-Befund unten, wo Calmar sich stark
verbessert, obwohl der rohe Return leicht schlechter wird — ein Hinweis, dass Calmar
hier Rauschen verstärkt, nicht immer echten Fortschritt misst).

| Symbol | Baseline OOS Return | Gate-On OOS Return | Baseline OOS Calmar | Gate-On OOS Calmar |
|--------|----------------------|----------------------|------------------------|------------------------|
| SOL/USD | −1,5 % | **−1,2 %** (besser) | −5,44 | −5,44 |
| ETH/USD | −1,3 % | −1,4 % (leicht schlechter) | −5,54 | −4,95 |
| AVAX/USD | −1,6 % | **−1,3 %** (besser) | −5,63 | −5,66 |
| LINK/USD | −1,4 % | **−1,2 %** (besser) | −5,59 | −5,16 |
| XRP/USD | *läuft* | | | |

**SOL/USD (`results/sweep_20260722_1529/report.md`):** Verbesserung sowohl Train
(−1,7 % statt −1,8 %) als auch OOS (−1,2 % statt −1,5 %, DD −1,3 % statt −1,6 %) —
konsistent in dieselbe Richtung.

**ETH/USD (`results/sweep_20260722_1532/report.md`):** **Gemischt** — roher OOS-Return
minimal schlechter (−1,4 % statt −1,3 %) und OOS-DD schlechter (−1,6 % statt −1,4 %),
aber Calmar durch Annualisierungs-Verstärkung viel besser (−4,95 vs. −5,54). Nach dem
methodischen Hinweis oben: **auf Return-Basis für ETH eher neutral/leicht negativ,
nicht eindeutig positiv wie bei SOL.**

**AVAX/USD (`results/sweep_20260722_1536/report.md`):** Auf Return-Basis **besser**
(−1,3 % statt −1,6 %, Train −1,1 % statt −1,8 % — deutlich), Calmar minimal
schlechter (Annualisierungs-Rauschen wie oben erklärt).

**LINK/USD (`results/sweep_20260722_1538/report.md`):** Besser auf allen Achsen —
Return −1,2 % statt −1,4 %, Calmar −5,16 statt −5,59, Train-Return −1,7 % statt −1,8 %.

**XRP/USD (`results/sweep_20260722_1541/report.md`):** Besser auf Return/DD (−1,4 %
statt −1,5 %), Calmar minimal schlechter (Rauschen).

## Stage-C-Fazit: erstes konsistentes Signal im gesamten Programm — aber klein

| Symbol | Baseline Return | Gate-On Return | Delta |
|--------|-------------------|-------------------|-------|
| SOL/USD | −1,5 % | −1,2 % | **+0,30 pp** |
| ETH/USD | −1,3 % | −1,4 % | −0,10 pp |
| AVAX/USD | −1,6 % | −1,3 % | **+0,30 pp** |
| LINK/USD | −1,4 % | −1,2 % | **+0,20 pp** |
| XRP/USD | −1,5 % | −1,4 % | **+0,10 pp** |
| **Gepoolter Mittelwert** | **−1,46 %** | **−1,30 %** | **+0,16 pp, 4/5 Symbole positiv** |

**Das ist das erste Mal im gesamten Programm (Kosten-Hebel H1, Trend-Filter H2-weich),
dass ein Hebel in der Mehrheit der Coins konsistent in dieselbe Richtung zeigt** —
anders als das reine Rauschen in Stage B. Der Effekt ist real, aber **klein**: +0,16
Prozentpunkte gepoolt, gegenüber einem Abstand zu Breakeven von ~1,46 Punkten (Return)
bzw. ~5,5 Punkten (Calmar). **Das dreht das Vorzeichen nicht um — es verkleinert den
Verlust um etwa ein Neuntel.** ETH ist die einzige Ausnahme (leicht schlechter).

## Diskriminator-Test: Ist das ein echter Edge oder ein "Weniger-von-etwas-Schlechtem"-Artefakt?

**Kritischer Einwand (bevor irgendeine Schlussfolgerung gezogen wird):** Bei einer
strukturell negativen Expectancy (wie hier in JEDEM bisherigen Test belegt) senkt das
Wegfiltern *irgendeines* Trade-Segments automatisch den Gesamtverlust — das ist reine
Mechanik, kein Edge. Das Grenzverhalten dieser Logik ist „alles wegfiltern → 0 Trades
→ 0 Verlust → 0 Gewinn". Um das zu unterscheiden: **Profit-Factor (pf) pro Trade**
messen — bleibt er flach/schlecht, ist es ein Trade-Weniger-Artefakt; steigt er
Richtung/über 1,0, wäre es ein echter Qualitätsgewinn.

Aus den bereits vorhandenen `all_runs.csv` (OOS-Phase, kein neuer Sweep nötig):

| Symbol | Baseline: Trades / PF / Hit-Rate | Gate-On: Trades / PF / Hit-Rate |
|--------|-----------------------------------|-------------------------------------|
| SOL/USD | 19 / 0,164 / 42,1 % | 12 / 0,189 / 41,7 % |
| ETH/USD | 16 / 0,132 / 37,5 % | 9 / 0,112 / 33,3 % |
| AVAX/USD | 25 / 0,075 / 20,0 % | 6 / **0,0 / 0,0 %** |
| LINK/USD | 14 / 0,128 / 42,9 % | 8 / 0,123 / 37,5 % |
| XRP/USD | 7 / 0,0 / 0,0 % | 5 / 0,0 / 0,0 % |

**Eindeutiges Ergebnis: Trade-Zahl sinkt konsequent (wie erwartet bei einem Gate),
aber der Profit-Factor verbessert sich NICHT** — bleibt in jedem Fall weit unter 1,0
(Bruttoverlust > Bruttogewinn, in jedem Segment, mit oder ohne Gate). Bei AVAX
verschlechtert sich die Trade-Qualität mit Gate sogar auf 0 % Hit-Rate. **Das ist
exakt das erwartete Verhalten eines „Weniger-von-etwas-Schlechtem"-Artefakts, kein
Edge.** Die weniger schlechten Return-Zahlen kommen daher, dass insgesamt weniger
verlustreiche Trades stattfinden — nicht daher, dass die verbleibenden Trades
irgendwie besser wären.

**Zusätzlich gegen das eigene Kill-Kriterium geprüft:** Ein OOS-Fenster, ~5–19
Trades/Symbol, 5 über BTC korrelierte (nicht unabhängige) Assets — Kill-Kriterium #2
(Bootstrap-CI schließt Null aus) ist damit gar nicht berechenbar, Kriterium #3 (≥30
Trades/Fenster) wird verfehlt. Selbst wenn der Effekt real wäre, würde er die
vorab festgelegte Latte nicht erfüllen.

## Phase-2-Gesamtfazit

**H2 ist in beiden getesteten Formen tot:** die weiche Variante (Trend-Filter-
Schwelle) zeigt reines Rauschen, die harte Variante (Regime-Hard-Gate) zeigt einen
Trade-Weniger-Artefakt ohne echte Verbesserung der Trade-Qualität. Damit sind alle
drei im Plan vorgesehenen Hebel (Kosten, Regime — weich und hart) getestet und
negativ. Der Grund, warum der Bot trotz z.T. hoher Win-Rate verliert, ist jetzt klar
belegt: **In keinem Marktsegment (ranging/trending, egal welche ADX-Schwelle) ist der
Profit-Factor auch nur in die Nähe von 1,0 gekommen** — die Strategie hat schlicht in
keinem getesteten Segment einen positiven Erwartungswert nach Gebühren.

## Offene Frage: Braucht es einen echten Code-Eingriff für H2?

Ein direkterer Test von „Grid komplett aus im Ranging-Regime" (die eigentliche
Original-Hypothese) würde eine neue Gate-Bedingung in `strategies/grid.py` brauchen,
die auf der PricePredictor-Regime-Klassifikation basiert — das ist ein echter
Code-Eingriff in die Strategie (nicht nur eine neue Sweep-Config-Achse wie bisher).
Geprüft: `levels_by_regime.ranging = 0` ist **nicht sicher** (Division durch Null in
`grid.py`, da `min_step_pct > 0` in der Live-Baseline immer eine Division durch
`levels` ausführt) — bräuchte einen expliziten Guard. Entscheidung, ob das gebaut wird,
hängt vom Ausgang der 5-Symbol-Verifikation oben ab.
