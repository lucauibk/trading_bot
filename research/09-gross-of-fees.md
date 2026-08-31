# Phase 9 — Brutto-Test (Gebühr = 0): Sind Fees der Engpass? (2026-08-31)

Bezug: User-Frage „testen wir andere Zutaten". Der Venue-Wechsel (Low-Fee-Venue wie
Hyperliquid, ~0,02–0,045 %) ist nur dann ein echter Hebel, wenn eine Strategie
**brutto** Geld verdient und nur an Gebühren stirbt. Test: die schon gebauten
Backtests mit `FEE = 0` laufen lassen.

## Grid (`scratchpad/grid_gross.py`, Live-Config, Lev 1×, 5 Symbole, 400d, Geometrie passt sich mit)

| Fee/Seite | Median Return | Median PF | Median Trades |
|-----------|---------------|-----------|---------------|
| 0,16 % | −1,2 % | 0,39 | 10 |
| 0,08 % | −1,0 % | 0,20 | 8 |
| 0,04 % | −0,8 % | 0,39 | 12 |
| **0,00 %** | **−0,9 %** | **0,42** | 10 |

Bei **null Gebühren**: Median-Return −0,9 %, **PF 0,42** — Bruttoverluste ~2,4× Bruttogewinne.
Bewegt sich kaum, wenn die Fee von 0,16 % auf 0 fällt. Der Grid ist **brutto negativ**;
das ist die Payoff-Asymmetrie aus `research/05` (short gamma), nicht die Fee.
(Kaveat: nur 8–12 Trades/Symbol bei Lev 1× — kleine Stichprobe, aber Richtung
eindeutig und konsistent mit allem anderen.)

## Trend-Breakout (`scripts/trend_breakout_backtest.py --fee 0`)

- **Long**, alle 6 Configs: In-Sample-PF **0,64–0,85 (alle < 1)**. OOS-PF ~1,0,
  Block-Bootstrap-CI schließt 0 **nicht** aus (Primary point +0,0006).
- **Short**, Primary: IS-PF **1,00** (exakt break-even), OOS-PF 0,73.

Auch ohne jede Gebühr verliert/break-eventet das In-Sample. Kein Bruttoertrag da.

## Cross-Sectional Momentum (`scripts/cross_sectional_backtest.py --fee 0`)

Fee-Drag war ohnehin klein (~0–3 pp/Jahr). Ohne Gebühr: L=8W Dev-CAGR +52 % → +58 %,
**M5 weiterhin FAIL (1/3)**. Die PIT-korrigierte Version (`research/08`) kehrt sich
OOS trotzdem um (−45 %). → Engpass ist Survivorship + Nicht-Stationarität, nicht Fees.

## Fazit

**Fees sind bei keiner der drei Strategie-Klassen der Engpass.** Alle sind auch
brutto ≤ 0 bzw. edge-los. Damit ist der Zutaten-Hebel **„anderer Venue / niedrigere
Gebühren" tot** — er hilft nur einer brutto-profitablen Strategie, und so eine
wurde in 9 Phasen nicht gefunden.

## Verbleibende Zutaten-Hebel (keiner davon „mehr Backtests")

| Hebel | Was es bräuchte | Realistischer Payoff |
|-------|-----------------|----------------------|
| Mehr Kapital | ~10–50 k € → Risikoprämie (Funding-Carry Low-Fee-Venue, Basis, Vault-Deposit) | ~4–15 %/Jahr mit echtem Tail-Risiko |
| Bezahlte prädiktive Daten | 30–800 €/Mon Orderflow/Cross-Venue + Frequenz + Infra | klein, decay-anfällig |
| Höhere Frequenz | VPS nahe Börse, Tick-Daten, Execution-Stack | Solo-Retail post-2015 nicht viabel |

Für **500 € auf öffentlichen Daten** gibt es keine Zutaten-Kombination mit
positiver Erwartung. Das ist jetzt empirisch (Brutto-Test), nicht nur strukturell
argumentiert.
