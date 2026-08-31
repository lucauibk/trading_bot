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

---

## Nachtrag 2026-08-31 — die zwei noch offenen Zweige zu Ende getestet

### Zweig „mehr Kapital": Funding-Carry mit Budget (`scripts/funding_carry_scale.py`)

Delta-neutral long Spot / short Perp, **900 Tage** (2,5 Jahre), Majors + liquide
Alts, Low-Fee-Venue-Annahme (Round-Trip 0,14 %). Always-on-Hedge.

| Sym | netto Funding 900d | annualisiert | neg. 8h-Perioden | maxDD (Funding-Leg) |
|-----|--------------------|--------------|-------------------|----------------------|
| BTC | +14,3 % | **+5,6 %** | 16 % | −0,4 % |
| ETH | +14,4 % | +5,6 % | 17 % | −0,6 % |
| LINK | +16,4 % | +6,3 % | 18 % | −0,2 % |
| DOGE | +14,5 % | +5,6 % | 24 % | −0,5 % |
| XRP | +11,6 % | +4,6 % | 29 % | −1,2 % |
| SOL | +7,1 % | +2,8 % | 34 % | −2,6 % |
| AVAX | +5,2 % | +2,1 % | 35 % | −2,4 % |
| BNB | −0,6 % | −0,2 % | 13 % | −3,7 % |
| **Portfolio (gleichgewichtet)** | | **+4,0 %/Jahr** | | |

**Anders als der 180-Tage-Test (`research/04`: −0,54 %) ist die Carry über 2,5 Jahre
universumsweit positiv** (7/8 Coins) — Fenster fing eine funding-positive Phase.
**Aber:** ~4 %/Jahr Portfolio-Carry ist ungefähr Stablecoin-Lending-Niveau, mit
deutlich mehr operativer Komplexität. Und die niedrige „maxDD" hier misst NUR die
Funding-Zahlungen — die echten Risiken (Basis-Blowout / Short-Leg-Liquidation im
Bull, Venue-Solvenz, Rebalance-Slippage, volatiles Collateral mit Margin-Management)
stehen nicht in dieser Zahl. Regime-abhängig: über einen vollen Zyklus inkl. 2022er
Bär eher 2–6 %.

**Urteil Zweig „mehr Kapital":** eine **echte, aber dünne Risikoprämie** — ~2–6 %/Jahr
bei $10–20k = $200–1200/Jahr brutto der obigen Risiken. Über Sparkonto, aber knapp,
und kein Weg zu „spürbarem monatlichem Einkommen". Kein Backtesting-Problem mehr —
eine Kapital- + Risiko-Toleranz-Entscheidung.

### Zweig „alle Grid-Parameter": breiter Sweep bei FEE=0

`scratchpad/grid_wide_gross.py` — 36 Configs (min_step_fee_multiple × levels ×
sl_mode × leverage), FEE=0, IS (erste ~62 %) / OOS-Split, 5 Symbole.
_(Ergebnis folgt.)_
