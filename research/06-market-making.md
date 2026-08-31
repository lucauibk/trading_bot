# Phase 6 — Market-Making: Feasibility-Screen (2026-08-31)

Bezug: `research/05-payoff-asymmetry.md` Gesamtfazit — nach dem Tod des
Short-Sleeves war echtes Market-Making (adaptive Quotes am Best-Bid/Ask, den Spread
*kassieren* statt zahlen) der letzte unberührte Hebel mit plausiblem Mechanismus.

## Vorregistriertes Kill-Kriterium

Der zeitgemittelte gequotete Spread muss den Maker-Round-Trip-Fee-Boden strukturell
übersteigen, über alle 5 Symbole — **bevor** Adverse Selection überhaupt modelliert
wird. Formal: `quoted_spread − 2·maker_fee > 0` im Median.

## Befund: dead on arithmetic

**Live-L2-Sample Kraken (2026-08-31, 5 Snapshots/Symbol, `ccxt.fetch_order_book`):**

| Symbol | Mid | Quoted Spread | Halb-Spread | Spread − Fee-Boden |
|--------|-----|---------------|-------------|--------------------|
| SOL/USD | 102,75 | 0,97 bps | 0,49 bps | **−31,0 bps** |
| ETH/USD | 2451,42 | 0,04 bps | 0,02 bps | **−32,0 bps** |
| AVAX/USD | 7,17 | 1,39 bps | 0,70 bps | **−30,6 bps** |
| LINK/USD | 11,28 | 2,94 bps | 1,47 bps | **−29,1 bps** |
| XRP/USD | 1,37 | 1,61 bps | 0,80 bps | **−30,4 bps** |

**Maker-Fee-Boden (Kraken Pro, 0,16 %/Seite, Round-Trip): 32,0 bps.**

Der gequotete Spread liegt bei **0,04–2,9 bps** — Faktor **10–800× unter** dem
Fee-Boden. Wer am Touch quotet und auf beiden Seiten gefüllt wird, zahlt 32 bps
Gebühren, um ~1 bps Spread zu kassieren. Adverse Selection (Preis läuft nach dem
Fill gegen einen) macht es nur schlimmer, kann es aber nicht retten.

**Kraken-Fee-Tiers** (`ccxt.kraken().fees['trading']['tiers']`): Maker 0,16 % bis
$50k/30d-Volumen, fällt auf 0,10 % bei $250k, **0,00 % erst ab $10M/30d**. Selbst
im Null-Fee-Tier ist ein ~1-bps-Spread gegen HFT-Konkurrenz kein retail-tauglicher
Edge — und $10M/Monat Umsatz sind mit ~500 USDT Kapital unerreichbar.

## Urteil

**Market-Making auf Kraken-Spot ist bei retail Fee-Tier tot** — nicht durch
Overfitting, sondern durch Arithmetik. Kein Mehr-Wochen-L2-Sammellauf gerechtfertigt
(das Kill-Kriterium ist schon aus einem Snapshot eindeutig verfehlt).

Damit sind **alle** Hebel aus dem vorregistrierten Programm + Backlog geprüft und
tot. Nächster Schritt: breite Recherche nach echtem Methodenwechsel (anderer Venue,
Instrument, Datenquelle, Strategie-Klasse) — `research/07-*` falls ein Kandidat den
Random-Walk- + Kosten-Filter besteht.
