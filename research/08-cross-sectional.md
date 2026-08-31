# Phase 8 — Cross-Sectional Momentum + BTC-Trend-Switch (Pre-Registration)

Stand: 2026-08-31. Vorregistriert **vor** jeder Backtest-Ausführung, danach nur
ergänzt (mit Datum). Disziplin wie `research/00`.

## 0. Warum dieser Hebel, nachdem alles andere tot ist

Alle bisherigen Tests waren: **intraday bis Tage**, **Kraken-Spot**, **5 BTC-Beta-Alts**,
**Richtungs-/Mean-Reversion-Prognose**. Diese Phase ändert **drei Dimensionen
gleichzeitig**:

1. **Frequenz:** wöchentlich (nicht 1h) → der ~0,3 % Round-Trip, der jeden bisherigen
   Hebel getötet hat, amortisiert sich über Wochen-Holds.
2. **Signal-Typ:** cross-sectional (Assets *ranken*), nicht directional (hoch/runter
   *vorhersagen*) — braucht nicht die Prognose, die mit F1 ≈ 0,50 / Brier 0,39 als
   nicht machbar belegt ist.
3. **Universum:** ~40 Coins statt 5 — cross-sectionale Effekte brauchen Breite.

Prior: Krypto-Cross-Sectional-Momentum hat publizierte Evidenz (Liu & Tsyvinski
2021 u.a.), aber die Samples sind 2017–2020 und der Effekt kann wegarbitragiert
sein. **Ehrlicher Prior: ~50/50.** Momentum-Crashes sind bekannt und heftig.

## 1. Hypothesen

| # | Hypothese |
|---|-----------|
| H8a | Ein Long-Only Top-Quintil-Cross-Sectional-Momentum-Portfolio (wöchentlich rebalanciert, Ranking nach trailing 4/8/12-Wochen-Return) schlägt BTC-Buy-and-Hold **netto nach Fees** in der risikoadjustierten Rendite (Sharpe) UND hat positive Netto-CAGR — in-sample UND out-of-sample. |
| H8b | Ein BTC-Trend-Risikoschalter (halte BTC wenn Close > 200-Tage-SMA, sonst USDT) schlägt BTC-Buy-and-Hold in Sharpe UND Max-Drawdown, netto nach Fees. |

## 2. Daten & Universum

- **Quelle:** Binance Daily-Closes (`ccxt`), wie `backtest/data.py` bereits nutzt
  (Kraken-Alt-Historie zu dünn/kurz). Neues Skript, kein Eingriff in `backtest/`.
- **Universum:** Top ~40 USDT-Spot-Paare nach aktuellem 24h-Quote-Volumen, **exkl.**
  Stablecoins, Gold-Token (XAUT/PAXG), tokenisierte Aktien (…B-Suffix), sowie Coins
  mit < 400 Tagen Historie (nicht genug Lookback).
- **⚠️ Survivorship Bias — bewusst akzeptiert:** Das Universum ist „Top-Coins die
  HEUTE existieren". Das überschätzt Momentum-Renditen (die Verlierer/Toten fehlen).
  Dieser Test ist damit eine **optimistische Obergrenze**: schlägt er BTC-B&H hier
  *nicht*, ist er erst recht tot. Schlägt er, braucht es einen zweiten Test mit
  point-in-time-Universum (historische Market-Caps) bevor irgendetwas live geht.

## 3. Dev/Vault-Split

- **Dev:** alle Daten **bis einschließlich 2025-12-31**. Hier: Parameterwahl
  (Lookback 4 vs. 8 vs. 12 Wochen, Quintil vs. Terzil, Long-only vs. Long-Short).
- **Vault:** **2026-01-01 → jetzt**. Nicht angeschaut, bis ein Dev-Kandidat besteht.
- Zusätzlich Dev in zwei Hälften (früh/spät) — Kill-Kriterium verlangt positives
  Vorzeichen in **beiden**.

## 4. Kosten (vorab fixiert)

- **0,15 % pro Seite** auf den Portfolio-Turnover je Rebalance (Binance Spot
  Maker/Taker ~0,1 %; konservativ 0,15 %). Round-Trip auf getauschtes Notional
  0,30 %.
- Turnover = Σ|w_neu − w_alt| / 2. Fee-Drag pro Rebalance = Turnover × 2 × 0,15 %.
- Kein Slippage-Modell zusätzlich (Daily-Close-Fills; die wöchentliche Frequenz und
  Top-Volumen-Coins machen das vertretbar — als Kaveat notiert).

## 5. Kill-Kriterien

**H8a — alle gleichzeitig:**

| # | Kriterium |
|---|-----------|
| M1 | Netto-CAGR (nach Fees) > 0 über das volle Dev-Fenster. |
| M2 | Netto-Sharpe > Sharpe(BTC-Buy-and-Hold) über dasselbe Fenster, mit Abstand ≥ 0,2. |
| M3 | Netto-CAGR positiv in **beiden** Dev-Hälften (nicht nur einer). |
| M4 | Der Bruttoeffekt überlebt die Kosten: Brutto-CAGR − Netto-CAGR (= Fee-Drag) < ½ der Netto-CAGR. Wenn die Fees mehr als ⅓ des Bruttoertrags fressen, ist es fragil. |
| M5 | Gilt für **mindestens 2 der 3** Lookbacks (4/8/12 W) — ein einzelner Gewinner ist Rosinenpickerei. |

**H8b — beide:**

| # | Kriterium |
|---|-----------|
| T1 | Netto-Sharpe > Sharpe(BTC-B&H) + 0,2. |
| T2 | Netto-Max-Drawdown betragsmäßig < 0,7 × Max-Drawdown(BTC-B&H). |

**Nicht erfüllt → ehrlicher Stopp, Dev-Fenster verbraucht, dokumentiert.** Bei
Bestehen: Vault-Lauf (einmalig) + point-in-time-Universum-Nachtest, bevor irgendein
Live-/Paper-Schritt.

## 6. Metriken (berichtet)

Je Config: Brutto/Netto-CAGR, Sharpe (annualisiert, wöchentliche Returns),
Sortino, Max-Drawdown, Ø-jährlicher Turnover, Fee-Drag (Prozentpunkte CAGR),
Trefferquote der Wochen, längste Underwater-Periode. Immer neben der
BTC-B&H- und der Equal-Weight-B&H-Benchmark.

---

## Befunde

_(wird nach den Läufen mit Datum angehängt — nicht rückwirkend editieren)_
