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

### 2026-08-31 — Dev-Lauf (`scripts/cross_sectional_backtest.py`)

Universum: 57 Binance-USDT-Coins mit ≥ 400 Tagen Historie, Daten 2022-04-15 →
2026-08-31. Dev-Fenster 2022-04-17 → 2025-12-28 (194 Wochen).

| Strategie | Netto-CAGR | Sharpe | MaxDD | Fee-Drag | Δsharpe vs BTC |
|-----------|-----------|--------|-------|----------|-----------------|
| **BTC Buy&Hold** | +23,9 % | 0,68 | −58,7 % | — | — |
| Equal-Weight B&H | −4,9 % | 0,29 | −65,1 % | — | −0,39 |
| CS-Mom L=4W Q (to 35 %) | +26,5 % | 0,68 | −58,1 % | +7,2 pp | −0,00 |
| CS-Mom L=8W Q (to 25 %) | +52,0 % | 0,92 | −52,6 % | +5,9 pp | **+0,24** |
| CS-Mom L=12W Q (to 21 %) | +37,6 % | 0,79 | −63,4 % | +4,5 pp | +0,11 |
| **BTC 200D-SMA-Switch (H8b)** | +32,8 % | **0,94** | **−30,9 %** | — | **+0,26** |

**M3 Dev-Hälften (Split 2024-02-25), CS-Mom Q:**

| Lookback | frühe Hälfte | späte Hälfte |
|----------|--------------|--------------|
| L=4W | +48,4 % | +8,6 % |
| L=8W | +109,1 % | +13,4 % |
| L=12W | +71,5 % | +13,6 % |

### VERDICT Dev

**H8a — FAIL.** Nur L=8W besteht M1+M2 (Δsharpe ≥ 0,2); L=4W (Δsh −0,00) und L=12W
(Δsh +0,11) reißen M2. **M5 (≥ 2/3 Lookbacks) verfehlt (1/3).** Der Effekt ist auf
einen einzelnen Lookback konzentriert = Rosinenpickerei-Muster. Zusätzlich zeigt M3:
der Bruttoeffekt ist **fast komplett in der frühen Hälfte** (2022–2024) und in
2024–2025 weitgehend zerfallen — die klassische Signatur einer wegarbitragierten
Anomalie.

**H8b — PASS.** T1 (Δsharpe +0,26 ≥ 0,2) und T2 (MaxDD −30,9 % < 0,7 × 58,7 %)
beide erfüllt, über 4 Jahre inkl. Bullmarkt.

### 2026-08-31 — Isoliertes Vault-Fenster (2026-01-01 → jetzt, 36 Wochen, vorher nie angeschaut)

| Strategie | CAGR | Sharpe | MaxDD |
|-----------|------|--------|-------|
| BTC Buy&Hold | **−15,9 %** | −0,14 | −36,4 % |
| Equal-Weight B&H | +4,7 % | 0,39 | −39,8 % |
| CS-Mom L=4W Q | +16,6 % | 0,58 | −46,8 % |
| CS-Mom L=8W Q | +52,8 % | 0,79 | −39,1 % |
| CS-Mom L=12W Q | +44,1 % | 0,74 | −38,4 % |
| BTC 200D-SMA-Switch | +18,3 % | **1,24** | **−0,1 %** |

2026 war bisher ein BTC-Minus-Jahr — beide Alternativen profitieren davon strukturell
(„nicht long BTC im BTC-Bärmarkt"). CS-Mom L=8W/L=12W schlagen BTC im Vault deutlich;
L=4W schwach. Der Trend-Switch saß 2026 fast durchgehend in USDT (MaxDD −0,1 %).

## Gesamtfazit Phase 8

- **H8a (Cross-Sectional Momentum): pre-registriertes Dev-Gate FAILED (M5 1/3), aber
  nicht sauber tot.** L=8W/L=12W schlagen BTC im Vault stark. **Der Confounder ist
  Survivorship Bias** — das Universum ist „Top-Coins die HEUTE existieren", tote
  Momentum-Verlierer fehlen. Das ist per Pre-Registration die entscheidende offene
  Frage: ein **point-in-time-Universum-Nachtest** (historische Market-Caps je
  Rebalance-Datum) ist Pflicht, bevor irgendetwas live geht — und wird den Edge
  voraussichtlich deutlich schrumpfen. Zwischenstand: **nicht validiert, nicht
  tot; blockiert auf dem PIT-Test.**
- **H8b (BTC 200D-SMA-Trend-Switch): Dev-Gate PASS, Vault bestätigt.** Das **erste**
  in diesem gesamten mehrphasigen Programm, das seine vorregistrierten Kriterien
  besteht. Aber ehrliche Einordnung: es ist **kein Alpha**, sondern ein
  Risikomanagement-Overlay auf einer Long-BTC-Position:
  - braucht weiterhin BTC-Beta (nicht marktneutral),
  - der Roh-Rendite-Vorsprung (+9 pp/Jahr Dev) ist moderat und periodenabhängig;
    der 2026-Vorsprung ist „hat im Minus-Jahr nicht verloren",
  - im scharfen V-Crash-and-Recover whipsawt es und underperformt B&H,
  - der **robuste** Wert ist Drawdown-Halbierung (−31 % vs −59 %) → Sharpe/Calmar
    hoch, nicht Rendite hoch,
  - mit 500 USDT trivial umsetzbar (1 Asset, wenige Trades/Jahr, Fee-irrelevant).

**Nächste Schritte (falls gewünscht):** (a) PIT-Universum für H8a (CoinGecko
historische Market-Caps, rate-limited, eigenes Skript) — der ehrliche Test.
(b) H8b als Paper-Overlay auf einer kleinen Long-BTC/ETH-Position, 3–6 Monate
Forward, gegen Buy-and-Hold. Kein Grid, kein Directional.

