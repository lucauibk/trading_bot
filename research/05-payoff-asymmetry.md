# Phase 5 — Payoff-Asymmetrie: Short-Trend-Sleeve (Pre-Registration)

Stand: 2026-08-31. Dieses Dokument wird **vor** jeder Backtest-Ausführung fixiert
und danach nicht mehr rückwirkend verändert (nur ergänzt, mit Datum) — gleiche
Disziplin wie `research/00-hypothesen.md`. Ziel: verhindern, dass ein struktureller
Vorteil vorgetäuscht wird, wo nur ein enger Bootstrap oder ein herausgepicktes
Fenster ist.

## 0. Ist-Stand & Reframe (Fakten, keine Interpretation)

- Der Grid-Bot ist in **jedem** getesteten Segment mit Profit-Factor < 1,0
  gemessen (`research/01`–`04`, `PROGRESS.md`, `project_ranging_gate_merged_2026-07-23`).
  Tot: Grid-Geometrie/Kosten (H1a–c), Regime-Gating weich + hart (H2),
  Cointegration auf den 5 gehandelten Coins (0/10 stationär), Funding-Carry
  (Universum −0,54 %), ML/LLM-Directional (Brier 0,39), Trend-Breakout+Trailing
  **long-only** (`PROGRESS.md`: alle 6 Configs PF 0,46–0,74), Mean-Reversion
  (PF 0,27–0,82).
- Nightly-Sweeps 19.–31.08. (`results/sweep_2026082*`) liefern durchweg **keinen**
  qualifizierenden Config.
- **Reframe (Advisor, 2026-08-31):** Das Kernproblem ist **nicht Fee-Drag,
  sondern Payoff-Asymmetrie.** Der Long-only-Grid ist *short gamma* — kassiert
  ~1,28 % pro Flip (`min_range = KRAKEN_FEE·levels·4`), Fees ~0,32 % Round-Trip
  (~25 % des Steps), und zahlt mehrere Prozent, wenn der Preis durch den
  Grid-Boden trendet (Floor-SL-Kaskade). Alle toten Hebel haben den
  Zweitrundeneffekt (Fees / Trade-Selektion) besteuert, nie die
  Erstrundenursache (Asymmetrie) behoben. Die enge Cross-Coin-Calmar-SD von 0,07
  ist strukturell zur *Payoff-Form*, nicht zum Fee-Schedule.
- **Screen für neue Ideen:** Behebt die Idee die Asymmetrie (statt sie zu
  besteuern) UND ist der Brutto-Edge groß gegen den ~0,32 % Fee-Boden? Es
  bestehen: (a) short-fähige Strategie, (b) echtes Market-Making (braucht
  L2-Orderbuch-Daten — teuer, eigenes Pre-Reg). Dieses Dokument testet (a).

## 1. Hypothesen (vorab fixiert, nicht erweiterbar ohne neues Pre-Reg-Datum)

| # | Hypothese | Framing |
|---|-----------|---------|
| H5a | Ein Short-Donchian-Breakdown + Chandelier-Trailing-Sleeve auf den 5 Alts hat bei echter Spot-Ökonomie (Lev 1×) OOS eine positive Per-Trade-Expectancy, deren 21-Tage-Block-Bootstrap-CI die Null ausschließt. | Standalone |
| H5b | Der Short-Sleeve feuert **zeitlich** in genau den Drawdown-Episoden, die die Floor-SLs des Long-Grids kaskadieren lassen, und deckt dort ≥ 25 % des Grid-Verlusts ab. | Portfolio |

**Warum kein Re-Test eines toten Ergebnisses:** `scripts/trend_breakout_backtest.py:8`
lief ausdrücklich long-only. Git-History hat keinen `short`/`hedge`-Commit. Der
`_hard_trend_down`-Gate in `strategies/grid.py` pausiert nur Buys, eröffnet nie
eine Gegenposition. Die Short-Seite von Trend-Following ist im gesamten Projekt
unberührt.

## 2. Fenster-Definitionen (nie vermischen)

- **IS-dev:** Trades mit `entry_ts < split_ts`, wobei
  `split_ts = data_start + split_days` (Default `--split-days 300`). **Nur hier**
  findet Config-Ranking der Sekundär-Configs statt.
- **OOS-dev:** Trades mit `split_ts <= entry_ts <= 2026-07-22T00:00:00Z`.
  **Einmal** gelesen, ausschließlich für die a-priori Primary-Config.
- **Vault:** Alle Daten `>= 2026-07-23`. Von diesem Experiment **nie geladen**.
  `--as-of 2026-07-22` ist Pflicht bei jedem Lauf (kappt `df = df[df.index <= as_of]`
  vor jeder Berechnung, wörtlich aus `scripts/sweep.py:383-392`).
- **Stress-Sub-Fenster:** Der Juni-2026-Crash liegt innerhalb IS/OOS-dev; wird
  deskriptiv separat berichtet, **nicht** zur Config-Wahl benutzt.
- **`--days`-Falle:** `backtest/data.py:41` zählt `days` von *jetzt* zurück, dann
  kappt `--as-of` oben. Heute (31.08.) − Dev-Cutoff (22.07.) ≈ 40 Tage → für eine
  ~360-Tage-Dev-Spanne `--days 405`. Jeder Lauf loggt pro Symbol
  `df.index[0] → df.index[-1]` + Kerzenzahl nach dem Cap.

## 3. A-priori Primary-Config (Anti-Data-Snooping)

- **Primary (das Urteil):** `entry_lb=20, chand_atr=3.0, use_regime=True`
  (kanonische Turtle/Chandelier-Einstellung; = Headline-Zeile des Long-Baselines).
- **Sekundär-Configs** (`(20,3,False), (20,4,True), (55,3,True), (55,4,True),
  (10,2.5,True)`): auf **IS-dev** gerankt; ihre OOS-Zahlen sind **rein deskriptiv**
  in einem separaten „SECONDARY — do-not-gate"-Block.
- **Robustheits-Check (kein Gate):** Median der 6 OOS-PF. Primary besteht, aber
  Median deutlich < 1,0 → im Writeup als fragil markiert.
- Ersetzt das bestehende `best = max(pf)` über 6 Configs
  (`trend_breakout_backtest.py:140-143`) — genau das Snooping-Muster, das das
  Programm verhindern soll.

## 4. Block-Bootstrap-Spec (die tragende statistische Wahl)

SOL/ETH/AVAX/LINK/XRP sind alle BTC-Beta: ein Downtrend feuert ~5 Shorts, die
gemeinsam gewinnen/verlieren. Ein i.i.d.-Per-Trade-Resample behandelt sie als
unabhängig und schrumpft die CI um ~√5 — so besteht ein toter Sleeve fälschlich
„CI schließt 0 aus".

**Moving-Block-Bootstrap über `entry_ts`:**
- Blocklänge: **21 Kalendertage**, jetzt fixiert, vor jedem Lauf.
- Verfahren: Evaluationsfenster in aufeinanderfolgende 21-Tage-Blöcke kacheln;
  `ceil(window_days / 21)` Blöcke mit Zurücklegen ziehen; Resample = alle Trades,
  deren `entry_ts` in einem gezogenen Block liegt (gepoolt über alle 5 Symbole);
  Expectancy (Mittelwert Per-Trade-`pnl`) je Resample; 10 000 Iterationen;
  2,5-/97,5-Perzentil. `np.random.default_rng(seed=0)`.
- **Beide CIs berichten** (i.i.d. + Block). **Die Block-CI ist das Gate.** Die
  i.i.d.-CI wird nur gezeigt, um die Inflation zu demonstrieren.

## 5. Kill-Kriterien — H5a (Standalone), alle gleichzeitig auf OOS-dev / Primary / gepoolt

| # | Kriterium | Quelle |
|---|-----------|--------|
| S1 | Leverage = 1× für die Headline-Expectancy/CI (`--leverage 1.0`); LEV>1-Zeilen sind Kontext (Harness modelliert keine Liquidation). | `research/00` KC#1 |
| S2 | **21-Tage-Block-Bootstrap-CI der gepoolten Per-Trade-Expectancy schließt 0 aus (positive Seite).** | `research/00` KC#2, gehärtet |
| S3 | ≥ 30 echte OOS-Trades gepoolt über den 5-Symbol-Basket (nicht pro Coin). | `research/00` KC#3 |
| S4 | Gepoolter Profit-Factor > 1,0 auf OOS-dev UND positive gepoolte Expectancy in der Mehrheit der Drittel (OOS-dev in 3 zusammenhängende Zeit-Drittel; „qualifizierend" = ≥ 10 gepoolte Trades). | `research/00` KC#4 |
| S5 | Carry-Sensitivität: ein Lauf mit gemessenem Funding (`research/04:19-28`, Portfolio-Ø ≈ −0,54 %/180d ≈ −0,00003/Tag → `--carry-daily -0.00003`) darf S2/S4 **nicht** von PASS auf FAIL kippen. Kippt es → „carry-fragil", kein Pass. | short ≠ long Kostenstruktur |

**Irgendein S1–S5 FAIL → Standalone-Short-Sleeve für tot erklärt**, OOS-dev als für
diese Hypothese verbraucht, dokumentiert in diesem File.

## 6. Kill-Kriterien — H5b (Portfolio)

Primärtest ist ein **Timing/Overlap-Test**, keine volle Portfolio-Calmar.

| # | Kriterium |
|---|-----------|
| P1 | Top-k (k=5 pro Symbol) Drawdown-Episoden der `GridStrategy`-Equity-Kurve auf denselben gekappten Fenstern (Episode = maximale zusammenhängende Spanne mit Drawdown < −5 %). Short-Sleeve-Kumulativ-PnL, summiert über exakt diese Episoden-Zeitspannen, ist **positiv** UND deckt **≥ 25 %** des Grid-Verlusts in diesen Episoden ab (gepoolt über 5 Symbole, Leverage- und Notional-gematcht; Sleeve-Notional = Grid-`investment` pro Symbol). |
| P2 | Sekundär: Equal-Weight-kombinierte Equity-Kurve (feste Pro-Symbol-Kapitalaufteilung, kein Cross-Margin) hat höheren Terminal-Return UND flacheren `max_drawdown` als Grid-allein auf OOS-dev, in ≥ 4/5 Symbolen. Allokations-Annahme explizit im Header. |
| P3 | Das Standalone-Ergebnis des Sleeves (H5a) ist **nicht schlechter als break-even** — verhindert, dass ein negativ-EV-Hedge nur durch Glück im Overlap-Fenster besteht. |

**P1 FAIL → der Mechanismus-Claim („Short feuert in den Grid-Floor-Kaskaden") ist
falsifiziert; kein struktureller Portfolio-Nutzen → tot für H5b.** P1 PASS aber
P2 FAIL → „Hedge-Timing stimmt, Sizing/Kosten fressen es" — dokumentiert,
schwach-positiv, kein grünes Licht.

## 7. Feasibility-Gate auf den „nächsten Schritt" (vorab)

Kraken ist spot-only; `LIVE_PARITY_OK=False`; `PROGRESS.md` offener Punkt #4 nennt
Margin/Perp-Parität „ein eigenes Projekt". PaperBroker kann aktuell nur
Long-Margin (`execution/paper.py:167`). **Ein Short-Sleeve kann auf dem aktuellen
Venue nicht ehrlich paper-getradet werden.** Grünes Licht (H5a + H5b PASS auf
OOS-dev) bedeutet daher: ein Margin/Perp-Parity-Projekt wird zur Voraussetzung;
der Paper-Toggle (Default aus) kommt erst hinter dieser Schicht und nach deren
eigenen Paritäts-Tests.

## 8. Nächste konkrete Schritte

1. Dieses File committen (Pre-Registration fixiert).
2. `scripts/trend_breakout_backtest.py` refactoren (`--side {long,short}`,
   `--as-of`, `--split-days`, `--leverage`, `--carry-daily`, Block-Bootstrap-CI,
   IS/OOS-Partition nach `entry_ts`). Long-Pfad bit-identisch erhalten.
3. **Regressions-Check ZUERST:**
   `python3 scripts/trend_breakout_backtest.py --side long --days 400 --leverage 3.0`
   → Primary + Sekundär PF müssen im protokollierten Band 0,46–0,74 landen
   (`PROGRESS.md`). Außerhalb → Refactor hat Semantik geändert, stoppen und fixen.
4. `scripts/short_sleeve_portfolio.py` neu (H5b).
5. Vorregistrierte Läufe (Details in `/Users/lucasturz/.claude-personal/plans/ich-will-das-du-indexed-cosmos.md`), Befunde hier anhängen.

---

## Befunde

_(wird nach den Läufen mit Datum angehängt — nicht rückwirkend editieren)_
