# Phase 10 — Offengelegte Trades (Congress + Insider) (Pre-Registration)

Stand: 2026-10-03. Vorregistriert **vor** jeder Rendite-Berechnung, danach nur
ergänzt (mit Datum). Disziplin wie `research/00`.

## 0. Warum, und was hier NICHT getestet wird

Anlass: Wunsch nach einem Copy-Trading-Modul „Politiker + Insider" (Quiver Quantitative
API). Statt Bot-Code erst die Frage: **Gibt es nach Meldedatum, Kosten und Faktor-Exposure
überhaupt einen Edge?** Daten sind bewusst kostenlos (kein Quiver-Abo).

Prior:
- **Congress:** ältere Studien positiv (Ziobrowski 2004/2011), neuere nach STOCK Act
  ≈ 0 (Eggers & Hainmueller 2013, Belmont et al. 2022). Copy-ETFs (NANC/KRUZ) ≈ Tech-Beta.
  **Ehrlicher Prior: ~15 %.**
- **Insider-Clusterkäufe:** solide Literatur (Lakonishok & Lee 2001, Cohen/Malloy/
  Pomorski 2012), Effekt aber v. a. Small/Micro-Caps und nach Faktor-Kontrolle kleiner.
  **Ehrlicher Prior: ~35 %**, dass nach Kosten + FF-Faktoren etwas übrig bleibt.

**Nicht getestet:** House (also auch **Pelosi**), Optionen, Echtzeit-Execution. Ein FAIL
von H10a heißt „Senate-Aktienkäufe 2013–2021 kopieren bringt nichts", nicht
„Pelosi-Copy ist tot".

## 1. Hypothesen

| # | Hypothese |
|---|-----------|
| H10a | Senate-Aktienkäufe (asset_type `Stock`, type `Purchase`, `amount_low ≥ 15 001 $`), Einstieg am Close des ersten Handelstags **nach** dem Meldedatum (`date_recieved`), schlagen den Markt netto und faktor-bereinigt. |
| H10b | Insider-Clusterkäufe (Definition §2.2), Einstieg am Close des ersten Handelstags **nach** dem Meldedatum des cluster-vollendenden Form 4, schlagen den Markt netto und faktor-bereinigt. |

Haltedauern je Hypothese: **20 / 60 / 120 Handelstage**, fixer Exit (kein SL/TP).

## 2. Daten

### 2.1 H10a — Senate
- Quelle: `timothycarambat/senate-stock-watcher-data` (GitHub), Dateien
  `data/transaction_report_for_MM_DD_YYYY.json`, Feld `date_recieved` = Meldedatum.
  Abdeckung 2012-07 → 2021-03. House-Stock-Watcher-S3 ist offline (403).
- Ticker: HTML-Link entfernen; `--`, leere Ticker, Anleihen, Optionen, „Other Securities",
  `Exchange` verworfen. Mehrere Käufe derselben Person, desselben Tickers, derselben
  Meldung → ein Event.

### 2.2 H10b — Insider (SEC Form 345 Data Sets, quartalsweise, 2013Q1 → aktuellste)
- Nur `DOCUMENT_TYPE == "4"` (4/A-Amendments verworfen), `NONDERIV_TRANS`,
  `TRANS_CODE == "P"`, `TRANS_ACQUIRED_DISP_CD == "A"`.
- Nur Reporting-Owner mit Relationship **Director oder Officer** (reine 10 %-Owner raus).
- Eine Accession = eine Meldung. Wert = Σ Shares × Preis; gewichteter Ø-Preis.
  Filter (point-in-time, aus Form-4-Werten, nicht aus yfinance): Ø-Preis ≥ 5 $.
- **Cluster:** Für einen Issuer ≥ 2 Meldungen, deren Owner-CIK-Mengen **disjunkt** sind
  (gemeinsame Einreichungen Fonds + GP zählen als eine), Transaktionsdaten innerhalb
  von 30 Kalendertagen, Summe ≥ 100 000 $. Event-Datum = Meldedatum der Meldung, die die
  Bedingung erstmals erfüllt. Danach 90 Kalendertage Sperre je Issuer (kein Doppel-Event).
- **Nachtrag 2026-10-04 (vor jeder Rendite-Berechnung):** Meldungen mit Meldeverzug
  > 30 Tage werden vor dem Clustering verworfen (2,2 % der Meldungen; v. a. verspätet
  gemeldete Plan-Käufe und Datums-Tippfehler). Anlass: Stichprobe CTBI, Kauf 2024-01-02,
  gemeldet 2024-07-23. Entspricht `max_filing_delay_days = 30` aus der Modul-Spec.

### 2.3 Kurse, Faktoren
- yfinance (Daily, split- und dividendenbereinigt für Renditen; `actions=True` für Splits).
- **Ticker-Reuse-Schutz (H10b):** Form-4-Ø-Preis vs. yfinance-Close am Transaktionstag,
  auf **unbereinigt (Split)** zurückgerechnet; Abweichung > 10 % → Event verworfen.
- **Liquiditätsfilter (beide):** Median-Dollar-Volumen der 20 Handelstage vor dem
  Einstieg ≥ 1 Mio. $.
- Benchmark/Faktoren: Fama-French 5 Faktoren (2×3) + Momentum, daily (Ken French
  Data Library). SPY nur für Portfolio-Simulation (K4).

## 3. Dev/Vault-Split

| | Dev (zwei Hälften) | Vault |
|-|--------------------|-------|
| H10a | 2013-01-01 → 2017-12-31 (2013–2015 / 2016–2017) | 2018-01-01 → 2021-03-10 |
| H10b | 2013-01-01 → 2023-12-31 (2013–2018 / 2019–2023) | 2024-01-01 → letzter Event mit voller Haltedauer |

Split nach Event-Datum. Vault nur einmal, nur nach Dev-PASS.

## 4. Kosten (fixiert)

- **Basis:** 0,10 % Kommission/Spread + 0,25 % FX (CHF/EUR → USD) **pro Seite**
  = 0,70 % Round-Trip, vom Event-Return abgezogen.
- **Sensitivität (berichtet, nicht entscheidend):** 0,50 % + 0,25 % pro Seite.
- **K4 zusätzlich:** Mindestkommission 1 $ pro Order (Retail-Broker), relevant bei
  60-$-Positionen.

## 5. Methode

- **Calendar-Time-Portfolio** je Haltedauer H: täglich gleichgewichtet über alle offenen
  Event-Positionen; Kosten am Einstiegs- und Ausstiegstag anteilig abgezogen. Tage ohne
  offene Position fallen raus.
- **Placebo:** je Event derselbe Ticker an einem zufälligen Handelstag im selben
  Dev-Fenster (Seed 0), gleiche Filter und Kosten. Trägt denselben Size-/Survivorship-Mix.
- Regression: `R_p − RF = α + β·(Mkt−RF) + s·SMB + h·HML + r·RMW + c·CMA + m·UMD`,
  Newey-West (20 Lags).

## 6. Kill-Kriterien (je Hypothese, alle gleichzeitig)

| # | Kriterium |
|---|-----------|
| C0 | **Abdeckung:** ≥ 70 % der Events haben gültige Kurse nach allen Checks. Sonst Ergebnis = **INCONCLUSIVE** (weder PASS noch FAIL). |
| K1 | Netto-Calendar-Portfolio minus Markt (Mkt = Mkt−RF + RF): Mittel > 0, 95 %-Block-Bootstrap-CI (zirkulär, 21-Tage-Blöcke, 10 000 Iter.) schließt 0 aus. |
| P1 | Netto-Event-Portfolio minus Placebo-Portfolio: Mittel > 0, CI wie K1 schließt 0 aus. |
| K2 | K1-Mittel positiv in **beiden** Dev-Hälften. |
| K5 | FF5+UMD-α (netto) > 0 mit Newey-West-t ≥ 2,0. |
| K3 | K1, P1, K2, K5 gelten für **≥ 2 von 3** Haltedauern. |
| K4 | Portfolio-Simulation (3000 $, 2 % je Position, max. 20 offen, Rest in SPY, Mindestkommission): Netto-CAGR ≥ SPY-CAGR **und** Sharpe ≥ Sharpe(SPY) + 0,2 — für die beste der bestehenden Haltedauern. |

**Nicht erfüllt → ehrlicher Stopp, dokumentiert.** Bei PASS: Vault-Lauf, danach
Survivorship-Nachtest (delistete Kurse) und House-Daten 2021+ (Capitol API / Bargo),
erst dann Diskussion über einen separaten Equity-Bot.

## 7. Bekannte Biases (benannt, nicht behoben)

- **Survivorship in beide Richtungen:** delistete Ticker fehlen bei yfinance —
  Pleiten (überschätzt Rendite) **und** Übernahmen mit Prämie (unterschätzt sie,
  bei Insider-Small-Caps häufig). Richtung unklar → deshalb P1 (Placebo) als Kriterium.
- Congress-Beträge sind Spannen; Senate-Datensatz ist ein Community-Scrape.
- Kurs-Fills am Daily-Close, keine Intraday-Slippage.

---

## Befunde

### 2026-10-04 — Dev-Läufe (`scripts/disclosure_event_study.py`)

Kosten Basis 0,35 %/Seite. „Ø/Tag" = Calendar-Time-Portfolio, %/J = × 252.
Logs: `data/research_cache/run_{a,b}_dev.log`, JSON: `data/research_cache/phase10_dev_{a,b}.json`.

#### H10b — Insider-Clusterkäufe (8 498 Events, 3 144 handelbar)

| Halte | Netto − Markt %/J (95 %-CI) | Netto − Placebo %/J | FF5+UMD-α %/J (t) | SMB | Depot 3000 $ vs. SPY CAGR |
|-------|-----------------------------|---------------------|-------------------|-----|---------------------------|
| 20 T  | **−12,5** [−21,4; −4,0] | +0,8 [−6,6; +7,8] | −10,6 (−3,34) | +0,71 | −0,7 % vs. +13,4 % |
| 60 T  | −3,1 [−11,1; +4,5] | +1,5 [−3,0; +6,0] | −0,9 (−0,41) | +0,72 | +7,2 % vs. +14,1 % |
| 120 T | −3,4 [−11,1; +4,0] | −0,4 [−4,0; +3,3] | −0,3 (−0,16) | +0,72 | +10,0 % vs. +14,2 % |

- **C0 verfehlt: Abdeckung 48 %** (3 620 Events ohne yfinance-Kurs, 468 Preis-Mismatch,
  333 Kurslücken; 933 zusätzlich illiquide). Abdeckung steigt von 46 % (2013) auf 77 %
  (2023) → Delistings und Ticker-Umbenennungen (SEC nennt den Ticker zum Meldezeitpunkt; z. B. TWTR, SIVB), keine Download-Lücke (kein Rate-Limit-Rest).
- Alle Kern-Kriterien K1/P1/K2/K5/K4 verfehlt, für alle drei Haltedauern. Kein einziger
  Punktschätzer netto über Markt; Placebo-Differenz ≈ 0 → das Cluster-Signal fügt dem
  Ticker-Profil nichts hinzu. SMB-Ladung +0,72 bestätigt den Small-Cap-Tilt.
- Auch brutto kein Edge: 20 T netto −12,5 %/J bei ≈ 8,8 %/J Kostendrag → brutto ≈ −3,7 %/J.

**VERDICT H10b Dev: INCONCLUSIVE (formal, C0).** Inhaltlich: auf der handelbaren Hälfte
nicht der Ansatz eines Edges. Damit ein PASS herauskäme, müssten die fehlenden 52 %
(delistet: Übernahmen *und* Pleiten) eine stark positive Überrendite tragen, die die
negative Hälfte mehr als ausgleicht. Mit kostenlosen Daten nicht prüfbar. Vault bleibt gesperrt.

#### H10a — Senate-Aktienkäufe (224 Events, 182 handelbar, Abdeckung 81 %)

| Halte | Netto − Markt %/J (95 %-CI) | Netto − Placebo %/J | FF5+UMD-α %/J (t) | Depot 3000 $ vs. SPY CAGR |
|-------|-----------------------------|---------------------|-------------------|---------------------------|
| 20 T  | −12,0 [−27,0; +3,3] | −0,4 [−22,9; +21,4] | −12,9 (−1,55) | +8,9 % vs. +13,4 % |
| 60 T  | +0,7 [−11,6; +13,6] | +5,1 [−7,6; +17,9] | +1,7 (+0,26) | +6,9 % vs. +9,6 % |
| 120 T | +2,8 [−6,0; +12,3] | +3,4 [−5,3; +12,6] | +3,6 (+0,73) | +8,3 % vs. +11,1 % |

- C0 erfüllt. K1/P1/K2/K5/K4 für alle Haltedauern verfehlt. K2: frühe Hälfte (2013–15)
  n = 40 positiv, späte Hälfte (2016–17) n = 142 negativ.
- Das Depot (2 % je Position, Rest SPY) liegt in jeder Variante **unter reinem SPY**.
- **Teststärke gering:** nur 182 Positionen, CI-Breite ≈ ±12 %/J. Ein FAIL heißt hier
  „kein nachweisbarer Edge", nicht „bewiesen null". Nur Senate 2015–2017; House und
  Pelosi ungetestet (kein kostenloser Datensatz mit Meldedatum).

**VERDICT H10a Dev: FAIL.** Vault (2018–2021) nicht geöffnet.

### Gesamturteil Phase 10

Copy-Trading offengelegter Trades ist mit öffentlich-kostenlosen Daten und Retail-Kosten
**nicht als Edge belegbar**: Senate FAIL (schwache Stärke), Insider-Cluster formal
INCONCLUSIVE, aber auf der messbaren Hälfte netto und brutto ≤ 0. Ein Quiver-Abo + Equity-
Broker-Integration für den Bot ist damit nicht gerechtfertigt. Offener Rest, falls jemals:
Insider mit point-in-time-Kursen inkl. Delistings (CRSP/bezahlte Daten) — Prior nach diesem
Lauf weiter gesunken.
