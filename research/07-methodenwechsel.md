# Phase 7 — Methodenwechsel-Recherche + realistische Profit/Risiko-Einschätzung (2026-08-31)

Bezug: `research/05` + `research/06` — alle Hebel aus dem vorregistrierten Programm
und dem Backlog sind geprüft und tot. Der User ist offen für einen kompletten
Methodenwechsel (anderer Venue / Instrument / Datenquelle / Strategie-Klasse). Eine
breite Recherche (Venue-Fee-Schedules, Funding-Daten, LP/IL-Studien,
Options-Marktstruktur, Event-Study-Literatur, On-Chain-Flow-Papers, Stand 2026)
wurde durchgeführt.

## Zwei Fakten, die die Ausgangslage verschieben

1. **Krakens Maker-Fee im untersten Tier ist 0,25–0,40 %, nicht 0,16 %.** Die
   0,16 % gelten nur für Stablecoin/FX-Paare bei mittlerem Volumen. Ein
   Maker-Rebate (−0,02 %) gibt es erst ab $10M/30d-Volumen. → Kraken-Spot ist für
   alles Fee-Sensitive der schlechtestmögliche Venue. (Anmerkung: `CLAUDE.md` /
   `KRAKEN_FEE = 0.0016` unterschätzt die reale Fee für dieses Konto.)
2. **EU/MiCA (Operator vermutlich Deutschland).** Seit 01.07.2026 nur noch ~324
   zugelassene CASPs; Bybit u.a. haben EEA-Zugang eingeschränkt. Deribit bedient
   „most of Europe" für berechtigte Clients, Hyperliquid ist KYC-frei aber
   EU-rechtliche Grauzone. **Steuer:** Perp-Funding-Einkommen vs. 1-Jahres-
   Haltefrist-Steuerfreiheit auf Spot — vor jedem Einsatz prüfen, die Steuer kann
   einen $20–40/Jahr-Edge dominieren.

## Bewertungsmaßstab: Dollar/Jahr auf 500 USDT gegen eine Hürde

Prozente sind bei dieser Größe irreführend. Relevante Frage je Idee:
**Dollar/Jahr auf 500 USDT netto nach Kosten, minus was das Geld als
Stablecoin-Lending verdienen würde.**

- **Hürde (Nichtstun):** USDC-Lending (Aave, Kraken/Coinbase Earn) ~4–9 % APR 2026
  = **$20–45/Jahr** auf 500 USDT, mit moderatem (nicht null) Venue/Contract-Risiko.
- **Kapital ist der bindende Constraint, nicht die Ideen.** Mehrere Mechanismen
  unten sind strukturell echt und bestehen den Random-Walk-Test. Sie scheitern
  hier, weil bei 500–3000 USDT der absolute Payout kleiner ist als das
  eingegangene Gegenpartei/Venue-Risiko.

## Ranking der Kandidaten

### 1. Einzahlung in einen Liquidations-/MM-Vault (Hyperliquid HLP o.ä.)

**Keine Strategie, die man betreibt — eine Einzahlung in die Strategie eines
anderen.** Mechanismus: der Vault ist Auto-Market-Maker + Backstop-Liquidator,
verdient Spread + Fee-Anteil + Übernahme liquidierter Positionen zu günstigen
Preisen. Gegenpartei: über-gehebelte Perp-Trader. Besteht den Random-Walk-Test
(prognostiziert keine Richtung).

- Realistisch: **~15–25 % APR** im Normaljahr, **−10 bis −20 % im schlechten
  Quartal**, historisch über 12-Monats-Fenster netto positiv. Tail: **−100 %** bei
  Bridge/Contract-Exploit oder Venue-Insolvenz (das IST die These, keine Fußnote).
- Auf ≤ $200 Einzahlung: **+$30–50/Jahr** erwartet in ~75 % der Jahre, ~1-zu-4
  Mehrjahres-Chance, den Großteil zu verlieren.
- Kill-Kriterium (vorab): schlechtestes rollendes 30-Tage-Fenster der letzten 18 Mon.
  < −15 %, ODER trailing-12M-Sharpe < 1,0, ODER Median-Monatsrendite < +0,8 %.

### 2. Spot-Perp Funding-Carry (delta-neutral) auf Low-Fee-Venue (NICHT Kraken)

Long Spot + Short Perp gleicher Größe, kassiert Funding solange positiv. Besteht den
Test (Funding ist vertraglich, keine Prognose) — Risikoprämie, kein Alpha.

- BTC/ETH-Funding mid-2026: **hohe einstellige %** annualisiert (Kompression von
  25 %+ vor zwei Jahren). Alts höher aber volatiler, ~35 % der 8h-Perioden negativ.
- Netto realistisch **4–10 % APR auf eingesetztes Kapital**; bei 500 USDT nur
  ~200–250 USDT effektiv arbeitend (zwei Legs + Margin-Puffer) → **~$10–25/Jahr**.
  Praktiker-Minimum ~$10.000.
- Braucht: Non-Kraken-Venue, always-on-Prozess mit Auto-Deleverage bei
  Funding-Flip (nicht-trivialer Code, hinter dem Execution-Parity-Lock),
  EU-Rechts-Check.
- Kill-Kriterium (vorab): Median netto-nach-Fee-APR < 12 % auf ≥ 3 von 5 Assets,
  ODER > 40 % der 8h-Perioden negativ, ODER schlechtestes 30-Tage-Kumulativ-Funding
  negativ um > ein Monats-Carry.

### 3. Kraken Passive-Maker / Rebate — tot, Kapital-infeasible

Maker-Fee bei diesem Tier +0,25 bis +0,40 %, Rebate erst ab $10M/30d. Keine
positiv-tragende Version bei 500 USDT.

### 4. Benchmark — Stablecoin-Lending / Nichtstun

~4–9 % APR = **$20–45/Jahr**. Das ist die ehrliche Latte. Option 1 schlägt sie in
Erwartung; Option 2 kaum oder nicht bei 500 USDT.

## Ausgeschlossen (mit konkretem Grund)

| Idee | Grund |
|------|-------|
| Optionen verkaufen (Deribit CC/CSP/Strangle) | Kapital-infeasible: Min-Order 0,1 BTC / 1 ETH bzw. USDC-Notional > 500 USDT + Margin; ein Naked-Short-Tail übersteigt das Konto. Erst ab > $25k. |
| Uniswap v3 Concentrated LP | Brutto-Edge negativ VOR eigenen Kosten: IL > Fee-Income für > 50 % der LPs / fast alle Pools. Ist verdeckt Short-Gamma — diversifiziert nicht mal gegen Option-Selling. |
| Token-Unlock-Shorting / Index-Rebalance | Effekt real aber front-run (Unlock-Druck 30 Tage vorher, Index-Flow „letzte 60 Sekunden"); braucht Shorten von Small-Caps, die man nicht leihen/sizen kann. |
| On-Chain-Flow-Signale (Netflow, Mint/Burn, Whales) | Published IC winzig (~0,2 % Tagesrendite pro 1σ, 1–2h-Horizont) — unter 0,32 % Round-Trip nicht handelbar; saubere Realtime-Daten $30–800/Mon > plausibler Edge. |
| Cross-Exchange / Triangular Arb | Tot für langsames Retail: Spreads schließen < 4 s gegen Bots; Withdrawal-Fees $5–25 + 15–40 min Latenz. |
| Dated-Futures Cash-and-Carry | Auf ~3 % annualisiert komprimiert, unter der Nichtstun-Hürde nach Fees. |
| stETH/ETH & Wrapped-Asset-Basis | Peg hält inzwischen ~0,2 %; Redemption = Tage ETH-Exposure; Gas + 0,32 % Hedge fressen die Dislokation. Zahlt nur in seltenen Stress-Events. |
| Airdrop-Farming | Nicht systematisch (Lotterie); Multi-Wallet durch Sybil-Detection entwertet. |
| Liquidations-Bots | Braucht Gas-kompetitive Infra + Mempool-Zugang; Retail wird von MEV-Searchern überboten. Retail-Version = Option 1. |

## Bottom Line

**Bei 500–3000 USDT gibt es keine systematische *Trading*-Strategie, die zuverlässig
die Kosten schlägt und Stablecoin-Lending übertrifft.** Jeder Preis-Prognose- und
Mikrostruktur-Edge stirbt am 0,3 %+ Retail-Round-Trip und an schnellerem Kapital.
Die Mechanismen, die den Random-Walk-Test bestehen — Funding-Carry,
Liquidations-Flow, Varianzprämie, LP-Fees, Lending-Raten — sind alle **vertraglich
gezahlte Risikoprämien**, und bei dieser Kontogröße ist der Dollar-Payout
($10–130/Jahr) kleiner als das eingegangene Tail-Risiko und weit kleiner als die
Zeit des Operators.

**Rationaler Umgang mit den 500 USDT: nicht aktiv traden.** In einer diversifizierten
Stablecoin-Lending-Position halten (~4–9 %, ~$20–45/Jahr), die Bot-Arbeit als
Lernprojekt behandeln, nicht als Einkommensquelle.

**Wenn eine aktive Sache: Option 1 (HLP-artiger Vault-Deposit)** — richtig
verstanden als Einzahlung in den Edge eines anderen, mit dem Smart-Contract/Venue-
Risiko als der gesamten These.

## Realistische Profit/Risiko-Spanne — Option 1 (bester aktiver Fall)

| Metrik | Schätzung |
|--------|-----------|
| Annualisierte Rendite | ~15–25 % im Normaljahr; −10 bis −20 % im schlechten Quartal; bisher über 12-Monats-Fenster netto positiv |
| Max Drawdown | 10–15 % bei normalen Vol-Spikes; **realistischer Tail: −100 %** bei Bridge/Contract-Exploit oder Venue-Insolvenz |
| Kapital | funktioniert in jeder Größe; Exposure auf Totalverlust-verträglichen Betrag deckeln — hier ≤ $200 von 500 |
| P(Edge echt, nicht overfit) | ~70–80 %, dass der Brutto-Liquidations-Flow-Edge strukturell echt ist und bleibt (kommt aus erzwungenem, preis-insensitivem Flow). ABER ~20–35 % Chance, dass in einem 2–3-Jahres-Fenster ein Exploit/De-Peg/Shutdown die Position auslöscht — das dominiert den Erwartungswert. |

**Erwartungswert auf $200 Deposit:** grob **+$30–50/Jahr** in den ~75 % der Jahre
ohne Bruch, gegen eine ~1-zu-4-Mehrjahres-Chance, den Großteil der $200 zu
verlieren. Nicht überzeugend genug, um die „einfach Stablecoins lenden"-Empfehlung
zu überstimmen für jemanden, dessen echter Constraint das Kapital ist.

## Was es bräuchte, um mit Trading nennenswert zu verdienen

Für z.B. $500–2000/Jahr (≈ spürbar): **10–50× das Kapital** ($10k–50k) UND eine der
Risikoprämien-Strategien sauber ausgeführt UND Akzeptanz der Tail-Risiken — und
selbst dann sind es ~10–20 %/Jahr, kein Geldregen. Größere Positionsgröße / Hebel
auf die bestehende (negative) Grid-Erwartung ändert daran nichts: Erwartungswert ist
linear, PF < 1 bleibt PF < 1 bei jeder Skalierung, und Hebel fügt Liquidationsrisiko
hinzu (siehe `feedback_risk_escalation_pushback`).
