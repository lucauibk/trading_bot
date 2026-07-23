# Relative-Value / Cointegration: Befund (Dev-Set)

Bezug: Nutzer-Ergänzung 2026-07-22 (drei neue Hypothesen jenseits des ursprünglichen
Plans — Funding-Carry, Cointegration-Pairs, echtes Market-Making). Dieser Abschnitt:
Cointegration-Pairs.

**Vorab festgelegtes Kill-Kriterium (User):** Spread muss im Dev-Set nachweisbar
stationär sein (ADF p<0,05) **und** das im Vault-Fenster bestätigen — nicht nur
einmalig.

## Methode

`scripts/cointegration_test.py`: Für alle 10 Paare der 5 gehandelten Coins
(SOL/ETH/AVAX/LINK/XRP) — OLS-Hedge-Ratio auf log-Preisen (`log(A) ~ log(B) + const`),
Augmented-Dickey-Fuller-Test auf den Residual-Spread. 180 Tage @ 1h, gekappt auf das
Dev-Set (≤ 2026-07-22, siehe `research/00-hypothesen.md` Dev/Vault-Split) — Vault
unberührt.

## Ergebnis: 0/10 Paare stationär

| Paar | Hedge-Ratio | ADF-Stat | p-Wert | Krit. 5% |
|------|------------|----------|--------|----------|
| SOL-ETH | 0,82 | −2,49 | 0,117 | −2,86 |
| SOL-AVAX | 0,63 | −2,85 | **0,051** | −2,86 |
| SOL-LINK | 1,10 | −1,97 | 0,299 | −2,86 |
| SOL-XRP | 0,86 | −2,36 | 0,154 | −2,86 |
| ETH-AVAX | 0,73 | −1,78 | 0,392 | −2,86 |
| ETH-LINK | 1,26 | −2,56 | 0,102 | −2,86 |
| ETH-XRP | 0,95 | −1,87 | 0,345 | −2,86 |
| AVAX-LINK | 1,45 | −1,11 | 0,710 | −2,86 |
| AVAX-XRP | 1,22 | −2,78 | **0,061** | −2,86 |
| LINK-XRP | 0,71 | −2,42 | 0,136 | −2,86 |

**Kein einziges Paar erreicht p<0,05.** SOL-AVAX (0,051) und AVAX-XRP (0,061) liegen
nah an der Schwelle, reißen sie aber. Diese zwei nicht als "fast bestanden"
weiterverfolgen — die Schwelle wurde vorab fixiert, genau um Post-hoc-Rosinenpickerei
zu verhindern (dieselbe Disziplin wie beim Rest des Programms).

## Einordnung

Alle 5 gehandelten Altcoins bewegen sich vermutlich überwiegend BTC-getrieben
gemeinsam, ohne stabile paarweise Gleichgewichtsbeziehung untereinander — plausibel,
da es sich um unterschiedliche L1/Infra-Coins ohne engen fundamentalen Bezug
zueinander handelt (anders als z.B. ETH-Staking-Derivate oder Coin/Wrapped-Pärchen,
die typischerweise cointegriert sind).

**Kill-Kriterium bereits im Dev-Set nicht erfüllt.** Vault-Bestätigung erübrigt sich
per User-eigener Kriterien-Definition ("UND das im Vault-Fenster bestätigen" setzt
voraus, dass Dev-Set zuerst besteht). **Diese Hypothese (Relative-Value/Cointegration
auf den 5 gehandelten Coins) ist tot.**

## Offene Erweiterung (nicht automatisch verfolgt)

Getestet wurden nur die 5 aktuell gehandelten Coins. Andere Paare (z.B. mit BTC selbst,
oder mit Coins außerhalb der aktuellen Liste) wurden nicht geprüft — das wäre eine
neue, separate Hypothese mit eigenem Pre-Registration-Bedarf, kein automatischer
Nachtest der hier gescheiterten.
