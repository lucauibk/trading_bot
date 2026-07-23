# Funding-Carry (long Spot / short Perp): Befund (Dev-Set, n=1 Fenster)

Bezug: Nutzer-Ergänzung 2026-07-22, Hypothese 1. Delta-neutral, sagt keine
Preisrichtung vorher — kassiert nur die Funding-Rate, solange gehedgt.

**Kill-Kriterium (User):** Funding muss Round-Trip-Fees + Borrow-Kosten strukturell
übersteigen.

## Methode

`scripts/funding_carry_test.py`: historische Funding-Rate-Historie (Binance-Perps via
ccxt, alle 8h, 539 Perioden ≈ 180 Tage, gekappt auf Dev-Set ≤ 2026-07-22) für die 5
gehandelten Coins. "Always-on"-Szenario: einmal hedgen, durchhalten, Netto-Funding
(positiv UND negativ) gegen EINEN Round-Trip (0,36 % Annahme: Spot-Maker 0,16 % +
Perp-Maker ~0,02 %, je Open+Close) verrechnen.

## Ergebnis: Universum-negativ, ein Ausreißer nicht verwertbar

| Symbol | Netto-Funding (180d) | Netto ggü. Round-Trip-Fee |
|--------|------------------------|-----------------------------|
| SOL/USD | −1,776 % | −2,136 % |
| ETH/USD | +0,145 % | −0,215 % |
| AVAX/USD | −0,027 % | −0,387 % |
| LINK/USD | +1,860 % | **+1,500 %** |
| XRP/USD | −1,104 % | −1,464 % |
| **Gleichgewichtetes Portfolio (alle 5)** | | **−0,54 %** |

**Nur 1/5 Coins (LINK) übersteigt die Gebührenschwelle. Auf das tatsächlich gehandelte
Universum gleichgewichtet angewendet, ist das Ergebnis −0,54 % — negativ.**

## Warum LINK nicht als Fund zählt (wichtige Selbstkorrektur)

LINK ist **das Maximum von fünf verrauschten Ziehungen, kein Befund** — exakt das
Mehrfachvergleichs-Problem, vor dem die Pre-Registration-Disziplin dieses gesamten
Programms schützen soll (dieselbe Form wie „positiv in 1 von 5 Fenstern" beim
Regime-Gate).

- **Nicht vorab wählbar:** „Long Spot/Short Perp auf LINK" funktioniert nur, wenn man
  vorher gewusst hätte, dass LINK der High-Funding-Coin wird. Nichts im Dev-Set sagt
  das im Voraus. Auf das tatsächlich gehandelte Universum angewendet, verliert die
  Strategie.
- **Nicht jetzt validierbar:** Ob das echter, anhaltender Carry auf LINK ist oder LINK
  in diesem Fenster zufällig 76 % positive Funding-Perioden hatte, lässt sich nur durch
  **zukünftige Persistenz** unterscheiden — eine Vault-/Forward-Frage, im Dev-Fenster
  strukturell unbeantwortbar.
- **Eigenes Kill-Kriterium nicht erfüllt:** n=1 Fenster, in-sample, keine
  Vault-Bestätigung; Kriterium „Bootstrap-CI schließt Null aus" ist bei einer
  einzelnen Realisierung nicht berechenbar.

**Der „positive_only"-Toggle-Test (nur in positiven Perioden gehedgt) wurde bewusst
NICHT gerechnet, um LINK zu retten** — bei Dutzenden Vorzeichenwechseln pro Coin und
0,36 % Round-Trip-Fee pro Wechsel würden die Fees die brutto 2,5 % positiven Perioden
voraussichtlich auffressen. Das wäre erneut Post-hoc-Rosinenpicken.

## Scope-Hinweis (wichtig, nicht verschweigen)

Funding-Carry ist **nicht der Grid-Bot** — es bräuchte ein Futures-/Short-Perp-Konto,
das der Bot nicht hat (`execution/kraken.py` ist spot-only, `LIVE_PARITY_OK=False`),
eine delta-neutrale Positionsverwaltung und eine andere Execution-Engine. Der
ursprüngliche Grid-Bot bleibt tot — dies ist eine Erweiterung der Fragestellung von
„diesen Bot profitabel machen" zu „irgendeine profitable Krypto-Strategie finden".
Selbst der beste Fall (LINK, ~3 %/Jahr brutto) ist dünne Belohnung für signifikantes
neues Infrastruktur-Risiko (Liquidation, Basis-Blowout, Exchange-Gegenparteirisiko).

## Fazit

**Universum-negativ.** LINK ist ein nicht validierbarer Einzelfall-Ausreißer, keine
belastbare Erkenntnis. Kein robuster, handelbarer, validierbarer Edge gefunden — und
selbst wenn: bräuchte neue Infrastruktur, die aktuell nicht existiert.

**Einzig ehrlich möglicher nächster Schritt, falls gewünscht:** ein **vorab
festgelegter Forward-Test** der Funding-Vorzeichen-Persistenz (live/paper committen
und abwarten), nicht weiteres In-Sample-Schneiden — das wäre ein neues, eigenes
Experiment mit eigener Pre-Registration, keine Fortsetzung dieses Befunds.
