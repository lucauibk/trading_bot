"""
Funding-Carry-Test (research/00-hypothesen.md Ergänzung 2026-07-22, Nutzer-Idee 1).

Delta-neutral: long Spot / short Perp. Sagt KEINE Preisrichtung vorher (anders als der
bereits toten Funding-IC-Test aus PROGRESS.md) — kassiert nur die Funding-Rate, solange
die Position gehedgt ist. Preis-P&L der Spot- und Perp-Seite heben sich (näherungsweise)
auf; das einzige P&L ist die Funding-Zahlung.

Kill-Kriterium (User, 2026-07-22): Funding muss Round-Trip-Fees + Borrow-Kosten
strukturell übersteigen.

Methode: historische Funding-Rate-Historie von Binance-Perps (ccxt
fetch_funding_rate_history, alle 8h) für die 5 gehandelten Coins, 180 Tage, gekappt auf
das Dev-Set (<= 2026-07-22, research/00-hypothesen.md Dev/Vault-Split). Zwei Szenarien:
  (a) "always_on": einmal hedgen, durchhalten, Netto-Funding über die volle Periode
      (positiv UND negativ) gegen EINEN Round-Trip verrechnen (Standard-Praxis:
      nicht bei jedem Vorzeichenwechsel aus-/einsteigen wegen Fee-Drag).
  (b) "positive_only": nur in Perioden mit Funding > 0 gehedgt sein (User-Formulierung
      wörtlich) — mit Fee-Annahme pro Ein-/Ausstieg, um Flip-Flop-Kosten sichtbar
      zu machen.

Fee-Annahme: Spot-Maker 0.16% (KRAKEN_FEE) + Perp-Maker ~0.02% (typischer Wert,
Binance/Kraken-Futures) je Order, also Round-Trip (Open Spot+Perp, Close Spot+Perp)
≈ 2 × (0.16% + 0.02%) = 0.36%. Konservativ als fixe Konstante, nicht symbolabhängig.

Usage: python3 scripts/funding_carry_test.py
"""

import datetime
import sys
from pathlib import Path

import ccxt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SYMBOLS_PERP = {
    "SOL/USD": "SOL/USDT:USDT",
    "ETH/USD": "ETH/USDT:USDT",
    "AVAX/USD": "AVAX/USDT:USDT",
    "LINK/USD": "LINK/USDT:USDT",
    "XRP/USD": "XRP/USDT:USDT",
}
DEV_SET_CUTOFF_MS = int(datetime.datetime(2026, 7, 22, tzinfo=datetime.timezone.utc)
                         .timestamp() * 1000)
ROUND_TRIP_FEE_PCT = 2 * (0.0016 + 0.0002)  # Spot-Maker + Perp-Maker, open+close


def fetch_all_funding(perp_symbol: str, days: int = 180) -> list:
    ex = ccxt.binance()
    ex.load_markets()
    since = ex.milliseconds() - days * 24 * 3600 * 1000
    all_rows = []
    while True:
        batch = ex.fetch_funding_rate_history(perp_symbol, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < 1000:
            break
        since = batch[-1]["timestamp"] + 1
    return [r for r in all_rows if r["timestamp"] <= DEV_SET_CUTOFF_MS]


def main():
    print(f"{'Symbol':<10} {'N':>5} {'Netto-Funding':>14} {'Round-Trip-Fee':>16} "
          f"{'Netto ggü. Fee (always_on)':>28}")
    print("=" * 78)

    results = []
    for spot_sym, perp_sym in SYMBOLS_PERP.items():
        rows = fetch_all_funding(perp_sym)
        rates = [r["fundingRate"] for r in rows]
        n = len(rates)
        net_funding_pct = sum(rates) * 100  # kumuliert, in %

        positive_periods = [r for r in rates if r > 0]
        negative_periods = [r for r in rates if r <= 0]
        pos_only_net_pct = sum(positive_periods) * 100

        always_on_net = net_funding_pct - ROUND_TRIP_FEE_PCT * 100
        print(f"{spot_sym:<10} {n:>5} {net_funding_pct:>13.3f}% "
              f"{ROUND_TRIP_FEE_PCT*100:>15.3f}% {always_on_net:>27.3f}%")

        results.append({
            "symbol": spot_sym, "n_periods": n,
            "net_funding_pct": net_funding_pct,
            "n_positive": len(positive_periods), "n_negative": len(negative_periods),
            "positive_only_gross_pct": pos_only_net_pct,
            "always_on_net_vs_fee_pct": always_on_net,
        })

    print("=" * 78)
    print(f"\nRound-Trip-Fee-Annahme: {ROUND_TRIP_FEE_PCT*100:.3f}% "
          f"(Spot 0.16% + Perp ~0.02%, je Open+Close)")
    print("\n'always_on': einmal hedgen, 180 Tage durchhalten, EIN Round-Trip.")
    n_pass = sum(1 for r in results if r["always_on_net_vs_fee_pct"] > 0)
    print(f"{n_pass}/5 Symbole: Netto-Funding übersteigt einen Round-Trip "
          f"strukturell (always_on).")

    print("\nPositive-Perioden vs. negative Perioden (roh, für 'positive_only'-"
          "Variante):")
    for r in results:
        print(f"  {r['symbol']}: {r['n_positive']} positiv / {r['n_negative']} "
              f"negativ von {r['n_periods']} — brutto positive Perioden: "
              f"{r['positive_only_gross_pct']:.3f}%")

    return results


if __name__ == "__main__":
    main()
