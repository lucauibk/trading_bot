"""
research/09 — Funding-Carry mit echtem Budget & Mehrjahres-Historie.

Delta-neutral (long Spot / short Perp): kassiert die Funding-Rate, solange gehedgt.
Erweiterung von funding_carry_test.py: 2–3 Jahre Historie, Majors + liquide Alts,
Low-Fee-Venue-Kosten (statt Kraken), realistische Rebalance-/Exit-Regel.

Bei $10–20k Kapital sind Fixkosten (Gas, Mindestgrößen) vernachlässigbar → die
prozentuale Carry ist die Frage. Kein Preis-Directional (Spot- und Perp-P&L heben
sich näherungsweise auf), einziges P&L = Funding + Fees.

  python3 scripts/funding_carry_scale.py --days 900
"""
import argparse, datetime as dt, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import ccxt

# Perp-Symbole (Binance USDT-M) — Majors zuerst, das ist der eigentliche Carry-Play
PERPS = {
    "BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT",
    "BNB": "BNB/USDT:USDT", "XRP": "XRP/USDT:USDT", "DOGE": "DOGE/USDT:USDT",
    "AVAX": "AVAX/USDT:USDT", "LINK": "LINK/USDT:USDT",
}
# Low-Fee-Venue-Annahme: Perp-Maker ~0,02 % + Spot-Maker ~0,05 %, open+close beide Legs
RT_FEE = 2 * (0.0002 + 0.0005)   # = 0,14 % Round-Trip
FUND_PER_DAY = 3                 # 8h-Perioden


def fetch_funding(perp, days):
    ex = ccxt.binance()
    ex.load_markets()
    since = ex.milliseconds() - days * 86400_000
    rows, cur = [], since
    while True:
        b = ex.fetch_funding_rate_history(perp, since=cur, limit=1000)
        if not b:
            break
        rows += b
        cur = b[-1]["timestamp"] + 1
        if len(b) < 1000:
            break
    return [(r["timestamp"], float(r["fundingRate"])) for r in rows]


def analyse(rates):
    """rates: Liste (ts, fundingRate) je 8h. Always-on: einmal hedgen, durchhalten."""
    if len(rates) < 30:
        return None
    fr = np.array([r for _, r in rates])
    n = len(fr)
    days = n / FUND_PER_DAY
    gross = fr.sum()                       # kumulatives Netto-Funding (Fraktion)
    net = gross - RT_FEE                   # einmal Round-Trip
    ann = (1 + net) ** (365 / days) - 1 if days > 0 else 0.0
    neg_share = (fr < 0).mean()
    # schlechtestes 30-Tage-Kumulativ-Funding (rollend, 90 Perioden)
    w = 30 * FUND_PER_DAY
    if n > w:
        roll = np.convolve(fr, np.ones(w), "valid")
        worst30 = roll.min()
    else:
        worst30 = gross
    # Equity-Kurve (nur Funding, ohne Fee) für Max-DD
    eq = np.cumprod(1 + fr)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return dict(days=round(days), gross=gross, net=net, ann=ann,
               neg_share=neg_share, worst30=worst30, mdd=mdd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=900)
    args = ap.parse_args()

    print(f"\nFunding-Carry (always-on hedge), {args.days}d, Low-Fee-Venue "
          f"(RT-Fee {RT_FEE*100:.2f}%)\n")
    hdr = f"{'Sym':<5} {'Tage':>5} {'brutto':>9} {'netto':>9} {'ann.':>9} {'neg%':>6} {'worst30d':>10} {'maxDD':>8}"
    print(hdr); print("-" * len(hdr))
    anns, worsts = [], []
    for sym, perp in PERPS.items():
        try:
            r = analyse(fetch_funding(perp, args.days))
        except Exception as e:
            print(f"{sym:<5} fail: {e}"); continue
        if not r:
            print(f"{sym:<5} zu wenig Daten"); continue
        print(f"{sym:<5} {r['days']:>5} {r['gross']*100:>+8.2f}% {r['net']*100:>+8.2f}% "
              f"{r['ann']*100:>+8.2f}% {r['neg_share']*100:>5.0f}% "
              f"{r['worst30']*100:>+9.2f}% {r['mdd']*100:>+7.1f}%")
        anns.append(r["ann"]); worsts.append(r["worst30"])
    if anns:
        print("-" * len(hdr))
        print(f"{'PORT':<5} {'':<5} {'':<9} {'':<9} {np.mean(anns)*100:>+8.2f}% "
              f"(gleichgewichtet)   worst30d Ø {np.mean(worsts)*100:+.2f}%")
        print(f"\nUrteil: ein 8–15%-Ziel bei $10–20k = $800–3000/Jahr — "
              f"Portfolio-Carry hier {np.mean(anns)*100:+.1f}%/Jahr, "
              f"{'ÜBER' if np.mean(anns) > 0.08 else 'UNTER'} der 8%-Schwelle.")


if __name__ == "__main__":
    main()
