"""
Cointegration-Test für Relative-Value/Pairs-Hypothese (research/00-hypothesen.md,
Ergänzung 2026-07-22: Nutzer-Idee "Relative-Value / Cointegration").

Für jedes der 10 Coin-Paare aus den 5 gehandelten Symbolen:
  1. OLS-Hedge-Ratio: log(A) ~ log(B) + const
  2. Residual-Spread = log(A) - hedge_ratio * log(B)
  3. Augmented-Dickey-Fuller-Test auf den Spread (H0: nicht stationär/Random Walk)

Nur Dev-Set (<= 2026-07-22, siehe research/00-hypothesen.md Dev/Vault-Split) — das
Vault-Fenster (ab 2026-07-23) wird hier nicht angerührt, Bestätigung folgt separat,
sobald genug Vault-Daten vorliegen.

Kill-Kriterium (User, 2026-07-22): Spread muss im Dev-Set nachweisbar stationär sein
(ADF p<0.05) UND das im Vault-Fenster bestätigen, nicht nur einmalig.

Usage: python3 scripts/cointegration_test.py
"""

import datetime
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.data import load_ohlcv  # noqa: E402

SYMBOLS = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]
DEV_SET_CUTOFF = datetime.datetime(2026, 7, 22, tzinfo=datetime.timezone.utc)


def main():
    closes = {}
    for sym in SYMBOLS:
        df = load_ohlcv(sym, "1h", 180)
        df = df[df.index <= DEV_SET_CUTOFF]  # Dev-Set-Cap, Vault-Fenster unberührt
        closes[sym] = df["close"]
        print(f"{sym}: {len(df)} candles, {df.index[0]} -> {df.index[-1]}")

    print("\n" + "=" * 90)
    print(f"{'Paar':<20} {'Hedge-Ratio':>12} {'ADF-Stat':>10} {'p-Wert':>10} "
          f"{'Krit. 5%':>10} {'Stationär?':>12}")
    print("=" * 90)

    results = []
    for a, b in itertools.combinations(SYMBOLS, 2):
        idx = closes[a].index.intersection(closes[b].index)
        log_a = np.log(closes[a].loc[idx])
        log_b = np.log(closes[b].loc[idx])

        X = add_constant(log_b)
        model = OLS(log_a, X).fit()
        hedge_ratio = model.params.iloc[1]
        spread = log_a - hedge_ratio * log_b

        adf_stat, p_value, _, _, crit, _ = adfuller(spread, autolag="AIC")
        crit_5pct = crit["5%"]
        stationary = "JA" if p_value < 0.05 else "nein"

        pair_label = f"{a.split('/')[0]}-{b.split('/')[0]}"
        print(f"{pair_label:<20} {hedge_ratio:>12.4f} {adf_stat:>10.3f} "
              f"{p_value:>10.4f} {crit_5pct:>10.3f} {stationary:>12}")
        results.append({
            "pair": pair_label, "hedge_ratio": hedge_ratio,
            "adf_stat": adf_stat, "p_value": p_value,
            "crit_5pct": crit_5pct, "n": len(spread),
            "stationary_dev_set": p_value < 0.05,
        })

    print("=" * 90)
    n_stationary = sum(1 for r in results if r["stationary_dev_set"])
    print(f"\n{n_stationary}/10 Paare im Dev-Set stationär (ADF p<0.05).")
    if n_stationary == 0:
        print("Kill-Kriterium NICHT erfüllt (bereits im Dev-Set) — kein Paar-Kandidat.")

    return results


if __name__ == "__main__":
    main()
