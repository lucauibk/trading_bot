"""
Funding Information-Coefficient Check (billigster Discriminator für Schritt 2)

Bevor ein separates Directional-Modell gebaut wird: testet ob das Funding-Signal
(funding_z, kausal berechnet) überhaupt Information über die Forward-48h-Rendite trägt.

Logik (Advisor-Empfehlung):
- Wenn IC ≈ 0 → kein LightGBM kann daraus Edge erzeugen → Schritt 2 wird 36% nicht klären → STOP
- Wenn |IC| ≳ 0.03–0.05 (signifikant) → Funding trägt Signal → Directional-Modell bauen lohnt

Kausalität strikt eingehalten:
- 7d-z-Score = trailing rolling (Kerze t nutzt nur Funding ≤ t)
- 8h→1h forward-fill: jede 1h-Kerze bekommt die letzte GEPOSTETE Funding-Rate

Nutzung:
  python3 scripts/funding_ic_check.py
  python3 scripts/funding_ic_check.py --symbol SOL/USD --days 720
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GRIDBOT_BACKTEST", "1")

import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from backtest.data import load_ohlcv
from data_fetcher import fetch_funding_history, perp_symbol_for

logger = logging.getLogger("funding_ic")

FWD_H          = 48    # Forward-Horizont (Stunden) = Directional-Haltedauer
Z_WINDOW_8H    = 21    # 7 Tage × 3 Funding-Punkte/Tag (8h-Intervall)
SYMBOLS        = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]


def build_causal_funding_features(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    """Baut kausale funding_rate + funding_z7d auf dem stündlichen OHLCV-Index.

    - forward-fill: jede 1h-Kerze erbt die letzte funding-Rate deren timestamp ≤ Kerzen-Zeit
    - z-Score: trailing rolling über 21 Funding-Punkte (7d), SHIFT damit nur Vergangenheit
    """
    f = funding.copy().sort_index()
    # Kausaler rollierender z-Score auf der 8h-Funding-Serie selbst
    roll_mean = f["rate"].rolling(Z_WINDOW_8H, min_periods=5).mean()
    roll_std  = f["rate"].rolling(Z_WINDOW_8H, min_periods=5).std()
    f["z"] = (f["rate"] - roll_mean) / (roll_std + 1e-9)

    # forward-fill auf 1h-Index: reindex + ffill (nur Vergangenheitswerte, da 8h-Punkte
    # immer vor der 1h-Kerze liegen die sie erbt)
    merged = pd.DataFrame(index=ohlcv.index)
    merged["funding_rate"] = f["rate"].reindex(ohlcv.index, method="ffill")
    merged["funding_z"]    = f["z"].reindex(ohlcv.index, method="ffill")
    return merged


def compute_ic(symbol: str, days: int) -> dict:
    logger.info("Lade OHLCV+Funding für %s (%d Tage)…", symbol, days)
    ohlcv = load_ohlcv(symbol, "1h", days)

    since_iso = (
        dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    perp = perp_symbol_for(symbol)
    try:
        funding = fetch_funding_history(perp, since_iso)
    except Exception as e:
        logger.warning("%s: Funding-Fetch fehlgeschlagen: %s", symbol, e)
        return {}
    if funding.empty:
        logger.warning("%s: keine Funding-Daten", symbol)
        return {}
    logger.info("%s: %d OHLCV-Kerzen, %d Funding-Punkte", symbol, len(ohlcv), len(funding))

    feats = build_causal_funding_features(ohlcv, funding)

    # Forward-48h-Rendite
    fwd_ret = ohlcv["close"].shift(-FWD_H) / ohlcv["close"] - 1.0

    df = pd.DataFrame({
        "funding_rate": feats["funding_rate"],
        "funding_z":    feats["funding_z"],
        "fwd_ret":      fwd_ret,
    }).dropna()

    if len(df) < 100:
        logger.warning("%s: zu wenig überlappende Punkte (%d)", symbol, len(df))
        return {}

    ic_rate_s, p_rate_s = spearmanr(df["funding_rate"], df["fwd_ret"])
    ic_z_s,    p_z_s    = spearmanr(df["funding_z"],    df["fwd_ret"])

    # NICHT-ÜBERLAPPENDE Stichprobe: jede 48. Zeile → unabhängige 48h-Fenster.
    # Überlappende 1h-Fenster teilen 47/48 ihres Horizonts → p-Werte massiv aufgebläht.
    # Dies ist die EHRLICHE Signifikanz (effektive N statt Roh-N).
    indep = df.iloc[::FWD_H]
    if len(indep) >= 30:
        ic_z_indep, p_z_indep = spearmanr(indep["funding_z"], indep["fwd_ret"])
    else:
        ic_z_indep, p_z_indep = 0.0, 1.0

    return {
        "symbol":      symbol,
        "n":           len(df),
        "n_indep":     len(indep),
        "ic_rate_spearman": round(float(ic_rate_s), 4),
        "ic_z_spearman":    round(float(ic_z_s), 4),
        "p_z_spearman":     round(float(p_z_s), 4),
        "ic_z_indep":       round(float(ic_z_indep), 4),
        "p_z_indep":        round(float(p_z_indep), 4),
    }


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--days", type=int, default=720)
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    results = []
    for sym in symbols:
        try:
            r = compute_ic(sym, args.days)
            if r:
                results.append(r)
        except Exception as e:
            logger.warning("%s fehlgeschlagen: %s", sym, e)

    if not results:
        print("\n❌ Keine Ergebnisse — Funding-Daten nicht verfügbar")
        return

    print(f"\n{'═' * 78}")
    print(f"  FUNDING INFORMATION COEFFICIENT vs Forward-{FWD_H}h-Rendite")
    print(f"{'═' * 78}")
    print(f"  {'Symbol':<10} {'N_indep':>8} {'IC(z)':>9} {'IC(z)indep':>11} {'p_indep':>9} {'signif?':>9}")
    print(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*11} {'-'*9} {'-'*9}")

    ic_z_vals = []
    for r in results:
        # EHRLICHE Signifikanz: nicht-überlappende Stichprobe
        signif = "JA" if r["p_z_indep"] < 0.05 and abs(r["ic_z_indep"]) >= 0.05 else "nein"
        ic_z_vals.append(r["ic_z_indep"])
        print(f"  {r['symbol']:<10} {r['n_indep']:>8} {r['ic_z_spearman']:>9} "
              f"{r['ic_z_indep']:>11} {r['p_z_indep']:>9} {signif:>9}")

    mean_abs_ic = np.mean([abs(v) for v in ic_z_vals])
    print(f"\n  Ø |IC(z)| (nicht-überlappend, ehrlich) über alle Symbole: {mean_abs_ic:.4f}")
    print(f"{'═' * 78}")

    # Urteil basiert auf EHRLICHER (nicht-überlappender) Signifikanz
    n_signif = sum(1 for r in results
                   if r["p_z_indep"] < 0.05 and abs(r["ic_z_indep"]) >= 0.05)
    print()
    if n_signif >= 3 and mean_abs_ic >= 0.03:
        print("  ✅ FUNDING TRÄGT SIGNAL — Directional-Modell bauen lohnt sich")
        print("     → Schritt 2 vollständig implementieren (train_directional + re-backtest)")
    elif n_signif >= 1:
        print("  ⚠️  SCHWACHES SIGNAL — grenzwertig, Edge unwahrscheinlich aber testbar")
        print("     → Directional-Modell bauen nur wenn Ressourcen da; Erwartung niedrig")
    else:
        print("  ❌ KEIN SIGNAL — Funding trägt keine Information über Forward-48h-Rendite")
        print("     → Kein LightGBM kann daraus Edge erzeugen; Schritt 2 würde 36% nicht klären")
        print("     → Directional bleibt AUS. Fokus vollständig auf Grid-Churn.")
    print()


if __name__ == "__main__":
    main()
