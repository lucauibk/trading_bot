"""
Trend-Breakout + Trailing-Stop Directional-Research

Testet den bisher UNGETESTETEN Schlüssel-Hebel: Trailing-Stop statt fixem TP.
Alle 3 gescheiterten Directional-Ansätze (ML, Funding, MTF) hatten FIXE Exits —
aber der Trend-Edge kommt vom Laufenlassen der Gewinner (klassisches Turtle/Chandelier).

Strategie (long-only, wie der Bot):
- Entry: Donchian-Breakout — close > höchstes High der letzten ENTRY_LOOKBACK Kerzen
- Regime-Filter (optional): nur wenn close > EMA200 (kein Kauf im Bärenmarkt)
- Exit: Chandelier-Trailing-Stop — highest_high_seit_entry − CHANDELIER_ATR × ATR
- Metrik: Expectancy (Profit-Factor, Ø-R, Total-PnL) — NICHT Win-Rate
  (Trend-Following hat niedrige WR ~35-45% aber große Gewinner)

Nutzung:
  python3 scripts/trend_breakout_backtest.py
"""
import argparse, logging, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GRIDBOT_BACKTEST", "1")

import numpy as np
import pandas as pd
import ta as ta_lib
from backtest.data import load_ohlcv

logger = logging.getLogger("trend_bt")

FEE = 0.0016
LEV = 3.0
SYMBOLS = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]


def _atr(df, period=14):
    return ta_lib.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)


def backtest(df: pd.DataFrame, entry_lb: int, chand_atr: float, use_regime: bool) -> list:
    """Ein Durchlauf, gibt Trade-Liste zurück."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = _atr(df).values
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values

    trades = []
    i = max(entry_lb, 200)
    n = len(df)
    while i < n - 1:
        # Entry: Breakout über Donchian-Oberband der letzten entry_lb Kerzen (ohne aktuelle)
        donchian_high = high[i - entry_lb:i].max()
        if not (close[i] > donchian_high):
            i += 1; continue
        if use_regime and not (close[i] > ema200[i]):
            i += 1; continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            i += 1; continue

        entry = close[i]
        entry_atr = atr[i]
        risk = chand_atr * entry_atr          # initiales Risiko (Chandelier-Abstand)
        highest = high[i]
        stop = highest - chand_atr * entry_atr
        exit_p = None
        j = i + 1
        while j < n:
            highest = max(highest, high[j])
            # Trailing-Stop nachziehen (nie senken)
            new_stop = highest - chand_atr * atr[j] if not np.isnan(atr[j]) else stop
            stop = max(stop, new_stop)
            if low[j] <= stop:
                exit_p = stop
                break
            j += 1
        if exit_p is None:
            exit_p = close[n - 1]; j = n - 1

        pnl = (exit_p - entry) / entry * LEV - 2 * FEE * LEV
        r = (exit_p - entry) / risk if risk > 0 else 0.0
        trades.append({"pnl": pnl, "r": r, "held": j - i, "win": exit_p > entry})
        i = j + 1   # kein Überlapp
    return trades


def summarize(trades):
    if not trades:
        return None
    n = len(trades)
    w = sum(1 for t in trades if t["win"])
    gw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = sum(-t["pnl"] for t in trades if t["pnl"] < 0)
    pf = gw / gl if gl > 1e-9 else float("inf")
    return {
        "n": n, "wr": w / n, "pf": pf,
        "total": sum(t["pnl"] for t in trades) * 100,
        "avg_r": sum(t["r"] for t in trades) / n,
        "avg_hold": sum(t["held"] for t in trades) / n,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=400)
    args = ap.parse_args()

    # Sweep-Achsen: Entry-Lookback × Chandelier-ATR × Regime-Filter
    configs = [
        (20, 3.0, True), (20, 3.0, False),
        (20, 4.0, True),
        (55, 3.0, True), (55, 4.0, True),
        (10, 2.5, True),
    ]

    print(f"\nLade {args.days}d OHLCV für {len(SYMBOLS)} Symbole…")
    data = {}
    for s in SYMBOLS:
        try:
            data[s] = load_ohlcv(s, "1h", args.days)
        except Exception as e:
            logger.warning("%s laden fehlgeschlagen: %s", s, e)

    print(f"\n{'='*82}")
    print("  TREND-BREAKOUT + TRAILING-STOP — Expectancy-Suche (aggregiert über Symbole)")
    print(f"{'='*82}")
    print(f"  {'Config (LB/ATR/Regime)':<26} {'N':>4} {'WR':>6} {'PF':>6} {'Ø-R':>6} {'Ø-Hold':>7} {'Total%':>9}")
    print(f"  {'-'*26} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*9}")

    best = None
    for lb, ca, reg in configs:
        alltr = []
        for s, df in data.items():
            alltr += backtest(df, lb, ca, reg)
        r = summarize(alltr)
        if not r:
            continue
        tag = f"LB={lb} ATR={ca} reg={'J' if reg else 'N'}"
        flag = ""
        if r["pf"] > 1.3 and r["n"] >= 30 and r["total"] > 0:
            flag = "  ✅"
            if best is None or r["pf"] > best[1]["pf"]:
                best = (tag, r)
        print(f"  {tag:<26} {r['n']:>4} {r['wr']:>5.0%} {r['pf']:>6.2f} {r['avg_r']:>+6.2f} {r['avg_hold']:>6.0f}h {r['total']:>+8.1f}{flag}")

    print(f"\n{'='*82}")
    if best:
        print(f"  ✅ EDGE GEFUNDEN: {best[0]}  |  PF={best[1]['pf']:.2f}  N={best[1]['n']}  Total={best[1]['total']:+.1f}%")
        print("     → Kandidat für Dashboard-Toggle-Directional (needs Paper-Validierung)")
    else:
        print("  ❌ KEIN Config besteht Gate (PF>1.3 & N≥30 & positiv)")
        print("     → Auch Trailing-Stop-Trend-Following zeigt keinen validierten Edge")
    print()


if __name__ == "__main__":
    main()
