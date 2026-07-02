"""
MTF-Directional OOS-Backtest — misst die Expectancy des MTF-Trend-Following-Pfads.

Der MTF-Trade-Code existiert bereits (strategies/grid.py:_check_mtf_entry), ist aber
schlafend + nie validiert. Dieser Backtest speist den echten MTFAnalyzer mit
Point-in-Time-Daten (kein Look-Ahead) und simuliert die Trades.

Metrik = EXPECTANCY (Profit-Factor, Ø-R), nicht nur Win-Rate — Trend-Following hat
niedrige WR aber große Gewinner.

Nutzung:
  python3 scripts/mtf_backtest.py --symbols SOL/USD,ETH/USD --oos-days 60
"""
import argparse, logging, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GRIDBOT_BACKTEST", "1")
# MTF-Analyzer loggt viel — auf WARNING drosseln
logging.getLogger("price_predictor.mtf_analyzer").setLevel(logging.WARNING)

import pandas as pd
from backtest.data import load_ohlcv
from price_predictor.mtf_analyzer import MTFAnalyzer

logger = logging.getLogger("mtf_backtest")

MAX_HOLD_H = 96     # Timeout 4 Tage
FEE = 0.0016
LEV = 3.0

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


class PITFeeder:
    """Point-in-Time OHLCV-Feeder: gibt nur Daten bis zum aktuellen Sim-Index zurück."""
    def __init__(self, df1h: pd.DataFrame):
        self.df1h = df1h
        self.i = 0
        self._cache_4h = None
        self._cache_1d = None

    def set_index(self, i: int):
        self.i = i

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        upto = self.df1h.iloc[: self.i + 1]
        if timeframe == "1h" or timeframe == "5m":   # 5m mit 1h approximiert
            # min. 60 Bars, sonst crasht compute_indicators (ADX-14) im Entry-Trigger
            return upto.tail(max(limit, 60))
        if timeframe == "4h":
            r = upto.resample("4h").agg(_AGG).dropna()
            return r.tail(limit)
        if timeframe == "1d":
            r = upto.resample("1D").agg(_AGG).dropna()
            return r.tail(limit)
        return upto.tail(limit)


def run(symbol: str, oos_days: int) -> dict:
    df = load_ohlcv(symbol, "1h", oos_days + 340)   # +340d Vorlauf: 1d-EMA200 braucht ≥300 Tages-Bars
    if len(df) < 24 * (oos_days + 60):
        logger.warning("%s: zu wenig Daten", symbol)
    feeder = PITFeeder(df)
    an = MTFAnalyzer(symbol, feeder.fetch)

    start = len(df) - oos_days * 24
    trades = []
    i = max(start, 200)
    while i < len(df) - 1:
        feeder.set_index(i)
        try:
            bias = an.refresh_bias()
            setup = an.find_retest_setup(bias)
        except Exception:
            i += 1; continue
        if not setup or setup["direction"] != "long":
            i += 1; continue
        price = float(df["close"].iloc[i])
        try:
            trig = an.check_entry_trigger(setup, price)
        except Exception:
            trig = None
        if not trig:
            i += 1; continue

        # Entry
        entry = float(trig["trigger_price"])
        atr = setup.get("atr") or entry * 0.02
        tp = setup["target"]
        sl = setup["zone_low"] - atr
        if sl >= entry or tp <= entry:   # ungültige Geometrie
            i += 1; continue
        risk = entry - sl

        # Exit simulieren
        outcome, exit_p, held = "timeout", None, 0
        for j in range(i + 1, min(i + 1 + MAX_HOLD_H, len(df))):
            h, l = float(df["high"].iloc[j]), float(df["low"].iloc[j])
            if l <= sl:
                outcome, exit_p, held = "loss", sl, j - i; break
            if h >= tp:
                outcome, exit_p, held = "win", tp, j - i; break
        if exit_p is None:
            exit_p = float(df["close"].iloc[min(i + MAX_HOLD_H, len(df) - 1)]); held = MAX_HOLD_H

        pnl_pct = (exit_p - entry) / entry * LEV - 2 * FEE * LEV
        r_mult = (exit_p - entry) / risk
        trades.append({"outcome": outcome, "pnl_pct": pnl_pct, "r": r_mult, "held": held})
        i += held + 1   # kein Überlapp

    if not trades:
        return {"symbol": symbol, "n": 0}
    n = len(trades)
    wins = [t for t in trades if t["outcome"] == "win"]
    gross_w = sum(t["pnl_pct"] for t in trades if t["pnl_pct"] > 0)
    gross_l = sum(-t["pnl_pct"] for t in trades if t["pnl_pct"] < 0)
    pf = gross_w / gross_l if gross_l > 1e-9 else float("inf")
    return {
        "symbol": symbol, "n": n, "wr": len(wins) / n,
        "pf": pf, "total_pnl_pct": sum(t["pnl_pct"] for t in trades) * 100,
        "avg_r": sum(t["r"] for t in trades) / n,
        "avg_hold": sum(t["held"] for t in trades) / n,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SOL/USD,ETH/USD")
    ap.add_argument("--oos-days", type=int, default=60)
    args = ap.parse_args()

    results = []
    for sym in args.symbols.split(","):
        logger.info("Backtest MTF %s (%dd OOS)…", sym, args.oos_days)
        try:
            r = run(sym.strip(), args.oos_days)
            results.append(r)
        except Exception as e:
            logger.warning("%s fehlgeschlagen: %s", sym, e)

    print(f"\n{'='*72}\n  MTF-DIRECTIONAL EXPECTANCY (OOS {args.oos_days}d)\n{'='*72}")
    print(f"  {'Symbol':<10} {'N':>4} {'WinRate':>8} {'PF':>6} {'Ø-R':>6} {'Total%':>9}")
    print(f"  {'-'*10} {'-'*4} {'-'*8} {'-'*6} {'-'*6} {'-'*9}")
    allt = []
    for r in results:
        if r.get("n", 0) == 0:
            print(f"  {r['symbol']:<10}    0   (keine Setups)"); continue
        allt.append(r)
        print(f"  {r['symbol']:<10} {r['n']:>4} {r['wr']:>7.1%} {r['pf']:>6.2f} {r['avg_r']:>6.2f} {r['total_pnl_pct']:>8.1f}")
    if allt:
        N = sum(r["n"] for r in allt)
        tot = sum(r["total_pnl_pct"] for r in allt)
        gpf = sum(r["pf"] for r in allt) / len(allt)
        print(f"\n  Σ Trades={N}  Ø-PF={gpf:.2f}  Total-PnL={tot:+.1f}%")
        print()
        if gpf > 1.3 and tot > 0:
            print("  ✅ EDGE — Profit-Factor > 1.3 & positiv → MTF aktivieren (mtf_auto_execute)")
        elif tot > 0:
            print("  ⚠️  GRENZWERTIG — positiv aber PF < 1.3; mehr Daten/Symbole prüfen")
        else:
            print("  ❌ KEIN EDGE — MTF bleibt schlafend, Directional endgültig aus")
    print()


if __name__ == "__main__":
    main()
