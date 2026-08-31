"""
Trend-Breakout + Trailing-Stop Directional-Research  (long UND short)

Ursprünglich (long-only) getestet: PROGRESS.md — alle 6 Configs PF 0,46–0,74, tot.
Phase 5 (research/05-payoff-asymmetry.md) erweitert das Skript um die bislang
UNGETESTETE Short-Seite: der Long-only-Grid ist short gamma und verliert genau in
schnellen Alt-Abstürzen — ein Short-Trend-Sleeve könnte diese Verlust-Seite in eine
zweite Profit-Quelle drehen.

Strategie:
- Entry long : close > höchstes High der letzten ENTRY_LOOKBACK Kerzen (Donchian-Breakout)
  Entry short: close < tiefstes Low  der letzten ENTRY_LOOKBACK Kerzen (Donchian-Breakdown)
- Regime-Filter (optional): long nur wenn close > EMA200, short nur wenn close < EMA200
- Exit: Chandelier-Trailing-Stop — long: extreme_high − ATR·k ; short: extreme_low + ATR·k
- Metrik: Per-Trade-Expectancy (Profit-Factor, Ø-R, Total-PnL) + Block-Bootstrap-CI
  (NICHT Win-Rate — Trend-Following hat niedrige WR aber große Gewinner)

Regressions-Check (muss PROGRESS.md-Band 0,46–0,74 reproduzieren):
  python3 scripts/trend_breakout_backtest.py --side long --days 400 --leverage 3.0

Vorregistrierter Short-Lauf (research/05):
  python3 scripts/trend_breakout_backtest.py --side short --as-of 2026-07-22 \
      --days 405 --split-days 300 --leverage 1.0 --carry-daily 0.0
"""
import argparse, datetime, logging, math, os, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GRIDBOT_BACKTEST", "1")

import numpy as np
import pandas as pd
import ta as ta_lib
from backtest.data import load_ohlcv

logger = logging.getLogger("trend_bt")

FEE = 0.0016
LEV = 3.0  # Modul-Default; wird von --leverage überschrieben
SYMBOLS = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]

# Vorregistrierte Sweep-Achsen (research/05 §3) — nicht erweitern.
CONFIGS = [
    (20, 3.0, True),   # <- a-priori PRIMARY
    (20, 3.0, False),
    (20, 4.0, True),
    (55, 3.0, True),
    (55, 4.0, True),
    (10, 2.5, True),
]
PRIMARY = (20, 3.0, True)


def _atr(df, period=14):
    return ta_lib.volatility.average_true_range(df["high"], df["low"], df["close"], window=period)


def backtest(df: pd.DataFrame, entry_lb: int, chand_atr: float, use_regime: bool,
             side: str = "long", lev: float = LEV, carry_daily: float = 0.0) -> list:
    """Ein Durchlauf über den gesamten (bereits gekappten) Frame, gibt Trade-Liste zurück.

    Long-Pfad ist verhaltensidentisch zur ursprünglichen Fassung (Regressions-Check).
    Short-Pfad spiegelt die Vorzeichen; der bekannte Quirk (Initial-Stop nutzt
    entry_atr, Trailing nutzt atr[j]) wird bewusst MITGESPIEGELT, nicht "gefixt" —
    sonst ist der Vergleich zur protokollierten Long-PF ungültig.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = _atr(df).values
    ema200 = df["close"].ewm(span=200, adjust=False).mean().values
    idx = df.index

    is_long = side == "long"
    trades = []
    i = max(entry_lb, 200)
    n = len(df)
    while i < n - 1:
        if is_long:
            donchian_high = high[i - entry_lb:i].max()
            if not (close[i] > donchian_high):
                i += 1; continue
            if use_regime and not (close[i] > ema200[i]):
                i += 1; continue
        else:
            donchian_low = low[i - entry_lb:i].min()
            if not (close[i] < donchian_low):
                i += 1; continue
            if use_regime and not (close[i] < ema200[i]):
                i += 1; continue
        if np.isnan(atr[i]) or atr[i] <= 0:
            i += 1; continue

        entry = close[i]
        entry_atr = atr[i]
        risk = chand_atr * entry_atr          # initiales Risiko (Chandelier-Abstand)
        exit_p = None
        j = i + 1
        if is_long:
            extreme = high[i]
            stop = extreme - chand_atr * entry_atr
            while j < n:
                extreme = max(extreme, high[j])
                new_stop = extreme - chand_atr * atr[j] if not np.isnan(atr[j]) else stop
                stop = max(stop, new_stop)          # Long-Stop nie senken
                if low[j] <= stop:
                    exit_p = stop
                    break
                j += 1
        else:
            extreme = low[i]
            stop = extreme + chand_atr * entry_atr
            while j < n:
                extreme = min(extreme, low[j])
                new_stop = extreme + chand_atr * atr[j] if not np.isnan(atr[j]) else stop
                stop = min(stop, new_stop)          # Short-Stop nie heben
                if high[j] >= stop:
                    exit_p = stop
                    break
                j += 1
        if exit_p is None:
            exit_p = close[n - 1]; j = n - 1

        held_days = (j - i) / 24.0
        if is_long:
            pnl = (exit_p - entry) / entry * lev - 2 * FEE * lev
            move = exit_p - entry
        else:
            pnl = (entry - exit_p) / entry * lev - 2 * FEE * lev + carry_daily * held_days
            move = entry - exit_p
        r = move / risk if risk > 0 else 0.0
        trades.append({
            "pnl": pnl, "r": r, "held": j - i, "win": pnl > 0,
            "entry_ts": idx[i], "exit_ts": idx[j],
        })
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
        "expectancy": sum(t["pnl"] for t in trades) / n,
    }


# ---------------------------------------------------------------------------
# Bootstrap-CI der gepoolten Per-Trade-Expectancy (research/05 §4)
# ---------------------------------------------------------------------------
def iid_bootstrap_expectancy(trades, n_iter=10000, seed=0):
    if not trades:
        return None
    rng = np.random.default_rng(seed)
    pnl = np.array([t["pnl"] for t in trades])
    means = rng.choice(pnl, size=(n_iter, len(pnl)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"point": float(pnl.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0)}


def block_bootstrap_expectancy(trades, block_days=21, n_iter=10000, seed=0):
    """Moving-Block-Bootstrap über entry_ts: die 5 Alts sind BTC-Beta, ein Downtrend
    feuert ~5 korrelierte Shorts. i.i.d. würde die CI um ~sqrt(5) zu eng machen."""
    if not trades:
        return None
    rng = np.random.default_rng(seed)
    ts0 = min(t["entry_ts"] for t in trades)
    ts1 = max(t["entry_ts"] for t in trades)
    span_days = (ts1 - ts0).total_seconds() / 86400.0 + 1e-9
    n_blocks = max(1, int(math.ceil(span_days / block_days)))
    buckets = defaultdict(list)
    for t in trades:
        b = int(((t["entry_ts"] - ts0).total_seconds() / 86400.0) // block_days)
        buckets[b].append(t["pnl"])
    keys = np.array(sorted(buckets.keys()))
    means = []
    for _ in range(n_iter):
        drawn = rng.choice(keys, size=n_blocks, replace=True)
        pool = [p for k in drawn for p in buckets[k]]
        if pool:
            means.append(float(np.mean(pool)))
    means = np.array(means)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"point": float(np.mean([t["pnl"] for t in trades])),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0), "n_blocks": n_blocks}


def thirds_expectancy_signs(trades, split_ts=None):
    """OOS-dev in 3 zusammenhängende Zeit-Drittel; Vorzeichen der gepoolten
    Expectancy je Drittel + ob es 'qualifiziert' (>=10 gepoolte Trades)."""
    if not trades:
        return []
    ts = sorted(t["entry_ts"] for t in trades)
    t0, t1 = ts[0], ts[-1]
    span = (t1 - t0).total_seconds() or 1.0
    out = []
    for k in range(3):
        a = t0 + datetime.timedelta(seconds=span * k / 3)
        b = t0 + datetime.timedelta(seconds=span * (k + 1) / 3)
        seg = [t for t in trades if (a <= t["entry_ts"] < b) or (k == 2 and t["entry_ts"] == b)]
        if seg:
            exp = sum(t["pnl"] for t in seg) / len(seg)
            out.append({"n": len(seg), "exp": exp, "qualifies": len(seg) >= 10,
                        "sign": "+" if exp > 0 else "-"})
        else:
            out.append({"n": 0, "exp": 0.0, "qualifies": False, "sign": "0"})
    return out


def _fmt_ci(ci):
    if ci is None:
        return "n/a"
    star = "  EXCLUDES 0" if ci["excludes_zero"] else ""
    return f"point={ci['point']:+.5f}  [{ci['ci_lo']:+.5f}, {ci['ci_hi']:+.5f}]{star}"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["long", "short"], default="long")
    ap.add_argument("--days", type=int, default=405)
    ap.add_argument("--as-of", type=str, default=None,
                    help="ISO-Datum. Daten danach werden vor jeder Berechnung verworfen "
                         "(Vault-Schutz). Pflicht für vorregistrierte research/05-Läufe.")
    ap.add_argument("--split-days", type=int, default=300,
                    help="IS/OOS-Grenze = data_start + split_days (nach entry_ts partitioniert).")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="Überschreibt LEV. Regressions-Check gegen PROGRESS.md: 3.0.")
    ap.add_argument("--carry-daily", type=float, default=0.0,
                    help="Per-Tag-Carry (Return-Fraktion), nur auf Short-Trades. "
                         "Portfolio-Ø aus research/04: -0.00003.")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--block-days", type=int, default=21)
    ap.add_argument("--fee", type=float, default=None,
                    help="Überschreibt FEE (0.0016). --fee 0 = Brutto-Test ohne Gebühren.")
    args = ap.parse_args()

    global FEE
    if args.fee is not None:
        FEE = args.fee
        logger.info("FEE überschrieben → %.4f", FEE)

    as_of_ts = None
    if args.as_of:
        as_of_ts = datetime.datetime.fromisoformat(args.as_of).replace(tzinfo=datetime.timezone.utc)
    elif args.side == "short":
        logger.warning("!!! --as-of NICHT gesetzt bei --side short — Vault-Fenster ist NICHT "
                       "geschützt. Für vorregistrierte research/05-Läufe --as-of 2026-07-22 setzen.")

    print(f"\nLade {args.days}d OHLCV für {len(SYMBOLS)} Symbole (side={args.side}, lev={args.leverage})…")
    data = {}
    for s in SYMBOLS:
        try:
            df = load_ohlcv(s, "1h", args.days)
        except Exception as e:
            logger.warning("%s laden fehlgeschlagen: %s", s, e)
            continue
        if as_of_ts is not None:
            df = df[df.index <= as_of_ts]
        data[s] = df
        logger.info("Data %s: %d Kerzen (%s → %s)", s, len(df), df.index[0], df.index[-1])

    if not data:
        print("Keine Daten geladen."); return

    data_start = max(df.index[0] for df in data.values())
    split_ts = data_start + datetime.timedelta(days=args.split_days)
    data_end = min(df.index[-1] for df in data.values())
    logger.info("IS-dev: %s → %s  |  OOS-dev: %s → %s", data_start, split_ts, split_ts, data_end)

    # --- alle Configs laufen, Trades pro Config gepoolt über Symbole + IS/OOS-Split ---
    per_cfg = {}
    for lb, ca, reg in CONFIGS:
        alltr = []
        for s, df in data.items():
            alltr += backtest(df, lb, ca, reg, side=args.side,
                              lev=args.leverage, carry_daily=args.carry_daily)
        is_tr = [t for t in alltr if t["entry_ts"] < split_ts]
        oos_tr = [t for t in alltr if split_ts <= t["entry_ts"] <= data_end]
        per_cfg[(lb, ca, reg)] = {"is": is_tr, "oos": oos_tr, "all": alltr}

    def _row(cfg):
        r_is = summarize(per_cfg[cfg]["is"])
        r_oos = summarize(per_cfg[cfg]["oos"])
        lb, ca, reg = cfg
        tag = f"LB={lb} ATR={ca} reg={'J' if reg else 'N'}"
        def g(r, k, d="  n/a"):
            return d if r is None else r[k]
        n_is = 0 if r_is is None else r_is["n"]
        n_oos = 0 if r_oos is None else r_oos["n"]
        pf_is = "n/a" if r_is is None else f"{r_is['pf']:.2f}"
        pf_oos = "n/a" if r_oos is None else f"{r_oos['pf']:.2f}"
        wr_oos = "n/a" if r_oos is None else f"{r_oos['wr']:.0%}"
        r_oos_v = "n/a" if r_oos is None else f"{r_oos['avg_r']:+.2f}"
        tot_oos = "n/a" if r_oos is None else f"{r_oos['total']:+.1f}"
        return (tag, n_is, pf_is, n_oos, pf_oos, wr_oos, r_oos_v, tot_oos)

    W = 88
    print(f"\n{'='*W}")
    print(f"  TREND-{args.side.upper()} + TRAILING-STOP — research/05  (IS/OOS nach entry_ts)")
    print(f"{'='*W}")
    hdr = f"  {'Config':<22} {'N_is':>5} {'PF_is':>6} {'N_oos':>6} {'PF_oos':>7} {'WR_oos':>7} {'Ø-R':>6} {'Tot%':>7}"
    print(hdr); print(f"  {'-'*(W-2)}")

    prim = _row(PRIMARY)
    print(f"  PRIMARY  {prim[0]:<13} {prim[1]:>5} {prim[2]:>6} {prim[3]:>6} {prim[4]:>7} {prim[5]:>7} {prim[6]:>6} {prim[7]:>7}")
    print(f"\n  SECONDARY — do-not-gate (nur deskriptiv, auf IS-dev gerankt):")
    sec = [c for c in CONFIGS if c != PRIMARY]
    sec.sort(key=lambda c: (summarize(per_cfg[c]["is"]) or {"pf": -1})["pf"], reverse=True)
    for c in sec:
        row = _row(c)
        print(f"    {row[0]:<20} {row[1]:>5} {row[2]:>6} {row[3]:>6} {row[4]:>7} {row[5]:>7} {row[6]:>6} {row[7]:>7}")

    # --- Primary-Config: Bootstrap-CIs + Drittel + VERDICT ---
    oos = per_cfg[PRIMARY]["oos"]
    r_oos = summarize(oos)
    block_ci = block_bootstrap_expectancy(oos, args.block_days, args.bootstrap, seed=0)
    iid_ci = iid_bootstrap_expectancy(oos, args.bootstrap, seed=0)
    thirds = thirds_expectancy_signs(oos)
    median_oos_pf = float(np.median([
        (summarize(per_cfg[c]["oos"]) or {"pf": 0.0})["pf"] for c in CONFIGS
    ]))

    print(f"\n{'-'*W}")
    print(f"  PRIMARY-CONFIG OOS-dev  (N={0 if r_oos is None else r_oos['n']})")
    print(f"  Block-Bootstrap-CI (GATE): {_fmt_ci(block_ci)}"
          + (f"  n_blocks={block_ci['n_blocks']}" if block_ci else ""))
    print(f"  i.i.d.-Bootstrap-CI (nur Inflation): {_fmt_ci(iid_ci)}")
    print(f"  Median-der-6 OOS-PF (Fragilitäts-Check): {median_oos_pf:.2f}")
    q = [t for t in thirds if t["qualifies"]]
    pos = [t for t in q if t["exp"] > 0]
    print(f"  OOS-Drittel Vorzeichen: {[t['sign'] for t in thirds]}  "
          f"(qualifizierend {len(q)}/3, davon positiv {len(pos)})")

    # VERDICT H5a
    pf_oos = None if r_oos is None else r_oos["pf"]
    n_oos = 0 if r_oos is None else r_oos["n"]
    s1 = abs(args.leverage - 1.0) < 1e-9
    s2 = bool(block_ci and block_ci["excludes_zero"])
    s3 = n_oos >= 30
    s4 = bool(pf_oos is not None and pf_oos > 1.0 and len(q) > 0 and len(pos) > len(q) / 2)
    print(f"\n  VERDICT H5a (Standalone {args.side}):")
    print(f"    S1 leverage == 1x ................. {'PASS' if s1 else 'FAIL'}  (lev={args.leverage})")
    print(f"    S2 Block-CI schließt 0 aus ........ {'PASS' if s2 else 'FAIL'}")
    print(f"    S3 >= 30 gepoolte OOS-Trades ...... {'PASS' if s3 else 'FAIL'}  (N={n_oos})")
    print(f"    S4 PF>1 & Mehrheit Drittel pos .... {'PASS' if s4 else 'FAIL'}  "
          f"(PF={'n/a' if pf_oos is None else f'{pf_oos:.2f}'})")
    print(f"    S5 Carry-Sensitivität ............. separat mit --carry-daily -0.00003 prüfen")
    overall = s1 and s2 and s3 and s4
    print(f"    OVERALL (S1–S4): {'PASS → S5 prüfen' if overall else 'DEAD'}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
