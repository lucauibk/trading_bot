"""
research/08 — Cross-Sectional Momentum (H8a) + BTC-Trend-Switch (H8b)

Wöchentliches Rebalancing, Long-Only Top-Quintil nach trailing L-Wochen-Return,
Fees 0,15 %/Seite auf Turnover. Benchmarks: BTC-B&H, Equal-Weight-B&H.
Dev bis 2025-12-31, Vault ab 2026-01-01 (nur mit --vault angeschaut).

Universum: Top-N Binance-USDT-Spot nach 24h-Quote-Volumen, exkl. Stables/Gold/
tokenisierte-Aktien, min. Historie. Survivorship Bias BEWUSST akzeptiert
(optimistische Obergrenze — schlägt es BTC hier nicht, ist es tot).

  python3 scripts/cross_sectional_backtest.py            # Dev-Lauf
  python3 scripts/cross_sectional_backtest.py --vault    # + Vault (nur nach Dev-Pass)
"""
import argparse, datetime as dt, json, logging, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cs")

CACHE = Path("/private/tmp/claude-502/-Users-lucasturz-Projects-trading-bot/"
             "44eeb447-d9b5-45f1-8be4-70683649ad5e/scratchpad/cs_daily.pkl")
DEV_END = pd.Timestamp("2025-12-31", tz="UTC")
FEE_PER_SIDE = 0.0015
N_UNIVERSE = 40
MIN_DAYS = 400
EXCL = ("USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "USD1", "RLUSD", "XUSD",
        "EUR", "GBP", "TRY", "BRL", "ARS", "XAUT", "PAXG", "WBTC", "WBETH", "BETH",
        "EURI", "AEUR")
EXCL_SUFFIX_B = ("SNDKB", "CRCLB", "MSTRB", "NVDAB", "COINB", "HOODB", "AAPLB")


def build_universe(b):
    tickers = b.fetch_tickers()
    mk = b.markets
    rows = []
    for s, t in tickers.items():
        if not s.endswith("/USDT"):
            continue
        m = mk.get(s, {})
        if not (m.get("spot") and m.get("active")):
            continue
        base = s.split("/")[0]
        if any(e == base or e in base for e in EXCL) or base in EXCL_SUFFIX_B:
            continue
        qv = t.get("quoteVolume")
        if qv:
            rows.append((s, qv))
    rows.sort(key=lambda r: -r[1])
    return [s for s, _ in rows[:N_UNIVERSE * 2]]  # Überhang, Historie-Filter kappt später


def fetch_daily(b, symbols, days=1600):
    since = b.milliseconds() - days * 86400_000
    out = {}
    for s in symbols:
        try:
            rows, cur = [], since
            while True:
                batch = b.fetch_ohlcv(s, "1d", since=cur, limit=1000)
                if not batch:
                    break
                rows += batch
                cur = batch[-1][0] + 86400_000
                if len(batch) < 1000:
                    break
                time.sleep(b.rateLimit / 1000)
            if len(rows) < MIN_DAYS:
                log.info("skip %s (%d Tage)", s, len(rows))
                continue
            df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            out[s] = df.set_index("ts")["c"]
            log.info("ok %s (%d Tage)", s, len(df))
            time.sleep(b.rateLimit / 1000)
        except Exception as e:
            log.warning("fail %s: %s", s, e)
    return pd.DataFrame(out).sort_index()


def load_data(force=False):
    if CACHE.exists() and not force:
        px = pd.read_pickle(CACHE)
        px.index = pd.to_datetime(px.index, utc=True)
        return px
    b = ccxt.binance({"enableRateLimit": True})
    b.load_markets()
    uni = build_universe(b)
    log.info("Universum-Kandidaten: %d", len(uni))
    px = fetch_daily(b, uni)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_pickle(CACHE)
    return px


# ---------------------------------------------------------------------------
def perf(weekly_ret: pd.Series, freq=52):
    r = weekly_ret.dropna()
    if len(r) < 5:
        return None
    eq = (1 + r).cumprod()
    yrs = len(r) / freq
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(freq)
    sharpe = (r.mean() * freq) / vol if vol > 0 else 0.0
    downside = r[r < 0].std() * np.sqrt(freq)
    sortino = (r.mean() * freq) / downside if downside > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    hit = (r > 0).mean()
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "maxdd": dd,
            "hit": hit, "n": len(r), "eq": eq}


def cs_momentum(px_w: pd.Series, ret_w: pd.DataFrame, lookback: int, frac: float):
    """Long-Only Top-`frac` nach trailing `lookback`-Wochen-Return, wöchentlich.
    Gibt (netto_ret, brutto_ret, avg_turnover)."""
    mom = px_w.pct_change(lookback)
    idx = px_w.index
    prev_w = pd.Series(0.0, index=px_w.columns)
    gross, net, turns = [], [], []
    for i in range(lookback + 1, len(idx) - 1):
        t, t1 = idx[i], idx[i + 1]
        scores = mom.iloc[i].dropna()
        # nur Coins mit gültigem Forward-Return und genug Historie
        valid = scores.index[px_w.iloc[i][scores.index].notna()
                             & ret_w.loc[t1, scores.index].notna()]
        scores = scores[valid]
        if len(scores) < 10:
            gross.append(np.nan); net.append(np.nan); continue
        k = max(1, int(np.ceil(len(scores) * frac)))
        winners = scores.nlargest(k).index
        w = pd.Series(0.0, index=px_w.columns)
        w[winners] = 1.0 / k
        turnover = (w - prev_w).abs().sum() / 2
        fee = turnover * 2 * FEE_PER_SIDE
        fwd = ret_w.loc[t1, winners].mean()
        gross.append(fwd)
        net.append(fwd - fee)
        turns.append(turnover)
        prev_w = w
    ix = idx[lookback + 2: len(idx)]
    return (pd.Series(net, index=ix[:len(net)]),
            pd.Series(gross, index=ix[:len(gross)]),
            float(np.nanmean(turns)) if turns else 0.0)


def btc_trend_switch(px_btc_d: pd.Series):
    """Daily 200-SMA-Switch, dann auf Wochen-Returns aggregiert. Fee bei jedem Flip."""
    sma = px_btc_d.rolling(200).mean()
    pos = (px_btc_d > sma).astype(float).shift(1).fillna(0.0)
    dret = px_btc_d.pct_change().fillna(0.0)
    flip = pos.diff().abs().fillna(0.0)
    strat_d = pos * dret - flip * FEE_PER_SIDE
    return strat_d.resample("W").apply(lambda x: (1 + x).prod() - 1)


def summarize(tag, s, bench_sharpe=None):
    p = perf(s)
    if p is None:
        print(f"  {tag:<34} — zu wenig Daten"); return None
    extra = ""
    if bench_sharpe is not None:
        extra = f"  Δsharpe={p['sharpe']-bench_sharpe:+.2f}"
    print(f"  {tag:<34} CAGR {p['cagr']:>+7.1%}  Sharpe {p['sharpe']:>+5.2f}  "
          f"Sortino {p['sortino']:>+5.2f}  MaxDD {p['maxdd']:>+6.1%}  "
          f"hit {p['hit']:.0%}  n={p['n']}{extra}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", action="store_true")
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--fee", type=float, default=None,
                    help="Überschreibt FEE_PER_SIDE (0.0015). --fee 0 = Brutto-Test.")
    args = ap.parse_args()

    global FEE_PER_SIDE
    if args.fee is not None:
        FEE_PER_SIDE = args.fee
        log.info("FEE_PER_SIDE überschrieben → %.4f", FEE_PER_SIDE)

    px = load_data(force=args.refetch)
    px = px.dropna(axis=1, thresh=MIN_DAYS)
    log.info("Universum final: %d Coins, %s → %s", px.shape[1], px.index[0].date(), px.index[-1].date())

    px_w = px.resample("W").last()
    ret_w = px_w.pct_change()

    def slice_end(df, vault):
        return df if vault else df[df.index <= DEV_END]

    W = 100
    for scope, vault in ([("DEV", False)] + ([("VAULT (voll)", True)] if args.vault else [])):
        pxw = slice_end(px_w, vault); rw = slice_end(ret_w, vault)
        pxd = slice_end(px, vault)
        print(f"\n{'='*W}\n  {scope}  ({pxw.index[0].date()} → {pxw.index[-1].date()}, {len(pxw)} Wochen)\n{'='*W}")

        btc = rw["BTC/USDT"].dropna()
        bh = summarize("BTC Buy&Hold", btc)
        ew = summarize("Equal-Weight Buy&Hold", rw.mean(axis=1))
        bench_sharpe = bh["sharpe"] if bh else 0.0

        print(f"  {'-'*(W-2)}")
        results = {}
        for lb in (4, 8, 12):
            for frac, fname in ((0.2, "Q"), (1/3, "T")):
                net, gross, to = cs_momentum(pxw, rw, lb, frac)
                p_net = perf(net); p_gross = perf(gross)
                if p_net is None:
                    continue
                drag = p_gross["cagr"] - p_net["cagr"]
                tag = f"CS-Mom L={lb}W {fname} (to={to:.0%})"
                print(f"  {tag:<34} CAGR {p_net['cagr']:>+7.1%}  Sharpe {p_net['sharpe']:>+5.2f}  "
                      f"MaxDD {p_net['maxdd']:>+6.1%}  drag {drag:>+5.1%}  Δsh {p_net['sharpe']-bench_sharpe:+.2f}")
                results[(lb, fname)] = (p_net, p_gross, drag)

        # H8b
        print(f"  {'-'*(W-2)}")
        sw = btc_trend_switch(pxd["BTC/USDT"].dropna())
        sw = sw[sw.index.isin(pxw.index)] if not vault else sw
        p_sw = summarize("BTC 200D-SMA-Switch (H8b)", sw, bench_sharpe)

        if not vault:
            # Dev-Hälften für M3
            mid = pxw.index[len(pxw)//2]
            print(f"\n  M3 Dev-Hälften (Split {mid.date()}):")
            for lb in (4, 8, 12):
                net, _, _ = cs_momentum(pxw, rw, lb, 0.2)
                a, bb = net[net.index <= mid], net[net.index > mid]
                pa, pb = perf(a), perf(bb)
                sa = f"{pa['cagr']:+.1%}" if pa else "n/a"
                sb = f"{pb['cagr']:+.1%}" if pb else "n/a"
                print(f"    L={lb}W Q:  früh {sa}   spät {sb}")

            # VERDICT
            print(f"\n  {'='*(W-4)}\n  VERDICT (Dev)")
            passes = 0
            for (lb, fn), (pn, pg, drag) in results.items():
                if fn != "Q":
                    continue
                m1 = pn["cagr"] > 0
                m2 = pn["sharpe"] > bench_sharpe + 0.2
                m4 = drag < 0.5 * pn["cagr"] if pn["cagr"] > 0 else False
                ok = m1 and m2 and m4
                passes += ok
                print(f"    L={lb}W Q: M1(CAGR>0)={m1}  M2(Δsharpe≥.2)={m2}  "
                      f"M4(drag<½CAGR)={m4}  → {'PASS' if ok else 'fail'}")
            print(f"    M5 (≥2/3 Lookbacks PASS): {'PASS' if passes >= 2 else 'FAIL'}  ({passes}/3)")
            t1 = p_sw and p_sw["sharpe"] > bench_sharpe + 0.2
            t2 = p_sw and abs(p_sw["maxdd"]) < 0.7 * abs(bh["maxdd"])
            print(f"    H8b: T1(Δsharpe≥.2)={bool(t1)}  T2(MaxDD<0.7×BTC)={bool(t2)}  "
                  f"→ {'PASS' if (t1 and t2) else 'FAIL'}")
            print(f"  {'='*(W-4)}")


if __name__ == "__main__":
    main()
