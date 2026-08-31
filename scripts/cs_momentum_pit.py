"""
research/08 Nachtest — Cross-Sectional Momentum mit point-in-time-Universum.

Der Phase-8-Lauf nutzte „Top-Coins die HEUTE nach Volumen existieren" → Survivorship
Bias. Hier: dynamisches Universum nach Market-Cap-Proxy (Preis × aktuelle
zirkulierende Menge), plus zwei Robustheits-Schnitte.

Residual-Bias, ehrlich benannt:
- **Supply-Proxy:** aktuelle circ. Menge × historischer Preis. Coins mit hoher
  Emission seit 2022 bekommen ihre FRÜHE Mcap überschätzt.
- **Binance-Delisting:** komplett von Binance delistete Coins (LUNA classic, FTT …)
  fehlen weiter. Kleineres Loch als „Top-by-heutigem-Volumen", aber nicht null.
- Ein voll sauberer Test bräuchte bezahlte historische Mcap (CoinGecko Pro, Kaiko).

  python3 scripts/cs_momentum_pit.py
"""
import json, logging, pickle, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.cross_sectional_backtest import (load_data, perf, MIN_DAYS, DEV_END,
                                              FEE_PER_SIDE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pit")

SCRATCH = Path("/private/tmp/claude-502/-Users-lucasturz-Projects-trading-bot/"
               "44eeb447-d9b5-45f1-8be4-70683649ad5e/scratchpad")
CG_CACHE = SCRATCH / "cg_markets.pkl"
V0 = pd.Timestamp("2026-01-01", tz="UTC")


def cg_supply_map():
    if CG_CACHE.exists():
        markets = pickle.load(open(CG_CACHE, "rb"))
    else:
        markets = []
        for page in (1, 2):
            url = (f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
                   f"&order=market_cap_desc&per_page=250&page={page}")
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            markets += json.load(urllib.request.urlopen(req, timeout=30))
        pickle.dump(markets, open(CG_CACHE, "wb"))
    # symbol -> circ supply, pick largest-mcap match per ticker
    best = {}
    for c in markets:
        sym = (c.get("symbol") or "").upper()
        cs = c.get("circulating_supply") or 0
        mc = c.get("market_cap") or 0
        if not sym or cs <= 0:
            continue
        if sym not in best or mc > best[sym][1]:
            best[sym] = (cs, mc)
    return {k: v[0] for k, v in best.items()}


def mcap_proxy(px_w: pd.DataFrame):
    supply = cg_supply_map()
    cols, sup = [], []
    for c in px_w.columns:
        base = c.split("/")[0]
        if base in supply:
            cols.append(c); sup.append(supply[base])
    log.info("Supply-Match für %d/%d Coins", len(cols), px_w.shape[1])
    return px_w[cols].mul(pd.Series(sup, index=cols), axis=1)


def cs_mom_universe(px_w, ret_w, mcap_w, lookback, frac, universe_fn):
    """universe_fn(week_index_i) -> set gültiger Coins für dieses Rebalance."""
    mom = px_w.pct_change(lookback)
    idx = px_w.index
    prev_w = pd.Series(0.0, index=px_w.columns)
    net = []
    for i in range(lookback + 1, len(idx) - 1):
        t1 = idx[i + 1]
        uni = universe_fn(i)
        scores = mom.iloc[i].reindex(uni).dropna()
        valid = [c for c in scores.index
                 if pd.notna(px_w.iloc[i].get(c)) and pd.notna(ret_w.loc[t1].get(c))]
        scores = scores[valid]
        if len(scores) < 8:
            net.append(np.nan); continue
        k = max(1, int(np.ceil(len(scores) * frac)))
        winners = scores.nlargest(k).index
        w = pd.Series(0.0, index=px_w.columns)
        w[winners] = 1.0 / k
        fee = (w - prev_w).abs().sum() / 2 * 2 * FEE_PER_SIDE
        net.append(ret_w.loc[t1, winners].mean() - fee)
        prev_w = w
    ix = idx[lookback + 2: len(idx)]
    return pd.Series(net, index=ix[:len(net)])


def run_block(name, px_w, ret_w, mcap_w, universe_fn, dev_only=True):
    print(f"\n  {name}")
    btc_dev = ret_w["BTC/USDT"]
    btc_dev = btc_dev[btc_dev.index <= DEV_END] if dev_only else btc_dev
    p_btc = perf(btc_dev)
    print(f"    {'BTC B&H':<26} CAGR {p_btc['cagr']:>+7.1%}  Sharpe {p_btc['sharpe']:>+5.2f}  MaxDD {p_btc['maxdd']:>+6.1%}")
    passes = 0
    for lb in (4, 8, 12):
        s = cs_mom_universe(px_w, ret_w, mcap_w, lb, 0.2, universe_fn)
        s_dev = s[s.index <= DEV_END] if dev_only else s
        p = perf(s_dev.dropna())
        if p is None:
            print(f"    L={lb}W Q: zu wenig Daten"); continue
        dsh = p["sharpe"] - p_btc["sharpe"]
        ok = p["cagr"] > 0 and dsh > 0.2
        passes += ok
        print(f"    {'L='+str(lb)+'W Q':<26} CAGR {p['cagr']:>+7.1%}  Sharpe {p['sharpe']:>+5.2f}  "
              f"MaxDD {p['maxdd']:>+6.1%}  Δsh {dsh:+.2f}  {'PASS' if ok else 'fail'}")
        # isoliertes Vault
        if dev_only:
            sv = perf(s[s.index >= V0].dropna())
            if sv:
                print(f"      └ Vault 2026: CAGR {sv['cagr']:>+7.1%}  Sharpe {sv['sharpe']:>+5.2f}")
    print(f"    → M5 (≥2/3): {'PASS' if passes >= 2 else 'FAIL'} ({passes}/3)")
    return passes >= 2


def main():
    px = load_data()
    px = px.dropna(axis=1, thresh=MIN_DAYS)
    px_w = px.resample("W").last()
    ret_w = px_w.pct_change()
    mcap_w = mcap_proxy(px_w).reindex(columns=px_w.columns)

    W = 96
    print(f"\n{'='*W}\n  CS-MOMENTUM — point-in-time-Universum-Nachtest (research/08)\n{'='*W}")
    print(f"  Coins mit Preis-Historie: {px_w.shape[1]}  |  mit Supply-Match: {mcap_w.notna().any().sum()}")

    ranks = mcap_w.rank(axis=1, ascending=False)  # 1 = größte Mcap

    # 1) DYN top-30: jedes Rebalance top-30 nach Mcap-Proxy
    def uni_dyn30(i):
        r = ranks.iloc[i].dropna()
        return set(r[r <= 30].index)
    p1 = run_block("[1] DYNAMISCH top-30 nach Mcap-Proxy (jedes Rebalance)",
                   px_w, ret_w, mcap_w, uni_dyn30)

    # 2) DYN top-15 large-cap (Survivorship-Loch am kleinsten)
    def uni_dyn15(i):
        r = ranks.iloc[i].dropna()
        return set(r[r <= 15].index)
    p2 = run_block("[2] DYNAMISCH top-15 (large-cap, kleinster Bias)",
                   px_w, ret_w, mcap_w, uni_dyn15)

    # 3) FROZEN: top-30 nach Mcap-Proxy in der ERSTEN Dev-Woche, fix gehalten
    first_i = 13  # nach Warmup genug Historie
    frozen = set(ranks.iloc[first_i].dropna().sort_values().index[:30])
    def uni_frozen(i):
        return frozen
    p3 = run_block(f"[3] FROZEN top-30 per {px_w.index[first_i].date()} (fix)",
                   px_w, ret_w, mcap_w, uni_frozen)

    print(f"\n{'='*W}")
    print(f"  GESAMT: DYN30 {'PASS' if p1 else 'FAIL'}  |  DYN15 {'PASS' if p2 else 'FAIL'}  |  FROZEN {'PASS' if p3 else 'FAIL'}")
    print(f"  (Phase-8-Original mit Survivorship-Universum: nur L=8W bestand, M5 FAIL)")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
