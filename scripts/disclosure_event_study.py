"""
research/10 — Event-Study „offengelegte Trades kopieren" gegen die vorregistrierten
Kill-Kriterien C0/K1/P1/K2/K5/K3/K4 (research/10-disclosed-trades.md §6).

  .venv-research/bin/python scripts/disclosure_event_study.py --hypothesis b
  .venv-research/bin/python scripts/disclosure_event_study.py --hypothesis a
  .venv-research/bin/python scripts/disclosure_event_study.py --hypothesis b --vault   # nur nach Dev-PASS

Methode (§5): Calendar-Time-Portfolio je Haltedauer, gleichgewichtet über offene
Event-Positionen, Kosten am Ein-/Ausstiegstag. Placebo = gleicher Ticker, zufälliger
Handelstag im selben Fenster. α aus FF5+UMD mit Newey-West.
"""
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.disclosure_data import (CACHE, build_insider_clusters, load_ff_factors,
                                     load_insider_filings, load_prices,
                                     load_senate_purchases)

log = logging.getLogger("event_study")

# research/10 §3 — Fenster nach Event-(Melde-)Datum
WINDOWS = {
    "a": {"dev": ("2013-01-01", "2017-12-31"), "split": "2016-01-01",
          "vault": ("2018-01-01", "2021-03-10")},
    "b": {"dev": ("2013-01-01", "2023-12-31"), "split": "2019-01-01",
          "vault": ("2024-01-01", "2026-12-31")},
}
COST_SIDE = 0.0010 + 0.0025            # §4 Basis: Kommission/Spread + FX
COST_SIDE_STRESS = 0.0050 + 0.0025     # §4 Sensitivität
MIN_DOLLAR_VOL = 1_000_000             # §2.3
PRICE_TOL = 0.10                       # §2.3 Ticker-Reuse-Schutz
COVERAGE_MIN = 0.70                    # C0
BLOCK = 21
N_BOOT = 10_000
NW_LAGS = 20
K4_CAPITAL, K4_PCT, K4_MAX_OPEN, K4_MIN_FEE = 3000.0, 0.02, 20, 1.0


# ---------------------------------------------------------------------------
# Events → Positionen
# ---------------------------------------------------------------------------
def load_events(hyp: str) -> pd.DataFrame:
    if hyp == "a":
        return load_senate_purchases()
    fil_cache = CACHE / "insider_filings.pkl"
    fil = pd.read_pickle(fil_cache) if fil_cache.exists() else load_insider_filings()
    # Delay-Filter gilt auch für ältere Caches
    d = (fil["filing_date"] - fil["tx_date"]).dt.days
    fil = fil[(d >= 0) & (d <= 30)]
    return build_insider_clusters(fil)


def unadjusted_close(px, ticker, date):
    """Split-bereinigter Close am letzten Handelstag ≤ date, auf den damaligen
    (unbereinigten) Kurs zurückgerechnet."""
    c = px["close"][ticker].loc[:date].dropna()
    if c.empty:
        return np.nan
    sp = px["splits"][ticker] if ticker in px["splits"] else pd.Series(dtype=float)
    later = sp[(sp.index > c.index[-1]) & (sp > 0)]
    return float(c.iloc[-1] * later.prod()) if len(later) else float(c.iloc[-1])


def resolve_positions(events, px, cal, hold, check_price):
    """Je Event: Einstiegs-/Ausstiegsindex im Handelskalender oder Grund fürs Verwerfen."""
    out, reasons = [], {}
    adj, close, vol = px["adj"], px["close"], px["volume"]
    for ev in events.itertuples(index=False):
        t = ev.ticker
        if t not in adj.columns:
            reasons["kein_kurs"] = reasons.get("kein_kurs", 0) + 1
            continue
        i0 = cal.searchsorted(ev.filing_date, side="right")  # erster Handelstag NACH Meldung
        i1 = i0 + hold
        if i1 >= len(cal):
            reasons["haltedauer_ueber_datenende"] = reasons.get("haltedauer_ueber_datenende", 0) + 1
            continue
        a0, a1 = adj[t].iloc[i0], adj[t].iloc[i1]
        if not (np.isfinite(a0) and np.isfinite(a1)) or a0 <= 0:
            reasons["kurs_luecke"] = reasons.get("kurs_luecke", 0) + 1
            continue
        if check_price and np.isfinite(ev.tx_price):
            ref = unadjusted_close(px, t, ev.tx_date)
            if not np.isfinite(ref) or abs(ref / ev.tx_price - 1) > PRICE_TOL:
                reasons["preis_mismatch"] = reasons.get("preis_mismatch", 0) + 1
                continue
        assert cal[i0] > ev.filing_date, "Look-Ahead: Einstieg vor/am Meldedatum"
        out.append({"event_id": ev.event_id, "ticker": t, "filing_date": ev.filing_date,
                    "i0": i0, "i1": i1})
    valid = pd.DataFrame(out)
    # Liquidität ist Strategie-Filter, nicht Abdeckung → separat gezählt
    n_cov = len(valid)
    if not valid.empty:
        dv = (close * vol)
        keep = []
        for r in valid.itertuples(index=False):
            w = dv[r.ticker].iloc[max(0, r.i0 - 20):r.i0].dropna()
            keep.append(len(w) >= 10 and w.median() >= MIN_DOLLAR_VOL)
        valid = valid[np.array(keep, dtype=bool)].reset_index(drop=True)
    reasons["illiquide"] = n_cov - len(valid)
    return valid, n_cov, reasons


def placebo_positions(valid, px, cal, hold, lo, hi, seed=0):
    """Gleicher Ticker, zufälliger Einstiegstag im selben Fenster, gleiche Filter."""
    rng = np.random.default_rng(seed)
    adj, dv = px["adj"], px["close"] * px["volume"]
    ia, ib = cal.searchsorted(pd.Timestamp(lo)), cal.searchsorted(pd.Timestamp(hi), side="right")
    out = []
    for r in valid.itertuples(index=False):
        for _ in range(50):
            i0 = int(rng.integers(ia, ib))
            i1 = i0 + hold
            if i1 >= len(cal):
                continue
            a0, a1 = adj[r.ticker].iloc[i0], adj[r.ticker].iloc[i1]
            if not (np.isfinite(a0) and np.isfinite(a1)) or a0 <= 0:
                continue
            w = dv[r.ticker].iloc[max(0, i0 - 20):i0].dropna()
            if len(w) < 10 or w.median() < MIN_DOLLAR_VOL:
                continue
            out.append({"ticker": r.ticker, "i0": i0, "i1": i1})
            break
    return pd.DataFrame(out)


def calendar_portfolio(pos, px, cal, cost_side):
    """Tägliche gleichgewichtete Rendite aller offenen Positionen (Tage ohne Position: NaN)."""
    n = len(cal)
    s, c = np.zeros(n), np.zeros(n)
    rets = px["adj"].pct_change(fill_method=None)
    for r in pos.itertuples(index=False):
        rr = rets[r.ticker].iloc[r.i0 + 1:r.i1 + 1].to_numpy(copy=True)
        rr = np.nan_to_num(rr, nan=0.0)
        rr[0] -= cost_side
        rr[-1] -= cost_side
        s[r.i0 + 1:r.i1 + 1] += rr
        c[r.i0 + 1:r.i1 + 1] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        port = np.where(c > 0, s / c, np.nan)
    return pd.Series(port, index=cal), pd.Series(c, index=cal)


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------
def block_bootstrap_mean(x: pd.Series, block=BLOCK, n_iter=N_BOOT, seed=0):
    """Zirkulärer Block-Bootstrap des Mittelwerts einer Tagesreihe. Anders als
    trend_breakout_backtest.block_bootstrap_expectancy (Trade-Buckets) hier auf der
    Calendar-Time-Reihe, weil sich 20–120-Tage-Haltedauern überlappen."""
    v = x.dropna().to_numpy()
    n = len(v)
    if n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_iter, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = v[idx.reshape(n_iter, -1)[:, :n]].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "n": n,
            "excl0": bool(lo > 0)}


def ff_alpha(port: pd.Series, ff: pd.DataFrame, lags=NW_LAGS):
    """OLS (R_p − RF) auf FF5+UMD, Newey-West-SE für α."""
    d = pd.concat([port.rename("p"), ff], axis=1, join="inner").dropna()
    if len(d) < 60:
        return None
    y = (d["p"] - d["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(d)), d[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return {"alpha_d": float(beta[0]), "alpha_ann": float(beta[0] * 252),
            "t": float(beta[0] / se[0]), "beta_mkt": float(beta[1]),
            "smb": float(beta[2]), "hml": float(beta[3]), "n": len(d)}


def perf_daily(r: pd.Series):
    r = r.dropna()
    if len(r) < 20:
        return None
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    vol = r.std() * np.sqrt(252)
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1),
            "sharpe": float(r.mean() * 252 / vol) if vol > 0 else 0.0,
            "maxdd": float((eq / eq.cummax() - 1).min())}


def k4_simulation(pos, px, cal, cost_side, spy="SPY"):
    """3000 $, 2 % je Position, max. 20 offen, Rest in SPY, Mindestkommission 1 $.
    SPY-Umschichtungen kostenfrei (begünstigt die Strategie leicht)."""
    if pos.empty:
        return None
    rets = px["adj"].pct_change(fill_method=None)
    spy_r = rets[spy].fillna(0.0).to_numpy()
    i_start, i_end = int(pos["i0"].min()), int(pos["i1"].max())
    entries = pos.groupby("i0")
    cash_spy, open_pos, eq_path = K4_CAPITAL, [], []
    for i in range(i_start, i_end + 1):
        if i > i_start:
            cash_spy *= 1 + spy_r[i]
            for p in open_pos:
                r = rets[p["ticker"]].iloc[i]
                p["val"] *= 1 + (0.0 if not np.isfinite(r) else r)
        still = []
        for p in open_pos:  # Exit am Close von i1
            if p["i1"] == i:
                cash_spy += p["val"] * (1 - cost_side) - K4_MIN_FEE
            else:
                still.append(p)
        open_pos = still
        if i in entries.groups:
            for r in entries.get_group(i).itertuples(index=False):
                if len(open_pos) >= K4_MAX_OPEN:
                    break
                equity = cash_spy + sum(p["val"] for p in open_pos)
                size = K4_PCT * equity
                fee = max(size * cost_side, K4_MIN_FEE)
                if cash_spy < size + fee:
                    break
                cash_spy -= size + fee
                open_pos.append({"ticker": r.ticker, "i1": int(r.i1), "val": size})
        eq_path.append(cash_spy + sum(p["val"] for p in open_pos))
    eq = pd.Series(eq_path, index=cal[i_start:i_end + 1])
    strat = perf_daily(eq.pct_change())
    bench = perf_daily(pd.Series(spy_r[i_start + 1:i_end + 1], index=cal[i_start + 1:i_end + 1]))
    return {"strat": strat, "spy": bench,
            "pass": bool(strat and bench and strat["cagr"] >= bench["cagr"]
                         and strat["sharpe"] >= bench["sharpe"] + 0.2)}


# ---------------------------------------------------------------------------
def evaluate(hyp, window_key, holds, seed=0):
    w = WINDOWS[hyp]
    lo, hi = w[window_key]
    events = load_events(hyp)
    events = events[(events["filing_date"] >= lo) & (events["filing_date"] <= hi)].reset_index(drop=True)
    log.info("H10%s %s: %d Events (%s → %s)", hyp, window_key, len(events), lo, hi)
    px = load_prices(sorted(set(events["ticker"])) + ["SPY"])
    cal = px["adj"]["SPY"].dropna().index
    px = {k: (v.reindex(cal) if isinstance(v, pd.DataFrame) else v) for k, v in px.items()}
    px["splits"] = px["splits"].fillna(0.0)
    ff = load_ff_factors()
    mkt = (ff["Mkt-RF"] + ff["RF"])

    results = {"hypothesis": hyp, "window": window_key, "range": [lo, hi],
               "n_events": len(events), "holds": {}}
    for h in holds:
        valid, n_cov, reasons = resolve_positions(events, px, cal, h, check_price=(hyp == "b"))
        coverage = n_cov / max(1, len(events) - reasons.get("haltedauer_ueber_datenende", 0))
        port, _ = calendar_portfolio(valid, px, cal, COST_SIDE)
        port_stress, _ = calendar_portfolio(valid, px, cal, COST_SIDE_STRESS)
        plc = placebo_positions(valid, px, cal, h, lo, hi, seed=seed)
        port_plc, _ = calendar_portfolio(plc, px, cal, COST_SIDE)

        k1 = block_bootstrap_mean((port - mkt).dropna())
        k1_stress = block_bootstrap_mean((port_stress - mkt).dropna())
        p1 = block_bootstrap_mean((port - port_plc).dropna())
        halves = []
        early = valid["filing_date"] < w["split"]
        for sub in (valid[early], valid[~early]):
            ps, _ = calendar_portfolio(sub, px, cal, COST_SIDE)
            ex = (ps - mkt).dropna()
            halves.append({"n_events": len(sub), "mean": float(ex.mean()) if len(ex) else None})
        alpha = ff_alpha(port, ff)
        k4 = k4_simulation(valid, px, cal, COST_SIDE)

        crit = {
            "C0": coverage >= COVERAGE_MIN,
            "K1": bool(k1 and k1["excl0"]),
            "P1": bool(p1 and p1["excl0"]),
            "K2": all(x["mean"] is not None and x["mean"] > 0 for x in halves),
            "K5": bool(alpha and alpha["alpha_d"] > 0 and alpha["t"] >= 2.0),
            "K4": bool(k4 and k4["pass"]),
        }
        results["holds"][h] = {"n_valid": len(valid), "n_placebo": len(plc),
                               "coverage": coverage, "drops": reasons, "K1": k1,
                               "K1_stress": k1_stress, "P1": p1, "halves": halves,
                               "alpha": alpha, "K4": k4, "crit": crit,
                               "core_pass": crit["K1"] and crit["P1"] and crit["K2"] and crit["K5"]}
    passing = [h for h, r in results["holds"].items() if r["core_pass"]]
    cov_ok = all(r["crit"]["C0"] for r in results["holds"].values())
    k3 = len(passing) >= 2
    k4_ok = any(results["holds"][h]["crit"]["K4"] for h in passing)
    if not cov_ok:
        verdict = "INCONCLUSIVE (C0 Abdeckung < 70 %)"
    else:
        verdict = "PASS" if (k3 and k4_ok) else "FAIL"
    results.update({"K3_passing_holds": passing, "verdict": verdict})
    return results


def _pct(x, d=3):
    return "n/a" if x is None else f"{x * 100:+.{d}f}%"


def report(res):
    print(f"\n=== H10{res['hypothesis']} — {res['window']} {res['range'][0]} → {res['range'][1]} "
          f"— {res['n_events']} Events ===")
    for h, r in res["holds"].items():
        c = r["crit"]
        print(f"\n-- Haltedauer {h} Handelstage: {r['n_valid']} Positionen "
              f"(Abdeckung {r['coverage']:.0%}, Placebo {r['n_placebo']}), Drops {r['drops']}")
        for key, label in (("K1", "Netto − Markt, Ø/Tag"), ("K1_stress", "  dito Stress-Kosten"),
                           ("P1", "Netto − Placebo, Ø/Tag")):
            b = r[key]
            if b:
                print(f"   {label:24s} {_pct(b['mean'])}  CI [{_pct(b['lo'])}, {_pct(b['hi'])}]"
                      f"  ≈ {b['mean'] * 252 * 100:+.1f} %/J  {'EXCL0' if b['excl0'] else ''}")
        print("   K2 Hälften Ø/Tag         " + "  /  ".join(
            f"{_pct(x['mean'])} (n={x['n_events']})" for x in r["halves"]))
        a = r["alpha"]
        if a:
            print(f"   K5 FF5+UMD α            {a['alpha_ann'] * 100:+.2f} %/J  t={a['t']:+.2f}  "
                  f"β_mkt={a['beta_mkt']:.2f}  SMB={a['smb']:+.2f}  HML={a['hml']:+.2f}")
        k4 = r["K4"]
        if k4 and k4["strat"]:
            s, b = k4["strat"], k4["spy"]
            print(f"   K4 Depot 3000$ CAGR {s['cagr'] * 100:+.1f}% Sharpe {s['sharpe']:.2f} "
                  f"MaxDD {s['maxdd'] * 100:.0f}%  |  SPY {b['cagr'] * 100:+.1f}% "
                  f"Sharpe {b['sharpe']:.2f} MaxDD {b['maxdd'] * 100:.0f}%")
        print("   Kriterien: " + "  ".join(f"{k}={'✓' if v else '✗'}" for k, v in c.items())
              + f"  → Kern {'PASS' if r['core_pass'] else 'FAIL'}")
    print(f"\nK3 bestanden für Haltedauern: {res['K3_passing_holds'] or '—'}")
    print(f"VERDICT H10{res['hypothesis']} {res['window']}: {res['verdict']}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", choices=["a", "b"], required=True)
    ap.add_argument("--vault", action="store_true")
    ap.add_argument("--hold", default="20,60,120")
    args = ap.parse_args()
    holds = [int(x) for x in args.hold.split(",")]
    flag = CACHE / f"phase10_dev_{args.hypothesis}.json"
    if args.vault:
        if not flag.exists() or json.load(open(flag)).get("verdict") != "PASS":
            sys.exit("Vault gesperrt: Dev-Lauf nicht bestanden (research/10 §3).")
    res = evaluate(args.hypothesis, "vault" if args.vault else "dev", holds)
    report(res)
    out = CACHE / f"phase10_{'vault' if args.vault else 'dev'}_{args.hypothesis}.json"
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(f"\nErgebnis gespeichert: {out.relative_to(CACHE.parent.parent)}")


if __name__ == "__main__":
    main()
