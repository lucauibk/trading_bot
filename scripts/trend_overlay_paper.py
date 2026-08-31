"""
research/08 H8b — Paper-Overlay-Ledger: BTC-200D-SMA-Trend-Switch forward vs Buy-and-Hold.

Overlay-Portfolio: 50 % BTC + 50 % ETH, ABER wenn BTC-Tagesschluss <= 200D-SMA →
komplett USDT (Cash, 0 % modelliert). Signal = BTC-Regime als markt-weiter
Risikoschalter (leichte Erweiterung des in research/08 getesteten BTC-only-H8b).
Benchmark: dasselbe BTC/ETH-Korb, nie geschaltet.

Fee 0,15 %/Seite auf getauschtes Notional bei jedem Flip.

Idempotent, für Cron gebaut. Einmalig backfillen, dann täglich laufen lassen:
  python3 scripts/trend_overlay_paper.py --backfill 2026-01-01   # Seed (einmalig)
  python3 scripts/trend_overlay_paper.py                          # täglich (Cron)
  python3 scripts/trend_overlay_paper.py --status                 # nur anzeigen
"""
import argparse, datetime as dt, json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import ccxt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("overlay")

LEDGER = Path(__file__).resolve().parent.parent / "data" / "trend_overlay_ledger.json"
CAPITAL = 200.0
FEE = 0.0015
SMA_N = 200
BASKET = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}


def fetch_daily(days=520):
    b = ccxt.binance({"enableRateLimit": True})
    since = b.milliseconds() - days * 86400_000
    out = {}
    for s in BASKET:
        rows, cur = [], since
        while True:
            batch = b.fetch_ohlcv(s, "1d", since=cur, limit=1000)
            if not batch:
                break
            rows += batch
            cur = batch[-1][0] + 86400_000
            if len(batch) < 1000:
                break
        df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.normalize()
        out[s] = df.set_index("ts")["c"]
    return pd.DataFrame(out).dropna().sort_index()


def load_ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"inception": None, "rows": []}


def save_ledger(led):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=2))


def step(px_row, sma_btc, prev):
    """Ein Tagesschritt. prev = letzte Ledger-Zeile (oder None). Gibt neue Zeile."""
    btc, eth = px_row["BTC/USDT"], px_row["ETH/USDT"]
    in_market = bool(btc > sma_btc)

    if prev is None:
        # Inception: beide starten bei CAPITAL, Overlay geht nach Signal in Markt/Cash
        bench_units = {s: CAPITAL * w / px_row[s] for s, w in BASKET.items()}
        if in_market:
            ov_units = dict(bench_units)
            ov_cash = 0.0
            ov_fee = 0.0
        else:
            ov_units = {s: 0.0 for s in BASKET}
            ov_cash = CAPITAL
            ov_fee = 0.0
        pos = "MARKT" if in_market else "CASH"
    else:
        bench_units = prev["_bench_units"]
        ov_units = dict(prev["_ov_units"])
        ov_cash = prev["_ov_cash"]
        ov_fee = 0.0
        was_in = prev["in_market"]
        if in_market and not was_in:               # Cash → Markt
            invest = ov_cash * (1 - FEE)
            ov_units = {s: invest * w / px_row[s] for s, w in BASKET.items()}
            ov_fee = ov_cash * FEE
            ov_cash = 0.0
            pos = "→MARKT"
        elif was_in and not in_market:             # Markt → Cash
            gross = sum(ov_units[s] * px_row[s] for s in BASKET)
            ov_cash = gross * (1 - FEE)
            ov_fee = gross * FEE
            ov_units = {s: 0.0 for s in BASKET}
            pos = "→CASH"
        else:
            pos = "MARKT" if in_market else "CASH"

    ov_eq = ov_cash + sum(ov_units[s] * px_row[s] for s in BASKET)
    bench_eq = sum(bench_units[s] * px_row[s] for s in BASKET)
    return {
        "date": px_row.name.strftime("%Y-%m-%d"),
        "btc": round(float(px_row["BTC/USDT"]), 2),
        "btc_sma200": round(float(sma_btc), 2),
        "in_market": in_market,
        "position": pos,
        "overlay_eq": round(ov_eq, 2),
        "bench_eq": round(bench_eq, 2),
        "fee_paid": round(ov_fee, 4),
        "_bench_units": bench_units,
        "_ov_units": ov_units,
        "_ov_cash": round(ov_cash, 6),
    }


def rebuild(px: pd.DataFrame, start: dt.date):
    sma = px["BTC/USDT"].rolling(SMA_N).mean()
    rows, prev = [], None
    for ts, row in px.iterrows():
        if ts.date() < start or pd.isna(sma.loc[ts]):
            continue
        prev = step(row, sma.loc[ts], prev)
        rows.append(prev)
    return rows


def print_status(led):
    rows = led["rows"]
    if not rows:
        print("Ledger leer."); return
    r = rows[-1]
    incep = led["inception"]
    ov_ret = r["overlay_eq"] / CAPITAL - 1
    bh_ret = r["bench_eq"] / CAPITAL - 1
    flips = sum(1 for x in rows if x["position"].startswith("→"))
    fees = sum(x["fee_paid"] for x in rows)
    # max drawdown je Kurve
    def mdd(key):
        eq = np.array([x[key] for x in rows]); peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1).min())
    print(f"\n{'='*66}")
    print(f"  H8b Paper-Overlay  —  Inception {incep}  →  {r['date']}  ({len(rows)} Tage)")
    print(f"{'='*66}")
    print(f"  BTC {r['btc']:,.0f}  vs 200D-SMA {r['btc_sma200']:,.0f}   Signal: "
          f"{'IN MARKT' if r['in_market'] else 'CASH'}   (heute: {r['position']})")
    print(f"  Flips bisher: {flips}   Fees gezahlt: {fees:.2f} USDT")
    print(f"  {'-'*62}")
    print(f"  {'':<22}{'Return':>12}{'MaxDD':>12}{'Equity':>14}")
    print(f"  {'Overlay (H8b)':<22}{ov_ret:>+11.1%}{mdd('overlay_eq'):>+12.1%}{r['overlay_eq']:>13.2f}")
    print(f"  {'Buy&Hold BTC/ETH':<22}{bh_ret:>+11.1%}{mdd('bench_eq'):>+12.1%}{r['bench_eq']:>13.2f}")
    print(f"  {'Δ (Overlay − B&H)':<22}{ov_ret-bh_ret:>+11.1%}")
    print(f"{'='*66}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=str, help="ISO-Datum: Ledger von hier neu aufbauen (Seed)")
    ap.add_argument("--status", action="store_true", help="nur anzeigen, nichts ändern")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    led = load_ledger()
    if args.status:
        print_status(led); return

    px = fetch_daily()

    if args.backfill:
        start = dt.date.fromisoformat(args.backfill)
        led = {"inception": args.backfill, "rows": rebuild(px, start)}
        save_ledger(led)
        log.info("Backfill: %d Tage ab %s", len(led["rows"]), args.backfill)
        print_status(led); return

    if not led["rows"]:
        log.warning("Ledger leer — erst --backfill YYYY-MM-DD laufen lassen.")
        return

    last_date = dt.date.fromisoformat(led["rows"][-1]["date"])
    new = px[px.index.date > last_date]
    if new.empty:
        if not args.quiet:
            print_status(led)
        log.info("Keine neuen Tage seit %s.", last_date)
        return

    sma = px["BTC/USDT"].rolling(SMA_N).mean()
    prev = led["rows"][-1]
    for ts, row in new.iterrows():
        prev = step(row, sma.loc[ts], prev)
        led["rows"].append(prev)
    save_ledger(led)
    log.info("+%d Tage angehängt (bis %s)", len(new), led["rows"][-1]["date"])
    if not args.quiet:
        print_status(led)


if __name__ == "__main__":
    main()
