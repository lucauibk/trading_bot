"""
research/05 Framing #2 (H5b) — feuert der Short-Trend-Sleeve ZEITLICH in genau den
Drawdown-Episoden, die die Floor-SLs des Long-Grids kaskadieren lassen?

P1 (primär, Timing-Test): Top-k Grid-Drawdown-Episoden (< dd_threshold) je Symbol
  aus der run_backtest-Equity-Kurve. Short-Sleeve-$-PnL summiert über exakt diese
  Episoden, gepoolt über 5 Symbole. GATE: positiv UND >= 25 % des Grid-$-Verlusts
  in diesen Episoden.
P2 (sekundär): Equal-Weight kombinierte Equity (Grid + Sleeve, feste
  Pro-Symbol-Kapitalaufteilung) — höherer Terminal-Return UND flacherer max_dd als
  Grid-allein auf OOS-dev, in >= 4/5 Symbolen.
P3: Standalone-Sleeve (H5a) mind. break-even — separat in trend_breakout_backtest.py.

Nutzung (vorregistriert):
  python3 scripts/short_sleeve_portfolio.py --as-of 2026-07-22 --days 405 \
      --split-days 300 --leverage 1.0 --dd-threshold 0.05 --top-k 5
"""
import argparse, datetime, json, logging, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("GRIDBOT_BACKTEST", "1")

import numpy as np
import pandas as pd

from backtest.data import load_ohlcv
from backtest.engine import run_backtest

logger = logging.getLogger("short_portfolio")

SYMBOLS = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]
INVESTMENT = 200.0            # Grid-investment je Symbol (wie scripts/sweep.py)
SLEEVE_NOTIONAL = INVESTMENT  # vorregistriert: Sleeve-Notional == Grid-investment
WARMUP = 60                   # run_backtest überspringt die ersten 60 Kerzen

# Primary-Config des Short-Sleeves (research/05 §3)
PRIMARY_LB, PRIMARY_ATR, PRIMARY_REGIME = 20, 3.0, True


def grid_equity_series(symbol: str, df: pd.DataFrame, leverage: float):
    """run_backtest der Live-Grid-Config; gibt (equity_series, halted, metrics)."""
    from strategies.grid import GridStrategy
    from strategies.grid_params import GridParams

    params_path = Path(__file__).resolve().parent.parent / "config" / "grid_params.json"
    overrides = json.loads(params_path.read_text()) if params_path.exists() else {}
    overrides["leverage"] = leverage
    strat = GridStrategy(
        [{"symbol": symbol, "investment": INVESTMENT, "levels": 8}],
        ml_enabled=False,
        params=GridParams.from_dict(overrides),
    )
    m = run_backtest(strat, df, symbol, initial_balance=INVESTMENT)
    eq = m["equity_curve"]  # [init] + ein Wert je Kerze ab i=60
    # eq[1:] richtet sich an df.index[60:60+len(eq)-1]  (Emergency-Stop truncatet!)
    body = eq[1:]
    idx = df.index[WARMUP:WARMUP + len(body)]
    s = pd.Series(body, index=idx, dtype=float)
    if len(s) and len(s) < len(df) - WARMUP:
        # halted → eingefrorene Equity bis zum gemeinsamen Ende forward-fillen
        tail_idx = df.index[WARMUP + len(body):]
        s = pd.concat([s, pd.Series(s.iloc[-1], index=tail_idx)])
    return s, m["halted"], m


def drawdown_episodes(eq: pd.Series, threshold: float, top_k: int):
    """Maximale zusammenhängende Spannen mit Drawdown < -threshold. Top-k nach Tiefe."""
    if eq.empty:
        return []
    peak = eq.cummax()
    dd = (eq - peak) / peak
    under = dd < -threshold
    episodes = []
    in_ep = False
    for ts, u in under.items():
        if u and not in_ep:
            in_ep = True; start = ts
        elif not u and in_ep:
            in_ep = False
            seg = dd.loc[start:prev_ts]
            episodes.append((start, prev_ts, float(seg.min()),
                             float(eq.loc[start] - eq.loc[prev_ts])))
        prev_ts = ts
    if in_ep:
        seg = dd.loc[start:prev_ts]
        episodes.append((start, prev_ts, float(seg.min()),
                         float(eq.loc[start] - eq.loc[prev_ts])))
    episodes.sort(key=lambda e: e[2])  # tiefster Drawdown zuerst
    return episodes[:top_k]


def short_trades(df: pd.DataFrame, leverage: float, carry_daily: float):
    from scripts.trend_breakout_backtest import backtest
    return backtest(df, PRIMARY_LB, PRIMARY_ATR, PRIMARY_REGIME,
                    side="short", lev=leverage, carry_daily=carry_daily)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=405)
    ap.add_argument("--as-of", type=str, default=None)
    ap.add_argument("--split-days", type=int, default=300)
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--carry-daily", type=float, default=0.0)
    ap.add_argument("--dd-threshold", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    as_of_ts = None
    if args.as_of:
        as_of_ts = datetime.datetime.fromisoformat(args.as_of).replace(tzinfo=datetime.timezone.utc)
    else:
        logger.warning("!!! --as-of NICHT gesetzt — Vault NICHT geschützt.")

    data = {}
    for s in SYMBOLS:
        df = load_ohlcv(s, "1h", args.days)
        if as_of_ts is not None:
            df = df[df.index <= as_of_ts]
        data[s] = df
        logger.info("Data %s: %d Kerzen (%s → %s)", s, len(df), df.index[0], df.index[-1])

    data_start = max(df.index[0] for df in data.values())
    split_ts = data_start + datetime.timedelta(days=args.split_days)
    logger.info("split_ts (IS/OOS) = %s", split_ts)

    W = 92
    print(f"\n{'='*W}")
    print(f"  H5b — Short-Sleeve × Long-Grid Overlap  (dd<{args.dd_threshold:.0%}, top-{args.top_k}, lev={args.leverage})")
    print(f"{'='*W}")

    pool_grid_loss = 0.0
    pool_short_pnl = 0.0
    p2_better = 0
    p2_rows = []

    for s in SYMBOLS:
        df = data[s]
        eq, halted, m = grid_equity_series(s, df, args.leverage)
        st = short_trades(df, args.leverage, args.carry_daily)
        eps = drawdown_episodes(eq, args.dd_threshold, args.top_k)

        print(f"\n  {s}   Grid: return={m['total_return_pct']:+.1f}%  maxDD={m['max_drawdown_pct']:.1f}%  "
              f"PF={m['profit_factor']:.2f}  halted={halted}   Short-Trades={len(st)}")
        if not eps:
            print(f"    (keine Grid-Drawdown-Episode < {args.dd_threshold:.0%})")
        for (a, b, depth, loss_usd) in eps:
            in_ep = [t for t in st if a <= t["entry_ts"] <= b]
            short_pnl_usd = sum(t["pnl"] for t in in_ep) * SLEEVE_NOTIONAL
            pool_grid_loss += max(loss_usd, 0.0)
            pool_short_pnl += short_pnl_usd
            print(f"    {a:%Y-%m-%d} → {b:%Y-%m-%d}  depth={depth:+.1%}  grid_loss=${loss_usd:+.2f}  "
                  f"short_n={len(in_ep):>2}  short_pnl=${short_pnl_usd:+.2f}")

        # --- P2: kombinierte Equity auf OOS-dev ---
        eq_oos = eq[eq.index >= split_ts]
        if len(eq_oos) > 2:
            # Sleeve-Equity: flach kompoundieren am exit_ts
            sleeve = pd.Series(0.0, index=eq_oos.index)
            for t in st:
                if t["exit_ts"] >= split_ts and t["exit_ts"] <= eq_oos.index[-1]:
                    sleeve.loc[sleeve.index >= t["exit_ts"]] += t["pnl"] * SLEEVE_NOTIONAL
            grid_only_ret = eq_oos.iloc[-1] / eq_oos.iloc[0] - 1.0
            grid_only_dd = ((eq_oos - eq_oos.cummax()) / eq_oos.cummax()).min()
            combined = eq_oos + sleeve
            comb_ret = combined.iloc[-1] / combined.iloc[0] - 1.0
            comb_dd = ((combined - combined.cummax()) / combined.cummax()).min()
            better = comb_ret > grid_only_ret and comb_dd > grid_only_dd
            p2_better += int(better)
            p2_rows.append((s, grid_only_ret, comb_ret, grid_only_dd, comb_dd, better))

    print(f"\n{'-'*W}")
    offset = (pool_short_pnl / pool_grid_loss) if pool_grid_loss > 1e-9 else float("nan")
    print(f"  P1  gepoolt über Episoden:  grid_loss=${pool_grid_loss:.2f}  "
          f"short_pnl=${pool_short_pnl:+.2f}  offset_ratio={offset:+.2%}")
    p1 = pool_short_pnl > 0 and offset >= 0.25
    print(f"      P1 GATE (short_pnl>0 UND offset>=25%): {'PASS' if p1 else 'FAIL'}")

    print(f"\n  P2  kombinierte OOS-Equity vs Grid-allein:")
    for (s, gr, cr, gd, cd, bt) in p2_rows:
        print(f"      {s:<9} return {gr:+.2%} → {cr:+.2%}   maxDD {gd:+.2%} → {cd:+.2%}   {'besser' if bt else '—'}")
    p2 = p2_better >= 4
    print(f"      P2 GATE (>=4/5 besser in Return UND maxDD): {'PASS' if p2 else 'FAIL'}  ({p2_better}/5)")

    print(f"\n  H5b Gesamt: P1={'PASS' if p1 else 'FAIL'}  P2={'PASS' if p2 else 'FAIL'}  "
          f"P3=siehe trend_breakout_backtest.py (H5a break-even?)")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
