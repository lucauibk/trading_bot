"""
Parameter sweep with out-of-sample validation for the grid strategy.

Train window: first --train-days; test window: the rest (OOS).
Configs are ranked on the train window (median Calmar across symbols,
hard constraints on drawdown/trades), the top ones re-run on the unseen
test window — final pick by OOS Calmar.

Usage:
  python3 scripts/sweep.py --days 180 --train-days 120
"""

import argparse
import csv
import datetime
import itertools
import json
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["GRIDBOT_BACKTEST"] = "1"

SYMBOLS = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]
INVESTMENT = 200.0
LEVERAGE = 3.0          # pinned so results don't depend on the live dashboard DB
WARMUP_CANDLES = 60     # run_backtest skips the first 60 candles


def build_param_grid() -> list:
    """~48 configs: every axis orthogonal to one implemented change."""
    sl_variants = [
        {"sl_mode": "per_position"},
        {"sl_mode": "floor", "floor_sl_atr_mult": 1.0},
        {"sl_mode": "floor", "floor_sl_atr_mult": 1.5},
    ]
    grid = []
    for sl, hold, trend, step, direc in itertools.product(
        sl_variants, [0, 2], [False, True], [0.0, 0.01], [False, True]
    ):
        cfg = {
            **sl,
            "momentum_hold_max": hold,
            "trend_filter_enabled": trend,
            "min_step_pct": step,
            "directional_enabled": direc,
            "leverage": LEVERAGE,
        }
        grid.append(cfg)
    return grid


_DFS = {}


def _init_worker(symbols, days):
    os.environ["GRIDBOT_BACKTEST"] = "1"
    logging.disable(logging.ERROR)  # silence per-trade INFO/WARNING spam
    global _DFS
    from backtest.data import load_ohlcv
    _DFS = {s: load_ohlcv(s, "1h", days) for s in symbols}


def _run_one(job):
    cfg_id, params_dict, symbol, phase, start_ts, end_ts = job
    from backtest.engine import run_backtest
    from strategies.grid import GridStrategy
    from strategies.grid_params import GridParams

    df = _DFS[symbol]
    df = df[(df.index >= start_ts) & (df.index < end_ts)]
    strategy = GridStrategy(
        [{"symbol": symbol, "investment": INVESTMENT, "levels": 8}],
        ml_enabled=False,
        params=GridParams.from_dict(params_dict),
    )
    try:
        m = run_backtest(strategy, df, symbol, initial_balance=INVESTMENT)
    except Exception as e:
        return {"cfg_id": cfg_id, "symbol": symbol, "phase": phase, "error": str(e)}
    return {
        "cfg_id": cfg_id, "symbol": symbol, "phase": phase, "params": params_dict,
        "return_pct": m["total_return_pct"], "pf": m["profit_factor"],
        "dd": m["max_drawdown_pct"], "calmar": m["calmar"],
        "trades": m["n_trades"], "hit": m["hit_rate_pct"], "halted": m["halted"],
    }


def aggregate(results: list) -> dict:
    """cfg_id → cross-symbol aggregate."""
    by_cfg = {}
    for r in results:
        if "error" in r:
            continue
        by_cfg.setdefault(r["cfg_id"], []).append(r)
    agg = {}
    for cfg_id, rows in by_cfg.items():
        agg[cfg_id] = {
            "params": rows[0]["params"],
            "median_calmar": median(r["calmar"] for r in rows),
            "median_return": median(r["return_pct"] for r in rows),
            "min_pf": min(r["pf"] for r in rows),
            "worst_dd": min(r["dd"] for r in rows),
            "total_trades": sum(r["trades"] for r in rows),
            "any_halted": any(r["halted"] for r in rows),
            "n_symbols": len(rows),
        }
    return agg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--max-dd", type=float, default=-15.0)
    parser.add_argument("--min-trades", type=int, default=100)
    # nightly_tune.run_sweep() invokes this per symbol with --symbol; without this
    # argument argparse aborted every nightly sweep with SystemExit(2) (#101).
    parser.add_argument("--symbol", type=str, default=None,
                        help="Restrict the sweep to a single symbol (default: all SYMBOLS)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    global SYMBOLS
    if args.symbol:
        if args.symbol not in SYMBOLS:
            parser.error(f"unknown --symbol {args.symbol!r}; known: {', '.join(SYMBOLS)}")
        SYMBOLS = [args.symbol]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("sweep")

    out_dir = ROOT / "results" / datetime.datetime.now().strftime("sweep_%Y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-warm OHLCV cache serially (avoid concurrent Binance fetches)
    from backtest.data import load_ohlcv
    dfs = {}
    for s in SYMBOLS:
        dfs[s] = load_ohlcv(s, "1h", args.days)
        log.info("Data %s: %d candles (%s → %s)", s, len(dfs[s]),
                 dfs[s].index[0], dfs[s].index[-1])

    data_start = max(df.index[0] for df in dfs.values())
    data_end = min(df.index[-1] for df in dfs.values())
    split_ts = data_start + datetime.timedelta(days=args.train_days)
    # test window starts WARMUP_CANDLES early: run_backtest skips them as indicator warmup
    test_start = split_ts - datetime.timedelta(hours=WARMUP_CANDLES)
    log.info("Train: %s → %s | Test (OOS): %s → %s", data_start, split_ts, split_ts, data_end)

    grid = build_param_grid()
    log.info("Param grid: %d configs × %d symbols = %d train runs",
             len(grid), len(SYMBOLS), len(grid) * len(SYMBOLS))

    train_jobs = [
        (i, cfg, sym, "train", data_start, split_ts)
        for i, cfg in enumerate(grid) for sym in SYMBOLS
    ]

    with Pool(args.jobs, initializer=_init_worker, initargs=(SYMBOLS, args.days)) as pool:
        train_results = pool.map(_run_one, train_jobs)

        errors = [r for r in train_results if "error" in r]
        if errors:
            log.warning("%d runs failed, e.g.: %s", len(errors), errors[0])

        agg = aggregate(train_results)
        ranked = sorted(
            (
                (cid, a) for cid, a in agg.items()
                if a["worst_dd"] > args.max_dd
                and a["total_trades"] >= args.min_trades
                and not a["any_halted"]
                and a["n_symbols"] == len(SYMBOLS)
            ),
            key=lambda kv: kv[1]["median_calmar"], reverse=True,
        )
        log.info("%d/%d configs pass constraints (worst_dd > %s%%, trades ≥ %d, no halt)",
                 len(ranked), len(agg), args.max_dd, args.min_trades)
        top = ranked[: args.top]

        oos_jobs = [
            (cid, agg[cid]["params"], sym, "test", test_start, data_end)
            for cid, _ in top for sym in SYMBOLS
        ]
        oos_results = pool.map(_run_one, oos_jobs)

    oos_agg = aggregate(oos_results)
    final = sorted(
        ((cid, oos_agg[cid]) for cid, _ in top if cid in oos_agg),
        key=lambda kv: kv[1]["median_calmar"], reverse=True,
    )

    # ── Outputs ──────────────────────────────────────────────────────────
    all_rows = [r for r in train_results + oos_results if "error" not in r]
    with open(out_dir / "all_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "cfg_id", "phase", "symbol", "return_pct", "pf", "dd",
            "calmar", "trades", "hit", "halted", "params"])
        w.writeheader()
        for r in all_rows:
            w.writerow({**{k: r[k] for k in w.fieldnames if k != "params"},
                        "params": json.dumps(r["params"])})

    lines = ["# Sweep Report", "",
             f"Symbole: {', '.join(SYMBOLS)} | {args.days}d "
             f"(Train {args.train_days}d / Test {args.days - args.train_days}d) | Leverage {LEVERAGE}×", "",
             "## Top-Configs: Train vs. OOS (Ranking nach OOS-median-Calmar)", "",
             "| # | cfg | OOS Calmar | OOS Ret% | OOS worstDD | Train Calmar | Train Ret% | Params |",
             "|---|-----|-----------|----------|-------------|--------------|------------|--------|"]
    for rank, (cid, oos) in enumerate(final, 1):
        tr = agg[cid]
        p = {k: v for k, v in oos["params"].items() if k != "leverage"}
        lines.append(
            f"| {rank} | {cid} | {oos['median_calmar']:.2f} | {oos['median_return']:.1f} "
            f"| {oos['worst_dd']:.1f} | {tr['median_calmar']:.2f} | {tr['median_return']:.1f} "
            f"| `{json.dumps(p)}` |")
    (out_dir / "report.md").write_text("\n".join(lines))

    if final:
        winner_id, winner = final[0]
        (out_dir / "winner.json").write_text(json.dumps(winner["params"], indent=2))
        log.info("WINNER cfg %d: %s", winner_id, winner["params"])
        log.info("OOS: median calmar %.2f | median return %.1f%% | worst DD %.1f%%",
                 winner["median_calmar"], winner["median_return"], winner["worst_dd"])
    log.info("Report: %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
