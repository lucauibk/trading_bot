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


def build_param_grid(aggressive: bool = False) -> list:
    """Ehrlicher Post-Phantom-Sweep (Review 2026-07-02): wenige, orthogonale Achsen.

    - sl_mode: floor vs per_position — der nie sauber OOS-verglichene Kern-Streit
    - min_step_fee_multiple: der real bindende Geometrie-Parameter (pinnt den Step
      auf 2×fee×mult in allen Regimen); war nie gesweept
    Directional bleibt AUS (5 Familien negativ getestet), momentum_hold 0.

    aggressive=True (Aggressiv-Paket 2026-07-13) erweitert um:
    - dca_size_mult {1.0, 1.3, 1.6}: Martingale-Sizing auf tiefe Levels
    - runner {aus, an mit tp=3 ATR}: Gewinner trailen statt am Level kappen
    - leverage {3, 5}
    """
    sl_variants = [
        {"sl_mode": "per_position"},
        {"sl_mode": "floor", "floor_sl_atr_mult": 1.0},
    ]
    if aggressive:
        dca_axis = [1.0, 1.3, 1.6]
        runner_axis = [{"runner_enabled": False},
                       {"runner_enabled": True, "runner_tp_atr": 3.0}]
        lev_axis = [3.0, 5.0]
    else:
        dca_axis = [1.0]
        runner_axis = [{"runner_enabled": False}]
        lev_axis = [LEVERAGE]

    grid = []
    for sl, step_mult, dca, runner, lev in itertools.product(
            sl_variants, [2.0, 3.0, 4.0, 6.0], dca_axis, runner_axis, lev_axis):
        cfg = {
            **sl,
            **runner,
            "min_step_fee_multiple": step_mult,
            "dca_size_mult": dca,
            "momentum_hold_max": 0,
            "trend_filter_enabled": True,
            "min_step_pct": 0.006,
            "max_inventory_notional_mult": 1.5,
            "directional_enabled": False,
            "leverage": lev,
        }
        grid.append(cfg)
    return grid


# ── Stage A (Ultra-Bot-Plan Phase 1): Geometrie + min_step_pct erweitern ────
#
# levels_by_regime/range_atr_mult_* wurden noch nie gesweept — build_param_grid()
# haelt sie immer auf dem Dataclass-Default. 3 benannte Presets statt freiem
# Kreuzprodukt pro Regime, um die Kombinatorik nicht explodieren zu lassen.
GEOMETRY_PRESETS = {
    "current": {   # Ist-Zustand (GridParams-Default)
        "levels_by_regime": {"ranging": 14, "trending": 6, "volatile": 20},
        "range_atr_mult_trending": 2.0,
        "range_atr_mult_volatile": 1.5,
    },
    "wide_few": {  # weniger, dafuer groessere Positionen -> weniger Fee-Drag
        "levels_by_regime": {"ranging": 10, "trending": 5, "volatile": 14},
        "range_atr_mult_trending": 2.5,
        "range_atr_mult_volatile": 2.0,
    },
    "tight_many": {  # mehr, kleinere Positionen -> mehr Fills
        "levels_by_regime": {"ranging": 18, "trending": 8, "volatile": 26},
        "range_atr_mult_trending": 1.5,
        "range_atr_mult_volatile": 1.2,
    },
}
MIN_STEP_PCT_AXIS = [0.0, 0.006, 0.010]
# 2 unveraenderte floor-Basis-Configs, um die sl_mode-Frage nicht ganz zu
# verlieren, aber NICHT mit den neuen Geometrie/min_step_pct-Achsen zu
# multiplizieren (floor erreichte im Diagnose-Lauf ohnehin nie die OOS-Phase).
FLOOR_BASELINE_STEP_MULTS = [4.0, 6.0]


def _load_live_baseline_config() -> dict:
    """Aktuelle Live-Config aus config/grid_params.json, gemergt mit
    GridParams-Defaults — als BASELINE-Zeile im Sweep-Report markiert, damit
    'schlaegt eine Config den Status quo?' ueberhaupt beantwortbar ist."""
    params_path = ROOT / "config" / "grid_params.json"
    overrides = json.loads(params_path.read_text()) if params_path.exists() else {}
    from strategies.grid_params import GridParams
    baseline = GridParams.from_dict(overrides).to_dict()
    baseline["leverage"] = LEVERAGE
    baseline["_label"] = "BASELINE (live config/grid_params.json)"
    baseline["_is_baseline"] = True
    return baseline


def build_param_grid_stage_a() -> list:
    """Ultra-Bot-Plan Phase 1 / Stage A: Geometrie-Presets x min_step_pct,
    nur fuer sl_mode=per_position (einziger Modus, der im Diagnose-Lauf die
    OOS-Phase erreicht hat), plus 2 unveraenderte floor-Basis-Configs, plus
    eine gelabelte Baseline-Zeile."""
    grid = [_load_live_baseline_config()]

    for preset_name, geometry in GEOMETRY_PRESETS.items():
        for step_mult in [2.0, 3.0, 4.0, 6.0]:
            for min_step_pct in MIN_STEP_PCT_AXIS:
                cfg = {
                    "sl_mode": "per_position",
                    "runner_enabled": False,
                    **geometry,
                    "min_step_fee_multiple": step_mult,
                    "min_step_pct": min_step_pct,
                    "dca_size_mult": 1.0,
                    "momentum_hold_max": 0,
                    "trend_filter_enabled": True,
                    "max_inventory_notional_mult": 1.5,
                    "directional_enabled": False,
                    "leverage": LEVERAGE,
                    "_label": f"stage_a/{preset_name}/step{step_mult}/minpct{min_step_pct}",
                }
                grid.append(cfg)

    for step_mult in FLOOR_BASELINE_STEP_MULTS:
        grid.append({
            "sl_mode": "floor",
            "floor_sl_atr_mult": 1.0,
            "runner_enabled": False,
            "min_step_fee_multiple": step_mult,
            "dca_size_mult": 1.0,
            "momentum_hold_max": 0,
            "trend_filter_enabled": True,
            "min_step_pct": 0.006,
            "max_inventory_notional_mult": 1.5,
            "directional_enabled": False,
            "leverage": LEVERAGE,
            "_label": f"floor_baseline/step{step_mult}",
        })

    return grid


_DFS = {}
_TIMEFRAME = "1h"
_REBUILD_EVERY = 1

# Sekunden pro Timeframe (für live-treue Rebuild-Kadenz: live = alle ~15 min)
_TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600}


def _init_worker(symbols, days, timeframe, rebuild_every):
    os.environ["GRIDBOT_BACKTEST"] = "1"
    logging.disable(logging.ERROR)  # silence per-trade INFO/WARNING spam
    global _DFS, _TIMEFRAME, _REBUILD_EVERY
    _TIMEFRAME = timeframe
    _REBUILD_EVERY = rebuild_every
    from backtest.data import load_ohlcv
    _DFS = {s: load_ohlcv(s, timeframe, days) for s in symbols}


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
        m = run_backtest(strategy, df, symbol, initial_balance=INVESTMENT,
                         rebuild_every=_REBUILD_EVERY)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--timeframe", default="1h", choices=["5m", "15m", "1h"],
                        help="5m empfohlen: 1h-Close-Fills unterschätzen den Live-Churn massiv")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--top-per-slmode", type=int, default=3,
                        help="Stratifizierte OOS-Auswahl: garantiert N Configs je sl_mode in der "
                             "Testphase (der alte Sweep filterte alle floor-Configs vor OOS raus)")
    parser.add_argument("--max-dd", type=float, default=-15.0)
    parser.add_argument("--min-trades", type=int, default=100)
    parser.add_argument("--symbol", action="append", dest="symbols", default=None,
                        help="Nur diese(s) Symbol(e) sweepen (mehrfach angebbar). "
                             "Default: alle. (nightly_tune.py übergibt --symbol — "
                             "war vorher ein argparse-Crash)")
    parser.add_argument("--rank-by", choices=["calmar", "return"], default="calmar",
                        help="return: Ranking nach median_return statt Calmar "
                             "(Aggressiv-Paket; --max-dd entsprechend lockern, z.B. -35)")
    parser.add_argument("--aggressive", action="store_true",
                        help="Aggressiv-Achsen aktivieren: dca_size_mult, runner, leverage 5")
    parser.add_argument("--grid", choices=["default", "stage_a"], default="default",
                        help="stage_a: Ultra-Bot-Plan Phase 1 — Geometrie-Presets x "
                             "min_step_pct statt des Standard-2-Achsen-Grids, inkl. "
                             "gelabelter BASELINE-Zeile aus config/grid_params.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("sweep")

    out_dir = ROOT / "results" / datetime.datetime.now().strftime("sweep_%Y%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_sec = _TF_SECONDS[args.timeframe]
    # Live rebuildet alle ~15 min (GRID_REBUILD_CYCLES × CHECK_INTERVAL)
    rebuild_every = max(1, 900 // tf_sec)
    log.info("Timeframe %s → rebuild_every=%d Candles", args.timeframe, rebuild_every)

    symbols = args.symbols or SYMBOLS
    rank_key = "median_calmar" if args.rank_by == "calmar" else "median_return"
    log.info("Symbole: %s | Ranking: %s%s", symbols, rank_key,
             " | AGGRESSIV-Achsen aktiv" if args.aggressive else "")

    # Pre-warm OHLCV cache serially (avoid concurrent Binance fetches)
    from backtest.data import load_ohlcv
    dfs = {}
    for s in symbols:
        dfs[s] = load_ohlcv(s, args.timeframe, args.days)
        log.info("Data %s: %d candles (%s → %s)", s, len(dfs[s]),
                 dfs[s].index[0], dfs[s].index[-1])

    data_start = max(df.index[0] for df in dfs.values())
    data_end = min(df.index[-1] for df in dfs.values())
    split_ts = data_start + datetime.timedelta(days=args.train_days)
    # test window starts WARMUP_CANDLES early: run_backtest skips them as indicator warmup
    test_start = split_ts - datetime.timedelta(seconds=WARMUP_CANDLES * tf_sec)
    log.info("Train: %s → %s | Test (OOS): %s → %s", data_start, split_ts, split_ts, data_end)

    grid = build_param_grid_stage_a() if args.grid == "stage_a" else build_param_grid(aggressive=args.aggressive)
    log.info("Param grid (%s): %d configs × %d symbols = %d train runs",
             args.grid, len(grid), len(symbols), len(grid) * len(symbols))

    train_jobs = [
        (i, cfg, sym, "train", data_start, split_ts)
        for i, cfg in enumerate(grid) for sym in symbols
    ]

    with Pool(args.jobs, initializer=_init_worker,
              initargs=(symbols, args.days, args.timeframe, rebuild_every)) as pool:
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
                and a["n_symbols"] == len(symbols)
            ),
            key=lambda kv: kv[1][rank_key], reverse=True,
        )
        log.info("%d/%d configs pass constraints (worst_dd > %s%%, trades ≥ %d, no halt)",
                 len(ranked), len(agg), args.max_dd, args.min_trades)

        # Stratifizierte OOS-Auswahl: Top-N GLOBAL plus Top-K je sl_mode.
        # Der Sweep 2026-06-22 ließ kein einziges floor-Config in die OOS-Phase
        # (Train-Calmar-Ranking) — der sl_mode-Vergleich war damit in-sample-only.
        top = ranked[: args.top]
        seen = {cid for cid, _ in top}
        by_mode: dict = {}
        for cid, a in ranked:
            mode = a["params"].get("sl_mode", "?")
            by_mode.setdefault(mode, [])
            if len(by_mode[mode]) < args.top_per_slmode:
                by_mode[mode].append((cid, a))
        for mode, entries in by_mode.items():
            for cid, a in entries:
                if cid not in seen:
                    top.append((cid, a))
                    seen.add(cid)
        log.info("OOS-Phase: %d Configs (stratifiziert: %s)",
                 len(top), {m: len(v) for m, v in by_mode.items()})

        # BASELINE-Zeile immer in die OOS-Phase forcieren, auch wenn sie nicht
        # ins Top-N-Ranking faellt — sonst ist "schlaegt eine Config den Status
        # quo?" nicht beantwortbar (Ultra-Bot-Plan Phase 1, Baseline-Pinning).
        baseline_ids = [cid for cid, a in agg.items() if a["params"].get("_is_baseline")]
        for cid in baseline_ids:
            if cid not in seen:
                top.append((cid, agg[cid]))
                seen.add(cid)
                log.info("BASELINE force-included in OOS-Phase (cfg %d)", cid)

        oos_jobs = [
            (cid, agg[cid]["params"], sym, "test", test_start, data_end)
            for cid, _ in top for sym in symbols
        ]
        oos_results = pool.map(_run_one, oos_jobs)

    oos_agg = aggregate(oos_results)
    final = sorted(
        ((cid, oos_agg[cid]) for cid, _ in top if cid in oos_agg),
        key=lambda kv: kv[1][rank_key], reverse=True,
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

    lines = ["# Sweep Report (ehrliche Metriken — ohne Phantom-PnL, ab Commit e5170a4)", "",
             f"Symbole: {', '.join(symbols)} | {args.days}d @ {args.timeframe} "
             f"(Train {args.train_days}d / Test {args.days - args.train_days}d) "
             f"| rebuild_every={rebuild_every} | Ranking: {rank_key} | Grid: {args.grid}", "",
             "## Top-Configs: Train vs. OOS", "",
             "| # | cfg | Label | OOS Calmar | OOS Ret% | OOS worstDD | Train Calmar | Train Ret% | Params |",
             "|---|-----|-------|-----------|----------|-------------|--------------|------------|--------|"]
    for rank, (cid, oos) in enumerate(final, 1):
        tr = agg[cid]
        # leverage bleibt sichtbar — ist seit dem Aggressiv-Paket eine Sweep-Achse
        p = dict(oos["params"])
        label = p.pop("_label", "")
        p.pop("_is_baseline", None)
        if label.startswith("BASELINE"):
            label = f"**{label}**"
        lines.append(
            f"| {rank} | {cid} | {label} | {oos['median_calmar']:.2f} | {oos['median_return']:.1f} "
            f"| {oos['worst_dd']:.1f} | {tr['median_calmar']:.2f} | {tr['median_return']:.1f} "
            f"| `{json.dumps(p)}` |")
    (out_dir / "report.md").write_text("\n".join(lines))

    if final:
        winner_id, winner = final[0]
        (out_dir / "winner.json").write_text(json.dumps(winner["params"], indent=2))
        # Metriken maschinenlesbar — nightly_tune.py liest den Calmar hieraus
        # statt fehleranfällig Logzeilen zu scrapen (#128: Logs gehen nach
        # stderr, und split()[-1] hätte den Worst-DD statt Calmar geparst).
        (out_dir / "winner_meta.json").write_text(json.dumps({
            "config_id": winner_id,
            "median_calmar": winner["median_calmar"],
            "median_return": winner["median_return"],
            "worst_dd": winner["worst_dd"],
        }, indent=2))
        log.info("WINNER cfg %d: %s", winner_id, winner["params"])
        log.info("OOS: median calmar %.2f | median return %.1f%% | worst DD %.1f%%",
                 winner["median_calmar"], winner["median_return"], winner["worst_dd"])
    log.info("Report: %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
