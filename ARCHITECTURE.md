# Architecture Overview

Quick-reference architecture map, one level above `CLAUDE.md`'s detailed doc.
Written after building `nq-video-bot/` (a deliberately minimal comparison bot)
made the onboarding cost of this project's layered design more visible —
see `nq-video-bot/docs/COMPARISON_REPORT.md`.

## Process topology

```
start.sh
  ├── dashboard/app.py   (Flask :5001, background process)
  └── main.py --mode paper|live   (background process)

Browser → dashboard (API calls + SSE /stream)
Dashboard ⟷ Bot: exclusively via SQLite data/trades.db (no direct IPC)
```

## Core event loop (`core/engine.py`)

```
every CHECK_INTERVAL (15s):
    read bot_status (leverage, stop_mode) + coin_settings from DB
    build MarketContext (core/context.py): BTC regime, funding, equity, positions
    for each symbol:
        PricePredictor → grid range (ATR/Bollinger)
        every PREDICTION_RECHECK ticks: ml/predictor.py → direction (LightGBM + Claude Haiku blend)
        strategies/grid.py.desired_orders() → target order book
        risk/manager.py.can_open() → pre-trade gate (daily DD, position caps, correlation)
        execution/paper.py or execution/kraken.py → place/cancel orders
    write trades/grid_state/equity/predictions to DB
```

## Layer responsibilities

| Layer | Purpose | Key files |
|---|---|---|
| Strategy | Grid logic, SL/TP, directional trades | `strategies/grid.py`, `grid_params.py` |
| Risk | Daily drawdown brake, position/correlation caps, sizing | `risk/manager.py`, `sizing.py`, `correlation.py` |
| ML | 34-feature LightGBM + Claude Haiku blend, walk-forward OOS gate | `ml/predictor.py`, `ml/model.py`, `ml/trainer.py` |
| Execution | Broker abstraction (paper vs. live Kraken) | `execution/paper.py`, `kraken.py`, `reconciler.py` |
| Market context | BTC regime, funding rate cache | `market/btc_context.py`, `perp.py` |
| Backtest | OOS parameter sweep engine | `backtest/engine.py`, `data.py`, `metrics.py` |
| Dashboard | Flask UI, SSE stream, bot process management | `dashboard/app.py`, `db.py` |

## Why the layering exists

Each abstraction earns its place from a specific constraint, not from
speculative design:

- **Strategy ABC + Broker ABC** (`core/strategy.py`, `execution/broker.py`):
  the same `GridStrategy` runs unchanged against `PaperBroker` and
  `KrakenBroker` — this is what makes backtest/paper/live share one code path.
- **Walk-forward OOS gate** (`MIN_OOS_F1 = 0.30` in `ml/trainer.py`): models
  are only saved if they clear this bar, because in-sample F1 alone
  overstates a model that hasn't been tested on unseen data.
- **Cross-coin daily drawdown brake** (`risk/manager.py`): a single circuit
  breaker across the whole portfolio, not per-symbol, because correlated
  coins can all draw down together (see `risk/correlation.py` bucketing).
- **Floor-stop-loss instead of per-position SL** (`strategies/grid.py`):
  a single SL level below the grid floor prevents individual level SLs from
  cascading into a dump — the per-position mode exists only as a backtest
  fallback (see `CLAUDE.md`'s Stop-Loss-Design section).

## Where to start reading

New to this repo (or a fresh Claude session)? Read in this order:
1. `CLAUDE.md` — full conventions, constants, DB schema, dashboard API.
2. `core/engine.py` — the actual event loop tying everything together.
3. `strategies/grid.py` — where most trading logic lives.
4. `risk/manager.py` — the one file every new strategy must integrate with.

For a look at how much this can be stripped down for a single-instrument,
rule-based, no-ML build, see `nq-video-bot/` — a parallel, deliberately naive
comparison project (not part of this bot's runtime).
