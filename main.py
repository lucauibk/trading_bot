"""
Grid Trading Bot – entry point.

Usage:
  python3 main.py --mode paper   # simulation (default)
  python3 main.py --mode live    # real money on Kraken
"""

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pathlib import Path
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            "logs/trading_bot.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("main")


def resolve_active_grids(symbols, per_coin, settings):
    """Resolve the active symbol list and per-coin grid config from dashboard
    coin_settings. A coin with enabled=0 is dropped entirely; an enabled coin's
    max_investment can only reduce its budget below per_coin (the broker's hard
    per-symbol cash bucket), never exceed it. A symbol absent from settings
    defaults to enabled with full per_coin budget. (#114)

    settings: {symbol: {"enabled": int, "max_investment": float, ...}}
    Returns (active_symbols, grids_config).
    """
    active_symbols = [s for s in symbols if settings.get(s, {}).get("enabled", 1)]
    grids_config = [
        {"symbol": s,
         "investment": min(settings.get(s, {}).get("max_investment", per_coin), per_coin),
         "levels": 8}
        for s in active_symbols
    ]
    return active_symbols, grids_config


def main():
    parser = argparse.ArgumentParser(description="Grid Trading Bot")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip live trading confirmation (for dashboard start)")
    args = parser.parse_args()

    paper = args.mode == "paper"

    # Live safety check
    if not paper:
        api_key = os.getenv("KRAKEN_API_KEY", os.getenv("BINANCE_API_KEY", ""))
        api_secret = os.getenv("KRAKEN_API_SECRET", os.getenv("BINANCE_API_SECRET", ""))
        if not api_key:
            logger.error("No API keys found. Set KRAKEN_API_KEY and KRAKEN_API_SECRET.")
            sys.exit(1)
        if not args.no_confirm:
            confirm = input("⚠️  LIVE TRADING – real money! Continue? (ja/nein): ")
            if confirm.strip().lower() != "ja":
                logger.info("Aborted.")
                sys.exit(0)
    else:
        api_key = api_secret = ""

    # Singleton
    from core.lifecycle import acquire_singleton
    acquire_singleton()

    import config as _cfg
    _cfg.PAPER_TRADING = paper

    try:
        from dashboard.db import set_status
        set_status(running=True, mode=args.mode, strategy="grid")
    except Exception:
        pass

    # Load symbols from config.yaml
    import yaml
    try:
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        symbols = cfg.get("symbols", ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"])
        initial_investment = float(cfg.get("initial_capital", 1000))
    except Exception:
        symbols = ["SOL/USD", "ETH/USD", "AVAX/USD", "LINK/USD", "XRP/USD"]
        initial_investment = 1000.0

    # Dashboard-set starting capital (paper mode) overrides config.yaml, if set
    if paper:
        try:
            from dashboard.db import get_initial_capital
            initial_investment = get_initial_capital()
        except Exception:
            pass

    per_coin = initial_investment / len(symbols)
    # Dashboard coin_settings are the control plane in BOTH paper and live modes
    # (CLAUDE.md: "Bot liest diese Felder jede Loop-Iteration"). Load them
    # unconditionally so `enabled` and `max_investment` take effect live too.
    settings = {}
    try:
        from dashboard.db import get_all_coin_settings
        settings = {r["symbol"]: r for r in get_all_coin_settings()}
    except Exception:
        pass

    # A disabled coin is removed entirely — it must NOT trade with a fallback budget.
    # (GridStrategy.init defaults any unconfigured symbol to 40 USDT, and the engine
    # iterates the symbol list it is handed, so filtering grids_config alone is not
    # enough: the engine's symbol list must be filtered too — see below.)
    active_symbols, grids_config = resolve_active_grids(symbols, per_coin, settings)

    # Build components
    from core.context import MarketContext
    from risk.correlation import CorrelationTracker
    from risk.manager import RiskManager

    ctx = MarketContext()
    corr = CorrelationTracker()
    risk = RiskManager(corr)

    from strategies.grid import GridStrategy
    from strategies.grid_params import GridParams
    import json as _json
    _params_path = Path("config/grid_params.json")
    if _params_path.exists():
        try:
            _raw = _json.loads(_params_path.read_text())
            grid_params = GridParams.from_dict(_raw)
            logger.info("Grid params loaded from config/grid_params.json: %s", _raw)
        except Exception as _e:
            logger.warning("Failed to load grid_params.json (%s) – using defaults", _e)
            grid_params = GridParams()
    else:
        grid_params = GridParams()
    strategy = GridStrategy(grids_config, risk_manager=risk, params=grid_params)

    if paper:
        from execution.paper import PaperBroker
        broker = PaperBroker(initial_balance=initial_investment, symbols=symbols)
        try:
            from dashboard.db import load_paper_balances
            saved = load_paper_balances()
            if saved and set(saved.keys()) == set(symbols):
                broker.load_balances(saved)
                logger.info("PaperBroker: Balances aus DB geladen: %s",
                            {k: f"{v:.2f}" for k, v in saved.items()})
        except Exception as _e:
            logger.debug("PaperBroker balance restore skipped: %s", _e)
        reconciler = None
    else:
        from execution.kraken import KrakenBroker
        from execution.reconciler import Reconciler
        broker = KrakenBroker(api_key, api_secret)
        reconciler = Reconciler(broker.reconcile_fills)

    from core.engine import Engine
    # Only trade the enabled coins — disabled coins are excluded from the engine's
    # per-tick loop (and thus from strategy.init) so they never build a grid.
    engine = Engine(strategy, broker, active_symbols, ctx, reconciler,
                    initial_capital=initial_investment)

    logger.info("Starting | mode=%s | symbols=%s | capital=%.0f USDT",
                args.mode.upper(), active_symbols, initial_investment)

    engine.run()

    total_profit = sum(
        getattr(strategy.get_state(s), "total_profit", 0) for s in active_symbols
    )
    logger.info("Bot stopped | total profit: %.4f USDT", total_profit)


if __name__ == "__main__":
    main()
