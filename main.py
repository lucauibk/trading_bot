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

    # Deaktivierte Coins (Dashboard-Toggle ODER persistierter Emergency-Halt)
    # wirklich ausschließen — vorher bekamen sie trotz enabled=0 das volle
    # Default-Budget, der Toggle war wirkungslos (P1-Fix Review 2026-07-02).
    try:
        from dashboard.db import get_all_coin_settings
        _settings = {r["symbol"]: r for r in get_all_coin_settings()}
    except Exception:
        _settings = {}
    disabled = [s for s in symbols if s in _settings and not _settings[s]["enabled"]]
    if disabled:
        logger.warning("Symbole deaktiviert (Dashboard/Emergency-Halt): %s", disabled)
        symbols = [s for s in symbols if s not in disabled]
    if not symbols:
        logger.error("Alle Symbole deaktiviert — nichts zu handeln. Exit.")
        return

    per_coin = initial_investment / len(symbols)
    overrides = {}
    if paper:
        overrides = {sym: r["max_investment"] for sym, r in _settings.items()
                     if r["enabled"]}
    grids_config = [
        # coin_settings can only reduce budget below per_coin, never exceed the
        # broker's hard per-symbol cash bucket (initial_capital / n_symbols).
        {"symbol": s, "investment": min(overrides.get(s, per_coin), per_coin), "levels": 8}
        for s in symbols
    ]

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
    engine = Engine(strategy, broker, symbols, ctx, reconciler, initial_capital=initial_investment)

    logger.info("Starting | mode=%s | symbols=%s | capital=%.0f USDT",
                args.mode.upper(), symbols, initial_investment)

    engine.run()

    total_profit = sum(
        getattr(strategy.get_state(s), "total_profit", 0) for s in symbols
    )
    logger.info("Bot stopped | total profit: %.4f USDT", total_profit)


if __name__ == "__main__":
    main()
