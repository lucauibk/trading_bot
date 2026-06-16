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

    per_coin = initial_investment / len(symbols)
    grids_config = [{"symbol": s, "investment": per_coin, "levels": 8} for s in symbols]

    # Build components
    from core.context import MarketContext
    from risk.correlation import CorrelationTracker
    from risk.manager import RiskManager

    ctx = MarketContext()
    corr = CorrelationTracker()
    risk = RiskManager(corr)

    from strategies.grid import GridStrategy
    strategy = GridStrategy(grids_config, risk_manager=risk)

    if paper:
        from execution.paper import PaperBroker
        broker = PaperBroker(initial_balance=initial_investment, symbols=symbols)
        reconciler = None
    else:
        from execution.kraken import KrakenBroker
        from execution.reconciler import Reconciler
        broker = KrakenBroker(api_key, api_secret)
        reconciler = Reconciler(broker.reconcile_fills)

    from core.engine import Engine
    engine = Engine(strategy, broker, symbols, ctx, reconciler)

    logger.info("Starting | mode=%s | symbols=%s | capital=%.0f USDT",
                args.mode.upper(), symbols, initial_investment)

    engine.run()

    total_profit = sum(
        getattr(strategy.get_state(s), "total_profit", 0) for s in symbols
    )
    logger.info("Bot stopped | total profit: %.4f USDT", total_profit)


if __name__ == "__main__":
    main()
