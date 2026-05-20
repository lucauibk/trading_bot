"""
Einstiegspunkt für den Trading Bot.

Verwendung:
  python3 main.py --mode paper --strategy ema
  python3 main.py --mode paper --strategy rsi
  python3 main.py --mode paper --strategy grid
  python3 main.py --mode live  --strategy grid
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Logs
Path("logs").mkdir(exist_ok=True)

# ── Singleton-Schutz: verhindert mehrere Bot-Instanzen ───────────────────────
_PIDFILE = Path(".bot.pid")
if _PIDFILE.exists():
    try:
        _existing_pid = int(_PIDFILE.read_text().strip())
        if _existing_pid != os.getpid():
            import os as _os
            _os.kill(_existing_pid, 0)  # prüfen ob Prozess noch läuft
            print(f"[main] Bot läuft bereits (PID {_existing_pid}) – Abbruch.")
            sys.exit(0)
    except (ProcessLookupError, ValueError):
        pass  # Prozess tot → PID-File veraltet, weitermachen
_PIDFILE.write_text(str(os.getpid()))

from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            "logs/trading_bot.log", maxBytes=5*1024*1024, backupCount=2, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("main")


def load_config() -> tuple[dict, dict]:
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("config/strategy_params.yaml") as f:
        params = yaml.safe_load(f)

    # API Keys aus Umgebungsvariablen einlesen (NIEMALS hardcoden)
    cfg["api_key"]    = os.getenv("KRAKEN_API_KEY", os.getenv("BINANCE_API_KEY", ""))
    cfg["api_secret"] = os.getenv("KRAKEN_API_SECRET", os.getenv("BINANCE_API_SECRET", ""))

    # Telegram aus Env überschreiben wenn gesetzt
    if os.getenv("TELEGRAM_TOKEN"):
        cfg.setdefault("telegram", {})["token"]   = os.getenv("TELEGRAM_TOKEN")
        cfg.setdefault("telegram", {})["chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")

    return cfg, params


def build_strategy(name: str, params: dict):
    if name == "ema":
        from src.strategy.ema_crossover import EMACrossoverStrategy
        return EMACrossoverStrategy(params["ema_crossover"])
    if name == "rsi":
        from src.strategy.rsi_mean_rev import RSIMeanRevStrategy
        return RSIMeanRevStrategy(params["rsi_mean_rev"])
    if name == "grid":
        return None  # Grid-Bot hat eigene Laufschleife
    raise ValueError(f"Unbekannte Strategie: {name}")


def run_grid(paper: bool):
    """Grid-Bot mit eigener Multi-Coin Logik starten."""
    import importlib.util, sys as _sys
    spec = importlib.util.spec_from_file_location("grid_bot", "grid_bot.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Paper/Live Flag temporär setzen
    import config as _cfg
    _cfg.PAPER_TRADING = paper
    mod.run()


def main():
    parser = argparse.ArgumentParser(description="Krypto Trading Bot")
    parser.add_argument("--mode",     choices=["paper", "live"], default="paper",
                        help="paper = Simulation, live = echtes Geld")
    parser.add_argument("--strategy", choices=["ema", "rsi", "grid"], default="grid",
                        help="Handelsstrategie")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Live-Bestätigung überspringen (für Dashboard-Start)")
    args = parser.parse_args()

    cfg, params = load_config()

    paper = args.mode == "paper"
    if not paper:
        if not cfg.get("api_key"):
            logger.error("Keine API Keys gefunden! Setze KRAKEN_API_KEY und KRAKEN_API_SECRET.")
            sys.exit(1)
        if not args.no_confirm:
            confirm = input("⚠️  LIVE TRADING – echtes Geld! Fortfahren? (ja/nein): ")
            if confirm.strip().lower() != "ja":
                logger.info("Abgebrochen.")
                sys.exit(0)

    cfg["paper_trading"] = paper
    logger.info("Modus: %s | Strategie: %s", args.mode.upper(), args.strategy.upper())

    try:
        from dashboard.db import set_status
        set_status(running=True, mode=args.mode, strategy=args.strategy)
    except Exception:
        pass

    if args.strategy == "grid":
        run_grid(paper)
        return

    strategy = build_strategy(args.strategy, params)
    from src.bot import TradingBot
    bot = TradingBot(cfg, params, strategy)
    bot.run()


if __name__ == "__main__":
    main()
