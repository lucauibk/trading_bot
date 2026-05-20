"""
Haupt-Orchestrierung: Daten holen → Indikatoren → Signal → Risiko → Order.
"""

import logging
import signal
import sys
import time
from typing import Optional

from src.data.fetcher import get_ohlcv, get_ticker
from src.risk.risk_manager import RiskManager
from src.execution.broker import create_broker
from src.strategy.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)

_running = True


def _shutdown(sig, frame):
    global _running
    _running = False
    logger.info("Graceful Shutdown eingeleitet…")


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


class TradingBot:

    def __init__(self, cfg: dict, strategy_params: dict, strategy: BaseStrategy):
        self.cfg      = cfg
        self.strategy = strategy
        self.symbols  = cfg.get("symbols", [cfg["symbol"]])
        self.tf       = cfg.get("timeframe", "1h")
        self.interval = cfg.get("check_interval", 60)
        self.exchange = cfg.get("exchange", "kraken")

        self.capital  = float(cfg.get("initial_capital", 1000))
        self.broker   = create_broker(cfg.get("paper_trading", True), cfg, self.capital)
        self.risk     = RiskManager(cfg.get("risk", {}), self.capital)

        self._open: dict[str, dict] = {}  # symbol → position info

        self._notify_startup()

    def run(self):
        logger.info("="*55)
        logger.info("Bot startet | %s | Strategie: %s | %d Symbole",
                    "PAPER" if self.cfg.get("paper_trading") else "LIVE",
                    self.strategy.name(), len(self.symbols))
        logger.info("="*55)

        while _running:
            try:
                self._tick()
            except Exception as e:
                logger.error("Tick-Fehler: %s", e)
                self._notify_error(str(e))

            for _ in range(self.interval):
                if not _running:
                    break
                time.sleep(1)

        logger.info("Bot gestoppt.")

    def _tick(self):
        for symbol in self.symbols:
            try:
                ticker = get_ticker(symbol, self.exchange)
                price  = ticker["last"]
                time.sleep(0.5)

                # Exit prüfen
                if symbol in self._open:
                    self._check_exit(symbol, price)
                    continue

                # Risiko prüfen
                bal = self.broker.get_balance()
                self.capital = bal.get("USDT", self.capital)
                if not self.risk.can_trade(self.capital, list(self._open.values())):
                    continue

                # Signal holen
                df = get_ohlcv(symbol, self.tf, limit=300, exchange_id="binance")
                sig = self.strategy.generate_signal(df)
                if not sig.is_valid:
                    continue

                # Position öffnen
                size = self.risk.calculate_position_size(self.capital, sig.entry, sig.stop_loss)
                if not size.valid:
                    logger.info("%s – Position nicht möglich: %s", symbol, size.reason)
                    continue

                self.broker.market_order(symbol, "buy", size.quantity)
                self._open[symbol] = {
                    "symbol":     symbol,
                    "entry":      sig.entry,
                    "stop_loss":  sig.stop_loss,
                    "take_profit": sig.take_profit,
                    "quantity":   size.quantity,
                    "risk_usdt":  size.risk_usdt,
                    "direction":  sig.signal,
                    "reason":     sig.reason,
                }
                logger.info("OPEN  %-12s @ %.4f | SL: %.4f | TP: %.4f | %s",
                            symbol, sig.entry, sig.stop_loss, sig.take_profit, sig.reason)
                self._notify_open(symbol, sig, size.quantity)

            except Exception as e:
                logger.warning("%s – Fehler: %s", symbol, e)
                time.sleep(1)

    def _check_exit(self, symbol: str, price: float):
        pos = self._open[symbol]
        exit_reason: Optional[str] = None

        if pos["direction"] == Signal.LONG:
            if price <= pos["stop_loss"]:
                exit_reason = "stop_loss"
            elif price >= pos["take_profit"]:
                exit_reason = "take_profit"

        # Strategie-Exit (z.B. RSI Überkauft)
        if exit_reason is None:
            df = get_ohlcv(symbol, self.tf, limit=100, exchange_id="binance")
            if hasattr(self.strategy, "should_exit") and self.strategy.should_exit(df, pos["direction"]):
                exit_reason = "signal_exit"

        if exit_reason:
            self.broker.market_order(symbol, "sell", pos["quantity"])
            pnl = (price - pos["entry"]) * pos["quantity"]
            logger.info("CLOSE %-12s @ %.4f | PnL: %+.2f USDT | %s",
                        symbol, price, pnl, exit_reason)
            self._notify_close(symbol, pos, price, pnl, exit_reason)
            del self._open[symbol]

    def _notify_startup(self):
        try:
            from notifier import notify_startup
            notify_startup(self.capital)
        except Exception:
            pass

    def _notify_open(self, symbol, sig, qty):
        try:
            from notifier import notify_trade_open
            notify_trade_open(symbol, sig.signal.value, sig.entry,
                              sig.stop_loss, sig.take_profit, qty)
        except Exception:
            pass

    def _notify_close(self, symbol, pos, price, pnl, reason):
        try:
            from notifier import notify_trade_close
            notify_trade_close(symbol, pos["direction"].value, pos["entry"], price, pnl, reason)
        except Exception:
            pass

    def _notify_error(self, msg):
        try:
            from notifier import notify_error
            notify_error(msg)
        except Exception:
            pass
