"""
Broker-Interface: PaperBroker (Simulation) und LiveBroker (Kraken via ccxt).
Beide haben dieselbe API – einfach austauschen.
"""

import logging
import time
import uuid
from typing import Optional

import ccxt

logger = logging.getLogger(__name__)

SLIPPAGE    = 0.0005   # 0.05% Slippage-Simulation
PAPER_FEE   = 0.001    # 0.1% Paper-Kommission
LIVE_FEE    = 0.002    # ~0.2% Kraken Taker-Fee


class PaperBroker:
    """Simuliert Orders lokal ohne echtes Geld."""

    def __init__(self, initial_capital: float):
        self._capital   = initial_capital
        self._positions: dict[str, dict] = {}
        self._history:   list[dict]      = []

    def get_balance(self) -> dict:
        unrealized = sum(
            (p["current_price"] - p["entry"]) * p["quantity"]
            for p in self._positions.values()
            if "current_price" in p
        )
        return {"USDT": self._capital, "unrealized_pnl": unrealized}

    def market_order(self, symbol: str, side: str, quantity: float) -> dict:
        from src.data.fetcher import get_ticker
        price = get_ticker(symbol)["last"]
        fill  = price * (1 + SLIPPAGE if side == "buy" else 1 - SLIPPAGE)
        fee   = fill * quantity * PAPER_FEE
        order = self._make_order(symbol, side, quantity, fill, "market", fee)

        if side == "buy":
            self._capital -= fill * quantity + fee
            self._positions[symbol] = {
                "symbol": symbol, "entry": fill,
                "quantity": quantity, "fee": fee,
            }
        else:
            pos = self._positions.pop(symbol, {})
            pnl = (fill - pos.get("entry", fill)) * quantity - fee
            self._capital += fill * quantity - fee
            order["pnl"] = pnl

        self._history.append(order)
        logger.info("[PAPER] %s %s %s | %.6f @ %.4f | Fee: %.4f",
                    side.upper(), symbol, order["id"][:8], quantity, fill, fee)
        return order

    def limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        fee   = price * quantity * PAPER_FEE
        order = self._make_order(symbol, side, quantity, price, "limit", fee)
        self._history.append(order)
        logger.info("[PAPER] LIMIT %s %s | %.6f @ %.4f", side.upper(), symbol, quantity, price)
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        logger.info("[PAPER] Storno %s", order_id[:8])
        return True

    def get_open_positions(self) -> list:
        return list(self._positions.values())

    def get_history(self) -> list:
        return self._history

    @staticmethod
    def _make_order(symbol, side, quantity, price, order_type, fee) -> dict:
        return {
            "id":       str(uuid.uuid4()),
            "symbol":   symbol,
            "side":     side,
            "type":     order_type,
            "quantity": quantity,
            "price":    price,
            "fee":      fee,
            "status":   "closed",
        }


class LiveBroker:
    """Echter Broker via ccxt (Kraken)."""

    def __init__(self, exchange_id: str, api_key: str, api_secret: str):
        cls = getattr(ccxt, exchange_id)
        self._ex = cls({
            "apiKey":          api_key,
            "secret":          api_secret,
            "enableRateLimit": True,
            "options":         {"defaultType": "spot"},
        })

    def get_balance(self) -> dict:
        bal = self._ex.fetch_balance()
        return {k: v for k, v in bal["free"].items() if v and v > 0}

    def market_order(self, symbol: str, side: str, quantity: float) -> dict:
        order = self._ex.create_market_order(symbol, side, quantity)
        logger.info("[LIVE] %s %s | %.6f | ID: %s", side.upper(), symbol, quantity, order["id"])
        time.sleep(0.5)
        return order

    def limit_order(self, symbol: str, side: str, quantity: float, price: float) -> dict:
        order = self._ex.create_limit_order(symbol, side, quantity, price)
        logger.info("[LIVE] LIMIT %s %s | %.6f @ %.4f | ID: %s",
                    side.upper(), symbol, quantity, price, order["id"])
        time.sleep(0.3)
        return order

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._ex.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning("Storno-Fehler %s: %s", order_id, e)
            return False

    def fetch_closed_orders(self, symbol: str, limit: int = 20) -> list:
        return self._ex.fetch_closed_orders(symbol, limit=limit)

    def get_open_positions(self) -> list:
        return []  # Kraken Spot: Positionen = Balance-Check


def create_broker(paper: bool, cfg: dict, initial_capital: float):
    """Factory – gibt PaperBroker oder LiveBroker zurück."""
    if paper:
        return PaperBroker(initial_capital)
    return LiveBroker(
        cfg["exchange"],
        cfg.get("api_key", ""),
        cfg.get("api_secret", ""),
    )
