"""
Broker ABC – all exchange integrations implement this interface.
Strategies never call the exchange directly; the engine uses the broker.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from core.strategy import Fill, Order


@dataclass
class BrokerOrder:
    """Internal broker representation of an open order."""
    client_id: str
    exchange_order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    filled_qty: float = 0.0
    status: str = "open"    # "open" | "filled" | "cancelled" | "partial"
    ts_placed: float = 0.0
    meta: dict = field(default_factory=dict)


class Broker(ABC):

    @abstractmethod
    def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        post_only: bool = True,
        client_id: str = "",
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> BrokerOrder:
        """Place a limit order. Returns the broker order with exchange_order_id set."""

    @abstractmethod
    def cancel(self, client_id: str) -> bool:
        """Cancel an order by client_id. Returns True if successfully cancelled."""

    @abstractmethod
    def cancel_all(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count cancelled."""

    @abstractmethod
    def reconcile_fills(self, since_ts: float) -> List[Fill]:
        """Fetch fills since a unix timestamp. Used to detect fills we may have missed."""

    @abstractmethod
    def get_open_orders(self, symbol: str) -> List[BrokerOrder]:
        """Return current open orders for a symbol."""

    @abstractmethod
    def get_balance(self, currency: str = "USD") -> float:
        """Return free balance for a currency."""

    def place_market(self, symbol: str, side: str, qty: float) -> Optional[BrokerOrder]:
        """Place a market order. Not all brokers support this; default raises NotImplementedError."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support market orders")

    def round_qty(self, symbol: str, qty: float) -> float:
        """Round qty to exchange-specific precision. Default: 6 decimal places."""
        return round(qty, 6)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to exchange-specific tick size. Default: 4 decimal places."""
        return round(price, 4)
