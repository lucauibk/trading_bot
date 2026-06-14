"""
Strategy ABC – all trading strategies implement this interface.

Engine calls hooks in this order each tick:
  1. on_candle(symbol, df, ctx)   – process new OHLCV data, update internal state
  2. desired_orders(symbol, ctx)  – return list of Orders the strategy wants open
  3. on_fill(fill, ctx)           – react to a confirmed fill
  4. on_tick(symbol, price, ctx)  – check exits, trailing stops, SL/TP

The engine reconciles desired_orders() with the broker state; the strategy
never calls the broker directly – it just declares intent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

from core.context import MarketContext


@dataclass
class Order:
    symbol: str
    side: str            # "buy" | "sell"
    price: float
    qty: float
    order_type: str = "limit"
    post_only: bool = True
    client_id: str = ""  # uuid set by engine if empty
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    meta: dict = field(default_factory=dict)  # strategy-specific payload


@dataclass
class Fill:
    client_id: str
    symbol: str
    side: str
    price: float
    qty: float
    fee: float
    ts: float            # unix timestamp
    exchange_order_id: str = ""
    meta: dict = field(default_factory=dict)


class Strategy(ABC):
    """Base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def init(self, symbols: List[str], ctx: MarketContext) -> None:
        """Called once at startup. Load state, configure parameters."""

    @abstractmethod
    def on_candle(self, symbol: str, df: pd.DataFrame, ctx: MarketContext) -> None:
        """Called when a new OHLCV candle is available. Update indicators."""

    @abstractmethod
    def desired_orders(self, symbol: str, price: float, ctx: MarketContext) -> List[Order]:
        """Return the full set of orders this strategy wants open for a symbol.
        The engine will cancel orders no longer in this list and place new ones."""

    @abstractmethod
    def on_fill(self, fill: Fill, ctx: MarketContext) -> None:
        """React to a confirmed fill. Update positions, place follow-up orders."""

    @abstractmethod
    def on_tick(self, symbol: str, price: float, ctx: MarketContext) -> List[Order]:
        """Intra-candle check. Return any immediate orders (e.g. SL/TP hits,
        trailing stop adjustments). Called every engine tick."""

    def status(self, symbol: str) -> dict:
        """Optional: return a dict for dashboard display."""
        return {}
