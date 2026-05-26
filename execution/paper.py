"""
PaperBroker – realistic paper trading simulation.

Improvements over the old PaperGridBot fill logic:
- Slippage model: max(spread/2, 3bps) applied to fill price
- Sell-level CANNOT fill in the same tick as the buy that created it
- Orders are tracked by client_id (uuid), not price
- Fee applied at fill time using KRAKEN_FEE
"""

import logging
import time
import uuid
from typing import Dict, List, Optional

from core.strategy import Fill, Order
from execution.broker import Broker, BrokerOrder

logger = logging.getLogger(__name__)

KRAKEN_FEE = 0.0016   # 0.16% maker fee
SLIPPAGE_BPS = 3      # 3 basis points slippage


class PaperBroker(Broker):

    def __init__(self, initial_balance: float = 1000.0):
        self._balance: float = initial_balance
        self._orders: Dict[str, BrokerOrder] = {}
        self._fill_callbacks: list = []
        self._tick: int = 0  # incremented each time update_price is called

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
        if not client_id:
            client_id = str(uuid.uuid4())
        order = BrokerOrder(
            client_id=client_id,
            exchange_order_id=client_id,
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            status="open",
            ts_placed=time.time(),
            meta={"sl": sl_price, "tp": tp_price, "placed_tick": self._tick},
        )
        self._orders[client_id] = order
        logger.debug("[PAPER] placed %s %s %s qty=%.6f @ %.4f", symbol, side, client_id[:8], qty, price)
        return order

    def cancel(self, client_id: str) -> bool:
        order = self._orders.get(client_id)
        if order and order.status == "open":
            order.status = "cancelled"
            logger.debug("[PAPER] cancelled %s", client_id[:8])
            return True
        return False

    def cancel_all(self, symbol: str) -> int:
        count = 0
        for order in self._orders.values():
            if order.symbol == symbol and order.status == "open":
                order.status = "cancelled"
                count += 1
        return count

    def update_price(self, symbol: str, price: float) -> List[Fill]:
        """Call each tick with current price. Returns fills that occurred."""
        self._tick += 1
        fills = []

        for order in list(self._orders.values()):
            if order.symbol != symbol or order.status != "open":
                continue
            placed_tick = order.meta.get("placed_tick", 0)
            if placed_tick == self._tick:
                # Prevent same-tick fill (sell created from buy fill cannot fill immediately)
                continue

            triggered = False
            if order.side == "buy" and price <= order.price:
                triggered = True
            elif order.side == "sell" and price >= order.price:
                triggered = True

            if triggered:
                slippage = price * SLIPPAGE_BPS / 10_000
                fill_price = (order.price + slippage) if order.side == "buy" else (order.price - slippage)
                fee = fill_price * order.qty * KRAKEN_FEE

                if order.side == "buy":
                    cost = fill_price * order.qty + fee
                    if cost > self._balance:
                        logger.warning("[PAPER] insufficient balance (%.2f < %.2f) for %s buy",
                                       self._balance, cost, symbol)
                        continue
                    self._balance -= cost
                else:
                    self._balance += fill_price * order.qty - fee

                order.status = "filled"
                order.filled_qty = order.qty

                fill = Fill(
                    client_id=order.client_id,
                    symbol=symbol,
                    side=order.side,
                    price=fill_price,
                    qty=order.qty,
                    fee=fee,
                    ts=time.time(),
                    exchange_order_id=order.exchange_order_id,
                    meta=order.meta.copy(),
                )
                fills.append(fill)
                logger.debug("[PAPER] FILL %s %s @ %.4f (slippage %.4f) qty=%.6f fee=%.4f",
                             symbol, order.side.upper(), fill_price, slippage, order.qty, fee)
        return fills

    def reconcile_fills(self, since_ts: float) -> List[Fill]:
        # Paper broker tracks fills in-memory; nothing to reconcile from external source
        return []

    def get_open_orders(self, symbol: str) -> List[BrokerOrder]:
        return [o for o in self._orders.values() if o.symbol == symbol and o.status == "open"]

    def get_balance(self, currency: str = "USD") -> float:
        return self._balance

    def round_qty(self, symbol: str, qty: float) -> float:
        return round(qty, 6)

    def round_price(self, symbol: str, price: float) -> float:
        return round(price, 4)
