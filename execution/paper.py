"""
PaperBroker – realistic paper trading simulation.

Balance accounting
------------------
Each symbol gets an isolated budget (initial_balance / n_symbols), so no coin
can starve the others.  Orders sized with leverage use *margin* accounting:

  buy  → deduct  fill_price × qty / leverage  +  fee
  sell → credit  bought_at  × qty / leverage  +  (fill_price − bought_at) × qty  −  fee
         = return of margin + leveraged P&L

Pre-seeded sell orders (placed during grid setup without a real buy fill) never
had margin deposited and therefore credit NOTHING on fill — on a real spot
exchange a sell without inventory is rejected, so any credit here would be
phantom profit (P0-Fix Review 2026-07-02):
  pre_seeded sell → credit  0

Fallback: if leverage/bought_at are not in order.meta the broker falls back to
the simple full-notional model (backward-compatible with tests that don't set meta).
"""

import logging
import time
import uuid
from typing import Dict, List, Optional

from core.strategy import Fill, Order
from execution.broker import Broker, BrokerOrder

logger = logging.getLogger(__name__)

KRAKEN_FEE   = 0.0016   # 0.16% maker fee
SLIPPAGE_BPS = 3        # 3 basis points slippage


class PaperBroker(Broker):

    def __init__(
        self,
        initial_balance: float = 1000.0,
        symbols: Optional[List[str]] = None,
    ):
        # Per-symbol balance isolation: each coin has its own cash bucket.
        # This prevents the first symbol in the list from consuming the whole pool.
        if symbols:
            per_coin = initial_balance / len(symbols)
            self._balances: Dict[str, float] = {s: per_coin for s in symbols}
        else:
            self._balances = {}

        # Fallback single-pool (used when symbol not in _balances)
        self._balance: float = initial_balance

        self._orders:          Dict[str, BrokerOrder] = {}
        self._fill_callbacks:  list                   = []
        self._tick:            int                    = 0

    # ── Internal balance helpers ──────────────────────────────────────────

    def _sym_balance(self, symbol: str) -> float:
        """Free cash available for *symbol*."""
        return self._balances.get(symbol, self._balance)

    def _deduct(self, symbol: str, amount: float) -> None:
        if symbol in self._balances:
            self._balances[symbol] -= amount
        else:
            self._balance -= amount

    def _credit(self, symbol: str, amount: float) -> None:
        if symbol in self._balances:
            self._balances[symbol] += amount
        else:
            self._balance += amount

    # ── Broker interface ──────────────────────────────────────────────────

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
        meta: Optional[dict] = None,
    ) -> BrokerOrder:
        if not client_id:
            client_id = str(uuid.uuid4())
        extra = meta or {}
        order = BrokerOrder(
            client_id=client_id,
            exchange_order_id=client_id,
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            status="open",
            ts_placed=time.time(),
            meta={"sl": sl_price, "tp": tp_price, "placed_tick": self._tick, **extra},
        )
        self._orders[client_id] = order
        logger.debug("[PAPER] placed %s %s %s qty=%.6f @ %.4f",
                     symbol, side, client_id[:8], qty, price)
        return order

    def cancel(self, client_id: str) -> bool:
        # Gecancelte Orders sofort entfernen: _orders wuchs sonst unbegrenzt und
        # update_price iterierte über jede jemals platzierte Order — O(n²) über
        # einen Dauerlauf bzw. Backtest (Perf-Fix Review 2026-07-02).
        order = self._orders.get(client_id)
        if order and order.status == "open":
            order.status = "cancelled"
            del self._orders[client_id]
            logger.debug("[PAPER] cancelled %s", client_id[:8])
            return True
        return False

    def cancel_all(self, symbol: str) -> int:
        count = 0
        for cid, order in list(self._orders.items()):
            if order.symbol == symbol and order.status == "open":
                order.status = "cancelled"
                del self._orders[cid]
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
                slippage  = price * SLIPPAGE_BPS / 10_000
                fill_price = (order.price + slippage) if order.side == "buy" else (order.price - slippage)
                fee        = fill_price * order.qty * KRAKEN_FEE

                if order.side == "buy":
                    leverage = float(order.meta.get("leverage", 1.0))
                    if leverage <= 0:
                        leverage = 1.0  # Guard: sonst negative cost → Credit beim Buy (#37)
                    # Margin deposit = notional / leverage (e.g. lev=3 → 1/3 of notional)
                    cost = fill_price * order.qty / leverage + fee
                    sym_bal = self._sym_balance(symbol)
                    if cost > sym_bal:
                        logger.warning(
                            "[PAPER] insufficient balance %.2f < %.2f for %s buy",
                            sym_bal, cost, symbol,
                        )
                        continue
                    self._deduct(symbol, cost)

                else:  # sell
                    leverage   = float(order.meta.get("leverage", 1.0))
                    if leverage <= 0:
                        leverage = 1.0
                    pre_seeded = bool(order.meta.get("pre_seeded", False))
                    bought_at  = float(order.meta.get("bought_at", fill_price))

                    if pre_seeded:
                        # Pre-seeded Sells hatten nie einen Buy → es gibt nichts
                        # zu krediteren. Jede Gutschrift wäre Cash aus dem Nichts,
                        # das auf Kraken Spot (Sell ohne Bestand wird abgelehnt)
                        # unmöglich ist (P0-Fix Review 2026-07-02).
                        credit = 0.0
                    else:
                        # Return margin + leveraged P&L
                        margin_return = bought_at * order.qty / leverage
                        pnl           = (fill_price - bought_at) * order.qty
                        credit        = margin_return + pnl - fee

                    self._credit(symbol, credit)

                order.status     = "filled"
                order.filled_qty = order.qty
                # Gefüllte Orders aus dem Buch entfernen (siehe cancel):
                # der Fill wird unten als Fill-Objekt zurückgegeben, die Order
                # selbst wird nie wieder gebraucht.
                del self._orders[order.client_id]

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
        return [o for o in self._orders.values()
                if o.symbol == symbol and o.status == "open"]

    def load_balances(self, balances: dict) -> None:
        """Restore per-symbol balances from a previous session."""
        for sym, val in balances.items():
            if sym in self._balances:
                self._balances[sym] = float(val)

    def sl_credit(self, symbol: str, amount: float) -> None:
        """Credit the margin + PnL back to the symbol bucket after a strategy-side SL.

        Strategy-handled stop losses never route through the broker's normal fill path,
        so the margin deducted on the original buy fill would otherwise be lost permanently.
        """
        self._credit(symbol, amount)
        logger.debug("[PAPER] SL credit %.4f for %s", amount, symbol)

    def get_balance(self, currency: str = "USD") -> float:
        """Total cash across all symbol buckets."""
        if self._balances:
            return sum(self._balances.values())
        return self._balance

    def round_qty(self, symbol: str, qty: float) -> float:
        return round(qty, 6)

    def round_price(self, symbol: str, price: float) -> float:
        return round(price, 4)
