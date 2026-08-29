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
had margin deposited, so they credit only the profit leg:
  pre_seeded sell → credit  (fill_price − bought_at) × qty  −  fee

Fallback: if leverage/bought_at are not in order.meta the broker falls back to
the simple full-notional model (backward-compatible with tests that don't set meta).
"""

import logging
import time
import uuid
from collections import deque
from typing import Dict, List, Optional

from core.strategy import Fill, Order
from execution.broker import Broker, BrokerOrder

logger = logging.getLogger(__name__)

KRAKEN_FEE   = 0.0016   # 0.16% maker fee
SLIPPAGE_BPS = 3        # 3 basis points slippage

# #135: cap how many *terminal* (filled/cancelled) orders are retained in
# ``_orders`` for queryability. Open orders are never subject to this cap — they
# live in ``_open_orders`` and are removed only when they go terminal. Production
# never reads a terminal order back (``cancel()`` is an idempotent no-op on an
# absent id), so this only bounds memory; the window is large enough that any
# same-tick read of a just-terminated order (as the tests do) is always served.
TERMINAL_RETAIN = 5000


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

        # Fallback single-pool (used when symbol not in _balances).
        # When per-symbol buckets already sum to initial_balance, this pool MUST be
        # 0.0 — otherwise it double-provisions the account and an unseeded symbol
        # (string mismatch, or a coin enabled after startup) transacts against a
        # hidden full-account pool, breaking per-coin isolation (#149). With 0.0 the
        # buy affordability guard rejects any unseeded-symbol buy instead.
        self._balance: float = 0.0 if self._balances else initial_balance

        # #135: ``_orders`` is the full (open + recently-terminal) map kept for
        # queryability; ``_open_orders`` holds ONLY resting orders and is the one
        # the per-tick price scan iterates, so update_price is O(open orders) per
        # symbol instead of O(every order ever placed). ``_terminal_ids`` is a
        # bounded FIFO of terminal client_ids: when it overflows, the oldest
        # terminal order is evicted from ``_orders`` so memory stays bounded.
        self._orders:          Dict[str, BrokerOrder] = {}
        self._open_orders:     Dict[str, BrokerOrder] = {}
        self._terminal_ids:    deque                  = deque(maxlen=TERMINAL_RETAIN)
        # #237: O(1) membership companion to _terminal_ids so place_limit can
        # cheaply detect (and clear) a reused-cid's stale terminal entry without
        # scanning the deque on every placement.
        self._terminal_set:    set                    = set()
        self._fill_callbacks:  list                   = []
        self._tick:            int                    = 0

    # ── Internal order-lifecycle helpers ──────────────────────────────────

    def _retire(self, order: BrokerOrder) -> None:
        """Move an order out of the open set once it is filled/cancelled (#135).

        The order stays in ``_orders`` (queryable) until it is pushed out of the
        bounded ``_terminal_ids`` FIFO, at which point it is dropped to keep the
        map from growing without bound over a long-running session.
        """
        self._open_orders.pop(order.client_id, None)
        if len(self._terminal_ids) == self._terminal_ids.maxlen:
            evicted = self._terminal_ids[0]  # about to be pushed out by append
            # #237: NEVER drop a cid that is currently OPEN again. client_ids are
            # reused across grid rebuilds (setup_grid recycles a level's cid), so a
            # level's OLD terminal cid can still sit in this FIFO while _sync_orders
            # has re-placed the SAME cid as a live resting order. Popping it from
            # _orders here would make cancel() a silent no-op on that live order,
            # which then fills anyway → orphan fill (engine.py) or the #192 SL
            # double-credit. _open_orders is the authority for "still live".
            if evicted not in self._open_orders:
                self._orders.pop(evicted, None)
            self._terminal_set.discard(evicted)
        self._terminal_ids.append(order.client_id)
        self._terminal_set.add(order.client_id)

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
        # #237: a reused client_id (grid rebuild recycles a level's cid) may still
        # carry a stale entry in the terminal FIFO. Clear it on re-placement so the
        # FIFO doesn't accumulate duplicate cids (which shrink the effective
        # retention window) and so a later overflow can never target this now-live
        # order. The _open_orders guard in _retire already makes eviction safe; this
        # keeps the FIFO honest as well.
        if client_id in self._terminal_set:
            self._terminal_set.discard(client_id)
            try:
                self._terminal_ids.remove(client_id)
            except ValueError:
                pass
        self._orders[client_id] = order
        self._open_orders[client_id] = order
        logger.debug("[PAPER] placed %s %s %s qty=%.6f @ %.4f",
                     symbol, side, client_id[:8], qty, price)
        return order

    def cancel(self, client_id: str) -> bool:
        # #237: _open_orders is the source of truth for whether an order is live
        # and fillable. A terminal-order GC eviction (#135) can drop a still-open
        # reused cid from _orders but NEVER from _open_orders, so consult it first —
        # otherwise cancel() silently no-ops on a live order and it fills again
        # (orphan fill / #192 SL double-credit).
        order = self._open_orders.get(client_id) or self._orders.get(client_id)
        if order and order.status == "open":
            order.status = "cancelled"
            self._retire(order)
            logger.debug("[PAPER] cancelled %s", client_id[:8])
            return True
        return False

    def cancel_all(self, symbol: str) -> int:
        count = 0
        # Iterate only resting orders (a copy, since _retire mutates the dict).
        for order in list(self._open_orders.values()):
            if order.symbol == symbol and order.status == "open":
                order.status = "cancelled"
                self._retire(order)
                count += 1
        return count

    def update_price(self, symbol: str, price: float, sells_only: bool = False) -> List[Fill]:
        """Call each tick with current price. Returns fills that occurred.

        `sells_only=True` fills only resting SELL orders (TP/exit) and skips resting
        BUY orders. This is used for emergency-stopped (#34) / dashboard-disabled
        (#184) coins: their open positions must still be able to exit via TP, but a
        resting buy must NOT fill — that would OPEN new risk on a coin whose contract
        is "new buys halted" (averaging down into a stopped-out loser). Default False
        keeps the normal two-sided grid behaviour untouched (#210)."""
        self._tick += 1
        fills = []

        # #135: scan only resting orders (was: every order ever placed).
        for order in list(self._open_orders.values()):
            if order.symbol != symbol or order.status != "open":
                continue
            if sells_only and order.side == "buy":
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
                    pre_seeded = bool(order.meta.get("pre_seeded", False))
                    bought_at  = float(order.meta.get("bought_at", fill_price))

                    if pre_seeded:
                        # No margin was deposited for pre-seeded sells → credit profit only
                        credit = (fill_price - bought_at) * order.qty - fee
                    else:
                        # Return margin + leveraged P&L
                        margin_return = bought_at * order.qty / leverage
                        pnl           = (fill_price - bought_at) * order.qty
                        credit        = margin_return + pnl - fee

                    self._credit(symbol, credit)

                order.status     = "filled"
                order.filled_qty = order.qty
                self._retire(order)  # #135: drop from the open-order scan set

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
        # #135: iterate only the resting-order index.
        return [o for o in self._open_orders.values()
                if o.symbol == symbol and o.status == "open"]

    def load_balances(self, balances: dict) -> None:
        """Restore per-symbol balances from a previous session.

        Active buckets are overwritten with their persisted value. A saved key that
        is NOT in the current (possibly reduced) active set — a coin disabled via the
        dashboard toggle (#184) or removed from ``config.yaml:symbols`` — is an
        *orphan*: its accumulated paper capital is real and must stay in equity.
        Discarding it (the previous ``if sym in self._balances`` guard) made the
        deposit-anchored drawdown brake (#132) see a phantom loss of exactly that
        bucket on restart, which could instantly FREEZE new buys on every remaining
        coin even though nothing was actually lost (#212).

        We therefore keep the orphan as its own *untraded* bucket instead of dropping
        it: no order ever routes to it (it is not in the active symbol set), so it is
        never debited/credited; ``get_balance()`` sums it so equity stays continuous;
        ``dict(self._balances)`` persists it across further restarts; and a later
        re-enable of the coin restores the saved value straight back into its active
        bucket. Parking the orphan in ``self._balance`` instead (as the issue first
        proposed) would resurrect the #149 hidden-pool hazard — a non-zero fallback
        pool lets an unseeded symbol transact against full-account cash — and would
        not survive the next save (only ``_balances`` is persisted), so it is kept
        out of ``_balance`` on purpose."""
        for sym, val in balances.items():
            self._balances[sym] = float(val)

    def sl_credit(self, symbol: str, amount: float) -> None:
        """Credit the margin + PnL back to the symbol bucket after a strategy-side SL.

        Strategy-handled stop losses never route through the broker's normal fill path,
        so the margin deducted on the original buy fill would otherwise be lost permanently.
        """
        self._credit(symbol, amount)
        logger.debug("[PAPER] SL credit %.4f for %s", amount, symbol)

    def get_balance(self, currency: str = "USD") -> float:
        """Total cash across all symbol buckets plus the fallback pool.

        Always include self._balance so any movement booked against the fallback
        pool (e.g. a pre-seeded sell for an unseeded symbol) stays visible to
        equity / mark-to-market instead of vanishing (#149)."""
        return sum(self._balances.values()) + self._balance

    def round_qty(self, symbol: str, qty: float) -> float:
        return round(qty, 6)

    def round_price(self, symbol: str, price: float) -> float:
        return round(price, 4)
