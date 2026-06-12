"""
Trading Engine – event-driven loop replacing the monolithic run() in grid_bot.py.

Responsibilities:
- Drive the main loop (ticker fetch → on_tick → desired_orders → broker reconcile)
- Manage market context updates (BTC, funding, correlations) on a schedule
- Handle graceful shutdown signals from Dashboard via DB
- Write to Dashboard DB for display
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.context import MarketContext
from core.lifecycle import ShutdownFlag
from core.strategy import Strategy
from execution.broker import Broker
from execution.reconciler import Reconciler

logger = logging.getLogger("core.engine")

CHECK_INTERVAL = 15        # seconds between loop iterations
PREDICTION_RECHECK = 5     # every N ticks refresh prediction (~75s)
GRID_REBUILD_CYCLES = 60   # every N ticks force grid rebuild (~15min)
BTC_REFRESH_CYCLES = 4     # every N ticks refresh BTC context (~1min)
FUNDING_REFRESH_CYCLES = 240  # every N ticks refresh funding (~1h)
# Per-symbol kill switch on realized loss. Must be wider than one full
# floor-SL flush (which can realize several % at once), otherwise a single
# flush permanently halts the symbol.
EMERGENCY_STOP_PCT = 0.12


class Engine:

    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        symbols: List[str],
        ctx: Optional[MarketContext] = None,
        reconciler: Optional[Reconciler] = None,
        initial_capital: float = 0.0,
    ):
        self.strategy = strategy
        self.broker = broker
        self.symbols = symbols
        self.ctx = ctx or MarketContext()
        self.reconciler = reconciler
        self._shutdown = ShutdownFlag()
        self._loop_count = 0
        self._initial_capital = initial_capital

        # Track which orders are currently open per symbol
        self._active_orders: Dict[str, Dict[str, object]] = {s: {} for s in symbols}
        self._waiting_for_fills = False  # set True by wait_fills stop mode

    def run(self):
        logger.info("Engine starting | symbols=%s", self.symbols)

        # Clear any stale stop_mode from previous session before first tick
        try:
            from dashboard.db import set_stop_mode
            set_stop_mode(None)
        except Exception:
            pass

        self.strategy.init(self.symbols, self.ctx)

        # Initialise equity so daily-drawdown brake has a valid baseline
        if self._initial_capital > 0:
            self.ctx.set_equity(self._initial_capital)

        self._refresh_btc()
        self._refresh_funding()

        from data_fetcher import fetch_ohlcv, fetch_ticker
        for sym in self.symbols:
            try:
                price = fetch_ticker(sym)["last"]
                df = fetch_ohlcv(sym, "1h", 500)
                self.strategy.on_candle(sym, df, self.ctx)
                if hasattr(self.strategy, "setup_grid"):
                    self.strategy.setup_grid(sym, price, self.ctx)
            except Exception as e:
                logger.error("Initial setup failed for %s: %s", sym, e)

        while self._shutdown.is_running():
            self._loop_count += 1
            try:
                self._tick()
            except Exception as e:
                logger.error("Engine tick error: %s", e, exc_info=True)

            for _ in range(CHECK_INTERVAL):
                if not self._shutdown.is_running():
                    break
                time.sleep(1)

        logger.info("Engine stopped after %d ticks", self._loop_count)
        self._cleanup()

    def _tick(self):
        from data_fetcher import fetch_ticker, fetch_ohlcv
        import notifier

        self._check_dashboard_stop()

        if self._loop_count % BTC_REFRESH_CYCLES == 0:
            self._refresh_btc()
        if self._loop_count % FUNDING_REFRESH_CYCLES == 0:
            self._refresh_funding()

        self._check_daily_drawdown()

        prices: Dict[str, float] = {}
        for sym in self.symbols:
            try:
                prices[sym] = float(fetch_ticker(sym)["last"])
                time.sleep(0.4)
            except Exception as e:
                logger.warning("Ticker failed %s: %s", sym, e)

        for sym in self.symbols:
            price = prices.get(sym)
            if price is None:
                continue

            state_obj = getattr(self.strategy, "get_state", lambda s: None)(sym)
            if state_obj and state_obj.total_profit <= -(state_obj.investment * EMERGENCY_STOP_PCT):
                logger.warning("[ENGINE] %s emergency stop (max loss)", sym)
                continue

            do_recheck = self._loop_count % PREDICTION_RECHECK == 0
            do_rebuild = self._loop_count % GRID_REBUILD_CYCLES == 0

            if do_recheck or do_rebuild:
                try:
                    df = fetch_ohlcv(sym, "1h", 500)
                    self.strategy.on_candle(sym, df, self.ctx)
                except Exception as e:
                    logger.warning("on_candle failed %s: %s", sym, e)

            out_of_range = False
            if state_obj and state_obj.grid_lines:
                lo = state_obj.grid_lines[0]
                hi = state_obj.grid_lines[-1]
                out_of_range = price < lo * 0.99 or price > hi * 1.01

            # Safety ticks (SL/TP) run even during freeze
            if hasattr(self.strategy, "on_tick_safety"):
                try:
                    self.strategy.on_tick_safety(sym, price, self.ctx)
                except Exception as e:
                    logger.warning("on_tick_safety failed %s: %s", sym, e)

            if not self.ctx.is_frozen():
                self.strategy.on_tick(sym, price, self.ctx)

            # For live mode: reconcile fills from exchange (paper has no reconciler)
            self._reconcile_fills()

            if (out_of_range or do_rebuild) and hasattr(self.strategy, "setup_grid"):
                self.strategy.setup_grid(sym, price, self.ctx)

            if not self.ctx.is_frozen():
                self._sync_orders(sym, price)

            self._update_dashboard(sym, price)

        self._log_equity()

        # wait_fills auto-termination: stop once all sell-with-bought_at positions are closed
        if self._waiting_for_fills:
            still_open = any(
                any(
                    not o.get("filled") and o.get("side") == "sell" and "bought_at" in o
                    for o in (getattr(self.strategy, "get_state", lambda s: {})(sym) or type(
                        "X", (), {"orders": {}})()).orders.values()
                )
                for sym in self.symbols
                if getattr(self.strategy, "get_state", lambda s: None)(sym)
            )
            if not still_open:
                logger.info("wait_fills: all positions closed, stopping.")
                self._shutdown.stop()

    def _reconcile_fills(self):
        """Process fills from live broker via reconciler. Paper fills handled in _sync_orders."""
        from execution.paper import PaperBroker
        if isinstance(self.broker, PaperBroker):
            return  # Paper fills are processed in process_paper_fills() inside _sync_orders
        if self.reconciler:
            fills = self.reconciler.reconcile()
            for fill in fills:
                self.strategy.on_fill(fill, self.ctx)
                if fill.client_id in self._active_orders.get(fill.symbol, {}):
                    del self._active_orders[fill.symbol][fill.client_id]

    def process_paper_fills(self, symbol: str, price: float):
        from execution.paper import PaperBroker
        if isinstance(self.broker, PaperBroker):
            fills = self.broker.update_price(symbol, price)
            for fill in fills:
                self.strategy.on_fill(fill, self.ctx)
                if fill.client_id in self._active_orders.get(symbol, {}):
                    del self._active_orders[symbol][fill.client_id]

    def _sync_orders(self, symbol: str, price: float):
        self.process_paper_fills(symbol, price)

        desired = {o.client_id: o for o in self.strategy.desired_orders(symbol, price, self.ctx)
                   if o.client_id}
        active = self._active_orders.get(symbol, {})

        for cid in list(active.keys()):
            if cid not in desired:
                self.broker.cancel(cid)
                del active[cid]

        for cid, order in desired.items():
            if cid not in active:
                try:
                    broker_order = self.broker.place_limit(
                        symbol=order.symbol,
                        side=order.side,
                        price=order.price,
                        qty=order.qty,
                        post_only=order.post_only,
                        client_id=cid,
                        sl_price=order.sl_price,
                    )
                    active[cid] = broker_order
                    if self.reconciler:
                        self.reconciler.track_order(
                            cid, broker_order.exchange_order_id,
                            symbol, order.side, order.price, order.qty
                        )
                except Exception as e:
                    logger.warning("place_limit failed %s %s: %s", symbol, order.side, e)

        self._active_orders[symbol] = active

    def _check_daily_drawdown(self):
        total_equity = self.ctx.total_equity
        if total_equity <= 0:
            return

        total_profit = 0.0
        for sym in self.symbols:
            state = getattr(self.strategy, "get_state", lambda s: None)(sym)
            if state:
                total_profit += state.total_profit

        if hasattr(self.strategy, "_risk") and self.strategy._risk:
            rm = self.strategy._risk
            rm.set_daily_start(total_equity)
            dd_ok = rm.daily_drawdown_ok(total_equity + total_profit)
            was_frozen = self.ctx.is_frozen()
            self.ctx.set_freeze(not dd_ok)
            if not dd_ok and not was_frozen:
                logger.warning("FREEZE: daily drawdown exceeded – no new buys")
            elif dd_ok and was_frozen:
                logger.info("Freeze lifted – new trading day")

    def _refresh_btc(self):
        try:
            from market.btc_context import get_btc_context
            btc = get_btc_context(force_refresh=True)
            if btc:
                self.ctx.set_btc(btc)
        except Exception as e:
            logger.debug("BTC context refresh failed: %s", e)

    def _refresh_funding(self):
        try:
            from market.perp import get_funding
            for sym in self.symbols:
                info = get_funding(sym)
                if info:
                    self.ctx.set_funding(sym, info)
        except Exception as e:
            logger.debug("Funding refresh failed: %s", e)

    def _check_dashboard_stop(self):
        try:
            from dashboard.db import get_stop_mode, set_stop_mode
            mode = get_stop_mode()
            if mode == "sell_all":
                set_stop_mode(None)
                logger.info("Dashboard: sell_all – initiating emergency sell")
                self._emergency_sell_all()
                self._shutdown.stop()
            elif mode == "wait_fills":
                set_stop_mode(None)
                for sym in self.symbols:
                    state = getattr(self.strategy, "get_state", lambda s: None)(sym)
                    if state:
                        state.with_position = False
                self._waiting_for_fills = True
                logger.info("Dashboard: wait_fills – sell-only mode active, will stop when all filled")
        except Exception:
            pass

    def _emergency_sell_all(self):
        """Cancel all open orders and close all positions at market."""
        for sym in self.symbols:
            try:
                # Cancel open broker orders
                self.broker.cancel_all(sym)
                self._active_orders[sym] = {}

                # For live broker: place market sell for open positions
                from execution.paper import PaperBroker
                if not isinstance(self.broker, PaperBroker):
                    positions = self.ctx.get_positions(sym)
                    for pos in positions:
                        if pos.qty > 0:
                            try:
                                self.broker.place_market(sym, "sell", pos.qty)
                                logger.info("[EMERGENCY] Market sell %s qty=%.6f", sym, pos.qty)
                            except Exception as e:
                                logger.error("Market sell failed %s: %s", sym, e)
                else:
                    # Paper broker: synthesise fills for all open sell-positions
                    from data_fetcher import fetch_ticker
                    price = float(fetch_ticker(sym)["last"])
                    state = getattr(self.strategy, "get_state", lambda s: None)(sym)
                    if state:
                        from core.strategy import Fill
                        for cid, o in list(state.orders.items()):
                            if o.get("side") == "sell" and not o.get("filled") and "bought_at" in o and not o.get("pre_seeded"):
                                fill = Fill(
                                    client_id=cid,
                                    symbol=sym, side="sell",
                                    price=price, qty=o["qty"],
                                    fee=price * o["qty"] * 2 * KRAKEN_FEE,
                                    ts=time.time(),
                                )
                                self.strategy.on_fill(fill, self.ctx)
            except Exception as e:
                logger.error("Emergency sell failed %s: %s", sym, e)

    def _update_dashboard(self, symbol: str, price: float):
        try:
            from dashboard.db import update_grid_state
            state = getattr(self.strategy, "get_state", lambda s: None)(symbol)
            if not state:
                return
            update_grid_state(
                symbol, price, state.orders,
                state.range_pct, state.investment,
                state.total_profit, state.trade_count,
                state._last_prediction,
                predicted_low=getattr(state, "_last_pred_low", 0),
                predicted_high=getattr(state, "_last_pred_high", 0),
                confidence=state._last_confidence,
                regime=state._last_regime,
                directional=state._directional or {},
            )
        except Exception:
            pass

    def _log_equity(self):
        """Log equity: broker balance + value of open positions at last tick price."""
        try:
            balance = self.broker.get_balance("USD")
            total = balance
            self.ctx.set_equity(total)
            from dashboard.db import log_equity, update_capital
            log_equity(total)
            update_capital(total)
        except Exception:
            pass

    def _cleanup(self):
        try:
            from core.lifecycle import release_singleton
            release_singleton()
            from dashboard.db import set_status
            set_status(running=False, mode="stopped", strategy="grid")
        except Exception:
            pass


KRAKEN_FEE = 0.0016  # used in emergency sell fee calculation
