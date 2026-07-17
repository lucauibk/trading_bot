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
# How long equity logging may stay skipped because of stale prices before we
# fall back to last-good prices.  A brief skip protects against the sleep-wake
# MTM spike (#20); but if a single symbol's ticker fails *permanently* (#26/#89),
# an unbounded skip would freeze the whole equity curve and the daily-drawdown
# brake forever.  After this grace window we log equity with the last known
# price for any still-stale symbol instead of blacking out entirely.
STALE_EQUITY_GRACE_SECONDS = 4 * CHECK_INTERVAL  # 60 s


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
        # Give the strategy a broker reference so SL events can credit the balance.
        if hasattr(strategy, "_broker"):
            strategy._broker = broker
        self._loop_count = 0
        self._initial_capital = initial_capital

        # Track which orders are currently open per symbol
        self._active_orders: Dict[str, Dict[str, object]] = {s: {} for s in symbols}
        self._waiting_for_fills = False  # set True by wait_fills stop mode
        # Last known prices (updated every tick) — used for mark-to-market equity
        self._last_prices: Dict[str, float] = {}
        # Timestamp of the last successful price fetch per symbol.  Used by
        # _log_equity() to guard against stale MTM after a Mac-sleep: if the
        # host was suspended, _last_prices holds pre-sleep values but the broker
        # balance already reflects post-wake reality → equity would spike/drop
        # by the full leveraged MTM delta.  We skip MTM for any symbol whose
        # cached price is older than 2× CHECK_INTERVAL (30 s).
        self._last_price_ts: Dict[str, float] = {}
        # When equity logging first started being skipped due to stale prices.
        # None while prices are fresh.  Used to bound the skip (see #89): after
        # STALE_EQUITY_GRACE_SECONDS we stop skipping and fall back to last-good
        # prices so a permanently-dead ticker cannot freeze the equity curve.
        self._equity_stale_since: Optional[float] = None
        # Cache last logged prediction per symbol to avoid writing on every 15s tick
        self._last_logged_pred: Dict[str, str] = {}
        # Rebuild cooldown: prevents out_of_range events from back-to-back grid
        # rebuilds when price hovers near the edge.  Scheduled rebuilds (do_rebuild)
        # always run unconditionally; only out_of_range rebuilds are rate-limited
        # to once per 20 ticks (~5 min).
        self._last_rebuild: Dict[str, int] = {}
        # Symbols currently in per-coin emergency stop (log-once + resume tracking).
        self._emergency_logged: set = set()

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
            self._refresh_correlations()

        self._check_daily_drawdown()

        prices: Dict[str, float] = {}
        for sym in self.symbols:
            try:
                prices[sym] = float(fetch_ticker(sym)["last"])
                time.sleep(0.4)
            except Exception as e:
                logger.warning("Ticker failed %s: %s", sym, e)

        # Cache for mark-to-market equity calculation
        now = time.time()
        self._last_prices.update(prices)
        for sym in prices:
            self._last_price_ts[sym] = now

        for sym in self.symbols:
            price = prices.get(sym)
            if price is None:
                continue

            state_obj = getattr(self.strategy, "get_state", lambda s: None)(sym)
            # Emergency stop (#34): a symbol past its per-coin realized-loss cap
            # must stop OPENING new risk — but its open positions still need SL/TP
            # protection. Previously this `continue`d before on_tick_safety, so
            # existing positions were left without a stop-loss and the unrealized
            # loss could grow unbounded. Now we treat it exactly like the daily
            # freeze: block new orders (on_tick + _sync_orders) below, but keep
            # running on_tick_safety and the dashboard update.
            emergency_stopped = bool(
                state_obj
                and state_obj.total_profit <= -(state_obj.investment * EMERGENCY_STOP_PCT)
            )
            if emergency_stopped and sym not in self._emergency_logged:
                logger.warning(
                    "[ENGINE] %s emergency stop (max loss) — new buys halted, "
                    "SL/TP still active", sym)
                self._emergency_logged.add(sym)
            elif not emergency_stopped and sym in self._emergency_logged:
                # Realized PnL recovered above the cap → allow trading again.
                self._emergency_logged.discard(sym)

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
                out_of_range = price < lo * 0.98 or price > hi * 1.02

            # Safety ticks (SL/TP) run even during freeze
            if hasattr(self.strategy, "on_tick_safety"):
                try:
                    self.strategy.on_tick_safety(sym, price, self.ctx)
                except Exception as e:
                    logger.warning("on_tick_safety failed %s: %s", sym, e)

            if not self.ctx.is_frozen() and not emergency_stopped:
                self.strategy.on_tick(sym, price, self.ctx)

            # For live mode: reconcile fills from exchange (paper has no reconciler)
            self._reconcile_fills()

            # Scheduled rebuild (every 60 ticks) always fires.  out_of_range
            # rebuilds are rate-limited to once per 20 ticks (~5 min) so that
            # price hovering near the grid edge does not cause a rebuild storm.
            rebuild_allowed = do_rebuild or (
                out_of_range and
                self._loop_count - self._last_rebuild.get(sym, 0) >= 40
            )
            if rebuild_allowed and hasattr(self.strategy, "setup_grid"):
                # Note: setup_grid runs even during a daily-drawdown freeze because
                # price can escape the grid while frozen.  setup_grid gates buy-seeding
                # internally via _buys_allowed (inventory cap + trend filter).  Even if
                # setup_grid seeds new buys, they won't be submitted to the broker while
                # frozen because _sync_orders is blocked below (if not is_frozen()).
                # on_tick_safety still fires SLs during freeze — intentional one-way
                # liquidation.  The real prevention against cascade losses is the
                # inventory cap (max_inventory_notional_mult) which stops accumulation
                # before a cascade can form, regardless of freeze state.
                self.strategy.setup_grid(sym, price, self.ctx)
                self._last_rebuild[sym] = self._loop_count

            if not self.ctx.is_frozen() and not emergency_stopped:
                self._sync_orders(sym, price)

            self._update_dashboard(sym, price)

        self._log_equity()
        self._update_prediction_outcomes(fetch_ohlcv)  # noqa: F821 (imported in _tick scope)

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
                        meta=order.meta,
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
            # Anchor baseline to configured starting capital so the brake always
            # measures against what the user deposited, not a mid-session equity.
            # total_profit is already baked into total_equity (cash balance), so
            # adding it again would double-count it — just use total_equity.
            baseline = self._initial_capital if self._initial_capital > 0 else total_equity
            rm.set_daily_start(baseline)
            dd_ok = rm.daily_drawdown_ok(total_equity)
            was_frozen = self.ctx.is_frozen()
            self.ctx.set_freeze(not dd_ok)
            if not dd_ok and not was_frozen:
                logger.warning("FREEZE: daily drawdown exceeded – no new buys")
                try:
                    from dashboard.db import set_frozen
                    set_frozen(True, f"Daily drawdown >{self.strategy._risk.max_daily_drawdown*100:.0f}%")
                except Exception:
                    pass
            elif dd_ok and was_frozen:
                logger.info("Freeze lifted – new trading day")
                try:
                    from dashboard.db import set_frozen
                    set_frozen(False)
                except Exception:
                    pass

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

    def _refresh_correlations(self):
        """Feed the CorrelationTracker so RiskManager.can_open()'s over-concentration
        bucket (step 5) has real data (#43). Without this the tracker stays empty,
        high_correlation_symbols() always returns [], and the bucket check is a
        silent no-op — all highly-correlated alts can open at once.

        30d correlation moves slowly, so this runs on the slow funding cadence and
        reuses the BTC close already fetched by btc_context (no extra fetch)."""
        try:
            risk = getattr(self.strategy, "_risk", None)
            corr = getattr(risk, "corr", None) if risk is not None else None
            if corr is None:
                return
            from market.btc_context import get_btc_close
            btc_close = get_btc_close()
            if btc_close is None or len(btc_close) < 100:
                return
            corr.update_btc(btc_close)
            get_state = getattr(self.strategy, "get_state", lambda s: None)
            for sym in self.symbols:
                st = get_state(sym)
                df = getattr(st, "_last_df", None) if st is not None else None
                if df is not None and "close" in getattr(df, "columns", []):
                    corr.update_symbol(sym, df["close"])
        except Exception as e:
            logger.debug("Correlation refresh failed: %s", e)

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
                                # #180: only the sell-side fee here — the buy fee
                                # was already deducted at buy-fill time
                                # (paper.py update_price). Charging a round-trip
                                # fee would double-count the buy fee, mirroring
                                # the floor-SL path (grid.py:752).
                                sell_fee = price * o["qty"] * KRAKEN_FEE
                                fill = Fill(
                                    client_id=cid,
                                    symbol=sym, side="sell",
                                    price=price, qty=o["qty"],
                                    fee=sell_fee,
                                    ts=time.time(),
                                )
                                self.strategy.on_fill(fill, self.ctx)
                                # #180: on_fill only updates in-memory total_profit
                                # and logs the trade — it never touches the broker
                                # cash bucket. Since cancel_all() above already
                                # cancelled the broker sell order, update_price will
                                # never credit margin+PnL back. Credit it here,
                                # identically to the floor-SL path (grid.py:747-754),
                                # so the persisted paper balance keeps the full
                                # margin instead of leaking it on every sell_all stop.
                                entry_lev = o.get("leverage", 1.0) or 1.0
                                bought_at = o["bought_at"]
                                credit = (
                                    bought_at * o["qty"] / entry_lev
                                    + (price - bought_at) * o["qty"]
                                    - sell_fee
                                )
                                self.broker.sl_credit(sym, credit)
            except Exception as e:
                logger.error("Emergency sell failed %s: %s", sym, e)

    def _update_dashboard(self, symbol: str, price: float):
        try:
            from dashboard.db import update_grid_state, log_prediction
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
                floor_sl=getattr(state, "floor_sl", 0.0),
            )
            # Log prediction only when it changes (avoids writing every 15s tick)
            current_pred = state._last_prediction
            if self._last_logged_pred.get(symbol) != current_pred:
                try:
                    log_prediction(
                        symbol,
                        current_pred,
                        getattr(state, "_last_confidence", 0.0),
                        getattr(state, "_last_pred_low", 0.0),
                        getattr(state, "_last_pred_high", 0.0),
                        # ML-specific fields for accurate calibration (separate from
                        # PricePredictor bounds which are a different system).
                        ml_score=getattr(state, "_direction_score", 0.0),
                        entry_price=self._last_prices.get(symbol, 0.0),
                    )
                    self._last_logged_pred[symbol] = current_pred
                except Exception as e:
                    logger.debug("log_prediction failed %s: %s", symbol, e)
        except Exception:
            pass

    def _log_equity(self):
        """Log equity: cash balance + margin-correct MTM of open positions.

        MTM per position = margin_returned + unrealized_pnl
                         = qty × bought_at / leverage + qty × (price - bought_at)
        Using full notional (qty × price) inflates equity by (lev-1)/lev × notional
        per position when leverage > 1, because PaperBroker only deducts margin on buy.

        Stale-price guard: after a Mac-sleep the host process is suspended for an
        arbitrary duration.  _last_prices still holds pre-sleep values, but when the
        machine wakes up the market has moved.  Using stale prices for MTM would create
        a spurious equity spike/drop that could falsely trigger the daily-drawdown brake
        or EMERGENCY_STOP_PCT.  We skip MTM for any symbol whose price is older than
        2× CHECK_INTERVAL (30 s); those positions are treated as cash-equivalent until
        the next fresh tick arrives.
        """
        try:
            balance = self.broker.get_balance("USD")
            try:
                from dashboard.db import get_leverage as _get_lev
                default_lev = float(_get_lev())
            except Exception:
                default_lev = 1.0
            mtm = 0.0
            price_age_limit = 2 * CHECK_INTERVAL  # 30 s
            now = time.time()
            stale_syms = []
            for sym in self.symbols:
                price = self._last_prices.get(sym, 0.0)
                if price <= 0:
                    continue
                age = now - self._last_price_ts.get(sym, 0.0)
                if age > price_age_limit:
                    stale_syms.append(sym)
                # NOTE: MTM is still computed (using the last-good price) for stale
                # symbols. Dropping their margin+unrealized would understate equity
                # by a full position and falsely trip the drawdown brake. Whether we
                # actually *log* this value is decided by the grace window below.
                state = getattr(self.strategy, "get_state", lambda s: None)(sym)
                if state is None:
                    continue
                for o in state.orders.values():
                    if (
                        not o.get("filled")
                        and o.get("side") == "sell"
                        and "bought_at" in o
                        and not o.get("pre_seeded")
                    ):
                        qty = o.get("qty", 0.0)
                        bought_at = o["bought_at"]
                        lev = o.get("leverage", default_lev) or 1.0
                        margin = qty * bought_at / lev
                        unrealized = qty * (price - bought_at)
                        mtm += margin + unrealized

            if stale_syms:
                if self._equity_stale_since is None:
                    self._equity_stale_since = now
                stale_for = now - self._equity_stale_since
                if stale_for < STALE_EQUITY_GRACE_SECONDS:
                    logger.warning(
                        "_log_equity: stale prices for %s (>%ds) — equity update skipped "
                        "(retaining last good value), likely waking from sleep.",
                        stale_syms, price_age_limit,
                    )
                    return  # brief skip: do NOT propagate — a sleep-wake spike would
                            # otherwise falsely trigger the daily-drawdown FREEZE.
                # Grace expired: a symbol's ticker is persistently dead (#89). Keep
                # logging equity using its last-good price rather than freezing the
                # entire equity curve + drawdown brake indefinitely.
                logger.warning(
                    "_log_equity: prices for %s stale for %ds (> grace %ds) — logging "
                    "equity with last-good prices to avoid a permanent equity freeze.",
                    stale_syms, int(stale_for), STALE_EQUITY_GRACE_SECONDS,
                )
            else:
                self._equity_stale_since = None

            total = balance + mtm
            self.ctx.set_equity(total)
            from dashboard.db import log_equity, update_capital, save_paper_balances
            log_equity(total)
            update_capital(total)
            if hasattr(self.broker, "_balances") and self.broker._balances:
                try:
                    save_paper_balances(dict(self.broker._balances))
                except Exception:
                    pass
        except Exception:
            pass

    def _update_prediction_outcomes(self, fetch_ohlcv_fn):
        """Füllt realized_high_6h/low_6h/hit für gereifte Predictions nach (alle ~15min)."""
        if self._loop_count % GRID_REBUILD_CYCLES != 0:
            return
        try:
            from dashboard.db import update_prediction_outcomes
            update_prediction_outcomes(fetch_ohlcv_fn)
        except Exception as e:
            logger.debug("update_prediction_outcomes failed: %s", e)

    def _cleanup(self):
        try:
            from core.lifecycle import release_singleton
            release_singleton()
            from dashboard.db import set_status
            set_status(running=False, mode="stopped", strategy="grid")
        except Exception:
            pass


KRAKEN_FEE = 0.0016  # used in emergency sell fee calculation
