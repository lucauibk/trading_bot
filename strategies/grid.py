"""
GridStrategy – grid trading using the Strategy ABC.

Grid builds limit buy orders below current price and limit sell orders above.
When a buy fills, a sell is placed one level higher (+ SL).
When a sell fills, a new buy is replenished.
"""

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ta as ta_lib

from core.context import MarketContext, Position
from core.strategy import Fill, Order, Strategy
from strategies.grid_params import GridParams

logger = logging.getLogger("strategies.grid")

# ── Constants ──────────────────────────────────────────────────────────────────
KRAKEN_FEE = 0.0016
MIN_STEP_FEE_MULTIPLE = 4.0
COMPOUND_EVERY_TRADES = 3
MAX_INVESTMENT_MULT = 3.0          # Compounding cap: max 3× initial investment

ADAPTIVE_SIZING = True             # Bullish → more budget on lower levels
SIZE_BIAS_FACTOR = 0.30

# Directional trade config (non-swept parts; swept ones live in GridParams)
DIRECTIONAL_RECHECK_SCORE_MIN = 0.25
DIRECTIONAL_DOWN_TRAIL_PCT = 0.005
DIRECTIONAL_COOLOFF_SECONDS = 4 * 3600
NO_DIRECTIONAL_HOURS = frozenset({5, 6, 7, 8})   # UTC hours with negative EV


def _calc_level_allocations(grid_lines: list, current_price: float,
                             investment: float, direction_score: float) -> dict:
    """Non-uniform budget allocation: bullish → more on lower buy levels (DCA into dips)."""
    n = len(grid_lines)
    if n == 0:
        # Defensive early-out: no grid lines → nothing to allocate. Must come
        # first, otherwise the `investment / n` uniform path below divides by 0.
        return {}
    if not ADAPTIVE_SIZING or abs(direction_score) < 0.05:
        base = investment / n
        return {p: base for p in grid_lines}

    sorted_lines = sorted(grid_lines)
    bias = direction_score * SIZE_BIAS_FACTOR
    weights = []
    for i, price in enumerate(sorted_lines):
        rank = i / (n - 1) if n > 1 else 0.5  # 0 = lowest level, 1 = highest
        if direction_score >= 0:
            w = 1.0 + bias * (0.5 - rank) * 2       # bullish: boost lower levels
        else:
            w = 1.0 + abs(bias) * (rank - 0.5) * 2  # bearish: boost upper levels
        weights.append(max(0.2, w))

    total_w = sum(weights)
    return {p: w / total_w * investment for p, w in zip(sorted_lines, weights)}


def _get_leverage() -> float:
    """Read current leverage from dashboard DB. Falls back to 1.0."""
    try:
        from dashboard.db import get_leverage
        return float(get_leverage() or 1.0)
    except Exception:
        return 1.0


class _GridState:
    """Per-symbol mutable state."""

    def __init__(self, symbol: str, investment: float, levels: int, range_pct: float):
        self.symbol = symbol
        self.investment = investment
        self._initial_investment = investment
        self.levels = levels
        self.range_pct = range_pct

        self.grid_lines: List[float] = []
        # client_id → {side, price, qty, filled, bought_at, sl_price, momentum_holds, ...}
        self.orders: Dict[str, dict] = {}
        self.price_to_id: Dict[float, str] = {}

        self.total_profit = 0.0
        self.trade_count = 0
        self.usdt_per_grid = investment / max(levels, 1)
        self._last_compound_at = 0
        self._compounded_profit = 0.0
        self._direction_score = 0.0
        self._last_prediction = "neutral"
        self._last_regime = ""
        self._last_confidence = 0.0
        self._last_pred_low = 0.0
        self._last_pred_high = 0.0
        self.with_position = False
        # Permanent sell-only latch for the graceful "wait_fills" shutdown. Unlike
        # with_position (recomputed every candle from the prediction), this is never
        # reset by _refresh_prediction, so the bot cannot re-arm buys after the user
        # requested a graceful stop. (#115)
        self.sell_only = False

        # Directional trade state
        self._directional: dict = {}
        self._directional_needs_recheck = False
        self._directional_sl_ts: float = 0.0

        # ATR for trailing stops
        self._atr: float = 0.0
        self._last_df: object = None

        # Floor-SL + hard trend filter state
        self.grid_lower: float = 0.0
        self.floor_sl: float = 0.0
        self._hard_trend_down: bool = False
        self._trend_up_count: int = 0


class GridStrategy(Strategy):
    name = "grid"

    def __init__(self, grids_config: list, risk_manager=None, ml_enabled: bool = True,
                 params: Optional[GridParams] = None):
        self._config = {g["symbol"]: g for g in grids_config}
        self._risk = risk_manager
        self._ml_enabled = ml_enabled
        self.p = params or GridParams()
        self._states: Dict[str, _GridState] = {}
        self._ml_predictor = None
        self._price_predictors: dict = {}
        self._broker = None  # set by engine; used to credit margin on SL
        # MTF analyzer state (symbol → MTFAnalyzer / bias / setup)
        self._mtf_analyzers: dict = {}
        self._mtf_bias: dict = {}
        self._mtf_setup: dict = {}
        self._mtf_bias_ts: dict = {}

    def _lev(self) -> float:
        """Pinned leverage from params (backtest/sweep determinism) or live dashboard value."""
        return self.p.leverage if self.p.leverage > 0 else _get_leverage()

    def init(self, symbols: List[str], ctx: MarketContext) -> None:
        for sym in symbols:
            cfg = self._config.get(sym, {"investment": 40.0, "levels": 8})
            self._states[sym] = _GridState(sym, cfg["investment"], cfg.get("levels", 8), 0.05)

        if not self._ml_enabled:
            return

        try:
            from ml.predictor import MLPredictor
            from data_fetcher import fetch_ohlcv
            self._ml_predictor = MLPredictor(fetch_ohlcv_fn=fetch_ohlcv)
            self._ml_predictor.initialize(symbols)
        except Exception as e:
            logger.warning("ML predictor unavailable: %s", e)

        try:
            from price_predictor.predictor import PricePredictor
            for sym in symbols:
                self._price_predictors[sym] = PricePredictor(
                    exchange_id="kraken", symbol=sym, timeframe="1h", limit=500
                )
                logger.info("PricePredictor ready for %s", sym)
        except Exception as e:
            logger.warning("PricePredictor unavailable: %s", e)

        try:
            from price_predictor.mtf_analyzer import MTFAnalyzer
            from data_fetcher import fetch_ohlcv
            for sym in symbols:
                self._mtf_analyzers[sym] = MTFAnalyzer(sym, fetch_ohlcv)
                self._mtf_bias[sym] = {"daily_bias": "neutral", "confirmed_4h": False, "daily_adx": 0.0}
                self._mtf_setup[sym] = None
                self._mtf_bias_ts[sym] = 0.0
            logger.info("MTFAnalyzer initialized for %d symbols", len(symbols))
        except Exception as e:
            logger.warning("MTFAnalyzer unavailable: %s", e)

        logger.info("GridStrategy initialized for %d symbols", len(symbols))

    def on_candle(self, symbol: str, df: pd.DataFrame, ctx: MarketContext) -> None:
        state = self._states.get(symbol)
        if not state:
            return

        try:
            atr = float(ta_lib.volatility.average_true_range(
                df["high"], df["low"], df["close"], window=14
            ).iloc[-1])
            state._atr = atr
        except Exception:
            pass
        state._last_df = df

        if self.p.trend_filter_enabled:
            self._update_trend_filter(state, df)

        self._refresh_prediction(symbol, df, ctx)
        self._refresh_mtf_setup(symbol)

    def _update_trend_filter(self, state: _GridState, df: pd.DataFrame) -> None:
        """ML-independent hard downtrend detection with 2-candle exit hysteresis."""
        try:
            tail = df.tail(150)
            close = tail["close"]
            ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            adx_ind = ta_lib.trend.ADXIndicator(tail["high"], tail["low"], close, window=14)
            adx = float(adx_ind.adx().iloc[-1])
            di_plus = float(adx_ind.adx_pos().iloc[-1])
            di_minus = float(adx_ind.adx_neg().iloc[-1])
        except Exception:
            return

        hard_down = (ema9 < ema21 < ema50) or (adx > self.p.trend_adx_min and di_minus > di_plus)
        if hard_down:
            if not state._hard_trend_down:
                logger.info("[TREND] %s hard downtrend → grid buys paused", state.symbol)
            state._hard_trend_down = True
            state._trend_up_count = 0
        elif state._hard_trend_down:
            state._trend_up_count += 1
            if state._trend_up_count >= 2:
                state._hard_trend_down = False
                logger.info("[TREND] %s downtrend cleared → grid buys resumed", state.symbol)

    def _deployed_notional(self, state: _GridState) -> float:
        """Aggregate leveraged notional of open (non-pre-seeded) grid positions.

        Uses the same filter as _log_equity: not filled, side=sell, has bought_at,
        not pre_seeded.  qty already bakes in leverage (qty = usdt*lev/price),
        so qty*bought_at is the actual leveraged notional currently deployed.
        """
        total = 0.0
        for o in state.orders.values():
            if (not o.get("filled") and o.get("side") == "sell"
                    and "bought_at" in o and not o.get("pre_seeded")):
                total += o.get("qty", 0.0) * o["bought_at"]
        return total

    def _buys_allowed(self, state: _GridState) -> bool:
        # Graceful wait_fills stop: no new buys, ever, regardless of prediction.
        if state.sell_only:
            return False
        if not state.with_position:
            return False
        if self.p.trend_filter_enabled and state._hard_trend_down:
            return False
        # Inventory/exposure cap: block new buys when deployed leveraged notional
        # reaches the threshold.  Auto-tightens with leverage (qty bakes in lev).
        # Covers desired_orders emission + smart-replenish + (via engine cancel-on-
        # absent) already-resting pending buys when the cap is freshly hit.
        if self.p.max_inventory_notional_mult > 0 and \
           self._deployed_notional(state) >= self.p.max_inventory_notional_mult * state.investment:
            return False
        # Optional confidence floor: skip buys if PricePredictor confidence is low.
        # Note: NOT ml/predictor.py MIN_CONFIDENCE (different, unrelated gate).
        if self.p.min_confidence_to_buy > 0 and \
           state._last_confidence < self.p.min_confidence_to_buy:
            return False
        return True

    def _refresh_prediction(self, symbol: str, df: pd.DataFrame, ctx: MarketContext):
        state = self._states[symbol]
        direction = "neutral"
        score = 0.0

        if self._ml_predictor is not None:
            try:
                direction = self._ml_predictor.predict(symbol)
                score = self._ml_predictor.get_score(symbol) or 0.0
                if direction == "down" and score > 0:
                    score = -score
                elif direction == "up" and score < 0:
                    score = -score
            except Exception as e:
                logger.debug("ML predict failed %s: %s", symbol, e)
        else:
            try:
                close = df["close"]
                ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
                ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
                rsi = float(ta_lib.momentum.rsi(close, window=14).iloc[-1]) if len(close) >= 14 else 50.0
                if ema9 < ema21 and rsi < 45:
                    direction = "down"
                    score = -0.5
                elif ema9 > ema21 and rsi > 55:
                    direction = "up"
                    score = 0.5
            except Exception:
                pass

        state._last_prediction = direction
        state._direction_score = score
        # Once a graceful wait_fills stop is latched, keep buys off — never let a
        # fresh "up"/"neutral" prediction flip with_position back on. (#115)
        state.with_position = False if state.sell_only else (direction != "down")

    def _build_grid_params(self, symbol: str, price: float, state: _GridState):
        """Compute (lower, upper, levels, range_pct, regime, confidence)."""
        lower = upper = regime = confidence = None

        pp = self._price_predictors.get(symbol)
        if pp is not None:
            try:
                result = pp.predict()
                lower = result["predicted_low"]
                upper = result["predicted_high"]
                regime = result["regime"]
                confidence = result["confidence"]
            except Exception:
                pass

        if (lower is None or upper is None) and state._last_df is not None:
            try:
                df = state._last_df
                close = df["close"]
                bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
                bb_lower = float(bb.bollinger_lband().iloc[-1])
                bb_upper = float(bb.bollinger_hband().iloc[-1])
                adx_val = float(ta_lib.trend.adx(df["high"], df["low"], close, window=14).iloc[-1])
                atr_pct = state._atr / price if price > 0 else 0.03
                if adx_val > 25:
                    half = state._atr * self.p.range_atr_mult_trending
                    lower, upper, regime = price - half, price + half, "trending"
                elif atr_pct > 0.03:
                    half = state._atr * self.p.range_atr_mult_volatile
                    lower, upper, regime = price - half, price + half, "volatile"
                elif bb_upper > bb_lower:
                    lower, upper, regime = bb_lower, bb_upper, "ranging"
                else:
                    half = state._atr * 1.5
                    lower, upper, regime = price - half, price + half, "ranging"
                lower = max(lower, 0.0)
                confidence = 0.6
            except Exception:
                pass

        levels = state.levels
        regime_configs = self.p.regime_levels
        if regime in regime_configs:
            levels = regime_configs[regime]

        min_range = KRAKEN_FEE * levels * MIN_STEP_FEE_MULTIPLE

        if lower and upper and upper > lower:
            range_pct = (upper - lower) / (2 * price)
            if range_pct < min_range:
                lower = price * (1 - min_range)
                upper = price * (1 + min_range)
                range_pct = min_range
        else:
            range_pct = max(state._atr * 2 / price if state._atr > 0 else 0.05, min_range)
            lower = price * (1 - range_pct)
            upper = price * (1 + range_pct)

        # Fee-aware minimum step: cap levels (not range — range reflects expected
        # volatility) so each grid step clears round-trip costs with margin.
        if self.p.min_step_pct > 0 and upper > lower:
            step_pct = (upper - lower) / levels / price
            if step_pct < self.p.min_step_pct:
                levels = max(2, int((upper - lower) / (price * self.p.min_step_pct)))

        return lower, upper, levels, range_pct, regime or "ranging", confidence or 0.5

    def desired_orders(self, symbol: str, price: float, ctx: MarketContext) -> List[Order]:
        state = self._states.get(symbol)
        if not state or not state.grid_lines:
            return []

        orders = []
        buys_ok = self._buys_allowed(state)
        for cid, o in state.orders.items():
            if o.get("filled"):
                continue
            if not buys_ok and o["side"] == "buy":
                continue
            # BTC crash gate for new grid buys (not for existing sell positions)
            if o["side"] == "buy" and "bought_at" not in o:
                btc = ctx.get_btc()
                if btc and btc.trend == "down" and btc.return_1h < -0.02:
                    continue
            orders.append(Order(
                symbol=symbol,
                side=o["side"],
                price=o["price"],
                qty=o["qty"],
                client_id=cid,
                sl_price=o.get("sl_price"),
                meta={"strategy": "grid", "leverage": self._lev(), **o},
            ))
        return orders

    def on_fill(self, fill: Fill, ctx: MarketContext) -> None:
        state = self._states.get(fill.symbol)
        if not state:
            return

        cid = fill.client_id
        order = state.orders.get(cid)
        if not order:
            return

        order["filled"] = True
        order["fill_price"] = fill.price

        if order["side"] == "buy":
            self._handle_buy_fill(fill, state, ctx)
        else:
            self._handle_sell_fill(fill, state, ctx)

        # Clean up filled order — don't keep it in state.orders forever
        state.orders.pop(cid, None)

    def _handle_buy_fill(self, fill: Fill, state: _GridState, ctx: MarketContext):
        buy_price = fill.price
        buy_cid = fill.client_id
        order = state.orders[buy_cid]
        lev = self._lev()
        qty = order["qty"]  # qty already has leverage factored in at setup

        try:
            idx = state.grid_lines.index(order["price"])
        except ValueError:
            idx = None

        if idx is not None and idx < len(state.grid_lines) - 1:
            sell_price = state.grid_lines[idx + 1]
            if self.p.sl_mode == "floor" and state.floor_sl > 0:
                # Per-cohort mode: use the floor stamped at seeding time so each
                # rebuild cohort has its own SL trigger.  A breach then only flushes
                # that cohort, not all accumulated positions from every rebuild.
                sl_price = (order.get("cohort_floor", state.floor_sl)
                            if self.p.floor_sl_per_cohort else state.floor_sl)
            else:
                step_pct = (sell_price - buy_price) / buy_price
                sl_pct = max(step_pct * self.p.per_pos_sl_step_mult, self.p.per_pos_sl_min_pct)
                # Hard-cap: no per-position SL can be wider than per_pos_sl_max_pct (4%)
                sl_pct = min(sl_pct, self.p.per_pos_sl_max_pct)
                sl_price = buy_price * (1 - sl_pct)

            sell_cid = str(uuid.uuid4())
            state.orders[sell_cid] = {
                "side": "sell",
                "price": sell_price,
                "qty": qty,
                "filled": False,
                "bought_at": buy_price,
                "sl_price": sl_price,
                "leverage": lev,
                "trailing_activated": False,
                "momentum_holds": 0,
            }
            state.price_to_id[sell_price] = sell_cid
            logger.info("[GRID] BUY fill %s @ %.4f | SL=%.4f | -> sell @ %.4f",
                        fill.symbol, buy_price, sl_price, sell_price)

        ctx.add_position(Position(
            symbol=fill.symbol, side="grid",
            entry_price=buy_price, qty=qty,
            usdt_value=buy_price * qty,
            leverage=lev,
        ))

    def _handle_sell_fill(self, fill: Fill, state: _GridState, ctx: MarketContext):
        sell_price = fill.price
        sell_cid = fill.client_id
        order = state.orders[sell_cid]
        buy_price = order.get("bought_at", sell_price)
        qty = order["qty"]
        # #57: log the leverage the position was entered with, not the live
        # value (which may have been changed via the dashboard mid-position).
        lev = order.get("leverage", self._lev())

        profit = (sell_price - buy_price) * qty
        fee = (sell_price + buy_price) * qty * KRAKEN_FEE
        net = profit - fee
        state.total_profit += net
        state.trade_count += 1

        holding_seconds = time.time() - order.get("entry_ts", time.time())
        logger.info("[GRID] SELL fill %s @ %.4f | bought @ %.4f | net=%.4f USDT",
                    fill.symbol, sell_price, buy_price, net)

        try:
            if os.getenv("GRIDBOT_BACKTEST"):
                raise ImportError("backtest: dashboard logging disabled")
            from dashboard.db import log_trade
            log_trade(
                fill.symbol, "GRID", buy_price, sell_price, net,
                "grid_fill", "grid", "paper",
                context={
                    "regime": state._last_regime,
                    "atr_pct": state.range_pct * 0.5,
                    "ml_prediction": state._last_prediction,
                    "ml_confidence": state._last_confidence,
                    "predicted_low": state._last_pred_low,
                    "predicted_high": state._last_pred_high,
                    "holding_seconds": holding_seconds,
                },
                leverage=lev,
            )
        except Exception:
            pass

        try:
            if os.getenv("GRIDBOT_BACKTEST"):
                raise ImportError("backtest: notifier disabled")
            import notifier
            notifier.notify_trade_close(fill.symbol, "GRID", buy_price, sell_price, net, "grid_fill")
        except Exception:
            pass

        # Smart-replenish: follow trend one level higher when bullish
        if self._buys_allowed(state):
            new_cid = str(uuid.uuid4())
            replenish_usdt = state.usdt_per_grid
            if state._direction_score > 0.1:
                # Follow trend upward: place replenish at next higher level
                try:
                    current_idx = state.grid_lines.index(order["price"])
                    if current_idx < len(state.grid_lines) - 1:
                        new_buy_price = state.grid_lines[current_idx + 1]
                    else:
                        new_buy_price = buy_price
                except (ValueError, IndexError):
                    new_buy_price = buy_price
            else:
                new_buy_price = buy_price
            new_qty = replenish_usdt * lev / new_buy_price
            state.orders[new_cid] = {
                "side": "buy",
                "price": new_buy_price,
                "qty": new_qty,
                "filled": False,
            }
            state.price_to_id[new_buy_price] = new_cid

        if not order.get("pre_seeded"):
            ctx.remove_position(fill.symbol, "grid")
        self._maybe_compound(sell_price, state)

    def _maybe_compound(self, price: float, state: _GridState):
        if state.total_profit <= 0:
            return
        trades_since = state.trade_count - state._last_compound_at
        if trades_since < COMPOUND_EVERY_TRADES:
            return
        delta = state.total_profit - state._compounded_profit
        if delta <= 0:
            return
        max_inv = state._initial_investment * MAX_INVESTMENT_MULT
        old = state.investment
        state.investment = min(old + delta, max_inv)
        state.usdt_per_grid = state.investment / state.levels
        state._last_compound_at = state.trade_count
        state._compounded_profit = state.total_profit
        logger.info("[COMPOUND] %s %.2f → %.2f USDT", state.symbol, old, state.investment)

    def on_tick(self, symbol: str, price: float, ctx: MarketContext) -> List[Order]:
        state = self._states.get(symbol)
        if not state:
            return []

        self._check_position_stops(symbol, price, state, ctx)
        self._check_directional(symbol, price, state, ctx)
        self._maybe_open_directional(symbol, price, state, ctx)
        self._check_mtf_entry(symbol, price, state, ctx)
        self._update_trailing_stops(symbol, price, state)

        return []

    def on_tick_safety(self, symbol: str, price: float, ctx: MarketContext):
        """SL/TP checks that run even during freeze (never skip stop-losses)."""
        state = self._states.get(symbol)
        if state:
            self._check_position_stops(symbol, price, state, ctx)
            # Directional exits (SL/TP/signal-flip) must also run during a freeze —
            # otherwise an open directional long can blow through its stop-loss while
            # the market dumps. _check_directional only ever *closes* a position, never
            # opens one, so it is safe on the freeze path (#104).
            self._check_directional(symbol, price, state, ctx)

    def _update_trailing_stops(self, symbol: str, price: float, state: _GridState):
        atr = state._atr
        if atr <= 0:
            return
        for order in state.orders.values():
            if order.get("filled") or order["side"] != "sell":
                continue
            if "bought_at" not in order or "sl_price" not in order:
                continue
            buy_price = order["bought_at"]
            current_sl = order["sl_price"]
            profit_in_atr = (price - buy_price) / atr

            if not order.get("trailing_activated") and profit_in_atr >= 1.0:
                new_sl = buy_price
                if new_sl > current_sl:
                    order["sl_price"] = new_sl
                    order["trailing_activated"] = True
                    logger.info("[TRAIL] %s SL moved to break-even %.4f", symbol, new_sl)
            elif order.get("trailing_activated"):
                trailing_sl = price - 1.5 * atr
                if trailing_sl > current_sl:
                    order["sl_price"] = trailing_sl
                    logger.debug("[TRAIL] %s SL trailed to %.4f", symbol, trailing_sl)

    def _refresh_mtf_setup(self, symbol: str) -> None:
        """Aktualisiert MTF-Bias (alle 30min) und Retest-Setup (jede on_candle Iteration)."""
        if symbol not in self._mtf_analyzers:
            return
        if time.time() - self._mtf_bias_ts.get(symbol, 0) > 1800:
            try:
                bias = self._mtf_analyzers[symbol].refresh_bias()
                self._mtf_bias[symbol] = bias
                self._mtf_bias_ts[symbol] = time.time()
            except Exception as e:
                logger.warning("MTF bias refresh %s: %s", symbol, e)
        try:
            bias = self._mtf_bias.get(symbol, {})
            self._mtf_setup[symbol] = self._mtf_analyzers[symbol].find_retest_setup(bias)
        except Exception as e:
            logger.warning("MTF setup refresh %s: %s", symbol, e)
        try:
            if not os.getenv("GRIDBOT_BACKTEST"):
                from dashboard.db import update_mtf_state
                update_mtf_state(symbol, self._mtf_bias.get(symbol, {}), self._mtf_setup.get(symbol))
        except Exception:
            pass

    def _check_mtf_entry(self, symbol: str, price: float,
                         state: _GridState, ctx: MarketContext) -> None:
        """Prüft 5m-Einstiegs-Trigger innerhalb der MTF-Retest-Zone."""
        if symbol not in self._mtf_analyzers or state._directional:
            return
        setup = self._mtf_setup.get(symbol)
        if not setup:
            return
        try:
            trigger = self._mtf_analyzers[symbol].check_entry_trigger(setup, price)
        except Exception as e:
            logger.debug("MTF entry check %s: %s", symbol, e)
            return
        if not trigger:
            return
        logger.info("[MTF] %s Entry-Trigger @ %.4f RSI=%.1f (zone %.4f–%.4f)",
                    symbol, trigger["trigger_price"], trigger["rsi"],
                    setup["zone_low"], setup["zone_high"])
        try:
            if not os.getenv("GRIDBOT_BACKTEST"):
                import notifier
                notifier.notify_mtf_zone_reached(
                    symbol, setup["direction"],
                    setup["zone_low"], setup["zone_high"],
                    price, setup["level"]
                )
        except Exception:
            pass
        # Bot ist long-only — SHORT-MTF-Setups nur als Alarm, kein Auto-Execute
        if setup["direction"] != "long":
            return
        # Auto-execute only if enabled in dashboard coin settings
        auto_exec = False
        try:
            if not os.getenv("GRIDBOT_BACKTEST"):
                from dashboard.db import get_mtf_auto_execute
                auto_exec = get_mtf_auto_execute(symbol)
        except Exception:
            pass
        if not auto_exec or not self._buys_allowed(state):
            return
        usdt = state.investment * self.p.directional_pct
        if self._risk is not None:
            allowed, reason = self._risk.can_open(symbol, usdt, ctx)
            if not allowed:
                logger.debug("[MTF] %s blocked by risk: %s", symbol, reason)
                return
        lev = self._lev()
        qty = usdt * lev / price
        tp = setup["target"]
        atr = setup.get("atr") or state._atr or price * 0.02
        sl = (setup["zone_low"] - atr if setup["direction"] == "long"
              else setup["zone_high"] + atr)
        state._directional = {
            "entry": price, "qty": qty, "usdt": usdt,
            "tp": tp, "sl": sl, "entry_ts": time.time(), "mtf": True,
        }
        logger.info("[MTF] %s AUTO DIRECTIONAL %s @ %.4f | TP=%.4f SL=%.4f",
                    symbol, setup["direction"].upper(), price, tp, sl)

    def _check_position_stops(self, symbol: str, price: float,
                               state: _GridState, ctx: MarketContext):
        lev = self._lev()
        for cid, order in list(state.orders.items()):
            if order.get("filled") or order["side"] != "sell":
                continue
            if "sl_price" not in order or "bought_at" not in order:
                continue

            # Momentum-hold: if score is strong bullish, delay SL by one cycle
            if price <= order["sl_price"]:
                holds = order.get("momentum_holds", 0)
                if state._direction_score > self.p.momentum_hold_score and holds < self.p.momentum_hold_max:
                    order["momentum_holds"] = holds + 1
                    logger.debug("[HOLD] %s SL delayed (score=%.2f, hold=%d)",
                                 symbol, state._direction_score, holds + 1)
                    continue

                buy_price = order["bought_at"]
                qty = order["qty"]
                profit = (price - buy_price) * qty
                fee = (price + buy_price) * qty * KRAKEN_FEE
                net = profit - fee
                state.total_profit += net
                state.trade_count += 1
                order["filled"] = True
                logger.warning("[SL] %s @ %.4f | bought @ %.4f | net=%.4f",
                               symbol, price, buy_price, net)
                try:
                    if os.getenv("GRIDBOT_BACKTEST"):
                        raise ImportError("backtest: dashboard logging disabled")
                    from dashboard.db import log_trade
                    log_trade(symbol, "SELL", buy_price, price, net,
                              "stop_loss", "grid", "paper",
                              context={
                                  "regime": state._last_regime,
                                  "ml_prediction": state._last_prediction,
                                  "ml_confidence": state._last_confidence,
                              },
                              leverage=lev)
                except Exception:
                    pass
                # Credit margin + PnL back to the paper broker balance.
                # The broker never sees the SL fill (the sell order is just
                # cancelled in _sync_orders), so without this the margin
                # from the original buy fill is permanently lost.
                if self._broker is not None:
                    try:
                        from execution.paper import PaperBroker
                        if isinstance(self._broker, PaperBroker):
                            # #57: return margin using the leverage the position
                            # was *entered* with (stored on the order), not the
                            # possibly-changed live leverage — otherwise the
                            # margin credited back drifts from what the buy fill
                            # deducted whenever the user changes leverage.
                            entry_lev = order.get("leverage", lev)
                            # #39: only the sell-side fee here. The buy fee was
                            # already deducted once at buy-fill time
                            # (paper.py update_price), so charging a round-trip
                            # fee would double-count the buy fee.
                            sl_fee = price * qty * KRAKEN_FEE
                            credit = buy_price * qty / entry_lev + (price - buy_price) * qty - sl_fee
                            self._broker.sl_credit(symbol, credit)
                    except Exception:
                        pass
                ctx.remove_position(symbol, "grid")
                # Remove from orders after SL
                state.orders.pop(cid, None)

    def _maybe_open_directional(self, symbol: str, price: float,
                                 state: _GridState, ctx: MarketContext):
        if not self.p.directional_enabled or not self._buys_allowed(state):
            return
        if state._directional:
            return

        # Negative-EV hour filter (UTC 05-08)
        if datetime.now(timezone.utc).hour in NO_DIRECTIONAL_HOURS:
            return

        # Cooloff after SL
        if (time.time() - state._directional_sl_ts) < DIRECTIONAL_COOLOFF_SECONDS:
            return

        if state._directional_needs_recheck:
            if state._last_prediction != "up" or state._direction_score < DIRECTIONAL_RECHECK_SCORE_MIN:
                return
            state._directional_needs_recheck = False

        if state._last_prediction != "up" or state._direction_score < self.p.directional_score_min:
            return

        # BTC context guard
        btc = ctx.get_btc()
        if btc and (btc.trend == "down" or btc.return_1h < -0.03):
            return

        atr = state._atr if state._atr > 0 else price * 0.02
        usdt = state.investment * self.p.directional_pct
        lev = self._lev()

        # RiskManager gate
        if self._risk is not None:
            allowed, reason = self._risk.can_open(symbol, usdt, ctx)
            if not allowed:
                logger.debug("[DIRECTIONAL] %s blocked by risk: %s", symbol, reason)
                return

        qty = usdt * lev / price
        tp = price + self.p.directional_tp_atr * atr
        sl = price - self.p.directional_sl_atr * atr

        state._directional = {
            "entry": price, "qty": qty, "usdt": usdt,
            "tp": tp, "sl": sl, "entry_ts": time.time(),
        }
        logger.info("[DIRECTIONAL] %s BUY @ %.4f | TP=%.4f SL=%.4f | score=%.2f",
                    symbol, price, tp, sl, state._direction_score)

    def _check_directional(self, symbol: str, price: float,
                            state: _GridState, ctx: MarketContext):
        if not state._directional:
            return
        d = state._directional
        hit_tp = price >= d["tp"]
        hit_sl = price <= d["sl"]

        pnl_pct = (price - d["entry"]) / d["entry"]
        score = state._direction_score
        signal_down = score < 0

        do_flip_sell = False
        if signal_down and pnl_pct >= 0:
            if "signal_down_price" not in d:
                d["signal_down_price"] = price
            trail_trigger = d["signal_down_price"] * (1 - DIRECTIONAL_DOWN_TRAIL_PCT)
            if price <= trail_trigger:
                do_flip_sell = True
        elif not signal_down and "signal_down_price" in d:
            del d["signal_down_price"]

        if not (hit_tp or hit_sl or do_flip_sell):
            return

        lev = self._lev()
        pnl = (price - d["entry"]) * d["qty"]
        fee = (price + d["entry"]) * d["qty"] * KRAKEN_FEE
        net = pnl - fee
        state.total_profit += net
        state.trade_count += 1
        reason = "TP" if hit_tp else ("SL" if hit_sl else "Signal-Flip")
        logger.info("[DIRECTIONAL] %s SELL @ %.4f | %s | net=%.4f", symbol, price, reason, net)

        if hit_sl:
            state._directional_needs_recheck = True
            state._directional_sl_ts = time.time()

        try:
            if os.getenv("GRIDBOT_BACKTEST"):
                raise ImportError("backtest: dashboard logging disabled")
            from dashboard.db import log_trade
            log_trade(symbol, "DIRECTIONAL", d["entry"], price, net,
                      f"directional_{reason.lower()}", "grid", "paper",
                      context={
                          "regime": state._last_regime,
                          "ml_prediction": state._last_prediction,
                          "ml_confidence": state._last_confidence,
                          "atr_pct": state.range_pct * 0.5,
                      },
                      leverage=lev)
        except Exception:
            pass

        try:
            if os.getenv("GRIDBOT_BACKTEST"):
                raise ImportError("backtest: notifier disabled")
            import notifier
            notifier.notify_trade_close(symbol, "DIRECTIONAL", d["entry"], price, net,
                                        f"directional_{reason.lower()}")
        except Exception:
            pass

        state._directional = {}

    def setup_grid(self, symbol: str, price: float, ctx: MarketContext,
                   lower: float = None, upper: float = None):
        state = self._states.get(symbol)
        if not state:
            return

        lower_p, upper_p, levels, range_pct, regime, confidence = self._build_grid_params(
            symbol, price, state
        )
        if lower is None:
            lower = lower_p
        if upper is None:
            upper = upper_p

        state.levels = levels
        state.range_pct = range_pct
        state._last_regime = regime
        state._last_confidence = confidence
        state._last_pred_low = lower
        state._last_pred_high = upper
        state.usdt_per_grid = state.investment / levels

        lev = self._lev()
        step = (upper - lower) / levels
        grid_lines = [lower + (i + 0.5) * step for i in range(levels)]
        state.grid_lines = grid_lines

        state.grid_lower = lower
        if self.p.sl_mode == "floor":
            atr = state._atr if state._atr > 0 else price * 0.02
            # NOTE (#62): the floor VALUE deliberately tracks the (possibly lower)
            # new grid bottom. It is NOT ratcheted upward — a global ratchet would
            # place new buy cohorts under an old, higher floor and stop them out on
            # fill. Only *open* positions' sl_price is ratcheted (never lowered),
            # done below at the open_positions loop.
            state.floor_sl = max(lower - self.p.floor_sl_atr_mult * atr, 0.0)

        # Adaptive sizing: more budget on lower levels when bullish
        allocations = _calc_level_allocations(grid_lines, price, state.investment,
                                              state._direction_score)

        # Preserve open positions through rebuild (real filled buys with pending sells).
        # Excludes pre_seeded sells: those are placeholder walls, not real positions,
        # and must be rebuilt fresh each cycle — otherwise they never get removed and
        # accumulate across rebuilds (state.orders growing without bound).
        open_positions = {
            cid: o for cid, o in state.orders.items()
            if not o.get("filled") and o.get("side") == "sell" and "bought_at" in o
            and not o.get("pre_seeded")
        }
        # Floor ratchet: never lower an existing position's SL on rebuild.
        # Skipped in per-cohort mode — each cohort keeps its own floor from
        # when it was seeded, so survivors should NOT be ratcheted to the
        # new (potentially lower or different) global floor.
        if self.p.sl_mode == "floor" and state.floor_sl > 0 and not self.p.floor_sl_per_cohort:
            for o in open_positions.values():
                if "sl_price" in o:
                    o["sl_price"] = max(o["sl_price"], state.floor_sl)
        state.orders = dict(open_positions)

        # Build new price → client_id map, recycling existing IDs where possible
        old_price_to_id = state.price_to_id
        state.price_to_id = {}

        # Evaluate once — covers all three buy-creation paths (setup, desired_orders,
        # smart-replenish) through this single gate.  Gating setup_grid here prevents
        # monotone inventory accumulation during downtrends (B in the plan).  The
        # engine's cancel-on-absent in _sync_orders will also retract any resting
        # pending buys whose client_ids are absent from desired_orders after this.
        buys_ok = self._buys_allowed(state)

        for i, gp in enumerate(grid_lines):
            # Reuse existing client_id for this price level if unchanged (avoids cancel-storm)
            existing_cid = old_price_to_id.get(gp)
            cid = existing_cid if existing_cid and existing_cid not in state.orders else str(uuid.uuid4())

            usdt = allocations.get(gp, state.usdt_per_grid)
            qty = usdt * lev / gp

            if gp < price:
                if buys_ok:
                    buy_order: dict = {"side": "buy", "price": gp, "qty": qty, "filled": False}
                    # In per-cohort floor mode, stamp each buy with the current cohort's
                    # floor so _handle_buy_fill can assign an individual sl_price.
                    if self.p.floor_sl_per_cohort and self.p.sl_mode == "floor" and state.floor_sl > 0:
                        buy_order["cohort_floor"] = state.floor_sl
                    state.orders[cid] = buy_order
                    state.price_to_id[gp] = cid
                # else: buy seeding blocked by inventory cap or trend filter.
                # desired_orders won't emit this level; engine _sync_orders will
                # cancel any already-resting pending buy order at this price.
            else:
                # Pre-seed sells above price so grid is two-sided from the start.
                # bought_at = current price (not a grid line above price) so that:
                #   a) P&L is positive when the sell fills
                #   b) no sl_price → _check_position_stops skips these (no phantom SL fires)
                sell_qty = usdt * lev / gp
                state.orders[cid] = {
                    "side": "sell", "price": gp, "qty": sell_qty, "filled": False,
                    "bought_at": price, "pre_seeded": True,
                }
                state.price_to_id[gp] = cid

        logger.info("[GRID] Built %s: %.4f–%.4f | %d levels | ±%.1f%% | regime=%s | lev=%.1f×",
                    symbol, lower, upper, levels, range_pct * 100, regime, lev)

    def get_state(self, symbol: str) -> Optional[_GridState]:
        return self._states.get(symbol)

    def status(self, symbol: str) -> dict:
        state = self._states.get(symbol)
        if not state:
            return {}
        return {
            "trade_count": state.trade_count,
            "total_profit": state.total_profit,
            "prediction": state._last_prediction,
            "score": state._direction_score,
            "regime": state._last_regime,
            "directional_open": bool(state._directional),
        }
