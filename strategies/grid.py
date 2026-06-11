"""
GridStrategy – grid trading using the Strategy ABC.

Grid builds limit buy orders below current price and limit sell orders above.
When a buy fills, a sell is placed one level higher (+ SL).
When a sell fills, a new buy is replenished.
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ta as ta_lib

from core.context import MarketContext, Position
from core.strategy import Fill, Order, Strategy

logger = logging.getLogger("strategies.grid")

# ── Constants ──────────────────────────────────────────────────────────────────
KRAKEN_FEE = 0.0016
MIN_STEP_FEE_MULTIPLE = 2.5
MAX_LOSS_PCT = 0.05
MOMENTUM_HOLD_SCORE = 0.35
MOMENTUM_HOLD_MAX = 2
COMPOUND_EVERY_TRADES = 3
MAX_INVESTMENT_MULT = 3.0          # Compounding cap: max 3× initial investment

ADAPTIVE_SIZING = True             # Bullish → more budget on lower levels
SIZE_BIAS_FACTOR = 0.30

# Directional trade config
DIRECTIONAL_ENABLED = True
DIRECTIONAL_SCORE_MIN = 0.12
DIRECTIONAL_PCT = 0.20
DIRECTIONAL_TP_ATR = 3.0
DIRECTIONAL_SL_ATR = 1.5
DIRECTIONAL_RECHECK_SCORE_MIN = 0.25
DIRECTIONAL_DOWN_TRAIL_PCT = 0.005
DIRECTIONAL_COOLOFF_SECONDS = 4 * 3600
NO_DIRECTIONAL_HOURS = frozenset({5, 6, 7, 8})   # UTC hours with negative EV


def _calc_level_allocations(grid_lines: list, current_price: float,
                             investment: float, direction_score: float) -> dict:
    """Non-uniform budget allocation: bullish → more on lower buy levels (DCA into dips)."""
    n = len(grid_lines)
    if not ADAPTIVE_SIZING or abs(direction_score) < 0.05 or n == 0:
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

        # Directional trade state
        self._directional: dict = {}
        self._directional_needs_recheck = False
        self._directional_sl_ts: float = 0.0

        # ATR for trailing stops
        self._atr: float = 0.0
        self._last_df: object = None


class GridStrategy(Strategy):
    name = "grid"

    def __init__(self, grids_config: list, risk_manager=None, ml_enabled: bool = True):
        self._config = {g["symbol"]: g for g in grids_config}
        self._risk = risk_manager
        self._ml_enabled = ml_enabled
        self._states: Dict[str, _GridState] = {}
        self._ml_predictor = None
        self._price_predictors: dict = {}

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

        self._refresh_prediction(symbol, df, ctx)

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
        state.with_position = direction != "down"

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
                    half = state._atr * 2.0
                    lower, upper, regime = price - half, price + half, "trending"
                elif atr_pct > 0.03:
                    half = state._atr * 1.5
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
        regime_configs = {"ranging": 14, "trending": 6, "volatile": 20}
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

        return lower, upper, levels, range_pct, regime or "ranging", confidence or 0.5

    def desired_orders(self, symbol: str, price: float, ctx: MarketContext) -> List[Order]:
        state = self._states.get(symbol)
        if not state or not state.grid_lines:
            return []

        orders = []
        for cid, o in state.orders.items():
            if o.get("filled"):
                continue
            if not state.with_position and o["side"] == "buy":
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
                meta={"strategy": "grid", **o},
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
        lev = _get_leverage()
        qty = order["qty"]  # qty already has leverage factored in at setup

        try:
            idx = state.grid_lines.index(order["price"])
        except ValueError:
            idx = None

        if idx is not None and idx < len(state.grid_lines) - 1:
            sell_price = state.grid_lines[idx + 1]
            step_pct = (sell_price - buy_price) / buy_price
            sl_pct = max(step_pct * 1.5, 0.008)
            sl_price = buy_price * (1 - sl_pct)

            sell_cid = str(uuid.uuid4())
            state.orders[sell_cid] = {
                "side": "sell",
                "price": sell_price,
                "qty": qty,
                "filled": False,
                "bought_at": buy_price,
                "sl_price": sl_price,
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
        lev = _get_leverage()

        profit = (sell_price - buy_price) * qty
        fee = (sell_price + buy_price) * qty * KRAKEN_FEE
        net = profit - fee
        state.total_profit += net
        state.trade_count += 1

        holding_seconds = time.time() - order.get("entry_ts", time.time())
        logger.info("[GRID] SELL fill %s @ %.4f | bought @ %.4f | net=%.4f USDT",
                    fill.symbol, sell_price, buy_price, net)

        try:
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
            import notifier
            notifier.notify_trade_close(fill.symbol, "GRID", buy_price, sell_price, net, "grid_fill")
        except Exception:
            pass

        # Smart-replenish: follow trend one level higher when bullish
        if state.with_position:
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
        self._update_trailing_stops(symbol, price, state)

        return []

    def on_tick_safety(self, symbol: str, price: float, ctx: MarketContext):
        """SL/TP checks that run even during freeze (never skip stop-losses)."""
        state = self._states.get(symbol)
        if state:
            self._check_position_stops(symbol, price, state, ctx)

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

    def _check_position_stops(self, symbol: str, price: float,
                               state: _GridState, ctx: MarketContext):
        lev = _get_leverage()
        for cid, order in list(state.orders.items()):
            if order.get("filled") or order["side"] != "sell":
                continue
            if "sl_price" not in order or "bought_at" not in order:
                continue

            # Momentum-hold: if score is strong bullish, delay SL by one cycle
            if price <= order["sl_price"]:
                holds = order.get("momentum_holds", 0)
                if state._direction_score > MOMENTUM_HOLD_SCORE and holds < MOMENTUM_HOLD_MAX:
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
                ctx.remove_position(symbol, "grid")
                # Remove from orders after SL
                state.orders.pop(cid, None)

    def _maybe_open_directional(self, symbol: str, price: float,
                                 state: _GridState, ctx: MarketContext):
        if not DIRECTIONAL_ENABLED or not state.with_position:
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

        if state._last_prediction != "up" or state._direction_score < DIRECTIONAL_SCORE_MIN:
            return

        # BTC context guard
        btc = ctx.get_btc()
        if btc and (btc.trend == "down" or btc.return_1h < -0.03):
            return

        atr = state._atr if state._atr > 0 else price * 0.02
        usdt = state.investment * DIRECTIONAL_PCT
        lev = _get_leverage()

        # RiskManager gate
        if self._risk is not None:
            allowed, reason = self._risk.can_open(symbol, usdt, ctx)
            if not allowed:
                logger.debug("[DIRECTIONAL] %s blocked by risk: %s", symbol, reason)
                return

        qty = usdt * lev / price
        tp = price + DIRECTIONAL_TP_ATR * atr
        sl = price - DIRECTIONAL_SL_ATR * atr

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

        lev = _get_leverage()
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

        lev = _get_leverage()
        step = (upper - lower) / levels
        grid_lines = [lower + (i + 0.5) * step for i in range(levels)]
        state.grid_lines = grid_lines

        # Adaptive sizing: more budget on lower levels when bullish
        allocations = _calc_level_allocations(grid_lines, price, state.investment,
                                              state._direction_score)

        # Preserve open positions through rebuild (real filled buys with pending sells)
        open_positions = {
            cid: o for cid, o in state.orders.items()
            if not o.get("filled") and o.get("side") == "sell" and "bought_at" in o
        }
        state.orders = dict(open_positions)

        # Build new price → client_id map, recycling existing IDs where possible
        old_price_to_id = state.price_to_id
        state.price_to_id = {}

        for i, gp in enumerate(grid_lines):
            # Reuse existing client_id for this price level if unchanged (avoids cancel-storm)
            existing_cid = old_price_to_id.get(gp)
            cid = existing_cid if existing_cid and existing_cid not in state.orders else str(uuid.uuid4())

            usdt = allocations.get(gp, state.usdt_per_grid)
            qty = usdt * lev / gp

            if gp < price:
                state.orders[cid] = {"side": "buy", "price": gp, "qty": qty, "filled": False}
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
