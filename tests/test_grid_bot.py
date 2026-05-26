"""
Tests for the new modular trading bot architecture.
Run: python3 -m pytest tests/ -v
"""

import time
import uuid

import numpy as np
import pandas as pd
import pytest


# ── PaperBroker ───────────────────────────────────────────────────────────────

class TestPaperBroker:

    def _broker(self, balance=1000.0):
        from execution.paper import PaperBroker
        return PaperBroker(initial_balance=balance)

    def test_buy_fills_when_price_hits(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        fills = broker.update_price("SOL/USD", 100.0)
        assert len(fills) == 1
        assert fills[0].side == "buy"
        assert fills[0].price > 100.0  # slippage applied

    def test_sell_fills_when_price_hits(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "sell", 110.0, 1.0, client_id="s1")
        broker.update_price("SOL/USD", 100.0)  # advance tick
        fills = broker.update_price("SOL/USD", 110.0)
        assert len(fills) == 1
        assert fills[0].side == "sell"

    def test_slippage_applied(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        fills = broker.update_price("SOL/USD", 100.0)
        assert len(fills) == 1
        assert fills[0].price > 100.0  # buy: price + slippage

    def test_balance_reduced_on_buy(self):
        broker = self._broker(balance=1000.0)
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        broker.update_price("SOL/USD", 100.0)
        assert broker.get_balance() < 1000.0

    def test_cancel_removes_order(self):
        broker = self._broker()
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        result = broker.cancel("b1")
        assert result is True
        fills = broker.update_price("SOL/USD", 99.0)
        assert len(fills) == 0

    def test_insufficient_balance_no_fill(self):
        broker = self._broker(balance=50.0)
        broker.place_limit("SOL/USD", "buy", 100.0, 1.0, client_id="b1")
        fills = broker.update_price("SOL/USD", 99.0)
        assert len(fills) == 0  # can't afford 100 USDT order with 50 balance


# ── Risk Sizing ───────────────────────────────────────────────────────────────

class TestRiskSizing:

    def test_kelly_fraction_basic(self):
        from risk.sizing import kelly_fraction
        f = kelly_fraction(0.55, 1.3, kelly_factor=0.25)
        assert 0 < f <= 0.25

    def test_kelly_fraction_bad_winrate(self):
        from risk.sizing import kelly_fraction
        f = kelly_fraction(0.0, 1.0)
        assert f > 0  # returns floor value

    def test_position_size_scales_with_equity(self):
        from risk.sizing import compute_position_usdt
        size_1k = compute_position_usdt(1000, 0.55, 1.3, 0.03)
        size_2k = compute_position_usdt(2000, 0.55, 1.3, 0.03)
        assert size_2k > size_1k

    def test_position_size_smaller_with_higher_vol(self):
        from risk.sizing import vol_target_size
        size_low_vol = vol_target_size(1000, 0.01, 0.02, 100.0)
        size_high_vol = vol_target_size(1000, 0.01, 0.08, 100.0)
        assert size_low_vol > size_high_vol

    def test_position_size_capped(self):
        from risk.sizing import compute_position_usdt
        size = compute_position_usdt(1000, 0.99, 10.0, 0.001, max_position_pct=0.10)
        assert size <= 100.0  # max 10% of 1000


# ── ML Features ──────────────────────────────────────────────────────────────

def _make_df(n=100):
    np.random.seed(42)
    prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.003,
        "low":    prices * 0.997,
        "close":  prices,
        "volume": np.random.uniform(1000, 5000, n),
    }, index=idx)


class TestMLFeatures:

    def test_technical_features_shape(self):
        from ml.features.technical import extract, FEATURE_NAMES
        df = _make_df()
        feats = extract(df)
        assert feats.shape == (len(FEATURE_NAMES),)
        assert not np.isnan(feats).any()

    def test_combined_features_shape(self):
        from ml.features.combined import extract_all, N_FEATURES
        df = _make_df()
        feats = extract_all(df)
        assert feats.shape == (N_FEATURES,)
        assert feats.shape == (34,)
        assert not np.isnan(feats).any()

    def test_perp_features_zeros_when_no_data(self):
        from ml.features.perp import extract
        feats = extract(None)
        assert (feats == 0).all()

    def test_market_features_zeros_when_no_data(self):
        from ml.features.market import extract
        feats = extract(None)
        assert (feats == 0).all()

    def test_seasonality_features_cyclic(self):
        from ml.features.seasonality import extract
        from datetime import datetime
        feats_midnight = extract(datetime(2024, 1, 1, 0, 0))
        feats_noon = extract(datetime(2024, 1, 1, 12, 0))
        assert feats_midnight[0] != feats_noon[0]

    def test_htf_features_no_crash(self):
        from ml.features.htf import extract
        df = _make_df(200)
        feats = extract(df)
        assert feats.shape[0] == 4
        assert not np.isnan(feats).any()


# ── Backtest Metrics ─────────────────────────────────────────────────────────

class TestBacktestMetrics:

    def test_sharpe_positive_returns(self):
        from backtest.metrics import sharpe
        import numpy as np
        np.random.seed(1)
        returns = list(0.01 + np.random.normal(0, 0.002, 100))
        s = sharpe(returns)
        assert s > 0

    def test_max_drawdown_negative(self):
        from backtest.metrics import max_drawdown
        equity = [1000, 1100, 900, 950, 1050]
        dd = max_drawdown(equity)
        assert dd < 0

    def test_hit_rate_correct(self):
        from backtest.metrics import hit_rate
        pnls = [10, -5, 8, -3, 12]
        assert hit_rate(pnls) == pytest.approx(0.6)

    def test_profit_factor(self):
        from backtest.metrics import profit_factor
        pnls = [10, -5, 8, -4]
        pf = profit_factor(pnls)
        assert pf == pytest.approx(18 / 9, rel=0.01)

    def test_summary_returns_dict(self):
        from backtest.metrics import summary
        pnls = [5, -2, 8, -3, 6]
        equity = [1000, 1005, 1003, 1011, 1008, 1014]
        result = summary(pnls, equity, days=30)
        assert "sharpe" in result
        assert "max_drawdown_pct" in result
        assert "hit_rate_pct" in result
        assert "profit_factor" in result


# ── Risk Manager ─────────────────────────────────────────────────────────────

class TestRiskManager:

    def _make_rm(self):
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager
        return RiskManager(CorrelationTracker())

    def test_can_open_basic(self):
        from core.context import MarketContext
        rm = self._make_rm()
        ctx = MarketContext()
        ctx.set_equity(1000.0)
        rm.set_daily_start(1000.0)
        ok, reason = rm.can_open("SOL/USD", 50.0, ctx)
        assert ok is True

    def test_blocks_on_daily_drawdown(self):
        from core.context import MarketContext
        rm = self._make_rm()
        ctx = MarketContext()
        ctx.set_equity(1000.0)
        rm.set_daily_start(1040.0)  # started at 1040 → 3.8% loss
        ok, reason = rm.can_open("SOL/USD", 50.0, ctx)
        assert ok is False
        assert "drawdown" in reason

    def test_blocks_on_btc_crash(self):
        from core.context import MarketContext, BTCContext
        rm = self._make_rm()
        ctx = MarketContext()
        ctx.set_equity(1000.0)
        rm.set_daily_start(1000.0)
        ctx.set_btc(BTCContext(
            trend="down", return_1h=-0.05, return_4h=-0.10,
            return_24h=-0.15, realized_vol_7d=0.8, dominance=0.5
        ))
        ok, reason = rm.can_open("SOL/USD", 50.0, ctx)
        assert ok is False
        assert "btc_crash" in reason


# ── GridStrategy ─────────────────────────────────────────────────────────────

class TestGridStrategy:

    def _strategy(self):
        from strategies.grid import GridStrategy
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}])

    def test_setup_grid_creates_orders(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")
        # ranging regime → 14 levels; grid_lines and orders are non-empty
        assert len(state.orders) > 0
        assert len(state.grid_lines) > 0

    def test_buy_fills_creates_sell(self):
        from core.context import MarketContext
        from core.strategy import Fill
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")

        buy_orders = [(cid, o) for cid, o in state.orders.items() if o["side"] == "buy"]
        assert len(buy_orders) > 0
        cid, order = buy_orders[0]

        sells_before = sum(1 for o in state.orders.values() if o["side"] == "sell")
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)

        sells_after = sum(1 for o in state.orders.values() if o["side"] == "sell")
        assert sells_after > sells_before

    def test_stop_loss_fires_on_tick(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        state = strategy.get_state("SOL/USD")

        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 105.0, "qty": 1.0,
            "filled": False, "bought_at": 100.0,
            "sl_price": 95.0, "trailing_activated": False,
        }

        strategy.on_tick("SOL/USD", 94.0, ctx)
        # After SL fires, order is removed from state.orders (cleanup behavior)
        assert cid not in state.orders, "SL order should be removed after firing"
        assert state.total_profit < 0  # SL → loss

    def test_trailing_stop_activates_at_breakeven(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = 2.0  # ATR = $2

        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 110.0, "qty": 1.0,
            "filled": False, "bought_at": 100.0,
            "sl_price": 96.0, "trailing_activated": False,
        }

        # Price moves up 1×ATR above entry (buy=100, ATR=2 → 102)
        strategy._update_trailing_stops("SOL/USD", 102.5, state)
        assert state.orders[cid]["trailing_activated"] is True
        assert state.orders[cid]["sl_price"] >= 100.0  # SL moved to break-even

    def test_compounding_increases_investment(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.total_profit = 10.0
        state.trade_count = 3  # COMPOUND_EVERY_TRADES = 3
        initial = state.investment
        strategy._maybe_compound(100.0, state)
        assert state.investment > initial
