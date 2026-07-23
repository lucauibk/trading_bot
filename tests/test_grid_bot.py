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
        from ml.features.market import extract, FEATURE_NAMES
        feats = extract(None)
        # btc_corr_30d defaults to a neutral 0.5 prior when btc context is missing
        # (commit d67be55); all other market features are 0.
        corr_idx = FEATURE_NAMES.index("btc_corr_30d")
        assert feats[corr_idx] == 0.5
        assert (np.delete(feats, corr_idx) == 0).all()

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
        # max_daily_drawdown is 0.10 (10%) per config.yaml (raised from 0.03 in dd0531b).
        # Use an 11% loss so the check still triggers regardless of the configured value.
        rm.set_daily_start(1124.0)  # 1000/1124 − 1 = −11.0% loss → exceeds 10% threshold
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
        state = strategy.get_state("SOL/USD")
        state.with_position = True  # mirror _refresh_prediction: buys only seed when with_position
        strategy.setup_grid("SOL/USD", 100.0, ctx)

        buy_orders = [(cid, o) for cid, o in state.orders.items() if o["side"] == "buy"]
        assert len(buy_orders) > 0
        cid, order = buy_orders[0]

        sells_before = sum(1 for o in state.orders.values() if o["side"] == "sell")
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)

        sells_after = sum(1 for o in state.orders.values() if o["side"] == "sell")
        assert sells_after > sells_before

    def test_per_pos_sl_max_pct_regime_override(self):
        """Ultra-Bot-Plan Phase 2: per_pos_sl_max_pct_by_regime overrides the
        flat cap for the listed regime and falls back to the default for
        others (strategies/grid.py's sl_max_pct_for_regime call sites)."""
        from core.context import MarketContext
        from core.strategy import Fill
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams

        params = GridParams.from_dict({
            "sl_mode": "per_position",
            "per_pos_sl_step_mult": 100.0,  # force sl_pct to hit the cap, not the step value
            "per_pos_sl_max_pct_by_regime": {"trending": 0.03},
        })
        strategy = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}], params=params)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True
        strategy.setup_grid("SOL/USD", 100.0, ctx)

        def _fill_one_buy_and_get_sl_pct():
            cid, order = next((c, o) for c, o in state.orders.items()
                               if o["side"] == "buy" and not o["filled"])
            fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                        price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
            strategy.on_fill(fill, ctx)
            sell = next(o for o in state.orders.values()
                        if o["side"] == "sell" and o.get("bought_at") == order["price"])
            return (order["price"] - sell["sl_price"]) / order["price"], sell

        state._last_regime = "trending"
        sl_pct_trending, sell1 = _fill_one_buy_and_get_sl_pct()
        assert sl_pct_trending == pytest.approx(0.03, abs=1e-6)

        state._last_regime = "ranging"
        sl_pct_ranging, sell2 = _fill_one_buy_and_get_sl_pct()
        assert sl_pct_ranging == pytest.approx(0.04, abs=1e-6)

    def test_per_position_sl_fires_on_tick(self):
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

    def test_pre_seeded_fill_books_no_pnl_and_no_cash(self):
        """P0-Regression (Review 2026-07-02): Pre-seeded Sells (kein echter Kauf)
        dürfen weder PnL buchen noch Cash krediteren — sonst Phantom-Profit."""
        from core.context import MarketContext
        from core.strategy import Fill
        from execution.paper import PaperBroker

        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")

        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 101.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "pre_seeded": True,
        }
        fill = Fill(client_id=cid, symbol="SOL/USD", side="sell",
                    price=101.0, qty=1.0, fee=0.16, ts=time.time())
        strategy.on_fill(fill, ctx)
        assert state.total_profit == 0.0, "Seed-Fill darf kein PnL buchen"
        assert state.trade_count == 0, "Seed-Fill ist kein Trade"

        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        broker.place_limit("SOL/USD", "sell", 101.0, 1.0,
                           meta={"pre_seeded": True, "bought_at": 100.0, "leverage": 3.0})
        broker.update_price("SOL/USD", 100.0)
        fills = broker.update_price("SOL/USD", 102.0)
        assert len(fills) == 1
        assert broker.get_balance() == 100.0, "Seed-Fill darf kein Cash krediteren"

    def test_orphan_buy_fill_is_adopted_not_dropped(self):
        """P0-Regression (Review 2026-07-02): Buy-Fill für einen cid, den ein
        Rebuild entfernt hat, darf nicht still verworfen werden (Margin-Leck) —
        die Position wird adoptiert (Sell + SL angelegt)."""
        from core.context import MarketContext
        from core.strategy import Fill

        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True
        strategy.setup_grid("SOL/USD", 100.0, ctx)

        ghost_cid = str(uuid.uuid4())  # nicht in state.orders (Rebuild hat ihn entfernt)
        fill = Fill(client_id=ghost_cid, symbol="SOL/USD", side="buy",
                    price=95.0, qty=1.0, fee=0.15, ts=time.time(),
                    meta={"leverage": 3.0})
        sells_before = sum(1 for o in state.orders.values()
                           if o["side"] == "sell" and "bought_at" in o and not o.get("pre_seeded"))
        strategy.on_fill(fill, ctx)
        sells_after = sum(1 for o in state.orders.values()
                          if o["side"] == "sell" and "bought_at" in o and not o.get("pre_seeded"))
        assert sells_after == sells_before + 1, "Orphan-Buy muss als Position adoptiert werden"
        adopted = [o for o in state.orders.values()
                   if o["side"] == "sell" and o.get("bought_at") == 95.0][0]
        assert adopted["sl_price"] < 95.0
        assert adopted["price"] > 95.0

    def test_remove_position_removes_only_one(self):
        """P1-Regression (Review 2026-07-02): remove_position darf nur EINE
        Position entfernen, nicht alle einer Seite — sonst zählt der
        RiskManager (max_open_positions, Korrelations-Bucket) falsch."""
        from core.context import MarketContext, Position
        ctx = MarketContext()
        for ep in (100.0, 98.0, 96.0):
            ctx.add_position(Position("SOL/USD", "grid", ep, 1.0, ep, 3.0))
        ctx.remove_position("SOL/USD", "grid", entry_price=98.0, qty=1.0)
        assert ctx.open_position_count() == 2
        remaining = {p.entry_price for p in ctx.get_positions("SOL/USD")}
        assert remaining == {100.0, 96.0}
        ctx.remove_position("SOL/USD", "grid")  # ohne Match-Hinweis: älteste
        assert ctx.open_position_count() == 1

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


# ── GridParams ────────────────────────────────────────────────────────────────

class TestGridParams:

    def test_defaults_match_legacy_behaviour(self):
        from strategies.grid_params import GridParams
        p = GridParams()
        # sl_mode default is "floor" (cascade-safe); per_position still available
        assert p.sl_mode == "floor"
        assert p.per_pos_sl_step_mult == 1.5
        assert p.per_pos_sl_min_pct == 0.008
        assert p.per_pos_sl_max_pct == 0.04   # hard-cap: no SL wider than 4%
        assert p.momentum_hold_score == 0.35
        assert p.momentum_hold_max == 2
        assert p.regime_levels == {"ranging": 14, "trending": 6, "volatile": 20}
        assert p.trend_filter_enabled is True  # sweep winner 2026-06-12
        assert p.min_step_pct == 0.0
        assert p.directional_enabled is True

    def test_from_dict_roundtrip(self):
        from strategies.grid_params import GridParams
        p = GridParams.from_dict({"sl_mode": "floor", "floor_sl_atr_mult": 1.5,
                                  "levels_by_regime": {"ranging": 10, "trending": 5}})
        d = p.to_dict()
        assert d["sl_mode"] == "floor"
        assert d["floor_sl_atr_mult"] == 1.5
        assert d["levels_by_regime"] == {"ranging": 10, "trending": 5}
        assert GridParams.from_dict(d) == p

    def test_from_dict_ignores_unknown_keys(self):
        from strategies.grid_params import GridParams
        p = GridParams.from_dict({"sl_mode": "floor", "nonsense_key": 42})
        assert p.sl_mode == "floor"


# ── Floor-SL ──────────────────────────────────────────────────────────────────

class TestFloorSL:

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"sl_mode": "floor", "leverage": 1.0, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _setup(self, strategy, price=100.0, atr=2.0):
        from core.context import MarketContext
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = atr
        state.with_position = True  # mirror _refresh_prediction: buys only seed when with_position
        strategy.setup_grid("SOL/USD", price, ctx)
        return ctx, state

    def test_buy_fill_uses_floor_sl(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0)
        ctx, state = self._setup(strategy)
        assert state.floor_sl > 0
        assert state.floor_sl == pytest.approx(state.grid_lower - 2.0)

        cid, order = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][0]
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)
        sells = [o for o in state.orders.values()
                 if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")]
        assert sells and all(o["sl_price"] == pytest.approx(state.floor_sl) for o in sells)

    def test_no_stop_inside_grid(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0, momentum_hold_max=0)
        ctx, state = self._setup(strategy)
        cid, order = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][0]
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)
        profit_before = state.total_profit
        # Price at the lowest grid line: inside the grid → no stop may fire
        strategy.on_tick("SOL/USD", min(state.grid_lines), ctx)
        assert state.total_profit == profit_before

    def test_floor_break_flushes_all_positions(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0, momentum_hold_max=0)
        ctx, state = self._setup(strategy)
        buys = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][:2]
        for cid, order in buys:
            fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                        price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
            strategy.on_fill(fill, ctx)
        open_pos = [o for o in state.orders.values()
                    if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")]
        assert len(open_pos) == 2

        strategy.on_tick("SOL/USD", state.floor_sl - 0.01, ctx)
        remaining = [o for o in state.orders.values()
                     if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")]
        assert remaining == []
        assert state.total_profit < 0

    def test_rebuild_never_lowers_sl(self):
        from core.strategy import Fill
        strategy = self._strategy(floor_sl_atr_mult=1.0)
        ctx, state = self._setup(strategy)
        cid, order = [(c, o) for c, o in state.orders.items() if o["side"] == "buy"][0]
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)
        pos = [o for o in state.orders.values()
               if o["side"] == "sell" and "sl_price" in o and not o.get("pre_seeded")][0]
        sl_before = pos["sl_price"]

        # Rebuild far lower → new floor far below; existing SL must NOT drop
        strategy.setup_grid("SOL/USD", 80.0, ctx)
        assert pos["sl_price"] >= sl_before


# ── Trend filter ──────────────────────────────────────────────────────────────

class TestTrendFilter:

    def _strategy(self):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"trend_filter_enabled": True, "leverage": 1.0})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _df(self, closes):
        closes = np.asarray(closes, dtype=float)
        idx = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
        return pd.DataFrame({
            "open": closes, "high": closes + 0.5, "low": closes - 0.5,
            "close": closes, "volume": 1000.0,
        }, index=idx)

    def test_downtrend_sets_flag_and_blocks_buys(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")

        down = self._df(np.linspace(120, 80, 200))
        strategy._update_trend_filter(state, down)
        assert state._hard_trend_down is True
        assert strategy._buys_allowed(state) is False

        state.with_position = True
        strategy.setup_grid("SOL/USD", 80.0, ctx)
        buys = [o for o in strategy.desired_orders("SOL/USD", 80.0, ctx) if o.side == "buy"]
        assert buys == []

    def test_hysteresis_needs_two_clear_candles(self):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._hard_trend_down = True

        up = self._df(np.linspace(80, 120, 200))
        strategy._update_trend_filter(state, up)
        assert state._hard_trend_down is True  # 1st clear candle: still paused
        strategy._update_trend_filter(state, up)
        assert state._hard_trend_down is False  # 2nd clear candle: resumed


# ── Ranging-Gate (research/00-hypothesen.md H2, hard variant) ─────────────────

class TestRangingGate:

    def _strategy(self, ranging_gate_enabled=True):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({
            "trend_filter_enabled": False,
            "ranging_gate_enabled": ranging_gate_enabled,
            "leverage": 1.0,
        })
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def test_ranging_regime_blocks_buys_when_gate_enabled(self):
        from core.context import MarketContext
        strategy = self._strategy(ranging_gate_enabled=True)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True
        state._last_regime = "ranging"
        assert strategy._buys_allowed(state) is False

    def test_trending_regime_unaffected_by_gate(self):
        from core.context import MarketContext
        strategy = self._strategy(ranging_gate_enabled=True)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True
        state._last_regime = "trending"
        assert strategy._buys_allowed(state) is True

    def test_ranging_regime_allowed_when_gate_disabled(self):
        from core.context import MarketContext
        strategy = self._strategy(ranging_gate_enabled=False)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True
        state._last_regime = "ranging"
        assert strategy._buys_allowed(state) is True  # Default aus — bestehendes Verhalten


# ── Fee-aware min step ────────────────────────────────────────────────────────

class TestMinStep:

    def test_levels_capped_to_min_step(self):
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"min_step_pct": 0.01, "leverage": 1.0})
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 20}],
                                ml_enabled=False, params=params)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = 1.0  # tight volatility → tight range → would violate min step

        strategy.setup_grid("SOL/USD", 100.0, ctx)
        step_pct = (state.grid_lines[1] - state.grid_lines[0]) / 100.0
        assert step_pct >= 0.0099


# ── Backtest equity fidelity ──────────────────────────────────────────────────

class TestBacktestEquity:

    def test_unrealized_losses_visible_in_equity_curve(self):
        """Open underwater positions must drag the equity curve down even
        without realized losses (regression for realized-only equity bug)."""
        from backtest.engine import run_backtest
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams

        # fast drop below grid floor (~91.6): fills resting buys before the
        # downtrend buy-pause can cancel them; price held below → losses stay unrealized.
        n_flat, n_drop, n_tail = 70, 4, 46
        closes = np.concatenate([
            100 + 0.3 * np.sin(np.arange(n_flat)),
            np.linspace(100, 88, n_drop),
            88 + 0.3 * np.sin(np.arange(n_tail)),
        ])
        idx = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": closes, "high": closes + 0.5, "low": closes - 0.5,
            "close": closes, "volume": 1000.0,
        }, index=idx)

        # Deep floor → nothing stops out → losses stay unrealized.
        # Trend filter off: it would block the buys this test depends on.
        params = GridParams.from_dict({
            "sl_mode": "floor", "floor_sl_atr_mult": 50.0,
            "directional_enabled": False, "leverage": 1.0,
            "trend_filter_enabled": False,
        })
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                                ml_enabled=False, params=params)
        metrics = run_backtest(strategy, df, "SOL/USD", initial_balance=100.0)
        assert min(metrics["equity_curve"]) < 100.0


# ── P0-Fixes 2026-07-07 (Issues #36, #48, #49, #63, #72) ─────────────────────

class TestP0Fixes:

    def _strategy(self):
        from strategies.grid import GridStrategy
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}])

    def _grid(self, price=100.0):
        from core.context import MarketContext
        strategy = self._strategy()
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True
        strategy.setup_grid("SOL/USD", price, ctx)
        return strategy, ctx, state

    def test_can_open_denies_uninitialized_equity(self):
        """#36: equity <= 0 muss verweigern, nicht alle Risk-Checks umgehen."""
        from core.context import MarketContext
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager
        rm = RiskManager(CorrelationTracker())
        ctx = MarketContext()  # total_equity startet bei 0
        ok, reason = rm.can_open("SOL/USD", 50.0, ctx)
        assert ok is False
        assert reason == "equity_uninitialized"

    def test_daily_start_snapshots_on_new_day(self):
        """#48: Tageswechsel muss die neue Equity snapshotten (kein No-Op)."""
        from datetime import date, timedelta
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager
        rm = RiskManager(CorrelationTracker())
        rm.set_daily_start(1000.0)
        rm.set_daily_start(1500.0)  # gleicher Tag → Baseline bleibt
        assert rm._daily_start == 1000.0
        rm._daily_date = date.today() - timedelta(days=1)
        rm.set_daily_start(1500.0)  # Tageswechsel → neue Baseline
        assert rm._daily_start == 1500.0

    def test_top_level_buy_fill_gets_sell_and_sl(self):
        """#49.1: Buy am obersten Grid-Level darf keine verwaiste Position
        ohne Sell/SL erzeugen."""
        from core.strategy import Fill
        strategy, ctx, state = self._grid()
        top = state.grid_lines[-1]
        cid = str(uuid.uuid4())
        state.orders[cid] = {"side": "buy", "price": top, "qty": 1.0, "filled": False}
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=top, qty=1.0, fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)
        sells = [o for o in state.orders.values()
                 if o["side"] == "sell" and o.get("bought_at") == top
                 and not o.get("pre_seeded")]
        assert len(sells) == 1, "Top-Level-Buy muss einen Paar-Sell bekommen"
        assert sells[0]["sl_price"] > 0
        assert sells[0]["price"] > top

    def test_buy_fill_replaces_preseeded_sell(self):
        """#49.2: Replenish-Sell am pre-seeded Boundary-Level darf nicht
        zusätzlich zum Platzhalter existieren (Over-Sell)."""
        from core.strategy import Fill
        strategy, ctx, state = self._grid(price=100.0)
        # höchster Buy: sein Sell-Level ist die erste Linie über dem Preis,
        # dort liegt nach setup_grid ein pre-seeded Sell
        buy_items = [(cid, o) for cid, o in state.orders.items() if o["side"] == "buy"]
        assert buy_items
        cid, order = max(buy_items, key=lambda kv: kv[1]["price"])
        idx = state.grid_lines.index(order["price"])
        sell_level = state.grid_lines[idx + 1]
        pre = [o for o in state.orders.values()
               if o["side"] == "sell" and o["price"] == sell_level and o.get("pre_seeded")]
        assert len(pre) == 1, "Testaufbau: pre-seeded Sell am Boundary-Level erwartet"

        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)

        sells_at_level = [o for o in state.orders.values()
                          if o["side"] == "sell" and o["price"] == sell_level
                          and not o.get("filled")]
        assert len(sells_at_level) == 1, "es darf nur EIN Sell an diesem Level liegen"
        assert not sells_at_level[0].get("pre_seeded")

    def test_replenish_lands_one_level_above_buy(self):
        """#72: bullisher Replenish muss EIN Level über dem Buy landen, nicht zwei."""
        strategy, ctx, state = self._grid()
        gl = state.grid_lines
        buy_price, sell_price = gl[1], gl[2]
        order = {"side": "sell", "price": sell_price, "qty": 1.0,
                 "filled": True, "bought_at": buy_price}
        state._direction_score = 0.5  # bullish → follow trend one level higher
        before = set(state.orders)
        strategy._replenish_after_sell(state, order, buy_price)
        new = [o for c, o in state.orders.items() if c not in before]
        assert len(new) == 1
        assert new[0]["price"] == gl[2], "Replenish gehört auf buy_idx+1 (= Sell-Level)"

    def test_momentum_holds_reset_on_recovery(self):
        """#63: Grace-Budget regeneriert sich, wenn der Preis den SL wieder verlässt."""
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({"momentum_hold_max": 2, "momentum_hold_score": 0.2})
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                                ml_enabled=False, params=params)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._direction_score = 0.9
        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 105.0, "qty": 1.0,
            "filled": False, "bought_at": 100.0,
            "sl_price": 95.0, "trailing_activated": False,
        }
        strategy._check_position_stops("SOL/USD", 94.0, state, ctx)  # Dip → Hold verbraucht
        assert state.orders[cid]["momentum_holds"] == 1
        strategy._check_position_stops("SOL/USD", 96.0, state, ctx)  # Recovery → Reset
        assert state.orders[cid]["momentum_holds"] == 0


# ── Fixes 2026-07-13 (#115, #149) ─────────────────────────────────────────────

class TestFixes20260713:

    def test_sell_only_survives_prediction_refresh(self):
        """#115: wait_fills-Stop darf von der nächsten Prediction nicht
        re-armed werden — sell_only hält with_position dauerhaft auf False."""
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                                ml_enabled=False)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")

        state.with_position = False
        state.sell_only = True

        # Simuliert _refresh_prediction mit bullischer Prediction:
        state._last_prediction = "up"
        state.with_position = ("up" != "down") and not state.sell_only
        assert state.with_position is False
        assert strategy._buys_allowed(state) is False

    def test_paper_broker_unseeded_symbol_has_no_hidden_pool(self):
        """#149: Bei Symbol-Buckets darf ein ungeseedetes Symbol nicht gegen
        einen verdeckten Voll-Pool handeln; get_balance bleibt konsistent."""
        from execution.paper import PaperBroker
        broker = PaperBroker(initial_balance=1000.0, symbols=["SOL/USD", "ETH/USD"])

        assert broker.get_balance() == pytest.approx(1000.0)
        # Ungeseedetes Symbol hat 0 Budget statt Zugriff auf 1000:
        assert broker._sym_balance("DOGE/USD") == 0.0

        # Credits eines ungeseedeten Symbols landen sichtbar in get_balance:
        broker._credit("DOGE/USD", 50.0)
        assert broker.get_balance() == pytest.approx(1050.0)

    def test_paper_broker_single_pool_unchanged(self):
        """#149-Regression: ohne symbols bleibt der Single-Pool-Modus intakt."""
        from execution.paper import PaperBroker
        broker = PaperBroker(initial_balance=500.0)
        assert broker.get_balance() == pytest.approx(500.0)
        broker._deduct("SOL/USD", 100.0)
        assert broker.get_balance() == pytest.approx(400.0)


# ── Aggressiv-Paket 2026-07-13 (DCA-Mult, Runner, Leverage 5×) ────────────────

class TestAggressivePackage:

    def test_dca_allocations_sum_and_bias(self):
        """dca_size_mult=1.6: Summe bleibt investment, unterste Linie > oberste."""
        from strategies.grid import _calc_level_allocations
        lines = [90.0, 95.0, 100.0, 105.0, 110.0]
        alloc = _calc_level_allocations(lines, 100.0, 500.0, 0.0, dca_mult=1.6)
        assert sum(alloc.values()) == pytest.approx(500.0)
        assert alloc[90.0] > alloc[110.0]
        # Monoton: jede tiefere Linie bekommt mehr
        sorted_lines = sorted(lines)
        vals = [alloc[p] for p in sorted_lines]
        assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))

    def test_dca_mult_one_is_legacy_behavior(self):
        """dca_size_mult=1.0 == exakt bisheriges Verhalten (uniform bei score~0)."""
        from strategies.grid import _calc_level_allocations
        lines = [90.0, 100.0, 110.0]
        alloc = _calc_level_allocations(lines, 100.0, 300.0, 0.0, dca_mult=1.0)
        assert all(v == pytest.approx(100.0) for v in alloc.values())

    def _runner_strategy(self, runner=True):
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict({
            "runner_enabled": runner, "runner_tp_atr": 3.0,
            "sl_mode": "per_position", "momentum_hold_max": 0,
        })
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                                ml_enabled=False, params=params)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state.with_position = True
        strategy.setup_grid("SOL/USD", 100.0, ctx)
        return strategy, ctx, state

    def test_runner_buy_fill_trails_instead_of_next_level(self):
        """Bei Hard-Uptrend: Sell = entry + 3×ATR mit runner-Flag statt nächstem Level."""
        from core.strategy import Fill
        strategy, ctx, state = self._runner_strategy(runner=True)
        state._hard_trend_up = True
        state._atr = 2.0

        cid, order = next((c, o) for c, o in state.orders.items() if o["side"] == "buy")
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)

        runner_sells = [o for o in state.orders.values()
                        if o["side"] == "sell" and o.get("runner")]
        assert len(runner_sells) == 1
        assert runner_sells[0]["price"] == pytest.approx(order["price"] + 3.0 * 2.0)
        assert "sl_price" in runner_sells[0]

    def test_no_runner_without_uptrend(self):
        """Ohne Hard-Uptrend bleibt der Sell am nächsten Grid-Level (kein runner-Flag)."""
        from core.strategy import Fill
        strategy, ctx, state = self._runner_strategy(runner=True)
        state._hard_trend_up = False
        state._atr = 2.0

        cid, order = next((c, o) for c, o in state.orders.items() if o["side"] == "buy")
        fill = Fill(client_id=cid, symbol="SOL/USD", side="buy",
                    price=order["price"], qty=order["qty"], fee=0.0, ts=time.time())
        strategy.on_fill(fill, ctx)

        assert not any(o.get("runner") for o in state.orders.values()
                       if o["side"] == "sell")

    def test_leverage_clamped_at_five(self):
        """set_leverage clampt jetzt auf 5.0 (vorher 3.0)."""
        from dashboard.db import get_leverage, set_leverage
        before = get_leverage()
        try:
            set_leverage(8.0)
            assert get_leverage() == pytest.approx(5.0)
            set_leverage(4.0)
            assert get_leverage() == pytest.approx(4.0)
        finally:
            set_leverage(before)


# ── Restart-Persistenz offener Positionen (Leak-Fix 2026-07-14) ───────────────

class TestPositionPersistence:

    def _strategy_with_position(self):
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        strategy = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                                ml_enabled=False)
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        cid = str(uuid.uuid4())
        state.orders[cid] = {
            "side": "sell", "price": 105.0, "qty": 1.5, "filled": False,
            "bought_at": 100.0, "sl_price": 96.0, "leverage": 3.0,
            "trailing_activated": False, "momentum_holds": 0,
            "entry_ts": 1234.5,
        }
        # Pre-seeded Sell und gefüllte Order dürfen NICHT exportiert werden
        state.orders["seed"] = {"side": "sell", "price": 110.0, "qty": 1.0,
                                "filled": False, "bought_at": 100.0, "pre_seeded": True}
        state.orders["done"] = {"side": "sell", "price": 102.0, "qty": 1.0,
                                "filled": True, "bought_at": 100.0}
        return strategy, ctx

    def test_export_only_real_open_positions(self):
        strategy, _ = self._strategy_with_position()
        exported = strategy.export_open_positions()
        assert list(exported.keys()) == ["SOL/USD"]
        assert len(exported["SOL/USD"]) == 1
        row = exported["SOL/USD"][0]
        assert row["bought_at"] == 100.0 and row["qty"] == 1.5
        assert row["sl_price"] == 96.0
        import json
        json.dumps(exported)  # muss JSON-serialisierbar sein

    def test_restore_roundtrip_recreates_position(self):
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        strategy, _ = self._strategy_with_position()
        exported = strategy.export_open_positions()

        fresh = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                             ml_enabled=False)
        ctx2 = MarketContext()
        fresh.init(["SOL/USD"], ctx2)
        n = fresh.restore_open_positions(exported, ctx2)
        assert n == 1

        state = fresh.get_state("SOL/USD")
        restored = [o for o in state.orders.values()
                    if o["side"] == "sell" and o.get("bought_at") == 100.0]
        assert len(restored) == 1
        assert restored[0]["sl_price"] == 96.0
        assert restored[0]["qty"] == 1.5
        # Risk-Exposure wieder registriert
        assert ctx2.open_position_count() == 1

    def test_restored_position_sl_still_fires(self):
        """Nach Restore muss der SL-Pfad greifen (kein ungestopptes Inventar)."""
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        strategy, _ = self._strategy_with_position()
        exported = strategy.export_open_positions()

        fresh = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                             ml_enabled=False)
        ctx2 = MarketContext()
        fresh.init(["SOL/USD"], ctx2)
        fresh.restore_open_positions(exported, ctx2)
        state = fresh.get_state("SOL/USD")

        fresh._check_position_stops("SOL/USD", 95.0, state, ctx2)  # unter SL 96
        assert all(o.get("filled") for o in state.orders.values()
                   if o.get("bought_at") == 100.0 and not o.get("pre_seeded"))

    def test_restore_unknown_symbol_is_skipped(self):
        from core.context import MarketContext
        from strategies.grid import GridStrategy
        fresh = GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                             ml_enabled=False)
        ctx = MarketContext()
        fresh.init(["SOL/USD"], ctx)
        n = fresh.restore_open_positions(
            {"DOGE/USD": [{"price": 1.0, "qty": 1.0, "bought_at": 0.9}]}, ctx)
        assert n == 0
