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
        # btc_corr_30d (index 3) is computed independently from OHLCV and is
        # preserved even without BTC context — pass a neutral 0.0 to get all-zeros.
        feats = extract(None, btc_corr=0.0)
        assert (feats == 0).all()

    def test_market_features_preserve_btc_corr_when_no_btc(self):
        from ml.features.market import extract
        # Without BTC context, only btc_corr_30d survives (default 0.5).
        feats = extract(None)
        assert feats[3] == 0.5
        assert (np.delete(feats, 3) == 0).all()

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

    def test_predict_maps_proba_index_to_real_class_label(self):
        # Regression for #88: when a class is absent from training, clf.classes_
        # is not [0,1,2]. predict() must translate the positional argmax back to
        # the real class label, not return the column index.
        from ml.model import TradingModel

        class _StubClf:
            classes_ = np.array([0, 2])  # class 1 (hold) never seen in training

            def predict_proba(self, x):
                # highest proba is column 1 -> real class 2 (buy), not label 1 (hold)
                return np.array([[0.2, 0.8]])

        m = TradingModel("TEST/USD")
        m._clf = _StubClf()
        m._n_samples = m.MIN_SAMPLES
        m._feature_names = []  # skip feature-count check
        label, conf = m.predict(np.zeros(34, dtype=np.float32))
        assert label == 2  # buy, not the positional index 1 (hold)
        assert conf == 0.8


# ── optimize.py ready-for-live drawdown gate ─────────────────────────────────

class TestReadyForLiveDrawdown:

    def test_equity_max_drawdown_reads_capital_column(self):
        """Regression for #102: the equity table's value column is `capital`, not
        `equity`. The drawdown helper must read it and compute a real drawdown."""
        import sqlite3
        from scripts.optimize import equity_max_drawdown

        con = sqlite3.connect(":memory:")
        # Mirror dashboard/db.py equity schema exactly.
        con.execute("CREATE TABLE equity (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "timestamp TEXT, capital REAL)")
        curve = [1000, 1100, 1200, 900, 950, 1000, 1050]  # peak 1200 → trough 900 = -25%
        for i, cap in enumerate(curve):
            con.execute("INSERT INTO equity (timestamp, capital) VALUES (?, ?)",
                        (f"2026-07-0{i+1}T00:00:00", cap))
        con.commit()

        dd = equity_max_drawdown(con)
        assert dd is not None, "drawdown must be computable from the `capital` column"
        assert dd == pytest.approx((900 - 1200) / 1200)  # -0.25

    def test_equity_max_drawdown_none_when_short(self):
        import sqlite3
        from scripts.optimize import equity_max_drawdown
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE equity (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "timestamp TEXT, capital REAL)")
        con.execute("INSERT INTO equity (timestamp, capital) VALUES ('2026-07-01', 1000)")
        con.commit()
        assert equity_max_drawdown(con) is None


# ── sweep.py CLI ──────────────────────────────────────────────────────────────

class TestSweepCLI:

    def test_parser_accepts_symbol(self):
        """Regression for #101: nightly_tune passes --symbol; sweep.py's parser must
        accept it (previously it aborted with SystemExit(2), killing every nightly
        sweep)."""
        from scripts.sweep import build_parser
        args = build_parser().parse_args(
            ["--symbol", "SOL/USD", "--days", "180", "--train-days", "120", "--jobs", "4"])
        assert args.symbol == "SOL/USD"
        assert args.days == 180 and args.train_days == 120 and args.jobs == 4

    def test_parser_symbol_optional(self):
        from scripts.sweep import build_parser
        args = build_parser().parse_args([])
        assert args.symbol is None


# ── MLPredictor error path (#117) ────────────────────────────────────────────

class TestPredictErrorPathClearsScore:
    """Regression for #117: a failed predict() must expire the cached score,
    not leave a stale conviction that adaptive/directional sizing reads."""

    def test_failed_predict_resets_stale_score(self, tmp_path, monkeypatch):
        import ml.data_store as ds
        monkeypatch.setattr(ds, "DB_PATH", tmp_path / "ml.db")   # no repo side effects
        from ml.predictor import MLPredictor

        def boom(*a, **k):
            raise RuntimeError("fetch down")

        p = MLPredictor(fetch_ohlcv_fn=boom)
        p._last_scores["SOL/USD"] = 0.9          # last successful strong-up conviction
        result = p.predict("SOL/USD")            # now fails → except path

        assert result == "neutral"
        assert p.get_score("SOL/USD") == 0.0     # expired, not frozen at 0.9


# ── Directional confidence gating ─────────────────────────────────────────────

class TestDirectionalConfidence:

    def test_hold_confidence_zeroed_for_direction(self):
        """Regression for #103: a confident LightGBM 'hold' must not contribute to
        directional confidence."""
        from ml.predictor import directional_lgbm_conf
        assert directional_lgbm_conf(1, 0.9) == 0.0   # hold → no directional confidence
        assert directional_lgbm_conf(2, 0.9) == 0.9   # buy  → unchanged
        assert directional_lgbm_conf(0, 0.9) == 0.9   # sell → unchanged

    def test_confident_hold_plus_llm_up_stays_below_gate(self):
        """End-to-end numeric check from the issue: LGBM confidently 'hold' (score 0,
        conf 0.9) + LLM 'up' (conf 0.65) must NOT clear MIN_CONFIDENCE once the hold
        confidence is zeroed."""
        from ml.predictor import directional_lgbm_conf, MIN_CONFIDENCE
        from ml.llm_analyst import blend_scores

        lgbm_conf = 0.9
        lgbm_score = 0.0  # hold
        llm_result = {"score": 0.65, "confidence": 0.65}

        # Buggy path (full hold confidence) would clear the gate:
        _, buggy_conf = blend_scores(lgbm_score, lgbm_conf, llm_result)
        assert buggy_conf >= MIN_CONFIDENCE

        # Fixed path (direction-aware confidence) stays below the gate:
        conf_dir = directional_lgbm_conf(1, lgbm_conf)
        _, fixed_conf = blend_scores(lgbm_score, conf_dir, llm_result)
        assert fixed_conf < MIN_CONFIDENCE


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

    def test_directional_sl_fires_on_safety_tick(self):
        """Regression for #104: an open directional position must be stopped out via
        on_tick_safety (the freeze/emergency safety path), not only via on_tick."""
        strategy = self._strategy(floor_sl_atr_mult=1.0)
        ctx, state = self._setup(strategy)
        state._directional = {
            "entry": 100.0, "qty": 1.0, "usdt": 20.0,
            "tp": 110.0, "sl": 97.0, "entry_ts": time.time(),
        }
        # Price breaks the directional SL. on_tick_safety is what runs during a
        # daily-drawdown freeze; the directional must still exit.
        strategy.on_tick_safety("SOL/USD", 96.0, ctx)
        assert state._directional == {}, "directional SL must fire on the safety tick"

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


# ── Hot-path retrain rollback (#35) ───────────────────────────────────────────

class TestHotPathRetrainRollback:
    """The per-tick online retrain must not replace a good model with a
    materially worse one — same rollback guard the daily refresh already has. #35.
    """

    def _make(self, new_f1):
        import threading
        from ml.trainer import ModelTrainer, RETRAIN_EVERY_N

        class _FakeStore:
            def count_new_labeled_since(self, ts):
                return RETRAIN_EVERY_N
            def get_labeled(self, symbol):
                return (np.zeros((120, 34), dtype=np.float32),
                        np.zeros(120, dtype=np.int32))

        class _FakeModel:
            MIN_SAMPLES = 100
            def __init__(self):
                self._last_oos_f1 = 0.55
                self._clf = "OLD_CLF"
                self._n_samples = 200
                self._lock = threading.Lock()
                self.saves = 0
            def train(self, X, y):  # simulate a successful (gate-passing) retrain
                self._clf = "NEW_CLF"
                self._n_samples = len(X)
                self._last_oos_f1 = new_f1
                self.saves += 1
            def _save(self):
                self.saves += 1

        model = _FakeModel()
        return ModelTrainer(_FakeStore(), {"SOL/USD": model}), model

    def test_worse_model_rolled_back(self):
        trainer, model = self._make(new_f1=0.34)  # 0.55 → 0.34, drop > 0.05
        trainer._maybe_retrain("SOL/USD")
        assert model._clf == "OLD_CLF"        # restored
        assert model._last_oos_f1 == 0.55
        assert model._n_samples == 200

    def test_similar_model_kept(self):
        trainer, model = self._make(new_f1=0.53)  # within 0.05 → keep
        trainer._maybe_retrain("SOL/USD")
        assert model._clf == "NEW_CLF"
        assert model._last_oos_f1 == 0.53
# ── Trainer 34-feature contract (#55) ─────────────────────────────────────────

class TestTrainerFeatureContract:
    """_extract_training_features must return None on failure (caller skips the
    sample) rather than silently falling back to a 16-feature vector. #55.
    """

    def test_returns_none_on_extraction_failure(self, monkeypatch):
        import ml.trainer as trainer
        def _boom(*a, **k):
            raise ValueError("simulated 34-feature extraction failure")
        monkeypatch.setattr(trainer, "extract_all_features", _boom)
        df = _make_df(60)
        out = trainer._extract_training_features(df, df, btc_corr=0.0)
        assert out is None  # NOT a 16-element fallback vector

    def test_returns_34_vector_on_success(self):
        import ml.trainer as trainer
        df = _make_df(120)
        out = trainer._extract_training_features(df, df, btc_corr=0.3)
        assert out is not None
        assert out.shape == (34,)
# ── Trainer stale-candle labeling (#91) ───────────────────────────────────────

class TestTrainerStaleCandleLabel:
    """A sample whose timestamp predates the labeling window must NOT be labeled
    from the clamped first candle (get_indexer nearest never returns -1). #91.
    """

    def _trainer(self, monkeypatch, tmp_path):
        import ml.data_store as data_store
        monkeypatch.setattr(data_store, "DB_PATH", tmp_path / "ml_training.db")
        from ml.data_store import MLDataStore
        from ml.trainer import ModelTrainer
        store = MLDataStore()
        return ModelTrainer(store, {}), store

    def test_out_of_window_sample_not_labeled(self, monkeypatch, tmp_path):
        trainer, store = self._trainer(monkeypatch, tmp_path)
        df = _make_df(60)  # hourly candles starting 2024-01-01
        df.index = df.index.tz_localize("UTC")  # match production UTC-aware index
        sym = "SOL/USD"
        feats = np.zeros(34, dtype=np.float32)

        in_window_ts  = int(df.index[20].timestamp())
        out_window_ts = int(df.index[0].timestamp()) - 100 * 24 * 3600  # 100 days earlier

        store.store(sym, in_window_ts,  feats, 100.0, 1)
        store.store(sym, out_window_ts, feats, 100.0, 1)

        trainer.label_and_maybe_retrain(sym, df)

        # in-window sample got labeled; out-of-window one was skipped (still NULL)
        unlabeled_ts = {ts for _, ts, _ in store.get_unlabeled_before(int(time.time()))}
        assert out_window_ts in unlabeled_ts, "stale sample must NOT be mislabeled"
        assert in_window_ts not in unlabeled_ts, "in-window sample should be labeled"
# ── Correlation-tracker wiring (#43) ──────────────────────────────────────────

class TestCorrelationWiring:
    """RiskManager.can_open() step 5 (over-concentration bucket) was dead because
    CorrelationTracker was never fed. Engine._refresh_correlations() now feeds it.
    """

    def test_refresh_correlations_populates_tracker(self, monkeypatch):
        import types
        import numpy as np
        import pandas as pd
        from core.engine import Engine
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager

        idx = pd.date_range("2024-01-01", periods=300, freq="1h")
        rng = np.random.RandomState(0)
        btc = pd.Series(100 + np.cumsum(rng.normal(0, 1.0, 300)), index=idx)
        # SOL strongly driven by BTC → high positive correlation (tiny idiosyncratic noise)
        sol = btc * 0.1 + pd.Series(rng.normal(0, 0.003, 300), index=idx)

        corr = CorrelationTracker()
        risk = RiskManager(corr)

        state = types.SimpleNamespace(_last_df=pd.DataFrame({"close": sol}, index=idx))
        strat = types.SimpleNamespace(_risk=risk, get_state=lambda s: state)
        fake_self = types.SimpleNamespace(strategy=strat, symbols=["SOL/USD"])

        monkeypatch.setattr("market.btc_context.get_btc_close", lambda: btc)

        # Call the unbound method with a fake self (avoids full Engine construction).
        Engine._refresh_correlations(fake_self)

        assert "SOL/USD" in corr._correlations
        assert corr._correlations["SOL/USD"] > 0.9
        assert "SOL/USD" in corr.high_correlation_symbols(0.85)

    def test_refresh_correlations_safe_without_btc_close(self, monkeypatch):
        import types
        from core.engine import Engine
        from risk.correlation import CorrelationTracker
        from risk.manager import RiskManager

        corr = CorrelationTracker()
        risk = RiskManager(corr)
        strat = types.SimpleNamespace(_risk=risk, get_state=lambda s: None)
        fake_self = types.SimpleNamespace(strategy=strat, symbols=["SOL/USD"])
        monkeypatch.setattr("market.btc_context.get_btc_close", lambda: None)

        # No BTC close available → no-op, must not raise.
        Engine._refresh_correlations(fake_self)
        assert corr._correlations == {}
# ── Emergency-stop keeps SL/TP alive (#34) ────────────────────────────────────

class TestEmergencyStopSL:
    """A symbol past its per-coin realized-loss cap must still receive
    on_tick_safety (SL/TP) while new-order paths are blocked — like the freeze.
    Previously the engine `continue`d before on_tick_safety, orphaning positions.
    """

    def _run_one_tick(self, monkeypatch, total_profit):
        import types
        from core.engine import Engine
        from execution.paper import PaperBroker
        from core.context import MarketContext

        calls = {"safety": 0, "on_tick": 0, "sync": 0}
        state = types.SimpleNamespace(total_profit=total_profit,
                                      investment=100.0, grid_lines=[])

        class FakeStrategy:
            _broker = None
            def get_state(self, s): return state
            def on_tick_safety(self, s, p, ctx): calls["safety"] += 1
            def on_tick(self, s, p, ctx): calls["on_tick"] += 1
            def on_candle(self, s, df, ctx): pass

        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        eng = Engine(FakeStrategy(), broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)

        # Neutralize everything except the per-symbol gating under test.
        for name in ("_check_dashboard_stop", "_check_daily_drawdown",
                     "_reconcile_fills", "_log_equity"):
            monkeypatch.setattr(eng, name, lambda *a, **k: None)
        monkeypatch.setattr(eng, "_update_dashboard", lambda s, p: None)
        monkeypatch.setattr(eng, "_update_prediction_outcomes", lambda f: None)
        monkeypatch.setattr(eng, "_sync_orders",
                            lambda s, p: calls.__setitem__("sync", calls["sync"] + 1))

        import data_fetcher
        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": 100.0})
        monkeypatch.setattr(data_fetcher, "fetch_ohlcv", lambda *a, **k: None)
        import core.engine as ce
        monkeypatch.setattr(ce.time, "sleep", lambda *a, **k: None)

        eng._loop_count = 1  # off every cadence (recheck/rebuild/btc/funding)
        eng._tick()
        return calls

    def test_healthy_symbol_trades_normally(self, monkeypatch):
        calls = self._run_one_tick(monkeypatch, total_profit=0.0)
        assert calls == {"safety": 1, "on_tick": 1, "sync": 1}

    def test_emergency_stopped_keeps_sl_blocks_orders(self, monkeypatch):
        # -12% of investment 100 = -12 → total_profit -20 trips the emergency stop.
        calls = self._run_one_tick(monkeypatch, total_profit=-20.0)
        assert calls["safety"] == 1   # SL/TP still runs (was 0 before the fix)
        assert calls["on_tick"] == 0  # no new buys
        assert calls["sync"] == 0     # no new orders submitted


class TestCoinSettingsLiveToggle:
    """#184: the dashboard coin `enabled` toggle must work as a *live* control.
    A coin disabled mid-run is re-read each tick (paper only) and frozen like an
    emergency stop — no new buys (on_tick/_sync_orders), but SL/TP stays live.
    """

    def _run_one_tick(self, monkeypatch, enabled):
        import types
        from core.engine import Engine
        from execution.paper import PaperBroker
        from core.context import MarketContext

        calls = {"safety": 0, "on_tick": 0, "sync": 0}
        state = types.SimpleNamespace(total_profit=0.0,
                                      investment=100.0, grid_lines=[])

        class FakeStrategy:
            _broker = None
            def get_state(self, s): return state
            def on_tick_safety(self, s, p, ctx): calls["safety"] += 1
            def on_tick(self, s, p, ctx): calls["on_tick"] += 1
            def on_candle(self, s, df, ctx): pass

        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        eng = Engine(FakeStrategy(), broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)

        for name in ("_check_dashboard_stop", "_check_daily_drawdown",
                     "_reconcile_fills", "_log_equity"):
            monkeypatch.setattr(eng, name, lambda *a, **k: None)
        monkeypatch.setattr(eng, "_update_dashboard", lambda s, p: None)
        monkeypatch.setattr(eng, "_update_prediction_outcomes", lambda f: None)
        monkeypatch.setattr(eng, "_sync_orders",
                            lambda s, p: calls.__setitem__("sync", calls["sync"] + 1))

        # Drive _refresh_coin_settings via the real DB helper (monkeypatched).
        import dashboard.db as ddb
        monkeypatch.setattr(
            ddb, "get_all_coin_settings",
            lambda: [{"symbol": "SOL/USD", "max_investment": 100.0,
                      "enabled": 1 if enabled else 0}],
        )

        import data_fetcher
        monkeypatch.setattr(data_fetcher, "fetch_ticker", lambda s: {"last": 100.0})
        monkeypatch.setattr(data_fetcher, "fetch_ohlcv", lambda *a, **k: None)
        import core.engine as ce
        monkeypatch.setattr(ce.time, "sleep", lambda *a, **k: None)

        eng._loop_count = 1
        eng._tick()
        return calls, eng

    def test_enabled_coin_trades_normally(self, monkeypatch):
        calls, eng = self._run_one_tick(monkeypatch, enabled=True)
        assert calls == {"safety": 1, "on_tick": 1, "sync": 1}
        assert eng._disabled_coins == set()

    def test_disabled_coin_keeps_sl_blocks_orders(self, monkeypatch):
        calls, eng = self._run_one_tick(monkeypatch, enabled=False)
        assert calls["safety"] == 1   # SL/TP still protects open positions
        assert calls["on_tick"] == 0  # no new buys
        assert calls["sync"] == 0     # no new orders submitted
        assert eng._disabled_coins == {"SOL/USD"}

    def test_refresh_is_paper_only(self, monkeypatch):
        # Live broker must not be touched (Live-Parität lock, #171) — refresh no-ops.
        from core.engine import Engine
        from core.context import MarketContext

        class FakeLiveBroker:  # not a PaperBroker
            def cancel_all(self, s): pass

        called = {"n": 0}
        import dashboard.db as ddb
        monkeypatch.setattr(ddb, "get_all_coin_settings",
                            lambda: called.__setitem__("n", called["n"] + 1) or [])
        eng = Engine(object(), FakeLiveBroker(), ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)
        eng._refresh_coin_settings()
        assert called["n"] == 0                 # DB never queried for a live broker
        assert eng._disabled_coins == set()
# ── Engine equity staleness (#89) ─────────────────────────────────────────────

class TestEquityStaleGuard:
    """A permanently-failing ticker must NOT freeze the whole equity curve /
    daily-drawdown brake forever. Brief staleness is skipped (sleep-wake guard,
    #20); persistent staleness falls back to last-good prices (#89).
    """

    class _FakeState:
        def __init__(self):
            # one open (unfilled) grid sell with a known entry → deterministic MTM
            self.orders = {
                "s1": {"side": "sell", "filled": False, "bought_at": 90.0,
                       "qty": 1.0, "leverage": 1.0},
            }
            self.total_profit = 0.0

    class _FakeStrategy:
        def __init__(self, state):
            self._state = state
        def get_state(self, sym):
            return self._state

    class _FakeBroker:
        def get_balance(self, currency="USD"):
            return 500.0

    def _engine(self):
        from core.engine import Engine
        from core.context import MarketContext
        state = self._FakeState()
        eng = Engine(self._FakeStrategy(state), self._FakeBroker(),
                     ["SOL/USD"], ctx=MarketContext())
        # last-good price 100 → MTM = margin(90) + unrealized(1*(100-90)) = 100
        eng._last_prices = {"SOL/USD": 100.0}
        return eng

    def test_brief_staleness_skips_equity_update(self):
        eng = self._engine()
        eng.ctx.set_equity(999.0)  # sentinel
        eng._last_price_ts = {"SOL/USD": time.time() - 10_000}  # very stale
        eng._log_equity()
        # within grace → skipped, sentinel retained (no false drawdown trigger)
        assert eng.ctx.total_equity == 999.0
        assert eng._equity_stale_since is not None

    def test_persistent_staleness_falls_back_to_last_good_price(self):
        from core.engine import STALE_EQUITY_GRACE_SECONDS
        eng = self._engine()
        eng.ctx.set_equity(999.0)  # sentinel
        eng._last_price_ts = {"SOL/USD": time.time() - 10_000}
        # pretend staleness began well before the grace window
        eng._equity_stale_since = time.time() - (STALE_EQUITY_GRACE_SECONDS + 30)
        eng._log_equity()
        # grace expired → equity logged with last-good price: 500 balance + 100 MTM
        assert eng.ctx.total_equity == pytest.approx(600.0)

    def test_fresh_prices_reset_stale_marker(self):
        eng = self._engine()
        eng._equity_stale_since = time.time() - 5.0  # was stale
        eng._last_price_ts = {"SOL/USD": time.time()}  # now fresh
        eng._log_equity()
        assert eng._equity_stale_since is None
        assert eng.ctx.total_equity == pytest.approx(600.0)
# ── cancel_all failure visibility (#61) ───────────────────────────────────────

class TestCancelAllLogging:
    """cancel_all must not silently swallow cancel failures — a stale live order
    left on the book after 'stop' can fill unexpectedly."""

    def test_cancel_all_counts_successes_and_logs_failures(self, caplog):
        import types
        import logging
        from execution.kraken import KrakenBroker

        orders = [types.SimpleNamespace(exchange_order_id=oid) for oid in ("a", "b", "c")]

        def cancel_order(oid):
            if oid == "b":
                raise RuntimeError("network down")  # not a ccxt err → no retry delay
            return {}

        fake = types.SimpleNamespace(
            get_open_orders=lambda s: orders,
            _ex=types.SimpleNamespace(cancel_order=cancel_order),
        )
        with caplog.at_level(logging.WARNING):
            count = KrakenBroker.cancel_all(fake, "SOL/USD")

        assert count == 2  # a and c cancelled; b failed
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "cancel failed" in msgs
        assert "still OPEN" in msgs
        assert "b" in msgs
# ── Live reconciler robustness (#76) ──────────────────────────────────────────

class TestReconcileFeeNone:
    """reconcile_fills must not drop the whole batch when one ccxt trade has
    fee=None (the key exists but is None, so t.get('fee', {}) returns None)."""

    def _fills(self, trades):
        import types
        from execution.kraken import KrakenBroker
        fake = types.SimpleNamespace(
            _ex=types.SimpleNamespace(fetch_my_trades=lambda *a, **k: trades),
            _client_to_exchange={},
        )
        return KrakenBroker.reconcile_fills(fake, 0.0)

    def test_fee_none_does_not_drop_batch(self):
        trades = [
            {"order": "o1", "symbol": "SOL/USD", "side": "buy", "price": 100.0,
             "amount": 1.0, "fee": {"cost": 0.16}, "timestamp": 1000, "id": "t1"},
            {"order": "o2", "symbol": "SOL/USD", "side": "sell", "price": 110.0,
             "amount": 1.0, "fee": None, "timestamp": 2000, "id": "t2"},  # fee=None
        ]
        fills = self._fills(trades)
        assert len(fills) == 2
        assert fills[0].fee == pytest.approx(0.16)
        assert fills[1].fee == 0.0

    def test_one_malformed_trade_skipped_others_survive(self):
        trades = [
            {"order": "o1", "symbol": "SOL/USD", "side": "buy", "price": 100.0,
             "amount": 1.0, "fee": {"cost": 0.1}, "timestamp": 1000, "id": "t1"},
            {"order": "bad", "id": "t2"},  # missing price/symbol → skipped
            {"order": "o3", "symbol": "ETH/USD", "side": "sell", "price": 3000.0,
             "amount": 0.5, "fee": {"cost": 2.4}, "timestamp": 3000, "id": "t3"},
        ]
        fills = self._fills(trades)
        assert len(fills) == 2
        assert {f.symbol for f in fills} == {"SOL/USD", "ETH/USD"}
# ── Floor-SL paper credit (#39 double-fee, #57 entry-leverage) ────────────────

class TestFloorSLCredit:
    """The floor-SL exit credits margin+PnL back to the PaperBroker manually
    (the broker never sees the SL fill). Regression guards for two bugs:
      #39 — a full round-trip fee was charged, double-counting the buy fee that
            was already deducted at buy-fill time.
      #57 — the live dashboard leverage was used to return margin instead of the
            leverage the position was entered with → balance drift on lev change.
    """

    def _sl_credit_amount(self, monkeypatch, entry_lev, live_lev):
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from execution.paper import PaperBroker
        from core.context import MarketContext

        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard logging

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(leverage=live_lev),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.orders["sell1"] = {
            "side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "sl_price": 99.0, "leverage": entry_lev,
            "momentum_holds": 0,
        }
        strat._states["SOL/USD"] = state

        captured = {}
        orig = broker.sl_credit
        monkeypatch.setattr(
            broker, "sl_credit",
            lambda symbol, amount: (captured.__setitem__("amount", amount),
                                    orig(symbol, amount))[1],
        )

        # price 98 < sl_price 99 → floor-SL fires
        strat._check_position_stops("SOL/USD", 98.0, state, MarketContext())
        return captured.get("amount")

    def test_sl_credit_charges_sell_fee_only(self, monkeypatch):
        from strategies.grid import KRAKEN_FEE
        amount = self._sl_credit_amount(monkeypatch, entry_lev=1.0, live_lev=1.0)
        # margin(100) + pnl(-2) - sell_fee(98 * KRAKEN_FEE); buy fee NOT re-charged
        expected = 100.0 + (98.0 - 100.0) - 98.0 * KRAKEN_FEE
        assert amount == pytest.approx(expected)
        assert amount == pytest.approx(97.8432)

    def test_sl_credit_uses_entry_leverage(self, monkeypatch):
        from strategies.grid import KRAKEN_FEE
        # Entered at lev=2, dashboard later switched to lev=1 → still use lev=2.
        amount = self._sl_credit_amount(monkeypatch, entry_lev=2.0, live_lev=1.0)
        expected = 100.0 / 2.0 + (98.0 - 100.0) - 98.0 * KRAKEN_FEE
        assert amount == pytest.approx(expected)


class TestEmergencySellAllCredit:
    """#180: the graceful "sell_all" paper stop must credit margin+PnL back to
    the broker cash bucket, exactly like the floor-SL path. Before the fix the
    synthetic sell only updated in-memory total_profit and logged the trade;
    the broker sell order was already cancelled, so update_price never credited
    the margin -> the persisted balance leaked margin+unrealized on every stop.
    """

    def _run_emergency(self, monkeypatch, entry_lev, sell_price):
        from core.engine import Engine, KRAKEN_FEE
        from execution.paper import PaperBroker
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from core.context import MarketContext

        monkeypatch.setenv("GRIDBOT_BACKTEST", "1")  # skip dashboard/notifier

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(leverage=entry_lev),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.orders["sell1"] = {
            "side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "sl_price": 99.0, "leverage": entry_lev,
            "momentum_holds": 0, "entry_ts": 0.0,
        }
        strat._states["SOL/USD"] = state

        eng = Engine(strat, broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)

        import data_fetcher
        monkeypatch.setattr(data_fetcher, "fetch_ticker",
                            lambda s: {"last": sell_price})

        bucket_before = broker._balances["SOL/USD"]
        eng._emergency_sell_all()
        return broker._balances["SOL/USD"], bucket_before, KRAKEN_FEE

    def test_bucket_credited_margin_and_pnl(self, monkeypatch):
        after, before, fee = self._run_emergency(monkeypatch, entry_lev=1.0,
                                                 sell_price=105.0)
        # margin(100) + pnl(+5) - sell_fee(105*fee); buy fee NOT re-charged.
        credit = 100.0 / 1.0 + (105.0 - 100.0) - 105.0 * fee
        assert after == pytest.approx(before + credit)

    def test_bucket_credit_uses_entry_leverage(self, monkeypatch):
        after, before, fee = self._run_emergency(monkeypatch, entry_lev=2.0,
                                                 sell_price=105.0)
        credit = 100.0 / 2.0 + (105.0 - 100.0) - 105.0 * fee
        assert after == pytest.approx(before + credit)


class TestPaperRestartMargin:
    """#183: normal SIGTERM/restart/crash never sells open positions, and only
    the cash balance is persisted (not the positions). Without settling them,
    each restart drops the persisted equity by the full margin + unrealized PnL.
    _mtm_close_paper_positions() must credit exactly margin+unrealized (== the MTM
    _log_equity reports) into the bucket so restart equity == displayed equity.
    """

    def _build_engine(self, entry_lev, last_price):
        from core.engine import Engine
        from execution.paper import PaperBroker
        from strategies.grid import GridStrategy, _GridState
        from strategies.grid_params import GridParams
        from core.context import MarketContext

        strat = GridStrategy(
            [{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
            ml_enabled=False,
            params=GridParams(leverage=entry_lev),
        )
        broker = PaperBroker(initial_balance=100.0, symbols=["SOL/USD"])
        strat._broker = broker

        state = _GridState("SOL/USD", 100.0, 6, 0.05)
        state.orders["sell1"] = {
            "side": "sell", "price": 110.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "sl_price": 99.0, "leverage": entry_lev,
        }
        strat._states["SOL/USD"] = state

        eng = Engine(strat, broker, ["SOL/USD"],
                     ctx=MarketContext(), initial_capital=100.0)
        eng._last_prices["SOL/USD"] = last_price
        return eng, broker, state

    def test_mtm_close_credits_margin_and_unrealized(self):
        eng, broker, state = self._build_engine(entry_lev=1.0, last_price=105.0)
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        # margin(100) + unrealized(+5); no synthetic fee -> equity continuity.
        assert broker._balances["SOL/USD"] == pytest.approx(before + 105.0)
        assert "sell1" not in state.orders  # credited exactly once

    def test_mtm_close_uses_entry_leverage(self):
        eng, broker, state = self._build_engine(entry_lev=2.0, last_price=105.0)
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        # margin(100/2=50) + unrealized(+5)
        assert broker._balances["SOL/USD"] == pytest.approx(before + 55.0)

    def test_mtm_close_skips_without_known_price(self):
        eng, broker, state = self._build_engine(entry_lev=1.0, last_price=105.0)
        eng._last_prices["SOL/USD"] = 0.0  # no known price
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        assert broker._balances["SOL/USD"] == pytest.approx(before)  # unchanged
        assert "sell1" in state.orders  # left intact, not lost

    def test_mtm_close_ignores_pre_seeded(self):
        eng, broker, state = self._build_engine(entry_lev=1.0, last_price=105.0)
        state.orders["seed1"] = {
            "side": "sell", "price": 120.0, "qty": 1.0, "filled": False,
            "bought_at": 100.0, "pre_seeded": True, "leverage": 1.0,
        }
        before = broker._balances["SOL/USD"]
        eng._mtm_close_paper_positions()
        # only the real position (sell1) is credited; pre-seeded had no margin.
        assert broker._balances["SOL/USD"] == pytest.approx(before + 105.0)
        assert "seed1" in state.orders
# ── Latent correctness traps (#78) ────────────────────────────────────────────

class TestLatentTraps:

    def test_level_allocations_empty_grid_no_zerodiv(self):
        # #78.1: n == 0 must early-out to {}, not fall into `investment / n`.
        from strategies.grid import _calc_level_allocations
        assert _calc_level_allocations([], 100.0, 40.0, 0.5) == {}

    def test_level_allocations_uniform_when_neutral(self):
        # Non-empty grid, neutral score → uniform split (sanity, unchanged path).
        from strategies.grid import _calc_level_allocations
        alloc = _calc_level_allocations([10.0, 11.0], 10.5, 40.0, 0.0)
        assert alloc == {10.0: 20.0, 11.0: 20.0}

    def test_risk_position_size_no_leverage_param(self):
        # #78.2: the misleading (silently-ignored) leverage param is gone.
        import inspect
        from risk.manager import RiskManager
        assert "leverage" not in inspect.signature(RiskManager.position_size).parameters

    def test_data_fetcher_get_balance_defaults_usd(self):
        # #78.3: default currency must be USD (Kraken/USD account), not USDT.
        import inspect
        import data_fetcher
        assert inspect.signature(data_fetcher.get_balance).parameters["currency"].default == "USD"


class TestMomentumHoldReset:
    """#165: the momentum-hold SL-delay budget must reset once price recovers
    above the SL, so it grants N ticks of grace *per contiguous dip episode*,
    not once over the whole position lifetime."""

    def _strategy(self, **overrides):
        from strategies.grid import GridStrategy
        from strategies.grid_params import GridParams
        params = GridParams.from_dict(
            {"sl_mode": "floor", "leverage": 1.0,
             "momentum_hold_score": 0.35, "momentum_hold_max": 1, **overrides})
        return GridStrategy([{"symbol": "SOL/USD", "investment": 100.0, "levels": 6}],
                            ml_enabled=False, params=params)

    def _setup(self, strategy):
        from core.context import MarketContext
        ctx = MarketContext()
        strategy.init(["SOL/USD"], ctx)
        state = strategy.get_state("SOL/USD")
        state._atr = 2.0
        # One open long grid position with a sell/SL leg.
        state.orders = {
            "c1": {"side": "sell", "price": 105.0, "qty": 1.0, "filled": False,
                   "bought_at": 100.0, "sl_price": 98.0, "momentum_holds": 0,
                   "leverage": 1.0},
        }
        state._direction_score = 0.9  # strongly bullish → holds are granted
        return ctx, state

    def test_recovery_resets_hold_budget(self):
        strategy = self._strategy()
        ctx, state = self._setup(strategy)

        # Dip #1: below SL → grace granted, no SL fire.
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)
        assert "c1" in state.orders, "first dip must be held, not stopped"
        assert state.orders["c1"]["momentum_holds"] == 1

        # Recovery above SL → budget must reset to 0.
        strategy._check_position_stops("SOL/USD", 101.0, state, ctx)
        assert state.orders["c1"]["momentum_holds"] == 0, "recovery must reset holds"

        # Dip #2 (an independent episode): must be held again, not stop out.
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)
        assert "c1" in state.orders, "independent later dip must still get grace"
        assert state.orders["c1"]["momentum_holds"] == 1

    def test_contiguous_dip_still_exhausts_budget(self):
        # Without any recovery, the budget is still finite: the 2nd contiguous
        # tick below SL stops out (max=1). Guards against the reset masking the cap.
        strategy = self._strategy()
        ctx, state = self._setup(strategy)
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)  # held (hold=1)
        assert "c1" in state.orders
        strategy._check_position_stops("SOL/USD", 97.0, state, ctx)  # budget spent → SL
        assert "c1" not in state.orders, "contiguous dip must still stop out after max"
