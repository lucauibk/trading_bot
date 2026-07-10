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
